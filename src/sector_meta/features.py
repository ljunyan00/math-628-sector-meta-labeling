"""Causal daily predictors; no fitting or full-sample quantiles."""

import numpy as np
import pandas as pd

TECHNICAL = ["return_1", "return_5", "return_20", "ma_gap", "rsi", "macd", "volatility", "relative_volume"]
MARKET = ["vix_lag1", "vix_change_lag1", "spy_return_20", "spy_drawdown_60"]
CROSS_SECTOR = ["sector_corr_60", "sector_dispersion_5", "relative_return_20"]
FEATURE_SETS = {"technical": TECHNICAL + ["ticker"],
                "market": TECHNICAL + MARKET + ["ticker"],
                "full": TECHNICAL + MARKET + CROSS_SECTOR + ["corr_drawdown", "ticker"]}


def build_features(prices, calendar, vix, cfg):
    closes = pd.DataFrame({t: prices[t].adj_close for t in cfg["sectors"]})
    returns = closes.pct_change(fill_method=None)
    pairs = [returns[a].rolling(60, min_periods=60).corr(returns[b])
             for i, a in enumerate(cfg["sectors"]) for b in cfg["sectors"][i + 1:]]
    corr = pd.concat(pairs, axis=1).mean(axis=1, skipna=False)
    dispersion = returns.std(axis=1, ddof=1).rolling(5, min_periods=5).mean()
    basket_return_20 = (1 + returns.mean(axis=1)).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    # Fill historical publication gaps BEFORE shifting by one exchange trading session.
    observed = vix.dropna()
    aligned = observed.reindex(observed.index.union(calendar)).sort_index().ffill().reindex(calendar)
    source_dates = pd.Series(observed.index, index=observed.index)
    source_dates = source_dates.reindex(source_dates.index.union(calendar)).sort_index().ffill().reindex(calendar).shift(1)
    lagged = aligned.shift(1)
    age = (pd.Series(calendar, index=calendar) - source_dates).dt.days
    lagged = lagged.where(age <= 7)
    spy = prices[cfg["market"]].adj_close
    context = pd.DataFrame({"vix_lag1": lagged,
                            "vix_change_lag1": aligned.pct_change(fill_method=None).shift(1).where(age <= 7),
                            "vix_source_date": source_dates,
                            "spy_return_20": spy.pct_change(20, fill_method=None),
                            "spy_drawdown_60": spy / spy.rolling(60, min_periods=60).max() - 1,
                            "sector_corr_60": corr, "sector_dispersion_5": dispersion})
    context["corr_drawdown"] = context.sector_corr_60 * context.spy_drawdown_60
    frames = []
    for ticker in cfg["sectors"]:
        price = closes[ticker]
        log_return = np.log(price).diff()
        sigma = log_return.ewm(span=cfg["ewm_span"], adjust=False, min_periods=cfg["ewm_span"]).std()
        ma20, ma50 = price.rolling(20).mean(), price.rolling(50).mean()
        delta = price.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        rsi = rsi.mask((gain == 0) & (loss == 0), 50)
        frame = pd.DataFrame({"ticker": ticker, "return_1": returns[ticker],
                              "return_5": price.pct_change(5, fill_method=None),
                              "return_20": price.pct_change(20, fill_method=None),
                              "ma_gap": ma20 / ma50 - 1, "rsi": rsi / 100,
                              "macd": (price.ewm(span=12, adjust=False).mean() - price.ewm(span=26, adjust=False).mean()) / price,
                              "volatility": sigma,
                              "relative_volume": prices[ticker].Volume / prices[ticker].Volume.rolling(20).mean(),
                              "relative_return_20": price.pct_change(20, fill_method=None) - basket_return_20,
                              "eligible": ma20 > ma50, "log_return": log_return,
                              "cusum_threshold": cfg["cusum_multiplier"] * sigma.shift(1)}, index=calendar)
        frames.append(frame.join(context).rename_axis("Date").reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "ticker"]).reset_index(drop=True)
