"""Chronological model comparison, frozen evaluation and research artifacts."""

from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from shutil import copyfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .backtest import buy_and_hold, performance, run_portfolio
from .cnn_models import MODELS, predict_fold
from .data import read_factors
from .pipeline import markdown_table, weekly_coverage
from .sequence_data import event_inputs
from .splits import make_splits, purged_train

COLUMNS = ["event_id", "ticker", "signal_date", "exit_date", "label", "vix_lag1"]
MODEL_FILES = ("sequence_data.py", "cnn_models.py", "cnn_experiment.py", "features.py", "events.py", "splits.py", "backtest.py", "data.py")


def provenance(cfg, spec, manifest):
    here = Path(__file__).parent
    return {
        "stage1_config": cfg,
        "cnn_config": spec,
        "input_sha256": {name: meta["sha256"] for name, meta in manifest["files"].items()},
        "code_sha256": {name: sha256((here / name).read_bytes()).hexdigest() for name in MODEL_FILES},
        "versions": {name: version(name) for name in ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "statsmodels"]},
    }


def forecast_metrics(y, p):
    both_classes = len(np.unique(y)) == 2
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "average_precision": average_precision_score(y, p) if both_classes else np.nan,
        "roc_auc": roc_auc_score(y, p) if both_classes else np.nan,
    }


def score_forecasts(frame, fold_col=None):
    groups = [("pooled", frame)]
    if fold_col is not None:
        groups.extend(frame.groupby(fold_col, sort=True))
    rows = [
        {"period": name, "model": model, "events": len(group), **forecast_metrics(group.label, group[model])}
        for name, group in groups for model in MODELS
    ]
    return pd.DataFrame(rows)


def save_calibration(frame, path):
    rows = []
    for model in MODELS:
        bins = pd.cut(frame[model], bins=np.linspace(0, 1, 11), include_lowest=True)
        summary = frame.groupby(bins, observed=True).agg(
            events=("label", "size"), observed_rate=("label", "mean"), mean_probability=(model, "mean"),
        )
        for bucket, values in summary.iterrows():
            rows.append({"model": model, "probability_bin": str(bucket), **values.to_dict()})
    pd.DataFrame(rows).to_csv(path, index=False)


def threshold_for_coverage(probabilities, coverage):
    count = int(np.ceil(len(probabilities) * coverage))
    if not 1 <= count <= len(probabilities):
        raise ValueError("A coverage threshold requires a nonempty sample and coverage in (0, 1].")
    return float(np.sort(np.asarray(probabilities, dtype=float))[-count])


def simulate(prices, events, cfg, start, end, mask, cost=None):
    if cost is None:
        cost = cfg["cost_bps_per_side"]
    curve, trades = run_portfolio(prices, events, cfg["sectors"], start, end, cost, events.event_id[mask])
    return curve, {
        "candidate_coverage": float(mask.mean()),
        "candidates_accepted": int(mask.sum()),
        "execution_rate": len(trades) / len(events),
        **performance(curve, trades),
    }


def select_vix(prices, events, cfg, start, end):
    rows = []
    for cutoff in cfg["vix_cutoff_grid"]:
        _, metrics = simulate(prices, events, cfg, start, end, events.vix_lag1.le(cutoff))
        rows.append({"cutoff": cutoff, **metrics})
    sweep = pd.DataFrame(rows)
    eligible = sweep.loc[(sweep.candidate_coverage >= cfg["minimum_candidate_coverage"]) & sweep.sharpe.notna()]
    if eligible.empty:
        raise ValueError("No valid VIX threshold met the minimum coverage.")
    best = eligible.sort_values(["sharpe", "cutoff"], ascending=[False, True]).iloc[0]
    return float(best.cutoff), sweep


def develop(root, prices, calendar, vix, cfg, spec, manifest):
    torch.set_num_threads(2)
    out = root / "output/cnn"
    out.mkdir(parents=True, exist_ok=True)
    end = cfg["development_end_exclusive"]
    start = f'{cfg["validation_years"][0]}-01-01'
    events, windows, returns, peers = event_inputs(prices, calendar, vix, cfg, spec, end)
    splits, _ = make_splits(events, returns.index, cfg)
    chunks = []
    for year in cfg["validation_years"]:
        training, validation = splits[f"train_{year}"], splits[f"validation_{year}"]
        chunk = events.iloc[validation][COLUMNS].copy()
        chunk["year"] = year
        predictions = predict_fold(events, windows, returns, peers, training, validation, cfg, spec)
        for model, scored in predictions.items():
            chunk[model] = scored
        chunks.append(chunk)
        print(f"{year}: completed four model fits on {len(training):,} training and {len(validation):,} validation events", flush=True)
    scored = pd.concat(chunks).sort_index()
    scored.to_csv(out / "development_predictions.csv", index=False)
    forecasts = score_forecasts(scored, "year")
    forecasts.to_csv(out / "development_forecast_metrics.csv", index=False)
    save_calibration(scored, out / "development_calibration.csv")
    opportunities = events.loc[scored.index]
    thresholds = {model: threshold_for_coverage(scored[model], spec["target_candidate_coverage"]) for model in MODELS}
    cutoff, vix_sweep = select_vix(prices, opportunities, cfg, start, end)
    vix_sweep.to_csv(out / "development_vix_cutoffs.csv", index=False)
    masks = {model: scored[model].ge(threshold) for model, threshold in thresholds.items()}
    masks["unfiltered"] = pd.Series(True, index=opportunities.index)
    masks["vix_filter"] = opportunities.vix_lag1.le(cutoff)
    portfolio = []
    for strategy, mask in masks.items():
        _, metrics = simulate(prices, opportunities, cfg, start, end, mask)
        portfolio.append({"strategy": strategy, **metrics})
    trading = pd.DataFrame(portfolio)
    trading.to_csv(out / "development_portfolio_metrics.csv", index=False)
    locked = {
        "provenance": provenance(cfg, spec, manifest),
        "probability_thresholds": thresholds,
        "development_acceptance": {model: float(masks[model].mean()) for model in MODELS},
        "vix_cutoff": cutoff,
    }
    (out / "frozen_spec.json").write_text(json.dumps(locked, indent=2, allow_nan=False) + "\n")
    print(f"Frozen specification: {out / 'frozen_spec.json'}")


def sharpe(returns, axis=0):
    values = np.asarray(returns)
    mean, deviation = values.mean(axis=axis), values.std(axis=axis, ddof=1)
    return np.divide(np.sqrt(252) * mean, deviation, out=np.full_like(mean, np.nan), where=deviation > 0)


def bootstrap_sharpe(curves, random_curves, spec, seed):
    strategies = ["cnn_2d", "cnn_1d", "xgboost", "unfiltered", "vix_filter"]
    values = np.column_stack([curves[k].daily_return.to_numpy() for k in strategies] + random_curves)
    length = len(values)
    block = spec["bootstrap_block_sessions"]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, length - block + 1, size=(spec["bootstrap_draws"], n_blocks))
    positions = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, :length]
    estimates = np.array([sharpe(values[index]) for index in positions])
    observed = sharpe(values)
    names = strategies[1:] + ["random_matched"]
    reference = np.column_stack([estimates[:, 1:len(strategies)], np.nanmean(estimates[:, len(strategies):], axis=1)])
    difference = estimates[:, 0, None] - reference
    points = observed[0] - np.r_[observed[1:len(strategies)], np.nanmean(observed[len(strategies):])]
    lo, hi = np.nanquantile(difference, [.025, .975], axis=0)
    return pd.DataFrame({"comparison": names, "cnn_2d_sharpe_difference": points, "ci_2.5%": lo, "ci_97.5%": hi})


def random_controls(prices, events, cfg, spec, start, end, matched_count, development_rate):
    rng = np.random.default_rng(cfg["seed"])
    samples = []
    matched_returns = []
    for draw in range(spec["random_draws"]):
        fixed = rng.random(len(events)) < development_rate
        matched = np.zeros(len(events), dtype=bool)
        matched[rng.choice(len(events), size=matched_count, replace=False)] = True
        for name, mask in [("random_development_rate", fixed), ("random_matched", matched)]:
            curve, metrics = simulate(prices, events, cfg, start, end, mask)
            samples.append({"strategy": name, "draw": draw, **metrics})
            if name == "random_matched":
                matched_returns.append(curve.daily_return.to_numpy())
    return pd.DataFrame(samples), matched_returns


def factor_estimates(root, curves):
    import statsmodels.api as sm

    path = root / "data/raw/F-F_Research_Data_Factors_weekly.csv"
    if not path.exists():
        return pd.DataFrame(columns=["strategy", "weeks", "term", "estimate", "hac_se"])
    factors = read_factors(path)
    records = []
    for key in ["unfiltered", "cnn_1d", "cnn_2d"]:
        joined = weekly_coverage(curves[key], factors)
        y = joined.portfolio_return - joined.RF
        x = sm.add_constant(joined[["Mkt-RF", "SMB", "HML"]])
        fitted = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        for term, estimate in fitted.params.items():
            records.append({"strategy": key, "weeks": len(joined), "term": term, "estimate": estimate, "hac_se": fitted.bse[term]})
    return pd.DataFrame(records)


def evaluate(root, prices, calendar, vix, cfg, spec, manifest):
    torch.set_num_threads(2)
    out = root / "output/cnn"
    frozen_path = out / "frozen_spec.json"
    if not frozen_path.exists():
        raise ValueError("Run cnn-develop to freeze the configuration before cnn-test.")
    locked = json.loads(frozen_path.read_text())
    if locked["provenance"] != provenance(cfg, spec, manifest):
        raise ValueError("The data, code, configuration or package versions changed since development.")
    start, end = cfg["test_start"], cfg["end_exclusive"]
    # Build pre-test labels separately so final training never resolves a label with holdout prices.
    train_events, train_windows, _, _ = event_inputs(prices, calendar, vix, cfg, spec, start)
    test_events, test_windows, returns, peers = event_inputs(prices, calendar, vix, cfg, spec, end, start)
    all_events = pd.concat([train_events, test_events], ignore_index=True)
    windows = np.concatenate([train_windows, test_windows])
    training, _ = purged_train(train_events, returns.index, start, cfg["boundary_gap_sessions"])
    holdout = np.arange(len(train_events), len(all_events))
    scored = test_events[COLUMNS].copy()
    for model, predictions in predict_fold(all_events, windows, returns, peers, training, holdout, cfg, spec).items():
        scored[model] = predictions
    scored.to_csv(out / "test_predictions.csv", index=False)
    forecasts = score_forecasts(scored)
    forecasts.to_csv(out / "test_forecast_metrics.csv", index=False)
    save_calibration(scored, out / "test_calibration.csv")
    masks = {model: scored[model].ge(locked["probability_thresholds"][model]) for model in MODELS}
    masks["unfiltered"] = pd.Series(True, index=scored.index)
    masks["vix_filter"] = scored.vix_lag1.le(locked["vix_cutoff"])
    curves, rows = {}, []
    for strategy, mask in masks.items():
        curves[strategy], metrics = simulate(prices, test_events, cfg, start, end, mask)
        rows.append({"strategy": strategy, **metrics})
    for strategy, tickers in [("spy_buy_hold", [cfg["market"]]), ("sector_buy_hold", cfg["sectors"])]:
        curves[strategy] = buy_and_hold(prices, tickers, start, end, cfg["cost_bps_per_side"])
        rows.append({"strategy": strategy, **performance(curves[strategy])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_portfolio_metrics.csv", index=False)
    pd.DataFrame({k: v.daily_return for k, v in curves.items()}).to_csv(out / "test_daily_returns.csv")
    print(f"Scored {len(scored):,} holdout events. Running random controls.", flush=True)
    random, random_returns = random_controls(
        prices, test_events, cfg, spec, start, end,
        int(masks["cnn_2d"].sum()), locked["development_acceptance"]["cnn_2d"],
    )
    random.to_csv(out / "test_random_draws.csv", index=False)
    controls = random.groupby("strategy", as_index=False).mean(numeric_only=True).drop(columns="draw")
    controls.to_csv(out / "test_random_summary.csv", index=False)
    intervals = bootstrap_sharpe(curves, random_returns, spec, cfg["seed"] + 1)
    intervals.to_csv(out / "test_sharpe_intervals.csv", index=False)
    rows = []
    for bps in cfg["cost_sensitivity_bps"]:
        for strategy in ["unfiltered", "xgboost", "cnn_1d", "cnn_2d"]:
            _, metrics = simulate(prices, test_events, cfg, start, end, masks[strategy], bps)
            rows.append({"cost_bps_per_side": bps, "strategy": strategy, **metrics})
    costs = pd.DataFrame(rows)
    costs.to_csv(out / "test_cost_sensitivity.csv", index=False)
    factors = factor_estimates(root, curves)
    factors.to_csv(out / "test_factor_attribution.csv", index=False)
    write_report(root, out, locked, len(training), forecasts, table, controls, intervals, costs, factors)
    print(f"Final report: {root / 'reports/CNN_Research_Report.md'}", flush=True)


def write_report(root, out, locked, n_train, forecast, portfolio, random, intervals, costs, factors):
    dev = pd.read_csv(out / "development_forecast_metrics.csv")
    dev_trading = pd.read_csv(out / "development_portfolio_metrics.csv")
    published = root / "reports" / "cnn"
    published.mkdir(parents=True, exist_ok=True)
    files = [
        "development_forecast_metrics.csv", "development_portfolio_metrics.csv", "development_calibration.csv",
        "development_vix_cutoffs.csv", "test_forecast_metrics.csv", "test_portfolio_metrics.csv",
        "test_calibration.csv", "test_random_summary.csv", "test_sharpe_intervals.csv",
        "test_cost_sensitivity.csv", "test_factor_attribution.csv", "frozen_spec.json",
    ]
    for name in files:
        copyfile(out / name, published / name)
    trade_cols = ["strategy", "candidate_coverage", "annualized_return", "sharpe", "max_drawdown", "mean_exposure", "trades"]
    val = dev.loc[dev.period == "pooled"].set_index("model")
    test = forecast.set_index("model")
    gap = test.at["cnn_2d", "log_loss"] - test.at["cnn_1d", "log_loss"]
    table = portfolio.set_index("strategy")
    sharpe_gap = table.at["cnn_2d", "sharpe"] - table.at["cnn_1d", "sharpe"]
    matched = random.set_index("strategy").loc["random_matched"]
    winner = test.log_loss.idxmin()
    best = val.log_loss.idxmin()
    matched_ci = intervals.set_index("comparison").loc["random_matched"]
    holdout_hard = "The interval spans zero." if matched_ci["ci_2.5%"] <= 0 <= matched_ci["ci_97.5%"] else "The interval excludes zero."
    date = datetime.now(ZoneInfo("America/Detroit")).date()
    text = f"""# Cross-Sector Return Maps for ETF Trade Selection

**Math 628 research report** · {date}

The lowest holdout log loss came from **{winner}**. The 2D CNN's log loss minus the 1D CNN's was **{gap:+.4f}**, and its net Sharpe minus the 1D CNN's was **{sharpe_gap:+.3f}**. The estimated Sharpe difference versus randomly accepting the same number of opportunities was **{matched_ci['cnn_2d_sharpe_difference']:+.3f}**. {holdout_hard}

## Method

Nine sector ETFs generated momentum opportunities under the Stage 1 trading rules. At each signal close, the recent target and peer-basket returns were supplied directly to a 1D CNN or transformed into absolute-difference and signed-product maps for a 2D CNN. Both CNNs and the two tabular baselines share the market, instrument and return-summary features. Each annual fit uses information available in its training set.

![Market and return-map input illustration](assets/cnn_input_examples.png)

See the [research plan](../docs/cnn_research_plan.md) for representation, chronology and references. The fixed-development 30% target defines each probability cutoff. A VIX cutoff of **{locked['vix_cutoff']:g}** was chosen on development net Sharpe. All scores use the same candidate set within each period. Three fixed seeds are averaged for each CNN.

## Development, 2015–2021

The best pooled development log loss came from **{best}**. Development thresholds were selected using these out-of-fold predictions. Development performance therefore describes the selection sample.

{markdown_table(dev.loc[dev.period == 'pooled'])}

{markdown_table(dev_trading[trade_cols])}

## Frozen holdout, 2022–2024

The final estimators trained on **{n_train:,}** eligible pre-2022 events. Each model's threshold is fixed, so realized test-period acceptance can differ across models.

{markdown_table(forecast)}

{markdown_table(portfolio[trade_cols])}

{markdown_table(random[trade_cols])}

The random controls show means across repeated allocations. The matched control accepts exactly the 2D CNN's number of opportunities and is a retrospective benchmark. The development-rate control uses the 2D CNN's frozen development acceptance probability. Sleeve conflicts can produce different realized trade counts and exposure even when acceptance counts match. The 2D CNN's mean capital exposure was {table.at['cnn_2d', 'mean_exposure']:.1%}, compared with {matched.mean_exposure:.1%} for the matched controls.

## Paired Sharpe uncertainty

The comparison is 2D CNN minus the specified benchmark. Each draw resamples aligned calendar blocks of {locked['provenance']['cnn_config']['bootstrap_block_sessions']} sessions. The random comparison takes the mean of the random portfolios' Sharpes on each resample. The intervals condition on the fitted models and random allocations.

{markdown_table(intervals)}

## Fee sensitivity

The following table uses the same trades and thresholds at each fee level.

{markdown_table(costs[['strategy', 'cost_bps_per_side', 'annualized_return', 'sharpe']])}

## Factor attribution and limitations

{markdown_table(factors) if not factors.empty else "The course factor file was unavailable for this run."}

Factor estimates are weekly and use heteroskedasticity- and autocorrelation-consistent standard errors with four lags. Nearby signal windows overlap and same-day ETFs share market movements. The holdout spans one three-year interval. The return maps remove the simultaneous sign of both original sequences. This comparison consequently tests the entire map representation plus its CNN, including the information discarded by the encoding.

The [frozen configuration](cnn/frozen_spec.json), [yearly metrics](cnn/development_forecast_metrics.csv), [holdout calibration](cnn/test_calibration.csv), and remaining summary tables in [reports/cnn](cnn/) accompany this report. Reproduction commands are in the [README](../README.md).
"""
    (root / "reports/CNN_Research_Report.md").write_text(text, encoding="utf-8")
