"""CUSUM candidates and one shared close-monitor/next-open trade engine."""

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "ticker", "signal_date", "entry_date", "exit_date", "entry_price",
                 "exit_price", "barrier", "barrier_width", "gross_return", "net_return", "label", "holding_sessions"]


def cusum_events(log_returns, thresholds):
    positive = negative = 0.0
    dates = []
    for date, value in log_returns.items():
        threshold = thresholds.loc[date]
        if not np.isfinite(value) or not np.isfinite(threshold) or threshold <= 0:
            positive = negative = 0.0
            continue
        positive, negative = max(0, positive + value), min(0, negative + value)
        if negative < -threshold:
            negative = 0.0
            dates.append(date)
        elif positive > threshold:
            positive = 0.0
            dates.append(date)
    return pd.DatetimeIndex(dates)


def net_trade_return(entry, exit_price, cost_bps):
    cost = cost_bps / 10_000
    return exit_price * (1 - cost) / (entry * (1 + cost)) - 1


def simulate_trade(prices, signal_date, sigma, holding_sessions=10, cost_bps=5):
    """Entry t+1 open; scheduled time exit at entry+10 open.

    Monitor closes entry ... entry+9. A hit exits at the following open;
    at the scheduled time exit a coincident close hit retains its hit category.
    A missing observable exit returns None, never a fabricated class.
    """
    i = prices.index.get_loc(signal_date)
    entry_i, time_i = i + 1, i + 1 + holding_sessions
    if entry_i >= len(prices) or not np.isfinite(sigma) or sigma <= 0:
        return None
    entry = float(prices.adj_open.iloc[entry_i])
    width = sigma * np.sqrt(holding_sessions)
    exit_i, barrier = time_i, "time"
    for j in range(entry_i, min(time_i, len(prices))):
        change = prices.adj_close.iloc[j] / entry - 1
        if change >= width or change <= -width:
            exit_i, barrier = j + 1, "profit" if change >= width else "loss"
            break
    if exit_i >= len(prices):
        return None
    exit_price = float(prices.adj_open.iloc[exit_i])
    net = net_trade_return(entry, exit_price, cost_bps)
    return dict(signal_date=signal_date, entry_date=prices.index[entry_i], exit_date=prices.index[exit_i],
                entry_price=entry, exit_price=exit_price, barrier=barrier, barrier_width=width,
                gross_return=exit_price / entry - 1, net_return=net, label=int(net > 0),
                holding_sessions=exit_i - entry_i)


def build_events(prices, features, cfg):
    records, rejected = [], []
    for ticker in cfg["sectors"]:
        frame = features.loc[features.ticker == ticker].set_index("Date")
        dates = cusum_events(frame.log_return, frame.cusum_threshold)
        for date in dates:
            if date < pd.Timestamp(cfg["sample_start"]) or not frame.at[date, "eligible"]:
                continue
            trade = simulate_trade(prices[ticker], date, frame.at[date, "volatility"],
                                   cfg["holding_sessions"], cfg["cost_bps_per_side"])
            if trade is None:
                rejected.append(dict(ticker=ticker, signal_date=date, reason="no_observable_exit"))
            else:
                trade.update(ticker=ticker, event_id=f"{ticker}_{date:%Y%m%d}")
                records.append(trade)
    events = pd.DataFrame(records, columns=EVENT_COLUMNS).sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    excluded = pd.DataFrame(rejected, columns=["ticker", "signal_date", "reason"])
    return events, excluded
