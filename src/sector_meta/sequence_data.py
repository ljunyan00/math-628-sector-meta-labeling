"""Causal return windows and numeric two-dimensional encodings."""

import numpy as np
import pandas as pd

from .events import build_events
from .features import FEATURE_SETS, build_features

SUMMARY_COLUMNS = [
    "window_pair_correlation", "window_relative_return", "window_relative_volatility",
    "window_same_direction", "sector_collectivity_60",
]
CONTEXT_COLUMNS = FEATURE_SETS["full"] + SUMMARY_COLUMNS


def daily_pairs(prices, sectors):
    returns = pd.DataFrame({ticker: prices[ticker].adj_close for ticker in sectors}).pct_change()
    peers = returns.rsub(returns.sum(axis=1), axis=0).div(len(sectors) - 1)
    peers.loc[returns.isna().any(axis=1)] = np.nan
    return returns, peers


def paired_windows(events, returns, peers, length):
    positions = returns.index.get_indexer(events.signal_date)
    tickers = returns.columns.get_indexer(events.ticker)
    if np.any(positions < length - 1) or np.any(tickers < 0):
        raise ValueError("Every event must have a known ticker and sufficient preceding sessions.")
    days = positions[:, None] + np.arange(1 - length, 1)
    windows = np.stack([
        returns.to_numpy()[days, tickers[:, None]],
        peers.to_numpy()[days, tickers[:, None]],
    ], axis=1)
    if not np.isfinite(windows).all():
        raise ValueError("Event return windows contain missing prices.")
    return windows


def fit_pair_scaling(returns, peers, start, last_train_signal):
    periods = slice(start, last_train_signal)
    history = np.stack([returns.loc[periods].to_numpy(), peers.loc[periods].to_numpy()], axis=1)
    means, scales = np.nanmean(history, axis=0), np.nanstd(history, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError("Both series need a nonzero training standard deviation for every ETF.")
    return means, scales


def standardize(windows, ticker_positions, scaling, clip):
    means, scales = scaling
    centered = windows - means.T[ticker_positions, :, None]
    return np.clip(centered / scales.T[ticker_positions, :, None], -clip, clip)


def return_maps(windows):
    target = windows[:, 0, :, None]
    peer = windows[:, 1, None, :]
    return np.stack([np.abs(target - peer), target * peer], axis=1)


def event_inputs(prices, calendar, vix, cfg, spec, stop, start=None):
    """Build labels and predictors from data strictly before the supplied end date."""
    end = pd.Timestamp(stop)
    calendar = calendar[calendar < end]
    limited_prices = {ticker: frame.loc[calendar] for ticker, frame in prices.items()}
    vix = vix.loc[vix.index < end]
    features = build_features(limited_prices, calendar, vix, cfg)
    events, _ = build_events(limited_prices, features, cfg)
    if start is not None:
        events = events.loc[events.signal_date >= pd.Timestamp(start)].reset_index(drop=True)
    returns, peers = daily_pairs(limited_prices, cfg["sectors"])
    windows = paired_windows(events, returns, peers, spec["window_sessions"])
    predictors = features.rename(columns={"Date": "signal_date"})
    events = events.merge(predictors, on=["signal_date", "ticker"], validate="one_to_one")
    target, peer = windows[:, 0, :], windows[:, 1, :]
    x, y = target - target.mean(axis=1, keepdims=True), peer - peer.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    events["window_pair_correlation"] = np.divide(
        (x * y).sum(axis=1), denominator, out=np.zeros(len(events)), where=denominator > 0,
    )
    events["window_relative_return"] = (1 + target).prod(axis=1) - (1 + peer).prod(axis=1)
    events["window_relative_volatility"] = (target - peer).std(axis=1)
    events["window_same_direction"] = ((target * peer) > 0).mean(axis=1)
    correlations = returns.rolling(60).corr().to_numpy().reshape(len(calendar), len(returns.columns), -1)
    indices = calendar.get_indexer(events.signal_date)
    matrices = correlations[indices]
    if not np.isfinite(matrices).all():
        raise ValueError("Correlation features require a complete 60-session warm-up.")
    events["sector_collectivity_60"] = np.linalg.eigvalsh(matrices)[:, -1] / len(returns.columns)
    return events, windows, returns, peers
