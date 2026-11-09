"""Deterministic tests for execution accounting and information boundaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sector_meta.backtest import buy_and_hold, performance, run_portfolio
from sector_meta.data import read_factors, read_prices
from sector_meta.events import cusum_events, simulate_trade
from sector_meta.features import build_features
from sector_meta.splits import make_splits, purged_train


def path_frame(n=18):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"adj_open": 100.0, "adj_close": 100.0}, index=dates)


def test_close_hit_executes_at_next_open_with_gap_and_costs():
    p = path_frame()
    p.iloc[1, p.columns.get_loc("adj_close")] = 110
    p.iloc[2, p.columns.get_loc("adj_open")] = 95
    trade = simulate_trade(p, p.index[0], .01)
    assert trade["entry_date"] == p.index[1]
    assert trade["exit_date"] == p.index[2]
    assert trade["exit_price"] == 95
    assert trade["barrier"] == "profit"
    assert trade["label"] == 0
    assert trade["net_return"] == pytest.approx(95 * .9995 / (100 * 1.0005) - 1)


@pytest.mark.parametrize("exit_price, expected_label", [(101, 1), (99, 0), (100, 0)])
def test_time_exit_does_not_imply_negative_class(exit_price, expected_label):
    p = path_frame()
    p.iloc[11, p.columns.get_loc("adj_open")] = exit_price
    trade = simulate_trade(p, p.index[0], .03)
    assert trade["exit_date"] == p.index[11]
    assert trade["holding_sessions"] == 10
    assert trade["barrier"] == "time"
    assert trade["label"] == expected_label


def test_incomplete_exit_is_excluded_but_observable_early_exit_is_valid():
    p = path_frame(5)
    assert simulate_trade(p, p.index[0], .01) is None
    p.iloc[1, p.columns.get_loc("adj_close")] = 90
    assert simulate_trade(p, p.index[0], .01)["exit_date"] == p.index[2]
    assert simulate_trade(p, p.index[-1], .01) is None


def test_barrier_on_last_close_needs_observed_next_open():
    p = path_frame(5)
    p.iloc[-1, p.columns.get_loc("adj_close")] = 120
    assert simulate_trade(p, p.index[0], .01) is None


def test_cusum_uses_supplied_time_varying_thresholds():
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.Series([np.nan, .01, .01, -.05, .01], index=dates)
    thresholds = pd.Series([np.nan, .015, .015, .02, .02], index=dates)
    assert cusum_events(returns, thresholds).tolist() == [dates[2], dates[3]]


def test_portfolio_skips_signals_while_open_and_matches_label_engine():
    p = path_frame(30)
    p.adj_close *= np.linspace(1, 1.03, len(p))
    p.adj_open *= np.linspace(1, 1.03, len(p))
    rows = []
    for k in [0, 1, 11, 12]:
        row = simulate_trade(p, p.index[k], .1)
        rows.append(dict(row, ticker="A", event_id=f"A_{k}"))
    events = pd.DataFrame(rows)
    daily, trades = run_portfolio({"A": p}, events, ["A"], p.index[0], p.index[-1] + pd.Timedelta(days=1))
    assert trades.event_id.tolist() == ["A_0", "A_11"]
    expected = events.set_index("event_id").loc[trades.event_id, "net_return"]
    np.testing.assert_allclose(trades.net_return, expected, atol=1e-12)
    assert daily.active_sleeves.max() == 1
    assert daily.exposure.between(0, 1).all()
    assert daily.equity.iloc[-1] == pytest.approx(np.prod(1 + trades.net_return))


def test_initial_loss_is_part_of_drawdown_and_entry_fee_is_recorded():
    p = path_frame()
    trade = dict(simulate_trade(p, p.index[0], .03), ticker="A", event_id="A_0")
    daily, ledger = run_portfolio({"A": p}, pd.DataFrame([trade]), ["A"], p.index[0], p.index[-1] + pd.Timedelta(days=1))
    assert daily.loc[p.index[1], "daily_return"] < 0
    assert performance(daily, ledger)["max_drawdown"] == pytest.approx(.9995 / 1.0005 - 1)


def test_buy_and_hold_costs_and_terminal_turnover():
    p = path_frame(5)
    p.iloc[-1, p.columns.get_loc("adj_close")] = 110
    daily = buy_and_hold({"A": p}, ["A"], p.index[0], p.index[-1] + pd.Timedelta(days=1))
    assert daily.equity.iloc[-1] == pytest.approx(1.1 * .9995 / 1.0005)
    assert daily.turnover.iloc[0] == pytest.approx(1 / 1.0005)
    assert daily.turnover.iloc[-1] == pytest.approx(1.1)
    assert (1 + daily.daily_return).prod() == pytest.approx(daily.equity.iloc[-1])


@pytest.fixture
def feature_inputs():
    cfg = json.loads((Path(__file__).parents[1] / "config/stage1.json").read_text())
    dates = pd.bdate_range("2019-01-01", periods=230)
    rng = np.random.default_rng(628)
    prices = {}
    for t in cfg["sectors"] + ["SPY"]:
        close = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, len(dates))))
        prices[t] = pd.DataFrame({"adj_close": close, "Volume": rng.integers(1000, 2000, len(dates))}, index=dates)
    vix = pd.Series(np.linspace(12, 30, len(dates)), index=dates)
    return cfg, dates, prices, vix


def test_vix_is_lagged_and_volatility_threshold_uses_prior_session(feature_inputs):
    cfg, dates, prices, vix = feature_inputs
    features = build_features(prices, dates, vix, cfg)
    f = features[features.ticker == "XLB"].set_index("Date")
    assert f.at[dates[120], "vix_lag1"] == vix.iloc[119]
    assert f.at[dates[120], "vix_source_date"] == dates[119]
    assert f.at[dates[120], "cusum_threshold"] == pytest.approx(2 * f.at[dates[119], "volatility"])


def test_future_price_and_vix_changes_do_not_change_past_features(feature_inputs):
    cfg, dates, prices, vix = feature_inputs
    before = build_features(prices, dates, vix, cfg)
    cutoff = dates[150]
    changed = {t: p.copy() for t, p in prices.items()}
    for p in changed.values():
        p.loc[p.index > cutoff, "adj_close"] *= 3
        p.loc[p.index > cutoff, "Volume"] *= 10
    vix_changed = vix.copy()
    vix_changed.loc[vix_changed.index > cutoff] *= 5
    after = build_features(changed, dates, vix_changed, cfg)
    pd.testing.assert_frame_equal(before[before.Date <= cutoff], after[after.Date <= cutoff])


def test_vix_missing_day_forward_fill_still_lags_one_session(feature_inputs):
    cfg, dates, prices, vix = feature_inputs
    vix.iloc[119] = np.nan
    f = build_features(prices, dates, vix, cfg).query("ticker == 'XLB'").set_index("Date")
    assert f.at[dates[120], "vix_lag1"] == vix.iloc[118]
    assert f.at[dates[120], "vix_source_date"] == dates[118]


def test_purge_checks_both_signal_gap_and_known_exit():
    dates = pd.bdate_range("2020-01-01", periods=60)
    events = pd.DataFrame({"signal_date": [dates[0], dates[20], dates[5]],
                           "exit_date": [dates[10], dates[25], dates[45]]})
    indices, cutoff = purged_train(events, dates, dates[30], 11)
    assert cutoff == dates[19]
    assert indices.tolist() == [0]


def test_validation_labels_cannot_cross_into_test():
    dates = pd.bdate_range("2020-01-01", "2022-02-01")
    events = pd.DataFrame({"signal_date": pd.to_datetime(["2020-01-03", "2021-12-29"]),
                           "exit_date": pd.to_datetime(["2020-01-15", "2022-01-05"])})
    cfg = {"validation_years": [2021], "test_start": "2022-01-01", "boundary_gap_sessions": 11}
    result, _ = make_splits(events, dates, cfg)
    assert result["validation_2021"].size == 0
    assert result["final_train"].tolist() == [0]


def test_prices_use_same_adjustment_for_open_and_close(tmp_path):
    p = pd.DataFrame({"Date": ["2020-01-01"], "Open": [100], "High": [105], "Low": [95], "Close": [102], "Adj Close": [51]})
    p.to_csv(tmp_path / "A.csv", index=False)
    result = read_prices(tmp_path, "A")
    assert result.adj_open.iloc[0] == 50
    assert result.adj_close.iloc[0] == 51


def test_factors_keep_actual_endpoint_and_convert_percentage_units(tmp_path):
    path = tmp_path / "factors.csv"
    path.write_text("Vintage note\n,Mkt-RF,SMB,HML,RF\n20241227,1.6,-.62,-.83,.056\n")
    result = read_factors(path)
    assert result.index[0] == pd.Timestamp("2024-12-27")
    assert result.iloc[0]["Mkt-RF"] == pytest.approx(.016)
    assert result.iloc[0].RF == pytest.approx(.00056)
