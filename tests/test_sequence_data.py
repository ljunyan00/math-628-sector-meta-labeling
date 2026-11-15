import numpy as np
import pandas as pd

from sector_meta.sequence_data import daily_pairs, fit_pair_scaling, paired_windows, return_maps


def test_window_uses_the_signal_close_and_excludes_target_from_peer_basket():
    dates = pd.date_range("2014-01-01", periods=5)
    prices = {
        "A": pd.DataFrame({"adj_close": [100, 110, 110, 1000, 1]}, index=dates),
        "B": pd.DataFrame({"adj_close": [100, 120, 132, 1, 1000]}, index=dates),
        "C": pd.DataFrame({"adj_close": [100, 100, 120, 1, 1000]}, index=dates),
    }
    target, peer = daily_pairs(prices, ["A", "B", "C"])
    events = pd.DataFrame({"ticker": ["A"], "signal_date": [dates[2]]})
    windows = paired_windows(events, target, peer, length=2)
    np.testing.assert_allclose(windows, [[[0.1, 0], [0.10, 0.15]]], atol=1e-15)
    means, _ = fit_pair_scaling(target, peer, dates[0], dates[2])
    np.testing.assert_allclose(means[:, 0], [0.05, 0.125], atol=1e-15)


def test_map_axes_preserve_target_peer_order_and_signed_product():
    maps = return_maps(np.array([[[1.0, -2.0], [-3.0, 4.0]]]))
    np.testing.assert_array_equal(maps[0, 0], [[4, 3], [1, 6]])
    np.testing.assert_array_equal(maps[0, 1], [[-3, 4], [6, -8]])
