import numpy as np
import pandas as pd
import pytest

from sector_meta.cnn_revision import historical_policy, scheduled_signal_cutoff, signed_maps
from sector_meta.sequence_data import return_maps


def test_signed_maps_recover_both_series_and_distinguish_joint_reversal():
    windows = np.array([[[1.0, -2.0, 0.5], [-3.0, 4.0, -0.5]]])
    np.testing.assert_array_equal(return_maps(windows), return_maps(-windows))
    signed = signed_maps(windows)
    np.testing.assert_allclose(signed_maps(-windows), -signed)
    common, spread = signed[0]
    np.testing.assert_allclose((common[:, 0] + spread[:, 0]) / np.sqrt(2), windows[0, 0])
    np.testing.assert_allclose((common[0] - spread[0]) / np.sqrt(2), windows[0, 1])


def test_last_eligible_signal_guarantees_next_open_exit_before_boundary():
    sessions = pd.bdate_range("2021-12-01", periods=32)
    boundary = sessions[24]
    cutoff = scheduled_signal_cutoff(sessions, boundary, holding_sessions=10)
    assert cutoff == sessions[13]
    assert sessions[sessions.get_loc(cutoff) - 1 + 1 + 10] < boundary
    assert sessions[sessions.get_loc(cutoff) + 1 + 10] == boundary


def test_annual_policy_falls_back_to_prior_and_refuses_unresolved_trades():
    history = pd.DataFrame({
        "signal_date": pd.to_datetime(["2018-12-04", "2018-12-05", "2018-12-06", "2018-12-07"]),
        "exit_date": pd.to_datetime(["2018-12-11", "2018-12-12", "2018-12-13", "2018-12-14"]),
        "label": [0, 1, 0, 1],
        "training_prior": [0.5] * 4,
        "cnn_2d_signed": [0.8, 0.2, 0.7, 0.3],
        "ridge_return": [0.01, -0.01, 0.02, -0.02],
    })
    models = ["cnn_2d_signed", "ridge_return"]
    policy = historical_policy(history, models, 0.5, [0.0, 0.5, 1.0], "2019-01-01")
    assert policy["cnn_2d_signed"] == {"cutoff": 0.7, "weight": 0.0}
    assert policy["ridge_return"] == {"cutoff": 0.01, "weight": None}
    with pytest.raises(ValueError, match="unavailable"):
        historical_policy(history, models, 0.5, [0.0, 1.0], "2018-12-14")
