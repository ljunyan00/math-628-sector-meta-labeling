"""Audit the frozen experiment and summarize retrospective checks without refitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import performance, run_portfolio
from .data import audit_and_load
from .sequence_data import event_inputs
from .splits import make_splits

MODELS = ["logistic", "xgboost", "cnn_1d", "cnn_2d"]


def binary_loss(labels, probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    labels = np.asarray(labels)
    return -(labels * np.log(p) + (1 - labels) * np.log1p(-p))


def check_scores(observed, expected):
    keys = ["event_id", "ticker", "signal_date", "exit_date", "label"]
    pd.testing.assert_frame_equal(observed[keys].reset_index(drop=True), expected[keys].reset_index(drop=True))
    values = observed[MODELS].to_numpy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Stored scores must be probabilities.")


def log_loss_intervals(predictions, calendar, baseline_p, spec, seed):
    comparisons = {"cnn_1d": predictions.cnn_1d, "xgboost": predictions.xgboost, "training_prior": baseline_p}
    actual = binary_loss(predictions.label, predictions.cnn_2d)
    daily = pd.DataFrame({key: actual - binary_loss(predictions.label, p) for key, p in comparisons.items()})
    daily["count"] = 1
    daily = daily.groupby(predictions.signal_date).sum().reindex(calendar, fill_value=0)
    block = spec["bootstrap_block_sessions"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(calendar) - block + 1, size=(spec["bootstrap_draws"], int(np.ceil(len(calendar) / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, :len(calendar)]
    sums = daily.to_numpy()[indices].sum(axis=1)
    sums = sums[sums[:, -1] > 0]
    draws = sums[:, :-1] / sums[:, -1, None]
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return pd.DataFrame({
        "comparison": list(comparisons), "cnn_2d_minus_reference": daily.drop(columns="count").sum().to_numpy() / len(predictions),
        "lower_95": low, "upper_95": high,
    })


def audit(root):
    out = root / "output/cnn"
    locked = json.loads((out / "frozen_spec.json").read_text())
    provenance = locked["provenance"]
    cfg, spec = provenance["stage1_config"], provenance["cnn_config"]
    for config_path, saved in [("config/stage1.json", cfg), ("config/cnn.json", spec)]:
        if json.loads((root / config_path).read_text()) != saved:
            raise ValueError(f"Current configuration differs from frozen configuration: {config_path}")
    for name, saved in provenance["code_sha256"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != saved:
            raise ValueError(f"Frozen research implementation changed: {name}")
    prices, calendar, vix, _, manifest = audit_and_load(root, cfg)
    if {name: item["sha256"] for name, item in manifest["files"].items()} != provenance["input_sha256"]:
        raise ValueError("The raw data vintage differs from the frozen experiment.")
    start, end = cfg["test_start"], cfg["end_exclusive"]
    development, _, returns, _ = event_inputs(prices, calendar, vix, cfg, spec, start)
    holdout, _, _, _ = event_inputs(prices, calendar, vix, cfg, spec, end, start)
    splits, _ = make_splits(development, returns.index, cfg)
    dev_pred = pd.read_csv(out / "development_predictions.csv", parse_dates=["signal_date", "exit_date"])
    test_pred = pd.read_csv(out / "test_predictions.csv", parse_dates=["signal_date", "exit_date"])
    expected = []
    prior = {}
    for year in cfg["validation_years"]:
        train = development.iloc[splits[f"train_{year}"]]
        fold = development.iloc[splits[f"validation_{year}"]]
        boundary = pd.Timestamp(year, 1, 1)
        if train.exit_date.max() >= boundary or fold.signal_date.min() < boundary or fold.exit_date.max() >= boundary + pd.DateOffset(years=1):
            raise ValueError(f"Unsafe date split for {year}.")
        expected.append(fold.assign(year=year))
        prior[year] = float(train.label.mean())
    check_scores(dev_pred, pd.concat(expected).sort_index())
    check_scores(test_pred, holdout)
    if dev_pred.event_id.duplicated().any() or test_pred.event_id.duplicated().any() or set(dev_pred.event_id) & set(test_pred.event_id):
        raise ValueError("Expected distinct development and holdout events.")
    if not dev_pred.year.equals(dev_pred.signal_date.dt.year.astype(dev_pred.year.dtype)):
        raise ValueError("Development forecast years do not match signal dates.")
    final_train = development.iloc[splits["final_train"]]
    if final_train.exit_date.max() >= pd.Timestamp(start):
        raise ValueError("A final-training label resolved inside the holdout.")
    baseline_p = float(final_train.label.mean())

    summaries = root / "reports/final"
    summaries.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for period, frame, p in [
        ("2015–2021", dev_pred, dev_pred.year.map(prior)), ("2022–2024", test_pred, baseline_p),
    ]:
        sample_rows.append(dict(period=period, events=len(frame), positive_rate=frame.label.mean(),
                                constant_log_loss=binary_loss(frame.label, p).mean()))
        forecast_table = pd.read_csv(out / ("test_forecast_metrics.csv" if period == "2022–2024" else "development_forecast_metrics.csv"))
        stored = forecast_table[forecast_table.period == "pooled"].set_index("model").log_loss
        for model in MODELS:
            np.testing.assert_allclose(binary_loss(frame.label, frame[model]).mean(), stored[model], atol=2e-7, rtol=0)
    pd.DataFrame(sample_rows).to_csv(summaries / "forecast_reference.csv", index=False, float_format="%.8f")
    holdout_dates = calendar[(calendar >= start) & (calendar < end)]
    log_loss_intervals(test_pred, holdout_dates, baseline_p, spec, cfg["seed"] + 2).to_csv(
        summaries / "paired_log_loss.csv", index=False, float_format="%.8f",
    )

    portfolio = pd.read_csv(out / "test_portfolio_metrics.csv").set_index("strategy")
    cost = cfg["cost_bps_per_side"]
    for model in MODELS:
        mask = test_pred[model].ge(locked["probability_thresholds"][model])
        curve, ledger = run_portfolio(prices, holdout, cfg["sectors"], start, end, cost, test_pred.loc[mask, "event_id"])
        realized = performance(curve, ledger)
        np.testing.assert_allclose([realized[x] for x in ["sharpe", "annualized_return", "mean_exposure", "trades"]],
                                   portfolio.loc[model, ["sharpe", "annualized_return", "mean_exposure", "trades"]].to_numpy(float),
                                   atol=1e-10, rtol=0)
        if int(mask.sum()) != int(portfolio.loc[model, "candidates_accepted"]):
            raise ValueError(f"Frozen cutoff acceptance mismatch for {model}.")
        np.testing.assert_allclose(ledger.net_return, ledger.event_id.map(holdout.set_index("event_id").net_return), atol=1e-12, rtol=0)

    count = int(portfolio.loc["cnn_2d", "candidates_accepted"])
    controls = pd.read_csv(out / "test_random_draws.csv")
    matched = controls.loc[controls.strategy == "random_matched"]
    if len(matched) != spec["random_draws"] or not matched.candidates_accepted.eq(count).all():
        raise ValueError("The random match did not preserve the CNN acceptance count.")
    rows = []
    for model in MODELS:
        # Sort by the saved score and then event ID to break bounded-precision ties.
        selected = test_pred.sort_values([model, "event_id"], ascending=[False, True]).head(count).event_id
        curve, ledger = run_portfolio(prices, holdout, cfg["sectors"], start, end, cost, selected)
        rows.append({"model": model, "accepted": count, "coverage": count / len(holdout), **performance(curve, ledger)})
    pd.DataFrame(rows).to_csv(summaries / "matched_candidate_diagnostic.csv", index=False, float_format="%.8f")
    print(f"Verified frozen input checksums and code, {len(dev_pred):,} development predictions, "
          f"{len(test_pred):,} holdout predictions, four portfolios, and {len(matched)} matched random draws.")
    print(f"Final review tables: {summaries}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    audit(parser.parse_args().root.resolve())
