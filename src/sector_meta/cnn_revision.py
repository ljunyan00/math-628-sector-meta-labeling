"""Run development-only ablations and fit decision policies on earlier forecasts."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import performance, run_portfolio
from .final_review import binary_loss, check_scores
from .sequence_data import event_inputs, fit_pair_scaling, standardize
from .splits import make_splits

BASE_MODELS = ("logistic", "xgboost", "cnn_1d", "cnn_2d")
PROBABILITY_MODELS = (*BASE_MODELS, "cnn_2d_signed")
RANKED_MODELS = (*PROBABILITY_MODELS, "ridge_return")


def signed_maps(windows):
    """Encode signed common movement and spread without dropping either series."""
    target = windows[:, 0, :, None]
    peer = windows[:, 1, None, :]
    return np.stack([target + peer, target - peer], axis=1) / np.sqrt(2)


def scheduled_signal_cutoff(calendar, boundary, holding_sessions):
    # A signal at index t enters t+1 and can exit as late as t+1+holding_sessions.
    boundary_index = calendar.searchsorted(pd.Timestamp(boundary))
    last_entry_gap = holding_sessions + 1
    if boundary_index < last_entry_gap:
        raise ValueError("The calendar needs enough history for the full trade horizon.")
    return calendar[boundary_index - last_entry_gap]


def historical_policy(history, models, target_coverage, weights, boundary):
    """Fit cutoffs and probability shrinkage only to completed past forecasts."""
    if history.empty or not 0 < target_coverage < 1:
        raise ValueError("Policy fitting requires earlier forecasts and coverage in (0, 1).")
    boundary = pd.Timestamp(boundary)
    if not (history.exit_date < boundary).all() or not (history.signal_date < boundary).all():
        raise ValueError("Policy history contains unavailable outcomes or forecasts.")
    rules = {}
    boundary_rank = int(np.ceil(target_coverage * len(history)))
    for model in models:
        scores = history[model].to_numpy()
        cutoff = float(np.sort(scores)[-boundary_rank])
        weight = None
        if model in PROBABILITY_MODELS:
            losses = [
                binary_loss(history.label, w * scores + (1 - w) * history.training_prior).mean()
                for w in weights
            ]
            weight = float(weights[int(np.argmin(losses))])
        rules[model] = {"cutoff": cutoff, "weight": weight}
    return rules


def predict_new_models(events, windows, returns, peers, train, validation, cfg, spec, revision):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from .cnn_models import cnn_predict, context_array

    last_train = events.signal_date.iloc[train].max()
    joined = np.r_[train, validation]
    scaling = fit_pair_scaling(returns, peers, cfg["sample_start"], last_train)
    ticker_positions = returns.columns.get_indexer(events.ticker.iloc[joined])
    maps = signed_maps(standardize(windows[joined], ticker_positions, scaling, spec["standardized_return_clip"]))
    context = context_array(events.iloc[joined], cfg["sectors"])
    split = len(train)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(context[:split])
    context = imputer.transform(context)
    context = StandardScaler().fit(context[:split]).transform(context)
    probabilities = cnn_predict(maps[:split], context[:split], events.label.iloc[train].to_numpy(),
                                maps[split:], context[split:], 2, spec)
    ridge = Ridge(alpha=revision["return_ridge_alpha"]).fit(context[:split], events.net_return.iloc[train])
    return probabilities, ridge.predict(context[split:])


def forecast_table(scored):
    rows = []
    for model in ("training_prior", *PROBABILITY_MODELS):
        versions = [("raw", model)]
        if model != "training_prior":
            versions.append(("shrunk", f"{model}_shrunk"))
        for method, column in versions:
            prediction = scored[column]
            rows.append({
                "model": model, "probabilities": method, "events": len(scored),
                "log_loss": binary_loss(scored.label, prediction).mean(),
                "brier": ((scored.label - prediction) ** 2).mean(),
            })
    return pd.DataFrame(rows)


def portfolio_table(prices, events, scored, cfg, spec):
    start = f"{int(scored.year.min())}-01-01"
    end = cfg["development_end_exclusive"]
    universe = events.loc[events.event_id.isin(scored.event_id)]
    selections = {"unfiltered": scored.event_id}
    matched_by_model = {}
    for model in RANKED_MODELS:
        selections[model] = scored.loc[scored[f"{model}_accepted"], "event_id"]
        matched = []
        for _, year in scored.groupby("year", sort=True):
            count = int(np.ceil(len(year) * spec["target_candidate_coverage"]))
            matched.append(year.sort_values([model, "event_id"], ascending=[False, True]).head(count).event_id)
        matched_by_model[model] = pd.concat(matched)
    rows = []
    for policy, group in [("historical_cutoff", selections), ("matched_retrospective", matched_by_model)]:
        for model, ids in group.items():
            daily, trades = run_portfolio(prices, universe, cfg["sectors"], start, end, cfg["cost_bps_per_side"], ids)
            rows.append({"policy": policy, "model": model, "accepted": len(ids), "coverage": len(ids) / len(scored),
                         **performance(daily, trades)})
    return pd.DataFrame(rows)


def run_revision(root, prices, calendar, vix, cfg, spec, manifest):
    import torch

    torch.set_num_threads(2)
    out = root / "output/cnn_revision"
    published = root / "reports/revision"
    out.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    revision = json.loads((root / "config/cnn_revision.json").read_text())
    weights = revision["probability_model_weights"]
    if sorted(set(weights)) != weights or not weights or weights[0] != 0 or weights[-1] > 1:
        raise ValueError("The shrinkage grid must increase from zero within [0, 1].")
    if revision["return_ridge_alpha"] <= 0:
        raise ValueError("Return ridge regularization must be positive.")

    prior_run = root / "output/cnn"
    frozen = json.loads((prior_run / "frozen_spec.json").read_text())["provenance"]
    hashes = {name: item["sha256"] for name, item in manifest["files"].items()}
    if (cfg != frozen["stage1_config"] or spec != frozen["cnn_config"] or hashes != frozen["input_sha256"]):
        raise ValueError("Run cnn-develop with the same configuration and data before cnn-revise.")
    for name, digest in frozen["code_sha256"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The original experiment code differs from the baseline run: {name}.")

    end = cfg["development_end_exclusive"]
    events, windows, returns, peers = event_inputs(prices, calendar, vix, cfg, spec, end)
    splits, _ = make_splits(events, returns.index, cfg)
    years = cfg["validation_years"]
    if len(years) < 2 or years != sorted(set(years)):
        raise ValueError("The revision requires ordered development folds and one warm-up fold.")
    expected = pd.concat([events.iloc[splits[f"validation_{year}"]].assign(year=year) for year in years]).sort_index()
    baseline_path = prior_run / "development_predictions.csv"
    baseline = pd.read_csv(baseline_path, parse_dates=["signal_date", "exit_date"])
    check_scores(baseline, expected)
    if not baseline.year.equals(baseline.signal_date.dt.year.astype(baseline.year.dtype)):
        raise ValueError("The baseline years do not match signal dates.")
    baseline = baseline.set_index("event_id")
    chunks = []
    for year in years:
        boundary = pd.Timestamp(year, 1, 1)
        next_year = boundary + pd.DateOffset(years=1)
        train = splits[f"train_{year}"]
        validation = splits[f"validation_{year}"]
        cutoff = scheduled_signal_cutoff(returns.index, next_year, cfg["holding_sessions"])
        validation = validation[events.signal_date.iloc[validation].to_numpy() < cutoff.to_datetime64()]
        if len(train) == 0 or len(validation) == 0 or events.exit_date.iloc[train].max() >= boundary:
            raise ValueError(f"Invalid training or validation split for {year}.")
        sample = events.iloc[validation][["event_id", "signal_date", "exit_date", "label", "net_return"]].copy()
        sample["year"] = year
        sample["training_prior"] = float(events.label.iloc[train].mean())
        for model in BASE_MODELS:
            sample[model] = baseline.loc[sample.event_id, model].to_numpy()
        sample["cnn_2d_signed"], sample["ridge_return"] = predict_new_models(
            events, windows, returns, peers, train, validation, cfg, spec, revision,
        )
        chunks.append(sample)
        print(f"{year}: scored {len(sample)} development events", flush=True)

    previous = chunks[0]
    scored_chunks, policy_rows = [], []
    for year, sample in zip(years[1:], chunks[1:]):
        rules = historical_policy(previous, RANKED_MODELS, spec["target_candidate_coverage"], weights, f"{year}-01-01")
        scored = sample.copy()
        for model, rule in rules.items():
            scored[f"{model}_accepted"] = sample[model].ge(rule["cutoff"])
            w = rule["weight"]
            if w is not None:
                scored[f"{model}_shrunk"] = w * sample[model] + (1 - w) * sample.training_prior
            policy_rows.append({"year": year, "model": model, "history_events": len(previous),
                                "history_last_exit": previous.exit_date.max(), "fitted_cutoff": rule["cutoff"],
                                "model_weight": w, "realized_coverage": scored[f"{model}_accepted"].mean()})
        scored_chunks.append(scored)
        previous = pd.concat([previous, sample])
    scored = pd.concat(scored_chunks)
    scored.to_csv(out / "development_predictions.csv", index=False)
    tables = {
        "forecast_metrics.csv": forecast_table(scored),
        "portfolio_metrics.csv": portfolio_table(prices, events, scored, cfg, spec),
        "yearly_policies.csv": pd.DataFrame(policy_rows),
    }
    for filename, table in tables.items():
        table.to_csv(published / filename, index=False, float_format="%.8f")
    reference = np.array([events.net_return.iloc[splits[f"train_{year}"]].mean() for year in years[1:]])
    stats = {
        "scored_events": len(scored),
        "scored_years": years[1:],
        "policy_warmup_year": years[0],
        "last_signal": scored.signal_date.max().date().isoformat(),
        "training_mean_return_rmse": float(np.sqrt(np.mean((scored.net_return - scored.year.map(dict(zip(years[1:], reference)))) ** 2))),
        "ridge_return_rmse": float(np.sqrt(np.mean((scored.net_return - scored.ridge_return) ** 2))),
        "baseline_development_predictions_sha256": sha256(baseline_path.read_bytes()).hexdigest(),
        "revision_config": revision,
    }
    (published / "run_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Completed {len(scored):,} development evaluations. Summary: {published}")
