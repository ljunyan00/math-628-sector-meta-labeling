"""Reproducible Stage 1 artifacts and an English progress report."""

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import buy_and_hold, performance, run_portfolio
from .data import audit_and_load, read_factors, versions
from .events import build_events, net_trade_return
from .features import build_features, FEATURE_SETS
from .splits import make_splits


def markdown_table(frame, digits=4):
    def fmt(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def weekly_coverage(daily, factors):
    rows = []
    endpoints = factors.index
    for i in range(1, len(endpoints)):
        before, end = endpoints[i - 1], endpoints[i]
        if before in daily.index and end in daily.index:
            r = daily.loc[(daily.index > before) & (daily.index <= end), "daily_return"]
            rows.append(dict(Date=end, portfolio_return=float((1 + r).prod() - 1)))
    returns = pd.DataFrame(rows, columns=["Date", "portfolio_return"]).set_index("Date")
    return returns.join(factors, how="inner")


def run_stage1(root, cfg):
    root = Path(root)
    out, processed = root / "output/stage1", root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    prices, calendar, vix, audit, manifest = audit_and_load(root, cfg)
    audit.to_csv(out / "data_audit.csv", index=False)
    full_features = build_features(prices, calendar, vix, cfg)
    features = full_features[full_features.Date >= cfg["sample_start"]].copy()
    features.to_csv(processed / "daily_features.csv", index=False)
    # Stage 1 does not construct or inspect held-out profitability labels.
    dev_end = cfg["development_end_exclusive"]
    dev_prices = {t: p.loc[p.index < dev_end] for t, p in prices.items()}
    dev_features = features[features.Date < dev_end]
    # Carry warm-up CUSUM state continuously into the 2005 research period.
    events, excluded = build_events(dev_prices, full_features[full_features.Date < dev_end], cfg)
    events.to_csv(processed / "development_events.csv", index=False)
    excluded.to_csv(out / "incomplete_development_events.csv", index=False)
    samples = events.merge(dev_features, left_on=["signal_date", "ticker"], right_on=["Date", "ticker"], validate="many_to_one").drop(columns="Date")
    if samples.event_id.tolist() != events.event_id.tolist():
        raise AssertionError("Sample-table row ordering no longer matches split indices.")
    samples.to_csv(processed / "development_samples.csv", index=False)
    feature_cols = FEATURE_SETS["full"]
    missing_features = samples[feature_cols].isna().sum().rename("missing_event_values").rename_axis("feature").reset_index()
    missing_features.to_csv(out / "feature_audit.csv", index=False)
    splits, fold_summary = make_splits(events, calendar, cfg)
    # There are no final-test labels at this stage, hence no test event indices to export.
    splits.pop("final_test", None)
    np.savez_compressed(processed / "chronological_splits.npz", **splits)
    fold_summary.to_csv(out / "fold_summary.csv", index=False)
    json_config = json.dumps(cfg, sort_keys=True, indent=2) + "\n"
    (out / "frozen_stage1_config.json").write_text(json_config)
    dictionary = {"feature_sets": FEATURE_SETS, "categorical": ["ticker"],
                  "not_predictors": ["event_id", "signal_date", "entry_date", "exit_date", "entry_price", "exit_price", "label", "net_return", "gross_return", "barrier", "holding_sessions", "barrier_width", "eligible", "log_return", "cusum_threshold", "vix_source_date"],
                  "note": "Use ONLY the explicit feature whitelist. All preprocessing must be fit within training folds."}
    (processed / "feature_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n")

    daily, trades = run_portfolio(prices, events, cfg["sectors"], cfg["sample_start"], dev_end, cfg["cost_bps_per_side"])
    daily.to_csv(out / "unfiltered_momentum_daily.csv")
    trades.to_csv(out / "unfiltered_momentum_trades.csv", index=False)
    expected = trades.event_id.map(events.set_index("event_id").net_return)
    if not np.allclose(trades.net_return, expected, atol=1e-12):
        raise AssertionError("Label and portfolio execution returns disagree.")
    benchmark_daily = {"Unfiltered event momentum": daily,
                       "SPY buy-and-hold": buy_and_hold(prices, [cfg["market"]], cfg["sample_start"], dev_end, cfg["cost_bps_per_side"]),
                       "Sector basket buy-and-hold": buy_and_hold(prices, cfg["sectors"], cfg["sample_start"], dev_end, cfg["cost_bps_per_side"])}
    metrics = pd.DataFrame([dict(strategy=name, **performance(frame, trades if name == "Unfiltered event momentum" else None))
                            for name, frame in benchmark_daily.items()])
    metrics.to_csv(out / "development_baseline_metrics.csv", index=False)
    for name, frame in list(benchmark_daily.items())[1:]:
        filename = "spy" if name.startswith("SPY") else "sector_basket"
        frame.to_csv(out / f"{filename}_daily.csv")
    # Fixed opportunity/exit paths; only fees and the corresponding binary labels change.
    costs = []
    for bps in cfg["cost_sensitivity_bps"]:
        paths = events.copy()
        paths["net_return"] = net_trade_return(paths.entry_price, paths.exit_price, bps)
        paths["label"] = paths.net_return.gt(0).astype(int)
        curve, ledger = run_portfolio(prices, paths, cfg["sectors"], cfg["sample_start"], dev_end, bps)
        costs.append(dict(cost_bps_per_side=bps, event_positive_rate=paths.label.mean(), **performance(curve, ledger)))
    cost_metrics = pd.DataFrame(costs)
    cost_metrics.to_csv(out / "development_cost_sensitivity.csv", index=False)
    sector_summary = events.groupby("ticker").agg(events=("label", "size"), positive_rate=("label", "mean"),
                                                    mean_net_return=("net_return", "mean"), mean_holding_sessions=("holding_sessions", "mean")).reset_index()
    sector_summary.to_csv(out / "development_event_summary.csv", index=False)
    barrier_summary = events.groupby("barrier").agg(events=("label", "size"), positive_rate=("label", "mean")).reset_index()
    barrier_summary.to_csv(out / "development_barrier_summary.csv", index=False)
    factors_path = root / "data/raw/F-F_Research_Data_Factors_weekly.csv"
    factor_status = "The local weekly factor file was not supplied; attribution inputs remain pending."
    if factors_path.exists():
        factors = read_factors(factors_path)
        factors.to_csv(processed / "weekly_factors_decimal.csv")
        joined = weekly_coverage(daily, factors)
        joined.to_csv(out / "development_weekly_factor_alignment.csv")
        factor_status = (f"The fixed local factor vintage contains {len(factors):,} weekly rows, ending {factors.index.max():%Y-%m-%d}. "
                         f"Exact observed endpoint matching produces {len(joined):,} complete development weeks. "
                         "The first partial week is excluded. Percentages were divided by 100; no factor regression has been fitted.")
    create_figure(out, benchmark_daily, events)
    vix_age = (dev_features.Date - dev_features.vix_source_date).dt.days
    summary = dict(stage=1, status="implemented_and_run", generated_at=datetime.now(ZoneInfo("America/Detroit")).isoformat(),
                   versions=versions(), config_sha256=sha256(json_config.encode()).hexdigest(),
                   price_sessions_per_ticker=len(calendar), sample_feature_rows=len(features), development_events=len(events),
                   incomplete_development_events=len(excluded), executed_development_trades=len(trades),
                   missing_event_feature_values=int(missing_features.missing_event_values.sum()),
                   final_test_labels_generated=False, vix_max_source_age_calendar_days=int(vix_age.max()),
                   input_checksums={name: item["sha256"] for name, item in manifest["files"].items()})
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # A report snapshot is checked in separately from generated local data.
    report = build_report(cfg, summary, audit, fold_summary, metrics, cost_metrics, sector_summary, barrier_summary, factor_status)
    reports = root / "reports"
    (reports / "assets").mkdir(parents=True, exist_ok=True)
    (reports / "artifacts").mkdir(exist_ok=True)
    shutil.copyfile(out / "development_overview.png", reports / "assets/development_overview.png")
    for name in ["run_summary.json", "data_audit.csv", "feature_audit.csv", "fold_summary.csv",
                 "development_baseline_metrics.csv", "development_cost_sensitivity.csv",
                 "development_event_summary.csv", "development_barrier_summary.csv"]:
        shutil.copyfile(out / name, reports / "artifacts" / name)
    if (out / "test_results.txt").exists():
        shutil.copyfile(out / "test_results.txt", reports / "artifacts/test_results.txt")
    (root / "reports/Stage_1_Progress_Report.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(f"Report: {root / 'reports/Stage_1_Progress_Report.md'}")


def create_figure(out, curves, events):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    colors = ["#206a8a", "#c97d35", "#67875d"]
    for (name, frame), color in zip(curves.items(), colors):
        axes[0].plot(frame.index, frame.equity, label=name, color=color, linewidth=1.3)
    axes[0].set(yscale="log", ylabel="Growth of $1 (log scale)", title="Stage 1 | Development-period infrastructure checks, 2005-2021")
    axes[0].legend(loc="upper left", frameon=False)
    yearly = events.groupby(events.signal_date.dt.year).size()
    axes[1].bar(yearly.index, yearly.values, color=colors[0], width=0.7)
    axes[1].set_xticks(yearly.index[::2])
    axes[1].set(xlabel="Decision year", ylabel="Labeled opportunities", title="CUSUM events passing the fixed momentum rule")
    fig.savefig(out / "development_overview.png", dpi=180)
    plt.close(fig)


def build_report(cfg, summary, audit, folds, metrics, costs, sectors, barriers, factor_status):
    key_metrics = metrics[["strategy", "annualized_return", "sharpe", "max_drawdown", "mean_exposure", "trades"]]
    return f'''# Stage 1 Progress Report

**Project:** Learning When to Stay Out: Regime-Aware Meta-Labeling for U.S. Sector ETF Momentum<br>
**Course:** Math 628 - Machine Learning in Finance<br>
**Report date:** {summary["generated_at"][:10]} (America/Detroit)<br>
**Status:** Stage 1 implemented and executed on downloaded historical data. Models and final-test evaluation remain pending.

## 1. Three-stage delivery plan

| Stage | Scope | Status |
| --- | --- | --- |
| 1 | Data acquisition and audit, causal features, CUSUM/triple-barrier labels, shared execution engine, chronological splits, development baselines | Implemented and run |
| 2 | Logistic Regression, Random Forest, XGBoost; three nested feature sets; fold-local preprocessing; development selection of settings, probability and VIX thresholds | Pending |
| 3 | Frozen 2022-2024 evaluation; random abstention and participation controls; bootstrap uncertainty; factor attribution; final results and report | Pending |

The proposal's H1-H3 concern incremental prediction and economic selection. Stage 1 establishes infrastructure and descriptive baselines; it does not test those hypotheses or demonstrate profitable model selection.

## 2. Data acquisition and audit

Nine sector ETFs and SPY were acquired from Yahoo Finance through yfinance. FRED VIXCLS was acquired separately. A 2004 warm-up precedes the 2005-2024 research sample. Download end dates are exclusive. Each raw file has a SHA-256 checksum, source and acquisition timestamp in `data/raw/manifest.json`; package versions are saved with the run.

{markdown_table(audit)}

The audit checks unique sorted dates, coverage relative to the downloaded SPY calendar, finite positive prices and volume, OHLC ordering, requested endpoints and cache checksums. It does not independently verify every exchange session against a second vendor. No missing ETF observations are forward-filled. Historical constituent portfolios are not reconstructed. [Sector history notes](../docs/sector_history.md) document the 2016 real estate separation, 2018 communication services changes and 2023 reclassifications with primary sources; a complete constituent-change history is outside this implementation.

{factor_status}

## 3. Features and timing

The table contains {summary["sample_feature_rows"]:,} ETF-date rows over 2005-2024. The development event table has {summary["missing_event_feature_values"]:,} missing values across the explicit full-model predictor list. No imputer, scaler, encoder or regime threshold has been fitted.

| Group | Implemented predictors |
| --- | --- |
| Technical | 1/5/20-session adjusted returns, MA20/MA50 minus one, normalized RSI, EMA12-minus-EMA26 divided by price, 30-span EWM log-return volatility, volume divided by its 20-session mean |
| Market | One-session-lagged VIX level/change, SPY 20-session return and rolling 60-session drawdown |
| Cross-sector | Mean of 36 pairwise 60-session return correlations, five-session mean daily cross-sector dispersion, ETF 20-session return minus daily-rebalanced equal-weight basket return |
| Interaction and identity | Correlation times SPY drawdown; ETF ticker |

VIX is forward-filled from historical observed values before shifting by one SPY trading session; its source date is retained. Values more than seven calendar days old are invalidated. Maximum development source age is {summary["vix_max_source_age_calendar_days"]} calendar days. All rolling features use past or current-close information only. The full-feature interaction is available for Logistic Regression in Stage 2. `feature_dictionary.json` supplies a strict predictor whitelist: realized returns, exit information and labels must never enter model inputs.

## 4. Opportunity and execution specification

An opportunity requires adjusted MA20 > MA50 and a symmetric CUSUM trigger. CUSUM uses daily log returns and twice the previous session's 30-span EWM volatility. Warm-up is carried through sampling before restricting signal dates to the research period.

Decisions occur after close t; entry is at open t+1. The symmetric arithmetic-return barrier width is volatility at t times sqrt(10). Closes are monitored from entry through the ninth session after entry; a hit executes at the next open. The scheduled time exit is at the open ten sessions after entry. A final monitored-close hit coinciding with that scheduled exit keeps its profit/loss category. Gaps can make a profit-barrier trade lose money, so the binary label always uses realized return after costs.

For entry price P0, exit price P1 and per-side cost c, net return is P1*(1-c)/(P0*(1+c))-1. Positive net return gives label 1; all other complete exits give label 0. The primary cost is 5 bps per side. Execution uses open and close prices multiplied by the same Adj Close/Close adjustment factor. This is a vendor-adjusted total-return-unit approximation with implicit distribution reinvestment; dividends are not added again. It is not a literal share-and-cash dividend ledger, and adjustment revisions remain a data-vintage limitation.

Portfolio execution uses nine initially equal sleeves, one position per ETF, no leverage, no cross-sleeve rebalancing and zero cash yield. Signals while a sleeve is occupied are skipped. Equity is marked at every close, including entry and exit costs. Hypothetical event labels may overlap; the executable portfolio does not allow overlapping positions within a sleeve. Every executed trade was reconciled against the shared label engine to 1e-12 absolute tolerance.

## 5. Development labels and chronological splits

There are {summary["development_events"]:,} complete development opportunities and {summary["executed_development_trades"]:,} executed unfiltered trades. Another {summary["incomplete_development_events"]} opportunities have no observable complete development exit and are excluded. Data are cut at 2022 before constructing these labels, so a late-2021 event cannot use a 2022 exit.

{markdown_table(sectors)}

{markdown_table(barriers)}

{markdown_table(folds)}

Each training fold excludes signal dates in the final eleven SPY sessions before the annual boundary and requires exit dates strictly before refitting. Validation events must resolve before the next annual boundary, a conservative convention that prevents 2021 validation labels from touching 2022. Split indices refer to row positions in `development_events.csv` and `development_samples.csv`.

Final-train indices are prepared for the 2022 boundary. **No 2022-2024 profitability labels, model scores or portfolio results have been generated.** Acquisition and coverage auditing of those dates are separate from outcome evaluation. The local files are not a cryptographically sealed holdout; the workflow enforces this separation procedurally.

## 6. Development baseline diagnostics

These are descriptive 2005-2021 results on the development sample. They are not final-test evidence. All rows use 5 bps per side. Buy-and-hold enters at the first observed open and liquidates at the final close; sector buy-and-hold starts equally allocated and does not rebalance. Event momentum is flat at the endpoint because incomplete paths are excluded.

{markdown_table(key_metrics)}

Returns, drawdowns and exposures are decimals. Sharpe uses sqrt(252) times mean daily return divided by its sample standard deviation, with zero risk-free rate. Annualized return uses 252 trading sessions. Max drawdown includes initial capital in the high-water mark. Exposure is the portfolio fraction invested; a lower drawdown by itself does not demonstrate better trade selection. Buy-and-hold has no event trade ledger, so its event-trade count is shown as zero.

![Development baseline diagnostics](assets/development_overview.png)

The cost sensitivity below preserves event dates and exit paths, recomputes fee-dependent labels and reruns sleeve accounting. No filter or model is reselected.

{markdown_table(costs[["cost_bps_per_side", "event_positive_rate", "annualized_return", "sharpe", "max_drawdown", "trades"]])}

## 7. Verification and reproducibility

The automated test suite covers next-open execution, adverse gap fills, time-exit profitability, incomplete paths, CUSUM timing, VIX lag, future-data perturbation invariance, overlap skipping, cost reconciliation, chronology and factor percentage conversion. A saved [test result](artifacts/test_results.txt) accompanies the report. Run the test command below to verify the current checkout.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m sector_meta.cli download --factors /path/to/F-F_Research_Data_Factors_weekly.csv
PYTHONPATH=src .venv/bin/python -m sector_meta.cli stage1
.venv/bin/python -m pytest -q
```

After acquisition, Stage 1 runs offline from checksum-verified cached CSV files. The configuration snapshot is generated in `output/stage1/frozen_stage1_config.json`; input checksums and run metadata are included in the [published run summary](artifacts/run_summary.json). Configuration SHA-256: `{summary["config_sha256"]}`. Reusing a modified cache fails verification. Re-downloading may change vendor revisions, so preserve the original manifest and cache for exact replication. Raw data remain local and are ignored by Git. The repository includes the report figure and small derived summary tables under `reports/`, so these remain viewable without downloading market data.

## 8. Remaining work and Stage 2 handoff

Stage 2 should fit fold-local preprocessing and compare Logistic Regression, Random Forest and XGBoost on exactly the same opportunities. The three feature lists are declared in `features.py`; the full Logistic Regression specification includes the predeclared correlation-by-drawdown interaction. Select hyperparameters and a probability cutoff from the declared grid using development performance, subject to at least 20% candidate coverage. Select the simple VIX benchmark cutoff using development folds. Save fold-level forecasts, forecast metrics, calibration and threshold sensitivity before freezing the final specification.

Stage 3 should generate test labels only after that freeze, evaluate the untouched 2022-2024 period without refitting, add development-rate and retrospective participation-matched random controls, paired calendar moving-block bootstrap intervals, risk-free cash sensitivity and weekly factor regression with autocorrelation-robust uncertainty. It should also complete permutation importance, conditional stress plots and final interpretation of the documented sector changes and limitations.

No classifier, threshold optimization, random abstention experiment, bootstrap interval, risk-free cash sensitivity or factor regression is claimed complete in Stage 1. Stage 1 is ready to supply their data and shared execution rules.

## Sources

- Research specification: the user-provided `Project_Proposal.pdf`, dated October 9, 2026.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) and [historical price API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html).
- [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS).
- Fixed course-local Fama-French weekly CSV, 202412 CRSP vintage; exact file provenance is in the acquisition manifest.
'''
