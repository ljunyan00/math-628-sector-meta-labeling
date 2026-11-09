# Learning When to Stay Out

Python implementation of the Math 628 proposal on regime-aware meta-labeling for U.S. sector ETF momentum.

## Delivery stages

1. **Stage 1 (implemented):** data acquisition/provenance, coverage audit, causal features, CUSUM events, triple-barrier labels, shared sleeve accounting, purged annual splits, development baseline diagnostics and an English progress report.
2. **Stage 2 (pending):** Logistic Regression, Random Forest, XGBoost; feature ablations; fold-local preprocessing; probability/VIX threshold selection.
3. **Stage 3 (pending):** frozen 2022-2024 evaluation, participation controls, uncertainty, factor attribution and final report.

Read [the Stage 1 progress report](reports/Stage_1_Progress_Report.md) for measured results and limitations. Stage 1 creates development labels only; final-test outcomes remain unevaluated.

The English report, its figure, small derived summary tables, input checksums and a saved test result are included in `reports/`. Raw vendor datasets and the local Python environment are excluded from version control. Published figures and tables describe the saved Stage 1 run; downloading a newer vendor data vintage may change reproduced values.

## Run

Python 3.11 or later is required. Commands are run from this repository.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m sector_meta.cli download --factors /path/to/F-F_Research_Data_Factors_weekly.csv
PYTHONPATH=src .venv/bin/python -m sector_meta.cli stage1
.venv/bin/python -m pytest -q
```

The `--factors` argument is optional for downloading prices/VIX; supply the course-local weekly file to prepare fixed-vintage attribution inputs. Once acquired, the `stage1` command is offline. For an editable install and a `sector-research` executable, use `.venv/bin/python -m pip install -e '.[test]'` instead. Stage 2 dependencies are declared separately under the `models` extra; training is not implemented yet.

`download` uses cached data after checking their checksums. `--refresh` explicitly downloads a new vintage and replaces cached files; archive the previous data and manifest first. Source connectivity failures stop acquisition; the program never replaces real data with synthetic data. Tests use deterministic synthetic fixtures solely to verify mechanics.

## Layout

| Path | Purpose |
| --- | --- |
| `config/stage1.json` | Frozen research conventions and predeclared threshold grids |
| `src/sector_meta/data.py` | Yahoo/FRED acquisition, checksums, input audit and weekly factor parser |
| `src/sector_meta/features.py` | Predictor construction and model feature whitelists |
| `src/sector_meta/events.py` | CUSUM and shared hypothetical trade execution |
| `src/sector_meta/backtest.py` | Nine sleeves, concurrent holdings and daily equity accounting |
| `src/sector_meta/splits.py` | Expanding annual folds, label purge and boundary gap |
| `src/sector_meta/pipeline.py` | Stage 1 outputs, reconciliation and progress report |
| `tests/` | Accounting and causal-timing tests |
| `data/raw/` | Local cached vendor data and acquisition manifest (ignored by Git) |
| `data/processed/` | Feature/event tables, split indices and factor inputs (ignored by Git) |
| `output/stage1/` | Development diagnostics, figure, metrics and run metadata |
| `reports/Stage_1_Progress_Report.md` | English progress report |
| `reports/assets/` | Report figures viewable directly on GitHub |
| `reports/artifacts/` | Small derived summary tables, run metadata and saved test result |

Use only columns listed in `data/processed/feature_dictionary.json` as predictors. Exit dates/prices, barrier outcomes, realized returns and labels are future information. Split arrays index the development event/sample tables; do not shuffle chronological samples. Preprocessing and probability threshold selection belong inside development folds.

## Execution conventions

Signals are after-close; entries and barrier exits use subsequent opens. The time exit is entry plus ten trading sessions at the open. All OHLC use the same vendor total-return adjustment factor; distributions are implicitly reinvested, never added twice. This adjusted-unit approximation is documented in the report. An entry fee buys fewer units and the exit fee reduces proceeds. Labels use the identical cash ratio as the portfolio ledger.

Nine sleeves begin equally funded, retain their own cash and do not rebalance. Signals while a sleeve is occupied are skipped. Development events requiring an unavailable exit after 2021 are omitted. Annual validation outcomes must resolve before the next annual boundary. The eleven-session training gap uses the common SPY calendar.

Baseline performance is descriptive development evidence. Classifier forecasts, final-test results, deployment claims and conclusions about H1-H3 are outside Stage 1.
