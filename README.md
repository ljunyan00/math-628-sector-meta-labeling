# Learning When to Stay Out

Math 628 research on whether the recent return patterns of a U.S. sector ETF and its peers improve selection of profitable momentum trades. The revised experiment compares a two-dimensional return map CNN with a temporal CNN and tabular benchmarks. See the [research plan](docs/cnn_research_plan.md).

## Delivery stages

1. **Stage 1 (implemented):** data acquisition/provenance, coverage audit, causal features, CUSUM events, triple-barrier labels, shared sleeve accounting, purged annual splits, development baseline diagnostics and an English progress report.
2. **Stage 2 (completed):** causal 20-session return windows and visualizations, followed by chronological comparisons of logistic regression, XGBoost, a temporal CNN and a return map CNN.
3. **Stage 3 (completed):** frozen 2022–2024 evaluation with participation controls, uncertainty, cost sensitivity and factor attribution.

The [final report](reports/Final_Report.md), [report PDF](reports/delivery/Math_628_Final_Research_Report.pdf), [PowerPoint](reports/delivery/Math_628_Final_Presentation.pptx) and [30-minute speaker notes](reports/Final_Presentation_Notes.md) cover the research and classroom presentation. The [Stage 1 progress report](reports/Stage_1_Progress_Report.md) documents the original data and backtest diagnostics. The [frozen-run results](reports/CNN_Research_Report.md) provide calibration, portfolio, cost and factor tables.

The English report, its figure, small derived summary tables, input checksums and a saved test result are included in `reports/`. Raw vendor datasets and the local Python environment are excluded from version control. Published figures and tables describe the saved Stage 1 run; downloading a newer vendor data vintage may change reproduced values.

## Run

Python 3.11 or later is required. Commands are run from this repository.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m sector_meta.cli download --factors /path/to/F-F_Research_Data_Factors_weekly.csv
PYTHONPATH=src .venv/bin/python -m sector_meta.cli stage1
PYTHONPATH=src .venv/bin/python -m sector_meta.cli cnn-visualize
.venv/bin/python -m pytest -q
```

The `--factors` argument is optional for downloading prices and VIX. Supply the course-local weekly file to prepare fixed-vintage attribution inputs. After acquisition, the remaining commands run offline. Python 3.12 was used for the saved runs.

To run the revised experiment, install the model dependencies before running development and holdout evaluation in sequence. On macOS, the XGBoost wheel may require the OpenMP runtime (`brew install libomp`). See the [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).

```bash
.venv/bin/python -m pip install -e '.[models]'
PYTHONPATH=src .venv/bin/python -m sector_meta.cli cnn-develop
PYTHONPATH=src .venv/bin/python -m sector_meta.cli cnn-test
PYTHONPATH=src .venv/bin/python -m sector_meta.final_review
```

The development command saves a frozen specification in `output/cnn/` with input checksums, model settings, thresholds and package versions. The holdout command checks that specification and writes the English report and aggregate results to `reports/`. Detailed predictions and daily portfolio returns remain under `output/cnn/`.

The final review checks the frozen source and data hashes, reconstructs the labeled opportunities, reconciles the saved scores and portfolio execution, and writes paired log-loss intervals and matched candidate comparisons to `reports/final/`. The matched comparisons are retrospective diagnostics at 323 accepted candidates per model. The frozen cutoffs continue to define the primary trading results.

If a macOS Apple Silicon run exits during XGBoost fitting after importing PyTorch, point the dynamic loader at the OpenMP library bundled with the PyTorch wheel for the experiment commands. For the tested Python 3.12 virtual environment, run `export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib"` before running development and holdout evaluation.

`download` uses cached data after checking their checksums. `--refresh` explicitly downloads a new vintage and replaces cached files; archive the previous data and manifest first. Source connectivity failures stop acquisition; the program never replaces real data with synthetic data. Tests use deterministic synthetic fixtures solely to verify mechanics.

## Layout

| Path | Purpose |
| --- | --- |
| `config/stage1.json` | Frozen research conventions and predeclared threshold grids |
| `config/cnn.json` | Revised input, training, participation and uncertainty settings |
| `src/sector_meta/sequence_data.py` | Past-only target and peer return windows and two-dimensional maps |
| `src/sector_meta/cnn_models.py` | Tabular baselines, temporal CNN and return map CNN |
| `src/sector_meta/cnn_experiment.py` | Development forecasts, frozen holdout and report |
| `src/sector_meta/final_review.py` | Frozen-run audit and retrospective diagnostics |
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
