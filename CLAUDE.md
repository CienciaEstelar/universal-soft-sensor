# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Industrial Soft-Sensor system for mining process optimization (Mining 4.0). Predicts quality variables (recovery, grade, silica) in near real-time using Gaussian Processes with automatic fallback to Gradient Boosting. Requires Python ≥ 3.10. The project is written in Spanish (code comments, UI, variable names) — maintain this convention.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -e ".[dev]"          # dev dependencies (pytest, black, ruff, mypy)

# Run tests
pytest tests/ -v                                    # all tests
pytest tests/test_modeling.py -v                    # single test file (also: test_core.py, test_pipeline.py)
pytest tests/ -k "test_name"                        # single test by name
pytest tests/ -m "validation and not integration"   # filter by custom marker

# Lint & format
ruff check .
black --check .
mypy .

# Pipeline stages — module form
python -m tools.scan_schema      # scan dataset structure
python -m core.pipeline          # ETL pipeline
python train_universal.py        # train model
python predict_universal.py      # run inference
streamlit run dashboard.py       # launch HMI dashboard

# Equivalent console-script entrypoints (after `pip install -e .`, see pyproject.toml)
mining-scan                      # = python -m tools.scan_schema
mining-pipeline                  # = python -m core.pipeline
mining-gp                        # = python -m core.models.mining_gp_pro
```

## Architecture

The system follows a staged pipeline: **Ingest -> Validate -> Preprocess -> Train -> Infer -> Dashboard**.

### Configuration Layer
- `config/settings.py` — Single source of truth (`CONFIG` singleton). All paths, GP parameters, and the critical `DEFAULT_SUBSAMPLE_STEP` are centralized here. Modules import `from config.settings import CONFIG`. Supports `.env` overrides.
- `config/dataset_config.json` — Declarative JSON defining dataset file, target column, include/exclude regex patterns for feature filtering, and data leakage prevention rules.

### Core Pipeline
- **Adapters** (`core/adapters/`) — Data ingestion layer. `UniversalAdapter` reads `dataset_config.json` and filters columns by regex patterns. `MiningCSVAdapter` handles chunked CSV streaming. `MiningDataAdapter` orchestrates the full ingestion flow.
- **Validation** (`core/validation/`) — `MiningSchema` uses pattern matching (not hardcoded column names) to detect physical variable categories (temperature, percentage, flow, pH, level) and enforce valid ranges. `MiningValidator` applies the schema.
- **Preprocessor** (`core/preprocessor.py`) — Statistical cleaning: null imputation (ffill/bfill/interpolate), outlier detection, constant column removal.
- **Pipeline** (`core/pipeline.py`) — `MiningPipeline` orchestrates ETL with chunked processing, checkpointing, and Rich progress bars.

### Modeling
- `core/models/mining_gp_pro.py` — `MiningGP` class: Gaussian Process with Matern kernels optimized via Optuna. Includes temporal feature engineering (lags, diffs, rolling windows), autocorrelation diagnostics, correlated feature removal, and automatic fallback to `GradientBoostingRegressor` when GP R² < 0.6.
- `train_universal.py` — Training orchestrator. Uses `MiningDataAdapter` for ingestion, then `MiningGP` for training. Creates temp files cleaned up via try/finally.

### Inference & UI
- `core/inference_engine.py` — `MiningInference` facade: loads saved models, generates features at inference time, de-scales predictions. Supports single-point and rolling series prediction.
- `predict_universal.py` — Inference simulation script.
- `dashboard.py` — Streamlit HMI with reactive inference and What-If scenario engine.
- `core/report_generator.py` — PDF audit report generation.

## Key Design Decisions

- **Subsample alignment**: The `DEFAULT_SUBSAMPLE_STEP` in `config/settings.py` must be the same for training and inference. Previously hardcoded differently in multiple files, now centralized. Never hardcode subsample values in individual modules.
- **Universal schema**: The validation schema uses regex pattern matching on column names, not hardcoded column lists. This makes it work across different datasets (gold_recovery, AI4I2020, etc.) without code changes.
- **No shuffle**: Temporal ordering is preserved throughout. Train/test splits are sequential, not random.
- **Dataset configuration is declarative**: New datasets are onboarded by editing `config/dataset_config.json`, not by modifying Python code. Note: the README references a `config/dataset_config.example.json` template that is not currently shipped — copy/adapt the existing `dataset_config.json` instead.

## Testing

Tests use synthetic data fixtures defined in `tests/conftest.py`. Custom markers: `integration`, `validation`, `schema`, `adapter`. The `trained_model` fixture is expensive (trains a real GP) — use sparingly.

## Tool Configuration

- **black**: line-length 100, target py310+
- **ruff**: line-length 100, includes E/W/F/I/B/C4/UP rules, ignores E501
- **pytest**: testpaths=tests, `-v --tb=short`, suppresses DeprecationWarning/UserWarning
