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
softsensor-scan                      # = python -m tools.scan_schema
softsensor-pipeline                  # = python -m core.pipeline
softsensor-gp                    # = python -m core.models.gp_model
```

## Architecture

The system follows a staged pipeline: **Ingest -> Validate -> Preprocess -> Train -> Infer -> Dashboard**.

### Configuration Layer
- `config/settings.py` — Single source of truth (`CONFIG` singleton). All paths, GP parameters, and the critical `DEFAULT_SUBSAMPLE_STEP` are centralized here. Modules import `from config.settings import CONFIG`. Supports `.env` overrides.
- `config/dataset_config.json` — Declarative JSON defining dataset file, target column, include/exclude substring patterns for feature filtering (not regex — avoids ReDoS), and data leakage prevention rules.

### Core Pipeline
- **Adapters** (`core/adapters/`) — Data ingestion layer. `UniversalAdapter` reads `dataset_config.json` and filters columns by substring match (not regex). `CSVAdapter` handles chunked CSV streaming. `DataAdapter` orchestrates the full ingestion flow.
- **Validation** (`core/validation/`) — `PhysicalSchema` uses pattern matching (not hardcoded column names) to detect physical variable categories (temperature, percentage, flow, pH, level) and enforce valid ranges. `PhysicalValidator` applies the schema.
- **Preprocessor** (`core/preprocessor.py`) — Statistical cleaning: null imputation (ffill/bfill/interpolate), outlier detection, constant column removal.
- **Pipeline** (`core/pipeline.py`) — `SoftSensorPipeline` orchestrates ETL with chunked processing, checkpointing, and Rich progress bars.

### Modeling
- `core/models/gp_model.py` — `SoftSensorGP` class: Gaussian Process with Matern kernels optimized via Optuna. Includes temporal feature engineering (lags, diffs, rolling windows), autocorrelation diagnostics, correlated feature removal, and automatic fallback to `GradientBoostingRegressor` when GP R² < 0.6.
- `train_universal.py` — Training orchestrator. Uses `DataAdapter` for ingestion, then `SoftSensorGP` for training. Creates temp files cleaned up via try/finally.

### Inference & UI
- `core/inference_engine.py` — `InferenceEngine` facade: loads saved models, generates features at inference time, de-scales predictions. Supports single-point and rolling series prediction.
- `predict_universal.py` — Inference simulation script.
- `dashboard.py` — Streamlit HMI with reactive inference and What-If scenario engine.
- `core/report_generator.py` — PDF audit report generation.

## Key Design Decisions

- **Subsample alignment**: The `DEFAULT_SUBSAMPLE_STEP` in `config/settings.py` must be the same for training and inference. Previously hardcoded differently in multiple files, now centralized. Never hardcode subsample values in individual modules.
- **Universal schema**: The validation schema uses substring pattern matching on column names, not hardcoded column lists. This makes it work across different datasets (gold_recovery, AI4I2020, etc.) without code changes.
- **No shuffle**: Temporal ordering is preserved throughout. Train/test splits are sequential, not random.
- **Dataset configuration is declarative**: New datasets are onboarded by editing `config/dataset_config.json`, not by modifying Python code. Note: the README references a `config/dataset_config.example.json` template that is not currently shipped — copy/adapt the existing `dataset_config.json` instead.

## Testing

Tests use synthetic data fixtures defined in `tests/conftest.py`. Custom markers: `integration`, `validation`, `schema`, `adapter`. The `trained_model` fixture is expensive (trains a real GP) — use sparingly.

## Validación Cross-Domain — 2026-07-20 (VERIFICADA)

> ⚠️ **Los números de la versión original de esta sección estaban inflados** por
> leakage autorregresivo (rezagos del target) y por un split degenerado en ZeMA.
> La auditoría de reproducibilidad los corrigió. Distinguir **dos regímenes**:
> *sensor-only* (solo sensores; desplegable) vs *AR* (con rezagos del target;
> cuasi-persistencia en prognostics). Artefactos: `results/verification/`. Detalle
> completo: `paper/paper.tex` y `SECURITY_AUDIT.md`.

Resultados **verificados** (régimen sensor-only, el honesto):

| Dataset | Tipo | R² sensor-only | R² con AR (inflado) | Modelo |
|---|---|---|---|---|
| 🛩️ NASA CMAPSS FD001 | Regresión RUL (turbofan) | **0.593** (RMSE 50.2) | 0.873 (cuasi-persistencia) | GP (Matérn ν=1.5) |
| 🔧 ZeMA Hydraulic Cooler | Regresión (3 niveles), split estratificado | **0.9998** (MAE 0.39) 🏆 | 0.994 | GradientBoosting (fallback) |
| ⚙️ AI4I 2020 | Clasificación binaria (fuera de alcance) | 0.169 ⚠️ | 0.266 | GradientBoosting (fallback) |

**Dos trampas metodológicas documentadas** (con artefactos de control): (1) split
secuencial ingenuo en ZeMA → test de una sola clase, R² indefinido; (2) rezagos
del target en CMAPSS → R² aparente 0.873 que es cuasi-persistencia, no señal real.

### Validación de negocio 2026-07-20 — minería real (4 datasets nuevos)

Se probó el pipeline contra datos REALES de proceso minero para responder si hay
caso de negocio. Mapa completo en `results/verification/FINDINGS.md`. Resumen:

- **Flotación de hierro (Kaggle, 737k filas):** la ley del concentrado (% sílica)
  NO es sensor-predecible. A resolución horaria honesta, persistencia R²=0.61 y los
  sensores no la superan a ningún horizonte (1-24h). El R²>0.9 de la literatura
  sobre este dataset es leakage del lag del target. Detectada además la trampa de
  persistencia inflada (target de lab repetido ~174 filas/hora → R²=0.998 fila-a-fila
  falso).
- **GeoMet cobre (Zenodo, geometalúrgico):** SÍ hay edge robusto. Recuperación LCT
  desde química+dureza → **R²=0.33, permutation p=0.005, GroupKFold por HOLEID,
  independiente de la ley de Cu (r=0.07)**. Modesto pero real. La dureza (F80/P80)
  es la que destraba la predicción — coherente físicamente. `xr` daba R²=0.77 pero
  es tautológico (r=0.88 con la ley de Cu de entrada) — descartado con el chequeo
  de tautología.
- **SRU refinería:** sensor-only instantáneo R²=−0.47 → el pipeline **necesita lags
  de INPUTS** (no solo del target). Deuda técnica #1 del ROADMAP.

**Veredicto de negocio:** el caso vive en recuperación de cobre (modesto pero
verificado con rigor de paper), no en la ley de concentrado. Scripts:
`run_flotation_*.py`, `run_geomet_*.py` en la raíz.

### Assets generados

```
data/nasa_cmaps_fd001.csv              ← CMAPSS preparado (20,631 filas, 27 cols)
data/zema_hydraulic_features.csv       ← ZeMA con feature extraction (2,205 filas, 104 features)
data/hydraulic_systems/                ← Datos crudos ZeMA (531 MB, 17 sensores)
data/ai4i2020.csv                      ← AI4I 2020 (10,000 filas)
models/gp_RUL_20260720_123357.pkl      ← Modelo CMAPSS
models/gradientboosting_cooler_condition_20260720_124004.pkl  ← Modelo ZeMA
models/gradientboosting_Machine failure_20260720_124535.pkl   ← Modelo AI4I
nasa-predictive-maintenance-rul.ipynb  ← Notebook referencia (wassimderbel)
ai4i-2020-predictive-maintenance.ipynb ← Notebook referencia (jiejiea)
```

### Lecciones aprendidas

- **El régimen de evaluación importa tanto como el modelo**: con rezagos del target,
  CMAPSS "sube" a R²=0.87 pero es cuasi-persistencia (leakage AR). Sin ellos
  (sensor-only, desplegable), R²=0.59 — moderado y honesto. Para prognostics de RUL,
  deshabilitar `add_lag_features`/`add_diff_features`.
- **GP funciona en regresión temporal, no es SOTA**: CMAPSS sensor-only R²=0.59
  (los métodos profundos especializados van a RMSE 12-16; nosotros ~50). Es baseline
  honesto, no récord.
- **Fallback automático funciona**: ZeMA GP→GB, AI4I GP→GB, sin intervención manual
- **Targets discretos (3-5 niveles)**: GP no converge bien (ZeMA R²=0.30 en CV), pero GB compensa
- **Clasificación binaria**: El pipeline no está diseñado para esto; R² es métrica incorrecta. Para AI4I se necesitaría: one-hot encoding de categóricas + modelos de clasificación + feature engineering de interacciones (Power, Power wear, Temp diff)
- **Feature extraction necesario para datos crudos**: ZeMA pasó de 43,680 a 104 features vía estadísticos por ciclo (mean, std, min, max, percentiles, trend)
- **Tiempos**: GP con 5,000 samples → ~20 min en Acer Nitro 5. Reducir GP_MAX_SAMPLES a 2,000 para iteración rápida (~1-2 min)
- **Kaggle API**: token configurado en `~/.kaggle/access_token` y `.kaggle_token` del proyecto. kagglehub instalado en `.venv`

### Configuración para reproducir

```bash
# CMAPSS
cp config/dataset_config.json config/dataset_config.json.bak
# Editar dataset_config.json → nasa_cmaps_fd001.csv, target=RUL
echo 'DATA_RAW_PATH=data/nasa_cmaps_fd001.csv\nGP_TARGET=RUL\nGP_MAX_SAMPLES=5000\nGP_TRIALS=20\nSUBSAMPLE_STEP=1' > .env
python -m core.pipeline && python train_universal.py
# Restaurar: cp config/dataset_config.json.bak config/dataset_config.json && rm .env
```

## Seguridad / Hardening — 2026-07-20

Auditoría hostil (red team) cerró 7 vectores. Ver `SECURITY_AUDIT.md` (detalle + PoCs)
y `tests/test_security.py` (11 tests de regresión). Reglas para no re-romper:

- **No cargar `.pkl` de origen no confiable**: `SoftSensorGP.load()` exige hash SHA-256
  (sidecar `.sha256` que `save()` genera). pickle = ejecución de código arbitrario.
- **Guard anti-leakage** (`_drop_target_leakage`): features con `|corr|>0.999` vs target
  se eliminan + warn; `strict_leakage=True` aborta. No desactivar sin motivo.
- **Métricas degeneradas**: `evaluate()` devuelve NaN (no 1.0/0.0) si el test tiene
  varianza cero. No "arreglar" para que dé un número.
- **Validación de entrada** en `load_data` (numérico/inf/min-filas/varianza) y contención
  de path traversal en el adapter. No bypassear.
- **Cap del GP** usa muestreo aleatorio seedeado (determinista, sin aliasing).

## Tool Configuration

- **black**: line-length 100, target py310+
- **ruff**: line-length 100, includes E/W/F/I/B/C4/UP rules, ignores E501
- **pytest**: testpaths=tests, `-v --tb=short`, suppresses DeprecationWarning/UserWarning
