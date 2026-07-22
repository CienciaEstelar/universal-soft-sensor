# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Industrial Soft-Sensor system for mining process optimization (Mining 4.0). Predicts quality variables (recovery, grade, silica) in near real-time using Gaussian Processes with automatic fallback to Gradient Boosting. Requires Python ≥ 3.10. The project is written in Spanish (code comments, UI, variable names) — maintain this convention.

## Key Design Decisions

- **Subsample alignment**: The `DEFAULT_SUBSAMPLE_STEP` in `config/settings.py` must be the same for training and inference. Previously hardcoded differently in multiple files, now centralized. Never hardcode subsample values in individual modules.
- **Universal schema**: The validation schema uses substring pattern matching on column names, not hardcoded column lists. This makes it work across different datasets (gold_recovery, AI4I2020, etc.) without code changes.
- **No shuffle**: Temporal ordering is preserved throughout. Train/test splits are sequential, not random.
- **Dataset configuration is declarative**: New datasets are onboarded by editing `config/dataset_config.json`, not by modifying Python code. Note: the README references a `config/dataset_config.example.json` template that is not currently shipped — copy/adapt the existing `dataset_config.json` instead.

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

## Estado 2026-07-21: foco minería/cobre + P0/P0b implementados

> Detalle completo y hallazgos honestos en `ROADMAP.md` (secciones -1, 0, 1b). Resumen:

- **Alcance**: el proyecto se re-enfoca a minería exclusivamente, cobre como prioridad #1.
  CMAPSS/ZeMA/AI4I2020 quedan como referencia técnica, no como objetivo de producto.
- **`config/dataset_config.json` apunta a GeoMet cobre** (antes AI4I2020). Backup del
  config anterior en `config/dataset_config.ai4i2020.bak.json`.
- **P0 (lags de inputs) y P0b (GroupKFold/split por grupo) implementados** en
  `core/models/gp_model.py` — nuevos parámetros `add_input_lags`, `group_column`,
  `parse_dates`, método público `permutation_test()`. Expuestos en CLI de `gp_model.py`
  y `train_universal.py`. 100/100 tests del repo pasan (11 nuevos en
  `TestGroupAwareTraining`, 6 en `TestInputLagFeatures`).
- **Bug real encontrado y corregido**: `UniversalAdapter.load_data()` hacía `ffill()`
  incondicional — para datasets sin timestamp real (ej. GeoMet, spatial/geometalúrgico)
  esto fabricaba valores tomando la fila vecina de OTRO sondaje. Ahora `ffill()` solo
  corre si hay índice temporal real; sin él, `dropna()` directo.
- **✅ Reconciliación CERRADA (confirmada con 200 permutaciones)**: se había marcado
  "cerrada" (R²=0.315, p=0.005) más temprano en la misma sesión, pero esa conclusión
  venía de un **segundo bug de no-determinismo**: `UniversalAdapter._apply_feature_selection()`
  armaba `keep_cols` con `set()` de Python (orden de iteración no determinista entre
  procesos), lo que hacía que `remove_correlated_features()` tirara una feature distinta
  según la corrida ("Si ppm" vs. "Fe ppm"), y con eso un R² distinto (0.315 vs 0.167-0.177).
  **Corregido**: `UniversalAdapter` ahora preserva el orden original de columnas
  (determinista). Con el fix ya en el código, se corrió el test de 200 permutaciones
  completo (no 25 ni 15 preliminares): **R²=0.319, p=0.005** — reconcilia con el 0.33 del
  script de auditoría aislado (`geomet_rigor.json`), diferencia de 0.011 atribuible a
  92 vs 102 filas por el fix del `ffill`. Esto SÍ califica como "señal real" por el
  criterio del propio `run_geomet_rigor.py` (r²>0.3, p<0.05). Artefacto:
  `results/verification/geomet_pipeline_reconciliation_v2.json`. Probados también
  PLS/Ridge/ElasticNet/RandomForest como alternativas al GB fijo — todos peores o
  negativos, sugiriendo que la relación es no-lineal; palancas de mejora (más datos,
  features de dominio, tuning) documentadas en ROADMAP.md sección 1b.
- **Segundo dataset de cobre (SINTÉTICO) para validar generalización del método**:
  se buscó un dataset real más grande (~10 candidatos evaluados); el único con target
  de recuperación real (Mu & Salas 2023, n=1112) resultó confidencial. Se usó en su
  lugar `porphyry_01` (Garrido et al. 2020, sintético, CC BY-NC-SA 4.0,
  github.com/exepulveda/geomet_datasets) — 147,231 bloques tras filtrar waste. Con
  200 permutaciones sobre n=20,000: **R²=0.7686, p=0.005** (SINTÉTICO — nunca comparar
  directamente con el 0.319 real de GeoMet). Confirma que el método detecta y valida
  señal fuerte cuando existe, a 200x el tamaño de GeoMet. Ver ROADMAP.md sección 1c.
- **Informe científico automático (P5, 2026-07-21)**: nuevo `core/scientific_report.py`.
  Cada entrenamiento genera ahora, además del panel de diagnóstico de siempre, un PDF
  consolidado (`informe_cientifico_*.pdf`) con estilo de publicación (SciencePlots),
  gráfico de test de permutación (si se corrió) y gráfico de importancia de features
  (permutation importance, model-agnóstico). Ver ROADMAP.md sección "P5".

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
