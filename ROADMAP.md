# ROADMAP — Proyecto Minero 4.0

> Prioridades, fuentes de datos reales y pasos hacia producción.
> Actualizado: 2026-05-10.

---

## 1. Datasets reales — dónde conseguirlos

El sistema fue diseñado para procesar datos de sensores de plantas de flotación,
pero nunca se ha probado con datos reales. Las métricas actuales (R²=0.9945,
MAPE=3.25%) provienen de un dataset sintético.

### Dataset principal (ya referenciado en el código)

**"Quality Prediction in a Mining Process" — Kaggle**

- URL: https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process
- Columnas: ~737k filas, 23 columnas de sensores + 2 de calidad (iron concentrate, silica concentrate)
- Frecuencia: datos reales de planta, intervalo ~20s
- Target default del sistema: `% Silica Concentrate` y `% Iron Concentrate`
- Archivo esperado por `settings.py`: `MiningProcess_Flotation_Plant_Database.csv`
- **Este es el dataset para el que se diseñó el sistema.**

Pasos para usarlo:
```bash
# 1. Descargar de Kaggle
# 2. Colocar en data/
mv MiningProcess_Flotation_Plant_Database.csv data/
# 3. Ejecutar pipeline
python -m tools.scan_schema
python -m core.pipeline
python train_universal.py
```

### Datasets alternativos para validación cruzada

| Dataset | Fuente | Utilidad |
|---|---|---|
| AI4I 2020 Predictive Maintenance | UCI / Kaggle | Probar el adaptador universal con sensores de distinta nomenclatura |
| Condition Monitoring of Hydraulic Systems | UCI | Series temporales multivariantes con fallas reales |
| SECOM Manufacturing | UCI | 1567 filas, 590 features — prueba de estrés para el GP (fuerza fallback a GBR) |
| Tennessee Eastman Process | Harvard Dataverse | Benchmark industrial clásico, simulación de planta química |

---

## 2. Hallazgos de auditoría pendientes (priorizados por esfuerzo/retorno)

### Sprint 1 — Bajo esfuerzo, alto retorno (1-2 horas cada uno)

| Orden | ID | Tarea | Archivo(s) |
|---|---|---|---|
| 1 | F4-01 | Crear `Makefile` con target `all: scan pipeline train predict` | raíz |
| 2 | F1-04 | Crear `.env.example` y `config/dataset_config.example.json` | raíz, config/ |
| 3 | F4-04 | Sidecar JSON con metadatos de entrenamiento junto a cada `.pkl` | `mining_gp_pro.py` |
| 4 | F3-01 | Reemplazar `bare except` en Optuna por excepciones específicas | `mining_gp_pro.py:672` |
| 5 | F4-03 | Reemplazar `np.random.seed(42)` global por `default_rng` local | `preprocessor.py:345` |

### Sprint 2 — Esfuerzo medio (2-4 horas cada uno)

| Orden | ID | Tarea |
|---|---|---|
| 6 | F3-03 | Migrar a logging estructurado: `console.print` → `logger.info` en módulos core |
| 7 | F2-07 | Eliminar rama de subsampling del pipeline (default=1 la neutraliza, pero sigue viva) |
| 8 | F2-04 | Restringir diagnóstico de autocorrelación a porción de train solamente |
| 9 | F3-07 | Documentar o consolidar `load_data`/`stream` en `_FileModeAdapter` vs `MiningDataAdapter` |

### Sprint 3 — Refactors mayores

| Orden | ID | Tarea |
|---|---|---|
| 10 | F2-06 | Reorden del pipeline: `load raw → split → FE causal → scale` |
| 11 | F4-05 | Eliminación completa de subsampling (depende de F2-07) |

---

## 3. Infraestructura faltante

- [ ] **`.env.example`** — Template con `MINING_DATA_RAW_PATH=`, `GP_TARGET=`, `SUBSAMPLE_STEP=`, etc.
- [ ] **`config/dataset_config.example.json`** — Template sanitizado del config actual
- [ ] **`Makefile`** — Targets: `all`, `scan`, `pipeline`, `train`, `predict`, `dashboard`, `test`, `lint`, `clean`
- [ ] **CI/CD** — GitHub Actions: pytest + ruff + black --check + mypy en cada PR
- [ ] **`mining-train` y `mining-predict`** en `[project.scripts]` de `pyproject.toml`
- [ ] **Pre-commit hooks** — Prevenir regresiones de leakage (prohibir `y.diff`/`y.rolling` sin `shift(1)` cuando el operando es target)

---

## 4. Fases hacia producción

### Fase 1 — Validación con datos reales (ahora)
- Descargar dataset Kaggle
- Ejecutar pipeline completo
- Comparar métricas contra baseline naive
- Publicar resultados reales (los que el sistema obtiene, no sintéticos)

### Fase 2 — Deuda técnica crítica (Sprint 1 + 2)
- Completar los 9 hallazgos rápidos
- Actualizar versiones (unificar 1.1.0 en `core/__init__.py`)
- Eliminar `AUDIT_REPORT.md` del repo local (ya fue borrado en GitHub)

### Fase 3 — CI/CD + templates
- GitHub Actions con test suite automática
- Templates de configuración para onboarding
- Makefile end-to-end

### Fase 4 — Madurez industrial
- Sidecar JSON con metadatos en cada artefacto
- Logging estructurado a archivo
- Tests de integración con dataset real
- Dashboard con datos vivos (no solo simulación)

---

## 5. Commits sugeridos (orden)

```
1. chore: sync with GitHub (delete AUDIT_REPORT.md, bump version in core/__init__.py)
2. feat: add ROADMAP.md with dataset sources and priorities
3. feat: add .env.example and dataset_config.example.json templates
4. feat: add Makefile with end-to-end targets
5. fix: replace bare except in Optuna objective (F3-01)
6. fix: replace global np.random.seed with local Generator (F4-03)
7. feat: add sidecar JSON metadata alongside model .pkl (F4-04)
8. refactor: restrict autocorrelation diagnosis to train split (F2-04)
9. refactor: remove subsample branch from pipeline (F2-07)
10. feat: add mining-train and mining-predict console scripts
11. feat: add GitHub Actions CI workflow
```
