# ROADMAP — Universal Soft-Sensor

> Prioridades y estado hacia producción.
> Actualizado: 2026-07-20 (post rename + validación cross-domain + hardening de seguridad).

---

## 0. Estado actual (qué YA está hecho)

- ✅ **Rename** de `proyecto_minero_4.0` → `universal-soft-sensor` (pipeline agnóstico al dominio).
- ✅ **Validación cross-domain con datos reales** (NASA CMAPSS, ZeMA Hydraulic, AI4I 2020),
  con artefactos de reproducción en `results/verification/`. Publicada como paper
  (`paper/paper.tex`, DOI Zenodo). Números **honestos y verificados**, con la distinción
  sensor-only vs autorregresivo y las dos trampas metodológicas documentadas.
- ✅ **Hardening de seguridad** (7 vectores de auditoría hostil cerrados, ver
  `SECURITY_AUDIT.md` + `tests/test_security.py`): leakage feature↔target, RCE por pickle,
  métricas degeneradas, path traversal, validación de entrada, muestreo del cap GP, doc.
- ✅ **Console scripts** `softsensor-*` en `pyproject.toml` (antes `mining-*`).

---

## 1. Datasets

### Cross-domain (ya validados, en `data/`)
- NASA CMAPSS FD001 (turbofan RUL) — regresión temporal.
- ZeMA Hydraulic Systems (condición del enfriador) — regresión multinivel.
- AI4I 2020 (fallo de máquina) — clasificación binaria, fuera del alcance actual.

### Caso de origen (flotación minera) — opcional, retrocompat
- **"Quality Prediction in a Mining Process"** (Kaggle): ~737k filas, sensores de planta.
  El default de `settings.py` sigue apuntando a `MiningProcess_Flotation_Plant_Database.csv`
  por retrocompatibilidad. Override con env `DATA_RAW_PATH` (alias legado `MINING_DATA_RAW_PATH`).

### Candidatos para ampliar la evidencia
| Dataset | Fuente | Utilidad |
|---|---|---|
| CMAPSS FD002–FD004 | NASA | Múltiples modos de falla / condiciones — fortalece la evidencia de RUL |
| SECOM Manufacturing | UCI | 590 features — stress-test del fallback a GBR |
| Tennessee Eastman Process | Harvard Dataverse | Benchmark industrial clásico |

---

## 2. Pendientes prioritarios (post-validación)

### Prioridad alta — cierran deuda científica y de producto
| ID | Tarea | Nota |
|---|---|---|
| **P0** | **Feature engineering de lags de INPUTS** (no solo del target) | 🔥 destapado por SRU y flotación: el pipeline solo lagea el target; procesos dinámicos con retardo de residencia necesitan historia de las ENTRADAS. Es el gap técnico que más limita el uso real. |
| P0b | **GroupKFold / splits con grupos** | evita el leakage por grupo (casi infló GeoMet a 0.93 falso); necesario para datos geometalúrgicos/por-lote |
| P1 | **Soporte de clasificación** (RF/XGBoost/SVC + one-hot automático + métricas F1/AUROC) | desbloquea AI4I y todo target binario/categórico |
| P2 | **Baselines naive + permutation test** reportados junto a cada métrica | aísla señal real de autocorrelación/leakage; ya probado a mano (`run_geomet_rigor.py`), falta integrarlo al pipeline |
| P3 | **Calibración de incertidumbre** (PICP / Sharpness) de los intervalos del GP | argumento de venta clave del GP, hoy sin validar |
| P4 | **Migrar `.pkl` → `skops`** (fix de raíz del RCE, ver V1 en SECURITY_AUDIT) | el hash SHA-256 ya mitiga, skops elimina |

### Prioridad media — deuda técnica (del audit original, aún válida)
| ID | Tarea |
|---|---|
| F2-04 | Restringir el diagnóstico de autocorrelación a la porción de train |
| F2-06 | Reorden del pipeline: `load raw → split → FE causal → scale` |
| F3-01 | Reemplazar `bare except` en el objetivo de Optuna por excepciones específicas |
| F3-03 | Logging estructurado (`console.print` → `logger.info`) en módulos core |
| F2-07 | Eliminar la rama de subsampling muerta del pipeline |

### Prioridad baja — DX / onboarding
- `Makefile` end-to-end (`scan`, `pipeline`, `train`, `predict`, `test`, `lint`).
- `config/dataset_config.example.json` (template sanitizado).
- Pre-commit hooks anti-leakage (prohibir `y.diff`/`y.rolling` sin `shift(1)` sobre el target).

---

## 3. Infraestructura faltante

- [ ] **CI/CD** — GitHub Actions: `pytest` (incl. `test_security.py`) + `ruff` + `black --check` en cada PR.
- [ ] **`.env.example`** actualizado (`DATA_RAW_PATH`, `GP_TARGET`, `GP_MAX_SAMPLES`, `GP_TRIALS`).
- [ ] **Sidecar de metadatos** JSON de entrenamiento junto a cada modelo (complementa el `.sha256` ya existente).
- [ ] **Dashboard** con datos vivos (hoy simulación).

---

## 4. Notas de integridad (no re-romper)

- **Régimen de evaluación**: para prognostics (RUL), `add_lag_features`/`add_diff_features`
  deben ir en `False` — los rezagos del target inflan la métrica (cuasi-persistencia).
- **Splits**: por unidad (CMAPSS), estratificado por clase (ZeMA). Verificar SIEMPRE que
  el test tenga varianza > 0 antes de reportar (el guard V3 ya lo fuerza).
- **Modelos**: no cargar `.pkl` de origen no confiable. El `load()` exige hash SHA-256.
