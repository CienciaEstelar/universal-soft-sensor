# 📋 Plan de rename — `proyecto_minero_4.0` → `universal-soft-sensor`

> Auditoría + plan de ejecución. **Nada se ejecuta hasta tu OK.**
> Fecha: 2026-07-20

---

## 0. Resumen de la auditoría

El proyecto **ya es agnóstico al dominio en su lógica** (schema por regex, adapter universal, GP genérico, soporte probado para flotación minera + AI4I2020 máquinas rotativas). Lo único "minero" que queda es **el naming y algunos defaults** — no la arquitectura.

- **10 clases con prefijo `Mining`** que no hacen nada específico de minería.
- **~250 menciones** de "mining/minero/flotación" (mayoría en docstrings/comentarios).
- **Acoplamiento real bajo:** los defaults de flotación en `settings.py` son todos overridables por env var — no hay nada hardcodeado que impida otro dominio.

**Conclusión:** es un rename + limpieza de identidad, no una reescritura.

---

## 1. Nombre elegido

| Campo | Antes | Después |
|---|---|---|
| Proyecto / carpeta | `proyecto_minero_4.0` | `universal-soft-sensor` |
| Paquete PyPI (`name`) | `proyecto-minero` | `universal-soft-sensor` |
| Módulo import | (usa `core`/`config` sueltos) | se mantiene `core`/`config` (ver nota §6) |
| Descripción | "Pipeline ETL y Soft-Sensor GP para proceso de flotación minera" | "Pipeline ETL + Soft-Sensor universal con Gaussian Processes e incertidumbre calibrada" |

> Si prefieres otro nombre (`gp-soft-sensor`, `industrial-gp-pipeline`), cámbialo aquí y el resto del plan aplica igual.

---

## 2. Mapa de renombrado de clases (10)

| Archivo | Clase actual | Clase nueva propuesta |
|---|---|---|
| `core/models/mining_gp_pro.py` | `MiningGP` | `SoftSensorGP` |
| `core/preprocessor.py` | `MiningPreprocessor` | `Preprocessor` |
| `core/pipeline.py` | `MiningPipeline` | `SoftSensorPipeline` |
| `core/validation/schema.py` | `MiningSchema` | `PhysicalSchema` |
| `core/validation/validator.py` | `MiningValidator` | `PhysicalValidator` |
| `core/inference_engine.py` | `MiningInference` | `InferenceEngine` |
| `core/report_generator.py` | `MiningReportEngine` | `ReportEngine` |
| `core/adapters/mining_data_adapter.py` | `MiningDataAdapter` | `DataAdapter` |
| `core/adapters/mining_csv_adapter.py` | `MiningCSVAdapter` | `CSVAdapter` |
| `core/adapters/mining_data_adapter.py` | `_FileModeAdapter` | (sin cambio — ya es genérico) |

**Archivos a renombrar (físico):**
- `core/models/mining_gp_pro.py` → `core/models/gp_model.py`
- `core/adapters/mining_data_adapter.py` → `core/adapters/data_adapter.py`
- `core/adapters/mining_csv_adapter.py` → `core/adapters/csv_adapter.py`

---

## 3. Archivos que importan esas clases (hay que actualizar imports)

Del grafo de imports auditado, estos archivos rompen si no se actualizan en el mismo commit:

**Código productivo:**
- `core/adapters/__init__.py` — re-exporta los adapters
- `core/__init__.py` — exports del paquete
- `core/pipeline.py` — usa preprocessor, validator, adapter
- `core/inference_engine.py` — usa el modelo GP
- `train_universal.py`, `predict_universal.py`, `dashboard.py` — orquestadores

**Tests (críticos — son el criterio de éxito):**
- `tests/test_modeling.py` → importa `MiningGP`, `MiningInference`
- `tests/test_core.py` → importa `MiningSchema`, `MiningValidator`, `MiningPreprocessor`, `MiningCSVAdapter`
- `tests/test_pipeline.py` → importa `MiningValidator`, `SCHEMA`
- `tests/conftest.py` → importa `MiningSchema`, `MiningValidator`, `MiningGP`

> `SCHEMA` y `PhysicalCategory` (constantes/enums) **ya son genéricos** — no se tocan, solo las clases `Mining*`.

---

## 4. `config/settings.py` — defaults y env vars

| Actual | Nuevo | Nota |
|---|---|---|
| env `MINING_DATA_RAW_PATH` | `DATA_RAW_PATH` | mantener alias viejo por retrocompat (leer ambas) |
| default `MiningProcess_Flotation_Plant_Database.csv` | `dataset.csv` (genérico) | sigue overridable por env/config |
| default target `_silica_concentrate` | `target` (o dejar vacío, forzar config) | |
| `mining_clean.csv` | `dataset_clean.csv` | |

> ⚠️ Decisión tuya: ¿mantener el caso flotación funcionando con sus defaults, o volver todo genérico? Recomiendo **mantener alias viejos** (leer `MINING_DATA_RAW_PATH` si existe) para no romper tu corrida de flotación.

---

## 5. `pyproject.toml` + metadata

- `name = "universal-soft-sensor"`
- `description` nueva
- `keywords`: quitar `"mining"`, `"flotation"`; agregar `"soft-sensor"`, `"predictive-maintenance"`, `"universal"`, `"uncertainty"`
- `proyecto_minero.egg-info/` → regenerar (borrar el viejo, `pip install -e .` lo recrea)

---

## 6. Nota sobre el paquete `core`/`config`

El proyecto importa como `core.*` y `config.*` (nombres genéricos sueltos, no un paquete namespaced tipo `universal_soft_sensor.core`). **Esto ya es genérico** — no dice "mining" en ningún import de módulo. Renombrar la carpeta raíz y el `name` de pyproject es suficiente; **no hace falta re-empaquetar** `core`/`config` bajo un namespace nuevo (sería un cambio mucho más invasivo, fuera de alcance salvo que lo pidas).

---

## 7. Docstrings y comentarios (~250 menciones)

Cosmético pero de volumen. Se hace con `sed` dirigido por archivo tras el rename de clases:
- "flotación minera" / "proceso minero" → "proceso industrial" / "el proceso"
- "Arquitectura Minera 4.0" (banner en `mining_gp_pro.py`) → nombre nuevo
- README.md — reescribir intro para reflejar multi-dominio (minería + mantenimiento predictivo + genérico)

---

## 8. Git

- El remote es `CienciaEstelar/proyecto_minero_4.0`. **Renombrar el repo en GitHub** es acción manual tuya (Settings → Rename). Git re-apunta el remote solo, o se actualiza con `git remote set-url`.
- Commits sugeridos (en orden, para que sea reversible):
  1. `refactor: rename Mining* classes to domain-agnostic names`
  2. `refactor: rename module files (mining_*.py → generic)`
  3. `refactor: genericize settings defaults + env vars (keep back-compat aliases)`
  4. `chore: rename package to universal-soft-sensor in pyproject`
  5. `docs: rewrite README/CLAUDE for multi-domain identity`

---

## 9. Orden de ejecución propuesto (cuando des OK)

1. **Rama nueva** `git checkout -b rename/universal-soft-sensor` (reversible, no toca `main`).
2. Renombrar clases (`sed` por archivo) + archivos físicos.
3. Actualizar todos los imports (código + tests).
4. **Correr `pytest`** → debe pasar igual que antes del rename. Este es el gate: si los tests pasan, el rename no rompió nada.
5. Genericizar `settings.py` con alias de retrocompat.
6. `pyproject.toml` + regenerar egg-info.
7. Barrido de docstrings/comentarios + README.
8. `pytest` de nuevo + `ruff` (el proyecto usa ruff, line-length 100).
9. Reporte final del diff para que revises antes de commitear.

---

## 10. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Romper imports al renombrar | Todo el rename + imports en un solo paso, `pytest` como gate |
| Perder el caso flotación | Mantener alias `MINING_DATA_RAW_PATH` y defaults leídos de env |
| Tocar `main` sin querer | Trabajar en rama `rename/*`, merge solo tras tu revisión |
| egg-info stale | Borrar y regenerar con `pip install -e .` |

---

**Nada de esto se ejecuta sin tu confirmación.** Cuando revises: dime "dale" y arranco por el paso 1, o ajusta lo que quieras (nombre, nombres de clase, si genericizar o no los defaults de flotación).
