# ROADMAP — Universal Soft-Sensor

> Prioridades y estado hacia producción.
> Actualizado: 2026-07-21 (foco minería/cobre + P0 lags de inputs + P0b GroupKFold).

---

## -1. Decisión de alcance (2026-07-21): minería, cobre primero

El proyecto deja de perseguir generalidad cross-domain (NASA CMAPSS, ZeMA, AI4I2020 —
turbofan/manufactura, no minería) como objetivo de producto. Esos dominios quedan como
**referencia técnica** de que el pipeline generaliza, no como caso de negocio. El foco
pasa a ser **exclusivamente minería, con cobre como prioridad #1**; otros minerales
(oro, hierro) se consideran solo si el pipeline funciona bien ahí también.

Consecuencia directa: `config/dataset_config.json` ahora apunta por defecto a
**GeoMet cobre** (antes AI4I2020 — quedó ahí tras la última validación cross-domain,
desalineado con este foco). Backup del config anterior en
`config/dataset_config.ai4i2020.bak.json`; template genérico en
`config/dataset_config.example.json`.

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
- 🟡 **P0 — mecanismo de lags de INPUTS implementado (2026-07-21)** — ver detalle abajo.
  El *mecanismo* está hecho y probado a nivel unitario; la *validación empírica* (re-correr
  SRU/flotación con `--input-lags` y confirmar que el R²=-0.47 sensor-only mejora) **NO se
  ha hecho todavía** — no asumir que el gap de SRU está cerrado sin volver a correr
  `run_flotation_*.py`/el caso SRU con la nueva feature activa.
- ✅ **P0b — GroupKFold/split por grupo implementado y RECONCILIADO (2026-07-21, confirmado)** —
  ver detalle abajo. El pipeline central soporta `group_column`/`parse_dates=False`, y
  `dataset_config.json` apunta a GeoMet cobre con esto activo end-to-end. Reconciliación
  cerrada con 200 permutaciones sobre código ya corregido: el pipeline integrado da
  **R²=0.319, p=0.005** — prácticamente idéntico al auditado (R²=0.33, p=0.005 en
  `geomet_rigor.json`, script aislado). La diferencia de 0.011 es atribuible a 92 vs 102
  filas (fix del bug de `ffill`, ver abajo). Artefacto:
  `results/verification/geomet_pipeline_reconciliation_v2.json`.
  De paso se encontraron y corrigieron dos bugs reales: `ffill()` incondicional (fabricaba
  valores no-temporales entre grupos distintos) y orden no-determinista de `set()` en
  `_apply_feature_selection` (causó una falsa alarma intermedia de "reconciliación
  cerrada" con un R² casual de 0.315 antes de encontrarse el segundo bug) — ver nota en
  sección 4.

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
| CMAPSS FD002–FD004 | NASA | Múltiples modos de falla / condiciones — fortalece la evidencia de RUL (referencia técnica, no prioridad de negocio tras el pivote) |
| SECOM Manufacturing | UCI | 590 features — stress-test del fallback a GBR (idem) |
| Más datasets geometalúrgicos/mineros REALES (cobre primero) | por buscar | **Sigue siendo prioridad de negocio** — se buscó exhaustivamente (Zenodo, MDPI, GitHub, ~10 candidatos evaluados) y el único hallazgo real con target de recuperación (Mu & Salas 2023, n=1112) resultó ser **confidencial, no descargable**. El único dataset abierto encontrado con la estructura correcta (porphyry_01, ver sección 1c) es sintético — sirve para validar el método, no para robustecer el caso de negocio. La vía real pasa por contacto institucional/académico, no por búsqueda web. |

---

## 1b. Reconciliación del R² de GeoMet cobre — CERRADA (2026-07-21, confirmada con 200 perms)

> Historial de esta sección en la misma sesión: "cerrada" (0.315, con bug) → "reabierta"
> (0.177, provisional con 25 perms, tras el fix de determinismo) → **cerrada de nuevo**,
> esta vez con 200 permutaciones completas sobre el código ya corregido: **R²=0.319,
> p=0.005**. La lectura de 25 perms (0.177) resultó ser ruido de muestra chica en el
> conteo de permutaciones, no el valor real — con 200 perms converge donde debía.

`config/dataset_config.json` apunta a GeoMet cobre (`data/geomet/flotation_chem_dureza.csv`,
target `LCT`, `group_column=HOLEID`). Se encontraron y corrigieron dos bugs de
reproducibilidad independientes en `UniversalAdapter`:
1. `ffill()` incondicional (fabricaba valores entre HOLEID distintos en datos no-temporales).
2. `keep_cols` como `set()` de Python en `_apply_feature_selection()` — orden de iteración
   no determinista entre procesos (hash randomization), hacía que `remove_correlated_features()`
   tirara una feature distinta ("Si ppm" vs. "Fe ppm") según la corrida. Corregido preservando
   el orden original de `df.columns`.

| Fuente | R² | p-value | Permutaciones | Filas | Notas |
|---|---|---|---|---|---|
| `results/verification/geomet_rigor.json` (script aislado, `run_geomet_rigor.py`) | 0.33 | 0.005 | 200 | 102 | GB fijo, `cross_val_predict` sobre TODO el dataset |
| Pipeline integrado, PRE-fix determinismo | 0.315 / 0.167–0.177 | 0.005 / — | 200 / 0 | 92 | variaba según qué feature correlacionada sobrevivía por azar |
| Pipeline integrado, POST-fix, 25 perms | 0.177 | 0.038 | 25 | 92 | lectura preliminar ruidosa, subestimó el R² real |
| **Pipeline integrado, POST-fix, 200 perms (DEFINITIVO)** | **0.319** | **0.005** | 200 | 92 | `results/verification/geomet_pipeline_reconciliation_v2.json` |

**Veredicto final**: el pipeline integrado y determinista reproduce el hallazgo del script
de auditoría aislado (0.319 vs. 0.33, diferencia de 0.011 atribuible a 92 vs. 102 filas por
el fix de `ffill`). Con el criterio de severidad del propio `run_geomet_rigor.py`
(`r2>0.3 and p<0.05` → "señal real"), **esto SÍ califica como señal real**, no como
"débil/dudosa". Features usadas (25, tras dropear "A" por constante y "Fe ppm" por
correlación >0.98): `Ag, Al, Au, C, Ca, Cl, Cu, F, K, Mg, Mn, Na, P, S, Si, Th, Ti, U ppm`,
`th1, th2, th3, F80, P80, M, Carbono Grafite ppm`.

**Comparación de modelos** (misma sesión, sin permutación, GroupKFold k=5, 25 features,
92 filas): GB fijo (n_est=200, depth=2) da el mejor resultado (R²≈0.18-0.32 según seed/CV);
RandomForest R²≈0.09; PLS (2-5 componentes), Ridge y ElasticNet dan **todos negativos**
(peor que predecir la media). La relación no es lineal/de bajo rango — PLS es la
herramienta clásica de quimiometría para este tipo de problema (features correlacionadas,
n chico) y que rinda peor que un árbol es información real: la señal es no-lineal o
depende de interacciones entre variables (ej. dureza × química), no de una combinación
lineal simple de ppm. No se intentó tuning de hiperparámetros del GB ni reducción de
dimensionalidad todavía — ver "Palancas para mejorar el R²" más abajo.

### Palancas para mejorar el R² (orden de impacto esperado, no probado)

| Palanca | Impacto esperado | Naturaleza | Nota |
|---|---|---|---|
| **Más datos (más sondajes/muestras)** | Alto | Inferencia, no hecho | n=92/21 grupos es chico para 25 features; el intervalo de confianza del R² es ancho. Es la palanca de mayor apalancamiento y la única que ataca la causa raíz (poder estadístico), no solo el modelo. |
| **Features de dominio geometalúrgico** (ratios Cu/Fe, S/Cu como proxy de mineralogía, índices de alteración) | Medio-alto | Hipótesis, no probado | Los ppm crudos ya se usan; ratios pueden capturar relaciones que el GB solo aprende indirectamente. Requiere criterio de mina (ventaja del usuario sobre un data scientist genérico). |
| **Tuning de hiperparámetros del GB fijo** | Medio | No probado | `permutation_test()` usa un GB con hiperparámetros fijos (`n_estimators=200, max_depth=2, lr=0.05`) para que el test sea comparable entre corridas — no se optimizó para performance. Un `Optuna` search podría subir el R² real. |
| **Reducción de dimensionalidad / selección de features** | Medio | Hipótesis, no probado | 25 features / 92 filas es un ratio ajustado. PCA o selección por importancia podrían reducir varianza del estimador. |
| Modelos no-lineales alternativos (SVR-RBF, redes chicas) | Bajo-medio | No probado | Dado que PLS/Ridge/ElasticNet (lineales) fallan y GB (no-lineal, basado en árboles) funciona, otro no-lineal podría ayudar, pero con n=92 el riesgo de overfitting es real. |

**Caveat obligatorio (anti p-hacking)**: cualquier cambio de arriba que "mejore" el R² en
una sola corrida debe re-validarse con `permutation_test()` (200 perms) antes de creerlo.
Con n=92 y 25+ features, es fácil que un cambio de semilla, de split, o de feature engineering
suba el R² por azar de muestreo, no por señal real — exactamente el patrón que ya causó dos
falsas alarmas en esta misma sesión (0.93 leakage, 0.315 no-determinismo). No reportar un
R² nuevo como mejora sin su p-value acompañante.

---

## 1c. Validación de generalización del método — dataset SINTÉTICO porphyry_01 (2026-07-21)

> ⚠️ **Este resultado NO es evidencia de negocio.** Es un dataset SINTÉTICO (simulado
> geoestadísticamente), no datos reales de una mina. Nunca comparar, mezclar ni reportar
> este R² junto al de GeoMet cobre (real, R²=0.319) sin la etiqueta explícita "sintético".
> Su único propósito es responder una pregunta distinta: ¿el método (features numéricas +
> GB con hiperparámetros fijos + `permutation_test`) generaliza a un dataset de cobre
> mucho más grande, o el 0.319 de GeoMet es un artefacto de tener solo 92 filas?

**Búsqueda de un segundo dataset de cobre real**: se evaluaron ~10 candidatos (Zenodo,
MDPI, GitHub, listas curadas como `awesome-mining-data`). Todos los datasets reales con
target de recuperación metalúrgica resultaron confidenciales — el caso más claro es
Mu & Salas (2023, *Processes* 11(6):1775, PUC Chile): dataset real de 1112 muestras de
sondaje de cobre, con "Data Availability Statement" explícito: *"The original data set is
not publicly available because of confidentiality agreements."* Esto no es casualidad:
datos de recuperación metalúrgica real tienen valor comercial directo para una minera
(definen blending, secuencia de extracción, NPV) y sistemáticamente no se liberan.

**Dataset usado**: `porphyry_01` de
[github.com/exepulveda/geomet_datasets](https://github.com/exepulveda/geomet_datasets)
(CC BY-NC-SA 4.0), del paper Garrido, Sepúlveda, Ortiz, Townley (2020), *"Simulation of
synthetic exploration and geometallurgical database of porphyry copper deposits for
educational purposes"*, Natural Resources Research 29:3527-3545. Block model simulado
de depósito pórfido de cobre, mismo linaje académico (geoestadística chilena) que GeoMet.

- 153,076 bloques originales → 147,231 tras filtrar waste (`cu==0`, zona de
  mineralización 5, "sin contenido económico" según el paper). El filtro es necesario:
  sin él, la correlación cu-rec trivial es 0.30 (predecir rec=0 donde cu=0 es gratis);
  con el filtro cae a 0.01 — confirma que el problema filtrado es genuinamente no-trivial,
  igual que GeoMet real. Script reproducible: `prepare_porphyry_synthetic_dataset.py`.
- Sin `group_column`: cada fila es un bloque espacial único post-interpolación, no hay
  múltiples muestras compartiendo un mismo sondaje (a diferencia de GeoMet/HOLEID).
- Config: `config/dataset_config.porphyry_synthetic.json` (separado de
  `dataset_config.json`, que sigue apuntando a GeoMet cobre real — no se tocó).
- Features tras feature engineering (6, de 12 candidatas): `chalcocite, bornite, cu, mo,
  as, bwi`. Se eliminaron `ton` (constante tras filtro), `tennantite` y `molibdenite`
  (correlación >0.98). Coordenadas x/y/z excluidas a propósito (régimen sensor-only).
- `load_data()` cachea automáticamente a 100,000 filas (cap de `GP_MAX_SAMPLES`).

**Resultado confirmado** (200 permutaciones, GB fijo n_est=200/depth=2/lr=0.05, KFold
k=5, subsample n=20,000 de las 100,000 disponibles, corrido en la máquina del usuario —
tardó 5147.8s ≈ 85.8 min):

```
real_r2 = 0.7686, p_value = 0.005, n_permutations = 200, n_splits = 5
```

Artefacto: `results/verification/porphyry_synthetic_permutation_confirmed.json`.

**Interpretación**: R²=0.77 (vs. 0.32 en GeoMet real) es exactamente lo esperable en un
dataset sintético — las relaciones simuladas son más limpias que las reales (sin ruido de
medición, sin variables omitidas, sin heterogeneidad no capturada). El hallazgo relevante
no es el valor del R², es que **el método detecta y confirma señal fuerte cuando existe**,
con p-value en el piso matemático posible (0.005 = 1/201) — evidencia de que el pipeline
no está artificialmente limitado a dar R²~0.3 por mal calibrado, y que escala sin romperse
a 200x el tamaño de GeoMet (n=20,000 vs. n=92).

**No se corrió sobre las 100,000 filas completas**: estimado en 7-10 horas (escalado desde
el tiempo medido en 20,000 filas, asumiendo costo entre lineal y ~n^1.2). No se justifica —
la pregunta que este dataset debía responder (¿generaliza el método?) ya quedó respondida
sin ambigüedad con n=20,000; a diferencia de GeoMet, donde el resultado con muestras
chicas de permutación (25) era genuinamente frágil (0.177 vs 0.315 según el bug), acá el
p-value ya tocó el piso posible y el R² es alto y estable — más muestra no cambiaría la
conclusión.

---

## 2. Pendientes prioritarios (post-validación)

### Prioridad alta — cierran deuda científica y de producto
| ID | Tarea | Nota |
|---|---|---|
| **P0** | ~~Feature engineering de lags de INPUTS~~ **Mecanismo implementado (2026-07-21)** | `SoftSensorGP(add_input_lags=True, input_lag_periods=[...], input_lag_columns=[...])` — lagea columnas de ENTRADA (causal, corre antes de `_create_lag_features` para no laguear derivados del target). Expuesto en CLI de `gp_model.py` (`--input-lags`/`--input-lag-periods`/`--input-lag-columns`) y de `train_universal.py` (mismos flags + fallback declarativo desde `dataset_config.json["feature_engineering"]`). También se agregó `--no-diffs` (faltaba, bloqueaba reproducir el régimen sensor-only honesto). 7 tests nuevos en `tests/test_modeling.py::TestInputLagFeatures`, 89/89 tests del repo pasan. **Pendiente**: re-correr SRU/flotación con esto activo y confirmar mejora real sobre el R²=-0.47 sensor-only — el mecanismo está verificado, el impacto empírico en esos datasets NO. |
| **P0b** | ~~GroupKFold / splits con grupos~~ **Mecanismo implementado y reconciliado (2026-07-21)** | `SoftSensorGP(group_column=..., parse_dates=False)` — split train/test con `GroupShuffleSplit` (ningún grupo en ambos lados), CV interna de Optuna con `GroupKFold`, más `permutation_test()` público (generaliza `run_geomet_rigor.py`, reutilizable fuera de GeoMet). Expuesto en CLI de ambos scripts (`--group-column`/`--no-parse-dates`/`--permutation-test`) y declarativo en `dataset_config.json["feature_engineering"]`. 11 tests nuevos en `TestGroupAwareTraining`, 100/100 tests del repo pasan. **Reconciliación CERRADA** (ver sección 1b): tras corregir un bug de no-determinismo (`keep_cols` como `set()` en `UniversalAdapter`), el pipeline integrado confirma con 200 permutaciones **R²=0.319, p=0.005** — señal real por el criterio del propio audit, prácticamente idéntica al 0.33 del script aislado. |
| P1 | **Soporte de clasificación** (RF/XGBoost/SVC + one-hot automático + métricas F1/AUROC) | ya no es prioritario tras el pivote a minería (AI4I2020 era el caso que lo motivaba) — degradar prioridad salvo que aparezca un target categórico minero real |
| **P2** | ~~Baselines naive + permutation test~~ **Implementado (2026-07-22)** | Permutation test ya estaba integrado en P0b. **Baseline naive agregado**: `evaluate(..., y_train_mean=...)` calcula `baseline_r2` = R² de predecir SIEMPRE la media del target de ENTRENAMIENTO sobre el test. Da el piso contra el que se lee el R² del modelo (un R²=0.32 no dice nada sin saber que el baseline da ~0.00). **Nota anti-tautología**: la media viene del TRAIN, no del test — la del test daría 0.0 por definición (R² se define contra la media del propio conjunto); con la del train puede ser negativo si difieren, y eso es información honesta. Consola y PDF muestran la fila con veredicto "el modelo lo supera / NO supera al no-modelo". 5 tests en `TestNaiveBaseline` (baseline ≈0 con medias iguales, negativo con media sesgada, None sin argumento, empate en 0 cuando el modelo predice la media del test, serialización). Verificado end-to-end. |
| **P3** | ~~Calibración de incertidumbre~~ **Implementado (2026-07-22)** | `evaluate(y_true, y_pred, y_std=...)` ahora calcula **NLL** (negative log-likelihood gaussiana media: `0.5·mean[log(2π·σ²) + (y-μ)²/σ²]`) y **Coverage@95** (fracción con `\|y-μ\|≤1.96σ`, ≈PICP). Responden lo que R²/RMSE no: si la σ del GP es confiable o son bandas decorativas. Idea adaptada del proyecto de cosmología `train_gp.py` (reconstrucción de potencial escalar). **Guard anti-trampa**: solo se calculan si hay σ real (toda σ>0); para el fallback GradientBoosting (σ=0) quedan `None`, no un número engañoso — calcular NLL sobre σ=0 daría división por cero enmascarada. Expuesto en la tabla de consola (con interpretación sobre/sub-confiado) y en el PDF científico. 7 tests nuevos en `TestUncertaintyCalibration` (incluye caso determinista de cobertura exacta, GB→None sin crash, backward-compat sin y_std). Verificado end-to-end: GP real sobre datos sintéticos da NLL=-0.79, Cov95=1.0, Sharpness=0.67 (sub-confiado, honesto). **Sharpness agregado (2026-07-22)**: ancho medio del IC 95% = `mean(2·1.96·σ)`, en unidades del target. Cierra el flanco de la cobertura — un modelo logra cobertura ~1.0 con bandas absurdamente anchas; se leen juntas (meta: cobertura ≈0.95 con la menor sharpness posible). Test dedicado `test_coverage_and_sharpness_together_expose_wide_band_trap` documenta el contraste (σ=0.1 vs σ=5.0: misma cobertura alta, sharpness ~50x peor). 11 tests en `TestUncertaintyCalibration`. |
| P4 | **Migrar `.pkl` → `skops`** (fix de raíz del RCE, ver V1 en SECURITY_AUDIT) | el hash SHA-256 ya mitiga, skops elimina |
| **P6** | ~~Test de extrapolación~~ **Implementado (2026-07-22)** | `SoftSensorGP.extrapolation_test(X, y, feature, low_pct, high_pct)` — método diagnóstico. Idea adaptada del "Test Extrapolación" de `train_gp.py` (cosmología). Entrena un GP fijo (Matérn ν=1.5, sin Optuna) SOLO en el interior del rango de una variable física (ej. dureza F80/P80, ley Cu) y evalúa en el EXTERIOR nunca visto. Pregunta central: **¿el GP ENSANCHA su σ fuera del rango (degrada con gracia) o se equivoca con confianza?** Firma sana: `std_ratio = σ_ext/σ_int > 1` → `graceful=True`. Protocolo anti-trampa: interior partido 80/20, σ_interior medido en holdout (no en puntos de entrenamiento, donde σ sería artificialmente bajo). Fuerza GP aunque el modelo de producción sea GB (el GB no da σ). Guards de zona mínima → `status="skipped"` en vez de número frágil. **Verificado sobre GeoMet cobre real (n=41)**: en F80, P80 y Cu ppm el GP ensancha σ 1.27-1.67x (`graceful=True` en las tres); con F80 (dureza — la variable que destraba la predicción) la cobertura exterior se mantiene en 0.92 pese a extrapolar. 6 tests en `TestExtrapolation`. |
| **P5** | ~~Informe científico automático~~ **Implementado (2026-07-21)** | Nuevo módulo `core/scientific_report.py`. Cada `train_from_file()` ahora genera automáticamente, además del panel de diagnóstico de siempre: (1) estilo de gráficos tipo publicación vía `SciencePlots` (`['science','no-latex']`, cae a `seaborn-v0_8-whitegrid` si no está instalado — ver `apply_scientific_style()`); (2) gráfico de test de permutación (distribución nula vs. R² real, solo si `run_permutation_test=True`); (3) gráfico de importancia de features vía `sklearn.inspection.permutation_importance` (model-agnóstico, funciona igual para GP y GB — a diferencia de `.feature_importances_`); (4) PDF consolidado (`informe_cientifico_*.pdf`) con portada de metadata de reproducibilidad (dataset, semilla, estilo gráfico, fecha) + tabla de métricas + los gráficos anteriores, vía `matplotlib.backends.backend_pdf.PdfPages`. Las figuras se pasan ya renderizadas al PDF (no se releen desde PNG) para evitar doble rasterizado. `permutation_test()` ahora también retorna `null_r2_distribution` (lista, no solo el resumen) — cambio backward-compatible. 3 tests nuevos pasando (contrato de `permutation_test` actualizado), 103/103 tests del repo pasan. |

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
- ~~`config/dataset_config.example.json` (template sanitizado)~~ ✅ creado 2026-07-21 (copia del config de AI4I2020, que queda como ejemplo genérico de clasificación).
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
- **Lags de inputs (P0)**: `add_input_lags=True` sin restringir `input_lag_columns` en
  datasets con muchas columnas puede disparar la dimensionalidad (columnas × periodos).
  El guard de leakage y `remove_correlated_features` podan lo redundante, pero conviene
  pasar `input_lag_columns` explícito en datasets anchos. `input_lag_periods` por defecto
  es corto (`[1,2,3]`, vs. `[1,5,10,20]` del target) a propósito: modela retardo de
  proceso, no autocorrelación de largo plazo — no asumir que sirve igual para ambos casos.
- **Grupos y datos no-temporales (P0b)**: con `group_column` seteado, el split/CV honesto
  depende de que TODAS las filas del mismo grupo queden juntas — nunca combinar
  `group_column` con `subsample_step>1` sin verificar que el diezmado no rompe la
  cardinalidad de grupos. `parse_dates=False` desactiva diagnóstico de autocorrelación y
  subsampleo (no tienen sentido físico sin tiempo); si un dataset SÍ es temporal, dejar
  `parse_dates=True` (default) — no usar `group_column` como sustituto de eso.
- **`ffill()` en `UniversalAdapter` es condicional desde 2026-07-21**: solo aplica si el
  dataset tiene `timestamp_column` real (índice temporal). Sin timestamp, usa `dropna()`
  directo — NO reintroducir el `ffill()` incondicional: en datos agrupados/spatiales
  fabrica valores tomando la fila vecina de OTRO grupo (bug real encontrado con GeoMet
  cobre, columna "Carbono Grafite ppm" con 12/60 NaN).
