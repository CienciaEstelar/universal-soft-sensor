Verificación independiente 2026-07-20 (sesión de auditoría del paper, post-rename).
Corridas del pipeline REAL (SoftSensorGP.train_from_file / flujo equivalente) sobre los 3 benchmarks.
Config: GP_MAX_SAMPLES=600-1000, n_trials=2-3 (métricas estables en ese rango).

DOS REGÍMENES DE EVALUACIÓN (distinción crítica):
- sensor-only  : sin lags/derivados del target — soft-sensing puro, desplegable. ES EL NÚMERO HONESTO.
- con lags (AR): el FE por defecto crea RUL_lag_1/5/10/20 etc. del TARGET. En CMAPSS eso es
  cuasi-persistencia (RUL_lag1 = RUL+1 dentro de un motor) — INVÁLIDO para prognostics estándar.
  Solo legítimo si el target pasado es observable en producción (ej. ensayos de laboratorio).

Artefactos:
- cmapss_sensor_only.json   : R2=0.593  RMSE=50.2  (HONESTO — no supera al notebook de referencia)
- cmapss_1000.json          : R2=0.873  RMSE=28.0  (con lags del target — cuasi-persistencia, documentado como trampa)
- zema_sensor_only.json     : R2=0.9998 MAE=0.39   (HONESTO — resultado estrella, split estratificado)
- zema_stratified.json      : R2=0.994  MAE=1.17   (con lags)
- zema.json                 : split secuencial puro DEGENERADO (test monoclase) — evidencia de la trampa #1
- ai4i_sensor_only.json     : R2=0.169  (resultado negativo, límite de aplicabilidad)
- ai4i.json                 : R2=0.266  (con lags)
- verify_one.py             : runner (NOLAGS=1 para sensor-only)

Split estratificado ZeMA: por clase, primer 80% de ciclos -> train, último 20% -> test
(orden intra-clase preservado; test 149/146/146).
