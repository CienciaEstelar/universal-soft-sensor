#!/usr/bin/env python3
"""
run_flotation_benchmark.py — Corre el Universal Soft-Sensor sobre el dataset REAL
de flotación (Quality Prediction in a Mining Process) y lo compara contra baselines
naive. Este es el test que define si hay caso de negocio.

Uso (desde la raíz del repo, con el .venv activado):
    python run_flotation_benchmark.py

Requisitos: data/MiningProcess_Flotation_Plant_Database.csv (176 MB, ya descargado).
Predice % Silica Concentrate. Descarta:
  - date (índice temporal, no feature)
  - % Iron Concentrate (CO-PRODUCTO de salida, no disponible al predecir = leakage)
Régimen sensor-only (sin lags del target) = el número honesto y desplegable.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "data", "MiningProcess_Flotation_Plant_Database.csv")
TARGET = "% Silica Concentrate"
LEAK = "% Iron Concentrate"   # co-producto de salida: NO disponible en tiempo real
OUT = os.path.join(HERE, "results", "verification", "flotation_silica.json")

# GP es O(n^3): cap de muestras. 2000 da buen número en ~pocos min. Subir si tenés paciencia/RAM.
os.environ.setdefault("GP_MAX_SAMPLES", "2000")

print("── Cargando dataset real de flotación (176 MB, decimal coma)...")
# decimal="," hace que pandas parsee los floats con coma correctamente.
# (Usar dtype=str rompe esto: "55,2" como string no lo entiende to_numeric.)
df = pd.read_csv(DATA, decimal=",", low_memory=False)
df = df.drop(columns=["date", LEAK])
for c in df.columns:
    if df[c].dtype == object:  # resto de columnas que quedaron como texto
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df[c] = df[c].astype("float32")
df = df.dropna().reset_index(drop=True)
if len(df) == 0:
    raise SystemExit("ERROR: 0 filas tras limpiar — revisá el separador decimal del CSV.")
print(f"   {df.shape[0]:,} filas, {df.shape[1]-1} features. Target: {TARGET}")

# ── Baselines naive (split temporal 80/20) ──
n = len(df); cut = int(n * 0.8)
y = df[TARGET].values.astype(float)
ytr, yte = y[:cut], y[cut:]
b_mean = r2_score(yte, np.full_like(yte, ytr.mean()))
pers = np.roll(yte, 1); pers[0] = ytr[-1]
b_pers = r2_score(yte, pers)
print(f"\n── BASELINES (lo que hay que superar):")
print(f"   media del train : R² = {b_mean:.4f}")
print(f"   persistencia     : R² = {b_pers:.4f}")

# ── Pipeline sensor-only (número honesto) ──
# Guardar CSV limpio submuestreado para el pipeline (evita recargar 176 MB).
sub = df.iloc[::8].reset_index(drop=True)   # 1/8 ≈ 90k filas, sobra
clean = os.path.join(HERE, "data", "flotation_silica_clean.csv")
sub.to_csv(clean, index=False)

from core.models.gp_model import SoftSensorGP

def run(add_lags):
    m = SoftSensorGP(target_col=TARGET, random_state=42,
                     add_lag_features=add_lags, add_diff_features=add_lags)
    t0 = time.time()
    mm = m.train_from_file(clean, test_size=0.2, n_trials=15, save_model=False)
    md = mm.to_dict() if hasattr(mm, "to_dict") else {"r2": mm.r2, "rmse": mm.rmse, "mae": mm.mae, "mape": mm.mape}
    md = {k: round(float(v), 4) for k, v in md.items()}
    md["model"] = m.model_type
    md["elapsed_s"] = round(time.time() - t0, 1)
    return md

print("\n── Pipeline SENSOR-ONLY (el número que vale para producción)...")
sensor_only = run(add_lags=False)
print("   →", sensor_only)

print("\n── Pipeline con lags del target (AR, referencia)...")
ar = run(add_lags=True)
print("   →", ar)

res = {
    "dataset": "MiningProcess_Flotation_Plant_Database.csv",
    "target": TARGET,
    "full_rows": int(n),
    "baseline_media_r2": round(float(b_mean), 4),
    "baseline_persist_r2": round(float(b_pers), 4),
    "sensor_only": sensor_only,
    "autoregressive": ar,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)

print("\n" + "=" * 60)
print("VEREDICTO:")
so = sensor_only["r2"]
print(f"  sensor-only R²={so:.4f}  vs  baseline media R²={b_mean:.4f}")
if so > b_mean + 0.15:
    print("  ✅ El modelo le GANA claramente al baseline. Hay señal real → hay caso.")
elif so > b_mean:
    print("  🟡 Le gana al baseline pero por poco. Señal débil, revisar features/alineación.")
else:
    print("  🔴 NO le gana al baseline. Sin señal sensor-only — necesita lags de INPUTS.")
print(f"\n  Artefacto guardado: {OUT}")
print("=" * 60)
