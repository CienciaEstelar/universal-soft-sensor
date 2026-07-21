#!/usr/bin/env python3
"""
run_geomet_recovery.py — ¿La química de mineral predice la RECUPERACIÓN de
flotación? (GeoMet, Zenodo 10.5281/zenodo.7051975)

flotation.csv es autocontenido: 53 muestras con su química (columnas '* ppm')
Y su respuesta de flotación (fr, xr, LCT). Test directo: química → recuperación.
Opcional: suma dureza/conminución (F80, P80, M, A...) por HOLEID.

Geometalúrgico = una fila por muestra, sin tiempo → split aleatorio, baseline
media, 5-fold CV. n=53 es CHICO: el R² es indicativo, no definitivo.

Uso:
    .venv/bin/python run_geomet_recovery.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "data", "geomet")
OUT = os.path.join(HERE, "results", "verification", "geomet_recovery.json")

flot = pd.read_csv(os.path.join(G, "flotation.csv"))
comm = pd.read_csv(os.path.join(G, "comminution.csv"))
print(f"── flotation: {flot.shape} | comminution: {comm.shape}")

# respuestas de flotación candidatas a target
RESPONSES = [c for c in ["fr", "xr", "LCT"] if c in flot.columns]
# features de química: columnas con 'ppm'
CHEM = [c for c in flot.columns if "ppm" in c.lower()]
COORDS = [c for c in ["X", "Y", "Z"] if c in flot.columns]
print(f"── respuestas (target): {RESPONSES}")
print(f"── química (features): {len(CHEM)} columnas")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
MODELS = {
    "Lineal": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                        learning_rate=0.05, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
}

def evaluate(df, features, target, tag):
    d = df[features + [target]].dropna()
    if len(d) < 20:
        print(f"   [{tag}] {target}: solo {len(d)} muestras válidas — se omite")
        return None
    X, y = d[features].values.astype(float), d[target].values.astype(float)
    row = {"n": int(len(d)), "target_mean": round(float(y.mean()), 3),
           "target_std": round(float(y.std()), 3)}
    print(f"\n   ── {tag} → target '{target}' (n={len(d)}, {len(features)} feat, "
          f"rango [{y.min():.2f},{y.max():.2f}]):")
    best = -np.inf
    for name, m in MODELS.items():
        pred = cross_val_predict(m, X, y, cv=kf)
        r2 = r2_score(y, pred); mae = mean_absolute_error(y, pred)
        print(f"        {name:<18} R²={r2:>8.4f}  MAE={mae:>7.3f}")
        row[name] = {"r2": round(float(r2), 4), "mae": round(float(mae), 3)}
        best = max(best, r2)
    row["best_r2"] = round(float(best), 4)
    return row

results = {}
print("\n══════ SOLO QUÍMICA (features = * ppm) ══════")
for tg in RESPONSES:
    r = evaluate(flot, CHEM, tg, "quimica")
    if r: results[f"chem::{tg}"] = r

# ── opcional: sumar dureza/conminución por HOLEID ──
COMM_FEAT = [c for c in comm.columns if c not in (["HOLEID", "X", "Y", "Z"])]
merged = flot.merge(comm[["HOLEID"] + COMM_FEAT], on="HOLEID", how="inner", suffixes=("", "_comm"))
print(f"\n══════ QUÍMICA + CONMINUCIÓN (merge por HOLEID: {len(merged)} muestras) ══════")
if len(merged) >= 20:
    feats2 = CHEM + [c for c in COMM_FEAT if c in merged.columns]
    for tg in RESPONSES:
        r = evaluate(merged, feats2, tg, "quimica+dureza")
        if r: results[f"chem+comm::{tg}"] = r
else:
    print(f"   overlap muy chico ({len(merged)} muestras) — se omite el merge.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)

# ── veredicto ──
print("\n" + "=" * 62)
best_key = max(results, key=lambda k: results[k]["best_r2"]) if results else None
best_r2 = results[best_key]["best_r2"] if best_key else -np.inf
print(f"Mejor combinación: {best_key}  →  R²={best_r2:.3f}  (n={results[best_key]['n']})" if best_key else "sin resultados")
if best_r2 > 0.5:
    print("✅ HAY SEÑAL FUERTE: la química predice la recuperación de flotación.")
    print("   Este es el caso de negocio: soft-sensor geometalúrgico de recuperación.")
elif best_r2 > 0.2:
    print("🟡 SEÑAL MODERADA. Hay algo real, pero con n=53 es indicativo.")
    print("   Justifica pedir más muestras / un dataset de planta real para confirmar.")
else:
    print("🔴 SIN SEÑAL clara con estas features. La recuperación tampoco es")
    print("   predecible desde la química acá. Es una respuesta honesta de negocio.")
print("\n⚠️  n=53 es una muestra chica: tomar el número como señal de dirección,")
print("   no como métrica definitiva. Un R² alto acá justifica el siguiente paso.")
print(f"\nArtefacto: {OUT}")
print("=" * 62)
