#!/usr/bin/env python3
"""
run_flotation_horizons.py — La corrida que decide si hay caso de negocio.

Pregunta: ¿a qué horizonte (h) el modelo con sensores SUPERA a la persistencia?
La persistencia gana a 1h (target deriva lento), pero se derrumba a horizontes
largos. Si el modelo con sensores le gana a 3/6/12/24h, ahí está la ventana de valor.

Prueba dos targets: % Silica Concentrate y % Iron Concentrate.
Agrega los ~174 samples/hora en features (mean/std/min/max/last) y cachea el
resultado en data/flotation_hourly_cache.parquet para no releer los 176 MB.

Uso:
    .venv/bin/python run_flotation_horizons.py
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "MiningProcess_Flotation_Plant_Database.csv")
CACHE = os.path.join(HERE, "data", "flotation_hourly_cache.parquet")
OUT = os.path.join(HERE, "results", "verification", "flotation_horizons.json")
TARGETS = ["% Silica Concentrate", "% Iron Concentrate"]
HORIZONS = [1, 3, 6, 12, 24]  # horas hacia adelante

# ── 1. Agregación horaria (con cache) ──
if os.path.exists(CACHE):
    print(f"── Usando cache: {CACHE}")
    H = pd.read_parquet(CACHE)
else:
    print("── Agregando dataset por hora (streaming, 176 MB, una sola vez)...")
    t0 = time.time()
    SENSORS = None; carry = None; parts = []
    def agg(df):
        g = df.groupby("date")
        a = g[SENSORS].agg(["mean", "std", "min", "max", "last"])
        a.columns = [f"{c}__{s}" for c, s in a.columns]
        for tg in TARGETS:
            a[tg] = g[tg].first()
        return a.reset_index()
    for chunk in pd.read_csv(RAW, decimal=",", chunksize=200000):
        chunk["date"] = pd.to_datetime(chunk["date"])
        if SENSORS is None:
            SENSORS = [c for c in chunk.columns if c not in (["date"] + TARGETS)]
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["date"].iloc[-1]
        done = chunk[chunk["date"] != last]; carry = chunk[chunk["date"] == last]
        if len(done):
            parts.append(agg(done))
    if carry is not None and len(carry):
        parts.append(agg(carry))
    H = pd.concat(parts, ignore_index=True).dropna().reset_index(drop=True)
    H = H.sort_values("date").reset_index(drop=True)
    H.to_parquet(CACHE)
    print(f"   {len(H)} horas, {H.shape[1]-1-len(TARGETS)} features. Cache guardado. t={time.time()-t0:.0f}s")

feat = [c for c in H.columns if c not in (["date"] + TARGETS)]
Xall = H[feat].values.astype(float)
print(f"── {len(H)} horas · {len(feat)} features · targets: {TARGETS}\n")

def eval_h(y, X, h):
    """Predice y[t+h] desde X[t]. Persistencia = y[t]. Split temporal 80/20."""
    n = len(y)
    Xf, yf, yp = X[:n-h], y[h:], y[:n-h]   # yp = persistencia (valor actual)
    cut = int(len(Xf) * 0.8)
    m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                  learning_rate=0.05, random_state=42)
    m.fit(Xf[:cut], yf[:cut])
    pred = m.predict(Xf[cut:])
    r2_model = r2_score(yf[cut:], pred)
    r2_pers = r2_score(yf[cut:], yp[cut:])
    return r2_model, r2_pers

results = {}
for tg in TARGETS:
    y = H[tg].values.astype(float)
    print(f"══ Target: {tg} ══")
    print(f"  {'horizonte':>10} | {'modelo R²':>10} | {'persist R²':>10} | veredicto")
    print("  " + "-"*52)
    results[tg] = {}
    for h in HORIZONS:
        rm, rp = eval_h(y, Xall, h)
        win = "✅ MODELO GANA" if rm > rp + 0.02 else ("≈ empate" if rm > rp - 0.02 else "persist gana")
        print(f"  {str(h)+'h':>10} | {rm:>10.4f} | {rp:>10.4f} | {win}")
        results[tg][f"{h}h"] = {"model_r2": round(float(rm), 4), "persist_r2": round(float(rp), 4)}
    print()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)

# ── Veredicto global ──
print("="*60)
best = None
for tg in TARGETS:
    for h in HORIZONS:
        d = results[tg][f"{h}h"]
        gap = d["model_r2"] - d["persist_r2"]
        if d["model_r2"] > 0.1 and gap > 0.02 and (best is None or gap > best[3]):
            best = (tg, h, d["model_r2"], gap)
if best:
    print(f"✅ HAY CASO: prediciendo {best[0]} a {best[1]}h, el modelo (R²={best[2]:.3f})")
    print(f"   le gana a la persistencia por +{best[3]:.3f}. Esa es tu ventana de valor.")
else:
    print("🔴 En NINGÚN horizonte/target el modelo supera claramente a la persistencia.")
    print("   Este dataset no es sensor-predecible → pivotar a recuperación (GeoMet) o")
    print("   posicionar el valor en nowcasting entre muestras de lab, no en forecasting.")
print(f"\nArtefacto: {OUT}")
print("="*60)
