#!/usr/bin/env python3
"""
run_geomet_rigor.py — ¿El R²=0.93 de la recuperación es REAL o autoengaño?

Tres tests de rigor sobre el resultado geometalúrgico (GeoMet):
  1. Integridad del join: ¿el merge flotation↔comminution duplica muestras?
     (si sí, un KFold aleatorio filtra el mismo HOLEID entre train y test).
  2. GroupKFold por HOLEID: el split HONESTO — ningún sondaje en train y test.
  3. Permutation test: barajar el target 200 veces; si el R² real no está muy
     por encima del R² con target barajado (p<0.05), es sobreajuste/leakage.
  + Chequeo de tautología: ¿'xr' es casi una copia de la ley de Cu de entrada?

Uso:
    .venv/bin/python run_geomet_rigor.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "data", "geomet")
OUT = os.path.join(HERE, "results", "verification", "geomet_rigor.json")
RNG = np.random.default_rng(42)

flot = pd.read_csv(os.path.join(G, "flotation.csv"))
comm = pd.read_csv(os.path.join(G, "comminution.csv"))
CHEM = [c for c in flot.columns if "ppm" in c.lower()]
TARGETS = [c for c in ["xr", "fr", "LCT"] if c in flot.columns]

# ── 1. Integridad del join ──
print("══ 1. INTEGRIDAD DEL JOIN ══")
print(f"  flotation: {len(flot)} filas, {flot['HOLEID'].nunique()} HOLEID únicos")
print(f"  comminution: {len(comm)} filas, {comm['HOLEID'].nunique()} HOLEID únicos")
comm_per_hole = comm.groupby("HOLEID").size()
print(f"  filas de comminution por HOLEID: min={comm_per_hole.min()} max={comm_per_hole.max()}")
merged = flot.merge(comm.drop(columns=[c for c in ["X","Y","Z"] if c in comm.columns]),
                    on="HOLEID", how="inner", suffixes=("", "_c"))
dup = len(merged) > len(flot)
print(f"  merge → {len(merged)} filas (flotation tenía {len(flot)})")
print(f"  {'🔴 EL MERGE DUPLICA muestras → KFold aleatorio FILTRA por HOLEID' if dup else '🟢 join 1:1, sin duplicación'}")

COMM_FEAT = [c for c in comm.columns if c not in (["HOLEID","X","Y","Z"]) and c in merged.columns]

def grouped_r2(df, feats, target):
    d = df[feats + [target, "HOLEID"]].dropna()
    X, y, grp = d[feats].values.astype(float), d[target].values.astype(float), d["HOLEID"].values
    ng = len(np.unique(grp))
    k = min(5, ng)
    gkf = GroupKFold(n_splits=k)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
    pred = cross_val_predict(m, X, y, cv=gkf, groups=grp)
    return r2_score(y, pred), len(d), ng, (X, y, grp, k)

def perm_test(X, y, grp, k, real_r2, n=200):
    gkf = GroupKFold(n_splits=k)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
    ge = 0
    for _ in range(n):
        yp = RNG.permutation(y)
        pr = cross_val_predict(m, X, yp, cv=gkf, groups=grp)
        if r2_score(yp, pr) >= real_r2: ge += 1
    return (ge + 1) / (n + 1)

results = {}
for feats, tag in [(CHEM, "quimica"), (CHEM + COMM_FEAT, "quimica+dureza")]:
    print(f"\n══ 2+3. {tag.upper()} — GroupKFold por HOLEID (split honesto) ══")
    for tg in TARGETS:
        r2, n, ng, packed = grouped_r2(merged if "dureza" in tag else flot, feats, tg)
        p = perm_test(*packed, r2)
        verdict = "✅ señal REAL" if (r2 > 0.3 and p < 0.05) else ("🟡 débil/dudosa" if r2 > 0.1 else "🔴 sin señal")
        print(f"  {tg:>4}: R²(grouped)={r2:>7.4f}  p-perm={p:.3f}  n={n} holes={ng}  → {verdict}")
        results[f"{tag}::{tg}"] = {"grouped_r2": round(float(r2),4), "p_perm": round(float(p),4),
                                   "n": n, "n_holes": ng}

# ── chequeo de tautología: xr vs ley de Cu ──
print("\n══ 4. TAUTOLOGÍA — ¿el target es casi una feature? ══")
cu = [c for c in flot.columns if c.strip().lower().startswith("cu")]
for tg in TARGETS:
    line=[]
    for c in cu:
        r = flot[[c, tg]].dropna().corr().iloc[0,1]
        line.append(f"{c}: r={r:+.2f}")
    print(f"  {tg} vs {' | '.join(line) if line else 'sin col Cu'}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print("\n" + "="*62)
best = max(results, key=lambda k: results[k]["grouped_r2"])
b = results[best]
real = b["grouped_r2"] > 0.3 and b["p_perm"] < 0.05
print(f"Mejor split honesto: {best} → R²={b['grouped_r2']:.3f}, p={b['p_perm']:.3f}")
if real:
    print("✅ SEÑAL CONFIRMADA con split honesto y permutation test. El caso es real.")
    print("   (aún con n chico: confirmar con más muestras, pero la dirección aguanta el rigor.)")
else:
    print("🔴 El R² se cae con el split honesto (GroupKFold) o no supera el permutation test.")
    print("   El 0.93 del KFold aleatorio era leakage por HOLEID duplicado. Sin caso confirmado aún.")
print(f"\nArtefacto: {OUT}")
print("="*62)
