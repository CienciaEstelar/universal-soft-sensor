"""
verify_geomet_number.py — zanja la reproducibilidad del R² de GeoMet cobre.

CONTEXTO: en el entorno de desarrollo con sklearn 1.7.2 (Python 3.10) el R²
reproduce en ~0.10-0.11, no en el 0.319 documentado (generado con sklearn 1.9.0,
Python 3.12). Este script mide el número EN TU ENTORNO por dos caminos
independientes y reporta la versión de sklearn, para determinar si el 0.319 es
correcto en tu .venv o si el resultado es frágil a la versión de librería.

USO (desde la raíz del proyecto, con tu .venv activado):
    source .venv/bin/activate
    python verify_geomet_number.py

Interpretación:
    - Si ambos R² dan ~0.31-0.33  → el 0.319 reproduce en tu entorno (sklearn 1.9.0);
      el problema es solo de versión y hay que PINEAR sklearn en requirements.txt.
    - Si dan ~0.10-0.11           → el 0.319 NO reproduce ni siquiera en tu .venv;
      el número oficial hay que corregirlo a ~0.11 (débil/dudoso por el propio
      rubro del proyecto) en dashboard/README/ROADMAP/CLAUDE.md.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

print(f"sklearn: {sklearn.__version__}  |  numpy: {np.__version__}")
print("=" * 64)

# ── Camino 1: script de auditoría exacto (run_geomet_rigor.py) ──────────────
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "geomet")
flot = pd.read_csv(os.path.join(G, "flotation.csv"))
comm = pd.read_csv(os.path.join(G, "comminution.csv"))
CHEM = [c for c in flot.columns if "ppm" in c.lower()]
merged = flot.merge(
    comm.drop(columns=[c for c in ["X", "Y", "Z"] if c in comm.columns]),
    on="HOLEID", how="inner", suffixes=("", "_c"),
)
COMM_FEAT = [c for c in comm.columns if c not in ["HOLEID", "X", "Y", "Z"] and c in merged.columns]
feats = CHEM + COMM_FEAT
d = merged[feats + ["LCT", "HOLEID"]].dropna()
X = d[feats].values.astype(float)
y = d["LCT"].values.astype(float)
grp = d["HOLEID"].values
k = min(5, len(np.unique(grp)))
gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
pred = cross_val_predict(gb, X, y, cv=GroupKFold(n_splits=k), groups=grp)
r2_audit = r2_score(y, pred)
print(f"[1] Auditoría (run_geomet_rigor.py, {X.shape[1]} feats, n={len(d)}, {len(np.unique(grp))} sondajes):")
print(f"    R² grouped = {r2_audit:.4f}")

# permutation test rápido (50 barajados) para el p-value
rng = np.random.default_rng(42)
ge = 0
NPERM = 50
for _ in range(NPERM):
    yp = rng.permutation(y)
    pr = cross_val_predict(gb, X, yp, cv=GroupKFold(n_splits=k), groups=grp)
    if r2_score(yp, pr) >= r2_audit:
        ge += 1
p_audit = (ge + 1) / (NPERM + 1)
print(f"    p-value  = {p_audit:.4f}  ({NPERM} permutaciones)")

# ── Camino 2: pipeline con el adapter (el que produjo el 0.319) ─────────────
try:
    import tempfile
    from core.adapters import DataAdapter
    from core.models.gp_model import SoftSensorGP

    adapter = DataAdapter("dataset_config.json")
    df = adapter.load_data()
    tmp = os.path.join(tempfile.mkdtemp(), "clean.csv")
    df.to_csv(tmp, index=False)
    m = SoftSensorGP(target_col="LCT", add_lag_features=False, add_diff_features=False,
                     subsample_step=1, parse_dates=False, group_column="HOLEID")
    Xa, ya, _ = m.load_data(filepath=tmp)
    res = m.permutation_test(Xa, ya, groups=m.groups_, n_permutations=50)
    print(f"[2] Pipeline adapter ({Xa.shape[1]} feats, n={Xa.shape[0]}):")
    print(f"    R² = {res['real_r2']:.4f}   p-value = {res['p_value']:.4f}  (50 permutaciones)")
except Exception as e:
    print(f"[2] Pipeline adapter: no se pudo correr ({e})")

print("=" * 64)
print("VEREDICTO:")
if r2_audit > 0.28:
    print(f"  ✅ El 0.319 REPRODUCE en tu entorno (R²={r2_audit:.3f}). Es cuestión de versión.")
    print("     → Pinear sklearn en requirements.txt y documentar la versión.")
elif r2_audit > 0.15:
    print(f"  🟡 Reproduce PARCIAL (R²={r2_audit:.3f}) — entre el 0.11 y el 0.32. Frágil.")
else:
    print(f"  🔴 NO reproduce (R²={r2_audit:.3f}). El 0.319 hay que corregirlo en las docs.")
