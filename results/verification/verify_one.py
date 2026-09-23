"""Corre el pipeline REAL (SoftSensorGP.train_from_file) sobre un CSV. Uso:
python3 verify_one.py <csv> <target> <out_json> <drop_cols_comma|-> <n_trials>"""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
csv, target, out_json, drops, trials = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.environ["GP_TARGET"] = target
os.environ["GP_MAX_SAMPLES"] = os.environ.get("VMAX","600")
CODEBASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODEBASE); os.chdir(CODEBASE)
import pandas as pd
df = pd.read_csv(csv)
if drops != "-":
    df = df.drop(columns=[c for c in drops.split(",") if c in df.columns])
tmp = out_json.replace(".json", "_data.csv")
df.to_csv(tmp, index=False)
from core.models.gp_model import SoftSensorGP
t0 = time.time()
m = SoftSensorGP(random_state=42)
metrics = m.train_from_file(tmp, test_size=0.2, n_trials=trials, save_model=False)
md = metrics.to_dict() if hasattr(metrics, "to_dict") else {"r2": metrics.r2, "rmse": metrics.rmse, "mae": metrics.mae, "mape": metrics.mape}
res = {"dataset": os.path.basename(csv), "target": target, "n_rows": len(df),
       "model_used": getattr(m, "model_type", getattr(m, "active_model", "?")),
       "elapsed_s": round(time.time()-t0, 1),
       "metrics": {k: round(float(v), 6) for k, v in md.items()}}
json.dump(res, open(out_json, "w"), indent=2)
print("DONE:", json.dumps(res))
