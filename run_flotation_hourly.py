import pandas as pd, numpy as np, warnings, json, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
F="data/MiningProcess_Flotation_Plant_Database.csv"
TARGET="% Silica Concentrate"; LEAK="% Iron Concentrate"
t0=time.time()

SENSORS=None; carry=None; hourly=[]
def agg(df):
    g=df.groupby("date")
    a=g[SENSORS].agg(["mean","std","min","max","last"])
    a.columns=[f"{c}_{s}" for c,s in a.columns]
    a[TARGET]=g[TARGET].first()
    return a.reset_index()

for chunk in pd.read_csv(F, decimal=",", chunksize=200000):
    chunk=chunk.drop(columns=[LEAK])
    chunk["date"]=pd.to_datetime(chunk["date"])
    if SENSORS is None:
        SENSORS=[c for c in chunk.columns if c not in ("date",TARGET)]
    if carry is not None: chunk=pd.concat([carry,chunk],ignore_index=True)
    last=chunk["date"].iloc[-1]
    done=chunk[chunk["date"]!=last]; carry=chunk[chunk["date"]==last]
    if len(done): hourly.append(agg(done))
if carry is not None and len(carry): hourly.append(agg(carry))
H=pd.concat(hourly,ignore_index=True).dropna().reset_index(drop=True)
print(f"horas agregadas: {len(H)} | features: {H.shape[1]-2} | t={time.time()-t0:.0f}s")

y=H[TARGET].values
feat=[c for c in H.columns if c not in ("date",TARGET)]
X=H[feat].values
n=len(H); cut=int(n*0.8)

# baselines HORARIOS
pers=np.roll(y,1); pers[0]=y[0]
b_pers=r2_score(y[cut:], pers[cut:])
b_mean=r2_score(y[cut:], np.full(n-cut, y[:cut].mean()))
print(f"\nBASELINE horario media   : R²={b_mean:.4f}")
print(f"BASELINE horario persist : R²={b_pers:.4f}  <-- el bar real")

def fit_eval(Xf, tag):
    Xtr,Xte,ytr,yte=Xf[:cut],Xf[cut:],y[:cut],y[cut:]
    m=GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,random_state=42)
    m.fit(Xtr,ytr); p=m.predict(Xte)
    r2=r2_score(yte,p); mae=mean_absolute_error(yte,p)
    print(f"  {tag:38} R²={r2:.4f}  MAE={mae:.4f}")
    return r2,mae

print("\nMODELO (sensores agregados por hora → sílica):")
r0,_=fit_eval(X, "nowcast (sensores hora t)")
# con lags de INPUTS (retardo de residencia): agregados de t-1..t-3
Xl=X.copy()
for k in (1,2,3):
    Xl=np.hstack([Xl, np.vstack([np.zeros((k,X.shape[1])), X[:-k]])])
r1,_=fit_eval(Xl[3:] if False else Xl, "con lags de inputs (t-1,t-2,t-3)")

res={"horas":int(n),"features_base":len(feat),
     "baseline_media":round(float(b_mean),4),"baseline_persist":round(float(b_pers),4),
     "modelo_nowcast_r2":round(float(r0),4),"modelo_inputlags_r2":round(float(r1),4)}
json.dump(res, open("flot_hourly.json","w"), indent=2)
print("\nVEREDICTO vs persistencia horaria:")
best=max(r0,r1)
print(f"  mejor modelo R²={best:.4f}  vs  persistencia R²={b_pers:.4f}")
print("  "+("✅ LE GANA A LA PERSISTENCIA — hay señal explotable, hay caso." if best>b_pers+0.02
        else "🟡 empata/no supera persistencia — la sílica horaria es casi un random walk; el valor está en nowcasting entre muestras de lab, no en forecasting."))
print("DONE", json.dumps(res))
