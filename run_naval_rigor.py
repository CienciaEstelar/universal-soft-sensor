#!/usr/bin/env python3
"""
run_naval_rigor.py — ¿El R²≈0.96 de la turbina naval es desempeño real o interpolación?

Caso NO minero traído del repo legacy (train_naval.py), donde se evaluó con un
split ALEATORIO 80/20. El dataset es SIMULADO (grilla uniforme de degradación,
sin ruido; ver prepare_naval_dataset.py), así que ese split mide interpolación
entre puntos vecinos casi idénticos. Cinco pruebas, para cada target:

  1. Split aleatorio 80/20        — réplica exacta del script legacy.
  2. GroupKFold por NIVEL de degradación — ningún nivel en train y test.
  3. Extrapolación a degradación NO vista — entrenar en el 60% más sano de los
     niveles, evaluar en el 40% más degradado. Es el uso real: detectar una
     turbina más degradada de lo que el modelo conoció. Se compara GB (árboles:
     no extrapolan por construcción) contra Ridge (lineal) y contra predecir la
     media del train.
  4. Sensibilidad a ruido de sensor — prueba 2 con ruido gaussiano multiplicativo
     de 0.1%, 0.5% y 1% en los 16 sensores. El simulador no tiene ruido; los
     niveles son SUPUESTOS de sensibilidad, no ruido medido de una planta real.
  5. GP en extrapolación — ¿ensancha σ fuera del rango visto o se equivoca con
     confianza? Protocolo del P6 (SoftSensorGP.extrapolation_test): σ_interior se
     mide en un holdout del interior, no en puntos de entrenamiento.
  5b. Prueba 5 con los mismos niveles de ruido de la prueba 4. Sin ruido el GP
     extrapola casi perfecto (el simulador es suave y determinista: el caso ideal
     para un GP); la pregunta relevante para una planta real es si eso sobrevive
     con ruido de sensor, y si la cobertura del IC 95% se mantiene fuera del rango.

⚠️ Todo R² de este script es sobre datos SIMULADOS. Reportar siempre con esa
etiqueta y nunca mezclar con resultados sobre datos reales.

Uso:
    .venv/bin/python prepare_naval_dataset.py   # una vez
    .venv/bin/python run_naval_rigor.py
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "naval_propulsion", "naval_propulsion.csv")
OUT = os.path.join(HERE, "results", "verification", "naval_rigor.json")
SEED = 42

SENSORS = ["lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
           "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"]
TARGETS = ["turbine_decay", "compressor_decay"]
HOLDOUT_FRAC = 0.4          # fracción de niveles más degradados que se extrapolan
NOISE_LEVELS = [0.001, 0.005, 0.01]
GP_N_TRAIN, GP_N_EVAL = 800, 1000
GP_RESTARTS = 2


def gb():
    # Mismos hiperparámetros que train_naval.py del repo legacy (comparabilidad).
    return make_pipeline(
        RobustScaler(),
        GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                  learning_rate=0.1, random_state=SEED),
    )


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 19)))


def metrics(y, p):
    # MAE en milésimas del coeficiente de degradación (el rango total de
    # turbine_decay es 25 milésimas; el de compressor_decay, 50).
    return {"r2": round(float(r2_score(y, p)), 4),
            "mae_milesimas": round(float(mean_absolute_error(y, p)) * 1000, 3)}


def grouped_cv(X, y):
    return cross_val_predict(gb(), X, y, groups=y, cv=GroupKFold(n_splits=5))


def gp_extrapolation(X, y, healthy, rng):
    """Protocolo P6: interior partido 80/20; σ_interior medido en el holdout."""
    # Sin columnas constantes (T1, P1): no aportan y cada una suma un
    # length_scale ARD que el optimizador tiene que ajustar en vano.
    X = X[:, X.std(axis=0) > 0]
    Xs = StandardScaler().fit(X[healthy]).transform(X)
    idx_in = rng.permutation(np.where(healthy)[0])
    n_fit = min(GP_N_TRAIN, int(len(idx_in) * 0.8))
    fit_idx, hold_idx = idx_in[:n_fit], idx_in[n_fit:n_fit + GP_N_EVAL]
    ext_idx = rng.permutation(np.where(~healthy)[0])[:GP_N_EVAL]

    y_mu, y_sd = y[fit_idx].mean(), y[fit_idx].std()
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=1.5)
              + WhiteKernel(1e-3))
    # 2 reinicios del optimizador: con ruido, un solo arranque puede caer en el
    # mínimo local "todo es ruido" (WhiteKernel absorbe la señal, R² interior≈0).
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=GP_RESTARTS, random_state=SEED)
    gp.fit(Xs[fit_idx], (y[fit_idx] - y_mu) / y_sd)

    def pred(ix):
        m, s = gp.predict(Xs[ix], return_std=True)
        return m * y_sd + y_mu, s * y_sd

    def cov(yt, m, s):
        return float(np.mean(np.abs(yt - m) <= 1.96 * s))

    mu_in, sd_in = pred(hold_idx)
    mu_ex, sd_ex = pred(ext_idx)
    std_ratio = float(sd_ex.mean() / sd_in.mean())
    return {
        "n_fit": int(n_fit), "n_holdout_interior": int(len(hold_idx)),
        "n_exterior": int(len(ext_idx)),
        "interior": {**metrics(y[hold_idx], mu_in),
                     "coverage95": round(cov(y[hold_idx], mu_in, sd_in), 3)},
        "exterior": {**metrics(y[ext_idx], mu_ex),
                     "coverage95": round(cov(y[ext_idx], mu_ex, sd_ex), 3),
                     "pred_min": round(float(mu_ex.min()), 4)},
        "std_ratio_ext_int": round(std_ratio, 2),
        "graceful": bool(std_ratio > 1),
        "kernel": str(gp.kernel_),
    }


def run_target(df, target):
    X = df[SENSORS].values.astype(float)
    y = df[target].values.astype(float)
    levels = np.sort(np.unique(y))
    n_hold = int(round(len(levels) * HOLDOUT_FRAC))
    threshold = levels[n_hold]              # train: y >= threshold (lado sano)
    healthy = y >= threshold
    rng = np.random.default_rng(SEED)
    res = {"n": int(len(y)), "n_niveles": int(len(levels)),
           "rango": [float(levels[0]), float(levels[-1])]}

    print(f"\n══ TARGET: {target} — {len(levels)} niveles [{levels[0]}–{levels[-1]}] ══")

    # 1. Réplica del split aleatorio legacy
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    res["1_split_aleatorio"] = metrics(yte, gb().fit(xtr, ytr).predict(xte))
    print(f"  1. split aleatorio (legacy):          R²={res['1_split_aleatorio']['r2']:.3f}")

    # 2. GroupKFold por nivel de degradación
    res["2_groupkfold_por_nivel"] = metrics(y, grouped_cv(X, y))
    print(f"  2. GroupKFold por nivel:              R²={res['2_groupkfold_por_nivel']['r2']:.3f}")

    # 3. Extrapolación a degradación no vista
    Xtr, ytr, Xte, yte = X[healthy], y[healthy], X[~healthy], y[~healthy]
    p_gb = gb().fit(Xtr, ytr).predict(Xte)
    p_rd = ridge().fit(Xtr, ytr).predict(Xte)
    res["3_extrapolacion"] = {
        "train": f"{target} >= {threshold} ({int(healthy.sum())} filas, "
                 f"{len(levels) - n_hold} niveles)",
        "test": f"{target} < {threshold} ({int((~healthy).sum())} filas, {n_hold} niveles)",
        "real_min": float(yte.min()),
        "gradient_boosting": {**metrics(yte, p_gb), "pred_min": round(float(p_gb.min()), 4)},
        "ridge_lineal": {**metrics(yte, p_rd), "pred_min": round(float(p_rd.min()), 4)},
        "media_del_train": metrics(yte, np.full_like(yte, ytr.mean())),
    }
    e = res["3_extrapolacion"]
    print(f"  3. extrapolación (<{threshold}):  GB R²={e['gradient_boosting']['r2']:.3f} "
          f"(pred mín {e['gradient_boosting']['pred_min']} vs real {e['real_min']}) | "
          f"Ridge R²={e['ridge_lineal']['r2']:.3f} | media R²={e['media_del_train']['r2']:.3f}")

    # 4. Sensibilidad a ruido de sensor (supuesto, no medido)
    res["4_ruido_sensor"] = {}
    for lvl in NOISE_LEVELS:
        Xn = X * (1 + np.random.default_rng(SEED).normal(0, lvl, X.shape))
        res["4_ruido_sensor"][f"{lvl:.1%}"] = metrics(y, grouped_cv(Xn, y))
    print("  4. ruido de sensor (GroupKFold):      " + " | ".join(
        f"{k}: R²={v['r2']:.3f}" for k, v in res["4_ruido_sensor"].items()))

    # 5. GP en extrapolación
    res["5_gp_extrapolacion"] = g = gp_extrapolation(X, y, healthy, rng)
    print(f"  5. GP extrapolación:  interior R²={g['interior']['r2']:.3f} "
          f"cov95={g['interior']['coverage95']:.2f} | exterior R²={g['exterior']['r2']:.3f} "
          f"cov95={g['exterior']['coverage95']:.2f} | σ_ext/σ_int={g['std_ratio_ext_int']} "
          f"→ {'ensancha σ (degrada con gracia)' if g['graceful'] else '🔴 NO ensancha σ'}")

    # 5b. GP en extrapolación con ruido de sensor (supuesto, no medido)
    res["5b_gp_extrapolacion_con_ruido"] = {}
    for lvl in NOISE_LEVELS:
        Xn = X * (1 + np.random.default_rng(SEED).normal(0, lvl, X.shape))
        g = gp_extrapolation(Xn, y, healthy, np.random.default_rng(SEED))
        res["5b_gp_extrapolacion_con_ruido"][f"{lvl:.1%}"] = g
        print(f"  5b. GP extrap. ruido {lvl:.1%}:  interior R²={g['interior']['r2']:.3f} "
              f"| exterior R²={g['exterior']['r2']:.3f} cov95={g['exterior']['coverage95']:.2f} "
              f"| σ_ext/σ_int={g['std_ratio_ext_int']}")
    return res


def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"❌ Falta {DATA}. Correr antes: .venv/bin/python prepare_naval_dataset.py")
    t0 = time.time()
    df = pd.read_csv(DATA)
    print("⚠️  Dataset SIMULADO (UCI 316, simulador de fragata) — no es evidencia de planta.")
    results = {
        "_meta": {
            "dataset": "UCI 316 Condition Based Maintenance of Naval Propulsion Plants "
                       "(Coraddu et al. 2016) — SIMULADO, CC BY 4.0",
            "sklearn": sklearn.__version__,
            "seed": SEED,
            "holdout_frac_extrapolacion": HOLDOUT_FRAC,
            "nota_ruido": "niveles de ruido = supuestos de sensibilidad, no medidos",
        }
    }
    for tg in TARGETS:
        results[tg] = run_target(df, tg)
    results["_meta"]["elapsed_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nArtefacto: {OUT}  ({results['_meta']['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
