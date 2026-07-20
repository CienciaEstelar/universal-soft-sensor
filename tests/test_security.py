"""
Tests de regresión de seguridad — cubren los vectores de la auditoría hostil
(SECURITY_AUDIT.md). Cada test verifica que un ataque conocido queda bloqueado.
No deben volver a pasar silenciosamente.
"""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from core.models.gp_model import SoftSensorGP


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def _train(df, target, tmp_path, **kw):
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    m = SoftSensorGP(target_col=target, random_state=42,
                     add_lag_features=False, add_diff_features=False, **kw)
    metrics = _quiet(m.train_from_file, str(p), test_size=0.2, n_trials=1, save_model=False)
    return m, metrics


def test_v2_leakage_feature_copy_is_dropped(tmp_path):
    """V2: una feature copia de un target ALEATORIO no debe producir R²≈1."""
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "press_b": rng.uniform(1, 5, n),
        "yq": rng.uniform(0, 100, n),   # target impredecible
    })
    df["leak_copy"] = df["yq"]          # leakage exacto
    _, metrics = _train(df, "yq", tmp_path)
    r2 = metrics.r2 if hasattr(metrics, "r2") else metrics["r2"]
    # sin la columna filtrada, un target aleatorio NO se puede predecir
    assert r2 < 0.5, f"leakage no neutralizado: R2={r2}"


def test_v2_strict_leakage_aborts(tmp_path):
    """V2: con strict_leakage=True el leakage debe abortar con error."""
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "yq": rng.uniform(0, 100, n),
    })
    df["leak_copy"] = df["yq"]
    with pytest.raises(ValueError):
        _train(df, "yq", tmp_path, strict_leakage=True)


def test_v3_zero_variance_test_split_r2_is_nan():
    """V3: si el TEST tiene varianza cero (degeneración tipo ZeMA), evaluate()
    debe reportar R²=NaN, no un número engañoso. Se prueba evaluate() directo
    porque este caso surge del SPLIT, no del target global (que V5 ya rechaza)."""
    m = SoftSensorGP(target_col="y")
    y_true = np.full(50, 7.0)            # test constante (SS_tot = 0)
    y_pred = np.full(50, 7.0)            # predicción "perfecta" trivial
    metrics = m.evaluate(y_true, y_pred)
    r2 = metrics.r2 if hasattr(metrics, "r2") else metrics["r2"]
    assert np.isnan(r2), f"R² sobre test de varianza cero debería ser NaN, fue {r2}"


def test_v3_all_zero_target_mape_is_nan():
    """V3: MAPE sobre y_true todo-cero debe ser NaN, no 0.0 ('perfecto')."""
    m = SoftSensorGP(target_col="y")
    y_true = np.zeros(50)
    y_pred = np.random.default_rng(1).normal(0, 1, 50)
    metrics = m.evaluate(y_true, y_pred)
    mape = metrics.mape if hasattr(metrics, "mape") else metrics["mape"]
    assert np.isnan(mape), f"MAPE de y_true todo-cero debería ser NaN, fue {mape}"


def test_v4_path_traversal_blocked(tmp_path):
    """V4: un filename con ../ en el config no debe escapar de data/."""
    import json
    import os
    from pathlib import Path
    from core.adapters.universal_adapter import UniversalAdapter

    cfg_dir = Path(__file__).parent.parent / "config"
    base = json.load(open(cfg_dir / "dataset_config.json"))
    base["files"]["filename"] = "../../../../../../etc/passwd"
    evil = cfg_dir / "_evil_test.json"
    evil.write_text(json.dumps(base))
    try:
        with pytest.raises(ValueError):
            UniversalAdapter("_evil_test.json")
    finally:
        try:
            os.remove(evil)
        except OSError:
            pass


def test_v1_load_unsigned_pkl_blocked(tmp_path):
    """V1: cargar un .pkl sin hash de referencia debe bloquearse por defecto."""
    rng = np.random.default_rng(6)
    n = 150
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "yq": rng.uniform(0, 100, n),
    })
    m, _ = _train(df, "yq", tmp_path)
    pkl = tmp_path / "model.pkl"
    _quiet(m.save, str(pkl))
    # borrar el sidecar → carga sin hash de referencia
    (tmp_path / "model.pkl.sha256").unlink()
    with pytest.raises(ValueError):
        _quiet(SoftSensorGP(target_col="yq").load, str(pkl))


def test_v1_tampered_pkl_blocked(tmp_path):
    """V1: un .pkl alterado (hash != sidecar) debe abortar la carga."""
    rng = np.random.default_rng(7)
    n = 150
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "yq": rng.uniform(0, 100, n),
    })
    m, _ = _train(df, "yq", tmp_path)
    pkl = tmp_path / "model.pkl"
    _quiet(m.save, str(pkl))
    pkl.write_bytes(pkl.read_bytes() + b"tampered")  # altera el binario
    with pytest.raises(ValueError):
        _quiet(SoftSensorGP(target_col="yq").load, str(pkl))


def test_v5_non_numeric_target_friendly_error(tmp_path):
    """V5: target string debe dar error claro, no stacktrace de sklearn."""
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": np.random.uniform(20, 80, n),
        "ys": ["texto"] * n,
    })
    with pytest.raises(ValueError, match="(?i)numérico"):
        _train(df, "ys", tmp_path)


def test_v5_too_few_rows_friendly_error(tmp_path):
    """V5: dataset diminuto debe dar error claro de datos insuficientes."""
    n = 4
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": np.random.uniform(20, 80, n),
        "yq": np.random.uniform(0, 100, n),
    })
    with pytest.raises(ValueError, match="(?i)insuficiente"):
        _train(df, "yq", tmp_path)


def test_v5_inf_target_friendly_error(tmp_path):
    """V5: target con inf debe dar error claro, no 'Input X contains infinity'."""
    rng = np.random.default_rng(9)
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "yq": rng.uniform(0, 100, n),
    })
    df.loc[3:8, "yq"] = np.inf
    with pytest.raises(ValueError, match="(?i)infinit"):
        _train(df, "yq", tmp_path)


def test_v6_sample_cap_is_deterministic(tmp_path):
    """V6: el muestreo aleatorio del cap O(n³) debe ser determinista (seedeado)."""
    rng = np.random.default_rng(10)
    n = 400
    df = pd.DataFrame({
        "date": pd.date_range("2023", periods=n, freq="h"),
        "temp_a": rng.uniform(20, 80, n),
        "press_b": rng.uniform(1, 5, n),
    })
    df["yq"] = df["temp_a"] * 0.5 + rng.normal(0, 3, n)
    from config.settings import CONFIG
    old = CONFIG.GP_MAX_TRAIN_SAMPLES
    CONFIG.GP_MAX_TRAIN_SAMPLES = 100  # forzar el cap O(n³)
    try:
        _, m1 = _train(df, "yq", tmp_path)
        _, m2 = _train(df, "yq", tmp_path)
    finally:
        CONFIG.GP_MAX_TRAIN_SAMPLES = old
    r1 = m1.r2 if hasattr(m1, "r2") else m1["r2"]
    r2 = m2.r2 if hasattr(m2, "r2") else m2["r2"]
    assert abs(r1 - r2) < 1e-9, f"muestreo del cap no determinista: {r1} vs {r2}"
