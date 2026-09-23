#!/usr/bin/env python3
"""
prepare_naval_dataset.py — Descarga y prepara data/naval_propulsion/naval_propulsion.csv

Caso NO minero: soft-sensor de degradación de turbina a gas (propulsión naval).
Traído desde el repo legacy proyecto_minero_4.0 (train_naval.py, 2026-07-10), donde
se evaluó con un split aleatorio 80/20 (R²≈0.96). Aquí se re-evalúa con rigor en
run_naval_rigor.py.

⚠️ ADVERTENCIA CRÍTICA — ESTE DATASET ES SIMULADO, NO REAL:
Proviene de:
    Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., Figari, M. (2016).
    "Machine learning approaches for improving condition-based maintenance of
    naval propulsion plants." Proc. IMechE Part M: J. Engineering for the
    Maritime Environment, 230(1), 136-153.
Repositorio: UCI Machine Learning Repository, dataset 316 ("Condition Based
Maintenance of Naval Propulsion Plants"). Licencia CC BY 4.0.

Los datos salen de un SIMULADOR NUMÉRICO de una fragata con propulsión por turbina
a gas (según el README upstream: "numerical simulator of a naval vessel (Frigate)").
Los estados de degradación están muestreados en una GRILLA UNIFORME:
    - kMc (compresor): [0.95, 1.00], paso 0.001  → 51 niveles
    - kMt (turbina):   [0.975, 1.00], paso 0.001 → 26 niveles
    - velocidad:       3 a 27 nudos, paso 3      → 9 niveles
    = 11,934 puntos en régimen estacionario, SIN ruido de sensor y SIN dimensión
      temporal (no es una serie de tiempo: cada fila es un punto de la grilla).

Consecuencias para la evaluación (ver run_naval_rigor.py):
    - Un split aleatorio mide INTERPOLACIÓN en una grilla densa, no desempeño en
      una turbina real.
    - El uso real (detectar degradación peor que la vista) es EXTRAPOLACIÓN.
    - Cualquier R² obtenido aquí debe reportarse SIEMPRE con la etiqueta "simulado"
      y nunca mezclarse con resultados sobre datos reales (GeoMet cobre, ZeMA, SRU).

Diferencias con el CSV del repo legacy: aquel agregaba una columna 'date' inventada
(2020-01-01 cada 1h) que NO existe en el dataset original y sugería una serie de
tiempo falsa. Aquí no se agrega: el CSV tiene solo las 18 columnas originales.

Uso:
    .venv/bin/python prepare_naval_dataset.py
    .venv/bin/python prepare_naval_dataset.py --from-zip ruta/al/archivo.zip   # sin red
"""
import argparse
import hashlib
import io
import os
import urllib.request
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "naval_propulsion")
OUT_CSV = os.path.join(OUT_DIR, "naval_propulsion.csv")
ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/316/"
    "condition+based+maintenance+of+naval+propulsion+plants.zip"
)
DATA_MEMBER = "UCI CBM Dataset/data.txt"
# SHA-256 de data.txt verificado el 2026-09-22 contra la descarga oficial de UCI
# (idéntico al archivo usado en el repo legacy proyecto_minero_4.0).
DATA_SHA256 = "de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6"

# Orden de columnas según Features.txt del dataset upstream.
COLUMNS = [
    "lp",     # Lever position [ ]
    "v",      # Ship speed [knots]
    "GTT",    # Gas Turbine shaft torque [kN m]
    "GTn",    # Gas Turbine rate of revolutions [rpm]
    "GGn",    # Gas Generator rate of revolutions [rpm]
    "Ts",     # Starboard Propeller Torque [kN]
    "Tp",     # Port Propeller Torque [kN]
    "T48",    # HP Turbine exit temperature [C]
    "T1",     # GT Compressor inlet air temperature [C]
    "T2",     # GT Compressor outlet air temperature [C]
    "P48",    # HP Turbine exit pressure [bar]
    "P1",     # GT Compressor inlet air pressure [bar]
    "P2",     # GT Compressor outlet air pressure [bar]
    "Pexh",   # Gas Turbine exhaust gas pressure [bar]
    "TIC",    # Turbine Injection Control [%]
    "mf",     # Fuel flow [kg/s]
    "compressor_decay",  # kMc — GT Compressor decay state coefficient
    "turbine_decay",     # kMt — GT Turbine decay state coefficient
]


def read_zip_bytes(from_zip: str = None) -> bytes:
    if from_zip:
        print(f"📦 Leyendo zip local: {from_zip}")
        with open(from_zip, "rb") as f:
            return f.read()
    print(f"⬇️  Descargando desde UCI: {ZIP_URL}")
    with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--from-zip", help="Usar un zip de UCI ya descargado (sin red)")
    args = parser.parse_args()

    with zipfile.ZipFile(io.BytesIO(read_zip_bytes(args.from_zip))) as zf:
        raw = zf.read(DATA_MEMBER)

    actual = hashlib.sha256(raw).hexdigest()
    if actual != DATA_SHA256:
        raise SystemExit(
            f"❌ SHA-256 de {DATA_MEMBER} no coincide con el verificado.\n"
            f"   esperado: {DATA_SHA256}\n   actual:   {actual}\n"
            f"   El archivo upstream cambió: revisar antes de re-reportar números."
        )
    print(f"🔒 SHA-256 verificado: {actual[:16]}…")

    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, names=COLUMNS)
    constant = [c for c in df.columns if df[c].nunique() == 1]

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"✅ {len(df):,} filas × {len(df.columns)} columnas → {OUT_CSV}")
    print(f"   turbine_decay: {df['turbine_decay'].nunique()} niveles "
          f"[{df['turbine_decay'].min()}–{df['turbine_decay'].max()}]")
    print(f"   compressor_decay: {df['compressor_decay'].nunique()} niveles "
          f"[{df['compressor_decay'].min()}–{df['compressor_decay'].max()}]")
    print(f"   columnas constantes (sin información): {constant}")
    print("⚠️  Dataset SIMULADO — reportar siempre con esa etiqueta.")


if __name__ == "__main__":
    main()
