#!/usr/bin/env python3
"""
prepare_geomet_dataset.py — Genera data/geomet/flotation_chem_dureza.csv

[P0b / ROADMAP] data/ está en .gitignore (833 MB de datasets locales, no
versionados). Este script reproduce de forma determinista el join
química×dureza que dataset_config.json usa para el caso de negocio de cobre
(target LCT) — mismo join que valida run_geomet_rigor.py, pero persistido a
disco para que UniversalAdapter (que solo lee UN CSV, no hace joins) pueda
consumirlo directamente.

Advertencia de integridad (heredada de run_geomet_rigor.py, sección
"INTEGRIDAD DEL JOIN"): el merge flotation↔comminution por HOLEID DUPLICA
muestras a propósito — comminution.csv tiene varias filas de dureza por
sondaje. El resultado (102 filas, ~21 HOLEID únicos) requiere SIEMPRE
group_column="HOLEID" en el entrenamiento (GroupShuffleSplit/GroupKFold);
un split que ignora el grupo infla el R² por leakage (~0.93 falso vs. ~0.33
real, ver results/verification/geomet_rigor.json).

Uso:
    .venv/bin/python prepare_geomet_dataset.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMET_DIR = os.path.join(HERE, "data", "geomet")


def main():
    flot_path = os.path.join(GEOMET_DIR, "flotation.csv")
    comm_path = os.path.join(GEOMET_DIR, "comminution.csv")
    out_path = os.path.join(GEOMET_DIR, "flotation_chem_dureza.csv")

    if not os.path.exists(flot_path) or not os.path.exists(comm_path):
        raise FileNotFoundError(
            f"Faltan los CSV crudos de GeoMet en {GEOMET_DIR}. "
            f"Descargar desde Zenodo 10.5281/zenodo.7051975 (flotation.csv, "
            f"comminution.csv) antes de correr este script."
        )

    flot = pd.read_csv(flot_path)
    comm = pd.read_csv(comm_path)

    merged = flot.merge(
        comm.drop(columns=[c for c in ["X", "Y", "Z"] if c in comm.columns]),
        on="HOLEID", how="inner", suffixes=("", "_c"),
    )

    print(f"flotation.csv:    {flot.shape} ({flot['HOLEID'].nunique()} HOLEID únicos)")
    print(f"comminution.csv:  {comm.shape} ({comm['HOLEID'].nunique()} HOLEID únicos)")
    print(f"merged (join):    {merged.shape} ({merged['HOLEID'].nunique()} HOLEID únicos)")
    if len(merged) > len(flot):
        print(
            "🔴 El merge DUPLICA muestras (esperado — comminution.csv tiene "
            "varias filas de dureza por sondaje). group_column='HOLEID' es "
            "OBLIGATORIO en el entrenamiento."
        )

    merged.to_csv(out_path, index=False)
    print(f"\n✅ Guardado: {out_path}")


if __name__ == "__main__":
    main()
