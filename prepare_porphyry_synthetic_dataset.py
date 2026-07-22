#!/usr/bin/env python3
"""
prepare_porphyry_synthetic_dataset.py — Descarga y prepara data/porphyry_synthetic/block_model.csv

[ROADMAP 2026-07-21] Segundo dataset de cobre para el proyecto, buscado tras la
reconciliación del R²=0.319 de GeoMet cobre (n=92) — ese resultado es real pero
descansa en una muestra chica (21 HOLEID), así que se buscó un dataset MÁS GRANDE
para validar si el método (GroupKFold/permutation test/GB fallback) generaliza a
escala. Se investigaron ~10 candidatos (Zenodo, MDPI, GitHub); la mayoría eran
datos reales pero CONFIDENCIALES (ej. Mu & Salas 2023, Processes: 1112 muestras
reales, "not publicly available because of confidentiality agreements") o no
tabulares (imágenes hiperespectrales). Este es el único candidato abierto,
tabular, con target de recuperación metalúrgica y volumen sustancialmente mayor.

⚠️ ADVERTENCIA CRÍTICA — ESTE DATASET ES SINTÉTICO, NO REAL:
Proviene de:
    Garrido, M., Sepúlveda, E., Ortiz, J., Townley, B. (2020).
    "Simulation of synthetic exploration and geometallurgical database of
    porphyry copper deposits for educational purposes."
    Natural Resources Research, 29, 3527-3545.
Repositorio: https://github.com/exepulveda/geomet_datasets (rama master,
datasets/porphyry_01/datafiles/block_model.csv). Licencia CC BY-NC-SA 4.0
(no comercial — coherente con el uso académico/portafolio de este proyecto).

Es un block model simulado geoestadísticamente (Bayesian/Hilbert-Kriging-style,
mismo linaje académico que el dataset GeoMet cobre real que ya usamos). NO es
evidencia de que el pipeline funcione en una mina real — es evidencia de que el
MÉTODO (features, GroupKFold, permutation test, fallback GP→GB) se comporta bien
a una escala 1000x mayor que GeoMet. Cualquier R² obtenido aquí debe reportarse
SIEMPRE con la etiqueta "sintético" y nunca mezclarse con el resultado real de
GeoMet cobre (R²=0.319, results/verification/geomet_pipeline_reconciliation_v2.json)
en ninguna tabla o comparación sin esa distinción explícita.

Filtrado de waste: el dataset trae 153,076 bloques, de los cuales 5,845 tienen
cu==rec==bwi==0 simultáneamente — corresponden a la zona de mineralización 5
("Waste and Gravel without economic content", ver README del dataset upstream).
Se filtran aquí (cu > 0.01) porque incluirlos infla trivialmente la separabilidad
(predecir rec=0 donde cu=0 no prueba nada) — confirmado empíricamente: la
correlación cu-rec cae de 0.30 (con waste) a 0.01 (sin waste), es decir, el
problema real (sin trampa) es genuinamente no-trivial, igual que GeoMet.

Uso:
    .venv/bin/python prepare_porphyry_synthetic_dataset.py
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "porphyry_synthetic")
RAW_URL = (
    "https://raw.githubusercontent.com/exepulveda/geomet_datasets/master/"
    "datasets/porphyry_01/datafiles/block_model.csv"
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "block_model_raw.csv")
    out_path = os.path.join(OUT_DIR, "block_model_filtered.csv")

    if not os.path.exists(raw_path):
        print(f"Descargando {RAW_URL} ...")
        urllib.request.urlretrieve(RAW_URL, raw_path)
        print(f"Guardado: {raw_path}")
    else:
        print(f"Ya existe (no se re-descarga): {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"block_model_raw.csv: {df.shape}")

    n_waste = int((df["cu"] == 0).sum())
    print(
        f"Filas de waste (cu==0, zona de mineralización 5, sin contenido "
        f"económico): {n_waste} — se filtran."
    )

    df_filtered = df[df["cu"] > 0.01].copy()
    print(f"Filtrado (cu > 0.01): {df_filtered.shape}")

    corr_raw = df["cu"].corr(df["rec"])
    corr_filtered = df_filtered["cu"].corr(df_filtered["rec"])
    print(f"Correlación cu-rec SIN filtrar: {corr_raw:.4f}")
    print(f"Correlación cu-rec CON filtro:  {corr_filtered:.4f}")
    if corr_raw > 0.15 and corr_filtered < 0.05:
        print(
            "🔴 Confirma trampa metodológica evitada: sin filtrar, cu==0 "
            "predice rec==0 trivialmente. Con el filtro, el problema es "
            "genuinamente no-trivial (como GeoMet cobre real)."
        )

    df_filtered.to_csv(out_path, index=False)
    print(f"\n✅ Guardado: {out_path}")
    print(f"   Filas finales: {len(df_filtered)}")
    print(f"   Columnas: {list(df_filtered.columns)}")


if __name__ == "__main__":
    main()
