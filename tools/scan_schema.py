"""
Script: tools/scan_schema.py
Descripción: Utilidad de diagnóstico para explorar la estructura del dataset.
             Ejecuta el Adapter una vez para extraer nombres de columnas
             sanitizados, tipos de datos y rangos iniciales.

Uso:
    python -m tools.scan_schema
    python -m tools.scan_schema --rows 1000
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Configurar path para imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CONFIG
from core.adapters import CSVAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SCAN_SCHEMA")


def escanear_estructura(max_rows: int = None, show_sample: bool = True) -> dict:
    """
    Escanea la estructura del dataset sin cargarlo completamente.
    
    Parameters
    ----------
    max_rows : int, optional
        Máximo de filas a leer para el análisis.
    show_sample : bool
        Si True, muestra muestra de datos.
        
    Returns
    -------
    dict
        Información estructural del dataset.
    """
    print("🛰️  Iniciando escaneo de estructura de datos...")
    print("=" * 60)
    
    try:
        # Verificar que existe el archivo
        CONFIG.validate()
        
        adapter = CSVAdapter(str(CONFIG.DATA_RAW_PATH))
        
        if max_rows:
            # Leer cantidad específica
            df_sample = adapter.read_all(max_rows=max_rows)
        else:
            # Obtener solo el primer chunk
            generador = adapter.stream()
            df_sample = next(generador)
        
        print(f"✅ Lectura exitosa. Dimensiones: {df_sample.shape}")
        print(f"   Archivo: {CONFIG.DATA_RAW_PATH}")
        
        # === COLUMNAS ===
        print("\n📋 LISTA DE COLUMNAS SANITIZADAS:")
        print("-" * 60)
        col_list = df_sample.columns.tolist()
        
        # Formato para copiar/pegar
        print("COLUMNS = [")
        for i, col in enumerate(col_list):
            comma = "," if i < len(col_list) - 1 else ""
            print(f'    "{col}"{comma}')
        print("]")
        
        # === TIPOS DE DATOS ===
        print("\n🔍 TIPOS DE DATOS DETECTADOS:")
        print("-" * 60)
        
        dtypes_info = []
        for col in df_sample.columns:
            dtype = df_sample[col].dtype
            n_nulls = df_sample[col].isna().sum()
            pct_nulls = (n_nulls / len(df_sample)) * 100
            
            dtypes_info.append({
                "columna": col,
                "tipo": str(dtype),
                "nulos": n_nulls,
                "pct_nulos": f"{pct_nulls:.2f}%"
            })
            
            print(f"  {col:40} → {dtype} (nulos: {n_nulls})")
        
        # === RANGOS NUMÉRICOS ===
        print("\n📊 RANGOS INICIALES (Mínimos y Máximos):")
        print("-" * 60)
        
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            desc = df_sample[numeric_cols].describe().T[['min', 'max', 'mean', 'std']]
            print(desc.to_string())
        else:
            print("  (No se encontraron columnas numéricas)")
        
        # === MUESTRA DE DATOS ===
        if show_sample:
            print("\n📝 MUESTRA DE DATOS (primeras 5 filas):")
            print("-" * 60)
            print(df_sample.head().to_string())
        
        # === INFO DEL ÍNDICE ===
        print("\n📅 INFORMACIÓN DEL ÍNDICE:")
        print("-" * 60)
        
        if isinstance(df_sample.index, pd.DatetimeIndex):
            print(f"  Tipo: DatetimeIndex")
            print(f"  Rango: {df_sample.index.min()} → {df_sample.index.max()}")
            print(f"  Frecuencia inferida: {pd.infer_freq(df_sample.index[:100])}")
        else:
            print(f"  Tipo: {type(df_sample.index).__name__}")
        
        print("\n" + "=" * 60)
        print("✅ Escaneo completado")
        
        return {
            "shape": df_sample.shape,
            "columns": col_list,
            "dtypes": {col: str(df_sample[col].dtype) for col in col_list},
            "numeric_ranges": desc.to_dict() if not numeric_cols.empty else {},
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Sugerencia: Verifica la ruta en config/settings.py")
        print(f"   o define MINING_DATA_RAW_PATH en tu archivo .env")
        return None
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        logger.exception("Error en escaneo")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Escanea la estructura del dataset minero"
    )
    parser.add_argument(
        "--rows", "-n",
        type=int,
        default=None,
        help="Número de filas a analizar (default: primer chunk)"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="No mostrar muestra de datos"
    )
    
    args = parser.parse_args()
    
    result = escanear_estructura(
        max_rows=args.rows,
        show_sample=not args.no_sample
    )
    
    if result is None:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
