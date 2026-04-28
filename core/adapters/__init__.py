"""
Adaptadores de ingesta de datos para Proyecto Minero 4.0.

API pública:
    - MiningDataAdapter   : adaptador unificado (modo config JSON o from_file)
    - IngestionStats      : estadísticas de la última carga
    - MiningCSVAdapter    : (legacy) lector CSV de bajo nivel — usar
                            MiningDataAdapter.from_file() para código nuevo
    - UniversalAdapter    : (legacy) carga por config JSON — usar
                            MiningDataAdapter() para código nuevo

Notas de migración:
    Antes existían clases shim en este __init__.py con los mismos nombres
    `MiningCSVAdapter` y `UniversalAdapter` que las definidas en sus módulos
    respectivos. Esa duplicación creaba dos clases distintas con el mismo
    nombre según el path de import y rompía isinstance(). Se eliminaron.
    Los nombres siguen disponibles como re-exports de las clases originales.
"""

from core.adapters.mining_data_adapter import MiningDataAdapter, IngestionStats
from core.adapters.mining_csv_adapter import MiningCSVAdapter
from core.adapters.universal_adapter import UniversalAdapter

__all__ = [
    "MiningDataAdapter",
    "IngestionStats",
    "MiningCSVAdapter",
    "UniversalAdapter",
]
