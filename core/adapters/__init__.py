"""
Adaptadores de ingesta de datos para Universal Soft-Sensor.

API pública:
    - DataAdapter   : adaptador unificado (modo config JSON o from_file)
    - IngestionStats      : estadísticas de la última carga
    - CSVAdapter    : (legacy) lector CSV de bajo nivel — usar
                            DataAdapter.from_file() para código nuevo
    - UniversalAdapter    : (legacy) carga por config JSON — usar
                            DataAdapter() para código nuevo

Notas de migración:
    Antes existían clases shim en este __init__.py con los mismos nombres
    `CSVAdapter` y `UniversalAdapter` que las definidas en sus módulos
    respectivos. Esa duplicación creaba dos clases distintas con el mismo
    nombre según el path de import y rompía isinstance(). Se eliminaron.
    Los nombres siguen disponibles como re-exports de las clases originales.
"""

from core.adapters.data_adapter import DataAdapter, IngestionStats
from core.adapters.csv_adapter import CSVAdapter
from core.adapters.universal_adapter import UniversalAdapter

__all__ = [
    "DataAdapter",
    "IngestionStats",
    "CSVAdapter",
    "UniversalAdapter",
]
