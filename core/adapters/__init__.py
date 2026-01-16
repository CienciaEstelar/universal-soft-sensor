"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/adapters/__init__.py
Versión: 2.0.0
═══════════════════════════════════════════════════════════════════════════════

Adaptadores de ingesta de datos para Proyecto Minero 4.0.

CAMBIO v2.0.0:
--------------
Se unificaron MiningCSVAdapter y UniversalAdapter en un solo MiningDataAdapter.

MIGRACIÓN:
----------
    # ANTES:
    from core.adapters import MiningCSVAdapter
    from core.adapters.universal_adapter import UniversalAdapter
    
    # AHORA:
    from core.adapters import MiningDataAdapter
    
    # Equivalencias:
    # - UniversalAdapter("config.json")  →  MiningDataAdapter("config.json")
    # - MiningCSVAdapter(filepath)       →  MiningDataAdapter.from_file(filepath)

DEPRECACIÓN:
------------
Los imports antiguos siguen funcionando por compatibilidad, pero mostrarán
un warning. Se eliminarán en v3.0.0.
"""

import warnings

# ═══════════════════════════════════════════════════════════════════════════
# NUEVO ADAPTER UNIFICADO (v2.0+)
# ═══════════════════════════════════════════════════════════════════════════
from core.adapters.mining_data_adapter import MiningDataAdapter, IngestionStats


# ═══════════════════════════════════════════════════════════════════════════
# COMPATIBILIDAD HACIA ATRÁS (Deprecated)
# ═══════════════════════════════════════════════════════════════════════════

class MiningCSVAdapter(MiningDataAdapter):
    """
    DEPRECATED: Usar MiningDataAdapter.from_file() en su lugar.
    
    Esta clase existe solo por compatibilidad hacia atrás.
    Se eliminará en v3.0.0.
    """
    
    def __init__(self, filepath: str, encoding: str = "utf-8"):
        warnings.warn(
            "MiningCSVAdapter está deprecado. "
            "Usa MiningDataAdapter.from_file(filepath) en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )
        # Usar el factory method del nuevo adapter
        instance = MiningDataAdapter.from_file(filepath, encoding)
        self.__dict__.update(instance.__dict__)
    
    def stream(self, chunk_size: int = 25000):
        """Mantiene compatibilidad con la API anterior."""
        return super().stream(chunk_size=chunk_size, apply_filtering=False)
    
    def read_all(self, max_rows=None):
        """Mantiene compatibilidad con la API anterior."""
        return self.load_raw(max_rows=max_rows)


class UniversalAdapter(MiningDataAdapter):
    """
    DEPRECATED: Usar MiningDataAdapter en su lugar.
    
    Esta clase existe solo por compatibilidad hacia atrás.
    Se eliminará en v3.0.0.
    """
    
    def __init__(self, config_filename: str = "dataset_config.json"):
        warnings.warn(
            "UniversalAdapter está deprecado. "
            "Usa MiningDataAdapter(config_filename) en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(config_filename)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS PÚBLICOS
# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    # Nuevo (recomendado)
    "MiningDataAdapter",
    "IngestionStats",
    # Deprecated (compatibilidad)
    "MiningCSVAdapter",
    "UniversalAdapter",
]
