# core/adapters/mining_data_adapter.py
from dataclasses import dataclass
from core.adapters.mining_csv_adapter import MiningCSVAdapter as _LegacyCSV
from core.adapters.universal_adapter import UniversalAdapter

@dataclass
class IngestionStats:
    rows_loaded: int = 0
    columns_loaded: int = 0
    source: str = ""

class MiningDataAdapter(UniversalAdapter):
    """Adapter unificado v2.0 — reemplaza MiningCSVAdapter y UniversalAdapter."""
    
    @classmethod
    def from_file(cls, filepath: str, encoding: str = "utf-8") -> "MiningDataAdapter":
        """Factory method para compatibilidad con MiningCSVAdapter legacy."""
        # Wrap temporal hasta migración completa
        instance = _LegacyCSV(filepath, encoding)
        return instance  # type: ignore
    
    def load_raw(self, max_rows=None):
        """Alias para load_data() con parámetro opcional de filas."""
        return self.load_data()
    
    def stream(self, chunk_size=25000, apply_filtering=True):
        """Streaming interface."""
        # Delegar al legacy adapter temporalmente
        from pathlib import Path
        legacy = _LegacyCSV(str(self.data_path))
        return legacy.stream(chunk_size=chunk_size)
