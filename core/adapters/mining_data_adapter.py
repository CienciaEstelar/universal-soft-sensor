"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/adapters/mining_data_adapter.py
Proyecto: Arquitectura Minera 4.0
Autor: Juan Galaz
Versión: 2.0.0
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Adaptador de Datos Unificado (v2.0).

    [FIX CRÍTICO] Este archivo FALTABA en el repositorio.
    El módulo core/adapters/__init__.py lo importaba con:

        from core.adapters.mining_data_adapter import MiningDataAdapter, IngestionStats

    Pero el archivo no existía, causando ModuleNotFoundError en TODA la codebase:
        - train_universal.py
        - predict_universal.py
        - dashboard.py
        - core/pipeline.py
        - Todos los tests

    ARQUITECTURA:
    ─────────────
    MiningDataAdapter unifica MiningCSVAdapter (ingesta raw) y UniversalAdapter
    (ingesta con filtrado por JSON config) en una sola interfaz.

        ┌─────────────────────────────────┐
        │       MiningDataAdapter         │  ← Clase pública principal
        └────────────────┬────────────────┘
                         │ hereda
        ┌────────────────▼────────────────┐
        │       UniversalAdapter          │  ← Carga por config JSON
        └────────────────┬────────────────┘
                         │ delega ingesta raw a
        ┌────────────────▼────────────────┐
        │       MiningCSVAdapter          │  ← CSV auto-detección
        └─────────────────────────────────┘

USO:
────
    # Modo config JSON (recomendado para entrenamiento)
    adapter = MiningDataAdapter("dataset_config.json")
    df = adapter.load_data()

    # Modo archivo directo (para scripts rápidos / predict)
    adapter = MiningDataAdapter.from_file("data/mi_dataset.csv")
    df = adapter.load_raw()

    # Streaming para pipeline ETL (datasets grandes)
    for chunk in adapter.stream(chunk_size=25000):
        process(chunk)

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from core.adapters.universal_adapter import UniversalAdapter
from core.adapters.mining_csv_adapter import MiningCSVAdapter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASS: Estadísticas de Ingesta
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionStats:
    """
    Estadísticas de una operación de ingesta de datos.

    Attributes
    ----------
    rows_loaded : int
        Filas cargadas exitosamente.
    columns_loaded : int
        Columnas presentes tras el filtrado.
    source : str
        Nombre del archivo de origen.
    filtered_columns : int
        Columnas eliminadas por reglas de exclusión.
    null_pct : float
        Porcentaje de valores nulos en el dataset resultante.
    """
    rows_loaded: int = 0
    columns_loaded: int = 0
    source: str = ""
    filtered_columns: int = 0
    null_pct: float = 0.0

    def __repr__(self) -> str:
        return (
            f"IngestionStats(rows={self.rows_loaded:,}, "
            f"cols={self.columns_loaded}, "
            f"source='{self.source}', "
            f"null_pct={self.null_pct:.2f}%)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: MiningDataAdapter
# ═══════════════════════════════════════════════════════════════════════════

class MiningDataAdapter(UniversalAdapter):
    """
    Adaptador de Datos Unificado v2.0 para Minería 4.0.

    Unifica MiningCSVAdapter y UniversalAdapter en una sola interfaz.
    Soporta dos modos de operación:

    MODO 1 — Config JSON (recomendado):
        Carga un archivo JSON que define el dataset, target, y reglas de filtrado.
        Ideal para entrenamiento y pipeline ETL.

        >>> adapter = MiningDataAdapter("dataset_config.json")
        >>> df = adapter.load_data()

    MODO 2 — Archivo directo (factory method):
        Carga directamente desde un CSV sin config JSON.
        Ideal para scripts de inferencia y exploración rápida.

        >>> adapter = MiningDataAdapter.from_file("data/mining.csv")
        >>> df = adapter.load_raw()

    Attributes
    ----------
    last_stats : IngestionStats
        Estadísticas de la última operación de carga.
    """

    def __init__(self, config_filename: str = "dataset_config.json"):
        """
        Inicializa el adaptador en modo config JSON.

        Parameters
        ----------
        config_filename : str
            Nombre del archivo JSON en config/. Por defecto 'dataset_config.json'.
        """
        super().__init__(config_filename)
        self.last_stats: Optional[IngestionStats] = None
        logger.info(f"MiningDataAdapter v2.0 iniciado — config: {config_filename}")

    # ═══════════════════════════════════════════════════════════════════════
    # FACTORY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_file(
        cls,
        filepath: str,
        encoding: str = "utf-8"
    ) -> "MiningDataAdapter":
        """
        Factory method: crea un adaptador desde un archivo CSV directamente.

        Reemplaza el uso legacy de MiningCSVAdapter(filepath).

        Parameters
        ----------
        filepath : str
            Ruta al archivo CSV.
        encoding : str
            Codificación del archivo.

        Returns
        -------
        MiningDataAdapter
            Instancia configurada para leer el CSV indicado.

        Example
        -------
        >>> adapter = MiningDataAdapter.from_file("data/flotation.csv")
        >>> df = adapter.load_raw(max_rows=10000)
        """
        # Usamos MiningCSVAdapter como backend de ingesta raw
        # y devolvemos un objeto que expone la interfaz unificada
        return _FileModeAdapter(filepath, encoding)

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS PÚBLICOS — Interfaz unificada
    # ═══════════════════════════════════════════════════════════════════════

    def load_data(self) -> pd.DataFrame:
        """
        Carga y filtra el dataset según las reglas del config JSON.

        Equivalente al load_data() de UniversalAdapter pero registra
        estadísticas de ingesta.

        Returns
        -------
        pd.DataFrame
            Dataset limpio y filtrado, indexado por timestamp.
        """
        df = super().load_data()

        self.last_stats = IngestionStats(
            rows_loaded=len(df),
            columns_loaded=len(df.columns),
            source=str(self.data_path.name),
            null_pct=round(df.isnull().mean().mean() * 100, 2),
        )

        logger.info(f"Ingesta completada: {self.last_stats}")
        return df

    def load_raw(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Carga el dataset completo SIN aplicar filtros de feature selection.

        Útil para exploración, diagnóstico y scripts de inferencia que
        necesitan ver todas las columnas.

        Parameters
        ----------
        max_rows : int, optional
            Límite de filas. None = carga todo.

        Returns
        -------
        pd.DataFrame
            Dataset raw con todas las columnas, indexado por timestamp.
        """
        csv_adapter = MiningCSVAdapter(str(self.data_path))
        df = csv_adapter.read_all(max_rows=max_rows)

        self.last_stats = IngestionStats(
            rows_loaded=len(df),
            columns_loaded=len(df.columns),
            source=str(self.data_path.name),
            null_pct=round(df.isnull().mean().mean() * 100, 2),
        )

        return df

    def stream(
        self,
        chunk_size: int = 25000,
        apply_filtering: bool = False
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generador de chunks para procesamiento incremental.

        Parameters
        ----------
        chunk_size : int
            Filas por chunk.
        apply_filtering : bool
            Si True, aplica las reglas de feature_engineering del JSON a cada chunk.
            Por defecto False (streaming raw para el pipeline ETL).

        Yields
        ------
        pd.DataFrame
            Chunk limpio y sanitizado.
        """
        csv_adapter = MiningCSVAdapter(str(self.data_path))

        for chunk in csv_adapter.stream(chunk_size=chunk_size):
            if apply_filtering:
                chunk = self._apply_feature_selection(chunk)
            yield chunk

    def get_target_column(self) -> str:
        """Retorna la columna objetivo definida en el config JSON."""
        return self.config["modeling"]["target_column"]

    def get_stats(self) -> Optional[IngestionStats]:
        """Retorna las estadísticas de la última carga. None si no se ha cargado aún."""
        return self.last_stats


# ═══════════════════════════════════════════════════════════════════════════
# CLASE INTERNA: Modo archivo directo (from_file)
# ═══════════════════════════════════════════════════════════════════════════

class _FileModeAdapter(MiningDataAdapter):
    """
    Adaptador en modo archivo directo.

    Subclase de MiningDataAdapter usada por el factory `from_file`. Comparte
    la interfaz pública (load_data/load_raw/stream/get_target_column/get_stats)
    para que `isinstance(x, MiningDataAdapter)` sea verdadero y los type hints
    no requieran `# type: ignore`.

    No usa el __init__ de la base porque no parte de un config JSON; en su
    lugar construye los atributos manualmente.
    """

    def __init__(self, filepath: str, encoding: str = "utf-8"):
        # Bypass intencional de super().__init__: no hay config JSON que cargar.
        self._csv_adapter = MiningCSVAdapter(filepath, encoding)
        self.data_path = Path(filepath)
        self.config = {
            "modeling": {"target_column": None},
            "feature_engineering": {
                "include_patterns": [],
                "exclude_patterns": [],
                "forced_drop": [],
            },
        }
        self.last_stats: Optional[IngestionStats] = None

    def load_data(self) -> pd.DataFrame:
        """Carga todos los datos raw (sin filtrado de features)."""
        return self.load_raw()

    def load_raw(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Carga el CSV completo."""
        df = self._csv_adapter.read_all(max_rows=max_rows)
        self.last_stats = IngestionStats(
            rows_loaded=len(df),
            columns_loaded=len(df.columns),
            source=self.data_path.name,
            null_pct=round(df.isnull().mean().mean() * 100, 2),
        )
        return df

    def stream(
        self,
        chunk_size: int = 25000,
        apply_filtering: bool = False,
    ) -> Generator[pd.DataFrame, None, None]:
        """Streaming directo del CSV. `apply_filtering` se ignora en este modo
        porque no hay reglas de feature_engineering cargadas desde config."""
        return self._csv_adapter.stream(chunk_size=chunk_size)

    def get_target_column(self) -> Optional[str]:
        return self.config["modeling"]["target_column"]

    def get_stats(self) -> Optional[IngestionStats]:
        return self.last_stats


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["MiningDataAdapter", "IngestionStats"]


# ═══════════════════════════════════════════════════════════════════════════
# CLI para testing rápido
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("=" * 60)
    print("🔌 Test MiningDataAdapter v2.0")
    print("=" * 60)

    # Modo 1: Config JSON
    try:
        adapter = MiningDataAdapter("dataset_config.json")
        df = adapter.load_data()
        print(f"\n✅ Modo config JSON: {adapter.last_stats}")
        print(f"   Target: {adapter.get_target_column()}")
    except FileNotFoundError as e:
        print(f"⚠️  Config JSON no encontrado (esperado en dev): {e}")

    # Modo 2: Archivo directo (si se pasa como argumento)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        adapter2 = MiningDataAdapter.from_file(filepath)
        df2 = adapter2.load_raw(max_rows=1000)
        print(f"\n✅ Modo archivo directo: {adapter2.last_stats}")
        print(f"   Columnas: {df2.columns.tolist()[:5]}...")
