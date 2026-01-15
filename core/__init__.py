"""
Core del Proyecto Minero 4.0

Módulos:
    - adapters: Ingesta de datos desde diversas fuentes
    - validation: Validación de datos contra esquema físico
    - preprocessor: Limpieza y preparación de datos
    - pipeline: Orquestación ETL completa
"""

from core.adapters import MiningCSVAdapter
from core.validation import SCHEMA, MiningValidator, ValidationStats
from core.preprocessor import MiningPreprocessor, CleaningStats
from core.pipeline import MiningPipeline

__all__ = [
    # Adapters
    "MiningCSVAdapter",
    # Validation
    "SCHEMA",
    "MiningValidator",
    "ValidationStats",
    # Preprocessing
    "MiningPreprocessor",
    "CleaningStats",
    # Pipeline
    "MiningPipeline",
]

__version__ = "1.0.0"
