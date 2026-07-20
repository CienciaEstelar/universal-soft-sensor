"""
Core del Universal Soft-Sensor

Módulos:
    - adapters: Ingesta de datos desde diversas fuentes
    - validation: Validación de datos contra esquema físico
    - preprocessor: Limpieza y preparación de datos
    - pipeline: Orquestación ETL completa
"""

from core.adapters import CSVAdapter
from core.validation import SCHEMA, PhysicalValidator, ValidationStats
from core.preprocessor import Preprocessor, CleaningStats
from core.pipeline import SoftSensorPipeline

__all__ = [
    # Adapters
    "CSVAdapter",
    # Validation
    "SCHEMA",
    "PhysicalValidator",
    "ValidationStats",
    # Preprocessing
    "Preprocessor",
    "CleaningStats",
    # Pipeline
    "SoftSensorPipeline",
]

__version__ = "1.0.0"
