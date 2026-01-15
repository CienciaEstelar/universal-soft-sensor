"""
Módulo: config/settings.py
Descripción: Configuración centralizada del proyecto.
             Usa variables de entorno con fallbacks sensatos.
             
Uso:
    from config.settings import CONFIG
    print(CONFIG.DATA_RAW_PATH)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Cargar .env si existe
load_dotenv()


def get_project_root() -> Path:
    """
    Detecta la raíz del proyecto buscando markers conocidos.
    Funciona tanto en desarrollo como en producción.
    """
    current = Path(__file__).resolve().parent
    
    # Subimos hasta encontrar la carpeta 'core' o un pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "core").is_dir() or (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: directorio actual
    return Path.cwd()


@dataclass
class ProjectConfig:
    """
    Configuración centralizada del proyecto minero.
    Todas las rutas y parámetros configurables en un solo lugar.
    """
    
    # === RUTAS BASE ===
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    
    # === RUTAS DE DATOS ===
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"
    
    @property
    def DATA_RAW_PATH(self) -> Path:
        """Ruta al dataset crudo original."""
        env_path = os.getenv("MINING_DATA_RAW_PATH")
        if env_path:
            return Path(env_path)
        return self.DATA_DIR / "MiningProcess_Flotation_Plant_Database.csv"
    
    @property
    def DATA_PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"
    
    @property
    def DATA_CLEAN_PATH(self) -> Path:
        """Ruta al dataset limpio (output del ETL)."""
        return self.DATA_PROCESSED_DIR / "mining_clean.csv"
    
    # === RUTAS DE OUTPUTS ===
    @property
    def MODELS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "models"
    
    @property
    def RESULTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "results"
    
    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "logs"
    
    # === CONFIGURACIÓN DEL PIPELINE ===
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "25000")))
    
    # === CONFIGURACIÓN DEL MODELO GP ===
    GP_TARGET_COLUMN: str = field(default_factory=lambda: os.getenv("GP_TARGET", "_silica_concentrate"))
    GP_MAX_TRAIN_SAMPLES: int = field(default_factory=lambda: int(os.getenv("GP_MAX_SAMPLES", "1800")))
    GP_OPTUNA_TRIALS: int = field(default_factory=lambda: int(os.getenv("GP_TRIALS", "15")))
    
    # === CONFIGURACIÓN DE PREPROCESAMIENTO ===
    PREPROCESSING_STRATEGY: str = field(default_factory=lambda: os.getenv("PREPROCESS_STRATEGY", "ffill"))
    FILL_VALUE: float = field(default_factory=lambda: float(os.getenv("FILL_VALUE", "0.0")))
    
    def __post_init__(self):
        """Crea directorios necesarios si no existen."""
        for dir_path in [self.DATA_PROCESSED_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Valida que existan los recursos críticos."""
        if not self.DATA_RAW_PATH.exists():
            raise FileNotFoundError(
                f"Dataset no encontrado: {self.DATA_RAW_PATH}\n"
                f"Define MINING_DATA_RAW_PATH en .env o coloca el archivo en data/"
            )
        return True
    
    def __repr__(self) -> str:
        return (
            f"ProjectConfig(\n"
            f"  PROJECT_ROOT={self.PROJECT_ROOT}\n"
            f"  DATA_RAW_PATH={self.DATA_RAW_PATH}\n"
            f"  DATA_CLEAN_PATH={self.DATA_CLEAN_PATH}\n"
            f"  GP_TARGET={self.GP_TARGET_COLUMN}\n"
            f")"
        )


# Instancia global lista para importar
CONFIG = ProjectConfig()


# =============================================================================
# CLI para verificar configuración
# =============================================================================
if __name__ == "__main__":
    print("🔧 Configuración del Proyecto Minero 4.0")
    print("=" * 50)
    print(CONFIG)
    print("=" * 50)
    
    try:
        CONFIG.validate()
        print("✅ Configuración válida")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
