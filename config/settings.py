"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: config/settings.py
Proyecto: Arquitectura Minera 4.0
Autor: Juan Galaz
Versión: 1.1.0
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Configuración centralizada del proyecto.
    Usa variables de entorno con fallbacks sensatos.
    
HISTORIAL DE CAMBIOS:
    [v1.1.0 - Enero 2026] CLEAN CODE UPDATE
        ✅ AGREGADO: DEFAULT_SUBSAMPLE_STEP como constante maestra
           - Antes: El valor de subsample estaba hardcodeado en múltiples archivos
             (train_universal.py usaba 10, inference.py usaba 50, mining_gp_pro.py usaba 50)
           - Ahora: Valor único centralizado aquí, todos los módulos lo importan
           - Beneficio: Cambiar el subsample en un solo lugar afecta todo el sistema
        
    [v1.0.0] Versión inicial con rutas y configuración GP

USO:
    from config.settings import CONFIG
    
    # Acceder a cualquier configuración:
    print(CONFIG.DATA_RAW_PATH)
    print(CONFIG.DEFAULT_SUBSAMPLE_STEP)  # <-- NUEVO en v1.1.0
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════
# Cargar variables de entorno desde .env (si existe)
# ═══════════════════════════════════════════════════════════════════════════
load_dotenv()


def get_project_root() -> Path:
    """
    Detecta la raíz del proyecto buscando markers conocidos.
    Funciona tanto en desarrollo como en producción.
    
    Returns:
        Path: Ruta absoluta a la raíz del proyecto
    """
    current = Path(__file__).resolve().parent
    
    # Subimos hasta encontrar la carpeta 'core' o un pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "core").is_dir() or (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: directorio actual de trabajo
    return Path.cwd()


@dataclass
class ProjectConfig:
    """
    Configuración centralizada del proyecto minero.
    
    Esta clase actúa como "Single Source of Truth" para todas las rutas
    y parámetros configurables del sistema. Cualquier valor que necesite
    ser consistente entre módulos debe definirse aquí.
    
    Attributes:
        PROJECT_ROOT: Raíz del proyecto (auto-detectada)
        CHUNK_SIZE: Tamaño de chunks para procesamiento por lotes
        GP_TARGET_COLUMN: Columna objetivo para el modelo GP
        GP_MAX_TRAIN_SAMPLES: Máximo de muestras para entrenamiento
        GP_OPTUNA_TRIALS: Número de trials de optimización Optuna
        DEFAULT_SUBSAMPLE_STEP: [NUEVO v1.1.0] Valor maestro de subsampleo
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUTAS BASE
    # ═══════════════════════════════════════════════════════════════════════
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUTAS DE DATOS (Properties para evaluación lazy)
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def DATA_DIR(self) -> Path:
        """Directorio raíz de datos."""
        return self.PROJECT_ROOT / "data"
    
    @property
    def DATA_RAW_PATH(self) -> Path:
        """
        Ruta al dataset crudo original.
        Puede ser sobreescrita con la variable de entorno MINING_DATA_RAW_PATH.
        """
        env_path = os.getenv("MINING_DATA_RAW_PATH")
        if env_path:
            return Path(env_path)
        return self.DATA_DIR / "MiningProcess_Flotation_Plant_Database.csv"
    
    @property
    def DATA_PROCESSED_DIR(self) -> Path:
        """Directorio para datos procesados (output del ETL)."""
        return self.DATA_DIR / "processed"
    
    @property
    def DATA_CLEAN_PATH(self) -> Path:
        """Ruta al dataset limpio (output del pipeline ETL)."""
        return self.DATA_PROCESSED_DIR / "mining_clean.csv"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUTAS DE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def MODELS_DIR(self) -> Path:
        """Directorio para modelos entrenados (.pkl)."""
        return self.PROJECT_ROOT / "models"
    
    @property
    def RESULTS_DIR(self) -> Path:
        """Directorio para reportes y gráficos."""
        return self.PROJECT_ROOT / "results"
    
    @property
    def LOGS_DIR(self) -> Path:
        """Directorio para archivos de log."""
        return self.PROJECT_ROOT / "logs"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DEL PIPELINE ETL
    # ═══════════════════════════════════════════════════════════════════════
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "25000"))
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DEL MODELO GP (Gaussian Process)
    # ═══════════════════════════════════════════════════════════════════════
    GP_TARGET_COLUMN: str = field(
        default_factory=lambda: os.getenv("GP_TARGET", "_silica_concentrate")
    )
    
    GP_MAX_TRAIN_SAMPLES: int = field(
        default_factory=lambda: int(os.getenv("GP_MAX_SAMPLES", "1800"))
    )
    
    GP_OPTUNA_TRIALS: int = field(
        default_factory=lambda: int(os.getenv("GP_TRIALS", "15"))
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE PREPROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════
    PREPROCESSING_STRATEGY: str = field(
        default_factory=lambda: os.getenv("PREPROCESS_STRATEGY", "ffill")
    )
    
    FILL_VALUE: float = field(
        default_factory=lambda: float(os.getenv("FILL_VALUE", "0.0"))
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # ██████████████████████████████████████████████████████████████████████
    # ███  NUEVO EN v1.1.0: CONSTANTE MAESTRA DE SUBSAMPLEO  ███████████████
    # ██████████████████████████████████████████████████████████████████████
    # ═══════════════════════════════════════════════════════════════════════
    #
    # FIX: "Subsample-Gate"
    # ---------------------
    # PROBLEMA ANTERIOR:
    #   - train_universal.py tenía: subsample_step = 10
    #   - inference.py tenía: df_processed = df_full.iloc[::50]
    #   - mining_gp_pro.py tenía: default subsample_step = 50
    #   
    #   Esto causaba desalineación de features durante inferencia porque
    #   el modelo se entrenaba con un subsample y predecía con otro.
    #
    # SOLUCIÓN:
    #   Centralizar el valor aquí. Todos los módulos ahora importan:
    #   
    #       from config.settings import CONFIG
    #       step = CONFIG.DEFAULT_SUBSAMPLE_STEP
    #
    # CÓMO CAMBIAR EL VALOR:
    #   Opción 1: Editar el default aquí (actualmente 10)
    #   Opción 2: Definir variable de entorno SUBSAMPLE_STEP=20
    #
    # ═══════════════════════════════════════════════════════════════════════
    # Default = 1 (sin subsampling). En ML supervisado para series temporales,
    # diezmar para "descorrelacionar" es un anti-patrón heredado de inferencia
    # estadística clásica: borra señal y desalinea train vs inference (que no
    # subsamplea, ver core/inference_engine.py). Cambiar solo si se necesita
    # reducir el costo cuadrático del GP en datasets muy grandes.
    DEFAULT_SUBSAMPLE_STEP: int = field(
        default_factory=lambda: int(os.getenv("SUBSAMPLE_STEP", "1"))
    )
    # ═══════════════════════════════════════════════════════════════════════
    
    def __post_init__(self):
        """
        Hook que se ejecuta después de inicializar el dataclass.
        Crea los directorios necesarios si no existen.
        """
        for dir_path in [self.DATA_PROCESSED_DIR, self.MODELS_DIR, 
                         self.RESULTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """
        Valida que existan los recursos críticos antes de ejecutar el pipeline.
        
        Returns:
            bool: True si la validación es exitosa
            
        Raises:
            FileNotFoundError: Si el dataset crudo no existe
        """
        if not self.DATA_RAW_PATH.exists():
            raise FileNotFoundError(
                f"❌ Dataset no encontrado: {self.DATA_RAW_PATH}\n"
                f"   Define MINING_DATA_RAW_PATH en .env o coloca el archivo en data/"
            )
        return True
    
    def __repr__(self) -> str:
        """Representación legible de la configuración actual."""
        return (
            f"ProjectConfig(\n"
            f"  PROJECT_ROOT           = {self.PROJECT_ROOT}\n"
            f"  DATA_RAW_PATH          = {self.DATA_RAW_PATH}\n"
            f"  DATA_CLEAN_PATH        = {self.DATA_CLEAN_PATH}\n"
            f"  GP_TARGET              = {self.GP_TARGET_COLUMN}\n"
            f"  DEFAULT_SUBSAMPLE_STEP = {self.DEFAULT_SUBSAMPLE_STEP}  ← [NUEVO v1.1.0]\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INSTANCIA GLOBAL - Lista para importar en cualquier módulo
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = ProjectConfig()


# ═══════════════════════════════════════════════════════════════════════════
# CLI: Ejecutar este archivo directamente para verificar configuración
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🔧 Configuración del Proyecto Minero 4.0 (v1.1.0)")
    print("=" * 60)
    print(CONFIG)
    print("=" * 60)
    
    try:
        CONFIG.validate()
        print("✅ Configuración válida - Todos los recursos encontrados")
    except FileNotFoundError as e:
        print(f"❌ Error de validación:\n{e}")
