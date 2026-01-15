"""
Módulo: core/validation/validator.py
Descripción: Filtra datos usando el Schema físico.
             Reporta detalladamente qué se elimina y por qué.
             
Características:
    - Validación por rangos físicos definidos en schema.py
    - Logging detallado de filas eliminadas por columna
    - Soporte para DatetimeIndex (filtra NaT)
    - Estadísticas de validación para monitoreo
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.validation.schema import SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Estadísticas de una ejecución de validación."""
    filas_entrada: int = 0
    filas_salida: int = 0
    filas_eliminadas_fecha: int = 0
    eliminadas_por_columna: Dict[str, int] = field(default_factory=dict)
    
    @property
    def filas_eliminadas_total(self) -> int:
        return self.filas_entrada - self.filas_salida
    
    @property
    def tasa_rechazo(self) -> float:
        if self.filas_entrada == 0:
            return 0.0
        return (self.filas_eliminadas_total / self.filas_entrada) * 100
    
    def __repr__(self) -> str:
        return (
            f"ValidationStats(entrada={self.filas_entrada}, "
            f"salida={self.filas_salida}, "
            f"rechazadas={self.filas_eliminadas_total} ({self.tasa_rechazo:.2f}%))"
        )


class MiningValidator:
    """
    Validador de datos de proceso minero.
    
    Filtra filas que no cumplen con los rangos físicos definidos
    en el Schema. Permite NaN (serán manejados por el Preprocessor).
    
    Uso:
        validator = MiningValidator()
        df_limpio = validator.validate(df_sucio)
        print(validator.last_stats)
    """
    
    def __init__(self, schema=None, log_threshold: int = 100):
        """
        Inicializa el validador.
        
        Parameters
        ----------
        schema : MiningSchema, optional
            Esquema de validación. Si es None, usa el global.
        log_threshold : int
            Umbral de filas inválidas para generar warning (por columna).
        """
        self.schema = schema or SCHEMA
        self.log_threshold = log_threshold
        self.last_stats: Optional[ValidationStats] = None
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra un DataFrame manteniendo solo filas físicamente válidas.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos crudos (posibles errores de sensor).
            
        Returns
        -------
        pd.DataFrame
            DataFrame filtrado (solo filas que cumplen rangos físicos).
            Los NaN se preservan para manejo posterior.
        """
        if df.empty:
            self.last_stats = ValidationStats()
            return df.copy()
        
        stats = ValidationStats(filas_entrada=len(df))
        df_work = df.copy()
        
        # 1. Filtrar filas con fecha inválida (NaT en índice)
        if isinstance(df_work.index, pd.DatetimeIndex):
            mask_fecha_valida = df_work.index.notna()
            invalidos_fecha = (~mask_fecha_valida).sum()
            
            if invalidos_fecha > 0:
                stats.filas_eliminadas_fecha = invalidos_fecha
                logger.warning(
                    f"Fechas inválidas (NaT): {invalidos_fecha} filas eliminadas"
                )
                df_work = df_work[mask_fecha_valida]
        
        # 2. Construir máscara de validación por rangos físicos
        mask_global = pd.Series(True, index=df_work.index)
        
        for col in df_work.columns:
            # Solo validar columnas numéricas
            if not pd.api.types.is_numeric_dtype(df_work[col]):
                continue
            
            min_val, max_val = self.schema.get_range(col)
            
            # Si no hay límites (infinito), no filtrar
            if min_val == -float("inf") and max_val == float("inf"):
                continue
            
            # Válido si: está en rango O es NaN (NaN se maneja después)
            is_valid = (
                ((df_work[col] >= min_val) & (df_work[col] <= max_val)) | 
                df_work[col].isna()
            )
            
            invalidos = (~is_valid).sum()
            
            if invalidos > 0:
                stats.eliminadas_por_columna[col] = invalidos
                
                # Logging según severidad
                if invalidos >= self.log_threshold:
                    logger.warning(
                        f"Columna '{col}': {invalidos} valores fuera de rango "
                        f"[{min_val}, {max_val}]"
                    )
                else:
                    logger.info(
                        f"Columna '{col}': {invalidos} valores fuera de rango"
                    )
            
            mask_global &= is_valid
        
        # 3. Aplicar filtro y calcular estadísticas
        df_filtrado = df_work[mask_global].copy()
        stats.filas_salida = len(df_filtrado)
        
        # Log resumen
        if stats.tasa_rechazo > 5.0:
            logger.warning(
                f"Alta tasa de rechazo: {stats.tasa_rechazo:.2f}% "
                f"({stats.filas_eliminadas_total}/{stats.filas_entrada} filas)"
            )
        else:
            logger.info(
                f"Validación completada: {stats.filas_salida}/{stats.filas_entrada} filas válidas"
            )
        
        self.last_stats = stats
        return df_filtrado
    
    def get_invalid_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera un reporte detallado de valores inválidos sin filtrar datos.
        
        Útil para diagnóstico y análisis de calidad de datos.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a analizar.
            
        Returns
        -------
        pd.DataFrame
            Resumen con columnas: columna, min_permitido, max_permitido,
            n_invalidos, pct_invalidos, ejemplo_valor_invalido
        """
        report_data = []
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            min_val, max_val = self.schema.get_range(col)
            
            if min_val == -float("inf") and max_val == float("inf"):
                continue
            
            # Encontrar inválidos (excluyendo NaN)
            mask_invalido = ~(
                ((df[col] >= min_val) & (df[col] <= max_val)) | 
                df[col].isna()
            )
            
            n_invalidos = mask_invalido.sum()
            
            if n_invalidos > 0:
                ejemplo = df.loc[mask_invalido, col].iloc[0]
                report_data.append({
                    "columna": col,
                    "min_permitido": min_val,
                    "max_permitido": max_val,
                    "n_invalidos": n_invalidos,
                    "pct_invalidos": (n_invalidos / len(df)) * 100,
                    "ejemplo_valor": ejemplo
                })
        
        return pd.DataFrame(report_data)


# =============================================================================
# __init__.py helper
# =============================================================================
__all__ = ["MiningValidator", "ValidationStats"]


# =============================================================================
# CLI para testing
# =============================================================================
if __name__ == "__main__":
    # Test básico
    import numpy as np
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    # Crear datos de prueba
    df_test = pd.DataFrame({
        "_iron_feed": [45.0, 50.0, 150.0, np.nan],  # 150 fuera de rango
        "ore_pulp_ph": [7.0, 8.5, 15.0, 6.0],       # 15 fuera de rango (pH max=14)
        "starch_flow": [100, 200, -50, 300],        # -50 fuera de rango (min=0)
    })
    
    print("📊 Test de MiningValidator")
    print("=" * 50)
    print("Datos de entrada:")
    print(df_test)
    print()
    
    validator = MiningValidator()
    df_clean = validator.validate(df_test)
    
    print("\nDatos válidos:")
    print(df_clean)
    print(f"\nEstadísticas: {validator.last_stats}")
    print(f"Eliminados por columna: {validator.last_stats.eliminadas_por_columna}")
