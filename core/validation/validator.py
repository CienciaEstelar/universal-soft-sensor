"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/validation/validator.py
Versión: 2.0.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Validador de datos que filtra filas con valores físicamente imposibles.
    Usa el Schema v2.0 con pattern matching universal.

═══════════════════════════════════════════════════════════════════════════════
CARACTERÍSTICAS:
═══════════════════════════════════════════════════════════════════════════════

    - Validación por rangos físicos (detectados automáticamente por categoría)
    - Preserva NaN para manejo posterior por el Preprocessor
    - Logging detallado con categoría física detectada
    - Estadísticas de validación para monitoreo
    - Reporte de diagnóstico para análisis de calidad de datos
    - Soporte para DatetimeIndex (filtra NaT)

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.0.0 - Enero 2026] INTEGRACIÓN CON SCHEMA UNIVERSAL
    ------------------------------------------------------
    - Ahora muestra la categoría física detectada en los logs
    - Nuevo método `diagnose()` para análisis detallado
    - Mejorado `get_invalid_summary()` con info de categoría
    - CLI actualizado con columnas de datasets reales
    
    [v1.0.0 - Inicial]
    ------------------
    - Validación básica por rangos
    - Estadísticas de rechazo

═══════════════════════════════════════════════════════════════════════════════
USO:
═══════════════════════════════════════════════════════════════════════════════

    from core.validation.validator import PhysicalValidator
    
    validator = PhysicalValidator()
    
    # Filtrar datos inválidos
    df_limpio = validator.validate(df_sucio)
    print(validator.last_stats)
    
    # Diagnóstico sin filtrar
    reporte = validator.diagnose(df)
    print(reporte)

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.validation.schema import SCHEMA, PhysicalCategory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASS: Estadísticas de Validación
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationStats:
    """
    Estadísticas de una ejecución de validación.
    
    Attributes
    ----------
    filas_entrada : int
        Número de filas antes de validar.
    filas_salida : int
        Número de filas después de validar.
    filas_eliminadas_fecha : int
        Filas eliminadas por tener fecha inválida (NaT).
    eliminadas_por_columna : Dict[str, int]
        Conteo de filas inválidas por cada columna.
    categorias_detectadas : Dict[str, str]
        Categoría física detectada para cada columna validada.
    """
    filas_entrada: int = 0
    filas_salida: int = 0
    filas_eliminadas_fecha: int = 0
    eliminadas_por_columna: Dict[str, int] = field(default_factory=dict)
    categorias_detectadas: Dict[str, str] = field(default_factory=dict)
    
    @property
    def filas_eliminadas_total(self) -> int:
        """Total de filas eliminadas (fecha + rangos)."""
        return self.filas_entrada - self.filas_salida
    
    @property
    def tasa_rechazo(self) -> float:
        """Porcentaje de filas rechazadas."""
        if self.filas_entrada == 0:
            return 0.0
        return (self.filas_eliminadas_total / self.filas_entrada) * 100
    
    @property
    def columnas_con_invalidos(self) -> List[str]:
        """Lista de columnas que tuvieron valores inválidos."""
        return list(self.eliminadas_por_columna.keys())
    
    def to_dict(self) -> dict:
        """Convierte las estadísticas a diccionario."""
        return {
            "filas_entrada": self.filas_entrada,
            "filas_salida": self.filas_salida,
            "filas_eliminadas_total": self.filas_eliminadas_total,
            "filas_eliminadas_fecha": self.filas_eliminadas_fecha,
            "tasa_rechazo_pct": round(self.tasa_rechazo, 2),
            "eliminadas_por_columna": self.eliminadas_por_columna,
            "categorias_detectadas": self.categorias_detectadas,
        }
    
    def __repr__(self) -> str:
        return (
            f"ValidationStats(entrada={self.filas_entrada}, "
            f"salida={self.filas_salida}, "
            f"rechazadas={self.filas_eliminadas_total} ({self.tasa_rechazo:.2f}%))"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: PhysicalValidator
# ═══════════════════════════════════════════════════════════════════════════

class PhysicalValidator:
    """
    Validador de datos de proceso industrial.
    
    Filtra filas que no cumplen con los rangos físicos definidos
    en el Schema. Usa pattern matching para detectar automáticamente
    la categoría física de cada columna.
    
    Comportamiento con NaN:
    - Los NaN se PRESERVAN (no se consideran inválidos)
    - El Preprocessor se encarga de imputarlos después
    
    Attributes
    ----------
    schema : PhysicalSchema
        Esquema de validación con reglas por categoría.
    log_threshold : int
        Umbral de filas inválidas para generar WARNING (por columna).
    last_stats : ValidationStats
        Estadísticas de la última validación ejecutada.
        
    Examples
    --------
    >>> validator = PhysicalValidator()
    >>> df_clean = validator.validate(df_raw)
    >>> print(validator.last_stats)
    ValidationStats(entrada=10000, salida=9856, rechazadas=144 (1.44%))
    """
    
    def __init__(self, schema=None, log_threshold: int = 100):
        """
        Inicializa el validador.
        
        Parameters
        ----------
        schema : PhysicalSchema, optional
            Esquema de validación. Si es None, usa el global (SCHEMA).
        log_threshold : int, default=100
            Umbral de filas inválidas para generar WARNING.
            Si una columna tiene más inválidos que este umbral,
            se loguea como WARNING en lugar de INFO.
        """
        self.schema = schema or SCHEMA
        self.log_threshold = log_threshold
        self.last_stats: Optional[ValidationStats] = None
    
    def validate(
        self, 
        df: pd.DataFrame,
        strict: bool = False,
        drop_unknown: bool = False
    ) -> pd.DataFrame:
        """
        Filtra un DataFrame manteniendo solo filas físicamente válidas.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos crudos (posibles errores de sensor).
        strict : bool, default=False
            Si True, también rechaza filas con valores en columnas UNKNOWN.
        drop_unknown : bool, default=False
            Si True, elimina columnas sin categoría reconocida.
            
        Returns
        -------
        pd.DataFrame
            DataFrame filtrado con solo filas válidas.
            Los NaN se preservan para manejo posterior.
            
        Notes
        -----
        El orden de filtrado es:
        1. Fechas inválidas (NaT en índice)
        2. Valores fuera de rango físico (por columna)
        """
        if df.empty:
            self.last_stats = ValidationStats()
            logger.warning("DataFrame vacío recibido")
            return df.copy()
        
        stats = ValidationStats(filas_entrada=len(df))
        df_work = df.copy()
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 1: Filtrar filas con fecha inválida (NaT en índice)
        # ═══════════════════════════════════════════════════════════════════
        if isinstance(df_work.index, pd.DatetimeIndex):
            mask_fecha_valida = df_work.index.notna()
            invalidos_fecha = (~mask_fecha_valida).sum()
            
            if invalidos_fecha > 0:
                stats.filas_eliminadas_fecha = invalidos_fecha
                logger.warning(
                    f"📅 Fechas inválidas (NaT): {invalidos_fecha} filas eliminadas"
                )
                df_work = df_work[mask_fecha_valida]
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 2: Identificar columnas a eliminar (si drop_unknown=True)
        # ═══════════════════════════════════════════════════════════════════
        if drop_unknown:
            cols_to_drop = []
            for col in df_work.columns:
                if pd.api.types.is_numeric_dtype(df_work[col]):
                    category = self.schema.get_category(col)
                    if category == PhysicalCategory.UNKNOWN:
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                logger.info(
                    f"🗑️  Eliminando {len(cols_to_drop)} columnas sin categoría: "
                    f"{cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}"
                )
                df_work = df_work.drop(columns=cols_to_drop)
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 3: Construir máscara de validación por rangos físicos
        # ═══════════════════════════════════════════════════════════════════
        mask_global = pd.Series(True, index=df_work.index)
        
        for col in df_work.columns:
            # Solo validar columnas numéricas
            if not pd.api.types.is_numeric_dtype(df_work[col]):
                continue
            
            # Detectar categoría física
            category = self.schema.get_category(col)
            stats.categorias_detectadas[col] = category.name
            
            # Obtener rango
            min_val, max_val = self.schema.get_range(col)
            
            # Si no hay límites (UNKNOWN sin strict), no filtrar
            if category == PhysicalCategory.UNKNOWN:
                if strict:
                    logger.debug(f"Columna '{col}' es UNKNOWN (modo strict)")
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # Regla de validación:
            # - Válido si está en rango [min, max]
            # - Válido si es NaN (se maneja después)
            # - Inválido si está fuera de rango
            # ═══════════════════════════════════════════════════════════════
            is_valid = (
                ((df_work[col] >= min_val) & (df_work[col] <= max_val)) | 
                df_work[col].isna()
            )
            
            invalidos = (~is_valid).sum()
            
            if invalidos > 0:
                stats.eliminadas_por_columna[col] = invalidos
                
                # Ejemplo de valor inválido para diagnóstico
                ejemplo = df_work.loc[~is_valid, col].iloc[0]
                
                # Logging según severidad
                msg = (
                    f"Columna '{col}' ({category.name}): "
                    f"{invalidos} valores fuera de [{min_val}, {max_val}] "
                    f"(ej: {ejemplo:.4f})"
                )
                
                if invalidos >= self.log_threshold:
                    logger.warning(f"⚠️  {msg}")
                else:
                    logger.info(f"ℹ️  {msg}")
            
            mask_global &= is_valid
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 4: Aplicar filtro y calcular estadísticas
        # ═══════════════════════════════════════════════════════════════════
        df_filtrado = df_work[mask_global].copy()
        stats.filas_salida = len(df_filtrado)
        
        # Log resumen
        if stats.tasa_rechazo > 10.0:
            logger.warning(
                f"🔴 Alta tasa de rechazo: {stats.tasa_rechazo:.2f}% "
                f"({stats.filas_eliminadas_total:,}/{stats.filas_entrada:,} filas)"
            )
        elif stats.tasa_rechazo > 5.0:
            logger.warning(
                f"🟡 Tasa de rechazo moderada: {stats.tasa_rechazo:.2f}% "
                f"({stats.filas_eliminadas_total:,}/{stats.filas_entrada:,} filas)"
            )
        else:
            logger.info(
                f"🟢 Validación OK: {stats.filas_salida:,}/{stats.filas_entrada:,} filas válidas "
                f"({stats.tasa_rechazo:.2f}% rechazado)"
            )
        
        self.last_stats = stats
        return df_filtrado
    
    def diagnose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera diagnóstico completo de calidad de datos SIN filtrar.
        
        Analiza cada columna y reporta:
        - Categoría física detectada
        - Rango permitido
        - Cantidad y porcentaje de valores inválidos
        - Cantidad de NaN
        - Estadísticas descriptivas
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a analizar.
            
        Returns
        -------
        pd.DataFrame
            Reporte de diagnóstico ordenado por porcentaje de inválidos.
        """
        report_data = []
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Detectar categoría y rango
            category = self.schema.get_category(col)
            min_val, max_val = self.schema.get_range(col)
            
            # Contar NaN
            n_nan = df[col].isna().sum()
            
            # Contar inválidos (excluyendo NaN)
            if category != PhysicalCategory.UNKNOWN:
                mask_invalido = ~(
                    ((df[col] >= min_val) & (df[col] <= max_val)) | 
                    df[col].isna()
                )
                n_invalidos = mask_invalido.sum()
                ejemplo = df.loc[mask_invalido, col].iloc[0] if n_invalidos > 0 else None
            else:
                n_invalidos = 0
                ejemplo = None
            
            # Estadísticas descriptivas (excluyendo NaN)
            col_clean = df[col].dropna()
            
            report_data.append({
                "columna": col,
                "categoria": category.name,
                "min_permitido": min_val if min_val != float("-inf") else "−∞",
                "max_permitido": max_val if max_val != float("inf") else "+∞",
                "n_total": len(df),
                "n_nan": n_nan,
                "pct_nan": round((n_nan / len(df)) * 100, 2),
                "n_invalidos": n_invalidos,
                "pct_invalidos": round((n_invalidos / len(df)) * 100, 2),
                "ejemplo_invalido": ejemplo,
                "valor_min_real": col_clean.min() if len(col_clean) > 0 else None,
                "valor_max_real": col_clean.max() if len(col_clean) > 0 else None,
                "media": col_clean.mean() if len(col_clean) > 0 else None,
            })
        
        # Crear DataFrame y ordenar por % inválidos
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values("pct_invalidos", ascending=False)
        
        return report_df.reset_index(drop=True)
    
    def get_invalid_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Versión simplificada de diagnose() - solo columnas con problemas.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a analizar.
            
        Returns
        -------
        pd.DataFrame
            Resumen solo de columnas con valores inválidos.
        """
        full_report = self.diagnose(df)
        
        # Filtrar solo columnas con inválidos o muchos NaN (>10%)
        problematic = full_report[
            (full_report["n_invalidos"] > 0) | 
            (full_report["pct_nan"] > 10)
        ]
        
        return problematic


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["PhysicalValidator", "ValidationStats"]


# ═══════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("🔍 Test de PhysicalValidator v2.0 - Pattern Matching Universal")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Test 1: Columnas estilo Gold Recovery (Kaggle)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📊 TEST 1: Columnas Gold Recovery")
    print("-" * 70)
    
    df_gold = pd.DataFrame({
        "rougher.input.feed_au": [2.5, 3.1, 150.0, np.nan, 4.2],  # 150 inválido
        "rougher.output.recovery": [65.0, 70.0, 102.0, 68.0, 71.0],  # 102 inválido
        "primary_cleaner.state.floatbank8_a_level": [450, 500, -600, 480, 520],  # -600 inválido
        "flotation_section_02_air_amount": [1200, 1300, -100, 1250, 1280],  # -100 inválido
    })
    
    print("Datos de entrada:")
    print(df_gold)
    
    validator = PhysicalValidator()
    df_clean = validator.validate(df_gold)
    
    print("\nDatos válidos:")
    print(df_clean)
    print(f"\n{validator.last_stats}")
    print(f"Categorías detectadas: {validator.last_stats.categorias_detectadas}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Test 2: Columnas estilo AI4I2020 (UCI)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 TEST 2: Columnas AI4I2020")
    print("-" * 70)
    
    df_ai4i = pd.DataFrame({
        "Air temperature [K]": [298.1, 298.2, 150.0, 299.0, 300.0],  # 150K inválido
        "Torque [Nm]": [42.8, 46.3, 600.0, 39.5, 40.2],  # 600 inválido
        "Rotational speed [rpm]": [1551, 1408, -500, 1433, 1425],  # -500 inválido
        "Tool wear [min]": [0, 3, 5, 7, 9],  # Todos válidos
        "Machine failure": [0, 0, 2, 0, 1],  # 2 inválido (debe ser 0 o 1)
    })
    
    print("Datos de entrada:")
    print(df_ai4i)
    
    df_clean = validator.validate(df_ai4i)
    
    print("\nDatos válidos:")
    print(df_clean)
    print(f"\n{validator.last_stats}")
    print(f"Categorías detectadas: {validator.last_stats.categorias_detectadas}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Test 3: Diagnóstico completo
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📋 TEST 3: Diagnóstico de Calidad")
    print("-" * 70)
    
    reporte = validator.diagnose(df_ai4i)
    print(reporte[["columna", "categoria", "n_invalidos", "pct_invalidos"]].to_string())
    
    print("\n" + "=" * 70)
    print("✅ Tests completados")
    print("=" * 70)
