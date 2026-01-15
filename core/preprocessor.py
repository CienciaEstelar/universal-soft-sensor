"""
Módulo: core/preprocessor.py
Descripción: Limpieza estadística y preparación numérica de datos de sensores.

Características:
    - Múltiples estrategias de imputación (ffill, bfill, interpolate)
    - Detección y manejo de outliers por IQR o Z-score
    - Reemplazo de valores infinitos
    - Logging estructurado de operaciones
    - Fail-safe absoluto (nunca rompe el pipeline)
    
Uso:
    from core.preprocessor import MiningPreprocessor
    
    preprocessor = MiningPreprocessor(estrategia_nulos="interpolate")
    df_limpio = preprocessor.clean_stream(df_crudo)
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Estadísticas de una operación de limpieza."""
    filas_procesadas: int = 0
    infinitos_reemplazados: int = 0
    nulos_imputados: int = 0
    outliers_detectados: int = 0
    
    def __repr__(self) -> str:
        return (
            f"CleaningStats(filas={self.filas_procesadas}, "
            f"inf={self.infinitos_reemplazados}, "
            f"nulos={self.nulos_imputados}, "
            f"outliers={self.outliers_detectados})"
        )


class MiningPreprocessor:
    """
    Preprocesador de datos para flujos numéricos continuos.
    
    Operaciones (en orden):
    1. Reemplazo de valores infinitos por NaN
    2. Detección de outliers (opcional)
    3. Imputación de valores nulos
    4. Relleno de NaN persistentes con valor por defecto
    
    Atributos:
        last_stats: Estadísticas de la última operación de limpieza.
    """
    
    ESTRATEGIAS_VALIDAS = {"ffill", "bfill", "interpolate", "mean", "median"}
    
    def __init__(
        self,
        estrategia_nulos: Literal["ffill", "bfill", "interpolate", "mean", "median"] = "ffill",
        valor_relleno_inicial: float = 0.0,
        detectar_outliers: bool = False,
        outlier_method: Literal["zscore", "iqr"] = "iqr",
        outlier_threshold: float = 3.0
    ) -> None:
        """
        Inicializa el preprocesador.
        
        Parameters
        ----------
        estrategia_nulos : str
            Estrategia para imputar valores nulos:
            - 'ffill': Forward fill (propagar último valor válido)
            - 'bfill': Backward fill (propagar siguiente valor válido)
            - 'interpolate': Interpolación lineal (mejor para series temporales)
            - 'mean': Reemplazar con media de la columna
            - 'median': Reemplazar con mediana de la columna
            
        valor_relleno_inicial : float
            Valor para NaN que persisten después de imputación
            (ej: al inicio de serie con ffill).
            
        detectar_outliers : bool
            Si True, marca outliers como NaN antes de imputar.
            
        outlier_method : str
            Método de detección: 'zscore' o 'iqr'.
            
        outlier_threshold : float
            Umbral para detección (3.0 estándar para z-score, 
            1.5 para IQR multiplicador).
        """
        if estrategia_nulos not in self.ESTRATEGIAS_VALIDAS:
            raise ValueError(
                f"Estrategia '{estrategia_nulos}' no válida. "
                f"Opciones: {self.ESTRATEGIAS_VALIDAS}"
            )
        
        self.estrategia_nulos = estrategia_nulos
        self.valor_relleno_inicial = valor_relleno_inicial
        self.detectar_outliers = detectar_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self.last_stats: Optional[CleaningStats] = None
        
        logger.debug(
            f"Preprocessor inicializado: estrategia={estrategia_nulos}, "
            f"outliers={detectar_outliers}"
        )
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detecta outliers usando Z-score."""
        mean = series.mean()
        std = series.std()
        
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detecta outliers usando IQR (Inter-Quartile Range)."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(False, index=series.index)
        
        lower_bound = q1 - (self.outlier_threshold * iqr)
        upper_bound = q3 + (self.outlier_threshold * iqr)
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _handle_outliers(self, df: pd.DataFrame, cols: pd.Index) -> tuple[pd.DataFrame, int]:
        """
        Detecta y reemplaza outliers por NaN.
        
        Returns
        -------
        tuple
            (DataFrame modificado, número de outliers detectados)
        """
        total_outliers = 0
        
        detector = (
            self._detect_outliers_zscore 
            if self.outlier_method == "zscore" 
            else self._detect_outliers_iqr
        )
        
        for col in cols:
            outlier_mask = detector(df[col])
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                df.loc[outlier_mask, col] = np.nan
                total_outliers += n_outliers
                
                if n_outliers > 100:
                    logger.warning(
                        f"Columna '{col}': {n_outliers} outliers detectados "
                        f"({self.outlier_method}, threshold={self.outlier_threshold})"
                    )
        
        return df, total_outliers
    
    def _imputar_nulos(self, df: pd.DataFrame, cols: pd.Index) -> tuple[pd.DataFrame, int]:
        """
        Imputa valores nulos según la estrategia configurada.
        
        Returns
        -------
        tuple
            (DataFrame modificado, número de nulos imputados)
        """
        nulos_antes = df[cols].isna().sum().sum()
        
        if self.estrategia_nulos == "ffill":
            df[cols] = df[cols].ffill()
            
        elif self.estrategia_nulos == "bfill":
            df[cols] = df[cols].bfill()
            
        elif self.estrategia_nulos == "interpolate":
            # Interpolación lineal - ideal para series temporales
            df[cols] = df[cols].interpolate(method='linear', limit_direction='both')
            
        elif self.estrategia_nulos == "mean":
            for col in cols:
                df[col] = df[col].fillna(df[col].mean())
                
        elif self.estrategia_nulos == "median":
            for col in cols:
                df[col] = df[col].fillna(df[col].median())
        
        nulos_despues = df[cols].isna().sum().sum()
        nulos_imputados = nulos_antes - nulos_despues
        
        return df, nulos_imputados
    
    def clean_stream(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y normaliza un bloque de datos numéricos.
        
        El método es tolerante a fallos:
        - Nunca modifica el DataFrame original
        - Si ocurre un error crítico, devuelve una copia segura
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos crudos pero tipados correctamente.
            
        Returns
        -------
        pd.DataFrame
            DataFrame limpio y seguro para análisis posterior.
        """
        stats = CleaningStats()
        
        try:
            # === Validación de entrada ===
            if not isinstance(df, pd.DataFrame):
                raise TypeError("El objeto recibido no es un pandas DataFrame")
            
            if df.empty:
                logger.debug("DataFrame vacío recibido, retornando copia")
                self.last_stats = stats
                return df.copy()
            
            df_limpio = df.copy()
            stats.filas_procesadas = len(df_limpio)
            
            # === Selección de columnas numéricas ===
            cols_numericas = df_limpio.select_dtypes(include=[np.number]).columns
            
            if cols_numericas.empty:
                logger.debug("Sin columnas numéricas, retornando sin cambios")
                self.last_stats = stats
                return df_limpio
            
            # === 1. Reemplazo de infinitos ===
            inf_mask = np.isinf(df_limpio[cols_numericas])
            stats.infinitos_reemplazados = inf_mask.sum().sum()
            
            if stats.infinitos_reemplazados > 0:
                df_limpio[cols_numericas] = df_limpio[cols_numericas].replace(
                    [np.inf, -np.inf], np.nan
                )
                logger.debug(f"Infinitos reemplazados: {stats.infinitos_reemplazados}")
            
            # === 2. Detección de outliers (opcional) ===
            if self.detectar_outliers:
                df_limpio, stats.outliers_detectados = self._handle_outliers(
                    df_limpio, cols_numericas
                )
                if stats.outliers_detectados > 0:
                    logger.info(f"Outliers detectados y marcados: {stats.outliers_detectados}")
            
            # === 3. Imputación de nulos ===
            df_limpio, stats.nulos_imputados = self._imputar_nulos(df_limpio, cols_numericas)
            
            if stats.nulos_imputados > 0:
                logger.debug(
                    f"Nulos imputados: {stats.nulos_imputados} "
                    f"(estrategia: {self.estrategia_nulos})"
                )
            
            # === 4. Relleno final de NaN persistentes ===
            nulos_persistentes = df_limpio[cols_numericas].isna().sum().sum()
            if nulos_persistentes > 0:
                df_limpio[cols_numericas] = df_limpio[cols_numericas].fillna(
                    self.valor_relleno_inicial
                )
                logger.debug(
                    f"NaN persistentes rellenados con {self.valor_relleno_inicial}: "
                    f"{nulos_persistentes}"
                )
            
            self.last_stats = stats
            return df_limpio
            
        except Exception as error:
            # Fail-safe absoluto: nunca romper el pipeline
            logger.error(f"Error durante limpieza de datos: {error}", exc_info=True)
            self.last_stats = stats
            return df.copy()
    
    def get_quality_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Genera un reporte de calidad de datos sin modificar el DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a analizar.
            
        Returns
        -------
        dict
            Métricas de calidad por columna.
        """
        cols_num = df.select_dtypes(include=[np.number]).columns
        
        report = {
            "total_filas": len(df),
            "total_columnas_numericas": len(cols_num),
            "por_columna": {}
        }
        
        for col in cols_num:
            serie = df[col]
            report["por_columna"][col] = {
                "nulos": serie.isna().sum(),
                "pct_nulos": serie.isna().mean() * 100,
                "infinitos": np.isinf(serie).sum() if serie.dtype in [np.float64, np.float32] else 0,
                "min": serie.min(),
                "max": serie.max(),
                "mean": serie.mean(),
                "std": serie.std(),
            }
        
        return report


# =============================================================================
# __init__.py helper
# =============================================================================
__all__ = ["MiningPreprocessor", "CleaningStats"]


# =============================================================================
# CLI para testing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
    
    # Crear datos de prueba con problemas comunes
    np.random.seed(42)
    df_test = pd.DataFrame({
        "sensor_a": [1.0, 2.0, np.inf, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],
        "sensor_b": [10.0, np.nan, 30.0, 40.0, 50.0, -np.inf, 70.0, 80.0, 90.0, 1000.0],  # 1000 es outlier
        "sensor_c": np.random.randn(10) * 10 + 50,
    })
    
    print("🔧 Test de MiningPreprocessor")
    print("=" * 60)
    print("Datos de entrada:")
    print(df_test)
    print()
    
    # Test con diferentes configuraciones
    configs = [
        {"estrategia_nulos": "ffill", "detectar_outliers": False},
        {"estrategia_nulos": "interpolate", "detectar_outliers": True, "outlier_method": "iqr"},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuración {i+1}: {config} ---")
        preprocessor = MiningPreprocessor(**config)
        df_clean = preprocessor.clean_stream(df_test)
        
        print(f"Estadísticas: {preprocessor.last_stats}")
        print("Resultado:")
        print(df_clean)
