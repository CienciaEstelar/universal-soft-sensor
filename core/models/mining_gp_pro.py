"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/models/mining_gp_pro.py
Proyecto: Arquitectura Minera 4.0
Autor: Juan Galaz
Versión: 4.1.0
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Soft-Sensor industrial para predicción en tiempo real de variables de
    proceso en plantas de flotación minera. Utiliza Gaussian Process (GP)
    con optimización bayesiana de hiperparámetros vía Optuna.

CARACTERÍSTICAS PRINCIPALES:
    • Diagnóstico automático de autocorrelación temporal
    • Feature engineering: lags, diferencias, promedios móviles
    • Eliminación automática de features constantes y correlacionados
    • Fallback inteligente a GradientBoosting si GP falla (R² < 0.6)
    • Cuantificación de incertidumbre (intervalos de confianza)

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v4.1.0 - Enero 2026] CLEAN CODE UPDATE
    ----------------------------------------
    
    ✅ FIX #1: Eliminado hardcode de "_iron_concentrate"
       
       ANTES (Línea ~177):
           drop_cols = [self.target_col, "_iron_concentrate"]  # ❌ Hardcode feo
       
       AHORA:
           drop_cols = [self.target_col]  # ✅ Solo el target, sistema universal
       
       RAZÓN: El código estaba pensado para un dataset específico (hierro).
              Al querer usar el sistema con oro o cobre, fallaba porque
              "_iron_concentrate" no existía. Ahora el sistema de 
              remove_correlated_features se encarga de eliminar columnas
              redundantes automáticamente.
    
    ✅ FIX #2: Subsample centralizado en CONFIG
       
       ANTES:
           def __init__(self, ..., subsample_step: int = 50, ...):  # ❌ Hardcode
       
       AHORA:
           def __init__(self, ..., subsample_step: int = None, ...):
               self.subsample_step = subsample_step or CONFIG.DEFAULT_SUBSAMPLE_STEP  # ✅
       
       RAZÓN: El valor de subsample estaba definido diferente en cada archivo
              (10 en train, 50 en inference). Esto causaba desalineación de
              features. Ahora todos usan el mismo valor desde config/settings.py

    [v4.0.0] Versión con fallback a GradientBoosting
    [v3.0.0] Versión con optimización Optuna
    [v2.0.0] Versión con diagnóstico de autocorrelación
    [v1.0.0] Versión inicial básica

═══════════════════════════════════════════════════════════════════════════════
USO BÁSICO:
═══════════════════════════════════════════════════════════════════════════════

    from core.models.mining_gp_pro import MiningGP
    
    # Entrenamiento completo desde archivo
    model = MiningGP(target_col="_silica_concentrate")
    metrics = model.train_from_file("data/processed/mining_clean.csv")
    
    # Predicción
    y_pred, y_std = model.predict(X_new)

═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════
import sys
import json
import joblib
import optuna
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, field

# Sklearn - Machine Learning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Rich - Interfaz de usuario bonita
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Configuración centralizada del proyecto
from config.settings import CONFIG

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# Silenciar warnings molestos de sklearn y numpy
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Estilo de gráficos matplotlib
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')  # Fallback para versiones antiguas


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES DE SOPORTE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelMetrics:
    """
    Contenedor de métricas de evaluación del modelo.
    
    Attributes:
        r2: Coeficiente de determinación (1.0 = perfecto)
        rmse: Error cuadrático medio (menor = mejor)
        mae: Error absoluto medio
        mape: Error porcentual absoluto medio
    """
    r2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    
    def to_dict(self) -> dict:
        """Convierte las métricas a diccionario (útil para JSON)."""
        return {"r2": self.r2, "rmse": self.rmse, "mae": self.mae, "mape": self.mape}
    
    def __repr__(self) -> str:
        return f"R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"


@dataclass
class TrainingArtifacts:
    """
    Artefactos generados durante el entrenamiento.
    Este objeto se serializa con joblib para persistencia.
    
    Attributes:
        model: El modelo entrenado (GP o GradientBoosting)
        scaler_X: Escalador de features (RobustScaler)
        scaler_y: Escalador del target
        feature_names: Lista de nombres de features usados
        target_column: Nombre de la columna objetivo
        best_params: Hiperparámetros óptimos encontrados
        metrics: Métricas de evaluación
        model_type: "GP" o "GradientBoosting"
        removed_features: Features eliminados durante limpieza
        training_date: Fecha/hora del entrenamiento
    """
    model: any
    scaler_X: RobustScaler
    scaler_y: RobustScaler
    feature_names: List[str]
    target_column: str
    best_params: Dict
    metrics: ModelMetrics
    model_type: str = "GP"
    removed_features: List[str] = field(default_factory=list)
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: MiningGP
# ═══════════════════════════════════════════════════════════════════════════

class MiningGP:
    """
    Soft-Sensor v4.1 - Versión Universal y Limpia.
    
    Esta clase implementa un sensor virtual (soft-sensor) para predecir
    variables de proceso minero en tiempo real, eliminando la necesidad
    de análisis de laboratorio que tardan horas.
    
    Cambios importantes en v4.1.0:
    -----------------------------
    1. Ya no tiene hardcode de columnas específicas como "_iron_concentrate"
    2. El subsample_step ahora viene de CONFIG.DEFAULT_SUBSAMPLE_STEP
    3. El sistema es verdaderamente "universal" para cualquier dataset minero
    
    Example:
        >>> model = MiningGP(target_col="rougher.output.recovery")
        >>> metrics = model.train_from_file("gold_data.csv", n_trials=30)
        >>> print(f"R² = {metrics.r2:.4f}")
    """
    
    def __init__(
        self, 
        target_col: str = None, 
        random_state: int = 42,
        subsample_step: int = None,  # ← [v4.1.0] Si es None, usa CONFIG
        add_lag_features: bool = True,
        lag_periods: List[int] = None,
        add_diff_features: bool = True,
        use_fallback_model: bool = True,
        remove_constant_features: bool = True,
        remove_correlated_features: bool = True,
        correlation_threshold: float = 0.98
    ):
        """
        Inicializa el Soft-Sensor.
        
        Args:
            target_col: Columna objetivo a predecir. Si es None, usa CONFIG.GP_TARGET_COLUMN
            random_state: Semilla para reproducibilidad
            subsample_step: Cada cuántas filas tomar una muestra.
                           [v4.1.0] Si es None, usa CONFIG.DEFAULT_SUBSAMPLE_STEP
            add_lag_features: Si True, agrega features de lag temporal
            lag_periods: Lista de periodos de lag [1, 5, 10, 20] por defecto
            add_diff_features: Si True, agrega diferencias y promedios móviles
            use_fallback_model: Si True, usa GradientBoosting cuando GP falla
            remove_constant_features: Si True, elimina features con std ≈ 0
            remove_correlated_features: Si True, elimina features muy correlacionados
            correlation_threshold: Umbral de correlación para eliminación (0.98 default)
        """
        # Interfaz de usuario
        self.console = Console()
        
        # Columna objetivo: usar la del argumento o la de CONFIG
        self.target_col = target_col or CONFIG.GP_TARGET_COLUMN
        self.random_state = random_state
        
        # ═══════════════════════════════════════════════════════════════════
        # [v4.1.0] FIX: Subsample centralizado
        # ═══════════════════════════════════════════════════════════════════
        # ANTES: subsample_step: int = 50  (hardcodeado)
        # AHORA: Si no se especifica, usa el valor de CONFIG
        # Esto garantiza consistencia entre entrenamiento e inferencia
        # ═══════════════════════════════════════════════════════════════════
        if subsample_step is not None:
            self.subsample_step = subsample_step
        else:
            self.subsample_step = CONFIG.DEFAULT_SUBSAMPLE_STEP
        # ═══════════════════════════════════════════════════════════════════
        
        # Configuración de feature engineering
        self.add_lag_features = add_lag_features
        self.lag_periods = lag_periods or [1, 5, 10, 20]
        self.add_diff_features = add_diff_features
        
        # Configuración de comportamiento
        self.use_fallback_model = use_fallback_model
        self.remove_constant_features = remove_constant_features
        self.remove_correlated_features = remove_correlated_features
        self.correlation_threshold = correlation_threshold
        
        # Escaladores (se ajustan durante fit)
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        
        # Estado interno
        self.model = None
        self.model_type = "GP"  # Puede cambiar a "GradientBoosting"
        self.feature_names: List[str] = []
        self.removed_features: List[str] = []
        self.best_params: Dict = {}
        self.metrics: Optional[ModelMetrics] = None
        self.data_diagnosis: Dict = {}
        
        # Log de inicialización
        logger.info(
            f"MiningGP v4.1.0 inicializado - "
            f"Target: {self.target_col}, "
            f"Subsample: {self.subsample_step} (desde CONFIG)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE DIAGNÓSTICO
    # ═══════════════════════════════════════════════════════════════════════
    
    def _diagnose_data(self, y_series: pd.Series) -> Dict:
        """
        Diagnóstico automático de la serie temporal objetivo.
        
        Analiza la autocorrelación para determinar qué tan "pegados" están
        los datos consecutivos. Alta autocorrelación = datos redundantes.
        
        Args:
            y_series: Serie temporal del target
            
        Returns:
            Dict con estadísticas y recomendaciones
        """
        diagnosis = {
            "n_samples": len(y_series),
            "mean": y_series.mean(),
            "std": y_series.std(),
            "cv": (y_series.std() / y_series.mean()) * 100 if y_series.mean() != 0 else 0,
            "autocorr_1": y_series.autocorr(lag=1),
            "autocorr_10": y_series.autocorr(lag=10) if len(y_series) > 10 else 0,
            "autocorr_50": y_series.autocorr(lag=50) if len(y_series) > 50 else 0,
        }
        
        # Encontrar el subsample recomendado (donde autocorr < 0.85)
        for lag in [10, 20, 30, 40, 50, 75, 100, 150, 200]:
            if len(y_series) > lag:
                ac = y_series.autocorr(lag=lag)
                if ac < 0.85:
                    diagnosis["recommended_subsample"] = lag
                    break
        else:
            diagnosis["recommended_subsample"] = 200
        
        # Clasificar severidad del problema de autocorrelación
        if diagnosis["autocorr_1"] > 0.98:
            diagnosis["severity"] = "CRÍTICA"
        elif diagnosis["autocorr_1"] > 0.95:
            diagnosis["severity"] = "ALTA"
        elif diagnosis["autocorr_1"] > 0.90:
            diagnosis["severity"] = "MODERADA"
        else:
            diagnosis["severity"] = "OK"
        
        return diagnosis
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE PREPROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════
    
    def _remove_problematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina features problemáticos automáticamente.
        
        Criterios de eliminación:
        1. Features constantes (std < 1e-8): No aportan información
        2. Features altamente correlacionados (r > threshold): Redundantes
        
        Args:
            df: DataFrame con features (sin el target)
            
        Returns:
            DataFrame limpio sin features problemáticos
        """
        df = df.copy()
        removed = []
        
        # --- Paso 1: Eliminar features constantes ---
        if self.remove_constant_features:
            for col in df.columns:
                if df[col].std() < 1e-8:
                    removed.append(f"{col} (constante)")
                    df = df.drop(columns=[col])
        
        # --- Paso 2: Eliminar features muy correlacionados ---
        if self.remove_correlated_features and len(df.columns) > 1:
            # Matriz de correlación absoluta
            corr_matrix = df.corr().abs()
            # Triángulo superior (para no duplicar comparaciones)
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Encontrar columnas con correlación > threshold
            to_drop = []
            for col in upper.columns:
                high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
                if high_corr:
                    to_drop.extend(high_corr)
            
            # Eliminar duplicados y dropear
            to_drop = list(set(to_drop))
            for col in to_drop:
                if col in df.columns:
                    removed.append(f"{col} (correlación > {self.correlation_threshold})")
                    df = df.drop(columns=[col])
        
        # --- Log de features eliminados ---
        if removed:
            self.console.print(f"[yellow]   ⚠️  Features eliminados automáticamente:[/yellow]")
            for r in removed[:5]:  # Mostrar máximo 5
                self.console.print(f"[dim]      - {r}[/dim]")
            if len(removed) > 5:
                self.console.print(f"[dim]      ... y {len(removed)-5} más[/dim]")
        
        self.removed_features = removed
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        """
        Crea features de ingeniería temporal.
        
        El Gaussian Process necesita contexto temporal para entender
        la dinámica del proceso. Agregamos:
        - Lags: valor de Y en t-1, t-5, t-10, t-20
        - Diferencias: cambio de Y entre t y t-1
        - Promedios móviles: suavizado de Y
        
        Args:
            df: DataFrame con datos
            y_col: Nombre de la columna objetivo
            
        Returns:
            DataFrame con features adicionales
        """
        df = df.copy()
        y = df[y_col]
        
        # --- Features de Lag ---
        if self.add_lag_features:
            for lag in self.lag_periods:
                df[f'{y_col}_lag_{lag}'] = y.shift(lag)
        
        # --- Features de Diferencia y Tendencia ---
        # Causales: solo usan información en t-1 o anterior, nunca y[t].
        if self.add_diff_features:
            df[f'{y_col}_diff_1'] = y.shift(1) - y.shift(2)
            df[f'{y_col}_diff_5'] = y.shift(1) - y.shift(6)
            df[f'{y_col}_rolling_mean_10'] = y.shift(1).rolling(10, min_periods=1).mean()
            df[f'{y_col}_rolling_std_10'] = y.shift(1).rolling(10, min_periods=1).std()
        
        return df
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL: CARGA DE DATOS
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_data(
        self, 
        filepath: str = None,
        max_rows: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Carga y preprocesa datos desde un archivo CSV.
        
        Pipeline completo:
        1. Leer CSV con límite de filas
        2. Diagnóstico de autocorrelación
        3. Subsampleo temporal (reduce autocorrelación)
        4. Feature engineering (lags, diferencias)
        5. Limpieza de features problemáticos
        6. Escalado robusto
        
        Args:
            filepath: Ruta al CSV. Si None, usa CONFIG.DATA_CLEAN_PATH
            max_rows: Máximo de filas a cargar (las últimas)
            
        Returns:
            Tuple de (X_scaled, y_scaled, dates_index)
        """
        filepath = filepath or str(CONFIG.DATA_CLEAN_PATH)
        
        self.console.print(f"[bold cyan]📥 Cargando datos desde:[/bold cyan] {filepath}")
        
        # Verificar existencia del archivo
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No encontrado: {filepath}")
        
        # Contar filas totales (para skip inteligente)
        with open(filepath, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # -1 por header
        
        skip_rows = max(0, total_rows - max_rows)
        self.console.print(f"[dim]   Archivo: {total_rows:,} filas totales[/dim]")
        
        # Leer CSV
        df = pd.read_csv(
            filepath, 
            index_col=0,
            parse_dates=True,
            skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
        )
        
        # Validar que existe el target
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target '{self.target_col}' no encontrado en el dataset.\n"
                f"Columnas disponibles: {list(df.columns[:10])}..."
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNÓSTICO DE AUTOCORRELACIÓN
        # ═══════════════════════════════════════════════════════════════════
        self.console.print(f"\n[bold yellow]🔬 Diagnóstico de Autocorrelación:[/bold yellow]")
        self.data_diagnosis = self._diagnose_data(df[self.target_col])
        
        # Mostrar diagnóstico en tabla bonita
        diag_table = Table(show_header=False, box=None, padding=(0, 2))
        diag_table.add_row("Autocorr lag-1:", f"{self.data_diagnosis['autocorr_1']:.4f}")
        diag_table.add_row("Autocorr lag-50:", f"{self.data_diagnosis['autocorr_50']:.4f}")
        
        sev = self.data_diagnosis['severity']
        sev_color = "red" if sev == "CRÍTICA" else "yellow" if sev in ["ALTA", "MODERADA"] else "green"
        diag_table.add_row("Severidad:", f"[{sev_color}]{sev}[/{sev_color}]")
        diag_table.add_row("Subsample recomendado:", f"cada {self.data_diagnosis['recommended_subsample']}")
        self.console.print(diag_table)
        
        # Aviso (NO destructivo) si la autocorrelación es crítica.
        # En ML supervisado para series temporales, alta autocorrelación NO
        # justifica diezmar: el modelo aprende pares (X, y), no requiere
        # independencia. Subsamplear elimina dinámica fina y desalinea train
        # vs inference (esta última no subsamplea).  Por eso solo emitimos
        # un warning y respetamos el valor del usuario.
        if self.data_diagnosis["autocorr_1"] > 0.98:
            recommended = self.data_diagnosis["recommended_subsample"]
            if self.subsample_step < recommended:
                self.console.print(
                    f"[yellow]   ⚠️  Autocorrelación lag-1 = "
                    f"{self.data_diagnosis['autocorr_1']:.4f} (CRÍTICA). "
                    f"Subsample del usuario = {self.subsample_step}. "
                    f"Si la dinámica del proceso es muy lenta, evaluar "
                    f"manualmente subsample ≈ {recommended}.[/yellow]"
                )

        # ═══════════════════════════════════════════════════════════════════
        # SUBSAMPLEO TEMPORAL
        # ═══════════════════════════════════════════════════════════════════
        # Default = 1 (sin subsampling). En ML supervisado, subsamplear una
        # serie para "descorrelacionar" es un anti-patrón heredado de la
        # inferencia estadística clásica: aquí solo destruye señal y crea
        # desalineación con la inferencia (que no subsamplea). Mantener > 1
        # solo si se quiere reducir el costo cuadrático del GP.
        if self.subsample_step > 1:
            df = df.iloc[::self.subsample_step]
            self.console.print(
                f"[dim]   Subsampleado 1/{self.subsample_step}: {len(df):,} filas[/dim]"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # FEATURE ENGINEERING
        # ═══════════════════════════════════════════════════════════════════
        df = self._create_lag_features(df, self.target_col)
        df = df.dropna()  # Los lags crean NaNs al inicio
        self.console.print(
            f"[dim]   Con feature engineering: {len(df):,} filas, {len(df.columns)} columnas[/dim]"
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # SEPARAR X e Y
        # ═══════════════════════════════════════════════════════════════════
        y_series = df[self.target_col]
        
        # ═══════════════════════════════════════════════════════════════════
        # [v4.1.0] FIX: Eliminado hardcode de "_iron_concentrate"
        # ═══════════════════════════════════════════════════════════════════
        # ANTES:
        #   drop_cols = [self.target_col, "_iron_concentrate"]  # ❌ Hardcode
        #
        # AHORA:
        #   Solo eliminamos el target. El sistema de remove_correlated_features
        #   se encargará de eliminar columnas redundantes automáticamente.
        #   Esto hace que el código sea verdaderamente "universal" y funcione
        #   con cualquier dataset minero (hierro, oro, cobre, etc.)
        # ═══════════════════════════════════════════════════════════════════
        drop_cols = [self.target_col]  # ✅ Solo el target, nada hardcodeado
        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # ═══════════════════════════════════════════════════════════════════
        
        # Eliminar features problemáticos (constantes, correlacionados)
        X_df = self._remove_problematic_features(X_df)
        
        # Guardar nombres de features para inferencia
        self.feature_names = X_df.columns.tolist()
        
        # Convertir a numpy arrays SIN escalar.
        # El escalado se hace en train_from_file() después del split para evitar
        # leakage de la mediana/IQR del test set hacia el train.
        X = X_df.values
        y = y_series.values.reshape(-1, 1)

        # Mostrar estadísticas finales
        new_autocorr = y_series.autocorr(lag=1) if len(y_series) > 1 else 0
        self.console.print(
            f"\n[green]✅ Datos listos: {X.shape[0]:,} filas, {X.shape[1]} features[/green]"
        )
        self.console.print(f"[dim]   Nueva autocorr lag-1: {new_autocorr:.4f}[/dim]")

        if new_autocorr > 0.9:
            self.console.print(
                f"[yellow]   ⚠️  Autocorrelación aún alta. "
                f"Considerar aumentar subsample.[/yellow]"
            )

        return X, y, df.index
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════
    
    def _train_gp(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        n_trials: int
    ) -> Tuple[any, Dict, float]:
        """
        Entrena Gaussian Process con optimización bayesiana de hiperparámetros.
        
        Usa Optuna para encontrar los mejores valores de:
        - alpha: ruido de regularización
        - length_scale: escala del kernel Matern
        - nu: suavidad del kernel (1.5 o 2.5)
        - noise_level: ruido del kernel WhiteKernel
        
        Args:
            X_train: Features de entrenamiento (escalados)
            y_train: Target de entrenamiento (escalado)
            n_trials: Número de trials de optimización
            
        Returns:
            Tuple de (modelo_sin_entrenar, mejores_params, score_cv)
        """
        
        def objective(trial):
            """Función objetivo para Optuna."""
            # Sugerir hiperparámetros
            alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
            length_scale = trial.suggest_float("length_scale", 1.0, 25.0, log=True)
            nu = trial.suggest_categorical("nu", [1.5, 2.5])
            noise = trial.suggest_float("noise_level", 0.001, 0.1, log=True)
            
            # Construir kernel compuesto
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                Matern(length_scale=length_scale, nu=nu, length_scale_bounds=(0.01, 100)) +
                WhiteKernel(noise_level=noise, noise_level_bounds=(0.01, 10))
            )
            
            # Crear modelo
            model = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=alpha,
                random_state=self.random_state,
                n_restarts_optimizer=2
            )
            
            # Subsamplear para velocidad en optimización
            max_samples = min(600, len(X_train))
            step = max(1, len(X_train) // max_samples)
            X_opt = X_train[::step][:max_samples]
            y_opt = y_train[::step][:max_samples]
            
            # Cross-validation temporal (respeta orden cronológico).
            # TimeSeriesSplit requiere n_samples >= n_splits + 1; en datasets
            # post-FE muy reducidos (ej. tests sintéticos) puede fallar, así
            # que ajustamos n_splits dinámicamente.
            n_splits = min(3, max(2, len(X_opt) - 1))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_opt):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_opt[train_idx], y_opt[train_idx].ravel())
                    pred = model.predict(X_opt[test_idx])
                    score = r2_score(y_opt[test_idx], pred)
                    scores.append(max(score, -1.0))  # Clamp negatives
                except:
                    return -1.0
            
            return np.mean(scores)
        
        # Ejecutar optimización con sampler seedeado para reproducibilidad
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Si todos los trials fallaron (datasets degenerados, NaN en CV, etc.)
        # study.best_params lanza ValueError. Devolvemos señal al caller para
        # que caiga al fallback en lugar de propagar la excepción.
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            self.console.print(
                "[yellow]   ⚠️  Todos los trials de Optuna fallaron. "
                "Señalando fallback al caller.[/yellow]"
            )
            return None, {}, -1.0

        best_params = study.best_params
        best_score = study.best_value
        
        # Reconstruir el mejor modelo
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(
                length_scale=best_params["length_scale"], 
                nu=best_params["nu"],
                length_scale_bounds=(0.01, 100)
            ) +
            WhiteKernel(
                noise_level=best_params["noise_level"],
                noise_level_bounds=(0.01, 10)
            )
        )
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=best_params["alpha"],
            n_restarts_optimizer=3,
            random_state=self.random_state
        )
        
        return model, best_params, best_score
    
    def _train_fallback(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Tuple[any, Dict]:
        """
        Modelo alternativo: GradientBoosting.
        
        Se usa cuando el GP no logra R² > 0.6, lo cual indica que
        el problema probablemente no es suave/estacionario.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Tuple de (modelo_sin_entrenar, params)
        """
        self.console.print(
            "[yellow]🔄 Gaussian Process falló. "
            "Usando GradientBoosting como alternativa...[/yellow]"
        )
        
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        params = {
            "model": "GradientBoosting", 
            "n_estimators": 150, 
            "max_depth": 4
        }
        
        return model, params
    
    def optimize_and_train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_trials: int = None
    ) -> None:
        """
        Optimiza hiperparámetros y entrena el modelo final.
        
        Pipeline:
        1. Optimizar GP con Optuna
        2. Evaluar CV score
        3. Si CV < 0.6, cambiar a GradientBoosting
        4. Entrenar modelo final con todos los datos
        
        Args:
            X: Features escalados
            y: Target escalado
            n_trials: Número de trials Optuna (usa CONFIG si es None)
        """
        n_trials = n_trials or CONFIG.GP_OPTUNA_TRIALS
        max_samples = CONFIG.GP_MAX_TRAIN_SAMPLES
        
        self.console.print(
            f"\n[bold yellow]⚡ Optimizando Gaussian Process "
            f"({n_trials} trials)...[/bold yellow]"
        )
        
        # Fase 1: Optimización
        model, params, cv_score = self._train_gp(X, y, n_trials)
        
        self.console.print(f"\n[bold]CV Score: R² = {cv_score:.4f}[/bold]")
        
        # Fase 2: Decidir modelo final
        if cv_score < 0.60 and self.use_fallback_model:
            self.console.print(
                f"[red]❌ GP no alcanzó R² > 0.6. Cambiando a modelo alternativo.[/red]"
            )
            model, params = self._train_fallback(X, y)
            self.model_type = "GradientBoosting"
        else:
            self.model_type = "GP"
        
        self.best_params = params
        
        # Mostrar parámetros
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Parámetro")
        table.add_column("Valor")
        for k, v in params.items():
            val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
            table.add_row(k, val_str)
        self.console.print(table)
        
        # Fase 3: Entrenar con datos limitados (GP es O(n³))
        if len(X) > max_samples:
            step = max(1, len(X) // max_samples)
            indices = np.arange(0, len(X), step)[:max_samples]
            X_train, y_train = X[indices], y[indices]
            self.console.print(
                f"[dim]Entrenando con {len(X_train):,} de {len(X):,} muestras "
                f"(límite de memoria)[/dim]"
            )
        else:
            X_train, y_train = X, y
        
        self.console.print(f"[bold blue]🚀 Entrenando {self.model_type}...[/bold blue]")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train.ravel())
        
        self.model = model
        self.console.print(f"[green]✅ {self.model_type} entrenado exitosamente[/green]")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE PREDICCIÓN Y EVALUACIÓN
    # ═══════════════════════════════════════════════════════════════════════
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera predicciones con incertidumbre.
        
        Args:
            X: Features escalados
            
        Returns:
            Tuple de (predicciones, desviaciones_estándar) en escala original
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_from_file() primero.")
        
        # Predecir (GP devuelve incertidumbre, GradientBoosting no)
        if self.model_type == "GP":
            y_pred_scaled, y_std_scaled = self.model.predict(X, return_std=True)
        else:
            y_pred_scaled = self.model.predict(X)
            y_std_scaled = np.zeros_like(y_pred_scaled)
        
        # Desescalar predicciones
        y_pred = self.scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()
        
        # Desescalar incertidumbre
        if hasattr(self.scaler_y, 'scale_') and self.scaler_y.scale_ is not None:
            y_std = y_std_scaled * self.scaler_y.scale_[0]
        else:
            y_std = y_std_scaled
        
        return y_pred, y_std
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Calcula métricas de evaluación.
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            ModelMetrics con R², RMSE, MAE, MAPE
        """
        # MAPE solo donde y_true != 0 para evitar división por cero
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        self.metrics = ModelMetrics(
            r2=r2_score(y_true, y_pred),
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            mape=mape
        )
        
        return self.metrics
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE PERSISTENCIA
    # ═══════════════════════════════════════════════════════════════════════
    
    def save(self, filepath: str = None) -> str:
        """
        Guarda el modelo y artefactos a disco.
        
        Args:
            filepath: Ruta destino. Si None, genera nombre automático.
            
        Returns:
            Ruta del archivo guardado
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        # Generar nombre si no se especifica
        if filepath is None:
            CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(
                CONFIG.MODELS_DIR / 
                f"{self.model_type.lower()}_{self.target_col}_{timestamp}.pkl"
            )
        
        # Empaquetar artefactos
        artifacts = TrainingArtifacts(
            model=self.model,
            scaler_X=self.scaler_X,
            scaler_y=self.scaler_y,
            feature_names=self.feature_names,
            target_column=self.target_col,
            best_params=self.best_params,
            metrics=self.metrics,
            model_type=self.model_type,
            removed_features=self.removed_features
        )
        
        joblib.dump(artifacts, filepath)
        self.console.print(f"[green]💾 Modelo guardado: {filepath}[/green]")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Carga modelo y artefactos desde disco.
        
        Args:
            filepath: Ruta al archivo .pkl
        """
        artifacts: TrainingArtifacts = joblib.load(filepath)
        
        self.model = artifacts.model
        self.scaler_X = artifacts.scaler_X
        self.scaler_y = artifacts.scaler_y
        self.feature_names = artifacts.feature_names
        self.target_col = artifacts.target_column
        self.best_params = artifacts.best_params
        self.metrics = artifacts.metrics
        self.model_type = getattr(artifacts, 'model_type', 'GP')
        self.removed_features = getattr(artifacts, 'removed_features', [])
        
        self.console.print(
            f"[green]📂 Modelo cargado: {filepath} ({self.model_type})[/green]"
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS DE VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_report(
        self, 
        y_true, 
        y_pred, 
        y_std, 
        dates, 
        output_dir=None
    ) -> List[str]:
        """
        Genera gráficos de diagnóstico del modelo.
        
        Crea un panel con 4 gráficos:
        1. Serie temporal: predicción vs real
        2. Scatter plot: correlación predicho vs real
        3. Histograma de errores
        4. Residuos vs predicción
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            y_std: Desviación estándar de predicciones
            dates: Índice temporal
            output_dir: Directorio de salida
            
        Returns:
            Lista de rutas de archivos generados
        """
        output_dir = Path(output_dir or CONFIG.RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # --- Gráfico 1: Serie Temporal ---
        n = min(200, len(y_true))  # Últimas 200 observaciones
        axes[0, 0].plot(dates[-n:], y_true[-n:], 'k-', lw=1, alpha=0.8, label='Real')
        axes[0, 0].plot(dates[-n:], y_pred[-n:], 'r--', lw=1.5, label='Predicción')
        
        # Banda de confianza 95%
        if np.any(y_std > 0):
            axes[0, 0].fill_between(
                dates[-n:], 
                y_pred[-n:] - 1.96 * y_std[-n:],
                y_pred[-n:] + 1.96 * y_std[-n:], 
                color='red', alpha=0.15, label='IC 95%'
            )
        
        axes[0, 0].set_title(f'Serie Temporal ({self.model_type})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # --- Gráfico 2: Scatter Plot ---
        axes[0, 1].scatter(y_true, y_pred, alpha=0.4, s=10, c='steelblue')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0, 1].plot(lims, lims, 'k--', lw=2, label='Línea perfecta')
        axes[0, 1].set_title(f'R² = {self.metrics.r2:.4f}')
        axes[0, 1].set_xlabel('Valor Real')
        axes[0, 1].set_ylabel('Valor Predicho')
        axes[0, 1].legend()
        
        # --- Gráfico 3: Histograma de Errores ---
        errors = y_true - y_pred
        axes[1, 0].hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1, 0].axvline(0, color='red', ls='--', lw=2, label='Error = 0')
        axes[1, 0].set_title('Distribución de Errores')
        axes[1, 0].set_xlabel('Error (Real - Predicho)')
        axes[1, 0].legend()
        
        # --- Gráfico 4: Residuos vs Predicción ---
        axes[1, 1].scatter(y_pred, errors, alpha=0.3, s=10, c='steelblue')
        axes[1, 1].axhline(0, color='red', ls='--', lw=2)
        axes[1, 1].set_title('Residuos vs Predicción (detecta heterocedasticidad)')
        axes[1, 1].set_xlabel('Valor Predicho')
        axes[1, 1].set_ylabel('Residuo')
        
        plt.suptitle(
            f'{self.model_type} | Target: {self.target_col}', 
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        # Guardar figura
        path = output_dir / f"{self.model_type.lower()}_report_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green]📊 Reporte guardado: {path}[/green]")
        
        return [str(path)]
    
    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE COMPLETO
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_from_file(
        self, 
        filepath=None, 
        test_size=0.2, 
        n_trials=None, 
        save_model=True
    ) -> ModelMetrics:
        """
        Pipeline completo: carga datos, entrena, evalúa y guarda.
        
        Este es el método principal para uso típico. Ejecuta todos los
        pasos necesarios de principio a fin.
        
        Args:
            filepath: Ruta al CSV. Si None, usa CONFIG.DATA_CLEAN_PATH
            test_size: Proporción de datos para test (default 20%)
            n_trials: Número de trials Optuna
            save_model: Si True, guarda el modelo entrenado
            
        Returns:
            ModelMetrics con los resultados de evaluación
        """
        # Paso 1: Cargar y preparar datos (X, y SIN escalar)
        X, y, dates = self.load_data(filepath)

        # Paso 2: Split temporal (respeta orden cronológico)
        test_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:test_idx], X[test_idx:]
        y_train, y_test = y[:test_idx], y[test_idx:]
        dates_test = dates[test_idx:]

        # Paso 3: Escalar — fit SOLO con train, transform en test (anti-leakage).
        X_train_s = self.scaler_X.fit_transform(X_train)
        X_test_s = self.scaler_X.transform(X_test)
        y_train_s = self.scaler_y.fit_transform(y_train)

        # Paso 4: Entrenar (en escala escalada)
        self.optimize_and_train(X_train_s, y_train_s, n_trials=n_trials)

        # Paso 5: Evaluar en test set (y_test ya está en escala original)
        y_test_real = y_test.ravel()
        y_pred, y_std = self.predict(X_test_s)
        metrics = self.evaluate(y_test_real, y_pred)
        
        # Mostrar resultados
        self.console.print("\n" + "=" * 50)
        self.console.print(f"[bold]🏆 RESULTADOS FINALES ({self.model_type})[/bold]")
        self.console.print("=" * 50)
        
        table = Table(header_style="bold green")
        table.add_column("Métrica")
        table.add_column("Valor")
        table.add_column("Interpretación")
        
        # R² con color según calidad
        r2_color = "green" if metrics.r2 > 0.7 else "yellow" if metrics.r2 > 0.5 else "red"
        r2_interp = "Excelente" if metrics.r2 > 0.8 else "Bueno" if metrics.r2 > 0.6 else "Pobre"
        table.add_row("R² Score", f"[{r2_color}]{metrics.r2:.4f}[/{r2_color}]", r2_interp)
        table.add_row("RMSE", f"{metrics.rmse:.4f}", "Error típico")
        table.add_row("MAE", f"{metrics.mae:.4f}", "Error absoluto promedio")
        table.add_row("MAPE", f"{metrics.mape:.2f}%", "Error porcentual")
        
        self.console.print(table)
        
        # Paso 5: Generar reporte visual
        self.generate_report(y_test_real, y_pred, y_std, dates_test)
        
        # Paso 6: Guardar modelo
        if save_model:
            self.save()
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS PÚBLICOS
# ═══════════════════════════════════════════════════════════════════════════
__all__ = ["MiningGP", "ModelMetrics", "TrainingArtifacts"]


# ═══════════════════════════════════════════════════════════════════════════
# CLI (Command Line Interface)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Punto de entrada para uso desde línea de comandos.
    
    Ejemplos:
        python mining_gp_pro.py --data data/clean.csv --target _silica_concentrate
        python mining_gp_pro.py --trials 30 --subsample 20
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Soft-Sensor GP v4.1.0 (Universal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python mining_gp_pro.py
  python mining_gp_pro.py --data data/gold.csv --target recovery
  python mining_gp_pro.py --trials 30 --subsample 20 --no-fallback
        """
    )
    parser.add_argument("--data", "-d", type=str, default=None,
                       help="Ruta al archivo CSV (default: usa CONFIG)")
    parser.add_argument("--target", "-t", type=str, default=None,
                       help="Columna objetivo (default: usa CONFIG)")
    parser.add_argument("--trials", "-n", type=int, default=None,
                       help="Número de trials Optuna (default: 15)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proporción de test (default: 0.2)")
    parser.add_argument("--subsample", "-s", type=int, default=None,
                       help="Subsample step (default: usa CONFIG)")
    parser.add_argument("--no-lags", action="store_true",
                       help="Desactivar features de lag")
    parser.add_argument("--no-fallback", action="store_true",
                       help="No usar GradientBoosting como alternativa")
    parser.add_argument("--no-save", action="store_true",
                       help="No guardar el modelo")
    
    args = parser.parse_args()
    
    try:
        model = MiningGP(
            target_col=args.target,
            subsample_step=args.subsample,
            add_lag_features=not args.no_lags,
            use_fallback_model=not args.no_fallback
        )
        
        metrics = model.train_from_file(
            filepath=args.data,
            test_size=args.test_size,
            n_trials=args.trials,
            save_model=not args.no_save
        )
        
        # Exit code basado en calidad del modelo
        exit(0 if metrics.r2 > 0 else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
