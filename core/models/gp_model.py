"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/models/gp_model.py
Proyecto: Universal Soft-Sensor
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

    from core.models.gp_model import SoftSensorGP
    
    # Entrenamiento completo desde archivo
    model = SoftSensorGP(target_col="_silica_concentrate")
    metrics = model.train_from_file("data/processed/mining_clean.csv")
    
    # Predicción
    y_pred, y_std = model.predict(X_new)

═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════
import os
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
from sklearn.model_selection import (
    TimeSeriesSplit, GroupKFold, KFold, GroupShuffleSplit, cross_val_predict
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Rich - Interfaz de usuario bonita
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Configuración centralizada del proyecto
from config.settings import CONFIG
from core.scientific_report import (
    apply_scientific_style,
    plot_permutation_test,
    plot_feature_importance,
    build_scientific_pdf,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# Silenciar warnings molestos de sklearn y numpy
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# [scientific_report] Estilo de gráficos: intenta SciencePlots (look de
# publicación IEEE/Nature) primero, cae a seaborn-whitegrid si no está
# instalado. Antes esto era un try/except inline duplicado aquí; ahora vive
# en un solo lugar (core/scientific_report.py) para no divergir entre módulos.
_ESTILO_GRAFICO_ACTIVO = apply_scientific_style()


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
        permutation_p_value: [P0b ROADMAP] p-value del permutation test, si
            se corrió (train_from_file(..., run_permutation_test=True)).
            None si no se corrió. Un R² alto con p-value alto (>0.05) es la
            firma clásica de sobreajuste/leakage — ver permutation_test().
        nll: [P6 ROADMAP — calibración] Negative Log-Likelihood gaussiana
            media sobre el test set: 0.5·mean[ log(2π·σ²) + (y-μ)²/σ² ].
            Mide qué tan bien calibrada está la incertidumbre σ del GP, no
            solo el error puntual: penaliza tanto errar la media como estar
            sobre-confiado (σ chico con error grande) o sub-confiado (σ grande
            innecesario). Menor = mejor. SOLO se calcula para el GP (que da σ
            real); para el fallback GradientBoosting (σ=0) queda None, porque
            la NLL no está definida sin incertidumbre — reportar un número ahí
            sería engañoso.
        coverage_95: [P6 ROADMAP — calibración] Fracción de puntos de test que
            caen dentro del intervalo de confianza del 95% (|y-μ| ≤ 1.96σ).
            Un GP bien calibrado da ~0.95. <0.95 = sobre-confiado (bandas muy
            angostas); >0.95 = sub-confiado (bandas muy anchas). Igual que NLL,
            solo aplica al GP; None para el fallback GB.
        sharpness: [P3 ROADMAP — calibración] Ancho medio del intervalo de
            confianza del 95%: mean(2·1.96·σ), en las MISMAS unidades que el
            target. Complemento obligatorio de coverage_95: la cobertura sola
            es engañable — un modelo logra cobertura ~1.0 con bandas
            absurdamente anchas (inútiles en la práctica). Se leen juntas: la
            meta es cobertura ≈0.95 CON la menor sharpness posible (bandas
            angostas pero honestas). Menor = mejor, PERO solo si la cobertura
            se mantiene cerca de 0.95. Solo aplica al GP; None para el GB.
        baseline_r2: [P2 ROADMAP — baseline naive] R² del predictor trivial que
            SIEMPRE predice la media del TARGET DE ENTRENAMIENTO, evaluado sobre
            el mismo test set. Es el "piso" contra el que se lee el R² del
            modelo: un R²=0.32 no significa nada sin saber que el baseline da,
            por ejemplo, 0.00 — recién ahí se ve que el modelo aporta señal
            real sobre no-modelo. CRÍTICO: la media viene del TRAIN, no del
            test; usar la media del test daría exactamente 0.0 por definición
            (tautológico, R² se define contra la media del propio conjunto).
            Con la media del train puede ser negativo si train y test difieren
            — y eso también es información honesta. None si no se provee la
            media del train a evaluate().
    """
    r2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    permutation_p_value: Optional[float] = None
    nll: Optional[float] = None
    coverage_95: Optional[float] = None
    sharpness: Optional[float] = None
    baseline_r2: Optional[float] = None

    def to_dict(self) -> dict:
        """Convierte las métricas a diccionario (útil para JSON)."""
        d = {"r2": self.r2, "rmse": self.rmse, "mae": self.mae, "mape": self.mape}
        if self.permutation_p_value is not None:
            d["permutation_p_value"] = self.permutation_p_value
        if self.nll is not None:
            d["nll"] = self.nll
        if self.coverage_95 is not None:
            d["coverage_95"] = self.coverage_95
        if self.sharpness is not None:
            d["sharpness"] = self.sharpness
        if self.baseline_r2 is not None:
            d["baseline_r2"] = self.baseline_r2
        return d

    def __repr__(self) -> str:
        base = f"R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"
        if self.baseline_r2 is not None:
            base += f", R²base={self.baseline_r2:.4f}"
        if self.permutation_p_value is not None:
            base += f", p-perm={self.permutation_p_value:.4f}"
        if self.nll is not None:
            base += f", NLL={self.nll:.4f}"
        if self.coverage_95 is not None:
            base += f", Cov95={self.coverage_95:.3f}"
        if self.sharpness is not None:
            base += f", Sharp={self.sharpness:.4f}"
        return base


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
# CLASE PRINCIPAL: SoftSensorGP
# ═══════════════════════════════════════════════════════════════════════════

class SoftSensorGP:
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
        >>> model = SoftSensorGP(target_col="rougher.output.recovery")
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
        add_input_lags: bool = False,  # [P0 ROADMAP] lags de variables de ENTRADA
        input_lag_periods: List[int] = None,
        input_lag_columns: List[str] = None,
        parse_dates: bool = True,  # [P0b ROADMAP] False para datasets sin dimensión temporal
        group_column: str = None,  # [P0b ROADMAP] columna de agrupamiento (ej. HOLEID)
        use_fallback_model: bool = True,
        remove_constant_features: bool = True,
        remove_correlated_features: bool = True,
        correlation_threshold: float = 0.98,
        strict_leakage: bool = False  # [SECURITY V2] True → abortar si hay leakage feature↔target
    ):
        """
        Inicializa el Soft-Sensor.

        Args:
            target_col: Columna objetivo a predecir. Si es None, usa CONFIG.GP_TARGET_COLUMN
            random_state: Semilla para reproducibilidad
            subsample_step: Cada cuántas filas tomar una muestra.
                           [v4.1.0] Si es None, usa CONFIG.DEFAULT_SUBSAMPLE_STEP
            add_lag_features: Si True, agrega features de lag temporal DEL TARGET
            lag_periods: Lista de periodos de lag del target [1, 5, 10, 20] por defecto
            add_diff_features: Si True, agrega diferencias y promedios móviles DEL TARGET
            add_input_lags: [P0 ROADMAP] Si True, agrega lags de las variables de
                           ENTRADA (no del target). Default False para no alterar el
                           comportamiento/dimensionalidad de modelos existentes.
                           Necesario en procesos con retardo/tiempo de residencia
                           (ver ROADMAP.md P0: SRU dio R²=-0.47 sensor-only instantáneo
                           porque el pipeline solo lageaba el target, nunca las entradas).
            input_lag_periods: Periodos de lag para las entradas [1, 2, 3] por defecto.
                           Intencionalmente más corto que lag_periods del target: modela
                           retardo de proceso, no autocorrelación de largo plazo.
            input_lag_columns: Columnas de entrada a laguear. Si None, se aplica a
                           TODAS las columnas numéricas de entrada (el target nunca se
                           incluye, y el guard de leakage + remove_correlated_features
                           corren después para podar lo redundante).
            parse_dates: [P0b ROADMAP] Si False, el índice del CSV NO se interpreta
                           como fecha y se desactivan el diagnóstico de autocorrelación
                           y el subsampleo temporal (ambos asumen dimensión temporal).
                           Usar False para datasets geometalúrgicos/spatial sin tiempo
                           (ej. una fila por sondaje/muestra, sin timestamp real).
            group_column: [P0b ROADMAP] Nombre de columna de agrupamiento (ej. "HOLEID"
                           en datos de sondajes). Si se especifica: (1) se excluye de
                           las features, (2) el split train/test usa GroupShuffleSplit
                           (ningún grupo aparece en ambos lados), (3) la CV interna de
                           Optuna usa GroupKFold en vez de TimeSeriesSplit. Motivación:
                           varias muestras del mismo sondaje están correlacionadas —
                           un split que las separa entre train/test infla el R² por
                           leakage de grupo (ver ROADMAP.md P0b y run_geomet_rigor.py,
                           que casi reportó R²=0.93 falso en vez del 0.33 real).
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
        
        # Configuración de feature engineering (lags/diffs del TARGET)
        self.add_lag_features = add_lag_features
        self.lag_periods = lag_periods or [1, 5, 10, 20]
        self.add_diff_features = add_diff_features

        # ═══════════════════════════════════════════════════════════════════
        # [P0 ROADMAP] Feature engineering de lags de ENTRADAS (no del target)
        # ═══════════════════════════════════════════════════════════════════
        # Motivación: el pipeline históricamente solo lageaba el target. En
        # procesos con retardo/tiempo de residencia (ej. SRU, columnas de
        # flotación) el efecto de un cambio de entrada aparece varios pasos
        # DESPUÉS — sin historia de las entradas el modelo solo ve el estado
        # instantáneo. Ver results/verification/ y ROADMAP.md P0.
        # ═══════════════════════════════════════════════════════════════════
        self.add_input_lags = add_input_lags
        self.input_lag_periods = input_lag_periods or [1, 2, 3]
        self.input_lag_columns = input_lag_columns  # None = todas las numéricas de entrada

        # ═══════════════════════════════════════════════════════════════════
        # [P0b ROADMAP] Datos no-temporales y split/CV por grupo
        # ═══════════════════════════════════════════════════════════════════
        self.parse_dates = parse_dates
        self.group_column = group_column
        self.groups_: Optional[np.ndarray] = None  # poblado por load_data() si hay group_column

        # Configuración de comportamiento
        self.use_fallback_model = use_fallback_model
        self.remove_constant_features = remove_constant_features
        self.remove_correlated_features = remove_correlated_features
        self.correlation_threshold = correlation_threshold
        self.strict_leakage = strict_leakage
        
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
            f"SoftSensorGP v4.1.0 inicializado - "
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
    
    def _drop_target_leakage(
        self, X_df: pd.DataFrame, y_series: pd.Series, thresh: float = 0.999
    ) -> pd.DataFrame:
        """
        [SECURITY V2] Detecta y elimina features que son copia (o cuasi-copia)
        del target — la fuente #1 de R² inflado y fraudulento en ML industrial.

        Calcula |corr(feature, target)| y elimina toda feature que supere
        `thresh`. Emite un WARNING ruidoso (no silencioso) porque, aunque el
        caso legítimo existe (p. ej. una réplica redundante de un sensor de
        salida), un usuario debe SIEMPRE enterarse de que una de sus features
        es prácticamente el target.

        Se puede endurecer a "abortar" con self.strict_leakage=True.
        """
        if X_df.shape[1] == 0 or len(y_series) < 3:
            return X_df
        y = pd.Series(np.asarray(y_series, dtype=float).ravel(), index=X_df.index)
        leaked = []
        for col in X_df.columns:
            try:
                x = pd.to_numeric(X_df[col], errors="coerce")
                if x.notna().sum() < 3 or x.std(skipna=True) == 0:
                    continue
                r = x.corr(y)
                if pd.notna(r) and abs(r) >= thresh:
                    leaked.append((col, float(r)))
            except Exception:
                continue
        if leaked:
            names = ", ".join(f"{c} (|r|={abs(r):.4f})" for c, r in leaked)
            msg = (f"⚠️  LEAKAGE DETECTADO: {len(leaked)} feature(s) cuasi-idéntica(s) "
                   f"al target y eliminada(s): {names}. Un R² alto SIN estas columnas "
                   f"es el número honesto.")
            self.console.print(f"[bold red]   {msg}[/bold red]")
            logger.warning(msg)
            if getattr(self, "strict_leakage", False):
                raise ValueError(f"Leakage feature↔target (strict_leakage=True): {names}")
            X_df = X_df.drop(columns=[c for c, _ in leaked])
        return X_df

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

    def _create_input_lag_features(
        self, df: pd.DataFrame, y_col: str, columns: List[str] = None
    ) -> pd.DataFrame:
        """
        [P0 ROADMAP] Crea lags de las variables de ENTRADA (no del target).

        A diferencia de `_create_lag_features` (que lagea el TARGET y por eso
        necesita cuidado especial anti-leakage vía shift(1) en diffs/rolling),
        laguear una columna de ENTRADA con shift(n), n>=1, es causal por
        construcción: nunca usa información futura respecto al target en la
        misma fila. No requiere el mismo blindaje, pero sí debe excluir
        siempre al target (nunca se lagea el target aquí — eso ya lo hace
        `_create_lag_features`).

        Debe llamarse ANTES de `_create_lag_features` en `load_data()`, sobre
        las columnas crudas: si corriera después, lagearía también las
        columnas `{target}_lag_N`/`{target}_diff_N` ya creadas, produciendo
        lags-de-lags sin sentido físico.

        Args:
            df: DataFrame con datos crudos (target + inputs), sin FE previo.
            y_col: Columna objetivo — siempre excluida de `columns`.
            columns: Columnas de entrada a laguear. Si None, usa TODAS las
                columnas numéricas del DataFrame excepto el target. El guard
                de leakage (`_drop_target_leakage`) y la poda de
                correlacionados corren después en `load_data()`, así que un
                exceso de columnas generadas aquí se filtra automáticamente.

        Returns:
            DataFrame con columnas adicionales `'{col}_lag_{n}'` por cada
            columna de entrada y periodo en `self.input_lag_periods`. Si
            `self.add_input_lags` es False, retorna `df` sin modificar.
        """
        if not self.add_input_lags:
            return df

        df = df.copy()

        if columns is None:
            # [P0b] group_column (ej. HOLEID) nunca se lagea por defecto: es un
            # identificador, no una señal de proceso — lagearlo produce ruido.
            excluded = {y_col}
            if self.group_column:
                excluded.add(self.group_column)
            target_columns = [
                c for c in df.columns
                if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            # Filtra a columnas presentes en df y nunca incluye al target,
            # aunque el caller lo haya pasado por error en `columns`.
            target_columns = [
                c for c in columns
                if c in df.columns and c != y_col and c != self.group_column
            ]

        for col in target_columns:
            for lag in self.input_lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

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
        # [P0b ROADMAP] Con parse_dates=True (default): la primera columna se
        # asume índice temporal (index_col=0), comportamiento original. Con
        # parse_dates=False (datasets sin dimensión temporal, ej.
        # geometalúrgicos): NO se fuerza ninguna columna como índice — si se
        # forzara index_col=0 igual, la primera columna (a veces el propio
        # group_column, ej. HOLEID) desaparecería de df.columns antes de
        # poder usarse como grupo.
        if self.parse_dates:
            df = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
                skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
            )
        else:
            df = pd.read_csv(
                filepath,
                skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
            )

        # Validar que existe el target
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target '{self.target_col}' no encontrado en el dataset.\n"
                f"Columnas disponibles: {list(df.columns[:10])}..."
            )

        # [P0b ROADMAP] Validar que la columna de agrupamiento existe, si se pidió.
        if self.group_column and self.group_column not in df.columns:
            raise ValueError(
                f"group_column '{self.group_column}' no encontrada en el dataset.\n"
                f"Columnas disponibles: {list(df.columns[:10])}..."
            )

        # ═══════════════════════════════════════════════════════════════════
        # [SECURITY V5] Validación amistosa de entrada — atrapa datos corruptos
        # ANTES del modelo, con mensajes claros en vez de stacktraces crípticos
        # de sklearn/RobustScaler aguas abajo.
        # ═══════════════════════════════════════════════════════════════════
        _yt = pd.to_numeric(df[self.target_col], errors="coerce")
        if _yt.notna().sum() == 0:
            raise ValueError(
                f"El target '{self.target_col}' no es numérico (no se pudo "
                f"convertir ningún valor a número). Un soft-sensor de regresión "
                f"requiere un target numérico continuo."
            )
        _MIN_FILAS = 10
        _n_validas = df.dropna(subset=[self.target_col]).shape[0]
        if _n_validas < _MIN_FILAS:
            raise ValueError(
                f"Datos insuficientes: solo {_n_validas} fila(s) con target válido "
                f"(mínimo {_MIN_FILAS}). Revisa el CSV, el split o los NaN del target."
            )
        if np.isinf(_yt.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(
                f"El target '{self.target_col}' contiene valores infinitos (inf). "
                f"Limpia o acota el target antes de entrenar."
            )
        if _yt.std(skipna=True) == 0:
            raise ValueError(
                f"El target '{self.target_col}' es CONSTANTE (varianza cero): no "
                f"hay nada que aprender ni métrica de regresión definida. Revisa el "
                f"target o el rango de datos seleccionado."
            )

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNÓSTICO DE AUTOCORRELACIÓN
        # ═══════════════════════════════════════════════════════════════════
        # [P0b ROADMAP] Autocorrelación y subsampleo asumen orden cronológico.
        # Con parse_dates=False (datasets geometalúrgicos/spatial, una fila =
        # una muestra sin tiempo real) ambos conceptos no aplican: se saltan
        # explícitamente en vez de calcular un número sin sentido físico.
        if self.parse_dates:
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

            # ═══════════════════════════════════════════════════════════════
            # SUBSAMPLEO TEMPORAL
            # ═══════════════════════════════════════════════════════════════
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
        else:
            self.data_diagnosis = {"skipped_non_temporal": True}
            self.console.print(
                "[dim]   parse_dates=False: diagnóstico de autocorrelación y "
                "subsampleo temporal omitidos (no aplican sin dimensión "
                "temporal).[/dim]"
            )
            if self.subsample_step > 1:
                self.console.print(
                    f"[yellow]   ⚠️  subsample_step={self.subsample_step} se ignora "
                    f"con parse_dates=False (no hay orden cronológico que "
                    f"diezmar).[/yellow]"
                )

        # ═══════════════════════════════════════════════════════════════════
        # FEATURE ENGINEERING
        # ═══════════════════════════════════════════════════════════════════
        # [P0 ROADMAP] Lags de ENTRADAS primero, sobre columnas crudas —
        # antes de que _create_lag_features agregue columnas derivadas del
        # target (evitaría laguear lags/diffs sin sentido físico).
        df = self._create_input_lag_features(df, self.target_col, columns=self.input_lag_columns)
        df = self._create_lag_features(df, self.target_col)
        df = df.dropna()  # Los lags crean NaNs al inicio
        self.console.print(
            f"[dim]   Con feature engineering: {len(df):,} filas, {len(df.columns)} columnas[/dim]"
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # SEPARAR X e Y
        # ═══════════════════════════════════════════════════════════════════
        y_series = df[self.target_col]

        # [P0b ROADMAP] Extraer la columna de agrupamiento ANTES de dropearla
        # de X. Se captura acá (post feature-engineering + dropna) para que
        # quede alineada fila a fila con X/y finales. Se usa en train_from_file
        # para un split honesto (GroupShuffleSplit) y en _train_gp para
        # GroupKFold en la CV interna de Optuna.
        if self.group_column:
            self.groups_ = df[self.group_column].values
        else:
            self.groups_ = None

        # ═══════════════════════════════════════════════════════════════════
        # [v4.1.0] FIX: Eliminado hardcode de "_iron_concentrate"
        # ═══════════════════════════════════════════════════════════════════
        # ANTES:
        #   drop_cols = [self.target_col, "_iron_concentrate"]  # ❌ Hardcode
        #
        # AHORA:
        #   Solo eliminamos el target (y, si aplica, group_column — es un
        #   identificador, no una feature de proceso). El sistema de
        #   remove_correlated_features se encarga de eliminar columnas
        #   redundantes automáticamente. Esto hace que el código sea
        #   verdaderamente "universal" y funcione con cualquier dataset
        #   minero (hierro, oro, cobre, etc.)
        # ═══════════════════════════════════════════════════════════════════
        drop_cols = [self.target_col]  # ✅ Solo el target, nada hardcodeado
        if self.group_column:
            drop_cols.append(self.group_column)
        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # ═══════════════════════════════════════════════════════════════════

        # [SECURITY V2] Guard anti-leakage feature↔target. Una feature que es
        # copia (o cuasi-copia) del target produce R²≈1.0 FALSO, incluso con
        # target impredecible. El filtro de correlacionados de abajo solo mira
        # feature↔feature, NO feature↔target, así que una columna que replica
        # el target sobrevive. Aquí se detecta y se elimina con aviso ruidoso.
        X_df = self._drop_target_leakage(X_df, y_series)

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
        n_trials: int,
        groups: np.ndarray = None
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
            groups: [P0b ROADMAP] Si se pasa (alineado fila a fila con
                X_train/y_train), la CV interna usa GroupKFold en vez de
                TimeSeriesSplit — ningún grupo aparece en train y validación
                del mismo fold. Necesario cuando las filas no son una serie
                temporal sino muestras agrupadas (ej. varias muestras por
                sondaje/HOLEID), donde TimeSeriesSplit no tiene sentido y
                un split que mezcla el mismo grupo infla el R² por leakage.

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

            # [P0b ROADMAP] Cross-validation: GroupKFold si hay grupos (datos
            # no-temporales/agrupados), TimeSeriesSplit si no (comportamiento
            # original, respeta orden cronológico). Ambos requieren
            # n_samples/n_grupos >= n_splits + 1 aprox.; en datasets post-FE
            # muy reducidos (ej. tests sintéticos, GeoMet n=53) ajustamos
            # n_splits dinámicamente y clampeamos al Nº de grupos únicos.
            if groups is not None:
                groups_opt = groups[::step][:max_samples]
                n_groups_opt = len(np.unique(groups_opt))
                n_splits = max(2, min(3, n_groups_opt))
                if n_splits > n_groups_opt or n_groups_opt < 2:
                    # Muy pocos grupos para una CV con sentido — señal de
                    # fallo al caller en vez de un GroupKFold inválido.
                    return -1.0
                cv = GroupKFold(n_splits=n_splits)
                split_iter = cv.split(X_opt, y_opt, groups=groups_opt)
            else:
                n_splits = min(3, max(2, len(X_opt) - 1))
                cv = TimeSeriesSplit(n_splits=n_splits)
                split_iter = cv.split(X_opt)

            scores = []
            for train_idx, test_idx in split_iter:
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
        n_trials: int = None,
        groups: np.ndarray = None
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
            groups: [P0b ROADMAP] Grupos alineados con X/y para GroupKFold en
                la CV interna (ver _train_gp). None = comportamiento original.
        """
        n_trials = n_trials or CONFIG.GP_OPTUNA_TRIALS
        max_samples = CONFIG.GP_MAX_TRAIN_SAMPLES

        self.console.print(
            f"\n[bold yellow]⚡ Optimizando Gaussian Process "
            f"({n_trials} trials)...[/bold yellow]"
        )

        # Fase 1: Optimización
        model, params, cv_score = self._train_gp(X, y, n_trials, groups=groups)
        
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
        # [SECURITY V6] Muestreo ALEATORIO seedeado en vez de por stride fijo.
        # El stride (np.arange(0, n, step)) puede aliasar con estructura
        # periódica del dataset (p. ej. ciclos de un banco de pruebas) y sesgar
        # la muestra. Un muestreo aleatorio con random_state es determinista
        # (reproducible) pero robusto a esa periodicidad. Se ordena para
        # preservar el orden temporal relativo dentro de la submuestra.
        if len(X) > max_samples:
            rng = np.random.default_rng(self.random_state)
            indices = np.sort(rng.choice(len(X), size=max_samples, replace=False))
            X_train, y_train = X[indices], y[indices]
            self.console.print(
                f"[dim]Entrenando con {len(X_train):,} de {len(X):,} muestras "
                f"(muestreo aleatorio seedeado, límite de memoria O(n³))[/dim]"
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
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
        y_train_mean: Optional[float] = None,
    ) -> ModelMetrics:
        """
        Calcula métricas de evaluación.

        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            y_std: [P6 ROADMAP — calibración] Desviación estándar predicha por
                el modelo (la incertidumbre σ). Opcional y backward-compatible:
                si es None (o si es todo ceros, como en el fallback
                GradientBoosting) NO se calculan NLL ni Coverage@95 — quedan en
                None. Solo el GP entrega una σ real; calcular calibración sobre
                σ=0 daría división por cero enmascarada en un número basura, así
                que se evita explícitamente en vez de reportar algo engañoso.
            y_train_mean: [P2 ROADMAP — baseline naive] Media del target de
                ENTRENAMIENTO. Si se provee, se calcula baseline_r2: el R² de
                predecir siempre esa media constante sobre el test. Es el piso
                honesto contra el que se lee el R² del modelo. DEBE ser la media
                del train, no del test (la del test daría 0.0 tautológico). None
                → no se calcula baseline_r2 (backward-compatible).

        Returns:
            ModelMetrics con R², RMSE, MAE, MAPE y —según se provea— NLL/Cov95/
            sharpness (con σ real) y baseline_r2 (con la media de train).
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        # [SECURITY V3] R² es indefinido si el target de test tiene varianza
        # cero (constante): SS_tot = 0 y r2_score puede devolver 1.0 o 0.0,
        # ambos engañosos. En ese caso NO se reporta un número que parezca
        # bueno — se devuelve NaN y se avisa, igual que no se reporta una
        # métrica sobre un test sin soporte.
        if np.var(y_true) == 0:
            msg = ("⚠️  Target de test con varianza CERO (constante): R² es "
                   "matemáticamente indefinido. Se reporta NaN, no un valor "
                   "engañoso. Revisa el split y el target.")
            self.console.print(f"[bold red]   {msg}[/bold red]")
            logger.warning(msg)
            r2 = float("nan")
        else:
            r2 = r2_score(y_true, y_pred)

        # MAPE solo donde y_true != 0. Si TODOS son cero, MAPE es indefinido:
        # se devuelve NaN (no 0.0, que parecería "predicción perfecta").
        mask = y_true != 0
        if mask.any():
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("nan")

        # [P6 ROADMAP — calibración] NLL y Coverage@95 SOLO si hay una σ real.
        # Condiciones para calcular (todas necesarias):
        #   - y_std provisto (no None),
        #   - misma longitud que y_true,
        #   - toda σ estrictamente > 0 (el fallback GB devuelve σ=0 → se omite;
        #     una σ=0 aislada haría NLL = +inf y rompería la media).
        # Si no se cumplen, ambas quedan None: preferimos "no medido" a un
        # número que parezca informativo pero no lo sea.
        nll = None
        coverage_95 = None
        sharpness = None
        if y_std is not None:
            y_std_arr = np.asarray(y_std, dtype=float).ravel()
            if y_std_arr.shape == y_true.shape and np.all(np.isfinite(y_std_arr)) and np.all(y_std_arr > 0):
                var = y_std_arr ** 2
                # NLL gaussiana media: 0.5·mean[ log(2π·σ²) + (y-μ)²/σ² ]
                nll = float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - y_pred) ** 2 / var))
                # Coverage empírico del IC 95% (banda ±1.96σ)
                coverage_95 = float(np.mean(np.abs(y_true - y_pred) <= 1.96 * y_std_arr))
                # Sharpness: ancho medio del IC 95% = mean(2·1.96·σ). En las
                # mismas unidades que el target. Complemento de coverage_95:
                # una cobertura alta con sharpness enorme = bandas inútiles.
                sharpness = float(np.mean(2 * 1.96 * y_std_arr))

        # [P2 ROADMAP — baseline naive] R² del predictor constante = media del
        # TRAIN. Solo si se provee la media y el test tiene varianza (si no, R²
        # es indefinido igual que arriba). Da contexto al R² del modelo: cuánto
        # supera al no-modelo. Puede ser negativo (media de train peor que la
        # de test), lo cual es información honesta, no un bug.
        baseline_r2 = None
        if y_train_mean is not None and np.var(y_true) > 0:
            baseline_pred = np.full_like(y_true, float(y_train_mean))
            baseline_r2 = float(r2_score(y_true, baseline_pred))

        self.metrics = ModelMetrics(
            r2=r2,
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            mape=mape,
            nll=nll,
            coverage_95=coverage_95,
            sharpness=sharpness,
            baseline_r2=baseline_r2,
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

        # [SECURITY V1] Sidecar con hash SHA-256 para verificar integridad y
        # procedencia al cargar. NO elimina el riesgo de fondo (pickle permite
        # ejecución de código arbitrario), pero permite detectar un .pkl
        # alterado/no confiable antes de deserializarlo. Ver load() y
        # SECURITY_AUDIT.md (roadmap de migración a skops).
        try:
            import hashlib
            digest = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()
            Path(filepath + ".sha256").write_text(digest + "\n")
        except Exception as e:
            logger.warning("No se pudo escribir el sidecar .sha256: %s", e)

        self.console.print(f"[green]💾 Modelo guardado: {filepath}[/green]")

        return filepath
    
    def load(self, filepath: str, expected_sha256: str = None,
             trust_unsigned: bool = None) -> None:
        """
        Carga modelo y artefactos desde disco.

        [SECURITY V1] ADVERTENCIA: los .pkl usan pickle, que ejecuta código
        arbitrario al deserializar. Cargar un modelo de origen NO CONFIABLE es
        equivalente a ejecutar un binario desconocido. Defensas:
          - Si existe un sidecar `<filepath>.sha256` (o se pasa `expected_sha256`),
            se verifica la integridad ANTES de deserializar; si no coincide,
            se aborta.
          - Si NO hay hash de referencia, se emite un WARNING y solo se procede
            si `trust_unsigned=True` (o la env SOFTSENSOR_TRUST_UNSIGNED_PKL=1).

        Args:
            filepath: Ruta al archivo .pkl
            expected_sha256: Hash esperado (si None, se busca el sidecar .sha256)
            trust_unsigned: Permitir cargar sin hash de referencia (default: False)
        """
        import hashlib
        p = Path(filepath)
        raw = p.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()

        ref = expected_sha256
        sidecar = Path(str(filepath) + ".sha256")
        if ref is None and sidecar.exists():
            ref = sidecar.read_text().strip().split()[0]

        if ref is not None:
            if actual != ref:
                raise ValueError(
                    f"Hash SHA-256 del modelo NO coincide con el esperado.\n"
                    f"  esperado: {ref}\n  actual:   {actual}\n"
                    f"El archivo pudo ser alterado o no es de confianza. Carga abortada."
                )
        else:
            if trust_unsigned is None:
                trust_unsigned = os.getenv("SOFTSENSOR_TRUST_UNSIGNED_PKL", "0") == "1"
            warn = ("⚠️  Cargando un .pkl SIN hash de referencia. pickle ejecuta "
                    "código arbitrario: solo cargá modelos que vos mismo generaste. "
                    "Ver SECURITY_AUDIT.md.")
            self.console.print(f"[bold yellow]   {warn}[/bold yellow]")
            logger.warning(warn)
            if not trust_unsigned:
                raise ValueError(
                    "Carga de .pkl sin verificación de hash bloqueada por seguridad. "
                    "Pasá expected_sha256=..., generá el sidecar .sha256, o forzá con "
                    "trust_unsigned=True / SOFTSENSOR_TRUST_UNSIGNED_PKL=1 si confiás en el origen."
                )

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
        output_dir=None,
        X_test=None,
        y_test_for_importance=None,
        permutation_result: Optional[Dict] = None,
    ) -> List[str]:
        """
        Genera gráficos de diagnóstico del modelo y, si hay suficiente
        información disponible, un informe científico consolidado en PDF.

        Siempre genera (comportamiento previo, sin cambios):
        1. Serie temporal: predicción vs real
        2. Scatter plot: correlación predicho vs real
        3. Histograma de errores
        4. Residuos vs predicción

        [scientific_report] Adicionalmente, si se pasan los argumentos
        opcionales, genera y ensambla en un único PDF:
        5. Gráfico de test de permutación (distribución nula vs. R² real) —
           requiere `permutation_result` con clave 'null_r2_distribution'
           (ver `permutation_test()`).
        6. Gráfico de importancia de features (permutation importance,
           model-agnóstico) — requiere `X_test` y `y_test_for_importance`
           en la MISMA escala con la que se entrenó `self.model`.

        Backward compatible: si no se pasan los argumentos nuevos, el
        comportamiento y el valor de retorno son idénticos a antes (solo el
        PNG del panel de diagnóstico).

        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            y_std: Desviación estándar de predicciones
            dates: Índice temporal
            output_dir: Directorio de salida
            X_test: Features de test (escaladas), para importancia de features
            y_test_for_importance: Target de test, en la escala de self.model
            permutation_result: Dict retornado por `self.permutation_test()`

        Returns:
            Lista de rutas de archivos generados (PNGs + PDF si aplica)
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
        
        serie_label = "Serie Temporal" if self.parse_dates else "Serie (orden de fila, sin tiempo real)"
        axes[0, 0].set_title(f'{serie_label} ({self.model_type})')
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
        # [scientific_report] No cerramos `fig` todavía — si se ensambla el
        # PDF consolidado más abajo, se reutiliza esta figura ya renderizada
        # en vez de releer el PNG desde disco (más rápido, sin doble
        # rasterizado). Se cierra al final de este método en cualquier caso.

        self.console.print(f"[green]📊 Reporte guardado: {path}[/green]")

        generated_paths = [str(path)]
        figures_for_pdf = [fig]

        # [scientific_report] Informe científico consolidado — solo si hay
        # información suficiente (no rompe el uso previo del método si no).
        permutation_png = None
        permutation_fig = None
        if permutation_result and "null_r2_distribution" in permutation_result:
            try:
                permutation_png_path = output_dir / f"{self.model_type.lower()}_permutation_{timestamp}.png"
                permutation_png, permutation_fig = plot_permutation_test(
                    null_r2_distribution=permutation_result["null_r2_distribution"],
                    real_r2=permutation_result["real_r2"],
                    p_value=permutation_result["p_value"],
                    target_col=self.target_col,
                    output_path=permutation_png_path,
                    keep_open=True,
                )
                generated_paths.append(permutation_png)
                figures_for_pdf.append(permutation_fig)
            except Exception as e:
                logger.warning(f"No se pudo generar el gráfico de permutación: {e}")
                permutation_png = None

        importance_png = None
        importance_fig = None
        if X_test is not None and y_test_for_importance is not None and self.model is not None:
            try:
                importance_png_path = output_dir / f"{self.model_type.lower()}_importance_{timestamp}.png"
                importance_png, importance_fig = plot_feature_importance(
                    model=self.model,
                    X_test=X_test,
                    y_test=y_test_for_importance,
                    feature_names=self.feature_names,
                    output_path=importance_png_path,
                    keep_open=True,
                )
                generated_paths.append(importance_png)
                figures_for_pdf.append(importance_fig)
            except Exception as e:
                logger.warning(f"No se pudo generar el gráfico de importancia de features: {e}")
                importance_png = None

        if permutation_png is not None or importance_png is not None:
            try:
                metadata = {
                    "dataset/target": self.target_col,
                    "modelo": self.model_type,
                    "estilo_grafico": _ESTILO_GRAFICO_ACTIVO,
                    "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "semilla_aleatoria": self.random_state,
                    "n_features": len(self.feature_names) if self.feature_names else "N/D",
                    "parse_dates": self.parse_dates,
                    "group_column": self.group_column,
                }
                metrics_rows = [
                    ("R² Score (holdout)", f"{self.metrics.r2:.4f}", "Excelente" if self.metrics.r2 > 0.8 else "Bueno" if self.metrics.r2 > 0.6 else "Moderado/Pobre"),
                ]
                # [P2 ROADMAP — baseline naive] Contexto del R²: el piso.
                if self.metrics.baseline_r2 is not None:
                    beats = (not np.isnan(self.metrics.r2)) and self.metrics.r2 > self.metrics.baseline_r2
                    metrics_rows.append((
                        "R² baseline (media train)", f"{self.metrics.baseline_r2:.4f}",
                        "El modelo lo supera" if beats else "El modelo NO supera al no-modelo",
                    ))
                metrics_rows += [
                    ("RMSE", f"{self.metrics.rmse:.4f}", "Error típico"),
                    ("MAE", f"{self.metrics.mae:.4f}", "Error absoluto promedio"),
                    ("MAPE", f"{self.metrics.mape:.2f}%", "Error porcentual"),
                ]
                # [P6 ROADMAP — calibración] Solo presentes para el GP (σ real).
                if self.metrics.coverage_95 is not None:
                    if self.metrics.coverage_95 < 0.90:
                        cov_interp = "Sobre-confiado (bandas angostas)"
                    elif self.metrics.coverage_95 > 0.99:
                        cov_interp = "Sub-confiado (bandas anchas)"
                    else:
                        cov_interp = "Incertidumbre bien calibrada (~95% nominal)"
                    metrics_rows.append((
                        "Cobertura IC 95%", f"{self.metrics.coverage_95:.3f}", cov_interp,
                    ))
                if self.metrics.sharpness is not None:
                    metrics_rows.append((
                        "Sharpness (ancho IC 95%)", f"{self.metrics.sharpness:.4f}",
                        "Ancho medio de la banda — leer junto a la cobertura",
                    ))
                if self.metrics.nll is not None:
                    metrics_rows.append((
                        "NLL (calibración)", f"{self.metrics.nll:.4f}",
                        "Negative log-likelihood gaussiana — menor = mejor",
                    ))
                if permutation_result:
                    # [scientific_report] R² agregado sobre el dataset completo
                    # (cross_val_predict + GroupKFold), el mismo número que el
                    # R² holdout de arriba pero calculado con un protocolo más
                    # robusto (todas las filas participan de test en algún
                    # fold). Se muestra por separado y con etiqueta explícita
                    # para no confundirlo con el holdout — son dos protocolos
                    # de evaluación válidos, no dos resultados contradictorios.
                    metrics_rows.append((
                        "R² agregado (CV, dataset completo)",
                        f"{permutation_result['real_r2']:.4f}",
                        "Protocolo oficial de reconciliación — ver ROADMAP.md",
                    ))
                    verdict = "señal real" if permutation_result["p_value"] < 0.05 else "no distinguible de azar"
                    metrics_rows.append((
                        "p-value (permutation)",
                        f"{permutation_result['p_value']:.4f}",
                        verdict,
                    ))

                pdf_path = output_dir / f"informe_cientifico_{self.model_type.lower()}_{timestamp}.pdf"
                build_scientific_pdf(
                    output_path=pdf_path,
                    metadata=metadata,
                    metrics_rows=metrics_rows,
                    figures=figures_for_pdf,  # cierra las figuras al terminar
                )
                figures_for_pdf = []  # ya cerradas por build_scientific_pdf
                generated_paths.append(str(pdf_path))
                self.console.print(f"[bold green]📄 Informe científico (PDF): {pdf_path}[/bold green]")
            except Exception as e:
                logger.warning(f"No se pudo ensamblar el informe científico PDF: {e}")

        # Cerrar cualquier figura que haya quedado abierta (ej. si el PDF no
        # se ensambló porque no había permutation_result ni X_test).
        for f in figures_for_pdf:
            plt.close(f)

        return generated_paths
    
    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE COMPLETO
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_from_file(
        self,
        filepath=None,
        test_size=0.2,
        n_trials=None,
        save_model=True,
        run_permutation_test=False,
        output_dir=None,
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
            output_dir: Directorio donde generate_report() escribe PNGs/PDF.
                Si None (default), usa CONFIG.RESULTS_DIR — el comportamiento
                de producción de siempre. Los TESTS deben pasar `tmp_path`
                acá explícitamente para no ensuciar los resultados reales del
                proyecto con PDFs/PNGs de corridas sintéticas (ver
                tests/conftest.py::trained_model y demás fixtures/tests que
                llaman train_from_file()).
            run_permutation_test: [P0b ROADMAP] Si True, corre permutation_test()
                sobre el DATASET COMPLETO (X, y, groups — antes del split
                train/test; 200 refits de un GB fijo, con GroupKFold si hay
                group_column) y adjunta el p-value a metrics.permutation_p_value.
                Usar el dataset completo (no solo el subset de entrenamiento)
                es intencional: reproduce el mismo protocolo agregado
                (cross_val_predict + GroupKFold) con el que se confirmó el
                R²=0.319 oficial en results/verification/geomet_pipeline_reconciliation_v2.json
                — así el R² que reporta este método es comparable al número
                ya documentado, en vez de un R² más ruidoso calculado sobre
                un subset más chico de grupos. GradientBoosting es invariante
                a escala, así que no hace falta re-escalar X. Default False:
                es caro (200 refits) y no siempre necesario — activarlo cuando
                se necesite descartar sobreajuste/leakage explícitamente (ej.
                datasets chicos como GeoMet).

        Returns:
            ModelMetrics con los resultados de evaluación
        """
        # Paso 1: Cargar y preparar datos (X, y SIN escalar)
        X, y, dates = self.load_data(filepath)
        groups = self.groups_  # None salvo que group_column esté configurado

        # ═══════════════════════════════════════════════════════════════════
        # Paso 2: Split train/test
        # ═══════════════════════════════════════════════════════════════════
        # [P0b ROADMAP] Con group_column: split por GRUPO (GroupShuffleSplit)
        # — ningún grupo (ej. HOLEID) queda partido entre train y test. Sin
        # group_column: comportamiento original, split secuencial que respeta
        # orden cronológico.
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
            train_idx, test_idx_arr = next(gss.split(X, y, groups=groups))
            X_train, X_test = X[train_idx], X[test_idx_arr]
            y_train, y_test = y[train_idx], y[test_idx_arr]
            dates_test = dates[test_idx_arr]
            groups_train = groups[train_idx]
            self.console.print(
                f"[dim]   [P0b] Split por grupo ('{self.group_column}'): "
                f"{len(np.unique(groups_train))} grupos en train, "
                f"{len(np.unique(groups[test_idx_arr]))} en test (sin solapamiento)[/dim]"
            )
        else:
            test_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:test_idx], X[test_idx:]
            y_train, y_test = y[:test_idx], y[test_idx:]
            dates_test = dates[test_idx:]
            groups_train = None

        # Paso 3: Escalar — fit SOLO con train, transform en test (anti-leakage).
        X_train_s = self.scaler_X.fit_transform(X_train)
        X_test_s = self.scaler_X.transform(X_test)
        y_train_s = self.scaler_y.fit_transform(y_train)

        # Paso 4: Entrenar (en escala escalada)
        self.optimize_and_train(X_train_s, y_train_s, n_trials=n_trials, groups=groups_train)

        # Paso 5: Evaluar en test set (y_test ya está en escala original)
        y_test_real = y_test.ravel()
        y_pred, y_std = self.predict(X_test_s)
        # [P2 ROADMAP] Media del TRAIN (escala original, y_train no está escalado)
        # para el baseline naive — NO la del test, que daría R²=0 tautológico.
        y_train_mean = float(np.asarray(y_train, dtype=float).mean())
        metrics = self.evaluate(y_test_real, y_pred, y_std=y_std, y_train_mean=y_train_mean)

        # [P0b ROADMAP] Permutation test opcional sobre el DATASET COMPLETO
        # (X, y, groups — no solo X_train) — mismo protocolo agregado
        # (cross_val_predict + GroupKFold) usado en la reconciliación oficial
        # (results/verification/geomet_pipeline_reconciliation_v2.json), para
        # que el R² reportado acá sea el mismo número, no uno distinto
        # calculado sobre un subset más chico y más ruidoso de grupos.
        perm_result = None  # [scientific_report] definido siempre: generate_report() lo acepta como None sin problema
        if run_permutation_test:
            self.console.print(
                "\n[bold yellow]🎲 Permutation test (200 refits sobre el dataset completo, puede tardar)...[/bold yellow]"
            )
            perm_result = self.permutation_test(X, y.ravel(), groups=groups)
            metrics.permutation_p_value = perm_result["p_value"]
            verdict = "✅ señal real" if perm_result["p_value"] < 0.05 else "🔴 no distinguible de azar"
            self.console.print(
                f"[dim]   p-value={perm_result['p_value']:.4f} "
                f"(R² agregado, dataset completo={perm_result['real_r2']:.4f}) → {verdict}[/dim]"
            )

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
        table.add_row("R² Score (holdout)", f"[{r2_color}]{metrics.r2:.4f}[/{r2_color}]", r2_interp)
        # [P2 ROADMAP — baseline naive] Piso contra el que se lee el R² del modelo.
        if metrics.baseline_r2 is not None:
            beats = (not np.isnan(metrics.r2)) and metrics.r2 > metrics.baseline_r2
            base_color = "green" if beats else "red"
            base_interp = "✅ el modelo lo supera" if beats else "🔴 el modelo NO supera al no-modelo"
            table.add_row(
                "R² baseline (media train)",
                f"[{base_color}]{metrics.baseline_r2:.4f}[/{base_color}]",
                base_interp,
            )
        table.add_row("RMSE", f"{metrics.rmse:.4f}", "Error típico")
        table.add_row("MAE", f"{metrics.mae:.4f}", "Error absoluto promedio")
        table.add_row("MAPE", f"{metrics.mape:.2f}%", "Error porcentual")
        # [P6 ROADMAP — calibración] Solo se llenan para el GP (σ real).
        if metrics.coverage_95 is not None:
            # Verde si la cobertura está cerca del 95% nominal (±5 pts);
            # amarillo si se aleja (sobre/sub-confianza).
            cov_ok = 0.90 <= metrics.coverage_95 <= 1.0
            cov_color = "green" if cov_ok else "yellow"
            if metrics.coverage_95 < 0.90:
                cov_interp = "Sobre-confiado (bandas angostas)"
            elif metrics.coverage_95 > 0.99:
                cov_interp = "Sub-confiado (bandas anchas)"
            else:
                cov_interp = "Incertidumbre bien calibrada"
            table.add_row(
                "Cobertura IC 95%",
                f"[{cov_color}]{metrics.coverage_95:.3f}[/{cov_color}]",
                cov_interp,
            )
        if metrics.sharpness is not None:
            table.add_row(
                "Sharpness (ancho IC 95%)",
                f"{metrics.sharpness:.4f}",
                "Ancho medio de la banda — leer junto a la cobertura",
            )
        if metrics.nll is not None:
            table.add_row("NLL (calibración)", f"{metrics.nll:.4f}", "Menor = mejor (media+incertidumbre)")
        if perm_result is not None:
            # [scientific_report] mismo R² que aparece en el PDF — dataset
            # completo, protocolo de reconciliación oficial. Se muestra junto
            # al holdout, no en su reemplazo, para que quede explícito que
            # son dos protocolos distintos y no un número contradictorio.
            table.add_row(
                "R² agregado (CV, dataset completo)",
                f"{perm_result['real_r2']:.4f}",
                "Protocolo oficial de reconciliación",
            )
        if metrics.permutation_p_value is not None:
            p_color = "green" if metrics.permutation_p_value < 0.05 else "red"
            table.add_row(
                "p-value (permutation)",
                f"[{p_color}]{metrics.permutation_p_value:.4f}[/{p_color}]",
                "Señal real" if metrics.permutation_p_value < 0.05 else "No distinguible de azar"
            )

        self.console.print(table)

        # Paso 5: Generar reporte visual (+ informe científico si hay datos suficientes)
        # [scientific_report] y_test_s: mismo espacio de escala en que se entrenó
        # self.model (fit sobre y_train_s) — necesario para que permutation_importance
        # calcule un R² comparable, aunque el R² como métrica es invariante a escala.
        y_test_s = self.scaler_y.transform(y_test).ravel()
        self.generate_report(
            y_test_real, y_pred, y_std, dates_test,
            output_dir=output_dir,
            X_test=X_test_s,
            y_test_for_importance=y_test_s,
            permutation_result=perm_result,
        )

        # Paso 6: Guardar modelo
        if save_model:
            self.save()

        return metrics

    def permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray = None,
        n_permutations: int = 200,
        n_splits: int = None
    ) -> Dict:
        """
        [P0b ROADMAP] ¿El R² observado es señal real o azar/leakage?

        Generalización reutilizable de run_geomet_rigor.py::perm_test: baraja
        el target `n_permutations` veces, mide R² vía cross_val_predict en
        cada barajado (GroupKFold si hay `groups`, si no KFold), y compara
        contra el R² real. p-value = fracción de barajados que igualan o
        superan el R² real (Monte Carlo, +1/+1 para evitar p=0 espurio).

        Usa un GradientBoostingRegressor de hiperparámetros FIJOS (no el
        modelo final, no Optuna) — misma razón que en el script original:
        200 refits con optimización bayesiana serían intratables, y el
        objetivo es diagnóstico de leakage/sobreajuste, no reproducir
        exactamente el modelo de producción.

        Args:
            X: Features (escalados o no — el resultado es relativo).
            y: Target, alineado fila a fila con X.
            groups: Si se pasa, usa GroupKFold (mismo criterio que el split
                honesto de train_from_file). Si None, KFold(shuffle=True).
            n_permutations: Número de barajados Monte Carlo (200 default).
            n_splits: Nº de folds. Si None: min(5, n_grupos) con grupos,
                min(5, n_muestras) sin ellos. Se clampea a >=2 y <=n_grupos.

        Returns:
            Dict con real_r2, p_value, n_permutations, n_splits.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        m = GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=self.random_state
        )

        if groups is not None:
            groups = np.asarray(groups)
            n_groups = len(np.unique(groups))
            k = max(2, min(n_splits or 5, n_groups))
            cv = GroupKFold(n_splits=k)
            cv_kwargs = {"groups": groups}
        else:
            k = max(2, min(n_splits or 5, len(X)))
            cv = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
            cv_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_pred = cross_val_predict(m, X, y, cv=cv, **cv_kwargs)
        real_r2 = r2_score(y, real_pred)

        rng = np.random.default_rng(self.random_state)
        ge = 0
        null_r2_distribution = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_permutations):
                y_perm = rng.permutation(y)
                perm_pred = cross_val_predict(m, X, y_perm, cv=cv, **cv_kwargs)
                perm_r2 = r2_score(y_perm, perm_pred)
                null_r2_distribution.append(float(perm_r2))
                if perm_r2 >= real_r2:
                    ge += 1
        p_value = (ge + 1) / (n_permutations + 1)

        return {
            "real_r2": float(real_r2),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "n_splits": k,
            # [scientific_report] distribución nula completa — permite graficar
            # el histograma de R² bajo H0 vs. el R² real, no solo reportarlo
            # como número. Backward-compatible: clave nueva, no rompe código
            # existente que solo lea real_r2/p_value/n_permutations/n_splits.
            "null_r2_distribution": null_r2_distribution,
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS PÚBLICOS
# ═══════════════════════════════════════════════════════════════════════════
__all__ = ["SoftSensorGP", "ModelMetrics", "TrainingArtifacts"]


# ═══════════════════════════════════════════════════════════════════════════
# CLI (Command Line Interface)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Punto de entrada para uso desde línea de comandos.
    
    Ejemplos:
        python gp_model.py --data data/clean.csv --target _silica_concentrate
        python gp_model.py --trials 30 --subsample 20
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Soft-Sensor GP v4.1.0 (Universal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python gp_model.py
  python gp_model.py --data data/gold.csv --target recovery
  python gp_model.py --trials 30 --subsample 20 --no-fallback
  python gp_model.py --input-lags --input-lag-periods 1,2,4,8   # [P0] lags de entradas
  python gp_model.py --no-lags --no-diffs                        # régimen sensor-only (RUL)
  python gp_model.py --data data/geomet/flotation.csv --target LCT \\
      --group-column HOLEID --no-parse-dates --no-lags --no-diffs \\
      --permutation-test                                          # [P0b] split honesto GeoMet
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
                       help="Desactivar features de lag DEL TARGET")
    parser.add_argument("--no-diffs", action="store_true",
                       help="Desactivar diferencias/rolling DEL TARGET (junto con "
                            "--no-lags, necesario para régimen sensor-only honesto "
                            "en prognostics — ver ROADMAP.md, nota de integridad)")
    parser.add_argument("--input-lags", action="store_true",
                       help="[P0 ROADMAP] Activar lags de variables de ENTRADA "
                            "(no del target) — para procesos con retardo/tiempo "
                            "de residencia")
    parser.add_argument("--input-lag-periods", type=str, default=None,
                       help="Periodos de lag de entrada, coma-separados "
                            "(default: 1,2,3). Ej: --input-lag-periods 1,2,4,8")
    parser.add_argument("--input-lag-columns", type=str, default=None,
                       help="Columnas de entrada a laguear, coma-separadas "
                            "(default: todas las columnas numéricas de entrada)")
    parser.add_argument("--no-fallback", action="store_true",
                       help="No usar GradientBoosting como alternativa")
    parser.add_argument("--no-save", action="store_true",
                       help="No guardar el modelo")
    parser.add_argument("--group-column", type=str, default=None,
                       help="[P0b ROADMAP] Columna de agrupamiento (ej. HOLEID) para "
                            "split/CV honesto vía GroupShuffleSplit/GroupKFold — "
                            "usar cuando varias filas comparten origen (sondaje, lote)")
    parser.add_argument("--no-parse-dates", action="store_true",
                       help="[P0b ROADMAP] El índice del CSV NO es una fecha real "
                            "(datasets geometalúrgicos/spatial) — desactiva diagnóstico "
                            "de autocorrelación y subsampleo temporal")
    parser.add_argument("--permutation-test", action="store_true",
                       help="[P0b ROADMAP] Correr permutation test (200 refits) tras "
                            "entrenar y reportar el p-value junto a las métricas")

    args = parser.parse_args()

    try:
        input_lag_periods = (
            [int(p.strip()) for p in args.input_lag_periods.split(",")]
            if args.input_lag_periods else None
        )
        input_lag_columns = (
            [c.strip() for c in args.input_lag_columns.split(",")]
            if args.input_lag_columns else None
        )

        model = SoftSensorGP(
            target_col=args.target,
            subsample_step=args.subsample,
            add_lag_features=not args.no_lags,
            add_diff_features=not args.no_diffs,
            add_input_lags=args.input_lags,
            input_lag_periods=input_lag_periods,
            input_lag_columns=input_lag_columns,
            parse_dates=not args.no_parse_dates,
            group_column=args.group_column,
            use_fallback_model=not args.no_fallback
        )

        metrics = model.train_from_file(
            filepath=args.data,
            test_size=args.test_size,
            n_trials=args.trials,
            save_model=not args.no_save,
            run_permutation_test=args.permutation_test
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
