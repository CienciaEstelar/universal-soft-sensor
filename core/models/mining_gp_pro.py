"""
Módulo: core/models/mining_gp_pro.py (VERSIÓN 4 - DEFINITIVA)
Descripción: 
    Soft-Sensor para procesos mineros con todas las correcciones:
    
FIXES INCLUIDOS:
    1. Elimina features constantes automáticamente
    2. Subsample adaptativo (default=50)
    3. Feature engineering (lags + diferencias)
    4. Fallback a GradientBoosting si GP falla
    5. Elimina features con alta multicolinealidad
    6. Noise level mínimo ajustado para evitar warnings
"""

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

# Sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel, RBF
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Rich UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Configuración
from config.settings import CONFIG

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Estilo de gráficos - usar uno que siempre existe
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')


@dataclass
class ModelMetrics:
    """Métricas de evaluación."""
    r2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    
    def to_dict(self) -> dict:
        return {"r2": self.r2, "rmse": self.rmse, "mae": self.mae, "mape": self.mape}
    
    def __repr__(self) -> str:
        return f"R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"


@dataclass
class TrainingArtifacts:
    """Artefactos de modelo entrenado."""
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


class MiningGP:
    """
    Soft-Sensor v4 - Versión definitiva con todas las correcciones.
    """
    
    def __init__(
        self, 
        target_col: str = None, 
        random_state: int = 42,
        subsample_step: int = 50,
        add_lag_features: bool = True,
        lag_periods: List[int] = None,
        add_diff_features: bool = True,
        use_fallback_model: bool = True,
        remove_constant_features: bool = True,  # NUEVO
        remove_correlated_features: bool = True,  # NUEVO
        correlation_threshold: float = 0.98  # NUEVO
    ):
        self.console = Console()
        self.target_col = target_col or CONFIG.GP_TARGET_COLUMN
        self.random_state = random_state
        self.subsample_step = subsample_step
        self.add_lag_features = add_lag_features
        self.lag_periods = lag_periods or [1, 5, 10, 20]
        self.add_diff_features = add_diff_features
        self.use_fallback_model = use_fallback_model
        self.remove_constant_features = remove_constant_features
        self.remove_correlated_features = remove_correlated_features
        self.correlation_threshold = correlation_threshold
        
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        
        self.model = None
        self.model_type = "GP"
        self.feature_names: List[str] = []
        self.removed_features: List[str] = []
        self.best_params: Dict = {}
        self.metrics: Optional[ModelMetrics] = None
        self.data_diagnosis: Dict = {}
        
        logger.info(f"MiningGP v4 inicializado - Target: {self.target_col}")
    
    def _diagnose_data(self, y_series: pd.Series) -> Dict:
        """Diagnóstico automático."""
        diagnosis = {
            "n_samples": len(y_series),
            "mean": y_series.mean(),
            "std": y_series.std(),
            "cv": (y_series.std() / y_series.mean()) * 100 if y_series.mean() != 0 else 0,
            "autocorr_1": y_series.autocorr(lag=1),
            "autocorr_10": y_series.autocorr(lag=10) if len(y_series) > 10 else 0,
            "autocorr_50": y_series.autocorr(lag=50) if len(y_series) > 50 else 0,
        }
        
        # Encontrar subsample recomendado
        for lag in [10, 20, 30, 40, 50, 75, 100, 150, 200]:
            if len(y_series) > lag:
                ac = y_series.autocorr(lag=lag)
                if ac < 0.85:  # Más estricto
                    diagnosis["recommended_subsample"] = lag
                    break
        else:
            diagnosis["recommended_subsample"] = 200
        
        # Severidad
        if diagnosis["autocorr_1"] > 0.98:
            diagnosis["severity"] = "CRÍTICA"
        elif diagnosis["autocorr_1"] > 0.95:
            diagnosis["severity"] = "ALTA"
        elif diagnosis["autocorr_1"] > 0.90:
            diagnosis["severity"] = "MODERADA"
        else:
            diagnosis["severity"] = "OK"
        
        return diagnosis
    
    def _remove_problematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina features constantes y altamente correlacionados."""
        df = df.copy()
        removed = []
        
        # 1. Eliminar constantes (std ≈ 0)
        if self.remove_constant_features:
            for col in df.columns:
                if df[col].std() < 1e-8:
                    removed.append(f"{col} (constante)")
                    df = df.drop(columns=[col])
        
        # 2. Eliminar altamente correlacionados
        if self.remove_correlated_features and len(df.columns) > 1:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = []
            for col in upper.columns:
                # Encontrar columnas con correlación > threshold
                high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
                if high_corr:
                    # Mantener la primera, eliminar las demás
                    to_drop.extend(high_corr)
            
            to_drop = list(set(to_drop))
            for col in to_drop:
                if col in df.columns:
                    removed.append(f"{col} (correlación > {self.correlation_threshold})")
                    df = df.drop(columns=[col])
        
        if removed:
            self.console.print(f"[yellow]   ⚠️  Features eliminados:[/yellow]")
            for r in removed[:5]:  # Mostrar máximo 5
                self.console.print(f"[dim]      - {r}[/dim]")
            if len(removed) > 5:
                self.console.print(f"[dim]      ... y {len(removed)-5} más[/dim]")
        
        self.removed_features = removed
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        """Crea features de lag y diferencias."""
        df = df.copy()
        y = df[y_col]
        
        if self.add_lag_features:
            for lag in self.lag_periods:
                df[f'{y_col}_lag_{lag}'] = y.shift(lag)
        
        if self.add_diff_features:
            df[f'{y_col}_diff_1'] = y.diff(1)
            df[f'{y_col}_diff_5'] = y.diff(5)
            df[f'{y_col}_rolling_mean_10'] = y.rolling(10, min_periods=1).mean()
            df[f'{y_col}_rolling_std_10'] = y.rolling(10, min_periods=1).std()
        
        return df
    
    def load_data(
        self, 
        filepath: str = None,
        max_rows: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Carga datos con todas las correcciones."""
        filepath = filepath or str(CONFIG.DATA_CLEAN_PATH)
        
        self.console.print(f"[bold cyan]📥 Cargando datos...[/bold cyan]")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No encontrado: {filepath}")
        
        # Contar filas
        with open(filepath, 'r') as f:
            total_rows = sum(1 for _ in f) - 1
        
        skip_rows = max(0, total_rows - max_rows)
        self.console.print(f"[dim]   Archivo: {total_rows:,} filas totales[/dim]")
        
        # Cargar
        df = pd.read_csv(
            filepath, 
            index_col=0,
            parse_dates=True,
            skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
        )
        
        if self.target_col not in df.columns:
            raise ValueError(f"Target '{self.target_col}' no encontrado")
        
        # === DIAGNÓSTICO ===
        self.console.print(f"\n[bold yellow]🔬 Diagnóstico:[/bold yellow]")
        self.data_diagnosis = self._diagnose_data(df[self.target_col])
        
        diag_table = Table(show_header=False, box=None, padding=(0, 2))
        diag_table.add_row("Autocorr lag-1:", f"{self.data_diagnosis['autocorr_1']:.4f}")
        diag_table.add_row("Autocorr lag-50:", f"{self.data_diagnosis['autocorr_50']:.4f}")
        
        sev = self.data_diagnosis['severity']
        sev_color = "red" if sev == "CRÍTICA" else "yellow" if sev in ["ALTA", "MODERADA"] else "green"
        diag_table.add_row("Severidad:", f"[{sev_color}]{sev}[/{sev_color}]")
        diag_table.add_row("Subsample recomendado:", f"cada {self.data_diagnosis['recommended_subsample']}")
        self.console.print(diag_table)
        
        # Auto-ajustar subsample
        if self.data_diagnosis["autocorr_1"] > 0.98:
            recommended = self.data_diagnosis["recommended_subsample"]
            if self.subsample_step < recommended:
                self.console.print(f"[yellow]   ⚠️  Auto-ajustando subsample: {self.subsample_step} → {recommended}[/yellow]")
                self.subsample_step = recommended
        
        # === SUBSAMPLE ===
        if self.subsample_step > 1:
            df = df.iloc[::self.subsample_step]
            self.console.print(f"[dim]   Subsampleado 1/{self.subsample_step}: {len(df):,} filas[/dim]")
        
        # === FEATURE ENGINEERING ===
        df = self._create_lag_features(df, self.target_col)
        df = df.dropna()
        self.console.print(f"[dim]   Con feature eng: {len(df):,} filas, {len(df.columns)} cols[/dim]")
        
        # === SEPARAR X e y ===
        y_series = df[self.target_col]
        drop_cols = [self.target_col, "_iron_concentrate"]
        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # === ELIMINAR FEATURES PROBLEMÁTICOS ===
        X_df = self._remove_problematic_features(X_df)
        
        self.feature_names = X_df.columns.tolist()
        X = X_df.values
        y = y_series.values.reshape(-1, 1)
        
        # Escalar
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Stats finales
        new_autocorr = y_series.autocorr(lag=1) if len(y_series) > 1 else 0
        
        self.console.print(f"\n[green]✅ Datos listos: {X.shape[0]:,} filas, {X.shape[1]} features[/green]")
        self.console.print(f"[dim]   Nueva autocorr lag-1: {new_autocorr:.4f}[/dim]")
        
        if new_autocorr > 0.9:
            self.console.print(f"[yellow]   ⚠️  Autocorrelación aún alta ({new_autocorr:.2f})[/yellow]")
        
        return X_scaled, y_scaled, df.index
    
    def _train_gp(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> Tuple[any, Dict, float]:
        """Entrena GP con Optuna - hiperparámetros optimizados."""
        
        def objective(trial):
            # Hiperparámetros con rangos ajustados
            alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)  # Más alto
            length_scale = trial.suggest_float("length_scale", 1.0, 25.0, log=True)
            nu = trial.suggest_categorical("nu", [1.5, 2.5])  # Quitamos 0.5 (muy ruidoso)
            noise = trial.suggest_float("noise_level", 0.001, 0.1, log=True)  # Mínimo mucho más alto
            
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                Matern(length_scale=length_scale, nu=nu, length_scale_bounds=(0.01, 100)) +
                WhiteKernel(noise_level=noise, noise_level_bounds=(0.01, 10))
            )
            
            model = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=alpha,
                random_state=self.random_state,
                n_restarts_optimizer=2
            )
            
            # CV con menos datos para velocidad
            max_samples = min(600, len(X_train))
            step = max(1, len(X_train) // max_samples)
            X_opt = X_train[::step][:max_samples]
            y_opt = y_train[::step][:max_samples]
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_opt):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_opt[train_idx], y_opt[train_idx].ravel())
                    pred = model.predict(X_opt[test_idx])
                    score = r2_score(y_opt[test_idx], pred)
                    scores.append(max(score, -1.0))  # Limitar score mínimo
                except:
                    return -1.0
            
            return np.mean(scores)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_score = study.best_value
        
        # Modelo final
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=best_params["length_scale"], 
                   nu=best_params["nu"],
                   length_scale_bounds=(0.01, 100)) +
            WhiteKernel(noise_level=best_params["noise_level"],
                       noise_level_bounds=(0.01, 10))
        )
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=best_params["alpha"],
            n_restarts_optimizer=3,
            random_state=self.random_state
        )
        
        return model, best_params, best_score
    
    def _train_fallback(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[any, Dict]:
        """Modelo alternativo: GradientBoosting."""
        self.console.print("[yellow]🔄 Usando GradientBoosting como alternativa...[/yellow]")
        
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        params = {"model": "GradientBoosting", "n_estimators": 150, "max_depth": 4}
        return model, params
    
    def optimize_and_train(self, X: np.ndarray, y: np.ndarray, n_trials: int = None) -> None:
        """Optimiza y entrena."""
        n_trials = n_trials or CONFIG.GP_OPTUNA_TRIALS
        max_samples = CONFIG.GP_MAX_TRAIN_SAMPLES
        
        self.console.print(f"\n[bold yellow]⚡ Optimizando GP ({n_trials} trials)...[/bold yellow]")
        
        # Intentar GP
        model, params, cv_score = self._train_gp(X, y, n_trials)
        
        self.console.print(f"\n[bold]CV Score: R² = {cv_score:.4f}[/bold]")
        
        # Decidir modelo
        if cv_score < 0.60 and self.use_fallback_model:
            self.console.print(f"[red]❌ GP falló. Cambiando a modelo alternativo.[/red]")
            model, params = self._train_fallback(X, y)
            self.model_type = "GradientBoosting"
        else:
            self.model_type = "GP"
        
        self.best_params = params
        
        # Mostrar params
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Parámetro")
        table.add_column("Valor")
        for k, v in params.items():
            val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
            table.add_row(k, val_str)
        self.console.print(table)
        
        # Entrenar final
        if len(X) > max_samples:
            step = max(1, len(X) // max_samples)
            indices = np.arange(0, len(X), step)[:max_samples]
            X_train, y_train = X[indices], y[indices]
            self.console.print(f"[dim]Entrenando con {len(X_train):,} de {len(X):,} muestras[/dim]")
        else:
            X_train, y_train = X, y
        
        self.console.print(f"[bold blue]🚀 Entrenando {self.model_type}...[/bold blue]")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train.ravel())
        
        self.model = model
        self.console.print(f"[green]✅ {self.model_type} entrenado[/green]")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicciones."""
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        if self.model_type == "GP":
            y_pred_scaled, y_std_scaled = self.model.predict(X, return_std=True)
        else:
            y_pred_scaled = self.model.predict(X)
            y_std_scaled = np.zeros_like(y_pred_scaled)
        
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if hasattr(self.scaler_y, 'scale_') and self.scaler_y.scale_ is not None:
            y_std = y_std_scaled * self.scaler_y.scale_[0]
        else:
            y_std = y_std_scaled
        
        return y_pred, y_std
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Métricas."""
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        self.metrics = ModelMetrics(
            r2=r2_score(y_true, y_pred),
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            mape=mape
        )
        return self.metrics
    
    def save(self, filepath: str = None) -> str:
        """Guardar."""
        if self.model is None:
            raise ValueError("No hay modelo")
        
        if filepath is None:
            CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(CONFIG.MODELS_DIR / f"{self.model_type.lower()}_{self.target_col}_{timestamp}.pkl")
        
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
        self.console.print(f"[green]💾 Guardado: {filepath}[/green]")
        return filepath
    
    def load(self, filepath: str) -> None:
        """Cargar."""
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
        self.console.print(f"[green]📂 Cargado: {filepath} ({self.model_type})[/green]")
    
    def generate_report(self, y_true, y_pred, y_std, dates, output_dir=None) -> List[str]:
        """Gráficos."""
        output_dir = Path(output_dir or CONFIG.RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Series temporales
        n = min(200, len(y_true))
        axes[0, 0].plot(dates[-n:], y_true[-n:], 'k-', lw=1, alpha=0.8, label='Real')
        axes[0, 0].plot(dates[-n:], y_pred[-n:], 'r--', lw=1.5, label='Predicción')
        if np.any(y_std > 0):
            axes[0, 0].fill_between(dates[-n:], y_pred[-n:]-1.96*y_std[-n:],
                                   y_pred[-n:]+1.96*y_std[-n:], color='red', alpha=0.15)
        axes[0, 0].set_title(f'Serie Temporal ({self.model_type})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter
        axes[0, 1].scatter(y_true, y_pred, alpha=0.4, s=10, c='steelblue')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0, 1].plot(lims, lims, 'k--', lw=2)
        axes[0, 1].set_title(f'R² = {self.metrics.r2:.4f}')
        axes[0, 1].set_xlabel('Real')
        axes[0, 1].set_ylabel('Predicho')
        
        # Errores
        errors = y_true - y_pred
        axes[1, 0].hist(errors, bins=50, color='steelblue', edgecolor='white')
        axes[1, 0].axvline(0, color='red', ls='--')
        axes[1, 0].set_title('Distribución de Errores')
        
        # Residuos
        axes[1, 1].scatter(y_pred, errors, alpha=0.3, s=10, c='steelblue')
        axes[1, 1].axhline(0, color='red', ls='--')
        axes[1, 1].set_title('Residuos vs Predicción')
        
        plt.suptitle(f'{self.model_type} | {self.target_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = output_dir / f"{self.model_type.lower()}_report_{timestamp}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        
        self.console.print(f"[green]📊 Reporte: {path}[/green]")
        return [str(path)]
    
    def train_from_file(self, filepath=None, test_size=0.2, n_trials=None, save_model=True) -> ModelMetrics:
        """Pipeline completo."""
        X, y, dates = self.load_data(filepath)
        
        test_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:test_idx], X[test_idx:]
        y_train, y_test = y[:test_idx], y[test_idx:]
        dates_test = dates[test_idx:]
        
        self.optimize_and_train(X_train, y_train, n_trials=n_trials)
        
        y_test_real = self.scaler_y.inverse_transform(y_test).ravel()
        y_pred, y_std = self.predict(X_test)
        metrics = self.evaluate(y_test_real, y_pred)
        
        # Resultados
        self.console.print("\n" + "=" * 50)
        self.console.print(f"[bold]🏆 RESULTADOS ({self.model_type})[/bold]")
        self.console.print("=" * 50)
        
        table = Table(header_style="bold green")
        table.add_column("Métrica")
        table.add_column("Valor")
        
        color = "green" if metrics.r2 > 0.5 else "yellow" if metrics.r2 > 0 else "red"
        table.add_row("R² Score", f"[{color}]{metrics.r2:.4f}[/{color}]")
        table.add_row("RMSE", f"{metrics.rmse:.4f}")
        table.add_row("MAE", f"{metrics.mae:.4f}")
        table.add_row("MAPE", f"{metrics.mape:.2f}%")
        self.console.print(table)
        
        self.generate_report(y_test_real, y_pred, y_std, dates_test)
        
        if save_model:
            self.save()
        
        return metrics


__all__ = ["MiningGP", "ModelMetrics", "TrainingArtifacts"]


def main():
    """CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Soft-Sensor GP v4 (Definitivo)")
    parser.add_argument("--data", "-d", type=str, default=None)
    parser.add_argument("--target", "-t", type=str, default=None)
    parser.add_argument("--trials", "-n", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--subsample", "-s", type=int, default=50)
    parser.add_argument("--no-lags", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    
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
        
        exit(0 if metrics.r2 > 0 else 1)
        
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()