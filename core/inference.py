"""
Módulo: core/inference.py (CORREGIDO)
Descripción:
    Motor de Inferencia Inteligente.
    CORRECCIÓN: Reconstruye la historia (Lags/Diffs) antes de predecir
    para evitar el problema de "Usando 0.0".
"""

import joblib
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Config
from config.settings import CONFIG

console = Console()

def load_latest_model():
    """Carga el último modelo .pkl disponible."""
    models = list(CONFIG.MODELS_DIR.glob("*.pkl"))
    if not models:
        console.print("[red]❌ No hay modelos en /models[/red]")
        return None
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    console.print(f"[bold cyan]🤖 Cerebro cargado:[/bold cyan] {latest_model.name}")
    return joblib.load(latest_model)

def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Recrea EXACTAMENTE las mismas características que usó el entrenamiento.
    """
    df = df.copy()
    y = df[target_col]
    
    # Estos deben coincidir con los usados en mining_gp_pro.py
    # Lags
    for lag in [1, 5, 10, 20]:
        df[f'{target_col}_lag_{lag}'] = y.shift(lag)
    
    # Diferencias
    df[f'{target_col}_diff_1'] = y.diff(1)
    df[f'{target_col}_diff_5'] = y.diff(5)
    
    # Rolling stats
    df[f'{target_col}_rolling_mean_10'] = y.rolling(10).mean()
    df[f'{target_col}_rolling_std_10'] = y.rolling(10).std()
    
    return df

def run_inference(csv_path: str = None, n_samples: int = 5):
    console.print(Panel.fit("🏭 Soft-Sensor: Modo Inferencia (Con Memoria)", style="bold blue"))

    # 1. Cargar Artefactos
    artifact = load_latest_model()
    if artifact is None: return

    model = artifact.model
    scaler_X = artifact.scaler_X
    scaler_y = artifact.scaler_y
    # Obtenemos las columnas que el modelo REALMENTE espera ver
    expected_features = artifact.feature_names 
    target_col = artifact.target_column

    # 2. Cargar Datos y Reconstruir Historia
    path = csv_path or CONFIG.DATA_CLEAN_PATH
    # Leemos todo (o un chunk grande) para poder calcular la historia
    df_full = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # --- AQUÍ ESTÁ EL FIX ---
    # Subsampleamos igual que en el entrenamiento para que los lags tengan sentido temporal
    # (Recordar: entrenamos con subsample=150)
    # Intentamos detectar el subsample aproximado o usamos uno seguro
    df_processed = df_full.iloc[::50] # Usamos 50 para asegurar continuidad
    
    # Regeneramos la ingeniería de features
    df_features = engineer_features(df_processed, target_col)
    
    # Eliminamos las primeras filas que quedaron con NaN por el shift
    df_valid = df_features.dropna()
    
    # 3. Muestreo Aleatorio (Simulación de "Ahora")
    # Ahora sí tomamos la muestra, porque ya tiene los lags calculados
    sample = df_valid.sample(n_samples)
    y_real = sample[target_col]

    # 4. Preparar Matriz X
    # Solo seleccionamos las columnas que el modelo pidió durante el entrenamiento
    X_input = pd.DataFrame(index=sample.index)
    
    for col in expected_features:
        if col in sample.columns:
            X_input[col] = sample[col]
        else:
            # Si aún falta algo (raro), avisamos
            console.print(f"[yellow]⚠️ Falta feature: {col} (poniendo 0.0)[/yellow]")
            X_input[col] = 0.0

    # Escalar
    X_scaled = scaler_X.transform(X_input)

    # 5. Predecir
    model_type = getattr(artifact, 'model_type', 'GP')
    
    if model_type == "GP":
        y_pred_scaled, y_std_scaled = model.predict(X_scaled, return_std=True)
        y_std = y_std_scaled * scaler_y.scale_[0] if hasattr(scaler_y, 'scale_') else y_std_scaled
    else:
        y_pred_scaled = model.predict(X_scaled)
        y_std = np.zeros_like(y_pred_scaled)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 6. Tabla de Resultados
    table = Table(title=f"💎 Predicción en Tiempo Real ({model_type})")
    table.add_column("Hora", style="dim")
    table.add_column("Real", style="green")
    table.add_column("Predicción", style="bold cyan")
    table.add_column("Diferencia", style="white")
    table.add_column("Confianza", justify="center")

    for i in range(len(sample)):
        diff = abs(y_real.iloc[i] - y_pred[i])
        
        # Semáforo simple
        if diff < 0.1: status = "✅ Excelente"
        elif diff < 0.3: status = "⚠️ Aceptable"
        else: status = "❌ Desviado"

        table.add_row(
            str(sample.index[i].time()),
            f"{y_real.iloc[i]:.2f}%",
            f"{y_pred[i]:.2f}%",
            f"{diff:.4f}",
            status
        )

    console.print(table)

if __name__ == "__main__":
    run_inference()