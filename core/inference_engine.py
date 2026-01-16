"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/inference_engine.py
Autor: Juan Galaz (Arquitectura Minera 4.0)
Versión: 1.1 (Documentación Extendida)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Motor de Inferencia dedicado para Producción.
    
    Este módulo actúa como una 'Fachada' (Facade Pattern) que abstrae la 
    complejidad de cargar modelos, generar features en tiempo real y 
    desescalar predicciones.

    RESPONSABILIDADES:
    1. Gestión de Artefactos: Carga automática del modelo más reciente (.pkl).
    2. Feature Engineering On-the-Fly: Recrea los lags y diffs necesarios
       para que el modelo entienda el contexto temporal.
    3. Seguridad: Valida que los datos de entrada coincidan con los del entrenamiento.

USO:
    engine = MiningInference()
    resultado = engine.predict_scenario(df_ultimas_50_horas)
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Importaciones internas
from config.settings import CONFIG
from core.models.mining_gp_pro import MiningGP, TrainingArtifacts

# Configuración de logger local
logger = logging.getLogger("InferenceEngine")

class MiningInference:
    """
    Clase controladora para la ejecución de modelos mineros en producción.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el motor de inferencia.

        Lógica de Carga:
        - Si se provee 'model_path', carga ese archivo específico.
        - Si NO se provee, busca automáticamente el .pkl más nuevo en 'models/'.

        Args:
            model_path (str, optional): Ruta absoluta al archivo .pkl. 
                                        Por defecto es None.
        """
        # Instancia vacía de MiningGP para acceder a sus métodos de utilidad
        # (como _create_lag_features) sin necesidad de re-implementarlos.
        self.model_wrapper = MiningGP() 
        self.loaded = False
        self.model_path = None
        
        # Estrategia de carga automática
        if model_path:
            self.load_model(Path(model_path))
        else:
            self._load_latest_model()

    def _load_latest_model(self) -> None:
        """
        Escanea el directorio de modelos y carga el archivo modificado más recientemente.
        
        Raises:
            FileNotFoundError: Si la carpeta 'models/' está vacía.
        """
        try:
            # Glob pattern para encontrar todos los pickles
            models = list(CONFIG.MODELS_DIR.glob("*.pkl"))
            
            if not models:
                msg = f"❌ No se encontraron modelos entrenados en: {CONFIG.MODELS_DIR}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            
            # Ordenar por fecha de modificación (st_mtime) descendente
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            logger.info(f"🔎 Modelo más reciente detectado: {latest_model.name}")
            
            self.load_model(latest_model)
            
        except Exception as e:
            logger.critical(f"💥 Error crítico buscando modelo: {e}")
            raise

    def load_model(self, path: Path) -> None:
        """
        Deserializa y carga los artefactos del modelo en memoria.

        Args:
            path (Path): Objeto Path apuntando al archivo .pkl.

        Raises:
            Exception: Si el archivo está corrupto o es de una versión incompatible.
        """
        logger.info(f"📂 Cargando artefactos desde disco...")
        try:
            # Usamos el método nativo .load() de la clase MiningGP
            # Esto restaura: modelo, scalers, nombres de columnas y configuración.
            self.model_wrapper.load(str(path))
            
            self.model_path = path
            self.loaded = True
            logger.info(f"✅ Inferencia lista. Modelo activo: {self.model_wrapper.model_type}")
            
        except Exception as e:
            logger.critical(f"❌ Error al cargar el modelo (¿Archivo corrupto?): {e}")
            raise

    def predict_scenario(self, df_recent_history: pd.DataFrame) -> Dict:
        """
        Ejecuta una predicción para el estado actual del proceso.

        IMPORTANTE - TEORÍA DE SERIES DE TIEMPO:
        Un modelo temporal no puede predecir con una sola fila de datos (t).
        Necesita el contexto histórico (t-1, t-5, etc.) para calcular
        lags y promedios móviles.
        
        Por eso, este método requiere un DataFrame con historia reciente,
        aunque solo devuelva la predicción para el último instante.

        Args:
            df_recent_history (pd.DataFrame): DataFrame con las últimas N filas
                                              (ej. 50 registros) de los sensores.

        Returns:
            Dict: Diccionario con la predicción, valor real (si existe) y metadatos.
                  Estructura: {
                      "timestamp": str,
                      "predicted_value": float,
                      "real_value": float | None,
                      "model_used": str,
                      "confidence_std": float
                  }

        Raises:
            RuntimeError: Si el modelo no ha sido cargado previamente.
        """
        if not self.loaded:
            raise RuntimeError("⚠️ Intento de predicción sin modelo cargado.")

        try:
            target_col = self.model_wrapper.target_col
            
            # -----------------------------------------------------------------
            # PASO 1: Feature Engineering en Tiempo Real
            # -----------------------------------------------------------------
            # Usamos el wrapper para generar lags (t-1, t-5) y diffs.
            # Esto garantiza que la transformación sea IDÉNTICA a la del entrenamiento.
            df_features = self.model_wrapper._create_lag_features(df_recent_history, target_col)
            
            # Nos interesa predecir SOLO para el último instante de tiempo (el "ahora")
            last_row = df_features.iloc[[-1]].copy()
            
            # -----------------------------------------------------------------
            # PASO 2: Alineación de Columnas (Schema Matching)
            # -----------------------------------------------------------------
            # El modelo espera un orden y número exacto de columnas.
            # Si en producción falta un sensor, debemos rellenarlo para no romper el modelo.
            expected_features = self.model_wrapper.feature_names
            
            X_input = pd.DataFrame(index=last_row.index)
            
            for feature in expected_features:
                if feature in last_row.columns:
                    X_input[feature] = last_row[feature]
                else:
                    # Fallback de seguridad: 0.0 si falta una columna calculada
                    # (Esto no debería pasar si el historial es suficiente)
                    logger.warning(f"⚠️ Feature faltante: {feature}. Imputando con 0.0")
                    X_input[feature] = 0.0
            
            # -----------------------------------------------------------------
            # PASO 3: Predicción y Desescalado
            # -----------------------------------------------------------------
            # Transformamos a la escala que conoce el modelo (Standard/Robust)
            X_values = X_input.values
            X_scaled = self.model_wrapper.scaler_X.transform(X_values)
            
            # Ejecutar predicción según el tipo de modelo cargado (GP o GBR)
            confidence_interval = 0.0
            
            if self.model_wrapper.model_type == "GP":
                # Gaussian Process devuelve valor + desviación estándar (incertidumbre)
                y_pred_sc, y_std_sc = self.model_wrapper.model.predict(X_scaled, return_std=True)
                confidence_interval = float(y_std_sc[0]) 
            else:
                # Gradient Boosting / Random Forest solo devuelve valor
                y_pred_sc = self.model_wrapper.model.predict(X_scaled)
            
            # Inversión del escalado para obtener unidades reales (ej. % de recuperación)
            y_pred_final = self.model_wrapper.scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()[0]
            
            # -----------------------------------------------------------------
            # PASO 4: Construcción de Respuesta
            # -----------------------------------------------------------------
            # Si el dataframe de entrada tenía el target, lo devolvemos para comparar
            real_value = last_row[target_col].values[0] if target_col in last_row else None
            
            return {
                "timestamp": str(last_row.index[0]),
                "predicted_value": float(y_pred_final),
                "real_value": float(real_value) if real_value else None,
                "model_used": self.model_wrapper.model_type,
                "confidence_std": confidence_interval
            }

        except Exception as e:
            logger.error(f"❌ Fallo durante la inferencia: {e}")
            raise