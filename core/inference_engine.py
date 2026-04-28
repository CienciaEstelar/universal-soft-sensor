"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/inference_engine.py
Versión: 1.2.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Motor de Inferencia dedicado para Producción.
    
    Este módulo actúa como una 'Fachada' (Facade Pattern) que abstrae la 
    complejidad de cargar modelos, generar features en tiempo real y 
    desescalar predicciones.

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v1.2.0 - Enero 2026] PREDICCIÓN EN SERIE + FEATURE IMPORTANCE
    ---------------------------------------------------------------
    - NUEVO: predict_series() para predicciones rolling (dashboard)
    - NUEVO: get_feature_importance() para explicabilidad real
    - NUEVO: calculate_confidence() convierte std a porcentaje
    - MEJORADO: Logging más detallado
    
    [v1.1.0]
    --------
    - Documentación extendida
    - Manejo de errores mejorado

═══════════════════════════════════════════════════════════════════════════════
USO:
═══════════════════════════════════════════════════════════════════════════════

    from core.inference_engine import MiningInference
    
    engine = MiningInference()
    
    # Predicción única (último punto)
    resultado = engine.predict_scenario(df_ultimas_50_horas)
    
    # Predicción en serie (para gráficos)
    serie = engine.predict_series(df_historico, n_points=100)
    
    # Feature importance (para XAI)
    importance = engine.get_feature_importance()

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Importaciones internas
from config.settings import CONFIG
from core.models.mining_gp_pro import MiningGP, TrainingArtifacts

logger = logging.getLogger("InferenceEngine")


class MiningInference:
    """
    Clase controladora para la ejecución de modelos mineros en producción.
    
    Attributes
    ----------
    model_wrapper : MiningGP
        Wrapper del modelo con scalers y configuración.
    loaded : bool
        Indica si hay un modelo cargado en memoria.
    model_path : Path
        Ruta al archivo .pkl del modelo activo.
    _feature_importance_cache : dict
        Cache de feature importance para evitar recálculo.
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
        self.model_wrapper = MiningGP() 
        self.loaded = False
        self.model_path = None
        self._feature_importance_cache = None
        
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
            models = list(CONFIG.MODELS_DIR.glob("*.pkl"))
            
            if not models:
                msg = f"❌ No se encontraron modelos entrenados en: {CONFIG.MODELS_DIR}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            
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
            self.model_wrapper.load(str(path))
            
            self.model_path = path
            self.loaded = True
            self._feature_importance_cache = None  # Invalidar cache
            
            logger.info(f"✅ Inferencia lista. Modelo activo: {self.model_wrapper.model_type}")
            
        except Exception as e:
            logger.critical(f"❌ Error al cargar el modelo (¿Archivo corrupto?): {e}")
            raise

    def predict_scenario(self, df_recent_history: pd.DataFrame) -> Dict:
        """
        Ejecuta una predicción para el estado actual del proceso (último punto).

        Args:
            df_recent_history (pd.DataFrame): DataFrame con las últimas N filas
                                              (ej. 50 registros) de los sensores.

        Returns:
            Dict: {
                "timestamp": str,
                "predicted_value": float,
                "real_value": float | None,
                "model_used": str,
                "confidence_std": float,
                "confidence_pct": float  # NUEVO: Confianza como porcentaje
            }

        Raises:
            RuntimeError: Si el modelo no ha sido cargado previamente.
        """
        if not self.loaded:
            raise RuntimeError("⚠️ Intento de predicción sin modelo cargado.")

        try:
            target_col = self.model_wrapper.target_col
            
            # PASO 1: Feature Engineering
            df_features = self.model_wrapper._create_lag_features(df_recent_history, target_col)
            last_row = df_features.iloc[[-1]].copy()
            
            # PASO 2: Alineación de Columnas
            expected_features = self.model_wrapper.feature_names
            X_input = pd.DataFrame(index=last_row.index)
            
            for feature in expected_features:
                if feature in last_row.columns:
                    X_input[feature] = last_row[feature]
                else:
                    logger.warning(f"⚠️ Feature faltante: {feature}. Imputando con 0.0")
                    X_input[feature] = 0.0
            
            # PASO 3: Predicción
            X_values = X_input.values
            X_scaled = self.model_wrapper.scaler_X.transform(X_values)
            
            confidence_std = 0.0
            
            if self.model_wrapper.model_type == "GP":
                y_pred_sc, y_std_sc = self.model_wrapper.model.predict(X_scaled, return_std=True)
                confidence_std = float(y_std_sc[0]) 
            else:
                y_pred_sc = self.model_wrapper.model.predict(X_scaled)
            
            # Desescalado
            y_pred_final = self.model_wrapper.scaler_y.inverse_transform(
                y_pred_sc.reshape(-1, 1)
            ).ravel()[0]
            
            # Valor real si existe
            real_value = last_row[target_col].values[0] if target_col in last_row else None
            
            # Calcular confianza como porcentaje
            confidence_pct = self.calculate_confidence(confidence_std, y_pred_final)
            
            return {
                "timestamp": str(last_row.index[0]),
                "predicted_value": float(y_pred_final),
                "real_value": float(real_value) if real_value is not None else None,
                "model_used": self.model_wrapper.model_type,
                "confidence_std": confidence_std,
                "confidence_pct": confidence_pct,
            }

        except Exception as e:
            logger.error(f"❌ Fallo durante la inferencia: {e}")
            raise

    def predict_series(
        self, 
        df_history: pd.DataFrame, 
        n_points: int = 100,
        min_history: int = 50
    ) -> pd.DataFrame:
        """
        Genera predicciones para una serie de puntos (rolling prediction).
        
        Útil para graficar la línea de predicción vs real en el dashboard.
        
        Args:
            df_history (pd.DataFrame): DataFrame con historia completa.
            n_points (int): Número de puntos a predecir.
            min_history (int): Mínimo de filas históricas necesarias para cada predicción.
            
        Returns:
            pd.DataFrame: DataFrame con columnas:
                - timestamp (index)
                - predicted: Valor predicho
                - real: Valor real (si existe)
                - confidence_std: Desviación estándar
                - confidence_pct: Confianza en porcentaje
                
        Example:
            >>> serie = engine.predict_series(df, n_points=100)
            >>> fig.add_trace(go.Scatter(x=serie.index, y=serie['predicted']))
        """
        if not self.loaded:
            raise RuntimeError("⚠️ Intento de predicción sin modelo cargado.")
        
        target_col = self.model_wrapper.target_col
        results = []
        
        # Determinar rango de índices a predecir
        total_rows = len(df_history)
        start_idx = max(min_history, total_rows - n_points)
        
        logger.info(f"🔮 Generando {total_rows - start_idx} predicciones rolling...")
        
        for i in range(start_idx, total_rows):
            try:
                # Ventana de historia para este punto
                window = df_history.iloc[max(0, i - min_history):i + 1]
                
                # Generar features
                df_features = self.model_wrapper._create_lag_features(window, target_col)
                
                if df_features.empty:
                    continue
                    
                last_row = df_features.iloc[[-1]]
                
                # Alinear columnas
                expected_features = self.model_wrapper.feature_names
                X_input = pd.DataFrame(index=last_row.index)
                
                for feature in expected_features:
                    if feature in last_row.columns:
                        X_input[feature] = last_row[feature]
                    else:
                        X_input[feature] = 0.0
                
                # Predecir
                X_scaled = self.model_wrapper.scaler_X.transform(X_input.values)
                
                if self.model_wrapper.model_type == "GP":
                    y_pred_sc, y_std_sc = self.model_wrapper.model.predict(X_scaled, return_std=True)
                    conf_std = float(y_std_sc[0])
                else:
                    y_pred_sc = self.model_wrapper.model.predict(X_scaled)
                    conf_std = 0.0
                
                # Desescalar
                y_pred = self.model_wrapper.scaler_y.inverse_transform(
                    y_pred_sc.reshape(-1, 1)
                ).ravel()[0]
                
                # Valor real
                real_val = last_row[target_col].values[0] if target_col in last_row.columns else None
                
                results.append({
                    "timestamp": last_row.index[0],
                    "predicted": float(y_pred),
                    "real": float(real_val) if real_val is not None else None,
                    "confidence_std": conf_std,
                    "confidence_pct": self.calculate_confidence(conf_std, y_pred),
                })
                
            except Exception as e:
                logger.debug(f"Skip punto {i}: {e}")
                continue
        
        if not results:
            logger.warning("⚠️ No se generaron predicciones")
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results.set_index("timestamp", inplace=True)
        
        logger.info(f"✅ Serie generada: {len(df_results)} predicciones")
        return df_results

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Obtiene la importancia de features del modelo.
        
        Para GradientBoosting/RandomForest: usa feature_importances_
        Para GaussianProcess: aproximación por varianza de coeficientes
        
        Args:
            top_n (int): Número de features top a retornar.
            
        Returns:
            Dict[str, float]: {nombre_feature: importancia_normalizada}
            
        Example:
            >>> imp = engine.get_feature_importance(top_n=5)
            >>> # {'rougher.input.feed_au': 0.35, 'air_amount': 0.22, ...}
        """
        if not self.loaded:
            raise RuntimeError("⚠️ Modelo no cargado.")
        
        # Usar cache si existe
        if self._feature_importance_cache is not None:
            return dict(list(self._feature_importance_cache.items())[:top_n])
        
        feature_names = self.model_wrapper.feature_names
        model = self.model_wrapper.model
        
        try:
            # Método 1: Modelos con feature_importances_ nativo
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
            # Método 2: GP - aproximación por longitud de escala inversa
            elif hasattr(model, 'kernel_') and hasattr(model.kernel_, 'get_params'):
                # Para GP con kernel RBF, length_scale indica sensibilidad
                # Menor length_scale = mayor importancia
                params = model.kernel_.get_params()
                
                if 'k2__length_scale' in params:
                    length_scales = np.atleast_1d(params['k2__length_scale'])
                    
                    if len(length_scales) == len(feature_names):
                        # Importancia inversamente proporcional a length_scale
                        importances = 1.0 / (length_scales + 1e-6)
                    else:
                        # GP isotrópico - usar varianza de datos como proxy
                        importances = np.ones(len(feature_names))
                else:
                    importances = np.ones(len(feature_names))
            else:
                # Fallback: importancia uniforme
                logger.warning("⚠️ Modelo sin feature_importances_. Usando uniforme.")
                importances = np.ones(len(feature_names))
            
            # Normalizar a suma = 1 (con guard anti-NaN si todas son cero)
            importances = np.array(importances)
            total = importances.sum()
            if total == 0:
                return dict.fromkeys(feature_names[:top_n], 0.0)
            importances = importances / total
            
            # Crear diccionario ordenado
            importance_dict = dict(zip(feature_names, importances))
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Cachear
            self._feature_importance_cache = importance_dict
            
            return dict(list(importance_dict.items())[:top_n])
            
        except Exception as e:
            logger.error(f"Error obteniendo feature importance: {e}")
            # Fallback seguro
            return {f: 1.0/len(feature_names) for f in feature_names[:top_n]}

    def calculate_confidence(self, std: float, prediction: float) -> float:
        """
        Convierte desviación estándar en porcentaje de confianza.
        
        Lógica:
        - Si std = 0 (GBR sin incertidumbre): retorna valor por defecto
        - Si std es pequeña relativa a la predicción: alta confianza
        - Usa coeficiente de variación (CV) para normalizar
        
        Args:
            std (float): Desviación estándar de la predicción.
            prediction (float): Valor predicho.
            
        Returns:
            float: Confianza en porcentaje [0-100].
        """
        # GBR no tiene incertidumbre calibrada
        if std == 0.0:
            return 85.0  # Valor conservador por defecto para GBR
        
        # Evitar división por cero
        if abs(prediction) < 1e-6:
            return 50.0
        
        # Coeficiente de variación
        cv = abs(std / prediction)
        
        # Mapear CV a confianza
        # CV = 0 → 99%, CV = 0.5 → 50%, CV >= 1 → ~5%
        # Función sigmoide invertida
        confidence = 100 * np.exp(-3 * cv)
        
        # Clampear a [5, 99]
        return float(np.clip(confidence, 5.0, 99.0))

    def get_model_info(self) -> Dict:
        """
        Retorna información del modelo cargado.
        
        Returns:
            Dict con metadata del modelo.
        """
        if not self.loaded:
            return {"status": "No model loaded"}
        
        return {
            "model_path": str(self.model_path),
            "model_type": self.model_wrapper.model_type,
            "target_column": self.model_wrapper.target_col,
            "n_features": len(self.model_wrapper.feature_names),
            "feature_names": self.model_wrapper.feature_names[:10],  # Primeras 10
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("🔮 Test de MiningInference v1.2")
    print("=" * 70)
    
    try:
        engine = MiningInference()
        print(f"\n📊 Info del modelo:")
        for k, v in engine.get_model_info().items():
            print(f"   {k}: {v}")
        
        print(f"\n🎯 Feature Importance (Top 5):")
        for feat, imp in engine.get_feature_importance(top_n=5).items():
            print(f"   {feat}: {imp:.4f}")
            
    except FileNotFoundError:
        print("⚠️ No hay modelo entrenado. Ejecuta train_universal.py primero.")
    except Exception as e:
        print(f"❌ Error: {e}")
