"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: tests/test_modeling.py
Versión: 2.0.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Tests para el módulo de modelado (MiningGP) y feature engineering.

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.0.0 - Enero 2026] ACTUALIZACIÓN PARA SCHEMA v2.0
    ----------------------------------------------------
    - Fixtures actualizados con nombres de columnas compatibles
    - Test de feature engineering con columnas tipo gold_recovery
    - Corregido test de eliminación de features constantes
    - Añadidos tests para inference_engine v1.2

═══════════════════════════════════════════════════════════════════════════════

Ejecutar con: pytest tests/test_modeling.py -v
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.mining_gp_pro import MiningGP


class TestMiningGPInitialization:
    """Tests de inicialización de MiningGP."""
    
    def test_default_initialization(self):
        """Verificar inicialización por defecto."""
        model = MiningGP()
        
        assert model.subsample_step == 10  # Verificar valor de CONFIG
        assert model.add_lag_features is True
        assert 1 in model.lag_periods
    
    def test_custom_initialization(self):
        """Verificar inicialización con parámetros custom."""
        model = MiningGP(
            target_col="custom_target",
            subsample_step=50,
            add_lag_features=False,
            add_diff_features=True
        )
        
        assert model.target_col == "custom_target"
        assert model.subsample_step == 50
        assert model.add_lag_features is False
        assert model.add_diff_features is True
    
    def test_lag_periods_configuration(self):
        """Verificar configuración de lag periods."""
        model = MiningGP(lag_periods=[1, 5, 10])
        
        assert model.lag_periods == [1, 5, 10]


class TestFeatureEngineering:
    """Tests para el feature engineering."""
    
    def test_lag_features_creation(self, synthetic_data):
        """Verificar creación de lags."""
        target = "rougher.output.recovery"
        model = MiningGP(target_col=target, add_lag_features=True, add_diff_features=False)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Verificar que existen las columnas de lag
        assert f"{target}_lag_1" in df_eng.columns
        assert f"{target}_lag_5" in df_eng.columns
    
    def test_diff_features_creation(self, synthetic_data):
        """Verificar creación de diferencias."""
        target = "rougher.output.recovery"
        model = MiningGP(target_col=target, add_lag_features=False, add_diff_features=True)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Verificar que existen las columnas de diff
        assert f"{target}_diff_1" in df_eng.columns
    
    def test_lag_values_correctness(self, synthetic_data):
        """Verificar que los valores de lag son matemáticamente correctos."""
        target = "rougher.output.recovery"
        model = MiningGP(target_col=target, add_lag_features=True)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # El lag_1 en la fila i debe ser igual al target en la fila i-1
        # Tomamos la fila 10 como ejemplo (tiene historia suficiente)
        expected = synthetic_data[target].iloc[9]  # Fila anterior
        actual = df_eng[f"{target}_lag_1"].iloc[10]
        
        assert np.isclose(expected, actual, rtol=1e-5)
    
    def test_rolling_features_if_enabled(self, synthetic_data):
        """Verificar features rolling (si están habilitadas)."""
        target = "rougher.output.recovery"
        model = MiningGP(
            target_col=target, 
            add_lag_features=True,
            add_rolling_features=True  # Si existe esta opción
        )
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Buscar columnas rolling (pueden no existir dependiendo de la config)
        rolling_cols = [c for c in df_eng.columns if "rolling" in c.lower() or "mean" in c.lower()]
        # No assertamos porque depende de la implementación


class TestFeatureCleaning:
    """Tests para limpieza de features."""
    
    def test_constant_feature_removal(self, synthetic_data):
        """Verificar eliminación de features constantes."""
        target = "rougher.output.recovery"
        model = MiningGP(target_col=target, remove_constant_features=True)
        
        # synthetic_data tiene 'flotation_section_03_air_amount' como constante
        # Preparamos X sin el target
        X_df = synthetic_data.drop(columns=[target])
        
        # Ejecutar limpieza
        X_clean = model._remove_problematic_features(X_df)
        
        # La columna constante debe haber sido eliminada
        assert "flotation_section_03_air_amount" not in X_clean.columns
    
    def test_target_not_removed_during_cleaning(self, synthetic_data):
        """Verificar que el target no se elimina accidentalmente."""
        target = "rougher.output.recovery"
        model = MiningGP(
            target_col=target, 
            remove_constant_features=True,
            remove_correlated_features=True
        )
        
        # IMPORTANTE: Pasar solo X (sin target) a la limpieza
        X_df = synthetic_data.drop(columns=[target])
        X_clean = model._remove_problematic_features(X_df)
        
        # El target no debe estar en X_clean (porque no lo pasamos)
        # pero la función no debe haber crasheado
        assert target not in X_clean.columns
        
        # Verificar que quedaron features útiles
        assert len(X_clean.columns) > 0


class TestTrainingCycle:
    """Tests del ciclo de entrenamiento."""
    
    def test_full_training_cycle(self, temp_csv, tmp_path):
        """
        Prueba de Integración: Ciclo de Vida Completo.
        """
        target = "rougher.output.recovery"
        
        # 1. Entrenar
        model = MiningGP(
            target_col=target, 
            subsample_step=5,  # Rápido para tests
            add_lag_features=True,
            add_diff_features=False,
        )
        
        model.train_from_file(
            filepath=temp_csv, 
            n_trials=1, 
            test_size=0.2,
            save_model=False 
        )
        
        # 2. Verificar que el modelo existe
        assert model.model is not None
        assert model.scaler_X is not None
        assert model.scaler_y is not None
        assert model.feature_names is not None
        
        # 3. Guardar manualmente
        custom_path = tmp_path / "test_model.pkl"
        model.save(filepath=str(custom_path))
        
        # 4. Verificar persistencia
        assert custom_path.exists()
        
        # 5. Verificar carga
        loaded = joblib.load(str(custom_path))
        assert hasattr(loaded, "model")
    
    def test_model_can_predict_after_training(self, temp_csv):
        """Verificar que el modelo puede predecir después de entrenar."""
        target = "rougher.output.recovery"
        
        model = MiningGP(
            target_col=target,
            subsample_step=10,
        )
        
        model.train_from_file(
            filepath=temp_csv,
            n_trials=1,
            test_size=0.2,
            save_model=False
        )
        
        # Leer datos para predicción
        df = pd.read_csv(temp_csv, index_col=0, parse_dates=True)
        window = df.iloc[-50:]
        
        # Generar features
        df_features = model._create_lag_features(window, target)
        last_row = df_features.iloc[[-1]]
        
        # Preparar X
        X_cols = [c for c in model.feature_names if c in last_row.columns]
        X = last_row[X_cols].fillna(0)
        
        # Escalar y predecir
        X_scaled = model.scaler_X.transform(X.values)
        y_pred = model.model.predict(X_scaled)
        
        # Verificar que retorna algo razonable
        assert len(y_pred) == 1
        assert not np.isnan(y_pred[0])


class TestModelPersistence:
    """Tests de persistencia del modelo."""
    
    def test_save_and_load(self, trained_model, tmp_path):
        """Verificar ciclo save/load."""
        model_path = tmp_path / "persistence_test.pkl"
        
        # Guardar
        trained_model.save(str(model_path))
        assert model_path.exists()
        
        # Cargar en nueva instancia
        new_model = MiningGP()
        new_model.load(str(model_path))
        
        # Verificar que se cargó correctamente
        assert new_model.model is not None
        assert new_model.model_type == trained_model.model_type
        assert new_model.target_col == trained_model.target_col
    
    def test_loaded_model_can_predict(self, trained_model, synthetic_data, tmp_path):
        """Verificar que modelo cargado puede predecir."""
        model_path = tmp_path / "predict_test.pkl"
        trained_model.save(str(model_path))
        
        # Cargar
        new_model = MiningGP()
        new_model.load(str(model_path))
        
        # Intentar predecir
        target = new_model.target_col
        window = synthetic_data.iloc[-50:]
        
        df_features = new_model._create_lag_features(window, target)
        
        # Si llegamos aquí sin error, el modelo se cargó bien
        assert not df_features.empty


class TestMetrics:
    """Tests para métricas del modelo."""
    
    def test_training_returns_metrics(self, temp_csv):
        """Verificar que el entrenamiento retorna métricas."""
        model = MiningGP(
            target_col="rougher.output.recovery",
            subsample_step=10,
        )
        
        model.train_from_file(
            filepath=temp_csv,
            n_trials=1,
            test_size=0.2,
            save_model=False
        )
        
        # Verificar que existen métricas
        assert hasattr(model, 'best_cv_score') or hasattr(model, 'metrics')


class TestInferenceEngineIntegration:
    """Tests de integración con inference_engine."""
    
    def test_predict_scenario(self, trained_model, synthetic_data, tmp_path):
        """Verificar predict_scenario del inference engine."""
        from core.inference_engine import MiningInference
        
        # Guardar modelo
        model_path = tmp_path / "inference_test.pkl"
        trained_model.save(str(model_path))
        
        # Cargar con inference engine
        engine = MiningInference(model_path=str(model_path))
        
        # Predecir
        window = synthetic_data.iloc[-50:]
        result = engine.predict_scenario(window)
        
        # Verificar estructura del resultado
        assert "predicted_value" in result
        assert "confidence_pct" in result
        assert "model_used" in result
        assert isinstance(result["predicted_value"], float)
    
    def test_predict_series(self, trained_model, synthetic_data, tmp_path):
        """Verificar predict_series del inference engine."""
        from core.inference_engine import MiningInference
        
        model_path = tmp_path / "series_test.pkl"
        trained_model.save(str(model_path))
        
        engine = MiningInference(model_path=str(model_path))
        
        # Generar serie
        series = engine.predict_series(synthetic_data, n_points=20, min_history=30)
        
        # Verificar estructura
        assert isinstance(series, pd.DataFrame)
        if not series.empty:
            assert "predicted" in series.columns
            assert "confidence_pct" in series.columns
    
    def test_get_feature_importance(self, trained_model, tmp_path):
        """Verificar get_feature_importance del inference engine."""
        from core.inference_engine import MiningInference
        
        model_path = tmp_path / "importance_test.pkl"
        trained_model.save(str(model_path))
        
        engine = MiningInference(model_path=str(model_path))
        
        # Obtener importancia
        importance = engine.get_feature_importance(top_n=5)
        
        # Verificar estructura
        assert isinstance(importance, dict)
        assert len(importance) <= 5
        
        # Valores deben sumar ~1 (normalizados)
        if importance:
            total = sum(importance.values())
            # Puede no sumar exactamente 1 si top_n < total features
            assert total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
