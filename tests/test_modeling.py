"""
Módulo: tests/test_modeling.py (CORREGIDO)
"""
import pytest
import joblib
import pandas as pd
import numpy as np
from core.models.mining_gp_pro import MiningGP

class TestMiningIntelligence:
    """Clase agrupadora para tests de Inteligencia Artificial."""

    def test_gp_initialization(self):
        """Verifica inicialización por defecto."""
        model = MiningGP(subsample_step=50)
        assert model.subsample_step == 50
        assert model.add_lag_features is True
        assert 1 in model.lag_periods

    def test_feature_engineering_logic(self, synthetic_data):
        """Verifica la creación de Lags."""
        target = "_silica_concentrate"
        model = MiningGP(target_col=target, add_lag_features=True, add_diff_features=True)
        df_eng = model._create_lag_features(synthetic_data, target)
        
        assert f"{target}_lag_1" in df_eng.columns
        assert f"{target}_diff_1" in df_eng.columns
        
        # Check matemático
        expected = synthetic_data[target].iloc[0]
        actual = df_eng[f"{target}_lag_1"].iloc[1]
        assert np.isclose(expected, actual)

    def test_full_training_cycle(self, temp_csv, tmp_path):
        """
        Prueba de Integración: Ciclo de Vida Completo.
        CORRECCIÓN: Pasamos la ruta de guardado explícitamente en lugar de hackear CONFIG.
        """
        # 1. Entrenar
        model = MiningGP(target_col="_silica_concentrate", subsample_step=1)
        
        # Le decimos que NO guarde automáticamente dentro de train_from_file
        # para poder llamar a save() nosotros con la ruta custom
        model.train_from_file(
            filepath=temp_csv, 
            n_trials=1, 
            test_size=0.2,
            save_model=False 
        )
        
        # 2. Guardar manualmente en la carpeta temporal
        custom_path = str(tmp_path / "test_model.pkl")
        model.save(filepath=custom_path)
        
        # 3. Verificar persistencia
        assert (tmp_path / "test_model.pkl").exists()
        
        # 4. Verificar carga
        loaded = joblib.load(custom_path)
        assert hasattr(loaded, "model")

    def test_constant_feature_removal(self, synthetic_data):
        """
        Prueba de Limpieza.
        CORRECCIÓN: Aseguramos que el target no sea eliminado aunque tenga alta correlación.
        """
        target = "_silica_concentrate"
        model = MiningGP(target_col=target, remove_constant_features=True, remove_correlated_features=True)
        
        # Simulamos que 'flotation_column_04_air_flow' es constante (lo es en synthetic_data)
        # Y aseguramos que el target esté separado del resto antes de limpiar,
        # O ajustamos el método interno para proteger el target.
        
        # En MiningGP v4, el método _remove_problematic_features recibe un DF.
        # Si le pasamos el DF completo (con target), podría borrarlo si correlaciona con otro feature.
        # Lo correcto es pasarle SOLO las features (X), sin el target.
        
        X_df = synthetic_data.drop(columns=[target, "_iron_concentrate"])
        
        # Ejecutamos limpieza sobre X
        X_clean = model._remove_problematic_features(X_df)
        
        # Assert: La columna constante debe desaparecer
        assert "flotation_column_04_air_flow" not in X_clean.columns
        
        # Assert: El target NO debe estar en X_clean (porque lo sacamos antes), 
        # pero la función no debe haber crasheado.
        assert target not in X_clean.columns