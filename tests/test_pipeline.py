"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: tests/test_pipeline.py
Versión: 2.0.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Tests unitarios y de integración para el pipeline completo.

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.0.0 - Enero 2026] COMPATIBILIDAD CON MÓDULOS v2.0
    -----------------------------------------------------
    - Tests actualizados para schema v2.0 (pattern matching)
    - Tests para MiningDataAdapter (reemplaza MiningCSVAdapter + UniversalAdapter)
    - Tests para inference_engine v1.2 (predict_series, get_feature_importance)
    - Eliminados tests obsoletos de adapters viejos

═══════════════════════════════════════════════════════════════════════════════

Ejecutar con: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TESTS: SCHEMA v2.0 (Pattern Matching Universal)
# =============================================================================

@pytest.mark.schema
class TestMiningSchema:
    """Tests para el esquema de validación universal v2.0."""
    
    def test_schema_import(self):
        """Verificar que el schema se importa correctamente."""
        from core.validation.schema import SCHEMA, PhysicalCategory
        assert SCHEMA is not None
        assert PhysicalCategory is not None
    
    def test_category_detection_gold_recovery(self):
        """Verificar detección de categorías para columnas de gold_recovery."""
        from core.validation.schema import SCHEMA, PhysicalCategory
        
        # Porcentajes de metales
        assert SCHEMA.get_category("rougher.input.feed_au") == PhysicalCategory.PERCENTAGE
        assert SCHEMA.get_category("rougher.output.recovery") == PhysicalCategory.PERCENTAGE
        
        # Niveles
        assert SCHEMA.get_category("primary_cleaner.state.floatbank8_a_level") == PhysicalCategory.LEVEL
        
        # Flujos
        assert SCHEMA.get_category("flotation_section_01_air_amount") == PhysicalCategory.FLOW_RATE
    
    def test_category_detection_ai4i2020(self):
        """Verificar detección de categorías para columnas de AI4I2020."""
        from core.validation.schema import SCHEMA, PhysicalCategory
        
        # Temperaturas
        assert SCHEMA.get_category("Air temperature [K]") == PhysicalCategory.TEMPERATURE_KELVIN
        assert SCHEMA.get_category("Process temperature [K]") == PhysicalCategory.TEMPERATURE_KELVIN
        
        # Mecánicas
        assert SCHEMA.get_category("Torque [Nm]") == PhysicalCategory.TORQUE
        assert SCHEMA.get_category("Rotational speed [rpm]") == PhysicalCategory.ROTATIONAL_SPEED
        
        # Tool wear
        assert SCHEMA.get_category("Tool wear [min]") == PhysicalCategory.TOOL_WEAR
        
        # Binarias
        assert SCHEMA.get_category("Machine failure") == PhysicalCategory.BINARY
        assert SCHEMA.get_category("TWF") == PhysicalCategory.BINARY
    
    def test_get_range_percentage(self):
        """Verificar rangos para categoría PERCENTAGE."""
        from core.validation.schema import SCHEMA
        
        min_val, max_val = SCHEMA.get_range("rougher.input.feed_au")
        assert min_val == 0.0
        assert max_val == 100.0
    
    def test_get_range_temperature_kelvin(self):
        """Verificar rangos para categoría TEMPERATURE_KELVIN."""
        from core.validation.schema import SCHEMA
        
        min_val, max_val = SCHEMA.get_range("Air temperature [K]")
        assert min_val == 200.0
        assert max_val == 600.0
    
    def test_get_range_unknown_returns_infinite(self):
        """Verificar fallback a infinito para columnas sin categoría."""
        from core.validation.schema import SCHEMA, PhysicalCategory
        
        category = SCHEMA.get_category("columna_completamente_random_xyz_123")
        assert category == PhysicalCategory.UNKNOWN
        
        min_val, max_val = SCHEMA.get_range("columna_completamente_random_xyz_123")
        assert min_val == float("-inf")
        assert max_val == float("inf")
    
    def test_add_override_rule(self, fresh_schema):
        """Verificar que se pueden agregar overrides."""
        fresh_schema.add_rule("mi_sensor_custom", 0, 999)
        
        min_val, max_val = fresh_schema.get_range("mi_sensor_custom")
        assert min_val == 0
        assert max_val == 999
    
    def test_override_takes_precedence(self, fresh_schema):
        """Verificar que override tiene prioridad sobre pattern matching."""
        # Por defecto, "_au" matchea PERCENTAGE [0, 100]
        min_val, max_val = fresh_schema.get_range("test_au")
        assert max_val == 100.0
        
        # Agregar override
        fresh_schema.add_rule("test_au", 0, 50)
        
        # Ahora debe usar el override
        min_val, max_val = fresh_schema.get_range("test_au")
        assert max_val == 50
    
    def test_analyze_columns(self, fresh_schema):
        """Verificar análisis de múltiples columnas."""
        columns = [
            "rougher.input.feed_au",
            "Air temperature [K]",
            "unknown_column"
        ]
        
        analysis = fresh_schema.analyze_columns(columns)
        
        assert len(analysis) == 3
        assert analysis["rougher.input.feed_au"]["category"] == "PERCENTAGE"
        assert analysis["Air temperature [K]"]["category"] == "TEMPERATURE_KELVIN"
        assert analysis["unknown_column"]["category"] == "UNKNOWN"
    
    def test_validate_value(self, fresh_schema):
        """Verificar validación de valores individuales."""
        # Valor válido
        is_valid, error = fresh_schema.validate_value("rougher.input.feed_au", 50.0)
        assert is_valid is True
        assert error is None
        
        # Valor inválido (> 100%)
        is_valid, error = fresh_schema.validate_value("rougher.input.feed_au", 150.0)
        assert is_valid is False
        assert "máximo" in error.lower()
        
        # NaN
        is_valid, error = fresh_schema.validate_value("rougher.input.feed_au", float("nan"))
        assert is_valid is False


# =============================================================================
# TESTS: VALIDATOR v2.0
# =============================================================================

@pytest.mark.validation
class TestMiningValidator:
    """Tests para el validador de datos v2.0."""
    
    def test_validator_import(self):
        """Verificar que el validador se importa correctamente."""
        from core.validation.validator import MiningValidator, ValidationStats
        assert MiningValidator is not None
        assert ValidationStats is not None
    
    def test_validate_filters_invalid_rows(self, synthetic_data_with_invalids, fresh_validator):
        """Verificar que se filtran filas con valores fuera de rango."""
        df_clean = fresh_validator.validate(synthetic_data_with_invalids)
        
        # Debe haber menos filas que el original
        assert len(df_clean) < len(synthetic_data_with_invalids)
    
    def test_validate_preserves_nan(self, fresh_validator):
        """Verificar que NaN se preservan (no se consideran inválidos)."""
        df = pd.DataFrame({
            "rougher.input.feed_au": [2.5, np.nan, 4.0],
            "other_col": [1.0, 2.0, 3.0]
        })
        
        df_clean = fresh_validator.validate(df)
        
        # La fila con NaN debe estar presente
        assert df_clean["rougher.input.feed_au"].isna().any()
    
    def test_validate_empty_df(self, fresh_validator):
        """Verificar manejo de DataFrame vacío."""
        df_empty = pd.DataFrame()
        df_result = fresh_validator.validate(df_empty)
        
        assert df_result.empty
    
    def test_validation_stats(self, synthetic_data_with_invalids, fresh_validator):
        """Verificar que se generan estadísticas correctas."""
        fresh_validator.validate(synthetic_data_with_invalids)
        
        stats = fresh_validator.last_stats
        assert stats is not None
        assert stats.filas_entrada == 10
        assert stats.filas_eliminadas_total > 0
        assert len(stats.categorias_detectadas) > 0
    
    def test_stats_include_categories(self, synthetic_data_with_invalids, fresh_validator):
        """Verificar que las estadísticas incluyen categorías detectadas."""
        fresh_validator.validate(synthetic_data_with_invalids)
        
        cats = fresh_validator.last_stats.categorias_detectadas
        assert "rougher.input.feed_au" in cats
        assert cats["rougher.input.feed_au"] == "PERCENTAGE"
    
    def test_diagnose_returns_report(self, synthetic_data_with_invalids, fresh_validator):
        """Verificar que diagnose() retorna un reporte completo."""
        report = fresh_validator.diagnose(synthetic_data_with_invalids)
        
        assert isinstance(report, pd.DataFrame)
        assert "columna" in report.columns
        assert "categoria" in report.columns
        assert "n_invalidos" in report.columns
        assert "pct_invalidos" in report.columns
    
    def test_get_invalid_summary(self, synthetic_data_with_invalids, fresh_validator):
        """Verificar que get_invalid_summary() solo muestra problemáticas."""
        summary = fresh_validator.get_invalid_summary(synthetic_data_with_invalids)
        
        # Solo columnas con problemas
        assert all(
            (summary["n_invalidos"] > 0) | (summary["pct_nan"] > 10)
        )


# =============================================================================
# TESTS: PREPROCESSOR
# =============================================================================

class TestMiningPreprocessor:
    """Tests para el preprocesador."""
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame con valores problemáticos."""
        return pd.DataFrame({
            "sensor_a": [1.0, 2.0, np.inf, 4.0, np.nan, 6.0],
            "sensor_b": [10.0, np.nan, 30.0, -np.inf, 50.0, 60.0],
        })
    
    def test_preprocessor_import(self):
        """Verificar que el preprocesador se importa."""
        from core.preprocessor import MiningPreprocessor
        assert MiningPreprocessor is not None
    
    def test_replaces_infinites(self, sample_df):
        """Verificar que infinitos se reemplazan."""
        from core.preprocessor import MiningPreprocessor
        
        preprocessor = MiningPreprocessor()
        df_clean = preprocessor.clean_stream(sample_df)
        
        assert not np.isinf(df_clean.values).any()
    
    def test_imputes_nulls_ffill(self, sample_df):
        """Verificar imputación forward fill."""
        from core.preprocessor import MiningPreprocessor
        
        preprocessor = MiningPreprocessor(estrategia_nulos="ffill")
        df_clean = preprocessor.clean_stream(sample_df)
        
        assert not df_clean.isna().any().any()
    
    def test_imputes_nulls_interpolate(self, sample_df):
        """Verificar imputación por interpolación."""
        from core.preprocessor import MiningPreprocessor
        
        preprocessor = MiningPreprocessor(estrategia_nulos="interpolate")
        df_clean = preprocessor.clean_stream(sample_df)
        
        assert not df_clean.isna().any().any()
    
    def test_invalid_strategy_raises(self):
        """Verificar que estrategia inválida lanza error."""
        from core.preprocessor import MiningPreprocessor
        
        with pytest.raises(ValueError):
            MiningPreprocessor(estrategia_nulos="invalid_strategy")


# =============================================================================
# TESTS: CONFIG
# =============================================================================

class TestConfig:
    """Tests para la configuración."""
    
    def test_config_import(self):
        """Verificar que CONFIG se importa correctamente."""
        from config.settings import CONFIG
        assert CONFIG is not None
    
    def test_config_has_required_attributes(self):
        """Verificar atributos requeridos."""
        from config.settings import CONFIG
        
        assert hasattr(CONFIG, 'DATA_DIR')
        assert hasattr(CONFIG, 'MODELS_DIR')
        assert hasattr(CONFIG, 'PROJECT_ROOT')
    
    def test_config_paths_are_pathlib(self):
        """Verificar que las rutas son objetos Path."""
        from config.settings import CONFIG
        
        assert isinstance(CONFIG.DATA_DIR, Path)
        assert isinstance(CONFIG.MODELS_DIR, Path)


# =============================================================================
# TESTS: INFERENCE ENGINE v1.2
# =============================================================================

class TestInferenceEngine:
    """Tests para el motor de inferencia v1.2."""
    
    def test_inference_import(self):
        """Verificar imports."""
        from core.inference_engine import MiningInference
        assert MiningInference is not None
    
    def test_calculate_confidence(self):
        """Verificar cálculo de confianza."""
        from core.inference_engine import MiningInference
        
        # Crear instancia sin cargar modelo
        engine = MiningInference.__new__(MiningInference)
        
        # std = 0 (GBR) → confianza por defecto
        conf = engine.calculate_confidence(0.0, 85.0)
        assert conf == 85.0
        
        # std pequeña → alta confianza
        conf = engine.calculate_confidence(1.0, 85.0)
        assert conf > 80.0
        
        # std grande → baja confianza
        conf = engine.calculate_confidence(50.0, 85.0)
        assert conf < 50.0
    
    def test_get_model_info_no_model(self):
        """Verificar get_model_info sin modelo cargado."""
        from core.inference_engine import MiningInference
        
        engine = MiningInference.__new__(MiningInference)
        engine.loaded = False
        
        info = engine.get_model_info()
        assert info["status"] == "No model loaded"


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Tests de integración que verifican el pipeline completo."""
    
    def test_schema_validator_integration(self, synthetic_data, fresh_validator):
        """Verificar que schema y validator trabajan juntos."""
        # Inyectar valores inválidos
        df = synthetic_data.copy()
        df.loc[df.index[0], "rougher.input.feed_au"] = 150.0  # > 100%
        
        # Validar
        df_clean = fresh_validator.validate(df)
        
        # La fila con 150% debe haber sido eliminada
        assert 150.0 not in df_clean["rougher.input.feed_au"].values
    
    def test_full_pipeline_smoke(self, synthetic_data, tmp_path):
        """
        Smoke test del pipeline: datos → validación → entrenamiento.
        """
        from core.validation.validator import MiningValidator
        from core.models.mining_gp_pro import MiningGP
        
        # 1. Validar
        validator = MiningValidator()
        df_valid = validator.validate(synthetic_data)
        
        assert len(df_valid) > 100  # Suficientes datos
        
        # 2. Guardar CSV
        csv_path = tmp_path / "pipeline_test.csv"
        df_valid.to_csv(csv_path)
        
        # 3. Entrenar (rápido)
        model = MiningGP(
            target_col="rougher.output.recovery",
            subsample_step=20,
        )
        
        model.train_from_file(
            filepath=str(csv_path),
            n_trials=1,
            test_size=0.3,
            save_model=False
        )
        
        # 4. Verificar que entrenó
        assert model.model is not None
        assert model.scaler_X is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
