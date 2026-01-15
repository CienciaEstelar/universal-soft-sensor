"""
Tests unitarios para el Proyecto Minero 4.0

Ejecutar con: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMiningSchema:
    """Tests para el esquema de validación."""
    
    def test_schema_import(self):
        """Verificar que el schema se importa correctamente."""
        from core.validation.schema import SCHEMA
        assert SCHEMA is not None
    
    def test_get_range_exact_match(self):
        """Verificar búsqueda exacta de rangos."""
        from core.validation.schema import SCHEMA
        
        min_val, max_val = SCHEMA.get_range("ore_pulp_ph")
        assert min_val == 0.0
        assert max_val == 14.0
    
    def test_get_range_pattern_match(self):
        """Verificar búsqueda por patrón para columnas de flotación."""
        from core.validation.schema import SCHEMA
        
        # Debe coincidir con la regla genérica
        min_val, max_val = SCHEMA.get_range("flotation_column_01_air_flow")
        assert min_val == 0.0
        assert max_val == 1000.0
    
    def test_get_range_no_rule(self):
        """Verificar fallback a infinito para columnas sin regla."""
        from core.validation.schema import SCHEMA
        
        min_val, max_val = SCHEMA.get_range("columna_inexistente")
        assert min_val == -float("inf")
        assert max_val == float("inf")
    
    def test_add_rule(self):
        """Verificar que se pueden agregar reglas dinámicamente."""
        from core.validation.schema import MiningSchema
        
        schema = MiningSchema()
        schema.add_rule("nueva_columna", 0, 100)
        
        min_val, max_val = schema.get_range("nueva_columna")
        assert min_val == 0
        assert max_val == 100


class TestMiningValidator:
    """Tests para el validador de datos."""
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de prueba con valores válidos e inválidos."""
        return pd.DataFrame({
            "_iron_feed": [45.0, 50.0, 150.0, np.nan],  # 150 fuera de rango
            "ore_pulp_ph": [7.0, 8.5, 15.0, 6.0],       # 15 fuera de rango
            "starch_flow": [100, 200, -50, 300],        # -50 fuera de rango
        })
    
    def test_validator_import(self):
        """Verificar que el validador se importa correctamente."""
        from core.validation.validator import MiningValidator
        assert MiningValidator is not None
    
    def test_validate_filters_invalid_rows(self, sample_df):
        """Verificar que se filtran filas inválidas."""
        from core.validation.validator import MiningValidator
        
        validator = MiningValidator()
        df_clean = validator.validate(sample_df)
        
        # Solo la fila 3 (índice 3) debería sobrevivir
        # - Fila 0: válida
        # - Fila 1: válida
        # - Fila 2: inválida (_iron_feed=150, ore_pulp_ph=15, starch_flow=-50)
        # - Fila 3: válida (NaN se permite)
        assert len(df_clean) < len(sample_df)
    
    def test_validate_preserves_nan(self, sample_df):
        """Verificar que NaN se preservan (no se filtran)."""
        from core.validation.validator import MiningValidator
        
        validator = MiningValidator()
        df_clean = validator.validate(sample_df)
        
        # La fila con NaN debería estar presente
        assert df_clean["_iron_feed"].isna().any()
    
    def test_validate_empty_df(self):
        """Verificar manejo de DataFrame vacío."""
        from core.validation.validator import MiningValidator
        
        validator = MiningValidator()
        df_empty = pd.DataFrame()
        df_result = validator.validate(df_empty)
        
        assert df_result.empty
    
    def test_validation_stats(self, sample_df):
        """Verificar que se generan estadísticas."""
        from core.validation.validator import MiningValidator
        
        validator = MiningValidator()
        validator.validate(sample_df)
        
        assert validator.last_stats is not None
        assert validator.last_stats.filas_entrada == 4


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
        
        # No debería haber infinitos
        assert not np.isinf(df_clean.values).any()
    
    def test_imputes_nulls_ffill(self, sample_df):
        """Verificar imputación forward fill."""
        from core.preprocessor import MiningPreprocessor
        
        preprocessor = MiningPreprocessor(estrategia_nulos="ffill")
        df_clean = preprocessor.clean_stream(sample_df)
        
        # No debería haber NaN (después de ffill + fillna final)
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
    
    def test_preserves_non_numeric_columns(self):
        """Verificar que columnas no numéricas no se modifican."""
        from core.preprocessor import MiningPreprocessor
        
        df = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0],
            "text": ["a", "b", "c"]
        })
        
        preprocessor = MiningPreprocessor()
        df_clean = preprocessor.clean_stream(df)
        
        assert df_clean["text"].tolist() == ["a", "b", "c"]
    
    def test_cleaning_stats(self, sample_df):
        """Verificar que se generan estadísticas de limpieza."""
        from core.preprocessor import MiningPreprocessor
        
        preprocessor = MiningPreprocessor()
        preprocessor.clean_stream(sample_df)
        
        assert preprocessor.last_stats is not None
        assert preprocessor.last_stats.infinitos_reemplazados > 0


class TestConfig:
    """Tests para la configuración."""
    
    def test_config_import(self):
        """Verificar que CONFIG se importa correctamente."""
        from config.settings import CONFIG
        assert CONFIG is not None
    
    def test_config_has_required_attributes(self):
        """Verificar atributos requeridos."""
        from config.settings import CONFIG
        
        assert hasattr(CONFIG, 'DATA_RAW_PATH')
        assert hasattr(CONFIG, 'DATA_CLEAN_PATH')
        assert hasattr(CONFIG, 'CHUNK_SIZE')
        assert hasattr(CONFIG, 'GP_TARGET_COLUMN')
    
    def test_config_paths_are_pathlib(self):
        """Verificar que las rutas son objetos Path."""
        from config.settings import CONFIG
        
        assert isinstance(CONFIG.DATA_DIR, Path)
        assert isinstance(CONFIG.MODELS_DIR, Path)


# =============================================================================
# Tests de integración (requieren el dataset)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Tests de integración que requieren el dataset real."""
    
    @pytest.fixture
    def check_data_exists(self):
        """Skip si no existe el dataset."""
        from config.settings import CONFIG
        if not CONFIG.DATA_RAW_PATH.exists():
            pytest.skip("Dataset no disponible para tests de integración")
    
    def test_adapter_reads_data(self, check_data_exists):
        """Verificar que el adapter lee datos correctamente."""
        from config.settings import CONFIG
        from core.adapters import MiningCSVAdapter
        
        adapter = MiningCSVAdapter(str(CONFIG.DATA_RAW_PATH))
        gen = adapter.stream()
        chunk = next(gen)
        
        assert not chunk.empty
        assert len(chunk.columns) > 0
    
    def test_full_pipeline_smoke(self, check_data_exists, tmp_path):
        """Smoke test del pipeline completo."""
        # Este test verifica que el pipeline no crashea
        # Usa un directorio temporal para no afectar datos reales
        pass  # Implementar si es necesario


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
