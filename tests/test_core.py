"""
Tests unitarios para el Universal Soft-Sensor

Ejecutar con: pytest tests/ -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhysicalSchema:
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
        
        # 'air_flow' matchea el patrón 'flow' → categoría FLOW_RATE
        min_val, max_val = SCHEMA.get_range("flotation_column_01_air_flow")

        # [FIX] ANTES: assert max_val == 1000.0
        #              ↑ Valor hardcodeado del schema v1. Siempre fallaba porque
        #                PhysicalCategory.FLOW_RATE tiene max = 50000.0
        # AHORA: validamos el rango real de FLOW_RATE
        assert min_val == 0.0
        assert max_val == 50000.0
    
    def test_get_range_no_rule(self):
        """Verificar fallback a infinito para columnas sin regla."""
        from core.validation.schema import SCHEMA
        
        min_val, max_val = SCHEMA.get_range("columna_inexistente")
        assert min_val == -float("inf")
        assert max_val == float("inf")
    
    def test_add_rule(self):
        """Verificar que se pueden agregar reglas dinámicamente."""
        from core.validation.schema import PhysicalSchema
        
        schema = PhysicalSchema()
        schema.add_rule("nueva_columna", 0, 100)
        
        min_val, max_val = schema.get_range("nueva_columna")
        assert min_val == 0
        assert max_val == 100


class TestPhysicalValidator:
    """Tests para el validador de datos."""
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de prueba con valores válidos e inválidos."""
        return pd.DataFrame({
            "_iron_feed": [45.0, 50.0, 150.0, np.nan],  # 150 fuera de rango
            "ore_pulp_ph": [7.0, 8.5, 15.0, 6.0],       # 15 fuera de rango
            "starch_flow": [100, 200, -50, 300],          # -50 fuera de rango
        })
    
    def test_validator_import(self):
        """Verificar que el validador se importa correctamente."""
        from core.validation.validator import PhysicalValidator
        assert PhysicalValidator is not None
    
    def test_validate_filters_invalid_rows(self, sample_df):
        """Verificar que se filtran filas inválidas."""
        from core.validation.validator import PhysicalValidator
        
        validator = PhysicalValidator()
        df_clean = validator.validate(sample_df)
        
        # Solo la fila 3 (índice 3) debería sobrevivir
        # - Fila 0: válida
        # - Fila 1: válida
        # - Fila 2: inválida (_iron_feed=150, ore_pulp_ph=15, starch_flow=-50)
        # - Fila 3: válida (NaN se permite)
        assert len(df_clean) < len(sample_df)
    
    def test_validate_preserves_nan(self, sample_df):
        """Verificar que NaN se preservan (no se filtran)."""
        from core.validation.validator import PhysicalValidator
        
        validator = PhysicalValidator()
        df_clean = validator.validate(sample_df)
        
        # La fila con NaN debería estar presente
        assert df_clean["_iron_feed"].isna().any()
    
    def test_validate_empty_df(self):
        """Verificar manejo de DataFrame vacío."""
        from core.validation.validator import PhysicalValidator
        
        validator = PhysicalValidator()
        df_empty = pd.DataFrame()
        df_result = validator.validate(df_empty)
        
        assert df_result.empty
    
    def test_validation_stats(self, sample_df):
        """Verificar que se generan estadísticas."""
        from core.validation.validator import PhysicalValidator
        
        validator = PhysicalValidator()
        validator.validate(sample_df)
        
        assert validator.last_stats is not None
        assert validator.last_stats.filas_entrada == 4


class TestPreprocessor:
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
        from core.preprocessor import Preprocessor
        assert Preprocessor is not None
    
    def test_replaces_infinites(self, sample_df):
        """Verificar que infinitos se reemplazan."""
        from core.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        df_clean = preprocessor.clean_stream(sample_df)
        
        # No debería haber infinitos
        assert not np.isinf(df_clean.values).any()
    
    def test_imputes_nulls_ffill(self, sample_df):
        """Verificar imputación forward fill."""
        from core.preprocessor import Preprocessor
        
        preprocessor = Preprocessor(estrategia_nulos="ffill")
        df_clean = preprocessor.clean_stream(sample_df)
        
        # No debería haber NaN (después de ffill + fillna final)
        assert not df_clean.isna().any().any()
    
    def test_imputes_nulls_interpolate(self, sample_df):
        """Verificar imputación por interpolación."""
        from core.preprocessor import Preprocessor
        
        preprocessor = Preprocessor(estrategia_nulos="interpolate")
        df_clean = preprocessor.clean_stream(sample_df)
        
        assert not df_clean.isna().any().any()
    
    def test_invalid_strategy_raises(self):
        """Verificar que estrategia inválida lanza error."""
        from core.preprocessor import Preprocessor
        
        with pytest.raises(ValueError):
            Preprocessor(estrategia_nulos="invalid_strategy")
    
    def test_preserves_non_numeric_columns(self):
        """Verificar que columnas no numéricas no se modifican."""
        from core.preprocessor import Preprocessor
        
        df = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0],
            "text": ["a", "b", "c"]
        })
        
        preprocessor = Preprocessor()
        df_clean = preprocessor.clean_stream(df)
        
        assert df_clean["text"].tolist() == ["a", "b", "c"]
    
    def test_cleaning_stats(self, sample_df):
        """Verificar que se generan estadísticas de limpieza."""
        from core.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        preprocessor.clean_stream(sample_df)
        
        assert preprocessor.last_stats is not None
        assert preprocessor.last_stats.infinitos_reemplazados > 0


@pytest.mark.adapter
class TestUniversalAdapterDeterminism:
    """
    [P0b ROADMAP][REGRESIÓN] Test contra el bug real encontrado en GeoMet cobre:
    UniversalAdapter._apply_feature_selection() armaba `keep_cols` como un
    `set()` de Python y hacía `df[list(keep_cols)]`. El orden de iteración de
    un set de strings NO es determinista entre procesos (hash randomization,
    PYTHONHASHSEED aleatorio por defecto desde Python 3.3). Esto hacía que
    remove_correlated_features() (aguas abajo, en SoftSensorGP) tirara una
    feature distinta entre corridas ("Si ppm" vs. "Fe ppm" en GeoMet), dando
    un R² distinto (0.315 vs 0.167-0.177) para el MISMO dataset y config.

    Fix: preservar el orden original de df.columns al construir keep_cols
    (ver core/adapters/universal_adapter.py, comentario "[P0b ROADMAP][FIX]").

    Estos tests corren el adapter en subprocesos con PYTHONHASHSEED distinto
    a propósito — un test in-process (misma corrida de pytest) NO habría
    detectado el bug original, porque el hash seed es fijo dentro de un
    mismo proceso.
    """

    @pytest.fixture
    def config_with_correlated_features(self):
        """
        Dataset sintético no-temporal con dos columnas casi perfectamente
        correlacionadas (>0.98) — replica la situación de GeoMet cobre
        ("Si ppm" vs "Fe ppm") donde remove_correlated_features() debe
        elegir cuál tirar, y esa elección dependía del orden no-determinista
        de keep_cols.

        NOTA: UniversalAdapter resuelve config_path y DATA_DIR relativos al
        PAQUETE (core/adapters/../../config, .../data), no al cwd del test
        — por eso este fixture escribe (y limpia) directamente dentro de
        config/ y data/ del repo real, con nombres únicos para no chocar
        con dataset_config.json.

        Returns:
            str: nombre del archivo de config JSON (relativo a config/).
        """
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        config_dir = project_root / "config"
        data_dir.mkdir(exist_ok=True)

        n = 60
        rng = np.random.RandomState(42)
        base = rng.uniform(0, 10, n)

        df = pd.DataFrame({
            "sample_id": range(1, n + 1),
            "feature_x_ppm": base,
            "feature_y_ppm": base * 1.0001 + rng.normal(0, 1e-6, n),  # corr > 0.999
            "feature_z_ppm": rng.uniform(-5, 5, n),
            "target": 10.0 + 2.0 * base + rng.normal(0, 0.5, n),
        })

        csv_name = "_test_determinism_correlated_features.csv"
        csv_path = data_dir / csv_name
        df.to_csv(csv_path, index=False)

        config = {
            "dataset_name": "test_determinism",
            "files": {
                "filename": csv_name,
                "timestamp_column": "no_existe_es_no_temporal",
                "separator": ",",
            },
            "modeling": {"target_column": "target", "problem_type": "regression"},
            "feature_engineering": {
                "include_patterns": ["ppm", "sample_id"],
                "exclude_patterns": [],
                "forced_drop": [],
            },
        }
        config_name = "_test_determinism_config.json"
        config_path = config_dir / config_name
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        yield config_name

        # NOTA: unlink() puede fallar por permisos según el filesystem/mount
        # (ej. carpetas de proyecto montadas desde el host). No es crítico
        # dejar estos archivos huérfanos (nombres con prefijo "_test_" y
        # gitignored vía data/), así que no fallamos el test por esto.
        for p in (csv_path, config_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def test_column_order_deterministic_within_process(
        self, config_with_correlated_features
    ):
        """
        Llamar load_data() repetidamente en el MISMO proceso debe dar
        siempre el mismo orden de columnas (condición necesaria, no
        suficiente — el bug real solo se manifestaba entre procesos).
        """
        from core.adapters.universal_adapter import UniversalAdapter

        config_filename = config_with_correlated_features
        orders = []
        for _ in range(5):
            adapter = UniversalAdapter(config_filename=config_filename)
            df = adapter.load_data()
            orders.append(list(df.columns))

        assert all(o == orders[0] for o in orders), (
            "El orden de columnas de UniversalAdapter.load_data() varió "
            "entre llamadas en el mismo proceso."
        )

    def test_column_order_deterministic_across_processes(
        self, config_with_correlated_features
    ):
        """
        [Regresión del bug real] Corre el adapter en 5 subprocesos frescos,
        cada uno con PYTHONHASHSEED distinto (por defecto, aleatorio por
        proceso). Antes del fix, el orden de columnas —y por lo tanto qué
        feature correlacionada sobrevivía a remove_correlated_features()—
        variaba entre estas corridas. Con el fix, debe ser idéntico siempre.
        """
        from core.adapters.universal_adapter import UniversalAdapter

        config_filename = config_with_correlated_features
        project_root = str(Path(__file__).parent.parent)

        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {project_root!r})\n"
            "from core.adapters.universal_adapter import UniversalAdapter\n"
            f"adapter = UniversalAdapter(config_filename={config_filename!r})\n"
            "df = adapter.load_data()\n"
            "print(json.dumps(list(df.columns)))\n"
        )

        column_orders = []
        for i in range(5):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(i)  # forzar seeds DISTINTOS entre corridas
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"Subproceso falló (PYTHONHASHSEED={i}): {result.stderr}"
            )
            column_orders.append(json.loads(result.stdout.strip().splitlines()[-1]))

        assert all(order == column_orders[0] for order in column_orders), (
            "UniversalAdapter da un orden de columnas distinto según "
            "PYTHONHASHSEED — regresión del bug de no-determinismo de "
            f"GeoMet cobre (set() en _apply_feature_selection). Órdenes "
            f"observados: {column_orders}"
        )

    def test_correlated_feature_removed_is_stable(
        self, config_with_correlated_features
    ):
        """
        Consecuencia directa del bug: con orden de columnas determinista,
        SoftSensorGP.remove_correlated_features() debe eliminar SIEMPRE la
        misma de las dos features correlacionadas (feature_y_ppm, que viene
        después de feature_x_ppm en el CSV original), no una al azar.
        """
        from core.adapters.universal_adapter import UniversalAdapter
        from core.models.gp_model import SoftSensorGP

        config_filename = config_with_correlated_features
        project_root = Path(__file__).parent.parent

        survivors = []
        for _ in range(3):
            adapter = UniversalAdapter(config_filename=config_filename)
            df = adapter.load_data()
            tmp_csv = project_root / "data" / "_test_determinism_reload_tmp.csv"
            df.to_csv(tmp_csv, index=False)

            model = SoftSensorGP(
                target_col="target", parse_dates=False, add_lag_features=False,
                add_diff_features=False,
            )
            model.load_data(filepath=str(tmp_csv))
            survivors.append(sorted(model.feature_names))
            try:
                tmp_csv.unlink()
            except OSError:
                pass

        assert all(s == survivors[0] for s in survivors), (
            f"El set de features sobrevivientes tras remove_correlated_features() "
            f"varió entre corridas: {survivors}"
        )


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
        from core.adapters import CSVAdapter
        
        adapter = CSVAdapter(str(CONFIG.DATA_RAW_PATH))
        gen = adapter.stream()
        chunk = next(gen)
        
        assert not chunk.empty
        assert len(chunk.columns) > 0
    
    def test_full_pipeline_smoke(self, check_data_exists, tmp_path):
        """Smoke test del pipeline completo."""
        pass  # Implementar si es necesario


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
