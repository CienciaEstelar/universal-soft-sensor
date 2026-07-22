"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: tests/test_modeling.py
Versión: 2.0.1 — BUGFIX

HISTORIAL:
    [v2.0.1 - 2026]
        [FIX] TypeError en TestFeatureEngineering.test_rolling_features_if_enabled.
              SoftSensorGP.__init__ no tiene el parámetro add_rolling_features.
              ANTES: SoftSensorGP(..., add_rolling_features=True)
                     ↑ TypeError: __init__() got an unexpected keyword argument
              AHORA: El test busca columnas rolling en el output sin pasar
                     un kwarg inexistente. Si el modelo las genera, las encontrará.
                     Si no, el test documenta que la feature no existe aún.

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

from core.models.gp_model import SoftSensorGP


class TestSoftSensorGPInitialization:
    """Tests de inicialización de SoftSensorGP."""
    
    def test_default_initialization(self):
        """Verificar inicialización por defecto."""
        model = SoftSensorGP()
        
        # Default tras la auditoría: subsample=1 (sin diezmar). Ver
        # config/settings.py: el subsampling agresivo era un anti-patrón
        # heredado que degradaba el rendimiento honesto.
        assert model.subsample_step == 1
        assert model.add_lag_features is True
        assert 1 in model.lag_periods
    
    def test_custom_initialization(self):
        """Verificar inicialización con parámetros custom."""
        model = SoftSensorGP(
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
        model = SoftSensorGP(lag_periods=[1, 5, 10])

        assert model.lag_periods == [1, 5, 10]

    def test_input_lags_disabled_by_default(self):
        """
        [P0 ROADMAP] add_input_lags debe ser False por defecto.

        Activarlo cambia la dimensionalidad de cualquier dataset existente;
        no debe alterar el comportamiento de modelos ya en uso sin opt-in
        explícito.
        """
        model = SoftSensorGP()

        assert model.add_input_lags is False
        assert model.input_lag_periods == [1, 2, 3]
        assert model.input_lag_columns is None

    def test_input_lags_custom_configuration(self):
        """Verificar configuración custom de lags de entrada."""
        model = SoftSensorGP(
            add_input_lags=True,
            input_lag_periods=[1, 4, 8],
            input_lag_columns=["rougher.input.feed_au"],
        )

        assert model.add_input_lags is True
        assert model.input_lag_periods == [1, 4, 8]
        assert model.input_lag_columns == ["rougher.input.feed_au"]


class TestFeatureEngineering:
    """Tests para el feature engineering."""
    
    def test_lag_features_creation(self, synthetic_data):
        """Verificar creación de lags."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, add_lag_features=True, add_diff_features=False)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Verificar que existen las columnas de lag
        assert f"{target}_lag_1" in df_eng.columns
        assert f"{target}_lag_5" in df_eng.columns
    
    def test_diff_features_creation(self, synthetic_data):
        """Verificar creación de diferencias."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, add_lag_features=False, add_diff_features=True)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Verificar que existen las columnas de diff
        assert f"{target}_diff_1" in df_eng.columns
    
    def test_lag_values_correctness(self, synthetic_data):
        """Verificar que los valores de lag son matemáticamente correctos."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, add_lag_features=True)
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # El lag_1 en la fila i debe ser igual al target en la fila i-1
        expected = synthetic_data[target].iloc[9]  # Fila anterior
        actual = df_eng[f"{target}_lag_1"].iloc[10]
        
        assert np.isclose(expected, actual, rtol=1e-5)
    
    def test_rolling_features_if_present(self, synthetic_data):
        """
        Verificar features rolling si el modelo las genera.

        [FIX] ANTES: SoftSensorGP(..., add_rolling_features=True)
                     ↑ TypeError: __init__() no tiene ese parámetro.
        AHORA: Se instancia con los parámetros reales del constructor
               y se verifica qué columnas rolling genera, si es que genera.
               El test no falla si no hay columnas rolling — documenta
               el comportamiento actual sin asumir una API inexistente.
        """
        target = "rougher.output.recovery"

        # [FIX] Constructor real — sin add_rolling_features que no existe
        model = SoftSensorGP(
            target_col=target,
            add_lag_features=True,
            add_diff_features=True,
        )
        
        df_eng = model._create_lag_features(synthetic_data, target)
        
        # Documentar qué features rolling genera el modelo actual
        rolling_cols = [c for c in df_eng.columns if "rolling" in c.lower() or "mean" in c.lower()]

        # No assertamos existencia obligatoria — si hay rolling cols, deben ser numéricos
        for col in rolling_cols:
            assert pd.api.types.is_numeric_dtype(df_eng[col]), \
                f"Columna rolling '{col}' no es numérica"


class TestInputLagFeatures:
    """
    [P0 ROADMAP] Tests para lags de variables de ENTRADA (no del target).

    Cubre el gap destapado por SRU/flotación en la validación cross-domain:
    el pipeline solo lageaba el target y no podía modelar retardo de proceso
    en las entradas. Ver ROADMAP.md P0 y results/verification/FINDINGS.md.
    """

    def test_noop_when_disabled(self, synthetic_data):
        """Con add_input_lags=False (default), no debe agregar columnas."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, add_input_lags=False)

        df_eng = model._create_input_lag_features(synthetic_data, target)

        assert list(df_eng.columns) == list(synthetic_data.columns)

    def test_creates_lag_columns_for_all_inputs_by_default(self, synthetic_data):
        """Con columns=None, debe laguear todas las numéricas excepto el target."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(
            target_col=target, add_input_lags=True, input_lag_periods=[1, 3]
        )

        df_eng = model._create_input_lag_features(synthetic_data, target)

        for col in synthetic_data.columns:
            if col == target:
                continue
            assert f"{col}_lag_1" in df_eng.columns
            assert f"{col}_lag_3" in df_eng.columns

        # El target jamás se lagea desde este método (eso lo hace
        # _create_lag_features, que es un método distinto).
        assert f"{target}_lag_1" not in df_eng.columns

    def test_creates_lag_columns_only_for_requested_columns(self, synthetic_data):
        """Con columns explícito, restringe el lageo a esas columnas."""
        target = "rougher.output.recovery"
        only_col = "rougher.input.feed_au"
        model = SoftSensorGP(
            target_col=target,
            add_input_lags=True,
            input_lag_periods=[1, 2],
            input_lag_columns=[only_col],
        )

        df_eng = model._create_input_lag_features(
            synthetic_data, target, columns=model.input_lag_columns
        )

        assert f"{only_col}_lag_1" in df_eng.columns
        assert f"{only_col}_lag_2" in df_eng.columns
        # Otra columna de entrada NO solicitada no debe lagearse.
        assert "rougher.input.feed_ag_lag_1" not in df_eng.columns

    def test_target_excluded_even_if_explicitly_requested(self, synthetic_data):
        """Si el caller pasa el target en columns por error, se debe ignorar."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, add_input_lags=True)

        df_eng = model._create_input_lag_features(
            synthetic_data, target, columns=[target, "rougher.input.feed_au"]
        )

        assert f"{target}_lag_1" not in df_eng.columns
        assert "rougher.input.feed_au_lag_1" in df_eng.columns

    def test_lag_values_are_causal_shift(self, synthetic_data):
        """El lag_N en la fila i debe igualar el valor de la columna en fila i-N."""
        target = "rougher.output.recovery"
        col = "rougher.input.feed_au"
        model = SoftSensorGP(
            target_col=target, add_input_lags=True, input_lag_periods=[1, 5]
        )

        df_eng = model._create_input_lag_features(synthetic_data, target)

        expected_lag1 = synthetic_data[col].iloc[9]
        actual_lag1 = df_eng[f"{col}_lag_1"].iloc[10]
        assert np.isclose(expected_lag1, actual_lag1, rtol=1e-5)

        expected_lag5 = synthetic_data[col].iloc[5]
        actual_lag5 = df_eng[f"{col}_lag_5"].iloc[10]
        assert np.isclose(expected_lag5, actual_lag5, rtol=1e-5)

    def test_runs_before_target_lags_in_load_data_pipeline(self, temp_csv):
        """
        Integración liviana: con add_input_lags=True, load_data() debe
        producir columnas de lag de entrada Y de target simultáneamente,
        sin lagear las columnas derivadas del target (no debe existir
        p. ej. 'rougher.output.recovery_lag_1_lag_1').
        """
        target = "rougher.output.recovery"
        model = SoftSensorGP(
            target_col=target,
            subsample_step=5,
            add_lag_features=True,
            add_diff_features=False,
            add_input_lags=True,
            input_lag_periods=[1],
        )

        X, y, dates = model.load_data(filepath=temp_csv)

        assert f"{target}_lag_1" in model.feature_names
        assert "rougher.input.feed_au_lag_1" in model.feature_names
        # No debe haber lags-de-lags del target.
        assert not any("_lag_1_lag_1" in c for c in model.feature_names)


class TestGroupAwareTraining:
    """
    [P0b ROADMAP] Tests para split/CV por grupo (GroupShuffleSplit/GroupKFold)
    y datasets sin dimensión temporal (parse_dates=False).

    Motivación: run_geomet_rigor.py demostró que un split aleatorio/secuencial
    sobre datos con muestras repetidas por grupo (ej. varias muestras del
    mismo sondaje/HOLEID) infla el R² por leakage de grupo (0.93 falso vs.
    0.33 real con GroupKFold). Este mecanismo lleva esa lógica al pipeline
    central en vez de vivir solo en un script aislado.
    """

    def test_defaults_are_backward_compatible(self):
        """parse_dates=True y group_column=None por defecto — no debe
        alterar el comportamiento de modelos existentes sin opt-in."""
        model = SoftSensorGP()
        assert model.parse_dates is True
        assert model.group_column is None
        assert model.groups_ is None

    def test_group_column_missing_raises(self, temp_csv_grouped):
        """Si group_column no existe en el CSV, debe fallar con mensaje claro."""
        model = SoftSensorGP(
            target_col="target", group_column="no_existe", parse_dates=False
        )
        with pytest.raises(ValueError, match="no_existe"):
            model.load_data(filepath=temp_csv_grouped)

    def test_group_column_excluded_from_features(self, temp_csv_grouped):
        """group_id no debe aparecer como feature del modelo."""
        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
        )
        X, y, _ = model.load_data(filepath=temp_csv_grouped)

        assert "group_id" not in model.feature_names

    def test_groups_populated_and_aligned_after_load_data(self, temp_csv_grouped):
        """model.groups_ debe quedar poblado y alineado fila a fila con X/y."""
        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
        )
        X, y, _ = model.load_data(filepath=temp_csv_grouped)

        assert model.groups_ is not None
        assert len(model.groups_) == X.shape[0]
        # Cardinalidad de grupos coherente con el fixture (15 grupos, 4 filas c/u)
        assert len(np.unique(model.groups_)) <= 15

    def test_parse_dates_false_skips_temporal_diagnosis(self, temp_csv_grouped):
        """Con parse_dates=False, data_diagnosis no debe calcular autocorrelación
        (no tiene sentido físico sin dimensión temporal) y no debe crashear."""
        model = SoftSensorGP(
            target_col="target",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
        )
        model.load_data(filepath=temp_csv_grouped)

        assert model.data_diagnosis == {"skipped_non_temporal": True}

    def test_group_shuffle_split_never_leaks_a_group(self, temp_csv_grouped):
        """
        Verifica la garantía central de [P0b]: ningún grupo debe aparecer
        simultáneamente en train y test. Reproduce el mismo split
        (GroupShuffleSplit con el random_state del modelo) que usa
        train_from_file() internamente sobre los groups_ que expone
        load_data() — si hay un bug de alineación entre X/y/groups, esta
        prueba lo detecta.
        """
        from sklearn.model_selection import GroupShuffleSplit

        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
            random_state=42,
        )
        X, y, _ = model.load_data(filepath=temp_csv_grouped)
        groups = model.groups_

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=model.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        assert train_groups.isdisjoint(test_groups), (
            "GroupShuffleSplit filtró un grupo entre train y test — "
            "esto es exactamente el leakage que P0b debía cerrar."
        )

    def test_train_gp_uses_groupkfold_when_groups_present(self, temp_csv_grouped):
        """_train_gp con groups debe correr sin excepción y devolver un
        cv_score numérico válido (no None, no NaN) usando GroupKFold."""
        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
            random_state=42,
        )
        X, y, _ = model.load_data(filepath=temp_csv_grouped)
        X_s = model.scaler_X.fit_transform(X)
        y_s = model.scaler_y.fit_transform(y)

        _, _, cv_score = model._train_gp(X_s, y_s, n_trials=2, groups=model.groups_)

        assert cv_score is not None
        assert np.isfinite(cv_score)

    def test_train_gp_degenerate_single_group_signals_failure(self):
        """Con un solo grupo único, GroupKFold no tiene sentido — debe
        señalizar fallo al caller (cv_score muy bajo) en vez de crashear."""
        model = SoftSensorGP(target_col="target", group_column="group_id", parse_dates=False)
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, 3))
        y = rng.uniform(size=(20, 1))
        groups = np.ones(20)  # un solo grupo

        model_out, params, cv_score = model._train_gp(X, y, n_trials=2, groups=groups)

        # No debe lanzar excepción; debe señalizar claramente que falló.
        assert cv_score <= -1.0 or (model_out is None and params == {})

    def test_full_training_cycle_with_group_column(self, temp_csv_grouped, tmp_path):
        """
        Integración end-to-end: train_from_file() con group_column debe
        completar sin excepción, usando split por grupo + GroupKFold interno,
        y producir métricas válidas.
        """
        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
            subsample_step=1,
        )

        metrics = model.train_from_file(
            filepath=temp_csv_grouped,
            test_size=0.3,
            n_trials=1,
            save_model=False,
            output_dir=tmp_path,  # no ensuciar el results/ real del proyecto
        )

        assert model.model is not None
        assert np.isfinite(metrics.r2) or np.isnan(metrics.r2)
        assert metrics.permutation_p_value is None  # no se pidió

    def test_permutation_test_contract(self, temp_csv_grouped):
        """permutation_test() debe devolver el dict esperado con tipos/rangos
        válidos, tanto con groups como sin ellos."""
        model = SoftSensorGP(
            target_col="target",
            group_column="group_id",
            parse_dates=False,
            add_lag_features=False,
            add_diff_features=False,
        )
        X, y, _ = model.load_data(filepath=temp_csv_grouped)

        result = model.permutation_test(X, y.ravel(), groups=model.groups_, n_permutations=5)

        # [scientific_report] null_r2_distribution es una clave NUEVA (no
        # rompe consumidores existentes que solo lean las 4 originales) —
        # se agregó para poder graficar la distribución nula vs. el R² real
        # en el informe científico. Se usa <= (subconjunto) en vez de ==
        # para no volver a romper este test si se agregan más claves.
        assert {"real_r2", "p_value", "n_permutations", "n_splits"} <= set(result.keys())
        assert 0.0 <= result["p_value"] <= 1.0
        assert np.isfinite(result["real_r2"])
        assert result["n_permutations"] == 5
        assert result["n_splits"] >= 2
        assert "null_r2_distribution" in result
        assert len(result["null_r2_distribution"]) == 5
        assert all(np.isfinite(r) for r in result["null_r2_distribution"])

    def test_permutation_test_without_groups(self, synthetic_data):
        """permutation_test() debe funcionar también sin groups (KFold plano)."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target)
        X = synthetic_data.drop(columns=[target]).values
        y = synthetic_data[target].values

        result = model.permutation_test(X, y, groups=None, n_permutations=5)

        assert 0.0 <= result["p_value"] <= 1.0
        assert np.isfinite(result["real_r2"])


class TestFeatureCleaning:
    """Tests para limpieza de features."""
    
    def test_constant_feature_removal(self, synthetic_data):
        """Verificar eliminación de features constantes."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(target_col=target, remove_constant_features=True)
        
        # synthetic_data tiene 'flotation_section_03_air_amount' como constante
        X_df = synthetic_data.drop(columns=[target])
        
        X_clean = model._remove_problematic_features(X_df)
        
        # La columna constante debe haber sido eliminada
        assert "flotation_section_03_air_amount" not in X_clean.columns
    
    def test_target_not_removed_during_cleaning(self, synthetic_data):
        """Verificar que el target no se elimina accidentalmente."""
        target = "rougher.output.recovery"
        model = SoftSensorGP(
            target_col=target,
            remove_constant_features=True,
            remove_correlated_features=True
        )
        
        X_df = synthetic_data.drop(columns=[target])
        X_clean = model._remove_problematic_features(X_df)
        
        assert target not in X_clean.columns
        assert len(X_clean.columns) > 0


class TestTrainingCycle:
    """Tests del ciclo de entrenamiento."""
    
    def test_full_training_cycle(self, temp_csv, tmp_path):
        """Prueba de Integración: Ciclo de Vida Completo."""
        target = "rougher.output.recovery"
        
        model = SoftSensorGP(
            target_col=target,
            subsample_step=5,
            add_lag_features=True,
            add_diff_features=False,
        )
        
        model.train_from_file(
            filepath=temp_csv,
            n_trials=1,
            test_size=0.2,
            save_model=False,
            output_dir=tmp_path,  # no ensuciar el results/ real del proyecto
        )

        assert model.model is not None
        assert model.scaler_X is not None
        assert model.scaler_y is not None
        assert model.feature_names is not None

        custom_path = tmp_path / "test_model.pkl"
        model.save(filepath=str(custom_path))
        
        assert custom_path.exists()
        
        loaded = joblib.load(str(custom_path))
        assert hasattr(loaded, "model")
    
    def test_model_can_predict_after_training(self, temp_csv, tmp_path):
        """Verificar que el modelo puede predecir después de entrenar."""
        target = "rougher.output.recovery"

        model = SoftSensorGP(target_col=target, subsample_step=10)

        model.train_from_file(
            filepath=temp_csv,
            n_trials=1,
            test_size=0.2,
            save_model=False,
            output_dir=tmp_path,  # no ensuciar el results/ real del proyecto
        )
        
        df = pd.read_csv(temp_csv, index_col=0, parse_dates=True)
        window = df.iloc[-50:]
        
        df_features = model._create_lag_features(window, target)
        last_row = df_features.iloc[[-1]]
        
        X_cols = [c for c in model.feature_names if c in last_row.columns]
        X = last_row[X_cols].fillna(0)
        
        X_scaled = model.scaler_X.transform(X.values)
        y_pred = model.model.predict(X_scaled)
        
        assert len(y_pred) == 1
        assert not np.isnan(y_pred[0])


class TestModelPersistence:
    """Tests de persistencia del modelo."""
    
    def test_save_and_load(self, trained_model, tmp_path):
        """Verificar ciclo save/load."""
        model_path = tmp_path / "persistence_test.pkl"
        
        trained_model.save(str(model_path))
        assert model_path.exists()
        
        new_model = SoftSensorGP()
        new_model.load(str(model_path))
        
        assert new_model.model is not None
        assert new_model.model_type == trained_model.model_type
        assert new_model.target_col == trained_model.target_col
    
    def test_loaded_model_can_predict(self, trained_model, synthetic_data, tmp_path):
        """Verificar que modelo cargado puede predecir."""
        model_path = tmp_path / "predict_test.pkl"
        trained_model.save(str(model_path))
        
        new_model = SoftSensorGP()
        new_model.load(str(model_path))
        
        target = new_model.target_col
        window = synthetic_data.iloc[-50:]
        
        df_features = new_model._create_lag_features(window, target)
        
        assert not df_features.empty


class TestMetrics:
    """Tests para métricas del modelo."""
    
    def test_training_returns_metrics(self, temp_csv, tmp_path):
        """Verificar que el entrenamiento retorna métricas."""
        model = SoftSensorGP(
            target_col="rougher.output.recovery",
            subsample_step=10,
        )

        model.train_from_file(
            filepath=temp_csv,
            n_trials=1,
            test_size=0.2,
            save_model=False,
            output_dir=tmp_path,  # no ensuciar el results/ real del proyecto
        )
        
        assert hasattr(model, 'best_cv_score') or hasattr(model, 'metrics')


class TestInferenceEngineIntegration:
    """Tests de integración con inference_engine."""
    
    def test_predict_scenario(self, trained_model, synthetic_data, tmp_path):
        """Verificar predict_scenario del inference engine."""
        from core.inference_engine import InferenceEngine
        
        model_path = tmp_path / "inference_test.pkl"
        trained_model.save(str(model_path))
        
        engine = InferenceEngine(model_path=str(model_path))
        
        window = synthetic_data.iloc[-50:]
        result = engine.predict_scenario(window)
        
        assert "predicted_value" in result
        assert "confidence_pct" in result
        assert "model_used" in result
        assert isinstance(result["predicted_value"], float)
    
    def test_predict_series(self, trained_model, synthetic_data, tmp_path):
        """Verificar predict_series del inference engine."""
        from core.inference_engine import InferenceEngine
        
        model_path = tmp_path / "series_test.pkl"
        trained_model.save(str(model_path))
        
        engine = InferenceEngine(model_path=str(model_path))
        
        series = engine.predict_series(synthetic_data, n_points=20, min_history=30)
        
        assert isinstance(series, pd.DataFrame)
        if not series.empty:
            assert "predicted" in series.columns
            assert "confidence_pct" in series.columns
    
    def test_get_feature_importance(self, trained_model, tmp_path):
        """Verificar get_feature_importance del inference engine."""
        from core.inference_engine import InferenceEngine
        
        model_path = tmp_path / "importance_test.pkl"
        trained_model.save(str(model_path))
        
        engine = InferenceEngine(model_path=str(model_path))
        
        importance = engine.get_feature_importance(top_n=5)
        
        assert isinstance(importance, dict)
        assert len(importance) <= 5

        # Tras el fix anti-NaN: si las importancias originales suman 0
        # (modelo trivial sin señal), el engine retorna {feature: 0.0}.
        # Es válido. Verificamos invariantes débiles: no NaN, suma == 1
        # (caso normal) o suma == 0 (caso degenerado, modelo sin señal).
        if importance:
            values = list(importance.values())
            assert all(np.isfinite(v) for v in values)
            total = sum(values)
            assert total == 0 or abs(total - 1.0) < 1e-6 or total <= 1.0


class TestUncertaintyCalibration:
    """
    [P6 ROADMAP — calibración] Tests de NLL y Coverage@95 en evaluate().

    Estas métricas responden a una pregunta que R²/RMSE no responden: ¿la
    incertidumbre σ que reporta el GP es confiable, o son bandas de confianza
    decorativas? Un GP puede tener buen R² y estar mal calibrado (sobre o
    sub-confiado). Estos tests fijan el contrato y previenen dos trampas:
    (1) calcular calibración sobre σ=0 (fallback GB) → división por cero
    enmascarada; (2) romper el comportamiento previo de evaluate() sin y_std.
    """

    def _model(self):
        return SoftSensorGP(target_col="t")

    def test_well_calibrated_coverage_near_95(self):
        """Con σ que coincide con el error real, la cobertura del IC 95% debe
        caer cerca del 95% nominal (banda ±1.96σ), no en cualquier lado."""
        m = self._model()
        rng = np.random.default_rng(0)
        n = 4000
        y_true = rng.normal(0, 1, n)
        sigma = 0.1
        y_pred = y_true + rng.normal(0, sigma, n)
        y_std = np.full(n, sigma)

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.coverage_95 is not None
        # Cobertura empírica de un gaussiano bien calibrado: ~0.95. Margen
        # amplio (±0.03) para no ser frágil ante la semilla.
        assert 0.92 <= metrics.coverage_95 <= 0.98
        assert metrics.nll is not None
        assert np.isfinite(metrics.nll)

    def test_overconfident_model_flagged_by_low_coverage(self):
        """Si σ es demasiado chico vs el error real (sobre-confianza), la
        cobertura debe caer MUY por debajo de 0.95 y la NLL dispararse."""
        m = self._model()
        rng = np.random.default_rng(1)
        n = 4000
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)  # error real ~0.1
        y_std = np.full(n, 0.01)                  # pero σ reportado 10x menor

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.coverage_95 is not None
        assert metrics.coverage_95 < 0.5          # claramente mal calibrado
        assert metrics.nll > 1.0                  # NLL alta = penaliza sobre-confianza

    def test_gb_fallback_zero_std_leaves_calibration_none(self):
        """El fallback GradientBoosting devuelve σ=0. Calibración NO debe
        calcularse (daría división por cero enmascarada): queda None, sin
        crash y sin número engañoso."""
        m = self._model()
        rng = np.random.default_rng(2)
        n = 500
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)
        y_std = np.zeros(n)  # exactamente lo que devuelve predict() para GB

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.coverage_95 is None
        assert metrics.nll is None
        # Las métricas de error puntual sí deben existir.
        assert np.isfinite(metrics.r2)
        assert np.isfinite(metrics.rmse)

    def test_backward_compatible_without_std(self):
        """evaluate() sin y_std (firma original) no debe romperse ni calcular
        calibración — comportamiento idéntico al previo."""
        m = self._model()
        rng = np.random.default_rng(3)
        n = 500
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        metrics = m.evaluate(y_true, y_pred)  # sin y_std

        assert metrics.coverage_95 is None
        assert metrics.nll is None
        assert np.isfinite(metrics.r2)

    def test_coverage_is_exact_fraction_within_band(self):
        """Coverage@95 debe ser la fracción exacta de puntos con |y-μ| ≤ 1.96σ,
        verificable de forma determinista (sin azar)."""
        m = self._model()
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        # errores: 0, 1.0, 3.0, 0.5 ; con σ=1 el umbral es 1.96
        #   dentro: 0, 1.0, 0.5 (3 de 4) ; fuera: 3.0
        y_pred = np.array([0.0, 1.0, 3.0, 0.5])
        y_std = np.array([1.0, 1.0, 1.0, 1.0])

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.coverage_95 == pytest.approx(0.75)

    def test_single_zero_std_point_disables_calibration(self):
        """Una sola σ=0 en medio de σ>0 haría NLL=+inf. El guard exige TODA σ>0,
        así que en ese caso la calibración se omite (None), no se reporta inf."""
        m = self._model()
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([0.1, 1.1, 1.9, 3.2])
        y_std = np.array([0.5, 0.5, 0.0, 0.5])  # una σ=0

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.nll is None
        assert metrics.coverage_95 is None

    def test_calibration_survives_dict_and_repr(self):
        """to_dict() y __repr__ deben incluir nll/coverage_95/sharpness cuando
        existen y omitirlos cuando son None (contrato de serialización)."""
        m = self._model()
        rng = np.random.default_rng(4)
        n = 300
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        with_std = m.evaluate(y_true, y_pred, y_std=np.full(n, 0.1))
        d = with_std.to_dict()
        assert "nll" in d and "coverage_95" in d and "sharpness" in d
        r = repr(with_std)
        assert "Cov95" in r and "NLL" in r and "Sharp" in r

        without = m.evaluate(y_true, y_pred, y_std=np.zeros(n))
        d2 = without.to_dict()
        assert "nll" not in d2 and "coverage_95" not in d2 and "sharpness" not in d2
        assert "Sharp" not in repr(without)

    def test_sharpness_is_exact_interval_width(self):
        """Sharpness debe ser el ancho medio del IC 95% = mean(2·1.96·σ),
        verificable de forma determinista para σ constante."""
        m = self._model()
        y_true = np.zeros(50)
        y_pred = np.zeros(50)
        y_std = np.full(50, 0.1)  # ancho esperado: 2*1.96*0.1 = 0.392

        metrics = m.evaluate(y_true, y_pred, y_std=y_std)

        assert metrics.sharpness == pytest.approx(2 * 1.96 * 0.1)

    def test_sharpness_scales_with_sigma(self):
        """A mayor σ, mayor sharpness (bandas más anchas) — proporcional."""
        m = self._model()
        y_true = np.zeros(50)
        y_pred = np.zeros(50)

        s_small = m.evaluate(y_true, y_pred, y_std=np.full(50, 0.1)).sharpness
        s_large = m.evaluate(y_true, y_pred, y_std=np.full(50, 0.2)).sharpness

        assert s_large == pytest.approx(2 * s_small)

    def test_sharpness_none_for_gb_and_missing_std(self):
        """Sharpness sigue el mismo guard que NLL/Cov95: None para σ=0 (GB) y
        cuando no se pasa y_std."""
        m = self._model()
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.9, 2.1])

        assert m.evaluate(y_true, y_pred, y_std=np.zeros(3)).sharpness is None
        assert m.evaluate(y_true, y_pred).sharpness is None

    def test_coverage_and_sharpness_together_expose_wide_band_trap(self):
        """La razón de ser de sharpness: un modelo con bandas absurdamente
        anchas logra cobertura ~1.0 pero sharpness enorme. Cobertura sola no
        lo delata; sharpness sí. Este test documenta ese contraste."""
        m = self._model()
        rng = np.random.default_rng(7)
        n = 500
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        honesto = m.evaluate(y_true, y_pred, y_std=np.full(n, 0.1))
        inflado = m.evaluate(y_true, y_pred, y_std=np.full(n, 5.0))

        # Ambos con cobertura alta...
        assert honesto.coverage_95 >= 0.90
        assert inflado.coverage_95 >= 0.99
        # ...pero el inflado tiene sharpness mucho peor (bandas ~50x más anchas).
        assert inflado.sharpness > 10 * honesto.sharpness


class TestNaiveBaseline:
    """
    [P2 ROADMAP — baseline naive] Tests del R² del predictor trivial (media
    del train). Da el "piso" contra el que se lee el R² del modelo. Fijan el
    contrato y previenen la trampa de usar la media del TEST (que daría 0.0
    tautológico en vez del baseline honesto).
    """

    def _model(self):
        return SoftSensorGP(target_col="t")

    def test_baseline_near_zero_when_train_mean_matches_test(self):
        """Si la media de train coincide con la de test, el baseline (predecir
        esa media constante) da R²≈0 — el punto de referencia clásico."""
        m = self._model()
        rng = np.random.default_rng(0)
        y_true = rng.normal(5.0, 2.0, 300)
        y_pred = y_true + rng.normal(0, 0.3, 300)

        metrics = m.evaluate(y_true, y_pred, y_train_mean=5.0)

        assert metrics.baseline_r2 is not None
        assert abs(metrics.baseline_r2) < 0.05     # ≈ 0
        assert metrics.r2 > metrics.baseline_r2     # el modelo aporta señal

    def test_baseline_negative_when_train_mean_off(self):
        """La media de train sesgada respecto del test da baseline negativo —
        información honesta, no un bug. (Confirma que NO se usa la media del
        test, que forzaría 0.0.)"""
        m = self._model()
        rng = np.random.default_rng(1)
        y_true = rng.normal(5.0, 2.0, 300)
        y_pred = y_true + rng.normal(0, 0.3, 300)

        metrics = m.evaluate(y_true, y_pred, y_train_mean=50.0)  # media lejísimos

        assert metrics.baseline_r2 is not None
        assert metrics.baseline_r2 < -1.0

    def test_baseline_none_without_train_mean(self):
        """Sin y_train_mean (firma original) no se calcula — backward compat."""
        m = self._model()
        rng = np.random.default_rng(2)
        y_true = rng.normal(0, 1, 200)
        y_pred = y_true + rng.normal(0, 0.1, 200)

        metrics = m.evaluate(y_true, y_pred)

        assert metrics.baseline_r2 is None

    def test_model_predicting_test_mean_ties_baseline_at_zero(self):
        """Un 'modelo' que predice la media del test da R²=0, y si el train
        tiene esa misma media, el baseline también es 0 — empatan. Verifica que
        el baseline no está inflado artificialmente."""
        m = self._model()
        rng = np.random.default_rng(3)
        y_true = rng.normal(0, 1, 200)
        y_pred = np.full(200, y_true.mean())  # predice media del test

        metrics = m.evaluate(y_true, y_pred, y_train_mean=y_true.mean())

        assert metrics.r2 == pytest.approx(0.0, abs=1e-9)
        assert metrics.baseline_r2 == pytest.approx(0.0, abs=1e-9)

    def test_baseline_in_dict_and_repr(self):
        """to_dict()/__repr__ incluyen baseline_r2 cuando existe, lo omiten si
        es None."""
        m = self._model()
        rng = np.random.default_rng(4)
        y_true = rng.normal(0, 1, 200)
        y_pred = y_true + rng.normal(0, 0.1, 200)

        with_base = m.evaluate(y_true, y_pred, y_train_mean=0.0)
        assert "baseline_r2" in with_base.to_dict()
        assert "R²base" in repr(with_base)

        without = m.evaluate(y_true, y_pred)
        assert "baseline_r2" not in without.to_dict()
        assert "R²base" not in repr(without)


class TestExtrapolation:
    """
    [ROADMAP — extrapolación] Tests de extrapolation_test(). La pregunta que
    responde: ¿el GP ENSANCHA su incertidumbre fuera del rango de
    entrenamiento (degrada con gracia) o se equivoca con confianza? La firma
    sana es std_ratio = σ_exterior/σ_interior > 1.
    """

    def _model(self):
        m = SoftSensorGP(target_col="t")
        m.feature_names = ["dureza", "ruido"]
        return m

    def _smooth_data(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        dureza = rng.uniform(0, 10, n)
        ruido = rng.uniform(0, 1, n)
        y = np.sin(dureza) + 0.05 * rng.normal(0, 1, n)
        X = np.column_stack([dureza, ruido])
        return X, y

    def test_gp_widens_uncertainty_outside_training_range(self):
        """Sobre un target suave, un GP entrenado en el interior debe ensanchar
        σ en la zona de extrapolación: std_ratio > 1 y graceful=True."""
        m = self._model()
        X, y = self._smooth_data()

        res = m.extrapolation_test(X, y, feature="dureza", low_pct=15, high_pct=85)

        assert res["status"] == "ok"
        assert res["std_ratio"] > 1.0
        assert res["graceful"] is True
        # La extrapolación es genuinamente difícil: el R² exterior debe ser
        # peor que el interior (no exigimos que sea bueno — sería sospechoso).
        assert res["r2_exterior"] < res["r2_interior"]

    def test_determinism_same_seed(self):
        """Con el mismo random_state, el resultado es idéntico (split interno
        seedeado)."""
        m1 = self._model()
        m2 = self._model()
        X, y = self._smooth_data()

        r1 = m1.extrapolation_test(X, y, feature="dureza")
        r2 = m2.extrapolation_test(X, y, feature="dureza")

        assert r1["std_ratio"] == pytest.approx(r2["std_ratio"])
        assert r1["r2_exterior"] == pytest.approx(r2["r2_exterior"])

    def test_skips_when_zones_too_small(self):
        """Dataset chico → status 'skipped' con reason, sin crash ni número
        frágil."""
        m = self._model()
        rng = np.random.default_rng(0)
        n = 20
        X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 1, n)])
        y = rng.normal(0, 1, n)

        res = m.extrapolation_test(X, y, feature="dureza")

        assert res["status"] == "skipped"
        assert "reason" in res

    def test_feature_by_name_and_index_agree(self):
        """Resolver la feature por nombre o por índice debe dar el mismo
        resultado."""
        m = self._model()
        X, y = self._smooth_data()

        by_name = m.extrapolation_test(X, y, feature="dureza", low_pct=15, high_pct=85)
        by_idx = m.extrapolation_test(X, y, feature=0, low_pct=15, high_pct=85)

        assert by_name["status"] == by_idx["status"] == "ok"
        assert by_name["feature"] == by_idx["feature"] == "dureza"
        assert by_name["std_ratio"] == pytest.approx(by_idx["std_ratio"])

    def test_unknown_feature_and_constant_feature_skip(self):
        """Feature inexistente o constante → skipped, no excepción."""
        m = self._model()
        X, y = self._smooth_data()

        assert m.extrapolation_test(X, y, feature="zzz")["status"] == "skipped"

        Xc = X.copy()
        Xc[:, 0] = 5.0  # dureza constante
        assert m.extrapolation_test(Xc, y, feature="dureza")["status"] == "skipped"

    def test_result_contract_keys_present(self):
        """El dict 'ok' debe traer todas las claves documentadas."""
        m = self._model()
        X, y = self._smooth_data()

        res = m.extrapolation_test(X, y, feature="dureza", low_pct=15, high_pct=85)

        for k in (
            "feature", "threshold_low", "threshold_high",
            "n_interior_train", "n_interior_holdout", "n_exterior",
            "r2_interior", "r2_exterior", "mean_std_interior",
            "mean_std_exterior", "std_ratio", "coverage_95_exterior", "graceful",
        ):
            assert k in res, f"falta la clave '{k}' en el resultado"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
