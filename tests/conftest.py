"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: tests/conftest.py
Versión: 2.0.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Configuración global y 'Fixtures' para la suite de pruebas con Pytest.
    
    Aquí definimos los recursos compartidos (como datos falsos o rutas temporales)
    que se inyectan automáticamente en los tests.

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.0.0 - Enero 2026] COMPATIBILIDAD CON MÓDULOS v2.0
    -----------------------------------------------------
    - Fixtures actualizados para schema universal (pattern matching)
    - Nuevos fixtures para gold_recovery y ai4i2020 datasets
    - Fixture para MiningDataAdapter
    - Marcadores adicionales para tests de validación

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL PATH
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """
    Hook de configuración inicial de Pytest.
    Registramos marcadores personalizados para clasificar los tests.
    """
    config.addinivalue_line("markers", "integration: tests lentos con dataset real")
    config.addinivalue_line("markers", "validation: tests del módulo de validación")
    config.addinivalue_line("markers", "schema: tests del schema universal")
    config.addinivalue_line("markers", "adapter: tests de adaptadores de datos")


# -----------------------------------------------------------------------------
# FIXTURES: DATOS SINTÉTICOS
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_data():
    """
    Genera un DataFrame sintético que imita la estructura del dataset gold_recovery.
    
    ACTUALIZADO v2.0: Usa nombres de columnas compatibles con el schema universal
    (pattern matching por categoría física).
    
    Returns:
        pd.DataFrame: DataFrame con 500 filas, índice temporal y columnas
                      que matchean las categorías del schema v2.0.
    """
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1h")
    
    df = pd.DataFrame({
        "date": dates,
        # Target: Recovery (PERCENTAGE)
        "rougher.output.recovery": np.random.uniform(60.0, 90.0, 500),
        # Features que matchean categorías del schema v2.0
        "rougher.input.feed_au": np.random.uniform(1.0, 5.0, 500),        # PERCENTAGE
        "rougher.input.feed_ag": np.random.uniform(5.0, 15.0, 500),       # PERCENTAGE
        "primary_cleaner.input.feed_size": np.random.uniform(20, 80, 500), # PARTICLE_SIZE
        "primary_cleaner.state.floatbank8_a_level": np.random.uniform(300, 600, 500),  # LEVEL
        "flotation_section_01_air_amount": np.random.uniform(1000, 2000, 500),  # FLOW_RATE
        "flotation_section_02_air_amount": np.random.uniform(1000, 2000, 500),  # FLOW_RATE
        # Columna constante (para probar eliminación)
        "flotation_section_03_air_amount": np.full(500, 1500.0),
    })
    
    # Inyectar correlación para que el modelo pueda aprender
    # "Si sube el feed_au, sube el recovery"
    df["rougher.output.recovery"] = (
        50.0 + 
        df["rougher.input.feed_au"] * 5 + 
        np.random.normal(0, 2, 500)
    ).clip(0, 100)
    
    df.set_index("date", inplace=True)
    
    return df


@pytest.fixture(scope="session")
def synthetic_data_ai4i():
    """
    Genera un DataFrame sintético que imita el dataset AI4I2020 (mantenimiento predictivo).
    
    Útil para probar que el schema universal funciona con diferentes dominios.
    
    Returns:
        pd.DataFrame: DataFrame con estructura de AI4I2020.
    """
    n = 300
    
    df = pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"M{i}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]": np.random.uniform(295, 305, n),      # TEMPERATURE_KELVIN
        "Process temperature [K]": np.random.uniform(305, 315, n),  # TEMPERATURE_KELVIN
        "Rotational speed [rpm]": np.random.uniform(1200, 2000, n), # ROTATIONAL_SPEED
        "Torque [Nm]": np.random.uniform(20, 80, n),                # TORQUE
        "Tool wear [min]": np.random.uniform(0, 250, n),            # TOOL_WEAR
        "Machine failure": np.random.choice([0, 1], n, p=[0.95, 0.05]),  # BINARY
        "TWF": np.zeros(n),  # BINARY
        "HDF": np.zeros(n),  # BINARY
        "PWF": np.zeros(n),  # BINARY
        "OSF": np.zeros(n),  # BINARY
        "RNF": np.zeros(n),  # BINARY
    })
    
    return df


@pytest.fixture(scope="session")
def synthetic_data_with_invalids():
    """
    DataFrame con valores inválidos para probar el validador.
    
    Incluye:
    - Valores fuera de rango físico
    - NaN (que deben preservarse)
    - Infinitos (que deben rechazarse)
    """
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1h")
    
    df = pd.DataFrame({
        "date": dates,
        "rougher.input.feed_au": [2.5, 3.1, 150.0, np.nan, 4.2, 2.0, 3.0, -5.0, 4.0, 5.0],
        "rougher.output.recovery": [65.0, 70.0, 102.0, 68.0, 71.0, 66.0, 69.0, 72.0, -10.0, 75.0],
        "primary_cleaner.state.floatbank8_a_level": [450, 500, -600, 480, 520, 490, 510, 2000, 485, 495],
        "flotation_section_01_air_amount": [1200, 1300, -100, 1250, 1280, np.inf, 1220, 1260, 1240, 1270],
    })
    
    df.set_index("date", inplace=True)
    return df


# -----------------------------------------------------------------------------
# FIXTURES: ARCHIVOS TEMPORALES
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_csv(tmp_path, synthetic_data):
    """
    Guarda los datos sintéticos en un archivo CSV temporal.
    
    Returns:
        str: Ruta absoluta al archivo CSV temporal.
    """
    p = tmp_path / "test_mining_data.csv"
    synthetic_data.to_csv(p)
    return str(p)


@pytest.fixture
def temp_config_json(tmp_path, temp_csv):
    """
    Crea un archivo de configuración JSON temporal para MiningDataAdapter.
    
    Returns:
        str: Nombre del archivo de configuración (sin ruta completa).
    """
    config = {
        "dataset_name": "test_synthetic",
        "description": "Dataset sintético para testing",
        "files": {
            "filename": Path(temp_csv).name,
            "timestamp_column": "date",
            "separator": ","
        },
        "modeling": {
            "target_column": "rougher.output.recovery",
            "problem_type": "regression"
        },
        "feature_engineering": {
            "include_patterns": [
                "rougher",
                "primary_cleaner",
                "flotation"
            ],
            "exclude_patterns": [],
            "forced_drop": []
        },
        "preprocessing": {
            "handle_nulls": "ffill"
        }
    }
    
    # Crear directorio config si no existe
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Crear directorio data y copiar CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Copiar el CSV al directorio data
    import shutil
    shutil.copy(temp_csv, data_dir / Path(temp_csv).name)
    
    # Guardar JSON
    config_path = config_dir / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)


# -----------------------------------------------------------------------------
# FIXTURES: INSTANCIAS DE MÓDULOS
# -----------------------------------------------------------------------------

@pytest.fixture
def fresh_schema():
    """
    Retorna una instancia NUEVA del schema (sin cache ni overrides).
    
    Útil para tests que modifican el schema y no quieren afectar otros tests.
    """
    from core.validation.schema import MiningSchema
    return MiningSchema()


@pytest.fixture
def fresh_validator(fresh_schema):
    """
    Retorna una instancia nueva del validador con schema limpio.
    """
    from core.validation.validator import MiningValidator
    return MiningValidator(schema=fresh_schema)


# -----------------------------------------------------------------------------
# FIXTURES: MODELOS MOCK
# -----------------------------------------------------------------------------

@pytest.fixture
def trained_model(synthetic_data, tmp_path):
    """
    Entrena un modelo MiningGP con datos sintéticos y lo retorna.
    
    Nota: Este fixture es costoso (entrena un modelo), usar con moderación.
    
    Returns:
        MiningGP: Modelo entrenado listo para inferencia.
    """
    from core.models.mining_gp_pro import MiningGP
    
    # Guardar datos en CSV temporal
    csv_path = tmp_path / "train_data.csv"
    synthetic_data.to_csv(csv_path)
    
    # Entrenar modelo
    model = MiningGP(
        target_col="rougher.output.recovery",
        subsample_step=10,  # Rápido para tests
        add_lag_features=True,
        add_diff_features=False,
    )
    
    model.train_from_file(
        filepath=str(csv_path),
        n_trials=1,
        test_size=0.2,
        save_model=False
    )
    
    return model
