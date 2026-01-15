"""
Módulo: tests/conftest.py
Descripción:
    Configuración global y 'Fixtures' para la suite de pruebas con Pytest.
    
    Aquí definimos los recursos compartidos (como datos falsos o rutas temporales)
    que se inyectan automáticamente en los tests. Esto evita repetir código
    de configuración en cada archivo de prueba.

Autor: Juan Galaz
Versión: 1.0.0
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL PATH
# -----------------------------------------------------------------------------
# Truco necesario para que Python encuentre la carpeta 'core' cuando ejecutamos
# pytest desde la raíz. Agregamos el directorio padre al sys.path.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """
    Hook de configuración inicial de Pytest.
    Registramos marcadores personalizados para clasificar los tests.
    """
    # 'integration': Tests que tocan disco o base de datos real (son más lentos)
    config.addinivalue_line("markers", "integration: tests lentos con dataset real")


# -----------------------------------------------------------------------------
# FIXTURES (RECURSOS COMPARTIDOS)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_data():
    """
    Genera un DataFrame sintético (falso) que imita la estructura física 
    de la planta de flotación.

    Propósito:
        Permite probar la lógica del modelo sin depender del archivo CSV real
        de 170MB, haciendo que los tests corran en milisegundos.

    Returns:
        pd.DataFrame: DataFrame con 500 filas, índice temporal y 5 columnas clave.
    """
    # 1. Creamos un índice temporal (fechas)
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1h")
    
    # 2. Generamos datos aleatorios para sensores simulados
    df = pd.DataFrame({
        "date": dates,
        # Target: Concentrado de sílice (variable a predecir)
        "_silica_concentrate": np.random.uniform(1.0, 5.0, 500),
        # Feature que debería eliminarse (colinealidad o fuga de datos)
        "_iron_concentrate": np.random.uniform(60.0, 68.0, 500), 
        # Feature útil
        "ore_pulp_ph": np.random.uniform(9.0, 11.0, 500),
        # Feature útil
        "flotation_column_01_air_flow": np.random.uniform(200, 300, 500),
        # Feature constante (ruido) para probar que el modelo V4 la elimina
        "flotation_column_04_air_flow": np.full(500, 295.0), 
    })
    
    # 3. Inyectamos una correlación matemática simple
    # Esto asegura que el modelo pueda aprender ALGO y no devuelva un R2 negativo
    # Lógica: "Si sube el pH, baja la sílice" (Relación inversa simulada)
    df["_silica_concentrate"] = 15.0 - df["ore_pulp_ph"] + np.random.normal(0, 0.1, 500)
    
    # Establecemos la fecha como índice (necesario para TimeSeriesSplit)
    df.set_index("date", inplace=True)
    
    return df


@pytest.fixture
def temp_csv(tmp_path, synthetic_data):
    """
    Guarda los datos sintéticos en un archivo CSV real dentro de una carpeta temporal.

    Args:
        tmp_path (Path): Fixture nativo de pytest que crea una carpeta temporal
                         que se autodestruye al finalizar el test.
        synthetic_data (pd.DataFrame): Los datos creados arriba.

    Returns:
        str: Ruta absoluta al archivo CSV temporal.
    """
    # Creamos la ruta del archivo temporal
    p = tmp_path / "test_mining_data.csv"
    
    # Guardamos el CSV físico
    synthetic_data.to_csv(p)
    
    # Devolvemos la ruta como string (que es lo que esperan nuestras funciones)
    return str(p)