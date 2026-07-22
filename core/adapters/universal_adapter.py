"""
Módulo: core/adapters/universal_adapter.py
Autor: Juan Galaz (Universal Soft-Sensor)
Versión: 1.2 (Documentación Extendida)

Descripción:
    Adaptador de Datos Universal (Data Driven Adapter).
    
    Este componente es la "Puerta de Entrada" de los datos al sistema.
    Su función es desacoplar el código Python de los datos específicos.
    En lugar de tener "ifs" para cada mina (Oro, Hierro, Cobre), este adaptador
    lee las reglas de un archivo JSON y se transforma para leer cualquier dataset.

Responsabilidades:
    1. Leer configuración desde 'config/dataset_config.json'.
    2. Cargar el CSV masivo de forma eficiente.
    3. Filtrar columnas "trampa" (Data Leakage) usando patrones de texto.
    4. Entregar un DataFrame limpio y listo para el entrenamiento.
"""

import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Set

# Intentamos importar la configuración global.
# Si falla, usamos rutas relativas como fallback.
try:
    from config.settings import CONFIG
    DATA_DIR = CONFIG.DATA_DIR
except ImportError:
    # Fallback si settings.py no está configurado aún
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Configuración del logger para trazabilidad
logger = logging.getLogger("Mining_Universal_Adapter")

class UniversalAdapter:
    """
    Clase adaptadora que ingesta datos basándose en reglas externas (JSON).
    """

    def __init__(self, config_filename: str = "dataset_config.json"):
        """
        Inicializa el adaptador.

        Args:
            config_filename (str): Nombre del archivo JSON que vive en la carpeta 'config/'.
                                   Define qué dataset cargar y cómo limpiarlo.
        """
        # 1. Construir ruta al archivo de configuración JSON
        # La ruta es: raiz/config/<nombre_archivo>
        self.config_path = Path(__file__).parent.parent.parent / "config" / config_filename
        
        # 2. Cargar las reglas en memoria
        self.config = self._load_config()
        
        # 3. Construir ruta al archivo de datos CSV
        # [SECURITY V4] Contención anti path-traversal: el `filename` viene del
        # config (potencialmente de un tercero). Sin esto, filename=
        # "../../../../etc/passwd" resolvería FUERA de data/ y permitiría leer
        # archivos arbitrarios del sistema. Se exige que el path resuelto quede
        # dentro de DATA_DIR.
        filename = self.config["files"]["filename"]
        data_root = DATA_DIR.resolve()
        candidate = (DATA_DIR / filename).resolve()
        if not str(candidate).startswith(str(data_root) + os.sep) and candidate != data_root:
            raise ValueError(
                f"filename '{filename}' escapa de la carpeta data/ "
                f"(resuelto: {candidate}). Path traversal bloqueado."
            )
        self.data_path = candidate

        logger.info(f"🔧 Adaptador Universal iniciado con reglas de: {config_filename}")

    def _load_config(self) -> Dict:
        """
        Lee el archivo JSON de disco y lo convierte en un diccionario Python.
        
        Returns:
            Dict: Configuración completa (files, modeling, feature_engineering).
        
        Raises:
            FileNotFoundError: Si no existe el archivo JSON en la carpeta config.
        """
        if not self.config_path.exists():
            msg = f"❌ Archivo de configuración no encontrado: {self.config_path}"
            logger.critical(msg)
            raise FileNotFoundError(msg)
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ El JSON está mal formado (faltan comillas o llaves): {e}")
            raise

    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [CEREBRO DEL ADAPTADOR]
        Selecciona las columnas útiles y elimina las peligrosas (Leakage)
        basándose en los patrones definidos en el JSON.

        Args:
            df (pd.DataFrame): DataFrame con todas las columnas crudas (80+).

        Returns:
            pd.DataFrame: DataFrame filtrado (ej. 40 columnas).
        """
        rules = self.config["feature_engineering"]
        target_col = self.config["modeling"]["target_column"]
        
        initial_cols = len(df.columns)
        
        # --- PASO 1: INCLUSIÓN (Whitelist) ---
        # Buscamos columnas que coincidan con los patrones permitidos (ej: "sensor_", "state.")
        keep_cols: Set[str] = set()
        
        for pattern in rules.get("include_patterns", []):
            # Magia de Python: Busca el patrón dentro del nombre de cada columna
            matches = [c for c in df.columns if pattern in c]
            keep_cols.update(matches)
            
        # IMPORTANTE: Siempre salvar la columna objetivo (Target), 
        # sin ella no podemos entrenar.
        if target_col in df.columns:
            keep_cols.add(target_col)
        else:
            logger.warning(f"⚠️ La columna target '{target_col}' no está en el CSV.")

        # Creamos un DF temporal solo con lo que pasó el filtro de inclusión.
        # [P0b ROADMAP][FIX] NO usar list(keep_cols): el orden de iteración de
        # un set de strings en Python es no-determinista ENTRE PROCESOS (hash
        # randomization, PYTHONHASHSEED aleatorio por defecto desde Py 3.3).
        # Esto no es cosmético: remove_correlated_features() (más adelante en
        # el pipeline) decide CUÁL de dos features >threshold correlacionadas
        # sobrevive según el orden de columnas — con list(keep_cols), dos
        # corridas del mismo dataset podían tirar una feature distinta y dar
        # un R² distinto. Detectado con GeoMet cobre: una corrida tiraba
        # "Si ppm", otra "Fe ppm", con R²=0.315 vs R²=0.167 respectivamente.
        # Preservar el orden original de df.columns es determinista y además
        # más predecible (ordenado como el CSV de origen).
        keep_cols_ordered = [c for c in df.columns if c in keep_cols]
        df_filtered = df[keep_cols_ordered].copy()
        
        # --- PASO 2: EXCLUSIÓN (Blacklist) ---
        # Eliminamos columnas que explícitamente no queremos (ej: ".output", fechas)
        drop_cols: Set[str] = set()
        
        for pattern in rules.get("exclude_patterns", []):
            matches = [c for c in df_filtered.columns if pattern in c]
            
            # PROTECCIÓN: Si el patrón de borrado coincide con el Target,
            # protegemos al Target. No queremos borrar lo que queremos predecir.
            if target_col in matches:
                matches.remove(target_col)
                
            drop_cols.update(matches)
            
        # También borramos columnas forzadas por nombre exacto (ej: "date")
        forced_drop = rules.get("forced_drop", [])
        drop_cols.update([c for c in forced_drop if c in df_filtered.columns])

        # Aplicamos el borrado final
        df_final = df_filtered.drop(columns=list(drop_cols))
        
        # Reporte de limpieza
        removed = initial_cols - len(df_final.columns)
        logger.info(f"🧹 Limpieza de Features: {initial_cols} columnas -> {len(df_final.columns)} columnas.")
        logger.info(f"   (Se eliminaron {removed} columnas irrelevantes o futuras)")
        
        return df_final

    def load_data(self) -> pd.DataFrame:
        """
        Ejecuta el proceso completo de Carga -> Limpieza -> Preparación.

        Returns:
            pd.DataFrame: Los datos listos para el modelo de IA.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"❌ No encuentro el CSV en: {self.data_path}")

        logger.info(f"⏳ Leyendo CSV masivo: {self.data_path.name} ...")
        
        # Obtenemos parámetros de lectura del JSON
        ts_col = self.config["files"]["timestamp_column"]
        sep = self.config["files"].get("separator", ",")

        # 1. Lectura optimizada con parseo de fechas
        try:
            df = pd.read_csv(self.data_path, sep=sep, parse_dates=[ts_col])
        except ValueError:
            # Si falla el parseo automático, leemos normal y convertimos después
            df = pd.read_csv(self.data_path, sep=sep)
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        # 2. Configuración de Serie de Tiempo (Índice)
        is_temporal = ts_col in df.columns
        if is_temporal:
            df.set_index(ts_col, inplace=True)
            df.sort_index(inplace=True) # El tiempo debe ser lineal

        # 3. Aplicar Inteligencia de Selección de Columnas
        df = self._apply_feature_selection(df)

        # 4. Manejo Básico de Nulos (Imputación)
        # [P0b ROADMAP][FIX] 'ffill' (Forward Fill) asume que la fila anterior
        # es el "último valor conocido" del MISMO sensor/proceso — válido en
        # series temporales (ej. sensor falla, se mantiene el último dato
        # hasta que vuelve). Para datasets SIN dimensión temporal (ej.
        # geometalúrgicos: una fila = una muestra de sondaje, sin orden con
        # significado), la fila "anterior" es una muestra de OTRO sondaje sin
        # relación real — ffill fabricaría silenciosamente un valor prestado
        # de una muestra distinta. Detectado con GeoMet cobre: "Carbono
        # Grafite ppm" tiene 12/60 NaN y el ffill contaminaba filas de un
        # HOLEID con el valor de otro. Sin timestamp real, se usa dropna()
        # directo (honesto: pierde filas incompletas en vez de inventarlas).
        if is_temporal:
            df = df.ffill().dropna()
        else:
            df = df.dropna()
        
        logger.info(f"✅ Datos cargados exitosamente: {len(df):,} filas.")
        
        return df

# Bloque de prueba: Se ejecuta solo si corres este script directamente
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        adapter = UniversalAdapter()
        df = adapter.load_data()
        print("\n--- 5 Primeras filas del Dataset Limpio ---")
        print(df.head())
    except Exception as e:
        print(f"Error en prueba: {e}")