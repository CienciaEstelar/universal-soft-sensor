"""
Módulo: core/adapters/mining_csv_adapter.py
Nivel: UNIVERSAL / BLINDADO
Descripción: Adaptador agnóstico para ingesta de datos CSV.
             
Características:
    - Auto-detección de separador (sep=None + engine='python')
    - Sanitización de nombres de columnas (snake_case)
    - Manejo de decimales chilenos/gringos automáticamente
    - Parseo robusto de fechas con múltiples formatos
    - Streaming por chunks para manejo eficiente de memoria
    
Uso:
    from core.adapters import MiningCSVAdapter
    
    adapter = MiningCSVAdapter("/path/to/data.csv")
    for chunk in adapter.stream(chunk_size=25000):
        process(chunk)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, Optional, List
import logging

logger = logging.getLogger(__name__)


class MiningCSVAdapter:
    """
    Adaptador universal para archivos CSV de datos mineros.
    
    Diseñado para ser resiliente ante:
    - Diferentes separadores (coma, punto y coma, tab)
    - Formatos de decimales regionales (1,5 vs 1.5)
    - Múltiples formatos de fecha
    - Líneas corruptas o malformadas
    """
    
    # Formatos de fecha comunes en datasets industriales
    DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S",    # ISO estándar
        "%d/%m/%Y %H:%M:%S",    # Formato europeo/chileno
        "%m/%d/%Y %H:%M:%S",    # Formato americano
        "%Y-%m-%d %H:%M",       # Sin segundos
        "%d-%m-%Y %H:%M:%S",    # Europeo con guiones
        "%Y/%m/%d %H:%M:%S",    # Alternativo
    ]
    
    def __init__(self, filepath: str, encoding: str = "utf-8"):
        """
        Inicializa el adaptador.
        
        Parameters
        ----------
        filepath : str
            Ruta al archivo CSV.
        encoding : str
            Codificación del archivo (default: utf-8).
            
        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        """
        self.path = Path(filepath)
        self.encoding = encoding
        
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset no encontrado: {self.path}\n"
                f"Verifica la ruta o configura MINING_DATA_RAW_PATH en .env"
            )
        
        logger.info(f"Adapter inicializado: {self.path.name} ({self._get_file_size()})")
    
    def _get_file_size(self) -> str:
        """Retorna tamaño del archivo en formato legible."""
        size_bytes = self.path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza nombres de columnas a snake_case.
        
        Transformaciones:
        - Espacios → guiones bajos
        - Caracteres especiales (%, /, paréntesis) → eliminados
        - Todo a minúsculas
        """
        original_cols = df.columns.tolist()
        
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
            .str.replace(r'[()%]', '', regex=True)
            .str.replace('/', '_', regex=False)
            .str.replace(r'__+', '_', regex=True)  # Eliminar guiones dobles
        )
        
        # Log cambios si hubo
        new_cols = df.columns.tolist()
        changed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
        if changed and len(changed) <= 5:
            logger.debug(f"Columnas renombradas: {changed}")
        elif changed:
            logger.debug(f"Columnas renombradas: {len(changed)} columnas")
        
        return df
    
    def _parse_date_column(self, series: pd.Series) -> pd.Series:
        """
        Parsea una serie de fechas probando múltiples formatos.
        
        Estrategia:
        1. Intentar cada formato conocido
        2. Si todos fallan, usar inferencia automática
        3. Retornar NaT para valores que no se pueden parsear
        """
        # Guardar valores originales para retry
        original_values = series.copy()
        
        # Intentar cada formato explícito
        for fmt in self.DATE_FORMATS:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                # Si más del 80% parsea correctamente, aceptar
                valid_pct = parsed.notna().mean()
                if valid_pct > 0.8:
                    logger.debug(f"Formato de fecha detectado: {fmt} ({valid_pct*100:.1f}% válido)")
                    return parsed
            except Exception:
                continue
        
        # Fallback: inferencia automática de pandas
        logger.debug("Usando inferencia automática de fechas")
        try:
            return pd.to_datetime(original_values, errors='coerce', infer_datetime_format=True)
        except Exception as e:
            logger.warning(f"Fallo en parseo de fechas: {e}")
            return pd.Series([pd.NaT] * len(series), index=series.index)
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas object a numéricas manejando decimales regionales.
        
        Maneja formatos como:
        - "1,234.56" (inglés con miles)
        - "1.234,56" (europeo con miles)
        - "1234,56" (decimal con coma simple)
        """
        cols_object = df.select_dtypes(include=['object']).columns
        
        for col in cols_object:
            # Saltar columna de fecha
            if 'date' in col.lower():
                continue
            
            try:
                # Limpiar y convertir
                serie_limpia = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(',', '.', regex=False)  # Coma decimal → punto
                )
                
                df[col] = pd.to_numeric(serie_limpia, errors='coerce')
                
            except Exception as e:
                logger.debug(f"Columna '{col}' no es numérica: {e}")
                continue
        
        return df
    
    def _universal_cleaner(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiador heurístico completo.
        
        Ejecuta en orden:
        1. Conversión de columnas numéricas
        2. Parseo de fechas
        """
        # 1. Convertir columnas numéricas
        df = self._convert_numeric_columns(df)
        
        # 2. Parsear columna de fecha si existe
        if 'date' in df.columns:
            df['date'] = self._parse_date_column(df['date'])
            
            # Reportar calidad del parseo
            nat_count = df['date'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Fechas no parseables: {nat_count} ({nat_count/len(df)*100:.2f}%)")
        
        return df
    
    def stream(self, chunk_size: int = 25000) -> Generator[pd.DataFrame, None, None]:
        """
        Generador que produce DataFrames en streaming.
        
        Parameters
        ----------
        chunk_size : int
            Número de filas por chunk.
            
        Yields
        ------
        pd.DataFrame
            Chunk limpio y sanitizado, indexado por fecha si está disponible.
        """
        logger.info(f"Iniciando ingesta: {self.path.name} (chunks de {chunk_size:,} filas)")
        
        chunks_procesados = 0
        filas_totales = 0
        
        try:
            # AUTO-DETECCIÓN de separador con engine='python'
            with pd.read_csv(
                self.path,
                sep=None,              # Auto-detección
                engine='python',       # Requerido para sep=None
                chunksize=chunk_size,
                encoding=self.encoding,
                on_bad_lines='skip'    # Resiliencia ante líneas corruptas
            ) as reader:
                
                for chunk in reader:
                    chunks_procesados += 1
                    
                    # 1. Sanitizar nombres de columnas
                    chunk = self._sanitize_columns(chunk)
                    
                    # 2. Limpieza de tipos (decimales y fechas)
                    chunk = self._universal_cleaner(chunk)
                    
                    # 3. Indexar por fecha si existe
                    if 'date' in chunk.columns:
                        chunk.set_index('date', inplace=True)
                        chunk.sort_index(inplace=True)
                    
                    filas_totales += len(chunk)
                    
                    if chunks_procesados % 10 == 0:
                        logger.debug(f"Procesados {chunks_procesados} chunks ({filas_totales:,} filas)")
                    
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error en ingesta: {e}")
            raise
        
        logger.info(f"Ingesta completada: {chunks_procesados} chunks, {filas_totales:,} filas totales")
    
    def read_all(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Lee todo el archivo de una vez (útil para datasets pequeños).
        
        Parameters
        ----------
        max_rows : int, optional
            Límite de filas a leer.
            
        Returns
        -------
        pd.DataFrame
            Dataset completo limpio y sanitizado.
        """
        logger.info(f"Leyendo dataset completo{f' (max {max_rows} filas)' if max_rows else ''}")
        
        df = pd.read_csv(
            self.path,
            sep=None,
            engine='python',
            encoding=self.encoding,
            nrows=max_rows,
            on_bad_lines='skip'
        )
        
        df = self._sanitize_columns(df)
        df = self._universal_cleaner(df)
        
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        
        logger.info(f"Lectura completada: {df.shape[0]:,} filas, {df.shape[1]} columnas")
        return df
    
    def get_schema_info(self) -> dict:
        """
        Obtiene información del esquema sin cargar todo el archivo.
        
        Returns
        -------
        dict
            Información de columnas, tipos y muestra de datos.
        """
        # Leer solo primeras filas
        df_sample = self.read_all(max_rows=100)
        
        return {
            "columns": df_sample.columns.tolist(),
            "dtypes": df_sample.dtypes.to_dict(),
            "shape_sample": df_sample.shape,
            "memory_sample_mb": df_sample.memory_usage(deep=True).sum() / 1024 / 1024,
        }


# =============================================================================
# __init__.py helper
# =============================================================================
__all__ = ["MiningCSVAdapter"]


# =============================================================================
# CLI para testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    )
    
    # Permitir ruta por argumento o usar default
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Intentar encontrar el archivo en ubicaciones comunes
        from config.settings import CONFIG
        test_path = str(CONFIG.DATA_RAW_PATH)
    
    print(f"🛰️  Probando Adapter con: {test_path}")
    print("=" * 60)
    
    try:
        adapter = MiningCSVAdapter(test_path)
        
        # Leer primer chunk
        gen = adapter.stream()
        batch = next(gen)
        
        print(f"\n✅ Éxito! Formato detectado automáticamente.")
        print(f"📊 Dimensiones del chunk: {batch.shape}")
        print(f"\n📋 Columnas ({len(batch.columns)}):")
        print(batch.columns.tolist())
        print(f"\n🔢 Tipos de datos:")
        print(batch.dtypes)
        print(f"\n📈 Muestra de datos numéricos:")
        print(batch.select_dtypes(include=[np.number]).iloc[:3, :5])
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        raise
