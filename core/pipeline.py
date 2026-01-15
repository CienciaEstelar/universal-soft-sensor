"""
Módulo: core/pipeline.py
Modo: ETL Completo (Extract - Transform - Load)
Descripción:
    - Ingesta + Validación + Limpieza
    - Guarda resultado en data/processed/mining_clean.csv
    - Maneja escritura incremental (append) para no saturar RAM
    - Soporta checkpointing para recuperación ante fallos
    
Uso:
    python -m core.pipeline
    
    # O desde código:
    from core.pipeline import MiningPipeline
    pipeline = MiningPipeline()
    pipeline.run()
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# Rich Progress
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.logging import RichHandler

# Componentes del proyecto
from config.settings import CONFIG
from core.adapters import MiningCSVAdapter
from core.validation import MiningValidator
from core.preprocessor import MiningPreprocessor


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    """
    Configura logging con salida a archivo y consola Rich.
    
    Parameters
    ----------
    log_file : Path, optional
        Ruta al archivo de log.
    level : str
        Nivel de logging.
        
    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Crear logger específico para el pipeline
    logger = logging.getLogger("PIPELINE")
    logger.setLevel(log_level)
    logger.handlers = []  # Limpiar handlers previos
    
    # Handler para archivo
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    # Handler para consola (solo errores para no interferir con Rich progress)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    return logger


class MiningPipeline:
    """
    Pipeline ETL para datos de proceso minero.
    
    Etapas:
    1. Extract: Lectura incremental desde CSV
    2. Transform: Validación + Limpieza
    3. Load: Escritura incremental a CSV limpio
    
    Atributos:
        stats: Diccionario con métricas de ejecución.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        estrategia_limpieza: str = "ffill",
        detectar_outliers: bool = False,
        enable_checkpointing: bool = True
    ) -> None:
        """
        Inicializa el pipeline.
        
        Parameters
        ----------
        chunk_size : int, optional
            Tamaño de chunk. Si None, usa CONFIG.
        estrategia_limpieza : str
            Estrategia de imputación para el preprocesador.
        detectar_outliers : bool
            Si True, detecta y maneja outliers.
        enable_checkpointing : bool
            Si True, guarda progreso para recuperación.
        """
        self.console = Console()
        self.chunk_size = chunk_size or CONFIG.CHUNK_SIZE
        self.enable_checkpointing = enable_checkpointing
        
        # Rutas de salida
        self.output_dir = CONFIG.DATA_PROCESSED_DIR
        self.output_file = CONFIG.DATA_CLEAN_PATH
        self.checkpoint_file = self.output_dir / ".checkpoint"
        
        # Logging
        log_file = CONFIG.LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.logger = setup_logging(log_file)
        
        # Estadísticas
        self.stats = {
            "start_time": None,
            "end_time": None,
            "chunks_procesados": 0,
            "filas_entrada": 0,
            "filas_salida": 0,
            "filas_rechazadas": 0,
        }
        
        # Componentes
        self.adapter: Optional[MiningCSVAdapter] = None
        self.validator: Optional[MiningValidator] = None
        self.preprocessor: Optional[MiningPreprocessor] = None
        
        self._init_components(estrategia_limpieza, detectar_outliers)
    
    def _init_components(self, estrategia: str, outliers: bool) -> None:
        """Inicializa los componentes del pipeline."""
        try:
            # Validar configuración
            CONFIG.validate()
            
            # Crear directorio de salida
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Limpiar archivo previo si existe (empezar de cero)
            if self.output_file.exists():
                self.output_file.unlink()
                self.logger.info(f"Archivo previo eliminado: {self.output_file}")
            
            # Inicializar componentes
            self.adapter = MiningCSVAdapter(str(CONFIG.DATA_RAW_PATH))
            self.validator = MiningValidator()
            self.preprocessor = MiningPreprocessor(
                estrategia_nulos=estrategia,
                detectar_outliers=outliers
            )
            
            self.logger.info("Pipeline inicializado correctamente")
            
        except FileNotFoundError as e:
            self.console.print(f"[bold red]❌ Error de configuración: {e}[/bold red]")
            raise
        except Exception as e:
            self.logger.critical(f"Error de inicialización: {e}")
            self.console.print(f"[bold red]🔥 Error crítico: {e}[/bold red]")
            raise
    
    def _save_checkpoint(self, batch_num: int, rows_processed: int) -> None:
        """Guarda checkpoint para recuperación."""
        if not self.enable_checkpointing:
            return
        
        with open(self.checkpoint_file, 'w') as f:
            f.write(f"{batch_num},{rows_processed},{datetime.now().isoformat()}")
    
    def _load_checkpoint(self) -> tuple[int, int]:
        """Carga checkpoint si existe. Retorna (último_batch, filas_procesadas)."""
        if not self.checkpoint_file.exists():
            return 0, 0
        
        try:
            content = self.checkpoint_file.read_text().strip()
            parts = content.split(',')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0
    
    def _clear_checkpoint(self) -> None:
        """Elimina archivo de checkpoint al completar exitosamente."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
    
    def run(self) -> dict:
        """
        Ejecuta el pipeline ETL completo.
        
        Returns
        -------
        dict
            Estadísticas de ejecución.
        """
        self.stats["start_time"] = datetime.now()
        total_rows = 0
        
        with Progress(
            SpinnerColumn("dots", style="bold yellow"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, style="dim white", complete_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        ) as progress:
            
            task_id = progress.add_task("💾 ETL en Progreso...", total=None)
            
            try:
                for i, raw_batch in enumerate(self.adapter.stream(chunk_size=self.chunk_size)):
                    self.stats["chunks_procesados"] += 1
                    self.stats["filas_entrada"] += len(raw_batch)
                    
                    # 1. VALIDAR
                    valid_batch = self.validator.validate(raw_batch)
                    
                    if valid_batch.empty:
                        self.logger.warning(f"Batch {i}: vacío después de validación")
                        continue
                    
                    # Registrar rechazadas
                    rechazadas = len(raw_batch) - len(valid_batch)
                    self.stats["filas_rechazadas"] += rechazadas
                    
                    # 2. LIMPIAR
                    clean_batch = self.preprocessor.clean_stream(valid_batch)
                    
                    # 3. GUARDAR (LOAD) - CSV Incremental
                    es_primer_batch = (i == 0)
                    
                    clean_batch.to_csv(
                        self.output_file,
                        mode='w' if es_primer_batch else 'a',
                        header=es_primer_batch,
                        index=True  # Guardamos la fecha (index)
                    )
                    
                    # 4. Actualizar métricas
                    rows = len(clean_batch)
                    total_rows += rows
                    self.stats["filas_salida"] += rows
                    
                    progress.update(task_id, advance=rows)
                    
                    # Checkpoint cada 10 batches
                    if self.enable_checkpointing and i % 10 == 0:
                        self._save_checkpoint(i, total_rows)
                    
                    self.logger.info(
                        f"Batch {i}: {rows} filas guardadas "
                        f"(rechazadas: {rechazadas})"
                    )
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]🛑 Detenido por usuario[/yellow]")
                self.logger.warning("Pipeline interrumpido por usuario")
                self.stats["end_time"] = datetime.now()
                return self.stats
                
            except Exception as e:
                self.logger.error(f"Error en pipeline: {e}", exc_info=True)
                self.console.print(f"\n[bold red]🔥 Error: {e}[/bold red]")
                self.stats["end_time"] = datetime.now()
                return self.stats
            
            # Completado exitosamente
            progress.update(
                task_id,
                description="[bold green]✅ ETL Completado",
                total=total_rows
            )
            
            self._clear_checkpoint()
        
        self.stats["end_time"] = datetime.now()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        # Resumen final
        self.console.print()
        self.console.print(f"[bold green]💾 Dataset guardado: {self.output_file}[/bold green]")
        self.console.print(f"[bold white]✨ Filas procesadas: {self.stats['filas_salida']:,}[/bold white]")
        self.console.print(f"[dim]⏱️  Tiempo total: {duration}[/dim]")
        
        if self.stats["filas_rechazadas"] > 0:
            pct_rechazado = (self.stats["filas_rechazadas"] / self.stats["filas_entrada"]) * 100
            self.console.print(
                f"[yellow]⚠️  Filas rechazadas: {self.stats['filas_rechazadas']:,} "
                f"({pct_rechazado:.2f}%)[/yellow]"
            )
        
        self.logger.info(f"Pipeline completado: {self.stats}")
        
        return self.stats


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    """Entry point para ejecución desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline ETL para datos de proceso minero"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=None,
        help="Tamaño de chunk (default: desde config)"
    )
    parser.add_argument(
        "--estrategia", "-e",
        choices=["ffill", "bfill", "interpolate", "mean", "median"],
        default="ffill",
        help="Estrategia de imputación de nulos"
    )
    parser.add_argument(
        "--outliers",
        action="store_true",
        help="Activar detección de outliers"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Desactivar checkpointing"
    )
    
    args = parser.parse_args()
    
    pipeline = MiningPipeline(
        chunk_size=args.chunk_size,
        estrategia_limpieza=args.estrategia,
        detectar_outliers=args.outliers,
        enable_checkpointing=not args.no_checkpoint
    )
    
    stats = pipeline.run()
    
    # Exit code basado en resultado
    if stats.get("filas_salida", 0) > 0:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
