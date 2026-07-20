"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: train_universal.py
Proyecto: Universal Soft-Sensor
Versión: 2.3.2 — BUGFIX

HISTORIAL:
    [v2.3.2 - 2026]
        [FIX] Resource leak: temp file nunca se eliminaba.
              prepare_data_phase() creaba train_input_{timestamp}.csv en cada
              ejecución pero nunca lo borraba. Con el dataset de flotación
              (~700MB) esto acumula gigabytes silenciosamente en sesiones
              largas de optimización con múltiples trials Optuna.
              Solución: try/finally en main() garantiza limpieza siempre,
              incluso si el entrenamiento falla a mitad.

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Interfaz y UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.adapters import DataAdapter
from core.models.gp_model import SoftSensorGP, ModelMetrics
from config.settings import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer_v2.3.2")
console = Console()

def prepare_data_phase() -> tuple:
    """
    FASE 1: Ingesta y Preparación.
    
    Returns:
        tuple: (ruta_archivo_filtrado, configuracion_dict, instancia_adapter)
    """
    console.print(Panel(
        "📥 [bold cyan]FASE 1: INGESTA Y FILTRADO UNIFICADO[/bold cyan]",
        border_style="cyan"
    ))
    
    adapter = DataAdapter("dataset_config.json")
    df = adapter.load_data()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    temp_path = CONFIG.DATA_PROCESSED_DIR / f"train_input_{timestamp}.csv"
    df.to_csv(temp_path)
    
    logger.info(f"✅ Datos preparados: {len(df)} registros listos para el modelo.")
    return temp_path, adapter.config, adapter

def train_model_phase(data_path: Path, ad_config: dict) -> tuple:
    """
    FASE 2: Entrenamiento Predictivo.
    
    Args:
        data_path: Ruta al CSV preparado en la Fase 1.
        ad_config: Configuración extraída del adaptador.
        
    Returns:
        tuple: (modelo_entrenado, objeto_metricas)
    """
    console.print(Panel(
        "🧠 [bold yellow]FASE 2: ENTRENAMIENTO Y OPTIMIZACIÓN (GP)[/bold yellow]",
        border_style="yellow"
    ))
    
    target = ad_config["modeling"]["target_column"]
    
    model = SoftSensorGP(
        target_col=target,
        use_fallback_model=True
    )
    
    metrics = model.train_from_file(
        filepath=str(data_path),
        n_trials=CONFIG.GP_OPTUNA_TRIALS,
        save_model=True
    )
    
    return model, metrics

def report_phase(dataset_name: str, model: SoftSensorGP, metrics: ModelMetrics):
    """
    FASE 3: Auditoría y Cierre.
    """
    console.print(Panel(
        "📊 [bold green]FASE 3: RESUMEN EJECUTIVO DE CALIDAD[/bold green]",
        border_style="green"
    ))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Métrica", style="dim")
    table.add_column("Valor")
    table.add_column("Evaluación")

    r2 = metrics.r2
    status = (
        "[bold green]🏆 Excelente[/]" if r2 > 0.8
        else "[yellow]👍 Aceptable[/]" if r2 > 0.5
        else "[red]⚠️ Pobre[/]"
    )
    
    table.add_row("R² Score (Precisión)", f"{r2:.4f}", status)
    table.add_row("MAPE (Error %)", f"{metrics.mape:.2f}%", "✅" if metrics.mape < 10 else "❗")
    table.add_row("Algoritmo Final", model.model_type, "🧠" if model.model_type == "GP" else "🌲")
    
    console.print(table)
    console.print(f"\n[dim]Modelo guardado en: {CONFIG.MODELS_DIR}[/dim]")

def main():
    """Punto de entrada principal del Pipeline."""
    temp_path = None  # [FIX] Inicializar antes del try para poder limpiar en finally

    try:
        console.print(Panel.fit(
            "🚀 [bold blue]PIPELINE UNIVERSAL v2.3.2[/bold blue]\n"
            "[italic]Mining Architecture 4.0[/italic]"
        ))
        
        console.print(
            f"[dim]⚙️ Config: Trials={CONFIG.GP_OPTUNA_TRIALS} | "
            f"Subsample={CONFIG.DEFAULT_SUBSAMPLE_STEP}[/dim]\n"
        )

        # Ejecución de las 3 fases
        # [FIX] temp_path se captura aquí para poder limpiarla en el finally
        temp_path, config, adapter = prepare_data_phase()
        model, metrics = train_model_phase(temp_path, config)
        report_phase(config.get('dataset_name', 'Mining_Dataset'), model, metrics)

    except Exception as e:
        console.print(f"\n[bold red]🔥 ERROR CRÍTICO:[/bold red] {str(e)}")
        sys.exit(1)

    finally:
        # [FIX] Limpiar el archivo temporal siempre, incluso si hubo error.
        # ANTES: el temp file nunca se eliminaba → acumulaba ~700MB por ejecución.
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
                logger.info(f"Archivo temporal eliminado: {temp_path.name}")
            except OSError as e:
                logger.warning(f"No se pudo eliminar el temporal {temp_path}: {e}")

if __name__ == "__main__":
    main()
