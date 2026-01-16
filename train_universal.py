"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: train_universal.py
Proyecto: Arquitectura Minera 4.0
Autor: Juan Galaz (Refactorizado por Gemini)
Versión: 2.3.1 - AUDIT-READY
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Orquestador de alto nivel para el entrenamiento de Soft-Sensors. 
    Implementa un pipeline de MLOps que automatiza la ingesta filtrada, 
    la optimización bayesiana de parámetros y la generación de reportes forenses.

FASE 1: Ingesta con MiningDataAdapter (Detección y Limpieza).
FASE 2: Modelado con MiningGP (Optimización Optuna y Gaussian Process).
FASE 3: Reporting (Métricas de precisión y persistencia .pkl).

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

# Módulos del Proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.adapters import MiningDataAdapter
from core.models.mining_gp_pro import MiningGP, ModelMetrics
from config.settings import CONFIG

# Configuración de Logging Industrial
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer_v2.3.1")
console = Console()

def prepare_data_phase() -> tuple:
    """
    FASE 1: Ingesta y Preparación.
    Utiliza el MiningDataAdapter para sanitizar el CSV y aplicar los filtros
    definidos en el esquema JSON.
    
    Returns:
        tuple: (ruta_archivo_filtrado, configuracion_dict, instancia_adapter)
    """
    console.print(Panel("📥 [bold cyan]FASE 1: INGESTA Y FILTRADO UNIFICADO[/bold cyan]", border_style="cyan"))
    
    adapter = MiningDataAdapter("dataset_config.json")
    df = adapter.load_data() # Hace el trabajo pesado de limpieza
    
    # Crear archivo temporal para el entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    temp_path = CONFIG.DATA_PROCESSED_DIR / f"train_input_{timestamp}.csv"
    df.to_csv(temp_path)
    
    logger.info(f"✅ Datos preparados: {len(df)} registros listos para el modelo.")
    return temp_path, adapter.config, adapter

def train_model_phase(data_path: Path, ad_config: dict) -> tuple:
    """
    FASE 2: Entrenamiento Predictivo.
    Instancia el modelo MiningGP y ejecuta la optimización de hiperparámetros.
    
    Args:
        data_path: Ruta al CSV preparado en la Fase 1.
        ad_config: Configuración extraída del adaptador.
        
    Returns:
        tuple: (modelo_entrenado, objeto_metricas)
    """
    console.print(Panel("🧠 [bold yellow]FASE 2: ENTRENAMIENTO Y OPTIMIZACIÓN (GP)[/bold yellow]", border_style="yellow"))
    
    target = ad_config["modeling"]["target_column"]
    
    # Instanciamos el modelo v4.1 (Gaussian Process)
    model = MiningGP(
        target_col=target,
        use_fallback_model=True # Si el R2 es muy bajo, usa Gradient Boosting
    )
    
    # Ejecución del entrenamiento con trials dinámicos desde CONFIG
    metrics = model.train_from_file(
        filepath=str(data_path),
        n_trials=CONFIG.GP_OPTUNA_TRIALS, # CORRECCIÓN AUDITORÍA: Ya no está hardcodeado
        save_model=True
    )
    
    return model, metrics

def report_phase(dataset_name: str, model: MiningGP, metrics: ModelMetrics):
    """
    FASE 3: Auditoría y Cierre.
    Genera un resumen ejecutivo en consola sobre la calidad del Soft-Sensor.
    """
    console.print(Panel("📊 [bold green]FASE 3: RESUMEN EJECUTIVO DE CALIDAD[/bold green]", border_style="green"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Métrica", style="dim")
    table.add_column("Valor")
    table.add_column("Evaluación")

    # Evaluación visual del R2
    r2 = metrics.r2
    status = "[bold green]🏆 Excelente[/]" if r2 > 0.8 else "[yellow]👍 Aceptable[/]" if r2 > 0.5 else "[red]⚠️ Pobre[/]"
    
    table.add_row("R² Score (Precisión)", f"{r2:.4f}", status)
    table.add_row("MAPE (Error %)", f"{metrics.mape:.2f}%", "✅" if metrics.mape < 10 else "❗")
    table.add_row("Algoritmo Final", model.model_type, "🧠" if model.model_type == "GP" else "🌲")
    
    console.print(table)
    console.print(f"\n[dim]Modelo guardado en: {CONFIG.MODELS_DIR}[/dim]")

def main():
    """Punto de entrada principal del Pipeline."""
    try:
        console.print(Panel.fit("🚀 [bold blue]PIPELINE UNIVERSAL v2.3.1[/bold blue]\n[italic]Mining Architecture 4.0[/italic]"))
        
        # Log de parámetros activos
        console.print(f"[dim]⚙️ Config: Trials={CONFIG.GP_OPTUNA_TRIALS} | Subsample={CONFIG.DEFAULT_SUBSAMPLE_STEP}[/dim]\n")

        # Ejecución de las 3 fases
        data_path, config, adapter = prepare_data_phase()
        model, metrics = train_model_phase(data_path, config)
        report_phase(config.get('dataset_name', 'Mining_Dataset'), model, metrics)

    except Exception as e:
        console.print(f"\n[bold red]🔥 ERROR CRÍTICO:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
