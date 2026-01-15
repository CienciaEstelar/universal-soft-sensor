"""
═══════════════════════════════════════════════════════════════════════════════
Script: train_universal.py
Autor: Juan Galaz (Arquitectura Minera 4.0)
Versión: 2.1 (Stability Patch - Enero 2026)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Orquestador Universal del Pipeline de Entrenamiento (MLOps).
    
    Este script es el "Director de Orquesta" que conecta:
    1. INGESTA: UniversalAdapter (Carga inteligente basada en JSON)
    2. MODELADO: MiningGP v4 (Gaussian Process con corrección de ruido)
    3. PERSISTENCIA: Guardado automático de modelos y métricas.

ACTUALIZACIONES CRÍTICAS (v2.1):
    ✅ CORRECCIÓN DE ESTABILIDAD (R² Negativo):
       Se aumentó n_trials a 50 y se ajustó subsample_step a 10.
       Esto permite que Optuna encuentre el óptimo global y evita que el
       Gaussian Process asuma excesivo ruido ("Noise Level" alto).
    
    ✅ SOPORTE TEMPORAL:
       Se garantiza el uso de MiningGP v4 que respeta la flecha del tiempo
       (shuffle=False implícito) para evitar Data Leakage.

USO:
    python train_universal.py

REQUISITOS:
    - config/dataset_config.json (Debe existir y estar bien formado)
    - core/models/mining_gp_pro.py (Debe incluir el fix de noise_level < 0.1)

FLUJO DE EJECUCIÓN:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. INGESTA (Adapter)                                           │
    │     → Lee JSON -> Carga CSV -> Filtra Columnas (Regex)          │
    ├─────────────────────────────────────────────────────────────────┤
    │  2. ENTRENAMIENTO (MiningGP)                                    │
    │     → Diagnóstico de Autocorrelación                            │
    │     → Optimización Optuna (50 trials)                           │
    │     → Entrenamiento (GP o GradientBoosting fallback)            │
    ├─────────────────────────────────────────────────────────────────┤
    │  3. REPORTE (Console UI)                                        │
    │     → Tabla de Métricas (R², RMSE, MAPE)                        │
    │     → Guardado de Artefactos (.pkl, .png)                       │
    └─────────────────────────────────────────────────────────────────┘

EXIT CODES:
    0 : Éxito (Modelo útil, R² > 0)
    1 : Fallo (Error técnico o R² negativo)

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# --- Librerías de UI Profesional (Rich) ---
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Módulos de la Arquitectura Minera ---
from core.adapters.universal_adapter import UniversalAdapter
from core.models.mining_gp_pro import MiningGP, ModelMetrics
from config.settings import CONFIG

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Universal_Trainer_v2.1")
console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA Y PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data_with_adapter(config_filename: str = "dataset_config.json") -> tuple:
    """
    Carga y limpia datos usando UniversalAdapter.
    Actúa como firewall de datos antes de entrar al modelo.
    
    Args:
        config_filename (str): Nombre del JSON en la carpeta config/.
        
    Returns:
        tuple: (Path al CSV temporal limpio, Dict de configuración).
    """
    console.print(Panel.fit("📥 FASE 1: Carga y Filtrado de Datos", style="bold cyan"))
    
    try:
        # 1. Instanciar Adaptador
        adapter = UniversalAdapter(config_filename)
        logger.info(f"📋 Dataset configurado: {adapter.config['dataset_name']}")
        
        # 2. Carga y Limpieza (Regex filter)
        df = adapter.load_data()
        
        # 3. Mostrar Estadísticas en Consola
        console.print(f"\n[bold green]✅ Datos cargados y filtrados:[/bold green]")
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_row("Registros:", f"{len(df):,}")
        stats_table.add_row("Features:", f"{len(df.columns) - 1}")
        stats_table.add_row("Target:", adapter.config["modeling"]["target_column"])
        stats_table.add_row("Rango temporal:", f"{df.index.min()} → {df.index.max()}")
        console.print(stats_table)
        
        # 4. Guardar Snapshot (CSV Procesado)
        CONFIG.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = adapter.config['dataset_name']
        temp_filepath = CONFIG.DATA_PROCESSED_DIR / f"{dataset_name}_filtered_{timestamp}.csv"
        
        df.to_csv(temp_filepath)
        logger.info(f"💾 CSV limpio guardado: {temp_filepath}")
        
        return temp_filepath, adapter.config
        
    except FileNotFoundError as e:
        logger.critical(f"❌ Archivo no encontrado: {e}")
        raise
    except KeyError as e:
        logger.critical(f"❌ JSON mal formado: {e}")
        raise
    except Exception as e:
        logger.critical(f"💥 Error fatal en ingesta: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# FASE 2: ENTRENAMIENTO (CORE ML)
# ═══════════════════════════════════════════════════════════════════════════

def train_model_with_gp(
    data_filepath: Path,
    adapter_config: dict,
    n_trials: int = 50,      # AUMENTADO PARA ESTABILIDAD
    subsample_step: int = 10 # OPTIMIZADO PARA DENSIDAD
) -> tuple:
    """
    Ejecuta el entrenamiento del Soft-Sensor (MiningGP v4).
    
    Ajustes de Estabilidad v2.1:
    - n_trials=50: Da tiempo a Optuna para salir de mínimos locales.
    - subsample_step=10: Mantiene alta densidad de datos para capturar transitorios.
    """
    console.print(Panel.fit("🧠 FASE 2: Entrenamiento con MiningGP v4", style="bold yellow"))
    
    try:
        target_col = adapter_config["modeling"]["target_column"]
        
        # 1. Configuración del Modelo
        model = MiningGP(
            target_col=target_col,
            subsample_step=subsample_step,
            add_lag_features=True,           # Vital para series de tiempo
            add_diff_features=True,          # Detecta cambios de tendencia
            use_fallback_model=True,         # Red de seguridad (GradientBoosting)
            remove_constant_features=True,   # Limpieza automática
            remove_correlated_features=True  # Evita colinealidad
        )
        
        logger.info(f"🎯 Target columna: {target_col}")
        logger.info(f"🔧 Configuración: Optuna Trials={n_trials} | Subsample=1/{subsample_step}")
        
        # 2. Ejecución del Pipeline (Train/Test Split temporal interno)
        metrics = model.train_from_file(
            filepath=str(data_filepath),
            test_size=0.2,
            n_trials=n_trials,
            save_model=True
        )
        
        return model, metrics
        
    except ValueError as e:
        logger.critical(f"❌ Error de validación de datos: {e}")
        raise
    except Exception as e:
        logger.critical(f"💥 Error durante el entrenamiento: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# FASE 3: REPORTING Y CIERRE
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary_report(
    dataset_name: str,
    model: MiningGP,
    metrics: ModelMetrics,
    data_filepath: Path
) -> None:
    """Genera el reporte final de calidad del modelo."""
    console.print(Panel.fit("📊 RESUMEN FINAL", style="bold green"))
    
    # Tabla Ejecutiva
    summary = Table(title="Auditoría del Modelo", show_header=True, header_style="bold cyan")
    summary.add_column("KPI", style="dim")
    summary.add_column("Resultado", style="white")
    
    summary.add_row("Dataset", dataset_name)
    summary.add_row("Modelo Usado", model.model_type)
    summary.add_row("Features Activos", str(len(model.feature_names)))
    
    # Semáforo de R²
    r2_val = metrics.r2
    r2_color = "green" if r2_val > 0.5 else "yellow" if r2_val > 0 else "red"
    summary.add_row("R² Score", f"[{r2_color}]{r2_val:.4f}[/{r2_color}]")
    summary.add_row("RMSE", f"{metrics.rmse:.4f}")
    summary.add_row("MAPE", f"{metrics.mape:.2f}%")
    
    console.print(summary)
    
    # Diagnóstico Final
    if r2_val < 0:
        console.print("\n[bold red]⛔ FALLO: El modelo no es predictivo.[/bold red]")
        console.print("[yellow]Acción sugerida:[/yellow] Revisa 'mining_gp_pro.py' y reduce 'noise_level' a 0.1.")
    elif r2_val > 0.7:
        console.print("\n[bold green]🏆 ÉXITO: Modelo de alta precisión listo para despliegue.[/bold green]")
    
    # Rutas de Salida
    console.print(f"\n[dim]Modelo guardado en: {CONFIG.MODELS_DIR}[/dim]")
    console.print(f"[dim]Reporte gráfico en: {CONFIG.RESULTS_DIR}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Función principal del pipeline."""
    try:
        console.print(Panel.fit(
            "🚀 Pipeline Universal de Entrenamiento v2.1\n"
            "Stability Fix Update\n"
            "Arquitectura Minera 4.0",
            style="bold blue"
        ))
        
        # --- PASO 1: DATOS ---
        data_filepath, adapter_config = prepare_data_with_adapter()
        
        # --- PASO 2: MODELADO ---
        # Usamos 50 trials y paso 10 para máxima estabilidad
        model, metrics = train_model_with_gp(
            data_filepath=data_filepath,
            adapter_config=adapter_config,
            n_trials=50,      
            subsample_step=10 
        )
        
        # --- PASO 3: REPORTE ---
        generate_summary_report(
            dataset_name=adapter_config['dataset_name'],
            model=model,
            metrics=metrics,
            data_filepath=data_filepath
        )
        
        # Exit Code (0=OK, 1=Fail)
        exit_code = 0 if metrics.r2 > 0 else 1
        logger.info(f"✅ Proceso finalizado. Exit Code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Interrumpido por usuario.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]🔥 ERROR FATAL: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()