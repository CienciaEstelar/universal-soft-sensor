"""
═══════════════════════════════════════════════════════════════════════════════
Script: train_universal.py
Proyecto: Arquitectura Minera 4.0
Autor: Juan Galaz
Versión: 2.2.0
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Orquestador Universal del Pipeline de Entrenamiento (MLOps).
    
    Este script es el "Director de Orquesta" que conecta:
    1. INGESTA: UniversalAdapter (Carga inteligente basada en JSON)
    2. MODELADO: MiningGP v4.1 (Gaussian Process con corrección de ruido)
    3. PERSISTENCIA: Guardado automático de modelos y métricas.

    El flujo completo es:
    
        CSV → UniversalAdapter → MiningGP → Modelo.pkl + Reporte.png
        
═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.2.0 - Enero 2026] CLEAN CODE UPDATE
    ----------------------------------------
    
    ✅ FIX: Subsample centralizado en CONFIG
    
       ANTES:
           # Línea ~95 (hardcodeado)
           subsample_step = 10  # ❌ Valor "mágico" suelto en el código
       
       AHORA:
           # Importamos desde config/settings.py
           from config.settings import CONFIG
           
           # Y usamos el valor centralizado
           # (MiningGP ya lo usa por defecto, no necesitamos pasarlo explícitamente)
       
       RAZÓN:
           Antes, si querías cambiar el subsample tenías que editar:
           - train_universal.py (valor 10)
           - inference.py (valor 50)  
           - mining_gp_pro.py (valor 50)
           
           Ahora solo editas config/settings.py y todo el sistema cambia.
    
    ✅ LIMPIEZA: Eliminado código redundante
       - Ya no pasamos subsample_step explícito a MiningGP
       - MiningGP usa CONFIG.DEFAULT_SUBSAMPLE_STEP por defecto

    [v2.1.0] Integración con UniversalAdapter
    [v2.0.0] Migración a MiningGP
    [v1.0.0] Versión inicial

═══════════════════════════════════════════════════════════════════════════════
USO:
═══════════════════════════════════════════════════════════════════════════════

    # Desde línea de comandos (modo estándar)
    python train_universal.py
    
    # El script lee la configuración desde:
    # - config/dataset_config.json  (reglas del dataset)
    # - config/settings.py          (parámetros globales)

═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════
import logging
import sys
from pathlib import Path
from datetime import datetime

# Rich - Interfaz de usuario profesional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Módulos internos de la Arquitectura Minera
from core.adapters.universal_adapter import UniversalAdapter
from core.models.mining_gp_pro import MiningGP, ModelMetrics
from config.settings import CONFIG  # ← [v2.2.0] Importamos la configuración centralizada

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LOGGING
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Universal_Trainer_v2.2")
console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data_with_adapter(config_filename: str = "dataset_config.json") -> tuple:
    """
    Carga y limpia datos usando UniversalAdapter.
    
    El UniversalAdapter actúa como un "firewall de datos":
    - Lee reglas desde un archivo JSON
    - Filtra columnas según patrones (include/exclude)
    - Elimina columnas que causan data leakage
    - Aplica forward-fill para valores nulos
    
    Args:
        config_filename: Nombre del archivo de configuración JSON
                        (debe estar en config/)
    
    Returns:
        Tuple de (ruta_csv_temporal, configuración_dict)
        
    Raises:
        FileNotFoundError: Si no encuentra el JSON o el CSV
        KeyError: Si el JSON está mal formado
    """
    console.print(Panel.fit(
        "📥 FASE 1: Carga y Filtrado de Datos", 
        style="bold cyan"
    ))
    
    try:
        # Inicializar adaptador (lee el JSON automáticamente)
        adapter = UniversalAdapter(config_filename)
        logger.info(f"📋 Dataset configurado: {adapter.config['dataset_name']}")
        
        # Cargar y filtrar datos según reglas del JSON
        df = adapter.load_data()
        
        # Mostrar estadísticas de carga
        console.print(f"\n[bold green]✅ Datos cargados y filtrados exitosamente:[/bold green]")
        
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_row("Registros:", f"{len(df):,}")
        stats_table.add_row("Features:", f"{len(df.columns) - 1}")  # -1 por el target
        stats_table.add_row("Target:", adapter.config["modeling"]["target_column"])
        stats_table.add_row("Rango temporal:", f"{df.index.min()} → {df.index.max()}")
        console.print(stats_table)
        
        # Guardar CSV temporal para que MiningGP lo lea
        # (MiningGP espera un archivo, no un DataFrame directamente)
        CONFIG.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = adapter.config['dataset_name']
        temp_filepath = CONFIG.DATA_PROCESSED_DIR / f"{dataset_name}_filtered_{timestamp}.csv"
        
        df.to_csv(temp_filepath)
        logger.info(f"💾 CSV temporal guardado: {temp_filepath}")
        
        return temp_filepath, adapter.config
        
    except FileNotFoundError as e:
        logger.critical(f"❌ Archivo no encontrado: {e}")
        raise
    except KeyError as e:
        logger.critical(f"❌ JSON mal formado - falta clave: {e}")
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
    n_trials: int = 50
) -> tuple:
    """
    Ejecuta el entrenamiento del Soft-Sensor usando MiningGP v4.1.
    
    Esta función es el corazón del pipeline. Conecta los datos preparados
    con el modelo de Gaussian Process y maneja todo el ciclo de vida:
    - Diagnóstico de autocorrelación
    - Optimización de hiperparámetros (Optuna)
    - Entrenamiento final
    - Evaluación y generación de reportes
    
    ═══════════════════════════════════════════════════════════════════════
    [v2.2.0] NOTA IMPORTANTE SOBRE SUBSAMPLE:
    ═══════════════════════════════════════════════════════════════════════
    
    ANTES (v2.1.0 y anteriores):
        # El valor estaba hardcodeado aquí
        model = MiningGP(
            target_col=target_col,
            subsample_step=10,  # ❌ Hardcode - fácil de olvidar actualizar
            ...
        )
    
    AHORA (v2.2.0):
        # MiningGP usa CONFIG.DEFAULT_SUBSAMPLE_STEP por defecto
        # No necesitamos pasar el valor explícitamente
        model = MiningGP(
            target_col=target_col,
            # subsample_step usa CONFIG automáticamente ✅
            ...
        )
    
    Si necesitas un subsample diferente para este entrenamiento específico,
    puedes pasarlo explícitamente:
        model = MiningGP(target_col=target_col, subsample_step=20)
    
    ═══════════════════════════════════════════════════════════════════════
    
    Args:
        data_filepath: Ruta al CSV filtrado (output de Fase 1)
        adapter_config: Diccionario con configuración del dataset
        n_trials: Número de trials para optimización Optuna
        
    Returns:
        Tuple de (modelo_entrenado, métricas)
    """
    console.print(Panel.fit(
        "🧠 FASE 2: Entrenamiento con MiningGP v4.1", 
        style="bold yellow"
    ))
    
    try:
        # Extraer columna objetivo del config
        target_col = adapter_config["modeling"]["target_column"]
        
        # ═══════════════════════════════════════════════════════════════════
        # [v2.2.0] CREACIÓN DEL MODELO
        # ═══════════════════════════════════════════════════════════════════
        # Ya NO pasamos subsample_step explícitamente.
        # MiningGP lo toma de CONFIG.DEFAULT_SUBSAMPLE_STEP por defecto.
        #
        # Esto garantiza que entrenamiento e inferencia usen el mismo valor,
        # evitando el bug de desalineación de features que teníamos antes.
        # ═══════════════════════════════════════════════════════════════════
        model = MiningGP(
            target_col=target_col,
            # subsample_step: usa CONFIG.DEFAULT_SUBSAMPLE_STEP (actualmente 10)
            add_lag_features=True,           # Agregar Y(t-1), Y(t-5), etc.
            add_diff_features=True,          # Agregar diferencias y rolling stats
            use_fallback_model=True,         # Usar GradientBoosting si GP falla
            remove_constant_features=True,   # Eliminar features con std ≈ 0
            remove_correlated_features=True  # Eliminar features redundantes
        )
        # ═══════════════════════════════════════════════════════════════════
        
        # Log de configuración
        logger.info(f"🎯 Target columna: {target_col}")
        logger.info(
            f"🔧 Configuración: "
            f"Optuna Trials={n_trials} | "
            f"Subsample=1/{CONFIG.DEFAULT_SUBSAMPLE_STEP} (desde CONFIG)"
        )
        
        # Ejecutar pipeline completo de entrenamiento
        metrics = model.train_from_file(
            filepath=str(data_filepath),
            test_size=0.2,      # 80% train, 20% test
            n_trials=n_trials,  # Número de experimentos Optuna
            save_model=True     # Guardar .pkl automáticamente
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
    """
    Genera el reporte final de calidad del modelo.
    
    Muestra un resumen ejecutivo con:
    - Dataset utilizado
    - Tipo de modelo final (GP o GradientBoosting)
    - Métricas clave con interpretación
    - Rutas de archivos generados
    
    Args:
        dataset_name: Nombre del dataset procesado
        model: Instancia del modelo entrenado
        metrics: Métricas de evaluación
        data_filepath: Ruta al CSV usado para entrenamiento
    """
    console.print(Panel.fit("📊 RESUMEN EJECUTIVO", style="bold green"))
    
    # Tabla de auditoría
    summary = Table(
        title="Auditoría del Modelo Entrenado", 
        show_header=True, 
        header_style="bold cyan"
    )
    summary.add_column("KPI", style="dim")
    summary.add_column("Resultado", style="white")
    summary.add_column("Estado", style="white")
    
    # Fila 1: Dataset
    summary.add_row("Dataset", dataset_name, "✅")
    
    # Fila 2: Tipo de modelo
    model_emoji = "🧠" if model.model_type == "GP" else "🌲"
    summary.add_row("Modelo Final", f"{model_emoji} {model.model_type}", "✅")
    
    # Fila 3: Features
    summary.add_row("Features Activos", str(len(model.feature_names)), "✅")
    
    # Fila 4: R² con color según calidad
    r2_val = metrics.r2
    if r2_val > 0.7:
        r2_display = f"[bold green]{r2_val:.4f}[/bold green]"
        r2_status = "🏆 Excelente"
    elif r2_val > 0.5:
        r2_display = f"[yellow]{r2_val:.4f}[/yellow]"
        r2_status = "👍 Aceptable"
    else:
        r2_display = f"[red]{r2_val:.4f}[/red]"
        r2_status = "⚠️ Revisar"
    summary.add_row("R² Score", r2_display, r2_status)
    
    # Fila 5: RMSE
    summary.add_row("RMSE", f"{metrics.rmse:.4f}", "-")
    
    # Fila 6: MAPE
    mape_status = "✅" if metrics.mape < 10 else "⚠️"
    summary.add_row("MAPE", f"{metrics.mape:.2f}%", mape_status)
    
    console.print(summary)
    
    # Mensaje final según resultado
    if r2_val < 0:
        console.print("\n[bold red]⛔ FALLO: El modelo no es predictivo (R² negativo).[/bold red]")
        console.print("[dim]   Posibles causas: datos insuficientes, target muy ruidoso, data leakage.[/dim]")
    elif r2_val > 0.7:
        console.print("\n[bold green]🏆 ÉXITO: Modelo de alta precisión listo para despliegue.[/bold green]")
    else:
        console.print("\n[yellow]⚠️ PRECAUCIÓN: Modelo funcional pero con margen de mejora.[/yellow]")
        console.print("[dim]   Sugerencia: Revisar features, probar más trials de Optuna.[/dim]")
    
    # Mostrar rutas de archivos generados
    console.print(f"\n[dim]📁 Archivos generados:[/dim]")
    console.print(f"[dim]   • Modelo: {CONFIG.MODELS_DIR}/*.pkl[/dim]")
    console.print(f"[dim]   • Reporte: {CONFIG.RESULTS_DIR}/*.png[/dim]")
    console.print(f"[dim]   • Datos: {data_filepath}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL (MAIN)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Función principal del pipeline de entrenamiento.
    
    Orquesta las 3 fases:
    1. Ingesta de datos (UniversalAdapter)
    2. Entrenamiento del modelo (MiningGP)
    3. Generación de reportes
    
    Exit codes:
        0: Éxito (R² > 0)
        1: Fallo (R² ≤ 0 o error)
        130: Interrumpido por usuario (Ctrl+C)
    """
    try:
        # Banner de inicio
        console.print(Panel.fit(
            "🚀 Pipeline Universal de Entrenamiento v2.2.0\n"
            "Clean Code Update - Enero 2026\n"
            "Arquitectura Minera 4.0",
            style="bold blue"
        ))
        
        # Mostrar configuración actual
        console.print(f"[dim]📋 Configuración activa:[/dim]")
        console.print(f"[dim]   • DEFAULT_SUBSAMPLE_STEP = {CONFIG.DEFAULT_SUBSAMPLE_STEP}[/dim]")
        console.print(f"[dim]   • GP_TARGET = {CONFIG.GP_TARGET_COLUMN}[/dim]")
        console.print(f"[dim]   • GP_TRIALS = {CONFIG.GP_OPTUNA_TRIALS}[/dim]")
        console.print()
        
        # === PASO 1: INGESTA DE DATOS ===
        data_filepath, adapter_config = prepare_data_with_adapter()
        
        # === PASO 2: ENTRENAMIENTO DEL MODELO ===
        model, metrics = train_model_with_gp(
            data_filepath=data_filepath,
            adapter_config=adapter_config,
            n_trials=50  # Puedes ajustar esto
        )
        
        # === PASO 3: REPORTE FINAL ===
        generate_summary_report(
            dataset_name=adapter_config['dataset_name'],
            model=model,
            metrics=metrics,
            data_filepath=data_filepath
        )
        
        # Determinar exit code
        exit_code = 0 if metrics.r2 > 0 else 1
        logger.info(f"✅ Pipeline finalizado. Exit Code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Proceso interrumpido por usuario (Ctrl+C).[/yellow]")
        sys.exit(130)
        
    except Exception as e:
        console.print(f"\n[bold red]🔥 ERROR FATAL: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
