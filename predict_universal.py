"""
═══════════════════════════════════════════════════════════════════════════════
Script: predict_universal.py
Proyecto: Universal Soft-Sensor
Versión: 1.2.1 — BUGFIX

HISTORIAL:
    [v1.2.1 - 2026]
        [FIX] TypeError en adapter.load_data(CONFIG.DATA_RAW_PATH).
              UniversalAdapter.load_data() no acepta argumentos posicionales.
              La ruta de datos viene definida desde el config JSON,
              no desde un argumento en tiempo de ejecución.
              ANTES: df_full = adapter.load_data(CONFIG.DATA_RAW_PATH)
              AHORA: df_full = adapter.load_data()

        [FIX] Construcción de config_path frágil.
              DataAdapter espera el NOMBRE del config (str filename),
              no un Path completo. El constructor resuelve la ruta internamente.
              ANTES: config_path = CONFIG.PROJECT_ROOT / "dataset_config.json"
                     adapter = DataAdapter(config_path)  ← TypeError
              AHORA: adapter = DataAdapter("dataset_config.json")

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
from pathlib import Path

# Librerías de UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Asegurar que el sistema encuentre los módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))

# Módulos del Proyecto
from core.inference_engine import InferenceEngine
from core.adapters import DataAdapter
from config.settings import CONFIG

# Configuración de Consola
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.NullHandler()]
)

def ejecutar_simulacion():
    """
    Orquestador de la simulación de inferencia.
    
    El flujo consiste en:
    1. Instanciar el motor de inferencia (carga el modelo .pkl).
    2. Conectar con la fuente de datos mediante el adaptador industrial.
    3. Definir escenarios de prueba (Último dato vs Dato aleatorio).
    4. Comparar resultados de la IA vs Sensores reales.
    """
    console.print(Panel.fit(
        "🔮 [bold white]SOFT-SENSOR MINERO[/bold white] | [italic]Simulation Mode v1.2.1[/italic]",
        style="purple",
        border_style="purple"
    ))

    try:
        # 1. INICIALIZACIÓN DEL MOTOR
        with console.status("[bold cyan]Cargando Motor de Inteligencia Artificial..."):
            engine = InferenceEngine()
        
        console.print(f"✔️  [dim]Modelo activo: {engine.model_path.name}[/dim]")
        console.print(f"✔️  [bold cyan]Algoritmo:[/bold cyan] {engine.model_wrapper.model_type}")

        # 2. CONEXIÓN A DATOS
        with console.status("[bold green]Accediendo a flujo de datos maestros..."):
            # [FIX] ANTES:
            #   config_path = CONFIG.PROJECT_ROOT / "dataset_config.json"
            #   adapter = DataAdapter(config_path)       ← Path object, no str filename
            #   df_full = adapter.load_data(CONFIG.DATA_RAW_PATH)  ← TypeError: 1 arg extra
            #
            # AHORA: DataAdapter espera el nombre del archivo (str),
            #         la ruta se resuelve internamente en el constructor.
            #         load_data() no acepta argumentos — la ruta viene del JSON.
            adapter = DataAdapter("dataset_config.json")
            df_full = adapter.load_data()
        
        # 3. DEFINICIÓN DE LA VENTANA DE SIMULACIÓN
        window_size = 100
        
        if len(df_full) < window_size:
            console.print(f"[bold red]❌ Error:[/bold red] Datos insuficientes (Mínimo {window_size} filas).")
            return

        # ESCENARIO A: Estado Actual (Últimos datos recibidos por el PLC)
        df_now = df_full.iloc[-window_size:]
        
        # ESCENARIO B: Validación Ciega (Punto aleatorio en la historia)
        random_idx = df_full.sample(1).index[0]
        pos = df_full.index.get_loc(random_idx)
        
        start_pos = max(0, pos - window_size + 1)
        df_random = df_full.iloc[start_pos : pos + 1]

        # 4. EJECUCIÓN DE INFERENCIA
        results = []
        
        with console.status("[bold yellow]Procesando Escenario Producción..."):
            res_now = engine.predict_scenario(df_now)
            res_now["nombre"] = "Último Registro (Producción)"
            results.append(res_now)
        
        with console.status("[bold yellow]Procesando Escenario Aleatorio..."):
            res_rnd = engine.predict_scenario(df_random)
            res_rnd["nombre"] = "Muestra de Control (Histórica)"
            results.append(res_rnd)

        # 5. REPORTE VISUAL DE RESULTADOS
        table = Table(
            title="\n[bold]TABLA DE PRECISIÓN: IA vs SENSORES REALES[/bold]",
            title_justify="left",
            header_style="bold magenta"
        )
        
        table.add_column("Escenario de Prueba", style="cyan", width=30)
        table.add_column("Timestamp", style="dim")
        table.add_column("Predicción Soft-Sensor", justify="right", style="bold green")
        table.add_column("Medición Real", justify="right", style="bold yellow")
        table.add_column("Error Relativo", justify="right")

        for r in results:
            pred = r['predicted_value']
            real = r['real_value']
            diff = abs(pred - real)
            error_pct = (diff / real * 100) if real != 0 else 0
            
            color = "green" if error_pct < 5 else "yellow" if error_pct < 15 else "red"
            
            table.add_row(
                r['nombre'],
                str(r['timestamp']),
                f"{pred:.4f}",
                f"{real:.4f}",
                f"[{color}]{error_pct:.2f}% (Δ {diff:.4f})[/{color}]"
            )

        console.print(table)
        console.print(
            f"\n[italic dim]Nota: Basado en el Subsample Step de "
            f"{CONFIG.DEFAULT_SUBSAMPLE_STEP} definido en settings.py[/italic dim]"
        )

    except Exception as e:
        console.print(Panel(f"[bold red]FALLO CRÍTICO EN SIMULACIÓN[/bold red]\n{str(e)}", title="Error"))
        logging.exception("Detalle técnico del error:")

if __name__ == "__main__":
    ejecutar_simulacion()
