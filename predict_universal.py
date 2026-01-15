"""
═══════════════════════════════════════════════════════════════════════════════
Script: predict_universal.py
Autor: Juan Galaz (Arquitectura Minera 4.0)
Versión: 1.1 (Documentación Extendida)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Herramienta CLI (Command Line Interface) para Simulación de Inferencia.
    
    Este script permite probar el modelo entrenado "como si" estuviera en producción.
    Utiliza el dataset histórico para simular la llegada de nuevos datos y 
    comparar la predicción de la IA contra lo que realmente ocurrió.

OBJETIVOS:
    1. Validar que el modelo (.pkl) se puede cargar correctamente.
    2. Verificar que la lógica de generación de features (lags) funciona.
    3. Medir visualmente la precisión en un dato "nuevo" (fuera de muestra).

USO:
    python predict_universal.py
"""

import sys
import logging
import pandas as pd

# Librerías de UI (Rich)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Módulos del Proyecto
from core.inference_engine import MiningInference
from core.adapters.universal_adapter import UniversalAdapter
from config.settings import CONFIG

# Configuración Visual
console = Console()
# Solo mostramos errores críticos para no ensuciar la tabla de resultados
logging.basicConfig(level=logging.ERROR) 

def main():
    """
    Función principal de simulación.
    
    Flujo:
    1. Carga el Motor de Inferencia (MiningInference).
    2. Carga datos históricos para simular el flujo de sensores.
    3. Selecciona dos escenarios (Último dato y Dato aleatorio).
    4. Ejecuta predicciones y muestra tabla comparativa.
    """
    console.print(Panel.fit("🔮 SISTEMA DE INFERENCIA MINERA v1.0", style="bold purple"))

    try:
        # ---------------------------------------------------------------------
        # PASO 1: Inicialización del Motor
        # ---------------------------------------------------------------------
        # El motor busca automáticamente el modelo más reciente en models/
        engine = MiningInference()
        
        console.print(f"[dim]Modelo cargado: {engine.model_path.name}[/dim]")
        console.print(f"[bold cyan]Algoritmo Activo:[/bold cyan] {engine.model_wrapper.model_type}")

        # ---------------------------------------------------------------------
        # PASO 2: Simulación de Sensores (Carga de Datos)
        # ---------------------------------------------------------------------
        # En un entorno real, esto se reemplazaría por una conexión a SQL/PI System/Kafka.
        # Aquí usamos el UniversalAdapter para leer el CSV y simular el presente.
        with console.status("[bold green]Conectando con flujo de datos (Simulado)...[/bold green]"):
            # Leemos la configuración para saber qué archivo cargar
            adapter = UniversalAdapter("dataset_config.json")
            df_full = adapter.load_data()
        
        # ---------------------------------------------------------------------
        # PASO 3: Definición de Escenarios de Prueba
        # ---------------------------------------------------------------------
        # Necesitamos una "ventana" de datos pasados para calcular lags (ej. t-1, t-10).
        # Si el modelo usa 'lag_20', necesitamos al menos 21 filas.
        simulation_window = 50 
        
        if len(df_full) < simulation_window:
            console.print("[red]❌ Error: Dataset insuficiente para calcular lags históricos.[/red]")
            return

        # --- Escenario A: El "Ahora" (Producción) ---
        # Tomamos las últimas 50 filas del archivo como si fueran los últimos 50 minutos.
        df_scenario_now = df_full.iloc[-simulation_window:]
        
        # --- Escenario B: Prueba Ciega (Validación Random) ---
        # Tomamos un punto aleatorio en la historia para ver si el modelo generaliza.
        # Nos aseguramos de tener espacio suficiente hacia atrás para la ventana.
        random_idx = df_full.sample(1).index[0]
        idx_pos = df_full.index.get_loc(random_idx)
        
        df_scenario_random = None
        if idx_pos > simulation_window:
             # Cortamos desde (índice - 50) hasta (índice)
             df_scenario_random = df_full.iloc[idx_pos-simulation_window : idx_pos+1]

        # ---------------------------------------------------------------------
        # PASO 4: Ejecución de Inferencia
        # ---------------------------------------------------------------------
        results = []
        
        # Predicción A
        pred_now = engine.predict_scenario(df_scenario_now)
        pred_now["escenario"] = "Último Registro (Fin del Dataset)"
        results.append(pred_now)
        
        # Predicción B (si fue posible generar el escenario)
        if df_scenario_random is not None:
            pred_random = engine.predict_scenario(df_scenario_random)
            pred_random["escenario"] = "Muestra Aleatoria (Validación Ciega)"
            results.append(pred_random)

        # ---------------------------------------------------------------------
        # PASO 5: Reporte de Resultados
        # ---------------------------------------------------------------------
        table = Table(title="Reporte de Predicción en Tiempo Real", show_header=True)
        table.add_column("Escenario", style="cyan")
        table.add_column("Fecha/Hora (Simulada)", style="dim")
        table.add_column("Predicción IA", style="bold green")
        table.add_column("Valor Real", style="bold yellow")
        table.add_column("Desviación (Error)", style="bold white")

        for res in results:
            real = res['real_value']
            pred = res['predicted_value']
            
            # Cálculo de error porcentual
            diff = abs(real - pred)
            error_pct = (diff / real) * 100 if real != 0 else 0
            
            # Semáforo de precisión (Verde < 5%, Amarillo < 15%, Rojo > 15%)
            color_diff = "green" if error_pct < 5 else "yellow" if error_pct < 15 else "red"
            
            table.add_row(
                res['escenario'],
                res['timestamp'],
                f"{pred:.4f}",
                f"{real:.4f}",
                f"[{color_diff}]{diff:.4f} ({error_pct:.2f}%)[/{color_diff}]"
            )

        console.print(table)
        console.print("\n[dim]Nota: Una desviación en [green]verde[/green] indica que el Soft-Sensor es preciso.[/dim]")

    except Exception as e:
        console.print(f"[bold red]🔥 Error fatal en simulación: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()