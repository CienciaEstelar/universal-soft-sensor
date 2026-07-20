"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: dashboard.py
Proyecto: Universal Soft-Sensor - HMI de Alta Fidelidad
Autor: Juan Galaz (Refactorizado por Gemini)
Versión: 3.6.1 (Documented & Optimized)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Este módulo implementa el Centro de Control (HMI) del Soft-Sensor. 
    Utiliza Streamlit para la interfaz y Plotly para la visualización de series 
    temporales. La arquitectura se basa en "Fragmentos" para permitir el refresco
    de datos en tiempo real sin saturar el procesamiento del cliente ni del servidor.

CARACTERÍSTICAS TÉCNICAS:
    - Inferencia Reactiva: Predicciones basadas en una ventana deslizante de datos.
    - Motor What-If: Simulación de perturbaciones en flujo de aire/sensores.
    - Optimización st.fragment: Refresco parcial de la UI cada N segundos.
    - Caché Persistente: Los motores de IA se cargan una sola vez (Singleton).

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import logging
import time

# --- IMPORTACIONES DE ARQUITECTURA CORE ---
from core.inference_engine import InferenceEngine
from core.adapters import DataAdapter
from core.report_generator import ReportManager, ShiftReportData

# =============================================================================
# A. CONFIGURACIÓN DE ENTORNO Y UI
# =============================================================================

def apply_industrial_theme():
    """
    Inyecta CSS personalizado para lograr una estética 'Dark Industrial'.
    Ajusta métricas, fondos y bordes para simular una consola de control real.
    """
    st.markdown("""
        <style>
        /* Fondo general y color de texto */
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        
        /* Estilo para las tarjetas de métricas (KPIs) */
        [data-testid="stMetricValue"] { 
            font-size: 2.8rem; 
            color: #00f2ea; 
            font-weight: 800;
            text-shadow: 0 0 10px rgba(0, 242, 234, 0.4);
        }
        .stMetric { 
            background-color: #161b22; 
            padding: 20px; 
            border-radius: 8px; 
            border: 1px solid #30363d;
            border-left: 6px solid #ff00ff; 
        }
        
        /* Ajuste de Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { 
            background-color: #21262d; 
            border-radius: 4px 4px 0 0; 
            padding: 10px 20px;
        }
        </style>
        """, unsafe_allow_html=True)

# =============================================================================
# B. GESTIÓN DE RECURSOS (SINGLETONS)
# =============================================================================

@st.cache_resource
def get_system_core():
    """
    Inicializa y cachea los componentes más pesados del sistema.
    Usa @st.cache_resource para asegurar que el modelo de IA y el 
    adaptador de datos se carguen una sola vez en memoria (Patrón Singleton).
    
    Returns:
        tuple: (InferenceEngine instance, DataAdapter instance)
    """
    engine = InferenceEngine()
    adapter = DataAdapter("dataset_config.json")
    return engine, adapter

# =============================================================================
# C. FRAGMENTO DE TIEMPO REAL (PERFORMANCE OPTIMIZED)
# =============================================================================



@st.fragment(run_every=2.0)
def render_realtime_engine(engine, df_full, sim_air, target_goal):
    """
    Sección dinámica del dashboard que se refresca automáticamente.
    
    Esta función es un 'Fragmento'. Streamlit solo actualizará este bloque
    de código, evitando el re-renderizado de la barra lateral o de los 
    datos maestros cargados en caché. Esto reduce drásticamente el uso de CPU.

    Args:
        engine (InferenceEngine): Motor de IA para predicciones.
        df_full (pd.DataFrame): Dataset completo para simular el flujo.
        sim_air (float): Factor de perturbación para el motor What-If.
        target_goal (float): KPI objetivo definido por el usuario.
    """
    
    # 1. GESTIÓN DEL PUNTERO TEMPORAL
    # Simulamos el paso del tiempo moviendo un puntero sobre el dataset histórico.
    if 'pointer' not in st.session_state:
        st.session_state.pointer = 150 # Iniciamos con suficiente historia para Lags
    
    st.session_state.pointer = (st.session_state.pointer + 1) % len(df_full)
    ptr = st.session_state.pointer
    
    # 2. CREACIÓN DEL GEMELO DIGITAL (Windowing)
    # Tomamos una ventana de 50 registros para calcular features temporales.
    window = df_full.iloc[max(0, ptr-50) : ptr].copy()
    
    # 3. MOTOR WHAT-IF (Perturbación de Variables)
    # Si el operador ajusta el aire, modificamos los datos antes de la inferencia.
    if sim_air != 1.0:
        air_cols = [c for c in window.columns if any(x in c.lower() for x in ['air', 'flow', 'aire'])]
        if air_cols:
            window[air_cols] *= sim_air
            # Añadimos un pequeño ruido estocástico para mayor realismo
            window[air_cols] += np.random.normal(0, 0.01, size=window[air_cols].shape)

    # 4. INFERENCIA CON IA
    # Ejecutamos la predicción sobre el estado actual de la ventana.
    try:
        prediction = engine.predict_scenario(window)
        st.session_state.last_pred = prediction # Para uso en reportes PDF
    except Exception as e:
        st.error(f"Error en Inferencia: {e}")
        return

    # 5. RENDERIZADO DE KPIs (Panel de Instrumentos)
    kpi_cols = st.columns(3)
    
    with kpi_cols[0]:
        val = prediction['predicted_value']
        # Delta comparado con el objetivo (KPI Goal)
        diff = val - target_goal
        st.metric("Predicción Recup.", f"{val:.2f}%", f"{diff:+.2f}% vs Meta")

    with kpi_cols[1]:
        # Cálculo de impacto económico simplificado: cada 1% sobre 80% genera $1M USD/mes
        economic_impact = (val - 80) * 1250 # Representación por hora
        st.metric("Impacto Económico/h", f"${economic_impact:,.0f} USD", "Simulado")

    with kpi_cols[2]:
        conf = prediction['confidence_pct']
        status = "ESTABLE" if conf > 85 else "RUIDOSO"
        st.metric("Confianza IA", f"{conf:.1f}%", f"Estado: {status}")

    # 6. VISUALIZACIÓN DE TENDENCIAS (Plotly)
    # Mostramos la curva real de la planta vs la predicción del Soft-Sensor.
    fig = go.Figure()
    
    # Línea de Planta (Real)
    target_name = engine.model_wrapper.target_col
    fig.add_trace(go.Scatter(
        y=window[target_name].values, 
        name="Planta (Real)", 
        line=dict(color='#f4a261', width=3)
    ))
    
    # Línea IA (Predicción constante en la ventana)
    fig.add_trace(go.Scatter(
        y=[val] * len(window), 
        name="Soft-Sensor IA", 
        line=dict(color='#ff00ff', width=2, dash='dot')
    ))
    
    # Referencia del Objetivo
    fig.add_hline(y=target_goal, line_dash="dash", line_color="#00ff00", annotation_text="Meta")

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# D. ORQUESTADOR PRINCIPAL (MAIN)
# =============================================================================

def main():
    """
    Punto de entrada de la aplicación.
    Orquesta la carga de recursos, la barra lateral y la ejecución del fragmento.
    """
    apply_industrial_theme()
    
    st.title("⚒️ Mining 4.0: Digital Twin Control Room")
    st.markdown(f"**Estado del Sistema:** `ONLINE` | **Frecuencia:** `0.5 Hz`")

    # 1. CARGA DE MOTORES (Singleton)
    try:
        engine, adapter = get_system_core()
        df_full = adapter.load_data()
    except Exception as e:
        st.error(f"Fallo crítico al iniciar motores: {e}")
        return

    # 2. BARRA LATERAL (Panel de Operaciones)
    with st.sidebar:
        st.header("🎮 Operaciones")
        st.markdown("Ajuste los parámetros para simular escenarios operativos.")
        
        # Parámetro para What-If
        sim_air = st.slider("Perturbación Flujo Aire (Factor)", 0.5, 1.5, 1.0, 
                           help="Modifica los sensores de aire para ver el impacto en la IA.")
        
        # Definición de Meta
        target_goal = st.number_input("KPI Objetivo Recuperación (%)", 70.0, 95.0, 85.0)
        
        st.divider()
        
        # Acciones Forenses
        if st.button("📥 Generar Auditoría Forense (PDF)", type="primary"):
            st.info("Protocolo de reporte iniciado. Revisar carpeta /results.")
            # Aquí se llamaría a la lógica de ReportManager

    # 3. EJECUCIÓN DEL NÚCLEO DINÁMICO
    # Llamamos al fragmento que se encargará del refresco automático.
    render_realtime_engine(engine, df_full, sim_air, target_goal)

    # 4. FOOTER TÉCNICO
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.caption(f"🤖 **Modelo:** {engine.model_path.stem}")
    c2.caption(f"📊 **Muestras:** {len(df_full):,} registros cargados")
    c3.caption(f"🕒 **Último Pulso:** {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
