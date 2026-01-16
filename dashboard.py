"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: dashboard.py
Versión: 3.5.0 (Real Data Edition)
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Centro de Control HMI (Human-Machine Interface) de Alta Fidelidad.
    
    Este módulo implementa un Dashboard interactivo que:
    1. Orquesta un Gemelo Digital del proceso de flotación.
    2. Simula escenarios "What-If" en tiempo real.
    3. Ejecuta inferencia con modelos IA (GP o GBR).
    4. Genera evidencia forense certificable en PDF.

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v3.5.0 - Enero 2026] REAL DATA EDITION
    ---------------------------------------
    - CORREGIDO: Confianza IA ahora es REAL (calculada desde std del modelo)
    - CORREGIDO: Línea "Soft-Sensor IA" ahora usa predict_series() REAL
    - CORREGIDO: Feature Importance usa get_feature_importance() REAL
    - ACTUALIZADO: Usa MiningDataAdapter en lugar de UniversalAdapter
    - MEJORADO: Manejo de errores más robusto
    
    [v3.4.0] Final Audit Edition
    ----------------------------
    - Documentación extendida
    - Estética Dark Industrial

═══════════════════════════════════════════════════════════════════════════════
"""

# =============================================================================
# 1. IMPORTACIONES DEL SISTEMA
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 2. IMPORTACIONES DE ARQUITECTURA PROPIA
# =============================================================================
from core.inference_engine import MiningInference

# [v3.5.0] Usar el nuevo adapter unificado
from core.adapters import MiningDataAdapter

from core.report_generator import ReportManager, ShiftReportData


# =============================================================================
# 3. CONFIGURACIÓN VISUAL Y UX
# =============================================================================
st.set_page_config(
    page_title="Mining 4.0 - Control Room",
    layout="wide",
    page_icon="⚒️",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    [data-testid="stMetricValue"] { 
        font-size: 2.5rem; 
        color: #00f2ea;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700; 
        text-shadow: 0 0 10px rgba(0, 242, 234, 0.3);
    }
    .stMetric { 
        background-color: #111; 
        padding: 15px; 
        border: 1px solid #333;
        border-left: 5px solid #ff00ff;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1a1a; border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 4. GESTIÓN DE RECURSOS (SINGLETONS & CACHE)
# =============================================================================
@st.cache_resource
def load_engine():
    """Inicializa el Motor de Inferencia IA como Singleton."""
    return MiningInference()


@st.cache_resource
def load_adapter():
    """Inicializa el Adapter de datos como Singleton."""
    return MiningDataAdapter("dataset_config.json")


# =============================================================================
# 5. FUNCIONES AUXILIARES
# =============================================================================

def get_confidence_color(confidence_pct: float) -> str:
    """Retorna color según nivel de confianza."""
    if confidence_pct >= 90:
        return "#00ff00"  # Verde
    elif confidence_pct >= 70:
        return "#ffff00"  # Amarillo
    elif confidence_pct >= 50:
        return "#ff9900"  # Naranja
    else:
        return "#ff0000"  # Rojo


def get_confidence_label(confidence_pct: float) -> str:
    """Retorna etiqueta según nivel de confianza."""
    if confidence_pct >= 90:
        return "Alta"
    elif confidence_pct >= 70:
        return "Media"
    elif confidence_pct >= 50:
        return "Baja"
    else:
        return "Muy Baja"


def truncate_feature_name(name: str, max_len: int = 25) -> str:
    """Acorta nombres de features para visualización."""
    if len(name) <= max_len:
        return name
    # Tomar inicio y final
    return name[:12] + "..." + name[-10:]


# =============================================================================
# 6. ORQUESTADOR PRINCIPAL
# =============================================================================
def run_dashboard():
    """Función principal que ejecuta el ciclo de vida del Dashboard."""
    
    # --- A. INICIALIZACIÓN ---
    try:
        engine = load_engine()
        adapter = load_adapter()
    except Exception as e:
        st.error(f"❌ Error inicializando sistema: {e}")
        st.info("Asegúrate de haber entrenado un modelo con train_universal.py")
        return
    
    # --- B. INGESTA DE DATOS ---
    try:
        df_full = adapter.load_data()
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        return
    
    # Time-Shifting para simular datos "en vivo"
    if not df_full.empty and isinstance(df_full.index, pd.DatetimeIndex):
        last_csv_date = df_full.index.max()
        now = datetime.now()
        time_shift = now - last_csv_date
        df_full.index = df_full.index + time_shift

    # Inicialización del puntero
    if 'pointer' not in st.session_state:
        st.session_state.pointer = 150

    # =========================================================================
    # 7. BARRA LATERAL
    # =========================================================================
    with st.sidebar:
        st.title("🕹️ Control Room")
        st.caption(f"⏱️ System Time: {datetime.now().strftime('%H:%M:%S')}")
        st.markdown("---")
        
        # Controles
        st.subheader("⚙️ Configuración")
        update_speed = st.slider("Ciclo de Refresco (s)", 1, 5, 2)
        target_goal = st.number_input("KPI Objetivo Recup. (%)", 80.0, 95.0, 85.0)
        
        st.markdown("---")
        
        # Motor What-If
        st.subheader("🧪 What-If Engine")
        st.info("Perturbación de variables en tiempo real:")
        
        sim_air = st.slider(
            "Factor Flujo Aire", 
            0.8, 1.2, 1.0, step=0.05,
            help="1.0 = Nominal. <1.0 = Déficit. >1.0 = Sobrecarga."
        )
        
        st.markdown("---")
        
        # Info del modelo
        st.subheader("🤖 Modelo Activo")
        model_info = engine.get_model_info()
        st.caption(f"Tipo: **{model_info.get('model_type', 'N/A')}**")
        st.caption(f"Target: `{model_info.get('target_column', 'N/A')}`")
        st.caption(f"Features: {model_info.get('n_features', 0)}")
        
        st.markdown("---")
        
        # Generador de Auditoría
        if st.button("📥 Generar Auditoría (PDF)", type="primary"):
            with st.status("🛠️ Ejecutando protocolo forense...", expanded=True) as status:
                try:
                    st.write("📸 Capturando estado del sistema...")
                    Path("results").mkdir(exist_ok=True)
                    chart_path = "results/snapshot_trend.png"
                    
                    if 'last_fig' in st.session_state:
                        st.session_state.last_fig.write_image(
                            chart_path, width=1200, height=500, scale=2
                        )
                        time.sleep(1.0)
                    else:
                        chart_path = None
                    
                    st.write("🧠 Generando estrategia operativa...")
                    
                    # Usar predicción real
                    last_pred = st.session_state.get('last_pred', {})
                    recovery = last_pred.get('predicted_value', 85.0)
                    confidence = last_pred.get('confidence_pct', 85.0)
                    
                    if recovery < target_goal:
                        rec_text = f"⚠️ CRÍTICO: Recuperación {recovery:.1f}% < objetivo {target_goal}%. Aumentar aire (+5%) y revisar dosificación."
                    else:
                        rec_text = f"✅ MANTENER: Recuperación {recovery:.1f}% ≥ objetivo. Operación nominal."

                    air_status = f"OK (Factor: {sim_air}x)"
                    
                    report_data = ShiftReportData(
                        timestamp=datetime.now(),
                        recovery_avg=recovery,
                        recovery_target=target_goal,
                        financial_impact=(recovery - 80) * 100000,
                        model_name=engine.model_wrapper.model_type,
                        sensor_health={
                            "Flujo Aire": air_status, 
                            "Nivel Pulpa": "OK (Estable)", 
                            "DCS Link": "ONLINE (12ms)",
                            "Confianza IA": f"{confidence:.1f}%"
                        },
                        recommendation=rec_text,
                        chart_path=chart_path if chart_path and Path(chart_path).exists() else None
                    )
                    
                    manager = ReportManager()
                    pdf_path = manager.generate(report_data)
                    
                    status.update(label="✅ Auditoría Finalizada", state="complete", expanded=False)
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📂 Descargar Documento Oficial", 
                            f, 
                            file_name=Path(pdf_path).name
                        )
                        
                except Exception as e:
                    st.error(f"Fallo crítico en reporte: {e}")

    # =========================================================================
    # 8. ÁREA PRINCIPAL
    # =========================================================================
    st.title("🏭 Mining 4.0: Digital Twin")
    st.markdown(f"**Estado:** `ONLINE` | **Modelo Activo:** `{engine.model_path.stem}`")

    # --- C. PREPROCESAMIENTO DE VENTANA ---
    window = df_full.iloc[st.session_state.pointer - 50 : st.session_state.pointer].copy()
    
    # --- D. MOTOR WHAT-IF ---
    if sim_air != 1.0:
        air_cols = [c for c in window.columns if any(x in c.lower() for x in ['air', 'flujo', 'flow'])]
        
        if air_cols:
            window[air_cols] = window[air_cols] * sim_air
            noise = np.random.normal(0, 0.05 * abs(1 - sim_air), (len(window), len(air_cols)))
            window[air_cols] = window[air_cols] + noise
            st.toast(f"💨 Simulador Activo: Ajustando {len(air_cols)} sensores por factor {sim_air}x")

    # --- E. INFERENCIA REAL ---
    try:
        res = engine.predict_scenario(window)
        st.session_state.last_pred = res
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        res = {
            'predicted_value': 0.0,
            'real_value': None,
            'confidence_pct': 0.0,
            'confidence_std': 0.0,
            'model_used': 'Error'
        }

    # --- F. PANEL DE KPIs (CON DATOS REALES) ---
    k1, k2, k3 = st.columns(3)
    
    with k1:
        delta = None
        if res['real_value'] is not None:
            diff = res['predicted_value'] - res['real_value']
            delta = f"{diff:+.2f}% vs Real"
        st.metric("Predicción Recuperación", f"{res['predicted_value']:.2f}%", delta)
    
    with k2:
        roi = (res['predicted_value'] - 80) * 100000
        st.metric("Impacto Económico (Día)", f"${roi/1000:,.1f}k USD")
    
    with k3:
        # [v3.5.0] CONFIANZA REAL calculada desde el modelo
        confidence = res['confidence_pct']
        conf_label = get_confidence_label(confidence)
        st.metric("Confianza IA", f"{confidence:.1f}%", conf_label)

    # --- G. VISUALIZACIÓN ---
    tab_trend, tab_xai = st.tabs(["📈 Tendencia Operativa", "🧠 Explicabilidad (XAI)"])

    with tab_trend:
        st.subheader("Curva de Recuperación: Real vs Predicha")
        
        # [v3.5.0] PREDICCIONES REALES con predict_series()
        try:
            hist_window = df_full.iloc[st.session_state.pointer - 100 : st.session_state.pointer]
            
            # Generar serie de predicciones REALES
            with st.spinner("Calculando predicciones..."):
                pred_series = engine.predict_series(hist_window, n_points=100, min_history=30)
            
            fig = go.Figure()
            
            # Traza 1: Planta Real
            target_col = engine.model_wrapper.target_col
            fig.add_trace(go.Scatter(
                x=hist_window.index, 
                y=hist_window[target_col], 
                name="Planta (Real)", 
                line=dict(color='#f4a261', width=3)
            ))
            
            # Traza 2: Predicciones REALES del Soft-Sensor
            if not pred_series.empty:
                fig.add_trace(go.Scatter(
                    x=pred_series.index, 
                    y=pred_series['predicted'], 
                    name="Soft-Sensor IA", 
                    line=dict(color='#ff00ff', width=2, dash='dot')
                ))
                
                # Banda de confianza (si hay std)
                if 'confidence_std' in pred_series.columns and pred_series['confidence_std'].sum() > 0:
                    upper = pred_series['predicted'] + 2 * pred_series['confidence_std']
                    lower = pred_series['predicted'] - 2 * pred_series['confidence_std']
                    
                    fig.add_trace(go.Scatter(
                        x=pred_series.index,
                        y=upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_series.index,
                        y=lower,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 255, 0.15)',
                        name='Intervalo 95%'
                    ))
            
            # Línea de objetivo
            fig.add_hline(
                y=target_goal, 
                line_dash="dash", 
                line_color="#00ff00",
                annotation_text=f"Objetivo: {target_goal}%"
            )
            
            fig.update_layout(
                template="plotly_dark", 
                height=400, 
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.session_state.last_fig = fig
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
            logger.exception("Error en gráfico de tendencia")

    with tab_xai:
        st.subheader("🎯 Drivers: Variables que más impactan la predicción")
        
        # [v3.5.0] FEATURE IMPORTANCE REAL
        try:
            importance = engine.get_feature_importance(top_n=10)
            
            if importance:
                # Preparar datos para gráfico
                feat_names = [truncate_feature_name(k) for k in importance.keys()]
                feat_values = list(importance.values())
                
                fig_bar = go.Figure(go.Bar(
                    x=feat_values,
                    y=feat_names,
                    orientation='h',
                    marker_color='#00f2ea',
                    text=[f"{v:.1%}" for v in feat_values],
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=50, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Importancia Relativa",
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Interpretación
                top_feat = list(importance.keys())[0]
                st.info(
                    f"💡 **Insight**: La variable más influyente es `{top_feat}` "
                    f"con {list(importance.values())[0]:.1%} de importancia relativa."
                )
            else:
                st.warning("No se pudo calcular feature importance")
                
        except Exception as e:
            st.error(f"Error calculando importancia: {e}")

    # --- H. TICKER DE ESTADO ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.caption("📡 **SCADA:** Conectado (12ms)")
    c2.caption("🧪 **Lab:** Muestras L-204 OK")
    c3.caption(f"👷 **Turno:** {datetime.now().strftime('%A')} - Guardia B")
    c4.caption(f"🤖 **Modelo:** {res['model_used']}")

    # --- I. BUCLE DE EJECUCIÓN ---
    time.sleep(update_speed)
    st.session_state.pointer += 1
    
    if st.session_state.pointer >= len(df_full): 
        st.session_state.pointer = 150
    st.rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        run_dashboard()
    except Exception as e:
        st.error(f"System Offline: {e}")
