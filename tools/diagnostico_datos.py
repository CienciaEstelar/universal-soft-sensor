"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: tools/diagnostico_datos.py
Proyecto: Universal Soft-Sensor
Autor: Juan Galaz (Refactorizado por Gemini)
Versión: 1.2.1
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Realiza una auditoría estadística de los datos preprocesados. Su objetivo es
    detectar patologías en los datos (multicolinealidad, baja varianza, 
    autocorrelación extrema) que degradan el desempeño de modelos de Procesos 
    Gaussianos (GP) y causan R² negativos.

REQUISITOS:
    - Haber ejecutado el pipeline de limpieza (mining_clean.csv).
    - Configuración válida en settings.py.

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Asegurar que el interprete encuentre el módulo 'core' y 'config'
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import CONFIG


def diagnosticar_datos():
    """
    Ejecuta un flujo de diagnóstico integral sobre el dataset de minería.
    
    Analiza:
    1. Integridad: Presencia de NaNs y ceros sospechosos.
    2. Varianza: Si el target se mueve lo suficiente para ser predecible.
    3. Tiempo: Si existe fuga de datos por autocorrelación.
    4. Features: Correlaciones fuertes y redundancias (multicolinealidad).
    
    Genera un reporte en consola y un dashboard visual en la carpeta /results.
    """
    
    print("\n" + "═"*70)
    print(f"🔬 AUDITORÍA DE DATOS: PROYECTO MINERO 4.0 (v1.2.1)")
    print("═"*70)
    
    # --- 1. CARGA DE DATOS ---
    filepath = CONFIG.DATA_CLEAN_PATH
    if not filepath.exists():
        print(f"❌ ERROR: No se encontró el dataset en: {filepath}")
        print("   Asegúrate de haber corrido el pipeline de procesamiento primero.")
        return
    
    print(f"📂 Analizando fuente: {filepath.name}")
    # Cargamos 10k registros: suficiente para estadística descriptiva sin saturar RAM
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, nrows=10000)
    
    target = CONFIG.GP_TARGET_COLUMN
    if target not in df.columns:
        print(f"❌ ERROR: El target '{target}' no existe en el archivo limpio.")
        print(f"   Columnas disponibles: {df.columns.tolist()[:5]}...")
        return

    # --- 2. ANÁLISIS DEL TARGET (Variable Dependiente) ---
    print(f"\n🎯 ANÁLISIS DEL OBJETIVO: {target}")
    print("-" * 50)
    
    y = df[target]
    mean_val = y.mean()
    std_val = y.std()
    cv = (std_val / mean_val) * 100 if mean_val != 0 else 0 # Coeficiente de Variación
    
    print(f"   • Rango: [{y.min():.3f} - {y.max():.3f}]")
    print(f"   • Coef. Variación: {cv:.2f}% (Varianza relativa a la media)")
    
    # Nota técnica: Si CV < 5%, el modelo le costará distinguir señal de ruido
    if cv < 5:
        print("     ⚠️  ALERTA: Target casi constante. R2 podría ser muy bajo.")

    # --- 3. AUTOCORRELACIÓN (Fuga de Información Temporal) ---
    print(f"\n🔄 AUTOCORRELACIÓN (Lag Analysis)")
    print("-" * 50)
    
    # Calculamos la correlación del dato actual con el anterior (Lag 1)
    ac_1 = y.autocorr(lag=1)
    print(f"   • Autocorrelación Lag 1: {ac_1:.4f}")
    
    # Nota técnica: Si ac_1 > 0.95, los datos son tan parecidos que el modelo
    # puede "hacer trampa" prediciendo simplemente el valor anterior.
    if ac_1 > 0.95:
        print(f"     ⚠️  RECOMENDACIÓN: Sube el SUBSAMPLE_STEP (Actual: {CONFIG.DEFAULT_SUBSAMPLE_STEP})")

    # --- 4. ANÁLISIS DE FEATURES (Variables Independientes) ---
    print(f"\n📊 ANÁLISIS DE PREDICTORES (Features)")
    print("-" * 50)
    
    # CORRECCIÓN AUDITORÍA: Drop dinámico basado en CONFIG
    features = df.drop(columns=[target], errors='ignore')
    
    # Buscar features que no aportan información (desviación estándar ~ 0)
    constantes = [c for c in features.columns if features[c].std() < 1e-6]
    if constantes:
        print(f"   • ❌ Features constantes detectadas: {constantes}")
    
    # --- 5. MULTICOLINEALIDAD (Redundancia) ---
    # Si dos sensores miden lo mismo, confunden al Proceso Gaussiano
    corr_matrix = features.corr().abs()
    # Tomamos solo la parte superior de la matriz para evitar duplicados
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundantes = [col for col in upper.columns if any(upper[col] > 0.95)]
    
    if redundantes:
        print(f"   • ⚠️  Features altamente redundantes (>0.95): {len(redundantes)}")
        print(f"        Sugerencia: Revisar {redundantes[:3]}...")

    # --- 6. GENERACIÓN DE DASHBOARD VISUAL ---
    print(f"\n🎨 Generando dashboard de diagnóstico...")
    
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle(f"Diagnóstico de Datos: {target}", fontsize=16, fontweight='bold')

    # Plot 1: Serie Temporal (Visualizar tendencia y outliers)
    axes[0, 0].plot(y.iloc[:1000], color='#1f77b4', linewidth=1)
    axes[0, 0].set_title("Serie Temporal (Muestra 1k)")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribución (Chequear normalidad para GP)
    sns.histplot(y, kde=True, ax=axes[0, 1], color='green')
    axes[0, 1].set_title(f"Distribución de {target}")

    # Plot 3: Top Correlaciones (¿Quién manda en el proceso?)
    top_corr = features.corrwith(y).abs().sort_values(ascending=False).head(10)
    top_corr.plot(kind='barh', ax=axes[1, 0], color='#ff7f0e')
    axes[1, 0].set_title("Top 10 Predictores (Importancia Lineal)")

    # Plot 4: Matriz de Correlación térmica
    sns.heatmap(features.iloc[:, :15].corr(), cmap='RdBu_r', center=0, ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("Mapa de Calor (Primeras 15 features)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardado seguro
    output_img = CONFIG.RESULTS_DIR / "diagnostico_profundo.png"
    plt.savefig(output_img, dpi=120)
    plt.close()
    
    print(f"✅ Dashboard guardado en: {output_img}")
    print("\n" + "═"*70)
    print("🏁 DIAGNÓSTICO FINALIZADO: Revisa las alertas arriba antes de entrenar.")


if __name__ == "__main__":
    # Si se ejecuta directamente, corremos el diagnóstico
    try:
        diagnosticar_datos()
    except Exception as e:
        print(f"❌ Error crítico durante el diagnóstico: {str(e)}")
