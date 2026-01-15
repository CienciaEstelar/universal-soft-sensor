"""
Script: tools/diagnostico_datos.py
Descripción: Diagnóstico completo de los datos limpios antes del modelado GP.
             Identifica problemas comunes que causan R² negativo.
             
Uso:
    python -m tools.diagnostico_datos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.settings import CONFIG


def diagnosticar_datos():
    """Ejecuta diagnóstico completo del dataset limpio."""
    
    print("🔬 DIAGNÓSTICO DE DATOS PARA MODELADO GP")
    print("=" * 70)
    
    # 1. Cargar datos
    filepath = CONFIG.DATA_CLEAN_PATH
    if not filepath.exists():
        print(f"❌ No se encontró: {filepath}")
        print("   Ejecuta primero: mining-pipeline")
        return
    
    print(f"\n📂 Cargando: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, nrows=10000)  # Solo 10k para diagnóstico
    
    print(f"   Dimensiones: {df.shape}")
    print(f"   Rango temporal: {df.index.min()} → {df.index.max()}")
    
    target = CONFIG.GP_TARGET_COLUMN
    
    # 2. Verificar target
    print(f"\n🎯 TARGET: {target}")
    print("-" * 50)
    
    if target not in df.columns:
        print(f"   ❌ Columna '{target}' NO ENCONTRADA")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        return
    
    y = df[target]
    print(f"   Min:    {y.min():.4f}")
    print(f"   Max:    {y.max():.4f}")
    print(f"   Mean:   {y.mean():.4f}")
    print(f"   Std:    {y.std():.4f}")
    print(f"   NaN:    {y.isna().sum()} ({y.isna().mean()*100:.2f}%)")
    print(f"   Zeros:  {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
    
    # 3. Verificar variabilidad del target
    print(f"\n📈 VARIABILIDAD DEL TARGET")
    print("-" * 50)
    
    cv = y.std() / y.mean() * 100  # Coeficiente de variación
    print(f"   Coef. Variación: {cv:.2f}%")
    
    if cv < 5:
        print("   ⚠️  ALERTA: Variabilidad MUY BAJA")
        print("      El target casi no varía - GP tendrá dificultades")
    elif cv < 10:
        print("   ⚠️  Variabilidad baja - considerar más features")
    else:
        print("   ✅ Variabilidad adecuada")
    
    # 4. Verificar autocorrelación (series temporales)
    print(f"\n🔄 AUTOCORRELACIÓN TEMPORAL")
    print("-" * 50)
    
    autocorr_1 = y.autocorr(lag=1)
    autocorr_10 = y.autocorr(lag=10)
    autocorr_100 = y.autocorr(lag=100)
    
    print(f"   Lag 1:   {autocorr_1:.4f}")
    print(f"   Lag 10:  {autocorr_10:.4f}")
    print(f"   Lag 100: {autocorr_100:.4f}")
    
    if autocorr_1 > 0.95:
        print("   ⚠️  ALERTA: Autocorrelación MUY ALTA")
        print("      Los datos consecutivos son casi idénticos")
        print("      Considera: subsamplear cada N registros")
    
    # 5. Verificar features
    print(f"\n📊 ANÁLISIS DE FEATURES")
    print("-" * 50)
    
    features = df.drop(columns=[target, "_iron_concentrate"], errors='ignore')
    
    print(f"   Total features: {len(features.columns)}")
    
    # Features constantes
    constantes = []
    for col in features.columns:
        if features[col].std() < 1e-6:
            constantes.append(col)
    
    if constantes:
        print(f"   ⚠️  Features CONSTANTES (eliminar): {constantes}")
    else:
        print("   ✅ No hay features constantes")
    
    # Features con alta correlación con target
    print(f"\n   Correlación con target ({target}):")
    correlaciones = features.corrwith(y).abs().sort_values(ascending=False)
    
    for col, corr in correlaciones.head(10).items():
        emoji = "🟢" if corr > 0.3 else "🟡" if corr > 0.1 else "🔴"
        print(f"      {emoji} {col}: {corr:.4f}")
    
    # 6. Verificar multicolinealidad
    print(f"\n🔗 MULTICOLINEALIDAD (Features correlacionados entre sí)")
    print("-" * 50)
    
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > 0.95:
                high_corr_pairs.append((idx, col, upper.loc[idx, col]))
    
    if high_corr_pairs:
        print(f"   ⚠️  {len(high_corr_pairs)} pares con correlación > 0.95:")
        for p1, p2, c in high_corr_pairs[:5]:
            print(f"      {p1} ↔ {p2}: {c:.4f}")
        print("   Considera eliminar features redundantes")
    else:
        print("   ✅ No hay multicolinealidad extrema")
    
    # 7. Recomendaciones
    print(f"\n💡 RECOMENDACIONES")
    print("=" * 70)
    
    recomendaciones = []
    
    if autocorr_1 > 0.95:
        recomendaciones.append(
            "• SUBSAMPLEAR: Toma cada 10-20 registros para reducir autocorrelación"
        )
    
    if cv < 10:
        recomendaciones.append(
            "• FEATURE ENGINEERING: Agregar lags, diferencias, o rolling stats"
        )
    
    if correlaciones.max() < 0.3:
        recomendaciones.append(
            "• FEATURES DÉBILES: Ningún feature tiene buena correlación con target.\n"
            "  Considera: lags temporales, interacciones, transformaciones"
        )
    
    if len(high_corr_pairs) > 5:
        recomendaciones.append(
            "• REDUCIR DIMENSIONALIDAD: PCA o eliminar features redundantes"
        )
    
    if not recomendaciones:
        print("✅ Los datos parecen adecuados para modelado GP")
    else:
        for r in recomendaciones:
            print(r)
    
    # 8. Guardar gráfico de diagnóstico
    print(f"\n📊 Generando gráfico de diagnóstico...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Serie temporal del target
    axes[0, 0].plot(df.index[:500], y.iloc[:500], 'b-', linewidth=0.5)
    axes[0, 0].set_title(f'Serie Temporal: {target} (primeros 500)')
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('Valor')
    
    # Histograma del target
    axes[0, 1].hist(y, bins=50, color='steelblue', edgecolor='white')
    axes[0, 1].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
    axes[0, 1].set_title(f'Distribución: {target}')
    axes[0, 1].legend()
    
    # Autocorrelación
    lags = range(1, 101)
    autocorrs = [y.autocorr(lag=l) for l in lags]
    axes[1, 0].bar(lags, autocorrs, color='steelblue', width=1)
    axes[1, 0].axhline(0.95, color='red', linestyle='--', label='Umbral 0.95')
    axes[1, 0].set_title('Autocorrelación por Lag')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelación')
    axes[1, 0].legend()
    
    # Top correlaciones con target
    top_corr = correlaciones.head(10)
    axes[1, 1].barh(top_corr.index, top_corr.values, color='steelblue')
    axes[1, 1].set_title(f'Top 10 Features Correlacionados con {target}')
    axes[1, 1].set_xlabel('|Correlación|')
    
    plt.tight_layout()
    
    output_path = CONFIG.RESULTS_DIR / "diagnostico_datos.png"
    CONFIG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"   Guardado: {output_path}")
    
    print("\n" + "=" * 70)
    print("🏁 Diagnóstico completado")
    

if __name__ == "__main__":
    diagnosticar_datos()
