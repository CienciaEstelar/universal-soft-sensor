"""
Módulo: core/scientific_report.py
Proyecto: Universal Soft-Sensor
Versión: 1.0.0 (2026-07-21)

DESCRIPCIÓN:
    Generación de reportes con estilo de publicación científica (SciencePlots,
    estilo IEEE/Nature: tipografía serif, líneas finas, márgenes ajustados) a
    partir de una corrida de entrenamiento de SoftSensorGP. Complementa, no
    reemplaza, a `core/report_generator.py` (que sigue siendo el reporte
    OPERACIONAL de turno con firmas, usado por el dashboard).

CONTENIDO GENERADO:
    1. Panel de diagnóstico (ya existía en gp_model.py::generate_report):
       serie/orden, scatter predicho-vs-real, histograma de errores, residuos.
    2. NUEVO — Gráfico de test de permutación: distribución nula (R² de datos
       barajados) vs. el R² real observado, con el p-value anotado. Visualiza
       directamente lo que el JSON de `permutation_test()` solo reporta como
       números — el tipo de evidencia que distingue "señal real" de "leakage
       o azar", que es el eje metodológico central de este proyecto.
    3. NUEVO — Gráfico de importancia de features vía permutation importance
       (sklearn.inspection), model-agnóstico (funciona igual para GP y GB,
       a diferencia de `.feature_importances_` que solo existe para árboles).
    4. NUEVO — PDF consolidado de una sola pieza (metadata + métricas +
       los 3 gráficos anteriores) vía matplotlib.backends.backend_pdf.

DEPENDENCIA OPCIONAL:
    `SciencePlots` (pip install SciencePlots) da el estilo tipográfico de
    publicación. Si no está instalado, cae de forma segura al estilo
    `seaborn-v0_8-whitegrid` que ya usaba el proyecto — nunca falla por falta
    de esta dependencia, solo se ve menos "paper".
"""

import logging
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def apply_scientific_style() -> str:
    """
    Aplica el estilo de gráficos más "científico" disponible.

    Intenta SciencePlots (['science', 'no-latex'] — no requiere instalación
    de LaTeX en el sistema, a diferencia del estilo 'science' puro). Si no
    está instalado, cae a seaborn-whitegrid; si tampoco existe (matplotlib
    muy antiguo), cae a ggplot.

    Returns:
        str: nombre del estilo efectivamente aplicado, para trazabilidad
             en el reporte (metadata de reproducibilidad).
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'no-latex'])
        return "scienceplots (science, no-latex)"
    except Exception:
        pass

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        return "seaborn-v0_8-whitegrid (fallback — instalar 'SciencePlots' para estilo de publicación)"
    except Exception:
        plt.style.use('ggplot')
        return "ggplot (fallback antiguo)"


def plot_permutation_test(
    null_r2_distribution: List[float],
    real_r2: float,
    p_value: float,
    target_col: str,
    output_path,
    keep_open: bool = False,
):
    """
    Visualiza el test de permutación: histograma de R² bajo la hipótesis
    nula (target barajado al azar) vs. el R² real observado.

    Este es el gráfico que le da cuerpo visual a la pregunta central del
    proyecto: "¿esto es señal real o es azar/leakage?" — un R² real que cae
    muy a la derecha de la distribución nula es evidencia visual directa de
    señal genuina, mucho más persuasivo que solo reportar un p-value.

    Args:
        null_r2_distribution: R² obtenido en cada permutación (barajado).
        real_r2: R² observado con el target sin barajar.
        p_value: p-value ya calculado por `permutation_test()`.
        target_col: nombre de la variable objetivo (para el título).
        output_path: ruta de salida (PNG).
        keep_open: si True, NO cierra la figura y la retorna junto al path —
            evita releer el PNG desde disco al ensamblar el PDF (más rápido,
            sin pérdida de calidad por doble rasterizado). Usar
            `plt.close(fig)` explícitamente cuando ya no se necesite.

    Returns:
        str, o (str, Figure) si keep_open=True.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    null_arr = np.asarray(null_r2_distribution, dtype=float)
    ax.hist(
        null_arr, bins=30, color='steelblue', alpha=0.75, edgecolor='white',
        label=f'Distribución nula (n={len(null_arr)} permutaciones)',
    )
    ax.axvline(
        real_r2, color='crimson', lw=2.5, ls='--',
        label=f'R² real = {real_r2:.4f}',
    )

    verdict = "señal real" if p_value < 0.05 else "no distinguible de azar"
    ax.set_title(
        f'Test de permutación — {target_col}\n'
        f'p = {p_value:.4f} ({verdict})'
    )
    ax.set_xlabel('R² (cross-validated, target barajado)')
    ax.set_ylabel('Frecuencia')
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if keep_open:
        return str(output_path), fig
    plt.close(fig)
    return str(output_path)


def plot_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_path,
    n_repeats: int = 5,
    random_state: int = 42,
    keep_open: bool = False,
):
    """
    Importancia de features vía permutation importance (sklearn.inspection),
    evaluada sobre datos de TEST (no de train) — mide la caída real de
    performance al barajar cada feature, no una heurística interna del
    modelo. Funciona igual para GP y GradientBoosting (a diferencia de
    `.feature_importances_`, que solo existe para modelos basados en árboles
    y no aplica a GaussianProcessRegressor).

    Args:
        model: estimador ya entrenado (SoftSensorGP.model — GP o GB fit).
        X_test: features de test, en la MISMA escala con la que se entrenó
            el modelo (ej. escalados si el modelo se entrenó escalado).
        y_test: target de test, en la misma escala que usa `model.predict`.
        feature_names: nombres de columnas, en el mismo orden que X_test.
        output_path: ruta de salida (PNG).
        n_repeats: número de barajados por feature. 5 por defecto — cada
            repetición es una re-predicción (no un reentrenamiento), pero
            para GP el costo de predict() no es trivial (kernel n_train x
            n_test), así que se mantiene bajo por defecto; subir a 10-20
            para un informe final si el tiempo lo permite.
        random_state: semilla para reproducibilidad.
        keep_open: si True, no cierra la figura y la retorna junto al path.

    Returns:
        str, o (str, Figure) si keep_open=True.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, random_state=random_state, scoring='r2',
        )

    order = np.argsort(result.importances_mean)
    names_sorted = [feature_names[i] for i in order]
    means_sorted = result.importances_mean[order]
    stds_sorted = result.importances_std[order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(feature_names))))
    ax.barh(
        names_sorted, means_sorted, xerr=stds_sorted,
        color='steelblue', alpha=0.85, edgecolor='white',
        error_kw={'ecolor': 'black', 'elinewidth': 1, 'capsize': 3},
    )
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Caída de R² al barajar la feature (permutation importance)')
    ax.set_title(f'Importancia de features (n_repeats={n_repeats}, sobre test set)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if keep_open:
        return str(output_path), fig
    plt.close(fig)
    return str(output_path)


def build_scientific_pdf(
    output_path,
    metadata: Dict,
    metrics_rows: List[tuple],
    image_paths: Optional[List[str]] = None,
    figures: Optional[List] = None,
) -> str:
    """
    Ensambla un PDF de una sola pieza: portada con metadata de
    reproducibilidad + tabla de métricas, seguida de una página por cada
    gráfico generado (diagnóstico, permutación, importancia de features).

    Acepta DOS formas de pasar los gráficos (se puede mezclar, pero se
    prefiere `figures` cuando está disponible — evita releer PNGs desde
    disco y volver a rasterizarlos, que es más lento y pierde nitidez):

    - `figures`: lista de objetos `matplotlib.figure.Figure` YA renderizados
      (ej. con `keep_open=True` en `plot_permutation_test`/
      `plot_feature_importance`). Se insertan directamente como páginas del
      PDF vía `pdf.savefig(fig)` — sin recodificar a imagen. Esta función
      se encarga de cerrarlas (`plt.close`) al terminar.
    - `image_paths`: rutas de PNG ya guardados en disco (comportamiento
      original) — más lento (lee + re-renderiza como imagen), pero útil
      para reconstruir un PDF a partir de imágenes ya existentes sin volver
      a tener las figuras en memoria.

    Args:
        output_path: ruta del PDF final.
        metadata: dict con claves como dataset, target, model_type,
            estilo_grafico, fecha, semilla — lo mínimo para que el reporte
            sea reproducible sin tener que ir a buscar el código.
        metrics_rows: lista de tuplas (nombre_metrica, valor, interpretacion)
            — mismo contenido que la tabla Rich que ya se imprime en consola.
        image_paths: rutas de los PNG a incluir (fallback si no hay `figures`).
        figures: figuras de matplotlib ya renderizadas, ruta rápida preferida.

    Returns:
        str: ruta del PDF generado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ancho de envoltura conservador para texto monospace fontsize=9 sobre una
    # página A4 (8.27in) con indent x=0.07 — evita que valores largos (ej. el
    # nombre del estilo gráfico con su nota de fallback) se corten fuera del
    # borde de la página en vez de pasar a una segunda línea.
    _WRAP_WIDTH = 88

    def _draw_wrapped(ax, x: float, y: float, text: str, line_step: float, **kwargs) -> float:
        """Dibuja `text` envuelto a `_WRAP_WIDTH` columnas, línea por línea.

        Returns:
            float: la coordenada `y` (en ejes, 0-1) después de la última línea
                dibujada, lista para que el llamador siga escribiendo debajo.
        """
        for line in textwrap.wrap(text, width=_WRAP_WIDTH) or [text]:
            ax.text(x, y, line, transform=ax.transAxes, **kwargs)
            y -= line_step
        return y

    with PdfPages(str(output_path)) as pdf:
        # --- Portada: metadata + métricas ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
        ax.axis('off')
        ax.set_title('Informe Científico — Universal Soft-Sensor', fontsize=16, fontweight='bold', pad=20)
        ax.text(
            0.05, 0.945,
            f"Generado automáticamente por Universal Soft-Sensor — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=8.5, color='#666666', style='italic', transform=ax.transAxes,
        )

        y = 0.89
        ax.text(0.05, y, "Metadata de reproducibilidad", fontsize=12, fontweight='bold', transform=ax.transAxes)
        y -= 0.035
        for key, value in metadata.items():
            y = _draw_wrapped(
                ax, 0.07, y, f"{key}: {value}", line_step=0.022,
                fontsize=9, family='monospace',
            )
            y -= 0.008  # respiro extra entre entradas de metadata

        y -= 0.03
        ax.text(0.05, y, "Métricas de evaluación", fontsize=12, fontweight='bold', transform=ax.transAxes)
        y -= 0.035
        for name, value, interp in metrics_rows:
            y = _draw_wrapped(
                ax, 0.07, y, f"{name}: {value}  ({interp})", line_step=0.022,
                fontsize=9, family='monospace',
            )
            y -= 0.008

        pdf.savefig(fig)
        plt.close(fig)

        # --- Ruta rápida: figuras ya renderizadas en memoria ---
        if figures:
            for f in figures:
                if f is None:
                    continue
                pdf.savefig(f)
                plt.close(f)

        # --- Ruta fallback: releer PNGs desde disco ---
        if image_paths:
            for img_path in image_paths:
                if not img_path or not Path(img_path).exists():
                    continue
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 apaisado
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

    logger.info(f"Informe científico PDF generado: {output_path}")
    return str(output_path)
