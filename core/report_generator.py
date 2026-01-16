"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/report_generator.py
Versión: 3.4.0 (Audit Engine)
Autor: Juan Galaz (Arquitectura Minera 4.0)
Fecha: 16 de Enero, 2026
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN GENERAL
------------------
Motor de Reportabilidad Forense para entornos industriales.
Este módulo es responsable de la materialización de la evidencia digital generada
por el Gemelo Digital.

OBJETIVO TÉCNICO
---------------
Transformar los datos volátiles de la sesión (predicciones, estados de sensores)
en documentos inmutables (PDF) que sirvan como auditoría de turno.

CARACTERÍSTICAS DE ROBUSTEZ
--------------------------
1. Sanitización de Texto: Elimina emojis y caracteres no-Latin-1 que rompen
   los sistemas de generación de PDF tradicionales.
2. Contrato de Datos (DTO): Uso de Dataclasses para asegurar tipado estricto
   antes de la generación del documento.
3. Manejo de Evidencia: Inserción inteligente de gráficos (Snapshots); si la
   imagen falla, el reporte se genera igual con una advertencia (Fail-Safe).

ARQUITECTURA
-----------
- ShiftReportData: Estructura de datos inmutable (el "Contrato").
- MiningReportEngine: Clase que define la identidad visual corporativa (Header/Footer).
- ReportManager: Controlador lógico que orquesta la creación del archivo.
═══════════════════════════════════════════════════════════════════════════════
"""

from fpdf import FPDF
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

# Configuración de Logging para trazabilidad de errores en producción
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CAPA DE DATOS (DATA TRANSFER OBJECTS)
# =============================================================================

@dataclass
class ShiftReportData:
    """
    DTO (Data Transfer Object) para el Reporte de Turno.
    
    Define el contrato estricto de los datos necesarios para generar una auditoría.
    Si falta algún campo obligatorio, el sistema fallará antes de intentar crear
    el PDF, garantizando la integridad de la información.
    """
    timestamp: datetime         # Momento exacto de la captura (Time-Stamp)
    recovery_avg: float         # Predicción promedio del modelo
    recovery_target: float      # KPI objetivo del negocio
    financial_impact: float     # Traducción monetaria del KPI técnico
    model_name: str             # Identificador del modelo (Trazabilidad de IA)
    sensor_health: Dict[str, str] # Estado de la red IoT {'Sensor': 'Estado'}
    recommendation: str         # Estrategia operativa sugerida por el motor "What-If"
    chart_path: Optional[str] = None # Ruta al archivo de evidencia visual (Snapshot)


# =============================================================================
# 2. MOTOR DE RENDERIZADO VISUAL (PDF ENGINE)
# =============================================================================

class MiningReportEngine(FPDF):
    """
    Extensión de FPDF que encapsula la identidad visual corporativa.
    Define cómo se ven los encabezados, pies de página y tarjetas de datos.
    """
    
    def header(self):
        """
        Renderiza el encabezado en CADA página del documento.
        Incluye el título del sistema y líneas divisorias.
        """
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100) # Gris Corporativo (Sobrio)
        # Título alineado a la derecha
        self.cell(0, 10, "MINING 4.0 | DIGITAL TWIN AUDIT SYSTEM", 0, 1, "R")
        # Línea de separación visual
        self.line(10, 20, 200, 20)
        self.ln(10) # Espacio de respiro

    def footer(self):
        """
        Renderiza el pie de página con paginación y certificación de versión.
        Vital para trazabilidad en auditorías impresas.
        """
        self.set_y(-15) # 1.5 cm desde el borde inferior
        self.set_font("Helvetica", "I", 8) # Itálica pequeña
        self.cell(0, 10, f"Pagina {self.page_no()} | Certificado por Inference Engine v3.4", 0, 0, "C")

    def section_title(self, title: str):
        """
        Genera un título de sección estandarizado (Azul Navy).
        """
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102) # Azul Institucional
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)

    def kpi_card(self, label: str, value: str):
        """
        Renderiza un par Clave-Valor con formato de tarjeta técnica.
        Asegura que las etiquetas y valores estén alineados perfectamente.
        """
        self.set_font("Helvetica", "", 12)
        self.set_text_color(0) # Negro puro
        self.cell(90, 8, f"{label}:", 0, 0) # Etiqueta
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, f"{value}", 0, 1)   # Valor en negrita


# =============================================================================
# 3. CONTROLADOR DE LÓGICA DE NEGOCIO (REPORT MANAGER)
# =============================================================================

class ReportManager:
    """
    Clase controladora responsable de orquestar la creación del reporte.
    Maneja la limpieza de datos, validación de rutas y escritura en disco.
    """
    
    def __init__(self, output_dir: str = "results/reports"):
        """
        Inicializa el gestor y asegura que el directorio de salida exista.
        """
        self.output_dir = Path(output_dir)
        # mkdir(parents=True) crea subdirectorios si no existen (ej: results/...)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_text(self, text: str) -> str:
        """
        [MÉTODO CRÍTICO] Limpieza de codificación.
        
        El estándar PDF antiguo (Latin-1) no soporta emojis modernos.
        Si intentamos escribir '⚠️' directamente, el reporte fallará.
        Esta función reemplaza iconos por texto seguro antes de renderizar.
        
        Args:
            text (str): Texto crudo potencialmente "sucio" con emojis.
        Returns:
            str: Texto seguro, compatible con ISO-8859-1.
        """
        replacements = {
            "⚠️": "[ALERTA]",
            "✅": "[OK]",
            "🚨": "[CRITICO]",
            "📉": "(-)",
            "📈": "(+)"
        }
        
        # 1. Reemplazo de emojis conocidos
        for char, safe_str in replacements.items():
            text = text.replace(char, safe_str)
        
        # 2. Eliminación forzada de cualquier otro caracter incompatible
        # encode('latin-1', 'replace') cambia caracteres desconocidos por '?'
        return text.encode('latin-1', 'replace').decode('latin-1')

    def generate(self, data: ShiftReportData) -> str:
        """
        Construye el documento PDF paso a paso.
        
        Args:
            data (ShiftReportData): DTO con toda la información del turno.
        Returns:
            str: Ruta absoluta del archivo generado.
        """
        pdf = MiningReportEngine()
        pdf.add_page()
        
        # --- PORTADA / TÍTULO ---
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 15, "Reporte de Turno - Auditoria de Proceso", 0, 1, "C")
        pdf.ln(5)

        # --- SECCIÓN 1: RESUMEN EJECUTIVO ---
        pdf.section_title("1. Resumen Ejecutivo")
        pdf.kpi_card("Fecha Emision", data.timestamp.strftime("%Y-%m-%d %H:%M"))
        pdf.kpi_card("Modelo IA", data.model_name)
        pdf.ln(5)
        
        # --- SECCIÓN 2: PERFORMANCE OPERACIONAL ---
        pdf.section_title("2. Performance Operacional")
        pdf.kpi_card("Recuperacion Promedio", f"{data.recovery_avg:.2f}%")
        # Formateo financiero (ej: $1,200.50)
        pdf.kpi_card("Impacto Economico", f"${data.financial_impact:,.2f} USD")
        
        # --- SECCIÓN 3: DIAGNÓSTICO DE TELEMETRÍA ---
        pdf.ln(5)
        pdf.section_title("3. Estado de Sensores")
        pdf.set_font("Courier", "", 10) # Fuente monoespaciada para logs
        
        for sensor_name, status in data.sensor_health.items():
            # Limpiamos el texto por si el dashboard envió algún emoji de estado
            clean_text = self._sanitize_text(f"[{status}] {sensor_name}")
            pdf.cell(0, 5, clean_text, 0, 1)

        # --- SECCIÓN 4: EVIDENCIA VISUAL (SNAPSHOT) ---
        pdf.ln(10)
        pdf.section_title("4. Tendencia de Proceso (Snapshot)")
        
        # Lógica Fail-Safe: Verificamos que la imagen exista y tenga tamaño > 0
        if data.chart_path and Path(data.chart_path).exists() and Path(data.chart_path).stat().st_size > 0:
            try:
                # Insertamos la imagen ajustada al ancho útil A4 (190mm)
                pdf.image(data.chart_path, x=10, w=190)
            except Exception as e:
                logger.error(f"Fallo al insertar imagen en PDF: {e}")
                pdf.set_text_color(200, 0, 0) # Rojo Alerta
                pdf.cell(0, 10, "[Error de Renderizado: Archivo de imagen corrupto]", 0, 1)
        else:
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 10, "[Imagen no disponible: Fallo en motor Kaleido]", 0, 1)
        
        pdf.set_text_color(0) # Reset a negro

        # --- SECCIÓN 5: ESTRATEGIA Y VALIDACIÓN (NUEVA PÁGINA) ---
        pdf.add_page()
        pdf.section_title("5. Validacion y Estrategia")
        
        # Subtítulo de IA
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 10, "RECOMENDACION DEL SISTEMA INTELIGENTE:", 0, 1)
        
        # Cuerpo de la recomendación (Sanitizado)
        pdf.set_font("Helvetica", "", 10)
        safe_recommendation = self._sanitize_text(data.recommendation)
        pdf.multi_cell(0, 5, safe_recommendation)
        
        pdf.ln(5)
        pdf.multi_cell(0, 5, "Se certifica que los datos presentados corresponden a la operacion real del Gemelo Digital. Documento generado automaticamente.")
        
        # --- BLOQUE DE FIRMAS ---
        pdf.ln(40) # Espacio vertical para firmar
        y = pdf.get_y()
        
        # Líneas de firma
        pdf.line(20, y, 90, y)      # Izquierda
        pdf.line(120, y, 190, y)    # Derecha
        
        # Textos de firma
        pdf.cell(90, 5, "Firma Jefe de Turno", 0, 0, "C")
        pdf.cell(10, 5, "", 0, 0) # Espaciador
        pdf.cell(70, 5, "Firma Gerente Planta", 0, 1, "C")

        # --- ESCRITURA A DISCO ---
        filename = f"Reporte_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.output_dir / filename
        
        pdf.output(str(output_path))
        logger.info(f"Reporte generado exitosamente: {output_path}")
        
        return str(output_path)