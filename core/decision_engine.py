"""
Módulo: core/decision_engine.py
Autor: Juan Galaz (Arquitectura Minera 4.0)
Versión: 1.2 (Hardened)

Descripción:
    Este módulo actúa como el "Juez Supremo" del sistema de monitoreo.
    Implementa el patrón de diseño "Sensor Fusion" para combinar múltiples fuentes
    de verdad antes de emitir una alerta operativa.

    Fuentes de Verdad (en orden de prioridad):
    1.  🔴 HARD GATES (Física): Sensores superando límites de diseño. (Veto absoluto).
    2.  🟠 MODELO NUMÉRICO (Estadística): Probabilidad de falla calculada por Gaussian Process.
    3.  🟡 IA GENERATIVA (Semántica): Interpretación contextual de Gemini.

    Responsabilidad:
    Evitar falsos positivos de la IA y garantizar la seguridad de la planta
    mediante bloqueos físicos (Safety Locks).
"""

import logging
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field

# Configuración del logger para trazabilidad de decisiones
logger = logging.getLogger("Mining_Decision_Engine")


# --- 1. Modelos de Datos (El Contrato) ---

class MiningAlert(BaseModel):
    """
    Estructura estandarizada para alertas operativas.
    Garantiza que el Agente siempre reciba datos con el mismo formato.
    """
    timestamp: str = Field(..., description="Hora UTC de la decisión.")
    sensor_id: str = Field(..., description="ID del componente afectado (ej: PUMP_01).")
    
    alert_level: str = Field(
        ..., 
        pattern="^(CRITICAL|WARNING|INFO|NORMAL)$",
        description="Nivel de severidad para el tablero SCADA."
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, le=1.0, 
        description="Nivel de certeza de la decisión (0.0 a 1.0)."
    )
    
    reason: str = Field(..., description="Explicación humana del porqué de la alerta.")
    
    action_required: str = Field(
        ..., 
        description="Acción recomendada para el operador humano."
    )
    
    safety_lock: bool = Field(
        default=False, 
        description="Si es True, solicita PARADA DE EMERGENCIA inmediata."
    )


# --- 2. Motor de Decisiones ---

class DecisionEngine:
    """
    Motor de inferencia híbrido (Física + IA).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el motor con los límites operativos de la planta.

        Args:
            config (Dict): Diccionario cargado desde 'config/settings.json'.
                           Debe contener la sección 'safety_limits'.
        """
        self.config = config
        
        # Cargamos límites operativos. Usamos valores por defecto seguros (fail-safe)
        # si no se encuentran en la configuración.
        limits = config.get("safety_limits", {})
        self.temp_limit = limits.get("temperature_max", 90.0)  # Grados Celsius
        self.vib_limit = limits.get("vibration_max", 15.0)     # mm/s RMS
        
        logger.info(f"⚙️ Decision Engine Iniciado. Límites: Temp={self.temp_limit}°C, Vib={self.vib_limit}mm/s")

    def _check_hard_gates(self, current_state: Dict[str, float]) -> Optional[MiningAlert]:
        """
        [CAPA 1: FÍSICA] Verifica violaciones de límites físicos inviolables.
        
        Esta capa tiene prioridad infinita. Si un sensor físico indica peligro,
        no importa lo que diga la IA, se debe detener el proceso.

        Args:
            current_state (Dict): Última lectura de sensores {sensor_name: value}.

        Returns:
            Optional[MiningAlert]: Alerta CRÍTICA si se rompe un límite, o None si está OK.
        """
        # 1. Verificación Térmica
        temp = current_state.get("sensor_temp", 0.0)
        if temp > self.temp_limit:
            msg = f"Límite Térmico Excedido: {temp:.1f}°C > {self.temp_limit}°C"
            logger.critical(f"🔥 HARD GATE: {msg}")
            
            return MiningAlert(
                timestamp="now", # Será reemplazado por el Agente
                sensor_id="TEMP_SENSOR_MAIN",
                alert_level="CRITICAL",
                confidence=1.0, # 100% Certeza (Es un dato físico)
                reason=msg,
                action_required="PARADA DE EMERGENCIA AUTOMÁTICA",
                safety_lock=True
            )

        # 2. Verificación de Vibración
        vib = current_state.get("sensor_vibration", 0.0)
        if vib > self.vib_limit:
            msg = f"Vibración Destructiva: {vib:.2f}mm/s > {self.vib_limit}mm/s"
            logger.critical(f"〰️ HARD GATE: {msg}")
            
            return MiningAlert(
                timestamp="now",
                sensor_id="VIB_SENSOR_MAIN",
                alert_level="CRITICAL",
                confidence=1.0,
                reason=msg,
                action_required="PARADA DE EMERGENCIA AUTOMÁTICA",
                safety_lock=True
            )
            
        return None

    def _fusion_logic(self, 
                      ai_diagnosis: Dict[str, Any], 
                      model_prob: float, 
                      current_state: Dict[str, float]) -> MiningAlert:
        """
        [CAPA 2: FUSIÓN] Combina probabilidad numérica con razonamiento semántico.
        
        Utiliza un promedio ponderado (Weighted Ensemble) para calcular el riesgo final.
        
        Lógica de Ponderación:
        - Modelo Numérico (GP): 60% peso (Detecta tendencias sutiles en series de tiempo).
        - Modelo IA (Gemini): 40% peso (Aporta contexto y explicación, pero puede alucinar).

        Args:
            ai_diagnosis (Dict): Respuesta JSON parseada de Gemini.
            model_prob (float): Probabilidad de falla (0.0 - 1.0) del modelo GP.
            current_state (Dict): Estado actual para contexto.

        Returns:
            MiningAlert: La decisión final ponderada.
        """
        
        # Extracción segura de valores de la IA (con valores por defecto defensivos)
        ai_risk_score = float(ai_diagnosis.get("risk_score", 0.0))
        ai_reason = ai_diagnosis.get("reason", "Análisis IA no disponible")
        
        # --- ALGORITMO DE FUSIÓN ---
        # Peso conservador: Confiamos más en la matemática dura (GP) que en el LLM.
        WEIGHT_GP = 0.6
        WEIGHT_AI = 0.4
        
        final_risk_score = (model_prob * WEIGHT_GP) + (ai_risk_score * WEIGHT_AI)
        
        logger.debug(f"🧮 Fusión: GP({model_prob:.2f}) * {WEIGHT_GP} + IA({ai_risk_score:.2f}) * {WEIGHT_AI} = {final_risk_score:.2f}")

        # --- UMBRALES DE DECISIÓN ---
        
        # Caso A: Riesgo Alto (Requiere Acción)
        if final_risk_score > 0.80:
            return MiningAlert(
                timestamp="now",
                sensor_id="SYS_INTEGRATED",
                alert_level="WARNING",
                confidence=final_risk_score,
                reason=f"Alta Probabilidad de Falla Combinada. IA destaca: {ai_reason}",
                action_required="Programar Mantenimiento Preventivo (Prioridad Alta)",
                safety_lock=False
            )
            
        # Caso B: Riesgo Medio (Observación)
        elif final_risk_score > 0.50:
            return MiningAlert(
                timestamp="now",
                sensor_id="SYS_INTEGRATED",
                alert_level="INFO",
                confidence=final_risk_score,
                reason="Tendencia leve al deterioro detectada por modelos híbridos.",
                action_required="Monitorear sensores en siguiente turno",
                safety_lock=False
            )

        # Caso C: Operación Normal
        return MiningAlert(
            timestamp="now",
            sensor_id="SYS_INTEGRATED",
            alert_level="NORMAL",
            confidence=0.95,
            reason="Operación estable validada (Física + IA + GP)",
            action_required="Ninguna",
            safety_lock=False
        )

    def evaluate(self, 
                 current_state: Dict[str, float], 
                 ai_diagnosis: Dict[str, Any], 
                 model_prob: float) -> MiningAlert:
        """
        Punto de entrada principal para la evaluación de estado.
        
        Flujo:
        1. Ejecuta _check_hard_gates(). Si hay peligro inminente, RETORNA INMEDIATAMENTE.
        2. Si la física es segura, ejecuta _fusion_logic() para análisis predictivo.

        Args:
            current_state: Diccionario con valores de sensores.
            ai_diagnosis: Diccionario con el análisis de Gemini.
            model_prob: Flotante con la probabilidad de falla del modelo GP.

        Returns:
            MiningAlert: Objeto con la decisión final, listo para ser enviado al Agente.
        """
        # 1. Seguridad Primero (Safety First)
        critical_alert = self._check_hard_gates(current_state)
        if critical_alert:
            return critical_alert
            
        # 2. Análisis Inteligente
        return self._fusion_logic(ai_diagnosis, model_prob, current_state)