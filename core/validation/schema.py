"""
═══════════════════════════════════════════════════════════════════════════════
Módulo: core/validation/schema.py
Versión: 2.0.0
Proyecto: Minero 4.0 - Pipeline Universal de IA Industrial
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
    Schema de validación física UNIVERSAL para datos de procesos industriales.
    
    A diferencia de la v1.0 (que tenía columnas hardcodeadas), esta versión
    usa PATTERN MATCHING para detectar el tipo de variable y aplicar rangos
    físicos universales.

═══════════════════════════════════════════════════════════════════════════════
DATASETS SOPORTADOS:
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┬────────────────────────────────────────────────────┐
    │ Dataset         │ Columnas que matchean                              │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ gold_recovery   │ rougher.input.feed_au, flotation_*_air_amount,    │
    │ (Kaggle)        │ primary_cleaner.state.floatbank*_level, etc.      │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ ai4i2020        │ Air temperature [K], Process temperature [K],     │
    │ (UCI)           │ Rotational speed [rpm], Torque [Nm], Tool wear    │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ Genérico        │ Cualquier dataset con nombres descriptivos        │
    │                 │ que contengan patrones reconocibles               │
    └─────────────────┴────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
HISTORIAL DE CAMBIOS:
═══════════════════════════════════════════════════════════════════════════════

    [v2.0.0 - Enero 2026] UNIVERSALIZACIÓN
    --------------------------------------
    - Eliminación de columnas hardcodeadas (_iron_feed, _silica_feed, etc.)
    - Implementación de pattern matching por categoría física
    - Soporte para múltiples datasets sin modificar código
    - Rangos basados en límites físicos universales
    
    MIGRACIÓN DESDE v1.0:
    
    # ANTES (v1.0 - hardcoded):
    RANGOS_FISICOS = {
        "_iron_feed": (0.0, 100.0),  # Solo funcionaba con dataset específico
    }
    
    # AHORA (v2.0 - universal):
    # El schema detecta automáticamente que "rougher.input.feed_iron" 
    # es un porcentaje de metal y aplica rango (0, 100)

═══════════════════════════════════════════════════════════════════════════════
USO:
═══════════════════════════════════════════════════════════════════════════════

    from core.validation.schema import SCHEMA
    
    # Obtener rango para cualquier columna
    min_val, max_val = SCHEMA.get_range("rougher.input.feed_au")
    # → (0.0, 100.0) porque detecta "_au" como porcentaje de metal
    
    min_val, max_val = SCHEMA.get_range("Air temperature [K]")
    # → (200.0, 500.0) porque detecta "temperature" 
    
    # Agregar regla específica (override)
    SCHEMA.add_rule("mi_columna_especial", 0.0, 999.0)
    
    # Listar todas las categorías
    SCHEMA.list_categories()

═══════════════════════════════════════════════════════════════════════════════
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMERACIÓN: Categorías Físicas
# ═══════════════════════════════════════════════════════════════════════════

class PhysicalCategory(Enum):
    """
    Categorías físicas universales para variables industriales.
    
    Cada categoría tiene:
    - Nombre descriptivo
    - Rango físico válido (min, max)
    - Unidad típica (para documentación)
    """
    # Formato: (min, max, unidad)
    TEMPERATURE_KELVIN = (200.0, 600.0, "K")
    TEMPERATURE_CELSIUS = (-50.0, 350.0, "°C")
    PERCENTAGE = (0.0, 100.0, "%")
    FLOW_RATE = (0.0, 50000.0, "m³/h o L/min")
    PH = (0.0, 14.0, "pH")
    DENSITY = (0.5, 5.0, "kg/L")
    PRESSURE = (0.0, 1000.0, "bar o kPa")
    TORQUE = (0.0, 500.0, "Nm")
    ROTATIONAL_SPEED = (0.0, 10000.0, "rpm")
    LEVEL = (-500.0, 1500.0, "mm o cm")
    TOOL_WEAR = (0.0, 500.0, "min")
    POWER = (0.0, 50000.0, "kW")
    CURRENT = (0.0, 1000.0, "A")
    PARTICLE_SIZE = (0.0, 1000.0, "µm")
    BINARY = (0.0, 1.0, "flag")
    UNKNOWN = (float("-inf"), float("inf"), "?")


# ═══════════════════════════════════════════════════════════════════════════
# REGLAS DE PATTERN MATCHING
# ═══════════════════════════════════════════════════════════════════════════

# Cada regla es: (lista_de_patrones, categoría, prioridad)
# Prioridad más alta = se evalúa primero (para resolver conflictos)

PATTERN_RULES: List[Tuple[List[str], PhysicalCategory, int]] = [
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORIDAD ALTA (100+): Patrones muy específicos
    # ═══════════════════════════════════════════════════════════════════════
    
    # Temperaturas en Kelvin (típico en datasets UCI)
    (["temperature [k]", "temp [k]", "_kelvin"], 
     PhysicalCategory.TEMPERATURE_KELVIN, 150),
    
    # Temperaturas en Celsius
    (["temperature [c]", "temp [c]", "_celsius", "temperature_c"], 
     PhysicalCategory.TEMPERATURE_CELSIUS, 150),
    
    # Tool wear específico (ai4i2020)
    (["tool wear", "tool_wear", "toolwear"], 
     PhysicalCategory.TOOL_WEAR, 140),
    
    # Torque específico
    (["torque [nm]", "torque_nm", "torque"], 
     PhysicalCategory.TORQUE, 130),
    
    # RPM / Velocidad rotacional
    (["rotational speed", "rotational_speed", "[rpm]", "_rpm"], 
     PhysicalCategory.ROTATIONAL_SPEED, 130),
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORIDAD MEDIA (50-99): Patrones de proceso minero
    # ═══════════════════════════════════════════════════════════════════════
    
    # Porcentajes de metales (gold_recovery dataset)
    # _au = oro, _ag = plata, _pb = plomo, _sol = sólidos
    (["_au", "_ag", "_pb", "_sol", "_fe", "_sio2", "iron", "silica",
      "recovery", "concentrate", "grade"], 
     PhysicalCategory.PERCENTAGE, 90),
    
    # pH del proceso
    (["_ph", ".ph", "pulp_ph", "ore_ph"], 
     PhysicalCategory.PH, 85),
    
    # Densidad de pulpa
    (["density", "densidad", "pulp_density"], 
     PhysicalCategory.DENSITY, 85),
    
    # Flujos y caudales
    (["flow", "amount", "caudal", "feed_rate", "starch", "amina", 
      "xanthate", "reagent"], 
     PhysicalCategory.FLOW_RATE, 80),
    
    # Niveles en celdas/tanques
    (["level", "nivel", "floatbank", "tank_level"], 
     PhysicalCategory.LEVEL, 80),
    
    # Aire en flotación
    (["air_amount", "air_flow", "airflow", "aeration"], 
     PhysicalCategory.FLOW_RATE, 80),
    
    # Tamaño de partícula
    (["particle_size", "feed_size", "p80", "d50", "granulometry"], 
     PhysicalCategory.PARTICLE_SIZE, 75),
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORIDAD BAJA (1-49): Patrones genéricos
    # ═══════════════════════════════════════════════════════════════════════
    
    # Temperatura genérica (sin unidad especificada, asumimos Kelvin)
    (["temperature", "temp"], 
     PhysicalCategory.TEMPERATURE_KELVIN, 40),
    
    # Potencia eléctrica
    (["power", "potencia", "kw", "watt"], 
     PhysicalCategory.POWER, 30),
    
    # Corriente eléctrica
    (["current", "corriente", "ampere", "_a"], 
     PhysicalCategory.CURRENT, 30),
    
    # Presión
    (["pressure", "presion", "bar", "kpa", "psi"], 
     PhysicalCategory.PRESSURE, 30),
    
    # Variables binarias / flags
    (["failure", "fault", "alarm", "flag", "status", "twf", "hdf", 
      "pwf", "osf", "rnf", "machine_failure"], 
     PhysicalCategory.BINARY, 20),
]


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: MiningSchema
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MiningSchema:
    """
    Schema de validación física universal para procesos industriales.
    
    Funciona en dos niveles:
    1. OVERRIDES: Reglas específicas por nombre exacto de columna
    2. PATTERNS: Detección automática por patrones en el nombre
    
    El orden de búsqueda es: override exacto → pattern matching → UNKNOWN
    
    Attributes
    ----------
    _overrides : Dict[str, Tuple[float, float]]
        Diccionario de reglas específicas (nombre_columna → (min, max))
    _pattern_cache : Dict[str, PhysicalCategory]
        Cache de categorías detectadas para evitar re-matching
        
    Examples
    --------
    >>> from core.validation.schema import SCHEMA
    >>> SCHEMA.get_range("rougher.input.feed_au")
    (0.0, 100.0)
    >>> SCHEMA.get_category("Air temperature [K]")
    <PhysicalCategory.TEMPERATURE_KELVIN>
    """
    
    _overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _pattern_cache: Dict[str, PhysicalCategory] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ordena las reglas por prioridad (descendente) una sola vez."""
        global PATTERN_RULES
        PATTERN_RULES = sorted(PATTERN_RULES, key=lambda x: x[2], reverse=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS PÚBLICOS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_range(self, col_name: str) -> Tuple[float, float]:
        """
        Obtiene el rango físico válido para una columna.
        
        Estrategia de búsqueda (en orden):
        1. Override exacto (si existe)
        2. Pattern matching por categoría
        3. Fallback a (-inf, +inf) para columnas desconocidas
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna (puede estar sanitizado o no).
            
        Returns
        -------
        Tuple[float, float]
            (valor_mínimo, valor_máximo) permitidos.
            
        Examples
        --------
        >>> SCHEMA.get_range("rougher.output.recovery")
        (0.0, 100.0)
        >>> SCHEMA.get_range("columna_random_xyz")
        (-inf, inf)
        """
        # 1. Buscar override exacto
        col_lower = col_name.lower().strip()
        if col_lower in self._overrides:
            return self._overrides[col_lower]
        
        # 2. Detectar categoría por patrón
        category = self.get_category(col_name)
        
        # 3. Retornar rango de la categoría
        return (category.value[0], category.value[1])
    
    def get_category(self, col_name: str) -> PhysicalCategory:
        """
        Detecta la categoría física de una columna por pattern matching.
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna.
            
        Returns
        -------
        PhysicalCategory
            Categoría detectada (o UNKNOWN si no matchea nada).
        """
        col_lower = col_name.lower().strip()
        
        # Check cache primero
        if col_lower in self._pattern_cache:
            return self._pattern_cache[col_lower]
        
        # Pattern matching
        for patterns, category, priority in PATTERN_RULES:
            for pattern in patterns:
                if pattern in col_lower:
                    self._pattern_cache[col_lower] = category
                    logger.debug(
                        f"Columna '{col_name}' → {category.name} "
                        f"(patrón: '{pattern}', prioridad: {priority})"
                    )
                    return category
        
        # Sin match → UNKNOWN
        self._pattern_cache[col_lower] = PhysicalCategory.UNKNOWN
        logger.debug(f"Columna '{col_name}' sin categoría reconocida → UNKNOWN")
        return PhysicalCategory.UNKNOWN
    
    def add_rule(self, col_name: str, min_val: float, max_val: float) -> None:
        """
        Agrega un override específico para una columna.
        
        Los overrides tienen prioridad sobre el pattern matching.
        
        Parameters
        ----------
        col_name : str
            Nombre exacto de la columna (case-insensitive).
        min_val : float
            Valor mínimo permitido.
        max_val : float
            Valor máximo permitido.
            
        Raises
        ------
        ValueError
            Si min_val > max_val.
        """
        if min_val > max_val:
            raise ValueError(
                f"min_val ({min_val}) no puede ser mayor que max_val ({max_val})"
            )
        
        col_lower = col_name.lower().strip()
        self._overrides[col_lower] = (min_val, max_val)
        logger.info(f"Override agregado: '{col_name}' → [{min_val}, {max_val}]")
    
    def remove_rule(self, col_name: str) -> bool:
        """
        Elimina un override específico.
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna.
            
        Returns
        -------
        bool
            True si se eliminó, False si no existía.
        """
        col_lower = col_name.lower().strip()
        if col_lower in self._overrides:
            del self._overrides[col_lower]
            logger.info(f"Override eliminado: '{col_name}'")
            return True
        return False
    
    def has_rule(self, col_name: str) -> bool:
        """Verifica si existe un override para la columna."""
        return col_name.lower().strip() in self._overrides
    
    def list_overrides(self) -> Dict[str, Tuple[float, float]]:
        """Retorna copia del diccionario de overrides."""
        return self._overrides.copy()
    
    def list_categories(self) -> Dict[str, Tuple[float, float, str]]:
        """
        Retorna todas las categorías físicas con sus rangos.
        
        Returns
        -------
        Dict[str, Tuple[float, float, str]]
            Diccionario {nombre_categoría: (min, max, unidad)}
        """
        return {cat.name: cat.value for cat in PhysicalCategory}
    
    def validate_value(
        self, 
        col_name: str, 
        value: float,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida un valor individual contra el rango de su columna.
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna.
        value : float
            Valor a validar.
        strict : bool, default=False
            Si True, también rechaza valores en categoría UNKNOWN.
            
        Returns
        -------
        Tuple[bool, Optional[str]]
            (es_válido, mensaje_de_error_o_None)
        """
        import math
        
        # NaN siempre es inválido
        if math.isnan(value):
            return (False, "Valor es NaN")
        
        # Infinitos siempre son inválidos
        if math.isinf(value):
            return (False, f"Valor es infinito: {value}")
        
        category = self.get_category(col_name)
        
        # En modo estricto, rechazar UNKNOWN
        if strict and category == PhysicalCategory.UNKNOWN:
            return (False, f"Columna '{col_name}' sin categoría definida")
        
        min_val, max_val = self.get_range(col_name)
        
        if value < min_val:
            return (False, f"Valor {value} < mínimo {min_val}")
        if value > max_val:
            return (False, f"Valor {value} > máximo {max_val}")
        
        return (True, None)
    
    def analyze_columns(self, columns: List[str]) -> Dict[str, dict]:
        """
        Analiza una lista de columnas y retorna sus categorías detectadas.
        
        Útil para diagnóstico y debugging.
        
        Parameters
        ----------
        columns : List[str]
            Lista de nombres de columnas.
            
        Returns
        -------
        Dict[str, dict]
            Diccionario con análisis de cada columna.
        """
        analysis = {}
        for col in columns:
            category = self.get_category(col)
            min_val, max_val = self.get_range(col)
            analysis[col] = {
                "category": category.name,
                "min": min_val,
                "max": max_val,
                "unit": category.value[2] if category != PhysicalCategory.UNKNOWN else "?",
                "has_override": self.has_rule(col),
            }
        return analysis
    
    def clear_cache(self) -> None:
        """Limpia el cache de pattern matching."""
        self._pattern_cache.clear()
        logger.debug("Cache de patrones limpiado")
    
    def __repr__(self) -> str:
        n_overrides = len(self._overrides)
        n_cached = len(self._pattern_cache)
        n_categories = len(PhysicalCategory)
        return (
            f"MiningSchema(overrides={n_overrides}, "
            f"cached={n_cached}, categories={n_categories})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INSTANCIA GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA = MiningSchema()


# ═══════════════════════════════════════════════════════════════════════════
# CLI PARA DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("📋 Schema de Validación Física v2.0 - Proyecto Minero 4.0")
    print("=" * 70)
    
    # Mostrar categorías
    print("\n🏷️  CATEGORÍAS FÍSICAS UNIVERSALES:")
    print("-" * 70)
    for cat_name, (min_v, max_v, unit) in SCHEMA.list_categories().items():
        if min_v == float("-inf"):
            print(f"  {cat_name:25} → [   -∞   ,    +∞   ] {unit}")
        else:
            print(f"  {cat_name:25} → [{min_v:>8.1f}, {max_v:>8.1f}] {unit}")
    
    # Test con columnas de ejemplo
    print("\n🧪 TEST DE PATTERN MATCHING:")
    print("-" * 70)
    
    test_columns = [
        # Gold Recovery dataset
        "rougher.input.feed_au",
        "rougher.output.recovery",
        "primary_cleaner.state.floatbank8_a_level",
        "flotation_section_02_air_amount",
        # AI4I2020 dataset
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        # Columna desconocida
        "columna_random_xyz_123",
    ]
    
    for col in test_columns:
        cat = SCHEMA.get_category(col)
        min_v, max_v = SCHEMA.get_range(col)
        status = "✅" if cat != PhysicalCategory.UNKNOWN else "❓"
        print(f"  {status} {col:45} → {cat.name:20} [{min_v}, {max_v}]")
    
    print("\n" + "=" * 70)
    print(f"Schema: {SCHEMA}")
    print("=" * 70)
