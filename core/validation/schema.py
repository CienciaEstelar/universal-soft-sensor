"""
Módulo: core/validation/schema.py
Descripción: Define los rangos físicos válidos para cada variable del proceso.
             Cualquier valor fuera de estos rangos se considera error de sensor.
             
Notas de Ingeniería:
    - Los rangos están basados en límites físicos reales de procesos de flotación.
    - Las columnas de flotación (01-07) usan reglas genéricas por patrón.
    - Si una columna no tiene regla definida, se permite cualquier valor.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _default_rangos() -> Dict[str, Tuple[float, float]]:
    """
    Factory function para crear el diccionario de rangos físicos.
    
    Unidades documentadas:
    - Porcentajes: 0-100 (%)
    - Flujos: típicamente m³/h o L/min según el proceso
    - pH: escala universal 0-14
    - Densidad: kg/L (densidad de pulpa típica)
    """
    return {
        # === Calidad del Mineral (Porcentajes 0-100%) ===
        "_iron_feed": (0.0, 100.0),           # % Fe en alimentación
        "_silica_feed": (0.0, 100.0),         # % SiO₂ en alimentación
        "_iron_concentrate": (0.0, 100.0),    # % Fe en concentrado (target secundario)
        "_silica_concentrate": (0.0, 100.0),  # % SiO₂ en concentrado (target principal)
        
        # === Reactivos y Flujos (Valores no negativos) ===
        "starch_flow": (0.0, 10000.0),        # Flujo de almidón (g/ton o similar)
        "amina_flow": (0.0, 10000.0),         # Flujo de amina (g/ton o similar)
        "ore_pulp_flow": (0.0, 10000.0),      # Flujo de pulpa (m³/h típico)
        
        # === Química del Proceso ===
        "ore_pulp_ph": (0.0, 14.0),           # pH - escala física universal
        "ore_pulp_density": (0.5, 5.0),       # Densidad pulpa (kg/L) - agua=1, mineral denso~3-4
        
        # === Columnas de Flotación (Reglas Genéricas) ===
        # Aplicadas por pattern matching para flotation_column_XX_air_flow y _level
        "flotation_column_air_flow": (0.0, 1000.0),  # Flujo de aire (Nm³/h o L/min)
        "flotation_column_level": (0.0, 1000.0),    # Nivel en celda (mm o cm)
    }


@dataclass
class MiningSchema:
    """
    Esquema de validación física para datos de proceso minero.
    
    Uso:
        from core.validation.schema import SCHEMA
        min_val, max_val = SCHEMA.get_range("ore_pulp_ph")
        
    Atributos:
        RANGOS_FISICOS: Diccionario con límites (min, max) por columna.
    """
    
    RANGOS_FISICOS: Dict[str, Tuple[float, float]] = field(default_factory=_default_rangos)
    
    def get_range(self, col_name: str) -> Tuple[float, float]:
        """
        Obtiene el rango válido para una columna.
        
        Estrategia de búsqueda:
        1. Búsqueda exacta por nombre
        2. Búsqueda por patrón (columnas de flotación seriadas)
        3. Fallback a (-inf, +inf) si no hay regla
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna (ya sanitizado).
            
        Returns
        -------
        Tuple[float, float]
            (valor_mínimo, valor_máximo) permitidos.
        """
        # 1. Búsqueda exacta
        if col_name in self.RANGOS_FISICOS:
            return self.RANGOS_FISICOS[col_name]
        
        # 2. Búsqueda por patrón para columnas de flotación
        # Ejemplo: "flotation_column_01_air_flow" → usa regla genérica
        if "flotation" in col_name:
            if "air_flow" in col_name:
                return self.RANGOS_FISICOS["flotation_column_air_flow"]
            if "level" in col_name:
                return self.RANGOS_FISICOS["flotation_column_level"]
        
        # 3. Sin regla definida → permitir todo
        logger.debug(f"Columna '{col_name}' sin regla de validación definida.")
        return (-float("inf"), float("inf"))
    
    def add_rule(self, col_name: str, min_val: float, max_val: float) -> None:
        """
        Agrega o actualiza una regla de validación.
        
        Parameters
        ----------
        col_name : str
            Nombre de la columna.
        min_val : float
            Valor mínimo permitido.
        max_val : float
            Valor máximo permitido.
        """
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) no puede ser mayor que max_val ({max_val})")
        
        self.RANGOS_FISICOS[col_name] = (min_val, max_val)
        logger.info(f"Regla agregada: {col_name} → [{min_val}, {max_val}]")
    
    def list_rules(self) -> Dict[str, Tuple[float, float]]:
        """Retorna copia del diccionario de reglas."""
        return self.RANGOS_FISICOS.copy()
    
    def has_rule(self, col_name: str) -> bool:
        """Verifica si existe una regla exacta para la columna."""
        return col_name in self.RANGOS_FISICOS


# =============================================================================
# Instancia global lista para importar
# =============================================================================
SCHEMA = MiningSchema()


# =============================================================================
# CLI para inspección
# =============================================================================
if __name__ == "__main__":
    print("📋 Esquema de Validación Física - Proyecto Minero 4.0")
    print("=" * 60)
    
    for col, (min_v, max_v) in SCHEMA.list_rules().items():
        print(f"  {col:40} → [{min_v:>10.2f}, {max_v:>10.2f}]")
    
    print("=" * 60)
    print(f"Total de reglas definidas: {len(SCHEMA.list_rules())}")
