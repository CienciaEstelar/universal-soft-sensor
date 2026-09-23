# 🔬 Hallazgos verificados — Universal Soft-Sensor

> Mapa honesto de dónde el pipeline encuentra señal real y dónde no.
> Todas las cifras salen de corridas reproducibles (scripts + JSON en esta carpeta).
> Sesión de validación: 2026-07-20.

---

## TL;DR — el sistema funciona como instrumento honesto

El pipeline **detecta señal donde la hay y reporta cero donde no la hay**, incluso
cuando eso contradice la literatura publicada. Encontró y neutralizó **4 modos de
autoengaño** (leakage autorregresivo, split degenerado, feature↔target tautológico,
persistencia inflada por estructura de datos). Un soft-sensor que reporta ceros
honestos vale más que uno que reporta 0.9 fraudulentos.

**Edge robusto confirmado:** recuperación metalúrgica de cobre (LCT) desde química
+ dureza, R²=0.33, sobrevive permutation test (p=0.005) y GroupKFold por sondaje,
e independiente de la ley de cabeza. Modesto pero real.

---

## Tabla maestra (todos los dominios probados)

| Dominio | Target | Régimen / split honesto | R² verificado | Veredicto |
|---|---|---|---|---|
| 🔧 ZeMA hidráulico | condición enfriador | sensor-only, estratificado | **0.9998** | ✅ señal fuerte real |
| ⛏️ GeoMet cobre | recuperación **LCT** | química+dureza, GroupKFold, perm p=0.005 | **0.33** | ✅ edge robusto modesto |
| 🛩️ NASA CMAPSS | RUL turbofan | sensor-only, split por unidad | 0.593 | 🟡 moderado (no SOTA) |
| ⛏️ GeoMet cobre | `xr` | química+dureza | 0.77 (pero r=0.88 con Cu) | ⚠️ tautológico — descartado |
| ⛏️ GeoMet cobre | `fr` | química+dureza, GroupKFold | 0.26 | 🟡 débil |
| ⚗️ SRU refinería | H₂S | sensor-only instantáneo | −0.47 | 🔴 falta lag de inputs |
| ⚗️ SRU refinería | H₂S | con lags | 0.57 | 🟡 (relearn persistencia) |
| ⛏️ Flotación hierro | % sílica concentrado | sensor-only horario | ~0.08 | 🔴 dominado por persistencia |
| ⛏️ Flotación hierro | sílica/hierro, 1-24h | forecast vs persistencia | negativo | 🔴 sin caso |
| ⚙️ AI4I 2020 | fallo (binario) | — | 0.17 | 🔴 fuera de alcance (clasificación) |
| 🚢 Turbina naval (**SIMULADO**) | degradación turbina | GroupKFold por nivel de degradación | 0.955 | 🟡 interpola bien; no extrapola con ruido (ver abajo) |
| 🚢 Turbina naval (**SIMULADO**) | degradación compresor | GroupKFold por nivel de degradación | 0.983 | 🟡 idem |

---

## Los 4 autoengaños que el sistema cazó

1. **Leakage autorregresivo (CMAPSS):** con lags del target, R² sube de 0.593 a 0.873
   — cuasi-persistencia, no señal. La mayoría de papers de RUL caen acá.
2. **Split degenerado (ZeMA):** el split secuencial ingenuo deja el test con una sola
   clase → R² indefinido. Corregido con split estratificado.
3. **Persistencia inflada por estructura (flotación hierro):** el target de lab se
   repite ~174 filas/hora → persistencia fila-a-fila da R²=0.998 falso. A resolución
   horaria honesta, la persistencia real es 0.61 y los sensores no la superan.
4. **Tautología feature↔target (GeoMet `xr`):** R²=0.77 parecía un triunfo, pero
   `xr` correlaciona 0.88 con la ley de Cu de entrada → predecir Cu desde Cu.
   El chequeo de tautología lo destapó.

---

## El edge robusto en detalle (GeoMet — recuperación de cobre)

**Pregunta:** ¿la química + dureza del mineral predicen la recuperación de flotación,
más allá de la ley de cabeza?

**Resultado (target LCT, locked-cycle test recovery):**
- Solo química: R²≈0 (la química sola no basta).
- Química + dureza (F80/P80/molienda): **R²=0.33**.
- Permutation test (200 barajadas): **p=0.005** → no es azar.
- GroupKFold por HOLEID (ningún sondaje en train y test): la señal se mantiene.
- Correlación LCT vs Cu de entrada: **r=0.07** → NO es tautología con la ley.

**Lectura metalúrgica:** que la dureza sea la que destraba la predicción tiene
sentido físico (la recuperación depende de qué tan bien se muele, no solo de la ley).
El modelo lo capta solo, sin que se lo digan.

**Salvedad honesta:** n=92 muestras es chico. El R²=0.33 es señal de dirección
robusta, no métrica final. Justifica conseguir más datos / un dataset de planta real
para confirmar y hacer crecer el número.

---

## Turbina a gas naval — caso NO minero (SIMULADO, 2026-09-22)

> ⚠️ **Datos simulados** (UCI 316, simulador de fragata, Coraddu et al. 2016, CC BY 4.0):
> grilla uniforme de degradación, sin ruido de sensor, sin tiempo. No es evidencia de
> desempeño en planta. Traído del repo legacy `proyecto_minero_4.0` (`train_naval.py`),
> donde se reportaba R²≈0.96 con split aleatorio. Artefacto: `naval_rigor.json`.

| Prueba | Turbina | Compresor |
|---|---|---|
| 1. Split aleatorio 80/20 (réplica legacy) | 0.959 | 0.983 |
| 2. GroupKFold por nivel de degradación | 0.955 | 0.983 |
| 3. Extrapolación a degradación peor que la vista — GB | **−5.97** (no predice bajo 0.984; real llega a 0.975) | **−4.62** |
| 3. idem — Ridge lineal | −1.13 | −2.07 |
| 3. idem — predecir la media del train | −20.5 | −19.6 |
| 4. Prueba 2 + ruido de sensor 0.1% / 0.5% / 1% | 0.926 / 0.772 / 0.580 | 0.956 / 0.802 / 0.604 |
| 5. GP extrapolando, sin ruido (R² / cobertura 95% / σ_ext÷σ_int) | 0.999 / 0.99 / 3.2 | 0.998 / 0.94 / 3.6 |
| 5b. GP extrapolando, ruido 0.1% | 0.924 / 0.96 / 1.2 | 0.887 / 0.91 / 1.1 |
| 5b. GP extrapolando, ruido 0.5% | **−19.5 / 0.11 / 1.0** | **−19.8 / 0.11 / 1.0** |
| 5b. GP extrapolando, ruido 1% | −11.8 / 0.29 / 1.05 | −9.0 / 0.36 / 1.02 |

**Lectura:**
- Entre niveles de degradación ya vistos, la señal física sensor→estado es real y
  fuerte (0.955–0.983), incluso dejando fuera niveles completos.
- El uso real (detectar degradación **peor** que la vista) es extrapolación, y ahí
  GradientBoosting falla por construcción: los árboles no predicen fuera del rango
  del train. Ridge falla menos pero también falla.
- El GP extrapola casi perfecto **sin ruido** y ensancha σ ~3x (degrada con gracia).
  Pero es el caso ideal para un GP: simulador suave y determinista.
- **Con ≥0.5% de ruido de sensor el GP se equivoca con confianza**: cobertura del IC
  95% de 0.11–0.36 fuera del rango y σ sin ensanchar (razón ≈1.0). Con 0.5% el R²
  interior colapsa a ≈0 (el kernel de ruido absorbe la señal); se repite idéntico con
  2 reinicios del optimizador, no es un mínimo local de un solo arranque.
- Los niveles de ruido son **supuestos de sensibilidad**, no ruido medido de una
  turbina real. Qué nivel es realista depende de la instrumentación.

**Implicación:** el caso sirve como demostración de que el pipeline encuentra señal
física directa cuando existe (contraste con flotación, donde no hay). No sirve como
validación industrial: el resultado depende críticamente del ruido, que el dataset no
tiene. El test P6 (`extrapolation_test`) es exactamente la herramienta que destapa esto.

---

## Scripts de reproducción (en la raíz del repo)

- `run_flotation_benchmark.py` — flotación hierro, sensor-only vs baseline.
- `run_flotation_hourly.py` — agregación horaria + baseline de persistencia honesto.
- `run_flotation_horizons.py` — multi-horizonte forecast vs persistencia.
- `run_geomet_recovery.py` — GeoMet, química(+dureza) → recuperación, 5-fold CV.
- `run_geomet_rigor.py` — GroupKFold + permutation test + chequeo de tautología.
- `prepare_naval_dataset.py` + `run_naval_rigor.py` — turbina naval (SIMULADO): split
  por nivel, extrapolación, ruido de sensor y GP con incertidumbre (~22 min).

Artefactos JSON: `flotation_*.json`, `geomet_recovery.json`, `geomet_rigor.json`.

---

## Deuda técnica que estos hallazgos destaparon

1. **Feature engineering de lags de INPUTS** (no solo del target): SRU y flotación lo
   necesitan. Es la mejora #1 del pipeline (ROADMAP P-input-lags).
2. **Soporte de clasificación**: AI4I fuera de alcance como regresor.
3. **GroupKFold / splits con grupos**: agregar al pipeline para datos con estructura
   de grupo (evita el leakage que casi infla GeoMet a 0.93).
