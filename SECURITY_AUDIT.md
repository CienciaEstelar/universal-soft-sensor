# 🔴 Auditoría Hostil (Red Team) — Universal Soft-Sensor

> Auditoría adversarial estilo "comprador que quiere romper el framework antes de pagar".
> Fecha: 2026-07-20. Metodología: ataques empíricos contra el código real, no checklist.
> Cada hallazgo fue **reproducido con un PoC ejecutable**.

---

## Resumen ejecutivo

Se probaron 10+ vectores de ataque. El pipeline es **sólido en el núcleo científico** (determinismo perfecto, anti-leakage temporal correcto, defensa DoS presente, validación física funcional), pero tiene **2 vulnerabilidades críticas** y **2 altas** que un comprador técnico encontraría en la primera hora. Ninguna es difícil de arreglar.

> ✅ **ESTADO 2026-07-20: TODOS LOS VECTORES BLINDADOS.** Los 7 hallazgos fueron
> corregidos en la rama `security/hardening-v2` y cubiertos por tests de
> regresión (`tests/test_security.py`, 11 tests). Suite completo: **79 passed,
> 2 skipped**. Cada fila de abajo lleva su fix y su test.

| # | Severidad | Vector | Estado | Fix |
|---|---|---|---|---|
| V1 | 🔴 **CRÍTICO** | RCE al cargar `.pkl` | ✅ **BLINDADO** | verificación SHA-256 (sidecar) + bloqueo de carga sin hash |
| V2 | 🔴 **CRÍTICO** | Leakage feature↔target → R²=1.0 falso | ✅ **BLINDADO** | `_drop_target_leakage` (\|r\|>0.999 → drop+warn; `strict_leakage` aborta) |
| V3 | 🟠 **ALTO** | Métricas engañosas en degenerados | ✅ **BLINDADO** | `evaluate()` → R²/MAPE = NaN en varianza-cero |
| V4 | 🟠 **ALTO** | Path traversal en filename del config | ✅ **BLINDADO** | contención `is_relative_to(data/)` en el adapter |
| V5 | 🟡 MEDIO | Errores crudos de sklearn | ✅ **BLINDADO** | validación amistosa en `load_data` (numérico/inf/min-filas/varianza) |
| V6 | 🟡 MEDIO | Cap del GP con muestreo por stride (aliasing) | ✅ **BLINDADO** | muestreo aleatorio seedeado (determinista + sin sesgo) |
| V7 | 🟢 BAJO | Doc dice "regex", implementa substring | ✅ **BLINDADO** | doc corregida en README/CLAUDE/paper |
| ✓ | — | Determinismo, DoS cap, anti-leakage temporal, split del scaler | **Ya estaba bien** | — |

---

## 🔴 V1 — RCE por deserialización de pickle (CRÍTICO)

**Qué.** `SoftSensorGP.load()` e `InferenceEngine.load_model()` usan `joblib.load()`, que es pickle. Cargar un `.pkl` de origen no confiable **ejecuta código arbitrario del atacante**.

**Por qué importa para un comprador.** El propio paper/README dice que el repo distribuye "modelos entrenados". `InferenceEngine` incluso auto-carga el "latest_model" de un directorio. Un comprador que baje un modelo preentrenado (del repo, de un colega, de un bucket) queda comprometido.

**PoC (ejecutado, `redteam/pickle_rce.py`):** un `.pkl` con `__reduce__` malicioso escribió `/tmp/pwned.txt` al cargarse. Payload ejecutado ✓.

**Fix.**
1. Nunca cargar `.pkl` de fuentes no confiables. Documentarlo explícitamente.
2. Migrar el formato de modelo a algo seguro: **`skops`** (diseñado para modelos sklearn, sin ejecución arbitraria) o serialización de solo-parámetros (JSON con los hiperparámetros + reconstrucción del estimador).
3. Como mínimo inmediato: firmar los `.pkl` distribuidos (hash SHA-256 en el repo) y verificar el hash antes de `load`.

---

## 🔴 V2 — Leakage feature↔target no detectado (CRÍTICO)

**Qué.** Si una feature es copia (o casi) del target, el pipeline **no lo detecta** y reporta R²≈1.0 — incluso cuando el target es ruido puro impredecible.

**PoC (ejecutado, `redteam/det_leak.py`):** target = `uniform(0,100)` **aleatorio** (R² real esperado ≈ 0), más una feature `leak_copy = target`. Resultado: **R²=1.0000**. El modelo "aprende" a copiar la feature filtrada.

**Causa raíz.** El pipeline elimina features altamente correlacionadas **entre sí** (feature↔feature, `r>0.98`), pero **nunca compara feature↔target**. Una columna que replica el target sobrevive el filtro.

**Por qué importa.** Es la fuente #1 de resultados fraudulentos en ML industrial. Un usuario que por error deje una columna downstream (o el mismo target con otro nombre) obtiene un "modelo perfecto" que colapsa en producción. Es exactamente el hermano del bug autorregresivo que ya documentamos en el paper.

**Fix.**
1. Antes de entrenar, calcular correlación **feature↔target**; si alguna feature supera un umbral (ej. `|r|>0.999`), **abortar con error claro** o excluirla con un `WARNING` prominente ("posible leakage: la columna X es cuasi-idéntica al target").
2. Extender la config `exclude_patterns` con detección semántica, no solo por nombre.

---

## 🟠 V3 — Métricas engañosas en casos degenerados (ALTO)

**Qué.** Con target de varianza cero (constante) el pipeline reporta **R²=1.0000**; con target todo-cero reporta **MAPE=0.0** ("perfecto"). Ambos son matemáticamente vacíos.

**PoC (ejecutado):** target constante `=42` → R²=1.0. Target `=0` → MAPE=0.0.

**Por qué importa.** Es el mismo tipo de "métrica que miente" que el split degenerado de ZeMA. Un pipeline serio debe **negarse a reportar** una métrica indefinida en vez de devolver un número que parece excelente.

**Fix.** En `evaluate()`:
- Si `var(y_true) == 0` → no calcular R² (devolver `NaN` + `ERROR: target de varianza cero, R² indefinido`).
- Si `y_true` tiene ceros y se pide MAPE → devolver `NaN` para esa fracción y avisar, no `0.0`.
- Verificar varianza del **test set** (no solo del train) antes de reportar.

---

## 🟠 V4 — Path traversal en el filename del config (ALTO)

**Qué.** `data_path = DATA_DIR / config["files"]["filename"]` sin contención. Un `filename = "../../../../etc/passwd"` resuelve **fuera** de `data/`.

**PoC (ejecutado):** `data/../../../../../../etc/passwd` → `/etc/passwd`. Sin ningún guard (`resolve()`, `is_relative_to`, `commonpath`) en el código.

**Por qué importa.** Menor que V1/V2 porque requiere un `dataset_config.json` malicioso, pero si el config viene de un tercero (marketplace de configs, CI, usuario no confiable), permite leer archivos arbitrarios del sistema.

**Fix.** Tras resolver el path, exigir contención:
```python
p = (DATA_DIR / filename).resolve()
if not p.is_relative_to(DATA_DIR.resolve()):
    raise ValueError("filename fuera de data/ — bloqueado")
```

---

## 🟡 V5 — Errores crudos en vez de validación amistosa (MEDIO)

Inputs adversariales (target con `inf`, target string, 1 sola fila, dataset vacío) **no crashean silenciosamente** (bien), pero devuelven stacktraces internos de sklearn (`RobustScaler`, `Input X contains infinity`) en vez de un mensaje de validación claro. Para un producto vendible, la capa de validación debería atrapar estos casos ANTES del modelo y explicar qué pasó. **No es inseguro, es falta de robustez de producto.**

**Fix.** Pre-chequeos en `load_data`: target numérico, ≥N filas mínimas, sin `inf`, varianza>0 — con mensajes claros.

---

## 🟡 V6 — `GP_MAX_SAMPLES` trunca con `iloc[:n]` (MEDIO)

La defensa DoS (cap de muestras para el GP O(n³)) **funciona** ✓, pero toma las **primeras n filas** (`iloc[:max]`), no una muestra representativa. En un dataset ordenado (como CMAPSS por unidad), esto entrena solo con los primeros motores → sesgo. **Es correcto como defensa de memoria, incorrecto como muestreo.**

**Fix.** Muestreo aleatorio seedeado o estratificado en vez de truncación cuando `len > max`.

---

## 🟢 V7 — Doc dice "regex", implementa substring (BAJO)

`include_patterns` se aplica con `pattern in columna` (substring), no regex. Esto **evita ReDoS** (bien, no es vulnerable), pero el paper y docstrings dicen "patrones regex". Inofensivo, pero corregir la doc para no prometer una capacidad que no existe (y que si se implementara como regex real, abriría ReDoS — mantener substring es la decisión segura).

---

## ✓ Lo que resistió los ataques (defensas confirmadas)

- **Determinismo perfecto**: dos corridas idénticas → R² idéntico hasta 1e-10 (Optuna seedeado). Reproducibilidad de nivel serio.
- **Anti-leakage temporal**: los lags usan `shift(1)+`, sin contaminación contemporánea.
- **Split anti-leakage del scaler**: `fit` solo en train, `transform` en test. Correcto.
- **Defensa DoS**: cap `GP_MAX_SAMPLES` evita el OOM del GP O(n³).
- **Validación física**: filtra correctamente valores fuera de rango (inf, 1e9 en temperatura → eliminados).
- **Sin ReDoS**: substring en vez de regex.

---

## Prioridad de blindaje (orden recomendado)

1. **V2 (leakage feature↔target)** — es el que destruye la credibilidad científica. 1 función, ~15 líneas.
2. **V3 (métricas degeneradas)** — barato, alto impacto en confianza. Guard en `evaluate()`.
3. **V1 (pickle RCE)** — migrar a skops o firmar hashes. El más grave en seguridad pura.
4. **V4 (path traversal)** — 3 líneas de contención.
5. V5/V6 — robustez de producto, no urgente.
6. V7 — doc.

**PoCs reproducibles:** `redteam/attack.py`, `redteam/pickle_rce.py`, `redteam/det_leak.py` (en outputs de la sesión).
