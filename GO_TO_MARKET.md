# 🎯 GO-TO-MARKET — Universal Soft-Sensor (recuperación geometalúrgica)

> Guion honesto para conseguir un piloto pagado. No es un plan de venta de software;
> es un plan para probar valor sobre datos reales y ganar el contrato por entrega,
> no por slides. Todas las afirmaciones se apoyan en `results/verification/FINDINGS.md`.

---

## 0. Regla de oro

**No competimos por mostrar el número más grande a un comprador que no distingue
honesto de inflado — competimos por ser el único que sobrevive cuando lo prueban
sobre data real que nadie vio antes.** Todo este documento se construye sobre esa idea.

---

## 1. Qué vendemos (posicionamiento)

Un **soft-sensor geometalúrgico de recuperación de cobre**: predice la recuperación
metalúrgica a partir de la química de sondaje + dureza del mineral, capturando la
parte que **la ley de cabeza NO te dice**.

Lo que NO vendemos: predicción de la ley del concentrado (probamos que está dominada
por persistencia, no es sensor-predecible — y el que te la vende con R²=0.9 te está
mostrando la ley de cabeza disfrazada).

---

## 2. La propuesta de valor — en pesos, no en R²

**El corazón del negocio (leer esto primero):** el ensayo metalúrgico que mide la
recuperación de verdad (tipo locked-cycle test / LCT) es **caro y lento**, así que
se corre en **pocas** muestras del cuerpo mineral. Los ensayos de **química de
sondaje** existen en **miles** de muestras (son baratos). Nuestro modelo **extiende
la recuperabilidad medida en el laboratorio caro a TODO el cuerpo mineral** usando la
química barata que ya tienen. Eso es el valor: no reemplazar el LCT, sino **cubrir
los miles de metros de sondaje donde nunca vas a correr un LCT.**

En pesos, eso se traduce en:
- **Planificación de blending / secuencia de mina** con recuperación predicha en todo
  el modelo de bloques, no solo en las pocas muestras con ensayo metalúrgico.
- **Menos sorpresas** metalúrgicas: anticipar mineral duro/refractario antes de planta.
- **Ahorro de ensayos:** priorizar dónde SÍ vale correr el LCT caro.

Punto clave que nos separa del que "predice la ley": nuestro modelo es
**independiente de la ley de cabeza** (correlación r=0.07 con el Cu de entrada; ver
FINDINGS.md). No estamos disfrazando la ley — capturamos la interacción química+dureza
que la ley sola no te da.

Pitch de una línea:
> *"Extendemos la recuperabilidad que hoy solo miden en pocas muestras de laboratorio
> a todo el cuerpo mineral, usando la química de sondaje que ya tienen. Lo pueden
> verificar sobre sus propios sondajes retenidos antes de pagar."*

> ⚠️ **Lo que NO afirmamos:** que le ganamos a su tabla geometalúrgica actual. Eso no
> lo sabemos hasta el piloto — y es exactamente lo que el criterio de aceptación
> (§4) mide. El R²=0.33 (n=96, data pública) prueba que **el método tiene señal real
> y no-tautológica**, no que supera el método del cliente. Prometer lo segundo antes
> del piloto es el error que hay que evitar.

---

## 3. El arma: la "prueba de 5 minutos" (cláusula, no promesa)

No convencemos al cliente de la diferencia entre honesto e inflado. Le damos UNA
prueba que la expone sola, y la ponemos como cláusula del piloto:

> **"Entrenamos con los sondajes que ustedes elijan y predecimos sobre un conjunto de
> sondajes que apartamos y NUNCA vimos (held-out por HOLEID). El número que cuenta es
> ese, no el de entrenamiento."**

Efecto: cualquier competidor con números inflados (leakage, tautología con la ley,
sobreajuste) ve su R² desplomarse en ese test. El nuestro se sostiene — porque ya lo
validamos así (GroupKFold por sondaje + permutation test, ver FINDINGS.md). No hace
falta que el cliente entienda estadística; la física de la prueba hace el trabajo.

---

## 4. Estructura del piloto (criterio de aceptación PRE-acordado)

- **Alcance:** 1 yacimiento / 1 dataset geometalúrgico del cliente.
- **Criterio de éxito, firmado antes de empezar:** el baseline (el "método actual"
  del cliente: su tabla geometalúrgica, su regla por dominio, o "usar solo la ley")
  y el margen X se **co-definen por escrito con el metalurgista del cliente ANTES**
  de correr nada. Sin baseline acordado no hay piloto — esto protege a ambas partes
  de discutir el resultado después. La métrica es error de predicción de recuperación
  sobre sondajes **held-out por HOLEID** (nunca vistos en entrenamiento).
- **Pago condicionado:** fee de setup + fee de éxito atado al criterio. Si no se
  cumple sobre held-out, no se cobra el fee de éxito. Esto nos separa de los que
  cobran por slides.
- **Duración:** semanas, no meses (el modelo corre en minutos; el tiempo es acceso
  a datos + acordar el criterio).
- **Entregable:** predicción + intervalo de incertidumbre + reporte que explica QUÉ
  variables mueven la recuperación (química vs dureza), no una caja negra.

---

## 5. A quién le vendemos

- **Aliado técnico (entra por acá):** el metalurgista / ingeniero de procesos /
  geometalurgista. Es el que huele un 0.99 tautológico a la legua y valora que no
  mintamos. Se le vende la técnica y la prueba held-out.
- **Decisor (firma):** superintendente / gerente de planta o de geometalurgia. Se le
  vende el resultado del piloto y el $ ahorrado, no el R².

---

## 6. Diferenciador vs competidores inflados

| Ellos (número inflado) | Nosotros (número verificado) |
|---|---|
| R²=0.99 en slides, sin held-out real | R²=0.33 que **sobrevive** held-out + permutation |
| Caja negra | Explicamos qué variables mueven la recuperación |
| Cobran por la promesa | Fee de éxito atado a la prueba |
| Colapsan en el piloto | Entregamos y ganamos el 2º contrato |

En la minería chilena (pueblo chico) fallar un piloto te quema para siempre.
Prometer poco y cumplir construye la reputación que trae el siguiente contrato.

---

## 7. Vías de entrada (programas de innovación — no venta fría)

Un estudiante solo no le vende en frío a Codelco. La puerta son los programas que
pagan pilotos y dan acceso a la minera:

- **Expande** (BHP + Codelco + AMSA + Fundación Chile) — desafíos de innovación abierta.
- **BHP / Escondida** — pilotos con startups (fondos dedicados).
- **Upscale Mining** (Centro de Innovación UC) — aceleración de soluciones mineras.
- **Start-Up Chile** — capital + red.
- **AMTC (U. de Chile)** — vía académica; ya hicieron gemelo digital para El Teniente.
- **Memoria de título con una minera** — tu estatus de estudiante es ventaja de acceso.

El paper con DOI + los artefactos de verificación son tu kit de credibilidad para
postular: te separan del 95% que llega con un notebook y un R² sin respaldo.

---

## 7b. Objeciones que te van a hacer (y la respuesta honesta)

Prep de reunión — estas son las preguntas reales de un gerente/metalurgista escéptico:

- **"0.33 es bajísimo, mi tabla geometalúrgica ya da eso."**
  → "Puede ser. Por eso no le pido que me crea: el piloto mide si le gano a SU tabla
  sobre sondajes que aparto. Si no le gano, no cobro el fee de éxito. El 0.33 solo
  prueba que el método tiene señal real, no tautológica — el resto lo decide su data."

- **"El LCT ya lo mido, ¿para qué predigo?"**
  → "El LCT lo corre en decenas de muestras; yo lo extiendo a los miles de metros de
  sondaje donde nunca va a correr un LCT. Es cubrir el cuerpo mineral completo con la
  química que ya tiene, no reemplazar su laboratorio."

- **"GeoMet es un depósito público, no mi pórfido."**
  → "Exacto, por eso el piloto es sobre SUS datos. GeoMet prueba que el método funciona
  con rigor (permutation test, sondajes held-out); la transferencia a su yacimiento es
  justo lo que el piloto confirma antes de escalar."

- **"¿Por qué le creo a un estudiante y no a [vendedor establecido]?"**
  → "No me crea. Deme sondajes que no vea, predigo, y comparamos contra su método. El
  código y la validación son públicos (paper con DOI). Los números grandes de otros
  ¿los pueden mostrar sobre datos held-out que no entrenaron?"

- **"n=96 es sobreajuste seguro."**
  → "Lo testeamos: permutation test p=0.005 y validación cruzada por sondaje (ningún
  hoyo en train y test a la vez). El número sobrevive el rigor. Con sus miles de
  muestras, crece y se estabiliza."

## 7c. Competidores reales (no el hombre de paja)

Existen actores serios, hay que nombrarlos y diferenciarse honesto:

- **SensoFlot (HighService, Chile):** sensor de interfase en flotación, +1-2% recuperación.
  Es hardware en planta, tiempo real. **Nosotros somos software offline sobre sondaje**
  (planificación, no control en vivo) — complementario, no competencia directa.
- **Petra Data Science, Genius Mining AI, etc.:** analítica minera con IA, ya hacen
  pilotos con BHP/Gold Fields. Más grandes y financiados. **Nuestra diferencia honesta:**
  agnóstico de proveedor, barato, local (Chile), y con validación reproducible y
  verificable (no caja negra, no números sin held-out).
- **Módulos de APC de AVEVA/AspenTech/Honeywell:** control de proceso en planta. Otro
  problema (proceso en vivo), no geometalurgia de sondaje.

Posición honesta: no somos los más grandes ni los más financiados. Somos los más
**verificables y baratos** para un problema específico (recuperabilidad geometalúrgica
desde química de sondaje) donde la integración es liviana (offline, no DCS).

## 8. Líneas rojas de integridad (no cruzar)

- **Nunca** vender el R²=0.93 tautológico. Se cae ante un metalurgista y te quema.
- **Nunca** prometer un número sobre datos que no probaste held-out.
- **Nunca** ocultar que la validación pública fue con n chico (53-96). Declararlo
  y convertirlo en el argumento del piloto ("con tus datos crece") es más fuerte.
- **Nunca** vender "predecir la ley del concentrado" — probamos que no hay señal.

---

## 9. Estado de madurez honesto

- ✅ Pipeline validado, blindado (7 vectores de seguridad, ver `SECURITY_AUDIT.md`),
  reproducible (artefactos + tests).
- ✅ Edge real modesto verificado con rigor (recuperación LCT, R²=0.33).
- ✅ **Ventaja de integración:** el modelo geometalúrgico es **offline** (corre sobre
  la base de sondajes para planificación de mina), NO un soft-sensor de proceso en
  tiempo real atado al DCS/PI. Eso hace la integración MUCHO más liviana: es una
  predicción batch sobre datos de ensayo que el cliente ya exporta, no una conexión
  en vivo a la planta. Menor barrera de entrada, piloto más rápido.
- 🚧 **No es un producto llave en mano todavía:** falta validación sobre una base de
  sondajes real grande, calibración de incertidumbre (PICP), y empaquetado del flujo
  (hoy son scripts). Pero el gap es de producto, no de método.
- 🎯 **Lo que se vende hoy es el piloto**, no el software. El software es el motor;
  el negocio es el resultado sobre los datos del cliente.

---

## 10. Próximos pasos concretos

1. Convertir `FINDINGS.md` en un **one-pager** visual (2 gráficos: el edge que
   sobrevive el permutation test, y la tabla honesta de dominios).
2. Postular a **Expande / Upscale Mining** con el kit (paper + one-pager + repo).
3. Buscar **una Memoria con una minera** para conseguir el primer dataset real grande.
4. Construir el **feature engineering de lags de inputs** (P0 del ROADMAP) — lo que
   destapamos como la mejora técnica #1.
5. Cuando haya dataset real: correr el piloto con la cláusula held-out del punto 3.
