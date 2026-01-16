# ⚒️ Proyecto Minero 4.0: Inteligencia Artificial para Procesos de Flotación

<div align="center">

**Pipeline ETL Industrial & Soft-Sensor Predictivo para Calidad en Tiempo Real.**

---

</div>

## 📋 Resumen Ejecutivo

Este proyecto implementa una solución de **Minería 4.0** diseñada para optimizar plantas de procesamiento de minerales. Sustituye los costosos y lentos análisis de laboratorio por un **Soft-Sensor de Inteligencia Artificial** capaz de predecir la calidad del concentrado (ej. % de Sílice o Recuperación de Oro) en tiempo real, basándose en los datos de los sensores de la planta.

El sistema robusto combina la elegancia matemática de los **Procesos Gaussianos (GP)** con la potencia industrial del **Gradient Boosting**, asegurando precisión incluso en condiciones operativas cambiantes.

### 🎯 Objetivo Principal

Predecir variables críticas del proceso de flotación (Target) utilizando variables operativas (Features) con una precisión superior al 95%, permitiendo el control avanzado de procesos (APC).

---

## 🏗️ Arquitectura del Sistema

El sistema se divide en dos pipelines macro: **Entrenamiento (Batch)** y **Inferencia (Real-time Simulation)**.


graph TD
    %% Estilos Mineros
    classDef data fill:#333,stroke:#f4a261,stroke-width:2px,color:white;
    classDef process fill:#2a9d8f,stroke:white,stroke-width:2px,color:white,rx:5,ry:5;
    classDef ai fill:#e76f51,stroke:white,stroke-width:2px,color:white,rx:15,ry:15;
    classDef storage fill:#264653,stroke:#e9c46a,stroke-width:2px,color:white,stroke-dasharray: 5 5;

    subgraph "🏭 PISO DE PLANTA (Origen de Datos)"
        RawData[(🗄️ Datos Crudos Sensores)]:::data
    end

    subgraph "🛠️ PIPELINE DE ENTRENAMIENTO (train_universal.py)"
        RawData --> Adapter[🔌 Universal Adapter\n(Auto-Schema & Regex Filter)]:::process
        Adapter --> Validator[🛡️ Validación Física\n(Rangos Operativos)]:::process
        Validator --> Preproc[🧹 Preprocesamiento Robusto\n(Imputación & Outliers)]:::process
        
        Preproc --> FeatureEng[⚙️ Feature Engineering\n(Lags temporales, Diffs)]:::process
        
        subgraph "🧠 NÚCLEO DE IA (MiningGP Pro v4)"
            FeatureEng --> Optuna[⚡ Optimización de Hiperparámetros\n(Optuna 50 trials)]:::ai
            Optuna --> TrainDecision{¿GP Estable?}:::ai
            TrainDecision -- Sí --> TrainGP[Entrenar Gaussian Process\n(Kernel Industrial)]:::ai
            TrainDecision -- No (Fallback) --> TrainGBR[🚜 Entrenar Gradient Boosting\n(Modo 'Tanque')]:::ai
        end
    end

    subgraph "💾 MODEL REGISTRY"
        TrainGP --> Artifacts[(📦 Artefactos .pkl\nModelo + Scalers + Metadata)]:::storage
        TrainGBR --> Artifacts
    end

    subgraph "🔮 MOTOR DE INFERENCIA (predict_universal.py)"
        NewData(📡 Datos Nuevos/Simulados):::data --> InferenceEngine[🚀 Inference Engine\n(Carga Automática & Feature Gen)]:::process
        Artifacts -.-> InferenceEngine
        InferenceEngine --> Prediction((🎯 Predicción\nValor + Incertidumbre)):::ai
    end

```

---

## ✨ Características Clave (Senior Level)

* **🛡️ Ingesta Universal & Segura**: Adaptador agnóstico capaz de leer CSVs masivos, detectando automáticamente timestamps y separadores. Incluye filtrado por Regex para evitar *data leakage* de columnas futuras.
* **🧠 Modelado Híbrido Inteligente (v4)**:
* Intenta modelar con **Gaussian Process** (ideal para incertidumbre) usando kernels Matérn restringidos físicamente.
* Si el GP no supera un umbral de calidad (R² < 0.6), activa automáticamente un **Fallback a Gradient Boosting** (más robusto ante datos ruidosos o no estacionarios).


* **⏳ Conciencia Temporal**: Respeta estrictamente la flecha del tiempo en el entrenamiento (`shuffle=False`) y genera features de lags/ventanas móviles para capturar la dinámica del proceso.
* **🚀 Motor de Inferencia Dedicado**: Módulo independiente para producción que carga el modelo campeón automáticamente y asegura que los datos de entrada tengan el mismo esquema que en el entrenamiento.

---

## 🏆 Resultados de Desempeño

El sistema ha sido probado en datasets de minería real (ej. Gold Recovery), logrando una precisión excepcional al activar el modo de respaldo (Gradient Boosting).

| Métrica | Resultado (Gradient Boosting) | Interpretación Minera |
| --- | --- | --- |
| **R² Score** | **0.9707** | El modelo explica el **97%** de la variabilidad del proceso. Excelente. |
| **MAPE** | **1.43%** | El error porcentual promedio es menor al 1.5%. Calidad de laboratorio. |
| **RMSE** | **1.74** | Desviación estándar baja en las mismas unidades de la variable objetivo. |

> 📉 **Nota:** Los gráficos detallados de ajuste y análisis de residuos se generan automáticamente en la carpeta `results/` después de cada entrenamiento.

---

## 🚀 Instalación y Configuración

### 1. Clonar y preparar entorno

```bash
git clone https://github.com/CienciaEstelar/proyecto_minero_4.0.git
cd proyecto_minero_4.0

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

```

### 2. Configurar Datos y Variables

1. Coloca tu archivo CSV de sensores en la carpeta `data/`.
2. Edita el archivo `config/dataset_config.json` para apuntar a tu archivo y definir tu columna objetivo (Target).
3. (Opcional) Copia `.env.example` a `.env` para ajustar parámetros avanzados.

---

## 🎮 Uso del Sistema

El proyecto cuenta con una interfaz de línea de comandos (CLI) profesional impulsada por la librería `rich`.

### 🏋️‍♂️ Entrenamiento (Training Pipeline)

Ejecuta el orquestador universal. Él se encargará de todo el flujo ETL y el modelado.

```bash
python train_universal.py

```

*Si el entrenamiento es exitoso, el modelo campeón se guardará automáticamente en la carpeta `models/`.*

### 🔮 Inferencia (Simulación de Producción)

Prueba el modelo guardado simulando datos en tiempo real.

```bash
python predict_universal.py

```

*Esto cargará el último modelo y mostrará una tabla comparativa de "Valor Real vs. Predicción IA" para validar el desempeño.*

---

## 📂 Estructura del Proyecto

```bash
proyecto_minero_4.0/
├── config/                  # ⚙️ Configuración del sistema (JSON y Python)
├── core/                    # 🧠 El Cerebro del sistema
│   ├── adapters/            # Conectores de datos (Ingesta)
│   ├── models/              # Lógica de los modelos de IA (GP Pro v4)
│   ├── validation/          # Reglas de negocio y física
│   ├── inference_engine.py  # Motor de predicción para producción
│   └── ...
├── data/                    # 🗄️ Almacenamiento de datos (ignorado por git)
├── models/                  # 📦 Registro de modelos entrenados (.pkl)
├── results/                 # 📊 Gráficos y reportes de desempeño
├── logs/                    # 📝 Trazabilidad de ejecución
├── train_universal.py       # 🚀 Orquestador de Entrenamiento (CLI)
├── predict_universal.py     # 🔮 Orquestador de Inferencia (CLI)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación

```

---

<div align="center">

**Desarrollado con ⛏️ y 🧠 para la Industria 4.0**

Juan Galaz | Arquitectura Minera

</div>
