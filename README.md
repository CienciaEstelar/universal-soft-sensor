# 🏭 Proyecto Minero 4.0

Pipeline ETL industrial y Soft-Sensor basado en Gaussian Process para predicción de calidad en procesos de flotación de mineral.

## 📋 Descripción

Este proyecto implementa un sistema completo para:

1. **Ingesta de datos** de sensores industriales (CSV con auto-detección de formato)
2. **Validación física** de rangos según el proceso de flotación
3. **Limpieza y preprocesamiento** robusto de datos de sensores
4. **Modelado predictivo** usando Gaussian Process para soft-sensing

El objetivo principal es predecir el **% de Sílice en concentrado** a partir de variables de proceso, funcionando como un "sensor virtual" (soft-sensor) que puede complementar o reemplazar mediciones de laboratorio costosas y con delay.

## 🏗️ Estructura del Proyecto

```
proyecto-minero-4.0/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuración centralizada
├── core/
│   ├── adapters/
│   │   └── mining_csv_adapter.py   # Ingesta universal de CSV
│   ├── validation/
│   │   ├── schema.py         # Rangos físicos válidos
│   │   └── validator.py      # Filtrado por validez física
│   ├── models/
│   │   └── mining_gp_pro.py  # Soft-Sensor GP
│   ├── preprocessor.py       # Limpieza de datos
│   └── pipeline.py           # Orquestador ETL
├── tools/
│   └── scan_schema.py        # Utilidad de diagnóstico
├── data/
│   ├── raw/                  # Datos crudos
│   └── processed/            # Datos limpios
├── models/                   # Modelos entrenados (.pkl)
├── results/                  # Gráficos y reportes
├── logs/                     # Logs de ejecución
├── .env.example              # Plantilla de configuración
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Instalación

### Opción 1: Instalación con pip (recomendado)

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/proyecto-minero.git
cd proyecto-minero

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar en modo editable
pip install -e .
```

### Opción 2: Instalación tradicional

```bash
pip install -r requirements.txt
```

### Configuración

```bash
# Copiar plantilla de configuración
cp .env.example .env

# Editar con tu ruta al dataset
nano .env
```

## 📊 Uso

### 1. Verificar estructura del dataset

```bash
python -m tools.scan_schema
```

### 2. Ejecutar pipeline ETL

```bash
# Usando el comando instalado
mining-pipeline

# O directamente
python -m core.pipeline

# Con opciones
python -m core.pipeline --estrategia interpolate --outliers
```

### 3. Entrenar Soft-Sensor

```bash
# Usando el comando instalado
mining-gp

# O directamente
python -m core.models.mining_gp_pro

# Con opciones
python -m core.models.mining_gp_pro --trials 20 --test-size 0.3
```

### Uso programático

```python
from core import MiningPipeline, MiningGP

# ETL
pipeline = MiningPipeline(estrategia_limpieza="interpolate")
stats = pipeline.run()

# Modelo
model = MiningGP(target_col="_silica_concentrate")
metrics = model.train_from_file()
print(f"R² = {metrics.r2:.4f}")

# Predicción
y_pred, y_std = model.predict(X_new)
```

## 🔧 Configuración

Variables de entorno (`.env`):

| Variable | Descripción | Default |
|----------|-------------|---------|
| `MINING_DATA_RAW_PATH` | Ruta al dataset crudo | `data/MiningProcess...csv` |
| `CHUNK_SIZE` | Filas por chunk | `25000` |
| `GP_TARGET` | Columna objetivo | `_silica_concentrate` |
| `GP_MAX_SAMPLES` | Máx. muestras para GP | `5000` |
| `GP_TRIALS` | Trials de Optuna | `15` |
| `PREPROCESS_STRATEGY` | Estrategia imputación | `ffill` |

## 📈 Resultados

El Soft-Sensor típicamente logra:
- **R² > 0.85** en predicción de % Sílice
- **Incertidumbre calibrada** (intervalos de confianza 95%)
- **Latencia < 1s** para predicciones en tiempo real

### Gráficos generados

- `control_chart_*.png`: Series temporales Real vs Predicho
- `scatter_fit_*.png`: Gráfico de ajuste con R²
- `error_analysis_*.png`: Distribución de residuos

## 🧪 Testing

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Con cobertura
pytest --cov=core --cov-report=html
```

## 📚 Documentación Técnica

### Arquitectura del Pipeline

```
CSV Crudo → Adapter (auto-detección) → Validator (rangos físicos) 
         → Preprocessor (imputación) → CSV Limpio
```

### Kernel del GP

El modelo usa un kernel industrial optimizado:

```
K(x, x') = σ² · Matérn(x, x'; ν, l) + σ_n² · δ(x, x')
```

- **Matérn (ν=1.5)**: Captura la dinámica suave de procesos físicos
- **WhiteKernel**: Modela el ruido de sensores
- **RobustScaler**: Maneja outliers típicos de sensores industriales

## 🤝 Contribuir

1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## 👤 Autor

**Juan Galaz**

---

*Desarrollado para optimización de procesos de flotación minera* 🏔️
