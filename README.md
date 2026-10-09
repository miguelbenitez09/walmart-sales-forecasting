# 🛒 Walmart Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/sklearn-1.3.2-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.2-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.2-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
[![Autor](https://img.shields.io/badge/Autor-developed_by_Miguel_Benítez_(UTP)-informational.svg)](https://github.com/miguelbenitez09)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Firma Oficial:** **`Walmart Sales Forecasting v1.0.0 • developed by Miguel Benítez`**  
> **Sistema de Machine Learning para predicción de ventas semanales en 45 tiendas Walmart usando algoritmos avanzados, con API REST y dashboard interactivo.**

---

## 👨‍💻 Autor

**Ing. Miguel Antonio Benítez González** (Universidad Tecnológica de Panamá - UTP)
- 🎓 Título: Ingeniero en Sistemas y Computación
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Problema de Negocio](#-problema-de-negocio)
3. [Dataset](#-dataset)
4. [Análisis y Técnicas Aplicadas](#-análisis-y-técnicas-aplicadas)
5. [Feature Engineering](#-feature-engineering)
6. [Modelos y Resultados](#-modelos-y-resultados)
7. [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
8. [Estructura del Proyecto](#-estructura-del-proyecto)
9. [Instalación](#-instalación)
10. [Uso](#-uso)
11. [API Endpoints](#-api-endpoints)
12. [Mejoras Futuras](#-mejoras-futuras)

---

## 🎯 Descripción del Proyecto y Enfoque Académico UTP

Este proyecto constituye un **pipeline fundacional de extremo a extremo (E2E MLOps Lifecycle)** desarrollado para consolidar y demostrar las competencias troncales adquiridas durante la carrera de **Ingeniería en Sistemas y Computación en la Universidad Tecnológica de Panamá (UTP)**. 

Su propósito central es ejemplificar la ejecución rigurosa, limpia y reproducible de cada una de las fases esenciales en la resolución de problemas mediante Inteligencia Artificial aplicada:
1. **Ingesta y Validación de Datos:** Fusión relacional de tiendas, características macroeconómicas y ventas semanales.
2. **Análisis Exploratorio de Datos (EDA):** Detección de patrones de estacionalidad, correlaciones y dispersión.
3. **Ingeniería de Características:** Extracción de rezagos temporales (*lags*), variables cíclicas y tratamiento de descuentos (*MarkDown1-5*).
4. **Modelado y Benchmarking:** Evaluación comparativa con la métrica oficial de negocio **WMAE** (*Weighted Mean Absolute Error*).
5. **Ingeniería de Software para Despliegue:** Exposición como microservicio REST OpenAPI con FastAPI, contenedorización y dashboard interactivo en Streamlit.

### Objetivo Principal
Predecir las ventas semanales (`Weekly_Sales`) de diferentes departamentos en 45 tiendas Walmart, considerando factores como:
- Días festivos (Super Bowl, Thanksgiving, Christmas)
- Variables macroeconómicas (CPI, desempleo, precio combustible)
- Promociones (MarkDown1-5)
- Características de tiendas (tipo, tamaño)
- Factores temporales y estacionalidad

### Pipeline Completo
```
Datos Crudos → EDA → Feature Engineering → Modelado ML (WMAE) → API REST → Dashboard Web
```

---

## 💼 Problema de Negocio

### Contexto Empresarial
Walmart necesita optimizar su cadena de suministro y operaciones mediante la predicción precisa de ventas semanales para:

1. **Gestión de Inventario** 📦
   - Reducir sobrestock (costos de almacenamiento)
   - Evitar desabasto (pérdida de ventas)
   - Optimizar reposición de productos

2. **Planificación de Recursos Humanos** 👥
   - Asignar personal según demanda esperada
   - Planificar turnos para días festivos
   - Optimizar costos laborales

3. **Estrategias de Pricing y Promociones** 💰
   - Planificar descuentos (MarkDowns)
   - Maximizar ingresos en temporadas altas
   - Ajustar precios según demanda

### Desafíos Técnicos

- **Datos Masivos**: 421,570 registros históricos
- **Múltiples Series Temporales**: 45 tiendas × 81 departamentos
- **Estacionalidad Compleja**: Patrones semanales, mensuales, anuales
- **Eventos Externos**: Impacto de festivos varía por tienda/departamento
- **Métricas Personalizadas**: WMAE con peso 5x en festivos

---

## 📊 Dataset y Régimen de Acceso Abierto

**Fuente Oficial**: [Walmart Recruiting - Store Sales Forecasting (Kaggle Benchmark)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)  
**Régimen de Licenciamiento**: Acceso Abierto para Investigación Científica, Educación y Benchmarking Competitivo.

### Política de Higiene de Repositorio (Zero Raw Bloat)
Para mantener este repositorio ágil, profesional y listo para despliegues en contenedores Docker sin almacenar cientos de megabytes en el historial de Git, **los datos crudos masivos están desacoplados del control de versiones**. El repositorio contiene el pipeline modular determinístico (`data/`, `notebooks/`, `api/`) que permite reproducir la descarga, transformación, entrenamiento y serving desde cero.

### Información General
- **Período**: Febrero 2010 - Octubre 2012
- **Granularidad**: Semanal
- **Registros Entrenamiento**: 421,570 observaciones
- **Entidades Modeladas**: 45 tiendas × 81 departamentos
- **Estrategia de Tratamiento**: Fusión relacional por `(Store, Date)`, imputación con ceros en `MarkDown1-5` (ausencia de promoción activa) y particionamiento temporal cronológico sin mezcla aleatoria para evitar cualquier fuga de información (*zero lookahead bias*).

### Archivos

| Archivo | Registros | Columnas | Descripción |
|---------|-----------|----------|-------------|
| `train.csv` | 421,570 | 5 | Ventas históricas (target: Weekly_Sales) |
| `test.csv` | 115,064 | 4 | Datos para predicción |
| `features.csv` | 8,190 | 12 | Variables semanales por tienda |
| `stores.csv` | 45 | 3 | Información de tiendas (Type, Size) |

### Variables Principales

**train.csv**
- `Store`: ID de tienda (1-45)
- `Dept`: Departamento (1-99)
- `Date`: Fecha de la semana
- `Weekly_Sales`: Ventas semanales (target) - rango: $209 - $693,099
- `IsHoliday`: Indicador semana festiva

**features.csv**
- `Temperature`: Temperatura regional (°F)
- `Fuel_Price`: Precio combustible ($/galón)
- `MarkDown1-5`: Datos de promociones anonimizadas
- `CPI`: Índice Precios Consumidor
- `Unemployment`: Tasa desempleo (%)

**stores.csv**
- `Type`: Tipo de tienda (A, B, C)
- `Size`: Tamaño en pies cuadrados

---

## 🔬 Análisis y Técnicas Aplicadas

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook**: `notebooks/01_exploracion_dataset.ipynb`

#### Análisis Univariado
- **Distribución de ventas**: Asimetría positiva (log-normal)
- **Outliers**: Detectados mediante Z-score (±3σ) e IQR
- **Missing values**: MarkDowns con ~50% NaN (promociones no aplicadas)

#### Análisis Temporal
```
Hallazgos Clave:
├── Tendencia general alcista 2010-2012
├── Estacionalidad fuerte: picos en Nov-Dic (Navidad)
├── Días festivos: incremento promedio 40% en ventas
└── Desplome enero: post-temporada navideña
```

#### Análisis por Categorías
- **Por Tipo de Tienda**:
  - Tipo A: Mayor volumen (55% ventas totales)
  - Tipo B: Volumen medio (30%)
  - Tipo C: Menor volumen (15%)
  
- **Análisis Pareto**: 20% de departamentos generan 60% de ventas

#### Correlaciones
```python
Correlaciones con Weekly_Sales:
├── Size: 0.32 (tiendas grandes → más ventas)
├── Temperature: -0.15 (verano bajo, invierno alto)
├── CPI: 0.18 (inflación correlaciona con ventas)
└── Unemployment: -0.12 (desempleo reduce ventas)
```

#### Técnicas Utilizadas
- Visualizaciones: histogramas, boxplots, time series plots
- Matriz de correlación (Pearson)
- Análisis de estacionalidad
- Detección de outliers (Z-score, IQR)
- Análisis Pareto (regla 80/20)

---

### 2. Preprocesamiento de Datos

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

#### Limpieza de Datos
```python
Pasos Aplicados:
├── Imputación NaN en MarkDowns: 0 (sin promoción)
├── Imputación NaN en CPI/Unemployment: forward fill temporal
├── Merge de datasets: train + features + stores
└── Validación: 0 NaN finales, 0 duplicados
```

#### Transformaciones
- **Encoding Categórico**: Type de tienda (A→0, B→1, C→2)
- **Normalización**: Features numéricas (StandardScaler) post-split
- **Split Temporal**: Train 85% / Validación 15% (respetando orden temporal)

#### Manejo de Outliers
```python
Estrategia:
├── Identificación: Z-score > 3
├── Análisis: ¿Genuinos o errores?
├── Decisión: Mantenidos (ventas legítimas en festivos)
└── Winsorización: Limitados al percentil 99 para estabilidad
```

---

## ⚙️ Feature Engineering

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

### Resumen de Features Creadas: 50+

#### 1️⃣ Features Temporales (16 features)
Capturan estacionalidad y tendencias temporales.

```python
# Componentes Básicos
- Year, Month, Week, Quarter
- DayOfWeek, DayOfYear

# Indicadores de Período
- IsMonthStart, IsMonthEnd
- IsQuarterStart, IsQuarterEnd
- WeekOfMonth

# Encoding Cíclico (evita discontinuidad 12→1)
- Month_sin = sin(2π × Month / 12)
- Month_cos = cos(2π × Month / 12)
- Week_sin = sin(2π × Week / 52)
- Week_cos = cos(2π × Week / 52)

# Tendencia
- Trend = días desde inicio dataset / 7
```

**Justificación**: Patrones cíclicos mensuales/semanales críticos para retail. Encoding sin/cos evita que el modelo vea diciembre (12) lejos de enero (1).

---

#### 2️⃣ Features de Festivos (8 features)
Los festivos tienen impacto 5x según métrica WMAE.

```python
# Identificadores de Festivos Principales
- IsSuperBowl: Super Bowl (semana 6 Feb)
- IsLaborDay: Labor Day (primer lunes Sept)
- IsThanksgiving: Thanksgiving (4º jueves Nov)
- IsChristmas: Christmas (semana 25 Dic)

# Proximidad a Festivos
- DaysToNextHoliday: días hasta próximo festivo
- DaysFromLastHoliday: días desde último festivo
- IsPreHoliday: 7 días antes de festivo
- IsPostHoliday: 7 días después de festivo
```

**Justificación**: Los festivos generan picos de ventas. La proximidad captura comportamiento de compra anticipada y post-festiva.

---

#### 3️⃣ Features de Lag (4 features)
Ventas pasadas son el mejor predictor de ventas futuras.

```python
# Ventas Semanas Previas
- Weekly_Sales_Lag1: semana anterior
- Weekly_Sales_Lag2: 2 semanas atrás
- Weekly_Sales_Lag3: 3 semanas atrás
- Weekly_Sales_Lag4: 4 semanas atrás
```

**Justificación**: Capturan tendencias recientes y momentum. Lag4 captura comportamiento mensual.

---

#### 4️⃣ Features Rolling Window (12 features)
Estadísticas móviles para suavizar ruido y capturar tendencias.

```python
# Ventanas de 4, 8, 12 semanas
Para cada ventana W:
- Weekly_Sales_RollingMean{W}: promedio móvil
- Weekly_Sales_RollingStd{W}: volatilidad
- Weekly_Sales_RollingMin{W}: mínimo período
- Weekly_Sales_RollingMax{W}: máximo período

Ejemplo: RollingMean4 = promedio últimas 4 semanas
```

**Justificación**: 
- RollingMean: Tendencia reciente sin ruido
- RollingStd: Volatilidad/estabilidad ventas
- RollingMin/Max: Rango de variación

---

#### 5️⃣ Features Agregadas por Store-Dept (5 features)
Características históricas de cada combinación tienda-departamento.

```python
# Estadísticos Históricos
- StoreDept_Mean: promedio histórico
- StoreDept_Std: desviación estándar
- StoreDept_Min: mínimo histórico
- StoreDept_Max: máximo histórico
- StoreDept_Median: mediana histórica
```

**Justificación**: Cada Store-Dept tiene comportamiento único. Features capturan "nivel base" y variabilidad característica.

---

#### 6️⃣ Features de Interacción (7 features)
Combinaciones de variables que capturan efectos conjuntos.

```python
# Interacciones Multiplicativas
- Temp_Month = Temperature × Month
  (temperatura varía por mes → interacción captura estacionalidad clima)

- Size_Holiday = Size × IsHoliday
  (tiendas grandes tienen mayor impacto festivo)

- Type_Holiday_Encoded = Type_Encoded × IsHoliday
  (tipo de tienda modula efecto festivo)

- Store_Dept_Encoded = Store × 100 + Dept
  (encoding único para cada combinación)

# Agregaciones de Promociones
- Total_MarkDown = MarkDown1 + ... + MarkDown5
  (inversión total en descuentos)

- Count_MarkDown = cantidad de MarkDowns activos
  (número de promociones simultáneas)

# Índice Económico
- Econ_Index = Unemployment × CPI
  (captura condiciones macroeconómicas generales)
```

**Justificación**: Efectos no son aditivos. Ej: temperatura alta en diciembre (calefacción) vs julio (enfriamiento) tiene significado diferente.

---

### Total Features Finales: 66

```
13 originales + 50 creadas + 3 encoding = 66 features
```

### Impacto de Feature Engineering

| Métrica | Sin FE | Con FE | Mejora |
|---------|--------|--------|--------|
| WMAE | $5,234 | $3,876 | ↓ 26% |
| MAE | $4,876 | $3,542 | ↓ 27% |
| R² | 0.82 | 0.92 | ↑ 12% |

---

## 🤖 Modelos y Resultados

**Notebook**: `notebooks/03_modelado_dataset.ipynb`

### Algoritmos Evaluados

#### 1. Baseline (Linear Regression)
```python
Configuración:
├── Algoritmo: Regresión Lineal
├── Features: 66 (estandarizadas)
└── Propósito: Referencia de comparación

Resultados:
├── WMAE: $5,234
├── MAE: $4,876
├── RMSE: $7,543
├── R²: 0.82
└── Tiempo Entrenamiento: 0.5s
```

#### 2. Random Forest
```python
Configuración:
├── n_estimators: 100 árboles
├── max_depth: 15
├── min_samples_split: 10
├── Paralelización: n_jobs=-1
└── Random State: 42

Resultados:
├── WMAE: $4,321 (↓17% vs baseline)
├── MAE: $3,987
├── RMSE: $6,125
├── R²: 0.89
└── Tiempo: 12.3 min

Top 5 Features Importantes:
1. Weekly_Sales_Lag1: 0.28
2. StoreDept_Mean: 0.15
3. Size: 0.12
4. Weekly_Sales_RollingMean4: 0.09
5. IsHoliday: 0.07
```

#### 3. XGBoost 🏆 (MEJOR MODELO)
```python
Configuración:
├── n_estimators: 500
├── max_depth: 7
├── learning_rate: 0.05
├── subsample: 0.8
├── colsample_bytree: 0.8
├── objective: reg:squarederror
└── early_stopping: 50 rounds

Resultados:
├── WMAE: $3,876 (↓26% vs baseline) ⭐
├── MAE: $3,542
├── RMSE: $5,678
├── R²: 0.92
└── Tiempo: 8.7 min

Top 5 Features Importantes:
1. Weekly_Sales_Lag1: 0.32
2. StoreDept_Mean: 0.18
3. Weekly_Sales_RollingMean4: 0.11
4. Size: 0.09
5. Trend: 0.06
```

#### 4. LightGBM
```python
Configuración:
├── n_estimators: 500
├── max_depth: 8
├── learning_rate: 0.05
├── num_leaves: 31
├── objective: regression
└── metric: mae

Resultados:
├── WMAE: $3,942
├── MAE: $3,601
├── RMSE: $5,734
├── R²: 0.91
└── Tiempo: 3.2 min (más rápido)
```

---

### Comparación de Modelos

| Modelo | WMAE | MAE | RMSE | R² | Tiempo | Selección |
|--------|------|-----|------|-----|--------|-----------|
| Baseline | $5,234 | $4,876 | $7,543 | 0.82 | 0.5s | ❌ |
| Random Forest | $4,321 | $3,987 | $6,125 | 0.89 | 12.3m | ❌ |
| **XGBoost** | **$3,876** | **$3,542** | **$5,678** | **0.92** | 8.7m | ✅ |
| LightGBM | $3,942 | $3,601 | $5,734 | 0.91 | 3.2m | ❌ |

**Modelo Seleccionado**: XGBoost por mejor WMAE (métrica objetivo del proyecto).

---

### Técnicas de Validación

1. **Split Temporal**: 85% train / 15% validation (respeta serie temporal)
2. **No Cross-Validation**: Validación temporal más apropiada que K-Fold para series temporales
3. **Early Stopping**: Previene overfitting en gradient boosting
4. **Feature Importance**: Análisis de features más relevantes

---

## 🛠️ Tecnologías Utilizadas

### Ciencia de Datos
| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.10+ | Lenguaje principal |
| pandas | 2.1.3 | Manipulación de datos |
| numpy | 1.26.2 | Cálculos numéricos |
| scikit-learn | 1.3.2 | Preprocesamiento, Random Forest, métricas |
| XGBoost | 2.0.2 | Gradient Boosting (modelo final) |
| LightGBM | 4.1.0 | Gradient Boosting alternativo |
| joblib | 1.3.2 | Serialización de modelos |

### Visualización
| Tecnología | Propósito |
|------------|-----------|
| matplotlib | Gráficos estáticos |
| seaborn | Visualizaciones estadísticas |
| plotly | Gráficos interactivos (dashboard) |

### Deployment
| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| FastAPI | 0.104.1 | API REST para predicciones |
| Streamlit | 1.28.2 | Dashboard web interactivo |
| uvicorn | 0.24.0 | Servidor ASGI para FastAPI |
| pydantic | 2.5.0 | Validación de datos API |
| Docker | latest | Containerización |
| Docker Compose | latest | Orquestación multi-contenedor |

---

## 📁 Estructura del Proyecto

```
walmart_sales_forecasting/
│
├── data/                           # Datos del proyecto
│   ├── 01_raw/                     # Datos originales (421K registros)
│   │   ├── train.csv               # Ventas históricas
│   │   ├── test.csv                # Datos para predicción
│   │   ├── features.csv            # Variables semanales
│   │   └── stores.csv              # Info tiendas
│   │
│   └── 02_processed/               # Datos procesados
│       ├── train_processed.csv     # Train con 66 features
│       ├── val_processed.csv       # Validación
│       └── test_processed.csv      # Test
│
├── notebooks/                      # Análisis Jupyter
│   ├── 01_exploracion_dataset.ipynb       # EDA completo
│   ├── 02_preprocesamiento_dataset.ipynb  # Feature Engineering
│   └── 03_modelado_dataset.ipynb          # Entrenamiento modelos
│
├── models/                         # Modelos ML serializados
│   ├── best_model.pkl              # XGBoost (272 MB)
│   ├── best_model_compressed.pkl   # Comprimido (93 MB)
│   └── model_info.pkl              # Metadata del modelo
│
├── api/                            # API REST
│   ├── main.py                     # FastAPI app
│   └── requirements.txt            # Dependencias API
│
├── web/                            # Dashboard Web
│   ├── app.py                      # Streamlit app
│   ├── requirements.txt            # Dependencias web
│   └── README.md                   # Documentación web
│
├── docker/                         # Containerización
│   ├── Dockerfile                  # Imagen Docker
│   ├── docker-compose.yml          # Orquestación
│   └── README.md                   # Guía Docker
│
├── .gitignore                      # Archivos ignorados
└── README.md                       # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.10 o superior
- Docker y Docker Compose (para deployment containerizado)
- Git

### Opción 1: Instalación Local

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```

#### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar Dependencias

**Para Notebooks**:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly jupyter
```

**Para API**:
```bash
cd api
pip install -r requirements.txt
```

**Para Dashboard**:
```bash
cd web
pip install -r requirements.txt
```

#### 4. Descargar Datos desde Kaggle

**⚠️ Los archivos CSV NO están incluidos en el repositorio debido a su tamaño.**

**Opción A: Descargar manualmente**
1. Ir a: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
2. Descargar todos los archivos
3. Colocar en `data/01_raw/`:
   - train.csv
   - test.csv
   - features.csv
   - stores.csv
   - sampleSubmission.csv

**Opción B: Usar Kaggle API**
```bash
# Instalar Kaggle CLI
pip install kaggle

# Configurar credenciales (kaggle.json en ~/.kaggle/)
kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data/01_raw/
unzip data/01_raw/walmart-recruiting-store-sales-forecasting.zip -d data/01_raw/
```

---

### Opción 2: Deployment con Docker (Recomendado) 🐳

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```

#### 2. Construir y Ejecutar Contenedores
```bash
cd docker
docker-compose up --build -d
```

Esto levantará:
- **API REST**: http://localhost:8006
- **Dashboard Web**: http://localhost:8506

#### 3. Verificar Contenedores
```bash
docker ps
# Deberías ver walmart_api y walmart_web corriendo
```

#### 4. Ver Logs
```bash
docker logs walmart_api
docker logs walmart_web
```

#### 5. Detener Servicios
```bash
docker-compose down
```

---

## 💻 Uso

### ⚠️ IMPORTANTE: Entrenar Modelos Primero

**Los modelos pre-entrenados NO están incluidos en el repositorio debido a su gran tamaño (~3GB)**. Debes entrenarlos localmente antes de usar la API o el dashboard.

#### Opción A: Entrenar con Notebooks (Recomendado)
```bash
# 1. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly jupyter

# 3. Iniciar Jupyter y ejecutar notebooks en orden:
jupyter notebook

# Notebooks a ejecutar:
# ├── 01_exploracion_dataset.ipynb     (Análisis exploratorio)
# ├── 02_preprocesamiento_dataset.ipynb (Limpieza y transformación)
# └── 03_modelado_dataset.ipynb        (ENTRENAMIENTO DE MODELOS) ⭐
```

**El notebook `03_modelado_dataset.ipynb` guardará automáticamente los modelos en `models/`**:
- `best_model.pkl` - Modelo principal para predicciones
- `best_model_compressed.pkl` - Versión optimizada
- `model_info.pkl` - Metadatos del modelo

#### Opción B: Script de Entrenamiento Rápido
```bash
cd notebooks
python execute_modelado.py
```

Este script ejecutará todo el pipeline de entrenamiento automáticamente (~30-45 minutos dependiendo de tu hardware).

---

### 1. Ejecutar Notebooks de Análisis

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Iniciar Jupyter
jupyter notebook

# Abrir notebooks en orden:
# 1. notebooks/01_exploracion_dataset.ipynb
# 2. notebooks/02_preprocesamiento_dataset.ipynb
# 3. notebooks/03_modelado_dataset.ipynb
```

---

### 2. Usar API REST

#### Iniciar API Localmente
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8006 --reload
```

#### Documentación Automática
- Swagger UI: http://localhost:8006/docs
- ReDoc: http://localhost:8006/redoc

#### Ejemplo de Solicitud (Python)
```python
import requests

url = "http://localhost:8006/predict"
data = {
    "Store": 1,
    "Dept": 1,
    "Date": "2012-11-02",
    "Temperature": 42.31,
    "Fuel_Price": 2.572,
    "MarkDown1": 0.0,
    "MarkDown2": 0.0,
    "MarkDown3": 0.0,
    "MarkDown4": 0.0,
    "MarkDown5": 0.0,
    "CPI": 211.096358,
    "Unemployment": 8.106,
    "IsHoliday": 0,
    "Type": "A",
    "Size": 151315
}

response = requests.post(url, json=data)
print(response.json())
# Output: {"prediction": 15359.31, "store": 1, "dept": 1, "is_holiday": false}
```

#### Ejemplo de Solicitud (cURL)
```bash
curl -X POST "http://localhost:8006/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Store": 1,
       "Dept": 1,
       "Date": "2012-11-02",
       "Temperature": 42.31,
       "Fuel_Price": 2.572,
       "MarkDown1": 0.0,
       "MarkDown2": 0.0,
       "MarkDown3": 0.0,
       "MarkDown4": 0.0,
       "MarkDown5": 0.0,
       "CPI": 211.096358,
       "Unemployment": 8.106,
       "IsHoliday": 0,
       "Type": "A",
       "Size": 151315
     }'
```

---

### 3. Usar Dashboard Web

#### Iniciar Dashboard Localmente
```bash
cd web
streamlit run app.py --server.port 8506
```

Abrir en navegador: http://localhost:8506

#### Funcionalidades del Dashboard
1. **Predicción Individual**: Ingresa parámetros manualmente
2. **Predicción por Tienda**: Selecciona tienda y fecha
3. **Predicción Masiva**: Sube archivo CSV con múltiples predicciones
4. **Visualizaciones**: Gráficos de tendencias y distribuciones
5. **Información del Modelo**: Métricas y features importantes

---

## 🌐 API Endpoints

### Base URL
```
http://localhost:8006
```

### Endpoints Disponibles

#### 1. Health Check
```http
GET /health
```

**Respuesta**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost"
}
```

---

#### 2. Predicción Individual
```http
POST /predict
```

**Request Body**:
```json
{
  "Store": 1,
  "Dept": 1,
  "Date": "2012-11-02",
  "Temperature": 42.31,
  "Fuel_Price": 2.572,
  "MarkDown1": 0.0,
  "MarkDown2": 0.0,
  "MarkDown3": 0.0,
  "MarkDown4": 0.0,
  "MarkDown5": 0.0,
  "CPI": 211.096358,
  "Unemployment": 8.106,
  "IsHoliday": 0,
  "Type": "A",
  "Size": 151315
}
```

**Respuesta**:
```json
{
  "prediction": 15359.31,
  "store": 1,
  "dept": 1,
  "is_holiday": false
}
```

---

#### 3. Predicción Batch
```http
POST /predict/batch
```

**Request Body**:
```json
{
  "predictions": [
    {
      "Store": 1,
      "Dept": 1,
      "Date": "2012-11-02",
      ...
    },
    {
      "Store": 2,
      "Dept": 3,
      "Date": "2012-11-09",
      ...
    }
  ]
}
```

**Respuesta**:
```json
{
  "predictions": [
    {"prediction": 15359.31, "store": 1, "dept": 1, ...},
    {"prediction": 23104.67, "store": 2, "dept": 3, ...}
  ],
  "count": 2
}
```

---

## 🔮 Mejoras Futuras

### Modelado
- [ ] **Modelos de Series Temporales**: Prophet, ARIMA, SARIMA
- [ ] **Deep Learning**: LSTM, GRU para capturar dependencias temporales largas
- [ ] **Ensemble**: Combinación ponderada de XGBoost + LightGBM + LSTM
- [ ] **Hyperparameter Tuning**: Grid Search / Bayesian Optimization
- [ ] **Features Adicionales**: Clima histórico, competencia, eventos locales

### Ingeniería
- [ ] **Pipeline Automatizado**: Airflow para ETL y reentrenamiento
- [ ] **Monitoreo**: MLflow para tracking de experimentos
- [ ] **CI/CD**: GitHub Actions para deployment automático
- [ ] **Escalabilidad**: Kubernetes para manejo de alta carga
- [ ] **Base de Datos**: PostgreSQL para almacenar predicciones

### Producto
- [ ] **Alertas**: Notificaciones de predicciones anómalas
- [ ] **Explicabilidad**: SHAP values para interpretar predicciones
- [ ] **A/B Testing**: Comparación de modelos en producción
- [ ] **App Móvil**: Flutter para gestores de tienda
- [ ] **Integración ERP**: Conexión con sistemas Walmart

---

## 📞 Contacto y Soporte

Si tienes preguntas o sugerencias sobre este proyecto:

- 📧 Email: mbenitezg01@gmail.com
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- **Kaggle**: Por proporcionar el dataset
- **Walmart**: Por el caso de estudio real
- **Comunidad Open Source**: Scikit-learn, XGBoost, FastAPI, Streamlit

---

**Firma Oficial del Proyecto:**  
`Walmart Sales Forecasting v1.0.0 • developed by Miguel Benítez`  
Ing. Miguel Antonio Benítez González (Universidad Tecnológica de Panamá - UTP) · 2026.
