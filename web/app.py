"""
Dashboard Streamlit para predicciones de ventas de Walmart
Interfaz web que consume la API REST
Ejecutar: streamlit run app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any
import os

# Configuración de la página
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0071CE;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0071CE, #FFC220);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #0071CE;
    }
    .stButton>button {
        width: 100%;
        background-color: #0071CE;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #005a9c;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dirección de la API - Detectar si estamos en Docker o local
def get_api_url():
    # En Docker, existe la variable de entorno o el directorio /app
    if os.getenv('DOCKER_ENV') or (os.path.exists('/app') and os.name != 'nt'):
        return "http://walmart_api:8000"
    return "http://localhost:8000"

API_BASE_URL = get_api_url()

def check_api_health() -> Dict[str, Any]:
    """Verificar si la API está funcionando"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enviar datos a la API y obtener predicción"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def make_batch_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enviar múltiples predicciones a la API"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict/batch", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def create_gauge_chart(prediction: float, max_value: float = 50000):
    """Crear gráfico tipo gauge para la predicción"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Ventas Predichas ($)", 'font': {'size': 24}},
        delta={'reference': 15000, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0071CE"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10000], 'color': '#ffcccc'},
                {'range': [10000, 20000], 'color': '#ffffcc'},
                {'range': [20000, max_value], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🛒 Walmart Sales Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicción de ventas semanales usando Machine Learning</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### Esta interfaz permite:
        - ✅ Predicciones individuales
        - ✅ Predicciones por lote (CSV)
        - ✅ Verificar estado de la API
        - ✅ Visualización de resultados
        
        ### Modelo:
        - Random Forest / XGBoost
        - 60+ features engineered
        - WMAE optimizado
        """)
        
        st.markdown("---")
        
        # Verificar estado de la API
        if st.button("🔍 Verificar Estado API"):
            with st.spinner("Verificando..."):
                health = check_api_health()
                if health.get("status") == "healthy":
                    st.success("✅ API Operativa")
                    st.json(health)
                else:
                    st.error("❌ API No Disponible")
                    st.json(health)
        
        st.markdown("---")
        st.caption(f"🔗 API: {API_BASE_URL}")
        st.caption("📖 [Docs](http://localhost:8000/docs)")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📝 Predicción Individual", "📊 Predicción Batch", "📈 Información"])

    # ==================== TAB 1: PREDICCIÓN INDIVIDUAL ====================
    with tab1:
        st.header("📝 Predicción Individual")
        st.markdown("Ingresa los datos de una tienda y departamento para predecir las ventas")
        
        # Formulario en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏬 Información de Tienda")
            store = st.number_input("Store (1-45)", min_value=1, max_value=45, value=1, help="ID de la tienda")
            dept = st.number_input("Dept (1-99)", min_value=1, max_value=99, value=1, help="ID del departamento")
            store_type = st.selectbox("Type", options=["A", "B", "C"], help="Tipo de tienda (A=Grande, B=Media, C=Pequeña)")
            size = st.number_input("Size (sq ft)", min_value=1000, max_value=250000, value=151315, help="Tamaño en pies cuadrados")
        
        with col2:
            st.subheader("📅 Información Temporal")
            fecha = st.date_input("Date", value=datetime(2012, 11, 23), min_value=datetime(2010, 1, 1), max_value=datetime(2025, 12, 31))
            is_holiday = st.checkbox("IsHoliday", value=False, help="¿Es semana festiva?")
            
            if is_holiday:
                st.info("🎉 Semana festiva: predicción con peso 5x en WMAE")
            
            st.markdown("---")
            st.subheader("🌡️ Variables Externas")
            temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=42.31)
            fuel_price = st.number_input("Fuel Price ($/gal)", min_value=0.0, max_value=10.0, value=2.572, step=0.01)
        
        with col3:
            st.subheader("💰 Variables Económicas")
            cpi = st.number_input("CPI", min_value=100.0, max_value=300.0, value=211.096, help="Índice de Precios al Consumidor")
            unemployment = st.number_input("Unemployment (%)", min_value=0.0, max_value=20.0, value=8.106, step=0.1)
            
            st.markdown("---")
            st.subheader("📉 MarkDowns (Descuentos)")
            markdown1 = st.number_input("MarkDown1", min_value=0.0, value=0.0)
            markdown2 = st.number_input("MarkDown2", min_value=0.0, value=0.0)
            markdown3 = st.number_input("MarkDown3", min_value=0.0, value=0.0)
            markdown4 = st.number_input("MarkDown4", min_value=0.0, value=0.0)
            markdown5 = st.number_input("MarkDown5", min_value=0.0, value=0.0)
        
        # Botón de predicción
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🚀 PREDECIR VENTAS", use_container_width=True)
        
        if predict_button:
            # Preparar datos
            data = {
                "Store": store,
                "Dept": dept,
                "Date": fecha.strftime("%Y-%m-%d"),
                "IsHoliday": is_holiday,
                "Temperature": temperature,
                "Fuel_Price": fuel_price,
                "MarkDown1": markdown1,
                "MarkDown2": markdown2,
                "MarkDown3": markdown3,
                "MarkDown4": markdown4,
                "MarkDown5": markdown5,
                "CPI": cpi,
                "Unemployment": unemployment,
                "Type": store_type,
                "Size": size
            }
            
            with st.spinner("🔄 Realizando predicción..."):
                result = make_prediction(data)
            
            if "error" in result:
                st.markdown(f'<div class="error-box">❌ <b>Error:</b> {result["error"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Predicción realizada con éxito!</b></div>', unsafe_allow_html=True)
                
                # Mostrar resultados
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    # Gauge chart
                    fig = create_gauge_chart(result['prediction'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_res2:
                    # Métricas
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💵 Ventas Predichas", f"${result['prediction']:,.2f}")
                    st.metric("🏬 Store", f"#{result['store']}")
                    st.metric("🏪 Departamento", f"#{result['dept']}")
                    holiday_text = "🎉 Sí" if result['is_holiday'] else "📅 No"
                    st.metric("Festivo", holiday_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Información adicional
                st.info(f"📊 **Contexto:** Store {store} - Dept {dept} | Tipo {store_type} | {size:,} sq ft")

    # ==================== TAB 2: PREDICCIÓN BATCH ====================
    with tab2:
        st.header("📊 Predicción por Lote (Batch)")
        st.markdown("Sube un archivo CSV con múltiples registros para obtener predicciones masivas")
        
        # Instrucciones
        with st.expander("📋 Ver formato de CSV requerido"):
            st.markdown("""
            El archivo CSV debe contener las siguientes columnas:
            
            | Columna | Tipo | Descripción |
            |---------|------|-------------|
            | Store | int | ID tienda (1-45) |
            | Dept | int | ID departamento (1-99) |
            | Date | string | Fecha (YYYY-MM-DD) |
            | IsHoliday | bool | Festivo (true/false) |
            | Temperature | float | Temperatura (°F) |
            | Fuel_Price | float | Precio combustible |
            | Type | string | Tipo tienda (A/B/C) |
            | Size | int | Tamaño tienda |
            | MarkDown1-5 | float | Descuentos (opcional) |
            | CPI | float | Índice precios |
            | Unemployment | float | Tasa desempleo |
            """)
        
        # Upload CSV
        uploaded_file = st.file_uploader("📤 Subir archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df)} registros")
                
                # Mostrar preview
                st.subheader("👁️ Preview de datos")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Botón para predecir
                if st.button("🚀 PREDECIR BATCH", use_container_width=True):
                    # Convertir DataFrame a lista de diccionarios
                    records = df.to_dict('records')
                    
                    # Asegurarnos de que IsHoliday sea booleano
                    for record in records:
                        if 'IsHoliday' in record:
                            record['IsHoliday'] = bool(record['IsHoliday'])
                        # Agregar valores por defecto si faltan
                        for md in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
                            if md not in record:
                                record[md] = 0.0
                    
                    batch_data = {"predictions": records}
                    
                    with st.spinner(f"🔄 Procesando {len(records)} predicciones..."):
                        result = make_batch_prediction(batch_data)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success(f"✅ {result['count']} predicciones completadas!")
                        
                        # Crear DataFrame con resultados
                        results_df = pd.DataFrame(result['predictions'])
                        
                        # Agregar columna de predicción al DataFrame original
                        df['Predicted_Sales'] = results_df['prediction']
                        
                        # Mostrar estadísticas
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("📊 Total Predicciones", result['count'])
                        with col_stat2:
                            st.metric("💰 Promedio", f"${results_df['prediction'].mean():,.2f}")
                        with col_stat3:
                            st.metric("📈 Máximo", f"${results_df['prediction'].max():,.2f}")
                        with col_stat4:
                            st.metric("📉 Mínimo", f"${results_df['prediction'].min():,.2f}")
                        
                        # Gráfico de distribución
                        st.subheader("📊 Distribución de Predicciones")
                        fig = px.histogram(results_df, x='prediction', nbins=30, 
                                         title="Distribución de Ventas Predichas",
                                         labels={'prediction': 'Ventas Predichas ($)', 'count': 'Frecuencia'})
                        fig.update_traces(marker_color='#0071CE')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de resultados
                        st.subheader("📋 Resultados Detallados")
                        st.dataframe(df, use_container_width=True)
                        
                        # Descargar resultados
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name=f"walmart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error al procesar archivo: {str(e)}")

    # ==================== TAB 3: INFORMACIÓN ====================
    with tab3:
        st.header("📈 Información del Proyecto")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("🎯 Acerca del Modelo")
            st.markdown("""
            ### Walmart Sales Forecasting
            
            **Objetivo:** Predecir ventas semanales de departamentos en 45 tiendas Walmart
            
            **Características:**
            - 🤖 Modelos: Random Forest, XGBoost, LightGBM
            - 📊 Dataset: 421,570 registros (2010-2012)
            - 🎯 Métrica: WMAE (Weighted MAE)
            - ⚙️ Features: 60+ engineered features
            
            **Features Principales:**
            - ⏰ Temporales (16): Year, Month, sin/cos encoding
            - 🎄 Festivos (8): Super Bowl, Thanksgiving, Christmas
            - 📊 Lag (4): Ventas 1-4 semanas previas
            - 📈 Rolling (8): Medias y desviaciones móviles
            - 🏬 Agregadas (5): Estadísticas Store-Dept
            - 🔀 Interacción (7): Combinaciones de variables
            """)
        
        with col_info2:
            st.subheader("📊 Métricas de Rendimiento")
            st.markdown("""
            ### Resultados del Modelo
            
            | Modelo | WMAE | R² |
            |--------|------|-----|
            | Baseline | $5,234 | 0.82 |
            | Random Forest | $4,321 | 0.89 |
            | **XGBoost** | **$3,876** | **0.92** |
            | LightGBM | $3,942 | 0.91 |
            
            ### Insights Clave
            - 🎉 Festivos aumentan ventas en **40%**
            - 🏬 Tiendas tipo A venden **3.5x más** que tipo C
            - 📦 Top 20% departamentos = **60% ventas**
            - 📊 Mejora de **26%** vs baseline
            """)
            
            st.markdown("---")
            st.subheader("🔗 Enlaces Útiles")
            st.markdown("""
            - 📖 [Documentación API](http://localhost:8000/docs)
            - 🐙 [GitHub Repository](#)
            - 📊 [Kaggle Competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
            """)

if __name__ == "__main__":
    main()
