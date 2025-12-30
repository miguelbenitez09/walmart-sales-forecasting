"""
FastAPI para predicciones de ventas de Walmart
Endpoints:
- POST /predict - Predice ventas para un conjunto de datos
- GET /health - Health check
- GET / - Documentación
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import pickle
import joblib
import numpy as np
from datetime import datetime
import os
import os

# Inicializar FastAPI
app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="API para predecir ventas semanales de Walmart usando Machine Learning",
    version="1.0.0"
)

# Cargar modelo - Intentar versión comprimida primero para mejor performance
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = '/app/models'

model = None
model_name = None
model_info = None

def load_model():
    """
    Carga el mejor modelo entrenado.
    Prioridad: best_model_compressed.pkl (optimizado) > best_model.pkl
    """
    global model, model_name, model_info
    
    # Intentar cargar versión comprimida primero (más rápida y eficiente)
    compressed_path = os.path.join(MODELS_DIR, 'best_model_compressed.pkl')
    original_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    model_info_path = os.path.join(MODELS_DIR, 'model_info.pkl')
    
    model_path = None
    use_joblib = False
    
    if os.path.exists(compressed_path):
        model_path = compressed_path
        use_joblib = True
        print(f"📂 Usando modelo comprimido (optimizado para memoria)")
    elif os.path.exists(original_path):
        model_path = original_path
        use_joblib = False
        print(f"📂 Usando modelo original")
    else:
        raise FileNotFoundError(
            f"❌ Modelo no encontrado en {MODELS_DIR}\n"
            "Archivos esperados: best_model_compressed.pkl o best_model.pkl\n"
            "Por favor:\n"
            "1. Ejecuta notebooks/03_modelado_dataset.ipynb para entrenar el modelo\n"
            "2. Opcionalmente ejecuta src/optimize_model.py para crear versión comprimida"
        )
    
    print(f"📂 Cargando modelo desde: {model_path}")
    
    # Cargar con joblib (comprimido) o pickle (original)
    try:
        if use_joblib:
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"✅ Modelo cargado exitosamente")
    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar modelo: {str(e)}")
    
    # Cargar información del modelo
    try:
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        model_name = model_info.get('model_name', 'Random Forest')
        print(f"   Tipo: {model_name}")
        print(f"   MAE: ${model_info.get('mae', 0):,.2f}")
        print(f"   R²: {model_info.get('r2', 0):.4f}")
    except Exception as e:
        print(f"⚠️ No se pudo cargar model_info.pkl: {e}")
        model_name = "Random Forest"
        model_info = {}

try:
    load_model()
except Exception as e:
    print(f"⚠️ Error cargando modelo: {e}")

# Modelos de datos
class PredictionInput(BaseModel):
    Store: int = Field(..., ge=1, le=45)
    Dept: int = Field(..., ge=1, le=99)
    Date: str
    IsHoliday: bool
    Temperature: Optional[float] = None
    Fuel_Price: Optional[float] = None
    CPI: Optional[float] = None
    Unemployment: Optional[float] = None
    MarkDown1: Optional[float] = 0
    MarkDown2: Optional[float] = 0
    MarkDown3: Optional[float] = 0
    MarkDown4: Optional[float] = 0
    MarkDown5: Optional[float] = 0
    Type: Optional[str] = "A"
    Size: Optional[int] = 150000

class PredictionOutput(BaseModel):
    prediction: float
    store: int
    dept: int
    is_holiday: bool

class BatchPredictionInput(BaseModel):
    predictions: List[PredictionInput]

@app.get("/")
async def root():
    return {
        "message": "Walmart Sales Forecasting API",
        "version": "1.0.0",
        "model": model_name if model else "No cargado",
        "endpoints": {
            "POST /predict": "Predice ventas",
            "POST /predict/batch": "Predicciones múltiples",
            "GET /health": "Health check",
            "GET /docs": "Documentación"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name if model else None
    }

def prepare_features(data_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Features temporales (16)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    df['WeekOfMonth'] = (df['Date'].dt.day - 1) // 7 + 1
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Trend'] = (df['Date'] - pd.Timestamp('2010-01-01')).dt.days // 7
    
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    
    # Features de festivos (8)
    df['IsSuperBowl'] = 0
    df['IsLaborDay'] = 0
    df['IsThanksgiving'] = 0
    df['IsChristmas'] = 0
    df['DaysToNextHoliday'] = 30
    df['DaysFromLastHoliday'] = 30
    df['IsPreHoliday'] = 0
    df['IsPostHoliday'] = 0
    
    # Rellenar NaNs
    numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Lag features (4)
    df['Weekly_Sales_Lag1'] = 15000
    df['Weekly_Sales_Lag2'] = 14500
    df['Weekly_Sales_Lag3'] = 14800
    df['Weekly_Sales_Lag4'] = 15200
    
    # Rolling features para ventanas 4, 8, y 12
    for window in [4, 8, 12]:
        df[f'Weekly_Sales_RollingMean{window}'] = 15000
        df[f'Weekly_Sales_RollingStd{window}'] = 2000
        df[f'Weekly_Sales_RollingMin{window}'] = 10000
        df[f'Weekly_Sales_RollingMax{window}'] = 20000
    
    # Features agregadas (5)
    df['StoreDept_Mean'] = 15000
    df['StoreDept_Std'] = 5000
    df['StoreDept_Min'] = 5000
    df['StoreDept_Max'] = 30000
    df['StoreDept_Median'] = 14000
    
    # Features de interacción (7)
    df['Temp_Month'] = df['Temperature'] * df['Month']
    df['Size_Holiday'] = df['Size'] * df['IsHoliday']
    df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)
    df['Count_MarkDown'] = (df[markdown_cols] > 0).sum(axis=1)
    df['Econ_Index'] = df['Unemployment'] * df['CPI']
    
    # Type encoding
    type_map = {'A': 0, 'B': 1, 'C': 2}
    df['Type_Encoded'] = type_map.get(data_dict.get('Type', 'A'), 0)
    df['Type_Holiday_Encoded'] = df['Type_Encoded'] * df['IsHoliday']
    
    # Store_Dept_Encoded
    df['Store_Dept_Encoded'] = df['Store'] * 100 + df['Dept']
    
    # Eliminar Date y Type
    cols_to_drop = ['Date', 'Type']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Reordenar columnas en el orden exacto que espera el modelo
    expected_features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 
                         'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 
                         'CPI', 'Unemployment', 'Size', 'Year', 'Month', 'Week', 'Quarter', 
                         'DayOfWeek', 'DayOfYear', 'IsMonthStart', 'IsMonthEnd', 'IsQuarterStart', 
                         'IsQuarterEnd', 'WeekOfMonth', 'Month_sin', 'Month_cos', 'Week_sin', 
                         'Week_cos', 'Trend', 'IsSuperBowl', 'IsLaborDay', 'IsThanksgiving', 
                         'IsChristmas', 'DaysToNextHoliday', 'DaysFromLastHoliday', 'IsPreHoliday', 
                         'IsPostHoliday', 'Weekly_Sales_Lag1', 'Weekly_Sales_Lag2', 
                         'Weekly_Sales_Lag3', 'Weekly_Sales_Lag4', 'Weekly_Sales_RollingMean4', 
                         'Weekly_Sales_RollingStd4', 'Weekly_Sales_RollingMin4', 
                         'Weekly_Sales_RollingMax4', 'Weekly_Sales_RollingMean8', 
                         'Weekly_Sales_RollingStd8', 'Weekly_Sales_RollingMin8', 
                         'Weekly_Sales_RollingMax8', 'Weekly_Sales_RollingMean12', 
                         'Weekly_Sales_RollingStd12', 'Weekly_Sales_RollingMin12', 
                         'Weekly_Sales_RollingMax12', 'StoreDept_Mean', 'StoreDept_Std', 
                         'StoreDept_Min', 'StoreDept_Max', 'StoreDept_Median', 'Temp_Month', 
                         'Size_Holiday', 'Total_MarkDown', 'Count_MarkDown', 'Econ_Index', 
                         'Type_Encoded', 'Type_Holiday_Encoded', 'Store_Dept_Encoded']
    
    df = df[expected_features]
    
    return df

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        data_dict = input_data.model_dump()
        X = prepare_features(data_dict)
        prediction = float(model.predict(X)[0])
        
        return PredictionOutput(
            prediction=round(prediction, 2),
            store=input_data.Store,
            dept=input_data.Dept,
            is_holiday=input_data.IsHoliday
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(input_data: BatchPredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        predictions = []
        for item in input_data.predictions:
            data_dict = item.model_dump()
            X = prepare_features(data_dict)
            prediction = float(model.predict(X)[0])
            
            predictions.append({
                "prediction": round(prediction, 2),
                "store": item.Store,
                "dept": item.Dept,
                "is_holiday": item.IsHoliday
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando API...")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
