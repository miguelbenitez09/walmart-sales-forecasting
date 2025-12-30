# Walmart Sales Forecasting - Quick Start Guide

## 🚀 Inicio Rápido con Docker (5 minutos)

### Prerequisitos
- Docker y Docker Compose instalados
- Git

### Pasos

1. **Clonar repositorio**
```bash
git clone https://github.com/miguelbenitez09/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```

2. **Levantar servicios**
```bash
cd docker
docker-compose up -d
```

3. **Acceder a servicios**
- API: http://localhost:8006/docs
- Dashboard: http://localhost:8506

4. **Probar API**
```bash
curl -X POST "http://localhost:8006/predict" \
  -H "Content-Type: application/json" \
  -d '{"Store":1,"Dept":1,"Date":"2012-11-02","Temperature":42.31,"Fuel_Price":2.572,"MarkDown1":0,"MarkDown2":0,"MarkDown3":0,"MarkDown4":0,"MarkDown5":0,"CPI":211.096358,"Unemployment":8.106,"IsHoliday":0,"Type":"A","Size":151315}'
```

¡Listo! 🎉

---

## 📊 Ejecutar Análisis Completo

### 1. Configurar entorno
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

### 2. Ejecutar notebooks
```bash
jupyter notebook
# Abrir en orden: 01, 02, 03
```

---

## 🔧 Troubleshooting

### Problema: Puerto ocupado
```bash
# Cambiar puertos en docker-compose.yml
ports:
  - "8007:8000"  # API en puerto 8007
  - "8507:8506"  # Web en puerto 8507
```

### Problema: Modelo no carga
```bash
# Verificar que models/ contenga best_model_compressed.pkl
ls models/
```

### Problema: Falta archivo de datos
Los datos están en `data/01_raw/`. Si faltan, descargar de Kaggle.

---

## 📚 Más Información

Ver [README.md](README.md) completo para:
- Análisis detallado del proyecto
- Explicación de técnicas aplicadas
- Documentación completa de API
- Feature engineering detallado
