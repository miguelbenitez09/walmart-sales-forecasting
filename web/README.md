# 🌐 Walmart Sales Forecasting - Web Interface

> **Dashboard interactivo con Streamlit para predicciones de ventas de Walmart.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [https://github.com/miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## 🚀 Uso

### Ejecución Local

```bash
streamlit run app.py --server.port 8501
```

Abre en `http://localhost:8501`

### Con Docker

Desde el directorio raíz del proyecto:

```bash
cd F_Docker
docker-compose up --build
```

Abre en `http://localhost:8506`

## ✨ Modos

1. **Predicción Individual**: Formulario manual
2. **Predicción por Lotes**: Sube CSV
3. **Análisis Histórico**: Visualiza datos pasados

## 📋 Requisitos

- Modelo en `models/`
- Python 3.10+
- `pip install -r requirements.txt`
