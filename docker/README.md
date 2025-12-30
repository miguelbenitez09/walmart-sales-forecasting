# 🐳 Docker Configuration - Walmart Sales Forecasting

> **Configuración de Docker para desplegar el proyecto de predicción de ventas de Walmart.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [https://github.com/miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## 🚀 Uso Rápido

```bash
cd F_Docker
docker-compose up -d --build
```

**Servicios disponibles:**
- API: `http://localhost:8006`
- Web Interface: `http://localhost:8506`
- API Docs: `http://localhost:8006/docs`

## 📦 Servicios

### 1. API (FastAPI)
- Puerto Host: 8006
- Puerto Container: 8000
- Container: `walmart_api`
- Endpoints: `/predict`, `/predict/batch`, `/health`

### 2. Web Interface (Streamlit)
- Puerto Host: 8506
- Puerto Container: 8506
- Container: `walmart_web`
- Dashboard interactivo con 3 modos de predicción

## 🛠️ Comandos

```bash
# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Rebuild
docker-compose up -d --build
```

## 📋 Requisitos

- Docker Desktop instalado
- Modelo entrenado en `models/best_model_xgboost.pkl`

## 🔧 Configuración

Edita `docker-compose.yml` para:
- Cambiar puertos
- Ajustar volúmenes
- Agregar variables de entorno

---

## 🌐 Integración con Portafolio

Este proyecto usa puertos únicos para evitar conflictos con otros proyectos:
- Credit Card: 8002 (API), 8502 (Web)
- Online Shoppers: 8004 (API), 8503 (Web)
- **Walmart**: 8006 (API), 8506 (Web)

Todos los proyectos pueden ejecutarse simultáneamente sin conflictos de puertos.
