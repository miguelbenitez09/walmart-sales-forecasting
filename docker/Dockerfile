# Dockerfile para Walmart Sales Forecasting API
FROM python:3.10-slim

WORKDIR /app

# Instalar curl para el healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY ../api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de API
COPY ../api/ ./api/

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["python", "api/main.py"]
