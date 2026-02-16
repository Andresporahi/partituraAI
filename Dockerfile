FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear carpetas temporales
RUN mkdir -p uploads output

EXPOSE 5000

CMD ["python", "app.py"]
