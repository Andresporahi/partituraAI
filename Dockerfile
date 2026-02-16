FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para librosa/soundfile/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Deshabilitar GPU de TensorFlow (Railway no tiene GPU) y suprimir warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_ENABLE_ONEDNN_OPTS=0

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
