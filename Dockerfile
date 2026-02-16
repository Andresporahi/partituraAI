FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copiar el resto del proyecto
COPY . .

# Crear carpetas temporales
RUN mkdir -p uploads output

# Puerto que Railway asigna dinámicamente
ENV PORT=5000
EXPOSE $PORT

# Usar gunicorn para producción
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
