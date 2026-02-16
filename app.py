"""
Partituras AI – Servidor Flask.
Funciona tanto en desarrollo local como en producción (Railway/Docker).
"""

import os
import uuid
import traceback

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from processor import audio_to_midi, midi_to_score

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PUBLIC_FOLDER = os.path.join(BASE_DIR, "public")

app = Flask(
    __name__,
    static_folder=os.path.join(PUBLIC_FOLDER, "static"),
    template_folder=PUBLIC_FOLDER,
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
CORS(app)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Rutas de la API
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(PUBLIC_FOLDER, "index.html")


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    """Recibe un archivo de audio, lo convierte a MIDI y genera partitura."""
    if "audio" not in request.files:
        return jsonify({"error": "No se envió ningún archivo de audio."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "El nombre del archivo está vacío."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Formato no soportado. Usa: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    job_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    audio_filename = f"{job_id}.{ext}"
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
    file.save(audio_path)

    try:
        # Leer BPM del usuario (opcional)
        user_bpm = request.form.get("bpm", None)
        if user_bpm:
            try:
                user_bpm = float(user_bpm)
            except (ValueError, TypeError):
                user_bpm = None

        # Paso 1: Audio → MIDI
        midi_filename = f"{job_id}.mid"
        midi_path = os.path.join(OUTPUT_FOLDER, midi_filename)
        midi_stats = audio_to_midi(audio_path, midi_path, user_bpm=user_bpm)

        # Paso 2: MIDI → Partitura
        output_base = os.path.join(OUTPUT_FOLDER, job_id)
        score_files = midi_to_score(
            midi_path, output_base,
            detected_bpm=midi_stats["detected_bpm"],
            notes_list=midi_stats["notes_list"],
        )

        response = {
            "job_id": job_id,
            "duration_seconds": midi_stats["duration_seconds"],
            "notes_detected": midi_stats["notes_detected"],
            "analysis": {
                "bpm": midi_stats["detected_bpm"],
                "key": score_files["key_info"]["key"],
                "key_name": score_files["key_info"]["key_name"],
                "key_confidence": score_files["key_info"]["confidence"],
                "time_signature": "4/4",
            },
            "files": {
                "midi": f"/api/download/{midi_filename}",
            },
        }

        if "musicxml" in score_files:
            response["files"]["musicxml"] = f"/api/download/{job_id}.musicxml"

        if "png" in score_files:
            response["files"]["png"] = f"/api/download/{job_id}.png"

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error procesando el audio: {str(e)}"}), 500


@app.route("/api/download/<filename>")
def download_file(filename):
    """Descarga un archivo generado."""
    safe_name = secure_filename(filename)
    return send_from_directory(OUTPUT_FOLDER, safe_name, as_attachment=True)


@app.route("/api/preview/<filename>")
def preview_file(filename):
    """Sirve un archivo para previsualización (sin forzar descarga)."""
    safe_name = secure_filename(filename)
    return send_from_directory(OUTPUT_FOLDER, safe_name)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_production = os.environ.get("RAILWAY_ENVIRONMENT") is not None

    print("=" * 60)
    print(f"  Partituras AI - http://localhost:{port}")
    print("=" * 60)

    if is_production:
        from waitress import serve
        print(f"  Modo producción (waitress) en puerto {port}")
        serve(app, host="0.0.0.0", port=port)
    else:
        app.run(debug=True, host="0.0.0.0", port=port)
