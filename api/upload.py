"""
Vercel Serverless Function: /api/upload
Recibe un archivo de audio, lo convierte a MIDI y genera partitura.
"""

import os
import uuid
import json
import base64
import traceback
import tempfile

from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs


ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}
TMP_DIR = "/tmp"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_multipart(body: bytes, content_type: str):
    """Parsea manualmente un multipart/form-data request."""
    boundary = content_type.split("boundary=")[-1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = boundary.encode()
    parts = body.split(b"--" + boundary_bytes)

    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        if b'name="audio"' not in part:
            continue

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        headers_raw = part[:header_end].decode("utf-8", errors="replace")
        file_data = part[header_end + 4:]

        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]

        filename = "upload.wav"
        if 'filename="' in headers_raw:
            fn_start = headers_raw.index('filename="') + 10
            fn_end = headers_raw.index('"', fn_start)
            filename = headers_raw[fn_start:fn_end]

        return filename, file_data

    return None, None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")

            if "multipart/form-data" not in content_type:
                self._json_response(400, {"error": "Se esperaba multipart/form-data."})
                return

            body = self.rfile.read(content_length)
            filename, file_data = parse_multipart(body, content_type)

            if not filename or not file_data:
                self._json_response(400, {"error": "No se envió ningún archivo de audio."})
                return

            if not allowed_file(filename):
                self._json_response(400, {
                    "error": f"Formato no soportado. Usa: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                return

            # Importar aquí para no cargar las libs en cold start si hay error temprano
            from api.processor import audio_to_midi, midi_to_score

            job_id = uuid.uuid4().hex[:12]
            ext = filename.rsplit(".", 1)[1].lower()

            audio_path = os.path.join(TMP_DIR, f"{job_id}.{ext}")
            midi_path = os.path.join(TMP_DIR, f"{job_id}.mid")
            output_base = os.path.join(TMP_DIR, job_id)

            with open(audio_path, "wb") as f:
                f.write(file_data)

            # Paso 1: Audio → MIDI
            midi_stats = audio_to_midi(audio_path, midi_path)

            # Paso 2: MIDI → Partitura
            score_files = midi_to_score(
                midi_path, output_base,
                detected_bpm=midi_stats["detected_bpm"],
                notes_list=midi_stats["notes_list"],
            )

            # Leer archivos generados y codificar en base64 para enviar al cliente
            files_data = {}

            with open(midi_path, "rb") as f:
                files_data["midi"] = {
                    "data": base64.b64encode(f.read()).decode("utf-8"),
                    "filename": f"{job_id}.mid",
                    "mime": "audio/midi",
                }

            if "musicxml" in score_files and os.path.exists(score_files["musicxml"]):
                with open(score_files["musicxml"], "r", encoding="utf-8") as f:
                    files_data["musicxml"] = {
                        "data": base64.b64encode(f.read().encode("utf-8")).decode("utf-8"),
                        "filename": f"{job_id}.musicxml",
                        "mime": "application/vnd.recordare.musicxml+xml",
                    }

            if "png" in score_files and os.path.exists(score_files["png"]):
                with open(score_files["png"], "rb") as f:
                    files_data["png"] = {
                        "data": base64.b64encode(f.read()).decode("utf-8"),
                        "filename": f"{job_id}.png",
                        "mime": "image/png",
                    }

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
                "files_data": files_data,
            }

            # Limpiar archivos temporales
            for path in [audio_path, midi_path]:
                try:
                    os.remove(path)
                except OSError:
                    pass
            for key in ["musicxml", "png"]:
                if key in score_files:
                    try:
                        os.remove(score_files[key])
                    except OSError:
                        pass

            self._json_response(200, response)

        except Exception as e:
            traceback.print_exc()
            self._json_response(500, {"error": f"Error procesando el audio: {str(e)}"})

    def _json_response(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
