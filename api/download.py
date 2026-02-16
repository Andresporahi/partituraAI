"""
Vercel Serverless Function: /api/download/<filename>
En modo serverless, los archivos se envían como base64 desde /api/upload.
Esta función existe como fallback para el modo local.
"""

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        body = json.dumps({
            "error": "En modo serverless, los archivos se descargan directamente desde el navegador."
        }).encode("utf-8")
        self.wfile.write(body)
