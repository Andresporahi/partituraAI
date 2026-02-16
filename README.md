# Partituras AI

Aplicación web que convierte audio vocal en partituras musicales. Sube un archivo de audio con tu voz cantando y obtén:

- **Archivo MIDI** con las notas detectadas
- **MusicXML** para abrir en cualquier editor de partituras (MuseScore, Finale, Sibelius)
- **Partitura visual** renderizada directamente en el navegador

## Cómo funciona

1. **Subir audio**: Arrastra o selecciona un archivo de audio (WAV, MP3, OGG, FLAC, M4A, WebM)
2. **Detección de pitch**: Se usa `librosa` con el algoritmo **pYIN** para detectar las frecuencias fundamentales de la voz
3. **Procesamiento inteligente**: Suavizado de pitch, corrección de octavas, detección de tempo y tonalidad
4. **Generación MIDI**: Las frecuencias se convierten a notas MIDI con cuantización rítmica inteligente
5. **Partitura**: Se genera un archivo MusicXML con `music21` y se renderiza en el navegador con **OpenSheetMusicDisplay**

## Requisitos (desarrollo local)

- Python 3.9 o superior
- (Opcional) [MuseScore](https://musescore.org/) o [LilyPond](https://lilypond.org/) para exportar a PDF/PNG

## Instalación local

```bash
cd "Partituras AI"

python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

Abre tu navegador en **http://localhost:5000**

## Deploy en Vercel

El proyecto está configurado para deploy automático en Vercel:

```bash
# Instalar Vercel CLI (si no lo tienes)
npm i -g vercel

# Deploy
vercel
```

O simplemente conecta el repositorio de GitHub a Vercel para deploy automático.

## Estructura del proyecto

```
Partituras AI/
├── app.py                 # Servidor Flask (desarrollo local)
├── requirements.txt       # Dependencias Python
├── vercel.json            # Configuración de Vercel
├── api/                   # Serverless functions (Vercel)
│   ├── __init__.py
│   ├── processor.py       # Lógica de procesamiento compartida
│   ├── upload.py          # POST /api/upload
│   └── download.py        # GET /api/download (fallback)
├── public/                # Archivos estáticos (Vercel)
│   ├── index.html
│   └── static/
│       ├── style.css
│       └── app.js
├── templates/             # Templates Flask (dev local)
│   └── index.html
├── static/                # Static Flask (dev local)
│   ├── style.css
│   └── app.js
├── uploads/               # Audio subidos (temporal, local)
└── output/                # Archivos generados (temporal, local)
```

## Tecnologías

| Componente | Tecnología |
|---|---|
| Backend | Python, Flask / Vercel Serverless |
| Detección de pitch | librosa (pYIN) |
| Generación MIDI | pretty_midi |
| Generación de partitura | music21 |
| Renderizado de partitura | OpenSheetMusicDisplay (OSMD) |
| Frontend | HTML5, CSS3, JavaScript vanilla |
| Deploy | Vercel |

## Consejos para mejores resultados

- Usa grabaciones **claras y sin ruido de fondo**
- Canta **una sola línea melódica** (monofónica), no acordes
- Formatos sin compresión (WAV, FLAC) dan mejores resultados que MP3
- Grabaciones de **5-60 segundos** funcionan mejor
