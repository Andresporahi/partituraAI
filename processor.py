"""
Módulo de procesamiento de audio: detección de pitch, generación MIDI y partitura.
Pipeline fiel a la voz: detecta solo las notas que realmente se cantan.
Usa CREPE (red neuronal vía ONNX Runtime) para detección de pitch.
"""

import os
import numpy as np
import librosa
import onnxruntime as ort
import pretty_midi
from scipy.ndimage import median_filter
from music21 import (
    stream, note, meter, tempo, metadata,
    key as m21key,
)
from music21.note import Rest

# ---------------------------------------------------------------------------
# MODELO CREPE ONNX (carga lazy, una sola vez)
# ---------------------------------------------------------------------------
_CREPE_SESSION = None
_CREPE_CENTS = np.arange(360) * 20 + 1997.3794084376191

def _get_crepe_session():
    """Carga el modelo CREPE ONNX una sola vez (singleton)."""
    global _CREPE_SESSION
    if _CREPE_SESSION is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "models", "crepe_small.onnx"
        )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        _CREPE_SESSION = ort.InferenceSession(model_path, opts)
    return _CREPE_SESSION


# ---------------------------------------------------------------------------
# 1. PREPROCESAMIENTO DE AUDIO
# ---------------------------------------------------------------------------

def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Normaliza, aplica pre-emphasis y recorta silencios."""
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y = librosa.effects.preemphasis(y, coef=0.97)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


# ---------------------------------------------------------------------------
# 2. DETECCIÓN DE PITCH CON CREPE (ONNX RUNTIME)
# ---------------------------------------------------------------------------

def _viterbi_decode(probs: np.ndarray) -> np.ndarray:
    """Decodificación Viterbi para suavizar la secuencia de pitch."""
    n_frames, n_bins = probs.shape
    log_probs = np.log(probs + 1e-10)

    max_transition = 12
    transition = np.zeros(n_bins)
    for i in range(n_bins):
        d = min(i, n_bins - i)
        if d <= max_transition:
            transition[i] = np.exp(-0.5 * (d / 4.0) ** 2)
    transition /= transition.sum()
    log_transition = np.log(transition + 1e-10)

    viterbi = np.zeros((n_frames, n_bins))
    backptr = np.zeros((n_frames, n_bins), dtype=int)
    viterbi[0] = log_probs[0]

    for t in range(1, n_frames):
        for j in range(n_bins):
            trans_scores = viterbi[t - 1] + np.roll(log_transition, j)
            backptr[t, j] = np.argmax(trans_scores)
            viterbi[t, j] = trans_scores[backptr[t, j]] + log_probs[t, j]

    path = np.zeros(n_frames, dtype=int)
    path[-1] = np.argmax(viterbi[-1])
    for t in range(n_frames - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


def detect_pitch_crepe(y: np.ndarray, sr: int) -> tuple:
    """
    Detección de pitch con CREPE vía ONNX Runtime.
    Retorna: (f0, voiced_flag, confidence, hop_length)
    """
    step_size_ms = 10
    crepe_sr = 16000
    frame_size = 1024
    step_samples = int(crepe_sr * step_size_ms / 1000)

    if sr != crepe_sr:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=crepe_sr)
    else:
        y_16k = y.copy()
    y_16k = y_16k.astype(np.float32)

    n_samples = len(y_16k)
    n_frames = max(1, 1 + (n_samples - frame_size) // step_samples)

    frames = np.zeros((n_frames, frame_size), dtype=np.float32)
    for i in range(n_frames):
        s = i * step_samples
        e = s + frame_size
        if e <= n_samples:
            frames[i] = y_16k[s:e]
        else:
            avail = n_samples - s
            if avail > 0:
                frames[i, :avail] = y_16k[s:]

    norms = np.maximum(np.max(np.abs(frames), axis=1, keepdims=True), 1e-8)
    frames = frames / norms

    session = _get_crepe_session()
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name

    all_probs = []
    for i in range(0, n_frames, 128):
        batch = frames[i:i + 128]
        all_probs.append(session.run([out], {inp: batch})[0])
    probs = np.vstack(all_probs)

    confidence = np.max(probs, axis=1)

    frequency = np.zeros(n_frames)
    if n_frames > 2:
        vpath = _viterbi_decode(probs)
    else:
        vpath = np.argmax(probs, axis=1)

    for i in range(n_frames):
        center = vpath[i]
        lo = max(0, center - 4)
        hi = min(360, center + 5)
        lp = probs[i, lo:hi]
        lc = _CREPE_CENTS[lo:hi]
        if np.sum(lp) > 0:
            cents = np.sum(lp * lc) / np.sum(lp)
        else:
            cents = _CREPE_CENTS[center]
        frequency[i] = 10.0 * 2.0 ** (cents / 1200.0)

    hop_length = int(sr * step_size_ms / 1000)

    voiced_flag = confidence >= 0.5
    f0 = frequency.copy().astype(np.float64)
    f0[~voiced_flag] = np.nan
    f0[f0 <= 0] = np.nan

    return f0, voiced_flag, confidence.copy(), hop_length


# ---------------------------------------------------------------------------
# 3. SUAVIZADO LIGERO POST-CREPE
# ---------------------------------------------------------------------------

def smooth_pitch(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Filtro de mediana pequeño para eliminar outliers residuales."""
    f0c = f0.copy()
    mask = voiced & ~np.isnan(f0)
    if np.sum(mask) < 5:
        return f0c
    f0c[mask] = median_filter(f0c[mask], size=3)
    return f0c


# ---------------------------------------------------------------------------
# 4. DETECCIÓN DE ENERGÍA RMS (para distinguir silencio real de transiciones)
# ---------------------------------------------------------------------------

def compute_rms_envelope(y: np.ndarray, sr: int, hop_length: int,
                         n_pitch_frames: int) -> np.ndarray:
    """
    Calcula la envolvente RMS del audio alineada con los frames de pitch.
    Retorna un array booleano: True = hay energía suficiente (sonido),
    False = silencio real.
    """
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # Alinear longitud con los frames de pitch
    if len(rms) >= n_pitch_frames:
        rms = rms[:n_pitch_frames]
    else:
        rms = np.pad(rms, (0, n_pitch_frames - len(rms)), mode='edge')

    # Umbral adaptativo: el 15% de la RMS máxima
    threshold = np.max(rms) * 0.15
    has_energy = rms >= threshold

    return has_energy


# ---------------------------------------------------------------------------
# 5. SEGMENTACIÓN EN NOTAS ESTABLES
# ---------------------------------------------------------------------------

def hz_to_midi_float(freq):
    """Convierte Hz a MIDI float."""
    if freq <= 0 or np.isnan(freq):
        return np.nan
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def detect_onsets(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Detecta los momentos donde comienza una nueva nota usando
    onset detection de librosa.
    """
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    return onset_frames


def extract_stable_notes(f0: np.ndarray, voiced: np.ndarray,
                         has_energy: np.ndarray, hop_length: int,
                         sr: int, onset_frames: np.ndarray = None) -> list:
    """
    Extrae notas combinando regiones estables de pitch con onsets.

    Una "nota" es una región continua donde:
    - Hay energía (RMS > umbral)
    - Hay pitch confiable (voiced)
    - El MIDI redondeado se mantiene igual
    - No hay un onset que marque inicio de nota nueva

    Los onsets fuerzan un corte de nota incluso si el pitch
    no cambió (ej: misma nota repetida con articulación).
    """
    frame_dur = hop_length / sr
    n = len(f0)

    onset_set = set()
    if onset_frames is not None:
        onset_set = set(onset_frames.tolist())

    # Convertir a MIDI redondeado para cada frame
    midi_round = np.full(n, np.nan)
    midi_float = np.full(n, np.nan)
    for i in range(n):
        if voiced[i] and has_energy[i] and not np.isnan(f0[i]):
            mf = hz_to_midi_float(f0[i])
            midi_float[i] = mf
            midi_round[i] = round(mf)

    # Construir boundaries: cambios de pitch + onsets + voiced/unvoiced
    boundaries = {0}
    for i in range(1, n):
        # Cambio voiced/unvoiced
        prev_valid = not np.isnan(midi_round[i - 1])
        curr_valid = not np.isnan(midi_round[i])
        if prev_valid != curr_valid:
            boundaries.add(i)
            continue
        # Cambio de pitch (> 2 semitonos)
        if prev_valid and curr_valid:
            if abs(midi_float[i] - midi_float[i - 1]) > 2.5:
                boundaries.add(i)
                continue
        # Onset detectado (nueva articulación)
        if i in onset_set and curr_valid:
            boundaries.add(i)

    boundaries = sorted(boundaries)

    # Procesar cada segmento
    notes = []
    for seg_idx in range(len(boundaries)):
        start_frame = boundaries[seg_idx]
        end_frame = (boundaries[seg_idx + 1]
                     if seg_idx + 1 < len(boundaries) else n)
        if end_frame <= start_frame:
            continue

        seg_midi = midi_float[start_frame:end_frame]
        valid_mask = ~np.isnan(seg_midi)
        if np.sum(valid_mask) < 2:
            continue

        valid_pitches = seg_midi[valid_mask]

        # Pitch promedio (media recortada)
        if len(valid_pitches) >= 4:
            sp = np.sort(valid_pitches)
            trim = max(1, len(sp) // 6)
            avg_midi = float(np.mean(sp[trim:-trim]))
        else:
            avg_midi = float(np.median(valid_pitches))

        midi_note = int(np.clip(round(avg_midi), 0, 127))
        start_time = start_frame * frame_dur
        end_time = end_frame * frame_dur

        if (end_time - start_time) >= 0.05:
            notes.append((midi_note, start_time, end_time, avg_midi))

    return notes


# ---------------------------------------------------------------------------
# 6. CORRECCIÓN DE OCTAVA
# ---------------------------------------------------------------------------

def fix_octave_jumps(notes: list) -> list:
    """Corrige saltos de octava usando mediana del contexto."""
    if len(notes) < 3:
        return notes

    corrected = list(notes)
    midis = [n[0] for n in corrected]

    for i in range(len(corrected)):
        ws = max(0, i - 5)
        we = min(len(corrected), i + 6)
        ctx = [midis[j] for j in range(ws, we) if j != i]
        if not ctx:
            continue
        med = np.median(ctx)
        diff = midis[i] - med
        if abs(diff) > 9:
            shift = round(diff / 12) * 12
            new_m = int(np.clip(midis[i] - shift, 0, 127))
            corrected[i] = (new_m, corrected[i][1], corrected[i][2], corrected[i][3])
            midis[i] = new_m

    return corrected


# ---------------------------------------------------------------------------
# 7. FUSIÓN Y LIMPIEZA
# ---------------------------------------------------------------------------

def merge_notes(notes: list, bpm: float) -> list:
    """
    Fusión conservadora:
    - Une notas iguales separadas por gaps menores a una corchea
    - Absorbe notas muy cortas (< corchea) en la vecina más cercana
    """
    if len(notes) < 2:
        return notes

    beat_dur = 60.0 / bpm
    eighth_dur = beat_dur / 2  # duración de una corchea en segundos

    # Paso 1: fusionar notas iguales con gaps pequeños
    merged = [list(notes[0])]
    for n in notes[1:]:
        prev = merged[-1]
        gap = n[1] - prev[2]
        if n[0] == prev[0] and gap < eighth_dur * 0.6:
            prev[2] = n[2]
        else:
            merged.append(list(n))

    # Paso 2: absorber notas más cortas que una corchea
    cleaned = []
    for n in merged:
        dur = n[2] - n[1]
        if dur < eighth_dur * 0.8 and len(cleaned) > 0:
            prev = cleaned[-1]
            if abs(n[0] - prev[0]) <= 2:
                prev[2] = n[2]
                continue
        cleaned.append(n)

    # Paso 3: segunda fusión tras absorción
    merged2 = [cleaned[0]]
    for n in cleaned[1:]:
        prev = merged2[-1]
        gap = n[1] - prev[2]
        if n[0] == prev[0] and gap < eighth_dur * 0.4:
            prev[2] = n[2]
        else:
            merged2.append(n)

    # Paso 4: filtrar por duración mínima (al menos media corchea)
    min_dur = eighth_dur * 0.5
    result = [tuple(n) for n in merged2 if (n[2] - n[1]) >= min_dur]

    return result


# ---------------------------------------------------------------------------
# 8. DETECCIÓN DE TEMPO
# ---------------------------------------------------------------------------

def detect_tempo(y: np.ndarray, sr: int) -> float:
    """Detecta el tempo del audio."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempos = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    if len(tempos) > 0:
        bpm = float(np.median(tempos))
    else:
        bpm = 120.0
    bpm = round(bpm / 2) * 2
    return float(max(40, min(200, bpm)))


# ---------------------------------------------------------------------------
# 9. CUANTIZACIÓN CONSCIENTE DEL COMPÁS
# ---------------------------------------------------------------------------

def quantize_to_grid(notes: list, bpm: float) -> list:
    """
    Cuantiza notas a una rejilla de corcheas (no semicorcheas ni fusas).
    Más grueso = notación más limpia.
    Luego elimina duplicados y asegura que las notas no se solapen.
    """
    beat_dur = 60.0 / bpm
    grid = beat_dur / 2  # corcheas

    quantized = []
    for midi_num, start, end, avg_midi in notes:
        qs = round(start / grid) * grid
        qe = round(end / grid) * grid
        if qe <= qs:
            qe = qs + grid
        quantized.append((midi_num, qs, qe, avg_midi))

    # Eliminar duplicados y solapamientos
    if len(quantized) < 2:
        return quantized

    result = [quantized[0]]
    for n in quantized[1:]:
        prev = result[-1]
        # Misma posición → fusionar
        if n[1] == prev[1]:
            if (n[2] - n[1]) > (prev[2] - prev[1]):
                result[-1] = n
            continue
        # Misma nota solapada → extender
        if n[0] == prev[0] and n[1] < prev[2]:
            result[-1] = (prev[0], prev[1], max(prev[2], n[2]), prev[3])
            continue
        # Si se solapan notas distintas, recortar la anterior
        if n[1] < prev[2]:
            result[-1] = (prev[0], prev[1], n[1], prev[3])
        result.append(n)

    return result


def snap_duration(ql: float) -> float:
    """Ajusta quarterLength al valor musical simple más cercano."""
    valid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(valid, key=lambda v: abs(v - ql))


# ---------------------------------------------------------------------------
# 10. DETECCIÓN DE TONALIDAD Y CORRECCIÓN
# ---------------------------------------------------------------------------

def detect_key_from_notes(notes: list, bpm: float = 120.0) -> dict:
    """Detecta la tonalidad más probable usando duraciones ponderadas."""
    if not notes:
        return {"key": "C major", "key_name": "Do Mayor", "confidence": 0}

    beat_dur = 60.0 / bpm
    s = stream.Stream()
    for midi_num, start, end, *_ in notes:
        dur_ql = max(0.5, (end - start) / beat_dur)
        n = note.Note(midi_num)
        n.quarterLength = dur_ql
        s.append(n)

    try:
        detected = s.analyze("key")
        key_str = str(detected)
        names = {
            "C major": "Do Mayor", "C minor": "Do menor",
            "C# major": "Do# Mayor", "C# minor": "Do# menor",
            "D- major": "Reb Mayor", "D- minor": "Reb menor",
            "D major": "Re Mayor", "D minor": "Re menor",
            "D# major": "Re# Mayor", "D# minor": "Re# menor",
            "E- major": "Mib Mayor", "E- minor": "Mib menor",
            "E major": "Mi Mayor", "E minor": "Mi menor",
            "F major": "Fa Mayor", "F minor": "Fa menor",
            "F# major": "Fa# Mayor", "F# minor": "Fa# menor",
            "G- major": "Solb Mayor", "G- minor": "Solb menor",
            "G major": "Sol Mayor", "G minor": "Sol menor",
            "G# major": "Sol# Mayor", "G# minor": "Sol# menor",
            "A- major": "Lab Mayor", "A- minor": "Lab menor",
            "A major": "La Mayor", "A minor": "La menor",
            "A# major": "La# Mayor", "A# minor": "La# menor",
            "B- major": "Sib Mayor", "B- minor": "Sib menor",
            "B major": "Si Mayor", "B minor": "Si menor",
        }
        conf = 0
        if hasattr(detected, 'correlationCoefficient'):
            conf = round(detected.correlationCoefficient, 2)
        return {
            "key": key_str,
            "key_name": names.get(key_str, key_str),
            "confidence": conf,
        }
    except Exception:
        return {"key": "C major", "key_name": "Do Mayor", "confidence": 0}


def snap_to_key(midi_num: int, avg_midi: float, key_obj) -> int:
    """Corrige notas fuera de la escala solo si están a 1 semitono."""
    try:
        scale_pcs = [p.pitchClass for p in key_obj.getScale().pitches]
        pc = midi_num % 12
        if pc in scale_pcs:
            return midi_num

        # Solo corregir si el MIDI float estaba cerca del borde
        deviation = abs(avg_midi - round(avg_midi))
        if deviation > 0.4:
            return midi_num

        dists = []
        for spc in scale_pcs:
            d = min(abs(pc - spc), 12 - abs(pc - spc))
            dists.append((d, spc))
        dists.sort()

        if dists[0][0] <= 1:
            closest = dists[0][1]
            diff = closest - pc
            if abs(diff) > 6:
                diff = diff - 12 if diff > 0 else diff + 12
            return int(np.clip(midi_num + diff, 0, 127))
        return midi_num
    except Exception:
        return midi_num


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def audio_to_midi(audio_path: str, midi_path: str,
                  user_bpm: float = None) -> dict:
    """
    Pipeline fiel a la voz:
    1. Preprocesamiento
    2. Detección de pitch con CREPE
    3. Suavizado ligero
    4. Detección de energía RMS (silencio real vs transiciones)
    5. Extracción de notas estables (no micro-segmentos)
    6. Corrección de octava
    7. Fusión conservadora
    8. Cuantización a rejilla de corcheas
    9. Generación MIDI
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    y_clean = preprocess_audio(y, sr)

    # Tempo: usar el del usuario si lo proporcionó, si no detectar
    if user_bpm and 30 <= user_bpm <= 300:
        detected_bpm = float(round(user_bpm))
    else:
        detected_bpm = detect_tempo(y_clean, sr)

    # Pitch con CREPE
    f0, voiced, confidence, hop_length = detect_pitch_crepe(y_clean, sr)
    f0 = smooth_pitch(f0, voiced)

    # Energía RMS para distinguir silencio real
    has_energy = compute_rms_envelope(y_clean, sr, hop_length, len(f0))

    # Detección de onsets (inicios de nota por articulación)
    onset_frames = detect_onsets(y_clean, sr, hop_length)

    # Extraer notas: regiones estables + onsets + RMS
    notes = extract_stable_notes(
        f0, voiced, has_energy, hop_length, sr, onset_frames
    )

    # Corrección de octava
    notes = fix_octave_jumps(notes)

    # Fusión relativa al BPM
    notes = merge_notes(notes, detected_bpm)

    # Cuantización a rejilla de corcheas
    notes = quantize_to_grid(notes, detected_bpm)

    # Generar MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=detected_bpm)
    voice = pretty_midi.Instrument(program=0, name="Voice")
    for midi_num, start, end, *_ in notes:
        voice.notes.append(pretty_midi.Note(
            velocity=100, pitch=midi_num, start=start, end=end
        ))
    midi.instruments.append(voice)
    midi.write(midi_path)

    return {
        "duration_seconds": round(duration, 2),
        "notes_detected": len(notes),
        "detected_bpm": detected_bpm,
        "midi_path": midi_path,
        "notes_list": notes,
    }


# ---------------------------------------------------------------------------
# GENERACIÓN DE PARTITURA
# ---------------------------------------------------------------------------

def simplify_notes_for_score(notes_list: list, bpm: float) -> list:
    """
    Simplificación final antes de la partitura:
    - Absorbe gaps pequeños extendiendo la nota anterior
    - Fusiona notas repetidas consecutivas que quedaron pegadas
    """
    if len(notes_list) < 2:
        return notes_list

    beat_dur = 60.0 / bpm
    min_rest = beat_dur * 0.4

    simplified = [list(notes_list[0])]
    for i in range(1, len(notes_list)):
        curr = list(notes_list[i])
        prev = simplified[-1]
        gap = curr[1] - prev[2]

        # Gap demasiado pequeño para ser silencio real
        if 0 < gap < min_rest:
            prev[2] = curr[1]

        # Fusionar si misma nota y continua
        if curr[0] == prev[0] and curr[1] <= prev[2] + 0.01:
            prev[2] = max(prev[2], curr[2])
        else:
            simplified.append(curr)

    return [tuple(n) for n in simplified]


def duration_to_ql(dur_seconds: float, bpm: float) -> float:
    """Convierte duración en segundos a quarter-note length."""
    return dur_seconds / (60.0 / bpm)


def midi_to_score(midi_path: str, output_base: str,
                  detected_bpm: float, notes_list: list) -> dict:
    """
    Genera partitura con análisis detallado:
    - Tonalidad detectada con duraciones correctas en quarter-notes
    - Construcción secuencial: nota por nota con silencios explícitos
    - Métrica 4/4 respetada: duraciones snapped a valores simples
    - Corrección tonal inteligente
    - Silencios solo donde realmente hay pausa (>= 1 corchea)
    """
    key_info = detect_key_from_notes(notes_list, detected_bpm)

    try:
        parts = key_info["key"].split()
        key_obj = m21key.Key(
            parts[0], parts[1] if len(parts) > 1 else "major"
        )
    except Exception:
        key_obj = m21key.Key("C", "major")

    # Simplificar notas antes de escribir
    notes_list = simplify_notes_for_score(notes_list, detected_bpm)

    beat_dur = 60.0 / detected_bpm

    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = "Transcripcion de Voz"
    s.metadata.composer = "Partituras AI"

    part = stream.Part()
    part.append(key_obj)
    part.append(meter.TimeSignature("4/4"))
    part.append(tempo.MetronomeMark(number=detected_bpm))

    # Construcción secuencial con silencios explícitos
    min_rest_ql = 0.5  # mínimo una corchea para crear silencio
    current_time = 0.0  # en segundos

    for midi_num, start, end, *extra in notes_list:
        avg_midi = extra[0] if extra else float(midi_num)

        # Silencio entre nota anterior y esta nota
        gap = start - current_time
        if gap > 0:
            rest_ql = duration_to_ql(gap, detected_bpm)
            rest_ql = snap_duration(rest_ql)
            if rest_ql >= min_rest_ql:
                r = Rest()
                r.quarterLength = rest_ql
                part.append(r)

        # Corrección tonal
        corrected = snap_to_key(midi_num, avg_midi, key_obj)

        # Duración en quarter-notes
        dur_ql = duration_to_ql(end - start, detected_bpm)
        dur_ql = snap_duration(dur_ql)

        n = note.Note(corrected)
        n.quarterLength = dur_ql
        part.append(n)

        current_time = end

    s.append(part)

    # Dejar que music21 rellene gaps restantes
    try:
        part.makeRests(fillGaps=True, inPlace=True)
    except Exception:
        pass

    xml_path = output_base + ".musicxml"
    s.write("musicxml", fp=xml_path)

    results = {"musicxml": xml_path, "key_info": key_info}

    try:
        png_path = output_base + ".png"
        s.write("lily.png", fp=png_path)
        results["png"] = png_path
    except Exception:
        pass

    return results
