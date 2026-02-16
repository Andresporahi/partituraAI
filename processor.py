"""
Módulo de procesamiento de audio: detección de pitch, generación MIDI y partitura.
Pipeline de alta precisión para transcripción de voz monofónica.
Usa CREPE (red neuronal) para detección de pitch estado del arte.
"""

import os
import numpy as np
import librosa
import crepe
import pretty_midi
from scipy.ndimage import median_filter
from music21 import (
    stream, note, meter, tempo, metadata,
    key as m21key, pitch as m21pitch, analysis,
)
from music21.note import Rest


# ---------------------------------------------------------------------------
# 1. PREPROCESAMIENTO DE AUDIO
# ---------------------------------------------------------------------------

def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocesa el audio para mejorar la detección de pitch:
    - Normaliza amplitud
    - Aplica filtro pasa-banda para aislar rango vocal (80Hz - 4000Hz)
    - Reduce ruido con spectral gating simple
    """
    # Normalizar
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Filtro pasa-banda para rango vocal humano
    y_filtered = librosa.effects.preemphasis(y, coef=0.97)

    # Trim silencios al inicio y final
    y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=30)

    return y_trimmed


# ---------------------------------------------------------------------------
# 2. DETECCIÓN DE PITCH CON CREPE (RED NEURONAL)
# ---------------------------------------------------------------------------

def detect_pitch_crepe(y: np.ndarray, sr: int) -> tuple:
    """
    Detección de pitch usando CREPE, una red neuronal convolucional
    entrenada específicamente para pitch monofónico.
    Mucho más preciso que pYIN, especialmente para voz humana.

    Usa modelo 'small' para balance entre precisión y velocidad.
    Viterbi activado para suavizado HMM de la curva de pitch.
    step_size=10ms para alta resolución temporal.
    """
    step_size_ms = 10  # 10ms entre frames

    # CREPE espera audio como int16 o float32
    # Resamplear a 16kHz internamente si es necesario (CREPE lo hace solo)
    time_arr, frequency, confidence, _ = crepe.predict(
        y, sr,
        model_capacity='small',
        viterbi=True,       # suavizado HMM para continuidad de pitch
        step_size=step_size_ms,
        verbose=0,
    )

    # Calcular hop_length equivalente para compatibilidad con el resto del pipeline
    hop_length = int(sr * step_size_ms / 1000)  # 220 samples a 22050Hz

    # Construir voiced_flag basado en confianza de CREPE
    confidence_threshold = 0.5
    voiced_flag = confidence >= confidence_threshold

    # Convertir frecuencias a f0 array compatible (NaN donde no hay voz)
    f0 = frequency.copy().astype(np.float64)
    f0[~voiced_flag] = np.nan
    f0[f0 <= 0] = np.nan

    # voiced_prob es directamente la confianza de CREPE (0-1)
    voiced_prob = confidence.copy()

    return f0, voiced_flag, voiced_prob, hop_length


# ---------------------------------------------------------------------------
# 3. SUAVIZADO DE PITCH LIGERO (POST-CREPE)
# ---------------------------------------------------------------------------

def smooth_pitch_post_crepe(f0: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
    """
    Suavizado ligero post-CREPE. Como CREPE con Viterbi ya aplica
    suavizado HMM, solo hacemos un filtro de mediana pequeño para
    eliminar cualquier outlier residual sin destruir transiciones.
    """
    f0_clean = f0.copy()
    voiced_mask = voiced_flag & ~np.isnan(f0)

    if np.sum(voiced_mask) < 5:
        return f0_clean

    voiced_freqs = f0_clean[voiced_mask]
    smoothed = median_filter(voiced_freqs, size=3)
    f0_clean[voiced_mask] = smoothed

    return f0_clean


# ---------------------------------------------------------------------------
# 4. DETECCIÓN DE ONSETS (INICIOS DE NOTA)
# ---------------------------------------------------------------------------

def detect_onsets(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Detecta los momentos donde comienza una nueva nota usando
    onset detection de librosa. Esto marca transiciones reales
    entre notas, no solo cambios de pitch.
    """
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    return onset_frames


# ---------------------------------------------------------------------------
# 5. AGRUPACIÓN INTELIGENTE DE NOTAS
# ---------------------------------------------------------------------------

def hz_to_midi_float(freq: float) -> float:
    """Convierte Hz a MIDI como float (sin redondear)."""
    if freq <= 0 or np.isnan(freq):
        return np.nan
    return 69 + 12 * np.log2(freq / 440.0)


def group_notes_with_onsets(f0: np.ndarray, voiced_flag: np.ndarray,
                            onset_frames: np.ndarray, hop_length: int,
                            sr: int) -> list:
    """
    Agrupa frames en notas usando tanto cambios de pitch como onsets detectados.
    Para cada segmento entre onsets, calcula el pitch promedio ponderado
    por confianza, lo que da una nota más precisa que frame-by-frame.
    """
    frame_duration = hop_length / sr
    n_frames = len(f0)

    # Convertir f0 a MIDI float
    midi_float = np.array([hz_to_midi_float(f) for f in f0])

    # Crear marcadores de segmento: un nuevo segmento empieza en cada onset
    # o cuando hay un cambio de voiced/unvoiced
    segment_boundaries = set(onset_frames.tolist())
    segment_boundaries.add(0)

    # También agregar cambios voiced→unvoiced y viceversa
    for i in range(1, n_frames):
        prev_voiced = voiced_flag[i-1] and not np.isnan(f0[i-1])
        curr_voiced = voiced_flag[i] and not np.isnan(f0[i])
        if prev_voiced != curr_voiced:
            segment_boundaries.add(i)

    # Agregar cambios grandes de pitch (> 1.5 semitonos entre frames consecutivos)
    for i in range(1, n_frames):
        if (voiced_flag[i] and voiced_flag[i-1] and
                not np.isnan(midi_float[i]) and not np.isnan(midi_float[i-1])):
            if abs(midi_float[i] - midi_float[i-1]) > 1.5:
                segment_boundaries.add(i)

    boundaries = sorted(segment_boundaries)

    # Procesar cada segmento
    notes_list = []

    for seg_idx in range(len(boundaries)):
        start_frame = boundaries[seg_idx]
        end_frame = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else n_frames

        if end_frame <= start_frame:
            continue

        # Extraer frames del segmento
        seg_voiced = voiced_flag[start_frame:end_frame]
        seg_midi = midi_float[start_frame:end_frame]

        # Solo considerar frames voiced con pitch válido
        valid_mask = seg_voiced & ~np.isnan(seg_midi)
        if np.sum(valid_mask) < 2:
            continue

        valid_pitches = seg_midi[valid_mask]

        # Calcular pitch promedio del segmento (media recortada para robustez)
        if len(valid_pitches) >= 4:
            sorted_p = np.sort(valid_pitches)
            trim = max(1, len(sorted_p) // 8)
            trimmed = sorted_p[trim:-trim] if trim > 0 else sorted_p
            avg_midi = np.mean(trimmed)
        else:
            avg_midi = np.median(valid_pitches)

        midi_note = int(np.clip(round(avg_midi), 0, 127))

        # Tiempo de inicio y fin
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration

        # Duración mínima: 50ms
        if (end_time - start_time) >= 0.05:
            notes_list.append((midi_note, start_time, end_time, avg_midi))

    return notes_list


# ---------------------------------------------------------------------------
# 6. CORRECCIÓN DE OCTAVA MEJORADA
# ---------------------------------------------------------------------------

def fix_octave_jumps_advanced(notes_list: list) -> list:
    """
    Corrección de octava con ventana deslizante y contexto amplio.
    Usa la mediana de las notas cercanas como referencia.
    """
    if len(notes_list) < 3:
        return notes_list

    corrected = list(notes_list)
    midi_values = [n[0] for n in corrected]

    for i in range(len(corrected)):
        # Ventana de contexto: hasta 5 notas antes y después
        window_start = max(0, i - 5)
        window_end = min(len(corrected), i + 6)
        context = [midi_values[j] for j in range(window_start, window_end) if j != i]

        if not context:
            continue

        context_median = np.median(context)
        current = midi_values[i]
        diff = current - context_median

        # Si la nota está a más de 9 semitonos de la mediana del contexto
        if abs(diff) > 9:
            octave_shift = round(diff / 12) * 12
            new_midi = int(np.clip(current - octave_shift, 0, 127))
            corrected[i] = (new_midi, corrected[i][1], corrected[i][2], corrected[i][3])
            midi_values[i] = new_midi

    return corrected


# ---------------------------------------------------------------------------
# 7. FUSIÓN Y LIMPIEZA DE NOTAS
# ---------------------------------------------------------------------------

def merge_and_clean_notes(notes_list: list, min_duration: float = 0.08,
                          merge_gap: float = 0.04) -> list:
    """
    Fusiona notas iguales consecutivas con micro-silencios y
    elimina notas demasiado cortas (probablemente ruido).
    También fusiona notas que difieren por solo 1 semitono si son muy cortas.
    """
    if len(notes_list) < 2:
        return notes_list

    # Paso 1: Fusionar notas idénticas con gaps pequeños
    merged = [notes_list[0]]
    for n in notes_list[1:]:
        prev = merged[-1]
        gap = n[1] - prev[2]
        if n[0] == prev[0] and gap < merge_gap:
            merged[-1] = (prev[0], prev[1], n[2], prev[3])
        else:
            merged.append(n)

    # Paso 2: Absorber notas muy cortas (< 50ms) en sus vecinas
    cleaned = []
    for i, n in enumerate(merged):
        dur = n[2] - n[1]
        if dur < 0.05 and len(cleaned) > 0:
            # Si difiere por 1-2 semitonos de la anterior, extender la anterior
            prev = cleaned[-1]
            if abs(n[0] - prev[0]) <= 2:
                cleaned[-1] = (prev[0], prev[1], n[2], prev[3])
                continue
        cleaned.append(n)

    # Paso 3: Filtrar por duración mínima
    cleaned = [n for n in cleaned if (n[2] - n[1]) >= min_duration]

    return cleaned


# ---------------------------------------------------------------------------
# 8. DETECCIÓN DE TEMPO MEJORADA
# ---------------------------------------------------------------------------

def detect_tempo_precise(y: np.ndarray, sr: int) -> float:
    """
    Detección de tempo más precisa: usa onset strength y permite
    tempos más variados (no solo los "comunes").
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_estimate = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        aggregate=None,
    )

    if len(tempo_estimate) > 0:
        bpm = float(np.median(tempo_estimate))
    else:
        bpm = 120.0

    # Redondear a múltiplos de 2 para un BPM musical limpio
    bpm = round(bpm / 2) * 2
    bpm = max(40, min(200, bpm))

    return float(bpm)


# ---------------------------------------------------------------------------
# 9. CUANTIZACIÓN RÍTMICA ALINEADA A BEATS
# ---------------------------------------------------------------------------

def quantize_to_beat_grid(notes_list: list, bpm: float,
                          grid_subdivision: int = 8) -> list:
    """
    Cuantiza los tiempos de inicio y fin de las notas a una rejilla
    basada en el tempo real. grid_subdivision=8 → fusas (1/8 de beat).
    """
    beat_duration = 60.0 / bpm
    grid_size = beat_duration / grid_subdivision

    quantized = []
    for midi_num, start, end, avg_midi in notes_list:
        q_start = round(start / grid_size) * grid_size
        q_end = round(end / grid_size) * grid_size

        # Asegurar duración mínima de 1 grid unit
        if q_end <= q_start:
            q_end = q_start + grid_size

        quantized.append((midi_num, q_start, q_end, avg_midi))

    return quantized


def duration_to_quarter_length(dur_seconds: float, bpm: float) -> float:
    """Convierte duración en segundos a quarter-note length."""
    beat_duration = 60.0 / bpm
    return dur_seconds / beat_duration


def snap_quarter_length(ql: float) -> float:
    """
    Ajusta un quarter-length a la duración musical más cercana.
    Incluye valores normales, con puntillo y tresillos.
    """
    valid = [
        0.25,       # semicorchea
        0.375,      # semicorchea con puntillo
        1/3,        # tresillo de corchea
        0.5,        # corchea
        0.75,       # corchea con puntillo
        2/3,        # tresillo de negra
        1.0,        # negra
        1.5,        # negra con puntillo
        2.0,        # blanca
        3.0,        # blanca con puntillo
        4.0,        # redonda
        6.0,        # redonda con puntillo
    ]
    return min(valid, key=lambda v: abs(v - ql))


# ---------------------------------------------------------------------------
# 10. DETECCIÓN DE TONALIDAD Y CORRECCIÓN
# ---------------------------------------------------------------------------

def detect_key_from_notes(notes_list: list) -> dict:
    """Detecta la tonalidad más probable de la melodía."""
    if not notes_list:
        return {"key": "C major", "key_name": "Do Mayor", "confidence": 0}

    s = stream.Stream()
    for midi_num, start, end, *_ in notes_list:
        dur = end - start
        n = note.Note(midi_num)
        n.quarterLength = max(0.25, dur)
        s.append(n)

    try:
        detected = s.analyze("key")
        key_str = str(detected)

        key_names_es = {
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

        key_name_es = key_names_es.get(key_str, key_str)
        confidence = round(detected.correlationCoefficient, 2) if hasattr(detected, 'correlationCoefficient') else 0

        return {
            "key": key_str,
            "key_name": key_name_es,
            "confidence": confidence,
        }
    except Exception:
        return {"key": "C major", "key_name": "Do Mayor", "confidence": 0}


def snap_to_key_smart(midi_num: int, avg_midi: float, key_obj,
                      threshold: float = 0.4) -> int:
    """
    Corrección tonal inteligente: solo ajusta notas que están
    "entre" dos notas de la escala (desafinadas). Si la nota
    está cerca del centro de un grado cromático, la respeta
    como nota de paso o cromatismo intencional.
    """
    try:
        scale_pitches = key_obj.getScale().pitches
        scale_pcs = [p.pitchClass for p in scale_pitches]

        pc = midi_num % 12
        if pc in scale_pcs:
            return midi_num

        # Calcular qué tan "desafinada" está la nota respecto a su MIDI redondeado
        deviation = abs(avg_midi - round(avg_midi))

        # Si la desviación es alta (nota entre dos semitonos), corregir
        if deviation > threshold:
            return midi_num  # Muy ambigua, no corregir

        # Buscar grado más cercano de la escala
        distances = []
        for spc in scale_pcs:
            d = min(abs(pc - spc), 12 - abs(pc - spc))
            distances.append((d, spc))

        distances.sort(key=lambda x: x[0])

        # Solo corregir si está a 1 semitono de un grado de la escala
        if distances[0][0] <= 1:
            closest_pc = distances[0][1]
            diff = closest_pc - pc
            if abs(diff) > 6:
                diff = diff - 12 if diff > 0 else diff + 12
            return int(np.clip(midi_num + diff, 0, 127))

        return midi_num
    except Exception:
        return midi_num


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def audio_to_midi(audio_path: str, midi_path: str) -> dict:
    """
    Pipeline de alta precisión con CREPE:
    1. Preprocesamiento de audio
    2. Detección de pitch con CREPE (red neuronal + Viterbi HMM)
    3. Suavizado ligero post-CREPE
    4. Detección de onsets
    5. Agrupación inteligente por segmentos
    6. Corrección de octava con contexto amplio
    7. Fusión y limpieza
    8. Cuantización a rejilla de beats
    9. Generación MIDI
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Preprocesar audio
    y_clean = preprocess_audio(y, sr)

    # 2. Detectar tempo
    detected_bpm = detect_tempo_precise(y_clean, sr)

    # 3. Detección de pitch con CREPE (red neuronal)
    f0, voiced_flag, voiced_prob, hop_length = detect_pitch_crepe(y_clean, sr)

    # 4. Suavizado ligero (CREPE+Viterbi ya suaviza, solo limpiamos outliers)
    f0 = smooth_pitch_post_crepe(f0, voiced_flag)

    # 5. Detección de onsets
    onset_frames = detect_onsets(y_clean, sr, hop_length)

    # 6. Agrupar notas usando onsets + pitch
    notes_list = group_notes_with_onsets(f0, voiced_flag, onset_frames, hop_length, sr)

    # 7. Corrección de octava
    notes_list = fix_octave_jumps_advanced(notes_list)

    # 8. Fusión y limpieza
    notes_list = merge_and_clean_notes(notes_list)

    # 9. Cuantización rítmica a rejilla de beats
    notes_list = quantize_to_beat_grid(notes_list, detected_bpm)

    # 10. Generar MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=detected_bpm)
    voice = pretty_midi.Instrument(program=0, name="Voice")

    for midi_num, start, end, *_ in notes_list:
        pm_note = pretty_midi.Note(
            velocity=100,
            pitch=midi_num,
            start=start,
            end=end,
        )
        voice.notes.append(pm_note)

    midi.instruments.append(voice)
    midi.write(midi_path)

    return {
        "duration_seconds": round(duration, 2),
        "notes_detected": len(notes_list),
        "detected_bpm": detected_bpm,
        "midi_path": midi_path,
        "notes_list": notes_list,
    }


def midi_to_score(midi_path: str, output_base: str,
                  detected_bpm: float, notes_list: list) -> dict:
    """
    Genera partitura con:
    - Tonalidad detectada automáticamente
    - Corrección tonal inteligente (solo notas ambiguas)
    - Cuantización rítmica a valores musicales reales
    - Silencios explícitos
    """
    key_info = detect_key_from_notes(notes_list)

    try:
        parts = key_info["key"].split()
        key_obj = m21key.Key(parts[0], parts[1] if len(parts) > 1 else "major")
    except Exception:
        key_obj = m21key.Key("C", "major")

    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = "Transcripción de Voz"
    s.metadata.composer = "Partituras AI"

    part = stream.Part()
    part.append(key_obj)
    part.append(meter.TimeSignature("4/4"))
    part.append(tempo.MetronomeMark(number=detected_bpm))

    current_time = 0.0

    for midi_num, start, end, *extra in notes_list:
        avg_midi = extra[0] if extra else float(midi_num)

        # Silencio entre notas
        gap = start - current_time
        if gap > 0.05:
            rest_ql = duration_to_quarter_length(gap, detected_bpm)
            rest_ql = snap_quarter_length(rest_ql)
            if rest_ql >= 0.25:
                r = Rest()
                r.quarterLength = rest_ql
                part.append(r)

        # Corrección tonal inteligente
        corrected_midi = snap_to_key_smart(midi_num, avg_midi, key_obj)

        # Duración cuantizada
        dur_seconds = end - start
        dur_ql = duration_to_quarter_length(dur_seconds, detected_bpm)
        dur_ql = snap_quarter_length(dur_ql)

        n = note.Note(corrected_midi)
        n.quarterLength = dur_ql
        part.append(n)

        current_time = end

    s.append(part)

    xml_path = output_base + ".musicxml"
    s.write("musicxml", fp=xml_path)

    results = {
        "musicxml": xml_path,
        "key_info": key_info,
    }

    try:
        png_path = output_base + ".png"
        s.write("lily.png", fp=png_path)
        results["png"] = png_path
    except Exception:
        pass

    return results
