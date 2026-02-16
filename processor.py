"""
Módulo de procesamiento de audio: detección de pitch, generación MIDI y partitura.
Compartido entre el servidor local (app.py) y las serverless functions de Vercel.
"""

import os
import numpy as np
import librosa
import pretty_midi
from scipy.ndimage import median_filter
from music21 import (
    stream, note, meter, tempo, metadata,
    key as m21key, pitch as m21pitch, analysis,
)
from music21.note import Rest


# ---------------------------------------------------------------------------
# Utilidades de conversión mejoradas
# ---------------------------------------------------------------------------

def hz_to_midi_note(freq: float) -> int:
    """Convierte frecuencia en Hz a número de nota MIDI (0-127)."""
    if freq <= 0:
        return 0
    midi_val = 69 + 12 * np.log2(freq / 440.0)
    return int(np.clip(round(midi_val), 0, 127))


def smooth_pitch_sequence(f0: np.ndarray, voiced_flag: np.ndarray,
                          kernel_size: int = 5) -> np.ndarray:
    """
    Aplica un filtro de mediana a la secuencia de pitch para eliminar
    saltos bruscos y glitches de detección.
    """
    f0_clean = f0.copy()
    voiced_mask = voiced_flag & ~np.isnan(f0)

    if np.sum(voiced_mask) < kernel_size:
        return f0_clean

    voiced_freqs = f0_clean[voiced_mask]
    smoothed = median_filter(voiced_freqs, size=kernel_size)
    f0_clean[voiced_mask] = smoothed

    return f0_clean


def fix_octave_jumps(notes_list: list, max_jump_semitones: int = 11) -> list:
    """
    Corrige saltos de octava erróneos. Si una nota salta más de
    max_jump_semitones respecto a sus vecinas, se ajusta a la octava
    más cercana.
    """
    if len(notes_list) < 3:
        return notes_list

    corrected = list(notes_list)

    for i in range(1, len(corrected) - 1):
        midi_prev = corrected[i - 1][0]
        midi_curr = corrected[i][0]
        midi_next = corrected[i + 1][0]

        neighbor_avg = (midi_prev + midi_next) / 2.0
        diff = midi_curr - neighbor_avg

        if abs(diff) > max_jump_semitones:
            octave_shift = round(diff / 12) * 12
            new_midi = midi_curr - octave_shift
            new_midi = int(np.clip(new_midi, 0, 127))
            corrected[i] = (new_midi, corrected[i][1], corrected[i][2])

    return corrected


def merge_repeated_notes(notes_list: list, min_gap: float = 0.05) -> list:
    """
    Fusiona notas consecutivas idénticas separadas por un silencio
    muy corto (< min_gap segundos), que probablemente son la misma nota.
    """
    if len(notes_list) < 2:
        return notes_list

    merged = [notes_list[0]]

    for midi_num, start, end in notes_list[1:]:
        prev_midi, prev_start, prev_end = merged[-1]
        if midi_num == prev_midi and (start - prev_end) < min_gap:
            merged[-1] = (prev_midi, prev_start, end)
        else:
            merged.append((midi_num, start, end))

    return merged


def detect_tempo_from_audio(y: np.ndarray, sr: int) -> float:
    """
    Detecta el BPM del audio usando librosa.beat.beat_track.
    Devuelve el tempo redondeado a valores musicales comunes.
    """
    tempo_estimate, _ = librosa.beat.beat_track(y=y, sr=sr)

    if hasattr(tempo_estimate, '__len__'):
        bpm = float(tempo_estimate[0])
    else:
        bpm = float(tempo_estimate)

    common_tempos = [60, 66, 72, 76, 80, 84, 88, 92, 96, 100,
                     104, 108, 112, 116, 120, 126, 132, 138, 144, 152, 160]

    closest = min(common_tempos, key=lambda t: abs(t - bpm))
    return closest


def quantize_duration(dur_quarters: float, grid: float = 0.25) -> float:
    """
    Cuantiza una duración en quarter-notes a la rejilla más cercana.
    grid=0.25 → semicorchea, grid=0.5 → corchea, etc.
    Permite duraciones con puntillo (1.5x).
    """
    standard_durations = []
    base = grid
    while base <= 8.0:
        standard_durations.append(base)
        standard_durations.append(base * 1.5)
        base *= 2

    standard_durations = sorted(set(standard_durations))

    closest = min(standard_durations, key=lambda d: abs(d - dur_quarters))
    return closest


def audio_to_midi(audio_path: str, midi_path: str) -> dict:
    """
    Pipeline mejorado: Audio → detección de pitch → suavizado →
    corrección de octava → fusión → generación MIDI.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    detected_bpm = detect_tempo_from_audio(y, sr)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    hop_length = 512
    frame_duration = hop_length / sr

    f0 = smooth_pitch_sequence(f0, voiced_flag, kernel_size=5)

    notes_list = []
    current_midi = None
    start_time = 0.0

    for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
        t = i * frame_duration
        if is_voiced and not np.isnan(freq):
            midi_num = hz_to_midi_note(freq)
            if midi_num != current_midi:
                if current_midi is not None:
                    notes_list.append((current_midi, start_time, t))
                current_midi = midi_num
                start_time = t
        else:
            if current_midi is not None:
                notes_list.append((current_midi, start_time, t))
                current_midi = None

    if current_midi is not None:
        notes_list.append((current_midi, start_time, len(f0) * frame_duration))

    min_duration = 0.06
    notes_list = [(m, s, e) for m, s, e in notes_list if (e - s) >= min_duration]
    notes_list = fix_octave_jumps(notes_list)
    notes_list = merge_repeated_notes(notes_list)
    notes_list = [(m, s, e) for m, s, e in notes_list if (e - s) >= min_duration]

    midi = pretty_midi.PrettyMIDI(initial_tempo=detected_bpm)
    voice = pretty_midi.Instrument(program=0, name="Voice")

    for midi_num, start, end in notes_list:
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


def detect_key_from_notes(notes_list: list) -> dict:
    """
    Usa music21 para analizar las notas y detectar la tonalidad
    más probable de la melodía.
    """
    if not notes_list:
        return {"key": "C major", "key_name": "Do Mayor", "confidence": 0}

    s = stream.Stream()
    for midi_num, start, end in notes_list:
        n = note.Note(midi_num)
        n.quarterLength = 1.0
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


def snap_to_key(midi_num: int, key_obj) -> int:
    """
    Si una nota no pertenece a la tonalidad detectada, la ajusta
    al grado más cercano de la escala.
    """
    try:
        scale_pitches = key_obj.getScale().pitches
        scale_pcs = [p.pitchClass for p in scale_pitches]

        pc = midi_num % 12
        if pc in scale_pcs:
            return midi_num

        distances = []
        for spc in scale_pcs:
            d = min(abs(pc - spc), 12 - abs(pc - spc))
            distances.append((d, spc))

        distances.sort(key=lambda x: x[0])
        closest_pc = distances[0][1]

        diff = closest_pc - pc
        if abs(diff) > 6:
            diff = diff - 12 if diff > 0 else diff + 12

        return int(np.clip(midi_num + diff, 0, 127))
    except Exception:
        return midi_num


def midi_to_score(midi_path: str, output_base: str,
                  detected_bpm: float, notes_list: list) -> dict:
    """
    Lee un archivo MIDI y genera una partitura con:
    - Tonalidad detectada automáticamente
    - Cuantización rítmica inteligente
    - Silencios explícitos entre notas
    - Armadura de clave correcta
    """
    key_info = detect_key_from_notes(notes_list)

    try:
        key_obj = m21key.Key(key_info["key"].split()[0],
                             key_info["key"].split()[1] if len(key_info["key"].split()) > 1 else "major")
    except Exception:
        key_obj = m21key.Key("C", "major")

    beats_per_second = detected_bpm / 60.0

    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = "Transcripción de Voz"
    s.metadata.composer = "Partituras AI"

    part = stream.Part()
    part.append(key_obj)
    part.append(meter.TimeSignature("4/4"))
    part.append(tempo.MetronomeMark(number=detected_bpm))

    current_time = 0.0

    for midi_num, start, end in notes_list:
        gap = start - current_time
        if gap > 0.1:
            rest_quarters = gap * beats_per_second
            rest_quantized = quantize_duration(rest_quarters)
            if rest_quantized >= 0.25:
                r = Rest()
                r.quarterLength = rest_quantized
                part.append(r)

        corrected_midi = snap_to_key(midi_num, key_obj)

        dur_seconds = end - start
        dur_quarters = dur_seconds * beats_per_second
        dur_quantized = quantize_duration(dur_quarters)

        n = note.Note(corrected_midi)
        n.quarterLength = dur_quantized
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
