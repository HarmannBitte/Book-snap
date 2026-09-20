"""Stage 9 - audio channel: ASR the soundtrack (faster-whisper, CPU-friendly).

For talk/podcast videos most books are *mentioned verbally*, which a purely
visual pipeline can never see. This stage produces a timestamped transcript
that `compile` fuses with the visual candidates.

Memory note: faster-whisper decodes the whole input plus VAD probabilities as
float32, which OOMs small boxes on long audio - so the input is transcribed
in `chunk_s` pieces (offsets restored afterwards).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from .ffmpeg_utils import ffmpeg_bin, probe


def transcribe(video: str, out_json: str, model_size: str = "tiny",
               device: str = "cpu", compute_type: str = "int8",
               vad_filter: bool = True, chunk_s: float = 300.0,
               beam_size: int = 1):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    total = float(probe(video).get("duration") or 0.0)
    ranges = [(s, min(chunk_s, total - s)) for s in
              np_arange(0.0, total, chunk_s)] if total else [(0.0, 0.0)]
    out = []
    for start, dur in ranges:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
        subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{start}",
                        "-t", f"{dur}", "-i", video, "-ac", "1", "-ar", "16000",
                        tmp, "-y"], check=True)
        segs, info = model.transcribe(tmp, vad_filter=vad_filter,
                                      beam_size=beam_size,
                                      condition_on_previous_text=False)
        for s in segs:
            out.append(dict(start=round(start + float(s.start), 1),
                            end=round(start + float(s.end), 1),
                            text=s.text.strip()))
        os.unlink(tmp)
    json.dump(dict(language=None, segments=out), open(out_json, "w"), indent=1)
    return out


def np_arange(a, b, step):
    out, x = [], a
    while x < b - 1e-9:
        out.append(x)
        x += step
    return out or [a]
