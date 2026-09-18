"""Stage 6 - recover scrolling credit / bibliography text.

Scrolling lists move, so the stability gate in `segment` correctly ignores
them - they must be captured separately:

  auto mode  : candidate windows = contiguous *moving* regions (windowed
               median motion >= stable_max). Each window is probed with OCR at
               a low rate; windows whose probe yields >= `min_lines` text lines
               are treated as scrolling-text and densely sampled.
  range mode : `--start/--end` dense-sample a known range.

Frames are OCR'd and unique lines stitched in reading order (the scroll
overlaps between consecutive samples, so dedupe by normalised text).
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import cv2
from scipy.ndimage import median_filter

from .ffmpeg_utils import ffmpeg_bin
from .ocr import ocr_image


def _raw_frame(video: str, t: float):
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", video,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)


def moving_windows(win: np.ndarray, fps: float, stable_max: float = 1.0, min_s: float = 3.0):
    moving = win >= stable_max
    windows, i, n = [], 0, len(moving)
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if (j - i) / fps >= min_s:
                windows.append((i / fps, (j - 1) / fps))
            i = j
        else:
            i += 1
    return windows


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def stitch(frames):
    seen, ordered = set(), []
    for t, lines in frames:
        for y, text, conf in lines:
            key = _norm(text)
            if len(key) < 8 or key in seen:
                continue
            seen.add(key)
            ordered.append(dict(t=round(t, 1), y=round(y, 1), text=text,
                                conf=round(float(conf), 2)))
    return ordered


def capture_scroll(video, ocr, win, fps, out_json, start=None, end=None,
                   step=0.5, probe_step=2.0, min_lines=12, stable_max=1.0):
    if start is not None and end is not None:
        ranges = [(float(start), float(end))]
    else:
        ranges = []
        for a, b in moving_windows(win, fps, stable_max):
            probe_lines = 0
            t = a
            while t <= b:
                blocks = ocr_image(ocr, _raw_frame(video, t))
                probe_lines = max(probe_lines, len(blocks))
                t += probe_step
            if probe_lines >= min_lines:
                ranges.append((a, b))
    frames = []
    for a, b in ranges:
        t = a
        while t <= b:
            blocks = ocr_image(ocr, _raw_frame(video, t))
            lines = sorted(
                (((b["box"][1] + b["box"][3]) / 2.0, b["text"], b["conf"]) for b in blocks),
                key=lambda r: r[0],
            )
            frames.append((t, lines))
            t += step
    ordered = stitch(frames)
    json.dump(ordered, open(out_json, "w"), indent=1)
    return ordered, ranges
