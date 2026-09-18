"""Stage 3 - one full-resolution representative frame per segment.

The representative is the most *static* frame inside the segment (minimum
windowed motion), which avoids grabbing mid-fade or mid-pan frames.
"""
from __future__ import annotations

import os

import numpy as np

from .ffmpeg_utils import grab_fullres


def representative_times(cuts: np.ndarray, win: np.ndarray, fps: float, min_len: int = 2):
    bounds = list(cuts) + [len(win)]
    reps = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a, b = int(a), int(b)
        if b - a < min_len:
            continue
        reps.append(a + int(np.argmin(win[a:b])))
    return sorted(set(round(r / fps, 2) for r in reps))


def extract_frames(video: str, cuts: np.ndarray, win: np.ndarray, fps: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    times = representative_times(cuts, win, fps)
    for i, t in enumerate(times):
        grab_fullres(video, t, os.path.join(out_dir, f"seg_{i:03d}_t{t:07.2f}.png"))
    return times
