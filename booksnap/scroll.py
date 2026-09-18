"""Stage 6 - recover scrolling credit / bibliography text.

Scrolling lists move, so the stability gate in `segment` correctly ignores
them - they must be captured separately:

  auto mode  : candidate windows = contiguous *moving* regions (windowed
               median motion >= stable_max) that are short enough to be a
               credit roll (<= max_window_s). Each window is probed with OCR;
               text-dense windows are densely sampled.
  range mode : `--start/--end` dense-sample a known range.

Frames are OCR'd and unique lines stitched in reading order. Because credit
rolls are frequently laid out in *columns* that scroll together, lines are
first assigned to a column by x-centre and each column is stitched
independently; the output concatenates columns left-to-right. This keeps
citation entries from interleaving (which corrupts entry boundaries).
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
from scipy.ndimage import median_filter

from .ffmpeg_utils import ffmpeg_bin
from .ocr import ocr_image

FRAME_W = 1920  # raw frames are grabbed at full resolution


def _raw_frame(video: str, t: float, height: int = 1080):
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", video,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, FRAME_W, 3)


def moving_windows(win, fps, stable_max=1.0, min_s=3.0, max_window_s=180.0):
    moving = win >= stable_max
    windows, i, n = [], 0, len(moving)
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            dur = (j - i) / fps
            if min_s <= dur <= max_window_s:
                windows.append((i / fps, (j - 1) / fps))
            i = j
        else:
            i += 1
    return windows


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _columns(lines, n_cols=2):
    """Assign (y, x, text, conf) lines to n_cols columns by x-centre."""
    if not lines:
        return [[] for _ in range(n_cols)]
    xs = sorted(l[1] for l in lines)
    edges = [xs[int((k + 1) * len(xs) / (n_cols + 1))] for k in range(n_cols - 1)]
    cols = [[] for _ in range(n_cols)]
    for l in lines:
        c = sum(1 for e in edges if l[1] > e)
        cols[min(c, n_cols - 1)].append(l)
    return cols


def stitch_column(lines):
    """Lines arrive in temporal order (frames) and top-to-bottom within a
    frame, which is exactly document order for an upward scroll - so do NOT
    re-sort by y across frames."""
    seen, out = set(), []
    for y, x, text, conf in lines:
        key = _norm(text)
        if len(key) < 6 or key in seen:
            continue
        seen.add(key)
        out.append(dict(y=round(y, 1), text=text, conf=round(float(conf), 2)))
    return out


def stitch(frames, n_cols=2):
    """frames: list of (t, [(y, x, text, conf), ...]); returns column-major lines."""
    per_col = [[] for _ in range(n_cols)]
    for _, lines in frames:
        lines = sorted(lines, key=lambda l: l[0])
        for c, col in enumerate(_columns(lines, n_cols)):
            per_col[c].extend(col)
    ordered, seen = [], set()
    for c in range(n_cols):
        for e in stitch_column(per_col[c]):
            k = _norm(e["text"])
            if k not in seen:
                seen.add(k)
                ordered.append(dict(column=c, **e))
    return ordered


def capture_scroll(video, ocr, win, fps, out_json, start=None, end=None,
                   step=0.5, probe_step=2.0, min_lines=12, stable_max=1.0,
                   max_window_s=180.0, n_cols=2):
    if start is not None and end is not None:
        ranges = [(float(start), float(end))]
    else:
        ranges = []
        for a, b in moving_windows(win, fps, stable_max, max_window_s=max_window_s):
            probe = 0
            t = a
            while t <= b:
                probe = max(probe, len(ocr_image(ocr, _raw_frame(video, t))))
                t += probe_step
            if probe >= min_lines:
                ranges.append((a, b))
    frames = []
    for a, b in ranges:
        t = a
        while t <= b:
            blocks = ocr_image(ocr, _raw_frame(video, t))
            lines = [((b_["box"][1] + b_["box"][3]) / 2.0,
                      (b_["box"][0] + b_["box"][2]) / 2.0,
                      b_["text"], b_["conf"]) for b_ in blocks]
            frames.append((t, lines))
            t += step
    ordered = stitch(frames, n_cols)
    json.dump(ordered, open(out_json, "w"), indent=1)
    return ordered, ranges
