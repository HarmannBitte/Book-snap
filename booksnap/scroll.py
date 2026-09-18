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

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .ffmpeg_utils import ffmpeg_bin, probe as probe_video
from .ocr import ocr_image


def _raw_frame(video: str, t: float, width: int, height: int):
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", video,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


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


def sharpness(img) -> float:
    """Variance of the Laplacian - motion-blurred frames score low."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def capture_scroll(video, ocr, win, fps, out_json, start=None, end=None,
                   step=0.5, probe_step=2.0, min_lines=12, stable_max=1.0,
                   max_window_s=180.0, n_cols=2, card_max_s=10.0, sharp_topk=3):
    info = probe_video(video)
    W, H = int(info["width"]), int(info["height"])
    if start is not None and end is not None:
        ranges = [(float(start), float(end))]
    else:
        ranges = []
        for a, b in moving_windows(win, fps, stable_max, max_window_s=max_window_s):
            nlines = 0
            t = a
            while t <= b:
                nlines = max(nlines, len(ocr_image(ocr, _raw_frame(video, t, W, H))))
                t += probe_step
            if nlines >= min_lines:
                ranges.append((a, b))
    frames = []
    for a, b in ranges:
        ts = np.arange(a, b + 1e-9, step)
        if (b - a) <= card_max_s:
            # animated title card: OCR only the sharpest frames (blur-resistant)
            scores = [(t, sharpness(_raw_frame(video, float(t), W, H))) for t in ts]
            ts = [t for t, _ in sorted(scores, key=lambda r: -r[1])[:sharp_topk]]
        for t in ts:
            blocks = ocr_image(ocr, _raw_frame(video, float(t), W, H))
            lines = [((b_["box"][1] + b_["box"][3]) / 2.0,
                      (b_["box"][0] + b_["box"][2]) / 2.0,
                      b_["text"], b_["conf"]) for b_ in blocks]
            frames.append((float(t), lines))
    ordered = stitch(frames, n_cols)
    json.dump(ordered, open(out_json, "w"), indent=1)
    return ordered, ranges
