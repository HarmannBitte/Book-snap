"""Stage 10 - book-spine extraction from shelf/backdrop footage.

Physical books on a set shelf are a different modality: tiny, angled,
partially occluded type at video resolution. Handling that works:

  1. pick the sharpest frames in the requested time range (Laplacian var),
  2. crop the shelf region (--roi x,y,w:h or --band = top fraction of frame),
  3. upscale (Lanczos) + unsharp-mask,
  4. OCR in horizontal tiles (very wide crops break text detectors),
  5. dedupe across frames keeping the highest-confidence reading.
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np

from .ffmpeg_utils import ffmpeg_bin, probe
from .scroll import sharpness


def _grab(video, t, w, h):
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", video,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def extract_spines(video, out_json, times=None, start=None, end=None, step=1.0,
                   topk=3, roi=None, band=0.22, upscale=4, tile_w=1400, ocr=None):
    from .ocr import RapidOCR, ocr_image
    ocr = ocr or RapidOCR()
    info = probe(video)
    W, H = int(info["width"]), int(info["height"])
    if times is None:
        times = np.arange(float(start), float(end) + 1e-9, float(step))
    scored = sorted(((float(t), sharpness(_grab(video, float(t), W, H))) for t in times),
                    key=lambda r: -r[1])[:topk]
    reads = {}
    for t, _ in scored:
        img = _grab(video, t, W, H)
        if roi:
            x, y, w, h = roi
            crop = img[y:y + h, x:x + w]
        else:
            crop = img[0:int(H * band), :]
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_LANCZOS4)
        crop = cv2.filter2D(crop, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
        ch, cw = crop.shape[:2]
        for i in range(0, cw, tile_w):
            tile = crop[:, i:i + tile_w]
            res = ocr_image(ocr, tile)
            for b in res:
                key = _norm(b["text"])
                if len(key) < 4:
                    continue
                if key not in reads or b["conf"] > reads[key]["conf"]:
                    reads[key] = dict(text=b["text"], conf=b["conf"], t=t,
                                      x=int(b["box"][0] + i), y=int(b["box"][1]))
    out = sorted(reads.values(), key=lambda r: -r["conf"])
    json.dump(out, open(out_json, "w"), indent=1)
    return out
