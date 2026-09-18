"""Stage 5 - locate embedded image panels (book covers / figures) in slides.

Slide backgrounds are flat (usually near-black); body text is sparse strokes.
Image panels - bright photos *and* dark jackets - are solid rectangles whose
pixels differ from the background colour. So:

  mask     : |V - background| > `delta`, background = median V of the frame
  panels   : connected components whose bounding-box fill density
             (mask pixels / bbox area) > `min_density`; solid panels score
             ~1.0 while text columns score ~0.1-0.3 and are rejected

Connected components of book-plausible size are cropped out. Side-by-side
covers merge into one wide component; `split_wide` halves components whose
aspect ratio exceeds `wide_ar`.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

B = 16  # analysis block size in pixels


def _panels(img: np.ndarray, delta: float = 10.0, close: int = 15):
    """(component mask, raw pixel mask) of 'differs from flat background'.

    The raw mask keeps text as sparse strokes (low fill density) while solid
    panels - bright photos and dark jackets alike - are dense. The closed mask
    merely merges strokes into blobs so connected components yield bboxes.
    """
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    bg = float(np.median(v))
    raw = (np.abs(v - bg) > delta).astype(np.uint8)
    k = np.ones((close, close), np.uint8)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return closed, raw


def _density(raw: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    sub = raw[y:y + h, x:x + w]
    return float(sub.mean()) if sub.size else 0.0


def detect_covers(src_dir: str, out_dir: str, pattern: str = "seg_*.png",
                  min_frame_frac=0.012, max_frame_frac=0.85, min_side=120,
                  wide_ar=1.15, split_wide=True, delta=10.0, min_density=0.6):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for path in sorted(glob.glob(os.path.join(src_dir, pattern))):
        img = cv2.imread(path)
        H, W = img.shape[:2]
        closed, raw = _panels(img, delta)
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        seg = os.path.basename(path)[:-4]
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            if _density(raw, x, y, bw, bh) < min_density:
                continue  # sparse text columns / titles, not solid panels
            px, py, pw, ph = int(x), int(y), int(bw), int(bh)
            ff = (pw * ph) / (W * H)
            if not (min_frame_frac <= ff <= max_frame_frac) or min(pw, ph) < min_side:
                continue
            boxes = [(px, py, pw, ph)]
            ar = pw / ph
            if split_wide and ar > wide_ar:  # likely two covers side by side
                boxes = [(px, py, pw // 2, ph), (px + pw // 2, py, pw - pw // 2, ph)]
            for (bx, by, bw2, bh2) in boxes:
                crop = img[by:by + bh2, bx:bx + bw2]
                fn = f"{seg}_r{len(manifest):03d}.png"
                cv2.imwrite(os.path.join(out_dir, fn), crop)
                manifest.append(dict(seg=seg, file=fn, x=int(bx), y=int(by),
                                     w=int(bw2), h=int(bh2),
                                     frame_frac=round(float((bw2 * bh2) / (W * H)), 3),
                                     ar=round(float(bw2 / bh2), 2)))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return manifest
