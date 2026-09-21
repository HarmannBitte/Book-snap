"""Stage 10 - book-spine extraction from shelf/backdrop footage.

Physical books on a set shelf are a different modality: tiny, angled,
partially occluded type at video resolution. Handling that works:

  1. pick the sharpest frames in the requested time range (Laplacian var),
  2. crop the shelf region (--roi x,y,w:h or --band = top fraction of frame),
  3. optional multi-frame median stack (locked-off camera -> free denoise),
  4. selective super-resolution: run several enhancement presets
     (see superres.PRESETS) over the crop,
  5. OCR in horizontal tiles (very wide crops break text detectors),
  6. dedupe across frames/presets keeping the highest-confidence reading.

Each read records which preset produced it, so the report can say whether a
spine needed binarisation or was legible in the baseline.
"""
from __future__ import annotations

import json
import subprocess

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


def _crop(img, H, roi, band):
    if roi:
        x, y, w, h = roi
        return img[y:y + h, x:x + w]
    return img[0:int(H * band), :]


def extract_spines(video, out_json, times=None, start=None, end=None, step=1.0,
                   topk=3, roi=None, band=0.22, upscale=4, tile_w=1400, ocr=None,
                   presets=("unsharp",), stack=False, min_len=3,
                   panorama=False, min_focus=0.0, vertical_pass=False):
    from .ocr import RapidOCR, ocr_image
    from .superres import enhance, stack_median, stitch_pan
    ocr = ocr or RapidOCR()
    if str(video).lower().endswith((".png", ".jpg", ".jpeg")):
        # still-image mode: benchmarks and one-off shelf photos need no video
        import cv2
        img = cv2.imread(str(video))
        if img is None:
            raise ValueError(f"cannot read image {video}")
        crop = _crop(img, img.shape[0], roi, band)
        crops = [(0.0, crop)]
    else:
        info = probe(video)
        W, H = int(info["width"]), int(info["height"])
        if times is None:
            times = np.arange(float(start), float(end) + 1e-9, float(step))
        time_crops = []
        for t in times:
            frame = _grab(video, float(t), W, H)
            cr = _crop(frame, H, roi, band)
            # Evaluate focus on the shelf crop itself, not the whole frame
            # (which is often dominated by foreground actors/clothing bokeh)
            s = sharpness(cr)
            time_crops.append((float(t), s, cr))
        time_crops.sort(key=lambda r: -r[1])
        if min_focus > 0.0:
            kept = [tc for tc in time_crops if tc[1] >= min_focus]
            if not kept:
                import sys
                print(f"warning: shelf crop sharpness ({time_crops[0][1]:.1f} max) below min-focus ({min_focus}); "
                      "skipping out-of-focus footage", file=sys.stderr)
            time_crops = kept
        crops = [(t, cr) for t, _, cr in time_crops[:topk]]
    variants = []
    if not crops:
        json.dump([], open(out_json, "w"), indent=1)
        return []
    if stack and len(crops) > 1:
        variants.append((crops[0][0], "stack",
                         stack_median([c for _, c in crops])))
    if panorama and len(crops) > 2:
        # a pan covers more shelf than any single frame: stitch it, then enhance
        pan = stitch_pan([c for _, c in crops])
        if pan is not None:
            variants.append((crops[0][0], "panorama", enhance(pan, "unsharp", upscale)))
    for t, crop in crops:
        for p in presets:
            variants.append((t, p, enhance(crop, p, upscale)))

    reads = {}
    for t, preset, img in variants:
        ch, cw = img.shape[:2]
        for i in range(0, cw, tile_w):
            tile = img[:, i:i + tile_w]
            for b in ocr_image(ocr, tile):
                key = _norm(b["text"])
                if len(key) < min_len:
                    continue
                if key not in reads or b["conf"] > reads[key]["conf"]:
                    reads[key] = dict(text=b["text"], conf=b["conf"], t=t,
                                      preset=preset, upscale=upscale,
                                      x=int(b["box"][0] + i), y=int(b["box"][1]),
                                      w=int(b["box"][2] - b["box"][0]),
                                      h=int(b["box"][3] - b["box"][1]))

        if vertical_pass:
            import cv2
            # 1. Rotated 90 deg CCW: English top-to-bottom vertical spine titles
            rot_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rch, rcw = rot_ccw.shape[:2]
            for j in range(0, rcw, tile_w):
                tile = rot_ccw[:, j:j + tile_w]
                for b in ocr_image(ocr, tile):
                    key = _norm(b["text"])
                    if len(key) < min_len:
                        continue
                    rx0, ry0 = b["box"][0] + j, b["box"][1]
                    rx1, ry1 = b["box"][2] + j, b["box"][3]
                    orig_x0 = cw - 1 - ry1
                    orig_y0 = rx0
                    orig_w = ry1 - ry0
                    orig_h = rx1 - rx0
                    if key not in reads or b["conf"] > reads[key]["conf"]:
                        reads[key] = dict(text=b["text"], conf=b["conf"], t=t,
                                          preset=f"{preset}_rotccw", upscale=upscale,
                                          x=int(orig_x0), y=int(orig_y0),
                                          w=int(orig_w), h=int(orig_h))

            # 2. Rotated 90 deg CW: Continental European bottom-to-top vertical spine titles
            rot_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rch, rcw = rot_cw.shape[:2]
            for j in range(0, rcw, tile_w):
                tile = rot_cw[:, j:j + tile_w]
                for b in ocr_image(ocr, tile):
                    key = _norm(b["text"])
                    if len(key) < min_len:
                        continue
                    rx0, ry0 = b["box"][0] + j, b["box"][1]
                    rx1, ry1 = b["box"][2] + j, b["box"][3]
                    orig_x0 = ry0
                    orig_y0 = ch - 1 - rx1
                    orig_w = ry1 - ry0
                    orig_h = rx1 - rx0
                    if key not in reads or b["conf"] > reads[key]["conf"]:
                        reads[key] = dict(text=b["text"], conf=b["conf"], t=t,
                                          preset=f"{preset}_rotcw", upscale=upscale,
                                          x=int(orig_x0), y=int(orig_y0),
                                          w=int(orig_w), h=int(orig_h))
    out = sorted(reads.values(), key=lambda r: -r["conf"])
    json.dump(out, open(out_json, "w"), indent=1)
    return out
