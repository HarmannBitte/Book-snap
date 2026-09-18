"""Stage 4 - OCR (RapidOCR / PaddleOCR-ONNX) over images -> text blocks."""
from __future__ import annotations

import glob
import json
import os

import cv2
from rapidocr_onnxruntime import RapidOCR


def ocr_image(ocr: RapidOCR, img):
    res, _ = ocr(img)
    blocks = []
    if res:
        for box, text, conf in res:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            blocks.append(dict(box=[min(xs), min(ys), max(xs), max(ys)],
                               text=text, conf=round(float(conf), 3)))
    return blocks


def ocr_directory(src_dir: str, out_json: str, pattern: str = "seg_*.png", verbose: bool = True):
    ocr = RapidOCR()
    paths = sorted(glob.glob(os.path.join(src_dir, pattern)))
    results = {}
    for i, path in enumerate(paths):
        results[os.path.basename(path)] = ocr_image(ocr, cv2.imread(path))
        if verbose and i % 25 == 0:
            print(f"ocr {i}/{len(paths)} {os.path.basename(path)}", flush=True)
    json.dump(results, open(out_json, "w"), indent=1)
    return results
