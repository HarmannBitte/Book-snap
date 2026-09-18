"""Selective super-resolution / enhancement for tiny on-screen text.

Whole-frame neural super-resolution is not viable here (CPU, ~1 GB RAM), and it
would be wasted anyway: the text we care about lives in a narrow shelf band.
So this module upscales *crops only*, and offers a handful of cheap, composable
enhancements. Callers run several presets and keep the highest-confidence OCR
read per string (see spines.extract_spines).

Presets (name -> operations applied after upscaling):
  lanczos   upscale only (baseline)
  unsharp   upscale + unsharp mask (default in spines; recovers stroke edges)
  clahe     upscale + CLAHE + unsharp (low-contrast spines)
  denoise   upscale + fastNlMeans + unsharp (compression ringing)
  binarize  upscale + adaptive threshold (faint light-on-dark spine type)

`stack_median` averages several frames of the same static shot (median, so a
walking person or a subtitle flash does not smear in) - for a locked-off
bookshelf this is the cheapest real resolution gain available.
"""
from __future__ import annotations

import cv2
import numpy as np

PRESETS = ("lanczos", "unsharp", "clahe", "denoise", "binarize")


def upscale(img, scale=4, interp=cv2.INTER_LANCZOS4):
    if scale == 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def unsharp(img, amount=1.0):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) * amount
    if amount != 1.0:  # keep the centre weight normalised
        kernel[1, 1] = 1 + 4 * amount
    return cv2.filter2D(img, -1, kernel)


def clahe(img, clip=2.5, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(img, h=6):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def binarize(img, block=25, c=11):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block, c)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


def enhance(crop, preset="unsharp", scale=4):
    """Upscale + preset. Returns a BGR image ready for OCR."""
    img = upscale(crop, scale)
    if preset == "lanczos":
        return img
    if preset == "unsharp":
        return unsharp(img)
    if preset == "clahe":
        return unsharp(clahe(img))
    if preset == "denoise":
        return unsharp(denoise(img))
    if preset == "binarize":
        return binarize(img)
    raise ValueError(f"unknown preset {preset!r} (choose from {PRESETS})")


def stack_median(images):
    """Per-pixel median across frames of the same (static) shot."""
    frames = [im for im in images if im is not None]
    if not frames:
        raise ValueError("no frames to stack")
    h = min(im.shape[0] for im in frames)
    w = min(im.shape[1] for im in frames)
    arr = np.stack([im[:h, :w].astype(np.uint8) for im in frames])
    return np.median(arr, axis=0).astype(np.uint8)
