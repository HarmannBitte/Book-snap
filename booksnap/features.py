"""Stage 1 - decode once, cache a compact per-frame feature stream.

Everything downstream (segmentation, benchmarking) reads this cache so the
video is decoded exactly one time.

Per proxy frame we store:
  gray   : G x G-aspect grayscale            (pixel diffs)
  blocks : B x B block-mean grayscale        (block-MAD, the winning signal)
  phash  : 64-bit perceptual hash            (hamming distance)
  hist   : 96-bin RGB histogram              (histogram intersection)
"""
from __future__ import annotations

import numpy as np
import cv2
import imagehash
from PIL import Image

from .ffmpeg_utils import frame_pipe

GW, GH = 96, 54      # grayscale probe size
BW, BH = 24, 14      # block grid for block-MAD


def build_features(video: str, out_npz: str, proxy_width: int = 480, fps: float = 10.0) -> int:
    grays, blocks, phashes, hists = [], [], [], []
    for _, rgb in frame_pipe(video, proxy_width, fps):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.resize(gray, (GW, GH), interpolation=cv2.INTER_AREA))
        blocks.append(
            cv2.resize(gray, (BW, BH), interpolation=cv2.INTER_AREA).astype(np.float32)
        )
        ph = imagehash.phash(Image.fromarray(rgb), hash_size=8)
        phashes.append(np.packbits(np.asarray(ph.hash).flatten().astype(np.uint8)))
        h = []
        for c in range(3):
            hh, _ = np.histogram(rgb[:, :, c], bins=32, range=(0, 256))
            h.append(hh)
        h = np.concatenate(h).astype(np.float64)
        hists.append(h / (h.sum() + 1e-9))

    np.savez_compressed(
        out_npz,
        gray=np.array(grays, np.uint8),
        blocks=np.array(blocks, np.float32),
        phash=np.array(phashes, np.uint8),
        hist=np.array(hists, np.float64),
        fps=np.array([fps]),
    )
    return len(grays)
