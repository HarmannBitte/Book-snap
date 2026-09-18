"""Stage 2 - the winning segmentation: stability-gated isolated-spike block-MAD.

Presentation videos are mostly *held still*; edits are isolated single-frame
spikes, while pans / fades / B-roll are sustained motion. So:

  1. stability gate : windowed median (default 2 s) of consecutive-frame
                      block-MAD < `stable_max`  ->  "static region"
  2. isolated spike : consecutive-frame block-MAD > `spike_thr` inside a
                      static region (adjacent spikes collapsed)  ->  cut

This beat pixel-diff, edge-diff, pHash, RGB-histogram and FFmpeg's
`select=gt(scene,0.2)` on our benchmark video (see examples/README.md).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def consecutive_mad(blocks: np.ndarray) -> np.ndarray:
    mad = np.zeros(len(blocks))
    mad[1:] = np.abs(blocks[1:] - blocks[:-1]).mean(axis=(1, 2))
    return mad


def segment(blocks: np.ndarray, fps: float, spike_thr: float = 1.5,
            stable_max: float = 1.0, window_s: float = 2.0):
    """Return (cuts, windowed_median_motion)."""
    n = len(blocks)
    mad = consecutive_mad(blocks)
    win = median_filter(mad, size=int(window_s * fps) | 1, mode="nearest")
    stable = win < stable_max
    idx = np.nonzero((mad > spike_thr) & stable)[0]
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])
    cuts = np.array([g[0] for g in groups], dtype=int)
    return cuts, win
