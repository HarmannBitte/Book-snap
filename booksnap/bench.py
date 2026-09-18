"""Development harness: benchmark detector families on a feature cache.

Compares naive "compare to last kept frame" segmenters (pixel-diff, block-MAD,
pHash, RGB-histogram, edge-diff) and scores cut precision/recall against a
consecutive-frame motion-spike oracle. Used to produce examples/README.md.
"""
from __future__ import annotations

import itertools

import numpy as np
import cv2
from scipy.ndimage import median_filter

from .segment import consecutive_mad


def naive_segments(blocks, gray, phash, hist, mode, thr, hold):
    n = len(blocks)
    gf = gray.astype(np.float32)
    cuts, rep, run = [0], 0, 0
    for i in range(1, n):
        if mode == "pixdiff":
            v = (np.abs(gf[i] - gf[rep]) > 25).mean() * 100
        elif mode == "blockmad":
            v = float(np.abs(blocks[i] - blocks[rep]).mean())
        elif mode == "phash":
            v = int((np.unpackbits(phash[i]) != np.unpackbits(phash[rep])).sum())
        else:
            v = 1.0 - float(np.minimum(hist[i], hist[rep]).sum())
        if v > thr:
            run += 1
            if run >= hold:
                cuts.append(i); rep = i; run = 0
        else:
            run = 0
    return np.array(cuts)


def motion_spikes(sig, k=8.0, gap=2):
    med = np.median(sig[1:])
    dev = np.median(np.abs(sig[1:] - med)) + 1e-9
    idx = np.nonzero(sig > med + k * 1.4826 * dev)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return np.array([g[int(np.argmax(sig[g]))] for g in groups])


def score(cuts, gt, tol):
    cuts = np.asarray(cuts)
    if len(cuts) == 0:
        return 0, 0.0, 0.0, 0.0
    tp = sum(1 for c in cuts if np.any(np.abs(gt - c) <= tol))
    prec = tp / len(cuts)
    rec = sum(1 for g in gt if np.any(np.abs(cuts - g) <= tol)) / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return len(cuts), prec, rec, f1


def run_bench(feat_npz, spike_thr=1.5, stable_max=1.0):
    d = np.load(feat_npz)
    blocks, gray, phash, hist = d["blocks"], d["gray"], d["phash"], d["hist"]
    fps = float(d["fps"][0])
    mad = consecutive_mad(blocks)
    gt = motion_spikes(mad)
    tol = int(round(0.25 * fps))
    rows = []
    grids = {"pixdiff": [5, 12, 20], "blockmad": [5, 8, 12],
             "phash": [8, 14, 22], "histrgb": [0.05, 0.1, 0.2]}
    for mode, thrs in grids.items():
        for thr, hold in itertools.product(thrs, [1, 3, 5]):
            cuts = naive_segments(blocks, gray, phash, hist, mode, thr, hold)
            n, p, r, f1 = score(cuts, gt, tol)
            rows.append(dict(mode=mode, thr=thr, hold=hold, segs=n + 1,
                             prec=round(p, 3), rec=round(r, 3), f1=round(f1, 3)))
    from .segment import segment
    cuts, win = segment(blocks, fps, spike_thr, stable_max)
    n, p, r, f1 = score(cuts, gt, tol)
    rows.append(dict(mode="STABILITY-GATED", thr=spike_thr, hold="-", segs=len(cuts) + 1,
                     prec=round(p, 3), rec=round(r, 3), f1=round(f1, 3)))
    rows.sort(key=lambda x: -x["f1"])
    return rows, len(gt)
