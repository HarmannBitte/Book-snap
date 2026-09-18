import numpy as np

from booksnap.segment import segment


def _blocks(seq):
    """seq: list of 24x14 frame levels (float) -> blocks array."""
    return np.array([np.full((14, 24), v, np.float32) for v in seq])


def test_hard_cut_detected():
    seq = [0.0] * 30 + [200.0] * 30          # static, cut, static
    cuts, win = segment(_blocks(seq), fps=10.0)
    assert list(cuts) == [30]


def test_slow_pan_not_a_cut():
    seq = [i * 0.4 for i in range(60)]       # gradual drift
    cuts, _ = segment(_blocks(seq), fps=10.0)
    assert list(cuts) == []


def test_sustained_motion_not_a_cut():
    seq = [0.0] * 20 + [100.0 if i % 2 else 0.0 for i in range(40)]  # B-roll flicker
    cuts, _ = segment(_blocks(seq), fps=10.0)
    assert list(cuts) == []


def test_localised_cover_pop_is_a_cut():
    base = np.zeros((14, 24), np.float32)
    cover = base.copy()
    cover[2:12, 14:22] = 180.0               # cover in one corner only
    blocks = np.array([base] * 25 + [cover] * 25)
    cuts, _ = segment(blocks, fps=10.0)
    assert list(cuts) == [25]
