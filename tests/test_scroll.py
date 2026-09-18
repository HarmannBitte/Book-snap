from booksnap.scroll import stitch, moving_windows
import numpy as np


def test_stitch_dedupes_overlapping_scroll():
    frames = [
        (0.0, [(10, 100, "alpha line one", 0.9), (20, 100, "beta line two", 0.9)]),
        (0.5, [(5, 100, "beta line two", 0.9), (15, 100, "gamma line three", 0.9)]),
    ]
    out = stitch(frames, n_cols=1)
    assert [e["text"] for e in out] == ["alpha line one", "beta line two",
                                        "gamma line three"]


def test_stitch_keeps_columns_separate():
    # two columns scrolling together: left entries must not interleave right ones
    frames = [
        (0.0, [(10, 200, "left one", 0.9), (10, 1200, "right one", 0.9)]),
        (0.5, [(20, 200, "left two", 0.9), (20, 1200, "right two", 0.9)]),
    ]
    out = stitch(frames, n_cols=2)
    left = [e["text"] for e in out if e["column"] == 0]
    right = [e["text"] for e in out if e["column"] == 1]
    assert left == ["left one", "left two"]
    assert right == ["right one", "right two"]


def test_moving_windows_excludes_long_talking_head():
    win = np.array([0.0] * 50 + [5.0] * 2000 + [0.0] * 50)
    w = moving_windows(win, fps=10.0, max_window_s=180.0)
    assert w == []
    w = moving_windows(win, fps=10.0, max_window_s=1000.0)
    assert len(w) == 1
