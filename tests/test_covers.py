import numpy as np

from booksnap.covers import attach_panel_text, detect_covers


def _write(tmp_path, name, img):
    import cv2
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def test_bright_panel_found_text_column_rejected(tmp_path):
    img = np.zeros((600, 800, 3), np.uint8)
    img[80:520, 500:760] = (200, 190, 180)          # solid bright panel
    img[100:500:12, 60:300] = 255                    # sparse text lines
    _write(tmp_path, "seg_000_t0000.00.png", img)
    m = detect_covers(str(tmp_path), str(tmp_path / "out"))
    assert len(m) == 1
    assert 500 <= m[0]["x"] <= 520 and m[0]["ar"] < 1.0


def test_dark_jacket_found(tmp_path):
    img = np.zeros((600, 800, 3), np.uint8)
    img[40:560, 480:760] = (26, 26, 26)              # near-black jacket on black bg
    img[300, 500:740] = (230, 230, 230)              # one light title line
    _write(tmp_path, "seg_001_t0010.00.png", img)
    m = detect_covers(str(tmp_path), str(tmp_path / "out"))
    assert len(m) == 1 and m[0]["x"] >= 470


def test_attach_panel_text_picks_caption_band():
    man = [dict(seg="seg_002_t0020.00", file="f.png", x=100, y=50, w=200, h=300,
                frame_frac=0.2, ar=0.67)]
    ocr = {"seg_002_t0020.00.png": [
        dict(box=[110, 360, 290, 380], text="AUTHOR NAME", conf=0.9),
        dict(box=[120, 390, 280, 410], text="Some Book Title", conf=0.9),
        dict(box=[10, 10, 90, 30], text="unrelated slide heading", conf=0.9),
    ]}
    attach_panel_text(man, ocr)
    assert "Some Book Title" in man[0]["text"]
    assert "unrelated" not in man[0]["text"]
