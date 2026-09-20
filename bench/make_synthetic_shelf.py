"""Round 8: synthetic held-out shelf for offline, CI-runnable scoring.

The real labelled shelf (bench/spine_labels_v2.json) is one video's backdrop,
and the vision channel's 1.0 recall on it is circular. A rendered shelf gives
a second, fully known ground truth that needs no network and no download:

  * titles are drawn as stacked words on standing spines (the layout the OCR
    path can read; vertical type is a documented failure mode, not benched),
  * realistic degradation: gaussian blur, JPEG re-compression, uneven light,
    sensor-ish noise, varied fonts/colours/widths,
  * a few spines carry an author surname band, exercising the author-band row,
  * the label file uses the same schema as the human-labelled bench, so
    `booksnap bench-spines` scores it unchanged.

Run:  python bench/make_synthetic_shelf.py [outdir]
Writes: synth_shelf.png, spine_labels_synth.json
The OpenLibrary cache that makes the gazetteer step offline lives next to
this file as synth_olcache.json (warm it once with --warm-cache).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# (title, author-or-None). Authors render as a small surname band on the
# spine, exactly like the real shelves that produced author_band rows.
BOOKS = [
    ("Sapiens", "Harari"),
    ("Thinking, Fast and Slow", "Kahneman"),
    ("The Selfish Gene", "Dawkins"),
    ("Consciousness Explained", "Dennett"),
    ("The Origin of Species", None),
    ("A Brief History of Time", "Hawking"),
    ("The Emperor's New Mind", "Penrose"),
    ("Weapons of Math Destruction", None),
    ("Noise", None),
    ("The Blank Slate", "Pinker"),
    ("Predictably Irrational", None),
    ("Antifragile", "Taleb"),
    ("The Black Swan", None),
    ("Fooled by Randomness", None),
    ("Skin in the Game", None),
    ("Silent Spring", None),
    ("The Double Helix", "Watson"),
    ("Cosmos", "Sagan"),
    ("Contact", None),
    ("The Demon-Haunted World", "Sagan"),
    ("Flow", None),
    ("Guns, Germs, and Steel", "Diamond"),
    ("Collapse", None),
    ("The Sixth Extinction", "Kolbert"),
]

WOOD = (96, 64, 40)
SPINE_COLOURS = [
    (178, 34, 34), (31, 58, 92), (28, 84, 62), (214, 178, 108),
    (74, 48, 96), (158, 82, 40), (40, 40, 46), (196, 188, 172),
    (96, 128, 60), (120, 40, 60),
]
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _rng(seed=8):
    return np.random.default_rng(seed)


def render(outdir=HERE, seed=8):
    rng = _rng(seed)
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), WOOD)
    draw = ImageDraw.Draw(img)
    shelves = 3
    shelf_h = H // shelves
    labels = []
    i = 0
    for s in range(shelves):
        y0 = s * shelf_h + 24
        y1 = (s + 1) * shelf_h - 26
        draw.rectangle([0, y1, W, y1 + 22], fill=(62, 40, 24))  # shelf board
        x = 12
        per = len(BOOKS) // shelves + (1 if s < len(BOOKS) % shelves else 0)
        for _ in range(per):
            title, author = BOOKS[i]
            bw = int(rng.integers(52, 100))
            bh = y1 - y0 - int(rng.integers(0, 26))
            col = SPINE_COLOURS[i % len(SPINE_COLOURS)]
            draw.rectangle([x, y1 - bh, x + bw, y1], fill=col)
            # spine highlight/shadow so edges are not perfectly crisp
            draw.rectangle([x, y1 - bh, x + 3, y1], fill=tuple(min(255, c + 34) for c in col))
            draw.rectangle([x + bw - 3, y1 - bh, x + bw, y1], fill=tuple(max(0, c - 30) for c in col))
            words = title.replace(",", "").split()

            def _fit(text, path, start):
                # shrink until the word fits inside the spine with margins
                fs = start
                while fs > 8:
                    f = ImageFont.truetype(path, fs)
                    if draw.textlength(text, font=f) <= bw - 10:
                        return f, fs
                    fs -= 1
                return ImageFont.truetype(path, 8), 8

            font, fs = _fit(max(words, key=len), FONTS[i % len(FONTS)], 24)
            ty = y1 - bh + 10
            light = (245, 240, 230) if sum(col) < 360 else (28, 24, 20)
            for w in words:
                if ty + fs > y1 - (26 if author else 8):
                    break
                draw.text((x + bw / 2, ty), w, font=font, fill=light, anchor="ma")
                ty += fs + 3
            if author:
                af, _ = _fit(author.upper(), FONTS[0], max(10, fs - 4))
                draw.text((x + bw / 2, y1 - 16), author.upper(), font=af, fill=light, anchor="ma")
            labels.append({"title": title, "author": author, "confident": True,
                           "shelf": s + 1, "x": int(x), "w": bw})
            x += bw + int(rng.integers(2, 7))
            i += 1
        if x < W - 60:  # bookend gap, like a real partly-filled shelf
            draw.rectangle([x + 8, y1 - 90, x + 26, y1], fill=(52, 34, 20))
    # degradation: blur + uneven light + noise + jpeg
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    light = Image.new("L", (W, H), 0)
    ld = ImageDraw.Draw(light)
    ld.ellipse([-400, -500, 1500, 900], fill=70)
    light = light.filter(ImageFilter.GaussianBlur(300))
    img = Image.composite(Image.eval(img, lambda p: min(255, p + 26)), img, light)
    arr = np.asarray(img).astype(np.int16)
    arr += rng.normal(0, 2.2, arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "synth_shelf.png")
    img.save(png, quality=72)
    lab = os.path.join(outdir, "spine_labels_synth.json")
    bands = [{"name": a, "confident": True, "shelf": b["shelf"]}
             for b in labels for a in [b["author"]] if a]
    json.dump({"video": "synthetic (bench/make_synthetic_shelf.py, seed 8)",
               "clip": "rendered 1920x1080 shelf, 3 boards, blur 0.7 + jpeg q72",
               "method": "ground truth by construction; every row is 'confident'",
               "books": labels, "author_bands": bands},
              open(lab, "w"), indent=1)
    return png, lab


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    if "--warm-cache" in sys.argv:
        out = HERE
    png, lab = render(out)
    print(png, lab)
