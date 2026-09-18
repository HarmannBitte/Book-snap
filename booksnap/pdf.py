"""Stage 8 - render the extracted slides as a PDF ("the video as a PDF")."""
from __future__ import annotations

import glob
import os
import subprocess

import img2pdf

from .ffmpeg_utils import ffmpeg_bin


def make_pdf(reps_dir: str, out_pdf: str, width: int = 1280, quality: int = 4,
             pattern: str = "seg_*.png") -> int:
    jpg_dir = os.path.join(os.path.dirname(out_pdf) or ".", "jpg")
    os.makedirs(jpg_dir, exist_ok=True)
    files = []
    for png in sorted(glob.glob(os.path.join(reps_dir, pattern))):
        jpg = os.path.join(jpg_dir, os.path.basename(png)[:-4] + ".jpg")
        subprocess.run([ffmpeg_bin(), "-v", "error", "-i", png,
                        "-vf", f"scale={width}:-2", "-q:v", str(quality), jpg, "-y"],
                       check=True)
        files.append(jpg)
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(files))
    return len(files)
