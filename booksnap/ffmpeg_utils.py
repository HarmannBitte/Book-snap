"""Shared helpers: locate an ffmpeg binary and spawn rawvideo frame pipes."""
from __future__ import annotations

import shutil
import subprocess
from typing import Iterator, Tuple

import numpy as np


def ffmpeg_bin() -> str:
    """Return a usable ffmpeg executable path."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # fall back to the static binary shipped with imageio-ffmpeg
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
        ) from exc


def probe(video: str) -> dict:
    """Minimal duration/resolution probe parsed from `ffmpeg -i` stderr."""
    p = subprocess.run([ffmpeg_bin(), "-i", video], capture_output=True, text=True)
    info = {"path": video}
    for line in p.stderr.splitlines():
        if "Duration" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            info["duration"] = float(h) * 3600 + float(m) * 60 + float(s)
        if "Video:" in line and "width" not in info:
            parts = line.split()
            for tok in parts:
                if "x" in tok and tok.replace("x", "").replace(",", "").isdigit():
                    w, h = tok.strip(",").split("x")
                    info["width"], info["height"] = int(w), int(h)
    return info


def frame_pipe(video: str, width: int, fps: float) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (index, rgb uint8 [H,W,3]) frames decoded+downscaled by ffmpeg."""
    height = width * 9 // 16
    cmd = [
        ffmpeg_bin(), "-v", "error", "-i", video,
        "-vf", f"fps={fps},scale={width}:{height}:flags=fast_bilinear",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    nbytes = width * height * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield idx, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def grab_fullres(video: str, t: float, out_png: str) -> None:
    """Seek to `t` seconds and write one full-resolution PNG."""
    subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", video,
         "-frames:v", "1", out_png, "-y"],
        check=True,
    )
