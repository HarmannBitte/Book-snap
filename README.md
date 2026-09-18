# Book-snap

Turn a **presentation-style video** (slide deck essays, lecture recordings,
conference talks) into:

1. one clean image per slide / visual state,
2. the **book covers** embedded in those slides, cropped out,
3. an OCR'd **book list** (on-screen covers + scrolling end-bibliography).

```
video ──▶ feature cache ──▶ segments ──▶ representative frames ──▶ OCR
                       └─▶ cover panels ─▶ cover OCR ─┐
                       └─▶ scroll capture (credits) ──┤──▶ books_candidates
```

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
# ffmpeg must be available (system ffmpeg or `pip install imageio-ffmpeg`)
```

## Usage

One command:

```bash
booksnap run lecture.mp4 --workdir out/
```

Or stage by stage (each stage is resumable / cache-friendly):

```bash
booksnap features lecture.mp4 out/feat.npz          # 1 decode, ~10 fps proxy
booksnap segment  out/feat.npz out/cuts.npy         # the detector (see below)
booksnap frames   lecture.mp4 out/cuts.npy out/feat.npz out/reps
booksnap ocr      out/reps out/ocr.json
booksnap covers   out/reps out/covers --ocr         # panel detection + crops
booksnap scroll   lecture.mp4 out/cuts_winmed.npy out/scroll_lines.json   # auto
booksnap compile  --ocr out/ocr.json --manifest out/covers/manifest.json \
                  --cover-ocr out/covers/cover_ocr.json \
                  --scroll out/scroll_lines.json \
                  --out-json out/books_candidates.json --out-md out/books_candidates.md
```

`booksnap scroll` auto-detects scrolling credit/bibliography sequences (moving
windows whose OCR probe is text-dense); pass `--start/--end` to force a range.

## The detector: stability-gated isolated-spike block-MAD

Naive frame-differencing over-segments presentation videos because they mix
*static slides*, *slow pans/fades* and *full-motion B-roll*. Book-snap instead:

1. **Stability gate** — a frame is in a *static region* when the windowed
   median (2 s) of consecutive-frame block-MAD < 1.0. This excludes B-roll.
2. **Isolated spike** — inside a static region, a cut is a single-frame
   block-MAD spike > 1.5 (adjacent spikes collapsed). Pans and fades never
   produce isolated spikes; edits do.

Block-mean MAD (24×14 grid) is used because a book cover appearing in one
corner of an otherwise-black slide is a *localised* change: block averaging
keeps it above codec noise, while perceptual hashes and colour histograms
dilute localised changes across the whole frame.

Measured on a 31:40 slide-deck essay video (see `examples/README.md`):

| Method | Segments |
|---|---|
| pixel-diff (classic OpenCV) | 1284 |
| FFmpeg `select=gt(scene,0.2)` | 746 |
| block-MAD (naive) | 745 |
| edge-diff | 565 |
| pHash | 504 |
| RGB histogram | 489 |
| **book-snap (stability-gated)** | **208** ← true slide count |

## Cover detection

Slide body text is sparse white-on-black; covers/photos are solid regions.
Two block-level masks are combined: **bright** (ink-fraction > 0.55) and
**dark-textured** (high local std, low mean) — the latter catches black
jackets with light type. Wide components (two covers side by side) are split.

## Limitations

* `compile` produces *candidates*; turning OCR text into clean
  author/title/year records is left to a human or an LLM pass.
* Covers smaller than ~1 % of frame area, or illegible at source resolution,
  are missed.
* Scrolling text faster than ~1 line per sampled frame can lose lines; lower
  `--scroll-step`.

## Repo layout

```
booksnap/
  ffmpeg_utils.py  decode/probe/seek helpers
  features.py      stage 1 feature cache
  segment.py       stage 2 detector (the winning one)
  frames.py        stage 3 representative frames
  ocr.py           stage 4 RapidOCR wrapper
  covers.py        stage 5 panel/cover detection
  scroll.py        stage 6 scrolling-credit recovery
  compile.py       stage 7 candidate book list
  bench.py         dev: detector benchmark harness
examples/          validation run + extracted book list
```

## License

MIT — see `LICENSE`.
