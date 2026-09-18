# Book-snap generality test — two new videos

Run with the packaged CLI (`booksnap run` / stage calls), 2026-09-18.

## Video 1 — "Why Biological Race Isn't Real" (Zach B. Hancock, 21:36, 720p)

Type: PowerPoint screen-recording essay.

| Stage | Result |
|---|---|
| segment | **27 segments** (correct: 27 distinct slides) |
| frames/ocr | 27 reps OCR'd |
| covers | 33 panels; all are **charts/figures/memorial slides**, no book jackets |
| scroll | no scrolling credits found (correct — none exist) |
| pdf | `v1/slides.pdf`, 27 pages |
| compile | 0 bibliography entries, 0 book covers; figure panels correctly flagged `cited=False` |

Verdict: pipeline behaves correctly on a clean slide deck; correctly reports
"no books shown". Known cosmetic issue: `split_wide` halves a wide figure and
both halves inherit the same caption text (duplicate candidate rows).

## Video 2 — "Nathan Cofnas on Race, IQ, and the Jason Arday Scandal…" (Coleman Hughes, 102:33, 720p)

Type: two-camera podcast with a **physical bookshelf backdrop**.

| Stage | Result |
|---|---|
| segment (fps 5) | **444 segments**, median 6.6 s = camera cuts (sane for a 2-cam talk) |
| covers | 2 panels, both the end-card logo (`THEFP.COM`); **no embedded cover graphics** |
| scroll auto | 5 short "moving + text-dense" windows = **camera pans across the bookshelf** |
| compile | 0 embedded covers; shelf spines not modelled |

### What video 2 exposed (new modalities, now documented)

1. **Animated/motion-blurred title cards & shelf pans** trip the scroll
   auto-detector (moving + text-dense). Added `sharpness()` (Laplacian
   variance) + sharp-frame top-k selection for windows ≤ `card_max_s` (10 s);
   helps for cards, not for shelf pans.
2. **Books as physical spines** are a modality the pipeline does not model.
   Demoed the right handling: crop the shelf band → 4× Lanczos + unsharp →
   tiled OCR. Result: only the largest spine ("Sapiens") is robustly legible;
   the rest ("Losing …", "Wealth, Power and Politics", "Victims"…) are at the
   resolution limit of the 720p source. Reliable spine extraction needs a
   1080p+ master or super-resolution — flagged as future work.
3. For talk/podcast videos the *audio* channel (Whisper transcript + title
   matching) is the appropriate book-detection path; visual-only extraction
   will under-report by design.

## Bugs found & fixed during the test

- `scroll._raw_frame` hardcoded 1080×1920 → now uses `probe()` dimensions.
- `capture_scroll` local variable `probe` shadowed the imported `probe()`
  → renamed; `run` completed.
- `stitch` sorted merged lines by y across frames (wrong for an upward
  scroll) → temporal order preserved; min-length filter 8→6.
- `_pdf` CLI namespace missing width/quality in manual calls (test-side).

## Repro

```bash
booksnap run v1.mp4 --workdir runs/v1
booksnap features v2.mp4 runs/v2/feat.npz --fps 5
booksnap segment  runs/v2/feat.npz runs/v2/cuts.npy
booksnap frames   v2.mp4 runs/v2/cuts.npy runs/v2/feat.npz runs/v2/reps
booksnap covers   runs/v2/reps runs/v2/covers --attach-ocr runs/v2/ocr.json
booksnap scroll   v2.mp4 runs/v2/cuts_winmed.npy runs/v2/scroll_lines.json
```
