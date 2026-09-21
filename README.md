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
booksnap audio    lecture.opus out/transcript.json    # chunked Whisper ASR
booksnap spines   talk.mp4 out/spines.json --start 2550 --end 2576 \
                  --presets lanczos,unsharp,clahe,denoise,binarize --stack
booksnap compile  --ocr out/ocr.json --manifest out/covers/manifest.json \
                  --cover-ocr out/covers/cover_ocr.json \
                  --scroll out/scroll_lines.json \
                  --audio out/transcript.json --spines out/spines.json \
                  --gazetteer --max-queries 60 \
                  --out-json out/books_candidates.json --out-md out/books_candidates.md
```

`booksnap run --resume` skips any stage whose artifact already exists in
`--workdir`, so an interrupted run costs nothing to restart.

### Spine super-resolution

Physical bookshelves are the hardest source: ~10 px glyphs at 720p. `spines`
crops the shelf band, optionally median-stacks the sharpest frames (a locked-off
camera gets a free denoise), then runs each crop through several enhancement
presets (`lanczos`, `unsharp`, `clahe`, `denoise`, `binarize`) and keeps the
highest-confidence OCR read per string, recording which preset produced it.
Measured on a real bookshelf pan (Coleman Hughes / Nathan Cofnas): 19 reads at
720p with one preset -> 51 reads at 720p with all five -> **88 reads at 1080p**,
where titles like *Losing Ground*, *Sapiens* and author spines become legible.
Source resolution dominates: always prefer `yt-dlp -f 137` when available.

### Verification, tiers and exports

`--gazetteer` checks every candidate (visual first, spoken titles interleaved so
neither channel starves) against OpenLibrary. Matching is fuzzy **and** phonetic
(`booksnap/fuzz.py`: token-set ratio + Metaphone + Jaro-Winkler), so OCR/ASR
damage like `LOSISG GROUND` or `Ethnic DeLema` still aligns to *Losing Ground* /
*The Ethnic Dilemma*. Guards: junk catalogue records, person-name titles
(middle-initial signal), place names, institutions, single fragments, and an
edition-count gate that rises with rule weakness (exact 3, prefix/fuzzy 4).

Each hit is tiered by corroboration, and only the top two tiers are exported:

| tier | meaning |
|---|---|
| `confirmed` | visual evidence (cover/spine), or the title was also heard |
| `verified` | spoken >=2 times, or once inside a book-ish context |
| `weak` | single uncorroborated mention - reported, not exported |

Exports: `books_candidates.md` / `.json` plus **`.bib`** and **`.ris`** for the
exportable tiers, with score, match rule and provenance per entry.

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

* `compile` produces *candidates*; the gazetteer tiers them but final curation
  of OCR text into clean author/title/year records is still human- or LLM-assisted.
* Covers smaller than ~1 % of frame area, or illegible at source resolution,
  are missed. Bookshelf spines need 1080p+ or the presets above.
* Scrolling text faster than ~1 line per sampled frame can lose lines; lower
  `--scroll-step`.
* `base` Whisper mis-hears proper nouns; gazetteer verification is what turns
  those garbles back into titles. A larger ASR model helps but needs more RAM.
* Gazetteer precision is bounded by entity ambiguity (places, journals and
  persons are also book titles) and by OpenLibrary catalogue noise. See
  `ROADMAP.md` for the measured numbers and what would close the gap.

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
  compile.py       stage 7 fusion, tiers, dedupe, bib/ris export
  pdf_report.py    stage 8 slide PDF
  audio.py         stage 9 chunked Whisper ASR
  spines.py        stage 10 shelf/spine OCR over enhancement presets
  superres.py      selective crop upscaling + enhancement presets
  gazetteer.py     stage 11 OpenLibrary title verification
  fuzz.py          fuzzy + phonetic (Metaphone/Jaro-Winkler) matching
  splitwords.py    dictionary DP split of fused caps OCR ("LOSINCTHIRACE")
  vlm.py           vision-model review loop (bundle/run/merge, optional API)
  bench.py         dev: detector benchmark harness
  bench_spines.py  scores the spine channel vs a labelled shelf
bench/             labelled shelf v2 + synthetic shelf generator/labels/cache
                   (`booksnap bench-spines`, `python bench/make_synthetic_shelf.py`)
examples/          validation run + extracted book list
```

## Discovery log

`docs/discovery_tree.json` + `docs/DISCOVERY_LOG.md` record every exploration
decision with its realized outcome (Dream-RSI-style: history as a replay
simulator). Screen ideas against it at zero executions:
`python tools/replay_discovery.py rank|path|dead|open|query WORD|render`.

## License

MIT — see `LICENSE`.
