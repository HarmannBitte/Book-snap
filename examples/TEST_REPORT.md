# Book-snap generality test — two new videos + two new channels

Run with the packaged CLI, 2026-09-18. New channels added this round:
`booksnap audio` (faster-whisper ASR, chunked to stay inside small-RAM boxes)
and `booksnap spines` (shelf-band crop → 4× Lanczos + unsharp → tiled OCR),
fused in `compile` via `fuse_audio` / `audio_title_candidates`.

## Video 1 — "Why Biological Race Isn't Real" (Zach B. Hancock, 21:36, 720p)

Type: PowerPoint screen-recording essay.

| Channel | Result |
|---|---|
| segment | 27 segments = 27 slides |
| covers | 33 panels, all charts/figures; 0 book jackets |
| scroll | none (correct) |
| audio (base, 265 segs) | **0 book-title cues** — the narration never names books |
| fusion | 0 confirmed (after word-boundary + book-like filters; earlier naive fusion produced 6 false "heard" hits from slide captions the narrator reads aloud) |
| pdf | `v1/slides.pdf`, 27 pages |

Verdict: correct negative result across all three channels.

## Video 2 — "Nathan Cofnas on Race, IQ, and the Jason Arday Scandal…" (Coleman Hughes, 102:33, 720p)

Type: two-camera podcast with a physical bookshelf backdrop.

| Channel | Result |
|---|---|
| segment (fps 5) | 444 camera-cut segments |
| covers | 2 panels = end-card logo only |
| spines (shelf pans 2560–2615 s) | 19 reads; robust: **"Sapiens"** (0.84); rest at 720p resolution limit ("LOSIHCAKCE", "WEATTSPOF", "ANDPOLTLS"…) |
| audio (base, 1304 segs, chunked 5 min) | title cues: **"Facing Reality"** (t=2086.8, a real book), "Ethnic DeLema" (ASR garble), "Simon Barron Cohen" (person, not a book) |
| fusion | garbled spine "Goleman" resolved to spoken "Coleman." (0.80); show-logo spines ("Conversations", "Coleman") matched the spoken show name — true matches, but programme title not books |

### Findings / limitations (honest)

1. **Spine text needs a better master.** At 720p only the largest spine is
   robust; 1080p+ or super-resolution is required for the rest.
2. **Audio cue recall is low.** Book titles are usually mentioned without a
   cue word ("…as I say in *Facing Reality*…" vs "the book X"). A proper
   solution is gazetteer/NER matching over the transcript, not regex cues.
3. **Fusion needs guards.** Naive fuzzy audio↔visual matching yields false
   positives whenever a narrator reads slide captions aloud (video 1).
   Word-boundary windows + a book-likeness filter (spine source, or book
   aspect ratio + ≤8 words + no terminal period) removed all of them.
4. **ASR memory:** faster-whisper decodes input + VAD probabilities as
   float32 → OOM on a 1 GB box for >~20 min audio; chunked transcription
   (5-min pieces, offsets restored) fixes it.
5. Show/podcast logos on the set match the spoken show name; filtering
   programme titles vs book titles remains a curation step.

## Round 3 - gazetteer verification (stage 11)

`booksnap/gazetteer.py`: title-case n-gram candidates from the transcript
(2-6 words, connectors allowed) + visual candidates, verified against
OpenLibrary (`search.json?title=`) with prefix/equality matching, an
author-name rejection (person != title), a junk-record rejection, a
book-context window for audio-derived candidates, and a disk cache.

| Video | Verified titles | Assessment |
|---|---|---|
| v1 | 0-1 ("The Princeton Guide to Evolution", Losos 2014 - flaky: depends on OL returning year/authors) | recall limited |
| v2 | "Facing Reality" (Murray 2021) TRUE; "New York City" FP (place); "Philosophical Psychology" borderline (journal) | precision ~1/3 |

Findings: cue-regexes (round 2) missed "Facing Reality"; the gazetteer finds
it with no cue word - recall win. Precision is capped by entity ambiguity
(places/journals/persons are also book titles) and by OpenLibrary catalogue
noise; production use should add a proper NER/WORK_OF_ART model or a curated
gazetteer. Rate-limiting: bursts of >~60 queries got us throttled (two runs
timed out); the disk cache + `--max-queries` cap fix repeatability.

**Critical bug found:** `title_candidates` infinite-looped on phrases like
"In the" (trailing connector strip via `rsplit(None, 1)` is a no-op on a
single token, and "in" is itself a connector). Fixed with a token-list pop +
regression test. This bug silently killed two 20-25 min runs.

## Bugs found & fixed during this round

- `_raw_frame` hardcoded 1080×1920 → probed dimensions.
- local `probe` shadowed imported `probe()` in `capture_scroll`.
- `stitch` re-sorted merged scroll lines by y across frames (wrong for an
  upward scroll) → temporal order; min-length 8→6.
- `heard_match` window was 4 chars too long (ratio 0.777 vs thr 0.78) and
  matched mid-word → n+1 window, word-boundary starts, thr 0.75.
- OOM on long audio → chunked `audio.transcribe`.

## Repro

```bash
booksnap run v1.mp4 --workdir runs/v1 --audio v1.wav
booksnap features v2.mp4 runs/v2/feat.npz --fps 5
booksnap segment  runs/v2/feat.npz runs/v2/cuts.npy
booksnap frames   v2.mp4 runs/v2/cuts.npy runs/v2/feat.npz runs/v2/reps
booksnap covers   runs/v2/reps runs/v2/covers
booksnap audio    v2.wav runs/v2/transcript.json --model base
booksnap spines   v2.mp4 runs/v2/spines.json --start 2560 --end 2615
booksnap compile  --manifest runs/v2/covers/manifest.json \
                  --cover-ocr runs/v2/covers/cover_ocr.json \
                  --scroll runs/v2/scroll_lines.json \
                  --audio runs/v2/transcript.json --spines runs/v2/spines.json \
                  --out-json runs/v2/books_candidates.json --out-md runs/v2/books_candidates.md
```
