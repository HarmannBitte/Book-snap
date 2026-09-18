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
