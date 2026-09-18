# Roadmap: closing the gap to a reliable book-extraction pipeline

Honest framing: "perfect" (100 % recall, 100 % precision) is not reachable from
720p video + noisy ASR. What is reachable is a **measurable, bounded error
budget** per video class, with every output line traceable to evidence. This
document lists the residual failure modes we actually measured on the two test
videos, the root cause of each, the fix, and the priority.

## 1. Measured state (v1 slide deck, v2 podcast)

| Channel | v1 | v2 | Verdict |
|---|---|---|---|
| Segment / frames | 27 slides | 444 cuts | solid |
| Covers (panel detect) | 33 panels, 0 books | 2 end cards | recall-limited |
| Cover OCR | – | 39 bibliography entries | solid |
| Scroll text | 0 lines | 117 lines | solid |
| Spines | – | 19 reads; 1 robust ("Sapiens") | resolution-limited |
| ASR | 265 segs | 1304 segs | word-error-limited |
| Audio title cues | 0 | 3 (1 true, 1 garble, 1 person) | precision-limited |
| Fusion | 0 | 1 (0.80) | exact-match only |
| Gazetteer (OpenLibrary) | 0–1 | 3 (1 true, 1 place, 1 journal) | precision ~1/3 |

## 2. Error taxonomy → root cause → fix

**E1. Sub-threshold resolution (spines, small print, cover text).**
Root cause: 720p source; OCR confidence < 0.75 on ~10 px glyphs.
Fix: (a) prefer 1080p+ at download time (`-f "bv*[height>=1080]/bv*"`);
(b) *selective* super-resolution — upscale only flagged low-confidence crops
(shelf regions, panels), never whole frames: Real-ESRGAN ONNX ×4 on CPU is
affordable for tens of crops, not for 444 frames; (c) multi-frame aggregation
already helps, extend to sub-pixel alignment before OCR.

**E2. ASR word errors on titles ("Ethnic DeLema", "Goleman" vs "Coleman").**
Root cause: `base int8` model, no beam search, no word confidence.
Fix: (a) `small`/`medium` int8 chunked when RAM allows, `beam_size=5`,
`word_timestamps=True`; (b) **phonetic/fuzzy retrieval** — match transcript
n-grams against a title index with Metaphone/Soundex + Jaro-Winkler ≥ 0.88
instead of exact/prefix. This single change converts garbles into recoverable
candidates. (c) keep per-word confidence and drop candidates whose words are
all low-confidence.

**E3. Gazetteer precision: persons, places, journals are also book titles.**
Root cause: no entity typing; OpenLibrary indexes non-book documents and junk
records.
Fix: (a) NER pass (spaCy `en_core_web_trf` or a small transformer) to keep only
`WORK_OF_ART` / unknown-proper-noun spans; (b) reject persons via a
person-name list + "authors-of-matched-book" check (partially done);
(c) restrict the index to `type=work`/edition records with a real
`first_publish_year`, ISBN or subject; (d) **require corroboration**: an
audio-only title claim needs either a visual hit, ≥2 independent mentions, or
a VLM confirmation before entering the final list.

**E4. Cover detection recall (3D mockups, angled covers, thumbnails, Kindle
screens, book lists rendered as text).**
Root cause: heuristics (panel shape + book-likeness) tuned on flat covers.
Fix: (a) a **VLM verification stage** on flagged panels — one vision-model call
per candidate panel: "is this a book cover? transcribe title/author" — this is
the single highest-leverage addition for both recall and precision;
(b) alternatively/locally, train a tiny detector (YOLOv8n) on synthetic covers;
(c) always keep the text-only fallback (bibliography/list OCR) as its own
source, which v2 already exercises.

**E5. Fusion is exact-match only.**
Root cause: normalized containment between audio text and visual titles.
Fix: rapidfuzz token-set ratio ≥ 0.90 with a phonetic tiebreaker; time-windowed
linking (mention within ±5 s of an on-screen panel) as evidence; speaker
diarization for attribution ("recommended by X") if quotes matter.

**E6. No labelled benchmark → we cannot claim or defend a number.**
Fix: hand-label 10–20 videos across classes (slides / podcast / vlog / review
channel), store as `bench/*.json`, compute per-channel precision/recall in CI,
and gate merges on regression. Calibrate the final confidence (Platt/ isotonic
on the labelled set) so thresholds mean something.

**E7. Operational fragility (rate limits, OOM, non-resumable runs).**
Fixed so far: chunked ASR, disk cache for OpenLibrary, `--max-queries`.
Remaining: per-stage artifact checkpointing + resume, exponential backoff with
jitter on all network calls, hard timeouts, a local catalogue index to remove
the network dependency, parallel per-segment workers, and pinned model
versions/hashes for reproducibility.

**E8. Output usefulness.**
Fix: deduplicate + normalize edition variants, attach ISBN/year/author from the
gazetteer, and emit BibTeX/RIS alongside Markdown/JSON. Every row keeps
provenance (timestamp, source channel, confidence, evidence snippet).

## 3. Prioritised plan

**P0 — high impact, days of work, runs on this box**
1. Fuzzy + phonetic matching in fusion and gazetteer (E2, E5).
2. Corroboration rule + entity typing lite (E3): audio-only claims demoted to
   `unverified` unless supported.
3. Per-stage resume/checkpoint + backoff + pinned models (E7).
4. Dedup, provenance, BibTeX/RIS output (E8).
5. Bench harness with 3 labelled videos and CI metrics (E6, start small).

**P1 — bigger levers, needs an API key or more RAM**
6. VLM verification stage on candidate panels/crops (E4) — biggest single win.
7. Selective super-resolution + OCR retry (E1).
8. NER model for spoken candidates (E3).
9. `small`/`medium` ASR with word timestamps when RAM allows (E2).

**P2 — infrastructure**
10. Local title index (OpenLibrary dump subset / Google Books cache) for offline
    fuzzy retrieval at scale (E3, E7).
11. Trained cover detector to replace heuristics (E4).
12. Full benchmark (20 videos), calibrated confidences, per-class thresholds
    and profiles: `slides`, `podcast`, `vlog`, `review` (E6).

## 4. Definition of done (proposed targets)

| Video class | Recall | Precision |
|---|---|---|
| Slide decks with bibliography | ≥ 0.95 | ≥ 0.95 |
| Podcasts with bookshelf + spoken titles | ≥ 0.85 | ≥ 0.90 |
| Reviews/vlogs (covers on screen) | ≥ 0.80 | ≥ 0.85 |

Plus: every emitted row has provenance and a calibrated confidence; CI reports
P/R per channel on the benchmark; a run is resumable and never exceeds the
network rate budget.

## 5. Known constraints

- Sandbox: ~1 GB RAM, no GPU, CPU-only. Chunking and *selective* upscaling are
  mandatory; whole-frame super-resolution and `large-v3` are not viable here.
- `shapely` not installable in this environment (already dropped; `pyclipper`
  covers the OCR dependency).
- GitHub PAT lacks `workflow` scope → `.github/workflows/ci.yml` sits on branch
  `ci`; a token with that scope (or a manual add on github.com) unblocks CI.
- OpenLibrary rate-limits bursts (~60 queries); any gazetteer growth must go
  through the disk cache or a local index.
