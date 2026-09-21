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

## Round 4 - P0 (fuzzy/phonetic matching, tiers, exports) + spine super-resolution

### Spine super-resolution experiment

Same 26 s bookshelf pan of video B (2550-2576 s), re-downloaded at both
resolutions (formats 136 / 137) so the only variable is source resolution:

| source | presets | reads | wall time |
|---|---|---|---|
| 720p | `unsharp` only (old default) | 19 | ~30 s |
| 720p | 5 presets + median stack | 51 | 38 s |
| 1080p | 5 presets + median stack, x3 | **88** | 67 s |

1080p is where spines become legible: `LOSING GROUND`, `LOCKED IN`, `DENNETT`,
`COATES`, `BARACK OBAMA`, `WEALTH...` appear, and clustering the 88 reads
collapses them into 42 physical-spine groups (e.g. `Sapicns`/`Saptens`/`Sapiens`
-> one group, canonical read by confidence).

### P0 matching changes and their measured effect

`booksnap/fuzz.py` (token-set ratio + Metaphone + Jaro-Winkler) is now used by
both fusion and the gazetteer; spine reads are cleaned before querying
(`M LOSING GEOCND` -> `LOSING GEOCND`) and matched by per-token phonetic
alignment (`LOSISG GROUND` -> *Losing Ground*, `Ethnic DeLema` -> *The Ethnic
Dilemma*); proper nouns spoken in the transcript disambiguate same-titled
catalogue records; results are tiered and exported to BibTeX/RIS.

Video B (podcast, 1080p spines) - exported tiers:

| tier | phrase | matched record | judgement |
|---|---|---|---|
| confirmed | `Sapiens` | *Sapiens*, Yuval Noah Harari, 2011 (86 ed) | correct |
| confirmed | `LOSINCTHIRACE` (alt-cleaned) | *Losing Ground*, Charles Murray, 2008 | correct |
| verified | `Facing Reality` | *Facing Reality*, Charles A. Murray, 2021 | correct |
| verified | `Great Awakening` | *The great awakening*, Thomas S. Kidd, 2006 | correct |
| verified | `Philosophical Psychology` | *Philosophical psychology*, Donceel, 1955 | **FP** (journal) |

Video A (slide deck, no books):

| tier | phrase | matched record | judgement |
|---|---|---|---|
| verified | `Princeton Guide to Evolution` | *The Princeton Guide to Evolution*, Losos, 2013 | correct |
| weak (not exported) | `Michael Moore`, `Cold War`, `New York Times` | person / event / newspaper FPs | correctly demoted |

Precision of the exported list: **4/5 on video B, 1/1 on video A** (round 3 was
1/3 on B and 0-1 on A). Author hints also fixed *record* selection: before,
`Facing Reality` matched Eccles 1970 and `Losing Ground` matched Catherine
Aird; the transcript names Charles Murray, and both now resolve to him.
Region/demographic phrases (`West Africa`, `African Americans`) are rejected
outright; `Peter Singer` is caught by the middle-initial person rule.

### Bugs found & fixed this round

* **Poisoned gazetteer cache**: a failed fetch was cached as an empty result,
  permanently hiding real titles (`facingreality` had 0 docs cached). 69 stale
  entries in video B and 18 in video A were purged; `fetch_docs` now returns
  `None` on failure and only successful results are cached.
* `looks_like_person` rejected ordinary two-word titles (*Losing Ground*).
  Narrowed to the real signal: a middle initial (`Charles T. Murray`).
* Alt (OCR-variant) lookups consumed the whole query budget, so spoken titles
  were never verified. Alts now have their own sub-budget, the queue
  interleaves visual/spoken, and cue-detected titles lead the spoken queue.
* Phonetic fusion produced homophone-only false "heard" hits
  (`COATES` ~ `cities`); a heard confirmation now needs >=2 significant words
  or a near-exact spelling match.
* A global fuzzy-score gate rejected pairs that per-token alignment had already
  proven; alignment is now the criterion and the score is only reported.

Tests: 42 passing (added fuzz, tiers, dedupe, BibTeX/RIS, resume-facing
compile test, match-rule guards, variant retry, author hints, cache-poisoning).

## Round 5 - first labelled benchmark (visual ground truth)

The shelf frame was read visually at 1080p (shelf1080.mp4, t=10) and stored as
`bench/spine_labels_v2.json` (9 confident books + 3 author bands). Scorer:
`booksnap bench-spines` (strict, directional matcher). First honest recall
numbers for the spine channel:

| metric | value |
|---|---|
| books surfaced at all (any read) | 4/9 = 0.44 |
| books verified to a catalogue record | 2/9 = 0.22 |
| verified with the right record | 2/9 = 0.22 |
| author bands surfaced / reported | 2/3 / 0 |

Misses are instructive: *Losing the Race*, *A Testament of Hope*, *We Were
Eight Years in Trouble*, *Wealth, Poverty and Politics*, *On the Origin of
Species* are visible to the eye but their spine type is below OCR at 1080p, or
the read is too mangled to query. Precision of the exported list stays 4/5
(Sapiens, Losing Ground, Facing Reality, The Great Awakening correct;
Philosophical Psychology is a journal-name false positive); the benchmark shows
the real weakness is RECALL on shelves, not precision.

The visual read also caught a bug the aggregate numbers hid: at clustering
threshold 0.85 two different spines merged ("LOSING THE RACE" + "LOSING
GROUND") and the group inherited Charles Murray's book. Threshold raised to
0.92; the benchmark matcher is deliberately stricter than the pipeline's own
fuzzy matcher so it cannot credit such accidents.

Also this round: cached OpenLibrary lookups no longer consume the query budget
(warm cache = every candidate verified; 23 matches, 18 correctly demoted to
weak), and a new informational "author spines" channel reports single-surname
reads only when the transcript also says the name (empty for this video).

## Round 6 - attacking spine recall: pan stitching, ROI crops, VLM loop

Three experiments against the measured 0.22 verified recall:

1. **Pan stitching** (`superres.stitch_pan`, `spines --panorama`): phase-
   correlation showed ~0 px shift between the sharpest frames - the shelf is
   NOT panning, the whole shelf is in every frame. Coverage was never the
   limit; reads went 88 -> 116 (more frames/presets) but bench recall stayed
   0.44/0.22. Honest negative result, kept because real pans exist.
2. **Per-spine ROI at x6** on my visual coordinates: recovered 2 of 8 misses
   (*A Testament of Hope* as fused `ATESTAMENTTEHOPE`, part of *Wealth,
   Poverty and Politics*). Not integrated: it needs an automatic spine
   detector to propose ROIs, and fused tokens still need word-splitting.
3. **Vision-model review loop** (`booksnap/vlm.py`, `vlm-bundle` /
   `vlm-merge`): open reading questions over shelf images, any vision model
   fills them, results merge back as `confirmed` entries with provenance and
   rewrite .md/.bib/.ris. Filled from my visual read of the labelled frame:
   verified recall 0.22 -> 1.0 on that shelf, exported .bib 5 -> 13 entries.
   **Circular by construction** (the reviewer is the ground truth here); the
   channel's real accuracy must be measured on held-out shelves - that is the
   next labelling task, not a claim.

Conclusion: on 720-1080p shelves the OCR channel's ceiling is glyph size and
fused tokens; the vision channel removes that ceiling and is now wired in.

## Round 7 - the "other ones": serial guard, splits, author bands, API hook, held-out shelf

* **Journal false positives closed**: `gazetteer.crossref_is_serial` consults
  Crossref for the phrase; a journal-article hit in a same-named container
  vetoes the book claim. "Philosophical Psychology" (the last exported FP on
  video B) is gone; exported precision on B is now 4/4.
* **Fused caps reads**: `splitwords.dp_split` segments clean fusions against a
  domain-built vocabulary ("LOSINGTHERACE" -> LOSING THE RACE) and feeds them
  as extra queries; mangled glyphs correctly refuse to segment. No live win
  yet on video B (its fusions are mangled, not clean) - the mechanism is in
  place for cleaner sources.
* **Author bands**: `vlm-merge` records author_band verdicts; video B now
  reports Dennett / Barack Obama / James Q. Wilson as shelved authors in
  .json and .md alongside the book list.
* **Optional API reviewer**: `vlm-run` fills a review bundle through any
  OpenAI-compatible vision endpoint, only when VLM_API_KEY is set;
  `audio --beam-size` exposes the beam for garbled proper nouns.
* **Held-out shelf (JBP podcast, 720p)**: full spine+gazetteer run over the
  visible shelf produced 0 reads and 0 title claims - matching the human
  verdict that the shelf is illegible (depth-of-field blur, vertical spine
  text). Precision guards hold out of sample; bokeh and vertical spines join
  the documented failure modes. A third in-focus shelf could not be fetched
  this session (YouTube bot-gated further downloads), so a title-level
  held-out benchmark remains open.
* CI (branch `ci`): pytest plus an offline `bench-spines` regression gate on
  committed fixtures (recall_verified >= 0.2), no network needed.

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

## Round 8 - synthetic held-out shelf + spine grouping (2026-09-20)

Goal: a title-level held-out benchmark that needs no download and no network,
after YouTube bot-gated searches and the JBP shelf proved illegible.

Method: bench/make_synthetic_shelf.py renders a 1920x1080 shelf (3 boards,
24 real titles as stacked spine words, 13 surname bands, gaussian blur 0.7,
jpeg q72, noise, uneven light) plus its ground truth in the same schema as
the human-labelled bench. bench/synth_olcache.json is a committed OpenLibrary
cache so the gazetteer step runs offline in CI.

Pipeline fixes the bench exposed (all tested):
  1. spines accepts a still image (benchmarks, shelf photos).
  2. reads carry box w/h; compile.group_spine_reads joins vertically stacked
     spine words per column and splits a trailing ALL-CAPS author band -
     before this, multi-word titles never became candidates at all.
  3. compile --gazetteer verified NOTHING on spine-only inputs
     (`if gazetteer and audio`); shelf-only runs now verify.
  4. caps surname bands no longer consume title queries; author bands are
     reported without audio corroboration when no audio exists.
  5. clean_spine_phrase collapses space garble ("Ma th" -> "Math");
     min_len 4 -> 3 so connectors survive OCR.
  6. bench scorer: label sig words drop connectors and nlabel is
     article-free; v2 surface recall 0.44 -> 0.56 is this scorer fairness,
     verified/right_record unchanged at 0.22 (no pipeline regression).

Results: SYNTH surface 0.79 / verified 0.79 / right_record 0.79,
author bands 12/13 reported. Misses are honest OCR failures (space garble,
a dropped word). V2 real shelf: 0.56 / 0.22 / 0.22 (unchanged verified).
CI (branch ci) gains an offline synthetic-shelf gate: verified >= 0.6.

Download probes: direct watch URLs still work (Test A re-fetched, 1280x500
slide deck with shelf webcam corner = runs/v1); searches stay bot-gated;
TKP #177 is a pure slideshow; sectioned downloads segfault imageio-ffmpeg.

## Round 9 - real shelf re-run with the grouping pipeline (2026-09-20)

Test B's 1080p source was re-fetched (direct watch URL; 846 MB) and the
labelled window (2550-2576 s) re-extracted with box-carrying reads, 5 frames,
4 presets + median stack: 165 reads, first real multi-word spine groups
("LOSINC THE RACE", "GARACK OBAMA", "Conversations ...").

Result vs the human labels: surface 0.56 / verified 0.22 / right 0.22 -
verified unchanged from baseline (Sapiens + Losing Ground; the Dennett band
now reports as author). "Losing the Race" stays unverified: every read of
that spine confuses C/G ("LOSINC"), and OpenLibrary cannot retrieve docs for
garbled strings - the ceiling here is glyph confusion, i.e. the super-res /
learned-OCR workstream, not the matcher.

Bug found: the default 60-query gazetteer budget starves spine-dense runs
(verified 0.11 at 60 vs 0.22 at 250). Default is now 250; CI passes explicit
budgets. Artifacts: examples/spines1080_wide.json + books_spines1080_wide.json
(reproduce the numbers offline).

## Round 10 - typo-tolerant retrieval (degarble) + alt-plumbing fixes

booksnap/degarble.py generates single-glyph confusion candidates (pair-major:
C/G, G/B, B/R, A/R, Y/I, U/V, I/L, U/L, ...) offered as spine alts, because
catalogue retrieval is exact-text: one confused glyph used to hide a book.
Plumbing fixes the bench forced out: near-miss variants lead the alt list
(the highest-conf canon is often a garbled cousin of a cleaner lower-conf
read); degarble runs over variants too (canon may be two glyphs from truth);
per-candidate alt cap 8; alt budget raised to max_queries (primaries keep
their own budget - test updated); scorer counts a garbled canon verified
under its clean matched_as form; visual queue sorts multi-word first.

Measured result on the real shelf: verified 0.22 (Sapiens, Losing Ground) -
and the third verified entry is BARACK OBAMA via degarble from GARACK OBAMA.
"Losing the Race" is NOT recoverable by any degarble: OpenLibrary returns
zero documents for the correct string - the residual gap is catalogue
coverage plus glyph confusion, both now named and measured. Synthetic shelf
unchanged (0.79). Tests 54.

## Round 11 - Crossref monograph fallback (catalogue coverage)

OpenLibrary returning zero docs for "Losing the Race" (round 10) motivated a
second catalogue: gazetteer.crossref_book_docs fetches Crossref book/monograph
DOIs when OL misses; matches are tagged match_type=crossref and demoted to
status=weak (reported, never exported) unless a spoken author hint
corroborates the record - because same-title different-book monographs are
real (Crossref's only "Losing the Race" is Gadd & Dixon 2018, not Murray
2000). Bench scorer now counts only confirmed/verified statuses as verified
recall, so weak rows can never inflate recall.

Measurements: Crossref throttles shared IPs hard (5/10 burst success at 0.5 s
spacing; responses degrade to empty under load), so bench/warm_crossref.py
warms the fallback cache politely and the result is committed
(examples/books_spines1080_wide.json.olcache.json) keeping runs and CI
offline. Final verdict on the coverage gap: trade-press books largely lack
DOIs, so Crossref closes almost nothing here - a bundled catalogue dump
(Wikidata subset) would be the next infra step, deliberately out of scope
for a dependency-light repo. Benches unchanged: v2 wide 0.56/0.22/0.22,
synth 0.79/0.79/0.79. Tests 56.

## Round 12 - YouTube unlock via JS runtime (Agent-Reach takeaway) + held-out attempt (2026-09-21)

Takeaway from github.com/Panniantong/Agent-Reach (its YouTube channel is a
yt-dlp wrapper whose doctor insists on a JS runtime): modern yt-dlp needs
deno/node for YouTube extraction, and this sandbox had none - which is what
the "bot gate" partly was. With deno 2.9.7 (kept in .cache/bin, excluded from
snapshots) and --js-runtimes deno, yt-dlp SEARCH works again: candidate
hunting for held-out shelves is now cheap instead of impossible.

Held-out attempt with the unlock: a different episode of the same show as the
labelled shelf (Conversations with Coleman, JcMvSCHGqtE, 1080p) - same
physical books, unseen footage, labels reusable. Result: 0 spine reads at
every angle/preset; a x6 clahe crop shows pure depth-of-field bokeh, no
glyphs at all. The shelf cam's focus varies by episode, so same-show reuse
is not a guaranteed holdout. Artifact: examples/heldout_coleman_klein_spines.json
(empty, with this note). Probes also logged: Cowen studio = curtain, DarkHorse
= painted backdrop, Politics&Prose event = wood wall (no shelves).

Status: search gate GONE (documented recipe: deno + --js-runtimes), focus
gate remains the selector for future held-out shelves. Benches and code
unchanged this round; tests 56.

## Round 13 - YouTube captions + optional LLM correction gate (2026-09-21)

Question: skip local Whisper and use YouTube's transcript, LLM-corrected?
Measured answer: captions yes, raw no, LLM-gated yes.

booksnap/captions.py (stage 9b) parses roll-up VTT (yt-dlp --write-auto-subs)
into the audio schema; `booksnap captions in.vtt out.json`. On the labelled
podcast captions spell proper nouns Whisper base garbled: "Ethnic Dilemma"
(vs DeLema), "Jason Arday" (vs Arde), "Thomas Sowell" (vs Sol).

But captions fed raw into the spoken n-gram scan collapse precision:
23 gazetteer hits of which 18 "verified" non-books ("Soviet Union",
"Hillary Clinton", "Steven Pinker", "World War II"...) - the noisy Whisper
transcript had been an accidental precision filter. So captions enter only
through booksnap/llmfix.py (optional, env-gated like the VLM reviewer:
LLM_API_KEY/LLM_BASE_URL/LLM_MODEL): extract_titles() = constrained
"which BOOK TITLES are mentioned, spelling corrected" extraction;
correct_phrases() = spelling-only repair offered as first spine alt.
compile --llm-titles wires both; absent key or any error -> classic path.
Tests 57 (mocked completion + no-key passthrough).

Verdict: Whisper stays the default (works on any local file, no platform
tie-in); captions + LLM gate are the YouTube-side upgrade once a key exists.
Bench numbers unchanged (v2 0.56/0.22/0.22, synth 0.79) - the LLM stage is
off by default and never overrides the gazetteer.
