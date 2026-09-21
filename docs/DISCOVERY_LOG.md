# Discovery Log (Dream-RSI-style)

Every exploration decision with its realized outcome; the tree is a
replay simulator - screen ideas with `tools/replay_discovery.py`
instead of re-running them. Verdicts: kept / dropped / open.

## Round 1

- **r1-seg** [kept] commit pre-6b4bf91
  - intent: isolate on-screen change moments
  - action: stability-gated isolated-spike detector over frame features
  - outcome: 208 segments on video-1; recall of bibliography scroll windows 100% (metric: segments=208)
- **r1-scroll** [kept] commit pre-6b4bf91
  - intent: recover scrolling credits/bibliography
  - action: line-stitch scroll OCR with ENTRY_START grouping
  - outcome: 19 bibliography entries recovered (metric: biblio=19)
  - branched from: r1-seg
- **r1-covers** [kept] commit pre-6b4bf91
  - intent: find embedded book-cover panels
  - action: aspect-ratio panel detection + panel OCR
  - outcome: 10 covers; shown-only tier separated (metric: covers=10)
  - branched from: r1-seg
- **r1-pdf** [kept] commit pre-6b4bf91
  - intent: human-auditable artifact
  - action: representative-frame PDF report
  - outcome: slides.pdf accepted (metric: -)
  - branched from: r1-covers

## Round 10

- **r10-degarble** [kept, impact 0.05] commit 6a8b65c
  - intent: exact-text retrieval hides garbled books
  - action: pair-major single-glyph confusion candidates as alts
  - outcome: BARACK OBAMA verifies from GARACK OBAMA (metric: +Obama)
  - branched from: r9-budget
- **r10-alts** [kept] commit 6a8b65c
  - intent: clean variants cut off / budget eaten
  - action: near-miss variants lead alts; per-cand cap; alt budget = max; multi-word-first queue
  - outcome: Losing Ground robust to canon garble (metric: robustness)
  - branched from: r10-degarble
- **r10-olzero** [kept] commit 6a8b65c
  - intent: why Losing the Race never verifies
  - action: probe OpenLibrary with the CORRECT string
  - outcome: zero docs: coverage gap, not matcher (metric: docs=0)
  - branched from: r10-degarble

## Round 11

- **r11-xref** [kept] commit ea64958
  - intent: second catalogue for coverage
  - action: crossref_book_docs monograph fallback + polite pool
  - outcome: throttle 5/10 bursts; only same-title DIFFERENT book exists (metric: coverage~0)
  - branched from: r10-olzero
- **r11-weak** [kept, impact 0.90] commit ea64958
  - intent: same-title different-book is real
  - action: crossref-only = status weak unless spoken author hint; scorer counts only confirmed/verified
  - outcome: precision guard; weak rows visible not exported (metric: export clean)
  - branched from: r11-xref

## Round 12

- **r12-deno** [kept, impact 0.70] commit b3d9093
  - intent: YouTube search was bot-gated
  - action: Agent-Reach takeaway: yt-dlp needs JS runtime; deno + --js-runtimes
  - outcome: search works again; held-out hunting cheap (metric: search=unlocked)
- **r12-heldout-ck** [kept] commit b3d9093
  - intent: same-show unseen footage as holdout
  - action: Conversations with Coleman Klein-ep 1080p, labels reusable
  - outcome: 0 reads: x6 = pure bokeh; focus varies per episode (metric: reads=0)
  - branched from: r12-deno
- **r12-probes** [kept] commit b3d9093
  - intent: find in-focus shelves
  - action: Cowen / DarkHorse / P&P frame probes
  - outcome: curtain / painted backdrop / wood wall: no shelves (metric: 0/3 shelves)
  - branched from: r12-deno

## Round 13

- **r13-captions** [kept, impact 0.30] commit d81165d
  - intent: free ASR instead of Whisper
  - action: captions.py VTT parser (stage 9b)
  - outcome: captions spell Dilemma/Arday/Sowell where Whisper garbles (metric: spelling better)
  - branched from: r12-deno
- **r13-rawcap-neg** [dropped] commit d81165d
  - intent: can captions replace Whisper raw?
  - action: compile with captions as audio, classic n-gram scan
  - outcome: 18 verified NON-BOOKS (Soviet Union, Hillary Clinton...): noise was a precision filter (metric: 18 FPs)
  - branched from: r13-captions
- **r13-llmfix** [kept, impact 0.30] commit d81165d
  - intent: gate captions through an LLM
  - action: llmfix.extract_titles/correct_phrases, env-keyed, compile --llm-titles
  - outcome: tested mocked; passthrough without key (metric: 57 tests)
  - branched from: r13-rawcap-neg

## Round 14

- **r14-manual-llm** [kept, impact 0.11] commit 4c66525
  - intent: exercise the gate without a key
  - action: agent as the LLM over cue-filtered captions + garbled canons
  - outcome: 4 mentions incl. Black Rednecks inferred from context; corrections applied (metric: mentions=4)
  - branched from: r13-llmfix
- **r14-silence** [kept, impact 0.90] commit 4c66525
  - intent: gate must replace the firehose
  - action: --llm-titles silences raw n-gram scan
  - outcome: export list 6/6 real books vs 18 non-books raw (metric: precision 6/6)
  - branched from: r14-manual-llm
- **r14-scorer2** [kept, impact 0.11] commit 4c66525
  - intent: unverified garble masked verified forms
  - action: verified-evidence-preferred attribution in scorer
  - outcome: real-shelf verified .22 -> .33 (first movement since r5) (metric: v2 .33)
  - branched from: r14-manual-llm

## Round 16

- **r16-live** [bug] commit 7fe9cdb
  - intent: prove the repo is fully functional end-to-end, live
  - action: cut 60 s shelf slice (2540-2600 s of Test B 1080p) -> booksnap run
  - outcome: crash at final stage: run->compile Namespace has no llm_titles (round-13 wiring drift; unit tests call compile directly, so only a live run exposes it) (metric: 57 tests green while 'run' was broken)
  - branched from: r13-llmfix
- **r16-fix1** [kept, impact 0.35] commit 7fe9cdb
  - intent: repair run->compile wiring
  - action: pass llm_titles through; expose --llm-titles on run; 3 regression tests capture the compile Namespace via --resume replay
  - outcome: run reaches compile; flag default False, pass-through True (metric: tests 57 -> 60)
  - branched from: r16-live
- **r16-oom2** [bug] commit 7fe9cdb
  - intent: spine stage on 1080p band crop inside run
  - action: band 0.78, upscale 4 (hardcoded), 3 presets + stack
  - outcome: silent OOM kill; exit code masked by pipe to tail - re-confirmed the logged RAM dead end (band ROI / upscale<=3) and a second wiring gap: run lacks --spines-upscale (metric: no artifacts, 0-byte output)
  - branched from: r16-live
- **r16-fix2** [kept, impact 0.20] commit 7fe9cdb
  - intent: make RAM-safe spines reachable from run
  - action: --spines-upscale flag passes through to the spines stage; 1 regression test
  - outcome: upscale 2 completes in RAM budget (metric: tests 60 -> 61)
  - branched from: r16-oom2
- **r16-demo** [kept, impact 0.15] commit 7fe9cdb
  - intent: full live proof, no network
  - action: booksnap run slice --resume --spines-band 0.78 --spines-upscale 2 --spines-stack
  - outcome: exit 0: 14 segments, 13 reps, OCR, 3 covers, spine reads to 0.89 (CONVERSATIONS WIT+ COLEMAN), compile -> books_candidates.{json,md}; bench-spines on committed artifacts reproduces real shelf 0.56/0.33/0.33 exactly (metric: live end-to-end + offline repro both green)
  - branched from: r16-fix2

## Round 17

- **r17-selector** [kept, impact 0.30] commit round17
  - intent: solve the focus / selector frontier on shelf footage
  - action: measure Laplacian variance on cropped shelf ROI rather than whole frame; add --min-focus / --spines-min-focus threshold
  - outcome: bokeh crops measure 9.5-26.5 vs in-focus shelves 406.9-414.9 (15-40x contrast); whole frame had falsely scored 115-131 on bokeh due to foreground host texture (metric: 15-40x focus SNR; skips bokeh automatically)
  - branched from: o-heldout
- **r17-compound** [kept, impact 0.20] commit round17
  - intent: recover fused ALL-CAPS titles misclassified as author bands
  - action: integrate dp_split into author_spine; fix build_vocab so unsegmented caps tokens do not pollute vocabulary
  - outcome: LOCKEDIN recognized as compound title rather than author band; segments to 'LOCKED IN' (metric: LOCKEDIN unblocked)
  - branched from: r8-group
- **r17-authorpop** [kept, impact 0.20] commit round17
  - intent: clean titles with trailing mixed-case author bands
  - action: group_spine_reads pops mixed-case/compressed author bands (JamQW) from bottom of column
  - outcome: WEALIT AND PO freed from glued JamQW; yields clean title candidate (metric: Wealth, Poverty and Politics unblocked)
  - branched from: r8-group
- **r17-corroborate** [kept, impact 0.23] commit round17
  - intent: corroborate physical 2-word titles containing prepositions
  - action: _corroborated accepts 2-word titles with physical visual source (spine/cover); bench_spines includes verified titles in match
  - outcome: Locked in (Pfaff) and Wealth, Poverty and Politics (Sowell) verified; real shelf verified recall 0.33 -> 0.56, synth 0.79 -> 0.83 (metric: real shelf verified .33 -> .56, right_record .33 -> .56, synth .79 -> .83)
  - branched from: r11-weak

## Round 2

- **r2-audio** [kept] commit pre-6b4bf91
  - intent: spoken title evidence
  - action: chunked faster-whisper (5-min chunks after OOM) + cue regexes
  - outcome: OOM at full-file; chunking stable; cues precise (metric: audio_titles=3)
  - branched from: r1-seg

## Round 3

- **r3-gaz** [kept] commit fcb200e
  - intent: kill false positives in exports
  - action: OpenLibrary verification with fuzzy+phonetic matching
  - outcome: test-A exported precision 1/1; person/place guards tested (metric: precision=1/1)
  - branched from: r2-audio

## Round 4

- **r4-roadmap** [kept] commit f390b58
  - intent: make errors taxonomic, targets explicit
  - action: ROADMAP with error taxonomy + per-modality targets
  - outcome: adopted as project contract (metric: -)
  - branched from: r3-gaz

## Round 5

- **r5-bench** [kept] commit 6b4bf91
  - intent: first ground truth for the spine channel
  - action: human-labelled shelf (9 books) + directional bench scorer
  - outcome: recall surface .44 / verified .22 / right .22; circularity documented (metric: v2=.44/.22/.22)
  - branched from: r4-roadmap
- **r5-xref-veto** [kept, impact 0.25] commit 6b4bf91
  - intent: last exported FP on video B
  - action: crossref_is_serial vetoes journal mentions
  - outcome: Philosophical Psychology removed; precision 4/4 (metric: precision=4/4)
  - branched from: r3-gaz
- **r5-superres** [kept] commit 6b4bf91
  - intent: raise glyph resolution
  - action: selective upscale + enhancement presets (unsharp/clahe/binarize/lanczos/denoise)
  - outcome: reads 88 -> 276 candidates; verified unchanged (metric: reads=276)
  - branched from: r5-bench

## Round 6

- **r6-stitch** [kept] commit 61c48c6
  - intent: wider shelf via pan stitching
  - action: phase-correlation stitch_pan
  - outcome: measured ~0 px shift: shelf does not pan; honest negative (metric: shift=0px)
  - branched from: r5-superres
- **r6-roix6** [kept] commit 61c48c6
  - intent: rescue missed spines
  - action: per-spine ROI x6 experiment
  - outcome: 2/8 misses rescued; not integrated (complexity > gain) (metric: 2/8)
  - branched from: r5-superres
- **r6-vlm** [kept] commit 61c48c6
  - intent: second eyes on the shelf
  - action: vlm bundle/run/merge review loop with provenance
  - outcome: labelled-shelf verified 1.0 but circular by construction (metric: v2+vlm=1.0(circ))
  - branched from: r5-bench

## Round 7

- **r7-split** [kept] commit 61c48c6
  - intent: fused caps reads
  - action: splitwords.dp_split dictionary segmentation
  - outcome: clean fusions split; mangled refused; no live win yet (metric: 0 live wins)
  - branched from: r5-xref-veto
- **r7-authbands** [kept] commit 61c48c6
  - intent: author evidence without title claims
  - action: author_band rows in outputs + vlm verdicts
  - outcome: Dennett / Obama / Wilson surfaced (metric: bands=3)
  - branched from: r6-vlm
- **r7-heldout-jbp** [kept] commit 61c48c6
  - intent: out-of-sample precision check
  - action: JBP EP531 shelf run (bokeh + vertical spines)
  - outcome: 0 reads, 0 claims: guards hold out of sample (metric: reads=0)
  - branched from: r5-bench

## Round 8

- **r8-synth** [kept, impact 0.79] commit 550820d
  - intent: held-out bench without downloads
  - action: rendered degraded shelf + ground truth + committed OL cache
  - outcome: synth .79/.79/.79, bands 12/13; CI-gateable offline (metric: synth=.79)
  - branched from: r7-heldout-jbp
- **r8-group** [kept, impact 0.46] commit 550820d
  - intent: multi-word titles never became candidates
  - action: box w/h in reads + column grouping + ALL-CAPS band split
  - outcome: synth verified 0 -> .46 pre-scorer-fix; real shelves gain phrases (metric: synth .46->.79)
  - branched from: r8-synth
- **r8-spineonly** [kept, impact 0.40] commit 550820d
  - intent: shelf-only runs verified nothing
  - action: gazetteer guard was 'and audio'; now (audio or spines)
  - outcome: spine-only verification enabled (metric: gaz 0->21 synth)
  - branched from: r8-group
- **r8-scorer** [kept] commit 550820d
  - intent: scorer punished article/connector drops
  - action: connector/article-tolerant label sig + article-free nlabel
  - outcome: v2 surface .44 -> .56 (fairness, not pipeline) (metric: v2 surface .56)
  - branched from: r8-synth

## Round 9

- **r9-wide** [kept] commit f9512e5
  - intent: run grouping on the real shelf
  - action: re-fetched Test B 1080p; 165 reads; first real multi-word groups
  - outcome: verified .22 unchanged; ceiling localized to glyphs (metric: v2=.56/.22/.22)
  - branched from: r8-group
- **r9-budget** [kept, impact 0.11] commit f9512e5
  - intent: spine-dense runs starved
  - action: default gazetteer budget 60 -> 250
  - outcome: verified .11 -> .22 on identical reads (metric: .11->.22)
  - branched from: r9-wide

## Declared-open fronts

- **o-heldout** [open]
  - intent: title-level held-out shelf
  - action: search unlocked; focus is the selector
  - outcome: - (metric: -)
  - branched from: r12-heldout-ck
- **o-vertical** [open]
  - intent: vertical spine layout
  - action: rotated tiles + synthetic vertical bench
  - outcome: - (metric: -)
  - branched from: d-rotjbp
- **o-catalog** [open]
  - intent: coverage without network
  - action: bundled Wikidata-subset catalogue
  - outcome: - (metric: -)
  - branched from: r11-xref
- **o-llmkey** [open]
  - intent: automate the gate
  - action: any OpenAI-compatible LLM_API_KEY
  - outcome: - (metric: -)
  - branched from: r14-manual-llm
- **o-asr** [open]
  - intent: better ASR
  - action: whisper base -> small
  - outcome: compute-gated (metric: -)
  - branched from: r2-audio

## Dead ends (tried, dropped, why)

- **d-stack** [dropped]
  - intent: denoise via temporal median
  - action: stack_median across shot cuts
  - outcome: ghosting: invalid across cuts (metric: -)
  - branched from: r1-seg
- **d-vlmcrops** [dropped]
  - intent: VLM over OCR crops
  - action: bundle crops as images
  - outcome: dead end: bundles must ask open questions over shelf images (metric: -)
  - branched from: r6-vlm
- **d-rotjbp** [dropped]
  - intent: read vertical spines
  - action: rotated-tile OCR prototype considered
  - outcome: bokeh+vertical = zero signal; not built (metric: -)
  - branched from: r7-heldout-jbp
- **d-segdl** [dropped]
  - intent: short clips via --download-sections
  - action: yt-dlp sectioned download
  - outcome: imageio-ffmpeg segfault; download full + local cut (metric: -)
  - branched from: r12-deno
- **d-720** [dropped]
  - intent: reuse surviving 720p shelf clip
  - action: re-extract at 720p
  - outcome: 33 garbled reads: too weak; re-fetch 1080p instead (metric: reads=33)
  - branched from: r9-wide

## Ranked ideas (curated impact, evidence in metrics)

1. 0.90 `r11-weak` - same-title different-book is real [export clean]
1. 0.90 `r14-silence` - gate must replace the firehose [precision 6/6]
1. 0.79 `r8-synth` - held-out bench without downloads [synth=.79]
1. 0.70 `r12-deno` - YouTube search was bot-gated [search=unlocked]
1. 0.46 `r8-group` - multi-word titles never became candidates [synth .46->.79]
1. 0.40 `r8-spineonly` - shelf-only runs verified nothing [gaz 0->21 synth]
1. 0.35 `r16-fix1` - repair run->compile wiring [tests 57 -> 60]
1. 0.30 `r13-captions` - free ASR instead of Whisper [spelling better]
1. 0.30 `r13-llmfix` - gate captions through an LLM [57 tests]
1. 0.30 `r17-selector` - solve the focus / selector frontier on shelf footage [15-40x focus SNR; skips bokeh automatically]
1. 0.25 `r5-xref-veto` - last exported FP on video B [precision=4/4]
1. 0.23 `r17-corroborate` - corroborate physical 2-word titles containing prepositions [real shelf verified .33 -> .56, right_record .33 -> .56, synth .79 -> .83]
1. 0.20 `r16-fix2` - make RAM-safe spines reachable from run [tests 60 -> 61]
1. 0.20 `r17-compound` - recover fused ALL-CAPS titles misclassified as author bands [LOCKEDIN unblocked]
1. 0.20 `r17-authorpop` - clean titles with trailing mixed-case author bands [Wealth, Poverty and Politics unblocked]
1. 0.15 `r16-demo` - full live proof, no network [live end-to-end + offline repro both green]
1. 0.11 `r9-budget` - spine-dense runs starved [.11->.22]
1. 0.11 `r14-manual-llm` - exercise the gate without a key [mentions=4]
1. 0.11 `r14-scorer2` - unverified garble masked verified forms [v2 .33]
1. 0.05 `r10-degarble` - exact-text retrieval hides garbled books [+Obama]

## External prior art consulted

- **dream-rsi.com / zhengkid/Dream-RSI** (technical report): the logging
  discipline adopted here - history as a replay simulator, decisions recorded
  with realized outcomes, alternatives screened at zero executions.
- **sonoxo/GPT-DOUG-Dream-RSI** (fork of the above): adds a Zyra/XUNIA swarm
  ecosystem stack, a DOJ NSD compliance-mapping pipeline with tests, a 24h
  compliance-audit CI workflow on integration pushes, and a compliance domain
  map. Assessment for this repo: the audit-on-push pattern is already present
  here as the offline bench gates; the swarm stack and the DOJ mapper are
  domain-specific weight with no analogue in book extraction - NOT adapted.
  Meta-lesson recorded: stacking unrelated ecosystems onto a discovery repo
  produces noise commits; this log stays single-project.
- **Panniantong/Agent-Reach** (round 12): yt-dlp JS-runtime diagnosis adopted
  (deno + --js-runtimes), which unlocked YouTube search.
