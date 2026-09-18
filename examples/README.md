# Validation run: "Biological Realism" (exquofonte)

Source: YouTube `11o3LTnGVa8`, 31:39, 1920×1080 @ 30 fps (≈56,997 frames).
Proxy: 480×270 @ 10 fps = 18,999 frames (single decode).

## Detector benchmark (this video)

Naive "compare to last kept frame" segmenters, best threshold per family:

| Detector | Segments | Median gap |
|---|---|---|
| pixel-diff (%px Δ>25) | 1284 | 0.1 s |
| edge-diff (Canny) | 565 | 0.1 s |
| block-MAD | 745 | 0.2 s |
| pHash (Hamming) | 504 | 0.7 s |
| RGB histogram | 489 | 0.6 s |
| FFmpeg `select=gt(scene,0.2)` | 746 | – |
| **book-snap stability-gated isolated-spike block-MAD** | **208** | 8.8 s (median segment) |

Cut precision vs a motion-spike oracle (0.3 s debounce, best per family):
block-MAD 0.685, histogram 0.622, pixel-diff 0.551, pHash 0.520.
The stability-gated detector is threshold-stable: 204–217 segments for spike
thresholds 1.0–3.0.

## Extraction results

* 208 representative full-res frames → OCR (RapidOCR, ~390 s).
* Scrolling end-bibliography recovered by dense 0.5 s sampling + stitching:
  83 unique lines → **19 books + 13 articles**.
* 9 book covers detected on screen and cropped:

| Book | Author | t (s) | In bibliography? |
|---|---|---|---|
| Born Together–Reared Apart | Nancy L. Segal | 595.7 | **no** |
| Blueprint: How DNA Makes Us Who We Are | Robert Plomin | 703.0 | yes |
| The Extended Phenotype | Richard Dawkins | 839.2 | yes |
| The Enigma of Reason | Mercier & Sperber | 899.9 | **no** |
| The Righteous Mind | Jonathan Haidt | 899.9 | yes |
| Thinking, Fast and Slow | Daniel Kahneman | 970.8 | yes |
| The Quest for Community | Robert A. Nisbet | 1311.8 | **no** |
| Multicultural America (Encyclopedia of the Newest Americans) | Ronald H. Bayor (ed.) | 1478.5 | **no** |
| Why Everyone (Else) Is a Hypocrite | Robert Kurzban | 1689.9 | yes |

→ 4 shown books never appear in the reference list (Segal, Mercier & Sperber,
Nisbet, Bayor). Union of cited + shown: **23 distinct books**.

Detection edge cases handled: side-by-side covers merge into one wide panel
(split at mid-line); the black Kurzban jacket is invisible to brightness-only
panel detection (caught by the dark-textured mask).
