"""Round 10: typo-tolerant catalogue retrieval for garbled spine OCR.

The real-shelf ceiling is glyph confusion: every read of one spine says
"LOSINC THE RACE", and OpenLibrary cannot retrieve documents for a string
it has never seen. The matcher is phonetic-tolerant but the *retrieval*
step is exact-text, so a single confused glyph hides a real book.

OCR glyph confusions observed on this project's shelves (and the usual
suspects) form symmetric pairs; a single substitution per candidate keeps
the query count linear in phrase length. Candidates are offered as spine
alts, and verify_all only spends queries on alts after the primary phrase
misses - clean reads cost nothing.
"""
from __future__ import annotations

import re

# symmetric single-glyph confusions seen in runs: LOSINC/LOSING (C,G),
# GARACK/BARACK and BAAACK/BARACK (G,B / A,R), VYCTIMS/VICTIMS (Y,I),
# EIGHT YESBS/YEARS (B,R), plus the classic OCR pairs.
PAIRS = (("C", "G"), ("G", "B"), ("B", "R"), ("A", "R"), ("Y", "I"),
         ("U", "V"), ("I", "L"), ("U", "L"), ("O", "Q"), ("D", "O"), ("F", "E"),
         ("M", "N"), ("S", "Z"))

_WORDS = re.compile(r"[A-Za-z]+")


def candidates(phrase, k=12):
    """Single-substitution degarble candidates, most-plausible order.

    Case is preserved positionally (spine OCR is usually all-caps; title
    case reads stay title case). Only one position is changed per
    candidate, so a phrase of n letters yields <= n * 1 candidates.
    """
    out = []
    # pair-major order: the most frequent confusions (C/G first) produce a
    # candidate before rare ones, so capped alt budgets see the useful ones
    for a, b in PAIRS:
        for m in _WORDS.finditer(phrase or ""):
            w = m.group(0)
            for i, ch in enumerate(w):
                up = ch.upper()
                for src, dst in ((a, b), (b, a)):
                    if up == src:
                        nw = w[:i] + (dst if ch.isupper() else dst.lower()) + w[i + 1:]
                        cand = phrase[:m.start()] + nw + phrase[m.end():]
                        if cand != phrase and cand not in out:
                            out.append(cand)
        if len(out) >= k:
            return out[:k]
    return out[:k]
