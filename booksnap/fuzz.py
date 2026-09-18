"""Fuzzy + phonetic string matching shared by the fusion and gazetteer stages.

Why: ASR mangles proper nouns ("Goleman" for "Coleman.", "Ethnic DeLema" for
"Ethnic Dilemma") and OCR mangles cover text, so exact/prefix comparison
between channels loses real matches. Two complementary scores:

  token_set_ratio  - order-insensitive word overlap, difflib-based, robust to
                     inserted/omitted words ("Facing Reality" vs
                     "Facing Reality: Why ...").
  phonetic score   - Jaro-Winkler over Metaphone codes, catches homophone-ish
                     ASR/OCR errors that share no spelling.

`fuzzy_score` returns the max of the two; callers pick their own threshold.
Pure stdlib: no rapidfuzz/jellyfish dependency.
"""
from __future__ import annotations

import re

_VOWELS = "aeiouy"
_DROP_INITIAL = (("kn", "n"), ("gn", "n"), ("pn", "n"), ("wr", "r"), ("ps", "s"),
                 ("ae", "e"), ("x", "s"))


def norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def metaphone(word):
    """Compact Metaphone: 'dilemma' and 'delema' both -> 'TLM'."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return ""
    for pre, rep in _DROP_INITIAL:
        if w.startswith(pre):
            w = rep + w[len(pre):]
            break
    w = re.sub(r"(.)\1+", r"\1", w)  # collapse doubles
    out, i, n = [], 0, len(w)
    while i < n:
        c, nxt = w[i], w[i + 1:i + 2]
        nxt2 = w[i + 1:i + 3]
        if c in "aeiou":
            if i == 0:
                out.append(c)
            i += 1
            continue
        if c == "b":
            if not (i == n - 1 and w[i - 1:i] == "m"):
                out.append("B")
        elif c == "c":
            if nxt2 in ("ia", "ih"):
                out.append("X")
            elif nxt == "h":
                out.append("X")
                i += 1
            else:
                out.append("K")
        elif c == "d":
            if nxt2 in ("ge", "gi", "gy"):
                out.append("J")
                i += 1
            else:
                out.append("T")
        elif c == "f":
            out.append("F")
        elif c == "g":
            if nxt == "h" and (i + 2 >= n or w[i + 2] not in _VOWELS):
                i += 2
                continue
            if nxt == "n" and i + 1 == n - 1:
                i += 2
                continue
            if nxt in "iey":
                out.append("J")
            else:
                out.append("K")
        elif c == "h":
            if i == 0 or w[i - 1] in _VOWELS and nxt in _VOWELS:
                out.append("H")
        elif c == "j":
            out.append("J")
        elif c == "k":
            out.append("K")
        elif c == "l":
            out.append("L")
        elif c == "m":
            out.append("M")
        elif c == "n":
            out.append("N")
        elif c == "p":
            if nxt == "h":
                out.append("F")
                i += 1
            else:
                out.append("B")
        elif c == "q":
            out.append("K")
        elif c == "r":
            out.append("R")
        elif c == "s":
            if nxt == "h" or nxt2 in ("io", "ia"):
                out.append("X")
                i += 1 if nxt == "h" else 0
            else:
                out.append("S")
        elif c == "t":
            if nxt == "h":
                out.append("0")
                i += 1
            elif nxt2 in ("ia", "io"):
                out.append("X")
            else:
                out.append("T")
        elif c == "v":
            out.append("F")
        elif c == "w":
            if nxt in _VOWELS:
                out.append("W")
        elif c == "x":
            out.append("KS")
        elif c == "z":
            out.append("S")
        i += 1
    return "".join(out)[:6]


def phonetic(s):
    return " ".join(metaphone(w) for w in re.findall(r"[a-z0-9']+", (s or "").lower()))


def jaro(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    win = max(len(a), len(b)) // 2 - 1
    if win < 0:
        win = 0
    fa = [False] * len(a)
    fb = [False] * len(b)
    m = 0
    for i, ca in enumerate(a):
        lo, hi = max(0, i - win), min(i + win + 1, len(b))
        for j in range(lo, hi):
            if fb[j] or b[j] != ca:
                continue
            fa[i] = fb[j] = True
            m += 1
            break
    if not m:
        return 0.0
    k = t = 0
    for i, ca in enumerate(a):
        if not fa[i]:
            continue
        while not fb[k]:
            k += 1
        if ca != b[k]:
            t += 1
        k += 1
    t //= 2
    return (m / len(a) + m / len(b) + (m - t) / m) / 3.0


def jaro_winkler(a, b, p=0.1):
    j = jaro(a, b)
    if j < 0.7:
        return j
    pre = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        pre += 1
    return j + pre * p * (1 - j)


def _words(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w]


def token_set_ratio(a, b):
    """Order-insensitive similarity in [0,1] (difflib over sorted token sets)."""
    import difflib
    A = " ".join(sorted(set(_words(a))))
    B = " ".join(sorted(set(_words(b))))
    if not A or not B:
        return 0.0
    if A == B:
        return 1.0
    inter = set(A.split()) & set(B.split())
    if inter:  # score on the shared prefix + remainder, like rapidfuzz
        base = " ".join(sorted(inter))
        rest_a = " ".join(sorted(set(A.split()) - inter))
        rest_b = " ".join(sorted(set(B.split()) - inter))
        s1 = difflib.SequenceMatcher(None, base, base + " " + rest_a,
                                     autojunk=False).ratio()
        s2 = difflib.SequenceMatcher(None, base, base + " " + rest_b,
                                     autojunk=False).ratio()
        s3 = difflib.SequenceMatcher(None, base + " " + rest_a,
                                     base + " " + rest_b, autojunk=False).ratio()
        return max(s1, s2, s3)
    return difflib.SequenceMatcher(None, A, B, autojunk=False).ratio()


def phonetic_score(a, b):
    return jaro_winkler(phonetic(a), phonetic(b))


def fuzzy_score(a, b):
    """Best of spelling-similarity and phonetic similarity."""
    return max(token_set_ratio(a, b), phonetic_score(a, b))


def similar(a, b, thr=0.88):
    return fuzzy_score(a, b) >= thr
