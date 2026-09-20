"""Split fused ALL-CAPS OCR reads into word hypotheses ("LOSINCTHIRACE").

Spine OCR at video resolution often glues words together. Such a token cannot
be queried against a title index, which is a pure recall loss. Classic remedy:
dictionary segmentation by dynamic programming. The dictionary here is
domain-built rather than shipped: frequent English words plus every word the
run has already seen (transcript, bibliography, verified titles, spine reads),
so "LOSING", "THE", "RACE" are all in-vocabulary from the video itself.

Only clean fusions are recoverable - genuinely mangled glyphs
("ATESTAMENTTEHOPE", where OF became TE) stay mangled, and that is correct:
inventing words OCR never saw would trade recall for precision.
"""
from __future__ import annotations

import re

CORE = {
    "the", "of", "and", "a", "an", "in", "on", "for", "to", "with", "without",
    "from", "by", "at", "or", "as", "is", "are", "was", "were", "be", "been",
    "not", "no", "my", "his", "her", "their", "our", "your", "its", "it",
    "lost", "losing", "loss", "ground", "race", "war", "peace", "power",
    "pathos", "hope", "testament", "wealth", "poverty", "politics", "locked",
    "origin", "species", "nature", "mind", "body", "self", "sapiens",
    "victims", "conversations", "reality", "facing", "great", "awakening",
    "american", "americans", "african", "history", "brief", "humankind",
    "murray", "coates", "dennett", "obama", "sowell", "wilson", "harari",
    "book", "books", "years", "year", "trouble", "eight", "we", "were",
    "between", "world", "me", "picture", "dorian", "gray", "late",
    "admissions", "quick", "dead", "black", "white", "new", "old", "good",
    "evil", "beyond", "good", "moral", "ethics", "ethic", "psychology",
    "philosophy", "philosophical", "economics", "capital", "capitalism",
    "democracy", "freedom", "liberty", "justice", "law", "order", "chaos",
    "god", "religion", "science", "scientific", "evolution", "genetics",
    "intelligence", "iq", "brain", "conscious", "consciousness", "language",
    "thought", "thinking", "fast", "slow", "noise", "signal", "theory",
    "practice", "art", "arts", "way", "ways", "life", "lives", "living",
    "death", "dying", "time", "times", "age", "ages", "century", "centuries",
    "world", "worlds", "earth", "water", "fire", "air", "blood", "bone",
    "bones", "skin", "eye", "eyes", "hand", "hands", "head", "heart",
}


def build_vocab(extra_texts=()):
    vocab = set(CORE)
    for text in extra_texts:
        for w in re.findall(r"[a-z]+", (text or "").lower()):
            if 2 <= len(w) <= 24:
                vocab.add(w)
    return vocab


def dp_split(token, vocab, max_parts=4, min_part=2):
    """Best segmentation of a fused token, or None if it does not segment.

    Score favours fewer parts and longer in-vocabulary parts; every part must
    be in the vocabulary, so garbage cannot leak through.
    """
    t = (token or "").lower()
    if not t.isalpha() or len(t) < 6 or t in vocab:
        return None
    n = len(t)
    best = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(min_part, n + 1):
        for j in range(max(0, i - 24), i - min_part + 1):
            if best[j] is None:
                continue
            part = t[j:i]
            if part not in vocab:
                continue
            score = best[j][0] + len(part) - 1.5  # long words beat many splits
            cand = (score, best[j][1] + [part])
            if len(cand[1]) <= max_parts and (best[i] is None or cand[0] > best[i][0]):
                best[i] = cand
    if best[n] is None or len(best[n][1]) < 2:
        return None
    return " ".join(best[n][1])


def split_candidates(texts, vocab, k=2):
    """Split hypotheses for fused single-token ALL-CAPS reads."""
    out = []
    for text in texts:
        words = (text or "").split()
        for w in words:
            if len(w) >= 8 and w.isupper() and w.isalpha():
                s = dp_split(w, vocab)
                if s and s.upper() not in out:
                    out.append(s.upper())
                    if len(out) >= k * 4:
                        return out
    return out
