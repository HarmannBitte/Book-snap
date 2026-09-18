"""Stage 11 - gazetteer verification of book-title candidates.

Cue-regexes miss most spoken titles ("...as I say in *Facing Reality*..."),
so instead:

  1. extract title-case n-gram candidates from the transcript (2-6 words,
     lowercase connectors allowed, >=2 capitalised words, not a whole
     sentence),
  2. verify each candidate against a book gazetteer - OpenLibrary by default
     (`search.json?title=`), accepting a hit only when the normalised
     gazetteer title contains/equals the normalised candidate,
  3. return candidates with bibliographic metadata (authors, first year).

Person names ("Nathan Cofnas") and show titles rarely match as *titles* in
the gazetteer, so verification doubles as a person/show filter. The lookup
function is injectable so tests never touch the network.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.parse
import urllib.request

BOOK_WORDS = {"book", "books", "memoir", "essay", "writes", "wrote", "writing",
              "author", "read", "reading", "published", "publication", "chapter",
              "title", "titled", "called", "edition", "press", "argues", "says",
              "said", "footnote", "bibliography", "cite", "cites", "monograph"}
ARTICLES = ("the", "a", "an")
CONNECTORS = {"of", "and", "the", "for", "to", "in", "a", "an", "on", "with",
              "without", "against", "beyond", "between", "under", "over", "or",
              "from", "up", "down", "out", "its", "his", "her", "their"}
RUN = re.compile(
    r"\b([A-Z][\w'’-]*(?:\s+(?:[A-Z][\w'’-]*|" +
    "|".join(sorted(CONNECTORS)) + r")){1,6})")
SENTENCE_END = re.compile(r"[.!?]")


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def title_candidates(segments, min_freq=1, max_len=48):
    """Title-case n-grams from a transcript, with frequencies."""
    freq = {}
    for seg in segments:
        text = seg["text"]
        for sent in re.split(SENTENCE_END, text):
            for m in RUN.finditer(sent):
                words = m.group(1).strip(" ,;:\"'").split()
                while len(words) > 1 and words[-1].lower() in CONNECTORS:
                    words.pop()  # drop trailing connectors
                phrase = " ".join(words)
                words = phrase.split()
                caps = sum(1 for w in words if w[0].isupper())
                if caps < 2 or len(phrase) > max_len:
                    continue
                # reject runs that are just the sentence start + one cap word
                if len(words) == 2 and m.start() == 0 and caps == 2:
                    continue
                freq[phrase] = freq.get(phrase, 0) + 1
    return [dict(phrase=p, freq=f) for p, f in
            sorted(freq.items(), key=lambda kv: -kv[1]) if f >= min_freq]


def openlibrary_lookup(phrase, timeout=5.0):
    """Return bibliographic metadata if OpenLibrary knows this title."""
    url = ("https://openlibrary.org/search.json?limit=5&fields=title,title_suggest,"
           "author_name,first_publish_year&title=" + urllib.parse.quote(phrase))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            docs = json.load(r).get("docs", [])
    except Exception:
        return None
    n = _norm(phrase)
    for d in docs:
        for field in ("title", "title_suggest"):
            t = d.get(field) or ""
            nt = _norm(t)
            for art in ARTICLES:  # allow "The X" to match candidate "X"
                if nt.startswith(art):
                    nt = nt[len(art):]
                    break
            if "|" in t or not d.get("author_name") or not d.get("first_publish_year"):
                continue  # catalogue junk record
            if (nt == n or nt.startswith(n)
                    or (len(n) >= 10 and
                        difflib.SequenceMatcher(None, n, nt, autojunk=False).ratio() >= 0.9)):
                authors = d.get("author_name", [])[:3]
                if _is_author(phrase, authors):
                    continue  # person name, not a title
                return dict(title=t, authors=authors,
                            year=d.get("first_publish_year"))
    return None


def _is_author(phrase, authors, thr=0.9):
    n = _norm(phrase)
    for a in authors or []:
        if difflib.SequenceMatcher(None, n, _norm(a), autojunk=False).ratio() >= thr:
            return True
    return False


def in_book_context(phrase, segments, window=80):
    """True if any transcript occurrence sits near a book-ish word."""
    for seg in segments:
        text = seg["text"]
        low = text.lower()
        start = 0
        while True:
            i = low.find(phrase.lower(), start)
            if i < 0:
                break
            ctx = set(re.findall(r"[a-z]+", low[max(0, i - window):i + len(phrase) + window]))
            if ctx & BOOK_WORDS:
                return True
            start = i + 1
    return False


def make_cached_lookup(cache_path):
    """Wrap openlibrary_lookup with a JSON disk cache (rate-limit friendly)."""
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}

    def lookup(phrase):
        key = _norm(phrase)
        if key not in cache:
            cache[key] = openlibrary_lookup(phrase)
            try:
                json.dump(cache, open(cache_path, "w"))
            except Exception:
                pass
        return cache[key] or None
    return lookup


def verify(candidates, lookup=openlibrary_lookup, max_queries=60, pause=0.25):
    out, done = [], 0
    for c in candidates:
        if done >= max_queries:
            break
        meta = lookup(c["phrase"])
        done += 1
        time.sleep(pause)
        if meta:
            out.append(dict(**c, **meta))
    return out


def verify_all(candidates, lookup=openlibrary_lookup, max_queries=60, pause=0.25,
               segments=None):
    """Verify candidates; audio-derived ones additionally need book context.

    Visual candidates (source set, freq==0) skip the context test - a cover or
    spine *is* its own context.
    """
    out, done = [], 0
    for c in candidates:
        c2 = dict(c)
        if done >= max_queries:
            c2.update(verified=False, skipped=True)
            out.append(c2); continue
        meta = lookup(c["phrase"])
        done += 1
        time.sleep(pause)
        ok = bool(meta)
        if ok and segments is not None and not c.get("source") and c.get("freq", 0) > 0:
            ok = in_book_context(c["phrase"], segments)
        c2.update(verified=ok, **(meta if ok else {}))
        out.append(c2)
    return out
