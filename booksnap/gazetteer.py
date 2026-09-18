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
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .fuzz import fuzzy_score, metaphone

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
# Heuristic place-name reject list: cities/regions are frequently *prefixes* of
# real titles ("New York City in 1979"), so they pass title matching while never
# being what the speaker meant. Extend as needed; it only demotes, never deletes.
PLACES = {
    "new york", "new york city", "london", "paris", "berlin", "chicago",
    "boston", "los angeles", "san francisco", "washington", "united states",
    "united kingdom", "great britain", "europe", "america", "africa", "asia",
    "china", "india", "japan", "russia", "germany", "france", "spain",
    "italy", "canada", "australia", "brazil", "mexico", "california", "texas",
    "virginia", "harlem", "brooklyn", "manhattan", "oxford", "cambridge",
    "princeton", "yale", "harvard", "stanford", "the west", "the south",
}


REGION = re.compile(
    r"^(north|south|east|west|central|sub-?saharan|south(?:east|west)|north(?:east|west)|"
    r"middle|near|far|greater|lower|upper|new|old|latin)\s+"
    r"(africa(n|ns)?|asia(n|ns)?|europe(an|ans)?|america(n|ns)?|india|indies|china|"
    r"japan|east|world|hemisphere)\b", re.I)
DEMOGRAPHIC = re.compile(
    r"^(african|black|white|asian|native|european|latin|jewish|muslim|christian|"
    r"immigrant|urban|rural|poor|young|old|black|hispanic)\s+"
    r"(americans?|africans?|asians?|europeans?|people|men|women|children|students|"
    r"youth|families|workers|voters)\b", re.I)
INSTITUTION_TAIL = {"university", "college", "school", "institute", "institute of",
                    "hospital", "foundation", "corporation", "company", "society",
                    "academy", "museum", "department", "ministry", "bank"}


def looks_like_place(phrase):
    return phrase.strip().lower().rstrip(".,;:") in PLACES


def looks_like_region(phrase):
    """'West Africa', 'South East Asia', 'African Americans' - geography and
    demographics, not titles. These were the last verified-tier false positives
    measured on the two test videos."""
    p = (phrase or "").strip().rstrip(".,;:")
    return bool(REGION.match(p) or DEMOGRAPHIC.match(p))


def looks_like_institution(phrase):
    """'Cambridge University', 'University of Austin' - not book titles."""
    words = [w.lower() for w in re.findall(r"[a-z]+", (phrase or "").lower())]
    if not 2 <= len(words) <= 4:
        return False
    return words[-1] in INSTITUTION_TAIL or words[0] in INSTITUTION_TAIL


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


def _fetch_json(url, timeout=5.0, tries=3):
    """GET with exponential backoff + jitter (OpenLibrary throttles bursts)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep((2 ** k) + random.random())
                continue
            return None
        except Exception:
            if k < tries - 1:
                time.sleep((2 ** k) + random.random())
                continue
            return None
    return None


FIELDS = "title,title_suggest,author_name,first_publish_year,edition_count,ratings_average"
JUNK_AUTHOR = re.compile(r"phone number|database|catalog|list|directory|index", re.I)
CAND_STOP = {"with", "and", "the", "of", "for", "in", "on", "a", "an", "to", "from",
             "his", "her", "their", "our", "your", "my", "it", "its", "is", "are",
             "was", "were", "that", "this", "these", "those", "but", "or", "as",
             "at", "by", "be", "we", "you", "he", "she", "they", "i", "not", "no"}


def fetch_docs(phrase, timeout=5.0):
    """Raw OpenLibrary records for a title query, or None if the fetch failed.

    None vs [] matters: an empty result is knowledge ("no such title") and may
    be cached, a failure is not and must be retried.
    """
    url = ("https://openlibrary.org/search.json?limit=8&fields=" + FIELDS +
           "&title=" + urllib.parse.quote(phrase))
    data = _fetch_json(url, timeout=timeout)
    if data is None:
        return None
    return data.get("docs", [])


def looks_like_person(s):
    """'Charles T. Murray', 'Daniel Dennett' -> True; 'Losing Ground' -> False."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z.']*", s or "")]
    if not 2 <= len(words) <= 4:
        return False
    if any(w.lower() in CONNECTORS or w.lower() in CAND_STOP for w in words):
        return False
    if not all(w[0].isupper() for w in words):
        return False
    return any(len(w.rstrip(".")) == 1 for w in words)  # middle initial


def _strip_article(nt):
    for art in ARTICLES:
        if nt.startswith(art):
            return nt[len(art):]
    return nt


RULE_RANK = {"exact": 0, "prefix": 1, "fuzzy": 2}


def _align(sig_words, title_words, thr=0.85):
    """Every significant candidate word must map to a distinct title word, by
    spelling (ratio >= thr) or by phonetic code (>=4 chars), and the title may
    add at most two words (subtitle). Recovers OCR/ASR damage: 'LOSISG GROUND'
    aligns to 'Losing Ground', 'Ethnic DeLema' to 'The Ethnic Dilemma'."""
    used, hits = set(), 0
    for w in sig_words:
        wl = w.lower()
        for t in title_words:
            if t in used:
                continue
            if (wl == t or fuzzy_score(wl, t) >= thr
                    or (len(wl) >= 4 and len(t) >= 4 and metaphone(wl) == metaphone(t))):
                used.add(t)
                hits += 1
                break
    return hits == len(sig_words) and len(title_words) - len(used) <= 2


def match_docs(phrase, docs, fuzzy_thr=0.86, min_editions=2, author_hints=None):
    """Decide whether any record is this phrase *as a book title*.

    Rules, in order of strictness:
      exact  normalised equality (article-insensitive)
      prefix candidate is the title's opening (>=2 tokens, or one long token
             that is exactly the title's first word - "Sapiens" for
             "Sapiens: A Brief History")
      fuzzy  fuzzy_score >= fuzzy_thr with at most one extra/missing token
    Always rejected: junk catalogue records, person-name titles (a surname on a
    spine is not a title), place names, and candidate phrases that are only
    stopwords. Fragments ("AND PO") are rejected too: prefix/fuzzy matches need
    >=2 significant tokens, and weaker rules demand more catalogue support
    (min_editions rises exact 2 -> prefix/fuzzy 3). Ties break on rule
    strictness, then edition count.
    """
    if (looks_like_place(phrase) or looks_like_institution(phrase)
            or looks_like_region(phrase)):
        return None
    if not (phrase or "").isupper() and looks_like_person(phrase):
        return None  # a spoken person name, not a title
    words = [w for w in re.findall(r"[A-Za-z0-9']+", phrase or "")]
    if not words or all(w.lower() in CAND_STOP for w in words):
        return None
    n = _norm(phrase)
    sig = [w for w in words if w.lower() not in CAND_STOP and w.lower() not in CONNECTORS]
    best = None
    for d in docs:
        authors = [a for a in (d.get("author_name") or []) if not JUNK_AUTHOR.search(a)]
        year = d.get("first_publish_year")
        if not authors or not year or year < 1450:
            continue  # catalogue junk / undated record
        if (d.get("edition_count") or 1) < min_editions:
            continue
        if _is_author(phrase, authors):
            continue  # person name, not a title
        for field in ("title", "title_suggest"):
            t = d.get(field) or ""
            if "|" in t or looks_like_person(t):
                continue
            nt = _strip_article(_norm(t))
            twords = set(re.findall(r"[a-z0-9]+", t.lower()))
            pwords = set(w.lower() for w in words)
            score = fuzzy_score(phrase, t)
            editions = d.get("edition_count") or 1
            if nt == n:
                rule, need = "exact", min_editions
            elif nt.startswith(n) and len(sig) >= 2:
                rule, score, need = "prefix", 0.99, min_editions + 1
            elif (len(words) == 1 and len(words[0]) >= 6 and len(sig) == 1
                  and score >= fuzzy_thr
                  and t.lower().split(":")[0].strip() == words[0].lower()):
                rule, need = "prefix", min_editions + 1  # "Sapiens" -> "Sapiens: A ..."
            elif len(sig) >= 2 and _align(sig, twords, fuzzy_thr - 0.01):
                # every significant token aligns (spelling or phonetics) and the
                # title adds at most a subtitle - the global ratio is reported,
                # not used as a gate, since alignment is the stronger test
                rule, need = "fuzzy", min_editions + 1
            else:
                continue
            if editions < need:
                continue
            cand = dict(title=t, authors=authors[:3], year=year,
                        score=round(score, 3), match_type=rule, editions=editions,
                        low_support=False)
            hinted = 0
            if author_hints:
                hinted = -sum(1 for a in authors
                              if any(w.lower() in author_hints
                                     for w in re.findall(r"[A-Za-z']+", a) if len(w) > 2))
            key = (RULE_RANK[rule], hinted, -editions, -cand["score"])
            if best is None or key < best[0]:
                best = (key, cand)
    if best:
        best[1]["low_support"] = (best[1].get("editions") or 1) < 3
    return best[1] if best else None


def openlibrary_lookup(phrase, timeout=5.0, fuzzy_thr=0.86, min_editions=2):
    """Fetch + match in one call (the injectable default for verify_all)."""
    return match_docs(phrase, fetch_docs(phrase, timeout=timeout),
                      fuzzy_thr=fuzzy_thr, min_editions=min_editions)


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


def make_cached_lookup(cache_path, fuzzy_thr=0.86, min_editions=2,
                       strong_editions=3, author_hints=None):
    """Disk-cached lookup. The cache stores *raw* OpenLibrary records, not
    match decisions, so matching rules can be retuned without re-querying."""
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}

    def lookup(phrase):
        key = _norm(phrase)
        entry = cache.get(key)
        if entry is None:
            docs = fetch_docs(phrase)
            if docs is None:  # failed fetch: do not poison the cache
                return match_docs(phrase, [], fuzzy_thr=fuzzy_thr,
                                  min_editions=min_editions, author_hints=author_hints)
            entry = dict(docs=docs)
            cache[key] = entry
            try:
                json.dump(cache, open(cache_path, "w"))
            except Exception:
                pass
        m = match_docs(phrase, entry.get("docs") or [], fuzzy_thr=fuzzy_thr,
                       min_editions=min_editions, author_hints=author_hints)
        if m:
            m["low_support"] = (m.get("editions") or 1) < strong_editions
        return m
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


def _corroborated(c, ctx):
    """A match to a thinly-held catalogue record (<3 editions) is only believed
    when something else supports it: >=2 significant words in the phrase, or a
    repeat mention, or a cue/context, or visual evidence.

    This is what separates 'Facing Reality' (cued, 2 words -> keep) from
    'LOCKEDIN' (one mangled caps token from a spine -> drop).
    """
    words = [w for w in re.findall(r"[A-Za-z0-9']+", c.get("phrase") or "")
             if w.lower() not in CAND_STOP and w.lower() not in CONNECTORS]
    return len(words) >= 2 and bool(c.get("source") or c.get("cued") or
                                    c.get("freq", 0) >= 2 or ctx)


def verify_all(candidates, lookup=openlibrary_lookup, max_queries=60, pause=0.25,
               segments=None, alt_budget=None):
    """Verify candidates against the gazetteer.

    Book context is recorded as `context_ok`, not used as a gate: a match with
    no context is still a real match, it is just weaker evidence. The caller
    (compile.tier) turns that into confirmed/verified/weak and only the first
    two are exported to BibTeX/RIS.
    """
    out, done, alts_done = [], 0, 0
    alt_budget = max_queries // 2 if alt_budget is None else alt_budget
    for c in candidates:
        c2 = dict(c)
        if done >= max_queries:
            c2.update(verified=False, skipped=True)
            out.append(c2); continue
        meta = lookup(c["phrase"])
        done += 1
        matched_as = c["phrase"]
        for alt in (c.get("alts") or []):  # OCR variants / cleaned reads
            if meta or alts_done >= alt_budget:
                break
            meta = lookup(alt)
            alts_done += 1
            time.sleep(pause)
            if meta:
                matched_as = alt
        ok = bool(meta)
        if ok and matched_as != c["phrase"]:
            c2["matched_as"] = matched_as
        ctx = True
        if ok and segments is not None and not c.get("source") and c.get("freq", 0) > 0:
            ctx = in_book_context(c["phrase"], segments)
        if ok and meta.get("low_support") and not _corroborated(c, ctx):
            ok = False  # thin catalogue record + thin evidence: do not export
        c2.update(verified=ok, context_ok=ctx, **(meta if meta else {}))
        out.append(c2)
    return out
