"""Stage 7 - merge OCR artifacts into a candidate book list.

Heuristics (deliberately simple; final curation is manual or LLM-assisted):

  bibliography entries : stitched scroll lines are grouped into entries at
                         Chicago-style "Surname," line starts; an entry is
                         classed as an *article* when it contains quotation
                         marks or a journal marker, else as a *book*.
  on-screen covers     : panels with a book-like aspect ratio (0.5-0.9) whose
                         associated text (see covers.attach_panel_text) is
                         non-trivial become "shown" candidates.
  cross-linking        : shown covers are fuzzy-matched against bibliography
                         entries by significant-token overlap, so
                         "shown but never cited" books are flagged
                         automatically.

Output: books_candidates.json / .md with provenance (timestamp + channel).
"""
from __future__ import annotations

import difflib
import json
import os
import re

from .fuzz import fuzzy_score, jaro_winkler, phonetic
from .gazetteer import CONNECTORS

ENTRY_START = re.compile(r"^[A-Z][a-zA-Z'’-]+,")  # Chicago style: "Surname, First ..."
ARTICLE = re.compile(r'["“”]|journal|proceedings|proc\.|nature|science|psychology|'
                     r'behaviour|behavior|entropy|genetics|ssrn|vol\.|no\.', re.I)
STOP = {"the", "and", "for", "with", "from", "who", "how", "why", "what", "when",
        "their", "its", "our", "are", "was", "were", "been", "being", "have",
        "has", "had", "other", "else", "into", "about", "than", "then", "them"}


def parse_bibliography(lines):
    """Group stitched scroll lines into entries at 'Surname,' line starts."""
    entries, cur = [], []
    for e in lines:
        text = e["text"].strip()
        if text.lower() == "bibliography":
            continue
        if ENTRY_START.match(text) and cur:
            entries.append(" ".join(cur).strip())
            cur = [text]
        else:
            cur.append(text)
    if cur:
        entries.append(" ".join(cur).strip())
    out = []
    for text in entries:
        kind = "article" if ARTICLE.search(text) else "book"
        m = re.search(r"(1[6-9]\d{2}|20\d{2})", text)
        out.append(dict(text=text, kind=kind, year=int(m.group(1)) if m else None))
    return out


def _norm_q(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def tokens(s: str):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 3 and t not in STOP}


def match_shown_to_bibliography(shown, biblio, thr=0.45):
    """Flag each shown cover with the bibliography entry it matches (if any)."""
    for s in shown:
        a = tokens(s.get("text") or s.get("ocr") or "")
        best, best_score = None, 0.0
        for i, e in enumerate(biblio):
            b = tokens(e["text"])
            if not a or not b:
                continue
            score = len(a & b) / min(len(a), len(b))
            if score > best_score:
                best, best_score = i, score
        s["cited"] = best is not None and best_score >= thr
        s["match_score"] = round(best_score, 2)
        s["match"] = biblio[best]["text"] if s["cited"] else None
    return shown


CUE = re.compile(
    r"\b(?:in|reading|finished|finished reading|his|her|their|the|a)?\s*"
    r"(?:book|books|memoir|essay collection)\s+"
    r"(?:called|titled|by|named)?\s*"
    r"((?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|and|the|a|for|to|in)){0,5})",
)
CUE2 = re.compile(
    r"\b(?:writes|wrote|argues in|says in|author of|title of)\s+"
    r"((?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|and|the|a|for|to|in)){0,5})",
)


def audio_title_candidates(audio_segments):
    """Title-case phrases introduced by book-ish cues in the transcript."""
    seen, out = set(), []
    for seg in audio_segments:
        for rx in (CUE, CUE2):
            for m in rx.finditer(seg["text"]):
                phrase = m.group(1).strip(" .,;:")
                key = phrase.lower()
                if len(phrase) >= 4 and key not in seen:
                    seen.add(key)
                    out.append(dict(phrase=phrase, t=seg["start"]))
    return out


def _words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3]


def heard_match(cand_text, audio_segments, thr=0.75, max_segs=400):
    """Fuzzy-search the transcript for a spoken form of `cand_text`.

    Two-phase: shortlist segments whose vocabulary matches candidate words by
    spelling (ratio >= 0.85) OR by phonetic code (Metaphone equality, so
    "Goleman" shortlists a segment containing "Coleman"), then slide a
    character window over those segments. A window is scored as
    max(spelling ratio, phonetic Jaro-Winkler) - the phonetic term is only
    computed for near-miss windows to keep it cheap.
    """
    c = cand_text.lower()
    cw = _words(c)
    if not cw:
        return None, 0.0
    cp = {phonetic(w) for w in cw}
    vocab = {}
    for i, seg in enumerate(audio_segments):
        for w in set(_words(seg["text"])):
            vocab.setdefault(w, []).append(i)
    short = set()
    for w in cw:
        pw = phonetic(w)
        for v, idxs in vocab.items():
            if (abs(len(v) - len(w)) <= 3 and
                    (difflib.SequenceMatcher(None, w, v, autojunk=False).ratio() >= 0.85
                     or phonetic(v) == pw)):
                short.update(idxs)
    best, best_r = None, 0.0
    for i in sorted(short)[:max_segs]:
        s = audio_segments[i]["text"].lower()
        n = len(c)
        for start in range(0, max(1, len(s) - n + 1)):
            if start and s[start - 1] not in " \t,.;:()!\"'":
                continue  # word-boundary only
            win = s[start:start + n + 1]
            r = difflib.SequenceMatcher(None, c, win, autojunk=False).ratio()
            if 0.55 <= r < 0.98:  # near miss: allow a homophone match
                r = max(r, jaro_winkler(phonetic(c), phonetic(win)))
            if r > best_r:
                best_r, best = r, audio_segments[i]["text"][start:start + n + 1].strip()
    return (best if best_r >= thr else None), round(best_r, 2)


SURNAME_ONLY = None  # placeholder, replaced below


def author_spine(c):
    """A spine whose read is a single proper noun ("DENNETT", "COATES").

    On a real shelf that is the *author* band of the spine, not the title. Such
    reads are still evidence (they name a person whose books may be shelved)
    but they must not be exported as verified book titles.
    """
    if c.get("source") != "spine":
        return False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z.']*", c.get("text") or "")]
    if len(words) != 1:
        return False
    w = words[0]
    return w[0].isupper() and (w.isupper() or w[1:].islower()) and len(w) >= 4


def book_like(c):
    """Only fuse audio for candidates that could plausibly be a book title."""
    t = (c.get("text") or c.get("ocr") or "").strip()
    if c.get("source") == "spine":
        return 6 <= len(t) <= 40
    ar = c.get("ar")
    if not (ar and 0.5 <= ar <= 0.9):
        return False
    words = t.split()
    return 1 <= len(words) <= 8 and len(t) <= 60 and not t.endswith(".")


def _heard_is_meaningful(text, spoken, score, spelling_floor=0.75):
    """Phonetic matching makes single short words collide ('COATES' ~ 'cities').

    A heard-confirmation needs a near-exact spelling match, or >=2 significant
    words, or one long word that also matches on spelling - a homophone alone
    is not evidence. (Heard flags only matter for entries the gazetteer already
    verified, so this guard protects the `confirmed` tier, not recall of text.)
    """
    import difflib
    words = [w for w in _words(text) if w not in STOP]
    if not words:
        return False
    spelling = difflib.SequenceMatcher(None, text.lower(), (spoken or "").lower(),
                                       autojunk=False).ratio()
    if spelling >= 0.9 or len(words) >= 2:
        return True
    return len(words[0]) >= 9 and spelling >= spelling_floor


def fuse_audio(candidates, audio_segments, hear_thr=0.75):
    for c in candidates:
        if not book_like(c):
            c.update(heard=False, heard_as=None, heard_score=0.0)
            continue
        text = c.get("text") or c.get("ocr") or ""
        phrase, r = heard_match(text, audio_segments, hear_thr)
        if phrase is not None and not _heard_is_meaningful(text, phrase, r):
            phrase, r = None, r  # homophone-only hit on a short word: not evidence
        c["heard"] = phrase is not None
        c["heard_as"] = phrase
        c["heard_score"] = r
    return candidates


def tier(entry):
    """Corroboration tier for a gazetteer-verified candidate.

    confirmed : visual evidence (cover/spine) or the title was also heard
    verified  : spoken >=2 times, or spoken once inside a book-ish context
    weak      : single spoken mention with no context -> reported, not exported
    """
    if entry.get("source") or entry.get("heard"):
        return "confirmed"
    if entry.get("freq", 0) >= 2 or entry.get("context_ok"):
        return "verified"
    return "weak"


def dedupe(entries, thr=0.92):
    """Merge near-duplicate titles, keeping the best evidence and provenance."""
    out = []
    for e in entries:
        key = e.get("title") or e.get("phrase") or ""
        hit = next((o for o in out
                    if fuzzy_score(o.get("title") or o["phrase"], key) >= thr), None)
        if hit is None:
            e = dict(e)
            e["provenance"] = [e.get("phrase")]
            out.append(e)
            continue
        hit["provenance"] = sorted(set(hit["provenance"] + [e.get("phrase")]))
        hit["freq"] = max(hit.get("freq", 0), e.get("freq", 0))
        if (e.get("score") or 0) > (hit.get("score") or 0):
            for k in ("title", "authors", "year", "score", "match_type"):
                if e.get(k) is not None:
                    hit[k] = e[k]
        if e.get("source") and not hit.get("source"):
            hit["source"] = e["source"]
    return out


def _bibkey(e):
    a = (e.get("authors") or ["anon"])[0]
    surname = re.sub(r"[^a-z]", "", a.split()[-1].lower()) or "anon"
    return f"{surname}{e.get('year') or ''}"


def write_bibtex(entries, path):
    lines = []
    for e in entries:
        auth = " and ".join(e.get("authors") or []) or "Unknown"
        lines.append(f"@book{{{_bibkey(e)},\n"
                     f"  title  = {{{e.get('title') or e['phrase']}}},\n"
                     f"  author = {{{auth}}},\n"
                     f"  year   = {{{e.get('year') or ''}}},\n"
                     f"  note   = {{score={e.get('score')} via {e.get('match_type')};"
                     f" provenance={'; '.join(e.get('provenance') or [])}}}\n}}")
    open(path, "w").write("\n\n".join(lines) + ("\n" if lines else ""))


def write_ris(entries, path):
    out = []
    for e in entries:
        out.append("TY  - BOOK")
        out.append(f"TI  - {e.get('title') or e['phrase']}")
        for a in e.get("authors") or []:
            out.append(f"AU  - {a}")
        if e.get("year"):
            out.append(f"PY  - {e['year']}")
        out.append(f"N1  - score={e.get('score')} match={e.get('match_type')}")
        out.append("ER  - ")
        out.append("")  # blank line between RIS records
    open(path, "w").write("\n".join(out) + ("\n" if out else ""))


def clean_spine_phrase(text):
    """Strip the debris that spine OCR glues onto a title.

    "M LOSING GEOCND" -> "LOSING GEOCND", "M-N LOSISG GROUND" -> "LOSISG GROUND".
    Dropping leading 1-2 char junk tokens is what makes the *query* findable;
    the remaining typos are handled by phonetic alignment in the gazetteer.
    """
    words = [w for w in re.split(r"\s+", (text or "").strip())
             if re.sub(r"[^A-Za-z0-9]", "", w)]
    while words and len(re.sub(r"[^A-Za-z0-9]", "", words[0])) <= 2:
        words.pop(0)
    while words and len(re.sub(r"[^A-Za-z0-9]", "", words[-1])) <= 1:
        words.pop()
    # collapse OCR space garble inside a word: "Ma th" -> "Math". Short tokens
    # that are connectors ("of", "in") are real words and stay put.
    merged = []
    for w in words:
        bare = re.sub(r"[^A-Za-z0-9]", "", w)
        prev = re.sub(r"[^A-Za-z0-9]", "", merged[-1]) if merged else ""
        if prev and len(bare) <= 2 and len(prev) <= 2 and w.lower() not in CONNECTORS:
            merged[-1] += w  # fragment chain: "Ma" + "th" -> "Math"
        else:
            merged.append(w)
    return " ".join(merged).strip(" -,.")


def spine_alts(group, k=3, vocab=None):
    """Cleaned query variants for a spine group, most legible first.

    Legibility proxy: more words, then higher OCR confidence - the canonical
    (highest-confidence) read is often the least complete one.
    """
    canon = (group.get("text") or "").strip()
    confs = group.get("variant_confs") or {}
    variants = sorted(group.get("variants") or [],
                      key=lambda v: (-len(v.split()), -confs.get(v, 0.0)))
    out = []
    # near-miss variants first: the canonical (highest-conf) read is often a
    # garbled cousin of a cleaner lower-conf read of the same spine, and the
    # variants list can be long enough that the clean one was cut off before
    for v in sorted(group.get("variants") or [], key=lambda v: -confs.get(v, 0.0)):
        cv = clean_spine_phrase(v)
        if (cv and cv.lower() != canon.lower()
                and fuzzy_score(cv, canon) >= 0.9 and cv not in out):
            out.append(cv)
        if len(out) >= 3:
            break
    for v in variants[:8]:
        cv = clean_spine_phrase(v)
        if cv and cv.lower() != canon.lower() and cv not in out:
            out.append(cv)
        if len(out) >= k:
            break
    if vocab is not None:  # dictionary splits of fused caps tokens
        from .splitwords import split_candidates
        for cand in split_candidates([canon] + variants, vocab, k=1):
            if cand not in out and len(out) < k + 1:
                out.append(cand)
    # glyph-confusion corrections ("LOSINC THE RACE" -> "LOSING THE RACE"):
    # retrieval is exact-text, so one confused glyph otherwise hides a book.
    # Degarble the variants too: the canon may be two glyphs from the truth
    # while some variant is only one away.
    from .degarble import candidates as degarble_candidates
    for v in sorted(group.get("variants") or [], key=lambda v: -confs.get(v, 0.0))[:4]:
        cv = clean_spine_phrase(v) or v
        for cand in degarble_candidates(cv, k=3):
            if cand not in out and len(out) < k + 9:
                out.append(cand)
    for cand in degarble_candidates(canon, k=8):
        if cand not in out and len(out) < k + 9:
            out.append(cand)
    return out


def group_spine_reads(spines):
    """Join vertically stacked spine words into one phrase per book.

    Spine titles are typeset one word per line, and OCR returns each line as
    its own read - so multi-word titles never became candidates at all (the
    synthetic shelf bench, round 8, made this measurable: single-word titles
    scored, everything else could not). Reads whose boxes overlap horizontally
    and sit within a line gap vertically are joined top-to-bottom; a trailing
    ALL-CAPS single word (the author band) is split back off into its own
    read so author_spine() still sees it. Reads without box geometry (files
    written before this existed) pass through untouched.
    """
    cols = []
    for r in sorted(spines, key=lambda r: r.get("y", 0)):
        if not r.get("w") or not r.get("h"):
            cols.append([r])
            continue
        placed = False
        for c in cols:
            a = c[-1]
            if not a.get("w") or not a.get("h"):
                continue
            xov = min(a["x"] + a["w"], r["x"] + r["w"]) - max(a["x"], r["x"])
            gap = r["y"] - (a["y"] + a["h"])
            if xov > 0.5 * min(r["w"], a["w"]) and -0.2 * a["h"] <= gap <= 2.0 * a["h"]:
                c.append(r)
                placed = True
                break
        if not placed:
            cols.append([r])
    out = []
    for c in cols:
        c.sort(key=lambda r: r.get("y", 0))
        if len(c) > 1 and c[-1]["text"].replace(" ", "").isupper() \
                and c[-1]["text"].replace(" ", "").isalpha():
            out.append(c.pop())  # author band stays its own read
        if not c:
            continue
        if len(c) == 1:
            out.append(c[0])
            continue
        m = dict(c[0])
        m["text"] = " ".join(r["text"] for r in c)
        m["conf"] = round(sum(r["conf"] for r in c) / len(c), 3)
        m["grouped"] = len(c)
        out.append(m)
    return out


def cluster_spines(shown, thr=0.92):
    """Collapse OCR variants of the same spine ("Sapicns"/"Saptens"/"Sapiens").

    Keeps the highest-confidence reading as canonical and records the variants,
    so the gazetteer is queried once per physical book instead of once per
    mis-read. The threshold is deliberately tight (0.92): at 0.85 two different
    spines merged ("LOSING THE RACE" + "LOSING GROUND") and the group inherited
    the wrong book. Non-spine candidates pass through untouched.
    """
    out, groups = [], []
    for c in shown:
        if c.get("source") != "spine":
            out.append(c)
            continue
        text = c.get("text") or ""
        grp = next((g for g in groups if fuzzy_score(g["text"], text) >= thr), None)
        if grp is None:
            grp = dict(c)
            grp["variants"] = [text]
            grp["variant_confs"] = {text: c.get("conf") or 0.0}
            groups.append(grp)
            out.append(grp)
            continue
        grp["variants"] = sorted(set(grp["variants"] + [text]))
        grp.setdefault("variant_confs", {})[text] = max(
            grp["variant_confs"].get(text, 0.0), c.get("conf") or 0.0)
        if (c.get("conf") or 0) > (grp.get("conf") or 0):
            for k in ("text", "ocr", "conf", "preset", "t", "seg"):
                if c.get(k) is not None:
                    grp[k] = c[k] if k != "seg" else grp["seg"]
            grp["ocr"] = grp["text"]
    return out


def compile_books(ocr_json, cover_manifest_json, cover_ocr_json, scroll_json,
                  out_json, out_md, audio_json=None, spines_json=None,
                  gazetteer=False, max_queries=250, fuzzy_thr=0.86,
                  out_bib=None, out_ris=None):
    ocr = json.load(open(ocr_json)) if ocr_json else {}
    manifest = json.load(open(cover_manifest_json)) if cover_manifest_json else []
    cover_ocr = json.load(open(cover_ocr_json)) if cover_ocr_json else {}
    scroll = json.load(open(scroll_json)) if scroll_json else []

    biblio = parse_bibliography(scroll) if scroll else []

    shown = []
    for p in manifest:
        if not (0.5 <= p["ar"] <= 0.9):
            continue
        text = p.get("text") or (cover_ocr or {}).get(p["file"], "") or ""
        text = text.strip()
        if len(text) < 6:
            continue
        shown.append(dict(cover=p["file"], seg=p["seg"], ar=p["ar"], ocr=text, text=text))
    spines = json.load(open(spines_json)) if spines_json else []
    spines = group_spine_reads(spines)
    for sp in spines:
        shown.append(dict(cover=None, seg=f"spine@{sp.get('t')}", ar=None,
                          ocr=sp["text"], text=sp["text"], source="spine",
                          conf=sp.get("conf"), preset=sp.get("preset")))
    n_raw = len(spines)
    shown = cluster_spines(shown)
    for s_ in shown:
        s_.setdefault("source", "cover")
    match_shown_to_bibliography(shown, biblio)

    audio = json.load(open(audio_json))["segments"] if audio_json else []
    audio_cands = audio_title_candidates(audio) if audio else []
    if audio:
        fuse_audio(shown, audio)

    gaz = []
    if gazetteer and (audio or spines):
        # shelf-only runs (no audio track worth querying) still get their spine
        # phrases verified; before round 8 the gazetteer silently skipped them
        from .gazetteer import title_candidates, verify_all, make_cached_lookup
        # Proper nouns the speaker actually says: used to pick between
        # same-titled catalogue records ("Facing Reality": Murray vs Eccles).
        hints = {w.lower() for seg in audio
                 for w in re.findall(r"\b[A-Z][a-z]{2,}\b", seg["text"])}
        lookup = make_cached_lookup(out_json + ".olcache.json", fuzzy_thr=fuzzy_thr,
                                    author_hints=hints)
        # Cue-detected titles ("...his book Facing Reality...") are the highest
        # quality spoken evidence, so they lead the spoken queue ahead of the
        # frequency-sorted n-gram scan.
        cued = [dict(phrase=a["phrase"], freq=0, cued=True) for a in audio_cands]
        seen_cued = {c["phrase"].lower() for c in cued}
        spoken = cued + [c for c in title_candidates(audio)
                         if c["phrase"].lower() not in seen_cued]
        # Visual evidence is stronger than a spoken n-gram, but a wall of spine
        # OCR must not starve the audio channel: reserve 40 % of the query
        # budget for spoken candidates.
        from .splitwords import build_vocab
        vocab = build_vocab(
            [seg["text"] for seg in audio] +
            [e["text"] for e in biblio] +
            [s_.get("text") or "" for s_ in shown])
        visual = []
        for s_ in shown:
            t = (s_.get("text") or "").strip()
            if not t:
                continue
            if (s_.get("source") == "spine" and author_spine(s_)
                    and (s_.get("text") or "").isupper()):
                continue  # caps surname band: author evidence, not a title query
            alts = spine_alts(s_, vocab=vocab) if s_.get("source") == "spine" else []
            visual.append(dict(phrase=t, freq=0, source=s_.get("source"), alts=alts,
                               conf=s_.get("conf") or 0.0))
        # multi-word phrases are title-like; single tokens (often junk or
        # author bands) must not burn the shared alt-query budget first
        visual.sort(key=lambda v: (-len(v["phrase"].split()), -v["conf"]))
        seen = {v["phrase"].lower() for v in visual}
        spoken = [c for c in spoken if c["phrase"].lower() not in seen]
        # interleave so neither channel can starve the other
        queue, vi, si = [], 0, 0
        while len(queue) < max_queries and (vi < len(visual) or si < len(spoken)):
            if vi < len(visual):
                queue.append(visual[vi]); vi += 1
            if len(queue) >= max_queries:
                break
            if si < len(spoken):
                queue.append(spoken[si]); si += 1
        gaz = verify_all(queue, lookup=lookup, max_queries=max_queries,
                         segments=audio)
        nq = _norm_q
        for s_ in shown:
            t = nq(s_.get("text") or "")
            hit = next((g for g in gaz if g.get("verified") and
                        (nq(g["phrase"]) == t or (len(t) >= 8 and t in nq(g["phrase"])))),
                       None)
            if hit:
                s_["gazetteer"] = {k: hit[k] for k in ("title", "authors", "year") if k in hit}

    for s_ in shown:
        if author_spine(s_):
            s_["author_spine"] = True
    for g in gaz:
        if g.get("verified"):
            # a single proper noun on a spine is the author band unless the
            # matched book's main title IS that word ("Sapiens" -> Harari, but
            # "DENNETT" -> "Daniel Dennett" is the author band of a Dennett book)
            main = _norm_q((g.get("title") or "").split(":")[0].split("(")[0])
            if (g.get("source") == "spine"
                    and main != _norm_q(g["phrase"])
                    and author_spine(dict(source="spine",
                                          text=g.get("matched_as") or g["phrase"]))):
                g["status"] = "author"
            elif (g.get("match_type") == "crossref"
                  and not any(w.lower() in hints
                              for a in (g.get("authors") or [])
                              for w in re.findall(r"[A-Za-z']+", a)
                              if len(w) > 2)):
                # title-only Crossref monograph: same-title different-book is
                # real (two "Losing the Race" monographs exist) - report weak,
                # export only when a spoken author hint corroborates the record
                g["status"] = "weak"
            else:
                g["status"] = tier(g)
    gaz_verified = dedupe([g for g in gaz if g.get("verified")])
    exportable = [g for g in gaz_verified
                  if g["status"] in ("confirmed", "verified")]
    verified_norm = {_norm_q(g["phrase"]) for g in gaz}  # incl. weak: it is a title hit
    spoken_names = {w.lower() for seg in audio
                    for w in re.findall(r"\b[A-Z][a-z]{2,}\b", seg["text"])}
    author_reads = sorted(
        {g["phrase"] for g in gaz_verified if g["status"] == "author"} |
        {s_["text"] for s_ in shown
         if s_.get("author_spine") and (s_.get("conf") or 0) >= 0.75
         and _norm_q(s_["text"]) not in verified_norm
         # on shelf-only runs (no audio) the caps band is the only evidence;
         # with audio, corroboration by a spoken mention is still required
         and (not spoken_names or (s_["text"] or "").lower() in spoken_names)})
    if gazetteer:
        write_bibtex(exportable, out_bib or out_json.replace(".json", ".bib"))
        write_ris(exportable, out_ris or out_json.replace(".json", ".ris"))

    data = dict(spine_reads=n_raw if spines_json else 0,
                spine_groups=len([s for s in shown if s.get("source") == "spine"]),
                bibliography=biblio, shown_covers=shown,
                shown_only=[s for s in shown if not s["cited"]],
                audio_titles=audio_cands,
                heard=[s for s in shown if s.get("heard")],
                gazetteer=gaz_verified,
                gazetteer_weak=[g for g in gaz_verified if g["status"] == "weak"],
                author_spines=author_reads)
    json.dump(data, open(out_json, "w"), indent=1)

    out_md = out_md or os.devnull  # md report is optional
    with open(out_md, "w") as f:
        f.write("# Book candidates\n\n## Bibliography entries (scroll OCR)\n\n")
        for e in biblio:
            f.write(f"- [{e['kind']}] {e['text']}\n")
        f.write("\n## On-screen cover candidates (panel OCR)\n\n")
        for s in shown:
            tag = "cited" if s["cited"] else "SHOWN-ONLY"
            f.write(f"- [{tag}] `{s['cover']}` ({s['seg']}): {s['ocr']}\n")
        f.write("\n## Shown but never cited\n\n")
        for s in data["shown_only"]:
            f.write(f"- `{s['cover']}` ({s['seg']}): {s['ocr']}\n")
        f.write("\n## Book titles mentioned in audio\n\n")
        for a in data["audio_titles"]:
            f.write(f"- t={a['t']}: {a['phrase']}\n")
        f.write("\n## Gazetteer-verified book titles\n\n")
        for g in data.get("gazetteer", []):
            if g["status"] == "weak":
                continue
            auth = ", ".join(g.get("authors") or []) or "?"
            f.write(f"- [{g['status']}] {g['phrase']}  ->  {g.get('title')} "
                    f"({auth}, {g.get('year')})  score={g.get('score')} "
                    f"({g.get('match_type')})  freq={g.get('freq')}\n")
        f.write("\n## Author spines (evidence of shelved authors, not titles)\n\n")
        for a in data.get("author_spines", []):
            f.write(f"- {a}\n")
        f.write("\n## Weak (single uncorroborated mention - not exported)\n\n")
        for g in data.get("gazetteer", []):
            if g["status"] != "weak":
                continue
            f.write(f"- [weak] {g['phrase']} -> {g.get('title')} "
                    f"({', '.join(g.get('authors') or [])}, {g.get('year')})\n")
        f.write("\n## Visual candidates confirmed by audio\n\n")
        for s in data["heard"]:
            f.write(f"- {s['text']!r} heard as {s['heard_as']!r} "
                    f"(score {s['heard_score']})\n")
    return data
