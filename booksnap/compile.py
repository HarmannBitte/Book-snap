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
import re

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

    Two-phase: match candidate words against the transcript vocabulary
    (ratio >= 0.85) to shortlist segments, then slide a character window of
    the candidate length over those segments.
    """
    c = cand_text.lower()
    cw = _words(c)
    if not cw:
        return None, 0.0
    vocab = {}
    for i, seg in enumerate(audio_segments):
        for w in set(_words(seg["text"])):
            vocab.setdefault(w, []).append(i)
    short = set()
    for w in cw:
        for v, idxs in vocab.items():
            if abs(len(v) - len(w)) <= 3 and difflib.SequenceMatcher(None, w, v,
                                                                     autojunk=False).ratio() >= 0.85:
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
            if r > best_r:
                best_r, best = r, audio_segments[i]["text"][start:start + n + 1].strip()
    return (best if best_r >= thr else None), round(best_r, 2)


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


def fuse_audio(candidates, audio_segments, hear_thr=0.75):
    for c in candidates:
        if not book_like(c):
            c.update(heard=False, heard_as=None, heard_score=0.0)
            continue
        phrase, r = heard_match(c.get("text") or c.get("ocr") or "", audio_segments, hear_thr)
        c["heard"] = phrase is not None
        c["heard_as"] = phrase
        c["heard_score"] = r
    return candidates


def compile_books(ocr_json, cover_manifest_json, cover_ocr_json, scroll_json,
                  out_json, out_md, audio_json=None, spines_json=None):
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
    for sp in spines:
        shown.append(dict(cover=None, seg=f"spine@{sp.get('t')}", ar=None,
                          ocr=sp["text"], text=sp["text"], source="spine",
                          conf=sp.get("conf")))
    for s_ in shown:
        s_.setdefault("source", "cover")
    match_shown_to_bibliography(shown, biblio)

    audio = json.load(open(audio_json))["segments"] if audio_json else []
    audio_cands = audio_title_candidates(audio) if audio else []
    if audio:
        fuse_audio(shown, audio)

    data = dict(bibliography=biblio, shown_covers=shown,
                shown_only=[s for s in shown if not s["cited"]],
                audio_titles=audio_cands,
                heard=[s for s in shown if s.get("heard")])
    json.dump(data, open(out_json, "w"), indent=1)

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
        f.write("\n## Visual candidates confirmed by audio\n\n")
        for s in data["heard"]:
            f.write(f"- {s['text']!r} heard as {s['heard_as']!r} "
                    f"(score {s['heard_score']})\n")
    return data
