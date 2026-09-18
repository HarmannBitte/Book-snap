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


def compile_books(ocr_json, cover_manifest_json, cover_ocr_json, scroll_json,
                  out_json, out_md):
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
    match_shown_to_bibliography(shown, biblio)

    data = dict(bibliography=biblio, shown_covers=shown,
                shown_only=[s for s in shown if not s["cited"]])
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
    return data
