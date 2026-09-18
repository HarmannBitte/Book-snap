"""Stage 7 - merge OCR artifacts into a candidate book list.

Heuristics (deliberately simple; final curation is manual or LLM-assisted):

  bibliography entries : stitched scroll lines are grouped into entries at
                         lines ending in a 4-digit year; an entry is classed
                         as an *article* when it contains quotation marks or a
                         journal marker, else as a *book*.
  on-screen covers     : panels with a book-like aspect ratio (0.5-0.9) whose
                         OCR text is non-trivial become "shown" candidates.

Output: books_candidates.json / .md with provenance (timestamp + channel).
"""
from __future__ import annotations

import json
import re

YEAR = re.compile(r"(1[6-9]\d{2}|20\d{2})\s*\.?$")
ENTRY_START = re.compile(r"^[A-Z][a-zA-Z'’-]+,")  # Chicago style: "Surname, First ..."
ARTICLE = re.compile(r'["“”]|journal|proceedings|proc\.|nature|science|psychology|'
                     r'behaviour|behavior|entropy|genetics|ssrn|vol\.|no\.', re.I)


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
        text = (cover_ocr or {}).get(p["file"], "").strip()
        if len(text) < 6:
            continue
        shown.append(dict(cover=p["file"], seg=p["seg"], ar=p["ar"], ocr=text))

    data = dict(bibliography=biblio, shown_covers=shown)
    json.dump(data, open(out_json, "w"), indent=1)

    with open(out_md, "w") as f:
        f.write("# Book candidates\n\n## Bibliography entries (scroll OCR)\n\n")
        for e in biblio:
            f.write(f"- [{e['kind']}] {e['text']}\n")
        f.write("\n## On-screen cover candidates (panel OCR)\n\n")
        for s in shown:
            f.write(f"- `{s['cover']}` ({s['seg']}): {s['ocr']}\n")
    return data
