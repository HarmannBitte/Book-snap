"""Score the spine channel against a labelled shelf (bench/*.json).

The labels come from a human/assistant visual read of an upscaled shelf frame -
the only ground truth this project has for physical books. Scoring is fuzzy
(token alignment, same rule the pipeline uses), and it distinguishes three
things a book-list extractor must not confuse:

  books        the title was found AND matched to the right catalogue record
  author_bands a surname read off a spine, reported as an author, not a title
  missed       a labelled book the pipeline never surfaced

Run: booksnap bench-spines bench/spine_labels_v2.json runs/v2/spines1080.json \
     runs/v2/books_1080.json
"""
from __future__ import annotations

import json

from .fuzz import fuzzy_score, norm
from .gazetteer import _align


def _sig(words):
    return [w for w in words if len(w) > 2]


def match_label(label, texts, thr=0.7):
    """Best spine read / verified phrase for a labelled title.

    Deliberately stricter and *directional* compared with the pipeline's own
    matcher: every significant word of the LABEL must align inside the read,
    the read may add at most one word, or the space-free normals must be equal
    ("Locked In" == "LOCKEDIN"). A loose bidirectional matcher would credit
    "On the Origin of Species" to a misread "Sapiens" and lie about recall.
    """
    words = _sig([w for w in label.lower().replace(",", " ").split()])
    nlabel = norm(label)
    best, best_t = 0.0, None
    for t in texts:
        tw = _sig([w for w in t.lower().replace(",", " ").split()])
        if not tw or not words:
            continue
        if nlabel and norm(t) == nlabel:
            s = 1.0
        elif (_align(words, set(tw), 0.8) and len(tw) - len(words) <= 1):
            s = max(fuzzy_score(label, t), 0.8)
        else:
            continue
        if s > best:
            best, best_t = s, t
    return (best_t if best >= thr else None), round(best, 2)


def score(labels_path, spines_json, books_json):
    labels = json.load(open(labels_path))
    spines = json.load(open(spines_json)) if spines_json else []
    books = json.load(open(books_json)) if books_json else {}
    reads = [r["text"] for r in spines]
    verified = {g["phrase"]: g for g in books.get("gazetteer", [])}
    author_reads = set(books.get("author_spines", []))
    all_texts = reads + list(verified) + list(author_reads)

    rows = []
    for b in labels.get("books", []):
        if not b.get("confident"):
            continue
        hit, s = match_label(b["title"], all_texts)
        g = verified.get(hit) if hit in verified else None
        right_record = bool(g) and (
            (b.get("author") or "").split()[-1].lower() in
            " ".join(g.get("authors") or []).lower()
            or fuzzy_score(b["title"], g.get("title") or "") >= 0.8)
        rows.append(dict(label=b["title"], found=hit, score=s,
                         verified=bool(g), right_record=right_record))
    arows = []
    for a in labels.get("author_bands", []):
        if not a.get("confident"):
            continue
        hit, s = match_label(a["name"], all_texts)
        arows.append(dict(label=a["name"], found=hit, score=s,
                          reported_author=bool(hit) and (hit in author_reads or
                                                         any(v.get("status") == "author"
                                                             for v in verified.values()
                                                             if v["phrase"] == hit))))
    n = len(rows)
    found = sum(1 for r in rows if r["found"])
    rec = sum(1 for r in rows if r["verified"])
    right = sum(1 for r in rows if r["right_record"])
    out = dict(books=rows, author_bands=arows,
               recall_surface=round(found / n, 2) if n else None,
               recall_verified=round(rec / n, 2) if n else None,
               recall_right_record=round(right / n, 2) if n else None)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="booksnap bench-spines")
    p.add_argument("labels")
    p.add_argument("spines")
    p.add_argument("books")
    a = p.parse_args(argv)
    out = score(a.labels, a.spines, a.books)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
