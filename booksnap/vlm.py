"""Stage 12 - vision-model review loop (the recall channel).

OCR on 10 px spine type caps recall around 0.2-0.4 (measured on the labelled
shelf). A vision model reading the crop does not have that ceiling - the
labels in bench/ were produced exactly this way, by an assistant looking at
the frame. This module makes that loop part of the pipeline instead of an
ad-hoc step:

  1. `booksnap vlm-bundle` writes review.json over shelf images (frames,
     stacked bands, panoramas - whatever the run produced): one open question
     per image, "list every book title/author you can read". Open questions,
     not OCR boxes: a spine OCR never read produces no crop, so box-level
     review could never recover the misses,
  2. any vision model fills it - an API, or an assistant in an interactive
     session (the bench/ labels were produced exactly this way),
  3. `booksnap vlm-merge` folds the filled review back in as `confirmed`
     entries with source "vlm" and full provenance, and rewrites .md/.bib/.ris.

Caveat that must survive refactors: on the shelf the labels came from, the
vlm channel's recall is 1.0 by construction (the reviewer IS the ground
truth). Its accuracy is only measurable on held-out shelves.

The merge never deletes pipeline results; it only adds and upgrades, and
every row says which channel produced it.
"""
from __future__ import annotations

import json
import os


def write_bundle(images, out_json):
    """One open reading question per shelf image."""
    items = [dict(id=i, image=img,
                  question="List every book title and every author band you "
                           "can read in this image. For each: title, author "
                           "(if legible), and rough box. Skip logos and set "
                           "dressing that are not books.")
             for i, img in enumerate(images)]
    json.dump(dict(items=items), open(out_json, "w"), indent=1)
    return items


def review_with_api(bundle_json, out_json, model=None, base_url=None,
                    api_key=None, timeout=120.0):
    """Fill a review bundle with any OpenAI-compatible vision endpoint.

    Activates only when credentials exist (arguments or VLM_API_KEY /
    VLM_BASE_URL env); the pipeline never calls a paid API by surprise. Each
    image goes out as a data URI with the bundle's open question, and the
    model's JSON answer is normalised into merge_review's entry format.
    """
    import base64
    import urllib.request
    api_key = api_key or os.environ.get("VLM_API_KEY")
    base_url = (base_url or os.environ.get("VLM_BASE_URL")
                or "https://api.openai.com/v1").rstrip("/")
    model = model or os.environ.get("VLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise SystemExit("no VLM_API_KEY set; fill the bundle by hand or "
                         "export the key to use an API reviewer")
    bundle = json.load(open(bundle_json))
    entries = []
    for item in bundle.get("items", []):
        with open(item["image"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = json.dumps(dict(
            model=model,
            messages=[dict(role="user", content=[
                dict(type="text", text=item["question"] +
                     " Answer as JSON: [{\"title\": ..., \"author\": ..., "
                     "\"kind\": \"book\"|\"author_band\"|\"other\"}]"),
                dict(type="image_url",
                     image_url=dict(url=f"data:image/png;base64,{b64}"))])],
            temperature=0)).encode()
        req = urllib.request.Request(
            base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.load(r)["choices"][0]["message"]["content"]
        try:
            rows = json.loads(content.strip().removeprefix("```json")
                              .removesuffix("```"))
        except Exception:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            kind = row.get("kind", "book")
            entries.append(dict(verdict="author_band" if kind == "author_band"
                                else "book" if kind == "book" else "other",
                                title=row.get("title"), author=row.get("author"),
                                image=item["image"], model=model))
    json.dump(dict(entries=entries), open(out_json, "w"), indent=1)
    return entries


def merge_review(review_json, books_json, out_json=None, out_md=None,
                 out_bib=None, out_ris=None):
    from .compile import tier, dedupe, write_bibtex, write_ris
    review = json.load(open(review_json))
    books = json.load(open(books_json))
    added = []
    for entry in review.get("entries", []):
        if entry.get("verdict") != "book" or not entry.get("title"):
            continue
        added.append(dict(phrase=entry["title"], title=entry["title"],
                          authors=[entry["author"]] if entry.get("author") else [],
                          year=entry.get("year"), score=1.0, match_type="vlm",
                          editions=None, source="vlm", status="confirmed",
                          provenance=[f"vlm:{entry.get('image', '?')}"],
                          ocr_hint=entry.get("ocr")))
    bands = []
    for entry in review.get("entries", []):
        if entry.get("verdict") == "author_band" and entry.get("title"):
            bands.append(dict(name=entry["title"], author=entry.get("author"),
                              provenance=f"vlm:{entry.get('image', '?')}"))
    seen = {b["name"].lower() for b in books.get("author_bands", [])}
    books["author_bands"] = books.get("author_bands", []) + [
        b for b in bands if b["name"].lower() not in seen]
    have = {(g.get("title") or "").lower() for g in books.get("gazetteer", [])}
    fresh = [a for a in added if a["title"].lower() not in have]
    books["gazetteer"] = books.get("gazetteer", []) + fresh
    books["gazetteer"] = dedupe(books["gazetteer"])
    for g in books["gazetteer"]:
        g.setdefault("status", tier(g))
    out_json = out_json or books_json
    json.dump(books, open(out_json, "w"), indent=1)
    exportable = [g for g in books["gazetteer"] if g["status"] in ("confirmed", "verified")]
    if out_bib:
        write_bibtex(exportable, out_bib)
    if out_ris:
        write_ris(exportable, out_ris)
    if out_md:
        with open(out_md, "a") as f:
            f.write("\n## Vision-model review (confirmed)\n\n")
            for g in fresh:
                f.write(f"- [vlm] {g['title']} ({', '.join(g.get('authors') or [])})"
                        f"  from {g['provenance'][0]}\n")
            if books.get("author_bands"):
                f.write("\n## Author bands seen on shelf (vision review)\n\n")
                for b in books["author_bands"]:
                    f.write(f"- {b['name']}  ({b['provenance']})\n")
    return fresh
