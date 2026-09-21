"""Dev utility: warm the Crossref monograph fallback cache politely.

Crossref throttles bursts hard from shared IPs (measured: 5/10 successes at
0.5 s spacing), so the fallback cache is warmed offline-ish, once, with long
pauses, and the result is committed next to the compiled books json. Runs
and CI then verify crossref-backed phrases with zero network.

Usage: python bench/warm_crossref.py <spines.json> <cache.json> [pause]
"""
from __future__ import annotations

import json
import sys
import time

from booksnap.compile import group_spine_reads, spine_alts
from booksnap.gazetteer import _norm, crossref_book_docs, match_docs


def main(spines_json, cache_path, pause=2.0):
    cache = {}
    try:
        cache = json.load(open(cache_path))
    except Exception:
        pass
    spines = json.load(open(spines_json))
    phrases = []
    for g in group_spine_reads(spines):
        if len(g["text"].split()) < 2:
            continue
        entry = cache.get(_norm(g["text"])) or {}
        if match_docs(g["text"], entry.get("docs") or []):
            continue  # OpenLibrary already verifies this canon
        phrases.append(g["text"])
        phrases.extend(a for a in spine_alts(g)[:4] if len(a.split()) >= 2)
    done = 0
    for p in phrases:
        key = "_crdocs:" + _norm(p)
        if key in cache:
            continue
        crossref_book_docs(p, cache=cache)
        done += 1
        json.dump(cache, open(cache_path, "w"))
        print(f"{done}/{len(phrases)} {p!r} -> "
              f"{len(cache.get(key, {}).get('docs', []))} docs", flush=True)
        time.sleep(pause)
    json.dump(cache, open(cache_path, "w"))
    print("warmed", done, "new phrases;", len(cache), "cache keys")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
