"""Round 13: optional LLM pass over transcripts / garbled phrases.

YouTube auto-captions are free and spell proper nouns better than local
Whisper, but fed raw into the n-gram scan they verify 18 non-books
("Soviet Union", "Hillary Clinton", ...) on the labelled podcast - the
noisy Whisper transcript had been an accidental precision filter. So the
captions enter only through a constrained LLM stage (this module), behind
environment keys exactly like the VLM reviewer:

    LLM_API_KEY   required; absent -> callers fall back to the classic path
    LLM_BASE_URL  default https://api.openai.com/v1 (any OpenAI-compatible)
    LLM_MODEL     default gpt-4o-mini

Two jobs, both JSON-in/JSON-out, both spelling-only or extraction-only:

  extract_titles(segments)  -> [{"phrase", "title", "t"}]
      "which BOOK TITLES are mentioned, with the ASR spelling corrected"
  correct_phrases(phrases)  -> [corrected strings, same order/length]
      spelling-only repair of OCR/ASR garble ("LOSINC THE RACE")

Any network/parsing error degrades to the input (or []), never to a crash:
the gazetteer remains the only authority on what a book is.
"""
from __future__ import annotations

import json
import os
import urllib.request

SYS_EXTRACT = (
    "You extract mentions of BOOK TITLES from podcast transcripts. Return a "
    "JSON list of objects {\"phrase\": the transcript span, \"title\": the "
    "correctly spelled book title, \"t\": segment start seconds}. Include "
    "only books (never people, journals, places, laws, events). If none, "
    "return []. No prose, no markdown.")
SYS_CORRECT = (
    "You correct OCR/ASR spelling of book titles and author names. Return a "
    "JSON list of corrected strings, same order and length as the input. "
    "Change spelling only; never add, drop or reorder items; when unsure "
    "return the input string unchanged. No prose, no markdown.")


def _client():
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return key, base, model


def _chat(client, system, user, timeout=60):
    key, base, model = client
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(dict(model=model, temperature=0,
                             messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": user}])
                        ).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    return payload["choices"][0]["message"]["content"]


def _json_list(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def extract_titles(segments, chunk_segs=80):
    client = _client()
    if not client:
        return []
    out = []
    for i in range(0, len(segments), chunk_segs):
        part = segments[i:i + chunk_segs]
        user = json.dumps([{"t": s.get("t", s.get("start")), "text": s["text"]}
                           for s in part])
        try:
            got = _json_list(_chat(client, SYS_EXTRACT, user))
        except Exception:
            continue
        for g in got if isinstance(got, list) else []:
            if isinstance(g, dict) and g.get("title"):
                out.append(dict(phrase=g.get("phrase") or g["title"],
                                title=g["title"], t=g.get("t")))
    return out


def correct_phrases(phrases, chunk=40):
    client = _client()
    if not client:
        return list(phrases)
    out = list(phrases)
    for i in range(0, len(phrases), chunk):
        part = phrases[i:i + chunk]
        try:
            got = _json_list(_chat(client, SYS_CORRECT, json.dumps(part)))
        except Exception:
            continue
        if isinstance(got, list) and len(got) == len(part):
            out[i:i + chunk] = [g if isinstance(g, str) else p
                                for g, p in zip(got, part)]
    return out
