"""Stage 9b - YouTube captions as a free, instant ASR alternative.

Round 13: local Whisper (base, beam 1) is slow, OOM-prone and garbles proper
nouns; YouTube's own auto-captions cost one yt-dlp call and no compute:

    yt-dlp --js-runtimes deno --skip-download --write-auto-subs \\
           --sub-langs en --sub-format vtt -o subs <url>

This module turns the roll-up VTT into the same segment schema the audio
channel produces, so `compile --audio` accepts it unchanged. Roll-up cues
repeat their previous lines; the parser diffs each cue against the last and
keeps only the new words, regrouping into ~10-word segments.

Captions are Google ASR, not truth: they garble too (differently). They are
evidence like any other channel - the gazetteer still decides.
"""
from __future__ import annotations

import json
import re

CUE = re.compile(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)[^\n]*\n(.*?)(?=\n\n|\Z)", re.S)
TAG = re.compile(r"<[^>]+>")


def _secs(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(path, words_per_seg=10):
    raw = open(path, encoding="utf-8", errors="replace").read()
    prev, buf, t0 = "", "", None
    segs = []
    for m in CUE.finditer(raw):
        start, end, body = _secs(m.group(1)), _secs(m.group(2)), m.group(3)
        text = re.sub(r"\s+", " ", TAG.sub(" ", body)).strip()
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)  # ASR stutter
        if not text:
            continue
        new = text[len(prev):] if prev and text.startswith(prev) else text
        prev = text
        new = new.strip()
        if not new:
            continue
        if t0 is None:
            t0 = start
        buf = (buf + " " + new).strip()
        if len(buf.split()) >= words_per_seg or (end - t0) > 6.0:
            segs.append(dict(start=round(t0, 2), end=round(end, 2),
                             t=round(t0, 2), text=buf))
            buf, t0 = "", None
    if buf:
        segs.append(dict(start=round(t0, 2), end=round(end, 2),
                         t=round(t0, 2), text=buf))
    return segs


def main(vtt_path, out_json):
    segs = parse_vtt(vtt_path)
    json.dump({"segments": segs}, open(out_json, "w"), indent=1)
    print(f"{len(segs)} segments -> {out_json}")
    return segs
