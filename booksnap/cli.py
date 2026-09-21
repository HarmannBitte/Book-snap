"""Command-line interface: `booksnap <stage> ...` or `booksnap run ...`."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _feat(args):
    from .features import build_features
    n = build_features(args.video, args.out, args.proxy_width, args.fps)
    print(f"cached {n} proxy frames -> {args.out}")


def _seg(args):
    from .segment import segment
    d = np.load(args.features)
    cuts, win = segment(d["blocks"], float(d["fps"][0]), args.spike_thr,
                        args.stable_max, args.window_s)
    np.save(args.out, cuts)
    np.save(args.out.replace(".npy", "_winmed.npy"), win)
    print(f"cuts={len(cuts)} segments={len(cuts) + 1}")


def _frames(args):
    from .frames import extract_frames
    cuts = np.load(args.cuts)
    win = np.load(args.cuts.replace(".npy", "_winmed.npy"))
    d = np.load(args.features)
    times = extract_frames(args.video, cuts, win, float(d["fps"][0]), args.out)
    print(f"wrote {len(times)} representative frames -> {args.out}")


def _ocr(args):
    from .ocr import ocr_directory
    res = ocr_directory(args.src, args.out, args.pattern)
    print(f"ocr'd {len(res)} images -> {args.out}")


def _covers(args):
    from .covers import detect_covers
    m = detect_covers(args.src, args.out, args.pattern)
    if args.attach_ocr and os.path.exists(args.attach_ocr):
        from .covers import attach_panel_text
        attach_panel_text(m, json.load(open(args.attach_ocr)))
        json.dump(m, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"panels={len(m)} -> {args.out}")
    if args.ocr:
        from .ocr import ocr_directory
        ocr_directory(args.out, os.path.join(args.out, "cover_ocr.json"),
                      pattern="seg_*_r*.png")


def _scroll(args):
    from .scroll import capture_scroll
    from .ocr import RapidOCR
    win = np.load(args.winmed)
    d = np.load(args.features) if args.features else None
    fps = float(d["fps"][0]) if d is not None else args.fps
    ocr = RapidOCR()
    ordered, ranges = capture_scroll(
        args.video, ocr, win, fps, args.out,
        start=args.start, end=args.end, step=args.step,
        probe_step=args.probe_step, min_lines=args.min_lines)
    print(f"scroll windows={[(round(a,1), round(b,1)) for a, b in ranges]} "
          f"unique_lines={len(ordered)} -> {args.out}")


def _pdf(args):
    from .pdf import make_pdf
    n = make_pdf(args.src, args.out, args.width, args.quality)
    print(f"pdf pages={n} -> {args.out}")


def _audio(args):
    from .audio import transcribe
    segs = transcribe(args.audio, args.out, args.model)
    print(f"transcript segments={len(segs)} -> {args.out}")


def _spines(args):
    from .spines import extract_spines
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    presets = tuple(p for p in (args.presets or "unsharp").split(",") if p)
    min_focus = getattr(args, "min_focus", 0.0)
    out = extract_spines(args.video, args.out, start=args.start, end=args.end,
                         step=args.step, topk=args.topk, roi=roi, band=args.band,
                         upscale=args.upscale, presets=presets, stack=args.stack,
                         min_len=args.min_len, panorama=args.panorama,
                         min_focus=min_focus)
    strong = sum(1 for r in out if r["conf"] >= 0.8)
    print(f"spine reads={len(out)} (conf>=0.80: {strong}) -> {args.out}")
    for r in out[:15]:
        print(f"   {r['conf']:.2f} [{r.get('preset')}] {r['text']}")


def _vlm_bundle(args):
    from .vlm import write_bundle
    items = write_bundle(args.images, args.out)
    print(f"review bundle with {len(items)} image(s) -> {args.out}")


def _vlm_merge(args):
    from .vlm import merge_review
    fresh = merge_review(args.review, args.books, out_json=args.out_json,
                         out_md=args.out_md, out_bib=args.out_bib,
                         out_ris=args.out_ris)
    print(f"merged {len(fresh)} vision-model entries")
    for g in fresh:
        print(f"   [vlm] {g['title']} ({', '.join(g.get('authors') or [])})")


def _bench_spines(args):
    from .bench_spines import score
    import json as _json
    out = score(args.labels, args.spines, args.books)
    print(_json.dumps(out, indent=1))


def _compile(args):
    from .compile import compile_books
    data = compile_books(args.ocr, args.manifest, args.cover_ocr, args.scroll,
                         args.out_json, args.out_md,
                         audio_json=args.audio, spines_json=args.spines,
                         gazetteer=args.gazetteer, max_queries=args.max_queries,
                                   llm_titles=args.llm_titles,
                         fuzzy_thr=args.fuzzy_thr,
                         out_bib=args.out_bib, out_ris=args.out_ris)
    tiers = {}
    for g in data["gazetteer"]:
        tiers[g["status"]] = tiers.get(g["status"], 0) + 1
    tstr = " ".join(f"{k}={v}" for k, v in sorted(tiers.items())) or "-"
    print(f"bibliography={len(data['bibliography'])} shown={len(data['shown_covers'])} "
          f"audio_titles={len(data['audio_titles'])} heard={len(data['heard'])} "
          f"gazetteer={len(data['gazetteer'])} ({tstr})")


def _have(path, resume):
    """Resume support: skip a stage whose artifact already exists."""
    ok = resume and os.path.exists(path) and os.path.getsize(path) > 0
    if ok:
        print(f"resume: reusing {path}")
    return ok


def _run(args):
    work = args.workdir
    resume = getattr(args, "resume", False)
    os.makedirs(work, exist_ok=True)
    feat = os.path.join(work, "feat.npz")
    cuts = os.path.join(work, "cuts.npy")
    reps = os.path.join(work, "reps")
    covers = os.path.join(work, "covers")

    if not _have(feat, resume):
        _feat(argparse.Namespace(video=args.video, out=feat,
                                 proxy_width=args.proxy_width, fps=args.fps))
    if not _have(cuts, resume):
        _seg(argparse.Namespace(features=feat, out=cuts, spike_thr=args.spike_thr,
                                stable_max=args.stable_max, window_s=args.window_s))
    if resume and os.path.isdir(reps) and os.listdir(reps):
        print(f"resume: reusing {reps}/ ({len(os.listdir(reps))} frames)")
    else:
        _frames(argparse.Namespace(video=args.video, cuts=cuts, features=feat, out=reps))
    ocr_json = os.path.join(work, "ocr.json")
    if not _have(ocr_json, resume):
        _ocr(argparse.Namespace(src=reps, out=ocr_json, pattern="seg_*.png"))
    if not _have(os.path.join(covers, "manifest.json"), resume):
        _covers(argparse.Namespace(src=reps, out=covers, pattern="seg_*.png", ocr=True,
                                   attach_ocr=ocr_json))
    _scroll(argparse.Namespace(video=args.video, winmed=cuts.replace(".npy", "_winmed.npy"),
                               features=feat, out=os.path.join(work, "scroll_lines.json"),
                               start=args.scroll_start, end=args.scroll_end,
                               step=args.scroll_step, probe_step=2.0,
                               min_lines=args.min_lines, fps=args.fps))
    if not args.skip_pdf:
        _pdf(argparse.Namespace(src=reps, out=os.path.join(work, "slides.pdf"),
                                width=1280, quality=4))
    if args.audio and not _have(os.path.join(work, "transcript.json"), resume):
        _audio(argparse.Namespace(audio=args.audio, out=os.path.join(work, "transcript.json"),
                                  model=args.audio_model))
    if (args.spines_start is not None and args.spines_end is not None
            and not _have(os.path.join(work, "spines.json"), resume)):
        _spines(argparse.Namespace(video=args.video, out=os.path.join(work, "spines.json"),
                                   start=args.spines_start, end=args.spines_end,
                                   step=1.0, topk=3, roi=args.spines_roi,
                                   band=args.spines_band,
                                   upscale=getattr(args, "spines_upscale", 4),
                                   presets=args.spines_presets, stack=args.spines_stack,
                                   min_len=3, panorama=args.spines_stack,
                                   min_focus=getattr(args, "spines_min_focus", 0.0)))
    _compile(argparse.Namespace(
        ocr=os.path.join(work, "ocr.json"),
        manifest=os.path.join(covers, "manifest.json"),
        cover_ocr=os.path.join(covers, "cover_ocr.json"),
        scroll=os.path.join(work, "scroll_lines.json"),
        audio=os.path.join(work, "transcript.json") if args.audio else None,
        spines=os.path.join(work, "spines.json") if args.spines_start is not None else None,
        gazetteer=args.gazetteer, max_queries=args.max_queries,
        llm_titles=getattr(args, "llm_titles", False),
        fuzzy_thr=args.fuzzy_thr, out_bib=None, out_ris=None,
        out_json=os.path.join(work, "books_candidates.json"),
        out_md=os.path.join(work, "books_candidates.md")))
    print(f"done. artifacts in {work}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="booksnap",
                                description="Slide & book-cover extraction from videos.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("features", help="decode once -> feature cache (.npz)")
    s.add_argument("video"); s.add_argument("out")
    s.add_argument("--proxy-width", type=int, default=480)
    s.add_argument("--fps", type=float, default=10.0)
    s.set_defaults(fn=_feat)

    s = sub.add_parser("segment", help="stability-gated isolated-spike segmentation")
    s.add_argument("features"); s.add_argument("out")
    s.add_argument("--spike-thr", type=float, default=1.5)
    s.add_argument("--stable-max", type=float, default=1.0)
    s.add_argument("--window-s", type=float, default=2.0)
    s.set_defaults(fn=_seg)

    s = sub.add_parser("frames", help="full-res representative frame per segment")
    s.add_argument("video"); s.add_argument("cuts"); s.add_argument("features"); s.add_argument("out")
    s.set_defaults(fn=_frames)

    s = sub.add_parser("ocr", help="OCR a directory of images")
    s.add_argument("src"); s.add_argument("out")
    s.add_argument("--pattern", default="seg_*.png")
    s.set_defaults(fn=_ocr)

    s = sub.add_parser("covers", help="detect embedded image panels / book covers")
    s.add_argument("src"); s.add_argument("out")
    s.add_argument("--pattern", default="seg_*.png")
    s.add_argument("--ocr", action="store_true")
    s.add_argument("--attach-ocr", help="full-frame ocr.json to associate panel captions")
    s.set_defaults(fn=_covers)

    s = sub.add_parser("scroll", help="recover scrolling credits/bibliography")
    s.add_argument("video"); s.add_argument("winmed"); s.add_argument("out")
    s.add_argument("--features")
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--start", type=float); s.add_argument("--end", type=float)
    s.add_argument("--step", type=float, default=0.5)
    s.add_argument("--probe-step", type=float, default=2.0)
    s.add_argument("--min-lines", type=int, default=12)
    s.set_defaults(fn=_scroll)

    s = sub.add_parser("audio", help="ASR the soundtrack (faster-whisper)")
    s.add_argument("audio"); s.add_argument("out")
    s.add_argument("--model", default="base",
                   help="whisper size; 'small'/'medium' fix proper nouns but "
                        "need more RAM (chunked, so ~1 GB ok for 'small')")
    s.add_argument("--beam-size", type=int, default=1,
                   help="beam >1 improves garbled titles at ~beam x runtime")
    s.set_defaults(fn=_audio)

    s = sub.add_parser("captions", help="parse a YouTube VTT into audio-schema segments")
    s.add_argument("vtt"); s.add_argument("out")
    s.set_defaults(fn=lambda a: __import__("booksnap.captions", fromlist=["main"]).main(a.vtt, a.out))

    s = sub.add_parser("spines", help="extract book-spine text from shelf footage")
    s.add_argument("video"); s.add_argument("out")
    s.add_argument("--start", type=float); s.add_argument("--end", type=float)
    s.add_argument("--step", type=float, default=1.0)
    s.add_argument("--topk", type=int, default=3)
    s.add_argument("--roi", help="x,y,w,h crop instead of top band")
    s.add_argument("--band", type=float, default=0.22)
    s.add_argument("--upscale", type=int, default=4)
    s.add_argument("--presets", default="unsharp",
                   help="comma-separated enhancement presets: "
                        "lanczos,unsharp,clahe,denoise,binarize")
    s.add_argument("--stack", action="store_true",
                   help="median-stack the sharpest frames before enhancing")
    s.add_argument("--panorama", action="store_true",
                   help="stitch the camera pan into one wide shelf image")
    s.add_argument("--min-len", type=int, default=3)
    s.add_argument("--min-focus", type=float, default=0.0,
                   help="minimum shelf crop sharpness; skips bokeh/blurred footage")
    s.set_defaults(fn=_spines)

    s = sub.add_parser("compile", help="merge artifacts into candidate book list")
    s.add_argument("--ocr"); s.add_argument("--manifest"); s.add_argument("--cover-ocr")
    s.add_argument("--scroll"); s.add_argument("--out-json"); s.add_argument("--out-md")
    s.add_argument("--audio"); s.add_argument("--spines")
    s.add_argument("--gazetteer", action="store_true")
    s.add_argument("--max-queries", type=int, default=250,
                   help="OpenLibrary query budget; spine-dense shelves need hundreds")
    s.add_argument("--fuzzy-thr", type=float, default=0.86)
    s.add_argument("--llm-titles", action="store_true",
                   help="route transcript/spine phrases through the optional LLM gate (LLM_API_KEY)")
    s.add_argument("--out-bib"); s.add_argument("--out-ris")
    s.set_defaults(fn=_compile)

    s = sub.add_parser("vlm-bundle", help="write a vision-model review bundle")
    s.add_argument("out"); s.add_argument("images", nargs="+")
    s.set_defaults(fn=_vlm_bundle)

    s = sub.add_parser("vlm-run", help="fill a review bundle via a vision API")
    s.add_argument("bundle"); s.add_argument("out")
    s.add_argument("--model"); s.add_argument("--base-url")
    s.set_defaults(fn=lambda a: print(
        f"{len(__import__('booksnap.vlm', fromlist=['review_with_api'])
              .review_with_api(a.bundle, a.out, a.model, a.base_url))} entries"))

    s = sub.add_parser("vlm-merge", help="merge a filled vision review")
    s.add_argument("review"); s.add_argument("books")
    s.add_argument("--out-json"); s.add_argument("--out-md")
    s.add_argument("--out-bib"); s.add_argument("--out-ris")
    s.set_defaults(fn=_vlm_merge)

    s = sub.add_parser("bench-spines", help="score the spine channel vs labels")
    s.add_argument("labels"); s.add_argument("spines"); s.add_argument("books")
    s.set_defaults(fn=_bench_spines)

    s = sub.add_parser("pdf", help="render representative frames as a PDF")
    s.add_argument("src"); s.add_argument("out")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--quality", type=int, default=4)
    s.set_defaults(fn=_pdf)

    s = sub.add_parser("run", help="run the whole pipeline")
    s.add_argument("video")
    s.add_argument("--workdir", default="booksnap_out")
    s.add_argument("--proxy-width", type=int, default=480)
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--spike-thr", type=float, default=1.5)
    s.add_argument("--stable-max", type=float, default=1.0)
    s.add_argument("--window-s", type=float, default=2.0)
    s.add_argument("--scroll-start", type=float); s.add_argument("--scroll-end", type=float)
    s.add_argument("--scroll-step", type=float, default=0.5)
    s.add_argument("--min-lines", type=int, default=12)
    s.add_argument("--skip-pdf", action="store_true")
    s.add_argument("--audio", help="audio file (opus/m4a/wav) to ASR")
    s.add_argument("--audio-model", default="base")
    s.add_argument("--spines-start", type=float); s.add_argument("--spines-end", type=float)
    s.add_argument("--spines-roi"); s.add_argument("--spines-band", type=float, default=0.22)
    s.add_argument("--spines-upscale", type=int, default=4,
                   help="OCR upscale factor; use 2-3 on 1080p band crops to stay in RAM budget")
    s.add_argument("--spines-presets", default="unsharp")
    s.add_argument("--spines-stack", action="store_true")
    s.add_argument("--spines-min-focus", type=float, default=0.0,
                   help="minimum shelf crop sharpness; skips bokeh/blurred footage")
    s.add_argument("--gazetteer", action="store_true")
    s.add_argument("--max-queries", type=int, default=250,
                   help="OpenLibrary query budget; spine-dense shelves need hundreds")
    s.add_argument("--fuzzy-thr", type=float, default=0.86)
    s.add_argument("--llm-titles", action="store_true",
                   help="route transcript/spine phrases through the optional LLM gate (LLM_API_KEY)")
    s.add_argument("--resume", action="store_true",
                   help="skip stages whose artifacts already exist in --workdir")
    s.set_defaults(fn=_run)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
