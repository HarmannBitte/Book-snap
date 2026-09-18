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


def _compile(args):
    from .compile import compile_books
    data = compile_books(args.ocr, args.manifest, args.cover_ocr, args.scroll,
                         args.out_json, args.out_md)
    print(f"bibliography={len(data['bibliography'])} shown_covers={len(data['shown_covers'])}")


def _run(args):
    work = args.workdir
    os.makedirs(work, exist_ok=True)
    feat = os.path.join(work, "feat.npz")
    cuts = os.path.join(work, "cuts.npy")
    reps = os.path.join(work, "reps")
    covers = os.path.join(work, "covers")

    _feat(argparse.Namespace(video=args.video, out=feat,
                             proxy_width=args.proxy_width, fps=args.fps))
    _seg(argparse.Namespace(features=feat, out=cuts, spike_thr=args.spike_thr,
                            stable_max=args.stable_max, window_s=args.window_s))
    _frames(argparse.Namespace(video=args.video, cuts=cuts, features=feat, out=reps))
    _ocr(argparse.Namespace(src=reps, out=os.path.join(work, "ocr.json"), pattern="seg_*.png"))
    _covers(argparse.Namespace(src=reps, out=covers, pattern="seg_*.png", ocr=True,
                               attach_ocr=os.path.join(work, "ocr.json")))
    _scroll(argparse.Namespace(video=args.video, winmed=cuts.replace(".npy", "_winmed.npy"),
                               features=feat, out=os.path.join(work, "scroll_lines.json"),
                               start=args.scroll_start, end=args.scroll_end,
                               step=args.scroll_step, probe_step=2.0,
                               min_lines=args.min_lines, fps=args.fps))
    if not args.skip_pdf:
        _pdf(argparse.Namespace(src=reps, out=os.path.join(work, "slides.pdf")))
    _compile(argparse.Namespace(
        ocr=os.path.join(work, "ocr.json"),
        manifest=os.path.join(covers, "manifest.json"),
        cover_ocr=os.path.join(covers, "cover_ocr.json"),
        scroll=os.path.join(work, "scroll_lines.json"),
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

    s = sub.add_parser("compile", help="merge artifacts into candidate book list")
    s.add_argument("--ocr"); s.add_argument("--manifest"); s.add_argument("--cover-ocr")
    s.add_argument("--scroll"); s.add_argument("--out-json"); s.add_argument("--out-md")
    s.set_defaults(fn=_compile)

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
    s.set_defaults(fn=_run)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
