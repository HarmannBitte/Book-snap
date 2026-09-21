"""Regression tests for `booksnap run` stage wiring.

Found live (2026-09-21): `run` crashed at its final compile stage with
AttributeError: 'Namespace' object has no attribute 'llm_titles' - the flag was
added to the standalone `compile` subparser in round 13 but the Namespace that
`run` constructs internally was never updated. Unit tests covered compile
directly, so only an end-to-end run could catch the drift. These tests replay
`run` against pre-seeded artifacts (--resume) and capture the Namespace handed
to _compile, guarding every attribute it reads.
"""
import json
import os

from booksnap import cli

# every attribute _compile reads off its Namespace
COMPILE_ATTRS = [
    "ocr", "manifest", "cover_ocr", "scroll", "audio", "spines",
    "gazetteer", "max_queries", "llm_titles", "fuzzy_thr",
    "out_bib", "out_ris", "out_json", "out_md",
]


def _stub_artifacts(wd):
    """Minimal artifacts so --resume skips every heavy stage."""
    os.makedirs(os.path.join(wd, "reps"), exist_ok=True)
    os.makedirs(os.path.join(wd, "covers"), exist_ok=True)
    with open(os.path.join(wd, "feat.npz"), "wb") as f:
        f.write(b"x")
    with open(os.path.join(wd, "cuts.npy"), "wb") as f:
        f.write(b"x")
    with open(os.path.join(wd, "reps", "seg_000_t0.png"), "wb") as f:
        f.write(b"x")
    with open(os.path.join(wd, "ocr.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(wd, "covers", "manifest.json"), "w") as f:
        json.dump({"panels": []}, f)


def _capture_compile_ns(tmp_path, monkeypatch, argv_extra):
    wd = str(tmp_path / "w")
    os.makedirs(wd, exist_ok=True)
    _stub_artifacts(wd)
    vid = os.path.join(wd, "video.mp4")
    with open(vid, "wb") as f:
        f.write(b"x")
    captured = {}
    monkeypatch.setattr(cli, "_scroll", lambda a: None)

    def fake_compile(a):
        captured["ns"] = a

    monkeypatch.setattr(cli, "_compile", fake_compile)
    cli.main(["run", vid, "--workdir", wd, "--resume", "--skip-pdf"] + argv_extra)
    return captured["ns"]


def test_run_compile_namespace_is_complete(tmp_path, monkeypatch):
    ns = _capture_compile_ns(tmp_path, monkeypatch, [])
    for attr in COMPILE_ATTRS:
        assert hasattr(ns, attr), f"run->compile Namespace missing {attr!r}"


def test_run_llm_titles_defaults_false(tmp_path, monkeypatch):
    ns = _capture_compile_ns(tmp_path, monkeypatch, [])
    assert ns.llm_titles is False


def test_run_llm_titles_flag_passes_through(tmp_path, monkeypatch):
    ns = _capture_compile_ns(tmp_path, monkeypatch, ["--llm-titles"])
    assert ns.llm_titles is True


def test_run_spines_upscale_passes_through(tmp_path, monkeypatch):
    """run used to hardcode upscale=4 (OOM on 1080p band crops); the flag must
    reach the spines stage Namespace."""
    wd = str(tmp_path / "w")
    os.makedirs(wd, exist_ok=True)
    _stub_artifacts(wd)
    vid = os.path.join(wd, "video.mp4")
    with open(vid, "wb") as f:
        f.write(b"x")
    captured = {}
    monkeypatch.setattr(cli, "_scroll", lambda a: None)
    monkeypatch.setattr(cli, "_compile", lambda a: None)

    def fake_spines(a):
        captured["ns"] = a
        with open(a.out, "w") as f:
            json.dump([], f)

    monkeypatch.setattr(cli, "_spines", fake_spines)
    cli.main(["run", vid, "--workdir", wd, "--resume", "--skip-pdf",
              "--spines-start", "0", "--spines-end", "10", "--spines-upscale", "2"])
    assert captured["ns"].upscale == 2


def test_run_spines_min_focus_passes_through(tmp_path, monkeypatch):
    wd = str(tmp_path / "w")
    os.makedirs(wd, exist_ok=True)
    _stub_artifacts(wd)
    vid = os.path.join(wd, "video.mp4")
    with open(vid, "wb") as f:
        f.write(b"x")
    captured = {}
    monkeypatch.setattr(cli, "_scroll", lambda a: None)
    monkeypatch.setattr(cli, "_compile", lambda a: None)

    def fake_spines(a):
        captured["ns"] = a
        with open(a.out, "w") as f:
            json.dump([], f)

    monkeypatch.setattr(cli, "_spines", fake_spines)
    cli.main(["run", vid, "--workdir", wd, "--resume", "--skip-pdf",
              "--spines-start", "0", "--spines-end", "10", "--spines-min-focus", "35.0"])
    assert captured["ns"].min_focus == 35.0
