"""P0: fuzzy/phonetic matching, corroboration tiers, dedupe, exports."""
import json
import os

from booksnap import fuzz
from booksnap.compile import compile_books, dedupe, tier, write_bibtex
from booksnap.gazetteer import looks_like_place, openlibrary_lookup, verify_all


def test_phonetic_catches_asr_errors():
    assert fuzz.metaphone("dilemma") == fuzz.metaphone("delema")
    assert fuzz.phonetic_score("Goleman", "Coleman.") == 1.0
    assert fuzz.fuzzy_score("Ethnic DeLema", "The Ethnic Dilemma") > 0.84
    assert fuzz.fuzzy_score("Charles Murray", "Facing Reality") < 0.6


def test_token_set_ratio_ignores_subtitles():
    assert fuzz.token_set_ratio("Facing Reality", "Facing Reality: Why We Must") > 0.95
    assert fuzz.token_set_ratio("Sapiens", "Sapiens: A Brief History") > 0.9


def test_place_names_are_not_titles():
    assert looks_like_place("New York City")
    assert not looks_like_place("Facing Reality")
    # no network: the guard runs before any request
    assert openlibrary_lookup("New York City") is None


def test_tiers():
    assert tier(dict(phrase="X", source="spine")) == "confirmed"
    assert tier(dict(phrase="X", heard=True)) == "confirmed"
    assert tier(dict(phrase="X", freq=2)) == "verified"
    assert tier(dict(phrase="X", freq=1, context_ok=True)) == "verified"
    assert tier(dict(phrase="X", freq=1, context_ok=False)) == "weak"


def test_dedupe_merges_variants():
    out = dedupe([
        dict(phrase="Facing Reality", title="Facing Reality", authors=["C. Murray"],
             year=2021, score=0.99, match_type="exact", freq=1),
        dict(phrase="facing reality", title="Facing Reality", authors=["C. Murray"],
             year=2021, score=0.9, match_type="fuzzy", freq=3),
        dict(phrase="Sapiens", title="Sapiens", authors=["Y. N. Harari"], year=2011,
             score=1.0, match_type="exact", freq=2),
    ])
    assert len(out) == 2
    fr = [o for o in out if o["title"] == "Facing Reality"][0]
    assert fr["freq"] == 3 and sorted(fr["provenance"]) == ["Facing Reality", "facing reality"]
    assert fr["score"] == 0.99


def test_bibtex_export(tmp_path):
    entries = [dict(phrase="Facing Reality", title="Facing Reality",
                    authors=["Charles Murray"], year=2021, score=1.0,
                    match_type="exact", provenance=["Facing Reality"])]
    p = tmp_path / "out.bib"
    write_bibtex(entries, str(p))
    text = p.read_text()
    assert text.startswith("@book{murray2021,")
    assert "author = {Charles Murray}" in text and "year   = {2021}" in text


def test_compile_exports_bib_ris_and_tiers(tmp_path, monkeypatch):
    """End-to-end (offline): a spine read verified by a fake lookup is
    'confirmed' and lands in the .bib; a single context-free spoken mention
    stays 'weak' and is excluded."""
    from booksnap import gazetteer as G

    (tmp_path / "scroll.json").write_text("[]")
    (tmp_path / "manifest.json").write_text("[]")
    (tmp_path / "spines.json").write_text(json.dumps(
        [dict(text="Sapiens", conf=0.84, t=100.0, x=5, y=5)]))
    (tmp_path / "transcript.json").write_text(json.dumps(dict(segments=[
        dict(start=0.0, end=2.0, text="I keep thinking about Deep Work lately."),
        dict(start=2.0, end=4.0, text="Totally unrelated remark about the weather."),
    ])))
    outj = tmp_path / "books.json"

    def fake_lookup(phrase):
        p = phrase.strip().lower()
        if p == "sapiens":
            return dict(title="Sapiens: A Brief History of Humankind",
                        authors=["Yuval Noah Harari"], year=2011, score=0.93,
                        match_type="fuzzy")
        if p == "deep work":
            return dict(title="Deep Work", authors=["Cal Newport"], year=2016,
                        score=1.0, match_type="exact")
        return None

    monkeypatch.setattr(G, "make_cached_lookup",
                        lambda path, **kw: fake_lookup)
    monkeypatch.setattr(G, "time", type("T", (), {"sleep": staticmethod(lambda x: None)}))

    data = compile_books(None, str(tmp_path / "manifest.json"), None,
                         str(tmp_path / "scroll.json"), str(outj),
                         str(tmp_path / "books.md"),
                         audio_json=str(tmp_path / "transcript.json"),
                         spines_json=str(tmp_path / "spines.json"), gazetteer=True)

    statuses = {g["title"]: g["status"] for g in data["gazetteer"]}
    assert statuses["Sapiens: A Brief History of Humankind"] == "confirmed"
    assert statuses["Deep Work"] == "weak"
    bib = (tmp_path / "books.bib").read_text()
    assert "Sapiens" in bib and "Deep Work" not in bib
    ris = (tmp_path / "books.ris").read_text()
    assert "TI  - Sapiens" in ris and "Deep Work" not in ris
    md = (tmp_path / "books.md").read_text()
    assert "[confirmed]" in md and "[weak] Deep Work" in md
    assert os.path.exists(str(outj))


def test_superres_presets_and_stack():
    import numpy as np
    from booksnap.superres import PRESETS, enhance, stack_median
    crop = (np.random.default_rng(0).random((30, 200, 3)) * 255).astype("uint8")
    for preset in PRESETS:
        out = enhance(crop, preset, scale=2)
        assert out.shape[:2] == (60, 400) and out.dtype == np.uint8
    noisy = [np.clip(crop.astype(int) + rng.integers(-20, 20, crop.shape), 0, 255).astype("uint8")
             for rng in (np.random.default_rng(i) for i in range(5))]
    stacked = stack_median(noisy)
    # median-stacking a static shot must reduce noise vs a single frame
    err_single = float(np.abs(noisy[0].astype(int) - crop.astype(int)).mean())
    err_stack = float(np.abs(stacked.astype(int) - crop.astype(int)).mean())
    assert err_stack < err_single


def test_spines_records_preset_and_keeps_best(monkeypatch, tmp_path):
    """Presets compete; the highest-confidence read per string wins.

    The fake OCR cannot see which preset produced a tile, so the enhancement
    spy tags pixel [0,0] with the preset index and the fake reads it back.
    """
    import numpy as np
    from booksnap import spines as SP
    from booksnap import superres as SR

    monkeypatch.setattr(SP, "probe", lambda v: dict(width=64, height=32))
    monkeypatch.setattr(SP, "_grab", lambda v, t, w, h: np.full((h, w, 3), 128, np.uint8))
    monkeypatch.setattr(SP, "sharpness", lambda img: 1.0)

    conf = {0: 0.51, 1: 0.70, 2: 0.93}   # lanczos, clahe, binarize
    orig_enhance = SR.enhance

    def spy(crop, preset="unsharp", scale=4):
        out = orig_enhance(crop, preset, scale=scale).copy()
        out[0, 0] = spy.tags[preset]
        return out

    spy.tags = {"lanczos": 0, "clahe": 1, "binarize": 2}
    monkeypatch.setattr(SR, "enhance", spy)

    import booksnap.ocr as O
    monkeypatch.setattr(O, "RapidOCR", type("FakeOCR", (), {}), raising=False)
    monkeypatch.setattr(O, "ocr_image", lambda ocr, tile: [
        dict(text="Sapiens", conf=conf[int(tile[0, 0, 0])], box=[0, 0, 10, 5])])

    out = SP.extract_spines("fake.mp4", str(tmp_path / "spines.json"),
                            start=0, end=1, step=1.0, topk=1, band=1.0,
                            upscale=1, presets=("lanczos", "clahe", "binarize"),
                            tile_w=4096)
    assert len(out) == 1
    assert out[0]["preset"] == "binarize" and out[0]["conf"] == 0.93


def test_match_rules_reject_people_places_institutions():
    from booksnap.gazetteer import match_docs
    person = [dict(title="Daniel Dennett", author_name=["Andrew Brook"],
                   first_publish_year=2002, edition_count=9)]
    assert match_docs("DENNETT", person) is None
    place = [dict(title="New York City in 1979", author_name=["Kathy Acker"],
                  first_publish_year=1981, edition_count=5)]
    assert match_docs("New York City", place) is None
    inst = [dict(title="Cambridge University handbook",
                 author_name=["University of Cambridge"], first_publish_year=1970,
                 edition_count=11)]
    assert match_docs("Cambridge University", inst) is None
    junk = [dict(title="Charles T. Murray",
                 author_name=["Phone Number List", "Databases List"],
                 first_publish_year=1906, edition_count=4)]
    assert match_docs("Charles Murray", junk) is None


def test_match_rules_accept_real_titles_and_gate_on_editions():
    from booksnap.gazetteer import match_docs
    sapiens = [dict(title="Sapiens", author_name=["Yuval Noah Harari"],
                    first_publish_year=2011, edition_count=86)]
    m = match_docs("Sapiens", sapiens)
    assert m and m["title"] == "Sapiens" and m["match_type"] == "exact"
    subtitle = [dict(title="Sapiens: A Brief History of Humankind",
                     author_name=["Yuval Noah Harari"], first_publish_year=2011,
                     edition_count=40)]
    assert match_docs("Sapiens", subtitle)["match_type"] == "prefix"
    obscure = [dict(title="Lockedin", author_name=["Kierra Emerson"],
                    first_publish_year=2018, edition_count=2)]
    m = match_docs("LOCKEDIN", obscure)
    assert m and m["low_support"] is True          # thin catalogue record
    assert match_docs("LOCKEDIN", obscure, min_editions=3) is None  # hard floor
    fragment = [dict(title="Liesl and Po", author_name=["Lauren Oliver"],
                     first_publish_year=2011, edition_count=20)]
    assert match_docs("AND PO", fragment) is None            # 1 significant token


def test_prefers_strictest_rule_then_most_editions():
    from booksnap.gazetteer import match_docs
    docs = [dict(title="Great Awakening vs the Great Reset", author_name=["A. Dugin"],
                 first_publish_year=2021, edition_count=4),
            dict(title="The Great Awakening", author_name=["Joseph Tracy"],
                 first_publish_year=2017, edition_count=18),
            dict(title="The great awakening", author_name=["Thomas Kidd"],
                 first_publish_year=2006, edition_count=7)]
    m = match_docs("Great Awakening", docs)
    assert m["title"] == "The Great Awakening" and m["match_type"] == "exact"


def test_variant_retry_finds_the_legible_read(monkeypatch):
    """Spine groups carry OCR variants; verify_all retries them when the
    canonical (highest-confidence) read fails to match."""
    from booksnap import gazetteer as G
    calls = []

    def fake(phrase):
        calls.append(phrase)
        if phrase == "M-N LOSISG GROUND":
            return dict(title="Losing Ground", authors=["Charles Murray"],
                        year=1984, score=0.95, match_type="fuzzy")
        return None

    monkeypatch.setattr(G, "time", type("T", (), {"sleep": staticmethod(lambda x: None)}))
    out = G.verify_all([dict(phrase="M LOSING GEOCND", freq=0, source="spine",
                             alts=["MO LOSING GROUND", "M-N LOSISG GROUND"])],
                       lookup=fake, max_queries=5)
    assert out[0]["verified"] and out[0]["title"] == "Losing Ground"
    assert out[0]["matched_as"] == "M-N LOSISG GROUND"
    assert calls == ["M LOSING GEOCND", "MO LOSING GROUND", "M-N LOSISG GROUND"]


def test_clean_spine_phrase_strips_ocr_debris():
    from booksnap.compile import clean_spine_phrase
    assert clean_spine_phrase("M LOSING GEOCND") == "LOSING GEOCND"
    assert clean_spine_phrase("M-N LOSISG GROUND") == "LOSISG GROUND"
    assert clean_spine_phrase("MO LOSING GROUND") == "LOSING GROUND"
    assert clean_spine_phrase("Sapiens") == "Sapiens"
    assert clean_spine_phrase("with") == "with"


def test_phonetic_alignment_recovers_mangled_titles():
    from booksnap.gazetteer import _align, match_docs
    assert _align(["LOSISG", "GROUND"], {"losing", "ground"})
    assert _align(["Ethnic", "DeLema"], {"the", "ethnic", "dilemma"})
    assert not _align(["WEALTTL"], {"wealth", "nations"})
    docs = [dict(title="Losing Ground", author_name=["Charles Murray"],
                 first_publish_year=1984, edition_count=25)]
    m = match_docs("LOSISG GROUND", docs)
    assert m and m["title"] == "Losing Ground" and m["match_type"] == "fuzzy"
    m2 = match_docs("Ethnic DeLema", [dict(title="The Ethnic Dilemma",
                                           author_name=["Some Author"],
                                           first_publish_year=2020, edition_count=5)])
    assert m2 and m2["match_type"] == "fuzzy"


def test_alt_lookups_do_not_starve_the_queue(monkeypatch):
    """Primary lookups own max_queries; alts get their own sub-budget."""
    from booksnap import gazetteer as G
    calls = []

    def fake(phrase):
        calls.append(phrase)
        if phrase == "Facing Reality":
            return dict(title="Facing Reality", authors=["Charles Murray"], year=2021,
                        score=1.0, match_type="exact")
        return None

    monkeypatch.setattr(G, "time", type("T", (), {"sleep": staticmethod(lambda x: None)}))
    cands = [dict(phrase=f"SPINE {i}", freq=0, source="spine", alts=[f"ALT {i}a", f"ALT {i}b"])
             for i in range(10)]
    cands.append(dict(phrase="Facing Reality", freq=2))
    out = G.verify_all(cands, lookup=fake, max_queries=11)
    assert any(o.get("title") == "Facing Reality" for o in out)
    assert calls.count("SPINE 0") == 1
    assert len([c for c in calls if c.startswith("ALT")]) <= 5  # sub-budget = 11//2


def test_regions_and_demographics_are_not_titles():
    from booksnap.gazetteer import looks_like_region, match_docs
    for p in ("West Africa", "West Africans", "South Asia", "North African",
              "South East Asia", "African Americans"):
        assert looks_like_region(p), p
    for p in ("Facing Reality", "Losing Ground", "The Great Awakening"):
        assert not looks_like_region(p), p
    docs = [dict(title="West Africa", author_name=["R. J. Harrison-Church"],
                 first_publish_year=1957, edition_count=25)]
    assert match_docs("West Africa", docs) is None


def test_homophone_only_heard_hits_are_not_evidence():
    from booksnap.compile import _heard_is_meaningful
    assert not _heard_is_meaningful("COATES", "cities", 1.0)
    assert _heard_is_meaningful("Sapiens", "Sapiens", 1.0)   # exact spelling
    assert _heard_is_meaningful("Conversations", "Conversations", 1.0)
    assert _heard_is_meaningful("Losing Ground", "losing ground", 1.0)


def test_cue_detected_titles_lead_the_spoken_queue(tmp_path, monkeypatch):
    """A wall of spine OCR must not starve a cue-detected spoken title."""
    from booksnap import gazetteer as G
    (tmp_path / "scroll.json").write_text("[]")
    (tmp_path / "manifest.json").write_text("[]")
    (tmp_path / "spines.json").write_text(json.dumps(
        [dict(text=f"SPINE {i:02d} WORD", conf=0.9, t=float(i), x=0, y=0) for i in range(40)]))
    (tmp_path / "transcript.json").write_text(json.dumps(dict(segments=[
        dict(start=0.0, end=3.0, text="As I argue in my book Facing Reality, this matters."),
    ] + [dict(start=3.0 + i, end=4.0 + i, text=f"Filler {i} about Nothing Special Here.")
         for i in range(30)])))
    queried = []

    def fake(phrase):
        queried.append(phrase)
        if phrase.strip().lower() == "facing reality":
            return dict(title="Facing Reality", authors=["Charles Murray"], year=2021,
                        score=1.0, match_type="exact", editions=12)
        return None

    monkeypatch.setattr(G, "make_cached_lookup", lambda path, **kw: fake)
    monkeypatch.setattr(G, "time", type("T", (), {"sleep": staticmethod(lambda x: None)}))
    data = compile_books(None, str(tmp_path / "manifest.json"), None,
                         str(tmp_path / "scroll.json"), str(tmp_path / "books.json"),
                         str(tmp_path / "books.md"),
                         audio_json=str(tmp_path / "transcript.json"),
                         spines_json=str(tmp_path / "spines.json"),
                         gazetteer=True, max_queries=20)
    assert "Facing Reality" in queried
    assert [g["title"] for g in data["gazetteer"]] == ["Facing Reality"]


def test_low_support_matches_need_corroboration():
    """'Facing Reality' (cued, 2 words) survives a thin record; a mangled
    single-token spine read does not."""
    from booksnap import gazetteer as G
    thin = dict(title="Facing Reality", authors=["Charles Murray"], year=2021,
                score=1.0, match_type="exact", editions=2, low_support=True)
    thin2 = dict(title="Lockedin", authors=["Kierra Emerson"], year=2018,
                 score=1.0, match_type="exact", editions=2, low_support=True)
    monkey = type("T", (), {"sleep": staticmethod(lambda x: None)})
    G.time = monkey
    cued = G.verify_all([dict(phrase="Facing Reality", freq=0, cued=True)],
                        lookup=lambda p: dict(thin))
    assert cued[0]["verified"] is True
    spine = G.verify_all([dict(phrase="LOCKEDIN", freq=0, source="spine")],
                         lookup=lambda p: dict(thin2))
    assert spine[0]["verified"] is False
    solid = G.verify_all([dict(phrase="LOCKEDIN", freq=0, source="spine")],
                         lookup=lambda p: dict(thin2, low_support=False, editions=40))
    assert solid[0]["verified"] is True


def test_failed_fetch_is_not_cached(tmp_path, monkeypatch):
    """A network failure must not poison the cache with an empty result."""
    from booksnap import gazetteer as G
    calls = []

    def fake_fetch(phrase, timeout=5.0):
        calls.append(phrase)
        return None if len(calls) == 1 else [
            dict(title="Facing Reality", author_name=["Charles Murray"],
                 first_publish_year=2021, edition_count=12)]

    monkeypatch.setattr(G, "fetch_docs", fake_fetch)
    lookup = G.make_cached_lookup(str(tmp_path / "cache.json"))
    assert lookup("Facing Reality") is None       # first attempt fails
    m = lookup("Facing Reality")                  # retried, not cached-as-empty
    assert m and m["title"] == "Facing Reality"
    assert calls == ["Facing Reality", "Facing Reality"]


def test_author_hints_pick_the_right_same_titled_record():
    from booksnap.gazetteer import match_docs
    docs = [dict(title="Facing reality", author_name=["J. C. Eccles"],
                 first_publish_year=1970, edition_count=22),
            dict(title="Facing Reality", author_name=["Charles Murray"],
                 first_publish_year=2021, edition_count=6)]
    plain = match_docs("Facing Reality", docs)
    assert plain["authors"] == ["J. C. Eccles"]      # edition count wins by default
    hinted = match_docs("Facing Reality", docs, author_hints={"charles", "murray"})
    assert hinted["authors"] == ["Charles Murray"]   # transcript names the author
    assert hinted["title"] == "Facing Reality"


def test_bench_matcher_is_directional_and_strict():
    from booksnap.bench_spines import match_label
    reads = ["Sapiens", "LOSING GROUND", "LOCKEDIN", "VYCYIMS", "LOSINCTHIRACE"]
    assert match_label("Sapiens", reads)[0] == "Sapiens"
    assert match_label("Locked In", reads)[0] == "LOCKEDIN"      # space-free norm
    assert match_label("Losing Ground", reads)[0] == "LOSING GROUND"
    assert match_label("Losing the Race", reads)[0] is None      # not Losing Ground
    assert match_label("On the Origin of Species", reads)[0] is None
    assert match_label("We Were Eight Years in Trouble", reads)[0] is None
