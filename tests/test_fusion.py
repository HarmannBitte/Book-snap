from booksnap.compile import audio_title_candidates, fuse_audio, heard_match

SEGMENTS = [
    dict(start=10.0, end=14.0,
         text="Yeah, I mean in his book Sapiens he argues that forager life was fine."),
    dict(start=20.0, end=24.0, text="She wrote a memoir called Losing Ground, I think?"),
    dict(start=30.0, end=34.0, text="We should talk about the weather tomorrow."),
]


def test_audio_title_candidates():
    c = audio_title_candidates(SEGMENTS)
    phrases = {x["phrase"] for x in c}
    assert "Sapiens" in phrases
    assert "Losing Ground" in phrases


def test_heard_match_clean_candidate():
    phrase, score = heard_match("Sapiens", SEGMENTS)
    assert phrase is not None and "Sapiens" in phrase
    assert score >= 0.78


def test_heard_match_garbled_spine_not_heard():
    phrase, score = heard_match("LOSIHCIIACE", SEGMENTS)
    assert phrase is None


def test_fuse_audio_sets_flags():
    cands = [dict(text="Sapiens", source="spine"), dict(text="WEALTILLNF", source="spine")]
    fuse_audio(cands, SEGMENTS)
    assert cands[0]["heard"] is True
    assert cands[1]["heard"] is False
