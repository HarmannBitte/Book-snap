from booksnap.compile import (match_shown_to_bibliography, parse_bibliography,
                              tokens)

LINES = [
    dict(text="Bibliography"),
    dict(text="Alexander, Richard D. The Biology of Moral Systems. 1987"),
    dict(text="Ball, Philip. How Life Works: A User's Guide to the New Biology. 2023."),
    dict(text="Caprara, Gian Vittorio, Shalom Schwartz. \"Personality and Politics:\""),
    dict(text="Political Psychology 27, no. 1 (2006): 1-28"),
]


def test_parse_groups_entries():
    e = parse_bibliography(LINES)
    assert len(e) == 3
    assert e[0]["kind"] == "book" and e[0]["year"] == 1987
    assert e[2]["kind"] == "article"


def test_match_flags_cited_and_shown_only():
    biblio = parse_bibliography(LINES)
    shown = [
        dict(text="blueprint how DNA makes us who we are ROBERT PLOMIN"),
        dict(text="Born Together Reared Apart The Landmark Minnesota Twin Study NANCY L. SEGAL"),
    ]
    biblio.append(dict(text="Plomin, Robert. Blueprint: How DNA Makes Us Who We Are. 2018.",
                       kind="book", year=2018))
    match_shown_to_bibliography(shown, biblio)
    assert shown[0]["cited"] is True
    assert shown[1]["cited"] is False


def test_tokens_drop_stopwords():
    assert "the" not in tokens("The Righteous Mind")
    assert "righteous" in tokens("The Righteous Mind")


def test_author_spine_fused_compound_not_author():
    from booksnap.compile import author_spine
    assert author_spine(dict(source="spine", text="DENNETT")) is True
    assert author_spine(dict(source="spine", text="OBAMA")) is True
    # Fused caps compound that segments via dp_split is a title, not an author surname
    assert author_spine(dict(source="spine", text="LOCKEDIN")) is False


def test_group_spine_reads_pops_mixed_case_author_band():
    from booksnap.compile import group_spine_reads
    reads = [
        dict(text="WEALIT", conf=0.75, x=3935, y=358, w=226, h=51),
        dict(text="AND PO", conf=0.84, x=3993, y=420, w=204, h=52),
        dict(text="JamQW", conf=0.70, x=3974, y=507, w=209, h=49),
    ]
    grouped = group_spine_reads(reads)
    texts = [g["text"] for g in grouped]
    assert "WEALIT AND PO" in texts
    assert "JamQW" in texts


def test_corroborated_allows_two_word_title_with_preposition():
    from booksnap.gazetteer import _corroborated
    # Real 2-word title with physical spine evidence: corroborated
    assert _corroborated(dict(phrase="Locked In", source="spine"), False) is True
    # Single stop word or single token without 2 words: rejected
    assert _corroborated(dict(phrase="IN", source="spine"), False) is False
    assert _corroborated(dict(phrase="THE", source="spine"), False) is False
