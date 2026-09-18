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
