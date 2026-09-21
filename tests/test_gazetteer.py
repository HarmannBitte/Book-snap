from booksnap.gazetteer import title_candidates, verify_all

SEGMENTS = [
    dict(start=1.0, end=5.0,
         text="As I say in Facing Reality, the numbers just do not add up."),
    dict(start=6.0, end=9.0, text="I asked Nathan Cofnas whether Facing Reality sold well."),
    dict(start=10.0, end=13.0, text="The weather tomorrow will be fine, I think."),
]

FAKE_DB = {"facingreality": dict(title="Facing Reality", authors=["Charles Murray"],
                                 year=2021)}


def fake_lookup(phrase, **kw):
    return FAKE_DB.get("".join(ch for ch in phrase.lower() if ch.isalnum()))


def test_title_candidates_mid_sentence():
    c = title_candidates(SEGMENTS)
    phrases = {x["phrase"] for x in c}
    assert "Facing Reality" in phrases
    assert c[[x["phrase"] for x in c].index("Facing Reality")]["freq"] == 2


def test_connector_only_tail_terminates():
    # regression: "In the" used to loop forever (single-token rsplit no-op)
    c = title_candidates([dict(start=0.0, end=2.0, text="In the end, Facing Reality won.")])
    assert any(x["phrase"] == "Facing Reality" for x in c)


def test_verify_all_flags_verified_and_rejected():
    c = title_candidates(SEGMENTS)
    out = verify_all(c, lookup=fake_lookup, pause=0.0)
    ver = {o["phrase"]: o for o in out}
    assert ver["Facing Reality"]["verified"] is True
    assert ver["Facing Reality"]["authors"] == ["Charles Murray"]
    assert ver["Nathan Cofnas"]["verified"] is False


def test_strip_article_preserves_words_starting_with_a():
    from booksnap.gazetteer import _strip_article, match_docs
    # "Antifragile" and "Arguments for democracy" must not have initial 'a' stripped
    assert _strip_article("Antifragile") == "antifragile"
    assert _strip_article("Arguments for Democracy") == "argumentsfordemocracy"
    assert _strip_article("The Way of All Flesh") == "wayofallflesh"
    assert _strip_article("A Brief History of Time") == "briefhistoryoftime"

    docs = [dict(title="Arguments for democracy", author_name=["Tony Benn"],
                 first_publish_year=1981, edition_count=2)]
    m = match_docs("Arguments for democracy", docs)
    assert m is not None
    assert m["match_type"] == "exact" and m["title"] == "Arguments for democracy"
