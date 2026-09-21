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


def test_catalog_crud_and_lookup(tmp_path):
    from booksnap.catalog import insert_books, lookup_catalog
    db_p = str(tmp_path / "test_cat.db")
    recs = [
        {"title": "The Selfish Gene", "authors": ["Richard Dawkins"], "year": 1976, "editions": 45},
        {"title": "Consciousness Explained", "authors": ["Daniel C. Dennett"], "year": 1991, "editions": 18},
    ]
    n = insert_books(recs, db_path=db_p)
    assert n == 2

    # Exact lookup
    docs = lookup_catalog("The Selfish Gene", db_path=db_p)
    assert len(docs) == 1 and docs[0]["title"] == "The Selfish Gene"
    assert docs[0]["first_publish_year"] == 1976

    # Article-stripped lookup
    docs2 = lookup_catalog("Selfish Gene", db_path=db_p)
    assert len(docs2) == 1 and docs2[0]["author_name"] == ["Richard Dawkins"]

    # Missing lookup
    assert lookup_catalog("Unknown Nonexistent Book", db_path=db_p) == []


def test_cached_lookup_with_catalog(tmp_path, monkeypatch):
    from booksnap import gazetteer as G
    from booksnap.catalog import insert_books
    db_p = str(tmp_path / "test_cat.db")
    insert_books([{"title": "Antifragile", "authors": ["Nassim Nicholas Taleb"],
                   "year": 2012, "editions": 22}], db_path=db_p)

    # Monkeypatch fetch_docs to raise an exception to ensure zero network is called
    def fail_fetch(p):
        raise RuntimeError("network called unexpectedly!")

    monkeypatch.setattr(G, "fetch_docs", fail_fetch)

    # With use_catalog=True and catalog DB path set
    import booksnap.catalog as C
    monkeypatch.setattr(C, "DEFAULT_DB_PATH", db_p)

    lk = G.make_cached_lookup(str(tmp_path / "cache.json"), use_catalog=True)
    m = lk("Antifragile")
    assert m is not None
    assert m["title"] == "Antifragile" and m["match_type"] == "exact"
    assert m["authors"] == ["Nassim Nicholas Taleb"]
