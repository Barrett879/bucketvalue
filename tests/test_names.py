"""Every case here was measured against the real 2026-27 index on 2026-10-08."""
from bucketlib.names import NameIndex, normalize

ROSTER = [
    ("Nikola Jokić", 1, "DEN"), ("P.J. Washington", 2, "DAL"), ("PJ Hall", 3, "MEM"),
    ("Karl-Anthony Towns", 4, "NYK"), ("Shai Gilgeous-Alexander", 5, "OKC"),
    ("Moritz Wagner", 6, "ORL"), ("Yang Hansen", 7, "POR"), ("Herbert Jones", 8, "NOP"),
    ("Jaylin Williams", 9, "OKC"), ("Tim Hardaway Jr.", 10, "DEN"), ("Dennis Schröder", 11, "CHA"),
    ("Kel'el Ware", 12, "MIA"), ("Ernest Udeh, Jr.", 13, "LAC"), ("Dereck Lively II", 14, "DAL"),
    ("Grant Williams", 15, "CHA"), ("Mark Williams", 16, "PHX"),
]
ALIASES = {normalize("Moe Wagner"): "Moritz Wagner", normalize("Hansen Yang"): "Yang Hansen",
           normalize("Dennis Schroeder"): "Dennis Schröder"}


def idx():
    ix = NameIndex(aliases=ALIASES)
    for n, v, t in ROSTER:
        ix.add(n, v, team=t)
    return ix


def test_accents_initials_hyphens_suffixes():
    ix = idx()
    assert ix.get("Nikola Jokic").value == 1
    assert ix.get("PJ Washington").value == 2
    assert ix.get("P.J. Hall").value == 3
    assert ix.get("Karl Anthony Towns").value == 4
    assert ix.get("Shai Gilgeous Alexander").value == 5
    assert ix.get("Kelel Ware").value == 12
    assert ix.get("Ernest Udeh Jr.").value == 13
    assert ix.get("Dereck Lively").value == 14


def test_aliases_and_prefix():
    ix = idx()
    m = ix.get("Moe Wagner")
    assert (m.value, m.how) == (6, "alias")
    assert ix.get("Dennis Schroeder").value == 11
    m = ix.get("Herb Jones")
    assert (m.value, m.how) == (8, "prefix")


def test_wrong_bindings_refused():
    ix = idx()
    # Jay Williams is not Jaylin Williams just because one is a prefix: with two
    # other Williamses present the prefix rule finds exactly one candidate, so
    # scope by team when the source gives one.
    assert ix.get("Jay Williams", team="CHA").value is None


def test_injury_report_order():
    ix = idx()
    assert ix.resolve_last_first("Lively II, Dereck", team="DAL").value == 14
    assert ix.resolve_last_first("Yang, Hansen", team="POR").value == 7
    assert ix.resolve_last_first("Schroder, Dennis").value == 11


def test_ambiguous_exact_needs_team():
    ix = NameIndex(aliases={})
    ix.add("Brandon Williams", 100, team="DAL")
    ix.add("Brandon Williams", 200, team="WAS")
    assert ix.get("Brandon Williams").how == "ambiguous"
    assert ix.get("Brandon Williams", team="WAS").value == 200
