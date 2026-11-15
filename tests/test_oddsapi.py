"""bucketlib.sources.oddsapi: the pure half of the licensed odds feed (The
Odds API). The settings reader, the request builders, the usage headers, the
error codes, /events and its match to the schedule, the event-odds parser and
the per-book merge of a refresh, all on plain dicts and on the hand-built
fixtures in tests/fixtures/oddsapi (CONSTRUCTED, see PROVENANCE.txt).

Nothing here reaches a network or reads a clock. Every time is a fixed past
instant passed in as an argument (2026-03-12, the serve fixture's date) or,
for the neutral-site games of this season, a literal date handed to a pure
function. The fake key is built at run time. The expected numbers were
counted by hand from the fixtures, never read back from the parser.

The contracts against props (FEED_COLUMNS covering props.FEED_REQUIRED, the
catalog equal to props' copy: BOOK_CATALOG, BOOK_ORDER, BOOK_CODE, BOOK_TITLE
and DFS_BOOKS) live with props' feed code in tests/test_props_feed.py.

The books (build amendment B35): the catalog, its exclusions, the default
list (BOOKS, pinned by test_the_default_books_are_pinned ALONE: every other
test reads it from the constant) and the settings reader. The constructed
fixtures were answered for the five books they hold (FIXTURE_BOOKS, read from
the PHI at DET file), so the parse and merge tests name those, as the calls
that made them did; test_a_constructed_ten_book_answer_parses_every_book and
test_a_pull_of_ten_books_after_a_pull_of_five_takes_nothing_down cover ten.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import http as H   # first: on macOS it loads CoreFoundation before curl_cffi
from bucketlib import props, teams  # noqa: E402
from bucketlib.sources import oddsapi as O  # noqa: E402
from bucketlib.util import ET, UTC  # noqa: E402

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "oddsapi"
SERVE = HERE / "fixtures" / "serve"
DAY = dt.date(2026, 3, 12)                                  # the serve fixture's date
NINE = dt.datetime(2026, 3, 12, 9, 0, tzinfo=ET)            # a morning pull (13:00Z)
ELEVEN = dt.datetime(2026, 3, 12, 11, 0, tzinfo=ET)         # a later refresh (15:00Z)
KEY = "K" * 40                                              # a fake key, built at run time
PHI_DET = "c4f1a9e2b7d3058f6a1e9c2b4d7f0953"                # the fixtures' invented event ids
DEN_SAS = "e7b2d9f4a1c6083e5b9d2f7a4c1e0958"
FOUR = ("player_points", "player_rebounds", "player_assists", "player_fantasy_points")
SIDES = {"both": 2, "more": 1, "less": 1}
# Vendor keys the catalog must never hold (build amendment B35), each named:
# the offshore books, every exchange and prediction market, the sweepstakes
# book, Hard Rock's state variants, courtside and rebet.
NEVER_BOOKS = ("betonlineag", "betus", "bovada", "lowvig", "mybookieag", "betanysports",
               "betopenly", "kalshi", "limitless", "novig", "polymarket", "polymarket_us",
               "prophetx", "fliff", "hardrockbet_az", "hardrockbet_fl", "hardrockbet_oh",
               "courtside", "rebet")


# -- Helpers -----------------------------------------------------------------------
def fixture(name: str):
    return json.loads((FIX / name).read_text())


def phi_det() -> dict:
    return fixture("event_odds_phi_det_constructed.json")


def den_sas() -> dict:
    return fixture("event_odds_den_sas_constructed.json")


# The books the constructed answers were asked for (PROVENANCE.txt), read from
# the PHI at DET answer, which holds every one of them, in its own order.
FIXTURE_BOOKS = tuple(b["key"] for b in phi_det()["bookmakers"])


def parse(payload, books=FIXTURE_BOOKS, markets=FOUR, at=NINE) -> pd.DataFrame:
    return O.parse_event_odds(payload, books=books, markets=markets, fetched_at=at)


def entry(payload: dict, book: str) -> dict:
    (bm,) = [b for b in payload["bookmakers"] if b["key"] == book]
    return bm


def market(payload: dict, book: str, key: str) -> dict:
    (mk,) = [m for m in entry(payload, book)["markets"] if m["key"] == key]
    return mk


def one(df: pd.DataFrame, book: str, name: str, stat: str = "Points") -> pd.Series:
    hit = df[(df["book"] == book) & (df["name"] == name) & (df["stat_type"] == stat)]
    assert len(hit) == 1, (book, name, stat, len(hit))
    return hit.iloc[0]


def outcomes_in(payload) -> int:
    """Every outcome a payload carries, counted here independently of the
    parser: outcomes inside a well-formed bookmakers -> markets -> outcomes
    nesting."""
    n = 0
    bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else None
    for bm in bookmakers if isinstance(bookmakers, list) else []:
        if isinstance(bm, dict) and isinstance(bm.get("markets"), list):
            for mk in bm["markets"]:
                if isinstance(mk, dict) and isinstance(mk.get("outcomes"), list):
                    n += len(mk["outcomes"])
    return n


def accounted(df: pd.DataFrame, payload=None) -> dict:
    """Every outcome lands in a kept line or in exactly one counter, in total
    and per book, and the kept outcomes are the rows' own sides."""
    a = df.attrs
    assert a["outcomes_seen"] == a["kept_outcomes"] + sum(a[k] for k in O.COUNTERS), a
    for book, t in a["by_book"].items():
        assert t["outcomes_seen"] == t["kept_outcomes"] + sum(t[k] for k in O.BOOK_COUNTERS), book
        assert t["rows"] == int((df["book"] == book).sum()), book
    assert a["rows"] == len(df)
    assert a["kept_outcomes"] == sum(SIDES[d] for d in df["direction"])
    if payload is not None and not a["unknown_team"]:
        assert a["outcomes_seen"] == outcomes_in(payload)
    return a


def ev(bookmakers, home="Detroit Pistons", away="Philadelphia 76ers", **top) -> dict:
    """A PHI at DET event-odds answer holding `bookmakers`."""
    out = {"id": PHI_DET, "sport_key": "basketball_nba", "sport_title": "NBA",
           "commence_time": "2026-03-12T23:00:00Z", "home_team": home, "away_team": away,
           "bookmakers": bookmakers}
    out.update(top)
    return out


def bk(key: str, *markets) -> dict:
    return {"key": key, "title": key, "markets": list(markets)}


def mk(key: str, *outcomes, last_update="2026-03-12T12:52:14Z") -> dict:
    m = {"key": key, "outcomes": [o for group in outcomes for o in group]}
    if last_update is not None:
        m["last_update"] = last_update
    return m


def pair(name: str, point, over=-110, under=-110) -> list:
    return [{"name": "Over", "description": name, "price": over, "point": point},
            {"name": "Under", "description": name, "price": under, "point": point}]


def side(name: str, player: str, point=24.5, price=-110) -> list:
    """One outcome, as a list so mk() can take it beside pair()."""
    return [{"name": name, "description": player, "price": price, "point": point}]


# -- The vendor's facts ----------------------------------------------------------------
def test_the_vendors_facts():
    assert O.REFERENCE_BOOK == "prizepicks"
    assert O.HOST == "api.the-odds-api.com" and O.SPORT == "basketball_nba"
    assert O.SEND_REGIONS is False                  # never regions, so never "us" (B35)
    assert O.MAX_BOOKS == 10


# -- The books (build amendment B35) ---------------------------------------------------
# C1, word for word: (key, code, title) in the board's column order.
CATALOG = (
    ("draftkings", "DK", "DraftKings"), ("fanduel", "FD", "FanDuel"), ("betmgm", "MGM", "BetMGM"),
    ("prizepicks", "PP", "PrizePicks"), ("underdog", "UD", "Underdog"),
    ("espnbet", "TSB", "theScore Bet"), ("hardrockbet", "HRB", "Hard Rock Bet"),
    ("betrivers", "BR", "BetRivers"), ("pick6", "P6", "DraftKings Pick6"),
    ("betparx", "PARX", "betPARX"), ("ballybet", "BALLY", "Bally Bet"),
    ("dabble_us_dfs", "DAB", "Dabble"), ("williamhill_us", "CZR", "Caesars"),
    ("fanatics", "FAN", "Fanatics"))
PICKEM = {"prizepicks", "underdog", "pick6", "dabble_us_dfs"}       # C5


def test_the_catalog_is_the_licensed_books_with_our_own_codes_and_titles():
    """C1: the state-licensed US sportsbooks and US DFS pick'em sites, in the
    board's column order, each with our own code (unique) and title; C5:
    the four pick'em books, which are every us_dfs key, use More and Less,
    every other book Over and Under; Caesars and Fanatics are served on paid
    plans only. props' copies are the same (tests/test_props_feed.py pins
    them too)."""
    assert O.BOOK_CATALOG == CATALOG
    assert O.BOOK_ORDER == tuple(k for k, _c, _t in CATALOG)
    assert O.BOOK_CODE == {k: c for k, c, _t in CATALOG}
    assert O.BOOK_TITLE == {k: t for k, _c, t in CATALOG}
    codes = [c for _k, c, _t in CATALOG]
    assert len(set(codes)) == len(codes) == len(set(O.BOOK_ORDER)) == 14
    assert all(re.fullmatch(r"[A-Z0-9]{2,5}", c) for c in codes)
    assert O.DFS_BOOKS == O.US_DFS_BOOKS == PICKEM and PICKEM <= set(O.BOOK_ORDER)
    assert O.PAID_BOOKS == {"williamhill_us", "fanatics"} <= set(O.BOOK_ORDER)
    for book in O.BOOK_ORDER:
        want = ("More", "Less") if book in PICKEM else ("Over", "Under")
        assert props.side_words(book) == want, book
    assert (props.BOOK_CATALOG, props.BOOK_ORDER, props.BOOK_CODE, props.BOOK_TITLE,
            props.DFS_BOOKS) == (O.BOOK_CATALOG, O.BOOK_ORDER, O.BOOK_CODE, O.BOOK_TITLE,
                                 O.DFS_BOOKS)


def test_the_never_list_is_the_excluded_set_and_shares_no_key_with_the_catalog():
    assert O.EXCLUDED_BOOKS == set(NEVER_BOOKS) and len(NEVER_BOOKS) == len(set(NEVER_BOOKS))
    assert not O.EXCLUDED_BOOKS & set(O.BOOK_ORDER)


@pytest.mark.parametrize("key", NEVER_BOOKS)
def test_a_book_the_site_never_buys_is_refused_by_the_catalog_the_reader_and_the_builder(key):
    """C1 (build amendment B35): an offshore book, an exchange or prediction
    market, the sweepstakes book, a Hard Rock state variant, courtside and
    rebet are never in the catalog (nor props' copy, nor the default); the
    settings reader drops one, whatever its case, with a fixed phrase that
    never holds its name; the request builder refuses it alone and beside a
    catalog book."""
    for catalog in (O.BOOK_ORDER, props.BOOK_ORDER, O.BOOKS):
        assert key not in catalog
    assert key not in O.BOOK_CODE and key not in O.BOOK_TITLE and key not in O.DFS_BOOKS
    assert key not in props.BOOK_TITLE and key not in props.DFS_BOOKS
    name = "BUCKETVALUE_ODDS_BOOKS"
    s = O.read_settings({name: f"betmgm, {key.upper()} ,underdog"})
    assert s.books == ("betmgm", "underdog")
    assert s.problems == (f"setting {name}: excluded book dropped",)
    alone = O.read_settings({name: key})
    assert alone.books == O.BOOKS
    assert alone.problems == (f"setting {name}: excluded book dropped",
                              f"setting {name}: empty list; default used")
    for p in s.problems + alone.problems:
        assert key not in p.lower(), p
    for books in ((key,), ("draftkings", key), (key.upper(),)):
        with pytest.raises(ValueError) as e:
            O.event_odds_request(PHI_DET, books, ("player_points",))
        assert str(e.value) == "bad books"


def test_the_default_books_are_pinned():
    """C2: THE one test to edit when the default list changes (build
    amendment B35). Ten books, as many as bill as one region (docs 529),
    each once, every one in the catalog, neither paid-only book among them;
    the setting's default is built from it. Every other test reads O.BOOKS."""
    assert O.BOOKS == ("draftkings", "fanduel", "betmgm", "prizepicks", "underdog", "espnbet",
                       "hardrockbet", "betrivers", "pick6", "betparx")
    assert len(set(O.BOOKS)) == len(O.BOOKS) <= O.MAX_BOOKS
    assert set(O.BOOKS) <= set(O.BOOK_ORDER) and not set(O.BOOKS) & O.PAID_BOOKS
    # A swap of any of the last five (or a new order of them) is this test's
    # edit alone (round BK's fix pass rehearsed six). The constructed vendor
    # answers every feed test buys post the first five, so dropping one of
    # those is a reviewed change, not a swap (docs/runbook.md section 8).
    assert set(FIXTURE_BOOKS) <= set(O.BOOKS), "the default must keep the fixtures' five books"
    assert "williamhill_us" not in O.BOOKS and "fanatics" not in O.BOOKS
    assert O.REFERENCE_BOOK in O.BOOKS
    assert O.SETTING_DEFAULTS["BUCKETVALUE_ODDS_BOOKS"] == ",".join(O.BOOKS)
    assert O.read_settings({}).books == O.BOOKS


def _default_list_literals(path: Path) -> int:
    """Tuple and list literals of `path` whose items are O.BOOKS' keys in
    order, and string constants that are the setting's default text."""
    import ast
    n = 0
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == len(O.BOOKS):
            items = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            n += items == list(O.BOOKS)
        elif isinstance(node, ast.Constant) and node.value == ",".join(O.BOOKS):
            n += 1
    return n


def test_the_default_list_is_written_once_in_the_code_and_once_in_the_tests():
    """C2: so a later swap is a one-line change plus one test. The list is
    spelled out in oddsapi alone (props keeps no copy) and pinned by
    test_the_default_books_are_pinned alone; every other module and test
    reads O.BOOKS."""
    repo = HERE.parent
    files = sorted({*repo.glob("bucketlib/**/*.py"), *repo.glob("pages/*.py"),
                    *repo.glob("scripts/*.py"), repo / "app.py", *repo.glob("tests/**/*.py")})
    assert len(files) > 40 and repo / "bucketlib" / "sources" / "oddsapi.py" in files
    found = {str(f.relative_to(repo)): k for f in files if (k := _default_list_literals(f))}
    assert found == {"bucketlib/sources/oddsapi.py": 1, "tests/test_oddsapi.py": 1}, found


def test_every_market_label_round_trips_through_props():
    """RC G8: the label a feed row carries is the stat props prices."""
    stat = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
            "player_threes": "fg3m", "player_blocks": "blk", "player_steals": "stl",
            "player_blocks_steals": "stocks", "player_turnovers": "tov",
            "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr",
            "player_points_assists": "pa", "player_rebounds_assists": "ra",
            "player_frees_made": "ftm", "player_frees_attempts": "fta",
            "player_fantasy_points": "fantasy"}
    assert set(O.MARKET_LABEL) == set(stat)
    for key, label in O.MARKET_LABEL.items():
        assert props.classify_stat(label, league="NBA") == (stat[key], "ok", None), key
    assert O.MAIN_MARKETS == (
        "player_points", "player_rebounds", "player_assists", "player_threes", "player_blocks",
        "player_steals", "player_blocks_steals", "player_turnovers",
        "player_points_rebounds_assists", "player_points_rebounds", "player_points_assists",
        "player_rebounds_assists")
    assert O.CORE_MARKETS == ("player_points",)
    assert "player_field_goals" not in O.MARKET_LABEL       # UNVERIFIED: made or attempted?


def test_feed_columns_are_props_parsed_columns_then_the_feeds_own():
    n = len(props.PARSED_COLUMNS)
    assert O.FEED_COLUMNS[:n] == props.PARSED_COLUMNS
    assert O.FEED_COLUMNS[n:] == ["book", "book_title", "market", "price_over", "price_under",
                                  "last_update", "fetched_at", "event_id", "event_home",
                                  "event_away", "game_id", "game_date", "tip_utc"]
    assert len(set(O.FEED_COLUMNS)) == len(O.FEED_COLUMNS)
    e = O.empty_feed()
    assert list(e.columns) == O.FEED_COLUMNS and e.empty
    assert str(e["fetched_at"].dtype) == "datetime64[ns, UTC]" and e["line"].dtype == float


# -- Settings ---------------------------------------------------------------------------
DEFAULT_SETTINGS = O.Settings(
    enabled=True, books=O.BOOKS, markets=("player_points",), core=("player_points",),
    reserve=None, first_pull_et=dt.time(8, 50), last_pull_lead=dt.timedelta(minutes=30),
    min_refresh=dt.timedelta(minutes=15), tip_guard=dt.timedelta(minutes=5),
    max_calls_10min=40, pp_topup=True, refresher=True, problems=())


def test_the_defaults_are_the_design_table():
    assert O.read_settings({}) == DEFAULT_SETTINGS
    assert O.SETTING_DEFAULTS == {
        "BUCKETVALUE_ODDS_ENABLED": "1", "BUCKETVALUE_ODDS_MARKETS": "player_points",
        "BUCKETVALUE_ODDS_CORE_MARKETS": "player_points",
        "BUCKETVALUE_ODDS_BOOKS": ",".join(O.BOOKS),       # pinned once, in the test below
        "BUCKETVALUE_ODDS_RESERVE": "", "BUCKETVALUE_ODDS_FIRST_PULL_ET": "08:50",
        "BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN": "30", "BUCKETVALUE_ODDS_MIN_REFRESH_MIN": "15",
        "BUCKETVALUE_ODDS_TIP_GUARD_MIN": "5", "BUCKETVALUE_ODDS_MAX_CALLS_10MIN": "40",
        "BUCKETVALUE_ODDS_PP_TOPUP": "1", "BUCKETVALUE_ODDS_REFRESHER": "1"}


def test_an_empty_mapping_is_read_as_empty_never_as_the_process_environment(monkeypatch):
    """m11: `environ or os.environ` would read the process environment for a
    test's empty dict. The reader reads only what it is handed."""
    monkeypatch.setenv("BUCKETVALUE_ODDS_ENABLED", "0")
    monkeypatch.setenv("BUCKETVALUE_ODDS_MARKETS", "player_rebounds")
    monkeypatch.setenv("BUCKETVALUE_ODDS_BOOKS", "prizepicks")
    assert O.read_settings({}) == DEFAULT_SETTINGS
    live = O.read_settings(os.environ)                # what live passes when environ is None
    assert live.enabled is False and live.markets == ("player_rebounds",)
    assert live.books == ("prizepicks",)
    for bad in (None, [], "BUCKETVALUE_ODDS_ENABLED=0", 0):
        with pytest.raises(TypeError, match="^read_settings needs a mapping$"):
            O.read_settings(bad)


@pytest.mark.parametrize("value, on", [
    ("0", False), (" 0 ", False), ("false", False), ("OFF", False), ("no", False),
    ("1", True), ("true", True), ("On", True), ("YES", True), ("", True), ("   ", True),
    (0, False), (1, True)])
def test_the_kill_switch_reads_its_words(value, on):
    s = O.read_settings({"BUCKETVALUE_ODDS_ENABLED": value})
    assert s.enabled is on and s.problems == ()


@pytest.mark.parametrize("value", ["disable", "2", "stop", "0.0", True, ["0"]])
def test_an_unreadable_kill_switch_switches_the_feed_off(value):
    """A typo in the kill switch never leaves the feed spending."""
    s = O.read_settings({"BUCKETVALUE_ODDS_ENABLED": value})
    assert s.enabled is False
    assert s.problems == ("setting BUCKETVALUE_ODDS_ENABLED: unreadable value; "
                          "the feed is switched off",)


def test_markets_are_a_trimmed_lowercased_deduplicated_comma_list():
    s = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": " player_points, Player_Rebounds ,,"
                                                     "player_points "})
    assert s.markets == ("player_points", "player_rebounds") and s.problems == ()
    s = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": ",".join(O.MAIN_MARKETS)})
    assert s.markets == O.MAIN_MARKETS and s.problems == ()
    s = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": ",".join(O.MARKET_LABEL)})
    assert s.markets == tuple(O.MARKET_LABEL)


REFUSED_MARKETS = {
    "player_points_q1": "period market dropped", "player_rebounds_q1": "period market dropped",
    "player_assists_q1": "period market dropped", "player_points_h1": "period market dropped",
    "player_first_basket": "yes/no market dropped",
    "player_first_team_basket": "yes/no market dropped",
    "player_double_double": "yes/no market dropped",
    "player_triple_double": "yes/no market dropped",
    "player_method_of_first_basket": "first-basket method market dropped",
    "player_points_alternate": "alternate market dropped",
    "player_fantasy_points_alternate": "alternate market dropped",
    "player_points_rebounds_assists_alternate": "alternate market dropped",
    "player_points_q1_alternate": "alternate market dropped",
    "player_field_goals": "unknown key dropped", "player_pointz": "unknown key dropped",
    "h2h": "unknown key dropped"}


@pytest.mark.parametrize("key", list(REFUSED_MARKETS))
def test_period_yes_no_alternate_and_unknown_markets_are_refused_so_none_is_billed(key):
    """M11: alternates are refused (the collapse threw away what it bought), and
    so are the period and Yes/No markets props would refuse anyway, so none of
    them is ever in a request. The phrase names the setting, never the key."""
    name = "BUCKETVALUE_ODDS_MARKETS"
    s = O.read_settings({name: f"player_rebounds,{key.upper()}",
                         "BUCKETVALUE_ODDS_CORE_MARKETS": "player_rebounds"})
    assert s.markets == ("player_rebounds",) and s.core == ("player_rebounds",)
    assert s.problems == (f"setting {name}: {REFUSED_MARKETS[key]}",)
    alone = O.read_settings({name: key})
    assert alone.markets == ("player_points",)
    assert alone.problems == (f"setting {name}: {REFUSED_MARKETS[key]}",
                              f"setting {name}: empty list; default used")
    with pytest.raises(ValueError, match="^bad markets$"):
        O.event_odds_request(PHI_DET, O.BOOKS, (key,))


def test_the_core_markets_are_what_the_full_list_keeps():
    both = "player_points,player_rebounds"
    s = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": both,
                         "BUCKETVALUE_ODDS_CORE_MARKETS": "player_rebounds"})
    assert s.core == ("player_rebounds",) and s.problems == ()
    s = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": "player_rebounds,player_assists"})
    assert s.core == ("player_rebounds",)          # the default core is not in this list
    assert s.problems == ("setting BUCKETVALUE_ODDS_CORE_MARKETS: not in the market list; "
                          "the first market used",)
    s = O.read_settings({"BUCKETVALUE_ODDS_CORE_MARKETS": "player_points_alternate"})
    assert s.core == ("player_points",)
    assert s.problems == (
        "setting BUCKETVALUE_ODDS_CORE_MARKETS: alternate market dropped",
        "setting BUCKETVALUE_ODDS_CORE_MARKETS: empty list; default used")


def test_the_reader_takes_any_catalog_book_in_the_order_named_each_once():
    """C3 (build amendment B35, which ends B6's rule that the setting may only
    narrow five books): any book of the catalog, in the order named, each
    once, whatever its case and spacing. Any other name is dropped with a
    fixed phrase that never carries the name, and an empty result, or an
    unreadable value, means the default (O.BOOKS)."""
    name = "BUCKETVALUE_ODDS_BOOKS"
    s = O.read_settings({name: " PrizePicks , underdog,prizepicks"})
    assert s.books == ("prizepicks", "underdog") and s.problems == ()
    s = O.read_settings({name: " BallyBet , dabble_us_dfs,pick6,ballybet"})
    assert s.books == ("ballybet", "dabble_us_dfs", "pick6") and s.problems == ()
    # Derived, never listed (round BK's review, BKR-3: a swap of the default
    # list edits the pin test alone): the catalog books outside B6's five
    # (the fixtures' books) and those outside the default, each named
    # backwards, upper case, spaced out and once more at the end.
    late = [b for b in O.BOOK_ORDER if b not in FIXTURE_BOOKS]
    off = [b for b in O.BOOK_ORDER if b not in O.BOOKS]           # catalog books off the default
    assert late and off
    for books in (late, off):
        named = tuple(reversed(books))[:O.MAX_BOOKS]
        s = O.read_settings({name: " , ".join(b.upper() for b in named + named[:1])})
        assert s.books == named and s.problems == (), books
    for book in O.BOOK_ORDER:                     # every catalog book, alone
        one = O.read_settings({name: book.upper()})
        assert one.books == (book,) and one.problems == (), book
    s = O.read_settings({name: "williamhill_us,fanatics"})        # paid plans: accepted
    assert s.books == ("williamhill_us", "fanatics") and s.problems == ()
    s = O.read_settings({name: "draftking,fanduel,bet365,BOVADA,betrivers"})
    assert s.books == ("fanduel", "betrivers")
    assert s.problems == (f"setting {name}: unknown key dropped",
                          f"setting {name}: excluded book dropped")
    for p in s.problems:
        assert not re.search(r"draftking\b|bet365|bovada", p, re.I), p
    for nothing in ("bet365,pinnacle", " , ,", "kalshi,novig"):
        s = O.read_settings({name: nothing})
        assert s.books == O.BOOKS, nothing
        assert s.problems[-1] == f"setting {name}: empty list; default used", nothing
    assert O.read_settings({name: ""}) == O.read_settings({})       # blank is unset
    for junk in (["draftkings"], True, {"a": 1}):
        s = O.read_settings({name: junk})
        assert s.books == O.BOOKS, junk
        assert s.problems == (f"setting {name}: unreadable value; default used",), junk


def test_an_eleventh_book_is_refused_because_it_doubles_the_bill():
    """Every 10 books named in `bookmakers` bill as one region (docs 529), so
    the reader keeps the first ten catalog books named and says it dropped
    the rest; a repeated name is not a second book; the request builder
    refuses an 11th whoever asks
    (test_the_builder_refuses_what_the_settings_reader_would)."""
    name = "BUCKETVALUE_ODDS_BOOKS"
    every = O.BOOK_ORDER
    assert len(every) > O.MAX_BOOKS == 10
    ten = O.read_settings({name: ",".join(every[:10])})
    assert ten.books == every[:10] and ten.problems == ()
    eleven = O.read_settings({name: ",".join(every[:11])})
    assert eleven.books == every[:10]
    assert eleven.problems == (f"setting {name}: more than 10 books; the rest dropped",)
    backwards = O.read_settings({name: ",".join(reversed(every))})
    assert backwards.books == tuple(reversed(every))[:10] and len(backwards.problems) == 1
    twice = O.read_settings({name: ",".join(every + every)})
    assert twice.books == every[:10]
    assert twice.problems == (f"setting {name}: more than 10 books; the rest dropped",)
    repeats = O.read_settings({name: ",".join([every[0]] * 12 + list(every[1:10]))})
    assert repeats.books == every[:10] and repeats.problems == ()
    for books in (every[:11], every, O.BOOKS + (every[-1],)):
        with pytest.raises(ValueError, match="^bad books$"):
            O.event_odds_request(PHI_DET, books, ("player_points",))
    path, params = O.event_odds_request(PHI_DET, every[:10], ("player_points",))
    assert len(params["bookmakers"].split(",")) == 10 and "regions" not in params


@pytest.mark.parametrize("name, value, field, want", [
    ("BUCKETVALUE_ODDS_RESERVE", "30", "reserve", 30),
    ("BUCKETVALUE_ODDS_RESERVE", "0", "reserve", 0),
    ("BUCKETVALUE_ODDS_RESERVE", " 30.0 ", "reserve", 30),
    ("BUCKETVALUE_ODDS_RESERVE", 1000, "reserve", 1000),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "9:05", "first_pull_et", dt.time(9, 5)),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "00:00", "first_pull_et", dt.time(0, 0)),
    ("BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN", "45", "last_pull_lead", dt.timedelta(minutes=45)),
    ("BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN", "0", "last_pull_lead", dt.timedelta(0)),
    ("BUCKETVALUE_ODDS_MIN_REFRESH_MIN", "1", "min_refresh", dt.timedelta(minutes=1)),
    ("BUCKETVALUE_ODDS_TIP_GUARD_MIN", "0", "tip_guard", dt.timedelta(0)),
    ("BUCKETVALUE_ODDS_TIP_GUARD_MIN", "10", "tip_guard", dt.timedelta(minutes=10)),
    ("BUCKETVALUE_ODDS_MAX_CALLS_10MIN", "1000", "max_calls_10min", 1000),
    ("BUCKETVALUE_ODDS_PP_TOPUP", "0", "pp_topup", False),
    ("BUCKETVALUE_ODDS_REFRESHER", "0", "refresher", False),
    ("BUCKETVALUE_ODDS_REFRESHER", "on", "refresher", True)])
def test_each_setting_reads_its_value(name, value, field, want):
    s = O.read_settings({name: value})
    assert getattr(s, field) == want and s.problems == ()


@pytest.mark.parametrize("name, value, field, phrase", [
    ("BUCKETVALUE_ODDS_RESERVE", "-5", "reserve", "out of range"),
    ("BUCKETVALUE_ODDS_RESERVE", "lots", "reserve", "unreadable value"),
    ("BUCKETVALUE_ODDS_RESERVE", "30.5", "reserve", "unreadable value"),
    ("BUCKETVALUE_ODDS_RESERVE", "1e3", "reserve", "unreadable value"),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "24:00", "first_pull_et", "unreadable value"),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "0850", "first_pull_et", "unreadable value"),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "8.50", "first_pull_et", "unreadable value"),
    ("BUCKETVALUE_ODDS_FIRST_PULL_ET", "08:60", "first_pull_et", "unreadable value"),
    ("BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN", "721", "last_pull_lead", "out of range"),
    ("BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN", "half an hour", "last_pull_lead",
     "unreadable value"),
    ("BUCKETVALUE_ODDS_MIN_REFRESH_MIN", "0", "min_refresh", "out of range"),
    ("BUCKETVALUE_ODDS_TIP_GUARD_MIN", "-1", "tip_guard", "out of range"),
    ("BUCKETVALUE_ODDS_MAX_CALLS_10MIN", "0", "max_calls_10min", "out of range"),
    ("BUCKETVALUE_ODDS_MAX_CALLS_10MIN", "1001", "max_calls_10min", "out of range"),
    ("BUCKETVALUE_ODDS_PP_TOPUP", "maybe", "pp_topup", "unreadable value"),
    ("BUCKETVALUE_ODDS_REFRESHER", "2", "refresher", "unreadable value")])
def test_a_bad_value_falls_back_to_its_default_and_says_so(name, value, field, phrase):
    s = O.read_settings({name: value})
    assert getattr(s, field) == getattr(DEFAULT_SETTINGS, field)
    assert s.problems == (f"setting {name}: {phrase}; default used",)


def test_problems_name_the_setting_never_its_value():
    """They go to the log. A value could be anything Barrett typed, a key
    pasted into the wrong secret included."""
    marker = "zz" + KEY
    env = {name: marker for name in O.SETTING_DEFAULTS}
    s = O.read_settings(env)
    assert s.problems
    for p in s.problems:
        assert KEY not in p and "zz" not in p, p
        assert re.fullmatch(r"setting BUCKETVALUE_ODDS_[A-Z0-9_]+: [a-z0-9/ ;-]+", p), p
    named = {p.split(":")[0].split()[1] for p in s.problems}
    assert named == set(O.SETTING_DEFAULTS)            # every setting says it fell back
    assert KEY not in repr(s)
    assert s == dataclasses.replace(DEFAULT_SETTINGS, enabled=False, problems=s.problems)


def test_the_reader_never_holds_the_key_and_has_no_layout_setting():
    """The key is live._read_key's alone (design 2.4.3). Layout B is the only
    board, so BUCKETVALUE_BOARD_LAYOUT is not a setting (amendment A10)."""
    env = {"BUCKETVALUE_ODDS_API_KEY": KEY, "BUCKETVALUE_ODDS_API_KEY_FILE": "~/odds_api_key",
           "THE_ODDS_API_KEY": KEY, "BUCKETVALUE_BOARD_LAYOUT": "C"}
    s = O.read_settings(env)
    assert s == DEFAULT_SETTINGS
    assert KEY not in repr(s)
    assert [f.name for f in dataclasses.fields(O.Settings)] == [
        "enabled", "books", "markets", "core", "reserve", "first_pull_et", "last_pull_lead",
        "min_refresh", "tip_guard", "max_calls_10min", "pp_topup", "refresher", "problems"]
    assert not any("LAYOUT" in n or "KEY" in n for n in O.SETTING_DEFAULTS)


# -- Requests ---------------------------------------------------------------------------
def test_the_events_request_is_free_and_carries_no_key():
    assert O.events_request() == ("/v4/sports/basketball_nba/events", {"dateFormat": "iso"})
    assert O.EVENTS_PATH == "/v4/sports/basketball_nba/events"
    assert O.PARTICIPANTS_PATH == "/v4/sports/basketball_nba/participants"


def test_the_event_odds_request_names_the_books_and_sends_no_regions():
    """Up to ten books in `bookmakers` bill as one region and `bookmakers` wins
    over `regions` (docs 529), so no `regions` is sent unless the probe says
    the vendor needs it (m5). No include flag and no key: live adds `apiKey`."""
    s = O.read_settings({})
    path, params = O.event_odds_request(PHI_DET, s.books, s.markets)
    assert path == f"/v4/sports/basketball_nba/events/{PHI_DET}/odds"
    assert params == {"bookmakers": ",".join(O.BOOKS),
                      "markets": "player_points", "oddsFormat": "american", "dateFormat": "iso"}
    _, topup = O.event_odds_request(DEN_SAS, ("prizepicks",), O.MAIN_MARKETS)
    assert topup["bookmakers"] == "prizepicks"
    assert topup["markets"] == ",".join(O.MAIN_MARKETS)
    _, dup = O.event_odds_request(PHI_DET, ("prizepicks", "prizepicks"), ("player_points",) * 2)
    assert dup["bookmakers"] == "prizepicks" and dup["markets"] == "player_points"


def test_regions_are_sent_only_when_the_probe_flips_send_regions(monkeypatch):
    monkeypatch.setattr(O, "SEND_REGIONS", True)
    _, params = O.event_odds_request(PHI_DET, O.BOOKS, ("player_points",))
    assert params["regions"] == "us"
    assert params["bookmakers"] == ",".join(O.BOOKS)


@pytest.mark.parametrize("bad", [
    "", PHI_DET[:31], PHI_DET + "0", PHI_DET[:31] + "/", PHI_DET[:31] + "?",
    "../../../v4/sports/upcoming/odds", PHI_DET[:16] + "-" + PHI_DET[17:], " " + PHI_DET[1:],
    None, 123, PHI_DET.encode(), [PHI_DET]])
def test_a_bad_event_id_is_refused_without_echoing_it(bad):
    with pytest.raises(ValueError) as e:
        O.event_odds_request(bad, O.BOOKS, ("player_points",))
    assert str(e.value) == "bad event id"


@pytest.mark.parametrize("books, markets, msg", [
    (("draftkings", "notabook"), ("player_points",), "bad books"),
    ((), ("player_points",), "bad books"),
    ("draftkings", ("player_points",), "bad books"),
    (None, ("player_points",), "bad books"),
    (O.BOOK_ORDER[:11], ("player_points",), "bad books"),
    (("draftkings", "bovada"), ("player_points",), "bad books"),
    (("DraftKings",), ("player_points",), "bad books"),
    (O.BOOKS, (), "bad markets"),
    (O.BOOKS, "player_points", "bad markets"),
    (O.BOOKS, ("player_points", 5), "bad markets"),
    (O.BOOKS, ("player_points_alternate",), "bad markets")])
def test_the_builder_refuses_what_the_settings_reader_would(books, markets, msg):
    with pytest.raises(ValueError) as e:
        O.event_odds_request(PHI_DET, books, markets)
    assert str(e.value) == msg


# -- Usage ------------------------------------------------------------------------------
def test_usage_comes_from_the_three_headers():
    u = O.usage_from_headers({"x-requests-remaining": "475", "x-requests-used": "25",
                              "x-requests-last": "1", "content-type": "application/json"})
    assert u == O.Usage(remaining=475, used=25, last=1) and u.allowance == 500
    mixed = O.usage_from_headers({"X-Requests-Remaining": "19000", "X-REQUESTS-USED": "1000",
                                  "X-Requests-Last": "0"})
    assert mixed == O.Usage(19000, 1000, 0) and mixed.allowance == 20000
    assert O.usage_from_headers({"x-requests-remaining": "499.0", "x-requests-used": 1,
                                 "x-requests-last": 2.0}) == O.Usage(499, 1, 2)


@pytest.mark.parametrize("headers, want", [
    ({}, O.Usage(None, None, None)),
    (None, O.Usage(None, None, None)),
    ("x-requests-remaining: 5", O.Usage(None, None, None)),
    ({"x-requests-remaining": "475"}, O.Usage(475, None, None)),
    ({"x-requests-used": "25", "x-requests-last": "1"}, O.Usage(None, 25, 1)),
    ({"x-requests-remaining": "", "x-requests-used": "lots", "x-requests-last": "-1"},
     O.Usage(None, None, None)),
    ({"x-requests-remaining": "4.5", "x-requests-used": True, "x-requests-last": None},
     O.Usage(None, None, None)),
    ({"x-requests-remaining": float("nan"), "x-requests-used": float("inf")},
     O.Usage(None, None, None))])
def test_missing_or_unreadable_usage_is_unknown_never_zero(headers, want):
    """Unknown means no paid call at all (design 3.1); a 0 here would read as
    an empty balance or a free call."""
    u = O.usage_from_headers(headers)
    assert u == want
    if want.remaining is None or want.used is None:
        assert u.allowance is None


# -- Errors -----------------------------------------------------------------------------
DOCUMENTED = (   # UT/error_codes.txt, every "## #" heading in page order
    "MISSING_KEY", "INVALID_KEY", "DEACTIVATED_KEY", "EXCEEDED_FREQ_LIMIT",
    "OUT_OF_USAGE_CREDITS", "MISSING_REGION", "INVALID_REGION", "INVALID_BOOKMAKERS",
    "MISSING_MARKET", "INVALID_MARKET", "INVALID_MARKET_COMBO", "INVALID_DATE_FORMAT",
    "INVALID_ODDS_FORMAT", "INVALID_ALL_SPORTS_PARAM", "INVALID_SPORT", "UNKNOWN_SPORT",
    "INVALID_SCORES_DAYS_FROM", "INVALID_EVENT_IDS", "INVALID_EVENT_ID", "EVENT_NOT_FOUND",
    "MISSING_HISTORICAL_TIMESTAMP", "INVALID_HISTORICAL_TIMESTAMP", "INVALID_COMMENCE_TIME_FROM",
    "INVALID_COMMENCE_TIME_TO", "INVALID_COMMENCE_TIME_RANGE",
    "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN", "INVALID_PARTICIPANT_ID",
    "INVALID_INCLUDE_LINKS", "INVALID_INCLUDE_SIDS", "INVALID_INCLUDE_BET_LIMITS",
    "INVALID_INCLUDE_MULTIPLIERS", "INVALID_INCLUDE_ROTATION_NUMBERS",
    "HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE", "INVALID_STATUS")


def test_all_34_documented_codes_are_known_in_page_order():
    """m4: v1 knew nine. INVALID_STATUS and INVALID_INCLUDE_MULTIPLIERS are
    what the probe's --evening run and step 5 may meet."""
    assert O.ERROR_CODES == DOCUMENTED
    assert len(O.ERROR_CODES) == 34 == len(set(O.ERROR_CODES))


@pytest.mark.parametrize("code", DOCUMENTED)
def test_each_code_is_read_from_the_scrubbed_error_body(code):
    """The body's field names are UNVERIFIED (RF 2.7), so the code is found
    anywhere in the snippet, after http.scrub has taken the key out."""
    body = json.dumps({"error_code": code, "message": f"rejected apiKey={KEY}",
                       "details_url": "https://the-odds-api.com/liveapi/guides/v4/"
                                      f"api-error-codes.html#{code.lower()}"})
    snippet = H.scrub(body, (KEY,))
    assert KEY not in snippet
    assert O.error_code(422, snippet) == code
    assert O.error_code(401, f"Error: {code}.") == code
    assert O.error_code(401, code) == code


@pytest.mark.parametrize("snippet, want", [
    ("INVALID_EVENT_IDS", "INVALID_EVENT_IDS"),
    ('{"error_code":"INVALID_EVENT_ID"}', "INVALID_EVENT_ID"),
    ("bad: INVALID_EVENT_IDS, see docs", "INVALID_EVENT_IDS"),
    ('{"error_code": "INVALID_MARKET_COMBO"}', "INVALID_MARKET_COMBO"),
    ("see INVALID_MARKET.", "INVALID_MARKET"),
    ("XINVALID_KEY", "HTTP_401"), ("INVALID_KEYS", "HTTP_401"), ("INVALID_KEY2", "HTTP_401"),
    ("MISSING_KEY_FILE", "HTTP_401"), ("invalid_key", "HTTP_401"),
    ("OUT_OF_USAGE_CREDITSX", "HTTP_401"), ("NOT_OUT_OF_USAGE_CREDITS", "HTTP_401"),
    ("INVALID_KEY then OUT_OF_USAGE_CREDITS", "INVALID_KEY"),
    ("OUT_OF_USAGE_CREDITS then INVALID_KEY", "OUT_OF_USAGE_CREDITS")])
def test_codes_count_only_as_whole_words(snippet, want):
    """INVALID_EVENT_ID is never read inside INVALID_EVENT_IDS, and no code
    inside a longer word; with two codes, the first in the text."""
    assert O.error_code(401, snippet) == want


@pytest.mark.parametrize("status, snippet, want", [
    (500, "Internal error <b>retry</b>", "HTTP_500"), (401, "", "HTTP_401"),
    (401, None, "HTTP_401"), (503, b"INVALID_KEY", "HTTP_503"), (429, 42, "HTTP_429"),
    (True, "x", "HTTP_0"), ("401", "", "HTTP_0"), (None, None, "HTTP_0"), (1000, "", "HTTP_0"),
    (302, "Found. Redirecting to https://elsewhere.example/", "HTTP_302")])
def test_with_no_code_the_status_stands_and_no_vendor_text_comes_back(status, snippet, want):
    got = O.error_code(status, snippet)
    assert got == want
    assert got in O.ERROR_CODES or re.fullmatch(r"HTTP_\d{1,3}", got)


@pytest.mark.parametrize("status, body, kind", [
    (401, '{"error_code": "OUT_OF_USAGE_CREDITS"}', "out_of_credits"),
    (401, '{"error_code": "INVALID_KEY"}', "bad_key"),
    (401, '{"error_code": "DEACTIVATED_KEY"}', "bad_key"),
    (401, '{"error_code": "MISSING_KEY"}', "bad_key"),
    (401, "", "bad_key"), (401, "Unauthorized", "bad_key"),
    (401, '{"error_code": "EXCEEDED_FREQ_LIMIT"}', "bad_key"),
    (200, "", "ok"), (404, '{"error_code": "EVENT_NOT_FOUND"}', "not_found"),
    (422, '{"error_code": "MISSING_REGION"}', "bad_request"),
    (429, '{"error_code": "OUT_OF_USAGE_CREDITS"}', "rate_limited"),
    (301, "", "server"), (302, "", "server"), (400, "", "server"), (403, "Forbidden", "server"),
    (500, "", "server"), (502, "", "server"), (503, "", "server"), (0, "", "server")])
def test_a_401_is_out_of_credits_only_when_its_code_says_so(status, body, kind):
    """Design 3.8: out of credits stops paid calls and keeps the free events
    call watching; every other 401 is a bad key and stops everything. The
    status decides first: OUT_OF_USAGE_CREDITS on a 429 is a rate limit."""
    code = O.error_code(status, H.scrub(body))
    assert O.failure_kind(status, code) == kind
    assert kind in O.FAILURE_KINDS


# -- Events -------------------------------------------------------------------------------
def test_every_vendor_team_name_maps_to_its_tricode():
    names = [p["full_name"] for p in fixture("participants_constructed.json")]
    codes = [teams.canonical(n) for n in names]
    assert len(names) == 30 and sorted(codes) == sorted(teams.CODES)
    events = [{"id": f"{i:032d}", "commence_time": "2026-03-12T23:00:00Z",
               "home_team": names[2 * i], "away_team": names[2 * i + 1]} for i in range(15)]
    df, stats = O.parse_events(events, DAY)
    assert stats["kept"] == 15 and stats["unknown_team"] == []
    assert sorted(df["home"].tolist() + df["away"].tolist()) == sorted(teams.CODES)
    clippers = [{"id": f"{i:032d}", "commence_time": "2026-03-12T23:00:00Z",
                 "home_team": name, "away_team": "Utah Jazz"}
                for i, name in enumerate(("LA Clippers", "Los Angeles Clippers"))]
    assert list(O.parse_events(clippers, DAY)[0]["home"]) == ["LAC", "LAC"]


def test_events_keep_todays_et_games_in_tip_order():
    """The endpoint lists in-play and later games too (docs 1372). Four of the
    slate's games start on the next UTC day and are still 2026-03-12 in ET."""
    df, stats = O.parse_events(fixture("events_constructed.json"), DAY)
    assert list(df.columns) == O.EVENT_COLUMNS
    assert str(df["commence_utc"].dtype) == "datetime64[ns, UTC]"
    assert stats == {"events_seen": 10, "kept": 9, "bad_event": 0, "unknown_team_events": 0,
                     "unknown_team": [], "other_date": 1, "duplicate_id": 0,
                     "bad_payload": False}
    assert list(zip(df["away"], df["home"])) == [
        ("PHX", "IND"), ("WAS", "ORL"), ("PHI", "DET"), ("BKN", "ATL"), ("MIL", "MIA"),
        ("DAL", "MEM"), ("DEN", "SAS"), ("BOS", "OKC"), ("CHI", "LAL")]
    assert set(df["et_date"]) == {DAY}
    assert df["commence_utc"].is_monotonic_increasing
    late = df[df["commence_utc"] >= pd.Timestamp("2026-03-13", tz="UTC")]
    assert list(late["home"]) == ["MEM", "SAS", "OKC", "LAL"]
    nxt, nstats = O.parse_events(fixture("events_constructed.json"), dt.date(2026, 3, 13))
    assert list(zip(nxt["away"], nxt["home"])) == [("GSW", "LAC")]   # "Los Angeles Clippers"
    assert nstats["other_date"] == 9


def test_unknown_teams_are_counted_and_named_never_raised():
    base = {"commence_time": "2026-03-12T23:00:00Z"}
    events = [{**base, "id": f"{1:032d}", "home_team": "Detroit Pistons",
               "away_team": "Philadelphia 76ers"},
              {**base, "id": f"{2:032d}", "home_team": "Team World", "away_team": "Team USA"},
              {**base, "id": f"{3:032d}", "home_team": "Detroit Pistons",
               "away_team": "Seattle SuperSonics Of The Pacific Northwest Basketball Club"},
              {**base, "id": f"{4:032d}", "home_team": {"name": "Detroit Pistons"},
               "away_team": None},
              {**base, "id": f"{5:032d}", "home_team": "Detroit\nPistons\x00 Reserve",
               "away_team": "Utah Jazz"}]
    events += [{**base, "id": f"{9 + i:032d}", "home_team": f"Club {i}", "away_team": "Utah Jazz"}
               for i in range(10)]
    df, stats = O.parse_events(events, DAY)
    assert list(df["home"]) == ["DET"]
    assert stats["unknown_team_events"] == 14 and stats["events_seen"] == 15
    names = stats["unknown_team"]
    assert len(names) == 8                                   # at most 8 kept for the log
    assert names[:5] == ["Team World", "Team USA", "Seattle SuperSonics Of The Pacific North",
                         "(not text)", "DetroitPistons Reserve"]
    assert all(len(n) <= 40 and n.isprintable() and "\n" not in n for n in names)


def test_a_logged_vendor_name_never_carries_a_key_shaped_token():
    """The names go to a WARNING line. Whatever the vendor sends, no run of 32
    or more letters and digits reaches it (http.scrub's rule), even when the
    rest of the name is cut away."""
    base = {"commence_time": "2026-03-12T23:00:00Z", "away_team": "Utah Jazz"}
    events = [{**base, "id": f"{1:032d}", "home_team": KEY},
              {**base, "id": f"{2:032d}", "home_team": "\x00" * 50 + KEY},
              {**base, "id": f"{3:032d}", "home_team": "Team " + KEY + " Reserve"}]
    _, stats = O.parse_events(events, DAY)
    assert stats["unknown_team"] == ["[redacted]", "Team [redacted] Reserve"]
    payload = ev([bk(KEY.lower(), mk("player_points", pair("Cade Cunningham", 25.5))),
                  bk("draftkings", mk("player_" + KEY.lower(), pair("Cade Cunningham", 25.5)))])
    a = parse(payload).attrs
    assert a["unknown_books"] == ["[redacted]"]
    assert a["unmapped_keys"] == ["player_[redacted]"]


def test_the_parsers_never_change_what_they_are_given():
    payload, events_payload = phi_det(), fixture("events_constructed.json")
    before, events_before = copy.deepcopy(payload), copy.deepcopy(events_payload)
    new = parse(payload, at=ELEVEN)
    events, _ = O.parse_events(events_payload, DAY)
    games = serve_games()
    games_before = games.copy()
    O.match_events(events, games)
    assert payload == before and events_payload == events_before
    pd.testing.assert_frame_equal(games, games_before)
    prev = first_pull()
    prev_before, new_before = prev.copy(), new.copy()
    O.merge_refresh(prev, new, FIXTURE_BOOKS)
    pd.testing.assert_frame_equal(prev, prev_before)
    pd.testing.assert_frame_equal(new, new_before)


@pytest.mark.parametrize("payload", [None, {}, "events", 5, [None, 1, "x", [], {}]])
def test_junk_events_answers_never_raise(payload):
    df, stats = O.parse_events(payload, DAY)
    assert df.empty and list(df.columns) == O.EVENT_COLUMNS
    assert stats["events_seen"] == (stats["kept"] + stats["bad_event"] + stats["other_date"]
                                    + stats["unknown_team_events"] + stats["duplicate_id"])


def test_bad_rows_in_an_events_answer_cost_only_themselves():
    good = {"id": PHI_DET, "commence_time": "2026-03-12T23:00:00Z",
            "home_team": "Detroit Pistons", "away_team": "Philadelphia 76ers"}
    rows = [good, dict(good),                                   # a repeated id
            {**good, "id": PHI_DET[:31]}, {**good, "id": None}, {**good, "id": 7},
            {**good, "commence_time": "2026-03-12T23:00:00"},   # naive: never assumed UTC
            {**good, "commence_time": "tonight"}, {**good, "commence_time": 1773356400},
            {**good, "away_team": "Detroit Pistons"}]           # a team against itself
    df, stats = O.parse_events(rows, DAY)
    assert list(df["event_id"]) == [PHI_DET]
    assert stats["duplicate_id"] == 1 and stats["bad_event"] == 7
    assert stats["events_seen"] == 9 == (stats["kept"] + stats["bad_event"]
                                         + stats["duplicate_id"])
    with pytest.raises(TypeError):
        O.parse_events(rows, dt.datetime(2026, 3, 12, 9, tzinfo=ET))   # a date, not a time


# Stamps a vendor could send as placeholders or junk: Go's zero time, and
# instants that leave the calendar once their offset is applied (or would, in
# ET). Before build 4's review (L1-1) each raised OverflowError out of the
# parser, which then took the whole walk down.
OUT_OF_RANGE = ("0001-01-01T00:00:00Z", "9999-12-31T23:00:00-12:00", "0001-01-01T00:00:00+05:00",
                "1899-12-31T23:59:59Z", "2201-01-01T00:00:00Z")


def test_out_of_range_commence_times_cost_only_their_own_event_l1_1():
    good = {"id": PHI_DET, "commence_time": "2026-03-12T23:00:00Z",
            "home_team": "Detroit Pistons", "away_team": "Philadelphia 76ers"}
    rows = [{**good, "id": "a" * 31 + str(i), "commence_time": t}
            for i, t in enumerate(OUT_OF_RANGE)] + [good]
    df, stats = O.parse_events(rows, DAY)
    assert list(df["event_id"]) == [PHI_DET]
    assert stats["bad_event"] == len(OUT_OF_RANGE) and stats["kept"] == 1
    assert stats["events_seen"] == len(rows) == stats["kept"] + stats["bad_event"]
    for t in OUT_OF_RANGE:
        assert O._utc(t) is None, t
    assert O._utc("2026-03-12T19:00:00-04:00") == dt.datetime(2026, 3, 12, 23, 0, tzinfo=UTC)


def test_out_of_range_stamps_in_an_event_odds_answer_never_raise_l1_1():
    """A commence_time or a market's last_update out of range is unread, as
    an unreadable one is (the lines stay; the stamp is NaN), never raised."""
    for t in OUT_OF_RANGE:
        payload = ev([bk("draftkings", mk("player_points", pair("Cade Cunningham", 25.5),
                                          last_update=t))], commence_time=t)
        df = parse(payload)
        assert list(zip(df["book"], df["name"], df["line"])) == [
            ("draftkings", "Cade Cunningham", 25.5)], t
        assert df["start_time"].isna().all() and df["last_update"].isna().all(), t
        accounted(df, payload)


def serve_games() -> pd.DataFrame:
    return pd.read_parquet(SERVE / "schedule.parquet")


def test_events_match_the_serve_fixtures_two_games():
    events, _ = O.parse_events(fixture("events_constructed.json"), DAY)
    m, stats = O.match_events(events, serve_games())
    assert list(m.columns) == O.MATCH_COLUMNS
    assert list(zip(m["game_id"], m["event_id"])) == [("0022500953", PHI_DET),
                                                       ("0022500958", DEN_SAS)]
    assert list(m["game_date"]) == ["2026-03-12"] * 2
    assert list(zip(m["away"], m["home"])) == [("PHI", "DET"), ("DEN", "SAS")]
    assert list(m["swapped"]) == [False, False]
    assert list(m["tip_utc"]) == [pd.Timestamp("2026-03-12T23:00Z"),
                                  pd.Timestamp("2026-03-13T01:00Z")]
    assert stats == {"events": 9, "games": 2, "matched": 2, "swapped": 0, "event_no_game": 7,
                     "game_no_event": 0, "refused_event": 0, "refused_game": 0,
                     "bad_event": 0, "bad_game": 0, "swapped_games": []}
    as_text = serve_games().assign(game_date=lambda d: d["game_date"].astype(str))
    assert O.match_events(events, as_text)[0].equals(m)      # an ISO date string reads the same


def neutral(eid: str, when: str, home: str, away: str) -> dict:
    return {"id": eid, "sport_key": "basketball_nba", "commence_time": when,
            "home_team": home, "away_team": away}


def test_a_swapped_home_and_away_still_matches_and_is_counted(caplog):
    """m18. The Mexico City game of 2026-11-07 is DEN at IND on the schedule;
    a vendor may list Denver as home."""
    games = pd.DataFrame([{"game_id": "0022600189", "game_date": dt.date(2026, 11, 7),
                           "tip_utc": pd.Timestamp("2026-11-07T22:00Z"), "home": "IND",
                           "away": "DEN"}])
    payload = [neutral("d" * 32, "2026-11-07T22:00:00Z", "Denver Nuggets", "Indiana Pacers")]
    events, _ = O.parse_events(payload, dt.date(2026, 11, 7))
    with caplog.at_level(logging.INFO, logger="bucketvalue"):
        m, stats = O.match_events(events, games)
    assert list(m["game_id"]) == ["0022600189"] and list(m["swapped"]) == [True]
    assert (m.loc[0, "home"], m.loc[0, "away"]) == ("IND", "DEN")          # the schedule's
    assert (m.loc[0, "event_home"], m.loc[0, "event_away"]) == ("DEN", "IND")
    assert stats["swapped"] == 1 and stats["swapped_games"] == ["0022600189"]
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert logged == ["oddsapi: event matched with home and away swapped: "
                      "0022600189 (DEN at IND)"]


def test_the_paris_and_manchester_games_three_days_apart_stay_apart():
    """m18. NOP at SAS in Paris on 2027-01-14 and SAS at NOP in Manchester on
    2027-01-17: the same pair of teams. Only the date keeps them apart. (Real
    schedule rows, typed in as literals; a pure function, no clock.)"""
    games = pd.DataFrame([
        {"game_id": "0022600075", "game_date": dt.date(2027, 1, 14),
         "tip_utc": pd.Timestamp("2027-01-14T19:00Z"), "home": "SAS", "away": "NOP"},
        {"game_id": "0022600076", "game_date": dt.date(2027, 1, 17),
         "tip_utc": pd.Timestamp("2027-01-17T15:30Z"), "home": "NOP", "away": "SAS"}])
    paris = neutral("a" * 32, "2027-01-14T19:00:00Z", "New Orleans Pelicans", "San Antonio Spurs")
    manchester = neutral("b" * 32, "2027-01-17T15:30:00Z", "New Orleans Pelicans",
                         "San Antonio Spurs")
    on14, s14 = O.parse_events([paris, manchester], dt.date(2027, 1, 14))
    on17, s17 = O.parse_events([paris, manchester], dt.date(2027, 1, 17))
    assert list(on14["event_id"]) == ["a" * 32] and s14["other_date"] == 1
    assert list(on17["event_id"]) == ["b" * 32] and s17["other_date"] == 1
    both = pd.concat([on14, on17], ignore_index=True)
    m, stats = O.match_events(both, games)
    assert dict(zip(m["game_id"], m["event_id"])) == {"0022600075": "a" * 32,
                                                      "0022600076": "b" * 32}
    assert dict(zip(m["game_id"], m["swapped"])) == {"0022600075": True, "0022600076": False}
    assert stats["refused_event"] == 0 and stats["matched"] == 2
    only17, s = O.match_events(on17, games)                 # the 14th's game has no event
    assert list(only17["game_id"]) == ["0022600076"]
    assert s["game_no_event"] == 1 and s["event_no_game"] == 0


def test_two_events_for_one_game_are_both_refused():
    events, _ = O.parse_events(fixture("events_constructed.json"), DAY)
    twin = events[events["event_id"] == PHI_DET].assign(event_id="f" * 32)
    m, stats = O.match_events(pd.concat([events, twin], ignore_index=True), serve_games())
    assert list(m["game_id"]) == ["0022500958"]
    assert stats["refused_event"] == 2 and stats["refused_game"] == 1
    assert stats["events"] == 10 == (stats["matched"] + stats["event_no_game"]
                                     + stats["refused_event"] + stats["bad_event"])
    assert stats["games"] == 2 == (stats["matched"] + stats["game_no_event"]
                                   + stats["refused_game"] + stats["bad_game"])


def test_games_without_an_event_or_without_teams_are_counted():
    events, _ = O.parse_events(fixture("events_constructed.json"), DAY)
    games = pd.concat([serve_games(), pd.DataFrame([
        {"game_id": "0022500961", "game_date": DAY, "home": "TOR", "away": "NYK"},
        {"game_id": "0062500001", "game_date": DAY, "home": None, "away": None},   # TBD
        {"game_id": "0022500962", "game_date": None, "home": "UTA", "away": "SAC"}])],
        ignore_index=True)
    m, stats = O.match_events(events, games)
    assert stats["matched"] == 2 and stats["game_no_event"] == 1 and stats["bad_game"] == 2
    for junk in (None, pd.DataFrame(), [], "games"):
        empty, s = O.match_events(junk, games)
        assert empty.empty and list(empty.columns) == O.MATCH_COLUMNS
        assert s["game_no_event"] == 3 and s["matched"] == 0
        assert O.match_events(events, junk)[1]["event_no_game"] == 9


# -- Parsing event odds ------------------------------------------------------------------
def test_phi_det_parses_to_one_line_per_book_player_and_market():
    payload = phi_det()
    df = parse(payload)
    assert list(df.columns) == O.FEED_COLUMNS
    assert df.groupby("book").size().to_dict() == {"draftkings": 10, "fanduel": 5, "betmgm": 4,
                                                   "prizepicks": 9, "underdog": 3}
    assert not df.duplicated(["book", "name", "market"]).any()
    a = accounted(df, payload)
    assert (a["outcomes_seen"], a["kept_outcomes"], a["rows"]) == (64, 59, 31)
    assert {k: a[k] for k in O.COUNTERS} == {
        "unknown_book": 0, "unmapped_market": 0, "refused_market": 0, "alt_dropped": 0,
        "unrequested_market": 0, "fantasy_dropped": 2, "bad_outcome": 1,
        "duplicate_outcome": 0, "extra_points": 2}
    assert a["books_answered"] == FIXTURE_BOOKS and a["bookmaker_entries"] == 5
    assert (a["event_id"], a["event_home"], a["event_away"]) == (PHI_DET, "DET", "PHI")


def test_sides_come_from_pairs_and_from_one_sided_lines():
    df = parse(phi_det())
    cade = one(df, "draftkings", "Cade Cunningham")
    assert (cade["line"], cade["direction"], cade["price_over"], cade["price_under"]) == (
        25.5, "both", -115.0, -105.0)
    jenkins = one(df, "fanduel", "Daniss Jenkins")             # Over only
    assert (jenkins["line"], jenkins["direction"], jenkins["price_over"]) == (7.5, "more", -125.0)
    assert math.isnan(jenkins["price_under"])
    duren = one(df, "betmgm", "Jalen Duren")                   # Under only
    assert (duren["line"], duren["direction"], duren["price_under"]) == (13.5, "less", -115.0)
    assert math.isnan(duren["price_over"])
    assert set(df["direction"]) == {"both", "more", "less"}


def test_a_missing_point_skips_the_outcome_and_is_never_read_as_0():
    """BetMGM's VJ Edgecombe Over has no point: only his
    Under at 16.5 stands, as an Under-only line."""
    df = parse(phi_det())
    vj = one(df, "betmgm", "VJ Edgecombe")
    assert (vj["line"], vj["direction"]) == (16.5, "less")
    assert df.attrs["by_book"]["betmgm"]["bad_outcome"] == 1
    assert (df["line"] > 0).all()


@pytest.mark.parametrize("point", [None, "", "abc", "NaN", float("nan"), float("inf"), -0.5,
                                   1000.5, True, [24.5], {"v": 24.5}, 10 ** 400, "1e400"])
def test_an_unreadable_point_is_counted_never_a_line(point):
    o = {"name": "Over", "description": "Cade Cunningham", "price": -110, "point": point}
    payload = ev([bk("draftkings", mk("player_points", [o], pair("Jalen Duren", 13.5)))])
    df = parse(payload)
    assert list(df["name"]) == ["Jalen Duren"]
    assert accounted(df, payload)["bad_outcome"] == 1


def test_an_absent_point_key_and_a_real_zero_are_different():
    absent = side("Over", "Cade Cunningham", point=None)
    del absent[0]["point"]
    zero = side("Over", "Jalen Duren", point=0) + side("Under", "Jalen Duren", point=0)
    text = side("Over", "Tobias Harris", point="12.5") + side("Under", "Tobias Harris", point=12.5)
    df = parse(ev([bk("draftkings", mk("player_points", absent, zero, text))]))
    assert sorted(zip(df["name"], df["line"], df["direction"])) == [
        ("Jalen Duren", 0.0, "both"), ("Tobias Harris", 12.5, "both")]
    assert df.attrs["bad_outcome"] == 1


def test_the_main_line_is_the_pair_closest_to_even_not_the_first_listed():
    """FanDuel lists Cade Cunningham's 24.5 (-142/+112) before his 25.5
    (-114/-106): the main line is 25.5, and 24.5's two outcomes are counted."""
    df = parse(phi_det())
    cade = one(df, "fanduel", "Cade Cunningham")
    assert (cade["line"], cade["price_over"], cade["price_under"]) == (25.5, -114.0, -106.0)
    assert df.attrs["by_book"]["fanduel"]["extra_points"] == 2


def test_dfs_and_one_sided_ties_take_the_first_point():
    """DFS prices are indicative, so for PrizePicks the first both-sided point
    stands; a both-sided point beats a one-sided one; with no pair, the first."""
    payload = ev([
        bk("prizepicks", mk("player_points", pair("Cade Cunningham", 26.5, 120, -150),
                            pair("Cade Cunningham", 25.5, -137, -137))),
        bk("draftkings", mk("player_points", side("Over", "Cade Cunningham", 24.5),
                            pair("Cade Cunningham", 25.5),
                            side("Over", "Jalen Duren", 14.5), side("Under", "Jalen Duren", 13.5),
                            side("Under", "Tobias Harris", 12.5, price=-120),
                            side("Under", "Tobias Harris", 11.5, price=-100)))])
    df = parse(payload)
    assert one(df, "prizepicks", "Cade Cunningham")["line"] == 26.5
    assert one(df, "draftkings", "Cade Cunningham")["line"] == 25.5
    duren = one(df, "draftkings", "Jalen Duren")
    assert (duren["line"], duren["direction"]) == (14.5, "more")
    harris = one(df, "draftkings", "Tobias Harris")
    assert (harris["line"], harris["direction"]) == (12.5, "less")
    a = accounted(df, payload)
    assert a["by_book"]["prizepicks"]["extra_points"] == 2
    assert a["by_book"]["draftkings"]["extra_points"] == 3


def test_dfs_prices_are_blanked_and_sportsbook_prices_kept():
    df = parse(phi_det())
    dfs = df[df["book"].isin(["prizepicks", "underdog"])]
    assert len(dfs) == 12 and dfs["price_over"].isna().all() and dfs["price_under"].isna().all()
    books = df[~df["book"].isin(["prizepicks", "underdog"])]
    offered_over = books["direction"].isin(["both", "more"])
    offered_under = books["direction"].isin(["both", "less"])
    assert books.loc[offered_over, "price_over"].notna().all()
    assert books.loc[offered_under, "price_under"].notna().all()
    for book in ("pick6", "dabble_us_dfs"):                  # any us_dfs book the settings name
        other = parse(ev([bk(book, mk("player_points", pair("Cade Cunningham", 25.5, -137, -137)))]),
                      books=(book,))
        assert len(other) == 1 and other["price_over"].isna().all()


def test_every_feed_row_is_a_standard_line_in_props_shape():
    """Demons and Goblins live only in `_alternate` markets, which the feed
    never buys (M11), so every row is "standard". The book goes in `book`,
    never in `source` or `odds_type`; vendor titles are never shown."""
    df = parse(phi_det())
    assert set(df["odds_type"]) == {"standard"} and set(df["source"]) == {"oddsapi"}
    assert set(df["league"]) == {"NBA"} and set(df["is_combo"]) == {False}
    for c in ("team", "opp", "position", "pick", "alt_pick", "pair", "pair_loose", "game_id",
              "game_date", "tip_utc"):
        assert df[c].isna().all(), c
    assert set(df["start_time"]) == {"2026-03-12T23:00:00Z"}
    assert set(df["event_id"]) == {PHI_DET}
    assert set(zip(df["event_home"], df["event_away"])) == {("DET", "PHI")}
    assert dict(zip(df["book"], df["book_title"])) == {b: O.BOOK_TITLE[b] for b in FIXTURE_BOOKS}
    assert "Underdog Fantasy" not in set(df["book_title"])
    assert set(df["market"]) == set(FOUR)
    assert set(df["stat_type"]) == {"Points", "Rebounds", "Assists", "Fantasy Score"}
    titles = parse(ev([bk(b, mk("player_points", pair("Cade Cunningham", 25.5)))
                       for b in O.BOOK_ORDER]), books=O.BOOK_ORDER)
    assert len(titles) == len(O.BOOK_ORDER) and not (titles["book_title"] == titles["book"]).any()
    assert dict(zip(titles["book"], titles["book_title"])) == O.BOOK_TITLE       # every book ours


def test_alternates_and_refused_markets_are_counted_and_never_kept():
    """M11: an `_alternate` key is never requested; if the vendor sends one
    anyway its lines are counted in alt_dropped, never kept as the standard
    line. Period and Yes/No markets are refused, unknown ones unmapped."""
    plain = parse(phi_det())
    payload = phi_det()
    entry(payload, "prizepicks")["markets"] += [
        mk("player_points_alternate", side("Over", "Cade Cunningham", 29.5, 100),
           side("Over", "Cade Cunningham", 21.5, -137), side("Over", "Jalen Duren", 17.5, 100)),
        mk("player_points_q1", pair("Cade Cunningham", 6.5)),
        mk("player_double_double", [{"name": "Yes", "description": "Jalen Duren", "price": 150},
                                    {"name": "No", "description": "Jalen Duren", "price": -190}]),
        mk("player_field_goals", pair("Cade Cunningham", 9.5)),
        mk("player_points_rebounds_assists_alternate", pair("Cade Cunningham", 39.5))]
    df = parse(payload)
    a = accounted(df, payload)
    pp = a["by_book"]["prizepicks"]
    assert (pp["alt_dropped"], pp["refused_market"], pp["unmapped_market"]) == (5, 4, 2)
    assert a["unmapped_keys"] == ["player_field_goals"]
    assert df.equals(plain)                                  # not one line more or changed
    assert not df["market"].str.endswith("_alternate").any()


def test_last_update_comes_from_the_market_level_only():
    """The bookmaker-level stamp is deprecated and absent on event odds (docs
    1053, 1501). Given one anyway, it is never read: a market with no stamp of
    its own has none."""
    payload = phi_det()
    entry(payload, "draftkings")["last_update"] = "2026-03-12T14:59:59Z"
    del market(payload, "draftkings", "player_points")["last_update"]
    df = parse(payload)
    dk = df[df["book"] == "draftkings"]
    assert dk.loc[dk["market"] == "player_points", "last_update"].isna().all()
    assert set(dk.loc[dk["market"] != "player_points", "last_update"]) == {
        pd.Timestamp("2026-03-12T12:52:14Z")}
    assert pd.Timestamp("2026-03-12T14:59:59Z") not in set(df["last_update"].dropna())
    assert set(df.loc[df["book"] == "prizepicks", "last_update"]) == {
        pd.Timestamp("2026-03-12T12:53:30Z")}


def test_fetched_at_is_the_callers_clock_on_every_row():
    df = parse(phi_det(), at=NINE)
    assert set(df["fetched_at"]) == {pd.Timestamp("2026-03-12T13:00:00Z")}
    assert str(df["fetched_at"].dtype) == "datetime64[ns, UTC]"
    for bad in (dt.datetime(2026, 3, 12, 9, 0), "2026-03-12T13:00:00Z", None, DAY):
        with pytest.raises(ValueError, match="^fetched_at must be an aware datetime$"):
            parse(phi_det(), at=bad)


def test_fantasy_lines_are_kept_for_prizepicks_only():
    df = parse(phi_det())
    fantasy = df[df["market"] == "player_fantasy_points"]
    assert list(zip(fantasy["book"], fantasy["name"], fantasy["line"])) == [
        ("prizepicks", "Cade Cunningham", 47.5)]
    assert df.attrs["by_book"]["underdog"]["fantasy_dropped"] == 2
    payload = ev([bk("draftkings", mk("player_fantasy_points", pair("Cade Cunningham", 47.5)))])
    assert parse(payload).empty and parse(payload).attrs["fantasy_dropped"] == 2


def test_only_the_requested_books_and_markets_count():
    payload = phi_det()
    pp = parse(payload, books=("prizepicks",))
    assert set(pp["book"]) == {"prizepicks"} and len(pp) == 9
    a = accounted(pp, payload)
    assert a["unknown_book"] == 46 and a["books_answered"] == ("prizepicks",)
    assert sorted(a["unknown_books"]) == ["betmgm", "draftkings", "fanduel", "underdog"]
    points = parse(payload, markets=("player_points",))
    assert set(points["market"]) == {"player_points"} and len(points) == 25
    b = accounted(points, payload)
    assert (b["unrequested_market"], b["fantasy_dropped"], b["kept_outcomes"]) == (14, 0, 47)
    extra = phi_det()
    extra["bookmakers"].append(bk("betrivers", mk("player_points", pair("Cade Cunningham", 25.5))))
    c = accounted(parse(extra), extra)
    assert c["unknown_book"] == 2 and c["unknown_books"] == ["betrivers"]


def test_den_sas_has_no_prizepicks_entry():
    payload = den_sas()
    df = parse(payload)
    assert df.groupby("book").size().to_dict() == {"draftkings": 11, "fanduel": 5, "betmgm": 2,
                                                   "underdog": 4}
    a = accounted(df, payload)
    assert a["books_answered"] == ("draftkings", "fanduel", "betmgm", "underdog")
    assert "prizepicks" not in a["by_book"]
    assert (a["outcomes_seen"], a["kept_outcomes"]) == (44, 44)
    assert one(df, "draftkings", "Nikola Jokic", "Rebounds")["line"] == 12.5
    assert set(zip(df["event_home"], df["event_away"])) == {("SAS", "DEN")}
    assert set(df["start_time"]) == {"2026-03-13T01:00:00Z"}


def test_one_line_even_when_a_book_repeats_itself():
    payload = ev([bk("draftkings", mk("player_points", pair("Cade Cunningham", 25.5),
                                      side("Over", "Cade Cunningham", 25.5, -120),
                                      pair("  cade   CUNNINGHAM ", 25.5))),
                  bk("draftkings", mk("player_points",
                                      pair("Cade Cunningham", 25.5, -130, 105)))])
    df = parse(payload)
    assert len(df) == 1 and df.iloc[0]["name"] == "Cade Cunningham"
    assert (df.iloc[0]["price_over"], df.iloc[0]["price_under"]) == (-110.0, -110.0)  # first seen
    a = accounted(df, payload)
    assert a["duplicate_outcome"] == 5 and a["bookmaker_entries"] == 2


JUNK = {
    "none": None, "a list": [], "text": "<html>Bad gateway</html>", "a number": 7,
    "an empty object": {},
    "no bookmakers key": {k: v for k, v in ev([]).items() if k != "bookmakers"},
    "bookmakers not a list": ev({"draftkings": []}),
    "bookmakers of junk": ev([None, 1, "x", [], {"markets": []}, {"key": 5, "markets": []}]),
    "markets not a list": ev([{"key": "draftkings", "markets": {"key": "player_points"}}]),
    "markets of junk": ev([bk("draftkings", None, 1, "x", {"key": "player_points"},
                              {"key": "player_points", "outcomes": "x"})]),
    "outcomes of junk": ev([bk("draftkings", {"key": "player_points",
                                              "outcomes": [None, 1, "x", [], {}]})]),
    "outcome fields of junk": ev([bk("draftkings", mk("player_points",
        [{"name": 5, "description": "Cade Cunningham", "price": -110, "point": 25.5},
         {"name": "Push", "description": "Cade Cunningham", "price": -110, "point": 25.5},
         {"name": "Over", "description": None, "price": -110, "point": 25.5},
         {"name": "Over", "description": "", "price": -110, "point": 25.5},
         {"name": "Over", "description": "x" * 81, "price": -110, "point": 25.5},
         {"name": "Over", "description": ["Cade Cunningham"], "price": -110, "point": 25.5},
         {"name": "Over", "description": "Cade Cunningham", "price": "abc", "point": 25.5},
         {"name": "Over", "description": "Cade Cunningham", "price": None, "point": 25.5},
         {"name": "Over", "description": "Cade Cunningham", "price": True, "point": 25.5},
         {"name": "Over", "description": "Cade Cunningham", "price": float("inf"),
          "point": 25.5},
         {"name": "Over", "description": "Cade Cunningham", "price": "1e999", "point": 25.5}]))]),
    "market keys of junk": ev([bk("draftkings", {"key": None, "outcomes": pair("A B", 1.5)},
                                  {"key": ["player_points"], "outcomes": pair("A B", 1.5)},
                                  {"outcomes": pair("A B", 1.5)})]),
    "teams of junk": ev([bk("draftkings", mk("player_points", pair("Cade Cunningham", 25.5)))],
                        home={"name": "Detroit Pistons"}, away=None),
    "one team twice": ev([bk("draftkings", mk("player_points", pair("Cade Cunningham", 25.5)))],
                         away="Detroit Pistons"),
    "deep nesting": ev([[[[[[[]]]]]], {"key": "draftkings", "markets": [[[[]]]]}]),
}


@pytest.mark.parametrize("name", list(JUNK))
def test_junk_shapes_never_raise_and_every_outcome_is_accounted_for(name):
    payload = JUNK[name]
    df = parse(payload)
    assert df.empty and list(df.columns) == O.FEED_COLUMNS
    a = accounted(df, payload)
    assert a["kept_outcomes"] == 0
    if name in ("none", "a list", "text", "a number", "no bookmakers key",
                "bookmakers not a list"):
        assert a["bad_payload"] and a["bookmaker_entries"] == 0
    if name in ("an empty object", "teams of junk", "one team twice"):
        assert a["unknown_team"] and not a["bad_payload"]


def test_good_lines_survive_their_junk_neighbours():
    payload = ev([None, bk("draftkings", None, mk("player_points", [None, {}],
                                                  pair("Cade Cunningham", 25.5))),
                  bk("fanduel", {"key": "player_points", "outcomes": "x"})],
                 commence_time="tonight")
    del payload["bookmakers"][1]["markets"][1]["last_update"]
    df = parse(payload)
    assert list(zip(df["book"], df["name"], df["line"])) == [
        ("draftkings", "Cade Cunningham", 25.5)]
    assert df["start_time"].isna().all() and df["last_update"].isna().all()
    a = accounted(df, payload)
    assert (a["bad_bookmaker"], a["bad_market"], a["bad_outcome"]) == (1, 2, 2)
    assert a["books_answered"] == ("draftkings", "fanduel")    # fanduel present, unreadable
    assert a["by_book"]["fanduel"]["rows"] == 0


# -- Merging a refresh --------------------------------------------------------------------
def first_pull(payload=None, at=NINE, markets=FOUR) -> pd.DataFrame:
    out, report = O.merge_refresh(None, parse(phi_det() if payload is None else payload, at=at,
                                              markets=markets), FIXTURE_BOOKS)
    assert report["changed"] and not report["empty"]
    return out


def by_book(df: pd.DataFrame, book: str) -> pd.DataFrame:
    return df[df["book"] == book].reset_index(drop=True)


def test_the_first_pull_stores_every_book():
    out, report = O.merge_refresh(None, parse(phi_det()), FIXTURE_BOOKS)
    assert len(out) == 31 and list(out.columns) == O.FEED_COLUMNS and out.attrs == {}
    assert report["books"] == {b: "replaced" for b in FIXTURE_BOOKS}
    assert (report["kept_rows"], report["n_pulled"], report["changed"]) == (31, 0, True)
    sas, rep = O.merge_refresh(None, parse(den_sas()), FIXTURE_BOOKS)
    assert rep["books"]["prizepicks"] == "absent" and "prizepicks" not in set(sas["book"])


def test_a_missing_book_keeps_its_previous_lines_and_their_own_time():
    """Stale beats empty, per book (spec rule 9)."""
    prev = first_pull()
    payload = phi_det()
    payload["bookmakers"] = [b for b in payload["bookmakers"] if b["key"] != "fanduel"]
    out, report = O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)
    assert report["books"]["fanduel"] == "absent" and report["unreadable"] == {}
    pd.testing.assert_frame_equal(by_book(out, "fanduel"), by_book(prev, "fanduel"))
    assert set(by_book(out, "fanduel")["fetched_at"]) == {pd.Timestamp(NINE)}
    assert set(out.loc[out["book"] != "fanduel", "fetched_at"]) == {pd.Timestamp(ELEVEN)}
    assert len(out) == 31 and report["changed"]


def test_a_line_taken_down_is_dropped_and_named():
    """The vendor drops a suspended market about 15 minutes later (docs 1053)."""
    prev = first_pull()
    payload = phi_det()
    pts = market(payload, "draftkings", "player_points")
    pts["outcomes"] = [o for o in pts["outcomes"] if o["description"] != "Tyrese Maxey"]
    out, report = O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)
    assert "Tyrese Maxey" not in set(by_book(out, "draftkings")["name"])
    assert "Tyrese Maxey" in set(by_book(out, "fanduel")["name"])
    assert report["pulled"] == [("draftkings", "Tyrese Maxey", "Points")]
    assert report["n_pulled"] == 1 and len(out) == 30


def test_the_pulled_list_holds_eight_and_counts_them_all():
    many = [o for i in range(20) for o in pair(f"Player Number{i:02d}", 10.5)]
    prev, _ = O.merge_refresh(None, parse(ev([bk("draftkings", mk("player_points", many))])),
                              FIXTURE_BOOKS)
    fewer = ev([bk("draftkings", mk("player_points", many[:22]))])     # 11 of the 20 left
    out, report = O.merge_refresh(prev, parse(fewer, at=ELEVEN), FIXTURE_BOOKS)
    assert len(out) == 11 and report["books"]["draftkings"] == "replaced"
    assert report["n_pulled"] == 9 and len(report["pulled"]) == 8


def test_an_empty_answer_keeps_everything():
    """bookmakers: [] is the free answer (docs 1637-1638): not a pull, and it
    never wipes a stored line."""
    prev = first_pull()
    payload = phi_det()
    payload["bookmakers"] = []
    out, report = O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)
    pd.testing.assert_frame_equal(out, prev)
    assert report["empty"] and not report["bad_payload"]
    assert (report["kept_rows"], report["changed"]) == (0, False)
    assert report["books"] == {b: "absent" for b in FIXTURE_BOOKS}


@pytest.mark.parametrize("spoil, counter, n", [
    (lambda m: m.update(key="player_field_goals"), "unmapped_market", 11),
    (lambda m: m.update(key="player_points_alternate"), "alt_dropped", 11),
    (lambda m: [o.pop("description") for o in m["outcomes"]], "bad_outcome", 11),
    (lambda m: m.update(outcomes=[]), "outcomes_seen", 0)])
def test_a_present_book_with_nothing_readable_keeps_its_previous_lines(spoil, counter, n):
    """M3: presence is decided by KEPT ROWS, not by the bookmaker entry. A
    feed change that leaves FanDuel's entry with nothing readable must not
    erase FanDuel; its counters go to the report for the WARNING."""
    prev = first_pull()
    payload = phi_det()
    spoil(market(payload, "fanduel", "player_points"))
    out, report = O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)
    pd.testing.assert_frame_equal(by_book(out, "fanduel"), by_book(prev, "fanduel"))
    assert report["books"]["fanduel"] == "unreadable"
    assert report["unreadable"]["fanduel"][counter] == n
    assert all(isinstance(v, int) for v in report["unreadable"]["fanduel"].values())
    assert report["books"]["draftkings"] == "replaced" and report["kept_rows"] == 26


def test_under_half_keeps_the_previous_lines_and_reports_suspect():
    """M3: a book never loses half a game between two pulls; a feed change
    does. DraftKings had 10 lines; 4 is under half, 5 is not."""
    prev = first_pull()

    def keep(n):
        payload = phi_det()
        dk = entry(payload, "draftkings")
        dk["markets"] = [market(payload, "draftkings", "player_points")]
        dk["markets"][0]["outcomes"] = dk["markets"][0]["outcomes"][:2 * n]
        return O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)

    out, report = keep(4)
    pd.testing.assert_frame_equal(by_book(out, "draftkings"), by_book(prev, "draftkings"))
    assert report["books"]["draftkings"] == "suspect"
    assert report["suspect"] == {"draftkings": (4, 10)} and report["n_pulled"] == 0
    half, report = keep(5)
    assert len(by_book(half, "draftkings")) == 5 and report["books"]["draftkings"] == "replaced"
    assert report["suspect"] == {} and report["n_pulled"] == 5


def test_a_prizepicks_only_answer_touches_prizepicks_only():
    """M4: the top-up asks bookmakers=prizepicks alone. Whatever comes back,
    only PrizePicks' stored lines change; every other book keeps its lines,
    its stamps and its absence."""
    prev = first_pull()
    payload = phi_det()
    for o in market(payload, "prizepicks", "player_points")["outcomes"]:
        if o["description"] == "Cade Cunningham":
            o["point"] = 26.5
    for o in market(payload, "draftkings", "player_points")["outcomes"]:
        o["point"] += 1.0                              # sent anyway: must not land
    for parsed, asked in ((parse(payload, at=ELEVEN), ("prizepicks",)),
                          (parse(payload, books=("prizepicks",), at=ELEVEN), "prizepicks")):
        out, report = O.merge_refresh(prev, parsed, asked)
        assert list(report["books"]) == ["prizepicks"] and report["books"]["prizepicks"] == "replaced"
        assert report["kept_rows"] == 9
        others = out[out["book"] != "prizepicks"].reset_index(drop=True)
        pd.testing.assert_frame_equal(others, prev[prev["book"] != "prizepicks"]
                                      .reset_index(drop=True))
        pp = by_book(out, "prizepicks")
        assert one(pp, "prizepicks", "Cade Cunningham")["line"] == 26.5
        assert set(pp["fetched_at"]) == {pd.Timestamp(ELEVEN)}
    sas, _ = O.merge_refresh(None, parse(den_sas()), FIXTURE_BOOKS)
    nothing = den_sas()
    nothing["bookmakers"] = []
    still, report = O.merge_refresh(sas, parse(nothing, books=("prizepicks",), at=ELEVEN),
                                    ("prizepicks",))
    pd.testing.assert_frame_equal(still, sas)
    assert report["empty"] and report["books"] == {"prizepicks": "absent"}


# A refresh judged on the markets it asked for (review I1, F1). Design 3.3's
# core mode asks for the core markets only once the full list stops fitting,
# so a game pulled at 9 AM with every market can be refreshed with points
# alone. The other markets were not asked for: they must stay as they were,
# and never count against the book (the half rule) or as "taken down by
# their book" (report["pulled"], which feeds design 6's visitor notice).
THREE = ("player_points", "player_rebounds", "player_assists")
FIVE_PLAYERS = tuple(f"Player Number{i}" for i in range(5))


def five_players(markets=O.MAIN_MARKETS, moved=None, drop=(), last_update="2026-03-12T12:52:14Z"):
    """A hand-built DraftKings answer for PHI at DET: the five players, one
    24.5 pair in each of `markets`. `moved` maps a player to a new POINTS
    line; `drop` holds (player, market) pairs left out."""
    moved = moved or {}
    return ev([bk("draftkings", *[
        mk(m, *[pair(p, moved.get(p, 24.5) if m == "player_points" else 24.5)
                for p in FIVE_PLAYERS if (p, m) not in drop], last_update=last_update)
        for m in markets])])


def test_parse_records_the_markets_the_call_asked_for():
    assert parse(phi_det()).attrs["markets_requested"] == FOUR            # the caller's order
    assert parse(phi_det(), markets=O.CORE_MARKETS).attrs["markets_requested"] == ("player_points",)
    junk = parse(phi_det(), markets=["player_points", None, 5, "player_points", "player_assists"])
    assert junk.attrs["markets_requested"] == ("player_points", "player_assists")
    assert parse(phi_det(), markets=None).attrs["markets_requested"] == ()
    for payload in ("<html>Bad gateway</html>", ev([], home="Team World")):    # early returns too
        assert parse(payload, markets=THREE).attrs["markets_requested"] == THREE


def test_a_core_refresh_after_a_three_market_pull_keeps_the_markets_it_did_not_ask_for():
    """The CONSTRUCTED PHI at DET answer, pulled at 9 AM for points, rebounds
    and assists, refreshed at 11 AM for points only with DraftKings' Cade
    Cunningham line moved. Judged on every market, the 5 rebounds and
    assists lines would be dropped and named as taken down."""
    prev = first_pull(markets=THREE)
    assert len(prev) == 30
    payload = phi_det()
    for o in market(payload, "draftkings", "player_points")["outcomes"]:
        if o["description"] == "Cade Cunningham":
            o["point"] += 1.0
    out, report = O.merge_refresh(prev, parse(payload, markets=O.CORE_MARKETS, at=ELEVEN), FIXTURE_BOOKS)
    assert report["books"] == {b: "replaced" for b in FIXTURE_BOOKS}
    assert (report["pulled"], report["n_pulled"], report["suspect"]) == ([], 0, {})
    assert (report["kept_rows"], report["changed"], len(out)) == (25, True, 30)
    others = out[out["market"] != "player_points"].reset_index(drop=True)
    pd.testing.assert_frame_equal(others, prev[prev["market"] != "player_points"]
                                  .reset_index(drop=True))
    assert len(others) == 5 and set(others["fetched_at"]) == {pd.Timestamp(NINE)}
    points = out[out["market"] == "player_points"]
    assert len(points) == 25 and set(points["fetched_at"]) == {pd.Timestamp(ELEVEN)}
    assert one(out, "draftkings", "Cade Cunningham")["line"] == \
        one(prev, "draftkings", "Cade Cunningham")["line"] + 1.0


def test_a_core_refresh_after_a_twelve_market_pull_replaces_the_points_lines():
    """The $30 plan's 12 main markets (design 15, item 2): 60 DraftKings
    lines at 9 AM. The core refresh answers its 5 points lines, one moved
    from 24.5 to 26.5. Judged against all 60, 5 is "under half": the moved
    line would stay 24.5 with its 9 AM time and nothing would change."""
    prev, _ = O.merge_refresh(None, parse(five_players(), markets=O.MAIN_MARKETS), FIXTURE_BOOKS)
    assert len(prev) == 60 and set(prev["market"]) == set(O.MAIN_MARKETS)
    core = five_players(O.CORE_MARKETS, moved={"Player Number0": 26.5},
                        last_update="2026-03-12T14:52:14Z")
    out, report = O.merge_refresh(prev, parse(core, markets=O.CORE_MARKETS, at=ELEVEN), FIXTURE_BOOKS)
    assert report["books"]["draftkings"] == "replaced" and report["suspect"] == {}
    assert (report["kept_rows"], report["n_pulled"], report["changed"], len(out)) == (5, 0, True, 60)
    moved = one(out, "draftkings", "Player Number0")
    assert (moved["line"], moved["fetched_at"]) == (26.5, pd.Timestamp(ELEVEN))
    assert set(out.loc[out["market"] == "player_points", "fetched_at"]) == {pd.Timestamp(ELEVEN)}
    others = out[out["market"] != "player_points"].reset_index(drop=True)
    pd.testing.assert_frame_equal(others, prev[prev["market"] != "player_points"]
                                  .reset_index(drop=True))


def test_the_half_rule_and_the_pulled_list_judge_only_the_markets_asked_for():
    """Inside the markets a call asked for, the rules stand: a core refresh
    with 2 of the 5 points lines is "suspect" (2 x 2 < 5) and keeps all 60;
    one with 4 of 5 replaces them and names the fifth as taken down; a full
    refresh missing one assists line names that line."""
    prev, _ = O.merge_refresh(None, parse(five_players(), markets=O.MAIN_MARKETS), FIXTURE_BOOKS)
    gone = [(p, "player_points") for p in FIVE_PLAYERS[2:]]
    out, report = O.merge_refresh(prev, parse(five_players(O.CORE_MARKETS, drop=gone),
                                              markets=O.CORE_MARKETS, at=ELEVEN), FIXTURE_BOOKS)
    pd.testing.assert_frame_equal(out, prev)
    assert report["suspect"] == {"draftkings": (2, 5)} and not report["changed"]
    one_gone = [("Player Number4", "player_points")]
    out, report = O.merge_refresh(prev, parse(five_players(O.CORE_MARKETS, drop=one_gone),
                                              markets=O.CORE_MARKETS, at=ELEVEN), FIXTURE_BOOKS)
    assert report["books"]["draftkings"] == "replaced" and len(out) == 59
    assert report["pulled"] == [("draftkings", "Player Number4", "Points")] and report["n_pulled"] == 1
    assert (out["market"] != "player_points").sum() == 55
    full = five_players(drop=[("Player Number1", "player_assists")])
    out, report = O.merge_refresh(prev, parse(full, markets=O.MAIN_MARKETS, at=ELEVEN), FIXTURE_BOOKS)
    assert report["pulled"] == [("draftkings", "Player Number1", "Assists")] and len(out) == 59


def test_a_core_prizepicks_top_up_keeps_prizepicks_other_markets():
    """The free PrizePicks-only poll (design 3.4) asks for the plan's markets,
    the core ones in core mode: PrizePicks' rebounds, assists and fantasy
    lines from the 9 AM pull stay, and so does every other book."""
    prev = first_pull()
    out, report = O.merge_refresh(prev, parse(phi_det(), books=("prizepicks",),
                                              markets=O.CORE_MARKETS, at=ELEVEN), ("prizepicks",))
    assert report["books"] == {"prizepicks": "replaced"} and report["n_pulled"] == 0
    assert report["kept_rows"] == 6 and len(out) == len(prev) == 31
    kept = out[(out["book"] != "prizepicks") | (out["market"] != "player_points")]
    pd.testing.assert_frame_equal(kept.reset_index(drop=True),
                                  prev[(prev["book"] != "prizepicks") | (prev["market"] != "player_points")]
                                  .reset_index(drop=True))
    pp_points = out[(out["book"] == "prizepicks") & (out["market"] == "player_points")]
    assert len(pp_points) == 6 and set(pp_points["fetched_at"]) == {pd.Timestamp(ELEVEN)}


def test_the_caller_may_state_the_markets_and_without_them_every_market_is_judged():
    """markets_requested= wins over the parse's record. A frame that carries
    no record (its attrs lost) falls back to judging every market, the rule
    before markets were tracked: there, the core refresh is "suspect"."""
    prev, _ = O.merge_refresh(None, parse(five_players(), markets=O.MAIN_MARKETS), FIXTURE_BOOKS)
    core = parse(five_players(O.CORE_MARKETS, moved={"Player Number0": 26.5}),
                 markets=O.CORE_MARKETS, at=ELEVEN)
    bare = core.copy()
    bare.attrs = {k: v for k, v in core.attrs.items() if k != "markets_requested"}
    out, report = O.merge_refresh(prev, bare, FIXTURE_BOOKS)
    assert report["suspect"] == {"draftkings": (5, 60)}
    pd.testing.assert_frame_equal(out, prev)
    for asked in (("player_points",), "player_points", ["player_points", None]):
        out, report = O.merge_refresh(prev, bare, FIXTURE_BOOKS, markets_requested=asked)
        assert report["books"]["draftkings"] == "replaced" and len(out) == 60, asked
        assert one(out, "draftkings", "Player Number0")["line"] == 26.5
    out, report = O.merge_refresh(prev, core, FIXTURE_BOOKS, markets_requested=O.MAIN_MARKETS)
    assert report["suspect"] == {"draftkings": (5, 60)}            # the caller's word wins
    out, report = O.merge_refresh(prev, core, FIXTURE_BOOKS, markets_requested=5)   # junk: every market
    assert report["suspect"] == {"draftkings": (5, 60)}


def test_changed_only_when_the_stored_lines_change():
    prev = first_pull()
    same, report = O.merge_refresh(prev, parse(phi_det(), at=NINE), FIXTURE_BOOKS)
    assert not report["changed"] and len(same) == len(prev)
    later, report = O.merge_refresh(prev, parse(phi_det(), at=ELEVEN), FIXTURE_BOOKS)
    assert report["changed"]                                    # new stamps: a newer "as of"
    payload = phi_det()
    market(payload, "underdog", "player_points")["outcomes"][0]["point"] = 26.5
    market(payload, "underdog", "player_points")["outcomes"][1]["point"] = 26.5
    moved, report = O.merge_refresh(prev, parse(payload, at=NINE), FIXTURE_BOOKS)
    assert report["changed"]
    assert one(moved, "underdog", "Cade Cunningham")["line"] == 26.5


@pytest.mark.parametrize("new", [None, "x", pd.DataFrame(), pd.DataFrame({"name": ["A B"]}),
                                 "unknown teams", "bad payload"])
def test_junk_never_wipes_the_stored_lines(new):
    prev = first_pull()
    if isinstance(new, str) and new == "unknown teams":
        new = parse(ev([bk("draftkings", mk("player_points", pair("Cade Cunningham", 25.5)))],
                       home="Team World"), at=ELEVEN)
    elif isinstance(new, str) and new == "bad payload":
        new = parse("<html>Bad gateway</html>", at=ELEVEN)
    out, report = O.merge_refresh(prev, new, FIXTURE_BOOKS)
    pd.testing.assert_frame_equal(out, prev)
    assert report["bad_payload"] and not report["empty"]
    assert (report["kept_rows"], report["changed"]) == (0, False)


def test_a_junk_previous_frame_is_treated_as_none():
    for prev in ("x", pd.DataFrame({"name": ["A B"]}), 5):
        out, report = O.merge_refresh(prev, parse(phi_det()), FIXTURE_BOOKS)
        assert len(out) == 31 and report["bad_prev"]
    out, report = O.merge_refresh(None, parse(phi_det()), FIXTURE_BOOKS)
    assert not report["bad_prev"]
    out, report = O.merge_refresh(first_pull(), parse(phi_det()), ())
    assert report["books"] == {} and report["kept_rows"] == 0 and not report["changed"]


# -- Build 4's fix pass: the mutation campaign's survivor OA05 ------------------------------
def test_a_market_a_book_answered_without_being_asked_is_judged_too_oa05():
    """Survivor OA05 (merge_refresh's docstring: "a market a book answered
    anyway is judged too"). The call said it asked for points only, but the
    answer carries rebounds as well, with Cunningham's DraftKings rebounds
    line moved: the stored rebounds lines are judged against the new ones
    and replaced, so each player keeps ONE rebounds line per book (the new
    one), never the old line beside the new. Assists, which the answer
    left out and the call never asked for, are untouched."""
    prev = first_pull()
    (old,) = [m for b in phi_det()["bookmakers"] if b["key"] == "draftkings" for m in b["markets"]
              if m["key"] == "player_rebounds"]
    point = next(o["point"] for o in old["outcomes"] if o["description"] == "Cade Cunningham")
    payload = copy.deepcopy(phi_det())
    for b in payload["bookmakers"]:
        for m in b["markets"]:
            for o in m["outcomes"]:
                if b["key"] == "draftkings" and m["key"] == "player_rebounds" \
                        and o["description"] == "Cade Cunningham":
                    o["point"] = point + 1
    fresh = parse(payload, at=ELEVEN, markets=("player_points", "player_rebounds"))
    out, report = O.merge_refresh(prev, fresh, FIXTURE_BOOKS, markets_requested=("player_points",))
    reb = out[(out["market"] == "player_rebounds") & (out["book"] == "draftkings")]
    cade = reb[reb["name"] == "Cade Cunningham"]
    assert cade["line"].tolist() == [point + 1]
    assert reb.groupby("name").size().max() == 1
    assert len(out[out["market"] == "player_assists"]) == len(prev[prev["market"] == "player_assists"])
    assert set(out[out["market"] == "player_assists"]["fetched_at"]) == {pd.Timestamp(NINE).tz_convert("UTC")}


# -- Ten books (build amendment B35) ---------------------------------------------------------
PRICES = {"sportsbook": (-118, -102), "pickem": (-137, -137)}


def books_answer(books=None, player="Cade Cunningham") -> dict:
    """A CONSTRUCTED event-odds answer for PHI at DET (the vendor's documented
    shape, UT/docs_v4.txt:1504-1586, as every file in tests/fixtures/oddsapi:
    never a captured answer). Each of `books`, by default every catalog
    book with the default books (O.BOOKS) first, posts `player`'s points,
    each at its own line, an Over and Under pair: a sportsbook at -118/-102,
    a pick'em book (C5's, PICKEM: the vendor's us_dfs keys) at the
    indicative -137/-137."""
    keys = books if books is not None else tuple(dict.fromkeys([*O.BOOKS, *O.BOOK_ORDER]))
    return ev([bk(b, mk("player_points", pair(player, 20.5 + i, *PRICES[
        "pickem" if b in PICKEM else "sportsbook"]))) for i, b in enumerate(keys)])


def test_a_constructed_ten_book_answer_parses_every_book():
    """T5 and C5: every catalog book posts, the ten default books first.
    Named with the catalog, every line is kept under our own title; the
    pick'em books (pick6 and dabble_us_dfs among them) have their prices
    blanked to NaN, the sportsbooks (espnbet and hardrockbet among them)
    keep theirs; each row's side words are its book's (props.side_words:
    More and Less on a pick'em book, Over and Under on a sportsbook). Named
    with the default books only, every other catalog book's entry is an
    unknown book, counted and never kept. Every count is read from the
    constants (round BK's review, BKR-3), so a swap of the default list
    leaves this test as it is."""
    payload = books_answer()
    keys = [b["key"] for b in payload["bookmakers"]]
    assert sorted(keys) == sorted(O.BOOK_ORDER) and keys[:len(O.BOOKS)] == list(O.BOOKS)
    df = parse(payload, books=O.BOOK_ORDER, markets=("player_points",))
    a = accounted(df, payload)
    assert list(df["book"]) == keys and a["books_answered"] == O.BOOK_ORDER
    n = len(keys)
    assert (a["outcomes_seen"], a["kept_outcomes"], a["rows"], a["unknown_book"]) == (2 * n, 2 * n,
                                                                                      n, 0)
    for i, book in enumerate(keys):
        r = one(df, book, "Cade Cunningham")
        assert (r["book_title"], r["line"], r["direction"]) == (O.BOOK_TITLE[book], 20.5 + i, "both")
        if book in PICKEM:                       # C5's list, named: never read from the code
            assert math.isnan(r["price_over"]) and math.isnan(r["price_under"]), book
            assert props.side_words(book) == ("More", "Less"), book
        else:
            assert (r["price_over"], r["price_under"]) == PRICES["sportsbook"], book
            assert props.side_words(book) == ("Over", "Under"), book
    for book in ("pick6", "dabble_us_dfs"):
        assert one(df, book, "Cade Cunningham")[["price_over", "price_under"]].isna().all(), book
    for book in ("espnbet", "hardrockbet"):
        assert one(df, book, "Cade Cunningham")["price_over"] == -118.0, book
    ten = parse(payload, books=O.BOOKS, markets=("player_points",))
    t = accounted(ten, payload)
    assert list(ten["book"]) == list(O.BOOKS) and t["books_answered"] == O.BOOKS
    unasked = [b for b in keys if b not in O.BOOKS]
    assert unasked and t["unknown_book"] == 2 * len(unasked)
    assert t["unknown_books"] == unasked[:O._SAMPLE]


@pytest.mark.parametrize("key", NEVER_BOOKS)
def test_a_book_the_site_never_buys_is_never_a_line_when_it_answers_anyway(key):
    """Round BK's review, BKR-4: a book the site never buys (C1's
    exclusions, each named) that sits in an answer anyway, beside every
    catalog book, is never a line, whichever books the call named (the
    default, or the whole catalog): its outcomes are counted as an unknown
    book's, it has no tally of its own and it is never among the books
    answered. The page never sees it (props.feed_board also drops a book
    outside the catalog), and neither does the stored frame. It answers
    first, so the log's sample of unknown books always holds it."""
    payload = books_answer()
    payload["bookmakers"].insert(0, bk(key, mk("player_points", pair("Cade Cunningham", 30.5),
                                                   pair("Jalen Duren", 9.5))))
    for books in (O.BOOKS, O.BOOK_ORDER):
        df = parse(payload, books=books, markets=("player_points",))
        a = accounted(df, payload)
        assert set(df["book"]) == set(books) and key not in a["by_book"], books
        assert a["books_answered"] == tuple(books), books
        assert a["unknown_book"] == 2 * (len(O.BOOK_ORDER) - len(books)) + 4, books
        assert a["unknown_books"][0] == key, books
    assert a["unknown_books"] == [key]                  # the catalog named: it alone is unknown
    out, _report = O.merge_refresh(None, df, O.BOOK_ORDER, markets_requested=("player_points",))
    assert key not in set(out["book"]) and set(out["book"]) == set(O.BOOK_ORDER)


def test_a_pull_of_ten_books_after_a_pull_of_five_takes_nothing_down():
    """T7 and C4: the first pull named the five books the fixtures hold; the
    refresh names the ten default books. The five new books' lines are
    stored and never counted as taken down, and the five are judged exactly
    as a five-book refresh judges them: the same rows, the same verdicts, the
    same taken-down list (Tyrese Maxey's DraftKings points line). A new book
    that posts nothing is absent, never a taken-down line. The other way
    round, a pull naming the five after one of ten never touches the books
    it did not name (books not requested are never touched)."""
    prev = first_pull()
    payload = phi_det()
    pts = market(payload, "draftkings", "player_points")
    pts["outcomes"] = [o for o in pts["outcomes"] if o["description"] != "Tyrese Maxey"]
    new_books = tuple(b for b in O.BOOKS if b not in FIXTURE_BOOKS)
    assert len(new_books) == len(O.BOOKS) - len(FIXTURE_BOOKS) > 0
    five_out, five = O.merge_refresh(prev, parse(payload, at=ELEVEN), FIXTURE_BOOKS)
    wider = copy.deepcopy(payload)
    wider["bookmakers"] += books_answer(new_books)["bookmakers"]
    out, ten = O.merge_refresh(prev, parse(wider, books=O.BOOKS, at=ELEVEN), O.BOOKS)
    assert ten["pulled"] == five["pulled"] == [("draftkings", "Tyrese Maxey", "Points")]
    assert ten["n_pulled"] == five["n_pulled"] == 1 and ten["suspect"] == five["suspect"] == {}
    assert ten["books"] == {**five["books"], **{b: "replaced" for b in new_books}}
    old = out[out["book"].isin(FIXTURE_BOOKS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, five_out)
    assert sorted(out.loc[~out["book"].isin(FIXTURE_BOOKS), "book"]) == sorted(new_books)
    quiet, rep = O.merge_refresh(prev, parse(phi_det(), books=O.BOOKS, at=ELEVEN), O.BOOKS)
    assert {b: rep["books"][b] for b in new_books} == {b: "absent" for b in new_books}
    assert rep["pulled"] == [] and rep["n_pulled"] == 0 and len(quiet) == len(prev)
    back, rep = O.merge_refresh(out, parse(phi_det(), at=ELEVEN), FIXTURE_BOOKS)
    assert set(rep["books"]) == set(FIXTURE_BOOKS) and rep["n_pulled"] == 0
    for book in new_books:
        pd.testing.assert_frame_equal(by_book(back, book), by_book(out, book))
