"""PrizePicks ingestion, matching, routing, comparison, the board store, the
local record, and the UI's escaping. Offline: there is no fetcher to test, on
purpose, and a test below fails if one appears.

Fixtures in tests/fixtures/props are CONSTRUCTED (see PROVENANCE.txt): no real
NBA paste exists yet. The board text copies the structure of DiamondValue's
real MLB paste. The clock is pinned to 11pm ET on opening night, 2026-10-20.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache, props, props_ui
from bucketlib.pmf import Pmf

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "props"
NOW = pd.Timestamp("2026-10-21T03:00:00Z").to_pydatetime()   # 11pm ET Oct 20
OPENER, NEXT = "2026-10-20", "2026-10-21"


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def roster():
    return pd.read_csv(FIX / "roster_constructed.csv")


@pytest.fixture(scope="module")
def schedule():
    s = pd.read_csv(FIX / "schedule_constructed.csv", dtype={"game_id": str})
    s["tip_utc"] = pd.to_datetime(s["tip_utc"], utc=True)
    return s


@pytest.fixture(scope="module")
def projections():
    return pd.read_csv(FIX / "projections_constructed.csv", dtype={"game_id": str})


@pytest.fixture(scope="module")
def feed_text():
    return (FIX / "feed_constructed.json").read_text()


@pytest.fixture(scope="module")
def board_text():
    return (FIX / "board_constructed.txt").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _quiet_and_clean():
    logging.disable(logging.CRITICAL)
    with props._BOARDS_LOCK:
        props._BOARDS.clear()
    props._load_reliability.cache_clear()
    yield
    logging.disable(logging.NOTSET)
    with props._BOARDS_LOCK:
        props._BOARDS.clear()
    props._load_reliability.cache_clear()


@pytest.fixture()
def no_reliability(monkeypatch, tmp_path):
    monkeypatch.setattr(props, "RELIABILITY_PATH", tmp_path / "absent.csv")
    props._load_reliability.cache_clear()


class FixedProvider:
    """Hands back the same Pmf for everything (or per stat)."""

    def __init__(self, pmf, per_stat=None, void=None):
        self._pmf, self._per = pmf, per_stat or {}
        self._void = void

    def pmf(self, person_id, game_id, stat):
        return self._per.get(stat, self._pmf)

    def void_risk(self, person_id, game_id, stat):
        return self._void if self._void is not None else float("nan")


def _line(name="Nikola Jokić", stat="Points", line=27.5, team="DEN", **kw):
    rec = {"name": name, "team": team, "position": "C", "stat_type": stat,
           "line": line, "start_time": None, "direction": "both",
           "odds_type": "standard", "opp": None, "is_combo": False,
           "league": None, "source": "board"}
    rec.update(kw)
    return rec


def _prepared(lines, roster, schedule, now=NOW, dates=None):
    df = pd.DataFrame(lines, columns=props.PARSED_COLUMNS)
    text_free = df.copy()
    kinds = [props.classify_stat(r.stat_type, r.name, bool(r.is_combo), r.league)
             for r in text_free.itertuples()]
    text_free["stat"] = [k[0] for k in kinds]
    resolved, unresolved = props.resolve_names(text_free, roster)
    parts = [x.assign(match="matched") for x in [resolved] if not x.empty]
    if not unresolved.empty:
        parts.append(unresolved.rename(columns={"reason": "match"}))
    out = pd.concat(parts, ignore_index=True)
    return props.route_lines(out, schedule, now=now, dates=dates)


# ── Stat labels ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label,key", [
    ("Points", "pts"), ("Rebounds", "reb"), ("Assists", "ast"),
    ("3-PT Made", "fg3m"), ("3-Pointers Made", "fg3m"), ("3PT Made", "fg3m"),
    ("3 pt made", "fg3m"), ("Blocked Shots", "blk"), ("Blocks", "blk"),
    ("Steals", "stl"), ("Turnovers", "tov"), ("Pts+Rebs", "pr"),
    ("Pts + Rebs", "pr"), ("Pts+Asts", "pa"), ("Rebs+Asts", "ra"),
    ("Pts+Rebs+Asts", "pra"), ("Pts+Rebds+Asts", "pra"), ("PTS + REBS + ASTS", "pra"),
    ("Blks+Stls", "stocks"), ("Blocks+Steals", "stocks"), ("Fantasy Score", "fantasy"),
    ("Free Throws Made", "ftm"), ("FT Attempted", "fta"),
    ("Free Throws Attempted", "fta"), ("Two Pointers Made", "fg2m"),
    ("Two Pointers Attempted", "fg2a"), ("FG Made", "fgm"), ("Field Goals Made", "fgm"),
    ("FG Attempted", "fga"), ("Field Goals Attempted", "fga"), ("3-PT Attempted", "fg3a"),
    ("Offensive Rebounds", "oreb"), ("Defensive Rebounds", "dreb"),
    ("Personal Fouls", "pf"), ("  points  ", "pts"),
])
def test_every_prizepicks_nba_label_maps(label, key):
    assert props.classify_stat(label) == (key, "ok", None)


@pytest.mark.parametrize("label", [
    "1H Points", "2H Rebounds", "1Q Points", "2Q Assists", "3Q Points", "4Q Points",
    "1st Half Points", "Points - 1st Quarter", "First Half Pts+Rebs+Asts",
    "Points in First 5 Minutes", "Rebounds in First 3 Minutes",
    "First 3 Minutes Points", "Assists 1st 3 Minutes",
])
def test_period_props_are_refused_with_a_reason(label):
    stat, kind, why = props.classify_stat(label)
    assert (stat, kind) == (None, "refused") and why == props.REFUSE_PERIOD


@pytest.mark.parametrize("label,word", [
    ("Dunks", "box score"), ("Double Doubles", "joint"), ("Triple-Doubles", "joint"),
    ("First Basket", "ordering"),
])
def test_other_refusals_carry_their_own_reason(label, word):
    stat, kind, why = props.classify_stat(label)
    assert kind == "refused" and word in why


def test_combo_player_squares_and_other_boards_are_refused():
    assert props.classify_stat("Points", name="LeBron James + Luka Dončić")[2] == props.REFUSE_COMBO
    assert props.classify_stat("Points", is_combo=True)[1] == "refused"
    assert "NBA1Q" in props.classify_stat("Points", league="NBA1Q")[2]
    assert props.classify_stat("Points", league="nba")[1] == "ok"


def test_an_unknown_label_is_unmapped_not_dropped():
    assert props.classify_stat("Minutes Played")[1] == "unmapped"


# ── Names that are not people ────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["LAL", "GSW", "UTAH", "LAL vs GSW", "LAL/GSW",
                                  "NYK @ BOS", "", None, "Points", "Trending",
                                  "vs DAL Wed 8:30pm", "Cur. 27"])
def test_team_codes_and_junk_are_not_people(name):
    assert props._person_like(name) is False


@pytest.mark.parametrize("name", ["Nikola Jokić", "P.J. Washington", "OG Anunoby",
                                  "Shai Gilgeous-Alexander", "Jaren Jackson Jr.",
                                  "Yang Hansen", "De'Aaron Fox"])
def test_nba_names_are_people(name):
    assert props._person_like(name) is True


# ── Direction ────────────────────────────────────────────────────────────────
def test_direction_is_never_inferred_from_odds_type():
    for odds in ("demon", "goblin", "standard"):
        assert props.offered_sides(None, odds) == "both"
        assert props.offered_sides("", odds) == props.ODDS_TYPE_SIDES[odds]
    assert props.offered_sides("more", "standard") == "more"
    assert props.offered_sides("garbage", "some_new_type") == "both"


@pytest.mark.parametrize("value,expected", [
    (None, "both"), ("over", "more"), (["over"], "more"), ("under", "less"),
    (["over", "under"], "both"), ([], "both"), ("OVER", "more"), ("nonsense", "both"),
])
def test_allowed_wager_types(value, expected):
    assert props.sides_from_allowed_wager_types(value) == expected


@pytest.mark.parametrize("lean,sides,ok", [
    ("More", "both", True), ("Less", "both", True), ("More", "more", True),
    ("Less", "more", False), ("More", "less", False), ("Even", "both", False),
    (None, "both", False),
])
def test_is_pickable(lean, sides, ok):
    assert props.is_pickable(lean, sides) is ok


# ── Feed parsing ─────────────────────────────────────────────────────────────
def test_feed_fixture_parses_per_row(feed_text):
    df = props.parse_any(feed_text)
    assert len(df) == 15
    assert df.attrs["skipped_noname"] == 1      # description "NYK" is not a person
    assert df.attrs["skipped_badline"] == 1     # "N/A"
    by = df.set_index(["name", "line"])
    assert by.loc[("Joel Embiid", 10.5), "direction"] == "more"     # allowed_wager_types
    assert by.loc[("Tyrese Maxey", 3.5), "direction"] == "both"     # a demon, unrestricted
    assert by.loc[("Tyrese Maxey", 3.5), "odds_type"] == "demon"
    assert by.loc[("Jalen Brunson", 7.5), "league"] == "NBA1Q"       # projection's league wins
    assert bool(by.loc[("LeBron James + Luka Doncic", 52.5), "is_combo"])
    assert df.attrs["allowed_wager_types"]


def test_feed_without_included_players_counts_team_codes(feed_text):
    raw = json.loads(feed_text)
    raw["included"] = []
    df = props.parse_any(json.dumps(raw))
    # every description is a team code ("PHI"), never a person: all 17 counted
    assert df.empty and df.attrs["skipped_noname"] == 17


def test_one_bad_line_score_costs_one_row_not_the_feed():
    def feed(bad=None):
        data, inc = [], []
        for i in range(60):
            data.append({"type": "projection", "id": str(i), "attributes": {
                "stat_type": "Points", "line_score": "N/A" if i == bad else 10.5 + i},
                "relationships": {"new_player": {"data": {"id": f"p{i}"}}}})
            inc.append({"type": "new_player", "id": f"p{i}",
                        "attributes": {"name": f"Player Number{chr(65 + i % 26)} Smith"}})
        return json.dumps({"data": data, "included": inc})
    assert len(props.parse_any(feed())) == 60
    got = props.parse_any(feed(bad=7))
    assert len(got) == 59 and got.attrs["skipped_badline"] == 1


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "nan", "1e400",
                                 "-3.5", "1e9", True, None, [], {}])
def test_non_finite_and_absurd_lines_are_rejected(raw):
    payload = {"data": [{"type": "projection", "attributes": {
        "stat_type": "Points", "line_score": raw},
        "relationships": {"new_player": {"data": {"id": "p"}}}}],
        "included": [{"type": "new_player", "id": "p",
                      "attributes": {"name": "Nikola Jokic", "team": "DEN"}}]}
    got = props.parse_any(json.dumps(payload))
    assert got.empty and got.attrs["skipped_badline"] == 1


def test_bare_nan_and_infinity_json_literals_are_rejected():
    """Python's json module accepts these unquoted."""
    for lit in ("NaN", "Infinity", "-Infinity"):
        text = ('{"data":[{"type":"projection","attributes":{"stat_type":"Points",'
                f'"line_score":{lit}}},"relationships":{{"new_player":{{"data":{{"id":"p"}}}}}}}}],'
                '"included":[{"type":"new_player","id":"p","attributes":{"name":"Nikola Jokic"}}]}')
        got = props.parse_any(text)
        assert got.empty and got.attrs["skipped_badline"] == 1, lit


def test_hostile_json_never_raises():
    atoms = ["x", 1, 1.5, None, True, [], {}, [1], {"a": 1}, {"type": "new_player"},
             {"attributes": None}, {"attributes": "x"}, {"id": None},
             {"relationships": "x"}, {"relationships": {"new_player": "x"}},
             {"relationships": {"new_player": {"data": "x"}}},
             {"type": "projection", "attributes": {"line_score": {"a": 1}}},
             {"type": "projection", "attributes": [], "relationships": []}]
    cases = []
    for a in atoms:
        for key in ("data", "included"):
            cases += [json.dumps({key: a}), json.dumps({key: [a]}),
                      json.dumps({key: [a], "included": [a]})]
    cases += ["", " ", "{}", "[]", "null", "true", '"s"', '{"data":[[[[]]]]}',
              '[{"a":1}]', '{"data":{"a":[1]}}', "abcdefghijklmno", "x" * 5000,
              "[" * 200_000, "{" * 100_000, '{"data": [' + "{}," * 50_000 + "{}]}",
              "\x00\x01\x02 binary \xff junk", b"\xff\xfe raw bytes"]
    for c in cases:
        got = props.parse_any(c)
        assert isinstance(got, pd.DataFrame) and list(got.columns) == props.PARSED_COLUMNS


def test_valid_json_is_never_shredded_through_the_board_parser():
    """DiamondValue: a JSON paste falling through to the text parser became
    junk rows. A feed with a stray trailing line is still the feed."""
    pretty = json.dumps({"data": [], "included": [{"name": "Nikola Jokic"}]}, indent=2)
    got = props.parse_any(pretty + "\n27.5\nPoints\n")
    assert got.empty and got.attrs.get("json_shape")
    # and if it is not JSON at all, JSON-looking lines are still never names
    broken = props.parse_any('{\n"name": "Nikola Jokic",\n27.5\nPoints\n')
    assert broken.empty


def test_parse_any_contains_a_parser_that_raises(monkeypatch):
    """The broad excepts are the last line of defence on a public page: if a
    future edit makes either parser raise, the visitor gets an empty result,
    not a traceback with server paths."""
    def boom(*_a, **_k):
        raise RecursionError("boom")
    monkeypatch.setattr(props, "parse_prizepicks_board", boom)
    got = props.parse_any("Nikola Jokic\nDEN - C\n27.5\nPoints\n")
    assert got.empty and list(got.columns) == props.PARSED_COLUMNS
    monkeypatch.setattr(props, "parse_prizepicks_json", boom)
    got = props.parse_any('{"data": []}')
    assert got.empty and got.attrs["json_shape"] == "unreadable feed"


def test_a_paste_over_the_size_cap_is_refused_quickly(monkeypatch):
    monkeypatch.setattr(props, "MAX_PASTE_CHARS", 1000)
    start = time.time()
    got = props.parse_any("x" * 5000)
    assert got.empty and got.attrs["too_large"] and time.time() - start < 0.5


# ── Board-text parsing ───────────────────────────────────────────────────────
def test_board_fixture_reads_the_real_structure(board_text):
    df = props.parse_any(board_text)
    got = {(r.name, r.stat_type): r for r in df.itertuples()}
    assert len(df) == 13
    j = got[("Nikola Jokić", "Points")]
    assert (j.line, j.odds_type, j.direction, j.team, j.position, j.opp) == \
        (34.5, "demon", "more", "DEN", "C", "DAL")                    # tag on the NAME line
    c = got[("Stephen Curry", "3-PT Made")]
    assert (c.odds_type, c.direction) == ("goblin", "more")
    assert got[("Nikola Jokić", "Rebounds")].direction == "both"
    assert got[("Luka Dončić", "Pts+Rebs+Asts")].position == "F-G"     # hyphenated
    assert got[("Jaren Jackson Jr.", "Blocked Shots")].position == "F-C"
    assert got[("Yang Hansen", "Dunks")].position == "C-F"
    nt = got[("Stephen Curry", "Points")]
    assert nt.team is None and nt.line == 24.5                          # TEAM - POS optional
    assert got[("P.J. Washington", "Rebounds")].line == 7.0            # whole line
    assert bool(got[("LeBron James + Luka Dončić", "Points")].is_combo)
    assert "DEN" not in set(df["name"])                                 # team header row
    assert 945.0 not in set(df["line"])                                 # trending count
    assert df.attrs["skipped_live"] == 1                                # SGA live square
    assert "Shai Gilgeous-Alexander" not in set(df["name"])


def test_demon_glued_to_the_stat_label_is_stripped():
    df = props.parse_prizepicks_board("Nikola Jokic\nDEN - C\n34.5\nPointsDemon\nMore\n")
    assert (df.iloc[0]["stat_type"], df.iloc[0]["odds_type"]) == ("Points", "demon")


def test_buttons_do_not_swallow_the_next_player():
    text = ("Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n"
            "Jamal Murray\nDEN - G\n2.5\n3-PT Made\nMore\n"
            "Aaron Gordon\nDEN - F\n5.5\nRebounds\nLess\n")
    df = props.parse_prizepicks_board(text)
    assert list(df["direction"]) == ["both", "more", "less"]


def test_board_without_buttons_leaves_direction_unknown():
    df = props.parse_prizepicks_board("Nikola Jokic\nDEN - C\n27.5\nPoints\n")
    assert df.iloc[0]["direction"] == "" and props.offered_sides("", "standard") == "both"


def test_a_stray_number_is_never_pinned_on_the_previous_player():
    text = ("Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n"
            "118.5\nTeam Points\nPopular\n12\nSomething Else\n")
    df = props.parse_prizepicks_board(text)
    assert len(df) == 1


def test_a_trending_count_after_an_abandoned_square_is_not_a_line():
    """A square that never reached its number leaves a name behind. The next
    square's trending count ('945', no K) must not be read as that player's
    line with the next player's name as its stat."""
    text = ("Nikola Jokic\nDEN - C\nTrending\n945\nLuka Doncic\nLAL - F-G\n"
            "Luka Doncic\n@ GSW Wed 10:00pm\n45.5\nPts+Rebs+Asts\nLess\nMore\n")
    df = props.parse_prizepicks_board(text)
    assert list(zip(df["name"], df["line"])) == [("Luka Doncic", 45.5)]


def test_live_squares_are_skipped_and_counted():
    text = ("Jalen Brunson\nNYK - G\nJalen Brunson\nNYK 54vs PHI 50\nCur. 18\n"
            "26.5\nPoints\nLess\nMore\n"
            "Joel Embiid\nPHI - C\nJoel Embiid\nPHI 50 @ NYK 54\n10.5\nRebounds\nMore\n")
    df = props.parse_prizepicks_board(text)
    assert df.empty and df.attrs["skipped_live"] == 2


def test_suffix_strip_is_linear():
    """The paste box is unauthenticated and one process serves every visitor.
    `(Demon|Goblin)+$` and a slicing while-loop were BOTH quadratic on
    GridironValue. 4x the input must cost about 4x, not 16x."""
    assert props._strip_board_suffix("PointsDemon") == ("Points", "demon")
    assert props._strip_board_suffix("PointsGoblinDemon") == ("Points", "goblin")
    assert props._strip_board_suffix("Points") == ("Points", "standard")
    # 16x the input. Linear grows ~16x (exponent 1), the slicing loop and the
    # regex ~256x (exponent 2); the bound is exponent 1.5, far from both.
    # Sized so a quadratic implementation FAILS in seconds-to-a-minute rather
    # than hanging the suite (the regex is the slow one, ~20 s at 320 KB).
    timings = []
    for n in (4_000, 64_000):
        s = "Demon" * n + "!" + "Demon" * n
        timings.append(min(_time(lambda: props._strip_board_suffix(s)) for _ in range(3)))
    exponent = math.log(max(timings[1], 1e-9) / max(timings[0], 1e-9)) / math.log(16)
    assert timings[1] < 0.05 or exponent < 1.5, (timings, exponent)


def test_parse_any_is_linear_on_hostile_board_text():
    def text(n):
        square = "Nikola JokicDemon\nDEN - G-F\nvs DAL 47m\n27.5\nPointsGoblin\nLess\nMore\n"
        return square * n + ("Demon" * 50 + "\n") * n + ("1\n" * n)
    t1 = min(_time(lambda: props.parse_any(text(2_000))) for _ in range(2))
    t2 = min(_time(lambda: props.parse_any(text(8_000))) for _ in range(2))
    assert t2 < max(t1 * 8.0, 1.0), (t1, t2)


def test_a_huge_paste_completes(monkeypatch):
    junk = ("Some Words Here\n12.5\nPoints\nLess\nMore\n" * 120_000)[:5_000_000]
    start = time.time()
    got = props.parse_any(junk)
    assert time.time() - start < 20
    assert len(got) <= props.MAX_ROWS_PER_PASTE


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ── Name resolution ──────────────────────────────────────────────────────────
def test_names_resolve_with_their_how(roster):
    lines = pd.DataFrame([
        _line("Nikola Jokic", team="DEN"), _line("Shai Gilgeous Alexander", team="OKC"),
        _line("PJ Washington", team="DAL"), _line("Jaren Jackson", team="MEM"),
        _line("Moe Wagner", team="ORL"), _line("Victor Wembanyama", team="SA"),
        _line("Nobody At All", team="LAL"), _line("Chris Smith", team=None),
        _line("Chris Smith", team="TOR"), _line("Jayson Tatum", team="NYK"),
    ])
    res, unres = props.resolve_names(lines, roster)
    assert len(res) + len(unres) == len(lines), "a line was dropped"
    pid = dict(zip(res["name"] + "|" + res["team"].astype(str), res["person_id"]))
    how = dict(zip(res["name"] + "|" + res["team"].astype(str), res["how"]))
    assert pid["Nikola Jokic|DEN"] == 203999                       # diacritics
    assert pid["Shai Gilgeous Alexander|OKC"] == 1628983           # hyphen
    assert pid["PJ Washington|DAL"] == 1629023                     # initials
    assert pid["Jaren Jackson|MEM"] == 1628991                     # suffix
    assert (pid["Moe Wagner|ORL"], how["Moe Wagner|ORL"]) == (1629021, "alias")
    assert pid["Victor Wembanyama|SA"] == 1641705                  # PrizePicks 'SA'
    assert pid["Chris Smith|TOR"] == 1699002                       # team scoping
    assert how["Jayson Tatum|NYK"] == "exact+other_team"          # recorded, not hidden
    reasons = dict(zip(unres["name"], unres["reason"]))
    assert reasons == {"Nobody At All": "unmatched", "Chris Smith": "ambiguous"}


def test_team_scoping_picks_the_right_namesake(roster):
    res, _ = props.resolve_names(pd.DataFrame([_line("Chris Smith", team="BOS")]), roster)
    assert res.iloc[0]["person_id"] == 1699001


# ── Routing ──────────────────────────────────────────────────────────────────
def test_a_feed_line_routes_to_the_et_date_of_its_own_start_time(feed_text, roster, schedule):
    # 11am ET Oct 20, before any tip (at NOW, 11pm, the Oct 20 games have tipped
    # and their lines are refused: test_a_feed_line_for_a_game_that_has_tipped...)
    lines, report = props.prepare_paste(feed_text, roster, schedule.assign(status="scheduled"),
                                        now=MORNING)
    brunson = lines[(lines["name"] == "Jalen Brunson") & (lines["stat"] == "pts")].iloc[0]
    # 2026-10-21T00:00Z is 8pm ET on Oct 20, NOT Oct 21
    assert (brunson["game_date"], brunson["game_id"]) == (OPENER, "0022600002")
    jokic = lines[lines["name"] == "Nikola Jokic"].iloc[0]
    assert (jokic["game_date"], jokic["game_id"]) == (NEXT, "0022600005")
    assert report["by_date"] == {OPENER: 11, NEXT: 2}


def test_an_evening_board_is_tomorrows_games(board_text, roster, schedule):
    lines, report = props.prepare_paste(board_text, roster, schedule, now=NOW)
    placed = lines[lines["game_date"].notna()]
    assert set(placed["game_date"]) == {NEXT}
    curry = placed[(placed["name"] == "Stephen Curry") & (placed["stat"] == "pts")].iloc[0]
    assert curry["game_id"] == "0022600006"        # no TEAM - POS line: roster team routes it
    assert report["unrouted"] == 2 and "Chris Smith" in report["unrouted_names"]


def test_a_tipped_game_is_never_the_destination(roster, schedule):
    """Tatum's team played (and finished) tonight; the next game is Oct 22."""
    df = _prepared([_line("Jayson Tatum", team="BOS")], roster, schedule)
    assert df.iloc[0]["route_how"] == "unrouted" and df.iloc[0]["game_date"] is None
    wide = _prepared([_line("Jayson Tatum", team="BOS")], roster, schedule,
                     dates=["2026-10-20", "2026-10-21", "2026-10-22"])
    assert (wide.iloc[0]["game_date"], wide.iloc[0]["game_id"]) == ("2026-10-22", "0022600008")


def test_a_live_game_is_never_the_destination(roster, schedule):
    """OKC@SAS tipped at 01:30Z and is LIVE (not final) at 03:00Z. A board line
    for an OKC player is not for that game, and OKC has no game tomorrow."""
    df = _prepared([_line("Shai Gilgeous-Alexander", team="OKC")], roster, schedule)
    assert df.iloc[0]["route_how"] == "unrouted"
    stale = schedule.assign(status="scheduled")    # a schedule that lags the tip
    df2 = _prepared([_line("Shai Gilgeous-Alexander", team="OKC")], roster, stale)
    assert df2.iloc[0]["route_how"] == "unrouted"


def test_routing_before_tip_uses_tonights_game(roster, schedule):
    morning = pd.Timestamp("2026-10-20T15:00:00Z").to_pydatetime()
    pre = schedule.assign(status="scheduled")          # the morning's view of it
    df = _prepared([_line("Jayson Tatum", team="BOS")], roster, pre, now=morning)
    assert (df.iloc[0]["game_date"], df.iloc[0]["game_id"]) == (OPENER, "0022600001")


def test_opponent_on_the_matchup_line_disambiguates(roster, schedule):
    dates = ["2026-10-20", "2026-10-21", "2026-10-22"]
    morning = pd.Timestamp("2026-10-20T15:00:00Z").to_pydatetime()
    pre = schedule.assign(status="scheduled")
    plain = _prepared([_line("Jayson Tatum", team="BOS")], roster, pre, now=morning, dates=dates)
    assert plain.iloc[0]["game_id"] == "0022600001"
    df = _prepared([_line("Jayson Tatum", team="BOS", opp="TOR")], roster, pre,
                   now=morning, dates=dates)
    assert df.iloc[0]["game_id"] == "0022600008"


def test_a_naive_or_malformed_start_time_is_unrouted_not_guessed(roster, schedule):
    df = _prepared([_line("Nikola Jokic", team="DEN", start_time="2026-10-21 20:30"),
                    _line("Nikola Jokic", team="DEN", stat="Rebounds", start_time="soon")],
                   roster, schedule)
    # naive: falls back to the schedule walk, never to a guessed zone
    assert set(df["route_how"]) <= {"team_schedule", "unrouted"}


# ── The alt-line ladder ──────────────────────────────────────────────────────
def test_ladder_collapses_to_the_standard_line():
    df = pd.DataFrame([_line(line=27.5), _line(line=31.5, odds_type="demon"),
                       _line(line=22.5, odds_type="goblin")])
    out = props.collapse_alt_lines(df)
    assert len(out) == 1 and out.iloc[0]["line"] == 27.5


def test_a_moved_line_replaces_the_stale_one_newest_wins():
    out = props.collapse_alt_lines(pd.DataFrame([_line(line=24.5), _line(line=26.5)]))
    assert len(out) == 1 and out.iloc[0]["line"] == 26.5
    alt_later = props.collapse_alt_lines(pd.DataFrame([_line(line=24.5),
                                                       _line(line=30.5, odds_type="demon")]))
    assert alt_later.iloc[0]["line"] == 24.5, "recency overrode the standard preference"


def test_collapse_keeps_other_stats_and_other_dates():
    df = pd.DataFrame([_line(), _line(stat="Rebounds", line=12.5)])
    assert len(props.collapse_alt_lines(df)) == 2
    two_days = pd.DataFrame([_line(game_date=OPENER), _line(game_date=NEXT)])
    assert len(props.collapse_alt_lines(two_days)) == 2


def test_the_freeze_key_has_no_line_and_no_false_comment_says_otherwise():
    assert "line" not in props.FREEZE_KEYS
    assert props.FREEZE_KEYS == ("game_id", "person_id", "stat")
    src = (REPO / "bucketlib" / "props.py").read_text()
    assert "FREEZE_KEYS includes" not in src


# ── Compare: lean, gap, ranking, partition, ledger ───────────────────────────
def _proj_row(pid=203999, gid="0022600005", games=300, **means):
    return pd.DataFrame([{"person_id": pid, "game_id": gid, "game_date": NEXT,
                          "name": "Nikola Jokić", "team": "DEN", "opp": "DAL",
                          "games_prior": games, **means}])


def _matched(lines):
    df = pd.DataFrame(lines)
    df["match"] = "matched"
    if "person_id" not in df.columns:
        df["person_id"] = 203999
    df["person_id"] = df["person_id"].fillna(203999)
    if "game_id" not in df.columns:
        df["game_id"] = "0022600005"
    df["game_id"] = df["game_id"].fillna("0022600005")
    return df


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gap_sign_equals_the_lean_on_whole_and_half_lines(seed, no_reliability):
    """The row can never contradict itself: Model > Line exactly when the lean
    is More, on every whole and half line, through compare() itself."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        n = int(rng.integers(2, 30))
        p = rng.dirichlet(np.ones(n) * float(rng.choice([0.3, 1, 4])))
        pmf = Pmf(np.arange(n), p)
        lines = [_line(line=float(x), person_id=203999)
                 for x in np.arange(0, n + 1, 0.5)]
        table, meta = props.compare(_matched(lines), FixedProvider(pmf), _proj_row())
        assert meta["matched"] == len(lines)
        for r in table.itertuples():
            if r.lean == "More":
                assert r.gap > 0, (r.line, r.model)
            elif r.lean == "Less":
                assert r.gap < 0, (r.line, r.model)
            assert r.gap == pytest.approx(r.model - r.line)


def test_compare_recentres_on_the_shipped_mean(no_reliability):
    """Spec 3.5: the distribution is re-centred on the SHIPPED mean, and that
    can decide the lean: a pmf centred at 20 leans Less on 22.5, the same shape
    re-centred on a shipped 26 leans More."""
    pmf = Pmf.from_negbin(20.0, 2.2)
    raw, _ = props.compare(_matched([_line(line=22.5)]), FixedProvider(pmf), _proj_row())
    assert raw.iloc[0]["lean"] == "Less"
    table, _ = props.compare(_matched([_line(line=22.5)]), FixedProvider(pmf),
                             _proj_row(pts=26.0))
    assert table.iloc[0]["mean"] == pytest.approx(26.0, rel=1e-8)
    assert table.iloc[0]["lean"] == "More" and table.iloc[0]["gap"] > 0


def test_combo_and_fantasy_shipped_means_are_exact_sums(projections):
    row = projections[projections["person_id"] == 203507].iloc[0]
    assert props.shipped_mean(row, "pra") == pytest.approx(29.0 + 11.5 + 6.2)
    assert props.shipped_mean(row, "fantasy") == pytest.approx(
        29.0 + 1.2 * 11.5 + 1.5 * 6.2 + 3 * 1.0 + 3 * 1.1 - 3.2)
    assert props.shipped_mean(row, "fg2m") == pytest.approx(11.3 - 0.4)


def test_ranking_is_tier_then_confidence_never_gap():
    t = pd.DataFrame([
        {"tier": "negative", "confidence": 0.49, "gap": 40.0, "player": "a", "stat": "pts", "tag": "worst_tier_huge_gap"},
        {"tier": "strong", "confidence": 0.05, "gap": 9.0, "player": "b", "stat": "pts", "tag": "strong_small"},
        {"tier": "strong", "confidence": 0.30, "gap": 0.5, "player": "c", "stat": "reb", "tag": "strong_big"},
        {"tier": "unmeasured", "confidence": 0.40, "gap": 3.0, "player": "d", "stat": "ast", "tag": "unmeasured"},
    ])
    got = props.rank_rows(t)["tag"].tolist()
    assert got == ["strong_big", "strong_small", "unmeasured", "worst_tier_huge_gap"]


def test_combos_inherit_their_weakest_component(monkeypatch, tmp_path):
    csv = tmp_path / "rel.csv"
    csv.write_text("stat,tier,side_edge_pts,note\npts,strong,5.1,x\nreb,weak,1.0,y\n"
                   "ast,negative,-2.0,z\nstl,moderate,,hand assigned\n")
    monkeypatch.setattr(props, "RELIABILITY_PATH", csv)
    props._load_reliability.cache_clear()
    assert props.reliability_for("pts")["tier"] == "strong"
    assert props.reliability_for("pr")["tier"] == "weak"
    assert props.reliability_for("pra")["tier"] == "negative"
    assert props.reliability_for("stl")["tier"] == "unmeasured"      # no measurement
    assert props.reliability_for("stocks")["tier"] == "unmeasured"
    assert props.reliability_for("fg3m")["tier"] == "unmeasured"


def test_a_missing_reliability_file_means_unmeasured(no_reliability):
    assert props.reliability_for("pts") == {"tier": "unmeasured", "edge": None, "note": ""}
    html = props_ui._tier_badge("unmeasured", None)
    assert "t-unk" in html and "not measured" in html


def test_hidden_rows_are_the_unoffered_side_and_even(no_reliability):
    pmf = Pmf([0, 1, 2, 3, 4], [0.05, 0.1, 0.2, 0.3, 0.35])
    lines = [_line(line=1.5, direction="less", stat="Points"),          # More lean, Less only
             _line(line=1.5, stat="Rebounds"),                          # More lean, both
             _line(line=1.0, stat="Assists")]                           # P(>)=P(<): Even
    flat = Pmf([0, 1, 2], [0.25, 0.5, 0.25])
    table, _ = props.compare(_matched(lines), FixedProvider(pmf, {"ast": flat}), _proj_row())
    shown, hidden, thin, void = props.board_rows(table)
    assert set(shown["stat"]) == {"reb"}
    assert set(hidden["stat"]) == {"pts", "ast"}
    assert set(hidden[hidden["stat"] == "ast"]["lean"]) == {"Even"}


def test_thin_and_void(no_reliability):
    pmf = Pmf.from_poisson(5.0)
    lines = [_line(line=3.5, person_id=1642905, game_id="0022600007"),
             _line(line=3.5, person_id=203999)]
    proj = pd.concat([_proj_row(), _proj_row(pid=1642905, gid="0022600007", games=0)])
    table, meta = props.compare(_matched(lines).assign(
        game_id=["0022600007", "0022600005"]), FixedProvider(pmf), proj)
    shown, hidden, thin, void = props.board_rows(table)
    assert set(thin["person_id"]) == {1642905}
    shown2, hidden2, thin2, void2 = props.board_rows(table, void_ids={203999})
    assert set(void2["person_id"]) == {203999} and shown2.empty


def test_board_rows_is_an_exact_partition():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(0, 40))
        t = pd.DataFrame({
            "person_id": rng.integers(1, 8, n), "lean": rng.choice(["More", "Less", "Even"], n),
            "sides": rng.choice(["both", "more", "less"], n), "thin": rng.random(n) < 0.3,
            "_rid": np.arange(n)})
        void_ids = set(rng.integers(1, 8, 2).tolist())
        parts = props.board_rows(t, void_ids)
        ids = [set(p["_rid"]) if "_rid" in p else set() for p in parts]
        assert sum(len(p) for p in parts) == n
        assert set().union(*ids) == set(range(n))
        for i in range(4):
            for j in range(i + 1, 4):
                assert not ids[i] & ids[j]
        assert all(pid not in void_ids for pid in parts[0]["person_id"]) if n else True
        assert all(props.is_pickable(l, s) for l, s in zip(parts[0]["lean"], parts[0]["sides"])) if n else True


def test_ledger_balances_on_both_fixtures(feed_text, board_text, roster, schedule,
                                          projections, no_reliability):
    provider = props.MeanDispersionProvider(projections)
    pre = schedule.assign(status="scheduled")
    for text, sched, now in ((feed_text, pre, MORNING), (board_text, schedule, NOW)):
        lines, report = props.prepare_paste(text, roster, sched, now=now)
        assert report["kept"] == len(lines)
        for d in report["by_date"]:
            table, meta = props.compare(props.lines_for_date(lines, d), provider, projections)
            parts = props.board_rows(table, void_ids={1630178})
            led = props.ledger(meta, *parts)
            assert led["balanced"], led
            assert led["posted"] == report["by_date"][d]


def test_ledger_balances_under_hostile_rows(roster, projections, no_reliability):
    class Exploding:
        def pmf(self, pid, gid, stat):
            if stat == "reb":
                raise RuntimeError("boom")
            return None if stat == "ast" else Pmf.from_poisson(20.0)
    lines = pd.DataFrame([
        _line(), _line(stat="Rebounds"), _line(stat="Assists"), _line(stat="Dunks"),
        _line(stat="Minutes Played"), _line(name="Nobody Here"), _line(name="Chris Smith", team=None),
        _line(name="LeBron James + Luka Dončić", is_combo=True), _line(name="Jayson Tatum", team="BOS"),
        _line(line=float("nan")),
    ])
    table, meta = props.compare(lines, Exploding(), projections, roster=roster)
    assert props.ledger(meta, *props.board_rows(table))["balanced"]
    assert meta["errors"] >= 1 and meta["ambiguous"] == 1 and meta["unmatched"] == 1
    assert meta["unprojected_why"]
    assert meta["refused_why"]["Points (two-player square)"] == props.REFUSE_COMBO


def test_compare_explains_an_empty_result(projections):
    _t, meta = props.compare(pd.DataFrame(), None, projections)
    assert meta["reason"] == "no lines pasted"
    _t, meta = props.compare(_matched([_line()]), None, pd.DataFrame())
    assert meta["reason"] == "no projections for this date" and meta["unprojected"] == 1


def test_a_period_board_pasted_as_full_game_is_flagged(no_reliability):
    rows, projs = [], []
    for i in range(12):
        rows.append(_line(line=6.5, person_id=1000 + i, game_id="g"))
        projs.append({"person_id": 1000 + i, "game_id": "g", "pts": 24.0, "games_prior": 50})
    table, meta = props.compare(_matched(rows).assign(person_id=[1000 + i for i in range(12)],
                                                      game_id="g"),
                                props.MeanDispersionProvider(pd.DataFrame(projs)),
                                pd.DataFrame(projs))
    assert meta["period_board_suspect"] is True


def test_void_risk_hook_is_carried_and_not_priced(no_reliability):
    assert props.REBOOT_VOIDS_MODELLED is False
    pmf = Pmf.from_poisson(30.0)
    table, _ = props.compare(_matched([_line(line=24.5)]), FixedProvider(pmf, void=0.07),
                             _proj_row())
    r = table.iloc[0]
    assert r["void_risk"] == pytest.approx(0.07)
    assert r["q_more"] == pytest.approx(r["p_over"] / (r["p_over"] + r["p_under"]))
    assert "void risk" in props_ui.ledger_html(table)
    no_hook, _ = props.compare(_matched([_line(line=24.5)]), FixedProvider(pmf), _proj_row())
    assert math.isnan(no_hook.iloc[0]["void_risk"])


# ── The board store ──────────────────────────────────────────────────────────
def test_tokens_are_unguessable_and_validated():
    t = props.new_board(pd.DataFrame([_line()]))
    assert props.valid_token(t) and len(t) >= 30
    assert len({props.new_board(pd.DataFrame()) for _ in range(50)}) == 50
    for bad in (None, "", "short", "x" * 200, "<script>alert(1)</script>xxxxxxxxxxxx",
                "../../etc/passwd/aaaaaaaaaaaaaaa", 12345):
        assert props.get_board(bad) is None


def test_two_visitors_boards_are_isolated():
    a = props.new_board(pd.DataFrame([_line(line=27.5)]))
    b = props.new_board(pd.DataFrame([_line(name="Stephen Curry", team="GSW", line=24.5)]))
    a2 = props.add_to_board(a, pd.DataFrame([_line(stat="Rebounds", line=12.5)]))
    assert a2 == a
    assert len(props.get_board(a)) == 2 and len(props.get_board(b)) == 1
    assert set(props.get_board(b)["name"]) == {"Stephen Curry"}
    props.clear_board(a)
    assert props.get_board(a) is None and len(props.get_board(b)) == 1
    got = props.get_board(b)
    got.loc[0, "name"] = "mutated"
    assert props.get_board(b).iloc[0]["name"] == "Stephen Curry", "get_board leaked a reference"


def test_an_unknown_or_expired_token_gets_a_new_board(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(props, "_clock", lambda: clock[0])
    t = props.new_board(pd.DataFrame([_line()]))
    clock[0] += props.BOARD_TTL_S - 1
    assert props.get_board(t) is not None                  # sliding: touched now
    clock[0] += props.BOARD_TTL_S + 1
    assert props.get_board(t) is None
    t2 = props.add_to_board(t, pd.DataFrame([_line()]))
    assert t2 != t and len(props.get_board(t2)) == 1


def test_the_store_is_bounded(monkeypatch):
    monkeypatch.setattr(props, "BOARD_MAX", 5)
    toks = [props.new_board(pd.DataFrame([_line()])) for _ in range(8)]
    assert props.board_count() == 5
    assert props.get_board(toks[0]) is None and props.get_board(toks[-1]) is not None
    monkeypatch.setattr(props, "BOARD_MAX_ROWS", 10)
    big = pd.DataFrame([_line(stat=f"Stat {i}") for i in range(30)])
    assert len(props.get_board(props.new_board(big))) == 10
    monkeypatch.setattr(props, "BOARD_TOTAL_ROWS", 25)
    for _ in range(5):
        props.new_board(big)
    with props._BOARDS_LOCK:
        assert sum(len(b.lines) for b in props._BOARDS.values()) <= 25


def test_the_store_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    t = props.new_board(pd.DataFrame([_line()]))
    props.add_to_board(t, pd.DataFrame([_line(stat="Rebounds")]))
    props.get_board(t)
    props.clear_board(t)
    assert list(tmp_path.iterdir()) == []
    tree = ast.parse((REPO / "bucketlib" / "props.py").read_text())
    store_fns = {"new_board", "get_board", "add_to_board", "clear_board", "_evict_locked", "_cap"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in store_fns:
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert not names & {"open", "write_json", "dc_path", "write_parquet",
                                "freeze_projections", "to_json", "to_parquet"}, node.name


def test_a_paste_survives_navigation_and_pill_equals_game_rows(
        board_text, roster, schedule, projections, no_reliability):
    """Paste on '/', follow a card: a new session reads the same token, and the
    card pill for a game equals the rows its game page shows."""
    token, report = props_ui.ingest(board_text, None, roster, schedule, now=NOW)
    assert token and report["on_board"] == report["kept"]
    provider = props.MeanDispersionProvider(projections)
    slate = props_ui.board_view(token, NEXT, provider, projections)       # session 1
    pills = props_ui.line_counts_by_game(slate.shown)
    assert pills
    for gid, n in pills.items():
        fresh = props_ui.board_view(token, NEXT, provider, projections)   # session 2
        game_rows = fresh.shown[fresh.shown["game_id"] == gid]
        per_player = props_ui.rows_by_person(game_rows)
        assert n == len(game_rows) == sum(len(v) for v in per_player.values())


def test_ingest_never_raises_and_reports_why(roster, schedule, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    token, report = props_ui.ingest("anything", None, roster, schedule, now=NOW)
    assert token is None and report.get("kept", 0) == 0
    msgs = props_ui.report_messages({**report, "view_date": NEXT})
    assert msgs and msgs[0][0] == "warning"
    monkeypatch.setattr(props, "prepare_paste", boom)
    token, report = props_ui.ingest("Nikola Jokic\n27.5\nPoints", None, roster, schedule)
    assert report["error"] is True
    assert props_ui.report_messages(report)[0][0] == "warning"


def test_report_points_at_other_dates_and_unrouted(feed_text, board_text, roster, schedule):
    _t, rep = props_ui.ingest(feed_text, None, roster, schedule.assign(status="scheduled"),
                              now=MORNING)
    msgs = props_ui.report_messages({**rep, "view_date": OPENER})
    # 13 placed; the combo square, Tatum's 1H line and Brunson's line from the
    # NBA1Q board are counted, never "loaded"
    assert rep["kept"] == 13 and rep["comparable"] == 10
    assert msgs[0][0] == "success" and msgs[0][1].startswith("10 lines loaded")
    assert any(k == "info" and "3 not priced" in t for k, t in msgs)
    assert any(k == "info" and "saved for Wed Oct 21, step the date" in t for k, t in msgs)
    _t, rep = props_ui.ingest(board_text, None, roster, schedule, now=NOW)
    msgs = props_ui.report_messages({**rep, "view_date": OPENER})
    assert any("could not be placed" in t for _k, t in msgs)
    assert any("live in-game square" in t for _k, t in msgs)
    empty = props_ui.report_messages({"nonempty": True, "kept": 0, "unrouted": 0,
                                      "parse": {"skipped_noname": 40}})
    assert empty[0][0] == "warning" and "included player list" in empty[0][1]


# ── The record (LOCAL ONLY) ──────────────────────────────────────────────────
@pytest.fixture()
def record_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "READ_ONLY", False)
    return tmp_path


def _frozen_table(line=24.5, model=26.1, lean="More", stat="pts", **kw):
    row = {"person_id": 203999, "game_id": "0022600005", "game_date": NEXT,
           "player": "Nikola Jokić", "stat": stat, "stat_label": "Points", "line": line,
           "model": model, "mean": model + 0.4, "gap": model - line, "lean": lean,
           "sides": "both", "odds_type": "standard", "tier": "unmeasured",
           "thin": False, "games_prior": 300,
           "as_of": "2026-10-21T15:00:00Z", "provisional": False,
           "tip_utc": pd.Timestamp("2026-10-22T00:30:00Z")}
    row.update(kw)
    return pd.DataFrame([row])


def test_freeze_first_seen_wins_and_stamps(record_dir):
    assert props.freeze_projections(_frozen_table(model=26.1), "m1@a", now=NOW) == 1
    assert props.freeze_projections(_frozen_table(model=99.0), "m1@b", now=NOW) == 0
    back = props.load_frozen(NEXT)
    assert len(back) == 1 and back.iloc[0]["model"] == 26.1
    assert back.iloc[0]["model_version"] == "m1@a"
    assert back.iloc[0]["as_of"] == "2026-10-21T15:00:00Z"
    assert back.iloc[0]["frozen_at"].endswith("+00:00")


def test_a_moved_line_is_not_a_second_pick(record_dir):
    props.freeze_projections(_frozen_table(line=24.5), "v", now=NOW)
    assert props.freeze_projections(_frozen_table(line=26.5), "v", now=NOW) == 0
    assert props.load_frozen(NEXT).iloc[0]["line"] == 24.5


def test_a_new_prop_still_freezes(record_dir):
    props.freeze_projections(_frozen_table(), "v", now=NOW)
    assert props.freeze_projections(_frozen_table(stat="reb"), "v", now=NOW) == 1
    assert len(props.load_frozen(NEXT)) == 2
    assert props.frozen_dates() == [NEXT]


def test_provisional_and_post_tip_lines_never_freeze(record_dir):
    assert props.freeze_projections(_frozen_table(provisional=True), "v", now=NOW) == 0
    after_tip = pd.Timestamp("2026-10-22T01:00:00Z").to_pydatetime()
    assert props.freeze_projections(_frozen_table(), "v", now=after_tip) == 0
    assert props.load_frozen(NEXT) is None


def test_read_only_refuses_to_freeze(record_dir, monkeypatch):
    monkeypatch.setattr(cache, "READ_ONLY", True)
    assert props.freeze_projections(_frozen_table(), "v", now=NOW) == 0
    assert props.load_frozen(NEXT) is None and list(record_dir.iterdir()) == []


def test_the_record_file_is_fenced_out_of_git():
    """LOCAL ONLY (Barrett): the record path must match a .gitignore rule."""
    probe = REPO / "cache" / f"props_record_2099-01-01_{props.VERSION}.json"
    assert props.record_path("2099-01-01").name == probe.name
    r = subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=REPO,
                       capture_output=True)
    if r.returncode == 128:
        pytest.skip("not a git checkout")
    assert r.returncode == 0, "cache/props_record_*.json is no longer git-ignored"


def _actuals(**stats):
    base = {"game_id": "0022600005", "person_id": 203999, "played": True}
    base.update(stats)
    return pd.DataFrame([base])


def test_grade_buckets(record_dir):
    t = _frozen_table(line=24.5, lean="More")
    assert props.grade(t, _actuals(pts=30.0)).iloc[0]["outcome"] == "correct"
    assert props.grade(t, _actuals(pts=20.0)).iloc[0]["outcome"] == "incorrect"
    assert props.grade(t, _actuals(played=False, pts=np.nan)).iloc[0]["outcome"] == "void"
    assert props.grade(t, pd.DataFrame()).iloc[0]["outcome"] == "pending"
    assert props.grade(t, _actuals(pts=np.nan)).iloc[0]["outcome"] == "pending"
    whole = _frozen_table(line=25.0, lean="Less")
    g = props.grade(whole, _actuals(pts=25.0)).iloc[0]
    assert g["outcome"] == "exact" and g["model_correct"] is None          # never a win
    assert props.grade(_frozen_table(lean="Even"), _actuals(pts=30.0)).iloc[0]["outcome"] == "no_lean"
    secs = _actuals(pts=np.nan).drop(columns="played").assign(seconds=0)
    assert props.grade(t, secs).iloc[0]["outcome"] == "void"


def test_grade_composes_fantasy_and_two_pointers():
    row = {"pts": 30, "reb": 10, "ast": 8, "stl": 2, "blk": 1, "tov": 4, "fgm": 11,
           "fg3m": 3, "fga": 20, "fg3a": 8}
    assert props.actual_value("fantasy", row) == pytest.approx(30 + 12 + 12 + 6 + 3 - 4)
    assert props.actual_value("fg2m", row) == 8 and props.actual_value("fg2a", row) == 12
    assert props.actual_value("pra", {"pts": 1, "reb": None, "ast": 1}) is None


def test_season_record_uses_the_common_side_and_breakeven():
    g = pd.DataFrame([{"outcome": "correct", "result": "Less"}] * 55
                     + [{"outcome": "incorrect", "result": "Less"}] * 5
                     + [{"outcome": "incorrect", "result": "More"}] * 40
                     + [{"outcome": "exact", "result": "Exact"}] * 3
                     + [{"outcome": "void", "result": None}] * 2)
    rec = props.season_record([g])
    assert rec["decided"] == 100 and rec["hit_rate"] == 55.0
    assert rec["baseline"] == 60.0 and rec["beats_coin"] is False
    assert rec["exact"] == 3 and rec["voids"] == 2 and rec["clears_breakeven"] is False
    lo, hi = props.wilson_interval(160, 300)
    assert lo < 0.5 < hi
    assert math.isnan(props.wilson_interval(0, 0)[0])
    assert props.season_record([])["decided"] == 0


def test_training_window():
    reg = {"trained_through": "2025-26"}
    assert props.is_in_training_window("2025-26", reg) is True
    assert props.is_in_training_window(2026, reg) is True
    assert props.is_in_training_window("2026-27", reg) is False
    assert props.is_in_training_window("2026-27", None) is False


def test_the_record_is_not_wired_into_any_public_page():
    """props_ui must not import or call the record. Pages are other people's
    files; the guard here is that the shared UI module cannot reach it."""
    tree = ast.parse((REPO / "bucketlib" / "props_ui.py").read_text())
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
           {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    record = {"freeze_projections", "load_frozen", "grade", "season_record",
              "frozen_dates", "record_path", "actual_value"}
    assert not used & record, used & record


# ── Paste only: no network, ever ─────────────────────────────────────────────
NETWORK = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "socket", "ftplib",
           "websocket", "websockets", "curl_cffi", "pycurl", "http", "nba_api",
           # Ways out that name no network library (audit 2026-10-09: a
           # subprocess curl and an importlib launder both passed the old guard).
           "subprocess", "importlib", "asyncio", "ssl", "smtplib", "poplib", "imaplib",
           "nntplib", "telnetlib", "xmlrpc", "socketserver", "webbrowser", "ctypes",
           "multiprocessing", "builtins", "pty", "selenium", "playwright", "mechanize"}
# Fetchers, process spawners and import-by-string, as a name, an attribute or
# an imported symbol, anywhere in the import closure: `pd.read_json(url)` is
# caught by the FEED_URL rule below, `from pandas.io.common import urlopen`,
# `np.DataSource(None).open(url)`, `os.system('curl ...')` and
# `sys.modules['urllib.request']` by this one.
BANNED_REFS = {"urlopen", "urlretrieve", "DataSource", "import_module", "__import__",
               "__builtins__", "system", "popen", "posix_spawn", "posix_spawnp", "spawnl",
               "spawnle", "spawnlp", "spawnv", "spawnve", "spawnvp", "execl", "execle",
               "execlp", "execv", "execve", "execvp", "fork", "forkpty", "modules",
               "read_html", "read_xml"}
BANNED_NAMES = {"exec", "eval", "compile", "globals", "breakpoint"}   # bare builtins only
PKG = REPO / "bucketlib"


def _module_path(dotted: str) -> Path | None:
    parts = dotted.split(".")
    for cand in (PKG.joinpath(*parts).with_suffix(".py"), PKG.joinpath(*parts, "__init__.py")):
        if cand.exists():
            return cand
    return None


def _scan(tree) -> tuple[set, set, list]:
    """(external top-level imports, internal bucketlib modules, banned references)."""
    ext, internal, dynamic = set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "bucketlib":
                    internal.add(".".join(a.name.split(".")[1:]) or "__init__")
                else:
                    ext.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            dynamic += [a.name for a in node.names if a.name in BANNED_REFS]
            if node.level:
                if node.module:
                    internal.add(node.module)
                    for a in node.names:
                        internal.add(f"{node.module}.{a.name}")
                else:
                    internal.update(a.name for a in node.names)
            elif node.module and node.module.split(".")[0] == "bucketlib":
                rest = node.module.split(".")[1:]
                if rest:
                    internal.add(".".join(rest))
                for a in node.names:
                    internal.add(".".join(rest + [a.name]))
            elif node.module:
                ext.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_REFS:
            dynamic.append(node.attr)
        elif isinstance(node, ast.Name) and (node.id in BANNED_REFS or node.id in BANNED_NAMES):
            dynamic.append(node.id)
    return ext, internal, dynamic


def _docstring_nodes(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _root_rules(tree) -> list[str]:
    """Rules for the props modules THEMSELVES (not the foundation they import):
    FEED_URL is referenced only as the argument of esc() (it is a link the
    visitor opens, never an argument to anything that could fetch it); no URL
    literal outside a docstring and FEED_URL's own definition; no getattr with
    a computed attribute name (the audit's importlib launder)."""
    bad = []
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    feed_def = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FEED_URL" for t in node.targets):
            feed_def |= {id(n) for n in ast.walk(node.value)}
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        is_ref = ((isinstance(node, ast.Name) and node.id == "FEED_URL"
                   and isinstance(node.ctx, ast.Load))
                  or (isinstance(node, ast.Attribute) and node.attr == "FEED_URL"))
        if is_ref:
            parent = parents.get(id(node))
            f = parent.func if isinstance(parent, ast.Call) else None
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if not (fname == "esc" and node in parent.args):
                bad.append(f"FEED_URL used outside esc() at line {node.lineno}")
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "://" in node.value and id(node) not in docs and id(node) not in feed_def):
            bad.append(f"URL literal at line {node.lineno}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "setattr", "delattr", "hasattr")
                and (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant))):
            bad.append(f"{node.func.id} with a computed name at line {node.lineno}")
    return bad


def _network_offenders(root_source: str, root_name: str = "props") -> list[str]:
    """Walk the bucketlib import closure of `root_source`; any module in it that
    imports a network library or references a fetcher, spawner or
    import-by-string is an offender, and the root must pass _root_rules."""
    offenders, seen = [], {root_name}
    root = ast.parse(root_source)
    offenders += [f"{root_name}: {b}" for b in _root_rules(root)]
    queue = [(root_name, root)]
    while queue:
        name, tree = queue.pop()
        ext, internal, dynamic = _scan(tree)
        bad = ext & NETWORK
        if bad:
            offenders.append(f"{name} imports {sorted(bad)}")
        if dynamic:
            offenders.append(f"{name} uses {sorted(set(dynamic))}")
        for mod in internal:
            if mod in seen:
                continue
            path = _module_path(mod)
            if path is None:
                continue
            seen.add(mod)
            queue.append((mod, ast.parse(path.read_text())))
    return offenders


def test_props_cannot_reach_the_network():
    """PASTE ONLY. PrizePicks 403s server requests and ToS 16(l) bars automated
    access. Checked by parsing imports, not grepping text: the docstrings name
    the endpoint while explaining why nothing calls it."""
    assert _network_offenders((PKG / "props.py").read_text(), "props") == []
    assert _network_offenders((PKG / "props_ui.py").read_text(), "props_ui") == []
    assert _network_offenders((PKG / "pmf.py").read_text(), "pmf") == []


@pytest.mark.parametrize("laundered", [
    "\ndef scrape():\n    import requests\n    return requests.get(FEED_URL)\n",
    "\ndef scrape():\n    from .http import fetch\n    return fetch(FEED_URL)\n",
    "\nfrom . import live\n",
    "\nimport bucketlib.http as _h\n",
    "\nfrom bucketlib.sources import espn\n",
    "\ndef scrape():\n    return __import__('urllib.request')\n",
    "\nimport importlib\ndef scrape():\n    return importlib.import_module('socket')\n",
])
def test_the_network_guard_catches_a_laundered_fetcher(laundered):
    """A guard nobody has watched fail is not known to work. Each of these is a
    one-line way to reintroduce a fetcher, including through a sibling module
    (GridironValue's first guard missed exactly that)."""
    src = (PKG / "props.py").read_text() + laundered
    assert _network_offenders(src, "props"), laundered


# ── UI escaping and house style ──────────────────────────────────────────────
EVIL = '<img src=x onerror=alert(1)>"\'&'


def test_a_pasted_name_with_html_renders_escaped(roster, projections, no_reliability):
    feed = {"data": [{"type": "projection", "attributes": {
        "stat_type": "Points", "line_score": 20.5, "start_time": "2026-10-22T00:30:00Z"},
        "relationships": {"new_player": {"data": {"id": "p"}}}},
        {"type": "projection", "attributes": {
            "stat_type": "<script>alert(document.cookie)</script>", "line_score": 2.5,
            "start_time": "2026-10-22T00:30:00Z"},
         "relationships": {"new_player": {"data": {"id": "q"}}}}],
        "included": [{"type": "new_player", "id": "p", "attributes": {
            "name": "<script>alert(document.cookie)</script> Evil Name", "team": "DEN"}},
            {"type": "new_player", "id": "q", "attributes": {"name": "Nikola Jokic",
                                                              "team": "DEN"}}]}
    lines, _rep = props.prepare_paste(json.dumps(feed), roster, pd.read_csv(
        FIX / "schedule_constructed.csv", dtype={"game_id": str}), now=NOW)
    table, meta = props.compare(props.lines_for_date(lines, NEXT),
                                props.MeanDispersionProvider(projections), projections)
    assert meta["unmatched_names"] and meta["unmapped"]
    html = "".join(props_ui.notes_html(meta, *props.board_rows(table)))
    assert "<script>" not in html
    assert "&lt;script&gt;alert(document.cookie)&lt;/script&gt; Evil Name" in html
    assert html.count("&lt;script&gt;") >= 2        # the pasted NAME and the pasted LABEL


def test_ledger_and_player_rows_escape_every_field(no_reliability):
    row = {"person_id": 1, "game_id": "g", "player": EVIL, "team": EVIL, "opp": EVIL,
           "stat": None, "stat_label": EVIL, "line": 2.0, "model": 2.4, "gap": 0.4,
           "lean": "More", "sides": "more", "odds_type": EVIL, "tier": EVIL,
           "tier_edge": None, "source": EVIL, "void_risk": 0.1, "tie_possible": True,
           "p_push": 0.2}
    for html in (props_ui.ledger_html(pd.DataFrame([row]), {1: 3}),
                 props_ui.render_player_lines([row])):
        assert "<img" not in html and "onerror=alert(1)>" not in html
        assert "&lt;img src=x onerror=alert(1)&gt;&quot;&#x27;&amp;" in html


def test_markdown_messages_escape_pasted_names():
    msgs = props_ui.report_messages({"nonempty": True, "kept": 1, "unrouted": 1,
                                     "unrouted_names": ["[x](javascript:alert) **b**"],
                                     "by_date": {}, "parse": {}})
    text = " ".join(t for _k, t in msgs)
    assert "[x](javascript" not in text and "\\[x\\]" in text


def test_demon_goblin_badges_and_footnote():
    assert "bv-odds demon" in props_ui._odds_badge("DEMON")
    assert "bv-odds goblin" in props_ui._odds_badge("goblin")
    assert props_ui._odds_badge("standard") == "" and props_ui._odds_badge(None) == ""
    assert props_ui.odds_footnote(pd.DataFrame([{"odds_type": "standard"}])) == ""
    foot = props_ui.odds_footnote(pd.DataFrame([{"odds_type": "demon"}, {"odds_type": "goblin"}]))
    assert "Demons and Goblins on this board" in foot and "21 August 2026" in foot
    assert "NBA" in foot and "per line" in foot and "not evidence" in foot


def test_count_header_and_gap_is_never_a_percentage(no_reliability):
    pmf = Pmf.from_poisson(26.0)
    table, meta = props.compare(_matched([_line(line=x) for x in (20.5, 26.0, 31.5)]),
                                FixedProvider(pmf), _proj_row())
    shown, hidden, thin, void = props.board_rows(table)
    assert props_ui.count_header(shown, hidden, thin, void).startswith(f"{len(shown)} of 3 lines shown (")
    html = props_ui.ledger_html(table)
    assert "%" not in html.split('class="bv-ledger-head"')[1].split("</div>")[0]
    assert ">Gap<" in html and ">Lean<" in html


def test_rendered_gap_sign_matches_the_lean(no_reliability):
    import re
    rng = np.random.default_rng(9)
    pmf = Pmf(np.arange(12), rng.dirichlet(np.ones(12)))
    table, _ = props.compare(_matched([_line(line=float(x)) for x in np.arange(0, 12, 0.5)]),
                             FixedProvider(pmf), _proj_row())
    for r in table.to_dict(orient="records"):
        gap = re.sub(r"<[^>]+>", "", props_ui._fmt_gap(r)).replace("&lt;", "<")
        if r["lean"] == "More":
            assert gap.startswith("+") or gap == "<0.01"
            assert "over" in props_ui._fmt_gap(r)
        elif r["lean"] == "Less":
            assert gap.startswith("-") or gap == "<0.01"
            assert "under" in props_ui._fmt_gap(r)


def test_top_per_player_keeps_one_row_each_with_a_count():
    t = pd.DataFrame({"person_id": [1, 2, 1, 3, 1, 2], "stat": list("abcdef")})
    rows, more = props_ui.top_per_player(t, n=2)
    assert list(rows["person_id"]) == [1, 2] and more == {1: 2, 2: 1}


def test_no_em_dashes_or_emojis_in_the_props_modules():
    """House style. util.SENTINEL is imported, never retyped."""
    for name in ("props.py", "props_ui.py", "pmf.py"):
        text = (PKG / name).read_text(encoding="utf-8")
        assert "—" not in text, f"em dash in {name}"
        assert "–" not in text, f"en dash in {name}"
        emoji = [ch for ch in text if ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF]
        assert not emoji, f"emoji in {name}: {emoji[:3]}"


def test_feed_url_is_one_constant_and_only_a_link():
    assert props.FEED_URL.startswith("https://api.prizepicks.com/projections?")
    assert f"league_id={props.NBA_LEAGUE_ID}" in props.FEED_URL
    assert "game_mode=pickem" in props.FEED_URL
    src = (PKG / "props_ui.py").read_text()
    assert "api.prizepicks.com" not in src, "the URL must live in one place"


# ── Streamlit smoke tests (AppTest, no browser) ──────────────────────────────
def _apptest(script: str):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string(script, default_timeout=60)


_SETUP = f"""
import sys, datetime as dt
sys.path.insert(0, {str(REPO)!r})
import pandas as pd
from bucketlib import props, props_ui
F = {str(FIX)!r} + "/"
roster = pd.read_csv(F + "roster_constructed.csv")
sched = pd.read_csv(F + "schedule_constructed.csv", dtype={{"game_id": str}})
sched["tip_utc"] = pd.to_datetime(sched["tip_utc"], utc=True)
proj = pd.read_csv(F + "projections_constructed.csv", dtype={{"game_id": str}})
NOW = pd.Timestamp("2026-10-21T03:00:00Z").to_pydatetime()
"""


def test_popover_paste_then_clear_end_to_end(board_text):
    at = _apptest(_SETUP + """
token = props_ui.render_update_lines_popover(dt.date(2026, 10, 20), sched, roster, now=NOW)
view = props_ui.board_view(token, "2026-10-21", props.MeanDispersionProvider(proj), proj)
props_ui.render_board(view.shown, view.hidden, view.thin, view.void, "Wed Oct 21",
                      meta=view.meta)
""")
    at.run()
    assert not at.exception
    other = props.new_board(pd.DataFrame([_line()]))          # another visitor
    at.text_area(key=props_ui.PASTE_KEY).input(board_text)
    at.button(key="bv_pp_add").click()
    at.run()
    assert not at.exception, at.exception
    token = at.query_params.get("board")
    token = token[0] if isinstance(token, list) else token
    assert props.valid_token(token) and props.get_board(token) is not None
    success = " ".join(s.value for s in at.success)
    assert "lines loaded" in success and "Wed Oct 21" in success
    assert at.text_area(key=props_ui.PASTE_KEY).value == ""          # applied once
    md = " ".join(m.value for m in at.markdown)
    assert "Model vs the board" in md and "lines shown (" in md
    at.button(key="bv_pp_clear").click()
    at.run()
    assert not at.exception
    assert props.get_board(token) is None
    assert props.get_board(other) is not None, "Clear all touched another visitor"
    assert not at.query_params.get("board")


def test_render_board_with_nothing_pasted_renders_nothing():
    at = _apptest(_SETUP + """
v = props_ui.board_view(None, "2026-10-21", props.MeanDispersionProvider(proj), proj)
n = props_ui.render_board(v.shown, v.hidden, v.thin, v.void, "Wed Oct 21", meta=v.meta)
import streamlit as st
st.write(f"returned {n}")
""")
    at.run()
    assert not at.exception
    assert "Model vs the board" not in " ".join(m.value for m in at.markdown)


# ═════════════════════════════════════════════════════════════════════════════
# Audit fixes (2026-10-09). Each test below was run and watched FAIL against the
# code the audit read, then the fix went in. The defect number is the audit's.
# ═════════════════════════════════════════════════════════════════════════════
MORNING = pd.Timestamp("2026-10-20T15:00:00Z").to_pydatetime()   # 11am ET Oct 20


def _feed(*rows):
    """A minimal feed: rows are (name, team, stat, line, start_time) tuples."""
    data, inc = [], []
    for i, (name, team, stat, line, start) in enumerate(rows):
        data.append({"type": "projection", "id": str(i), "attributes": {
            "stat_type": stat, "line_score": line, "start_time": start},
            "relationships": {"new_player": {"data": {"id": f"p{i}"}}}})
        attrs = {"name": name}
        if team:
            attrs["team"] = team
        inc.append({"type": "new_player", "id": f"p{i}", "attributes": attrs})
    return json.dumps({"data": data, "included": inc})


# ── Defect 1: one square's state leaked into the next ───────────────────────
def test_a_team_square_never_lends_its_state_to_the_next_player(roster, schedule):
    txt = ("DEN\nDEN - TM\nvs DAL Wed 8:30pm\n118.5\nTeam Points\n"
           "Stephen Curry\n24.5\nPoints\nLess\nMore\n")
    df = props.parse_prizepicks_board(txt)
    assert len(df) == 1
    assert df.iloc[0]["team"] is None and df.iloc[0]["opp"] is None
    lines, _ = props.prepare_paste(txt, roster, schedule, now=NOW)
    curry = lines.iloc[0]
    assert (curry["how"], curry["game_id"]) == ("exact", "0022600006")
    # a namesake never inherits the team of the square before it
    smith = props.parse_prizepicks_board(
        "BOS - TM\n118.5\nTeam Points\nChris Smith\n4.5\nAssists\nLess\nMore\n")
    assert smith.iloc[0]["team"] is None
    # a LIVE team square does not make the next pre-game square live
    live = props.parse_prizepicks_board(
        "DEN - TM\nDEN 50 vs DAL 48\n118.5\nTeam Points\n"
        "Stephen Curry\n24.5\nPoints\nLess\nMore\n")
    assert len(live) == 1 and live.attrs["skipped_live"] == 0
    # a team square after an abandoned player square: the total is not his line
    abandoned = props.parse_prizepicks_board(
        "Nikola Jokic\nDEN - C\nDEN\nDEN - TM\n118.5\nTeam Points\n"
        "Stephen Curry\n24.5\nPoints\nMore\n")
    assert list(zip(abandoned["name"], abandoned["stat_type"])) == [("Stephen Curry", "Points")]
    assert abandoned.iloc[0]["team"] is None
    # Each reset is needed on its own (a test that only the OTHER fix makes pass
    # asserts nothing). The nameless square's number ends ITS odds: a marker
    # inside a team square never becomes the next player's Demon...
    marker = props.parse_prizepicks_board(
        "DEN - TM\nDemon\n118.5\nTeam Points\nStephen Curry\n24.5\nPoints\nLess\nMore\n")
    assert marker.iloc[0]["odds_type"] == "standard"
    # ...and a new name ends whatever came before it with no number in between
    # (a team square abandoned before its line, or a stray TEAM - POS line).
    for pre in ("DEN - TM\nvs DAL Wed 8:30pm\n", "BOS - G\n", "NYK 54 vs PHI 50\n"):
        df = props.parse_prizepicks_board(pre + "Stephen Curry\n24.5\nPoints\nLess\nMore\n")
        assert len(df) == 1 and df.iloc[0]["team"] is None and df.iloc[0]["opp"] is None, pre


# ── Defect 2: the record's guards failed open ───────────────────────────────
def test_the_schedule_tip_rides_on_every_routed_row(board_text, roster, schedule,
                                                    projections, no_reliability):
    lines, _ = props.prepare_paste(board_text, roster, schedule, now=NOW)
    day = props.lines_for_date(lines, NEXT)
    table, _ = props.compare(day, props.MeanDispersionProvider(projections), projections)
    assert len(table) and table["tip_utc"].notna().all()
    curry = table[(table["player"] == "Stephen Curry")].iloc[0]
    assert pd.Timestamp(curry["tip_utc"]) == pd.Timestamp("2026-10-22T02:00:00Z")


def test_the_record_fails_closed(record_dir, board_text, roster, schedule, projections,
                                 no_reliability):
    lines, _ = props.prepare_paste(board_text, roster, schedule, now=NOW)
    table, _ = props.compare(props.lines_for_date(lines, NEXT),
                             props.MeanDispersionProvider(projections), projections)
    assert len(table)
    # the fixture projections never say whether they are provisional: not frozen
    assert table["provisional"].isna().all()
    assert props.freeze_projections(table, "v", now=NOW) == 0
    # every game has tipped: nothing freezes, whatever the projection row says
    late = pd.Timestamp("2026-10-23T12:00:00Z").to_pydatetime()
    assert props.freeze_projections(table.assign(provisional=False), "v", now=late) == 0
    # an unknown tip is never assumed to be in the future
    assert props.freeze_projections(_frozen_table(tip_utc=None), "v", now=NOW) == 0
    assert props.freeze_projections(_frozen_table(provisional=None), "v", now=NOW) == 0
    assert props.load_frozen(NEXT) is None
    for bad in (None, "", "   "):
        with pytest.raises(ValueError):
            props.freeze_projections(_frozen_table(), bad, now=NOW)
    # and the honest case still freezes
    assert props.freeze_projections(table.assign(provisional=False), "v", now=NOW) > 0


# ── Defect 3: the thin filter failed open ───────────────────────────────────
def test_thin_fails_closed_without_a_game_count(board_text, roster, schedule, projections,
                                               no_reliability):
    lines, _ = props.prepare_paste(board_text, roster, schedule, now=NOW)
    day = props.lines_for_date(lines, NEXT)
    bare = projections.drop(columns="games_prior")
    table, meta = props.compare(day, props.MeanDispersionProvider(bare), bare)
    shown, hidden, thin, void = props.board_rows(table)
    assert len(table) and shown.empty and len(thin) == len(table) - len(hidden)
    assert meta["games_unknown"] is True
    html = "".join(props_ui.notes_html(meta, shown, hidden, thin, void))
    assert "no prior-game count" in html
    # a table that never went through compare carries no 'thin' column at all
    informed, thin2 = props.filter_thin(pd.DataFrame({"person_id": [1, 2]}))
    assert informed.empty and len(thin2) == 2


# ── Defect 4: the no-network guard had holes ────────────────────────────────
@pytest.mark.parametrize("laundered", [
    "\ndef scrape():\n    return pd.read_json(FEED_URL)\n",
    "\nimport subprocess\ndef scrape():\n    return subprocess.run(['curl', FEED_URL])\n",
    "\ndef scrape():\n    return np.DataSource(None).open(FEED_URL)\n",
    "\nimport importlib\ndef scrape():\n"
    "    return getattr(importlib, 'import_' + 'module')('urllib.request')\n",
    "\nfrom pandas.io.common import urlopen\n",
    "\nimport os\ndef scrape():\n    return os.system('curl ' + FEED_URL)\n",
    "\ndef scrape():\n    u = FEED_URL\n    return u\n",
    "\ndef scrape():\n    return pd.read_csv('https://api.prizepicks.com/projections')\n",
    "\nimport sys\ndef scrape():\n    return sys.modules['urllib.request']\n",
    "\nimport asyncio\n",
    "\ndef scrape(mod, name):\n    return getattr(mod, name)\n",
])
def test_the_network_guard_catches_the_audits_launders(laundered):
    src = (PKG / "props.py").read_text() + laundered
    assert _network_offenders(src, "props"), laundered


def test_the_ui_module_may_only_link_the_feed():
    """props_ui puts FEED_URL in an <a href>, through esc(). Anything else that
    touches it is a fetch in waiting."""
    src = (PKG / "props_ui.py").read_text()
    assert _network_offenders(src, "props_ui") == []
    bad = src + "\ndef fetch():\n    return pd.read_json(props.FEED_URL)\n"
    assert _network_offenders(bad, "props_ui")


@pytest.fixture()
def socket_guard(monkeypatch):
    """Every way out of the process through a socket, recorded and refused."""
    import socket
    calls = []

    def refuse(name):
        def _f(*a, **k):
            calls.append(name)
            raise OSError(f"network blocked in test ({name})")
        return _f
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, refuse(name))
    return calls


def test_the_whole_paste_path_runs_without_a_socket(socket_guard, feed_text, board_text,
                                                    roster, schedule, projections,
                                                    record_dir, no_reliability):
    provider = props.MeanDispersionProvider(projections)
    for text, now in ((feed_text, MORNING), (board_text, NOW)):
        token, report = props_ui.ingest(text, None, roster, schedule, now=now)
        assert token and not report["error"]
        for d in report["by_date"]:
            v = props_ui.board_view(token, d, provider, projections, roster, void_ids=("1630178",))
            props_ui.ledger_html(v.shown)
            props_ui.notes_html(v.meta, v.shown, v.hidden, v.thin, v.void)
            for rows in props_ui.rows_by_person(v.shown).values():
                props_ui.render_player_lines(rows)
            props.freeze_projections(v.shown.assign(provisional=False), "v", now=now)
        props_ui.report_messages({**report, "view_date": NEXT})
    assert socket_guard == []


def test_the_socket_guard_is_known_to_fire(socket_guard):
    """A guard nobody has watched fail is not known to work."""
    import urllib.request
    with pytest.raises(OSError):
        urllib.request.urlopen("http://127.0.0.1:9/", timeout=1)
    assert socket_guard


# ── Defect 5: memory on the shared process ──────────────────────────────────
def test_the_paste_cap_is_sized_to_a_real_feed():
    assert props.MAX_PASTE_CHARS <= 8_000_000


def test_board_text_stops_reading_at_its_line_cap(monkeypatch):
    monkeypatch.setattr(props, "MAX_TEXT_LINES", 50)
    square = "Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n"
    df = props.parse_prizepicks_board(square * 100)       # 600 lines
    assert len(df) <= 50 // 6 + 1 and df.attrs["truncated"]


def test_json_with_too_many_containers_is_refused_before_it_is_built(monkeypatch, feed_text):
    real_cap = props.MAX_JSON_CONTAINERS
    assert real_cap >= 200_000                       # ~60k in a 3,000-row feed
    monkeypatch.setattr(props, "MAX_JSON_CONTAINERS", 1000)
    got = props.parse_any("[" + "{}," * 5000 + "{}]")
    assert got.empty and got.attrs["too_large"]
    assert len(props.parse_any(feed_text)) == 15      # a real-shaped feed is far under it


def test_the_board_store_is_bounded_in_bytes(monkeypatch, roster, schedule):
    """The audit's hostile board: thousands of distinct 70-character labels."""
    rows = "\n".join(f"Nikola Jokic\nDEN - C\n{i % 900}.5\n{'X' * 70}{i}\nMore"
                     for i in range(3000))
    first, _rep = props_ui.ingest(rows, None, roster, schedule, now=MORNING)
    one = props.board_info(first)["nbytes"]
    assert one / props.board_info(first)["rows"] < 1100      # compact columns
    assert set(props.get_board(first).columns) <= set(props.BOARD_COLUMNS)
    monkeypatch.setattr(props, "BOARD_MAX_BYTES", int(one * 2.5))
    tokens = [props_ui.ingest(rows, None, roster, schedule, now=MORNING)[0] for _ in range(5)]
    with props._BOARDS_LOCK:
        held = sum(b.nbytes for b in props._BOARDS.values())
        measured = sum(int(b.lines.memory_usage(deep=True).sum()) for b in props._BOARDS.values())
        count = len(props._BOARDS)
    assert held == measured <= props.BOARD_MAX_BYTES and count == 2
    assert props.get_board(tokens[-1]) is not None           # the newest paste survives
    assert props.get_board(first) is None                    # the oldest went first


# ── Defect 6: feed lines for games that had tipped were priced ──────────────
def test_a_feed_line_for_a_game_that_has_tipped_is_not_routed(roster, schedule):
    text = _feed(("Jayson Tatum", "BOS", "Points", 26.5, "2026-10-20T23:00:00Z"),
                 ("Nikola Jokic", "DEN", "Points", 29.5, "2026-10-22T00:30:00Z"))
    lines, report = props.prepare_paste(text, roster, schedule, now=NOW)
    tatum = lines[lines["name"] == "Jayson Tatum"].iloc[0]
    assert pd.isna(tatum["game_date"]) and tatum["route_how"] == "tipped"
    assert report["by_date"] == {NEXT: 1} and "Jayson Tatum" in report["unrouted_names"]
    pre = schedule.assign(status="scheduled")
    early, _ = props.prepare_paste(text, roster, pre, now=MORNING)
    assert early[early["name"] == "Jayson Tatum"].iloc[0]["game_id"] == "0022600001"


# ── Defect 7: one bad row discarded the whole paste ─────────────────────────
@pytest.mark.parametrize("start", ["2999-01-01T00:00:00Z", "9999-12-31T23:59:59Z",
                                   "0001-01-01T00:00:00Z"])
def test_an_absurd_start_time_costs_one_row(roster, schedule, start):
    text = _feed(("Nikola Jokic", "DEN", "Points", 29.5, "2026-10-22T00:30:00Z"),
                 ("Some Unknown Guy", None, "Points", 25.5, start))
    token, report = props_ui.ingest(text, None, roster, schedule, now=NOW)
    assert token and not report["error"], report
    board = props.get_board(token)
    assert "Nikola Jokic" in set(board["name"])
    assert report["by_date"] == {NEXT: 1}


def test_classify_resolve_and_route_catch_errors_per_row(roster, schedule, monkeypatch):
    real_classify, real_ts = props.classify_stat, props._parse_ts

    def classify(stat_type, *a, **k):
        if stat_type == "Boom":
            raise RuntimeError("classify")
        return real_classify(stat_type, *a, **k)

    def parse_ts(value):
        if value == "boom":
            raise RuntimeError("route")
        return real_ts(value)
    monkeypatch.setattr(props, "classify_stat", classify)
    monkeypatch.setattr(props, "_parse_ts", parse_ts)
    text = _feed(("Nikola Jokic", "DEN", "Points", 29.5, "2026-10-22T00:30:00Z"),
                 ("Nikola Jokic", "DEN", "Boom", 9.5, "2026-10-22T00:30:00Z"),
                 ("Nikola Jokic", "DEN", "Rebounds", 12.5, "boom"))
    lines, report = props.prepare_paste(text, roster, schedule, now=NOW)
    assert len(lines) == 3
    by = lines.set_index("stat_type")
    assert by.loc["Points", "game_id"] == "0022600005"
    assert by.loc["Boom", "kind"] == "unmapped"
    assert by.loc["Rebounds", "route_how"] == "error"

    class Flaky:
        def get(self, name, team=None):
            if name == "Boom Name":
                raise RuntimeError("resolve")
            return props.build_name_index(roster).get(name, team=team)
    res, unres = props.resolve_names(pd.DataFrame([_line("Boom Name"), _line()]), roster,
                                     index=Flaky())
    assert len(res) == 1 and list(unres["how"]) == ["error"]
    assert list(unres["reason"]) == ["unmatched"]


# ── Defect 8: compare re-resolved resolved lines and broke ──────────────────
def test_compare_trusts_the_paste_time_match(roster, projections, no_reliability):
    import warnings
    # resolved at paste time against an EMPTY roster: nothing matched, so the
    # frame has a 'match' column and no 'person_id' column
    empty_roster = roster.iloc[0:0]
    lines, _ = props.prepare_paste(
        "Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n"
        "Chris Smith\n4.5\nAssists\nLess\nMore\n", empty_roster,
        pd.DataFrame(columns=["game_id", "tip_utc", "away", "home", "status"]), now=NOW)
    assert "match" in lines.columns and "person_id" not in lines.columns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table, meta = props.compare(lines, props.MeanDispersionProvider(projections),
                                    projections, roster)
    assert props.ledger(meta, *props.board_rows(table))["balanced"]
    assert meta["unmatched"] == 2 and meta["matched"] == 0
    # without a roster, an ambiguous verdict stays ambiguous
    amb = pd.DataFrame([_line("Chris Smith", team=None)]).assign(match="ambiguous")
    _t, meta2 = props.compare(amb, None, projections, None)
    assert meta2["ambiguous"] == 1 and meta2["unmatched"] == 0


def test_the_saved_fuzz_input_that_broke_compare(roster, schedule, projections, no_reliability):
    """The shape of the audit fuzzer's saved input (d2_30): no line resolved at
    paste time (a combo-flagged player record is refused before resolution,
    the other name is unknown), but the combo-flagged name IS on the roster, so
    a second resolution inside compare() matched it and the merge raised."""
    data, inc = [], []
    for i, (name, team, combo) in enumerate((("Yang Hansen", "POR", True),
                                             ("Lal Gsw", None, False))):
        data.append({"type": "projection", "id": str(i), "attributes": {
            "stat_type": "Points", "line_score": 6.5, "start_time": "2026-10-22T00:00:00Z"},
            "relationships": {"new_player": {"data": {"id": f"p{i}"}}}})
        attrs = {"name": name, "combo": combo}
        if team:
            attrs["team"] = team
        inc.append({"type": "new_player", "id": f"p{i}", "attributes": attrs})
    lines, _ = props.prepare_paste(json.dumps({"data": data, "included": inc}),
                                   roster, schedule.assign(status="scheduled"), now=MORNING)
    assert "person_id" not in lines.columns
    table, meta = props.compare(props.lines_for_date(lines, NEXT),
                                props.MeanDispersionProvider(projections), projections, roster)
    assert props.ledger(meta, *props.board_rows(table))["balanced"]
    assert meta["n_refused"] == 1 and meta["unmatched"] == 1


# ── Defect 9: hostile void ids and dates ────────────────────────────────────
def test_hostile_void_ids_never_raise(no_reliability):
    pmf = Pmf.from_poisson(30.0)
    table, _ = props.compare(_matched([_line(line=24.5)]), FixedProvider(pmf), _proj_row())
    for ids in (["abc"], ["203999", "abc", None, float("nan"), float("inf"), "1e400", True],
                ["<script>"], ["9" * 400]):
        shown, hidden, thin, void = props.board_rows(table, ids)
        assert len(shown) + len(hidden) + len(thin) + len(void) == len(table)
    assert len(props.board_rows(table, ["203999"])[3]) == 1
    assert len(props.board_rows(table, [203999.0])[3]) == 1


def test_board_view_never_raises_on_a_bad_date(roster, schedule, projections):
    token, _ = props_ui.ingest("Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n", None,
                               roster, schedule, now=NOW)
    provider = props.MeanDispersionProvider(projections)
    for date in ("not-a-date", "<script>", None, "99999-99-99"):
        v = props_ui.board_view(token, date, provider, projections)
        assert v.shown.empty


# ── Defect 10: the banner counted lines that can never show ─────────────────
def test_the_banner_counts_only_lines_that_can_be_compared(roster, schedule):
    text = ("Nobody Known Here\nDEN - C\n27.5\nPoints\nLess\nMore\n"
            "Paolo Banchero\nORL - F\n6.5\n1Q Points\nLess\nMore\n")
    _t, report = props_ui.ingest(text, None, roster, schedule, now=NOW)
    assert report["kept"] == 2 and report["comparable"] == 0
    msgs = props_ui.report_messages({**report, "view_date": NEXT})
    assert not any(k == "success" for k, _ in msgs), msgs
    assert msgs[0][0] == "warning"
    assert "not matched" in msgs[0][1] and "not priced" in msgs[0][1]
    assert props_ui.toast_count(report) == 0


def test_the_banner_count_on_the_board_fixture(board_text, roster, schedule):
    _t, report = props_ui.ingest(board_text, None, roster, schedule, now=NOW)
    # Jokic x2, Doncic, Curry x2, Washington, Antetokounmpo, Jackson Jr.
    assert report["comparable"] == 8
    assert (report["comparable"] + sum(report["not_comparable"].values())
            + report["unrouted"]) == report["kept"]
    msgs = props_ui.report_messages({**report, "view_date": NEXT})
    assert msgs[0] == ("success", msgs[0][1]) and msgs[0][1].startswith("8 lines loaded")
    assert props_ui.toast_count(report) == 8


# ── Defect 11: tests that a regression would have walked past ───────────────
def _evil(k: str) -> str:
    return f"<img src=x onerror=alert('{k}')>"


def _escaped(k: str) -> str:
    return f"&lt;img src=x onerror=alert(&#x27;{k}&#x27;)&gt;"


def test_every_html_sink_escapes_its_own_field(monkeypatch, no_reliability):
    monkeypatch.setattr(props, "TIER_LABEL", {**props.TIER_LABEL, "unmeasured": _evil("tier")})
    base = {"person_id": 1, "game_id": "g", "stat": "pts", "line": 2.5, "model": 3.1,
            "gap": 0.6, "sides": "both", "odds_type": "standard", "tier": "unmeasured",
            "tier_edge": None, "source": "model", "void_risk": float("nan"),
            "tie_possible": False, "p_push": 0.0, "games_prior": 1.0}
    shown = pd.DataFrame([{**base, "player": _evil("player"), "team": _evil("team_only"),
                           "opp": None, "stat_label": "Points", "lean": _evil("lean")}])
    thin = pd.DataFrame([{**base, "player": _evil("thin"), "team": "DEN", "opp": "DAL",
                          "lean": "More"}])
    void = pd.DataFrame([{**base, "player": _evil("void"), "team": "DEN", "opp": "DAL",
                          "lean": "More"}])
    hidden = pd.DataFrame([{**base, "player": "x", "lean": "Even"}])
    meta = {"posted": 9, "matched": 4, "unmatched": 1, "ambiguous": 1, "n_refused": 2,
            "unprojected": 1, "unmatched_names": [_evil("unmatched")],
            "ambiguous_names": [_evil("ambiguous")],
            "refused": {_evil("refused_label"): 1}, "refused_why": {_evil("refused_label"): _evil("why")},
            "unmapped": {_evil("unmapped"): 1}, "unprojected_why": {_evil("unprojected"): 1},
            "nonexact": [(_evil("pasted"), _evil("roster"), "alias")]}
    html = "".join(props_ui.notes_html(meta, shown, hidden, thin, void))
    html += props_ui.ledger_html(shown) + props_ui.render_player_lines(shown)
    html += props_ui.window_html(shown, hidden, thin, void, _evil("scope"))
    html += props_ui.none_matched_html(3, _evil("scope_notice"))
    assert "<img" not in html
    for k in ("player", "team_only", "lean", "tier", "thin", "void", "unmatched", "ambiguous",
              "refused_label", "why", "unmapped", "unprojected", "pasted", "roster", "scope",
              "scope_notice"):
        assert _escaped(k) in html, k


def test_the_even_band_is_a_band_not_a_point(no_reliability):
    b = props.EVEN_BAND
    assert b > 0
    assert props.lean_from(0.5 + b / 2) == "Even" and props.lean_from(0.5 - b / 2) == "Even"
    assert props.lean_from(0.5 + 2 * b) == "More" and props.lean_from(0.5 - 2 * b) == "Less"
    q = 0.5 + b / 2                                   # inside the band, but above a coin
    pmf = Pmf([0, 1], [1 - q, q])
    table, _ = props.compare(_matched([_line(line=0.5)]), FixedProvider(pmf), _proj_row())
    assert table.iloc[0]["q_more"] == pytest.approx(q) and table.iloc[0]["lean"] == "Even"
    shown, hidden, _thin, _void = props.board_rows(table)
    assert shown.empty and len(hidden) == 1


def test_pills_match_an_independent_board_rows(board_text, roster, schedule, projections,
                                               no_reliability):
    """The pill test compared board_view with itself, so board_view returning
    the unfiltered table passed. This recomputes the partition independently
    and needs rows that are NOT shown to exist."""
    token, _ = props_ui.ingest(board_text, None, roster, schedule, now=NOW)
    provider = props.MeanDispersionProvider(projections)
    void_ids = {203999}
    table, _meta = props.compare(props.lines_for_date(props.get_board(token), NEXT),
                                 provider, projections)
    shown, hidden, thin, void = props.board_rows(table, void_ids)
    assert len(hidden) + len(thin) + len(void) > 0, "the fixture no longer exercises the filter"
    view = props_ui.board_view(token, NEXT, provider, projections, void_ids=void_ids)
    expected = shown.groupby("game_id").size().to_dict()
    assert props_ui.line_counts_by_game(view.shown) == {str(k): int(v) for k, v in expected.items()}
    assert len(view.shown) == len(shown) < len(table)


def test_the_token_gate_keeps_hostile_keys_out_of_the_store():
    for bad in (None, "", "short", "x" * 65, "<script>alert(1)</script>xxxxxxxxxx",
                "../../etc/passwd/aaaaaaaaaaaaaa", "a" * 19, "a b" * 10, 12345, b"a" * 30):
        assert props.valid_token(bad) is False, bad
    assert props.valid_token("a" * 20) and props.valid_token("A_-9" * 8)
    hostile = "<script>alert(1)</script>xxxxxxxxxxxxxxxx"
    t = props.add_to_board(hostile, pd.DataFrame([_line()]))
    assert t != hostile and props.valid_token(t)
    with props._BOARDS_LOCK:
        assert hostile not in props._BOARDS


def test_one_feed_row_that_raises_costs_that_row(monkeypatch):
    real = props.sides_from_allowed_wager_types

    def flaky(value):
        if value == ["boom"]:
            raise RuntimeError("only this row")        # not a type any except names
        return real(value)
    monkeypatch.setattr(props, "sides_from_allowed_wager_types", flaky)
    raw = json.loads(_feed(("Nikola Jokic", "DEN", "Points", 29.5, "2026-10-22T00:30:00Z"),
                           ("Jamal Murray", "DEN", "Points", 19.5, "2026-10-22T00:30:00Z")))
    raw["data"][1]["attributes"]["allowed_wager_types"] = ["boom"]
    got = props.parse_any(json.dumps(raw))
    assert list(got["name"]) == ["Nikola Jokic"] and got.attrs["skipped_error"] == 1


# ── Defect 12: the record's scope was not the board's ───────────────────────
def test_the_record_freezes_only_what_the_board_shows(record_dir):
    t = pd.concat([
        _frozen_table(stat="pts"),                                   # shown
        _frozen_table(stat="reb", lean="More", sides="less"),        # hidden: side not offered
        _frozen_table(stat="ast", lean="Even"),                      # hidden: Even
        _frozen_table(stat="stl", thin=True),                        # thin
        _frozen_table(stat="blk", person_id=1629023),                # void
    ], ignore_index=True)
    assert props.freeze_projections(t, "v", now=NOW, void_ids={1629023}) == 1
    assert list(props.load_frozen(NEXT)["stat"]) == ["pts"]


def test_season_record_drops_the_training_window():
    g = pd.DataFrame([{"outcome": "correct", "result": "More", "game_date": "2026-03-01"}] * 30
                     + [{"outcome": "incorrect", "result": "Less", "game_date": "2026-11-01"}] * 10
                     + [{"outcome": "correct", "result": "More", "game_date": "2026-11-02"}] * 10)
    reg = {"trained_through": "2025-26"}
    out = props.season_record([g], registry=reg)
    assert out["decided"] == 20 and out["in_sample_dropped"] == 30
    assert props.season_record([g])["decided"] == 50


# ── Defect 13: Fantasy ties, and the board cap's silent drop ────────────────
def test_a_fantasy_line_on_the_tenths_grid_can_tie_and_says_so(no_reliability):
    pmf = Pmf(np.round(np.arange(340, 371) / 10, 1), np.ones(31) / 31)
    table, _ = props.compare(_matched([_line(stat="Fantasy Score", line=35.5),
                                       _line(stat="Points", line=27.5)]),
                             FixedProvider(pmf, {"pts": Pmf.from_poisson(27.0)}), _proj_row())
    fan = table[table["stat"] == "fantasy"].iloc[0]
    assert fan["p_push"] > 0 and bool(fan["tie_possible"]) is True
    assert "landing exactly on 35.5" in props_ui._stat_cell(fan.to_dict())
    pts = table[table["stat"] == "pts"].iloc[0]
    assert bool(pts["tie_possible"]) is False


def test_the_board_cap_counts_what_it_drops(monkeypatch, roster, schedule):
    monkeypatch.setattr(props, "BOARD_MAX_ROWS", 3)
    text = "".join(f"Nikola Jokic\nDEN - C\n{i}.5\n{lab}\nLess\nMore\n"
                   for i, lab in enumerate(["Points", "Rebounds", "Assists", "Steals", "Blocks"]))
    token, report = props_ui.ingest(text, None, roster, schedule, now=NOW)
    assert report["on_board"] == 3 and report["dropped_by_cap"] == 2
    msgs = " ".join(t for _k, t in props_ui.report_messages({**report, "view_date": NEXT}))
    assert "limit of 3 lines" in msgs and "2 oldest" in msgs
