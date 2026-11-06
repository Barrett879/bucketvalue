"""A typed list of picks, pasted into the same box as the PrizePicks board
(Barrett, 2026-11-06: "can you incorporate these into the script").

The first real list was for Magic at Suns on 2026-11-05 and arrived after
that game had ended. The roster and schedule rows below are real ones
(state/current at the time), cut to what the tests read, so routing is
exercised against the case that motivated it. An adversarial review of the
first version (wf_c2c4584b-ab2) found 24 problems; the tests named after
them pin each fix; a second review (wf_a391f788-8fa) found 26 more, pinned
the same way ("review 2"), a third (wf_1f71c1af-4f2) 31 more ("review 3"), a
fourth (wf_60c99f43-7a2) 25 more ("review 4"), a fifth (wf_25601194-36e) 33
more ("review 5"), a sixth (wf_23d96fe3-c45) 31 more ("review 6"), and a
seventh (wf_515a4685-47c) 37 more ("review 7").
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import props, props_ui, teams
from bucketlib.pmf import Pmf

FIX = Path(__file__).resolve().parent / "fixtures" / "props"

FIRST_LIST = """Suns:

Take the OVER on Miles Bridges for a combined 6.5 Rebounds and Assists (Combined Rebounds and Assists will be 7 or more)

Oso Ighodaro over 7.5 Points + Rebounds (Combined Points and Rebounds will be 8 or more)



Magic:

Wendell Carter Jr. under 23.5 Points + Rebounds + Assists (Combined Points, Rebounds, and Assists will be 23 or less)
"""

BEFORE_TIP = pd.Timestamp("2026-11-06T01:30:00Z").to_pydatetime()    # 8:30 PM ET Nov 5
AFTER_GAME = pd.Timestamp("2026-11-06T06:13:00Z").to_pydatetime()    # 1:13 AM ET Nov 6
MAGIC_AT_SUNS = "0022600187"
HEAT_AT_WARRIORS = "0022600188"
BLAZERS_AT_LAKERS = "0022600032"
# The committed layout of a version-1 browser copy (commit 4bcbc60). Written
# out, never read from props: this is the format already sitting in visitors'
# browsers, and the test must fail if the code stops reading it.
V1_COLUMNS = ["name", "team_code", "stat_type", "line", "direction", "odds_type",
              "opp", "is_combo", "league", "game_date", "game_id", "tip_utc"]
FIRE, CHECK, MONEY, BALL = chr(0x1F525), chr(0x2705), chr(0x1F4B0), chr(0x1F3C0)
EM, BUL, STAR = chr(0x2014), chr(0x2022), chr(0x2B50)
EN = chr(0x2013)
KEYCAP_0, KEYCAP_1, KEYCAP_2, KEYCAP_3, KEYCAP_4 = (f"{d}" + chr(0xFE0F) + chr(0x20E3)
                                                     for d in range(5))
ARROW = chr(0x2192)
FRIDAY_11AM = pd.Timestamp("2026-11-06T16:00:00Z").to_pydatetime()
FRIDAY_LATE = pd.Timestamp("2026-11-07T04:30:00Z").to_pydatetime()     # 11:30 PM ET Fri


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


@pytest.fixture(scope="module")
def roster():
    return pd.DataFrame([
        {"person_id": 1628970, "name": "Miles Bridges", "team": "PHX"},
        {"person_id": 1628969, "name": "Mikal Bridges", "team": "NYK"},
        {"person_id": 1642345, "name": "Oso Ighodaro", "team": "PHX"},
        {"person_id": 1628976, "name": "Wendell Carter Jr.", "team": "ORL"},
        {"person_id": 1628975, "name": "Jevon Carter", "team": "ORL"},
        {"person_id": 1631094, "name": "Paolo Banchero", "team": "ORL"},
        {"person_id": 1629029, "name": "Luka Doncic", "team": "LAL"},
        {"person_id": 201939, "name": "Stephen Curry", "team": "GSW"},
        {"person_id": 201950, "name": "Jrue Holiday", "team": "POR"},
        {"person_id": 1630166, "name": "Deni Avdija", "team": "POR"},
        {"person_id": 1629008, "name": "Michael Porter Jr.", "team": "BKN"},
        {"person_id": 1628983, "name": "Shai Gilgeous-Alexander", "team": "OKC"},
        {"person_id": 1642860, "name": "Will Riley", "team": "WAS"},
        {"person_id": 1628973, "name": "Jalen Brunson", "team": "NYK"},
        {"person_id": 1630700, "name": "Dyson Daniels", "team": "ATL"},
        {"person_id": 1641730, "name": "Noah Clowney", "team": "BKN"},
        {"person_id": 203076, "name": "Anthony Davis", "team": "WAS"},
        {"person_id": 1629627, "name": "Zion Williamson", "team": "NOP"},
        {"person_id": 1629622, "name": "Max Strus", "team": "LAC"},
        {"person_id": 1631108, "name": "Max Christie", "team": "DAL"},
        {"person_id": 1641744, "name": "Zach Edey", "team": "MEM"},
        {"person_id": 1628404, "name": "Josh Hart", "team": "NYK"},
        {"person_id": 1630162, "name": "Anthony Edwards", "team": "MIN"},
        {"person_id": 1630530, "name": "Trey Murphy III", "team": "NOP"},
        {"person_id": 1626157, "name": "Karl-Anthony Towns", "team": "NYK"},
    ])


@pytest.fixture(scope="module")
def schedule():
    rows = [
        ("0022600174", "2026-11-04", "2026-11-05T00:00:00Z", "final", "WAS", "ATL"),
        ("0022600183", "2026-11-04", "2026-11-05T02:00:00Z", "final", "PHX", "POR"),
        (MAGIC_AT_SUNS, "2026-11-05", "2026-11-06T02:00:00Z", "scheduled", "PHX", "ORL"),
        (HEAT_AT_WARRIORS, "2026-11-05", "2026-11-06T03:00:00Z", "scheduled", "GSW", "MIA"),
        ("0022600025", "2026-11-06", "2026-11-07T00:00:00Z", "scheduled", "ATL", "WAS"),
        ("0022600029", "2026-11-06", "2026-11-07T01:00:00Z", "scheduled", "MEM", "OKC"),
        ("0022600031", "2026-11-06", "2026-11-07T01:00:00Z", "scheduled", "NOP", "MIN"),
        (BLAZERS_AT_LAKERS, "2026-11-06", "2026-11-07T03:00:00Z", "scheduled", "LAL", "POR"),
        ("0022600190", "2026-11-07", "2026-11-08T00:30:00Z", "scheduled", "ATL", "BKN"),
        ("0022600191", "2026-11-07", "2026-11-08T00:30:00Z", "scheduled", "CLE", "NYK"),
        ("0022600192", "2026-11-07", "2026-11-08T01:00:00Z", "scheduled", "OKC", "WAS"),
        ("0022600193", "2026-11-07", "2026-11-08T01:30:00Z", "scheduled", "GSW", "ORL"),
        ("0022600194", "2026-11-07", "2026-11-08T02:00:00Z", "scheduled", "PHX", "HOU"),
        ("0022600198", "2026-11-08", "2026-11-09T00:00:00Z", "scheduled", "MIN", "UTA"),
        ("0022600201", "2026-11-08", "2026-11-09T03:30:00Z", "scheduled", "LAC", "NOP"),
        ("0022600216", "2026-11-10", "2026-11-11T04:00:00Z", "scheduled", "POR", "OKC"),
        ("0022600226", "2026-11-11", "2026-11-12T03:00:00Z", "scheduled", "POR", "OKC"),
    ]
    s = pd.DataFrame(rows, columns=["game_id", "game_date", "tip_utc", "status", "home", "away"])
    s["game_type"] = "RS"
    s["tip_utc"] = pd.to_datetime(s["tip_utc"], utc=True)
    s["game_date"] = pd.to_datetime(s["game_date"]).dt.date
    return s


def _prep(text, roster, schedule, now=BEFORE_TIP):
    lines, report = props.prepare_paste(text, roster, schedule, now=now)
    return lines.set_index("name", drop=False), report


def _one(text) -> dict:
    df = props.parse_typed_lines(text)
    assert len(df) == 1, (text, df.attrs)
    r = df.iloc[0].to_dict()
    r["stat"] = props.classify_stat(r["stat_type"])[0]
    return r


# ── The first list, as sent ──────────────────────────────────────────────────
def test_the_first_list_reads_as_three_picks():
    df = props.parse_any(FIRST_LIST)
    got = df[["name", "team", "stat_type", "line", "pick", "source"]].to_dict("records")
    assert got == [
        {"name": "Miles Bridges", "team": "PHX", "stat_type": "Rebs+Asts", "line": 6.5,
         "pick": "more", "source": "typed"},
        {"name": "Oso Ighodaro", "team": "PHX", "stat_type": "Pts+Rebs", "line": 7.5,
         "pick": "more", "source": "typed"},
        {"name": "Wendell Carter Jr.", "team": "ORL", "stat_type": "Pts+Rebs+Asts",
         "line": 23.5, "pick": "less", "source": "typed"},
    ]
    assert df.attrs["typed_headings"] == ["PHX", "ORL"]
    # The side written is the visitor's pick, never the sides the line offers.
    assert set(df["direction"]) == {""} and set(df["odds_type"]) == {"standard"}


def test_before_tip_the_first_list_lands_on_magic_at_suns(roster, schedule):
    lines, report = props.prepare_paste(FIRST_LIST, roster, schedule, now=BEFORE_TIP)
    assert list(lines["game_id"]) == [MAGIC_AT_SUNS] * 3
    assert list(lines["opp"]) == ["ORL", "ORL", "PHX"]
    assert list(lines["person_id"]) == [1628970, 1642345, 1628976]
    assert report["comparable"] == 3 and report["tipped"] == 0
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    assert msgs[0] == ("success", "3 lines loaded. 3 are for Thu Nov 5: see Model vs the "
                                  "board on the slate and on each game page.")


def test_after_the_game_the_first_list_is_late_never_the_next_game(roster, schedule):
    """Sent at 1:13 AM, after Magic at Suns had ended. Each team plays again
    on Nov 7 (PHX v HOU, ORL at GSW): the lines must not be filed there."""
    lines, report = props.prepare_paste(FIRST_LIST, roster, schedule, now=AFTER_GAME)
    assert lines["game_date"].isna().all() and set(lines["route_how"]) == {"tipped"}
    assert report["tipped"] == report["unrouted"] == 3 and report["comparable"] == 0
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    text = " ".join(t for _k, t in msgs)
    assert ("3 lines are for a game that has already started (Miles Bridges, Oso Ighodaro, "
            "Wendell Carter Jr\\.), so they cannot be compared. A line has to be pasted "
            "before its game tips.") in text
    assert "could not be placed" not in text            # not also called unknown


def test_without_its_headings_a_typed_line_routes_like_board_text(roster, schedule):
    """No game named: the player's next game today or tomorrow, as a board
    square without a matchup line goes."""
    lines, _ = props.prepare_paste("Miles Bridges over 6.5 Rebounds and Assists",
                                   roster, schedule, now=AFTER_GAME)
    assert (lines.iloc[0]["game_date"], lines.iloc[0]["game_id"]) == ("2026-11-07", "0022600194")


# ── How picks are written ────────────────────────────────────────────────────
@pytest.mark.parametrize("text,name,stat,line,pick", [
    ("Miles Bridges o6.5 R+A", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges O 6.5 Reb+Ast", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges 6.5 Rebs+Asts Over", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges Rebs+Asts over 6.5", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges: Combined Rebounds and Assists will be 7 or more",
     "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges 7+ rebounds and assists", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges at least 7 rebounds and assists", "Miles Bridges", "ra", 6.5, "more"),
    ("Wendell Carter Jr. 23 or fewer points, rebounds and assists",
     "Wendell Carter Jr.", "pra", 23.5, "less"),
    ("Wendell Carter Jr. no more than 23 PRA", "Wendell Carter Jr.", "pra", 23.5, "less"),
    ("Wendell Carter Jr. under 23.5 P + R + A", "Wendell Carter Jr.", "pra", 23.5, "less"),
    ("Wendell Carter Jr. o23.5 A + P + R", "Wendell Carter Jr.", "pra", 23.5, "more"),
    ("Wendell Carter Jr. P + R + A under 23.5", "Wendell Carter Jr.", "pra", 23.5, "less"),
    ("Miles Bridges over 6.5 R + A", "Miles Bridges", "ra", 6.5, "more"),
    ("Jalen Brunson over 2.5 TO", "Jalen Brunson", "tov", 2.5, "more"),
    ("Over 6.5 Rebounds and Assists for Miles Bridges", "Miles Bridges", "ra", 6.5, "more"),
    ("Take the under on Jalen Brunson's points at 25.5", "Jalen Brunson", "pts", 25.5, "less"),
    ("1. Stephen Curry over 4.5 3-PT Made", "Stephen Curry", "fg3m", 4.5, "more"),
    ("- Stephen Curry over 4.5 3 pointers made", "Stephen Curry", "fg3m", 4.5, "more"),
    ("Stephen Curry u 4.5 threes", "Stephen Curry", "fg3m", 4.5, "less"),
    ("Shai Gilgeous-Alexander more 31.5 points", "Shai Gilgeous-Alexander", "pts", 31.5, "more"),
    ("Jalen Brunson higher 11.5 assists", "Jalen Brunson", "ast", 11.5, "more"),
    ("Paolo Banchero over 1.5 blocks + steals", "Paolo Banchero", "stocks", 1.5, "more"),
    ("Luka Doncic over 26.5 Fantasy Score", "Luka Doncic", "fantasy", 26.5, "more"),
    ("Wendell Carter Jr. over 7.5 rebounds", "Wendell Carter Jr.", "reb", 7.5, "more"),
    ("miles bridges over 6.5 r+a", "miles bridges", "ra", 6.5, "more"),
    ("Miles Bridges, Rebs+Asts, 6.5", "Miles Bridges", "ra", 6.5, None),
    ("Miles Bridges,Rebs+Asts,6.5", "Miles Bridges", "ra", 6.5, None),
    ("Miles Bridges;Rebs+Asts;6.5", "Miles Bridges", "ra", 6.5, None),
    ("Luka Doncic, 26.5, points, under", "Luka Doncic", "pts", 26.5, "less"),
    ("Luka Doncic 26.5 points", "Luka Doncic", "pts", 26.5, None),
    ("Miles Bridges Rebounds + Assists (Over 6.5)", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges 6.5 R+A (Over)", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges O/U 6.5 R+A: Under", "Miles Bridges", "ra", 6.5, "less"),
    # Prediction-market wording: a Yes or No on the line's own side.
    ("Miles Bridges 7+ R+A (No)", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A Yes", "Miles Bridges", "ra", 6.5, "more"),
    ("Miles Bridges 7+ R+A (NO)", "Miles Bridges", "ra", 6.5, "less"),
    ("Will Miles Bridges record 7+ rebounds and assists? No", "Miles Bridges", "ra", 6.5,
     "less"),
    ("Will Riley over 10.5 points", "Will Riley", "pts", 10.5, "more"),
    ("Will Riley record 10+ points?", "Will Riley", "pts", 9.5, "more"),
    # A side or a note after the line, in Name Stat Line order (review 2, P5).
    ("Jalen Brunson Points 25.5 - Under", "Jalen Brunson", "pts", 25.5, "less"),
    ("Miles Bridges - Rebs+Asts - 6.5 - More", "Miles Bridges", "ra", 6.5, "more"),
    (f"Miles Bridges Rebs+Asts 6.5 {EM} OVER", "Miles Bridges", "ra", 6.5, "more"),
    ("Jalen Brunson Points O25.5 LOCK", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson Points O25.5 !!", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, LOCK", "Jalen Brunson", "pts", 25.5, "more"),
    # Glued, closing and column forms (review 2, P11).
    ("Jalen Brunson o25.5pts", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson 25.5PTS U", "Jalen Brunson", "pts", 25.5, "less"),
    ("Miles Bridges 4.5 R+A U", "Miles Bridges", "ra", 4.5, "less"),
    ("Miles Bridges 6.5+ Rebs+Asts", "Miles Bridges", "ra", 6.5, "more"),
    ("Give me Jalen Brunson points over 25.5.", "Jalen Brunson", "pts", 25.5, "more"),
    ("Miles Bridges\tRebs+Asts\t6.5\tOver\t1", "Miles Bridges", "ra", 6.5, "more"),
    ("1 Miles Bridges over 6.5 R+A", "Miles Bridges", "ra", 6.5, "more"),
    # A whole-number line keeps its number beside a stat with a digit in it.
    ("Stephen Curry, 3PT, 4", "Stephen Curry", "fg3m", 4.0, None),
    ("Miles Bridges, Rebs+Asts, 7", "Miles Bridges", "ra", 7.0, None),
    # A first name that is a tout word, and capitals throughout (review 2, P7, P8).
    ("Max Strus over 10.5 points", "Max Strus", "pts", 10.5, "more"),
    ("MAX PLAY: Max Christie over 2.5 threes", "Max Christie", "fg3m", 2.5, "more"),
    ("ZACH EDEY OVER 10.5 REBOUNDS", "ZACH EDEY", "reb", 10.5, "more"),
    ("JOSH HART OVER 8.5 REBOUNDS", "JOSH HART", "reb", 8.5, "more"),
    # A note is not a period (review 2, P13).
    ("Jalen Brunson over 25.5 points (2nd half of a back-to-back)", "Jalen Brunson", "pts",
     25.5, "more"),
    ("Jalen Brunson over 25.5 points (scored 30 in the first meeting)", "Jalen Brunson",
     "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, second half of a back to back", "Jalen Brunson",
     "pts", 25.5, "more"),
    # A parenthesis that changes the stat changes it (review 2, P2).
    ("Zach Edey over 4.5 rebounds (offensive)", "Zach Edey", "oreb", 4.5, "more"),
    ("Zach Edey over 4.5 rebounds (OREB)", "Zach Edey", "oreb", 4.5, "more"),
    ("Stephen Curry over 10.5 threes (attempted)", "Stephen Curry", "fg3a", 10.5, "more"),
    ("Stephen Curry over 10.5 3PT (attempts)", "Stephen Curry", "fg3a", 10.5, "more"),
    ("Luka Doncic over 45.5 points (fantasy)", "Luka Doncic", "fantasy", 45.5, "more"),
    ("Jalen Brunson over 12.5 points (+ assists)", "Jalen Brunson", "pa", 12.5, "more"),
    ("Stephen Curry over 4.5 threes (made)", "Stephen Curry", "fg3m", 4.5, "more"),
    ("Miles Bridges over 6.5 R+A (+EV)", "Miles Bridges", "ra", 6.5, "more"),
    # A combo written with commas, then a note: never its first stat (review 3, R1).
    ("Jalen Brunson over 32.5 points, rebounds and assists combined", "Jalen Brunson", "pra",
     32.5, "more"),
    ("Josh Hart over 14.5 points, rebounds combined", "Josh Hart", "pr", 14.5, "more"),
    ("Wendell Carter Jr. under 23.5 Points, Rebounds, and Assists for sure",
     "Wendell Carter Jr.", "pra", 23.5, "less"),
    ("Miles Bridges over 6.5 rebounds, assists combined", "Miles Bridges", "ra", 6.5, "more"),
    ("Josh Hart over 2.5 steals, blocks combined", "Josh Hart", "stocks", 2.5, "more"),
    # A parenthesis that changes the stat, however it is spelled; a gloss that
    # only mentions one does not (review 3, F2, R10).
    ("Stephen Curry over 10.5 threes (3PA)", "Stephen Curry", "fg3a", 10.5, "more"),
    ("Zach Edey over 4.5 rebounds (off.)", "Zach Edey", "oreb", 4.5, "more"),
    ("Zach Edey over 4.5 rebounds (O-reb)", "Zach Edey", "oreb", 4.5, "more"),
    ("Jalen Brunson over 32.5 points (+ rebounds + assists)", "Jalen Brunson", "pra", 32.5,
     "more"),
    ("Jalen Brunson over 12.5 points (and assists)", "Jalen Brunson", "pa", 12.5, "more"),
    ("Jalen Brunson over 25.5 points (weak defensive team)", "Jalen Brunson", "pts", 25.5,
     "more"),
    ("Miles Bridges over 6.5 R+A (Rebounds + Assists)", "Miles Bridges", "ra", 6.5, "more"),
    # A qualifier with "only" or "just" still changes the stat; "off" and "def"
    # among other words are "off the bench", "def a lock"; "plus" and "with"
    # add a stat as "and" does (review 4, A2, A10).
    ("Josh Hart over 2.5 rebounds (offensive only)", "Josh Hart", "oreb", 2.5, "more"),
    ("Stephen Curry over 10.5 threes (attempts only)", "Stephen Curry", "fg3a", 10.5, "more"),
    ("Luka Doncic over 45.5 points (fantasy only)", "Luka Doncic", "fantasy", 45.5, "more"),
    ("Jalen Brunson over 25.5 points (off the bench)", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 32.5 points (plus assists)", "Jalen Brunson", "pa", 32.5, "more"),
    ("Jalen Brunson over 32.5 points (with assists)", "Jalen Brunson", "pa", 32.5, "more"),
    # A Yes or No inside a note is no answer: "no doubt" flipped the pick
    # (review 4, A1); an answer with a full stop is still one (A3).
    ("Jalen Brunson over 25.5 points no doubt", "Jalen Brunson", "pts", 25.5, "more"),
    ("Josh Hart under 8.5 rebounds no doubt", "Josh Hart", "reb", 8.5, "less"),
    ("Jalen Brunson over 25.5 points - no brainer", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points with no Towns", "Jalen Brunson", "pts", 25.5, "more"),
    ("JALEN BRUNSON OVER 25.5 POINTS NO DOUBT", "JALEN BRUNSON", "pts", 25.5, "more"),
    ("Miles Bridges over 6.5 R+A, yes sir", "Miles Bridges", "ra", 6.5, "more"),
    ("Will Jalen Brunson score 26+ points? No.", "Jalen Brunson", "pts", 25.5, "less"),
    ("Miles Bridges 7+ R+A No.", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A: No LOCK", "Miles Bridges", "ra", 6.5, "less"),
    # A qualifier, a note word or a result word later in a note is context
    # (review 4, B8, A7).
    ("Jalen Brunson over 25.5 points, coming off a big game", "Jalen Brunson", "pts", 25.5,
     "more"),
    ("Josh Hart over 8.5 rebounds, below average rebounding opponent", "Josh Hart", "reb", 8.5,
     "more"),
    ("Jalen Brunson over 25.5 points, bad def tonight", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points - Knicks lost last night", "Jalen Brunson", "pts", 25.5,
     "more"),
    ("Josh Hart over 8.5 rebounds, he missed the last game", "Josh Hart", "reb", 8.5, "more"),
    # A handle that opens with an underscore (review 4, B10), and a line with
    # no side closed by "tonight" after a line's .5 (B9).
    ("@_hoopspicks: Jalen Brunson over 25.5 points", "Jalen Brunson", "pts", 25.5, "more"),
    ("@__hoops__: Jalen Brunson over 25.5 points", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson 25.5 points tonight", "Jalen Brunson", "pts", 25.5, None),
    ("Jalen Brunson, Points, 25.5, LOCK", "Jalen Brunson", "pts", 25.5, None),
    # A note in either order, and a side in a parenthesis (review 3, F6, F8).
    ("Jalen Brunson PRA O35.5 - Max play", "Jalen Brunson", "pra", 35.5, "more"),
    ("Jalen Brunson o25.5 Points LOCK", "Jalen Brunson", "pts", 25.5, "more"),
    ("Points O25.5 Jalen Brunson LOCK", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson o25.5 Points lets go", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson o6.5 Assists LOCK OF THE DAY", "Jalen Brunson", "ast", 6.5, "more"),
    ("Jalen Brunson Points 25.5 (U)", "Jalen Brunson", "pts", 25.5, "less"),
    # Capitals, glued forms, list numbers and keycaps (review 3, F7, F9, F10, R11, F18).
    ("JALEN BRUNSON'S POINTS OVER 25.5", "JALEN BRUNSON", "pts", 25.5, "more"),
    ("Jalen Brunson o25.5pts.", "Jalen Brunson", "pts", 25.5, "more"),
    ("Josh Hart 9+reb", "Josh Hart", "reb", 8.5, "more"),
    ("Karl-Anthony Towns 26+pts", "Karl-Anthony Towns", "pts", 25.5, "more"),
    ("1 Amen Thompson over 10.5 points", "Amen Thompson", "pts", 10.5, "more"),
    (f"{KEYCAP_4} miles bridges over 6.5 r+a", "miles bridges", "ra", 6.5, "more"),
    (f"{KEYCAP_3} Luka Doncic, Points, 27", "Luka Doncic", "pts", 27.0, None),
    ("3PM over 4.5 Stephen Curry", "Stephen Curry", "fg3m", 4.5, "more"),
    ("3pm Jalen Brunson o25.5 pts", "Jalen Brunson", "pts", 25.5, "more"),
    # Notes that are not a period (review 3, F12).
    ("Jalen Brunson over 25.5 points (2nd half of B2B)", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points - 2nd half of back to back", "Jalen Brunson", "pts", 25.5,
     "more"),
    ("Jalen Brunson over 25.5 points (30+ in the first 5 games)", "Jalen Brunson", "pts", 25.5,
     "more"),
    ("Jalen Brunson over 25.5 points (in first place)", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (in the first full game back)", "Jalen Brunson", "pts",
     25.5, "more"),
    ("Jalen Brunson over 25.5 points (in the first round of the Cup)", "Jalen Brunson", "pts",
     25.5, "more"),
    ("Jalen Brunson over 25.5 points (2nd half of the season he averaged 28)",
     "Jalen Brunson", "pts", 25.5, "more"),
    # Review 5: a lone letter after a dash or a comma is no Assists (B2); an
    # addition written "w/", "including" or "included" joins the stat (A7); a
    # comma run of three with an everyday closer is still PRA (B11); a note
    # that only mentions another stat is a gloss.
    ("Jalen Brunson over 25.5 points - a lock", "Jalen Brunson", "pts", 25.5, "more"),
    ("Josh Hart over 8.5 rebounds - a lock for sure", "Josh Hart", "reb", 8.5, "more"),
    ("Jalen Brunson over 25.5 points, a lock", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 32.5 points (w/ assists)", "Jalen Brunson", "pa", 32.5, "more"),
    ("Jalen Brunson over 43.5 points (including rebounds and assists)", "Jalen Brunson", "pra",
     43.5, "more"),
    ("Jalen Brunson over 32.5 points (assists included)", "Jalen Brunson", "pa", 32.5, "more"),
    ("Jalen Brunson over 35.5 points, rebounds and assists all day", "Jalen Brunson", "pra",
     35.5, "more"),
    ("Jalen Brunson over 25.5 points (no assists line today)", "Jalen Brunson", "pts", 25.5,
     "more"),
    # Review 5, A2, A5, A12, B3: a market's No set off on its own is the
    # answer wherever it sits; one that starts a note, or answers the note's
    # own question, is not.
    ("Miles Bridges 7+ R+A No - fade", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A No for sure", "Miles Bridges", "ra", 6.5, "less"),
    # (Review 6, B6: beside a written Over, "No, he's been cold" may be the
    # answer or a note on the Over, so it is counted unclear: see below.)
    ("Mikal Bridges 16+ points No, he's been cold", "Mikal Bridges", "pts", 15.5, "less"),
    ("Jalen Brunson 26+ points NO at 38c", "Jalen Brunson", "pts", 25.5, "less"),
    ("Miles Bridges No 7+ R+A", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges: No on 7+ R+A", "Miles Bridges", "ra", 6.5, "less"),
    ("Will Miles Bridges get 7+ R+A? I say no, fade", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (No.)", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (NO.)", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (No, fade)", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A: Yes or No? No", "Miles Bridges", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A No. Yes on Hart though", "Miles Bridges", "ra", 6.5, "less"),
    ("Will Anthony Edwards score 28+ points vs NO? No", "Anthony Edwards", "pts", 27.5, "less"),
    ("Jalen Brunson over 25.5 points? No doubt", "Jalen Brunson", "pts", 25.5, "more"),
    ("Josh Hart under 8.5 rebounds? No doubt", "Josh Hart", "reb", 8.5, "less"),
    ("Josh Hart over 8.5 rebounds - is Towns playing? No.", "Josh Hart", "reb", 8.5, "more"),
    ("Jalen Brunson over 25.5 points. Fade? No.", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, no defense on the other side", "Jalen Brunson", "pts",
     25.5, "more"),
    ("NO DOUBT Jalen Brunson over 25.5 points", "Jalen Brunson", "pts", 25.5, "more"),
    ("Miles Bridges 7+ R+A No - Brunson cold", "Miles Bridges", "ra", 6.5, "less"),
    ("JALEN BRUNSON OVER 25.5 POINTS, KNICKS HAVE NO DEFENSE", "JALEN BRUNSON", "pts", 25.5,
     "more"),
    ("Jalen Brunson over 25.5 points, this is no lock", "Jalen Brunson", "pts", 25.5, "more"),
    ("Jalen Brunson over 32.5 points (combined with assists)", "Jalen Brunson", "pa", 32.5,
     "more"),
])
def test_a_pick_reads_however_it_is_written(text, name, stat, line, pick):
    r = _one(text)
    assert (r["name"], r["stat"], r["line"], r["pick"]) == (name, stat, line, pick)


@pytest.mark.parametrize("text", [
    f"{FIRE} Miles Bridges OVER 6.5 Reb+Ast (-120) 2u",
    f"{CHECK} Miles Bridges o6.5 R+A -115",
    f"{MONEY} Miles Bridges O6.5 R+A {FIRE}{FIRE}",
    "POTD: Miles Bridges over 6.5 R+A",
    "Leg 1: Miles Bridges over 6.5 R+A",
    "Pick #2 - Miles Bridges over 6.5 R+A",
    "Miles Bridges o6.5 R+A +105",
    "Miles Bridges over 6.5 R+A (2u)",
    "Miles Bridges over 6.5 R+A 2 units",
    "Best Bet: Miles Bridges over 6.5 R+A",
    # Review 2, P6 and P12: keycap numbers, league tags, thread numbers,
    # markdown, a handle, a date, a tip time, a multiplier, mid-line bullets.
    f"{KEYCAP_1} Miles Bridges over 6.5 R+A",
    "NBA: Miles Bridges over 6.5 R+A", "[NBA] Miles Bridges over 6.5 R+A",
    "NBA PROP: Miles Bridges over 6.5 R+A", "NBA Miles Bridges over 6.5 R+A",
    "1/ Miles Bridges over 6.5 R+A", "2/5 Miles Bridges over 6.5 R+A",
    "**Miles Bridges over 6.5 R+A**", "__Miles Bridges__ over 6.5 R+A",
    "`Miles Bridges over 6.5 R+A`", "@hoopspicks: Miles Bridges over 6.5 R+A",
    "11/5 Miles Bridges over 6.5 R+A", "7:30pm Miles Bridges over 6.5 R+A",
    "2x Miles Bridges over 6.5 R+A", f"Miles Bridges {BUL} Rebs+Asts {BUL} 6.5 {BUL} More",
    "Miles Bridges / Rebs+Asts / 6.5 / More",
    # Review 3, F7 and F11: a two-keycap number and a trailing keycap, a
    # hashtag, a handle with no colon, a bracketed thread number, a dated
    # prefix, "2x:", "x2", single underscores, arrows, a 3 PM tip, and
    # bullets with no spaces.
    f"{KEYCAP_1}{KEYCAP_0} Miles Bridges over 6.5 R+A", f"Miles Bridges over 6.5 R+A {KEYCAP_2}",
    "#NBA Miles Bridges over 6.5 R+A", "@hoopspicks Miles Bridges over 6.5 R+A",
    "@hoopspicks - Miles Bridges over 6.5 R+A", "[1/3] Miles Bridges over 6.5 R+A",
    "11/05/26 Miles Bridges over 6.5 R+A", "Nov 5: Miles Bridges over 6.5 R+A",
    "2x: Miles Bridges over 6.5 R+A", "x2 Miles Bridges over 6.5 R+A",
    "_Miles Bridges_ over 6.5 R+A", f"Miles Bridges {ARROW} over 6.5 R+A",
    "Miles Bridges >> over 6.5 R+A", "3pm Miles Bridges over 6.5 R+A",
    f"Miles Bridges{BUL}Rebs+Asts{BUL}6.5{BUL}More",
    # Review 5, A21: a name written as an @-mention, a source's name and a
    # colon, and a markdown heading mark.
    "@Miles Bridges over 6.5 R+A", "hoopspicks: Miles Bridges over 6.5 R+A",
    "## Miles Bridges over 6.5 R+A",
])
def test_tout_decoration_is_set_aside(text):
    r = _one(text)
    assert (r["name"], r["stat"], r["line"], r["pick"], r["odds_type"]) == \
        ("Miles Bridges", "ra", 6.5, "more", "standard")


def test_teams_odds_and_opponents_on_the_line():
    df = props.parse_typed_lines(
        "Miles Bridges | Rebs+Asts | 6.5 | demon\n"
        "Paolo Banchero over 3.5 blocks (Goblin)\n"
        "Luka Doncic (LAL) under 27.5 points\n"
        "Luka Doncic over 11.5 rebounds vs Blazers tonight\n"
        "PHX Miles Bridges over 6.5 R+A\n"
        "Suns' Oso Ighodaro over 7.5 P+R\n"
        "Michael Porter Jr. @ATL o20.5 pts\n"
        "Michael Porter Jr. over 20.5 points (@ATL)\n")
    assert list(df["odds_type"]) == ["demon", "goblin"] + ["standard"] * 6
    assert list(df["team"])[2:] == ["LAL", None, "PHX", "PHX", None, None]
    assert list(df["opp"])[3:] == ["POR", None, None, "ATL", "ATL"]
    assert list(df["name"])[4:] == ["Miles Bridges", "Oso Ighodaro", "Michael Porter Jr.",
                                    "Michael Porter Jr."]


def test_an_at_sign_names_the_opponent_never_his_team(roster, schedule):
    """'@ATL' glued is tout shorthand for the opponent. Read as his team it
    filed Porter under Hawks v Wizards, a day early (review P5)."""
    morning = pd.Timestamp("2026-11-06T16:00:00Z").to_pydatetime()
    lines, _ = _prep("Michael Porter Jr. @ATL o20.5 pts\n", roster, schedule, now=morning)
    r = lines.iloc[0]
    assert (r["team_code"], r["opp"], r["game_id"]) == (None, "ATL", "0022600190")


@pytest.mark.parametrize("text", [
    "Last night we went 2 of 3",
    "Suns -3.5",
    "Phoenix Suns over 112.5 points",          # a team total, not a player
    "Over 220.5 total points",                 # a game total
    "We like Bridges to go over/no 6.5",       # no side to read
    "Miles Bridges over 6.5 R+A and Oso Ighodaro over 7.5 P+R",   # two lines in one
    "Bridges has gone over 6.5 R+A in 8 of his last 10",
    "Miles Bridges over 6.5 Rebounds and Assists under 7.5",
    "Miles Bridges over -6.5 R+A",             # a minus number is a spread
    "Miles Bridges 7+ R+A Yes No",             # two answers
    "Miles Bridges 7+ R+A for sure Yes No",    # ...after a remark (review 5)
    "Miles Bridges 6.5 R+A No",                # a No with no side to turn
    "Suns:", "Magic @ Suns", "",
    # A result or an average in a copied post (review 3, R7).
    f"{CHECK} Josh Hart 12 rebounds, cashed", "OG Anunoby 3 threes, easy money",
    "Jalen Brunson Points 27.1 avg", "Josh Hart Rebounds 9.8 L5",
    "Jalen Brunson 40 points, a season high", "Jalen Brunson 27 pts cash",
    f"{CHECK} Jalen Brunson o26.5 pts, cashed",
    # A game total written in codes (review 3, F18).
    "LAL BOS over 220.5 points", "PHX ORL u225.5 points",
])
def test_lines_that_are_not_a_players_pick_add_nothing(text):
    assert props.parse_typed_lines(text).empty


def test_a_parenthesis_that_changes_the_stat_never_prices_the_stat_without_it():
    """'threes (attempted)' was priced as threes made, against the made pmf
    (review 2, P2). A word naming no stat priced here leaves the line
    unrecognised; a gloss with the line's own number in it stays a gloss."""
    combo = _one("Jalen Brunson over 12.5 points (combo)")
    assert combo["stat"] is None and props.classify_stat(combo["stat_type"])[1] == "unmapped"
    assert _one(FIRST_LIST.splitlines()[2])["stat"] == "ra"
    # A note or a parenthesis naming another stat leaves the line
    # unrecognised: priced on its first stat it was a confident lean on the
    # wrong pmf (review 3, R1, F2).
    for text in ("Jalen Brunson over 25.5 points in the paint",
                 "Jalen Brunson over 25.5 points (rebounds)",
                 "Jalen Brunson over 32.5 points, rebounds and blocks and assists",
                 # A qualifier among other words still changes the stat: priced
                 # as plain Rebounds it led the discrepancies (review 4, A2).
                 "Josh Hart over 8.5 rebounds (offensive glass)",
                 "Josh Hart over 2.5 rebounds (offensive boards are his thing)",
                 "Stephen Curry over 10.5 threes (attempted, not made)",
                 # A qualifier right after the stat, with no parenthesis (A9).
                 "Josh Hart over 2.5 rebounds offensive",
                 "Jalen Duren over 4.5 rebounds offensive only",
                 # A stat run across a comma with a note after it: R+A, or
                 # Rebounds and a note? Neither is read (review 4, B1).
                 "Josh Hart over 8.5 rebounds, assists are a sneaky play too",
                 "Jalen Brunson over 25.5 points, rebounds and assists are traps",
                 "Jalen Brunson over 6.5 assists, points are the safer play",
                 "Josh Hart over 8.5 rebounds; assists too",
                 "Josh Hart over 8.5 rebounds - assists too",
                 "Josh Hart o8.5 reb, pts too",
                 # Review 5: two stats with only a comma between them and no
                 # word saying combo (A9); a qualifier after a filler word
                 # (A4, B5); "off" beside another word (A7); another stat named
                 # as if added; a note word first ("per game", A16); and an en
                 # dash before the remark (A17).
                 "Jalen Brunson over 6.5 assists, rebounds no",
                 "Josh Hart over 8.5 rebounds, assists I like",
                 "Josh Hart over 2.5 rebounds, only offensive",
                 "Stephen Curry over 9.5 threes, counting attempts",
                 "Jalen Brunson over 12.5 points, only in the paint",
                 "Josh Hart over 2.5 rebounds (off glass)",
                 "Josh Hart over 2.5 rebounds (off, not total)",
                 "Jalen Brunson over 12.5 points (assists too)",
                 "Jalen Brunson over 12.5 points (in the paint)",
                 "Jalen Brunson over 25.5 points per game",
                 "Jalen Brunson over 12.5 points, lots of paint touches",
                 f"Josh Hart over 8.5 rebounds {EN} assists too",
                 # Review 6, A7, A15: the word that changes the stat after
                 # another filler word or spelled another way, and an article
                 # "a" after "+" (B2).
                 "Karl-Anthony Towns over 2.5 rebounds, exclusively offensive",
                 "Josh Hart over 1.5 rebounds, but only offensive",
                 "Jarrett Allen over 3.5 rebounds, all offensive",
                 "Josh Hart over 2.5 rebounds, strictly offensive",
                 "Josh Hart over 2.5 rebounds, on the offensive glass",
                 "Josh Hart over 2.5 rebounds, offense only",
                 "Josh Hart over 2.5 rebounds (O boards)",
                 "Josh Hart over 2.5 rebounds (o-boards)",
                 "Josh Hart over 2.5 rebounds (offense)",
                 "Mikal Bridges over 7.5 points, all from the paint",
                 "Jalen Brunson over 12.5 points, mostly in the paint",
                 "Jalen Brunson over 12.5 points, in transition",
                 "Jalen Brunson over 12.5 points, on fastbreaks",
                 "Stephen Curry over 9.5 threes, as attempts",
                 "Stephen Curry over 9.5 threes, his attempts",
                 "Stephen Curry over 9.5 threes (shots)",
                 "Jalen Brunson over 25.5 points + a lock",
                 "Josh Hart over 8.5 reb + a lock",
                 # Lower-case letters joined by "and", and a run of four.
                 "Josh Hart over 10.5 r and a",
                 "Jalen Brunson over 40.5 Pts + R + A + S"):
        r = _one(text)
        assert r["stat"] is None and props.classify_stat(r["stat_type"])[1] == "unmapped", text
    # ...and the words that only gloss it still do (review 4, B8; review 6).
    for text in ("Jalen Brunson over 25.5 points, coming off a big game",
                 "Jalen Brunson over 25.5 points, bad def tonight",
                 "Jalen Brunson over 25.5 points, no defense on the other side",
                 "Jalen Brunson over 25.5 points (vs bad defense)",
                 "Josh Hart over 8.5 rebounds (including offensive)",
                 "Jalen Brunson over 25.5 points - a lock"):
        assert _one(text)["stat"] in ("pts", "reb"), text


# ── ESPN's two-letter codes ──────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Zion Williamson (NO) over 22.5 points", "Zion Williamson NO over 22.5 points",
    "NO Zion Williamson over 22.5 points", "Zion Williamson - NO - Over 22.5 Points",
    "Zion Williamson over 22.5 points (NO)", "Zion Williamson over 22.5 points NO",
    "Zion Williamson over 22.5 points NO.", "Zion Williamson over 22.5 points NO LOCK",
])
def test_espns_no_is_new_orleans_for_a_pelican(text, roster, schedule):
    """ESPN writes New Orleans 'NO'. Read as a market No, every Pelicans Over
    written that way was tagged Less beside a More lean (review 2, P1)."""
    r = _one(text)
    assert (r["name"], r["team"], r["pick"], r["alt_pick"]) == \
        ("Zion Williamson", "NOP", "more", "less")
    lines, _ = _prep(text, roster, schedule, now=FRIDAY_11AM)
    assert (lines.iloc[0]["pick"], lines.iloc[0]["game_id"]) == ("more", "0022600031")


def test_a_no_before_anyone_elses_name_is_the_markets_no(roster, schedule):
    """Miles Bridges is no Pelican, so his 'NO' is the market's No (review
    P6). After a '7+' a NO is the answer; after a side word it is read both
    ways, like a NO beside the name (review 3, F1)."""
    assert (_one("NO Miles Bridges 7+ R+A")["pick"], _one("NO Miles Bridges 7+ R+A")["alt_pick"]) \
        == ("more", "less")
    lines, _ = _prep("NO Miles Bridges 7+ R+A", roster, schedule)
    r = lines.iloc[0]
    assert (r["pick"], r["team_code"], r["game_id"]) == ("less", "PHX", MAGIC_AT_SUNS)
    assert _one("Miles Bridges 7+ R+A (NO)")["alt_pick"] is None
    lines, _ = _prep("Jalen Brunson over 25.5 points NO", roster, schedule, now=FRIDAY_11AM)
    assert (lines.iloc[0]["pick"], lines.iloc[0]["team_code"]) == ("less", "NYK")
    # A question answered NO is the No for a Pelican too, bare or in a
    # parenthesis: only a title-case "No" or an implied side was tested
    # (review 4, A13).
    for text in ("Will Zion Williamson go over 22.5 points? NO",
                 "Will Zion Williamson go over 22.5 points? (NO)",
                 "Will Trey Murphy III hit over 2.5 threes? NO",
                 # With more words after it, a NO is the market's for a
                 # Pelican too: "- fade", a price (review 5, B3).
                 "Zion Williamson over 22.5 points NO - fade",
                 "Zion Williamson over 22.5 points NO at 38c",
                 "Trey Murphy III 3+ threes NO at 38c"):
        lines, _ = _prep(text, roster, schedule, now=FRIDAY_11AM)
        r = lines.iloc[0]
        assert (r["pick"], r["roster_team"], r["game_id"]) == ("less", "NOP", "0022600031"), text


def test_a_no_on_a_line_that_names_nobody_is_the_markets_no(roster, schedule):
    """'NO 7+ assists' under a player's name was read as New Orleans, so the
    line, and every line after it, was lost without a word (review 3, R4)."""
    df = props.parse_typed_lines("Jalen Brunson\nYES 26+ points\nNO 7+ assists\nYES 4+ rebounds")
    assert list(zip(df["name"], df["stat_type"], df["line"], df["pick"])) == [
        ("Jalen Brunson", "Points", 25.5, "more"), ("Jalen Brunson", "Assists", 6.5, "less"),
        ("Jalen Brunson", "Rebounds", 3.5, "more")]
    assert df["alt_pick"].isna().all()
    assert list(props.parse_typed_lines("Josh Hart:\nNO 9+ rebounds\nYES 5+ assists")["pick"]) \
        == ["less", "more"]
    lines, _ = _prep("NO over 25.5 points for Jalen Brunson\n"
                     "Jalen Brunson NYK NO over 6.5 assists\n", roster, schedule, now=FRIDAY_11AM)
    assert list(zip(lines["pick"], lines["team_code"], lines["game_id"])) == \
        [("less", "NYK", "0022600191")] * 2


@pytest.mark.parametrize("text,name,team", [
    ("Stephen Curry GS over 4.5 threes", "Stephen Curry", "GSW"),
    ("Jalen Brunson NY over 25.5 points", "Jalen Brunson", "NYK"),
    ("Victor Wembanyama SA over 10.5 rebounds", "Victor Wembanyama", "SAS"),
])
def test_espns_two_letter_codes_are_teams_beside_a_name(text, name, team):
    """Kept in the name, they matched nobody (review 2, P1)."""
    r = _one(text)
    assert (r["name"], r["team"]) == (name, team)


def test_a_pick_without_a_full_name_is_counted_not_guessed():
    df = props.parse_typed_lines("Bridges over 6.5 R+A\nLast night we hit over 6.5 R+A")
    assert df.empty and df.attrs["typed_noname"] == 2
    msgs = props_ui.report_messages({"nonempty": True, "kept": 0, "parse": df.attrs})
    assert msgs[0][1].startswith("2 lines in that paste read like picks but name no player "
                                 "in full, so nothing was added.")
    one = props.parse_typed_lines("Bridges over 6.5 R+A")
    msgs = props_ui.report_messages({"nonempty": True, "kept": 0, "parse": one.attrs})
    assert msgs[0][1].startswith("1 line in that paste reads like a pick but names no player")
    mixed = props_ui.report_messages({"nonempty": True, "kept": 1, "comparable": 1,
                                      "parse": one.attrs})
    assert ("info", "1 line reads like a pick but names no player in full (first and last "
                    "name), so it was skipped.") in mixed


def test_a_pick_with_more_than_one_number_is_counted_not_guessed():
    """A note with numbers after a pick, or two picks on one line, was
    dropped without a word (review 2, P11)."""
    text = ("Jalen Brunson o25.5 pts - hit in 8 of last 10\n"
            f"Jalen Brunson o25.5 pts 5{STAR}\n"
            "Miles Bridges over 6.5 R+A and Oso Ighodaro over 7.5 P+R\n")
    df = props.parse_typed_lines(text)
    assert df.empty and df.attrs["typed_unread"] == 3 and df.attrs["typed_noname"] == 0
    msgs = props_ui.report_messages({"nonempty": True, "kept": 0, "parse": df.attrs})
    assert msgs[0][1].startswith("3 lines in that paste look like picks but hold more than "
                                 "one number, so nothing was added.")
    assert props.parse_typed_lines("Last night we went 2 of 3").attrs["typed_unread"] == 0
    mixed = props.parse_typed_lines(text + "Miles Bridges over 6.5 R+A\n")
    msgs = props_ui.report_messages({"nonempty": True, "kept": 1, "comparable": 1,
                                     "parse": mixed.attrs})
    assert ("info", "3 lines look like picks but hold more than one number, so they were "
                    "skipped. Write one pick per line with its line as the only number.") in msgs


# ── A period prop is refused wherever the period is written ──────────────────
@pytest.mark.parametrize("text", [
    "Miles Bridges over 12.5 points (1st half)",
    "Miles Bridges over 12.5 points (1H)",
    "Miles Bridges over 3.5 Q1 points",
    "Miles Bridges over 2.5 R+A (1Q)",
    "Miles Bridges over 3.5 points (first 5 minutes)",
    "1H Miles Bridges over 12.5 points",
    "Miles Bridges over 12.5 points 1st half",
    "1Q: Miles Bridges over 2.5 points", "Q1 Miles Bridges over 6.5 points",
    "Miles Bridges 1H over 12.5 points", "Miles Bridges over 25.5 points - huge Q4 usage",
])
def test_a_half_or_quarter_line_is_never_priced_as_a_full_game(text, roster, schedule):
    """A parenthesised '(1H)' was read as a gloss and the half line priced
    against the full game: a confident More on nothing (review P1). A marker
    before the name is the same period prop, never a name problem (review 2,
    P13)."""
    lines, report = props.prepare_paste("Suns:\n" + text, roster, schedule, now=BEFORE_TIP)
    assert len(lines) == 1 and report["comparable"] == 0
    assert report["not_comparable"]["not priced"] == 1
    assert set(lines["refusal"]) == {props.REFUSE_PERIOD}
    assert props.classify_stat("Q1 Points")[1] == "refused"


# ── A name and the picks under it ────────────────────────────────────────────
def test_a_name_alone_names_every_pick_under_it():
    df = props.parse_typed_lines(
        "Paolo Banchero\nOver 22.5 Points\nOver 8.5 Rebounds\nUnder 4.5 Assists\n"
        "Miles Bridges\n- over 6.5 R+A\n")
    assert list(zip(df["name"], df["stat_type"], df["pick"])) == [
        ("Paolo Banchero", "Points", "more"), ("Paolo Banchero", "Rebounds", "more"),
        ("Paolo Banchero", "Assists", "less"), ("Miles Bridges", "Rebs+Asts", "more")]


def test_a_name_line_with_its_matchup_names_the_picks_under_it():
    """'Jalen Brunson vs Heat' above his picks lost every one (review 2, P11)."""
    df = props.parse_typed_lines("Jalen Brunson vs Heat\nOver 25.5 points\nOver 6.5 assists\n"
                                 "Jalen Brunson - NYK @ MIA\nOver 2.5 threes\n"
                                 "Miles Bridges props:\nOver 6.5 R+A\n")
    assert list(zip(df["name"], df["team"], df["opp"])) == [
        ("Jalen Brunson", None, "MIA"), ("Jalen Brunson", None, "MIA"),
        ("Jalen Brunson", "NYK", "MIA"), ("Miles Bridges", None, None)]


def test_a_name_is_not_lent_past_other_words_or_to_a_total():
    df = props.parse_typed_lines(
        "Miles Bridges\nover 6.5 rebounds and assists\nsome words about the matchup\n"
        "over 7.5 P+R\n"
        "Oso Ighodaro\nover 7.5 P+R\nOver 220.5 total points\nOver 112.5 points\n")
    assert list(df["name"]) == ["Miles Bridges", "Oso Ighodaro"]


def test_a_copied_post_with_stray_number_lines_is_still_read():
    """A tweet's '12 Reposts' gave the board parser a junk square and the
    typed reader never ran (review P7)."""
    post = ("Hoops Picks Daily\n@hoopspicks\nNBA picks tonight\nMiles Bridges over 6.5 R+A\n"
            "Oso Ighodaro over 7.5 P+R\nWendell Carter Jr. under 23.5 PRA\n"
            "9:41 PM - Nov 5, 2026\n12\nReposts\n45\nLikes\n")
    df = props.parse_any(post)
    assert set(df["source"]) == {"typed"}
    assert list(df["name"]) == ["Miles Bridges", "Oso Ighodaro", "Wendell Carter Jr."]
    numbered = props.parse_any("Best Bets Tonight\n1\nMiles Bridges over 6.5 R+A\n2\n"
                               "Oso Ighodaro over 7.5 P+R\n3\n")
    assert list(numbered["name"]) == ["Miles Bridges", "Oso Ighodaro"]


# ── Headings group a list; the schedule names a game ─────────────────────────
def test_a_section_label_ends_the_team_heading_above_it(roster, schedule):
    """'Bonus play:' after 'Magic:' put Curry under Magic at Suns (review P3).
    The roster now files him either way, so the heading itself is checked
    too: it scopes the match of a name the roster cannot place (review 2, I8)."""
    text = ("Suns:\nMiles Bridges over 6.5 Rebounds and Assists\nMagic:\n"
            "Wendell Carter Jr. under 23.5 PRA\n\nBonus play:\nStephen Curry over 4.5 threes\n")
    assert list(props.parse_typed_lines(text)["team"]) == ["PHX", "ORL", None]
    lines, report = _prep(text, roster, schedule)
    assert lines.loc["Stephen Curry", "game_id"] == HEAT_AT_WARRIORS
    assert lines.loc["Miles Bridges", "game_id"] == MAGIC_AT_SUNS
    assert report["comparable"] == 3


def test_the_roster_team_wins_over_a_heading(roster, schedule):
    lines, _ = _prep("Suns:\nMiles Bridges over 6.5 R+A\nStephen Curry over 4.5 threes\n",
                     roster, schedule)
    curry = lines.loc["Stephen Curry"]
    assert (curry["team_code"], curry["game_id"], curry["how"]) == \
        ("GSW", HEAT_AT_WARRIORS, "exact")             # never "another team"


def test_a_short_first_name_under_another_teams_heading_still_matches(roster, schedule):
    """'Steph Curry' under 'Suns:' was unmatched: the prefix match looked only
    on the heading's team (review 2, P14)."""
    lines, _ = _prep("Suns:\nMiles Bridges over 6.5 R+A\nSteph Curry over 4.5 threes\n",
                     roster, schedule)
    curry = lines.loc["Steph Curry"]
    assert (curry["person_id"], curry["team_code"], curry["game_id"]) == \
        (201939, "GSW", HEAT_AT_WARRIORS)


def test_two_headings_from_two_games_route_each_on_its_own(roster, schedule):
    """Read as one game, every line was lost (review P4, I1)."""
    lines, report = _prep("Suns:\nMiles Bridges over 6.5 R+A\nWarriors:\n"
                          "Stephen Curry over 4.5 threes\n", roster, schedule)
    assert list(lines["game_id"]) == [MAGIC_AT_SUNS, HEAT_AT_WARRIORS]
    assert report["comparable"] == 2


def test_a_game_two_headings_played_yesterday_does_not_make_them_late(roster, schedule):
    """PHX beat POR on Nov 4. A Suns and Blazers list on Nov 5 is for each
    team's next game, not a late list for that one (review P4)."""
    lines, report = _prep("Suns:\nMiles Bridges over 6.5 R+A\nBlazers:\n"
                          "Jrue Holiday over 5.5 assists\n", roster, schedule)
    assert list(lines["game_id"]) == [MAGIC_AT_SUNS, BLAZERS_AT_LAKERS]
    assert report["tipped"] == 0 and report["comparable"] == 2


def test_a_third_heading_does_not_bring_back_the_misfiling(roster, schedule):
    """Three headings once named no game, and the late list was filed under
    each team's next game again (review I2)."""
    text = FIRST_LIST + "\nLakers:\nLuka Doncic over 9.5 Assists\n"
    lines, report = _prep(text, roster, schedule, now=AFTER_GAME)
    late = ["Miles Bridges", "Oso Ighodaro", "Wendell Carter Jr."]
    assert set(lines.loc[late, "route_how"]) == {"tipped"} and report["tipped"] == 3
    assert lines.loc["Luka Doncic", "game_id"] == BLAZERS_AT_LAKERS


def test_two_headings_that_meet_later_do_not_move_tonights_list(roster, schedule):
    """Thunder and Wizards meet on Saturday and each plays someone else on
    Friday: a Friday-morning list is Friday's (review 2, P3)."""
    lines, _ = _prep("Thunder:\nShai Gilgeous-Alexander over 30.5 points\nWizards:\n"
                     "Anthony Davis over 11.5 rebounds\n", roster, schedule, now=FRIDAY_11AM)
    assert list(lines["game_id"]) == ["0022600029", "0022600025"]


def test_a_list_for_tomorrow_is_not_late_because_two_of_its_teams_played_tonight(
        roster, schedule):
    """Hawks and Wizards tipped at 7 PM. At 11:30 PM a list with Nets and
    Thunder too pairs off into Saturday's two games; half of it was called
    late (review 2, P9). The two headings alone are that game's, sent late."""
    text = ("Hawks:\nDyson Daniels over 5.5 assists\nNets:\nNoah Clowney over 6.5 rebounds\n"
            "Wizards:\nAnthony Davis over 11.5 rebounds\nThunder:\n"
            "Shai Gilgeous-Alexander over 30.5 points\n")
    lines, report = _prep(text, roster, schedule, now=FRIDAY_LATE)
    assert list(lines["game_id"]) == ["0022600190"] * 2 + ["0022600192"] * 2
    assert report["tipped"] == 0
    lines, report = _prep("Hawks:\nDyson Daniels over 5.5 assists\nWizards:\n"
                          "Anthony Davis over 11.5 rebounds\n", roster, schedule, now=FRIDAY_LATE)
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 2


def test_a_heading_pairs_only_on_the_next_game_of_both(roster, schedule):
    """7:30 PM Friday: Hawks at Wizards is on, and the Thunder play at 8. The
    Wizards' next game (Saturday) is against the Thunder, but the Thunder's
    is tonight's: that pairs nothing, so the list stays tonight's and the
    Hawks and Wizards lines are late, not Saturday's."""
    text = ("Hawks:\nDyson Daniels over 5.5 assists\nWizards:\nAnthony Davis over 11.5 rebounds\n"
            "Thunder:\nShai Gilgeous-Alexander over 30.5 points\n")
    lines, report = _prep(text, roster, schedule,
                          now=pd.Timestamp("2026-11-07T00:30:00Z").to_pydatetime())
    assert list(lines["route_how"]) == ["tipped", "tipped", "team_schedule"]
    assert lines.loc["Shai Gilgeous-Alexander", "game_id"] == "0022600029"
    assert report["tipped"] == 2


@pytest.mark.parametrize("when", ["2026-11-06T08:01:00Z", "2026-11-06T15:00:00Z",
                                  "2026-11-06T16:59:00Z"])
def test_the_late_list_is_still_late_the_next_morning(roster, schedule, when):
    """Six hours after tip the two headings named no game, and the list was
    filed under Saturday's games (review 2, I1). Neither team plays on
    Friday, so until noon the game they last played is the list's."""
    lines, report = _prep(FIRST_LIST, roster, schedule, now=pd.Timestamp(when).to_pydatetime())
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 3


@pytest.mark.parametrize("when", ["2026-11-06T17:00:00Z", "2026-11-07T03:59:00Z"])
def test_from_noon_the_next_day_two_headings_are_a_list_for_their_next_games(
        roster, schedule, when):
    """All day after a game, any list naming its two teams was refused as
    'already started', a list for their next games too (review 3, R5)."""
    lines, report = _prep(FIRST_LIST, roster, schedule, now=pd.Timestamp(when).to_pydatetime())
    assert list(lines["game_id"]) == ["0022600194", "0022600194", "0022600193"]
    assert report["tipped"] == 0


def test_a_list_for_the_day_after_next_is_not_called_late(roster, schedule):
    """NOP and MIN met on Friday, neither plays Saturday, both play Sunday. A
    Saturday-afternoon list for Sunday was rejected in full (review 3, R5);
    the morning after, it is Friday's list sent late."""
    text = ("Pelicans:\nZion Williamson over 22.5 points\nTrey Murphy III over 2.5 threes\n"
            "Timberwolves:\nAnthony Edwards over 27.5 points\n")
    lines, report = _prep(text, roster, schedule,
                          now=pd.Timestamp("2026-11-07T19:00:00Z").to_pydatetime())
    assert list(lines["game_id"]) == ["0022600201", "0022600201", "0022600198"]
    assert report["tipped"] == 0
    lines, report = _prep(text, roster, schedule,
                          now=pd.Timestamp("2026-11-07T15:00:00Z").to_pydatetime())
    assert report["tipped"] == 3


def test_a_back_to_back_does_not_take_last_nights_list_the_next_morning(roster, schedule):
    """OKC won at MEM on Friday and hosts WAS on Saturday. On Saturday morning
    the Thunder and Grizzlies list is Friday's, sent late: it was filed
    under the Thunder's next game and Edey was 'not placed' (review 3, F3)."""
    text = ("Thunder:\nShai Gilgeous-Alexander over 30.5 points\nGrizzlies:\n"
            "Zach Edey over 10.5 rebounds\n")
    lines, report = _prep(text, roster, schedule,
                          now=pd.Timestamp("2026-11-07T15:00:00Z").to_pydatetime())
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 2
    lines, _ = _prep(text, roster, schedule,
                     now=pd.Timestamp("2026-11-07T19:00:00Z").to_pydatetime())
    assert lines.loc["Shai Gilgeous-Alexander", "game_id"] == "0022600192"


def test_a_late_list_is_not_filed_under_the_rematch(roster, schedule):
    """WAS hosted ATL on Nov 4 and they meet again on Nov 6. The next morning
    their list is the Nov 4 game's. Every earlier late-list test had no
    rematch inside the dates, so dropping the `late` mark passed them all,
    and this list went to the rematch, reported as loaded (review 3, F5)."""
    text = "Wizards:\nAnthony Davis over 11.5 rebounds\nHawks:\nDyson Daniels over 5.5 assists\n"
    lines, report = _prep(text, roster, schedule,
                          now=pd.Timestamp("2026-11-05T15:00:00Z").to_pydatetime())
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 2


def test_a_list_two_days_after_its_game_routes_like_any_other(roster, schedule):
    lines, _ = _prep(FIRST_LIST, roster, schedule,
                     now=pd.Timestamp("2026-11-07T05:01:00Z").to_pydatetime())
    assert list(lines["game_id"]) == ["0022600194", "0022600194", "0022600193"]


@pytest.mark.parametrize("text", [
    "Magic @ Suns 9:00 PM ET", "ORL @ PHX 9pm", "Magic vs Suns - 9:00 PM", "Magic @ Suns 11/5",
    "Thu 11/5: Magic @ Suns", "Orlando Magic at Phoenix Suns, 9 PM", "Magic 4-4 @ Suns 5-3",
    "Magic (4-4) @ Suns (5-3)", "ORL-PHX", "ORL/PHX", "Magic/Suns", "Magic x Suns",
    "Magic & Suns",
    # Review 3, F4: every time zone, a time without AM or PM, the network, a
    # weekday night, a year, an ISO date, the day before the month.
    "Magic @ Suns 7:00 PM MST", "Magic @ Suns | 9 ET", "Magic @ Suns 9:00 PM ET on ESPN",
    "Thursday Night: Magic @ Suns", "Magic @ Suns Nov 5, 2026", "Magic @ Suns 9ET",
    "Magic @ Suns 21:00", "Magic @ Suns 2026-11-05", "Magic @ Suns 5 Nov",
    "Magic at Suns Thursday Night", "Magic @ Suns 9:00 PM Eastern", "Magic @ Suns 7 PM PDT",
    # Review 4, A5: two zones, "Eastern Time", a network without "on", the
    # Cup, "at 9 PM", "9.00pm", "2100".
    "Magic @ Suns 9 PM ET / 7 PM MST", "Magic @ Suns 9pm ET/7pm MST",
    "Magic @ Suns 9:00 PM Eastern Time", "Magic @ Suns 7 PM Mountain Time",
    "Magic @ Suns 9:00 PM ET | ESPN", "Magic @ Suns 9:00 PM ET, ESPN",
    "Magic (4-4) vs Suns (5-3) 9 PM ET ESPN", "NBA Cup: Magic @ Suns", "Magic @ Suns NBA Cup",
    "Magic @ Suns at 9 PM", "Magic @ Suns @ 9pm", "Magic @ Suns on NBA League Pass",
    "Magic @ Suns 9.00pm", "Magic @ Suns 2100",
    # Review 5, A6 and A18: more tip-time, network and Cup spellings, and the
    # time written before the matchup.
    "Magic @ Suns on Amazon Prime", "Magic @ Suns at 9", "Magic at Suns at 9:00",
    "Tonight @ 9: Magic @ Suns", "Magic @ Suns 9p ET", "Magic @ Suns 7:00 PM Arizona time",
    "Magic @ Suns 7 PM local", "Magic @ Suns - Cup game", "Magic @ Suns: NBA Cup group play",
    "Magic @ Suns on FanDuel Sports Network", "Magic @ Suns 21h00",
    "9 PM ET / 7 PM MST: Magic @ Suns",
])
def test_a_game_heading_is_read_with_its_time_date_and_record(text, roster, schedule):
    """A tip time beside the matchup made it no heading at all, and the late
    list was filed under each team's next game again (review 2, P4)."""
    assert props._typed_heading(text) == ("ORL", "PHX")
    lines, report = _prep(text + "\nMiles Bridges over 6.5 R+A\nWendell Carter Jr. under 23.5 PRA",
                          roster, schedule, now=AFTER_GAME)
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 2


@pytest.mark.parametrize("heading", ["Knicks & Nets:", "Knicks and Nets:", "Knicks/Nets:",
                                     "Knicks - Nets:", "Knicks x Nets"])
def test_two_teams_joined_loosely_group_a_list_unless_they_play(heading, roster, schedule):
    """'Knicks & Nets' pinned each line to a game the two do not play, and
    every line was lost with two false reasons (review 3, R6). Whether a
    joined pair is one game is the schedule's to say."""
    lines, report = _prep(f"{heading}\nJalen Brunson over 25.5 points\n"
                          "Noah Clowney over 6.5 rebounds\n", roster, schedule, now=FRIDAY_11AM)
    assert list(lines["game_id"]) == ["0022600191", "0022600190"] and report["comparable"] == 2
    together, _ = _prep(heading.replace("Knicks", "Magic").replace("Nets", "Suns")
                        + "\nMiles Bridges over 6.5 R+A\nWendell Carter Jr. under 23.5 PRA\n",
                        roster, schedule)
    assert list(together["game_id"]) == [MAGIC_AT_SUNS] * 2
    assert list(together["opp"]) == ["ORL", "PHX"]


def test_a_game_heading_gives_each_player_the_other_team(roster, schedule):
    lines, _ = _prep("Magic @ Suns\nMiles Bridges over 6.5 R+A\n"
                     "Wendell Carter Jr. under 23.5 PRA\nStephen Curry over 4.5 threes",
                     roster, schedule)
    assert list(lines["opp"]) == ["ORL", "PHX", None]
    assert list(lines["game_id"]) == [MAGIC_AT_SUNS, MAGIC_AT_SUNS, HEAT_AT_WARRIORS]


def test_a_rematch_does_not_claim_a_list_sent_during_the_first_game(roster, schedule):
    """OKC at POR on Nov 10 and again on Nov 11. A list naming them while the
    first game is on is that game's (review P11); the next morning it is the
    rematch's."""
    text = "Thunder:\nShai Gilgeous-Alexander over 30.5 points\nBlazers:\nDeni Avdija over 7.5 rebounds\n"
    during = pd.Timestamp("2026-11-11T04:30:00Z").to_pydatetime()
    lines, report = _prep(text, roster, schedule, now=during)
    assert set(lines["route_how"]) == {"tipped"} and report["tipped"] == 2
    morning = pd.Timestamp("2026-11-11T15:00:00Z").to_pydatetime()
    lines, _ = _prep(text, roster, schedule, now=morning)
    assert set(lines["game_id"]) == {"0022600226"}


def test_typed_games_keeps_an_opponent_the_line_gave(schedule):
    rows = pd.DataFrame([
        {"source": "typed", "opp": "HOU", "pair": None, "team_code": "PHX",
         "roster_team": "PHX", "match": "matched", "how": "exact"},
        {"source": "typed", "opp": None, "pair": "ORL PHX", "team_code": None,
         "roster_team": "ORL", "match": "matched", "how": "exact"},
        {"source": "typed", "opp": None, "pair": "ORL PHX", "team_code": "BOS",
         "roster_team": "BOS", "match": "matched", "how": "exact"},
        {"source": "board", "opp": None, "pair": None, "team_code": "ORL",
         "roster_team": "PHX", "match": "matched", "how": "exact+other_team"},
    ])
    out = props.typed_games(rows, schedule, now=BEFORE_TIP, headings=["PHX", "ORL"])
    assert list(out["opp"]) == ["HOU", "PHX", None, None]
    assert list(out["how"]) == ["exact", "exact", "exact", "exact+other_team"]   # board kept


@pytest.mark.parametrize("text,code", [
    ("Suns", "PHX"), ("Suns:", "PHX"), ("Phoenix Suns picks", "PHX"), ("PHX", "PHX"),
    ("Phoenix", "PHX"), ("76ers:", "PHI"), ("Sixers", "PHI"), ("Trail Blazers", "POR"),
    ("Golden State", "GSW"), ("No", None), ("NO", None), ("Sa", None), ("Phx", None),
    ("@ATL", None), ("Los Angeles", None), ("LA", None), ("Miles Bridges", None),
    ("Suns 5-3:", "PHX"), ("Suns 9 PM:", "PHX"), ("Suns tonight:", "PHX"),
    ("Suns 7:00 PM MST:", "PHX"), ("Suns 9 ET:", "PHX"),
    ("Jalen Brunson vs Heat", None), ("Shai Gilgeous-Alexander", None),
])
def test_team_words(text, code):
    head = props._typed_heading(text)
    assert (head[0] if head else None) == code


def test_nicknames_stay_out_of_canonical():
    """canonical() reads feed codes and URL parameters: a typed nickname must
    not start resolving there."""
    assert teams.canonical("Suns") is None and teams.from_words("Suns") == "PHX"
    assert teams.from_words("Phoenix", cities=False) is None
    assert all(teams.from_words(c) == c for c in teams.CODES)


# ── Stats ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("phrase,key", [
    ("Rebounds and Assists", "ra"), ("Points + Rebounds", "pr"),
    ("Points, Rebounds, and Assists", "pra"), ("R+A", "ra"), ("P+R+A", "pra"),
    ("P + R + A", "pra"), ("A + P + R", "pra"), ("R + A", "ra"), ("TO", "tov"),
    ("pts + ast", "pa"), ("Steals and Blocks", "stocks"), ("combined points", "pts"),
    ("Points Rebounds Assists", "pra"), ("boards", "reb"), ("dimes", "ast"),
    ("threes", "fg3m"), ("PRA", "pra"), ("first half points", None),
    ("double doubles", None), ("a", None), ("p", None), ("points points", None),
    # Review 5, B2 and A9: a letter beside a stat word, two stats with only a
    # comma or a semicolon between them.
    ("points, a", None), ("rebounds and a", None), ("assists, rebounds", None),
    ("rebounds; assists", None),
])
def test_typed_stat(phrase, key):
    assert props.typed_stat(phrase) == key


def test_classify_reads_typed_phrases_and_still_refuses_what_it_refused():
    assert props.classify_stat("Rebounds and Assists")[:2] == ("ra", "ok")
    assert props.classify_stat("1st Half Points and Rebounds")[1] == "refused"
    assert props.classify_stat("Double Doubles")[1] == "refused"
    assert props.classify_stat("Dunks")[1] == "refused"
    for name in ("Nikola Jokic", "Andre Drummond", "Brandon Miller"):
        assert props.typed_stat(name) is None


# ── The pick on the board ────────────────────────────────────────────────────
class _Provider:
    def __init__(self, pmf):
        self._pmf = pmf

    def pmf(self, person_id, game_id, stat):
        return self._pmf

    def void_risk(self, person_id, game_id, stat):
        return float("nan")


def _compared(roster, schedule, text, pmf):
    lines, _ = props.prepare_paste(text, roster, schedule, now=BEFORE_TIP)
    proj = pd.DataFrame([{"person_id": int(p), "game_id": MAGIC_AT_SUNS,
                          "game_date": "2026-11-05", "name": n, "team": t,
                          "opp": "ORL" if t == "PHX" else "PHX", "games_prior": 300}
                         for p, n, t in roster[["person_id", "name", "team"]].itertuples(
                             index=False)])
    return props.compare(lines, _Provider(pmf), proj)


def test_a_pick_the_model_disagrees_with_is_shown_never_hidden(roster, schedule,
                                                                no_reliability):
    low = Pmf.from_poisson(3.0)                     # every line here leans Less
    table, meta = _compared(roster, schedule, FIRST_LIST, low)
    assert meta["matched"] == 3 and set(table["lean"]) == {"Less"}
    assert list(table.sort_values("line")["pick"]) == ["more", "more", "less"]
    shown, hidden, thin, void = props.board_rows(table)
    assert len(shown) == 3 and hidden.empty           # Over picks against a Less lean
    assert set(table["sides"]) == {"both"}


def test_a_pick_on_a_coin_flip_is_named_in_the_hidden_note(roster, schedule, no_reliability):
    """The board hides an Even lean (no side to show); a visitor looking for
    their own pick there is told which one it was, on the slate and on the
    Game page (review I10)."""
    table, meta = _compared(roster, schedule, FIRST_LIST, Pmf([6, 7], [0.5, 0.5]))
    shown, hidden, thin, void = props.board_rows(table)
    assert list(hidden["player"]) == ["Miles Bridges"] and set(hidden["lean"]) == {"Even"}
    said = (" The pasted pick among them: Miles Bridges Rebs+Asts (picked More; the model has "
            "no lean on it).")
    assert props_ui.hidden_picks_text(hidden) == said
    notes = " ".join(props_ui.notes_html(meta, shown, hidden, thin, void))
    assert said in notes and "1 sits on a coin flip with no lean" in notes
    # The Game page has no hidden-lines note: it says the pick is not shown
    # and why (review 2, I11). The page itself is rendered in
    # tests/test_pages_smoke.py (review 2, I5: this was a grep of its source).
    assert props_ui.hidden_picks_text(hidden, game_page=True) == (
        " Not shown here: the pasted pick Miles Bridges Rebs+Asts (picked More; the model has "
        "no lean on it).")
    assert props_ui.hidden_picks_text(hidden.assign(pick=None)) == ""


def test_the_pick_tag_sits_beside_the_lean():
    """On a pick that agrees with the lean too, the common case (review 2,
    I6), and worded for anyone holding the link, never as the reader's own
    pick (review 2, I12)."""
    row = {"stat": "ra", "lean": "Less", "pick": "more", "line": 6.5}
    for lean, pick in (("Less", "more"), ("More", "more"), ("Less", "less"), ("More", "less")):
        cell = props_ui._stat_cell({**row, "lean": lean, "pick": pick})
        assert f">Pick: {pick.title()}</span>" in cell and f"The model leans {lean}." in cell
        assert "The pick in the pasted list" in cell and "You" not in cell
    assert "Pick:" not in props_ui._stat_cell({**row, "pick": None})
    assert "Pick:" not in props_ui._stat_cell({**row, "pick": "<b>more</b>"})
    assert "no lean" in props_ui._stat_cell({**row, "lean": "Even", "pick": "less"})
    ledger = props_ui.ledger_html(pd.DataFrame([{**row, "player": "Miles Bridges",
                                                 "person_id": 1, "model": 9.0, "gap": 2.5}]))
    assert "Pick: More" in ledger


def test_board_and_feed_rows_never_carry_a_pick(no_reliability):
    """Both PrizePicks paths, and compare's table after them (review I7)."""
    board = props.parse_any((FIX / "board_constructed.txt").read_text(encoding="utf-8"))
    feed = props.parse_any((FIX / "feed_constructed.json").read_text())
    assert set(board["source"]) == {"board"} and len(feed) > 0
    assert board["pick"].isna().all() and feed["pick"].isna().all()
    croster = pd.read_csv(FIX / "roster_constructed.csv")
    csched = pd.read_csv(FIX / "schedule_constructed.csv", dtype={"game_id": str})
    csched["tip_utc"] = pd.to_datetime(csched["tip_utc"], utc=True)
    proj = pd.read_csv(FIX / "projections_constructed.csv", dtype={"game_id": str})
    now = pd.Timestamp("2026-10-21T03:00:00Z").to_pydatetime()
    for text in ((FIX / "board_constructed.txt").read_text(encoding="utf-8"),
                 (FIX / "feed_constructed.json").read_text()):
        lines, _ = props.prepare_paste(text, croster, csched, now=now)
        table, meta = props.compare(lines, props.MeanDispersionProvider(proj), proj)
        assert meta["matched"] > 0 and table["pick"].isna().all()


# ── A typed pick and a board square for the same line ────────────────────────
_SQUARES = ("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n6.5\nRebs+Asts\n{sides}"
            "Oso Ighodaro\nPHX - F\nOso Ighodaro\nvs ORL Thu 9:00pm\n8.5\nPts+Rebs\nLess\nMore\n")


def _board_after(roster, schedule, *texts):
    token = None
    for text in texts:
        lines, _ = props.prepare_paste(text, roster, schedule, now=BEFORE_TIP)
        token = props.add_to_board(token, lines)
    return props.get_board(token).set_index("name")


def test_a_board_paste_keeps_the_picks_on_its_own_numbers(roster, schedule):
    """Paste your list, then the board: a pick and a square on the same
    number are one row with both facts (review I3). A square at another
    number (8.5 against the 7.5 typed) no longer replaces the pick: the pick
    is the bet the visitor asked about (review 2, I3)."""
    b = _board_after(roster, schedule, FIRST_LIST, _SQUARES.format(sides="Less\nMore\n"))
    assert (b.loc["Miles Bridges", "pick"], b.loc["Miles Bridges", "direction"]) == \
        ("more", "both")
    assert (b.loc["Oso Ighodaro", "pick"], b.loc["Oso Ighodaro", "line"]) == ("more", 7.5)
    assert b.loc["Wendell Carter Jr.", "pick"] == "less"


def test_a_typed_pick_keeps_the_sides_the_board_offered(roster, schedule):
    b = _board_after(roster, schedule, _SQUARES.format(sides="More\n"),
                     "Suns:\nMiles Bridges under 6.5 R+A\nMagic:")
    assert (b.loc["Miles Bridges", "pick"], b.loc["Miles Bridges", "direction"]) == \
        ("less", "more")


def test_a_board_repaste_says_what_it_offers_now(roster, schedule):
    """Between board pastes the newest square says which sides it offers: a
    'More only' square re-pasted without buttons is unknown again, as before
    picks existed (review 2, I9)."""
    b = _board_after(roster, schedule, _SQUARES.format(sides="More\n"), _SQUARES.format(sides=""))
    assert b.loc["Miles Bridges", "direction"] == ""


_CURRY = "Stephen Curry\nGSW - G\nStephen Curry\nvs MIA Thu 10:00pm\n3.5\n3-PT Made\nLess\nMore\n"


@pytest.mark.parametrize("texts", [
    (_CURRY, "Stephen Curry over 1.5 threes"),
    (_CURRY, "Stephen Curry over 1.5 threes (goblin)"),
    ("Stephen Curry over 1.5 threes (goblin)", _CURRY),
])
def test_a_typed_pick_below_the_boards_line_is_never_lost_or_held(roster, schedule, texts):
    """Below the board's 3.5, the pick was held as a period line ('Paste the
    full-game board'), or dropped by the ladder while the paste said loaded
    (review 2, I3)."""
    token = None
    for text in texts:
        token, report = props_ui.ingest(text, token, roster, schedule, now=BEFORE_TIP)
        assert report["period_held"] == 0, report
        assert not any("held back" in m for _k, m in props_ui.report_messages(
            {**report, "nonempty": True}))
    b = props.get_board(token)
    assert len(b) == 1 and (b.iloc[0]["line"], b.iloc[0]["pick"]) == (1.5, "more")


def test_a_list_of_goblin_picks_is_not_withheld_as_a_period_board(roster, schedule,
                                                                  no_reliability):
    """Picks far below the medians are the visitor's own Goblins, never a
    quarter board; the same lines with no side still are (review 2, I3)."""
    picks = ("Miles Bridges over 3.5 rebounds\nOso Ighodaro over 2.5 rebounds\n"
             "Wendell Carter Jr. over 4.5 rebounds\n")
    table, meta = _compared(roster, schedule, picks, Pmf.from_poisson(9.0))
    assert meta["matched"] == 3 and not meta["period_board_suspect"]
    plain = ("Miles Bridges, Rebounds, 3.5\nOso Ighodaro, Rebounds, 2.5\n"
             "Wendell Carter Jr., Rebounds, 4.5\n")
    table, meta = _compared(roster, schedule, plain, Pmf.from_poisson(9.0))
    assert meta["period_board_suspect"] and meta["period_withheld"] == 3
    # Beside board lines at full-game numbers, the picks lend no line to the
    # ratio, so the board lines are never withheld for them (review 3, R3).
    squares = ("Paolo Banchero\nORL - F\nPaolo Banchero\n@ PHX Thu 9:00pm\n8.5\nRebounds\n"
               "Less\nMore\nJevon Carter\nORL - G\nJevon Carter\n@ PHX Thu 9:00pm\n8.5\n"
               "Rebounds\nLess\nMore\n")
    table, meta = _compared(roster, schedule, squares + picks, Pmf.from_poisson(9.0))
    assert meta["matched"] == 5 and not meta["period_board_suspect"]


# ── The pick survives a reload ───────────────────────────────────────────────
def test_the_browser_copy_keeps_the_pick(roster, schedule):
    lines, _ = props.prepare_paste(FIRST_LIST, roster, schedule, now=BEFORE_TIP)
    token = props.add_to_board(None, lines)
    payload = props.saved_board(props.get_board(token), BEFORE_TIP, "s1", dt.date(2026, 11, 5))
    payload = json.loads(json.dumps(payload, allow_nan=False))
    assert payload["v"] == 2 and payload["cols"] == V1_COLUMNS + ["pick"]
    back, _pasted, dropped = props.restore_saved(payload, roster, dt.date(2026, 11, 5),
                                                 BEFORE_TIP)
    assert dropped == 0 and list(back["pick"]) == ["more", "more", "less"]


def test_a_board_without_picks_is_saved_as_version_1(roster, schedule):
    """Readable by the code before picks too, so a rollback or an older
    process keeps the copy instead of clearing it (review I11)."""
    lines, _ = props.prepare_paste(_SQUARES.format(sides="Less\nMore\n"), roster, schedule,
                                   now=BEFORE_TIP)
    token = props.add_to_board(None, lines)
    payload = props.saved_board(props.get_board(token), BEFORE_TIP, "s1", dt.date(2026, 11, 5))
    assert payload["v"] == 1 and payload["cols"] == V1_COLUMNS
    assert all(len(r) == len(V1_COLUMNS) for r in payload["rows"])


def test_a_copy_saved_before_picks_still_comes_back(roster):
    """A version-1 copy is already in visitors' browsers; its layout is
    written out above, never taken from props (review I4)."""
    row = {"name": "Miles Bridges", "team_code": "PHX", "stat_type": "Rebs+Asts",
           "line": 6.5, "direction": "both", "odds_type": "standard", "opp": "ORL",
           "is_combo": False, "league": None, "game_date": "2026-11-05",
           "game_id": MAGIC_AT_SUNS, "tip_utc": "2026-11-06T02:00:00+00:00"}
    payload = {"v": 1, "id": "s1", "cols": list(V1_COLUMNS),
               "pasted_utc": "2026-11-05T23:00:00+00:00", "rows": [[row[c] for c in V1_COLUMNS]]}
    back, _p, dropped = props.restore_saved(payload, roster, dt.date(2026, 11, 5), BEFORE_TIP)
    assert dropped == 0 and pd.isna(back.iloc[0]["pick"]) and back.iloc[0]["match"] == "matched"
    assert back.iloc[0]["direction"] == "both" and back.iloc[0]["opp"] == "ORL"


@pytest.mark.parametrize("bad", ["sideways", 1, ["more"], {"more": 1}, True, "MORE", "",
                                 " more", "more\x00"])
def test_a_pick_from_the_browser_is_validated(roster, bad):
    cols = V1_COLUMNS + ["pick"]
    row = {"name": "Miles Bridges", "team_code": "PHX", "stat_type": "Rebs+Asts",
           "line": 6.5, "direction": "", "odds_type": "standard", "opp": "ORL",
           "is_combo": False, "league": None, "game_date": "2026-11-05",
           "game_id": MAGIC_AT_SUNS, "tip_utc": None, "pick": bad}
    payload = {"v": 2, "id": "s1", "cols": cols, "pasted_utc": "2026-11-05T23:00:00+00:00",
               "rows": [[row[c] for c in cols]]}
    back, _p, dropped = props.restore_saved(payload, roster, dt.date(2026, 11, 5), BEFORE_TIP)
    assert dropped == 0 and pd.isna(back.iloc[0]["pick"])


# ── The paste box stays safe ─────────────────────────────────────────────────
def test_picks_typed_into_a_board_paste_are_read_beside_it(roster, schedule):
    """A board copied with a list under it kept the board and dropped every
    pick without a word (review 2, P10, I2). The board's own rows are still
    board rows, which a reader trying the list first would lose (review 2,
    I7), and a typed reading with no side is a square the board holds."""
    board_text = (FIX / "board_constructed.txt").read_text(encoding="utf-8")
    alone = props.parse_any(board_text)
    both = props.parse_any(board_text + "\nMy picks:\nLuka Doncic over 45.5 PRA\n")
    assert list(both["source"]) == ["board"] * len(alone) + ["typed"]
    assert both.iloc[-1][["name", "line", "pick"]].tolist() == ["Luka Doncic", 45.5, "more"]
    assert len(props.parse_any(board_text + "\nLuka Doncic, PRA, 45.5\n")) == len(alone)
    square = _SQUARES.format(sides="Less\nMore\n").split("Oso Ighodaro")[0]
    lines, report = _prep(square + FIRST_LIST, roster, schedule)
    assert report["comparable"] == 3
    assert (lines.loc["Miles Bridges", "pick"], lines.loc["Miles Bridges", "direction"]) == \
        ("more", "both")


def test_the_board_and_the_feed_still_win_over_a_typed_reading():
    """Board text is read as board text, and valid JSON never reaches the
    typed reader, even with a pick-shaped line after it (review I8)."""
    board = props.parse_any("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n"
                            "6.5\nRebs+Asts\nLess\nMore")
    assert list(board["source"]) == ["board"] and list(board["direction"]) == ["both"]
    empty = props.parse_any('{"data": []}')
    assert empty.empty and empty.attrs["json_shape"] == props.EMPTY_FEED
    tail = props.parse_any('{"data": []}\nMiles Bridges over 6.5 Rebounds + Assists')
    assert tail.empty and tail.attrs["json_shape"] == props.EMPTY_FEED


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _word(k: int) -> str:
    """A different capitalised word for each k ('Bac', 'Cac', ...)."""
    return "".join(chr(97 + (k // 26 ** i) % 26) for i in range(3)).title()


# Every line DIFFERENT: copies of one line hid a reader that scanned every
# earlier line, which cost 35 s on 5,000 different lines (review 2, I4).
_SHAPES = (
    lambda k: (f"Take the OVER on {_word(k)} {_word(k + 7)} for a combined {k % 40}.5 "
               f"Rebounds and Assists (Combined Rebounds and Assists will be {k % 40 + 1} or "
               "more)\n"),
    lambda k: "vs " * 39 + f"{k % 1000}.{(k // 1000) % 100}\n",
    lambda k: (f"{_word(k)}:\n" if k % 3 == 0 else f"{_word(k)} {_word(k + 3)}\n" if k % 3 == 1
               else f"over {k % 30}.5 points\n"),
)


def test_the_typed_reader_is_linear():
    """4x the lines costs about 4x, never 16x (review I6: the first version
    of this test passed a quadratic reader). The big paste is the most lines
    of the longest shape (40 words) that MAX_TYPED_WORDS reads in full."""
    n = props.MAX_TYPED_WORDS // 40
    for shape in _SHAPES:
        def paste(n):
            return "".join(shape(k) for k in range(n))
        small_text, big_text = paste(n // 4), paste(n)
        props.parse_typed_lines(paste(200))                       # warm
        small = min(_time(lambda: props.parse_typed_lines(small_text)) for _ in range(2))
        big = min(_time(lambda: props.parse_typed_lines(big_text)) for _ in range(2))
        assert big < 7.0 * small + 0.05, (shape(1)[:30], small, big)


def test_a_hostile_paste_is_bounded():
    """150,000 lines of 'vs vs ... 6.5' cost the shared process ~20 s through
    the typed reader before MAX_TYPED_LINES (review P8). Different lines, so
    a cost kept per line seen shows (review 2, I4)."""
    paste = "".join(_SHAPES[1](k) for k in range(150_000))
    took = _time(lambda: props.parse_any(paste))
    board = _time(lambda: props.parse_prizepicks_board(paste))
    assert took < board + 3.0, (took, board)
    df = props.parse_typed_lines(("Miles Bridges over 6.5 R+A\n") * (props.MAX_TYPED_LINES + 10))
    assert len(df) == props.MAX_TYPED_LINES and df.attrs["truncated"]


def test_a_paste_of_long_lines_stops_at_the_word_budget():
    """A long line costs per word, so 5,000 lines of 40 junk words cost the
    shared process seconds within MAX_TYPED_LINES (review P8): the reader also
    stops at MAX_TYPED_WORDS, and the line that crosses it is the last read.
    A board paste (about 2 words a line) and any real list stay far under it."""
    long = "Miles Bridges over 6.5 R+A - he has been great lately and should stay hot\n"
    assert len(long.split()) == 15
    df = props.parse_typed_lines(long * 3000)
    assert len(df) == (props.MAX_TYPED_WORDS + 14) // 15 and df.attrs["truncated"]
    df = props.parse_typed_lines(long * 2000)
    assert len(df) == 2000 and not df.attrs["truncated"]
    assert props.MAX_TYPED_WORDS >= 4 * 2 * props.MAX_TYPED_LINES     # a board paste at the line cap


def test_the_heading_noise_lookahead_changes_nothing():
    """The heading noise's lookahead skips only the places where none of its
    pieces can start (review P8: without it a hostile paste of long lines cost
    seconds, the noise tried at every character). With it and without it,
    every heading, time, date, network and run of junk below reads the
    same."""
    plain = [re.compile(props._heading_noise_body(n), re.IGNORECASE)
             for n in (props._NETWORKS, props._NETWORKS_WIDE)]
    fast = [props._HEADING_NOISE, props._HEADING_NOISE_WIDE]
    words = [
        "Magic", "@", "Suns", "vs", "at", "on", "x", "9", "12", "2026-11-05", "9:00", "9:00 PM",
        "9pm", "9 p.m.", "9P", "@9", "@ 9:30", "at 9", "9 ET", "21:00", "2100", "4-4", "11/5",
        "11/05/26", "11.5.26", "Nov 5", "Nov. 5, 2026", "5 Nov", "5th Nov", "Thursday", "Thu.",
        "Thursday Night", "ET", "Eastern Time", "Arizona time", "Phoenix time", "local", "ESPN",
        "| ESPN", "ESPN2", "ESPN+", "NBA League Pass", "League Pass", "Amazon Prime Video",
        "FanDuel Sports Network", "FanDuel Sports Network Southwest", "NBC Sports Bay Area",
        "NBC Washington", "MSG2", "MSG Networks", "KJZZ", "fdsnsw", "NBCSBA", "Peacock", "truTV",
        "Emirates NBA Cup", "NBA Cup group play", "Cup game", "In-Season Tournament",
        "West Group B", "Game 1", "tonight", "today", "tip", "tip-off", "tipoff 9 PM",
        "9 o'clock", "9ish", "21h", "9h30", "7:00 Arizona", "start time", "Max", "HBO Max",
        "Bally", "Suns Live", "late game", "main slate", "early window", "76ers", "(1H)", "1Q",
        "-", "/", "|", ":", ",", "_", "#", "a9pm", "x@9", "9PMET", "MST", "PDT", "az",
        chr(0xE9) + "t", chr(0xFF19) + "pm", chr(0x663) + ":00", chr(0xBD), chr(0x2014),
    ]
    rng = random.Random(8)
    lines = [" ".join(words), "".join(words), "vs " * 99 + "6.5", "at " * 99 + "6.5"]
    for _ in range(6000):
        picked = rng.choices(words, k=rng.randint(1, 9))
        lines.append(rng.choice([" ", "", " ", "@", ":", "-", " | "]).join(picked))
    for ln in lines:
        for a, b in zip(plain, fast):
            assert a.sub(" ", ln) == b.sub(" ", ln), ln
        assert props._has_digit(ln) == any(ch.isdigit() for ch in ln), ln


@pytest.mark.parametrize("hostile", [
    "x" * 280, ("Miles Bridges over 6.5 " + "+" * 250), "( " * 140, ") " * 140,
    "Miles Bridges over 6.5 R+A (" + "(" * 250, chr(0x2014) + " " * 0 + (chr(0x2014) + " ") * 119 + "Miles Bridges o6.5 R+A",
    "Miles Bridges over 6.5 " + "R+A " * 70, "o" + "6.5 " * 70, "@" * 280, "@ " * 140,
    "Miles Bridges over 1e400 points", "Miles Bridges over NaN points",
    "Miles Bridges over 99999.5 points", "(" + "@ATL " * 50 + ")",
    "\x00Miles Bridges\x00 over 6.5 R+A", "vs " * 39 + "6.5", "no " * 60 + "7+ R+A",
    FIRE * 140, "Will " * 50 + "7+ R+A?",
])
def test_hostile_typed_lines_never_raise(hostile):
    """parse_typed_lines itself, whose errors parse_any would hide (review I5)."""
    df = props.parse_typed_lines(hostile)
    assert df.attrs["skipped_error"] == 0
    assert np.isfinite(df["line"].astype(float)).all() if len(df) else True
    assert (df["line"].astype(float) <= props.MAX_LINE_VALUE).all() if len(df) else True
    beside = props.parse_typed_lines(FIRST_LIST + hostile + "\n")
    assert beside.attrs["skipped_error"] == 0 and len(beside) >= 3


# ── Review 3 ─────────────────────────────────────────────────────────────────
class _ByStat:
    """A provider with a pmf per stat, `default` for the rest."""

    def __init__(self, by_stat, default):
        self._by, self._default = by_stat, default

    def pmf(self, person_id, game_id, stat):
        return self._by.get(stat, self._default)

    def void_risk(self, person_id, game_id, stat):
        return float("nan")


def _proj(roster):
    return pd.DataFrame([{"person_id": int(p), "game_id": MAGIC_AT_SUNS,
                          "game_date": "2026-11-05", "name": n, "team": t,
                          "opp": "ORL" if t == "PHX" else "PHX", "games_prior": 300}
                         for p, n, t in roster[["person_id", "name", "team"]].itertuples(
                             index=False)])


@pytest.mark.parametrize("label", ["1Q props:", "First quarter picks:", "1H:", "Q1 Props",
                                   "1Q Points"])
def test_a_period_in_a_section_label_refuses_the_picks_under_it(label, roster, schedule):
    """'1Q props:' over plain picks showed quarter lines as full-game ones at
    q near 1 (review 3, R2). A team heading under it keeps the period; a
    full-game section ends it."""
    text = (f"{label}\nSuns:\nMiles Bridges over 2.5 points\nOso Ighodaro over 1.5 rebounds\n"
            "Full game:\nWendell Carter Jr. under 23.5 PRA\n")
    lines, report = _prep(text, roster, schedule)
    assert report["not_comparable"]["not priced"] == 2 and report["comparable"] == 1
    assert set(lines.loc[["Miles Bridges", "Oso Ighodaro"], "refusal"]) == {props.REFUSE_PERIOD}
    assert lines.loc["Wendell Carter Jr.", "game_id"] == MAGIC_AT_SUNS


def test_a_quarter_board_pasted_over_a_typed_pick_is_still_caught(roster, schedule,
                                                                  no_reliability):
    """A typed full-game pick, then a quarter board for the same game: the
    square under the pick was dropped without a word, and the other two were
    shown as full-game lines at q near 1 (review 3, R3). The pick itself is
    never held or withheld."""
    token, _ = props_ui.ingest("Miles Bridges over 24.5 points", None, roster, schedule,
                               now=BEFORE_TIP)
    quarter = ("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n6.5\nPoints\nLess\n"
               "More\nOso Ighodaro\nPHX - F\nOso Ighodaro\nvs ORL Thu 9:00pm\n2.5\nRebounds\n"
               "Less\nMore\nWendell Carter Jr.\nORL - F\nWendell Carter Jr.\n@ PHX Thu 9:00pm\n"
               "2.5\nRebounds\nLess\nMore\n")
    token, report = props_ui.ingest(quarter, token, roster, schedule, now=BEFORE_TIP)
    assert report["period_held"] == 1
    assert any("held back" in m for _k, m in props_ui.report_messages(
        {**report, "nonempty": True}))
    provider = _ByStat({"pts": Pmf.from_poisson(24.0)}, Pmf.from_poisson(9.0))
    table, meta = props.compare(props.get_board(token), provider, _proj(roster))
    assert meta["period_board_suspect"] and meta["period_withheld"] == 2
    assert list(zip(table["player"], table["line"], table["pick"])) == \
        [("Miles Bridges", 24.5, "more")]


def test_a_board_line_a_pick_sets_aside_is_never_called_loaded(roster, schedule):
    """A board re-paste at 7.5 for a pick typed at 6.5: the board keeps the
    pick (review 2, I3), and the paste said '1 line loaded' for a line that
    never shows (review 3, R8, F14). It is said instead, and not counted."""
    square = "Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n7.5\nRebs+Asts\nLess\nMore\n"
    token, _ = props_ui.ingest("Miles Bridges over 6.5 R+A", None, roster, schedule,
                               now=BEFORE_TIP)
    token, report = props_ui.ingest(square, token, roster, schedule, now=BEFORE_TIP)
    assert props_ui.toast_count(report) == 0 and report["pick_kept"] == 1
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    assert msgs == [("warning", "1 line in that paste was set aside for a pick you typed for "
                                "the same player and stat at another number (Miles Bridges "
                                "Rebs\\+Asts). The board keeps your pick's number; type the pick "
                                "again at the board's number to compare that line instead.")]
    b = props.get_board(token)
    assert len(b) == 1 and (b.iloc[0]["line"], b.iloc[0]["pick"]) == (6.5, "more")
    # Viewed from another date, the set-aside line is still no line "saved"
    # for its own (review 4, A12: the view above was the line's date).
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-04"})
    assert not any("saved for" in m for _k, m in msgs)
    # Pasted together, the same. Typed after the board, the pick replaces the
    # older line as any newer paste does, and nothing in that paste is set
    # aside.
    _lines, report = props.prepare_paste(square + "Miles Bridges over 6.5 R+A\n", roster,
                                         schedule, now=BEFORE_TIP)
    assert (report["pick_kept"], report["comparable"]) == (1, 1)
    token, _ = props_ui.ingest(square, None, roster, schedule, now=BEFORE_TIP)
    token, report = props_ui.ingest("Miles Bridges over 6.5 R+A", token, roster, schedule,
                                    now=BEFORE_TIP)
    assert report["pick_kept"] == 0 and props_ui.toast_count(report) == 1


def test_lines_a_board_paste_could_not_read_as_picks_are_counted():
    """Typed under a board, a pick with a surname only, or with two numbers,
    was dropped in silence; pasted alone it was counted (review 3, F13)."""
    board_text = (FIX / "board_constructed.txt").read_text(encoding="utf-8")
    alone = props.parse_any(board_text)
    assert (alone.attrs["typed_noname"], alone.attrs["typed_unread"]) == (0, 0)
    both = props.parse_any(board_text + "\nMy picks:\nBridges over 6.5 R+A\n"
                           "Jalen Brunson o25.5 pts - hit in 8 of last 10\n"
                           "Luka Doncic over 45.5 PRA\n")
    assert (both.attrs["typed_noname"], both.attrs["typed_unread"]) == (1, 1)
    assert list(both["source"]).count("typed") == 1
    msgs = props_ui.report_messages({"nonempty": True, "kept": len(both),
                                     "comparable": len(both), "parse": both.attrs})
    text = " ".join(m for _k, m in msgs)
    assert "names no player in full" in text and "more than one number" in text


def test_a_board_square_is_no_section_label_and_no_unnamed_pick():
    """Inside a board paste the board's own text is not a typed list: its
    '1Q Points' square once made every pick typed after it a quarter prop,
    and a square run together onto one line was counted as a pick that
    names no player in full."""
    board_text = (FIX / "board_constructed.txt").read_text(encoding="utf-8")
    assert "\n1Q Points\n" in board_text
    both = props.parse_any(board_text + "\nNikola Jokic over 34.5 points is the play\n")
    mine = both[both["source"] == "typed"]
    assert list(zip(mine["name"], mine["stat_type"])) == [("Nikola Jokic", "Points")]
    lines = [x for x in board_text.splitlines() if x.strip()]
    squares, cur = [], []
    for i, ln in enumerate(lines):
        cur.append(ln)
        if ln in ("Less", "More") and (lines[i + 1] if i + 1 < len(lines) else None) \
                not in ("Less", "More"):
            squares.append(cur)
            cur = []
    joined = "\n".join(("\n".join(q) if i % 2 == 0 else " ".join(q))
                       for i, q in enumerate(squares + [cur]))
    run_together = props.parse_any(joined)
    assert len(run_together) > 0 and run_together.attrs["typed_noname"] == 0


def test_a_board_paste_brings_the_headings_of_the_list_under_it(roster, schedule):
    """A board square with the late list under it: without the list's
    headings its three picks were filed under each team's next game, and the
    one test of a board with a list ran before tip (review 3, F15)."""
    square = "Stephen Curry\nGSW - G\nStephen Curry\nvs ORL Sat 8:30pm\n4.5\n3-PT Made\nLess\nMore\n"
    lines, report = _prep(square + FIRST_LIST, roster, schedule, now=AFTER_GAME)
    late = ["Miles Bridges", "Oso Ighodaro", "Wendell Carter Jr."]
    assert set(lines.loc[late, "route_how"]) == {"tipped"} and report["tipped"] == 3
    assert lines.loc["Stephen Curry", "game_id"] == "0022600193" and report["comparable"] == 1


def test_a_held_period_square_lends_a_typed_pick_nothing(roster, schedule):
    """A 'More only' square held back as a period line lent its sides to an
    Under typed on its number, and the pick was hidden (review 3, F16)."""
    text = ("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n24.5\nPoints\nLess\nMore\n"
            "Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n12.5\nPoints\nMore\n"
            "Miles Bridges under 12.5 points\n")
    lines, report = _prep(text, roster, schedule)
    assert report["period_held"] == 1
    kept = lines.loc["Miles Bridges"]
    assert (kept["line"], kept["pick"], kept["direction"]) == (12.5, "less", "")


def test_a_hidden_pick_is_told_why_of_the_models_lean(roster, schedule, no_reliability):
    """A More pick on a More-only square, the model on Less: the Game page
    said the pick sat on a side the line does not offer, when More is just
    what it offers (review 3, R9, F19)."""
    token = None
    for text in (_SQUARES.format(sides="More\n"), "Miles Bridges over 6.5 R+A"):
        lines, _ = props.prepare_paste(text, roster, schedule, now=BEFORE_TIP)
        token = props.add_to_board(token, lines)
    table, meta = props.compare(props.get_board(token), _Provider(Pmf.from_poisson(3.0)),
                                _proj(roster))
    shown, hidden, thin, void = props.board_rows(table)
    mine = hidden[hidden["player"] == "Miles Bridges"]
    assert list(zip(mine["lean"], mine["sides"], mine["pick"])) == [("Less", "more", "more")]
    assert props_ui.hidden_picks_text(mine, game_page=True) == (
        " Not shown here: the pasted pick Miles Bridges Rebs+Asts (picked More; the model "
        "leans Less, a side this line does not offer).")
    notes = " ".join(props_ui.notes_html(meta, shown, mine, thin, void))
    assert "1 leans a side the line does not offer" in notes


def test_one_line_is_one_line_in_the_paste_report(roster, schedule):
    """'1 line loaded. 1 are for Thu Nov 5', which a single typed pick now
    reaches often (review 3)."""
    _lines, report = props.prepare_paste("Miles Bridges over 6.5 R+A", roster, schedule,
                                         now=BEFORE_TIP)
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    assert msgs[0] == ("success", "1 line loaded. 1 is for Thu Nov 5: see Model vs the board "
                                  "on the slate and on each game page.")


def test_a_name_line_with_a_tip_time_names_the_picks_under_it():
    """A player's card copied with its tip time lost every pick under it
    (review 3, F10)."""
    df = props.parse_typed_lines("Jalen Brunson @ Heat 7:30 PM\nOver 25.5 points\n"
                                 "Over 6.5 assists\nJosh Hart vs Heat 7:30pm ET\n"
                                 "over 8.5 rebounds\n")
    assert list(zip(df["name"], df["opp"], df["stat_type"])) == [
        ("Jalen Brunson", "MIA", "Points"), ("Jalen Brunson", "MIA", "Assists"),
        ("Josh Hart", "MIA", "Rebounds")]


def test_the_cheapest_typed_lines_scale_linearly():
    """A per-line cost that grows with the lines before it hides inside the
    7x + 50 ms bound above when parsing itself dominates: round 1's scan of
    every earlier line passed it at 8x on the cheapest shape (review 3, F20).
    On that shape at the cap, 4x the lines must cost under 6x, no slack."""
    shape = _SHAPES[2]

    def paste(n):
        return "".join(shape(k) for k in range(n))
    small_text = paste(props.MAX_TYPED_LINES // 4)
    big_text = paste(props.MAX_TYPED_LINES)
    props.parse_typed_lines(paste(200))                           # warm
    small = min(_time(lambda: props.parse_typed_lines(small_text)) for _ in range(3))
    big = min(_time(lambda: props.parse_typed_lines(big_text)) for _ in range(3))
    assert big < 6.0 * small, (small, big)


# ── Review 4 ─────────────────────────────────────────────────────────────────
_BODY = "Miles Bridges over 6.5 R+A\nWendell Carter Jr. under 23.5 PRA\n"


def test_a_yes_or_no_inside_a_note_is_no_answer(roster, schedule):
    """'no doubt' after an Over was the market's No, and in capitals ESPN's
    New Orleans as well: either way the board showed 'Pick: Less' (review 4,
    A1)."""
    for text in ("JALEN BRUNSON OVER 25.5 POINTS NO DOUBT", "Jalen Brunson over 25.5 points NO doubt",
                 # An all-caps NO inside a note is neither New Orleans nor an
                 # answer (review 5).
                 "JALEN BRUNSON OVER 25.5 POINTS, KNICKS HAVE NO DEFENSE"):
        r = _one(text)
        assert (r["pick"], r["alt_pick"]) == ("more", None), text
        lines, _ = _prep(text, roster, schedule, now=FRIDAY_11AM)
        assert lines.iloc[0]["pick"] == "more", text


@pytest.mark.parametrize("text", [
    "Anthony Edwards over 27.5 points vs NO", "Anthony Edwards over 27.5 points @NO",
    "Anthony Edwards over 27.5 points @ NO", "Anthony Edwards vs NO over 27.5 points",
    "Anthony Edwards over 27.5 points (@NO)", "MIN @ NO\nAnthony Edwards over 27.5 points",
    # Every opponent mark, not only "vs" and "@" (review 5, A14).
    "Anthony Edwards over 27.5 points against NO", "Anthony Edwards over 27.5 points at NO",
    "Anthony Edwards over 27.5 points v NO", "Anthony Edwards over 27.5 points vs. NO",
    "Anthony Edwards over 27.5 points versus NO",
])
def test_vs_no_is_new_orleans_the_opponent(text, roster, schedule):
    """ESPN's 'vs NO' after an Over was read as the market's No, and the
    board showed 'Pick: Less' (review 4, A4)."""
    lines, _ = _prep(text, roster, schedule, now=FRIDAY_11AM)
    r = lines.iloc[0]
    assert (r["pick"], r["opp"], r["game_id"]) == ("more", "NOP", "0022600031")


def test_a_late_list_under_a_loose_game_heading_is_late_all_the_next_day(roster, schedule):
    """'ORL-PHX' was read as two team headings, so from noon the day after,
    last night's list was filed under each team's next game and called
    loaded, while 'Magic @ Suns' said late (review 4, B4)."""
    for heading in ("ORL-PHX 9:00 PM ET", "Magic/Suns 9 PM", "Magic & Suns", "Magic x Suns"):
        for when in ("2026-11-06T17:01:00Z", "2026-11-07T04:30:00Z"):
            lines, report = _prep(f"{heading}\n{_BODY}", roster, schedule,
                                  now=pd.Timestamp(when).to_pydatetime())
            assert set(lines["route_how"]) == {"tipped"}, (heading, when)
            assert report["tipped"] == 2 and report["comparable"] == 0
    # Two separate team headings still follow the noon rule (review 3, R5).
    lines, _ = _prep(f"Magic:\n{_BODY.splitlines()[1]}\nSuns:\n{_BODY.splitlines()[0]}\n",
                     roster, schedule, now=pd.Timestamp("2026-11-06T17:01:00Z").to_pydatetime())
    assert set(lines["route_how"]) == {"team_schedule"}
    # Over team sub-headings the loose pair still names the game (review 5,
    # A15).
    sub = (f"ORL-PHX 9:00 PM ET\nMagic:\n{_BODY.splitlines()[1]}\nSuns:\n"
           f"{_BODY.splitlines()[0]}\n")
    for when in ("2026-11-06T17:01:00Z", "2026-11-06T20:00:00Z"):
        lines, _ = _prep(sub, roster, schedule, now=pd.Timestamp(when).to_pydatetime())
        assert set(lines["route_how"]) == {"tipped"}, when


@pytest.mark.parametrize("remark", [
    "What a 4th quarter!", "1H recap: 3-1", "Brunson went off in the 4th quarter last night",
    "No 1Q props today, full game only", "My full-game picks (no 1Q or 1H):",
    "Fade alert: the Suns have been awful in the first half", "Q4 Brunson is different",
    "Suns (blew a 2H lead last night):", "Not betting the 1H tonight:",
    "Full game picks, 1H later:",
    # Review 5: the period's picks posted elsewhere, a remark's verb, and a
    # back-to-back note.
    "2H plays coming later in the discord", "Live 2H bets in my discord",
    "Brunson owns the fourth quarter", "Warriors in the first of a back to back",
    # Review 6: the verbs a negation reaches the period across.
    "Not touching the 1H tonight", "Not playing the 1Q tonight",
])
def test_a_remark_that_names_a_period_is_no_section_label(remark, roster, schedule):
    """Any line naming a period refused every full-game pick under it as a
    period prop (review 4, A6, B2)."""
    lines, report = _prep(f"{remark}\n{_BODY}", roster, schedule)
    assert report["comparable"] == 2 and not report["not_comparable"].get("not priced")


def test_a_period_label_and_a_headings_own_period(roster, schedule):
    """A label with no colon, or with a few more words, is still a label; a
    heading's own period ends at the next heading, and 'full game' with no
    colon ends a label's (review 4, A11, B2)."""
    for label in ("First quarter props", "1Q props for the late games:", "1Q PRA"):
        lines, report = _prep(f"{label}\n{_BODY}", roster, schedule)
        assert report["not_comparable"]["not priced"] == 2, label
    text = ("Magic @ Suns (1H)\nMiles Bridges over 12.5 points\nHeat @ Warriors\n"
            "Stephen Curry over 4.5 threes\n")
    lines, report = _prep(text, roster, schedule)
    assert lines.loc["Miles Bridges", "refusal"] == props.REFUSE_PERIOD
    assert lines.loc["Stephen Curry", "game_id"] == HEAT_AT_WARRIORS and report["comparable"] == 1
    text = "1Q props:\nMiles Bridges over 2.5 points\nFull game props\n" + _BODY.splitlines()[1]
    lines, report = _prep(text, roster, schedule)
    assert lines.loc["Miles Bridges", "refusal"] == props.REFUSE_PERIOD
    assert lines.loc["Wendell Carter Jr.", "stat_type"] == "Pts+Rebs+Asts"
    assert report["comparable"] == 1        # a refused line still has its game_id


def test_a_board_paste_takes_a_period_only_from_a_typed_label():
    """A board's period tab ('1Q', '1st Quarter') above typed picks made them
    quarter props while the squares stayed full game (review 4, B2)."""
    square = "Stephen Curry\nGSW - G\nStephen Curry\nvs ORL Sat 8:30pm\n4.5\n3-PT Made\nLess\nMore\n"
    for tab, period in (("1Q", False), ("1st Quarter", False), ("1Q props:", True),
                        # A label naming the picks needs no colon (review 5, B1),
                        # and a colon alone makes one (review 6, A12).
                        ("1Q Props", True), ("1Q Locks", True), ("## 1Q props", True),
                        ("My 1Q picks", True), ("1Q:", True), ("First quarter:", True),
                        ("1st Half:", True)):
        df = props.parse_any(f"{tab}\n{square}Jalen Brunson over 25.5 points\n")
        mine = df[df["source"] == "typed"]
        assert len(mine) == 1, tab
        st = mine.iloc[0]["stat_type"]
        assert (props.classify_stat(st)[2] == props.REFUSE_PERIOD and st.startswith("Points ")
                if period else st == "Points"), (tab, st)


def test_a_pick_that_reports_its_result_is_counted_never_filed(roster, schedule):
    """'Jalen Brunson over 25.5 points - Knicks lost last night' vanished
    without a word, and a line with two numbers lost its count (review 4,
    A7, B7). A result word before the line or right after the stat is a
    recap, counted; later in a note it is context."""
    for text in ("Jalen Brunson over 25.5 points - cashed", "Cashed: Jalen Brunson over 25.5 points",
                 f"{CHECK} Jalen Brunson o26.5 pts, cashed", "Josh Hart under 8.5 rebounds (L) missed"):
        df = props.parse_typed_lines(text)
        assert df.empty and df.attrs["typed_recap"] == 1, text
    df = props.parse_typed_lines("Josh Hart over 8.5 rebounds, cashed 8 of the last 10")
    assert df.empty and df.attrs["typed_unread"] == 1
    # After a mark or a closer, "cashed" anywhere after the stat, and a result
    # in a parenthesis (review 5, A10, B6); "pushed" in a note about the team
    # is context.
    for text in ("Jalen Brunson o25.5 pts = cashed", "Jalen Brunson o25.5 pts LOCK cashed",
                 "Jalen Brunson over 25.5 points (cashed)",
                 "Jalen Brunson over 25.5 points - he cashed it",
                 "Josh Hart under 8.5 rebounds - that one lost",
                 "Josh Hart over 8.5 rebounds - it pushed"):
        df = props.parse_typed_lines(text)
        assert df.empty and df.attrs["typed_recap"] == 1, text
    assert len(props.parse_typed_lines("Jalen Brunson over 6.5 assists, Knicks pushed the pace "
                                       "last game")) == 1
    _lines, report = props.prepare_paste("Jalen Brunson over 25.5 points - cashed\n"
                                         "Josh Hart over 8.5 rebounds\n", roster, schedule,
                                         now=FRIDAY_11AM)
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-07"})
    # Wherever the word sat, and advice only for a pick taken as a result
    # (review 5, B12).
    assert ("info", "1 line reads like a result, not a pick (a word such as cashed, missed or "
                    "won), so it was skipped. If it is a pick for a game still to come, type it "
                    "without that word to compare it.") in msgs
    two = props_ui.report_messages({"nonempty": True, "kept": 1, "comparable": 1,
                                    "parse": {"typed_recap": 2}})
    assert ("info", "2 lines read like results, not picks (a word such as cashed, missed or won), "
                    "so they were skipped. If any is a pick for a game still to come, type it "
                    "without that word to compare it.") in two
    _lines, report = props.prepare_paste("Jalen Brunson over 25.5 points - cashed", roster,
                                         schedule, now=FRIDAY_11AM)
    msgs = props_ui.report_messages({**report, "nonempty": True})
    assert msgs[0][0] == "warning" and "reads like a result, not a pick" in msgs[0][1]


def test_a_handle_in_place_of_the_name_borrows_no_name(roster, schedule):
    """A player's handle stood where his name goes, so the line borrowed the
    name above and his pick was filed under Josh Hart (review 4, B3)."""
    for line in ("@jalenbrunson1 o25.5 pts", "#JalenBrunson over 25.5 points",
                 "#KAT over 11.5 rebounds"):
        df = props.parse_typed_lines(f"Josh Hart\nOver 8.5 rebounds\n{line}\n")
        assert list(zip(df["name"], df["stat_type"], df["line"])) == \
            [("Josh Hart", "Rebounds", 8.5)], line
        assert df.attrs["typed_noname"] == 1


def test_a_pick_above_the_board_names_the_pick_when_the_board_is_held(roster, schedule):
    """A pick typed above the board's number (an untagged Demon, a typo),
    then the full-game board: told it looked like a quarter board, paste the
    full-game board (review 4, A8, B5). The message names the pick; held
    below a board line, it is the quarter-board message as before."""
    square = "Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n6.5\nRebs+Asts\nLess\nMore\n"
    token, _ = props_ui.ingest("Miles Bridges over 12.5 R+A", None, roster, schedule,
                               now=BEFORE_TIP)
    token, report = props_ui.ingest(square, token, roster, schedule, now=BEFORE_TIP)
    assert (report["period_held"], report["period_held_pick"]) == (1, 1)
    assert props_ui.toast_count(report) == 0
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    assert msgs == [("warning", "1 line sat far below the pick you typed for the same player "
                                "and stat (Miles Bridges Rebs\\+Asts), so it was held back and "
                                "the board keeps your pick. If that paste is the full-game "
                                "board, type the pick again at the board's number to compare "
                                "it.")]
    # The board's own line is the standard, never the pick's: the same board
    # line after the pick is set aside for it, not held.
    board, _ = props.prepare_paste(square, roster, schedule, now=BEFORE_TIP)
    pick, _ = props.prepare_paste("Miles Bridges over 12.5 R+A", roster, schedule,
                                  now=BEFORE_TIP)
    merged = props.collapse_alt_lines(pd.concat([board, pick, board], ignore_index=True),
                                      new_from=2)
    assert (merged.attrs["period_held"], merged.attrs["pick_kept"]) == (0, 1)
    held = props_ui.report_messages({"nonempty": True, "kept": 1, "comparable": 1,
                                     "period_held": 1, "period_held_pick": 0})
    assert held[0] == ("warning", "1 line sat far below the full-game line already on your "
                                  "board, so it was held back: that looks like a quarter or half "
                                  "board. Paste the full-game board.")
    # Two lines, each below its own pick (review 5, A20).
    held = props_ui.report_messages({"nonempty": True, "kept": 2, "comparable": 2,
                                     "period_held": 2, "period_held_pick": 2,
                                     "period_held_pick_names": ["Jalen Brunson Points",
                                                                "Josh Hart Rebounds"]})
    assert held[0] == ("warning", "2 lines sat far below the picks you typed for the same "
                                  "players and stats (Jalen Brunson Points, Josh Hart Rebounds), "
                                  "so they were held back and the board keeps your picks. If that "
                                  "paste is the full-game board, type those picks again at the "
                                  "board's numbers to compare them.")


def test_two_goblins_and_a_pick_are_no_quarter_board(roster, schedule, no_reliability):
    """Goblin rungs sit near half the mean by design; with a typed pick
    lending the third player, two real Goblin lines were withheld as a
    quarter or half board (review 4, B6)."""
    goblins = ("Paolo BancheroGoblin\nORL - F\nPaolo Banchero\n@ PHX Thu 9:00pm\n4.5\nRebounds\n"
               "More\nJevon CarterGoblin\nORL - G\nJevon Carter\n@ PHX Thu 9:00pm\n4.5\n"
               "Rebounds\nMore\nMiles Bridges over 6.5 R+A\n")
    table, meta = _compared(roster, schedule, goblins, Pmf.from_poisson(9.0))
    assert meta["matched"] == 3 and not meta["period_board_suspect"]
    assert meta["period_withheld"] == 0


def test_the_other_lines_are_explained_when_every_line_is_set_aside(roster, schedule):
    """With its only comparable line set aside for a pick, the paste's Dunks
    square was never explained (review 4, B11)."""
    square = ("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n7.5\nRebs+Asts\nLess\n"
              "More\nOso Ighodaro\nPHX - F\nOso Ighodaro\nvs ORL Thu 9:00pm\n0.5\nDunks\nLess\n"
              "More\n")
    token, _ = props_ui.ingest("Miles Bridges over 6.5 R+A", None, roster, schedule,
                               now=BEFORE_TIP)
    token, report = props_ui.ingest(square, token, roster, schedule, now=BEFORE_TIP)
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-05"})
    assert [k for k, _m in msgs] == ["warning", "info"]
    assert msgs[1] == ("info", "Also in that paste and not compared: 1 not priced (period "
                               "props, Dunks, two-player squares and the like).")


def test_a_line_on_no_game_is_never_set_aside_for_a_pick(roster, schedule):
    """A late pick and a board square for the same started game both sat on
    no date, so the square was 'set aside for a pick you typed' and the
    toast hid a line that showed (review 4, B12)."""
    square = "Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n7.5\nRebs+Asts\nLess\nMore\n"
    other = "Stephen Curry\nGSW - G\nStephen Curry\nvs ORL Sat 8:30pm\n4.5\n3-PT Made\nLess\nMore\n"
    token, _ = props_ui.ingest("Magic @ Suns\nMiles Bridges over 6.5 R+A", None, roster,
                               schedule, now=AFTER_GAME)
    token, report = props_ui.ingest(square + other, token, roster, schedule, now=AFTER_GAME)
    assert report["pick_kept"] == 0 and props_ui.toast_count(report) == 1
    msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-06"})
    assert not any("set aside" in m for _k, m in msgs)


# ── Review 5 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label", [
    "1Q Locks", "## 1Q props", "1Q props I like", "1H plays to tail", "1Q props below",
    "No doubt 1Q locks:", "Not your usual 1Q props:", "1Q props, no parlays:",
    "1Q props no one is talking about:", "Been on a heater lately so here are my 1Q props:",
    "1Q bets I'm on:", "First quarter overs I like:", "1Q props Saturday", "FREE 1Q Props",
    "First 5 minutes props:", "Tonight's First 5 Minutes Props", "First five minutes props:",
    "1Q Props (full game below):", "Tonight's picks (1H, early game)",
    # A line naming the picks stays a label even with a remark's words.
    "1Q props went 3-0 yesterday",
    # Review 6: a negation's reach ends at a mark (A3, B4); hyphens, "qtr",
    # "Quarter 1", "opening", the first minutes (B4, A10); a brag beside
    # the period in a parenthesis, a label with the period as its subject,
    # a label that is also a brag (B4); a colon label with a remark's words
    # (A12).
    "No more waiting, 1Q props:", "No overthinking, 1Q plays:", "No hedging, 1Q props only:",
    "No more guessing, 1Q locks:", "No missing: 1Q props", "No fading the 1Q:",
    "No 2H, 1H props:", "First-half props:", "1st-quarter picks", "Second-half plays:",
    "First qtr props:", "Fourth qtr props:", "Quarter 1 props:", "Half 1 props:",
    "Opening quarter props:", "Opening 5 minutes props:", "Through 5 minutes props:",
    "Tonight's picks (1Q, been hitting)", "My plays (all 1Q, went 4-0 last night)",
    "Card (1H, cashed yesterday)", "Locks (first quarter - this has hit 6 straight)",
    "1Q is the move", "1Q looks juicy tonight", "What I have for the 1Q", "1Q Is The Move",
    "Q1 props, cashed all 3 last night", "1H is where the value is:", "1Q Is the Move",
    "Who has the edge in the 1Q:", "1Q has been kind to these guys:",
])
def test_a_period_label_is_read_however_the_tout_titles_it(label, roster, schedule):
    """Round 4's label shape missed '1Q Locks', '1Q props I like', a negation
    anywhere on the line and more than six words before a colon, and the
    quarter picks under them were priced as full-game lines at q near 1
    (review 5, A1, B1, A8, A13). Refusing is the safe side: a line naming a
    period is its section's label unless it is a remark."""
    lines, report = _prep(f"{label}\n{_BODY}", roster, schedule)
    assert report["not_comparable"].get("not priced") == 2 and report["comparable"] == 0, label


def test_a_name_lines_period_goes_with_the_picks_under_it(roster, schedule):
    """'Jalen Brunson (1Q)' above his picks lost its period, so the quarter
    lines were priced as full-game ones, and 'Jalen Brunson 1Q:' lost the
    picks with a false message (review 5, A3). Then only his NAMELESS picks
    took it: 'Jalen Brunson 1Q props', 'Main Slate 1H props:' or 'Knicks
    Starters 1Q:' read as a name line, and the named quarter picks under them
    were priced as full-game lines (review 6, A1, B1). A name line cannot be
    told from such a label without the roster, so its period labels every
    pick under it as a label's does, past the next name line, until a label
    naming none ('Full game:'). Refusing Josh Hart's line there is the safe
    side, and the report says why (typed_period_label)."""
    for head in ("Jalen Brunson (1Q)", "Jalen Brunson 1Q:", "1Q: Jalen Brunson",
                 "Jalen Brunson - 1Q", "Jalen Brunson first quarter:", "Jalen Brunson (NYK, 1Q)",
                 "Jalen Brunson 1Q props", "JALEN BRUNSON 1Q PROPS", "Main Slate 1H props:",
                 "Knicks Starters 1Q:", "Late Games 1Q:", "What I Like 1Q:"):
        text = (f"{head}\nOver 6.5 points\nJalen Brunson over 1.5 assists\nJosh Hart\n"
                "Over 8.5 rebounds\nJosh Hart over 2.5 steals\nFull game:\n"
                "Josh Hart over 4.5 assists\n")
        lines, report = _prep(text, roster, schedule, now=FRIDAY_11AM)
        refused = lines[lines["refusal"].eq(props.REFUSE_PERIOD)]
        kinds = sorted(refused["stat_type"].str.split().str[0])
        # The nameless pick right under it takes the name line's name, or,
        # under a label that names nobody ("Main Slate" is heading noise), is
        # counted as naming no one (review 7).
        assert kinds == ["Assists", "Points", "Rebounds", "Steals"] or (
            kinds == ["Assists", "Rebounds", "Steals"]
            and report["parse"]["typed_noname"] == 1), (head, kinds)
        assert report["comparable"] == 1, head
        assert lines[lines["refusal"].isna()]["stat_type"].tolist() == ["Assists"], head
        msgs = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-07"})
        assert any("lines sat under a line that names a quarter, a half or the first minutes"
                   in m[1] for m in msgs), (head, msgs)
    # A name line saying "full game" ends it, as a label naming none does.
    lines, report = _prep("Jalen Brunson (1Q)\nOver 6.5 points\nJosh Hart (full game)\n"
                          "Over 8.5 rebounds\n", roster, schedule, now=FRIDAY_11AM)
    assert lines.loc["Josh Hart", "stat_type"] == "Rebounds" and report["comparable"] == 1


def test_a_game_headings_period_holds_over_its_own_teams(roster, schedule):
    """'Magic @ Suns (1H)', then 'Suns:' and 'Magic:': the period ended at
    the first team heading and the half lines were priced as full game; a
    parenthesis with a tip time beside the period was dropped whole; a game
    line with its period ('Magic @ Suns 1H:') ran on into the next game
    (review 5, B4, A19)."""
    for head in ("Magic @ Suns (1H)", "Magic @ Suns (1H, 9 PM ET)", "Magic @ Suns First Half",
                 "Magic @ Suns 1H:", "ORL-PHX (1H)", "Magic @ Suns first-half"):
        # Under it: the game's own team headings, a loose pair of the same
        # teams (review 6, A13) and a sub-label naming one of them ("Magic
        # players:", review 6, B9).
        for sub_a, sub_b in (("Suns:", "Magic:"), ("ORL-PHX", "Magic & Suns"),
                             ("Suns players:", "Orlando side:")):
            text = (f"{head}\n{sub_a}\nMiles Bridges over 12.5 points\n{sub_b}\n"
                    "Wendell Carter Jr. under 11.5 PRA\nHeat @ Warriors\n"
                    "Stephen Curry over 4.5 threes\n")
            lines, report = _prep(text, roster, schedule)
            assert lines.loc["Miles Bridges", "refusal"] == props.REFUSE_PERIOD, (head, sub_a)
            assert lines.loc["Wendell Carter Jr.", "refusal"] == props.REFUSE_PERIOD, (head, sub_b)
            assert lines.loc["Stephen Curry", "game_id"] == HEAT_AT_WARRIORS, head
            assert report["comparable"] == 1, (head, sub_a)
    # A sub-label naming another team is a section of its own, as before.
    lines, _ = _prep("Magic @ Suns (1H)\nWarriors players:\nStephen Curry over 4.5 threes\n",
                     roster, schedule)
    assert lines.loc["Stephen Curry", "stat_type"] == "3-PT Made"


def test_a_recap_naming_another_player_ends_the_name_line_above():
    """A recap line for Josh Hart between Jalen Brunson's picks lent
    Brunson's name to the line after it (review 5, A11, B7); a recap with no
    name of its own keeps the name line."""
    df = props.parse_typed_lines("Jalen Brunson\nOver 6.5 assists\nJosh Hart o8.5 reb - cashed\n"
                                 "Over 2.5 threes\n")
    assert list(zip(df["name"], df["stat_type"])) == [("Jalen Brunson", "Assists")]
    assert df.attrs["typed_recap"] == 1
    df = props.parse_typed_lines("Jalen Brunson\nOver 6.5 assists\nOver 2.5 threes - cashed\n"
                                 "Over 4.5 rebounds\n")
    assert list(zip(df["name"], df["stat_type"])) == [("Jalen Brunson", "Assists"),
                                                      ("Jalen Brunson", "Rebounds")]


def test_a_tout_tag_under_a_name_line_keeps_the_name_above():
    """'#NBA over 25.5 points' under 'Jalen Brunson' was dropped, and the
    visitor was told to write the full name (review 5, B10). A tag that may
    stand for a player still names nobody (review 4, B3)."""
    df = props.parse_typed_lines("Jalen Brunson\n#NBA over 25.5 points\n#LOCK over 6.5 assists\n"
                                 "@hoopspicks: over 2.5 threes\n#Knicks over 4.5 rebounds\n")
    assert list(zip(df["name"], df["stat_type"])) == [
        ("Jalen Brunson", "Points"), ("Jalen Brunson", "Assists"),
        ("Jalen Brunson", "3-PT Made"), ("Jalen Brunson", "Rebounds")]
    assert df.attrs["typed_noname"] == 0


def test_a_loose_pair_that_met_last_night_and_plays_again_today_is_todays_list(roster, schedule):
    """'Hawks & Wizards' the day after the two met, with both playing again
    today, was pinned to last night's game and called already started; two
    team headings were filed under today's games (review 5, B9)."""
    for heading in ("Hawks & Wizards", "ATL-WAS"):
        lines, report = _prep(f"{heading}\nDyson Daniels over 1.5 steals\n"
                              "Will Riley over 10.5 points\n", roster, schedule,
                              now=pd.Timestamp("2026-11-07T15:00:00Z").to_pydatetime())
        assert list(lines["game_id"]) == ["0022600190", "0022600192"], heading
        assert report["tipped"] == 0 and report["comparable"] == 2, heading


def test_a_quarter_board_copied_with_its_goblin_rungs_is_still_caught(roster, schedule,
                                                                      no_reliability):
    """One standard square and three Goblin rungs, all far below the
    full-game projections: Goblins never set the ratio, so the quarter board
    was priced as full game and led the discrepancies (review 5, B8). Two
    real Goblins and a pick are still no quarter board (review 4, B6)."""
    board = ("Miles Bridges\nPHX - F\nMiles Bridges\nvs ORL Thu 9:00pm\n6.5\nPoints\nLess\nMore\n"
             "Oso IghodaroGoblin\nPHX - F\nOso Ighodaro\nvs ORL Thu 9:00pm\n1.5\nRebounds\nMore\n"
             "Jevon CarterGoblin\nORL - G\nJevon Carter\n@ PHX Thu 9:00pm\n2.5\nRebounds\nMore\n"
             "Paolo BancheroGoblin\nORL - F\nPaolo Banchero\n@ PHX Thu 9:00pm\n3.5\nPoints\nMore\n")
    lines, _ = props.prepare_paste(board, roster, schedule, now=BEFORE_TIP)
    table, meta = props.compare(lines, _ByStat({"pts": Pmf.from_poisson(24.0)},
                                               Pmf.from_poisson(9.0)), _proj(roster))
    assert meta["period_board_suspect"] and meta["period_withheld"] == 4 and table.empty


# ── Review 6 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,stat,line,pick", [
    # A No in more words than one, after the pick's own question or first
    # after the line: read as a note, the board showed "Pick: More" (A2).
    ("Will Jalen Brunson score 26+ points? No way", "pts", 25.5, "less"),
    ("Will Miles Bridges record 7+ rebounds and assists? No chance", "ra", 6.5, "less"),
    ("Will Jalen Brunson score 26+ points? No shot", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? NO WAY", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? No way Jose", "pts", 25.5, "less"),
    ("Jalen Brunson 26+ points - no chance", "pts", 25.5, "less"),
    ("Jalen Brunson 26+ points? no way, fade", "pts", 25.5, "less"),
    ("Jalen Brunson over 25.5 points? No way", "pts", 25.5, "less"),
    ("Jalen Brunson under 25.5 points? No chance", "pts", 25.5, "more"),
    # Other words for No, and parentheses that answer (A6, A14).
    ("Will Mikal Bridges score 16+ points? Nope", "pts", 15.5, "less"),
    ("Will Jalen Brunson score 26+ points? Nah", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Not a chance", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Absolutely not", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Yeah", "pts", 25.5, "more"),
    ("Jalen Brunson 26+ points Nope", "pts", 25.5, "less"),
    ("Karl-Anthony Towns 12+ rebounds (No 2u)", "reb", 11.5, "less"),
    ("Miles Bridges 7+ R+A (No for me)", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (answer: No)", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (I say no)", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (No - fade)", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A (NO: fade)", "ra", 6.5, "less"),
    ("Jalen Brunson 26+ points (Nope)", "pts", 25.5, "less"),
    ("Jalen Brunson 26+ points No lol", "pts", 25.5, "less"),
    ("Jalen Brunson 26+ points No imo", "pts", 25.5, "less"),
    ("Miles Bridges 7+ R+A No fade", "ra", 6.5, "less"),
    ("No question: Jalen Brunson over 25.5 points", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, there's no way he misses", "pts", 25.5, "more"),
    ("Miles Bridges 7+ R+A - yes? no.", "ra", 6.5, "less"),
    # The pick's own question with a stat of several words or the game in
    # it: the answer was dropped (B1).
    ("Will Stephen Curry get 5+ 3-PT Made? No", "fg3m", 4.5, "less"),
    ("Stephen Curry 5+ Three Pointers Made? No", "fg3m", 4.5, "less"),
    ("Will Victor Wembanyama record 3+ blocked shots? No", "blk", 2.5, "less"),
    ("Stephen Curry 5+ 3-PT Made No - fade", "fg3m", 4.5, "less"),
    ("Will Jalen Brunson score 26+ points on Saturday? No", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points in his next game? No", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points for the Knicks? No", "pts", 25.5, "less"),
    ("Will Jalen Brunson record 10+ assists in Saturday's game? No", "ast", 9.5, "less"),
    ("Will Josh Hart get 11+ R + A? No", "ra", 10.5, "less"),
    ("Will Josh Hart grab 9+ rebounds? No shot", "reb", 8.5, "less"),
    # An all-caps NO opening a note after the stat is the note's (B3).
    ("JALEN BRUNSON OVER 25.5 POINTS, NO TOWNS TONIGHT", "pts", 25.5, "more"),
    ("JALEN BRUNSON OVER 25.5 POINTS NO MATTER WHAT", "pts", 25.5, "more"),
    ("JALEN BRUNSON UNDER 25.5 POINTS - NO MITCHELL ROBINSON", "pts", 25.5, "less"),
    ("Jalen Brunson over 25.5 points, NO Towns tonight", "pts", 25.5, "more"),
    ("ZION WILLIAMSON OVER 22.5 POINTS NO MATTER WHAT", "pts", 22.5, "more"),
    # Letters joined to a stat word by "+", "&" or "/", and capitals joined
    # by "and" (A5, B2, B14): priced as the first stat, or not at all.
    ("Josh Hart over 8.5 reb + a", "ra", 8.5, "more"),
    ("Jalen Brunson over 35.5 pts + reb + a", "pra", 35.5, "more"),
    ("Jalen Brunson over 35.5 Pts + R + A", "pra", 35.5, "more"),
    ("Jalen Brunson over 35.5 Pts & R & A", "pra", 35.5, "more"),
    ("Mikal Bridges over 20.5 pts + r", "pr", 20.5, "more"),
    ("Josh Hart over 10.5 Boards + A", "ra", 10.5, "more"),
    ("Josh Hart over 3.5 stl + b", "stocks", 3.5, "more"),
    ("Jalen Brunson Pts + A over 30.5", "pa", 30.5, "more"),
    ("Josh Hart over 10.5 R and A", "ra", 10.5, "more"),
    ("Jalen Brunson over 35.5 P, R and A", "pra", 35.5, "more"),
    # A word that changes the stat, spelled out (A7): the stat written.
    ("Karl-Anthony Towns over 2.5 rebounds (OREBs)", "oreb", 2.5, "more"),
    ("Jalen Brunson over 32.5 points (inc. assists)", "pa", 32.5, "more"),
    ("Jalen Brunson over 32.5 points (along with assists)", "pa", 32.5, "more"),
    ("Jalen Brunson over 32.5 points (alongside assists)", "pa", 32.5, "more"),
    ("Jalen Brunson over 35.5 points (incl. rebounds and assists)", "pra", 35.5, "more"),
    ("Jalen Brunson over 35.5 points (counting rebounds and assists)", "pra", 35.5, "more"),
    ("Jalen Brunson over 32.5 points (includes assists)", "pa", 32.5, "more"),
])
def test_a_market_answer_and_a_stat_read_as_written(text, stat, line, pick):
    """Each was read with the other side, or as another stat, at q near 1
    (review 6). The visitor's No turns the pick; a note's words never do."""
    r = _one(text)
    assert (r["stat"], r["line"], r["pick"]) == (stat, line, pick), (text, r["stat_type"])


@pytest.mark.parametrize("text", [
    # A No opening the line beside a written side: a reply or a pass (B5).
    "No, I like Jalen Brunson over 25.5 points", "No play: Jalen Brunson over 25.5 points",
    "No bet on Jalen Brunson over 25.5 points", "No, I'm taking Josh Hart under 8.5 rebounds",
    "No - Jalen Brunson over 25.5 points",
    # Beside a written side, a No with more words after it, or in a
    # parenthesis with a clause, argues either side (B6).
    "Jalen Brunson over 12.5 points (no, not a typo)", "Jalen Brunson over 25.5 points, no for real",
    "Jalen Brunson over 25.5 points - no but seriously, hammer it",
    "Jalen Brunson over 25.5 points. No, I'm not kidding",
    "Josh Hart over 8.5 rebounds No, he's been cold", "Josh Hart over 8.5 rebounds No he's cold",
    "Jalen Brunson over 25.5 points (No - Towns out)",
    # A No in more words than one with a clause after it: "No way he misses"
    # means Yes (A2).
    "Will Jalen Brunson score 26+ points? No way he does",
    "Jalen Brunson over 25.5 points, no way he misses",
    # The note's own question, answered, on a line with no written side.
    "Miles Bridges 7+ R+A - is Towns playing? No",
    # In capitals with a comma after it, a reply as well; a word after the
    # pick's own answer, or a lone No after a note beside a written side.
    "NO, I LIKE JALEN BRUNSON OVER 25.5 POINTS",
    "Will Jalen Brunson score 26+ points? No Towns though",
    "Jalen Brunson over 25.5 points, great spot tonight. No",
])
def test_a_side_that_cannot_be_told_is_counted_never_guessed(text, roster, schedule):
    """Read as the market's No, each showed 'Pick: Less' beside a written
    Over, or the opposite for the market's own question (review 6, B5, B6).
    Which side it takes cannot be told from the words, so it is counted and
    said, never guessed."""
    df = props.parse_typed_lines(text)
    assert df.empty and df.attrs["typed_unclear"] == 1, (text, df.attrs)
    lines, report = _prep(text, roster, schedule, now=FRIDAY_11AM)
    assert report["parse"]["typed_unclear"] == 1, text
    said = props_ui.report_messages({**report, "nonempty": True, "view_date": "2026-11-06"})
    assert said == [("warning", "1 line in that paste reads like a pick, but which side it "
                     "takes could not be told (a Yes or No that may belong to a note), so nothing "
                     "was added. Write the side you take, as in Jalen Brunson under 25.5 points.")]


def test_the_unclear_and_period_label_notes_are_plural_and_plain():
    """The two new notes of review 6, worded for one line and for several,
    with no dash or emoji (the site's style). The period note names the
    first minutes too, and never counts more lines than were not priced
    (review 7, A13)."""
    for n, unclear, plabel in ((1, "1 line reads like a pick, but which side it takes",
                                "1 line sat under a line that names a quarter, a half or the "
                                "first minutes (such as 1Q props), so it was read as a pick for "
                                "that period"),
                               (3, "3 lines read like picks, but which side they take",
                                "3 lines sat under a line that names a quarter, a half or the "
                                "first minutes (such as 1Q props), so they were read as picks for "
                                "that period")):
        msgs = props_ui.report_messages({"parse": {"typed_unclear": n, "typed_period_label": n},
                                         "not_comparable": {"not priced": n}, "kept": 5,
                                         "comparable": 2, "nonempty": True})
        text = [m[1] for m in msgs]
        assert any(t.startswith(unclear) for t in text), text
        assert any(t.startswith(plabel) for t in text), text
        for t in text:
            assert EM not in t and EN not in t and t.isascii(), t
    # Two period picks that the paste merged into one line: one is said.
    msgs = props_ui.report_messages({"parse": {"typed_period_label": 2}, "kept": 1,
                                     "not_comparable": {"not priced": 1}, "comparable": 0,
                                     "nonempty": True})
    assert any(m[1].startswith("1 line sat under a line that names") for m in msgs), msgs


@pytest.mark.parametrize("text", [
    "Jalen Brunson over 6.5 points (first qtr)", "Jalen Brunson over 6.5 points - 1st-quarter",
    "Jalen Brunson over 12.5 points (first-half)", "Jalen Brunson first-half over 12.5 points",
    "Jalen Brunson (first-half)\nOver 12.5 points",
    "Knicks @ Cavs first-half\nJalen Brunson over 12.5 points",
    "Jalen Brunson over 6.5 points (Q1 only)", "Jalen Brunson over 6.5 points, opening quarter",
])
def test_a_period_written_on_the_pick_however_it_is_spelled(text, roster, schedule):
    """A hyphen, 'qtr' or 'opening' was no period, and the half and quarter
    lines were priced as full game (review 6, B4)."""
    lines, report = _prep(text, roster, schedule, now=FRIDAY_11AM)
    assert list(lines["refusal"]) == [props.REFUSE_PERIOD] and report["comparable"] == 0, text


def test_a_result_is_counted_and_a_hope_is_still_a_pick(roster, schedule):
    """A result in a parenthesis or alone after the stat was filed as a pick
    for the next game (review 6, A8, A16); 'it cashes tonight' was counted as
    a result (B13)."""
    for text in ("Jalen Brunson over 25.5 points (cashed it)",
                 "Jalen Brunson over 25.5 points (it cashed)",
                 "Jalen Brunson over 25.5 points (won easily)",
                 "Jalen Brunson over 25.5 points (W)",
                 "Josh Hart under 8.5 rebounds (L)", "Josh Hart under 8.5 rebounds (push)",
                 "Jalen Brunson over 25.5 points (hit)", "Jalen Brunson over 25.5 points - hit",
                 "Jalen Brunson over 25.5 points - easy dub", "Jalen Brunson over 25.5 points: HIT",
                 "Josh Hart under 8.5 rebounds - loss",
                 "Josh Hart under 8.5 rebounds - barely missed",
                 "Josh Hart under 8.5 rebounds - tough one, lost",
                 "Josh Hart under 8.5 rebounds - just missed"):
        df = props.parse_typed_lines(text)
        assert df.empty and df.attrs["typed_recap"] == 1, text
    for text in ("Jalen Brunson over 25.5 points, it cashes tonight",
                 "Josh Hart over 8.5 rebounds - this line cashes often",
                 "Jalen Brunson over 25.5 points (won 3 straight)",
                 "Jalen Brunson over 25.5 points - hit this"):
        assert _one(text)["stat"] in ("pts", "reb"), text


def test_a_recap_of_the_name_lines_own_player_keeps_the_name(roster, schedule):
    """'Brunson o25.5 pts - cashed' under 'Jalen Brunson' ended his name line,
    and his next pick vanished with no count (review 6, B10). A pick with no
    name and no name line above it is counted now."""
    for recap in ("Brunson o25.5 pts - cashed", "Jalen Brunson o25.5 pts last night - cashed",
                  "Last night: Brunson o25.5 pts cashed"):
        df = props.parse_typed_lines(f"Jalen Brunson\nOver 6.5 assists\n{recap}\nOver 2.5 threes\n")
        assert list(zip(df["name"], df["stat_type"])) == [("Jalen Brunson", "Assists"),
                                                          ("Jalen Brunson", "3-PT Made")], recap
    df = props.parse_typed_lines("Jalen Brunson\nOver 6.5 assists\nJB o25.5 pts - cashed\n"
                                 "Over 2.5 threes\n")
    assert list(df["stat_type"]) == ["Assists"]
    assert df.attrs["typed_recap"] == 1 and df.attrs["typed_noname"] == 1


def test_a_players_handle_is_never_decoration():
    """'@Money23Green' (Draymond Green), '@HarrisonBarnes' and '#Pickett'
    hold a part of a tout's tag, so their lines were priced as the name
    line's player, and '#Pickett' replaced his pick (review 6, A4, B7). A tag
    is decoration only when it is all decoration, or a team's own hashtag
    (A11); '#potd' and '#PrizePicks' still are (A17)."""
    for tag in ("@Money23Green", "@HarrisonBarnes", "#Pickett", "@JalenPickett",
                "@MarvinBagleyIII", "#Steinbach", "@KingJames", "@PJWashington",
                "@SunsDevinBooker"):
        df = props.parse_typed_lines(f"Stephen Curry\nOver 4.5 threes\n{tag} over 6.5 assists\n")
        assert list(zip(df["name"], df["stat_type"])) == [("Stephen Curry", "3-PT Made")], tag
        assert df.attrs["typed_noname"] == 1, tag
    for tag in ("#potd", "#PrizePicks", "#BOL", "#fire", "@hoops_daily", "#NBAPicks",
                "#DubNation", "#LakeShow", "#BleedGreen", "#ThunderUp", "#GoSpursGo",
                "#KnicksTape"):
        df = props.parse_typed_lines(f"Stephen Curry\n{tag} over 4.5 threes\n")
        assert list(zip(df["name"], df["stat_type"])) == [("Stephen Curry", "3-PT Made")], tag
        assert df.attrs["typed_noname"] == 0, tag


@pytest.mark.parametrize("text,game", [
    # A national network before a city (B8).
    ("Tonight on ESPN Washington at Atlanta", ("WAS", "ATL")),
    ("ESPN Washington @ Atlanta", ("WAS", "ATL")),
    ("TNT Philadelphia at Cleveland", ("PHI", "CLE")),
    ("Warriors @ Kings on NBC Sports Bay Area", ("GSW", "SAC")),
    # Spellings of the tip, the Cup and the network outside round 5's (A9,
    # A17).
    ("Magic @ Suns tip 9 ET", ("ORL", "PHX")), ("Magic @ Suns tipoff 9 PM ET", ("ORL", "PHX")),
    ("Magic @ Suns - tip-off 9 PM", ("ORL", "PHX")),
    ("Magic @ Suns | Tip: 9 PM ET", ("ORL", "PHX")),
    ("Magic @ Suns - NBA Cup East Group B", ("ORL", "PHX")),
    ("Magic @ Suns - Cup Group C", ("ORL", "PHX")),
    ("Magic @ Suns ESPN/ESPN+", ("ORL", "PHX")), ("Magic @ Suns 9 o'clock", ("ORL", "PHX")),
    ("Magic @ Suns 9ish", ("ORL", "PHX")), ("Magic @ Suns 7:00 Arizona", ("ORL", "PHX")),
    ("Magic @ Suns 21h", ("ORL", "PHX")), ("Magic @ Suns 11.5.26", ("ORL", "PHX")),
    ("Magic @ Suns 7 PM Phoenix time", ("ORL", "PHX")),
    ("Magic @ Suns tip time 9 ET", ("ORL", "PHX")),
    ("Knicks @ Cavs on MSG", ("NYK", "CLE")), ("Knicks @ Cavs on MSG Networks", ("NYK", "CLE")),
    ("Heat @ Magic on FanDuel Sports Network Sun", ("MIA", "ORL")),
])
def test_a_game_heading_is_read_through_its_noise(text, game):
    assert props._typed_heading(text) == game


def test_a_network_before_a_city_leaves_the_late_list_late(roster, schedule):
    """'Tonight on ESPN Washington at Atlanta' lost Washington, so the list
    for the game under way was filed under Saturday's and called loaded
    (review 6, B8)."""
    lines, report = _prep("Tonight on ESPN Washington at Atlanta\nDyson Daniels over 1.5 steals\n"
                          "Will Riley over 10.5 points\n", roster, schedule, now=FRIDAY_LATE)
    assert set(lines["route_how"]) == {"tipped"} and report["comparable"] == 0


# ── Review 7 ─────────────────────────────────────────────────────────────────
HALF = chr(0xBD)
KEYCAP_1Q = "1" + chr(0xFE0F) + chr(0x20E3) + "Q"


@pytest.mark.parametrize("label", [
    # A tip time or a count made the label pick-shaped, and its other words
    # read as a name, so the label was filed as a refused row (A1).
    "What I Like 1Q 7 PM:", "Knicks Starters 1Q 7 PM:", "Late Games First Quarter (10 PM):",
    "Main Slate 1Q 3 picks:", "Big Board 1Q 2 Plays", "Late Games 1H 4 Plays:",
    "Hot Hands 1Q 5 Star:", "Late Slate 1H 10 PM:",
    # The sportsbook's spellings and glued labels (A3).
    f"1st {HALF} props:", "1st 1/2 props:", "Q-1 props:", "First 1/2:", "1st Qrtr props",
    "1st Qrt props:", "1Qs:", "1st hlf props:", "Quarter One props:", "Half One props:",
    "First Frame props:", "First twelve minutes:", f"{KEYCAP_1Q} props:", "#1QProps",
    "1Q_props", "#FirstQuarter", "FIRSTQUARTER:", "1Qprops:",
    # A negation of the word after the period, not of the period (A7).
    "No 1Q unders, only overs:", "No 1Q parlays, singles only:",
    "Not taking 1Q unders, overs only:", "No 1Q parlays:", "No 1Q fades tonight:",
    # Colon labels that would be remarks without their colon (A11).
    "Knicks have been great in the 1Q:", "Brunson has owned the 1Q lately:",
    "They are elite in the 1H:", "Q4 Brunson is different:",
])
def test_a_period_label_is_read_with_its_numbers_and_spellings(label, roster, schedule):
    """Each priced the quarter or half picks under it as full-game lines at q
    near 1 (review 7, A1, A3, A7, A11)."""
    lines, report = _prep(f"{label}\n{_BODY}", roster, schedule)
    assert report["not_comparable"].get("not priced") == 2 and report["comparable"] == 0, label
    assert report["unrouted"] == 0, label           # the label line is no row of its own


@pytest.mark.parametrize("head", [
    "Jalen Brunson (1Q, 7 PM)", "Jalen Brunson 1Q 7 PM", "Jalen Brunson (first quarter, 7 PM)",
    "Jalen Brunson 1H - 8 PM ET", "Jalen Brunson (1Q) 7 PM", "Jalen Brunson first 5 minutes",
])
def test_a_name_lines_period_holds_with_a_tip_time_beside_it(head, roster, schedule):
    """The tip time made the name line a refused row, so the period never
    reached the picks under it (review 7, A1)."""
    lines, report = _prep(f"{head}\nOver 6.5 points\nJalen Brunson over 1.5 assists\n", roster,
                          schedule, now=FRIDAY_11AM)
    assert list(lines["refusal"]) == [props.REFUSE_PERIOD] * 2 and report["comparable"] == 0, head
    assert set(lines["name"]) == {"Jalen Brunson"}, head


@pytest.mark.parametrize("text,stat,line,pick", [
    # A spaced separator between stat words and letters joins them (A2).
    ("Jalen Brunson over 35.5 Pts / R / A", "pra", 35.5, "more"),
    ("Josh Hart over 10.5 Reb / A", "ra", 10.5, "more"),
    ("Mikal Bridges over 30.5 Pts / A", "pa", 30.5, "more"),
    ("OG Anunoby over 2.5 Stl / B", "stocks", 2.5, "more"),
    (f"Jalen Brunson over 35.5 Pts {BUL} R {BUL} A", "pra", 35.5, "more"),
    ("Jalen Brunson over 35.5 Pts | R | A", "pra", 35.5, "more"),
    (f"Jalen Brunson over 35.5 Pts{BUL}R{BUL}A", "pra", 35.5, "more"),
    ("Jalen Brunson Pts / R / A over 35.5", "pra", 35.5, "more"),
    ("Jalen Brunson 36+ Pts / R / A", "pra", 35.5, "more"),
    # ...and a separated row is still read as fields.
    ("Miles Bridges / R+A / 6.5", "ra", 6.5, None),
    (f"Miles Bridges {BUL} R+A {BUL} 6.5", "ra", 6.5, None),
    ("Jalen Brunson | Pts | 25.5 | Over", "pts", 25.5, "more"),
    ("Josh Hart over 8.5 REB + A", "ra", 8.5, "more"),
    # Other words for No, and "Yeah, no" (A5, A6).
    ("Will Jalen Brunson score 26+ points? Yeah, no. Fade", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Yeah, no", "pts", 25.5, "less"),
    ("Miles Bridges 7+ R+A yeah no", "ra", 6.5, "less"),
    ("Will Jalen Brunson score 26+ points? Yeah no way", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Yeah, no doubt", "pts", 25.5, "more"),
    ("Will Jalen Brunson score 26+ points? There's no way", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? There is no way", "pts", 25.5, "less"),
    ("Will Josh Hart grab 9+ rebounds? Nooo", "reb", 8.5, "less"),
    ("Will Jalen Brunson score 26+ points? Nahh", "pts", 25.5, "less"),
    ("Miles Bridges 7+ R+A? Never", "ra", 6.5, "less"),
    ("Will Jalen Brunson score 26+ points? Fat chance", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Doubt it", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Certainly not", "pts", 25.5, "less"),
    ("Miles Bridges 7+ R+A? Not a shot", "ra", 6.5, "less"),
    ("Jalen Brunson over 25.5 points? Fade", "pts", 25.5, "less"),
    ("Will Jalen Brunson score 26+ points? Absolutely", "pts", 25.5, "more"),
    ("Will Jalen Brunson score 26+ points? Lock it in", "pts", 25.5, "more"),
    ("Will Jalen Brunson score 26+ points? vs MIA", "pts", 25.5, "more"),
    # The stat written, spelled Basketball-Reference's way (A8).
    ("Karl-Anthony Towns over 7.5 rebs (DRB)", "dreb", 7.5, "more"),
    ("Josh Hart over 2.5 rebounds (ORBs)", "oreb", 2.5, "more"),
    # Quinten Post's surname is no name-tail word (A10).
    ("Quinten Post over 10.5 points", "pts", 10.5, "more"),
    ("Will Quinten Post score 11+ points? No", "pts", 10.5, "less"),
])
def test_review_7_reads_the_stat_and_the_side_as_written(text, stat, line, pick):
    r = _one(text)
    assert (r["stat"], r["line"], r["pick"]) == (stat, line, pick), (text, r["stat_type"])
    assert "Post" not in text or r["name"] == "Quinten Post"


@pytest.mark.parametrize("text", [
    # Letters after a comma, an article "A" in capitals (A2, A4).
    "Jalen Brunson over 35.5 Pts, R, A", "Josh Hart over 10.5 Reb, A",
    "JALEN BRUNSON OVER 25.5 POINTS + A LOCK", "JOSH HART OVER 8.5 REBOUNDS + A LOCK",
    "Jalen Brunson over 25.5 Points + A Lock Tonight", "Jalen Brunson over 25.5 PTS & A LOCK",
    "Jalen Brunson over 25.5 points + A+ spot",
    # Words that change the stat (A8, A11).
    "Josh Hart over 2.5 rebs, ORB", "Stephen Curry over 9.5 threes (tries)",
    "Jalen Brunson over 12.5 assists (potential)", "Josh Hart over 8.5 rebounds, chances",
    "Jalen Brunson over 12.5 points, from three", "Jalen Brunson over 12.5 points (from 3)",
    "Jalen Brunson over 12.5 points, from the line", "Jalen Brunson over 12.5 points (3s only)",
    "Jalen Brunson over 2.5 steals (deflections)", "Jalen Brunson over 4.5 points, all bench",
    "Josh Hart over 2.5 rebounds, mostly off", "Jalen Brunson over 25.5 points, a per game thing",
    "Jalen Brunson over 4.5 points, mainly bench",
])
def test_review_7_never_prices_a_stat_it_cannot_read(text):
    r = _one(text)
    assert r["stat"] is None and props.classify_stat(r["stat_type"])[1] == "unmapped", text


@pytest.mark.parametrize("text", [
    "Will Jalen Brunson score 26+ points? He's been hot",
    "Will Jalen Brunson score 26+ points? Let's see",
    "Miles Bridges 7+ R+A Yes No",
])
def test_words_after_the_picks_question_that_say_no_side_are_counted(text):
    """'? Fat chance' and '? Never' kept the implied More (review 7, A6). An
    answer the reader cannot place is counted, and so are a copied market's
    two buttons (A5)."""
    df = props.parse_typed_lines(text)
    assert df.empty and df.attrs["typed_unclear"] == 1, (text, df.attrs)


def test_a_period_on_the_pick_in_a_sportsbooks_spelling(roster, schedule):
    for text in (f"Jalen Brunson over 12.5 points (1st {HALF})",
                 "Jalen Brunson 1st 1/2 over 12.5 points"):
        lines, report = _prep(text, roster, schedule, now=FRIDAY_11AM)
        assert list(lines["refusal"]) == [props.REFUSE_PERIOD], text


def test_a_negated_period_still_ends_its_own_section(roster, schedule):
    """A negation that denies the period's picks still cancels it (review 4,
    B2); only one denying another word keeps the label (review 7, A7)."""
    for label in ("No 1Q today", "No 1Q props today, full game only", "Not betting the 1H tonight:"):
        lines, report = _prep(f"{label}\n{_BODY}", roster, schedule)
        assert report["comparable"] == 2, label


@pytest.mark.parametrize("text,game", [
    ("Magic @ Suns on KTVK", ("ORL", "PHX")), ("Magic @ Suns on Suns Live", ("ORL", "PHX")),
    ("Magic @ Suns | late game", ("ORL", "PHX")), ("Magic @ Suns on Max", ("ORL", "PHX")),
    ("Magic @ Suns on Bally", ("ORL", "PHX")),
    # Region suffixes beyond Bay Area and Sun (A12).
    ("Heat @ Magic on FanDuel Sports Network Florida", ("MIA", "ORL")),
    ("Cavs @ Knicks on FanDuel Sports Network Ohio", ("CLE", "NYK")),
    ("Pistons @ Mavs on FDSN Detroit", ("DET", "DAL")),
])
def test_review_7_heading_noise(text, game):
    assert props._typed_heading(text) == game


def test_a_two_word_team_sub_label_keeps_the_games_period(roster, schedule):
    """'Golden State starters:' under 'Heat @ Warriors (1H)': the two-word
    team names in _names_team were unpinned (review 7, A12)."""
    lines, report = _prep("Heat @ Warriors (1H)\nGolden State starters:\n"
                          "Stephen Curry over 2.5 threes\n", roster, schedule)
    assert list(lines["refusal"]) == [props.REFUSE_PERIOD] and report["comparable"] == 0


@pytest.mark.parametrize("text,stat,line,pick", [
    # A No that a clause after it may turn, only beside an implied side when
    # the clause holds no second negation and no word of the Yes (B6).
    ("Mikal Bridges 16+ points No, he's been cold", "pts", 15.5, "less"),
    ("Miles Bridges 7+ R+A No - Brunson cold", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A No. Yes on Hart though", "ra", 6.5, "less"),
    ("Miles Bridges 7+ R+A No at 38c", "ra", 6.5, "less"),
    # A No inside a word with a digit is no answer (B4).
    ("Jalen Brunson over 25.5 points (no B2B)", "pts", 25.5, "more"),
    ("Josh Hart under 8.5 rebounds (no b2b)", "reb", 8.5, "less"),
    ("Jalen Brunson 26+ points (no B2B)", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (no 76ers)", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (No. 1 option tonight)", "pts", 25.5, "more"),
    # Letters after letters are the combo; "+EV" and "& more" are notes (B3,
    # B17).
    ("Miles Bridges over 6.5 R + A lock it in", "ra", 6.5, "more"),
    ("Jalen Brunson over 35.5 P + R + A, lets ride", "pra", 35.5, "more"),
    ("Josh Hart over 8.5 rebounds +EV", "reb", 8.5, "more"),
    ("Jalen Brunson over 25.5 Pts + 2u", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points +", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points & more", "pts", 25.5, "more"),
    # Matchup and usage notes beside an offensive or defensive word (B16).
    ("Jalen Brunson over 25.5 points, weak defensive team tonight", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, bad defensive matchup", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, elite offensive rating", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, Hawks rank 29th defensively", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points - he attacks the rim", "pts", 25.5, "more"),
    ("Stephen Curry over 4.5 threes - shots are falling", "fg3m", 4.5, "more"),
    ("Jalen Brunson over 25.5 points, great transition team", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points, lives at the rim", "pts", 25.5, "more"),
    ("Jalen Brunson 26+ points? Yes. No Towns, more shots", "pts", 25.5, "more"),
    # A note about the player's games, a hope, a bet (B19).
    ("Josh Hart over 8.5 rebounds (missed last game)", "reb", 8.5, "more"),
    ("Jalen Brunson over 25.5 points (missed shootaround)", "pts", 25.5, "more"),
    ("Josh Hart over 8.5 rebounds - missed last game", "reb", 8.5, "more"),
    ("Jalen Brunson over 25.5 points (hit it)", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (hit this one)", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (W tonight)", "pts", 25.5, "more"),
    ("Jalen Brunson over 25.5 points (push it)", "pts", 25.5, "more"),
])
def test_review_7_notes_that_say_no_side_and_change_no_stat(text, stat, line, pick):
    r = _one(text)
    assert (r["stat"], r["line"], r["pick"]) == (stat, line, pick), (text, r["stat_type"])


@pytest.mark.parametrize("text", [
    "JALEN BRUNSON OVER 25.5 POINTS + A LOCK", "Jalen Brunson Over 25.5 Points & A Lock",
    "Jalen Brunson over 25.5 PTS & A FREE PLAY", "Josh Hart over 8.5 Reb + A Lock",
    "Jalen Brunson over 25.5 points/a lock", "Karl-Anthony Towns over 2.5 rebounds, exclusively offensive",
    "Jalen Brunson over 12.5 points, at the rim",
    "Jalen Brunson over 12.5 points, in transition", "Stephen Curry over 9.5 threes, shots taken",
])
def test_review_7_a_letter_that_may_be_an_article_is_never_assists(text):
    """'POINTS + A LOCK' was priced as Pts+Asts in capitals (review 7, B3)."""
    r = _one(text)
    assert r["stat"] is None and props.classify_stat(r["stat_type"])[1] == "unmapped", text


@pytest.mark.parametrize("text", [
    # A No that opens a Yes beside an implied side (B6).
    "Josh Hart 9+ rebounds no there's no way he misses", "Jalen Brunson 26+ points no you can't fade him",
    "Jalen Brunson 26+ points no never doubt him", "Josh Hart 9+ rebounds no it is a lock",
    "Jalen Brunson 26+ points no your boy is cooking",
    # An idiom beside a written side with no question (B7).
    "Josh Hart under 8.5 rebounds no shot", "Jalen Brunson over 25.5 points - no way",
    "Jalen Brunson over 25.5 points (no way)", "JALEN BRUNSON OVER 25.5 POINTS NO WAY",
    "Josh Hart under 8.5 rebounds (no chance)",
    # A No closed by the Yes's words, or beside both sides, next to a
    # written side (the author's own regressions in round 7).
    "Jalen Brunson over 25.5 points nope, lock it", "Josh Hart under 8.5 rebounds nope, lock it",
    "Jalen Brunson over 25.5 points, no chance he goes under",
    # A corrected answer (B8).
    "Will Jalen Brunson score 26+ points? No... actually yes",
    "Will Jalen Brunson score 26+ points? Yes. No wait, no", "Jalen Brunson 26+ points No, wait, yes",
])
def test_review_7_a_side_turned_by_its_own_words_is_counted(text):
    df = props.parse_typed_lines(text)
    assert df.empty and df.attrs["typed_unclear"] == 1, (text, df.attrs)


def test_chatter_with_a_no_is_no_unclear_pick_and_a_nameless_one_keeps_the_name(roster, schedule):
    """'No, Brunson had 40 last night' was counted as a pick whose side could
    not be told (review 7, B12), and an unclear line under a name line ended
    the name block, so the clean picks after it were refused (B13)."""
    for text in ("No, Brunson had 40 last night", "Nope, he only had 12 last night"):
        df = props.parse_typed_lines(text)
        assert df.empty and not df.attrs.get("typed_unclear"), text
    df = props.parse_typed_lines("Jalen Brunson\nOver 25.5 points\nOver 6.5 assists, nah he's a lock\n"
                                 "Over 2.5 threes\n")
    assert list(zip(df["name"], df["stat_type"])) == [("Jalen Brunson", "Points"),
                                                      ("Jalen Brunson", "3-PT Made")]
    assert df.attrs["typed_unclear"] == 1 and df.attrs["typed_noname"] == 0


@pytest.mark.parametrize("text", [
    "Jalen Brunson over 25.5 points CASHES", "Jalen Brunson o25.5 pts cashes again",
    "Josh Hart under 8.5 rebounds cashes", "Jalen Brunson over 25.5 points (cashes)",
    "Jalen Brunson over 25.5 points - this one cashes",
])
def test_a_present_tense_result_is_a_recap(text):
    """'... CASHES' with a check mark is last night's winner, filed as a pick
    for the next game (review 7, B5). With a hope ('tonight', 'often') it is
    a pick."""
    df = props.parse_typed_lines(text)
    assert df.empty and df.attrs["typed_recap"] == 1, (text, df.attrs)


def test_a_full_game_name_line_ends_only_its_own_period(roster, schedule):
    """'Josh Hart (full game)' under '1Q props:' ended the whole quarter
    section, and the next quarter line was priced as full game (review 7,
    B2); a sub-label saying 'full game' ends a heading's period (B22)."""
    for head in ("1Q props:", "Knicks @ Cavaliers (1Q)"):
        lines, report = _prep(f"{head}\nJalen Brunson over 6.5 points\nJosh Hart (full game)\n"
                              "Over 8.5 rebounds\nMikal Bridges over 3.5 points\n", roster, schedule,
                              now=FRIDAY_11AM)
        assert lines.loc["Josh Hart", "stat_type"] == "Rebounds", head
        assert lines.loc["Mikal Bridges", "refusal"] == props.REFUSE_PERIOD, head
        assert lines.loc["Jalen Brunson", "refusal"] == props.REFUSE_PERIOD, head
    lines, report = _prep("Magic @ Warriors (1H)\nPaolo Banchero over 10.5 points\nMagic full game:\n"
                          "Paolo Banchero over 22.5 points\n", roster, schedule)
    assert sorted(lines["stat_type"]) == ["Points", "Points 1h"]


def test_a_recap_of_another_player_with_the_same_surname_ends_the_name_line():
    """'Jaylin Williams o6.5 reb - cashed' under 'Jalen Williams' kept his name
    line, and the next nameless pick was his (review 7, B9)."""
    for head, recap in (("Jalen Williams", "Jaylin Williams o6.5 reb - cashed"),
                        ("Wendell Carter Jr.", "Jevon Carter o1.5 threes - cashed"),
                        ("Josh Hart", "Brunson o25.5 pts cashed (Hart assisted)")):
        df = props.parse_typed_lines(f"{head}\nOver 6.5 rebounds\n{recap}\nOver 4.5 assists\n")
        assert list(zip(df["name"], df["stat_type"])) == [(head, "Rebounds")], recap
        assert df.attrs["typed_noname"] == 1, recap


def test_a_tag_is_decoration_only_with_a_touts_word_in_it():
    """'#TheKing' and '#Buckets' are a player's nickname: decoration, they were
    lent the name above (review 7, B10). '#PlayerProps' and '#LockOfTheDay'
    are decoration (B18); 'Post' is Quinten Post's surname (B1)."""
    for text in ("Joel Embiid\nOver 30.5 points\n#TheKing over 7.5 assists",
                 "Joel Embiid\nOver 30.5 points\n#Buckets over 7.5 assists",
                 "Joel Embiid\nOver 30.5 points\n@King23 over 7.5 assists",
                 "Zach Edey\nOver 10.5 rebounds\nPost over 6.5 rebounds"):
        df = props.parse_typed_lines(text)
        assert len(df) == 1 and df.attrs["typed_noname"] == 1, text
    for tag in ("#PlayerProps", "#LockOfTheDay", "#BetOfTheDay", "#MoneyMaker", "#LetsCash",
                "#NBAPlayerProps"):
        df = props.parse_typed_lines(f"Joel Embiid\n{tag} over 30.5 points\n")
        assert list(df["name"]) == ["Joel Embiid"] and df.attrs["typed_noname"] == 0, tag


@pytest.mark.parametrize("text,game", [
    ("Wizards at Hawks on NBC Washington", ("WAS", "ATL")),
    ("76ers @ Cavaliers on NBC Philadelphia", ("PHI", "CLE")),
    ("Celtics @ Hornets on NBC Boston", ("BOS", "CHA")),
    ("Tonight on ESPN Washington at Atlanta", ("WAS", "ATL")),
])
def test_a_networks_city_station_is_read_both_ways(text, game):
    """NBC's own stations brand by city, so round 6's fix for 'ESPN
    Washington' lost 'on NBC Washington' (review 7, B11)."""
    assert props._typed_heading(text) == game


@pytest.mark.parametrize("label,priced", [
    # A parenthesis putting its period off, the word "minor", the verbs a
    # negation reaches across, a hyphenated season note (B20).
    ("My card (1H later, full game now)", True), ("Tonight's picks (1Q tomorrow, full game tonight)", True),
    ("Through 3 minor injuries he still plays", True), ("Not chasing 1H props tonight", True),
    ("Not posting 1Q props tonight", True), ("Second-half of the season Brunson", True),
    # More spellings of a quarter (B23).
    ("First Q props:", False), ("Second Q plays:", False), ("Third Q:", False), ("Q 1 props:", False),
    ("Qtr1 props:", False), ("Quarter1 props:", False), ("1QProps", False), ("Q1Props:", False),
    ("1st period props:", False), ("Opening period props:", False),
])
def test_review_7_labels_and_their_spellings(label, priced, roster, schedule):
    lines, report = _prep(f"{label}\n{_BODY}", roster, schedule)
    if priced:
        assert report["comparable"] == 2, label
    else:
        assert report["not_comparable"].get("not priced") == 2 and report["comparable"] == 0, label


def test_the_period_label_note_advises_taking_the_period_off():
    """'paste them without that line' led a name line's visitor to a
    nameless refusal (review 7, B21)."""
    msgs = props_ui.report_messages({"parse": {"typed_period_label": 2}, "kept": 2,
                                     "not_comparable": {"not priced": 2}, "comparable": 0,
                                     "nonempty": True})
    said = [m[1] for m in msgs if "sat under a line" in m[1]]
    assert said and said[0].endswith("take the period off that line and paste again."), msgs
