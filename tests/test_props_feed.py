"""bucketlib.props' feed path (design 2.5 and 14 steps 12 and 13; layout B,
amendments A1, A2, A4 and A9): prepare_feed, resolve_event_names, compare()'s
new columns, compare_books, feed_board, american_breakeven and side_words,
plus the contracts between props and sources.oddsapi.

The feed frames come from the REAL oddsapi parser (parse_events, match_events,
parse_event_odds and merge_refresh, as the live layer runs them) on the
CONSTRUCTED fixtures in tests/fixtures/oddsapi (see PROVENANCE.txt: invented
lines on the serve fixture's real 2026-03-12 slate), against that date's real
roster and projections in tests/fixtures/serve. The visitor's paste is the
serve fixture's board text through the real paste path (prepare_paste, the
token board, its browser copy and the restore). Every time is a fixed past
instant; nothing here reads a clock or reaches a network. The expected numbers
were counted by hand from the fixtures (PROVENANCE.txt lists what each holds),
never read back from the code.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import props
from bucketlib.sources import oddsapi as O
from bucketlib.util import ET, UTC

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "oddsapi"
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402

DAY = H.DATE                                                # 2026-03-12
EARLY, LATE = H.EARLY, H.LATE                               # PHI at DET 7 PM, DEN at SAS 9 PM
NINE = dt.datetime(2026, 3, 12, 9, 0, tzinfo=ET)            # the morning pull
NINE_OH_FIVE = dt.datetime(2026, 3, 12, 9, 5, tzinfo=ET)    # the morning view that restores
ELEVEN = dt.datetime(2026, 3, 12, 11, 0, tzinfo=ET)         # a paste after the morning pull
NOON = dt.datetime(2026, 3, 12, 12, 0, tzinfo=ET)           # a later refresh
LAST_NIGHT = dt.datetime(2026, 3, 11, 21, 41, tzinfo=ET)    # yesterday evening's paste
FOUR = ("player_points", "player_rebounds", "player_assists", "player_fantasy_points")
SPORTSBOOKS = ("draftkings", "fanduel", "betmgm")

# Person ids from the serve fixture's roster (names with accents are read from
# the roster, never typed here).
CUNNINGHAM, DUREN, JENKINS, EDGECOMBE, MAXEY, GRIMES = (1630595, 1631105, 1642450, 1642845,
                                                       1630178, 1629656)
JOKIC, VALANCIUNAS, CAM_JOHNSON, HARDAWAY, BRAUN, CASTLE = (203999, 202685, 1629661, 203501,
                                                           1631128, 1642264)

# Feed lines per (book, game), counted by hand from the two event-odds
# fixtures: PrizePicks has not posted DEN at SAS; Underdog's fantasy line is
# dropped; FanDuel's Cunningham keeps one main line; BetMGM's Duren is an
# Under alone and Edgecombe's pointless Over is skipped.
POSTED = {("prizepicks", EARLY): 9, ("underdog", EARLY): 3, ("underdog", LATE): 4,
          ("draftkings", EARLY): 10, ("draftkings", LATE): 11, ("fanduel", EARLY): 5,
          ("fanduel", LATE): 5, ("betmgm", EARLY): 4, ("betmgm", LATE): 2}
BOOK_ROWS = {"prizepicks": 9, "underdog": 7, "draftkings": 21, "fanduel": 10, "betmgm": 6}
# The board text's 11 lines, and the 6 keys the feed's PrizePicks lines share
# with it (all PHI at DET; the paste's 5 DEN at SAS lines are its own).
PASTED = 11
SHARED = {(CUNNINGHAM, "pts"), (CUNNINGHAM, "ast"), (DUREN, "reb"), (JENKINS, "pts"),
          (EDGECOMBE, "pts"), (MAXEY, "pts")}


# -- The world -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world():
    return H.load()


@pytest.fixture(scope="module")
def roster(world):
    return world.roster


@pytest.fixture(scope="module")
def proj(world):
    return world.proj


@pytest.fixture(scope="module")
def provider(proj):
    return props.MeanDispersionProvider(proj)


@pytest.fixture(scope="module")
def outs(proj):
    """The players whose picks are void: the rows the served frame reads Out
    (props_ui.void_ids' rule). Maxey is one."""
    return props.void_id_set(proj.loc[proj["inj_status"] == "Out", "person_id"].tolist())


@pytest.fixture()
def boards():
    """Tokens a test made, forgotten when it ends (the store is process-wide)."""
    made: list = []
    yield made
    for token in made:
        props.clear_board(token)


def fixture(name: str):
    return json.loads((FIX / name).read_text())


def phi_det() -> dict:
    return fixture("event_odds_phi_det_constructed.json")


def den_sas() -> dict:
    return fixture("event_odds_den_sas_constructed.json")


def without(payload: dict, book: str) -> dict:
    p = copy.deepcopy(payload)
    p["bookmakers"] = [b for b in p["bookmakers"] if b["key"] != book]
    return p


def moved(payload: dict, book: str, player: str, point: float,
          market: str = "player_points") -> dict:
    """A copy of `payload` with `player`'s line at `book` moved to `point`."""
    p = copy.deepcopy(payload)
    hit = 0
    for bm in p["bookmakers"]:
        for mk in bm["markets"] if bm["key"] == book else []:
            for oc in mk["outcomes"] if mk["key"] == market else []:
                if oc["description"] == player:
                    oc["point"], hit = point, hit + 1
    assert hit == 2, (book, player, hit)
    return p


def added(payload: dict, book: str, player: str, point: float,
          market: str = "player_points") -> dict:
    """A copy of `payload` with one more Over/Under pair at `book`."""
    p = copy.deepcopy(payload)
    (mk,) = [m for b in p["bookmakers"] if b["key"] == book for m in b["markets"]
             if m["key"] == market]
    mk["outcomes"] += [{"name": side, "description": player, "price": -110, "point": point}
                       for side in ("Over", "Under")]
    return p


def matched(world) -> pd.DataFrame:
    events, _ = O.parse_events(fixture("events_constructed.json"), DAY)
    games = world.sched[["game_id", "game_date", "home", "away", "tip_utc"]]
    m, stats = O.match_events(events, games)
    assert stats["matched"] == 2
    return m.set_index("event_id")


def feed(world, payloads=None, at=NINE, game_date=None) -> pd.DataFrame:
    """The day's feed frame as the live layer holds it: each payload through
    the REAL parser and merged as a first pull, with the event match's game
    id, ISO date and tip on every row (live._fill). `game_date` replaces the
    date with another spelling of it (the schedule hands python dates)."""
    m = matched(world)
    frames = []
    for p in (phi_det(), den_sas()) if payloads is None else payloads:
        df = O.parse_event_odds(p, books=O.BOOKS, markets=FOUR, fetched_at=at)
        out, _ = O.merge_refresh(None, df, O.BOOKS, markets_requested=FOUR)
        row = m.loc[df.attrs["event_id"]]
        out["game_id"] = row["game_id"]
        out["game_date"] = row["game_date"] if game_date is None else game_date
        out["tip_utc"] = row["tip_utc"]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def tables(world, provider, frame=None) -> props.FeedTables:
    lines, report = props.prepare_feed(feed(world) if frame is None else frame,
                                       world.roster, world.proj)
    assert report["error"] is None
    return props.compare_books(lines, provider, world.proj, world.roster)


def paste_lines(world, at, text=None) -> pd.DataFrame:
    lines, report = props.prepare_paste(world.board_text if text is None else text,
                                        world.roster, world.sched, now=at, dates=[DAY])
    return lines


def board(boards, lines, at) -> str:
    token = props.add_to_board(None, lines, pasted_utc=at)
    boards.append(token)
    return token


def compared(world, provider, token):
    """The paste compared as the page compares it today (props_ui.board_view):
    (table, meta, the board's paste time)."""
    lines = props.lines_for_date(props.get_board(token), DAY)
    table, meta = props.compare(lines, provider, world.proj, world.roster)
    return table, meta, props.board_info(token)["pasted_utc"]


def restored(world, boards, at=LAST_NIGHT, now=NINE_OH_FIVE, edit=None) -> str:
    """Last night's paste, kept in the visitor's browser and restored by the
    keeper this morning (props_ui.board_keeper: saved_board, the browser's
    JSON, restore_saved, add_to_board with the copy's own paste time)."""
    lines = paste_lines(world, at)
    if edit is not None:
        lines = edit(lines.copy())
    first = board(boards, lines, at)
    saved = props.saved_board(props.get_board(first), props.board_info(first)["pasted_utc"],
                              "s1", at.date())
    back = props.restore_saved(json.loads(json.dumps(saved)), world.roster, now.date(), now)
    assert back is not None
    lines, pasted, dropped = back
    assert dropped == 0 and pasted == at.astimezone(UTC)
    return board(boards, lines, pasted)


def keyed(df: pd.DataFrame) -> dict:
    """{(person_id, stat): row} of a frame holding one row per key."""
    out = {}
    for rec in df.to_dict(orient="records"):
        k = (int(rec["person_id"]), rec["stat"])
        assert k not in out, k
        out[k] = rec
    return out


def balanced(fb: props.FeedBoard, outs) -> dict:
    led = props.ledger(fb.meta, *props.board_rows(fb.reference, outs))
    assert led["balanced"], led
    return led


# -- Contracts between props and sources.oddsapi ----------------------------------------
def test_the_feed_frame_carries_every_column_prepare_feed_needs():
    assert set(props.FEED_REQUIRED) <= set(O.FEED_COLUMNS)
    assert len(set(props.FEED_REQUIRED)) == len(props.FEED_REQUIRED)
    assert O.SOURCE == props.FEED_SOURCE == "oddsapi"
    assert props.FEED_SOURCE != props.TYPED_SOURCE
    lines, report = props.prepare_feed(O.empty_feed(), pd.DataFrame(), pd.DataFrame())
    assert report["error"] == "no lines" and lines.empty            # not "missing columns"


def test_the_books_titles_and_codes_are_one_list_in_layout_bs_order():
    assert props.BOOK_ORDER == ("prizepicks", "underdog", "draftkings", "fanduel", "betmgm")
    assert set(O.BOOKS) == set(props.BOOK_ORDER)
    assert O.BOOK_TITLE == props.BOOK_TITLE == {
        "prizepicks": "PrizePicks", "underdog": "Underdog", "draftkings": "DraftKings",
        "fanduel": "FanDuel", "betmgm": "BetMGM"}
    assert props.BOOK_CODE == {"prizepicks": "PP", "underdog": "UD", "draftkings": "DK",
                               "fanduel": "FD", "betmgm": "MGM"}
    assert O.DFS_BOOKS == props.DFS_BOOKS == {"prizepicks", "underdog"}
    assert O.REFERENCE_BOOK == props.PASTE_BOOK == "prizepicks"


def test_the_table_gains_the_feeds_columns_and_the_stored_layouts_do_not():
    assert props.TABLE_EXTRA == ("book", "book_title", "origin", "price_over", "price_under",
                                 "fetched_at", "last_update")
    assert props.TABLE_COLUMNS[:39][-3:] == ["tip_utc", "tie_possible", "pick"]
    assert props.TABLE_COLUMNS[39:] == list(props.TABLE_EXTRA)
    assert len(props.TABLE_COLUMNS) == len(set(props.TABLE_COLUMNS)) == 46
    # The token board and the browser copy keep their layouts (design 2.5).
    assert props.BOARD_COLUMNS == (
        "name", "team_code", "stat_type", "stat", "line", "direction", "odds_type", "opp",
        "is_combo", "league", "match", "person_id", "how", "roster_name", "roster_team",
        "game_date", "game_id", "tip_utc", "pick")
    assert props.SAVED_VERSION == 2
    assert props.SAVED_COLUMNS == ("name", "team_code", "stat_type", "line", "direction",
                                   "odds_type", "opp", "is_combo", "league", "game_date",
                                   "game_id", "tip_utc", "pick")


# -- Names inside the event's two teams (RC G9, m20) -------------------------------------
def test_the_fixtures_names_all_resolve_inside_their_events(world):
    lines, report = props.prepare_feed(feed(world), world.roster, world.proj)
    assert report["error"] is None and report["rows"] == report["kept"] == 53
    assert {b: t["rows"] for b, t in report["books"].items()} == BOOK_ROWS
    assert report["comparable"] == 53 and set(lines["match"]) == {"matched"}
    assert set(lines["how"]) == {"exact"}
    by = {(r["name"], r["book"], r["stat"]): r for r in lines.to_dict(orient="records")}
    jokic = by[("Nikola Jokic", "draftkings", "pts")]          # the roster spells him with accents
    assert jokic["person_id"] == JOKIC and jokic["team"] == jokic["team_code"] == "DEN"
    assert jokic["roster_name"] == world.roster.set_index("person_id").loc[JOKIC, "name"]
    assert by[("Jonas Valanciunas", "draftkings", "pts")]["person_id"] == VALANCIUNAS
    assert set(lines["route_how"]) == {"event"} and set(lines["projectable"]) == {True}
    assert report["games"] == {
        EARLY: {"game_date": "2026-03-12", "tip_utc": "2026-03-12T23:00:00+00:00",
                "event_away": "PHI", "event_home": "DET"},
        LATE: {"game_date": "2026-03-12", "tip_utc": "2026-03-13T01:00:00+00:00",
               "event_away": "DEN", "event_home": "SAS"}}


def test_a_namesake_on_another_team_never_makes_a_feed_name_ambiguous(world):
    """A third-team Cameron Johnson makes the team-less lookup ambiguous; the
    event's two teams keep the line on Denver's, exact and by prefix."""
    namesake = pd.DataFrame([{"person_id": 9900001, "first_name": "Cameron",
                              "last_name": "Johnson", "team": "BKN", "position": "",
                              "two_way": False, "name": "Cameron Johnson"}])
    roster = pd.concat([world.roster, namesake], ignore_index=True)
    payload = added(added(den_sas(), "draftkings", "Cameron Johnson", 10.5),
                    "fanduel", "Cam Johnson", 10.5)
    frame = feed(world, [payload])
    _, unres = props.resolve_names(frame[frame["name"] == "Cameron Johnson"], roster)
    assert list(unres["reason"]) == ["ambiguous"]               # what one-team scoping gives
    lines, _ = props.prepare_feed(frame, roster, world.proj)
    cam = lines[lines["name"].isin(["Cameron Johnson", "Cam Johnson"])]
    assert sorted(zip(cam["name"], cam["person_id"], cam["how"], cam["team"])) == [
        ("Cam Johnson", CAM_JOHNSON, "prefix", "DEN"),
        ("Cameron Johnson", CAM_JOHNSON, "exact", "DEN")]


def test_the_full_roster_fallback_takes_exact_or_alias_matches_in_this_game_only(world, provider):
    """m20: a roster lagging a trade (Hardaway and Braun moved to Brooklyn here,
    while the projections still have them in DEN at SAS) and a player the
    roster puts on a team outside the event (Herbert Jones, given a projection
    row in DEN at SAS). Exact and alias matches with a projection in THIS
    game resolve, marked +other_team; a prefix, or the same exact name in
    the other game, never does."""
    roster = world.roster.copy()
    roster.loc[roster["person_id"].isin([HARDAWAY, BRAUN]), "team"] = "BKN"
    herb = pd.DataFrame([{"person_id": 9900002, "first_name": "Herbert", "last_name": "Jones",
                          "team": "NOP", "position": "", "two_way": False,
                          "name": "Herbert Jones"}])
    roster = pd.concat([roster, herb], ignore_index=True)
    proj = world.proj.copy()
    herb_row = proj[proj["person_id"] == CASTLE].assign(person_id=9900002,
                                                        player_name="Herbert Jones")
    proj = pd.concat([proj, herb_row], ignore_index=True)
    late = added(added(added(den_sas(), "draftkings", "Tim Hardaway Jr.", 11.5),
                       "draftkings", "Herb Jones", 7.5), "draftkings", "Chris Braun", 10.5)
    early = added(phi_det(), "draftkings", "Tim Hardaway Jr.", 11.5)
    lines, report = props.prepare_feed(feed(world, [early, late]), roster, proj)
    got = {(r["name"], r["game_id"]): r for r in lines.to_dict(orient="records")
           if r["book"] == "draftkings" and r["stat"] == "pts"}
    # The same exact name in the other game stays unmatched: matched there,
    # it would be his line for the wrong game.
    assert got[("Tim Hardaway Jr.", EARLY)]["match"] == "unmatched"
    hardaway = got[("Tim Hardaway Jr.", LATE)]
    assert (hardaway["match"], hardaway["person_id"], hardaway["how"]) == \
        ("matched", HARDAWAY, "exact+other_team")
    assert hardaway["team"] == hardaway["team_code"] == hardaway["roster_team"] == "BKN"
    herbert = got[("Herb Jones", LATE)]
    assert (herbert["match"], herbert["person_id"], herbert["how"]) == \
        ("matched", 9900002, "alias+other_team")
    assert got[("Chris Braun", LATE)]["match"] == "unmatched"           # a prefix: never
    # Braun's own line ("Christian Braun") is exact, so it still resolves.
    assert got[("Christian Braun", LATE)]["how"] == "exact+other_team"
    dk = report["books"]["draftkings"]
    assert dk["unmatched"] == 2 and sorted(dk["unmatched_names"]) == ["Chris Braun",
                                                                      "Tim Hardaway Jr."]
    ft = props.compare_books(lines, props.MeanDispersionProvider(proj), proj, roster)
    meta = ft.metas[("draftkings", LATE)]
    assert ("Herb Jones", "Herbert Jones", "alias+other_team") in meta["nonexact"]
    # The traded players are listed as matched on another team (the page's
    # "Matched by nickname, partial name or another team").
    assert {(a, h) for a, _b, h in meta["nonexact"]} == {
        ("Tim Hardaway Jr.", "exact+other_team"), ("Christian Braun", "exact+other_team"),
        ("Herb Jones", "alias+other_team")}
    assert meta["unmatched_names"] == ["Chris Braun"]
    assert props.ledger(meta, *props.board_rows(ft.tables[("draftkings", LATE)]))["balanced"]


def _namesake(person_id: int, first: str, last: str, team: str) -> dict:
    return {"person_id": person_id, "first_name": first, "last_name": last, "team": team,
            "position": "", "two_way": False, "name": f"{first} {last}"}


def test_a_namesake_on_a_third_team_no_longer_blocks_the_other_team_fallback_b13(world):
    """Build amendment B13. The other-team fallback (a roster lagging a
    trade, m20) used the full roster's index, so a namesake anywhere else in
    the league made a traded player's line ambiguous. It now looks the name
    up among the players with a projection row in THIS event's game only,
    exact or alias matches only. Hardaway is moved to Brooklyn in the roster
    while the projections keep him in DEN at SAS, and a second Tim Hardaway
    Jr. plays for Miami; Herbert Jones (projected in DEN at SAS, rostered at
    New Orleans) has a namesake at Memphis. Over the full roster both names
    are ambiguous; among this game's players each is one man, so the lines
    resolve, marked +other_team. A namesake projected in this game too keeps
    the name ambiguous, and a prefix still never matches."""
    roster = world.roster.copy()
    roster.loc[roster["person_id"] == HARDAWAY, "team"] = "BKN"
    roster = pd.concat([roster, pd.DataFrame([
        _namesake(9900002, "Herbert", "Jones", "NOP"),
        _namesake(9900003, "Tim", "Hardaway Jr.", "MIA"),
        _namesake(9900004, "Herbert", "Jones", "MEM")])], ignore_index=True)
    proj = world.proj.copy()
    herb_row = proj[proj["person_id"] == CASTLE].assign(person_id=9900002,
                                                        player_name="Herbert Jones")
    proj = pd.concat([proj, herb_row], ignore_index=True)
    full = props.build_name_index(roster)
    assert full.get("Tim Hardaway Jr.").how == full.get("Herb Jones").how == "ambiguous"
    late = added(added(added(den_sas(), "draftkings", "Tim Hardaway Jr.", 11.5),
                       "draftkings", "Herb Jones", 7.5), "draftkings", "Timothy Hardaway Jr.", 11.5)
    frame = feed(world, [late])

    def dk_points(projections):
        lines, report = props.prepare_feed(frame, roster, projections)
        assert report["error"] is None
        return {r["name"]: r for r in lines.to_dict(orient="records")
                if r["book"] == "draftkings" and r["stat"] == "pts"}
    got = dk_points(proj)
    hardaway, herbert = got["Tim Hardaway Jr."], got["Herb Jones"]
    assert (hardaway["match"], hardaway["person_id"], hardaway["how"], hardaway["team"]) == (
        "matched", HARDAWAY, "exact+other_team", "BKN")
    assert (herbert["match"], herbert["person_id"], herbert["how"], herbert["team"]) == (
        "matched", 9900002, "alias+other_team", "NOP")
    assert got["Timothy Hardaway Jr."]["match"] == "unmatched"        # a prefix: never
    both = pd.concat([proj, proj[proj["person_id"] == CASTLE].assign(
        person_id=9900003, player_name="Tim Hardaway Jr.")], ignore_index=True)
    again = dk_points(both)
    assert again["Tim Hardaway Jr."]["match"] == "ambiguous"         # two of this game's players
    assert again["Herb Jones"]["person_id"] == 9900002


def test_unknown_names_are_counted_and_listed_and_non_people_dropped_with_a_count(world,
                                                                                  provider):
    payload = added(added(phi_det(), "draftkings", "Nobody Atall", 9.5),
                    "draftkings", "DET", 110.5)
    lines, report = props.prepare_feed(feed(world, [payload]), world.roster, world.proj)
    dk = report["books"]["draftkings"]
    assert (dk["rows"], dk["kept"], dk["not_a_person"], dk["unmatched"]) == (12, 11, 1, 1)
    assert dk["unmatched_names"] == ["Nobody Atall"] and dk["comparable"] == 10
    assert report["not_a_person"] == 1 and "DET" not in set(lines["name"])
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    meta = ft.metas[("draftkings", EARLY)]
    assert (meta["posted"], meta["matched"], meta["unmatched"]) == (11, 10, 1)
    assert meta["unmatched_names"] == ["Nobody Atall"]


# -- Dates and keys (m3) -------------------------------------------------------------------
@pytest.mark.parametrize("spelling", [DAY, pd.Timestamp(DAY), "2026-03-12",
                                      "2026-03-12T00:00:00", dt.datetime(2026, 3, 12, 0, 0)])
def test_game_dates_become_the_iso_string_a_pasted_row_carries(world, provider, boards,
                                                               spelling):
    """The schedule hands python dates (pipeline.state_games). Left as they
    were, no feed key would ever meet a pasted one (m3): the merge would see
    no overlap and the PrizePicks table would hold both rows."""
    lines, _ = props.prepare_feed(feed(world, game_date=spelling), world.roster, world.proj)
    assert set(lines["game_date"]) == {"2026-03-12"}
    assert all(isinstance(d, str) for d in lines["game_date"])
    assert set(lines["tip_utc"]) == {"2026-03-12T23:00:00+00:00", "2026-03-13T01:00:00+00:00"}
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    table, meta, pasted = compared(world, provider, board(boards, paste_lines(world, NINE), NINE))
    assert set(table["game_date"]) == {"2026-03-12"} and set(table["tip_utc"]) <= set(lines["tip_utc"])
    fb = props.feed_board(ft, table, meta, pasted, selected="prizepicks")
    assert fb.paste_won + fb.feed_won == len(SHARED)
    assert len(fb.reference) == BOOK_ROWS["prizepicks"] + PASTED - len(SHARED)


def test_one_row_per_game_player_stat_and_book_and_the_newest_fetch_wins(world):
    morning = feed(world)
    later = feed(world, [moved(phi_det(), "draftkings", "Cade Cunningham", 26.5)], at=NOON)
    lines, report = props.prepare_feed(pd.concat([morning, later], ignore_index=True),
                                       world.roster, world.proj)
    phi_rows = 9 + 3 + 10 + 5 + 4                    # the PHI at DET rows, sent twice
    assert report["rows"] == 53 + phi_rows and report["duplicates"] == phi_rows
    assert report["kept"] == len(lines) == 53
    assert report["books"]["draftkings"]["duplicates"] == 10
    key = lines[["game_date", "person_id", "stat", "book"]].astype(str)
    assert not key.duplicated().any()
    cade = lines[(lines["person_id"] == CUNNINGHAM) & (lines["stat"] == "pts")]
    assert sorted(cade["book"]) == sorted(O.BOOKS)          # one per book, every book kept
    dk = cade[cade["book"] == "draftkings"].iloc[0]
    assert dk["line"] == 26.5 and dk["fetched_at"] == pd.Timestamp(NOON).tz_convert("UTC")
    fd = cade[cade["book"] == "fanduel"].iloc[0]
    assert fd["fetched_at"] == pd.Timestamp(NOON).tz_convert("UTC")   # newest copy, same line
    sas = lines[(lines["game_id"] == LATE) & (lines["book"] == "draftkings")]
    assert set(sas["fetched_at"]) == {pd.Timestamp(NINE).tz_convert("UTC")}


def test_the_paste_collapse_key_and_its_ladder_are_unchanged():
    """The feed is deduplicated on its own key (with the book). The paste's
    collapse key never gained the book, so a re-paste replaces a line exactly
    as before (design 2.0)."""
    rows = pd.DataFrame([
        {"name": "Cade Cunningham", "stat_type": "Points", "stat": "pts", "line": 25.5,
         "odds_type": "standard", "game_date": "2026-03-12", "person_id": CUNNINGHAM,
         "book": "draftkings"},
        {"name": "Cade Cunningham", "stat_type": "Points", "stat": "pts", "line": 26.5,
         "odds_type": "standard", "game_date": "2026-03-12", "person_id": CUNNINGHAM,
         "book": "fanduel"}])
    assert list(props._collapse_key(rows)) == [f"2026-03-12|p{CUNNINGHAM}|pts"] * 2
    kept = props.collapse_alt_lines(rows)
    assert len(kept) == 1 and kept.iloc[0]["line"] == 26.5          # the newest paste wins


# -- compare() per book and game -------------------------------------------------------------
def test_compare_carries_the_book_its_title_origin_prices_and_stamps(world, provider, boards):
    frame = feed(world)
    frame.loc[frame["book"] == "underdog", "book_title"] = "Underdog Fantasy"   # the vendor's
    # The parser blanks DFS prices; compare() refuses them on its own as well.
    frame.loc[frame["book"].isin(["prizepicks", "underdog"]), ["price_over", "price_under"]] = -137.0
    ft = tables(world, provider, frame)
    nine, read_at = pd.Timestamp(NINE).tz_convert("UTC"), pd.Timestamp("2026-03-12T12:52:14Z")
    dk = keyed(ft.tables[("draftkings", EARLY)])[(CUNNINGHAM, "pts")]
    assert (dk["book"], dk["book_title"], dk["origin"]) == ("draftkings", "DraftKings", "feed")
    assert (dk["line"], dk["price_over"], dk["price_under"]) == (25.5, -115.0, -105.0)
    assert dk["fetched_at"] == nine and dk["last_update"] == read_at
    fd = keyed(ft.tables[("fanduel", EARLY)])[(CUNNINGHAM, "pts")]
    assert (fd["line"], fd["price_over"], fd["price_under"]) == (25.5, -114.0, -106.0)
    for book in ("prizepicks", "underdog"):                       # DFS: never a price
        row = keyed(ft.tables[(book, EARLY)])[(CUNNINGHAM, "pts")]
        assert math.isnan(row["price_over"]) and math.isnan(row["price_under"]), book
        assert props.american_breakeven(row["price_over"]) is None
    ud = keyed(ft.tables[("underdog", EARLY)])[(CUNNINGHAM, "pts")]
    assert ud["book_title"] == "Underdog"                          # props' own title
    for (book, _gid), table in ft.tables.items():
        assert list(table.columns) == props.TABLE_COLUMNS
        assert set(table["book"]) == {book} and set(table["origin"]) == {"feed"}
        assert set(table["source"]) == {"baseline"}               # the provider's, untouched
    table, _meta, _at = compared(world, provider, board(boards, paste_lines(world, NINE), NINE))
    pasted = keyed(table)[(CUNNINGHAM, "pts")]
    assert (pasted["book"], pasted["book_title"], pasted["origin"]) == \
        ("prizepicks", "PrizePicks", "pasted")
    assert math.isnan(pasted["price_over"]) and pasted["fetched_at"] is None
    assert pasted["last_update"] is None


def test_each_rows_lean_and_gap_are_against_its_own_books_line(world, provider):
    """Design 7: no consensus line. Maxey is 24.5 at PrizePicks, 25.5 at
    Underdog and 26.5 at the three sportsbooks."""
    ft = tables(world, provider)
    pmf = provider.pmf(MAXEY, EARLY, "pts")
    seen = {}
    for book in O.BOOKS:
        row = keyed(ft.tables[(book, EARLY)])[(MAXEY, "pts")]
        p_lt, p_eq, p_gt = pmf.split(row["line"])
        assert row["p_over"] == pytest.approx(p_gt) and row["p_under"] == pytest.approx(p_lt)
        assert row["gap"] == pytest.approx(row["model"] - row["line"])
        assert row["lean"] == props.lean_from(p_gt / (p_gt + p_lt))
        seen[book] = row["line"]
    assert seen == {"prizepicks": 24.5, "underdog": 25.5, "draftkings": 26.5, "fanduel": 26.5,
                    "betmgm": 26.5}


def test_every_book_and_game_gets_its_own_balanced_meta(world, provider, outs):
    ft = tables(world, provider)
    assert set(ft.metas) == set(ft.tables) == set(POSTED)
    assert ("prizepicks", LATE) not in ft.metas                    # PrizePicks has not posted it
    for key, meta in ft.metas.items():
        table = ft.tables[key]
        assert meta["posted"] == POSTED[key] == meta["matched"] == len(table), key
        led = props.ledger(meta, *props.board_rows(table, outs))
        assert led["balanced"], (key, led)
    assert list(ft.metas)[:3] == [("prizepicks", EARLY), ("underdog", EARLY), ("underdog", LATE)]
    assert ft.report["rows"] == 53 and ft.report["error"] is None   # prepare_feed's report


def test_one_book_and_game_that_fails_costs_only_itself(world, provider, monkeypatch, caplog):
    lines, _ = props.prepare_feed(feed(world), world.roster, world.proj)
    real = props.compare

    def flaky(rows, *a, **k):
        if set(rows["book"]) == {"fanduel"} and set(rows["game_id"]) == {LATE}:
            raise RuntimeError("boom")
        return real(rows, *a, **k)
    monkeypatch.setattr(props, "compare", flaky)
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    bad = ft.metas[("fanduel", LATE)]
    assert ft.tables[("fanduel", LATE)].empty
    assert (bad["posted"], bad["unprojected"], bad["unprojected_why"]) == \
        (5, 5, {"internal error": 5})
    assert props.ledger(bad, *props.board_rows(ft.tables[("fanduel", LATE)]))["balanced"]
    assert len(ft.tables[("fanduel", EARLY)]) == 5
    assert "compare_books" in caplog.text and "RuntimeError" in caplog.text and "boom" not in caplog.text


# -- The merge (M5): newest wins; typed picks, Demons and Goblins always do ---------------------
def _moved_feed(world, at=NINE):
    """The feed with PrizePicks' Cunningham points moved 25.5 -> 26.5 and
    Maxey's 24.5 -> 25.5 since the board text was pasted."""
    pp = moved(moved(phi_det(), "prizepicks", "Cade Cunningham", 26.5),
               "prizepicks", "Tyrese Maxey", 25.5)
    return feed(world, [pp, den_sas()], at=at)


def test_a_restored_old_paste_loses_to_the_morning_feed(world, provider, boards, outs):
    """M5: the keeper restores last night's paste on this morning's first
    view. Under "the paste always wins" its numbers beat the 9 AM feed all
    day. Newest wins: the feed's line is shown for every shared key."""
    token = restored(world, boards)
    table, meta, pasted = compared(world, provider, token)
    assert len(table) == PASTED and pasted == LAST_NIGHT.astimezone(UTC)
    ft = tables(world, provider, _moved_feed(world))
    fb = props.feed_board(ft, table, meta, pasted, selected="prizepicks")
    assert (fb.feed_won, fb.paste_won) == (len(SHARED), 0)
    ref = keyed(fb.reference)
    for k in SHARED:
        assert ref[k]["origin"] == "feed", k
    assert ref[(CUNNINGHAM, "pts")]["line"] == 26.5 and ref[(MAXEY, "pts")]["line"] == 25.5
    assert ref[(JOKIC, "pra")]["origin"] == "pasted"          # a key only the paste holds
    assert fb.paste_times == {"pasted_utc": pd.Timestamp(LAST_NIGHT).tz_convert("UTC"),
                              "feed_oldest": pd.Timestamp(NINE).tz_convert("UTC"),
                              "feed_newest": pd.Timestamp(NINE).tz_convert("UTC")}
    led = balanced(fb, outs)
    assert led["posted"] == led["matched"] == len(fb.reference) == \
        BOOK_ROWS["prizepicks"] + PASTED - len(SHARED)
    assert fb.meta["pasted_utc"] == pasted


def test_a_paste_newer_than_the_pull_wins_and_a_later_pull_wins_back(world, provider, boards):
    table, meta, pasted = compared(world, provider,
                                   board(boards, paste_lines(world, ELEVEN), ELEVEN))
    fb = props.feed_board(tables(world, provider, _moved_feed(world, at=NINE)), table, meta,
                          pasted, selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (len(SHARED), 0)
    ref = keyed(fb.reference)
    assert ref[(CUNNINGHAM, "pts")]["line"] == 25.5 and ref[(CUNNINGHAM, "pts")]["origin"] == "pasted"
    assert fb.paste_times["feed_oldest"] is None and fb.paste_times["feed_newest"] is None
    fb = props.feed_board(tables(world, provider, _moved_feed(world, at=NOON)), table, meta,
                          pasted, selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (0, len(SHARED))
    assert keyed(fb.reference)[(CUNNINGHAM, "pts")]["line"] == 26.5
    # The same instant is not later: the feed keeps the key.
    fb = props.feed_board(tables(world, provider, _moved_feed(world, at=ELEVEN)), table, meta,
                          pasted, selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (0, len(SHARED))


def test_typed_picks_demons_and_goblins_always_win(world, provider, boards, outs):
    """Last night's paste, with a typed pick on Cunningham's points, a Goblin
    on Duren's rebounds and a Demon on Edgecombe's points: those three beat
    this morning's feed; its three plain lines do not."""
    def edit(lines):
        cade = (lines["person_id"] == CUNNINGHAM) & (lines["stat"] == "pts")
        lines.loc[cade, ["line", "pick"]] = [24.5, "more"]
        lines.loc[(lines["person_id"] == DUREN) & (lines["stat"] == "reb"),
                  ["line", "odds_type"]] = [8.5, "goblin"]
        lines.loc[(lines["person_id"] == EDGECOMBE) & (lines["stat"] == "pts"),
                  ["line", "odds_type"]] = [20.5, "demon"]
        return lines
    table, meta, pasted = compared(world, provider, restored(world, boards, edit=edit))
    fb = props.feed_board(tables(world, provider, _moved_feed(world)), table, meta, pasted,
                          selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (3, 3)
    ref = keyed(fb.reference)
    assert (ref[(CUNNINGHAM, "pts")]["line"], ref[(CUNNINGHAM, "pts")]["pick"]) == (24.5, "more")
    assert (ref[(DUREN, "reb")]["line"], ref[(DUREN, "reb")]["odds_type"]) == (8.5, "goblin")
    assert (ref[(EDGECOMBE, "pts")]["line"], ref[(EDGECOMBE, "pts")]["odds_type"]) == \
        (20.5, "demon")
    for k in ((CUNNINGHAM, "ast"), (JENKINS, "pts"), (MAXEY, "pts")):
        assert ref[k]["origin"] == "feed", k
    balanced(fb, outs)


def test_a_board_with_no_paste_time_counts_as_older(world, provider, boards):
    table, meta, _at = compared(world, provider,
                                board(boards, paste_lines(world, ELEVEN), ELEVEN))
    table = table.copy()
    table.loc[table["person_id"] == JENKINS, "pick"] = "less"
    fb = props.feed_board(tables(world, provider, _moved_feed(world)), table, meta, None,
                          selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (1, len(SHARED) - 1)       # only the typed pick
    assert fb.meta["pasted_utc"] is None and fb.paste_times["pasted_utc"] is None


def test_paste_rows_the_paste_path_withheld_leave_the_feeds_line(world, provider, boards, outs):
    """A quarter board pasted as full-game: compare() withholds its PHI at
    DET rows (props.py's period-board check), so they are not in the paste
    table, and the feed's line stays for those keys."""
    quarter = {(CUNNINGHAM, "pts"): 6.5, (CUNNINGHAM, "ast"): 2.5, (DUREN, "reb"): 2.5,
               (JENKINS, "pts"): 1.5, (EDGECOMBE, "pts"): 4.5, (MAXEY, "pts"): 6.5}

    def edit(lines):
        for (pid, stat), x in quarter.items():
            lines.loc[(lines["person_id"] == pid) & (lines["stat"] == stat), "line"] = x
        return lines
    table, meta, pasted = compared(world, provider, restored(world, boards, edit=edit))
    assert meta["period_board_suspect"] and meta["period_withheld"] == len(quarter)
    assert set(table["game_id"]) == {LATE}
    fb = props.feed_board(tables(world, provider), table, meta, pasted, selected="prizepicks")
    assert (fb.paste_won, fb.feed_won) == (0, 0)
    ref = keyed(fb.reference)
    for k in quarter:
        assert ref[k]["origin"] == "feed" and ref[k]["line"] > quarter[k], k
    led = balanced(fb, outs)
    assert led["n_refused"] == len(quarter) and led["matched"] == BOOK_ROWS["prizepicks"] + 5


# -- Layout B: one selected book (amendments A1 and A9) -----------------------------------
@pytest.fixture()
def pasted_board(world, provider, boards):
    """This morning's paste (the board text at 11 AM) compared as the page
    compares it: (table, meta, paste time)."""
    return compared(world, provider, board(boards, paste_lines(world, ELEVEN), ELEVEN))


def test_the_reference_is_the_selected_books_rows_for_every_game_and_no_others(
        world, provider, pasted_board, outs):
    ft = tables(world, provider)
    for book in props.BOOK_ORDER:
        fb = props.feed_board(ft, *pasted_board, selected=book)
        assert set(fb.reference["book"]) == {book}, book
        want = set()
        for gid in (EARLY, LATE):
            if (book, gid) in ft.tables:
                want |= set(keyed(ft.tables[(book, gid)]))
        if book == "prizepicks":
            want |= set(keyed(pasted_board[0]))
            assert set(fb.reference["origin"]) == {"feed", "pasted"}
        else:
            assert set(fb.reference["origin"]) == {"feed"}, book
        assert set(keyed(fb.reference)) == want, book
        assert list(fb.reference["person_id"]) == list(props.rank_rows(fb.reference)["person_id"])
        assert not set(fb.context["book"]) & {book}
        balanced(fb, outs)


def test_missing_games_lists_the_games_the_selected_book_has_not_posted(world, provider, outs):
    """PrizePicks has not posted DEN at SAS: none of its rows, and never
    another book's, stand in for it (amendment A6), and the game is listed.
    With BetMGM gone from both games, both are listed, in tip order."""
    ft = tables(world, provider)
    fb = props.feed_board(ft, None, None, None, selected="prizepicks")
    assert fb.missing_games == [LATE] and fb.ref_by_game == {EARLY: "prizepicks"}
    assert set(fb.reference["game_id"]) == {EARLY} and set(fb.reference["book"]) == {"prizepicks"}
    assert len(fb.reference) == BOOK_ROWS["prizepicks"]
    balanced(fb, outs)
    for book in ("underdog", *SPORTSBOOKS):
        fb = props.feed_board(ft, None, None, None, selected=book)
        assert fb.missing_games == [] and fb.ref_by_game == {EARLY: book, LATE: book}, book
    no_mgm = tables(world, provider, feed(world, [without(phi_det(), "betmgm"),
                                                  without(den_sas(), "betmgm")]))
    fb = props.feed_board(no_mgm, None, None, None, selected="betmgm")
    assert fb.missing_games == [EARLY, LATE] and fb.ref_by_game == {}
    assert fb.reference.empty and fb.context.empty
    assert props.ledger(fb.meta, *props.board_rows(fb.reference))["balanced"]
    assert fb.meta["posted"] == 0


def test_missing_games_come_in_tip_order_not_id_order(world, provider):
    """The two fixture games' ids sort as their tips do, so PHI at DET is
    moved after DEN at SAS here (a 10 PM ET tip)."""
    frame = feed(world, [without(phi_det(), "betmgm"), without(den_sas(), "betmgm")])
    frame.loc[frame["game_id"] == EARLY, "tip_utc"] = pd.Timestamp("2026-03-13T03:00:00Z")
    fb = props.feed_board(tables(world, provider, frame), None, None, None, selected="betmgm")
    assert fb.missing_games == [LATE, EARLY]
    fb = props.feed_board(tables(world, provider, frame), None, None, None, selected="fanduel")
    assert list(fb.ref_by_game) == [LATE, EARLY]


def test_a_paste_posts_prizepicks_lines_for_a_game_the_feed_lacks(world, provider,
                                                                   pasted_board, outs):
    ft = tables(world, provider)
    fb = props.feed_board(ft, *pasted_board, selected="prizepicks")
    assert fb.missing_games == [] and fb.ref_by_game == {EARLY: "prizepicks", LATE: "prizepicks"}
    late = fb.reference[fb.reference["game_id"] == LATE]
    assert len(late) == 5 and set(late["origin"]) == {"pasted"}
    # Every game with lines anywhere is in exactly one of the two, for every pick.
    for book in props.BOOK_ORDER:
        fb = props.feed_board(ft, *pasted_board, selected=book)
        assert set(fb.ref_by_game).isdisjoint(fb.missing_games)
        assert set(fb.ref_by_game) | set(fb.missing_games) == {EARLY, LATE}
        assert set(fb.reference["game_id"]) <= set(fb.ref_by_game)


def test_the_merged_prizepicks_rows_are_the_reference_and_the_context(world, provider, boards):
    """With a typed pick on Cunningham's points (24.5, More) over the morning
    feed's 25.5: PrizePicks picked, it is the reference row; DraftKings
    picked, it is the PrizePicks line in the context, beside the other
    books, in BOOK_ORDER."""
    def edit(lines):
        cade = (lines["person_id"] == CUNNINGHAM) & (lines["stat"] == "pts")
        lines.loc[cade, ["line", "pick"]] = [24.5, "more"]
        return lines
    table, meta, pasted = compared(world, provider, restored(world, boards, edit=edit))
    ft = tables(world, provider)
    pp = props.feed_board(ft, table, meta, pasted, selected="prizepicks")
    mine = keyed(pp.reference)[(CUNNINGHAM, "pts")]
    assert (mine["line"], mine["origin"], mine["pick"]) == (24.5, "pasted", "more")
    dk = props.feed_board(ft, table, meta, pasted, selected="draftkings")
    assert keyed(dk.reference)[(CUNNINGHAM, "pts")]["line"] == 25.5
    cade = dk.context[(dk.context["person_id"] == CUNNINGHAM) & (dk.context["stat"] == "pts")]
    assert list(cade["book"]) == ["prizepicks", "underdog", "fanduel", "betmgm"]
    ctx_pp = cade.iloc[0]
    assert (ctx_pp["line"], ctx_pp["origin"], ctx_pp["pick"]) == (24.5, "pasted", "more")
    # The context is every other book's row for the reference keys, each once,
    # PrizePicks' being the merged ones ...
    ref_keys = set(keyed(dk.reference))
    pairs = list(zip(dk.context["person_id"], dk.context["stat"], dk.context["book"]))

    def keys_of(book):
        return {k for (b, _g), t in ft.tables.items() if b == book for k in keyed(t)}
    want = {(p, s, "prizepicks") for p, s in set(keyed(pp.reference)) & ref_keys}
    want |= {(p, s, b) for b in ("underdog", "fanduel", "betmgm")
             for p, s in keys_of(b) & ref_keys}
    assert set(pairs) == want and len(pairs) == len(want) == 8 + 7 + 9 + 6   # PP UD FD MGM
    assert (JOKIC, "pra") not in {(p, s) for p, s, _ in pairs}   # a paste key DK lacks
    assert (CASTLE, "pts", "prizepicks") in set(pairs)         # a pasted line DK also has
    # ... in the reference's order, then BOOK_ORDER.
    order = list(zip(dk.reference["person_id"], dk.reference["stat"]))
    seen = list(dict.fromkeys((p, s) for p, s, _ in pairs))
    assert seen == [k for k in order if k in set(seen)]
    rank = [props.BOOK_ORDER.index(b) for b in dk.context["book"]]
    for k in seen:
        mine = [r for (p, s, _b), r in zip(pairs, rank) if (p, s) == k]
        assert mine == sorted(mine), k


def test_the_meta_balances_for_every_pick(world, provider, pasted_board, outs):
    payload = added(phi_det(), "draftkings", "Nobody Atall", 9.5)
    ft = tables(world, provider, feed(world, [payload, den_sas()]))
    want = {"prizepicks": BOOK_ROWS["prizepicks"] + PASTED - len(SHARED),
            "underdog": 7, "draftkings": 22, "fanduel": 10, "betmgm": 6}
    for book in props.BOOK_ORDER:
        fb = props.feed_board(ft, *pasted_board, selected=book)
        led = balanced(fb, outs)
        assert led["posted"] == want[book], book
        assert sum(fb.meta["by_tier"].values()) == led["matched"] == len(fb.reference)
        assert (fb.meta["pasted_utc"] is not None) == (book == "prizepicks"), book
    dk = props.feed_board(ft, *pasted_board, selected="draftkings").meta
    assert (dk["unmatched"], dk["unmatched_names"]) == (1, ["Nobody Atall"])
    assert len(dk["unresolved"]) == 1


def test_an_unknown_book_is_refused(world, provider):
    ft = tables(world, provider)
    for bad in ("bet365", "PrizePicks", "", None, 5, "betrivers"):
        with pytest.raises(ValueError):
            props.feed_board(ft, None, None, None, selected=bad)


# -- Storage stays apart (RC R5) -------------------------------------------------------------
def test_feed_rows_never_reach_the_token_board_or_the_browser_copy(world, provider, boards,
                                                                   monkeypatch):
    token = board(boards, paste_lines(world, ELEVEN), ELEVEN)
    before = props.get_board(token)
    table, meta, pasted = compared(world, provider, token)
    count = props.board_count()

    def never(*a, **k):
        raise AssertionError("the feed path touched the visitor's board store")
    for name in ("new_board", "add_to_board", "saved_board", "restore_saved", "_cap",
                 "_store_locked", "get_board", "board_info"):
        monkeypatch.setattr(props, name, never)
    ft = tables(world, provider)
    boards_seen = [props.feed_board(ft, table, meta, pasted, selected=b) for b in props.BOOK_ORDER]
    monkeypatch.undo()
    assert len(boards_seen) == 5 and props.board_count() == count
    after = props.get_board(token)
    pd.testing.assert_frame_equal(after, before)
    saved = props.saved_board(after, pasted, "s1", DAY)
    names = {(r[0], r[2]) for r in saved["rows"]}
    assert len(saved["rows"]) == PASTED
    assert ("Quentin Grimes", "Points") not in names              # the feed's PrizePicks-only line
    assert ("Tobias Harris", "Points") not in names               # a sportsbook-only line
    assert not set(saved["cols"]) & {"book", "origin", "price_over", "price_under",
                                     "fetched_at", "last_update", "source"}


def test_the_token_board_refuses_feed_rows_handed_to_it_b14(world, provider, boards):
    """Build amendment B14, defense in depth for RC R5: the feed path never
    hands the token board a feed row (the test above), and if a later
    mistake did, the board drops every one. Feed lines (their line-level
    source is FEED_SOURCE) through add_to_board or new_board store nothing;
    mixed into a paste, only the paste is kept; a feed line for a key the
    board already holds, at another number, never displaces the board's own
    pasted line; and a compared feed row (origin "feed", the merged board's
    reference) is refused too. The browser copy then carries the paste's
    lines only."""
    paste = paste_lines(world, ELEVEN)
    feed_lines, _ = props.prepare_feed(feed(world), world.roster, world.proj)
    assert set(feed_lines["source"]) == {props.FEED_SOURCE} and len(feed_lines) == 53
    alone = board(boards, paste, ELEVEN)
    want = props.get_board(alone)
    assert len(want) == PASTED
    assert props.get_board(board(boards, feed_lines, ELEVEN)).empty
    made = props.new_board(feed_lines)
    boards.append(made)
    assert props.get_board(made).empty
    mixed = board(boards, pd.concat([paste, feed_lines], ignore_index=True), ELEVEN)
    # the same rows and values (person_id comes back as objects after the concat)
    pd.testing.assert_frame_equal(props.get_board(mixed), want, check_dtype=False)
    cade = (feed_lines["person_id"] == CUNNINGHAM) & (feed_lines["stat"] == "pts") \
        & (feed_lines["book"] == "prizepicks")
    pasted_line = keyed(want)[(CUNNINGHAM, "pts")]["line"]
    stray = feed_lines[cade].assign(line=pasted_line + 3.0)
    assert len(stray) == 1
    after = props.add_to_board(alone, stray, pasted_utc=NOON)
    boards.append(after)
    pd.testing.assert_frame_equal(props.get_board(after), want)
    ref = props.feed_board(tables(world, provider), None, None, None, selected="prizepicks").reference
    assert set(ref["origin"]) == {"feed"} and not ref.empty
    assert props.get_board(board(boards, ref, ELEVEN)).empty
    saved = props.saved_board(props.get_board(mixed), ELEVEN, "s1", DAY)
    assert len(saved["rows"]) == PASTED


# -- Prices, side words and robustness ----------------------------------------------------------
@pytest.mark.parametrize("price, want", [
    (-110, 0.5238095), (-130, 0.5652174), (120, 0.4545455), (-100, 0.5), (100, 0.5),
    (-110.0, 0.5238095), (np.float64(-110.0), 0.5238095), (np.int64(150), 0.4)])
def test_american_breakeven_is_the_probability_the_price_implies(price, want):
    assert props.american_breakeven(price) == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("price", [None, float("nan"), float("inf"), -float("inf"), True,
                                   False, "-110", 0, 50, -50, 99.9, [-110], pd.NaT])
def test_american_breakeven_refuses_what_is_not_an_american_price(price):
    assert props.american_breakeven(price) is None


def test_side_words_are_more_and_less_on_dfs_rows_and_over_and_under_on_sportsbooks(world,
                                                                                   provider):
    for book in ("prizepicks", "underdog", "PrizePicks", None, "", float("nan")):
        assert props.side_words(book) == ("More", "Less"), book
    for book in ("draftkings", "fanduel", "betmgm", "DraftKings", "betrivers"):
        assert props.side_words(book) == ("Over", "Under"), book
    ft = tables(world, provider)
    leans = set().union(*(set(t["lean"]) for t in ft.tables.values()))
    assert leans <= {"More", "Less", "Even"}                      # stored as before, every book


@pytest.mark.parametrize("junk", [
    None, [], "lines", 5, pd.DataFrame(), pd.DataFrame({"name": ["Cade Cunningham"]}),
    "missing fetched_at"])
def test_prepare_feed_never_raises_on_a_frame_it_cannot_read(world, provider, junk):
    if isinstance(junk, str) and junk == "missing fetched_at":
        junk = feed(world).drop(columns=["fetched_at"])
    lines, report = props.prepare_feed(junk, world.roster, world.proj)
    assert lines.empty and report["error"] in ("no feed", "no lines", "missing columns")
    if report["error"] == "missing columns":
        assert report["missing"]
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    assert ft.tables == {} and ft.metas == {}
    fb = props.feed_board(ft, None, None, None, selected="prizepicks")
    assert fb.reference.empty and fb.missing_games == [] and fb.meta["posted"] == 0


def test_prepare_feed_counts_junk_cells_and_keeps_the_rest(world, provider):
    frame = feed(world, [phi_det()])
    junk = frame.iloc[:8].copy()
    junk["line"] = ["abc", float("nan"), -1.0, 1e9, None, True, 25.5, 25.5]
    junk.loc[junk.index[6], "game_date"] = "not a date"
    junk.loc[junk.index[7], "game_id"] = None
    names = frame.iloc[8:12].copy()
    names["name"] = [None, 123, "PHI", "Cade Cunningham"]
    names["book"] = ["draftkings", "draftkings", "draftkings", None]
    rest = frame.iloc[12:].copy()
    rest["fetched_at"] = "junk"
    rest["tip_utc"] = "9999-99-99"
    lines, report = props.prepare_feed(pd.concat([junk, names, rest], ignore_index=True),
                                       world.roster, world.proj)
    assert report["error"] is None
    assert report["bad_rows"] == 8 + 1 and report["not_a_person"] == 3
    assert report["kept"] == len(lines) == len(rest)
    assert report["rows"] == len(frame)
    assert sum(t["rows"] for t in report["books"].values()) == len(frame)
    assert report["books"]["unknown"]["bad_rows"] == 1
    assert lines["tip_utc"].isna().all() and lines["fetched_at"].isna().all()
    assert lines["last_update"].notna().all()                    # the parser's own stamps
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    for key, meta in ft.metas.items():
        assert props.ledger(meta, *props.board_rows(ft.tables[key]))["balanced"], key


def test_prepare_feed_survives_its_own_failure(world, monkeypatch, caplog):
    def broken(*a, **k):
        raise KeyError("person_id")
    monkeypatch.setattr(props, "resolve_event_names", broken)
    lines, report = props.prepare_feed(feed(world), world.roster, world.proj)
    assert lines.empty and report["error"] == "internal error"
    assert lines.attrs["feed_report"] is report
    assert "prepare_feed failed (KeyError)" in caplog.text


# -- Build 4's fix pass: the mutation campaign's survivors PR01, PR04, PR05, PR07, PR08 -------
def test_a_feed_row_with_no_fetch_time_never_beats_a_dated_paste_pr01(world, provider, boards):
    """Survivor PR01 (M5, _paste_wins): a feed row with no fetch time counts
    as older than any dated paste, so the paste's plain row wins that key;
    a board with no paste time still loses to it."""
    assert props._paste_wins({"pick": None, "odds_type": "standard"}, {"fetched_at": None},
                             NINE.astimezone(UTC)) is True
    assert props._paste_wins({"pick": None, "odds_type": "standard"}, {"fetched_at": None},
                             None) is False
    ft = tables(world, provider)
    pp = ft.tables[("prizepicks", EARLY)].copy()
    pp.loc[pp["person_id"] == JENKINS, "fetched_at"] = pd.NaT
    ft = props.FeedTables(tables={**ft.tables, ("prizepicks", EARLY): pp}, metas=ft.metas,
                          report=ft.report)
    table, meta, _at = compared(world, provider, board(boards, paste_lines(world, LAST_NIGHT),
                                                       LAST_NIGHT))
    fb = props.feed_board(ft, table, meta, LAST_NIGHT.astimezone(UTC), selected="prizepicks")
    ref = keyed(fb.reference)
    assert ref[(JENKINS, "pts")]["origin"] == "pasted"
    assert ref[(CUNNINGHAM, "pts")]["origin"] == "feed"               # dated, and newer
    assert fb.paste_won == 1


def test_a_book_whose_lines_for_a_game_all_went_unmatched_has_still_posted_it_pr04(world, provider):
    """Survivor PR04 (amendment A6): a book has lines for a game when it
    POSTED some, matched to a player or not. FanDuel's DEN at SAS lines all
    carry names nobody on the roster has: the game is FanDuel's (its board
    explains the unmatched names), never "No FanDuel lines yet"."""
    late = copy.deepcopy(den_sas())
    for bm in late["bookmakers"]:
        for m in bm["markets"] if bm["key"] == "fanduel" else []:
            for oc in m["outcomes"]:
                oc["description"] = "Zzyzx " + oc["description"].split()[-1] + " Qq"
    ft = tables(world, provider, feed(world, [phi_det(), late]))
    meta = ft.metas[("fanduel", LATE)]
    assert int(meta["posted"]) > 0 and int(meta["matched"]) == 0
    fb = props.feed_board(ft, None, None, None, selected="fanduel")
    assert fb.ref_by_game.get(LATE) == "fanduel" and LATE not in fb.missing_games


def test_a_repeated_key_in_a_books_table_still_balances_the_ledger_pr05(world, provider, outs):
    """Survivor PR05: should a book's compared table ever hold one key twice
    (compare() and the parser leave one row per key, so it is defence in
    depth), the merge keeps one and the summed meta takes the extra row
    out, so the ledger still balances."""
    ft = tables(world, provider)
    pp = ft.tables[("prizepicks", EARLY)]
    twice = pd.concat([pp, pp.iloc[[0]]], ignore_index=True)
    meta = dict(ft.metas[("prizepicks", EARLY)])
    meta["posted"] = int(meta["posted"]) + 1
    meta["matched"] = int(meta["matched"]) + 1
    ft = props.FeedTables(tables={**ft.tables, ("prizepicks", EARLY): twice},
                          metas={**ft.metas, ("prizepicks", EARLY): meta}, report=ft.report)
    fb = props.feed_board(ft, None, None, None, selected="prizepicks")
    assert len(fb.reference) == len(pp)
    balanced(fb, outs)


def test_a_namesake_projected_in_another_game_never_blocks_the_fallback_pr07(world):
    """Survivor PR07 (B13): the fallback indexes only the players with a
    projection row in THIS event's game. The Memphis Herbert Jones gets a
    projection row in PHI at DET; Herb Jones in DEN at SAS still resolves to
    the Herbert Jones projected there."""
    roster = pd.concat([world.roster, pd.DataFrame([
        _namesake(9900002, "Herbert", "Jones", "NOP"),
        _namesake(9900004, "Herbert", "Jones", "MEM")])], ignore_index=True)
    proj = world.proj.copy()
    proj = pd.concat([proj,
                      proj[proj["person_id"] == CASTLE].assign(person_id=9900002,
                                                               player_name="Herbert Jones"),
                      proj[proj["person_id"] == CUNNINGHAM].assign(person_id=9900004,
                                                                   player_name="Herbert Jones")],
                     ignore_index=True)
    assert set(proj.loc[proj["person_id"] == 9900004, "game_id"]) == {EARLY}
    lines, report = props.prepare_feed(feed(world, [added(den_sas(), "draftkings", "Herb Jones",
                                                          7.5)]), roster, proj)
    got = {r["name"]: r for r in lines.to_dict(orient="records")
           if r["book"] == "draftkings" and r["stat"] == "pts"}
    herb = got["Herb Jones"]
    assert (herb["match"], herb["person_id"], herb["how"]) == ("matched", 9900002, "alias+other_team")


def test_a_name_two_players_share_inside_the_event_stays_ambiguous_pr08(world):
    """Survivor PR08 (B13): the other-team fallback is for a name nobody in
    the event's two teams has. When the two-team lookup finds two players
    by the name (a namesake on the other team of the event, here a second
    Nikola Jokic rostered at San Antonio with no projection row), the line
    stays ambiguous: the fallback never picks the projected one."""
    twin = world.roster[world.roster["person_id"] == JOKIC].assign(person_id=9900006, team="SAS")
    roster = pd.concat([world.roster, twin], ignore_index=True)
    lines, report = props.prepare_feed(feed(world, [den_sas()]), roster, world.proj)
    got = [r for r in lines.to_dict(orient="records")
           if r["book"] == "draftkings" and r["stat"] == "pts" and r["name"] == "Nikola Jokic"]
    assert len(got) == 1 and got[0]["match"] == "ambiguous"
    assert pd.isna(got[0]["person_id"]) or got[0]["person_id"] is None
