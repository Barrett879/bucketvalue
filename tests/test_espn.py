"""ESPN source: every trap the 2026-10-08 research and probes found, pinned
against trimmed REAL payloads in tests/fixtures/espn/ (offline)."""
import copy
import datetime as dt
import importlib.util
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import cache, http, util
from bucketlib.sources import espn

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "espn"


def fx(name):
    return cache.read_json(FIX / name)


@pytest.fixture(scope="module")
def months():
    return fx("scoreboard_months.json.gz")


@pytest.fixture(scope="module")
def odds_fx():
    return fx("odds_events.json.gz")


@pytest.fixture(scope="module")
def nba_games():
    logs = fx("gamelog_t.json.gz")
    return pd.concat([espn.nba_games_from_gamelog(http.result_sets(p)["LeagueGameLog"])
                      for p in logs.values()], ignore_index=True)


@pytest.fixture(scope="module")
def all_events(months):
    return pd.concat([espn.parse_events(p)[0] for p in months.values()], ignore_index=True)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def odds(odds_fx, eid):
    d = odds_fx[eid]
    return espn.parse_odds(d, d["season"])


def no_network(*a, **k):
    raise AssertionError(f"unexpected network call {a} {k}")


# ── small parsers ────────────────────────────────────────────────────────────
def test_parse_line_reads_american_strings_and_prefixes():
    assert espn.parse_line("+9.5") == 9.5
    assert espn.parse_line("-7.5") == -7.5
    assert espn.parse_line("o220.5") == 220.5      # scoreboard total.over.close.line
    assert espn.parse_line("u220.5") == 220.5
    assert espn.parse_line("EVEN") == 0.0
    assert espn.parse_line("PK") == 0.0
    assert espn.parse_line(232.5) == 232.5
    for junk in (None, "", "abc", True, float("nan")):
        assert espn.parse_line(junk) is None


def test_parse_american_treats_zero_as_missing():
    assert espn.parse_american("+235") == 235
    assert espn.parse_american("-110") == -110
    assert espn.parse_american("EVEN") == 100
    assert espn.parse_american(0) is None          # BOVADA 2015-16 sentinel
    assert espn.parse_american(None) is None


def test_details_favourite():
    assert espn.details_favourite("PHI -8.5", "BKN", "PHI") == "away"
    assert espn.details_favourite("NY -5.5", "NYK", "PHI") == "home"   # ESPN abbreviation
    assert espn.details_favourite("UTAH -1", "BKN", "UTA") == "away"
    assert espn.details_favourite("EVEN", "BKN", "UTA") == "even"
    assert espn.details_favourite("LAL -3", "BKN", "UTA") is None      # not in this game
    assert espn.details_favourite(None, "BKN", "UTA") is None


def test_season_months_cover_bubble_and_short_seasons():
    m = espn.season_months("2019-20")
    assert "202007" in m and "202010" in m and "202004" not in m and "202003" in m
    assert espn.season_months("2020-21")[0] == "202012" and espn.season_months("2020-21")[-1] == "202107"
    assert espn.season_months("2023-24") == ["202310", "202311", "202312", "202401", "202402",
                                             "202403", "202404", "202405", "202406"]


# ── scoreboard / events ──────────────────────────────────────────────────────
def test_range_form_is_refused_before_any_request(monkeypatch):
    # dates=A-B is an HTTP 400 that reads like "no games".
    monkeypatch.setattr(espn.http, "fetch", no_network)
    with pytest.raises(ValueError, match="range form"):
        espn.fetch_scoreboard("20261001-20261031")
    with pytest.raises(ValueError):
        espn.scoreboard_month("2026-10")


def test_month_at_limit_is_split_into_day_queries(monkeypatch, tmp_cache):
    calls = []

    def ev(i, day):
        return {"id": str(400000000 + i), "date": f"2016-03-{day:02d}T23:00Z",
                "season": {"year": 2016, "slug": "regular-season"},
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitions": [{"type": {"abbreviation": "STD"}, "competitors": [
                    {"homeAway": "home", "team": {"id": "2"}, "score": "100"},
                    {"homeAway": "away", "team": {"id": "1"}, "score": "90"}]}]}

    def fake(url, params=None, **kw):
        calls.append(params["dates"])
        assert params["limit"] == str(espn.MONTH_LIMIT)
        if params["dates"] == "201603":
            return {"events": [ev(i, 1) for i in range(espn.MONTH_LIMIT)]}  # truncated
        day = int(params["dates"][-2:])
        return {"events": [ev(day, day), ev(day + 100, day)]}

    monkeypatch.setattr(espn.http, "fetch", fake)
    out = espn.scoreboard_month("201603")
    assert calls[0] == "201603" and len(calls) == 1 + 31
    assert len(out["events"]) == 62
    # Settled month: the cached copy is reused, never refetched.
    monkeypatch.setattr(espn.http, "fetch", no_network)
    assert len(espn.scoreboard_month("201603")["events"]) == 62


def test_current_month_cache_expires(monkeypatch, tmp_cache):
    ym = util.today_et().strftime("%Y%m")
    path = cache.dc_path(f"raw/espn_scoreboard/{ym}.json.gz")
    cache.write_json(path, {"events": []}, stamped=True, gz=True)
    # Fresh stamp: reused.
    monkeypatch.setattr(espn.http, "fetch", no_network)
    assert espn.scoreboard_month(ym) == {"events": []}
    # Stale stamp: refetched.
    cache.write_json(path, {"fetched_at": 0, "data": {"events": []}}, gz=True)
    monkeypatch.setattr(espn.http, "fetch", lambda *a, **k: {"events": []})
    seen = []
    monkeypatch.setattr(espn, "fetch_scoreboard", lambda d, limit=500: seen.append(d) or {"events": []})
    espn.scoreboard_month(ym)
    assert seen == [ym]


def _stamp(iso):
    return dt.datetime.fromisoformat(iso).timestamp()


def _cache_month(ym, payload, fetched_iso):
    path = cache.dc_path(f"raw/espn_scoreboard/{ym}.json.gz")
    cache.write_json(path, {"fetched_at": _stamp(fetched_iso), "data": payload}, gz=True)


def test_month_cached_before_it_ended_is_refetched(monkeypatch, tmp_cache, months):
    # Settled is judged by WHEN THE COPY WAS FETCHED, not today's date. A
    # copy saved mid-month holds that month's later games as SCHEDULED; the
    # calendar rule reused it forever (verifier: 202610 cached Oct 8, read in
    # December, 88 of 88 events still STATUS_SCHEDULED, no lines).
    mid = {"events": months["202002"]["events"][:2]}
    full = months["202002"]
    seen = []
    monkeypatch.setattr(espn, "fetch_scoreboard", lambda d, limit=500: seen.append(d) or full)
    _cache_month("202002", mid, "2020-02-10T17:00:00+00:00")
    assert len(espn.scoreboard_month("202002")["events"]) == len(full["events"])
    assert seen == ["202002"]
    # The refetch is stamped now (long after month end): settled for good.
    monkeypatch.setattr(espn, "fetch_scoreboard", no_network)
    assert len(espn.scoreboard_month("202002")["events"]) == len(full["events"])


def test_month_settles_only_after_the_grace_days(monkeypatch, tmp_cache, months):
    p = months["202002"]
    monkeypatch.setattr(espn, "fetch_scoreboard", no_network)
    _cache_month("202002", p, "2020-03-04T17:00:00+00:00")   # Feb 29 + 4 days, ET
    assert espn.scoreboard_month("202002") == p
    seen = []
    monkeypatch.setattr(espn, "fetch_scoreboard", lambda d, limit=500: seen.append(d) or p)
    _cache_month("202002", p, "2020-03-03T17:00:00+00:00")   # + 3 days: not yet
    espn.scoreboard_month("202002")
    assert seen == ["202002"]


def test_empty_month_is_never_settled(monkeypatch, tmp_cache, months):
    # A 200 with events: [] must not be trusted forever.
    _cache_month("202002", {"events": []}, "2021-01-01T17:00:00+00:00")
    seen = []
    monkeypatch.setattr(espn, "fetch_scoreboard",
                        lambda d, limit=500: seen.append(d) or months["202002"])
    assert len(espn.scoreboard_month("202002")["events"]) == len(months["202002"]["events"])
    assert seen == ["202002"]


def test_fetched_after_uses_the_et_date():
    # 03:30 UTC on Mar 4 is still Mar 3 in New York.
    assert not espn.fetched_after(_stamp("2020-03-04T03:30:00+00:00"), dt.date(2020, 2, 29), 3)
    assert espn.fetched_after(_stamp("2020-03-04T05:30:00+00:00"), dt.date(2020, 2, 29), 3)
    assert not espn.fetched_after(None, dt.date(2020, 2, 29), 3)


def test_events_reports_raw_counts_per_month(monkeypatch, months):
    pages = {"202002": months["202002"]}
    monkeypatch.setattr(espn, "scoreboard_month", lambda ym, **k: pages.get(ym, {"events": []}))
    ev = espn.events("2019-20", months=["202002", "202003"])
    assert ev.attrs["month_counts"] == {"202002": 8, "202003": 0}


def test_final_event_without_a_score_is_kept_and_counted(months):
    p = copy.deepcopy(months["202002"])
    e = next(x for x in p["events"] if x["id"] == "401161374")
    e["competitions"][0]["competitors"][0]["score"] = ""
    df, dropped = espn.parse_events(p)
    assert dropped["final_score_missing"] == 1 and dropped["final_score_missing_ids"] == ["401161374"]
    r = df.set_index("event_id").loc["401161374"]
    assert pd.isna(r["home_score"]) or pd.isna(r["away_score"])
    assert len(df) == 6                                     # kept, not dropped


def test_parse_events_drops_and_counts_non_nba(months):
    df, dropped = espn.parse_events(months["202002"])
    # Rising Stars WORLD VS USA is type STD with fake team ids; the All-Star
    # game is type ALLSTAR. Both gone, both counted.
    assert dropped["non_nba_team"] == 1 and dropped["allstar"] == 1
    assert set(dropped["dropped_ids"]) == {"401197814", "401197813"}
    assert not df["event_id"].isin(["401197814", "401197813"]).any()
    assert len(df) == 6


def test_london_lions_preseason_opponent_is_dropped():
    j = fx("live_sb_20261008.json.gz")["data"]
    j = copy.deepcopy(j)
    j["events"][0]["competitions"][0]["competitors"][1]["team"]["id"] = "134478"
    df, dropped = espn.parse_events(j)
    assert dropped["non_nba_team"] == 1 and len(df) == 5


def test_event_dates_are_et_not_utc(months):
    df, _ = espn.parse_events(months["202002"])
    r = df.set_index("event_id").loc["401161374"]          # BOS-PHI, tip 01:30Z
    assert str(r["start_utc"]) == "2020-02-02 01:30:00+00:00"
    assert r["et_date"] == dt.date(2020, 2, 1)
    assert r["home"] == "BOS" and r["away"] == "PHI"


def test_postponed_and_cup_final_are_kept_with_their_tags(all_events):
    e = all_events.set_index("event_id")
    assert e.loc["401267343", "status"] == "STATUS_POSTPONED"
    assert e.loc["401809839", "comp_type"] == "CC" and e.loc["401809839", "season_slug"] == "regular-season"
    assert bool(e.loc["401810002", "neutral_site"]) is True  # Mexico City


def test_parse_events_rejects_structural_surprises(months):
    bad = copy.deepcopy(months["202002"])
    bad["events"][0]["competitions"][0]["competitors"][1]["homeAway"] = "home"
    with pytest.raises(ValueError, match="one home"):
        espn.parse_events(bad)
    dup = copy.deepcopy(months["202002"])
    dup["events"].append(copy.deepcopy(dup["events"][0]))
    with pytest.raises(ValueError, match="twice"):
        espn.parse_events(dup)


def test_events_dedupes_across_months_and_filters_season(monkeypatch, months):
    a = copy.deepcopy(months["202002"])
    b = {"events": [copy.deepcopy(a["events"][0]), copy.deepcopy(months["202101"]["events"][0])]}
    pages = {"202002": a, "202003": b}
    monkeypatch.setattr(espn, "scoreboard_month", lambda ym, **k: pages.get(ym, {"events": []}))
    ev = espn.events("2019-20", months=["202002", "202003"])
    assert ev.attrs["dropped"]["duplicate_across_months"] == 1
    assert ev.attrs["dropped"]["other_season"] == 1        # the 2020-21 event
    assert ev["event_id"].is_unique and (ev["season"] == "2019-20").all()
    # The same id with different teams is a parsing error, not a duplicate.
    b2 = copy.deepcopy(b)
    b2["events"][0]["competitions"][0]["competitors"][0]["team"]["id"] = "13"
    pages["202003"] = b2
    with pytest.raises(ValueError, match="differs"):
        espn.events("2019-20", months=["202002", "202003"])


# ── closing lines ────────────────────────────────────────────────────────────
def test_2015_16_books_median_and_opening_provider(odds_fx):
    r = odds(odds_fx, "400827970")                         # DEN @ GSW
    assert (r["home"], r["away"]) == ("GSW", "DEN")
    assert r["home_line"] == -17.5 and r["total"] == 211.0
    # consensus/numberfire/teamrankings and Opening (23) are NOT books.
    assert r["n_books"] == 6 and "consensus" not in r["books"] and "Opening" not in r["books"]
    # ...and they are excluded as KNOWN non-books, not as unknown providers.
    assert r["qc_flags"] == ""
    assert r["open_home_line"] == -14.5 and r["open_total"] == 214.5 and r["has_open"]


def test_opening_pickem_is_zero_not_missing(odds_fx):
    r = odds(odds_fx, "400827926")                         # Opening 'EVEN'
    assert r["open_home_line"] == 0.0 and r["open_total"] == 193.0 and r["has_open"]
    assert r["home_line"] == 2.5                           # ATL (away) favoured


def test_no_opener_between_2017_and_2023_even_if_fields_appear(odds_fx):
    r = odds(odds_fx, "400975071")
    assert math.isnan(r["open_home_line"]) and math.isnan(r["open_total"]) and not r["has_open"]
    d = copy.deepcopy(odds_fx["401160740"])
    for it in d["pages"][0]["items"]:
        it["initialSpread"], it["initialOverUnder"] = -3.0, 210.0
    r = espn.parse_odds(d, d["season"])
    assert not r["has_open"]                               # era-gated, never a fake opener


def test_book_outliers_are_dropped(odds_fx):
    r = odds(odds_fx, "400975071")                         # Caesars 31 says WSH -15.5
    assert r["home_line"] == 1.0 and "outlier:31" in r["qc_flags"]
    # The whole book goes, total included: one bad market marks the snapshot.
    assert r["n_books"] == 5 and r["n_books_total"] == 5
    r = odds(odds_fx, "401160740")                         # Wynn flipped the favourite
    assert r["home_line"] == 5.5 and "outlier:32" in r["qc_flags"]
    r = odds(odds_fx, "401160741")                         # DK stale 0.5 / 212
    assert r["home_line"] == 5.5 and r["total"] == 222.5 and "outlier:40" in r["qc_flags"]


def test_partial_game_total_takes_its_spread_with_it(odds_fx):
    # Caesars 31: total 59.0 (a quarter/half market) with spread -2 vs -6.
    r = odds(odds_fx, "401161469")
    assert "partial_or_bad_total:31" in r["qc_flags"]
    assert r["n_books"] == 5 and r["home_line"] == -6.5


def test_live_feeds_never_count(odds_fx):
    r = odds(odds_fx, "401267311")
    assert "live:44" in r["qc_flags"] and "live:46" in r["qc_flags"]
    d = copy.deepcopy(odds_fx["401810164"])                # DK only: DET -9.5
    items = d["pages"][0]["items"]
    live = copy.deepcopy(items[0])
    live["provider"] = {"id": "200", "name": "DraftKings - Live Odds"}
    live["spread"], live["overUnder"] = -24.5, 250.5
    live["homeTeamOdds"]["close"]["pointSpread"]["american"] = "-24.5"
    live["awayTeamOdds"]["close"]["pointSpread"]["american"] = "+24.5"
    newlive = copy.deepcopy(live)
    newlive["provider"] = {"id": "777", "name": "Some Book LIVE odds"}  # unknown id, live name
    items += [live, newlive]
    d["pages"][0]["count"] = len(items)
    r = espn.parse_odds(d, d["season"])
    assert r["home_line"] == -9.5 and r["n_books"] == 1
    assert "live:200" in r["qc_flags"] and "live:777" in r["qc_flags"]


def test_unknown_provider_is_excluded_and_reported(odds_fx):
    d = copy.deepcopy(odds_fx["401810164"])
    nb = copy.deepcopy(d["pages"][0]["items"][0])
    nb["provider"] = {"id": "4242", "name": "NewBook"}
    d["pages"][0]["items"].append(nb)
    r = espn.parse_odds(d, d["season"])
    assert r["n_books"] == 1 and "unknown_provider:4242:NewBook" in r["qc_flags"]


def test_aggregator_zero_is_not_a_pickem(odds_fx):
    # Postponed 401267339 has only numberfire (0.0/0.0) and teamrankings.
    r = odds(odds_fx, "401267339")
    assert math.isnan(r["home_line"]) and r["n_books"] == 0 and "no_pregame_books" in r["qc_flags"]


def test_espn_bet_2022_23_close_prices_are_never_read(odds_fx):
    d = copy.deepcopy(odds_fx["401468361"])
    d["pages"][0]["items"] = [it for it in d["pages"][0]["items"] if it["provider"]["id"] == "58"]
    assert d["pages"][0]["items"][0]["homeTeamOdds"]["close"]["pointSpread"]["american"] == "-110"
    r = espn.parse_odds(d, d["season"])
    assert r["home_line"] == -1.5 and r["total"] == 225.5 and r["line_field"] == "current"


def test_from_2023_24_close_beats_a_stale_or_inplay_current(odds_fx):
    # ESPN BET current total 217.5 vs its own close 234.5 (others ~233.5).
    r = odds(odds_fx, "401584954")
    assert r["line_field"] == "close" and r["total"] == 233.5 and r["home_line"] == 9.5
    # 2024 playoffs: three books' current totals were in-play (181-184).
    r = odds(odds_fx, "401654757")
    assert r["home_line"] == -5.5 and r["total"] == 206.5
    # open{} objects give the opener from 2023-24 on.
    assert r["has_open"] and r["open_home_line"] == -4.5


def test_close_spread_must_mirror_the_away_side(odds_fx):
    d = copy.deepcopy(odds_fx["401810164"])
    it = d["pages"][0]["items"][0]
    it["homeTeamOdds"]["close"]["pointSpread"]["american"] = "-110"   # a price, not a line
    it["awayTeamOdds"]["close"]["pointSpread"]["american"] = "-110"
    it["spread"] = -9.0
    r = espn.parse_odds(d, d["season"])
    assert r["home_line"] == -9.0 and r["line_field"] == "mixed"


def test_decimal_value_is_ignored_for_lines(odds_fx):
    # homeTeamOdds.open.pointSpread carries value 1.952 next to american '+6.5'.
    r = odds(odds_fx, "401705127")
    assert r["open_home_line"] == 6.5 and r["home_line"] == 5.5 and r["n_books"] == 1


def test_espn_bet_and_draftkings_overlap(odds_fx):
    r = odds(odds_fx, "401809800")
    assert r["n_books"] == 2 and r["books"] == "DraftKings|ESPN BET"
    assert r["home_line"] == 4.5 and r["open_home_line"] == 2.0     # median of +2.5 / +1.5


def test_per_book_sign_mismatch_drops_that_book(odds_fx):
    d = copy.deepcopy(odds_fx["400827970"])
    five = next(it for it in d["pages"][0]["items"] if it["provider"]["id"] == "18")
    five["details"] = "DEN -17.5"                          # says away favoured, spread says home
    r = espn.parse_odds(d, d["season"])
    assert "book_sign_mismatch:18" in r["qc_flags"] and r["n_books"] == 5


def test_details_never_vetoes_a_close_line(odds_fx):
    # `details` describes the CURRENT line. From 2023-24 the spread is read
    # from close{}, and near pick'em the two legitimately disagree: real
    # 401585288, Betfair closed BKN +1 (with nine other books) while its
    # details said 'BKN -1'. Alone (the 2024-26 one-book eras), the old vote
    # withheld that valid close as a sign_conflict.
    d = copy.deepcopy(odds_fx["401585288"])
    d["pages"][0]["items"] = [it for it in d["pages"][0]["items"] if it["provider"]["id"] == "50"]
    d["pages"][0]["count"] = 1
    assert d["pages"][0]["items"][0]["details"] == "BKN -1"
    r = espn.parse_odds(d, d["season"])
    assert (r["home"], r["home_line"], r["line_field"]) == ("BKN", 1.0, "close")
    assert "sign_conflict" not in r["qc_flags"] and "details_vs_close:50" in r["qc_flags"]
    # Same with the single DraftKings book of 2025-26.
    d = copy.deepcopy(odds_fx["401810164"])
    d["pages"][0]["items"][0]["details"] = "ATL -9.5"      # close says DET -9.5
    r = espn.parse_odds(d, d["season"])
    assert r["home_line"] == -9.5 and r["total"] == 232.5
    assert "sign_conflict" not in r["qc_flags"] and "details_vs_close:100" in r["qc_flags"]
    # The full ten-book payload is unchanged by the rule.
    r = odds(odds_fx, "401585288")
    assert r["home_line"] == 1.0 and r["n_books"] == 10


def test_consensus_sign_conflict_still_withholds_a_current_line(odds_fx):
    # Before 2023-24 the spread IS the current value, so details vote. Four
    # books at GSW +1.5 with unreadable details, two at -1.5 naming GSW: the
    # median says DEN favoured, every readable details says GSW.
    d = copy.deepcopy(odds_fx["400827970"])
    books = [it for it in d["pages"][0]["items"] if int(it["provider"]["id"]) in espn.PREGAME_BOOKS]
    assert len(books) == 6
    for i, it in enumerate(books):
        it["spread"], it["details"] = (1.5, "") if i < 4 else (-1.5, "GS -1.5")
    r = espn.parse_odds(d, d["season"])
    assert math.isnan(r["home_line"]) and "sign_conflict" in r["qc_flags"]
    assert r["total"] == 211.0


def test_2016_17_sportsinteraction_and_sportsbook_are_pregame_books(odds_fx):
    # Real 400899603 (CLE-TOR): ids 1 and 3 price in line with the others in
    # every sampled 2016-17 game; they used to be dropped as unknown.
    r = odds(odds_fx, "400899603")
    assert r["n_books"] == 9 and r["qc_flags"] == ""
    assert "SportsInteraction.com" in r["books"] and "Sportsbook.com" in r["books"]
    assert r["home_line"] == -6.0 and r["total"] == 209.0
    assert r["open_home_line"] == -6.5 and r["has_open"]     # provider 23, 2016-17


def test_known_live_ids_are_live_even_without_live_in_the_name(odds_fx):
    # Defense in depth: the id list catches a live feed ESPN renames.
    d = copy.deepcopy(odds_fx["401810164"])
    live = copy.deepcopy(d["pages"][0]["items"][0])
    live["provider"] = {"id": "59", "name": "ESPN Bet"}
    d["pages"][0]["items"].append(live)
    r = espn.parse_odds(d, d["season"])
    assert "live:59" in r["qc_flags"] and "unknown_provider:59" not in r["qc_flags"]


def test_spread_bounds(odds_fx):
    d = copy.deepcopy(odds_fx["400827970"])
    for it in d["pages"][0]["items"]:
        if it["provider"]["id"] == "18":
            it["spread"] = -30.0
    r = espn.parse_odds(d, d["season"])
    assert "spread_out_of_bounds:18" in r["qc_flags"] and r["n_books"] == 5 and r["home_line"] == -17.5


def test_team_refs_must_match_the_caller(odds_fx):
    d = odds_fx["401810164"]
    assert espn.parse_odds(d, d["season"], home="DET", away="ATL")["home_line"] == -9.5
    with pytest.raises(ValueError, match="caller says"):
        espn.parse_odds(d, d["season"], home="ATL", away="DET")


def test_cup_final_has_lines(odds_fx):
    r = odds(odds_fx, "401809839")
    assert (r["home"], r["away"]) == ("NYK", "SAS") and r["home_line"] == -2.5


def test_odds_pagination_and_count(monkeypatch):
    pages = {None: {"count": 3, "pageIndex": 1, "pageCount": 2, "items": [{"a": 1}, {"b": 2}]},
             "2": {"count": 3, "pageIndex": 2, "pageCount": 2, "items": [{"c": 3}]}}
    monkeypatch.setattr(espn.http, "fetch", lambda url, params=None, **k: pages[params.get("page")])
    got = espn.fetch_odds_pages("401")
    assert sum(len(p["items"]) for p in got) == 3
    pages[None]["count"] = 4
    with pytest.raises(http.BadPayload, match="count=4"):
        espn.fetch_odds_pages("401")


def test_absent_is_cached_blocked_is_raised(monkeypatch, tmp_cache):
    def absent(url, **k):
        raise http.Absent(url, "HTTP 404")

    monkeypatch.setattr(espn.http, "fetch", absent)
    r = espn.game_odds("999", "2019-20")
    assert r["qc_flags"] == "odds_absent" and math.isnan(r["home_line"])
    monkeypatch.setattr(espn.http, "fetch", no_network)
    assert espn.game_odds("999", "2019-20")["qc_flags"] == "odds_absent"   # resumable

    def blocked(url, **k):
        raise http.Blocked(url, "HTTP 403 Akamai")

    monkeypatch.setattr(espn.http, "fetch", blocked)
    with pytest.raises(http.Blocked):
        espn.game_odds("1000", "2019-20")
    assert not espn.odds_cache_path("2019-20", "1000").exists()       # never cached as empty


def test_absent_marker_expires_but_real_odds_never_do(monkeypatch, tmp_cache, odds_fx):
    # A transient burst of 404s must not poison every later rerun.
    calls = []

    def absent(url, **k):
        calls.append(url)
        raise http.Absent(url, "HTTP 404")

    monkeypatch.setattr(espn.http, "fetch", absent)
    espn.game_odds("999", "2019-20")
    assert len(calls) == 1 and espn.odds_is_cached("999", "2019-20")
    espn.game_odds("999", "2019-20")
    assert len(calls) == 1                                  # fresh marker: reused
    espn.game_odds("999", "2019-20", retry_absent=True)
    assert len(calls) == 2                                  # forced re-ask
    path = espn.odds_cache_path("2019-20", "999")
    old = util.now_utc().timestamp() - espn.ABSENT_TTL - 60
    cache.write_json(path, {"fetched_at": old, "data": {"event_id": "999", "absent": True}}, gz=True)
    assert not espn.odds_is_cached("999", "2019-20")
    espn.game_odds("999", "2019-20")
    assert len(calls) == 3                                  # expired marker: re-asked
    # A real payload for a finished game is final: even an ancient one is reused.
    real = {k: v for k, v in odds_fx["401810164"].items() if k != "season"}
    cache.write_json(espn.odds_cache_path("2025-26", "401810164"), {"fetched_at": 0, "data": real}, gz=True)
    monkeypatch.setattr(espn.http, "fetch", no_network)
    assert espn.odds_is_cached("401810164", "2025-26")
    assert espn.game_odds("401810164", "2025-26")["home_line"] == -9.5


def test_empty_item_list_is_no_lines_not_an_error():
    r = espn.parse_odds({"event_id": "1", "pages": [{"count": 0, "items": []}]}, "2026-27")
    assert r["qc_flags"] == "no_items" and r["n_books"] == 0


# ── crosswalk ────────────────────────────────────────────────────────────────
def test_crosswalk_on_real_fixtures(all_events, nba_games):
    xw = espn.crosswalk(all_events, nba_games).set_index("event_id")
    assert xw.loc["401161374", "how"] == "exact" and xw.loc["401161374", "game_id"] == "0021900732"
    # Feb-2020 corrupted timestamps: DET@POR shows 2020-02-24T12:00Z at the United Center.
    assert xw.loc["401161490", ["how", "game_id"]].tolist() == ["date_pm1", "0021900848"]
    assert xw.loc["401161490", "nba_game_date"] == dt.date(2020, 2, 23)
    assert xw.loc["401161528", ["how", "game_id"]].tolist() == ["date_pm1", "0021900886"]
    # Postponed ids never match, including both legs of the CLE@WAS two-game series.
    for e in ("401267339", "401267343", "401267357", "401267364"):
        assert xw.loc[e, "how"] == "not_final" and xw.loc[e, "game_id"] is None
    # Neutral site where the NBA log prints both rows as '@'.
    assert xw.loc["401810002", ["how", "game_id"]].tolist() == ["exact", "0022500147"]
    assert xw.loc["401809837", "game_id"] == "0022501229"
    # Cup finals are real games but not in the Regular Season log.
    for e in ("401809839", "401607495"):
        assert xw.loc[e, "how"] == "unmatched" and "006" in xw.loc[e, "hint"]
    fin = xw[xw["status"] == "STATUS_FINAL"]
    assert set(fin["how"]) <= {"exact", "date_pm1", "unmatched"}
    assert (fin["how"] == "unmatched").sum() == 2


def test_cup_final_matches_when_the_006_game_is_supplied(all_events, nba_games):
    cup = pd.DataFrame([{"game_id": "0062500001", "game_date": dt.date(2025, 12, 16),
                         "home": "NYK", "away": "SAS"}])   # synthetic row for the 006 id
    xw = espn.crosswalk(all_events, pd.concat([nba_games, cup], ignore_index=True)).set_index("event_id")
    assert xw.loc["401809839", ["how", "game_id"]].tolist() == ["exact", "0062500001"]


def _ev(eid, d, home="BOS", away="NYK", status="STATUS_FINAL"):
    return {"event_id": eid, "season": "2020-21", "et_date": d, "home": home, "away": away,
            "status": status, "season_slug": "regular-season", "comp_type": "STD"}


def test_fallback_never_reuses_a_claimed_game_and_flags_ambiguity():
    d = dt.date(2021, 1, 10)
    games = pd.DataFrame([
        {"game_id": "0022000101", "game_date": d, "home": "BOS", "away": "NYK"},
        {"game_id": "0022000102", "game_date": d + dt.timedelta(days=1), "home": "BOS", "away": "NYK"},
        {"game_id": "0022000103", "game_date": d + dt.timedelta(days=5), "home": "BOS", "away": "NYK"},
        {"game_id": "0022000104", "game_date": d + dt.timedelta(days=7), "home": "BOS", "away": "NYK"},
    ])
    ev = pd.DataFrame([
        _ev("1", d + dt.timedelta(days=1)),                # exact -> 102
        _ev("2", d + dt.timedelta(days=2)),                # +-1 -> 102 (claimed) only -> unmatched
        _ev("3", d - dt.timedelta(days=1)),                # +-1 -> 101 (free)
        _ev("4", d + dt.timedelta(days=6)),                # 103 and 104 both one day away
        _ev("5", d, status="STATUS_POSTPONED"),
    ])
    xw = espn.crosswalk(ev, games).set_index("event_id")
    assert xw.loc["1", ["how", "game_id"]].tolist() == ["exact", "0022000102"]
    assert xw.loc["2", "how"] == "unmatched"
    assert xw.loc["3", ["how", "game_id"]].tolist() == ["date_pm1", "0022000101"]
    assert xw.loc["4", "how"] == "ambiguous"
    assert xw.loc["5", "how"] == "not_final"
    # 104 (d+7) lies outside the events' date span, so it is not reported.
    assert set(xw.attrs["unclaimed_games"]["game_id"]) == {"0022000103"}


def test_crosswalk_input_invariants(all_events, nba_games):
    dup = pd.concat([nba_games, nba_games.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate game_id"):
        espn.crosswalk(all_events, dup)
    bad = nba_games.copy()
    bad.loc[0, "home"] = "GS"                              # ESPN code, not canonical
    with pytest.raises(ValueError, match="non-canonical"):
        espn.crosswalk(all_events, bad)
    pair = nba_games.iloc[:1].copy()
    pair["game_id"] = "0021999999"
    with pytest.raises(ValueError, match="duplicate"):
        espn.crosswalk(all_events, pd.concat([nba_games, pair], ignore_index=True))
    with pytest.raises(ValueError, match="duplicate event_id"):
        espn.crosswalk(pd.concat([all_events, all_events.iloc[:1]]), nba_games)


def test_gamelog_neutral_rows_and_bad_matchups():
    logs = fx("gamelog_t.json.gz")
    df = http.result_sets(logs["2025-26"])["LeagueGameLog"]
    g = espn.nba_games_from_gamelog(df).set_index("game_id")
    assert bool(g.loc["0022500147", "neutral"]) and not bool(g.loc["0022500142", "neutral"])
    # The team-row shape gives the same games.
    rows = pd.DataFrame({"game_id": df["GAME_ID"], "game_date": df["GAME_DATE"],
                         "team": df["TEAM_ABBREVIATION"], "home": df["MATCHUP"].str.contains(" vs. ")})
    g2 = espn.nba_games_from_team_rows(rows).set_index("game_id")
    pd.testing.assert_frame_equal(g.sort_index(), g2.sort_index())
    broken = df.copy()
    broken.loc[broken.index[0], "MATCHUP"] = broken.loc[broken.index[0], "MATCHUP"].replace(" @ ", " vs. ")
    gid = broken.loc[broken.index[0], "GAME_ID"]
    other = broken[(broken["GAME_ID"] == gid)].index[1]
    broken.loc[other, "MATCHUP"] = broken.loc[other, "MATCHUP"].replace(" @ ", " vs. ")
    with pytest.raises(ValueError, match="one home"):
        espn.nba_games_from_gamelog(broken)
    alien = df.copy()
    alien.loc[alien.index[0], ["TEAM_ID", "TEAM_ABBREVIATION"]] = [1, "LON"]
    with pytest.raises(ValueError, match="unknown teams"):
        espn.nba_games_from_gamelog(alien)


def test_season_from_game_id():
    assert espn.season_from_game_id("0021900002") == "2019-20"
    assert espn.season_from_game_id(22500147) == "2025-26"
    with pytest.raises(ValueError):
        espn.season_from_game_id("401809839")


# ── live adapters ────────────────────────────────────────────────────────────
def _sb(name):
    j = fx(name)
    return j["data"], dt.datetime.fromisoformat(j["captured_at"])


def test_live_scoreboard_odds_vanish_at_tip_rows_do_not():
    data, cap = _sb("live_sb_20261008.json.gz")
    df = espn.parse_scoreboard(data, cap).set_index("event_id")
    assert len(df) == 6
    started = df[df["status"] != "STATUS_SCHEDULED"]
    assert len(started) == 5 and started["home_line"].isna().all()
    lal = df.loc["401898717"]
    assert (lal["home"], lal["away"], lal["provider_id"]) == ("LAL", "SAC", "100")
    assert lal["home_line"] == -10.5 and lal["total"] == 229.5
    assert lal["open_home_line"] == -6.5 and lal["open_total"] == 228.5   # 'o228.5'
    assert lal["home_ml"] == -380 and lal["away_ml"] == 300
    assert lal["captured_at"] == pd.Timestamp(cap).tz_convert("UTC")


def test_live_scoreboard_home_line_sign():
    data, cap = _sb("live_sb_20261020.json.gz")
    df = espn.parse_scoreboard(data, cap).set_index("event_id")
    assert df.loc["401909088", ["home", "home_line", "open_home_line", "open_total"]].tolist() == \
        ["DET", -1.5, -2.5, 224.5]
    assert df.loc["401909089", "home_line"] == -5.5 and df.loc["401909089", "home"] == "NYK"
    assert (df["season_slug"] == "regular-season").all() and df["et_date"].eq(dt.date(2026, 10, 20)).all()


def test_live_scoreboard_skips_live_feed_and_checks_sign():
    data, cap = _sb("live_sb_20261008.json.gz")
    data = copy.deepcopy(data)
    comp = next(e for e in data["events"] if e["id"] == "401898717")["competitions"][0]
    live = copy.deepcopy(comp["odds"][0])
    live["provider"] = {"id": "200", "name": "DraftKings - Live Odds"}
    live["spread"] = -30.0
    comp["odds"].insert(0, live)
    r = espn.parse_scoreboard(data, cap).set_index("event_id").loc["401898717"]
    assert r["home_line"] == -10.5 and "live:200" in r["qc_flags"]
    comp["odds"][1]["details"] = "SAC -10.5"
    r = espn.parse_scoreboard(data, cap).set_index("event_id").loc["401898717"]
    assert math.isnan(r["home_line"]) and "sign_conflict" in r["qc_flags"]
    with pytest.raises(ValueError, match="aware"):
        espn.parse_scoreboard(data, dt.datetime(2026, 10, 8, 20, 0))


def test_live_scoreboard_prefers_draftkings_wherever_it_sits():
    data, cap = _sb("live_sb_20261008.json.gz")
    data = copy.deepcopy(data)
    comp = next(e for e in data["events"] if e["id"] == "401898717")["competitions"][0]
    other = copy.deepcopy(comp["odds"][0])
    other["provider"] = {"id": "58", "name": "ESPN BET"}
    other["spread"], other["details"] = -8.5, "LAL -8.5"
    comp["odds"].insert(0, other)                          # DK is now second
    r = espn.parse_scoreboard(data, cap).set_index("event_id").loc["401898717"]
    assert (r["provider_id"], r["home_line"]) == ("100", -10.5)


def test_live_scoreboard_drops_a_known_live_id_with_a_plain_name():
    data, cap = _sb("live_sb_20261008.json.gz")
    data = copy.deepcopy(data)
    comp = next(e for e in data["events"] if e["id"] == "401898717")["competitions"][0]
    comp["odds"] = [copy.deepcopy(comp["odds"][0])]
    comp["odds"][0]["provider"] = {"id": "200", "name": "DraftKings"}   # live id, no 'live'
    r = espn.parse_scoreboard(data, cap).set_index("event_id").loc["401898717"]
    assert math.isnan(r["home_line"]) and r["provider_id"] is None and "live:200" in r["qc_flags"]


def test_scoreboard_converts_an_aware_datetime_to_its_et_date(monkeypatch):
    asked = []
    monkeypatch.setattr(espn, "fetch_scoreboard", lambda d, limit=500: asked.append(d) or {"events": []})
    # 02:00 UTC on Oct 21 is 10pm ET on Oct 20: the 20th's slate.
    espn.scoreboard(dt.datetime(2026, 10, 21, 2, 0, tzinfo=util.UTC))
    espn.scoreboard(dt.date(2026, 10, 21))
    espn.scoreboard("2026-10-22")
    assert asked == ["20261020", "20261021", "20261022"]
    with pytest.raises(ValueError, match="naive"):
        espn.scoreboard(dt.datetime(2026, 10, 21, 2, 0))


def test_injuries_team_from_group_and_id_from_links():
    j = fx("injuries.json.gz")
    df = espn.parse_injuries(j["data"], dt.datetime.fromisoformat(j["captured_at"]))
    by = df.set_index("name")
    # The item id (533614) is the injury, not the athlete.
    assert by.loc["Luguentz Dort", "espn_athlete_id"] == "4397020"
    assert "533614" not in set(df["espn_athlete_id"])
    # athlete.team is stale after a trade: Aldama says MEM inside the DAL group.
    assert by.loc["Santi Aldama", "team"] == "DAL" and by.loc["Brandon Ingram", "team"] == "LAC"
    assert df.attrs["stale_athlete_team"] == 2
    assert by.loc["Henri Veesaar", "fantasy_status"] == "OFS"
    assert by.loc["Brandon Ingram", "return_date"] == dt.date(2026, 11, 1)
    # Teams with no entries are simply absent from the payload.
    assert "HOU" in df.attrs["teams_without_entries"]


def test_injuries_invariants():
    j = fx("injuries.json.gz")
    cap = dt.datetime.fromisoformat(j["captured_at"])
    bad = copy.deepcopy(j["data"])
    bad["injuries"][0]["id"] = "999999"
    with pytest.raises(ValueError, match="unknown team"):
        espn.parse_injuries(bad, cap)
    bad = copy.deepcopy(j["data"])
    bad["injuries"][0]["injuries"][0]["athlete"]["links"] = []
    with pytest.raises(ValueError, match="one id"):
        espn.parse_injuries(bad, cap)
    dup = copy.deepcopy(j["data"])
    orig = dup["injuries"][0]["injuries"][0]
    older = copy.deepcopy(orig)
    older["date"], older["status"] = "2020-01-01T00:00Z", "STALE"
    dup["injuries"][0]["injuries"].append(older)
    df = espn.parse_injuries(dup, cap)
    assert df.attrs["duplicates_dropped"] == 1 and df["espn_athlete_id"].is_unique
    aid = espn._athlete_id_from_links(orig["athlete"])
    assert df.set_index("espn_athlete_id").loc[aid, "status"] == orig["status"]   # latest kept
    # ...whichever order the feed lists them in.
    newer = copy.deepcopy(orig)
    newer["date"], newer["status"] = "2099-01-01T00:00Z", "NEWEST"
    dup["injuries"][0]["injuries"].insert(0, newer)
    df = espn.parse_injuries(dup, cap)
    assert df.set_index("espn_athlete_id").loc[aid, "status"] == "NEWEST"


def test_roster():
    j = fx("roster_dal.json.gz")
    cap = dt.datetime.fromisoformat(j["captured_at"])
    df = espn.parse_roster(j["data"], "DAL", cap).set_index("full_name")
    assert df.loc["Santi Aldama", ["espn_athlete_id", "date_of_birth", "jersey", "position"]].tolist() == \
        ["4593125", dt.date(2001, 1, 10), "34", "F"]
    with pytest.raises(ValueError, match="asked for"):
        espn.parse_roster(j["data"], "MEM", cap)


def test_match_athletes_confirms_names_with_birth_dates():
    espn_df = pd.DataFrame([
        {"espn_athlete_id": "4593125", "full_name": "Santi Aldama", "date_of_birth": dt.date(2001, 1, 10), "team": "DAL"},
        {"espn_athlete_id": "1", "full_name": "Max Christie", "date_of_birth": dt.date(1999, 1, 1), "team": "DAL"},
        {"espn_athlete_id": "2", "full_name": "Moe Cisse", "date_of_birth": dt.date(2002, 9, 10), "team": "DAL"},
        {"espn_athlete_id": "3", "full_name": "Nobody Here", "date_of_birth": None, "team": "DAL"},
    ])
    nba_df = pd.DataFrame([
        {"person_id": 1630583, "name": "Santi Aldama", "birthdate": "2001-01-10T00:00:00", "team": "DAL"},
        {"person_id": 1631108, "name": "Max Christie", "birthdate": "2003-02-10T00:00:00", "team": "DAL"},
        {"person_id": 1642000, "name": "Moussa Cisse", "birthdate": "2002-09-10T00:00:00", "team": "DAL"},
    ])
    m = espn.match_athletes(espn_df, nba_df).set_index("espn_athlete_id")
    assert m.loc["4593125", "person_id"] == 1630583 and m.loc["4593125", "how"] == "name_exact+dob"
    assert m.loc["1", "how"] == "dob_conflict" and pd.isna(m.loc["1", "person_id"])
    assert m.loc["2", ["person_id", "how"]].tolist() == [1642000, "dob+team"]
    assert m.loc["3", "how"] == "none"


# ── build_lines script ───────────────────────────────────────────────────────
def _load_script():
    spec = importlib.util.spec_from_file_location("build_lines", REPO / "scripts" / "build_lines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bl():
    return _load_script()


def test_kaggle_conventions(bl):
    k = pd.DataFrame([
        {"season": 2024, "date": "2024-01-15", "home": "phi", "away": "hou", "spread": 8.5,
         "total": 227.5, "whos_favored": "home", "regular": True, "playoffs": False},
        {"season": 2024, "date": "2024-01-16", "home": "gs", "away": "utah", "spread": 3.0,
         "total": 230.0, "whos_favored": "away", "regular": True, "playoffs": False},
        {"season": 2024, "date": "2024-04-16", "home": "no", "away": "lal", "spread": 0.0,
         "total": 220.0, "whos_favored": "away", "regular": False, "playoffs": False},  # play-in
    ])
    out = bl.kaggle_lines(k)
    assert out["season"].tolist() == ["2023-24"] * 3           # END year -> label
    assert out["k_home_line"].tolist() == [-8.5, 3.0, 0.0]
    assert out["home"].tolist() == ["PHI", "GSW", "NOP"] and out["away"].tolist()[1] == "UTA"
    assert len(out) == 3                                       # regular=False & playoffs=False kept
    bad = k.copy()
    bad.loc[0, "home"] = "xyz"
    with pytest.raises(ValueError, match="unknown Kaggle team"):
        bl.kaggle_lines(bad)


def test_agreement_rates(bl):
    lines = pd.DataFrame({"et_date": [dt.date(2024, 1, 15), dt.date(2024, 1, 16), dt.date(2024, 1, 17)],
                          "home": ["PHI", "GSW", "BOS"], "away": ["HOU", "UTA", "NYK"],
                          "season": ["2023-24"] * 3, "home_line": [-8.5, 3.5, -2.0],
                          "total": [227.5, 231.0, 220.0], "n_books": [1, 1, 1], "books": ["x"] * 3})
    kag = pd.DataFrame({"et_date": [dt.date(2024, 1, 15), dt.date(2024, 1, 16), dt.date(2024, 1, 17)],
                        "home": ["PHI", "GSW", "BOS"], "away": ["HOU", "UTA", "NYK"],
                        "k_home_line": [-8.5, 3.0, 1.0], "k_total": [227.5, 230.0, 220.0]})
    res = bl.agreement(lines, kag).set_index("season")
    assert res.loc["ALL", "overlap"] == 3
    assert res.loc["ALL", "line_exact"] == pytest.approx(33.3)
    assert res.loc["ALL", "line_le0.5"] == pytest.approx(66.7)
    assert res.loc["ALL", "total_le1.0"] == 100.0
    assert res.loc["ALL", "fav_flips"] == 1


def test_kaggle_check_refuses_unless_local_only_is_ignored(bl, monkeypatch):
    class R:
        returncode = 1

    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(bl.http, "fetch", no_network)
    with pytest.raises(SystemExit, match="refusing"):
        bl.kaggle_frame()


def test_local_only_is_really_ignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "cache/local_only/probe"], cwd=REPO)
    assert r.returncode == 0


def _offline_build(monkeypatch, bl, months, odds_fx):
    logs = fx("gamelog_t.json.gz")
    monkeypatch.setattr(espn, "scoreboard_month", lambda ym, **k: months.get(ym, {"events": []}))
    monkeypatch.setattr(bl, "load_gamelog",
                        lambda season, stype, **k: http.result_sets(logs[season])["LeagueGameLog"]
                        if stype == "Regular Season" else pd.DataFrame(
                            columns=["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION"]))
    asked = []

    def fake_fetch(url, params=None, **k):
        eid = re.search(r"/events/(\d+)/", url).group(1)
        asked.append(eid)
        if eid in odds_fx:
            return odds_fx[eid]["pages"][0]
        return {"count": 0, "pageIndex": 1, "pageCount": 1, "items": []}

    monkeypatch.setattr(espn.http, "fetch", fake_fetch)
    return asked


def test_build_season_offline_preview(bl, monkeypatch, tmp_cache, months, odds_fx):
    asked = _offline_build(monkeypatch, bl, months, odds_fx)
    lines, cov = bl.build_season("2025-26", sample=None, months=["202511", "202512"],
                                 template=None, workers=2, write=True)
    assert set(asked) == {"401810002", "401809997", "401809800", "401809837", "401809838",
                          "401809839", "401810164"}
    assert lines["game_id"].is_unique and lines["game_id"].str.startswith("00225").all()
    assert len(lines) == 6                                  # the Cup final is unmatched
    assert [u[0] for u in cov["unmatched"]] == ["401809839"]
    row = lines.set_index("event_id").loc["401810164"]
    assert (row["home_line"], row["total"], row["n_books"], row["has_open"]) == (-9.5, 232.5, 1, True)
    assert not cov["complete"]
    assert not (tmp_cache / "game_lines_2025-26_v1.parquet").exists()     # never the season table
    prev = tmp_cache / "raw" / "espn_lines_preview" / "game_lines_2025-26.parquet"
    assert cache.read_parquet_meta(prev)["complete"] is False
    back = pd.read_parquet(prev)
    assert list(back.columns) == bl.LINE_COLUMNS and back["et_date"].iloc[0] == dt.date(2025, 11, 1)


def test_full_build_with_empty_months_is_refused(bl, monkeypatch, tmp_cache, months, odds_fx):
    # Only 202002 of 2019-20's ten months has events (the verifier's
    # simulation): the old build wrote 6 rows as complete=True.
    _offline_build(monkeypatch, bl, months, odds_fx)
    assert bl.main(["--seasons", "2019-20", "--workers", "1"]) == 1
    assert not (tmp_cache / "game_lines_2019-20_v1.parquet").exists()
    prev = tmp_cache / "raw" / "espn_lines_preview" / "game_lines_2019-20.parquet"
    meta = cache.read_parquet_meta(prev)
    assert meta["complete"] is False
    gate = " ".join(meta["gate_failures"])
    assert "empty months" in gate and "202003" in gate and "final NBA games" in gate
    cov = meta["coverage"]
    assert cov["month_counts"]["202002"] == 8 and cov["month_counts"]["201910"] == 0
    # 79 games in the fixture log, 6 claimed; March's 7 lie outside the
    # events' date span, which the span-only count never saw.
    assert (cov["expected_final_games"], cov["missing_games"], cov["unclaimed_nba_games"]) == (79, 73, 66)


def test_allow_incomplete_writes_the_table_marked_incomplete(bl, monkeypatch, tmp_cache, months, odds_fx):
    _offline_build(monkeypatch, bl, months, odds_fx)
    assert bl.main(["--seasons", "2019-20", "--workers", "1", "--allow-incomplete"]) == 0
    meta = cache.read_parquet_meta(tmp_cache / "game_lines_2019-20_v1.parquet")
    assert meta["complete"] is False and meta["gate_failures"]


def _one_month_season(monkeypatch, bl, months, odds_fx, nba_games, tmp_cache, keep):
    """2019-20 reduced to Feb 2020, with a games table holding `keep` game ids."""
    _offline_build(monkeypatch, bl, months, odds_fx)
    monkeypatch.setattr(espn, "season_months", lambda season: ["202002"])
    g = nba_games[nba_games["game_id"].isin(keep)]
    p = tmp_cache / "games_2019-20.parquet"
    g.to_parquet(p)
    return str(tmp_cache / "games_{season}.parquet")


FEB_2020_CLAIMED = ["0021900732", "0021900827", "0021900847", "0021900848", "0021900867", "0021900886"]


def _line_for_every_event(monkeypatch):
    def fake(eid, season, home=None, away=None, **k):
        return {"event_id": eid, "home": home, "away": away, "home_line": -3.0, "total": 220.0,
                "n_books": 1, "n_books_total": 1, "books": "x", "open_home_line": math.nan,
                "open_total": math.nan, "has_open": False, "line_field": "current",
                "qc_flags": "", "source": "espn_core"}
    monkeypatch.setattr(espn, "game_odds", fake)


def test_gate_passes_and_writes_the_complete_table(bl, monkeypatch, tmp_cache, months, odds_fx,
                                                   nba_games):
    tmpl = _one_month_season(monkeypatch, bl, months, odds_fx, nba_games, tmp_cache, FEB_2020_CLAIMED)
    _line_for_every_event(monkeypatch)
    lines, cov = bl.build_season("2019-20", sample=None, months=None, template=tmpl,
                                 workers=1, write=True)
    assert cov["gate_failures"] == [] and cov["complete"] is True
    meta = cache.read_parquet_meta(tmp_cache / "game_lines_2019-20_v1.parquet")
    assert meta["complete"] is True and meta["coverage"]["match_counts"]["date_pm1"] == 2
    assert len(lines) == 6 and lines["game_id"].is_unique


def test_gate_sees_a_missing_game_outside_the_events_date_span(bl, monkeypatch, tmp_cache, months,
                                                               odds_fx, nba_games):
    # The verifier's "missing first or last month": a final game AFTER the
    # last ESPN event (here 2020-03-01) was invisible to the span check.
    march = nba_games.loc[pd.to_datetime(nba_games["game_date"]) == pd.Timestamp("2020-03-01"),
                          "game_id"].iloc[0]
    tmpl = _one_month_season(monkeypatch, bl, months, odds_fx, nba_games, tmp_cache,
                             FEB_2020_CLAIMED + [march])
    _line_for_every_event(monkeypatch)
    _, cov = bl.build_season("2019-20", sample=None, months=None, template=tmpl, workers=1, write=True)
    assert cov["unclaimed_nba_games"] == 0 and cov["missing_games"] == 1
    assert any("1 of 7 final NBA games" in g for g in cov["gate_failures"])
    assert not (tmp_cache / "game_lines_2019-20_v1.parquet").exists()


def test_gate_line_coverage_and_unmatched_rules(bl):
    base = {"empty_months": [], "missing_games": 0, "expected_final_games": 1311,
            "missing_examples": [], "unmatched_blocking": 0, "lines_rows": 1311,
            "with_line_pct": 96.95, "odds_absent": 0}
    assert bl.completeness_gate(base) == []                 # 2014-15's real coverage passes
    low = bl.completeness_gate({**base, "with_line_pct": 61.0, "odds_absent": 500})
    assert len(low) == 1 and "line coverage 61.0%" in low[0] and "--retry-absent" in low[0]
    assert bl.completeness_gate({**base, "unmatched_blocking": 1})
    # Before opening night: no final games, no lines, nothing to refuse.
    assert bl.completeness_gate({**base, "expected_final_games": 0, "lines_rows": 0,
                                 "with_line_pct": None}) == []


def test_only_an_unkeyable_cup_final_is_tolerated(bl, all_events, nba_games):
    ev = all_events[all_events["season"] == "2025-26"]
    xw = espn.crosswalk(ev, nba_games)                       # Regular Season log: no 006
    empty = pd.DataFrame(columns=bl.LINE_COLUMNS)
    cov = bl.coverage("2025-26", ev, {}, xw, xw.iloc[0:0], xw.iloc[0:0], empty, {}, True)
    assert cov["unmatched_blocking"] == 0 and len(cov["unmatched"]) == 1    # the CC final
    xw2 = xw.copy()
    xw2.loc[xw2["comp_type"] == "CC", "comp_type"] = "STD"
    cov = bl.coverage("2025-26", ev, {}, xw2, xw.iloc[0:0], xw.iloc[0:0], empty, {}, True)
    assert cov["unmatched_blocking"] == 1


def test_lines_et_date_is_the_nba_date_on_pm1_rows(bl, monkeypatch, tmp_cache, months, odds_fx):
    # ESPN's corrupted Feb-2020 timestamps put 401161528 on 2020-02-29; the
    # game (0021900886) was played on the 28th.
    _offline_build(monkeypatch, bl, months, odds_fx)
    lines, _ = bl.build_season("2019-20", sample=None, months=["202002"], template=None,
                               workers=1, write=False)
    r = lines.set_index("event_id")
    assert r.loc["401161528", ["match", "et_date", "espn_et_date"]].tolist() == \
        ["date_pm1", dt.date(2020, 2, 28), dt.date(2020, 2, 29)]
    assert r.loc["401161490", ["et_date", "espn_et_date"]].tolist() == [dt.date(2020, 2, 23), dt.date(2020, 2, 24)]
    ex = r[r["match"] == "exact"]
    assert (ex["et_date"] == ex["espn_et_date"]).all()


@pytest.mark.parametrize("err", [http.BadPayload("u", "HTML error page"), ValueError("parser")])
def test_first_non_blocked_error_cancels_the_queue(bl, monkeypatch, tmp_cache, err):
    calls = []

    def boom(eid, season, **k):
        calls.append(eid)
        time.sleep(0.02)      # give the main thread time to cancel, as a real request would
        raise err

    monkeypatch.setattr(espn, "game_odds", boom)
    targets = pd.DataFrame({"event_id": [str(400000000 + i) for i in range(300)],
                            "home": "BOS", "away": "NYK"})
    with pytest.raises(type(err)) as ei:
        bl.fetch_lines(targets, "2019-20", workers=1)
    # The old loop let all 300 run before the error surfaced.
    assert len(calls) <= 3
    assert any("400000000" in n for n in getattr(ei.value, "__notes__", []))


def test_gamelog_saved_mid_season_is_refetched(bl, monkeypatch, tmp_cache):
    logs = fx("gamelog_t.json.gz")
    path = cache.dc_path("raw/espn_xwalk/leaguegamelog_2019-20_RegularSeason.json.gz")
    calls = []
    monkeypatch.setattr(bl.http, "stats_call", lambda *a, **k: calls.append(k) or logs["2019-20"])
    cache.write_json(path, {"fetched_at": _stamp("2020-03-01T17:00:00+00:00"), "data": logs["2019-20"]},
                     gz=True)
    bl.load_gamelog("2019-20", "Regular Season")
    assert len(calls) == 1                                  # mid-season copy: refetched
    bl.load_gamelog("2019-20", "Regular Season")
    assert len(calls) == 1                                  # now settled
    cache.write_json(path, {"fetched_at": _stamp("2020-08-15T17:00:00+00:00"), "data": logs["2019-20"]},
                     gz=True)
    bl.load_gamelog("2019-20", "Regular Season")
    assert len(calls) == 1
    bl.load_gamelog("2019-20", "Regular Season", refresh=True)
    assert len(calls) == 2


def test_cli_passes_refresh_and_retry_absent(bl, monkeypatch):
    seen = {}

    def fake(season, **k):
        seen.update(k)
        return pd.DataFrame(columns=bl.LINE_COLUMNS), {}

    monkeypatch.setattr(bl, "build_season", fake)
    assert bl.main(["--seasons", "2019-20", "--refresh", "--retry-absent", "--no-write"]) == 0
    assert seen["refresh"] and seen["retry_absent"] and not seen["allow_incomplete"]


def test_build_season_prefers_the_schedule_table(bl, monkeypatch, tmp_cache, months, odds_fx):
    asked = _offline_build(monkeypatch, bl, months, odds_fx)
    monkeypatch.setattr(bl, "load_gamelog", no_network)
    logs = fx("gamelog_t.json.gz")
    g = espn.nba_games_from_gamelog(http.result_sets(logs["2025-26"])["LeagueGameLog"])
    sched = pd.DataFrame({
        "game_id": list(g["game_id"]) + ["0062500001", "0012500001", "0062600001"],
        "game_type": ["RS"] * len(g) + ["CUP_FINAL", "PRE", "CUP_FINAL"],
        "game_date": pd.to_datetime(list(g["game_date"]) + [dt.date(2025, 12, 16), dt.date(2025, 10, 5),
                                                             dt.date(2026, 12, 15)]),
        "home": list(g["home"]) + ["NYK", "DAL", None],    # last row: TBD teams
        "away": list(g["away"]) + ["SAS", "LAL", None],
        "is_neutral": list(g["neutral"]) + [True, False, True],
    })
    cache.write_parquet(sched, tmp_cache / "schedule_2025-26_v1.parquet")
    lines, cov = bl.build_season("2025-26", sample=None, months=["202511", "202512"],
                                 template=None, workers=2, write=False)
    assert cov["nba_games_source"] == "schedule_2025-26_v1.parquet"
    assert cov["unmatched"] == []
    cup = lines.set_index("event_id").loc["401809839"]
    assert cup["game_id"] == "0062500001" and cup["home_line"] == -2.5 and cup["comp_type"] == "CC"
    assert "401809839" in asked


def test_build_season_stops_on_blocked(bl, monkeypatch, tmp_cache, months, odds_fx):
    _offline_build(monkeypatch, bl, months, odds_fx)

    def blocked(url, **k):
        raise http.Blocked(url, "timeout")

    monkeypatch.setattr(espn.http, "fetch", blocked)
    with pytest.raises(SystemExit, match="blocked"):
        bl.build_season("2025-26", sample=None, months=["202511"], template=None, workers=2, write=True)
    assert not (tmp_cache / "raw" / "espn_lines_preview" / "game_lines_2025-26.parquet").exists()


def test_sample_is_spread_and_deterministic(bl):
    df = pd.DataFrame({"event_id": [str(i) for i in range(100)]})
    s = bl._sample(df, 5)
    assert s["event_id"].tolist() == ["0", "25", "50", "74", "99"]
    assert bl._sample(df, 500) is df


# ── hygiene ──────────────────────────────────────────────────────────────────
def test_all_network_goes_through_bucketlib_http():
    for p in (REPO / "bucketlib/sources/espn.py", REPO / "scripts/build_lines.py"):
        src = p.read_text()
        assert not re.search(r"^\s*(import|from)\s+(requests|urllib|curl_cffi|httpx)\b", src, re.M), p


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("BUCKETVALUE_NETWORK") != "1",
                    reason="live ESPN call; set BUCKETVALUE_NETWORK=1")
def test_live_scoreboard_smoke():
    df = espn.scoreboard(util.today_et())
    assert list(df.columns) == espn.SCOREBOARD_COLUMNS
