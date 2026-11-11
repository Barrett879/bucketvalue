"""bucketlib.store's side of the licensed odds feed (design 2.7 and 14 step 15;
layout B, amendment A3; build amendments B2, B10 and N4): feed_on,
feed_start and feed_result, the work order (_feed_work, feed_work), the feed
tables cache (_feed_tables, feed_tables, _strict_digest), feed_info, the
loading watch, the pill and chip wording behind a feed flag, and view_params
carrying the picked book.

Nothing here reaches a network or holds a key: conftest blocks the vendor and
deletes the key variables for the session, and the tests that turn the feed on
set a fake key built at run time ("K" * 40, from tests/test_live_feed.py)
through monkeypatch. The vendor is test_live_feed's fake (Vendor), answering
from the CONSTRUCTED fixtures in tests/fixtures/oddsapi; the anchor's pool is
its synchronous SyncPool, so no task outlives a test; the refresher stays off
(conftest). The world is the serve fixture's 2026-03-12 (a fixed past date)
through serve_harness, with every clock the pages read pinned, or a
CONSTRUCTED schedule built here (ids invented; the two real games of
2026-03-12 among them). Every expected count was worked out by hand from that
construction (the comments say how), never read back from the code.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from bucketlib import dist as D
from bucketlib import http
from bucketlib import live
from bucketlib import oddsbudget as OB
from bucketlib import pipeline as P
from bucketlib import props, simprov, store
from bucketlib import state as S
from bucketlib.util import ET, UTC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_live_feed as LF  # noqa: E402  (the fake vendor, the pools, the fake key)
import test_props_feed as PF  # noqa: E402  (the feed frame through the real parser)

KEY = LF.KEY                                   # "K" * 40, built at run time
DAY = H.DATE                                   # 2026-03-12
EARLY, LATE = H.EARLY, H.LATE                  # PHI at DET 7 PM ET, DEN at SAS 9 PM ET
MORNING = LF.MORNING                           # 08:55 ET on Mar 12: the first pulls are due
MORNING_UTC = LF.MORNING_UTC


# -- Fixtures ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fresh_feed_state(monkeypatch):
    """No anchor and no cached work order or tables before or after a test,
    and the once-per-process off line said afresh. (The caches are held here
    as they were, since a test may replace a name in store.)"""
    caches = (store._feed_work, store._season_games, store._feed_tables)
    live.reset_feed_anchor_for_tests()
    for fn in caches:
        fn.clear()
    monkeypatch.setattr(store, "_off_said", set())
    yield
    live.reset_feed_anchor_for_tests()
    for fn in caches:
        fn.clear()


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    return caplog


@pytest.fixture
def key_on(monkeypatch):
    """The feed on: a fake key in the process environment (the refresher
    stays off: conftest)."""
    monkeypatch.setenv(live.KEY_ENV, KEY)


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "bucketvalue"]


def _pin_store(monkeypatch, at: dt.datetime) -> None:
    monkeypatch.setattr(store, "now_utc", lambda: at.astimezone(UTC))


def _raises(name):
    def f(*a, **k):
        raise AssertionError(f"{name} ran with the feed off")
    return f


# -- A constructed schedule ----------------------------------------------------------------
TEAMS = ("BOS", "NYK", "MIA", "CHI", "LAL", "GSW")


def _game(gid, day, hh, mm, away, home, *, kind="RS", postponed=False, tbd=False) -> dict:
    tip = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=ET).astimezone(UTC)
    return {"game_id": gid, "season": "2025-26", "game_type": kind, "game_date": day,
            "tip_utc": pd.Timestamp(tip), "status": "scheduled",
            "status_text": "TBD" if tbd else "7:30 pm ET", "home": home, "away": away,
            "postponed": postponed, "time_tbd": tbd}


def season() -> pd.DataFrame:
    """CONSTRUCTED: three regular-season games a day at 7:30 PM ET from Feb 26
    to Apr 3, 2026 (ids g<mmdd><i>). On Mar 12 the first two are the serve
    fixture's real PHI at DET (7 PM) and DEN at SAS (9 PM), the third stays
    (no committed file names it), and a play-in game and a postponed game are
    added (neither is projected). On Mar 20 a fourth game has no tip time
    yet (a made-up 00:00 ET). Projected games: Feb 9, March 31 x 3 + 1 = 94,
    April 9, so the busiest month is March with 94."""
    rows, d = [], dt.date(2026, 2, 26)
    while d <= dt.date(2026, 4, 3):
        day = [_game(f"g{d:%m%d}{i}", d, 19, 30, TEAMS[i], TEAMS[i + 3]) for i in range(3)]
        if d == DAY:
            day[0] = _game(EARLY, d, 19, 0, "PHI", "DET")
            day[1] = _game(LATE, d, 21, 0, "DEN", "SAS")
            day += [_game("p0312", d, 18, 0, "ATL", "ORL", kind="PI"),
                    _game("x0312", d, 20, 0, "UTA", "POR", postponed=True)]
        if d == dt.date(2026, 3, 20):
            day.append(_game("t0320", d, 0, 0, "OKC", "HOU", tbd=True))
        rows += day
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def _committed_file(path: Path, ids) -> Path:
    """A committed serve file holding these game ids (and a column the work
    order must not need)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": list(ids), "person_id": list(range(len(ids)))}).to_parquet(path)
    return path


@pytest.fixture
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    return tmp_path / "state"


# -- feed_on and the no-key path (M9) ------------------------------------------------------
def test_the_feed_is_on_only_with_a_key_and_the_switch_on(monkeypatch, logs):
    """design 2.7: a key and the kill switch, read on this call; an empty
    mapping is no key (m11); the off line is said once per process."""
    assert store.feed_on() is False
    assert store.feed_on({}) is False
    assert store.feed_on({live.KEY_ENV: "   "}) is False
    assert store.feed_on({live.KEY_ENV: KEY}) is True
    for off in ("0", "no", "maybe"):                  # B6: an unreadable value is off
        assert store.feed_on({live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": off}) is False
    assert store.feed_on({live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": "1"}) is True
    monkeypatch.setenv(live.KEY_ENV, KEY)
    assert store.feed_on() is True and store.feed_on({}) is False
    said = _messages(logs)
    assert said.count("oddsapi: no key; feed off") == 1
    assert said.count("oddsapi: switched off") == 1
    assert all(KEY not in m for m in said) and KEY not in logs.text


def test_with_no_key_feed_start_is_off_before_any_feed_code_runs_m9(monkeypatch, state_dir):
    """M9: with no key, feed_start returns "off" before the work order, the
    schedule helpers or the anchor, so nothing of the feed can raise (a work
    order that raises included) and "down" cannot appear; feed_result gives
    FeedLive.off() without reading memory. No vendor call (conftest's block
    fails the test on any)."""
    for mod, name in ((store, "feed_work"), (store, "_feed_work"), (store, "_committed"),
                      (store, "_order_sched_key"), (P, "state_games"), (live, "feed_kick"),
                      (live, "ensure_refresher"), (live, "feed_anchor"), (live, "feed_view"),
                      (live, "prop_lines_memory")):
        monkeypatch.setattr(mod, name, _raises(name))
    _pin_store(monkeypatch, MORNING)
    for when in (DAY, DAY - dt.timedelta(days=1)):
        ticket = store.feed_start(when, season())
        assert ticket == store.FeedTicket(when, "off")
        got = store.feed_result(ticket)
        assert got.state == "off" and got.value.empty and got.per_game == {} and got.loading == 0
    assert live.ANCHOR_NAME not in sys.modules


def test_the_kill_switch_is_off_too_even_with_a_key(monkeypatch, key_on):
    monkeypatch.setenv("BUCKETVALUE_ODDS_ENABLED", "0")
    monkeypatch.setattr(store, "feed_work", _raises("feed_work"))
    monkeypatch.setattr(live, "feed_kick", _raises("feed_kick"))
    _pin_store(monkeypatch, MORNING)
    assert store.feed_start(DAY, season()).mode == "off"
    assert live.ANCHOR_NAME not in sys.modules


def test_another_date_reads_memory_and_never_kicks(monkeypatch, key_on):
    """design 1.1: today only. Another date's ticket is "memory": no work
    order, no kick; its result is memory's, read without waiting."""
    monkeypatch.setattr(store, "feed_work", _raises("feed_work"))
    monkeypatch.setattr(live, "feed_kick", _raises("feed_kick"))
    seen = []
    monkeypatch.setattr(live, "prop_lines_memory",
                        lambda date, **k: seen.append(date) or live.FeedLive.off())
    _pin_store(monkeypatch, MORNING)
    for when in (DAY - dt.timedelta(days=1), DAY + dt.timedelta(days=1)):
        ticket = store.feed_start(when, season())
        assert ticket == store.FeedTicket(when, "memory")
        store.feed_result(ticket)
    assert seen == [DAY - dt.timedelta(days=1), DAY + dt.timedelta(days=1)]


def test_a_kick_that_raises_still_gives_kicked_and_the_page_reads_memory(monkeypatch, key_on, logs):
    """design 2.7: an exception inside feed_start logs its type and still
    returns "kicked", so feed_result reads memory (waiting PAGE_WAIT_S)."""
    def boom(*a, **k):
        raise RuntimeError("vendor-said-xyzzy " + KEY)
    monkeypatch.setattr(store, "feed_work", boom)
    waits = []
    monkeypatch.setattr(live, "feed_view", lambda date, **k: waits.append((date, k)) or live.FeedLive.off())
    _pin_store(monkeypatch, MORNING)
    ticket = store.feed_start(DAY, season())
    assert ticket == store.FeedTicket(DAY, "kicked")
    store.feed_result(ticket)
    assert waits == [(DAY, {"wait_s": store.PAGE_WAIT_S})] and store.PAGE_WAIT_S == 3.0
    assert _messages(logs) == ["oddsapi: feed start failed (RuntimeError)"]
    assert "xyzzy" not in logs.text and KEY not in logs.text


def test_a_result_that_raises_is_down_and_logs_only_the_type(monkeypatch, key_on, logs):
    def boom(*a, **k):
        raise ValueError("vendor-said-xyzzy")
    monkeypatch.setattr(live, "feed_view", boom)
    monkeypatch.setattr(live, "prop_lines_memory", boom)
    for mode in ("kicked", "memory"):
        got = store.feed_result(store.FeedTicket(DAY, mode))
        assert got.state == "down" and got.value.empty
    assert _messages(logs) == ["oddsapi: feed result failed (ValueError)"] * 2
    assert "xyzzy" not in logs.text
    assert store.feed_result(store.FeedTicket(DAY, "nonsense")).state == "off"


def test_feed_start_kicks_the_walk_then_starts_the_refresher_inside_its_try_n4(
        monkeypatch, key_on, logs):
    """Build amendment N4 (build 3's review, F2): for today store.feed_start
    calls live.feed_kick with the work order, then live.ensure_refresher,
    once each, inside its try. With no game to kick (an empty schedule) the
    refresher still starts: it buys from the evening's order or today's
    files. Another date ("memory") and the feed off ("off") start neither. A
    refresher that will not start is logged by its type, and the page still
    reads memory. conftest keeps the real refresher off; these are
    recorders."""
    calls = []
    work = {"date": DAY.isoformat(), "games": []}
    monkeypatch.setattr(store, "feed_work", lambda day, sched: calls.append(("work", day)) or work)
    monkeypatch.setattr(live, "feed_kick", lambda w, **k: calls.append(("kick", w is work)))
    monkeypatch.setattr(live, "ensure_refresher", lambda *a, **k: calls.append(("refresher",)))
    _pin_store(monkeypatch, MORNING)
    assert store.feed_start(DAY, season()) == store.FeedTicket(DAY, "kicked")
    assert calls == [("work", DAY), ("kick", True), ("refresher",)]
    for nothing in (season().iloc[:0], None):
        calls.clear()
        assert store.feed_start(DAY, nothing) == store.FeedTicket(DAY, "kicked")
        assert calls == [("refresher",)]
    calls.clear()
    for when in (DAY - dt.timedelta(days=1), DAY + dt.timedelta(days=1)):
        assert store.feed_start(when, season()).mode == "memory"
    assert calls == []

    def boom(*a, **k):
        raise RuntimeError("vendor-said-xyzzy " + KEY)
    with monkeypatch.context() as m:
        m.setattr(live, "ensure_refresher", boom)
        assert store.feed_start(DAY, season()) == store.FeedTicket(DAY, "kicked")
    assert calls == [("work", DAY), ("kick", True)]
    assert _messages(logs) == ["oddsapi: feed start failed (RuntimeError)"]
    assert "xyzzy" not in logs.text and KEY not in logs.text
    calls.clear()
    monkeypatch.delenv(live.KEY_ENV)
    assert store.feed_start(DAY, season()) == store.FeedTicket(DAY, "off") and calls == []


# -- The work order: buyable games (m26), the horizon (B2), its rows (N4) ---------------------
def test_buyable_games_are_projected_and_in_the_dates_committed_file_m26(monkeypatch, state_dir):
    """m26: a game is bought only when the site projects it AND the date's
    committed projections file names it, else its features file when there is
    no projections file; with neither, nothing. The third RS game of Mar 12 is
    projected but named by no file; the play-in and the postponed game are
    named by the file but are not projected."""
    _pin_store(monkeypatch, dt.datetime(2026, 3, 12, 8, 55, tzinfo=ET))
    sched = season()
    proj = _committed_file(P.projections_path(DAY), [EARLY, LATE, "p0312", "x0312"])
    _committed_file(P.features_path(DAY), [LATE])
    work = store.feed_work(DAY, sched)
    assert [g["game_id"] for g in work["games"]] == [EARLY, LATE]
    proj.unlink()                                              # features file only: DEN at SAS
    assert [g["game_id"] for g in store.feed_work(DAY, sched)["games"]] == [LATE]
    P.features_path(DAY).unlink()                              # neither: a bot-outage day
    empty = store.feed_work(DAY, sched)
    assert empty["games"] == [] and len(empty["horizon"]) == 100       # the horizon stays


def test_the_work_order_is_plain_data_with_teams_on_every_row_n4(monkeypatch, state_dir):
    """design 2.4.5 and N4: plain values only, and every games and horizon
    row carries its teams (the refresher matches vendor events by the date
    and the team pair). The tip is an ISO UTC string; a TBD tip is flagged."""
    at = dt.datetime(2026, 3, 12, 21, 30, tzinfo=ET)
    _pin_store(monkeypatch, at)
    _committed_file(P.projections_path(DAY), [EARLY, LATE])
    work = store.feed_work(DAY, season())
    assert set(work) == {"date", "written_at", "games", "horizon", "serve_files", "busiest"}
    assert work["date"] == "2026-03-12" and work["written_at"] == "2026-03-13T01:30:00+00:00"
    assert work["games"][0] == {"game_id": EARLY, "game_date": "2026-03-12",
                                "tip_utc": "2026-03-12T23:00:00+00:00", "tbd": False,
                                "away": "PHI", "home": "DET"}
    keys = {"game_id", "game_date", "tip_utc", "tbd", "away", "home"}
    assert all(set(r) == keys and r["away"] and r["home"] for r in work["games"] + work["horizon"])
    tbd = [r for r in work["horizon"] if r["game_id"] == "t0320"]
    assert tbd == [{"game_id": "t0320", "game_date": "2026-03-20",
                    "tip_utc": "2026-03-20T04:00:00+00:00", "tbd": True, "away": "OKC",
                    "home": "HOU"}]
    assert work["busiest"] == 94                                    # March: 31 x 3 + the TBD game
    plain = (str, int, bool, type(None), list, dict)

    def walk(v):
        assert isinstance(v, plain), type(v)
        for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()):
            walk(x)
    walk(work)
    # The refresher reads tomorrow's committed files from tonight's order (2.4.5).
    assert work["serve_files"] == {
        d.isoformat(): [str(P.projections_path(d).absolute()), str(P.features_path(d).absolute())]
        for d in (DAY, DAY + dt.timedelta(days=1))}
    assert all(Path(p).is_absolute() for ps in work["serve_files"].values() for p in ps)


@pytest.mark.parametrize("today, first, last, rows", [
    # A plain day: the period's own opening day (its window's first day, the
    # last day of the month before) to 08:00 ET on the 1st after. Feb 28 (3),
    # March (94) and Apr 1 (3).
    (dt.date(2026, 3, 12), dt.date(2026, 2, 28), dt.date(2026, 4, 1), 100),
    # The window W(Apr) opens at 05:00 ET on Mar 31: the old period may still
    # run, so the horizon starts at ITS opening day (B2) and runs to the new
    # period's end: Feb 28 (3), March (94) and Apr 1-3 (9).
    (dt.date(2026, 3, 31), dt.date(2026, 2, 28), dt.date(2026, 5, 1), 106),
    (dt.date(2026, 4, 1), dt.date(2026, 2, 28), dt.date(2026, 5, 1), 106),
    # Past the window: April's own period, from Mar 31 (3) and April (9).
    (dt.date(2026, 4, 2), dt.date(2026, 3, 31), dt.date(2026, 5, 1), 12),
    # Inside W(Mar) on Mar 1, and on Feb 28 once it opens: February's period
    # opened on Jan 31. Feb 26-28 (9), March (94), Apr 1 (3).
    (dt.date(2026, 3, 1), dt.date(2026, 1, 31), dt.date(2026, 4, 1), 106),
    (dt.date(2026, 2, 28), dt.date(2026, 1, 31), dt.date(2026, 4, 1), 106),
])
def test_the_horizon_starts_at_the_earliest_possible_period_opening_b2(
        monkeypatch, state_dir, today, first, last, rows):
    """B2: the work order's horizon holds the PERIOD'S projected games from
    the earliest possible period opening, tipped ones too, to the horizon
    end; inside a reset window that is the previous month's period opening
    day."""
    _pin_store(monkeypatch, dt.datetime.combine(today, dt.time(12), tzinfo=ET))
    assert store._horizon_days(today) == (first, last)
    work = store.feed_work(today, season())
    days = [r["game_date"] for r in work["horizon"]]
    assert len(days) == rows and days == sorted(days)
    assert min(days) >= first.isoformat() and max(days) <= last.isoformat()
    assert not {"p0312", "x0312"} & {r["game_id"] for r in work["horizon"]}   # never projected


def test_the_horizon_holds_every_period_the_governor_can_pick_on_that_day_b2():
    """At every hour of a month's turn, both periods oddsbudget.period_of can
    name (no balance known: the old one; nothing used: the new one) open on
    or after the horizon's first day and end by its last: so the governor's
    own period filter always finds the period's games, tipped ones too."""
    day = dt.date(2026, 3, 29)
    while day <= dt.date(2026, 4, 3):
        first, last = store._horizon_days(day)
        for hour in range(24):
            t = dt.datetime.combine(day, dt.time(hour, 30), tzinfo=ET)
            for used, allowance in ((None, None), (0, 500), (480, 500)):
                p = OB.period_of(t, used, allowance, None, 1, [])
                assert first <= p.opens.astimezone(ET).date(), (day, hour, used)
                assert p.horizon_end.astimezone(ET).date() <= last, (day, hour, used)
        day += dt.timedelta(days=1)


def test_the_work_order_is_cached_on_its_schedule_columns_and_the_files(monkeypatch, state_dir):
    """design 2.4.5: cached on the schedule's strict digest and the committed
    files' (path, mtime_ns, size). A score moving during a game is not a
    column the order reads, so it hits; a tip that moves, or a file the bot
    rewrites, builds it again; a schedule that cannot be hashed is never
    cached."""
    _pin_store(monkeypatch, MORNING)
    built = []
    real = store._work_order
    monkeypatch.setattr(store, "_work_order", lambda *a, **k: built.append(1) or real(*a, **k))
    sched = season()
    path = _committed_file(P.projections_path(DAY), [EARLY])
    first = store.feed_work(DAY, sched)
    scored = sched.assign(status=["live"] * len(sched), status_text=["Q2 4:10"] * len(sched))
    assert store.feed_work(DAY, scored) == first and len(built) == 1
    moved = sched.copy()
    moved.loc[moved["game_id"] == EARLY, "tip_utc"] = pd.Timestamp("2026-03-12T23:30:00Z")
    assert store.feed_work(DAY, moved)["games"][0]["tip_utc"] == "2026-03-12T23:30:00+00:00"
    assert len(built) == 2
    _committed_file(path, [EARLY, LATE])                       # the bot publishes again
    assert [g["game_id"] for g in store.feed_work(DAY, sched)["games"]] == [EARLY, LATE]
    assert len(built) == 3
    odd = sched.assign(home=[["list"]] + list(sched["home"].iloc[1:]))
    assert store._order_sched_key(odd) is None
    store.feed_work(DAY, odd)
    store.feed_work(DAY, odd)
    assert len(built) == 5


# -- The feed tables cache (m19) -------------------------------------------------------------
@pytest.fixture(scope="module")
def world():
    return H.load()


@pytest.fixture(scope="module")
def feed_frame(world):
    """The day's feed frame as the live layer holds it: both constructed
    payloads through the real parser, merged as first pulls at 9 AM ET."""
    return PF.feed(world)


def _feed_live(frame, generation=3, state="ok") -> live.FeedLive:
    return live.FeedLive(state=state, value=frame, asof=MORNING_UTC, per_game={},
                         generation=generation)


@pytest.fixture
def counted(monkeypatch):
    """prepare_feed counted (the tables' first step), and a cheap provider:
    the cache is under test here, not the simulation."""
    calls = []
    real = props.prepare_feed
    monkeypatch.setattr(props, "prepare_feed", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(simprov, "provider_for", lambda frame, params: props.MeanDispersionProvider(frame))
    return calls


def test_the_tables_are_the_production_compare_of_the_feeds_lines(world, feed_frame):
    """feed_tables is props.prepare_feed then props.compare_books with the
    date's shared simulator provider (simprov.provider_for), table for table."""
    params = D.SimParams()
    served = types.SimpleNamespace(frame=world.proj)
    got = store.feed_tables(DAY, _feed_live(feed_frame), served, params, world.roster)
    lines, _ = props.prepare_feed(feed_frame, world.roster, world.proj)
    want = props.compare_books(lines, simprov.provider_for(world.proj, params), world.proj,
                               world.roster)
    assert isinstance(got, props.FeedTables) and set(got.tables) == set(want.tables)
    assert len(got.tables) == 9                       # PrizePicks has not posted DEN at SAS
    for k in want.tables:
        pd.testing.assert_frame_equal(got.tables[k], want.tables[k])
        assert got.metas[k]["posted"] == want.metas[k]["posted"]


def test_the_tables_cache_keys_on_the_generation_and_strict_digests_m19(world, feed_frame, counted):
    params = D.SimParams()
    served = types.SimpleNamespace(frame=world.proj)
    fl = _feed_live(feed_frame)
    first = store.feed_tables(DAY, fl, served, params, world.roster)
    assert store.feed_tables(DAY, fl, served, params, world.roster) is first     # read-only, shared
    assert len(counted) == 1
    store.feed_tables(DAY, _feed_live(feed_frame, generation=4), served, params, world.roster)
    assert len(counted) == 2                                     # the ledger moved
    fewer = feed_frame[feed_frame["book"] != "fanduel"]
    store.feed_tables(DAY, _feed_live(fewer, generation=4), served, params, world.roster)
    assert len(counted) == 3                                     # the feed's rows changed
    store.feed_tables(DAY, fl, types.SimpleNamespace(frame=world.proj.iloc[1:]), params, world.roster)
    assert len(counted) == 4                                     # the frame changed
    store.feed_tables(DAY, fl, served, params, world.roster.iloc[1:])
    assert len(counted) == 5                                     # the roster changed
    store.feed_tables(DAY, fl, served, D.SimParams(beta={"fga": 1.0}), world.roster)
    assert len(counted) == 6                                     # the parameters changed
    # Same length, different content: never a len() fallback (m19).
    a = world.roster.copy()
    b = a.assign(name=a["name"].iloc[::-1].to_numpy())
    assert len(a) == len(b) and store._strict_digest(a) != store._strict_digest(b)
    assert store._strict_digest(a) == store._strict_digest(a.copy())
    assert store._strict_digest(a.rename(columns={"name": "nm"})) != store._strict_digest(a)


def test_a_digest_that_cannot_be_made_is_never_cached_m19(world, feed_frame, counted):
    """None means do not cache: an unhashable roster is compared every time."""
    params = D.SimParams()
    served = types.SimpleNamespace(frame=world.proj)
    odd = world.roster.assign(position=[["G"]] * len(world.roster))
    assert store._strict_digest(odd) is None and store._strict_digest(None) is None
    assert store._params_digest(object()) is None
    for _ in range(3):
        assert isinstance(store.feed_tables(DAY, _feed_live(feed_frame), served, params, odd),
                          props.FeedTables)
    assert len(counted) == 3
    store.feed_tables(DAY, _feed_live(feed_frame), served, object(), world.roster)
    assert len(counted) == 4                                     # parameters with no digest


def test_no_tables_with_the_feed_off_no_lines_no_frame_or_no_parameters(world, feed_frame, counted):
    """None when there is nothing to compare (the feed off, no line held, no
    frame). Lines held with no simulator parameters to compare them are
    tables marked failed, so the board explains itself (build 3's review,
    F1); nothing is computed for any of them."""
    served = types.SimpleNamespace(frame=world.proj)
    params = D.SimParams()
    assert store.feed_tables(DAY, live.FeedLive.off(), served, params, world.roster) is None
    assert store.feed_tables(DAY, _feed_live(feed_frame, state="off"), served, params,
                             world.roster) is None
    assert store.feed_tables(DAY, _feed_live(feed_frame.iloc[:0]), served, params,
                             world.roster) is None
    assert store.feed_tables(DAY, _feed_live(feed_frame), None, params, world.roster) is None
    assert store.feed_tables(DAY, _feed_live(feed_frame), types.SimpleNamespace(frame=world.proj.iloc[:0]),
                             params, world.roster) is None
    assert store.feed_tables(DAY, _feed_live(feed_frame), served, None, world.roster) == \
        props.FeedTables(failed=True)
    assert counted == []


def test_the_tables_never_raise(world, feed_frame, monkeypatch, logs):
    """A step that raises gives tables marked failed (the feed holds lines),
    logged by the exception's type only (build 3's review, F1)."""
    def boom(*a, **k):
        raise KeyError("vendor-said-xyzzy")
    monkeypatch.setattr(props, "compare_books", boom)
    assert store.feed_tables(DAY, _feed_live(feed_frame), types.SimpleNamespace(frame=world.proj),
                             D.SimParams(), world.roster) == props.FeedTables(failed=True)
    assert _messages(logs) == ["oddsapi: feed tables failed (KeyError)"]
    assert "xyzzy" not in logs.text


def test_lines_left_with_nothing_to_compare_are_failed_tables_never_cached_f1(
        world, feed_frame, counted, monkeypatch, logs):
    """Build 3's review, F1: the feed holds lines and the frame is there, but
    no table can be built. prepare_feed never raises: its own error (here a
    step inside it that raises) leaves no line to compare, and neither does a
    feed whose every row is dropped. Each gives tables marked failed, logged
    by type, and a failure is never cached: the next view tries again, and
    once the step works the real tables come back (and are cached)."""
    served = types.SimpleNamespace(frame=world.proj)
    params = D.SimParams()
    fl = _feed_live(feed_frame)
    real = props._prepare_feed

    def broken(*a, **k):
        raise ValueError("vendor-said-xyzzy")
    monkeypatch.setattr(props, "_prepare_feed", broken)
    for n in (1, 2):
        got = store.feed_tables(DAY, fl, served, params, world.roster)
        assert got == props.FeedTables(failed=True) and len(counted) == n
    monkeypatch.setattr(props, "_prepare_feed", real)
    junk = feed_frame.assign(line=-1.0)                 # every line outside the paste rule: dropped
    assert store.feed_tables(DAY, _feed_live(junk), served, params, world.roster).failed
    assert len(counted) == 3
    good = store.feed_tables(DAY, fl, served, params, world.roster)
    assert not good.failed and len(good.tables) == 9 and len(counted) == 4
    assert store.feed_tables(DAY, fl, served, params, world.roster) is good and len(counted) == 4
    said = _messages(logs)
    assert [m for m in said if m.startswith("oddsapi:")] == \
        ["oddsapi: feed tables failed (_NothingCompared)"] * 3
    assert said.count("prepare_feed failed (ValueError)") == 2 and "xyzzy" not in logs.text


# -- B10: the anchor check through store.feed_start --------------------------------------------
def test_the_anchor_survives_both_streamlit_caches_between_two_kicks_through_feed_start_b10(
        monkeypatch, tmp_path, key_on):
    """B10, design 8.2's anchor check made through store.feed_start: two page
    views of today with st.cache_data.clear() and st.cache_resource.clear()
    between them. The second makes zero extra paid calls, the ledger is the
    same object, and the rate cap's times survive. The serve fixture's day
    at 08:55 ET, the fake vendor, and the anchor's pool synchronous."""
    w = H.install(monkeypatch, tmp_path, MORNING)
    # install() stubs the live layer's feed calls for the page tests (design
    # 8.3); this test is about the real ones.
    monkeypatch.setattr(live, "feed_kick", H.REAL_FEED_KICK)
    monkeypatch.setattr(live, "feed_view", H.REAL_FEED_VIEW)
    vendor = LF.Vendor()
    monkeypatch.setattr(http, "vendor_get", vendor)
    live._anchor_module().pool = LF.SyncPool()
    mem = live.feed_anchor()
    first = store.feed_start(DAY, w.sched)
    assert first == store.FeedTicket(DAY, "kicked")
    assert [c.event_id for c in vendor.odds_calls()] == [LF.PHI_EID, LF.DEN_EID]
    assert mem["calls"] == [MORNING_UTC, MORNING_UTC] and mem["generation"] == 2
    assert [g["game_id"] for g in mem["work"]["games"]] == [EARLY, LATE]
    times = list(mem["calls"])
    st.cache_data.clear()
    st.cache_resource.clear()
    H.pin(monkeypatch, MORNING + dt.timedelta(minutes=1))
    second = store.feed_start(DAY, w.sched)
    assert second.mode == "kicked" and live.feed_anchor() is mem
    assert len(vendor.odds_calls()) == 2 and len(vendor.events_calls()) == 1
    assert mem["calls"] == times
    got = store.feed_result(second)
    assert got.state == "ok" and got.generation == 2 and len(got.value)
    assert set(got.per_game) == {EARLY, LATE}
    assert all(c.keyed for c in vendor.calls)


# -- feed_info ------------------------------------------------------------------------------
def test_feed_info_is_plain_copies_and_carries_the_pages_today(monkeypatch):
    stamp = MORNING_UTC
    fl = live.FeedLive(state="stale", value=pd.DataFrame({"x": [1, 2]}), asof=stamp,
                       per_game={EARLY: {"asof": stamp, "books": {"draftkings": stamp},
                                         "pretip": False, "pp_missing": True, "reason": None}},
                       loading=1, generation=5,
                       report={"pulled": [("draftkings", "Jalen Duren", "Rebounds")], "n_pulled": 1})
    info = store.feed_info(fl, DAY)
    five = ("draftkings", "fanduel", "betmgm", "prizepicks", "underdog")   # the setting's default
    assert info == {"state": "stale", "asof": stamp, "loading": 1, "generation": 5, "lines": 2,
                    "per_game": {EARLY: {"asof": stamp, "books": {"draftkings": stamp},
                                         "pretip": False, "pp_missing": True, "reason": None}},
                    "report": {"pulled": [("draftkings", "Jalen Duren", "Rebounds")], "n_pulled": 1},
                    "today": DAY, "books": five}
    info["per_game"][EARLY]["books"]["fanduel"] = stamp
    info["report"]["pulled"].append(("x", "y", "z"))
    assert fl.per_game[EARLY]["books"] == {"draftkings": stamp} and len(fl.report["pulled"]) == 1
    off = store.feed_info(live.FeedLive.off(), DAY)
    assert off["state"] == "off" and off["lines"] == 0 and off["per_game"] == {}
    # Build amendment B25: the books the site buys, BUCKETVALUE_ODDS_BOOKS read on the call (book
    # keys only, a name outside the five dropped), from the process environment by default.
    narrow = {"BUCKETVALUE_ODDS_BOOKS": "draftkings, FanDuel,pinnacle,prizepicks,underdog"}
    assert store.feed_info(fl, DAY, environ=narrow)["books"] == (
        "draftkings", "fanduel", "prizepicks", "underdog")
    monkeypatch.setenv("BUCKETVALUE_ODDS_BOOKS", "betmgm")
    assert store.feed_info(fl, DAY)["books"] == ("betmgm",)


# -- The pill and chip wording behind a feed flag (design 2.7) --------------------------------
def test_the_pill_says_lines_shown_with_the_feed_and_your_lines_without(world):
    row = world.sched[world.sched["game_id"] == LATE].iloc[0]
    for n, unit in ((1, "line"), (3, "lines")):
        feed = store.render_slate_card(row, DAY, {}, n_lines=n, feed=True)
        assert f'title="{n} {unit} shown for this game"' in feed
        assert f'<b>{n}</b><span class="u">{unit}</span>' in feed
        assert "your" not in feed
        plain = store.render_slate_card(row, DAY, {}, n_lines=n)
        assert plain == store.render_slate_card(row, DAY, {}, n_lines=n, feed=False)
        assert f'title="{n} of your lines shown for this game"' in plain
    slate = store.render_slate(world.sched, DAY, {}, line_counts={EARLY: 2}, feed=True)
    assert 'title="2 lines shown for this game"' in slate
    assert store.render_slate(world.sched, DAY, {}, line_counts={EARLY: 2}) == \
        store.render_slate(world.sched, DAY, {}, line_counts={EARLY: 2}, feed=False)


def test_the_teal_chip_counts_the_boards_lines_with_the_feed(world):
    view = pd.DataFrame([{"person_id": 1630595, "game_id": EARLY, "name": "Cade Cunningham",
                          "position": "G", "status": "", "reason": None, "out": False,
                          "pplay": "", "section": "main", "starter": True, "p_play": 1.0,
                          "n_lines": 2, **{k: 1.0 for k, _ in store.STAT_COLS}}])
    feed = store.render_projection_table("DET", view, feed=True)
    assert 'title="2 lines on the board; open him under this table">2</span>' in feed
    plain = store.render_projection_table("DET", view)
    assert plain == store.render_projection_table("DET", view, feed=False)
    assert 'title="2 PrizePicks lines on your board; open him under this table">2</span>' in plain
    one = store.render_projection_table("DET", view.assign(n_lines=1), feed=True)
    assert 'title="1 line on the board; open him under this table">1</span>' in one


# -- view_params and the picked book (amendment A3) -------------------------------------------
def test_view_params_carry_the_book_only_with_the_feed_on_and_not_prizepicks(monkeypatch):
    base = store.view_params("tok", (5, 3))
    assert base == {"board": "tok", "out": "3,5"} == store.view_params("tok", (5, 3), book=None)
    # The feed off: a ?book= is never carried, whatever the page passes.
    for book in ("draftkings", "prizepicks", "nope"):
        assert store.view_params("tok", (5, 3), book=book) == base
    monkeypatch.setenv(live.KEY_ENV, KEY)
    assert store.view_params("tok", (5, 3), book="draftkings") == {**base, "book": "draftkings"}
    assert list(store.view_params("", (), book="betmgm")) == ["book"]
    for book in ("prizepicks", "nope", "", None, "DraftKings", ["draftkings"]):
        assert store.view_params("tok", (5, 3), book=book) == base, book
    url = store.game_url(DAY, "PHI", "DET", store.view_params("", (), book="underdog"))
    assert url == "/Game?date=2026-03-12&away=PHI&home=DET&book=underdog"


# -- The loading watch (design 2.7, M6 item 3) ------------------------------------------------
def test_the_watch_reruns_once_the_generation_moves_or_nothing_loads_within_its_runs():
    def now(generation, loading):
        return types.SimpleNamespace(generation=generation, loading=loading)
    assert store.watch_reruns(1, 4, now(5, 1)) is True           # a game landed
    assert store.watch_reruns(1, 4, now(4, 0)) is True           # nothing loading any more
    assert store.watch_reruns(1, 4, now(4, 2)) is False          # still loading, nothing new
    assert store.watch_reruns(store.WATCH_MAX_RUNS, 4, now(5, 0)) is True
    assert store.watch_reruns(store.WATCH_MAX_RUNS + 1, 4, now(5, 0)) is False   # a minute: stop
    assert store.watch_reruns(1, 4, None) is False
    assert (store.WATCH_EVERY_S, store.WATCH_MAX_RUNS) == (3, 20)


WATCH_SCRIPT = f"""
import sys
sys.path.insert(0, {str(HERE.parent)!r})
import streamlit as st
from bucketlib import store
store.feed_loading_watch("2026-03-12", 1)
st.markdown("after the watch")
st.text_input("A widget drawn after the watch", key="later")
"""


def test_the_watch_shows_the_loading_line_and_never_reruns_inside_the_pages_own_run(monkeypatch):
    """The page's own run draws the watch inline: it shows the loading line
    and counts its run, and never calls st.rerun() there (that would cut the
    script short and drop every later widget's state), even though the
    generation moved. Its timed runs (which AppTest does not drive) are the
    ones that rerun the app. Memory says 2 games still load, generation 9."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(live, "feed_view",
                        lambda date, **k: types.SimpleNamespace(generation=9, loading=2))
    at = AppTest.from_string(WATCH_SCRIPT, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    md = [m.value for m in at.markdown]
    assert md == ['<div class="bv-note">Lines for 2 games are still loading; this board fills in by '
                  'itself.</div>', "after the watch"]
    assert at.session_state[store.WATCH_RUNS_KEY] == 1
    at.text_input(key="later").input("kept").run()
    assert not at.exception and at.text_input(key="later").value == "kept"
    assert at.session_state[store.WATCH_RUNS_KEY] == 2


def test_the_watch_reruns_the_whole_app_from_its_own_timed_runs_m6(monkeypatch):
    """Design 2.7 and M6 item 3, the watch's real job (build 3's review, F3).
    In its own timed run (store._fragment_run True, as in a fragment's run;
    AppTest drives no timer) it calls st.rerun() with no scope, a full app
    rerun, so the cards, the pills and the board fill in: once the ledger's
    generation moved past the page's (1 here) or nothing is loading any more.
    Never while the same generation still loads, and from its run past
    WATCH_MAX_RUNS (a minute) never again: it only shows the line. The rerun
    is recorded, not made, so each run goes on to the end of the page."""
    from streamlit.testing.v1 import AppTest
    reruns = []
    now = {"view": types.SimpleNamespace(generation=1, loading=2)}
    monkeypatch.setattr(st, "rerun", lambda *a, **k: reruns.append((a, k)))
    monkeypatch.setattr(store, "_fragment_run", lambda: True)
    monkeypatch.setattr(live, "feed_view", lambda date, **k: now["view"])
    at = AppTest.from_string(WATCH_SCRIPT, default_timeout=30)
    line = ('<div class="bv-note">Lines for 2 games are still loading; this board fills in by '
            'itself.</div>')
    for generation, loading, want in ((1, 2, 0), (2, 2, 1), (1, 0, 1), (2, 0, 1), (1, 2, 0)):
        now["view"] = types.SimpleNamespace(generation=generation, loading=loading)
        reruns.clear()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert reruns == [((), {})] * want, (generation, loading)
        md = [m.value for m in at.markdown]
        assert md == ([line] if loading == 2 else []) + ["after the watch"]
    assert at.session_state[store.WATCH_RUNS_KEY] == 5
    now["view"] = types.SimpleNamespace(generation=2, loading=0)
    for runs_before, want in ((store.WATCH_MAX_RUNS - 1, 1), (store.WATCH_MAX_RUNS, 0),
                              (store.WATCH_MAX_RUNS + 5, 0)):
        at.session_state[store.WATCH_RUNS_KEY] = runs_before
        reruns.clear()
        at.run()
        assert not at.exception and reruns == [((), {})] * want, runs_before
        assert at.session_state[store.WATCH_RUNS_KEY] == runs_before + 1


def test_the_watch_knows_its_own_timed_run_from_the_pages_run(monkeypatch):
    """store._fragment_run reads the script run context Streamlit sets for
    each run: fragment_ids_this_run names the fragments a fragment's own run
    runs, and is None or empty in a full script run (where an st.rerun()
    would cut the page short); with no run context at all it is False too.
    The field is Streamlit 1.59.1's (ScriptRunContext)."""
    import dataclasses
    from streamlit.runtime.scriptrunner_utils.script_run_context import ScriptRunContext
    assert "fragment_ids_this_run" in {f.name for f in dataclasses.fields(ScriptRunContext)}
    seen = []
    for ctx, want in ((None, False), (types.SimpleNamespace(fragment_ids_this_run=None), False),
                      (types.SimpleNamespace(fragment_ids_this_run=[]), False),
                      (types.SimpleNamespace(fragment_ids_this_run=["a1b2"]), True)):
        monkeypatch.setattr(store, "get_script_run_ctx",
                            lambda suppress_warning=False, _ctx=ctx: seen.append(suppress_warning) or _ctx)
        assert store._fragment_run() is want, ctx
    assert seen == [True] * 4


# -- Build 4's fix pass: the campaign's survivor ST01, and review L3-4 -------------------------
def test_with_no_key_the_settings_are_never_read_st01(monkeypatch, logs):
    """Survivor ST01 (design 2.7, M9): feed_on asks for the key first, so with
    no key the settings are never read (not even a broken reader can raise
    a warning onto the log), and with a key a settings reader that raises
    is a feed that is off, logged by its type."""
    reads = []

    def broken(env):
        reads.append(1)
        raise RuntimeError("settings " + KEY)
    monkeypatch.setattr(store.O, "read_settings", broken)
    for env in ({}, {live.KEY_ENV: "  "}, {"BUCKETVALUE_ODDS_ENABLED": "0"}):
        assert store.feed_on(env) is False
    assert reads == [] and "feed check failed" not in logs.text
    assert store.feed_on({live.KEY_ENV: KEY}) is False and reads == [1]
    assert "oddsapi: feed check failed (RuntimeError)" in _messages(logs)
    assert KEY not in logs.text


WATCH_EPISODE_SCRIPT = f"""
import sys
sys.path.insert(0, {str(HERE.parent)!r})
import streamlit as st
from bucketlib import store
store.feed_loading_watch("2026-03-12", st.session_state.get("gen", 1))
"""


def test_the_watchs_run_cap_is_per_loading_episode_l3_4(monkeypatch):
    """Build 4's review, L3-4: the watch's WATCH_MAX_RUNS cap (a minute of
    reruns) counts one loading episode, never the whole session. A session
    that spent its runs on the morning's loads (the count past the cap: no
    rerun) gets a fresh count when the page draws the watch for a new
    episode (the ledger's generation has moved since), so the evening's
    lines rerun the page once they land."""
    from streamlit.testing.v1 import AppTest
    reruns = []
    now = {"view": types.SimpleNamespace(generation=1, loading=2, loading_games=())}
    monkeypatch.setattr(st, "rerun", lambda *a, **k: reruns.append((a, k)))
    monkeypatch.setattr(store, "_fragment_run", lambda: True)
    monkeypatch.setattr(live, "feed_view", lambda date, **k: now["view"])
    at = AppTest.from_string(WATCH_EPISODE_SCRIPT, default_timeout=30)
    at.run()
    assert at.session_state[store.WATCH_RUNS_KEY] == 1
    at.session_state[store.WATCH_RUNS_KEY] = store.WATCH_MAX_RUNS + 3          # the morning's
    now["view"] = types.SimpleNamespace(generation=1, loading=0, loading_games=())
    reruns.clear()
    at.run()
    assert reruns == [] and at.session_state[store.WATCH_RUNS_KEY] == store.WATCH_MAX_RUNS + 4
    at.session_state["gen"] = 5                                                # the evening's episode
    now["view"] = types.SimpleNamespace(generation=5, loading=1, loading_games=())
    reruns.clear()
    at.run()
    assert reruns == [] and at.session_state[store.WATCH_RUNS_KEY] == 1
    now["view"] = types.SimpleNamespace(generation=6, loading=0, loading_games=())
    at.run()
    assert reruns == [((), {})] and at.session_state[store.WATCH_RUNS_KEY] == 2
