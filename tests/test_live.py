"""bucketlib.live, the app's only network layer: stale beats empty, a failure
is ANY exception from a third-party payload, an empty payload never replaces
a good copy, the newer of memory and the bot's committed copy is served, a
line seen before tip survives ESPN dropping it, the injury walk never
outlasts its budget, and the box score after tip (spec 4 and 5, rule 9).

Every test goes through the real live functions with only the transport
(http.fetch, the injury store) and the clock replaced. No network.
"""
from __future__ import annotations

import copy
import datetime as dt
import gzip
import json
import types
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import context as CTX
from bucketlib import http as H
from bucketlib import live
from bucketlib import state as S
from bucketlib.sources import box as BX
from bucketlib.sources import cdn, espn
from bucketlib.sources import injury as INJ
from bucketlib.util import ET

FIX = Path(__file__).parent / "fixtures"
SCHEDULE = json.loads((FIX / "cdn" / "schedule_2026-27_trim.json").read_text())
PLAYERS = json.loads((FIX / "cdn" / "playerIndex_trim.json").read_text())
REPORT_PDF = FIX / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf"
BOX_GID = "0022500953"


def _scoreboard() -> dict:
    sb = C.read_json(FIX / "espn" / "live_sb_20261020.json.gz")
    return sb if "events" in sb else sb["data"]


def _box() -> dict:
    with gzip.open(FIX / "serve" / f"box_{BOX_GID}.json.gz", "rt") as f:
        return json.load(f)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A fake clock for live (time.time and time.monotonic), empty memory and
    breaker, the committed state in a temp dir, and http.fetch answering
    whatever env.answer holds (a payload, or an exception to raise)."""
    clock = [1_000_000.0]
    monkeypatch.setattr(live, "time", types.SimpleNamespace(time=lambda: clock[0],
                                                            monotonic=lambda: clock[0]))
    monkeypatch.setattr(live, "_good", {})
    monkeypatch.setattr(live, "_fail", {})
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    e = types.SimpleNamespace(clock=clock, answer=None, calls=[])

    def fetch(url, **kw):
        e.calls.append((url, kw))
        if isinstance(e.answer, BaseException):
            raise e.answer
        return copy.deepcopy(e.answer)
    monkeypatch.setattr(H, "fetch", fetch)
    return e


# ── A failure is any exception, and it never costs the good copy ─────────────
def test_a_null_field_serves_the_last_good_schedule_and_trips_the_breaker(env):
    """gameDateTimeUTC null on one game: cdn.parse_schedule raises
    AttributeError. That is a feed failure: the memory copy is served stale,
    the failure is remembered, and nothing refetches inside the fail TTL."""
    env.answer = SCHEDULE
    good = live.schedule()
    assert good.source == "live" and len(good.value)
    env.clock[0] += 601
    bad = copy.deepcopy(SCHEDULE)
    bad["leagueSchedule"]["gameDates"][0]["games"][0]["gameDateTimeUTC"] = None
    env.answer = bad
    got = live.schedule()
    assert got.stale and got.source == "memory" and got.value is good.value
    assert "AttributeError" in got.error and "schedule" in live._fail
    n = len(env.calls)
    env.clock[0] += 30
    assert live.schedule().stale and len(env.calls) == n          # the breaker answered


def test_an_odd_typed_failure_with_nothing_cached_is_the_outage_runtime_error(env):
    """A list where the payload's dict belongs (AttributeError in the parser):
    the page's outage notice (RuntimeError), never Streamlit's error page."""
    env.answer = [{"leagueSchedule": {}}]
    with pytest.raises(RuntimeError, match="AttributeError"):
        live.schedule()


def test_an_empty_schedule_never_replaces_the_good_one(env):
    env.answer = SCHEDULE
    good = live.schedule().value
    env.clock[0] += 601
    env.answer = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": []}}
    got = live.schedule()
    assert got.stale and got.source == "memory" and got.value is good
    assert "schedule parsed empty" in got.error
    assert "game_date" in got.value.columns


def test_an_empty_schedule_with_no_memory_serves_the_committed_copy(env):
    env.answer = SCHEDULE
    committed = live._schedule_fetch()
    C.write_parquet(committed, S.current_path("schedule"))
    env.answer = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": []}}
    got = live.schedule()
    assert got.source == "committed" and got.stale and len(got.value) == len(committed)


def test_a_short_player_index_is_a_failure(env, monkeypatch):
    """A roster pull with a handful of players is broken, never 'the league
    emptied': cdn.MIN_ROSTER, the one threshold the bot refuses under too,
    read at call time."""
    env.answer = PLAYERS                                 # the trim: a few players
    with pytest.raises(RuntimeError, match="rostered players"):
        live.roster()
    assert not hasattr(live, "MIN_ROSTER")
    monkeypatch.setattr(cdn, "MIN_ROSTER", 1)
    live._fail.clear()
    assert live.roster().source == "live"


# ── Lines ────────────────────────────────────────────────────────────────────
def test_an_espn_error_body_keeps_the_last_good_lines(env):
    d = dt.date(2026, 10, 20)
    env.answer = _scoreboard()
    good = live.odds_on(d)
    assert good.source == "live" and len(good.value) >= 2
    env.clock[0] += 301
    env.answer = {"code": 2500, "message": "service unavailable"}
    got = live.odds_on(d)
    assert got.stale and got.source == "memory" and got.value == good.value
    env.clock[0] += 301
    env.answer = {}
    assert live.odds_on(d).value == good.value


def test_the_lines_come_through_espns_one_scoreboard_request_on_the_apps_budget(env, monkeypatch):
    """live.odds_on asks espn.fetch_scoreboard (its date check, limit and
    events-list check) with the app's short budget and no retries, never a
    request of its own."""
    asked = []
    real = espn.fetch_scoreboard

    def spy(dates, *a, **k):
        asked.append((dates, k))
        return real(dates, *a, **k)
    monkeypatch.setattr(espn, "fetch_scoreboard", spy)
    env.answer = _scoreboard()
    assert ("BOS", "DET") in live.odds_on(dt.date(2026, 10, 20)).value
    assert asked == [("20261020", {"timeout": live.TIMEOUT, "retries": 0})]
    url, kw = env.calls[-1]
    assert url == espn.SITE + "scoreboard" and kw["timeout"] == live.TIMEOUT and kw["retries"] == 0
    assert kw["params"] == {"dates": "20261020", "limit": str(espn.MONTH_LIMIT)}


def test_an_error_body_with_nothing_cached_is_the_outage_not_no_lines(env):
    env.answer = {"code": 2500}
    got = live.odds_on(dt.date(2026, 10, 20))
    assert got.stale and got.source == "none" and got.value == {}


def test_a_line_seen_before_tip_survives_espn_dropping_it_at_tip(env, monkeypatch):
    # odds_on tells a started game by its status OR its tip against the WALL
    # clock (util.now_utc, which env's fake does not move). The fixture is the
    # 2026-10-20 slate (BOS at DET tips 19:00Z), so the wall clock is pinned to
    # that noon ET: unpinned, this test began failing at the opener's real tip.
    wall = {"now": dt.datetime(2026, 10, 20, 16, 0, tzinfo=dt.timezone.utc)}
    monkeypatch.setattr(live, "now_utc", lambda: wall["now"])
    d = dt.date(2026, 10, 20)
    sb = _scoreboard()
    env.answer = sb
    before = live.odds_on(d).value
    assert ("BOS", "DET") in before
    after = copy.deepcopy(sb)
    for e in after["events"]:
        c = e["competitions"][0]
        sides = {x["homeAway"]: x["team"].get("abbreviation") for x in c["competitors"]}
        if sides.get("home") == "DET":
            c["odds"] = []
            e["status"]["type"]["name"] = c["status"]["type"]["name"] = "STATUS_IN_PROGRESS"
    env.clock[0] += 301
    env.answer = after
    got = live.odds_on(d)
    assert got.source == "live" and not got.stale
    kept = got.value[("BOS", "DET")]
    assert kept["pre_tip"] and kept["home_line"] == before[("BOS", "DET")]["home_line"]
    assert kept["total"] == before[("BOS", "DET")]["total"]
    # A game that has NOT started and loses its line is not carried: it is
    # "not posted" again (a pulled line), never a stale number.
    pulled = copy.deepcopy(sb)
    for e in pulled["events"]:
        c = e["competitions"][0]
        sides = {x["homeAway"]: x["team"].get("abbreviation") for x in c["competitors"]}
        if sides.get("home") == "DET":
            c["odds"] = []
    live._good.clear()
    env.answer = sb
    live.odds_on(d)
    env.clock[0] += 301
    env.answer = pulled
    assert ("BOS", "DET") not in live.odds_on(d).value
    # Past its tip a game has started even while ESPN still says SCHEDULED (a
    # late status flip): the line it showed before tip is kept, not dropped.
    live._good.clear()
    env.answer = sb
    live.odds_on(d)
    wall["now"] = dt.datetime(2026, 10, 20, 19, 5, tzinfo=dt.timezone.utc)
    env.clock[0] += 301
    env.answer = pulled
    late = live.odds_on(d).value[("BOS", "DET")]
    assert late["pre_tip"] and late["home_line"] == before[("BOS", "DET")]["home_line"]


# ── Memory or committed: the newer one ───────────────────────────────────────
def _report_at(rep, title):
    return INJ.ParsedReport({**rep.meta, "report_ts_et": title},
                            rep.rows.assign(report_ts_et=title),
                            rep.team_rows.assign(report_ts_et=title))


def _serve_sched():
    return pd.read_parquet(FIX / "serve" / "schedule.parquet")


def test_a_newer_committed_report_beats_an_older_one_in_memory(env, monkeypatch):
    """The process remembers the noon report; the bot has since committed
    the 10:30 PM one; the live walk fails. The committed (newer) report is
    served, stale, and memory is left alone for the next good fetch."""
    rep = INJ.parse_report(REPORT_PDF.read_bytes())
    sched = _serve_sched()
    late = _report_at(rep, dt.datetime(2026, 3, 12, 22, 30, tzinfo=ET))
    CTX.write_committed(late, sched)

    def down(*a, **k):
        raise H.Blocked("injury", "test: ak-static down")
    monkeypatch.setattr(INJ, "latest_report", down)
    live._good[live.INJURY_KEY] = (env.clock[0] - live.INJURY_TTL - 1, rep)
    got = live.injury_report(sched)
    assert got.stale and got.source == "committed"
    assert pd.Timestamp(got.value.meta["report_ts_et"]) == pd.Timestamp(late.meta["report_ts_et"])
    assert live._good[live.INJURY_KEY][1] is rep
    # Memory newer than the committed copy: memory wins.
    newest = _report_at(rep, dt.datetime(2026, 3, 12, 23, 0, tzinfo=ET))
    live._good[live.INJURY_KEY] = (env.clock[0] - live.INJURY_TTL - 1, newest)
    live._fail.clear()
    got = live.injury_report(sched)
    assert got.source == "memory" and got.value is newest


def test_a_newer_committed_schedule_beats_an_older_memory_copy(env):
    env.answer = SCHEDULE
    mem = live.schedule().value
    env.clock[0] += 3600
    C.write_parquet(mem.iloc[:5], S.current_path("schedule"),
                    meta={"fetched_at": env.clock[0] - 60})            # committed a minute ago
    env.answer = H.Blocked(live.cdn.SCHEDULE_URL, "test: down")
    got = live.schedule()
    assert got.source == "committed" and len(got.value) == 5
    # An older committed copy loses to memory.
    C.write_parquet(mem.iloc[:5], S.current_path("schedule"),
                    meta={"fetched_at": env.clock[0] - 7200})
    live._fail.clear()
    assert live.schedule().source == "memory"


# ── The injury walk's budget ─────────────────────────────────────────────────
def test_no_injury_fetch_can_outlast_the_walks_budget(env, monkeypatch):
    """Each slot answers slowly (absent after 0.7 of the budget). The first
    fetch gets the full per-PDF timeout; the second starts with less than
    one timeout left, so it gets exactly what is left: no fetch can end past
    the budget (it could by a whole timeout before)."""
    fetches = []                                    # (start, the timeout it ran under)
    lag = 0.7 * live.INJURY_BUDGET

    def live_store(timeout=8.0):
        def fetcher(url):
            fetches.append((env.clock[0], timeout))
            env.clock[0] += lag
            raise H.Absent(url, "test: not published")
        return INJ.ReportStore(fetcher=fetcher, persist=False)
    monkeypatch.setattr(INJ, "live_store", live_store)
    monkeypatch.setattr(live, "now_et", lambda: dt.datetime(2026, 3, 12, 12, 5, tzinfo=ET))
    t0 = env.clock[0]
    with pytest.raises(RuntimeError, match="budget"):
        live.injury_report(_serve_sched())
    assert len(fetches) == 2 and fetches[0][1] == live.INJURY_TIMEOUT
    for start, timeout in fetches:
        assert start + timeout <= t0 + live.INJURY_BUDGET + 1e-9, (start - t0, timeout)


def test_every_fetch_fits_the_spec_budget():
    """Spec 4: ~5 s, no retries. The schedule, roster, lines and box score
    share TIMEOUT."""
    assert live.TIMEOUT <= 5.0 and live.INJURY_TIMEOUT <= live.INJURY_BUDGET <= 5.0


# ── Game state ───────────────────────────────────────────────────────────────
def _row(**kw):
    base = {"status": "scheduled", "status_text": "7:00 pm ET",
            "tip_utc": pd.Timestamp("2026-11-03T00:00:00Z"), "postponed": False}
    return pd.Series({**base, **kw})


def test_a_postponed_game_is_never_called_live():
    late = dt.datetime(2026, 11, 2, 23, 0, tzinfo=ET)        # 4 hours after its tip
    assert live.game_state(_row(), now=late) == "live"
    assert live.game_state(_row(postponed=True, status_text="PPD"), now=late) == "postponed"
    assert live.game_state(_row(postponed=True, status="final"), now=late) == "final"
    assert live.game_state(_row(postponed=float("nan")), now=late) == "live"   # a missing flag is not a postponement


def test_a_game_whose_time_is_not_set_is_never_called_live_by_the_clock():
    tbd = _row(status_text="TBD", tip_utc=pd.Timestamp("2026-12-08T05:00:00Z"), time_tbd=True)
    noon = dt.datetime(2026, 12, 8, 12, 0, tzinfo=ET)
    assert live.game_state(tbd, now=noon) == "upcoming"
    assert live.time_tbd(tbd) and not live.time_tbd(_row(time_tbd=False))
    # A frame parsed before the column existed: the feed's own 'TBD' text.
    assert live.time_tbd(_row(status_text="TBD"))
    assert live.game_state(_row(status="live", status_text="TBD", time_tbd=True), now=noon) == "live"


# ── The box score after tip ──────────────────────────────────────────────────
def test_the_box_score_is_read_through_the_cdn_with_its_own_ttl_and_breaker(env, monkeypatch):
    env.answer = _box()
    b = live.box(BOX_GID)
    assert b.source == "live" and b.value.final and len(b.value.lines) == 23
    url, kw = env.calls[-1]
    assert "cdn.nba.com" in url and BOX_GID in url and "stats.nba.com" not in url
    assert kw["retries"] == 0 and kw["timeout"] == live.TIMEOUT
    cade = b.value.lines.set_index("player_name").loc["Cade Cunningham"]
    assert (cade["pts"], cade["ast"]) == (8, 13)
    n = len(env.calls)
    env.clock[0] += live.BOX_TTL - 1
    assert live.box(BOX_GID).source == "memory" and len(env.calls) == n
    # In progress: not final (the box module's own final rule, box.is_final).
    live._good.clear()
    live_payload = _box()
    live_payload["game"]["gameStatus"] = 2
    env.answer = live_payload
    assert live.box(BOX_GID).value.final is False
    asked, real_final = [], BX.is_final
    monkeypatch.setattr(BX, "is_final", lambda g, gid, where="cdn": asked.append((gid, where)) or True)
    live._good.clear()
    assert live.box(BOX_GID).value.final is True and asked == [(BOX_GID, "cdn")]
    monkeypatch.setattr(BX, "is_final", real_final)
    # Another game's payload is a failure, remembered for BOX_FAIL_TTL.
    live._good.clear()
    env.clock[0] += live.BOX_TTL + 1
    env.answer = _box()
    with pytest.raises(RuntimeError):
        live.box("0022500958")
    n = len(env.calls)
    with pytest.raises(RuntimeError):
        live.box("0022500958")
    assert len(env.calls) == n
