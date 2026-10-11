"""The bot (scripts/daily_update.py): the projections step, the freeze, the
finals step's one definition of ingested, and that nothing is written
outside state/.

Offline. The projections tests run the bot's own step_projections into a tmp
STATE_DIR on the real 2026-03-12 fixture (tests/fixtures/pipeline: the history
features the bot would commit, the real 12:00 PM report, production models),
with only the feature BUILD stubbed (serve parity proves it bit for bit in
tests/test_serve_parity.py) and a fake clock. The end-to-end test runs main()
with the network stubbed at bucketlib.http.fetch (fixture payloads; any other
host fails the test) and the committed seed, so every step's real code runs,
features.build included.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import http as H
from bucketlib import models as M
from bucketlib import pipeline as P
from bucketlib import serve as V
from bucketlib import state as S
from bucketlib.sources import box, cdn, cdnbox, espn
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
DATE = dt.date(2026, 3, 12)
MORNING = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)
EVENING = dt.datetime(2026, 3, 12, 19, 0, tzinfo=ET)     # the three 7:00 PM ET games froze at 6:55
EARLY = {"0022500953", "0022500954", "0022500955"}


def _daily_update():
    spec = importlib.util.spec_from_file_location("daily_update", REPO / "scripts" / "daily_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mdl() -> M.Models:
    try:
        return M.load()
    except M.ModelMismatch as e:
        pytest.skip(f"production models not loadable here: {e}")


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    return pd.read_parquet(FIX / "pipeline" / "features.parquet")


@pytest.fixture(scope="module")
def sched() -> pd.DataFrame:
    return pd.read_parquet(FIX / "pipeline" / "schedule.parquet")


@pytest.fixture(scope="module")
def report() -> INJ.ParsedReport:
    return INJ.parse_report((FIX / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())


def odds(sched: pd.DataFrame, shift: float = 0.0, drop=()) -> pd.DataFrame:
    """Made-up snapshots in the scoreboard's shape (historical lines are never committed)."""
    g = sched[~sched["game_id"].isin(set(drop))]
    n = len(g)
    return pd.DataFrame({"away": g["away"].to_numpy(), "home": g["home"].to_numpy(),
                         "home_line": np.linspace(-9.5, 7.5, n) + shift,
                         "total": np.linspace(214.5, 236.5, n) + shift,
                         "captured_at": pd.Timestamp("2026-03-12 15:00", tz="UTC")})


@pytest.fixture
def bot(tmp_path, monkeypatch, feats, mdl):
    """The bot module with its state in a tmp dir, a settable clock, and the
    feature build answered from the fixture (the rows serve.history_features
    returns for the date's games, and the history's names). `du.edit` maps
    the fixture rows to what the next build returns (a changed history)."""
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "state" / "seed")
    clock = [MORNING.astimezone(UTC)]
    monkeypatch.setattr(du, "now_utc", lambda: clock[0])
    builds, edit = [], [lambda f: f]

    def history_features(d, games, roster, sched, seeds):
        builds.append(d)
        rows = edit[0](feats[feats["game_id"].isin(set(games["game_id"]))].reset_index(drop=True))
        return rows, feats[["person_id", "player_name"]].drop_duplicates()
    monkeypatch.setattr(du, "history_features", history_features)
    monkeypatch.setattr(M, "load", lambda dir=M.MODELS_DIR: mdl)
    du.RED.clear()
    du.clock, du.builds, du.edit = clock, builds, edit
    return du


ROSTER = pd.DataFrame({"person_id": [1], "team": ["DET"], "name": ["x"], "two_way": [False]})


def run(du, sched, report, snaps, at, today=DATE, held=None):
    du.clock[0] = at.astimezone(UTC)
    du.step_projections(sched, ROSTER, report, {DATE: snaps}, today, False, held)
    return C.read_parquet(P.projections_path(DATE))


# ── The freeze ───────────────────────────────────────────────────────────────
def test_a_game_past_tip_minus_five_is_carried_byte_for_byte_and_the_rest_recomputed(
        bot, sched, report):
    first = run(bot, sched, report, odds(sched), MORNING)
    assert bot.RED == [] and set(first["game_id"]) == set(sched["game_id"])
    assert not first["provisional"].any() and bot.builds == [DATE]
    assert C.read_parquet(P.features_path(DATE)) is not None
    second = run(bot, sched, report, odds(sched, shift=1.0), EVENING)
    assert bot.RED == []
    early = second["game_id"].isin(EARLY)
    pd.testing.assert_frame_equal(second[early].reset_index(drop=True),
                                  first[first["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    late = second[~early].set_index(["game_id", "person_id"])
    was = first.set_index(["game_id", "person_id"]).loc[late.index]
    assert (late["computed_at"] == pd.Timestamp(EVENING).tz_convert(UTC)).all()
    assert (late["home_line"] - was["home_line"] == 1.0).all()
    assert not np.array_equal(late["mu_min"].to_numpy(), was["mu_min"].to_numpy())
    # Once every game is past its freeze point nothing is built or written.
    before = P.projections_path(DATE).read_bytes()
    run(bot, sched, report, odds(sched, shift=2.0), dt.datetime(2026, 3, 13, 1, 0, tzinfo=ET))
    assert P.projections_path(DATE).read_bytes() == before and bot.builds == [DATE, DATE]


def test_an_unchanged_slate_is_not_a_new_file_at_the_cron_cadence(bot, sched, report):
    """The crons run every 30 minutes and two model features move with the
    clock alone: with nothing else changed the files stay byte-identical (no
    commit), at +30 minutes as at +1."""
    run(bot, sched, report, odds(sched), MORNING)
    before = {p: p.read_bytes() for p in (P.projections_path(DATE), P.features_path(DATE))}
    for later in (1, 30, 120):
        run(bot, sched, report, odds(sched), MORNING + dt.timedelta(minutes=later))
        assert bot.RED == []
        assert {p: p.read_bytes() for p in before} == before, f"+{later} min rewrote a file"


def test_an_input_or_a_crossed_horizon_rewrites_that_game_only(bot, sched, report):
    first = run(bot, sched, report, odds(sched), MORNING)
    key = ["game_id", "person_id"]

    def rewritten(out, at):
        return set(out.loc[out["computed_at"] == pd.Timestamp(at).tz_convert(UTC), "game_id"])

    # 5:05 PM: t_120 of the 7:00 PM games (5:00 PM) has passed since 12:20.
    five = dt.datetime(2026, 3, 12, 17, 5, tzinfo=ET)
    out = run(bot, sched, report, odds(sched), five)
    assert rewritten(out, five) == EARLY
    rest = ~out["game_id"].isin(EARLY)
    pd.testing.assert_frame_equal(out[rest].reset_index(drop=True),
                                  first[~first["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    # A new history for one game (a final ingested, a roster move): that game.
    gid = "0022500958"
    bot.edit[0] = lambda f: f.assign(career_games=f["career_games"].where(f["game_id"] != gid,
                                                                          f["career_games"] + 1))
    at = five + dt.timedelta(minutes=10)
    out2 = run(bot, sched, report, odds(sched), at)
    assert rewritten(out2, at) == {gid}
    moved = out2.set_index(key).loc[out2["game_id"].eq(gid).to_numpy()]
    assert (moved["games_prior"] == first.set_index(key).loc[moved.index, "games_prior"] + 1).all()
    # A moved line: that game.
    snaps = odds(sched)
    snaps.loc[snaps["home"] == "LAL", "home_line"] += 0.5
    at = at + dt.timedelta(minutes=10)
    out3 = run(bot, sched, report, snaps, at)
    assert rewritten(out3, at) == {"0022500960"}
    assert bot.RED == []


def test_a_game_first_seen_after_its_freeze_is_never_written(bot, sched, report):
    gid = "0022500953"                         # 7:00 PM ET tip: frozen from 6:55
    # The morning run had no line for it: withheld (never RED), not written.
    first = run(bot, sched, report, odds(sched, drop=[gid]), MORNING)
    assert bot.RED == [] and gid not in set(first["game_id"])
    # Its line arrives after its freeze point: still never written, while the
    # untipped games are recomputed and the other frozen ones carried.
    second = run(bot, sched, report, odds(sched, shift=1.0), EVENING)
    assert bot.RED == [] and gid not in set(second["game_id"])
    assert set(second["game_id"]) == set(first["game_id"])
    carried = second[second["game_id"].isin(EARLY)].reset_index(drop=True)
    pd.testing.assert_frame_equal(carried, first[first["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    # A first run at all after the early games froze writes only the rest.
    P.projections_path(DATE).unlink()
    third = run(bot, sched, report, odds(sched), EVENING)
    assert set(third["game_id"]) == set(sched["game_id"]) - EARLY


# ── Provisional ──────────────────────────────────────────────────────────────
def test_tomorrow_is_provisional(bot, sched, report):
    out = run(bot, sched, report, odds(sched), MORNING, today=DATE - dt.timedelta(days=1))
    assert bot.RED == [] and len(out) and out["provisional"].all()


def test_a_game_whose_teams_miss_a_final_is_provisional_and_only_that_game(bot, sched, report):
    """Completeness is per game (spec 4: "every Final game of both teams
    before D"): DET-ATL missing from the state holds back DET's and ATL's
    games only; the other seven stay freezable."""
    missed = pd.DataFrame([{"game_id": "0022500940", "season": "2025-26", "game_type": "RS",
                            "game_date": DATE - dt.timedelta(days=1),
                            "tip_utc": pd.Timestamp("2026-03-12 00:00", tz="UTC"), "status": "final",
                            "home": "DET", "away": "ATL"}])
    full = pd.concat([missed, sched], ignore_index=True)
    out = run(bot, full, report, odds(sched), MORNING)
    stale = {"0022500953", "0022500956"}                     # DET-PHI, ATL-BKN
    assert bot.RED == [] and set(out["game_id"]) == set(sched["game_id"])
    assert out.groupby("game_id")["provisional"].all().to_dict() == {
        g: g in stale for g in sched["game_id"]}
    prov, gaps = V.provisional_by_game(DATE, DATE, full, "2025-26")
    assert {g for g, p in prov.items() if p} == stale and gaps["0022500953"] == ["0022500940"]


def test_a_previous_game_still_live_makes_its_teams_games_provisional(bot, sched, report):
    """A first leg still live (or not yet flipped to Final) when the run lands
    is missing from the state just the same: the second leg is provisional,
    never the frozen projection, and the run is not RED for it."""
    live = pd.DataFrame([{"game_id": "0022500941", "season": "2025-26", "game_type": "RS",
                          "game_date": DATE - dt.timedelta(days=1),
                          "tip_utc": pd.Timestamp("2026-03-12 02:30", tz="UTC"), "status": "live",
                          "home": "LAL", "away": "CHI"}])
    out = run(bot, pd.concat([live, sched], ignore_index=True), report, odds(sched), MORNING)
    assert bot.RED == []
    assert out.groupby("game_id")["provisional"].all().to_dict() == {
        g: g == "0022500960" for g in sched["game_id"]}


# ── What is RED ──────────────────────────────────────────────────────────────
def test_no_report_today_is_red_and_writes_nothing(bot, sched):
    assert run(bot, sched, None, odds(sched), MORNING) is None
    assert len(bot.RED) == 1 and "no injury report" in bot.RED[0] and "report_age_h" in bot.RED[0]


def test_no_report_for_tomorrow_is_a_warning(bot, sched, capsys):
    assert run(bot, sched, None, odds(sched), MORNING, today=DATE - dt.timedelta(days=1)) is None
    assert bot.RED == [] and "WARN projections 2026-03-12" in capsys.readouterr().out


def test_an_exception_inside_a_date_is_red(bot, sched, report, monkeypatch):
    def broken(*a, **k):
        raise KeyError("a contract break")
    monkeypatch.setattr(bot.P, "lines_from_odds", broken)
    assert run(bot, sched, report, odds(sched), MORNING) is None
    assert len(bot.RED) == 1 and "KeyError" in bot.RED[0]


def test_an_untipped_game_left_unprojected_is_red_today(bot, sched, report):
    """Only the missing-line policy may leave an untipped game unprojected:
    a game with no feature rows (its teams missing from the roster pull) or a
    side under MIN_TEAM_ROWS rows is RED today, named, and the rest of the
    slate is still published."""
    lost, thin = "0022500960", "0022500959"

    def cut(f):
        f = f[f["game_id"] != lost]
        side = (f["game_id"] == thin) & (f["team"] == "OKC")
        return f[~side | (side.cumsum() <= 3)]
    bot.edit[0] = cut
    out = run(bot, sched, report, odds(sched), MORNING)
    assert len(bot.RED) == 1 and "2 untipped game(s) without a whole roster" in bot.RED[0]
    assert f"{lost} CHI 0 / LAL 0" in bot.RED[0] and f"{thin} BOS" in bot.RED[0] \
        and "OKC 3" in bot.RED[0]
    assert set(out["game_id"]) == set(sched["game_id"]) - {lost}


def test_an_unprojected_game_tomorrow_is_a_warning(bot, sched, report, capsys):
    bot.edit[0] = lambda f: f[f["game_id"] != "0022500960"]
    run(bot, sched, report, odds(sched), MORNING, today=DATE - dt.timedelta(days=1))
    assert bot.RED == [] and "WARN projections 2026-03-12: 1 untipped game(s)" in capsys.readouterr().out


def test_a_failing_date_does_not_stop_the_other(bot, sched, monkeypatch):
    nxt = DATE + dt.timedelta(days=1)
    two = pd.concat([sched, sched.assign(game_id=sched["game_id"].str.replace("00225", "00226"),
                                         game_date=nxt, tip_utc=sched["tip_utc"] + pd.Timedelta(days=1))],
                    ignore_index=True)
    seen = []

    def project_date(d, *a, **k):
        seen.append(d)
        if d == DATE:
            raise KeyError("a contract break")
    monkeypatch.setattr(bot, "project_date", project_date)
    bot.step_projections(two, ROSTER, None, {}, DATE, False)
    assert seen == [DATE, nxt]
    assert len(bot.RED) == 1 and "KeyError" in bot.RED[0] and str(DATE) in bot.RED[0]


# ── The report's title is a stamp; a report out of reach; play-in ────────────
def retitled(report: INJ.ParsedReport, ts: dt.datetime) -> INJ.ParsedReport:
    """The same report under another title time (what a re-issue is)."""
    return INJ.ParsedReport({**report.meta, "report_ts_et": ts}, report.rows.assign(report_ts_et=ts),
                            report.team_rows.assign(report_ts_et=ts))


@pytest.fixture(scope="module")
def reissue() -> INJ.ParsedReport:
    """The real 12:15 PM report of 2026-03-12: row for row the 12:00 PM report
    under a new title (the NBA re-titles an unchanged report every 15
    minutes)."""
    return INJ.parse_report((FIX / "injury" / "Injury-Report_2026-03-12_12_15PM.pdf").read_bytes())


def _step_injuries(du, sched, rep):
    """The bot's step 4 with the walk answered by `rep` (no network)."""
    import types
    from unittest import mock
    with mock.patch.object(INJ, "latest_report", lambda *a, **k: types.SimpleNamespace(report=rep)):
        return du.step_injuries(sched, False)


def test_a_same_content_reissue_rewrites_no_committed_file(bot, sched, report, reissue):
    """12:05: the 12:00 PM report is committed (step 4) and projected. 12:20:
    the 12:15 PM re-issue, the same rows under a new title, is not a new
    committed report (context.write_committed returns False) and the run
    leaves projections_v1 and both committed injury files byte-identical: a
    title is a stamp, so a re-issue every 15 minutes is no commit every run."""
    from bucketlib import context as CTX
    assert reissue.meta["report_ts_et"] > report.meta["report_ts_et"]
    rep = _step_injuries(bot, sched, report)
    run(bot, sched, rep, odds(sched), dt.datetime(2026, 3, 12, 12, 5, tzinfo=ET))
    assert bot.RED == []
    files = [P.projections_path(DATE), *CTX.committed_paths()]
    before = {p: p.read_bytes() for p in files}
    assert not CTX.write_committed(reissue, sched)
    rep = _step_injuries(bot, sched, reissue)
    assert rep.meta["report_ts_et"] == reissue.meta["report_ts_et"]
    run(bot, sched, rep, odds(sched), dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET))
    assert bot.RED == []
    assert {p: p.read_bytes() for p in files} == before


def test_a_report_out_of_reach_for_every_game_is_red_today_and_writes_nothing(bot, sched, report,
                                                                               capsys):
    """A report titled 31 hours before the run (the committed one after a day
    of ak-static outage) is out of reach for every game: no injury report.
    RED today, naming why, and no projections written; a warning when the
    date is tomorrow."""
    stale = retitled(report, MORNING - dt.timedelta(hours=31))
    assert run(bot, sched, stale, odds(sched), MORNING) is None
    assert len(bot.RED) == 1 and "out of reach for every game" in bot.RED[0]
    assert "nothing new written" in bot.RED[0] and not P.projections_path(DATE).exists()
    bot.RED.clear()
    capsys.readouterr()
    assert run(bot, sched, stale, odds(sched), MORNING, today=DATE - dt.timedelta(days=1)) is None
    out = capsys.readouterr().out
    assert bot.RED == [] and "WARN projections 2026-03-12: nothing new written: " in out
    assert "out of reach for every game" in out


def test_a_play_in_game_gets_no_projection_rows_and_no_red(bot, sched, report, capsys):
    """Play-in is ingested as state but never projected (the models trained on
    regular-season rows only): a play-in game on the date has no rows, is
    not a game "without a whole roster", and the run stays green."""
    pi = pd.DataFrame([{"game_id": "0052500101", "season": "2025-26", "game_type": "PI",
                        "game_date": DATE, "tip_utc": pd.Timestamp("2026-03-12 23:30", tz="UTC"),
                        "status": "scheduled", "home": "CHA", "away": "NOP"}])
    out = run(bot, pd.concat([sched, pi], ignore_index=True), report, odds(sched), MORNING)
    assert bot.RED == [] and set(out["game_id"]) == set(sched["game_id"])
    assert "0052500101" not in capsys.readouterr().out
    # A date with only the play-in game: nothing to project, said in words.
    bot.step_projections(pi.assign(game_date=DATE + dt.timedelta(days=5)), ROSTER, report, {},
                         DATE + dt.timedelta(days=5), False)
    assert bot.RED == [] and "no regular-season games today or tomorrow (preseason and play-in " \
        "are never projected)" in capsys.readouterr().out


# ── A game frozen on an old report ───────────────────────────────────────────
def _froze_on_old(out: str) -> dict[str, str]:
    return {line.split(": ", 1)[1].split(" ")[0]: line for line in out.splitlines()
            if line.startswith("WARN projections 2026-03-12: ") and "froze on rows" in line}


def test_a_game_frozen_on_rows_older_than_a_committed_report_is_a_warning(bot, sched, report,
                                                                          capsys):
    """The three 7:00 PM games froze at 6:55 PM on rows the bot computed at
    12:20 with the 12:00 PM report. Whether a committed report titled before
    6:55 PM was newer decides a WARN naming the game and both report times
    (never RED: the carried rows are the record either way)."""
    at = dt.datetime(2026, 3, 12, 19, 20, tzinfo=ET)
    six45 = retitled(report, dt.datetime(2026, 3, 12, 18, 45, tzinfo=ET))
    seven15 = retitled(report, dt.datetime(2026, 3, 12, 19, 15, tzinfo=ET))
    run(bot, sched, report, odds(sched), MORNING)
    capsys.readouterr()
    # 6:58 PM: the run's own report, titled 6:45 PM, is newer than the rows.
    run(bot, sched, six45, odds(sched), dt.datetime(2026, 3, 12, 18, 58, tzinfo=ET))
    warns = _froze_on_old(capsys.readouterr().out)
    assert bot.RED == [] and set(warns) == EARLY
    for w in warns.values():
        assert "computed with the 2026-03-12 12:00 ET report, but the 2026-03-12 18:45 ET report " \
               "was committed before its freeze point (2026-03-12 18:55 ET)" in w
    assert "(PHI at DET)" in warns["0022500953"]
    # 7:20 PM with the 7:15 PM report: the bot holds no report titled before
    # 6:55 PM newer than 12:00 PM, so nothing to say.
    run(bot, sched, seven15, odds(sched), at)
    assert bot.RED == [] and _froze_on_old(capsys.readouterr().out) == {}
    # The pair the run found (committed by a run whose projections never
    # reached those games) was titled 6:45 PM: a warning per frozen game.
    run(bot, sched, seven15, odds(sched), at, held=six45.meta["report_ts_et"])
    assert set(_froze_on_old(capsys.readouterr().out)) == EARLY and bot.RED == []
    # Rows computed with the newest pre-freeze report: no warning.
    P.projections_path(DATE).unlink()
    run(bot, sched, six45, odds(sched), dt.datetime(2026, 3, 12, 18, 50, tzinfo=ET))
    run(bot, sched, seven15, odds(sched), at, held=six45.meta["report_ts_et"])
    assert bot.RED == [] and _froze_on_old(capsys.readouterr().out) == {}


def test_rows_carried_past_the_freeze_as_provisional_never_warn_of_an_old_report(bot, sched, report,
                                                                                 capsys):
    """A provisional row is never the frozen projection, so a game that
    reached its freeze point on provisional rows (here published the day
    before, as tomorrow's) did not freeze on an old report: no WARN, though
    the 6:45 PM report was committed before 6:55 PM and the rows carry the
    12:00 PM one."""
    six45 = retitled(report, dt.datetime(2026, 3, 12, 18, 45, tzinfo=ET))
    seven15 = retitled(report, dt.datetime(2026, 3, 12, 19, 15, tzinfo=ET))
    first = run(bot, sched, report, odds(sched), MORNING, today=DATE - dt.timedelta(days=1))
    assert first["provisional"].all()
    capsys.readouterr()
    out = run(bot, sched, six45, odds(sched), dt.datetime(2026, 3, 12, 18, 58, tzinfo=ET))
    assert out.loc[out["game_id"].isin(EARLY), "provisional"].all()
    assert bot.RED == [] and _froze_on_old(capsys.readouterr().out) == {}
    run(bot, sched, seven15, odds(sched), dt.datetime(2026, 3, 12, 19, 20, tzinfo=ET),
        held=six45.meta["report_ts_et"])
    assert bot.RED == [] and _froze_on_old(capsys.readouterr().out) == {}


def test_main_hands_step_6_the_committed_report_as_the_run_found_it(tmp_path, monkeypatch, sched,
                                                                    report):
    """The pair step 4 replaces is still known to step 6: main() reads the
    committed title BEFORE step 4 writes the new report."""
    from bucketlib import context as CTX
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    _pin(du, monkeypatch, dt.datetime(2026, 3, 12, 19, 20, tzinfo=ET))
    assert du.committed_title() is None
    assert CTX.write_committed(report, sched)                      # the 12:00 PM pair
    late = retitled(report, dt.datetime(2026, 3, 12, 19, 15, tzinfo=ET))

    def injuries(sched_, dry):
        assert CTX.write_committed(late, sched_)                   # 7 hours on: re-stamped
        return late
    seen = []
    monkeypatch.setattr(du, "step_schedule", lambda dry: sched)
    monkeypatch.setattr(du, "step_roster", lambda dry: ROSTER)
    monkeypatch.setattr(du, "step_finals", lambda *a: None)
    monkeypatch.setattr(du, "step_injuries", injuries)
    monkeypatch.setattr(du, "step_odds", lambda *a: {})
    monkeypatch.setattr(du, "step_projections", lambda *a: seen.append(a))
    assert du.main(["--date", "2026-03-12"]) == 0, du.RED
    (args,) = seen
    assert args[2] is late and args[-1] == pd.Timestamp(report.meta["report_ts_et"])
    assert du.committed_title() == pd.Timestamp(late.meta["report_ts_et"])


# ── Finals: state.ingested_ids is the one definition ─────────────────────────
def _box(gid: str) -> dict:
    return C.read_json(FIX / "cdnbox" / f"cdn_{gid}.json.gz")["data"]


def _sched_row(payload: dict, status: str = "final") -> dict:
    """The cdn.parse_schedule row of a box score's game."""
    g = payload["game"]
    tip = pd.Timestamp(g["gameTimeUTC"])
    return {"game_id": g["gameId"], "season": "2019-20", "game_type": "RS", "status": status,
            "game_date": tip.tz_convert("America/New_York").date(), "tip_utc": tip,
            "home": g["homeTeam"]["teamTricode"], "away": g["awayTeam"]["teamTricode"]}


def _stuck(payload: dict) -> dict:
    """The same box frozen mid-game (17 cdn boxes of 2019-20 are)."""
    p = json.loads(json.dumps(payload))
    p["game"]["gameStatus"], p["game"]["gameStatusText"] = 2, "Q4 0:00"
    return p


def _no_team_stats(payload: dict) -> dict:
    p = json.loads(json.dumps(payload))
    p["game"]["homeTeam"]["statistics"] = None
    return p


@pytest.fixture
def finals_bot(tmp_path, monkeypatch):
    """The bot with its state in a tmp dir and box scores from `du.boxes`
    ({game_id: payload, or an exception to raise})."""
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    du.boxes = {}

    def cdn_box(gid):
        b = du.boxes[gid]
        if isinstance(b, Exception):
            raise b
        return b
    monkeypatch.setattr(du, "cdn_box", cdn_box)
    du.RED.clear()
    return du


GOOD, BAD = "0021900001", "0021900464"


@pytest.mark.parametrize("bad", ["stuck", "bad_payload", "no_team_stats", "bad_request"])
@pytest.mark.parametrize("order", ["bad_first", "good_first"])
def test_one_bad_box_score_never_stops_the_others(finals_bot, bad, order):
    """A box frozen mid-game (box.NotFinal), a bad payload, a payload the
    parser chokes on (TypeError): that game is reported, and every other
    final of the run is still written."""
    du = finals_bot
    good, other = _box(GOOD), _box(BAD)
    du.boxes = {GOOD: good, BAD: {"stuck": _stuck(other),
                                  "bad_payload": H.BadPayload(cdn.BOX_URL, "test: HTML 200"),
                                  "no_team_stats": _no_team_stats(other),
                                  "bad_request": H.BadRequest(cdn.BOX_URL, "test: 400")}[bad]}
    rows = [_sched_row(other), _sched_row(good)]
    sched = pd.DataFrame(rows if order == "bad_first" else rows[::-1])
    du.step_finals(sched, dt.date(2019, 12, 31), dry=False)
    assert S.ingested_ids("2019-20") == {GOOD}
    assert du.RED and all(BAD in r for r in du.RED) and not any(GOOD in r for r in du.RED)
    if bad == "stuck":
        assert any("--override " + BAD in r for r in du.RED)


def test_a_box_not_final_yet_is_a_warning_while_the_tip_is_recent(finals_bot, monkeypatch, capsys):
    """The schedule can say Final before the box does: a warning (retried next
    run) until STUCK_AFTER past the tip, RED after that."""
    du = finals_bot
    other = _box(BAD)
    du.boxes = {BAD: _stuck(other)}
    sched = pd.DataFrame([_sched_row(other)])
    tip = pd.Timestamp(other["game"]["gameTimeUTC"]).to_pydatetime()
    today = sched["game_date"].iloc[0]
    monkeypatch.setattr(du, "now_utc", lambda: tip + dt.timedelta(hours=3))
    du.step_finals(sched, today, dry=False)
    assert du.RED == [] and f"WARN finals: {BAD}" in capsys.readouterr().out
    monkeypatch.setattr(du, "now_utc", lambda: tip + du.STUCK_AFTER + dt.timedelta(minutes=1))
    du.step_finals(sched, today, dry=False)
    assert len(du.RED) == 1 and BAD in du.RED[0] and "still not final" in du.RED[0]


def test_a_game_before_today_not_final_yet_is_a_warning_then_red(finals_bot, monkeypatch, capsys):
    """After midnight a 10:30 PM ET game is still live: not ingested, which is
    normal for a few hours (a warning; the next games of its teams are
    provisional meanwhile), and RED once it is STUCK_AFTER past its tip (a
    stuck feed). Live games are never fetched."""
    du = finals_bot
    du.boxes = {}                                   # any fetch fails the test (KeyError -> RED)
    tip = dt.datetime(2026, 11, 11, 3, 30, tzinfo=UTC)          # 10:30 PM ET on Nov 10
    sched = pd.DataFrame([{"game_id": "0022600215", "season": "2026-27", "game_type": "RS",
                           "status": "live", "game_date": dt.date(2026, 11, 10), "tip_utc": tip,
                           "home": "DEN", "away": "ORL"}])
    monkeypatch.setattr(du, "now_utc", lambda: tip + dt.timedelta(hours=1, minutes=47))  # 12:17 AM ET
    du.step_finals(sched, dt.date(2026, 11, 11), dry=False)
    out = capsys.readouterr().out
    assert du.RED == [] and "WARN finals: 1 state game(s) before today not ingested yet" in out
    assert "0022600215 (live)" in out
    monkeypatch.setattr(du, "now_utc", lambda: tip + dt.timedelta(hours=7))
    du.step_finals(sched, dt.date(2026, 11, 11), dry=False)
    assert len(du.RED) == 1 and "0022600215 (live)" in du.RED[0]


def test_a_final_game_with_a_committed_override_is_never_refetched(tmp_path, monkeypatch):
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    gid = "0021900001"
    payload = C.read_json(FIX / "cdnbox" / f"cdn_{gid}.json.gz")["data"]
    part, meta = box.parse_cdn(payload, gid)
    frames = {"participation": part, "games": pd.DataFrame([meta]),
              "player_games": cdnbox.player_lines(payload), "team_games": cdnbox.team_lines(payload)}
    S.write_override("2019-20", {k: v.assign(season="2019-20", season_type="RS") for k, v in frames.items()})

    def fetch(game_id):
        raise AssertionError(f"re-fetched {game_id}, which is committed as an override")
    monkeypatch.setattr(du, "cdn_box", fetch)
    sched = pd.DataFrame([_sched_row(payload)])
    du.RED.clear()
    du.step_finals(sched, dt.date(2019, 12, 31), dry=False)
    assert du.RED == []
    assert not list(S.season_dir("2019-20").glob("*.parquet"))   # nothing written beside the override
    assert S.ingested_ids("2019-20") == {gid}


# ── End to end: main() writes only under state/ ──────────────────────────────
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


@pytest.fixture
def network(monkeypatch):
    """bucketlib.http.fetch answered from fixtures: the cdn schedule and
    roster trims, ESPN's 2026-10-20 scoreboard, and for any other date its
    2026-10-08 preseason scoreboard (one priced preseason game, which must
    never be snapshotted into state),
    the fixture PDF for the first injury slot asked (a March report: not a
    real report for an October schedule, so the walk goes on) and Absent for
    the rest. Any other URL fails the test (stats.nba.com above all)."""
    calls = []
    sb, pre = (C.read_json(FIX / "espn" / f"live_sb_{d}.json.gz") for d in ("20261020", "20261008"))
    sb, pre = (x["data"] if "events" not in x else x for x in (sb, pre))
    pdf = (FIX / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes()

    def fetch(url, *, params=None, expect="json", **kw):
        calls.append(url)
        if url == cdn.SCHEDULE_URL:
            return json.loads((FIX / "cdn" / "schedule_2026-27_trim.json").read_text())
        if url == cdn.PLAYER_INDEX_URL:
            return json.loads((FIX / "cdn" / "playerIndex_trim.json").read_text())
        if url.startswith(espn.SITE + "scoreboard"):
            return sb if params["dates"] == "20261020" else pre
        if url.startswith(INJ.BASE_URL):
            if sum(u.startswith(INJ.BASE_URL) for u in calls) == 1:
                return pdf
            raise H.Absent(url, "test: not published")
        if url.startswith(cdn.BOX_URL.split("{")[0]):
            raise H.Absent(url, "test: no box score")
        raise AssertionError(f"the bot fetched an unexpected URL: {url}")
    monkeypatch.setattr(H, "fetch", fetch)
    return calls


def _pin(du, monkeypatch, at: dt.datetime) -> None:
    """The bot's clock (every clock read in the bot goes through these two)."""
    monkeypatch.setattr(du, "now_utc", lambda: at.astimezone(UTC))
    monkeypatch.setattr(du, "now_et", lambda: at.astimezone(ET))


def test_the_bot_writes_only_under_state(tmp_path, monkeypatch, network, capsys):
    if not (S.SEED_DIR / S.MANIFEST).exists():
        pytest.skip("no committed seed")
    from bucketlib.sources import statsnba
    du = _daily_update()
    # The evening before opening night (the runbook's first manual run), on a
    # pinned clock: on the real one the fixture's games freeze on Oct 20.
    _pin(du, monkeypatch, dt.datetime(2026, 10, 19, 18, 0, tzinfo=ET))
    state = tmp_path / "state"
    monkeypatch.setattr(S, "STATE_DIR", state)      # the committed seed is read where it is
    monkeypatch.setattr(cdn, "MIN_ROSTER", 1)       # the fixture roster is a trim of six players
    # cache/ is pointed at a tmp dir too (every module that holds the path),
    # so a write there is caught without ever landing in the real cache,
    # however the checkout's cache/ is linked.
    cache = tmp_path / "cache"
    for mod in (C, INJ, statsnba):
        monkeypatch.setattr(mod, "CACHE_DIR", cache)
    monkeypatch.setattr(INJ, "_default_store", None)
    before = _tree(REPO)

    assert du.main(["--date", "2026-10-19", "--dry-run"]) == 0, du.RED
    assert not state.exists() or not any(p.is_file() for p in state.rglob("*"))
    capsys.readouterr()
    assert du.main(["--date", "2026-10-19"]) == 0, du.RED
    out = capsys.readouterr().out

    after = _tree(REPO)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert changed == [], f"the bot wrote outside state/: {changed[:10]}"
    assert not cache.exists() or not any(p.is_file() for p in cache.rglob("*")), \
        f"the bot wrote under cache/: {[str(p) for p in cache.rglob('*')][:5]}"
    written = sorted(str(p.relative_to(state)) for p in state.rglob("*.parquet"))
    assert written == ["current/roster_v1.parquet", "current/schedule_v1.parquet",
                       "season/2026-27/odds_2026-10-20.parquet",
                       "serve/2026-10-20/features_v1.parquet"]
    feats = C.read_parquet(state / "serve" / "2026-10-20" / "features_v1.parquet")
    assert len(feats) and set(feats["game_id"]) <= {"0022600001", "0022600002", "0022600003"}
    assert C.read_parquet_meta(state / "serve" / "2026-10-20" / "features_v1.parquet")[P.HISTORY] \
        == {"through": None, "games": 0, "digest": P.history_stamp("2026-27", "2026-10-20")["digest"]}
    # The fixture roster is a trim of six players: every game is short a side,
    # which is named (a warning: the slate is tomorrow's), and no real report
    # lists an October game, so nothing is projected (and that is said).
    assert "WARN projections 2026-10-20: 3 untipped game(s) without a whole roster" in out
    assert "0022600001 BOS 0 / DET" in out
    assert "WARN projections 2026-10-20: nothing new written: no injury report at all" in out
    assert not any("stats.nba.com" in u for u in network)


def test_the_apps_fallbacks_read_what_the_bot_writes(tmp_path, monkeypatch, network):
    """live.schedule() and live.roster() fall back to state/current, the files
    step_schedule and step_roster commit (one path, through bucketlib.state)."""
    from bucketlib import live
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cdn, "MIN_ROSTER", 1)
    sched, roster = du.step_schedule(False), du.step_roster(False)
    assert S.current_path("schedule").exists() and S.current_path("roster").exists()

    def down(url, **kw):
        raise H.Blocked(url, "test: host down")
    monkeypatch.setattr(H, "fetch", down)
    monkeypatch.setattr(live, "_good", {})
    monkeypatch.setattr(live, "_fail", {})
    for got, want in ((live.schedule(), sched), (live.roster(), roster)):
        assert got.source == "committed" and got.stale
        # Same values; parquet hands the ET zone back as another tz object.
        pd.testing.assert_frame_equal(got.value, want, check_dtype=False)


def test_an_exception_in_a_step_is_red_and_the_other_steps_run(tmp_path, monkeypatch, network):
    du = _daily_update()
    _pin(du, monkeypatch, dt.datetime(2026, 10, 9, 12, 0, tzinfo=ET))
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cdn, "MIN_ROSTER", 1)

    def boom(*a):
        raise RuntimeError("boom")
    ran = []
    monkeypatch.setattr(du, "step_finals", boom)
    monkeypatch.setattr(du, "step_projections", lambda *a: ran.append(a[4]))   # today
    assert du.main(["--date", "2026-10-09", "--dry-run"]) == 1
    assert du.RED == ["finals step raised RuntimeError: boom"]
    assert ran == [dt.date(2026, 10, 9)]


def test_the_report_resolves_against_the_historys_spellings(bot, sched, report):
    """The bot resolves injury names against History.names (every spelling the
    history carries), not only the roster's current spelling."""
    first = run(bot, sched, report, odds(sched), MORNING)
    listed = first[first["inj_status"].notna()].iloc[0]
    pid = int(listed["person_id"])
    # The roster now spells him in a way the report does not; the history
    # still carries the report's spelling.
    bot.edit[0] = lambda f: f.assign(player_name=f["player_name"].mask(f["person_id"] == pid,
                                                                       "Zz Renamed"))
    out = run(bot, sched, report, odds(sched), MORNING + dt.timedelta(minutes=30))
    me = out[out["person_id"] == pid].iloc[0]
    assert me["player_name"] == "Zz Renamed" and me["inj_status"] == listed["inj_status"]


# ── Stale beats empty: ANY failed pull keeps the committed copy ──────────────
def _schedule_payload() -> dict:
    return json.loads((FIX / "cdn" / "schedule_2026-27_trim.json").read_text())


def _mismatched_dates() -> dict:
    p = _schedule_payload()
    p["leagueSchedule"]["gameDates"][0]["games"][0]["gameDateEst"] = "1999-01-01T00:00:00Z"
    return p


def _unknown_team() -> dict:
    p = json.loads((FIX / "cdn" / "playerIndex_trim.json").read_text())
    rs = p["resultSets"][0]
    i = rs["headers"].index("TEAM_ID")
    rs["rowSet"][0][i] = 1612709999
    return p


@pytest.mark.parametrize("step,url,bad", [
    ("schedule", cdn.SCHEDULE_URL, H.BadPayload(cdn.SCHEDULE_URL, "test: HTML 200")),
    ("schedule", cdn.SCHEDULE_URL, H.Absent(cdn.SCHEDULE_URL, "test: gone")),
    ("schedule", cdn.SCHEDULE_URL, _mismatched_dates()),          # the parser's ValueError
    ("roster", cdn.PLAYER_INDEX_URL, H.BadPayload(cdn.PLAYER_INDEX_URL, "test: truncated JSON")),
    ("roster", cdn.PLAYER_INDEX_URL, _unknown_team()),             # unknown TEAM_ID: ValueError
])
def test_a_failed_pull_keeps_the_committed_copy(tmp_path, monkeypatch, network, step, url, bad):
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cdn, "MIN_ROSTER", 1)
    fn = {"schedule": du.step_schedule, "roster": du.step_roster}[step]
    good = fn(False)
    assert good is not None and S.current_path(step).exists() and du.RED == []
    before = S.current_path(step).read_bytes()
    real = H.fetch

    def fetch(u, **kw):
        if u == url:
            if isinstance(bad, Exception):
                raise bad
            return bad
        return real(u, **kw)
    monkeypatch.setattr(H, "fetch", fetch)
    got = fn(False)
    assert got is not None and len(got) == len(good)
    assert len(du.RED) == 1 and step in du.RED[0]
    assert S.current_path(step).read_bytes() == before


def test_a_truncated_injury_pdf_keeps_the_committed_report(tmp_path, monkeypatch, sched, report):
    from bucketlib import context as CTX
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    assert CTX.write_committed(report, sched)

    def truncated(*a, **k):
        raise H.BadPayload(INJ.BASE_URL, "test: truncated PDF")
    monkeypatch.setattr(INJ, "latest_report", truncated)
    rep = du.step_injuries(sched, False)
    assert rep is not None and rep.meta["report_ts_et"] == report.meta["report_ts_et"]
    assert len(du.RED) == 1 and "BadPayload" in du.RED[0]


# ── Odds: one capture per change, regular season and play-in only ────────────
def _scoreboard(at: dt.datetime, slug: str = "regular-season", move: float = 0.0) -> pd.DataFrame:
    x = C.read_json(FIX / "espn" / "live_sb_20261020.json.gz")
    x = x["data"] if "events" not in x else x
    sb = espn.parse_scoreboard(x, at.astimezone(UTC))
    sb.loc[sb["home"] == "DET", "home_line"] += move
    return sb.assign(season_slug=slug)


@pytest.fixture
def odds_bot(tmp_path, monkeypatch):
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    du.boards = {}                       # {date: scoreboard frame, or an exception}

    def scoreboard(d):
        b = du.boards.get(d, pd.DataFrame(columns=espn.SCOREBOARD_COLUMNS))
        if isinstance(b, Exception):
            raise b
        return b
    monkeypatch.setattr(espn, "scoreboard", scoreboard)
    du.RED.clear()
    return du


OPENER = dt.date(2026, 10, 20)


def _odds_path(d: dt.date):
    return S.odds_path(d)            # the one path the bot writes and the app reads


def test_an_unchanged_board_leaves_the_odds_file_byte_identical(odds_bot):
    """Each capture is stamped with its fetch time: a run whose lines did not
    move is no new capture, so the file (and the commit) stays as it was."""
    du = odds_bot
    t0 = dt.datetime(2026, 10, 20, 9, 17, tzinfo=ET)
    du.boards = {OPENER: _scoreboard(t0)}
    du.step_odds(OPENER, False)
    first = _odds_path(OPENER).read_bytes()
    for minutes in (30, 60, 90):
        du.boards = {OPENER: _scoreboard(t0 + dt.timedelta(minutes=minutes))}
        out = du.step_odds(OPENER, False)
        assert _odds_path(OPENER).read_bytes() == first, f"+{minutes} min rewrote the file"
        assert len(out[OPENER]) == 3
    # A moved line is a new capture: first and latest are kept for that game.
    later = t0 + dt.timedelta(hours=2)
    du.boards = {OPENER: _scoreboard(later, move=-1.0)}
    out = du.step_odds(OPENER, False)[OPENER]
    assert _odds_path(OPENER).read_bytes() != first and len(out) == 4
    det = out[out["home"] == "DET"].sort_values("captured_at")
    assert det["home_line"].tolist() == [-1.5, -2.5]
    assert P.lines_from_odds(out, pd.DataFrame({"game_id": ["g"], "away": ["BOS"], "home": ["DET"]})
                             )["home_line"].tolist() == [-2.5]
    assert du.RED == []


def test_playoff_and_preseason_lines_never_enter_state(odds_bot):
    du = odds_bot
    at = dt.datetime(2026, 10, 20, 9, 17, tzinfo=ET)
    for slug in ("post-season", "preseason"):
        du.boards = {OPENER: _scoreboard(at, slug=slug)}
        assert du.step_odds(OPENER, False)[OPENER] is None
        assert not _odds_path(OPENER).exists()
    du.boards = {OPENER: _scoreboard(at, slug="play-in-season")}
    assert len(du.step_odds(OPENER, False)[OPENER]) == 3


def test_a_scoreboard_that_does_not_parse_keeps_the_committed_snapshots(odds_bot):
    """Tomorrow's scoreboard raising (a parser ValueError) neither loses
    today's capture nor tomorrow's committed one, and the run is RED."""
    du = odds_bot
    at = dt.datetime(2026, 10, 19, 21, 17, tzinfo=ET)
    du.boards = {OPENER: _scoreboard(at)}
    du.step_odds(OPENER - dt.timedelta(days=1), False)            # Oct 19: tomorrow is the opener
    committed = C.read_parquet(_odds_path(OPENER))
    du.boards = {OPENER: ValueError("test: one competitor")}
    out = du.step_odds(OPENER - dt.timedelta(days=1), False)
    pd.testing.assert_frame_equal(out[OPENER], committed)
    assert len(du.RED) == 1 and "ValueError" in du.RED[0]


# ── Camp rosters, --date, the probe ──────────────────────────────────────────
def test_a_camp_roster_is_red_today_and_a_warning_tomorrow(bot, sched, report, capsys):
    big = pd.DataFrame({"person_id": range(1, 22), "team": "DET", "name": "x", "two_way": False})
    roster = pd.concat([big, pd.DataFrame({"person_id": [99], "team": ["PHI"], "name": ["y"],
                                           "two_way": [False]})], ignore_index=True)
    bot.clock[0] = MORNING.astimezone(UTC)
    bot.step_projections(sched, roster, report, {DATE: odds(sched)}, DATE, False)
    assert len(bot.RED) == 1 and "more than 18 players for DET 21" in bot.RED[0]
    bot.RED.clear()
    capsys.readouterr()
    bot.step_projections(sched, roster, report, {DATE: odds(sched)}, DATE - dt.timedelta(days=1), False)
    assert bot.RED == [] and "WARN projections 2026-03-12: the roster pull lists more than 18" \
        in capsys.readouterr().out
    assert bot.oversized(sched, ROSTER) == {}


def test_a_future_date_is_refused(tmp_path, monkeypatch, capsys):
    """--date replays the past: the injury walk, the odds and the freeze read
    the real clock, so a future "today" would publish freezable rows from
    stale inputs."""
    du = _daily_update()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    _pin(du, monkeypatch, dt.datetime(2026, 10, 19, 18, 0, tzinfo=ET))
    ran = []
    for step in ("step_schedule", "step_roster", "step_odds", "step_projections"):
        monkeypatch.setattr(du, step, lambda *a, _s=step: ran.append(_s))
    assert du.main(["--date", "2026-10-20"]) == 2 and ran == []
    assert "refusing --date 2026-10-20" in capsys.readouterr().out
    assert du.main(["--date", "2026-10-19", "--dry-run"]) == 0 and "step_projections" in ran


def test_the_probe_covers_every_source_the_bot_reads():
    """scripts/probe_hosts.py (run on Actions) reaches every URL shape the bot
    fetches: a source missing from the probe is an unproven assumption."""
    spec = importlib.util.spec_from_file_location("probe_hosts", REPO / "scripts" / "probe_hosts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    urls = [t[1] for t in mod.TARGETS]
    for want in (cdn.SCHEDULE_URL, cdn.PLAYER_INDEX_URL):
        assert want in urls, want
    for prefix in (cdn.BOX_URL.split("{")[0], INJ.BASE_URL, espn.SITE + "scoreboard"):
        assert any(u.startswith(prefix) for u in urls), prefix
