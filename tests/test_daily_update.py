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


def run(du, sched, report, snaps, at, today=DATE):
    du.clock[0] = at.astimezone(UTC)
    du.step_projections(sched, ROSTER, report, {DATE: snaps}, today, False)
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


def test_a_date_whose_state_misses_a_final_is_provisional(bot, sched, report):
    missed = pd.DataFrame([{"game_id": "0022500940", "season": "2025-26", "game_type": "RS",
                            "game_date": DATE - dt.timedelta(days=1),
                            "tip_utc": pd.Timestamp("2026-03-12 00:00", tz="UTC"), "status": "final",
                            "home": "DET", "away": "ATL"}])
    full = pd.concat([missed, sched], ignore_index=True)
    out = run(bot, full, report, odds(sched), MORNING)
    assert bot.RED == [] and out["provisional"].all()
    assert P.provisional_for(DATE, DATE, full, "2025-26") == (True, ["0022500940"])


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


# ── Finals: state.ingested_ids is the one definition ─────────────────────────
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
    tip = pd.Timestamp(payload["game"]["gameTimeUTC"])
    sched = pd.DataFrame([{"game_id": gid, "season": "2019-20", "game_type": "RS", "status": "final",
                           "game_date": tip.tz_convert("America/New_York").date(),
                           "home": None, "away": None}])
    du.RED.clear()
    du.step_finals(sched, dt.date(2019, 12, 31), dry=False)
    assert du.RED == []
    assert not list(S.season_dir("2019-20").glob("*.parquet"))   # nothing written beside the override


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
    monkeypatch.setattr(du, "MIN_ROSTER", 1)        # the fixture roster is a trim of six players
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
    monkeypatch.setattr(du, "MIN_ROSTER", 1)
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
    monkeypatch.setattr(du, "MIN_ROSTER", 1)

    def boom(*a):
        raise RuntimeError("boom")
    ran = []
    monkeypatch.setattr(du, "step_finals", boom)
    monkeypatch.setattr(du, "step_projections", lambda *a: ran.append(a[-2]))
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
