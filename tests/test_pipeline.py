"""bucketlib.pipeline: the projection frame, the freeze, the app's live frame.

Offline, on real data: the nine games of 2026-03-12 as the bot would commit
their history features (tests/fixtures/pipeline, cut by
scripts/make_pipeline_fixture.py from the training table), the committed real
injury report of that day (12:00 PM ET), the production models (models/,
verified by models.load). The lines are made up: historical lines are never
committed. Every check runs production code; where a property is "the same
as", the other side is production code too (context.context_for,
models.predict).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import context as X
from bucketlib import features as F
from bucketlib import live as LV
from bucketlib import livectx as L
from bucketlib import models as M
from bucketlib import pipeline as P
from bucketlib import serve as V
from bucketlib import state as S
from bucketlib.sources import box, cdnbox
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

FIX = Path(__file__).resolve().parent / "fixtures"
DATE = dt.date(2026, 3, 12)
SEASON = "2025-26"
MORNING = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)    # the 12:00 PM report is 20 minutes old
EVENING = dt.datetime(2026, 3, 12, 19, 0, tzinfo=ET)     # 23:00 UTC: the three 7:00 PM ET games froze at 6:55
EARLY = {"0022500953", "0022500954", "0022500955"}       # tip 23:00 UTC


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
def games(sched) -> pd.DataFrame:
    return P.state_games(sched, DATE)


@pytest.fixture(scope="module")
def report() -> INJ.ParsedReport:
    return INJ.parse_report((FIX / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())


@pytest.fixture(scope="module")
def reissue() -> INJ.ParsedReport:
    """The 12:15 PM report of the same day: row for row the 12:00 PM report
    under a new title (the NBA re-titles an unchanged report every 15
    minutes). Fetched from ak-static by the 2026-10-09 audit."""
    return INJ.parse_report((FIX / "injury" / "Injury-Report_2026-03-12_12_15PM.pdf").read_bytes())


def retitled(report: INJ.ParsedReport, ts: dt.datetime) -> INJ.ParsedReport:
    return INJ.ParsedReport({**report.meta, "report_ts_et": ts}, report.rows.assign(report_ts_et=ts),
                            report.team_rows.assign(report_ts_et=ts))


@pytest.fixture(scope="module")
def names(feats) -> pd.DataFrame:
    return feats[["person_id", "player_name"]].drop_duplicates()


def made_up_lines(games: pd.DataFrame, shift: float = 0.0) -> pd.DataFrame:
    n = len(games)
    return pd.DataFrame({"game_id": games["game_id"].to_numpy(),
                         "home_line": np.linspace(-9.5, 7.5, n) + shift,
                         "total": np.linspace(214.5, 236.5, n) + shift})


@pytest.fixture(scope="module")
def base(feats, games, sched, report, mdl, names) -> pd.DataFrame:
    return P.project(feats, games, sched, report, made_up_lines(games), mdl, MORNING, names=names)


# ── project(): the column contract ───────────────────────────────────────────
def test_project_frame_has_the_contract_columns_and_dtypes(base, feats):
    assert list(base.columns) == P.COLUMNS
    assert {c: str(t) for c, t in base.dtypes.items()} == P.DTYPES
    assert len(base) == len(feats) and not base.duplicated(["person_id", "game_id"]).any()
    assert base["provisional"].dtype == bool and not base["provisional"].any()
    assert not base[["frozen", "adjusted"]].any().any()
    # Section 3 says these columns exist; the contract test pins every group.
    for group in (P.IDENTITY, P.AVAILABILITY, P.MEANS, P.SIM_INPUTS, P.PROPS, P.PROVENANCE):
        assert set(group) <= set(base.columns)
    assert set(P.MEANS) == {"min", *M.STATS, *M.COMPOSED}


def test_project_is_apply_lines_then_context_then_predict(base, feats, games, sched, report, mdl,
                                                          names):
    """The frame's numbers are models.predict on context.context_for of
    features.apply_lines: the production path, nothing recomputed."""
    rows = F.apply_lines(feats, made_up_lines(games))
    ctx = X.context_for(rows, games, sched, report, MORNING, names)
    pred = M.predict(mdl, ctx)
    want = pd.concat([ctx[["game_id", "person_id", "career_games", *X.DISPLAY_COLS, "home_line",
                           "total"]], pred], axis=1).set_index(["game_id", "person_id"]).sort_index()
    got = base.set_index(["game_id", "person_id"]).sort_index()
    for c in ["p_play", "mu_min", *M.STATS, *M.COMPOSED, *(f"rate_{s}" for s in M.STATS),
              "home_line", "total"]:
        assert np.array_equal(got[c].to_numpy(), want[c].to_numpy(), equal_nan=True), c
    assert np.array_equal(got["min"].to_numpy(), want["mu_min"].to_numpy())
    assert np.array_equal(got["games_prior"].to_numpy(), want["career_games"].to_numpy(), equal_nan=True)
    for c in X.DISPLAY_COLS:
        assert got[c].tolist() == want[c].tolist(), c
    assert (got["model_version"] == mdl.version).all()
    assert (got["computed_at"] == pd.Timestamp(MORNING).tz_convert(UTC)).all()
    known = got["team_state"] != "no_report"
    assert known.all()     # the report is in reach for every game at 12:20
    assert (got["report_ts_et"] == pd.Timestamp(report.meta["report_ts_et"])).all()
    # Unknown is never healthy: Denver had not submitted at noon.
    den = X.display_status(base[base["team"] == "DEN"])
    assert (den["status"] == X.NOT_FILED).all()


def test_report_time_only_on_games_the_report_was_in_reach_for(feats, games, sched, report, mdl,
                                                                names):
    """One report per call: a game whose cutoff (tip - 5 min) is before the
    report's title is no_report, with no report time and an unknown badge."""
    title = dt.datetime(2026, 3, 12, 19, 0, tzinfo=ET)          # after the 7:00 PM games' cutoff
    later = INJ.ParsedReport({**report.meta, "report_ts_et": title},
                             report.rows.assign(report_ts_et=title),
                             report.team_rows.assign(report_ts_et=title))
    gid = "0022500953"
    rows = feats[~feats["game_id"].isin(EARLY - {gid})]
    out = P.project(rows, games, sched, later, made_up_lines(games), mdl,
                    dt.datetime(2026, 3, 12, 19, 10, tzinfo=ET), names=names)
    early = out["game_id"] == gid
    assert early.any() and (~early).any()
    assert (out.loc[early, "team_state"] == "no_report").all() and out.loc[early, "report_ts_et"].isna().all()
    assert (X.display_status(out[early])["status"] == X.NO_REPORT).all()
    assert (out.loc[~early, "report_ts_et"] == pd.Timestamp(title)).all()


def test_the_frame_survives_the_parquet_round_trip(base, tmp_path):
    p = tmp_path / "projections_v1.parquet"
    C.write_parquet(base, p)
    back = C.read_parquet(p)
    pd.testing.assert_frame_equal(back, base, check_exact=True)
    pd.testing.assert_frame_equal(P.conform(back), base, check_exact=True)


def test_provisional_is_never_missing_and_is_what_the_caller_says(feats, games, sched, report, mdl,
                                                                   names):
    out = P.project(feats, games, sched, report, made_up_lines(games), mdl, MORNING,
                    provisional=True, names=names)
    assert out["provisional"].dtype == bool and out["provisional"].all()
    with pytest.raises(ValueError, match="missing values"):
        P.conform(out.assign(provisional=np.where(out.index == 0, np.nan, True)))


def test_provisional_is_set_per_game_from_a_map_and_a_missing_game_is_provisional(
        base, feats, games, sched, report, mdl, names):
    """project() takes provisional_for's {game_id: bool} map as is: each row
    carries its own game's flag, and a game the map does not name is
    provisional (fail closed). Nothing else in the frame moves."""
    g = list(games["game_id"])
    prov = {g[0]: True, **{x: False for x in g[2:]}}           # g[1] is not in the map
    out = P.project(feats, games, sched, report, made_up_lines(games), mdl, MORNING,
                    provisional=prov, names=names)
    assert out["provisional"].dtype == bool
    assert out.groupby("game_id")["provisional"].agg(["all", "any"]).to_dict("index") == {
        x: {"all": x in g[:2], "any": x in g[:2]} for x in g}
    pd.testing.assert_frame_equal(out.drop(columns="provisional"), base.drop(columns="provisional"),
                                  check_exact=True)


# ── project(): the missing-line policy and the null check ─────────────────────
def test_a_game_without_a_line_is_withheld_and_moves_nothing_else(base, feats, games, sched, report,
                                                                  mdl, names):
    lines = made_up_lines(games)
    lines.loc[0, "total"] = np.nan                         # a spread but no total
    lines = lines[lines["game_id"] != games["game_id"].iloc[1]]   # no line at all
    out = P.project(feats, games, sched, report, lines, mdl, MORNING, names=names)
    held = sorted(games["game_id"].iloc[:2])
    assert out.attrs["withheld"] == held
    assert not out["game_id"].isin(held).any()
    rest = base[~base["game_id"].isin(held)].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), rest, check_exact=True)
    none = P.project(feats, games, sched, report, None, mdl, MORNING, names=names)
    assert none.empty and none.attrs["withheld"] == sorted(games["game_id"])
    assert list(none.columns) == P.COLUMNS


def test_no_report_is_a_null_skew_not_a_healthy_slate(feats, games, sched, mdl, names):
    with pytest.raises(M.NullSkew, match="report_age_h"):
        P.project(feats, games, sched, None, made_up_lines(games), mdl, MORNING, names=names)


def test_no_report_is_refused_even_where_the_null_check_cannot_see_it(feats, games, sched, mdl,
                                                                       names):
    """check_nulls skips strata under models.NULL_MIN_ROWS rows, so a small
    slate with no report (a thin opening-night roster, a trimmed fixture)
    passes it; project() refuses it anyway."""
    one = feats[feats["game_id"] == games["game_id"].iloc[0]].head(M.NULL_MIN_ROWS - 1)
    small = F.apply_lines(one, made_up_lines(games))
    ctx = X.context_for(small, games, sched, None, MORNING, names)
    assert ctx["report_age_h"].isna().all()
    M.check_nulls(mdl, ctx)                     # the null check alone lets it through
    with pytest.raises(P.NoReport, match="no injury report"):
        P.project(one, games, sched, None, made_up_lines(games), mdl, MORNING, names=names)


def test_a_report_out_of_reach_for_every_game_is_no_report(feats, games, sched, report, mdl, names):
    """A report held but titled 30+ hours before the cutoff (the committed or
    in-memory report after a day of ak-static outage) is out of reach for
    every game: every team no_report, every status unknown, report_age_h
    null, the same slate as no report at all. project() refuses it as
    NoReport, naming the title, whatever the slate size: a one-game slate,
    and a slate small enough that check_nulls skips every stratum."""
    stale = retitled(report, MORNING - dt.timedelta(hours=31))
    gid = games["game_id"].iloc[-1]
    one = feats[feats["game_id"] == gid]
    with pytest.raises(P.NoReport, match=r"titled 2026-03-11 05:20 ET is out of reach for every "
                                         r"game.*30 hours"):
        P.project(one, games, sched, stale, made_up_lines(games), mdl, MORNING, names=names)
    small = one.head(M.NULL_MIN_ROWS - 1)
    ctx = X.context_for(F.apply_lines(small, made_up_lines(games)), games, sched, stale, MORNING, names)
    assert (ctx["team_state"] == "no_report").all() and ctx["report_age_h"].isna().all()
    M.check_nulls(mdl, ctx)                     # the null check alone lets it through
    with pytest.raises(P.NoReport, match="out of reach"):
        P.project(small, games, sched, stale, made_up_lines(games), mdl, MORNING, names=names)
    # 29 hours old is in reach (the training rule's 30): the same slate projects.
    ok = P.project(one, games, sched, retitled(report, MORNING - dt.timedelta(hours=29)),
                   made_up_lines(games), mdl, MORNING, names=names)
    assert len(ok) == len(one) and (ok["team_state"] != "no_report").all()


def test_a_slate_with_many_players_out_is_projected_not_withheld(feats, games, sched, report, mdl,
                                                                 names):
    """tm_out_max_min is NaN exactly on rows listed Out, so a late slate
    with a high Out share (2025-26 at t_15: 0.37-0.40 of 10+|0 rows) read as
    a null skew and the whole date was withheld. Marked out the way a
    report lists them (status Out), 45% of this slate's 10+|0 players still
    project, and the Out rows are exactly the NaN ones."""
    ten = feats[(M.null_strata(feats) == "10+|0").to_numpy()]
    # Spread over every team, as a report lists them (a whole team Out is
    # another slate: own_exp_min_share has no team minutes to share).
    k = ten.groupby(["game_id", "team"]).cumcount()
    out_ids = set(ten.loc[(k % 20 < 9).to_numpy(), "person_id"].astype(int))
    frame = P.project(feats, games, sched, report, made_up_lines(games), mdl, MORNING,
                      marked_out=out_ids, names=names)
    assert len(frame) == len(feats)
    assert (frame.loc[frame["person_id"].isin(out_ids), "inj_status"] == "Out").all()
    ctx = X.context_for(F.apply_lines(feats, made_up_lines(games)), games, sched, report, MORNING,
                        names, marked_out=out_ids)
    ten_ctx = (M.null_strata(ctx) == "10+|0").to_numpy()
    assert ctx.loc[ten_ctx, "tm_out_max_min"].isna().mean() > 0.45
    assert (ctx["tm_out_max_min"].isna() == ctx["inj_out"].eq(1.0)).all()


def test_feature_rows_that_already_carry_lines_are_refused(feats, games, sched, report, mdl, names):
    with pytest.raises(ValueError, match="line columns"):
        P.project(F.apply_lines(feats, made_up_lines(games)), games, sched, report,
                  made_up_lines(games), mdl, MORNING, names=names)


# ── Mark-outs ────────────────────────────────────────────────────────────────
def _healthy_starter(frame: pd.DataFrame, games: set[str]) -> pd.Series:
    """The highest-minute player of `games` whose team filed and who is not on it."""
    ok = (frame["game_id"].isin(games) & frame["inj_status"].isna()
          & (frame["team_state"] == "listed"))
    return frame[ok].sort_values("mu_min", ascending=False).iloc[0]


def test_a_mark_out_moves_teammates_minutes_up_and_touches_no_other_team(base, feats, games, sched,
                                                                         report, mdl, names):
    star = _healthy_starter(base, set(games["game_id"]))
    out = P.project(feats, games, sched, report, made_up_lines(games), mdl, MORNING,
                    marked_out={int(star["person_id"])}, names=names)
    key = ["game_id", "person_id"]
    a, b = base.set_index(key).sort_index(), out.set_index(key).sort_index()
    me = (star["game_id"], star["person_id"])
    team = (b["team"] == star["team"]).to_numpy() & (b.index.get_level_values(0) == star["game_id"])
    assert b.loc[me, "inj_status"] == "Out" and b.loc[me, "inj_reason"] == X.MARKED_REASON
    assert b.loc[me, "p_play"] < a.loc[me, "p_play"]
    mates = team & (b.index != me)
    gain = b.loc[mates, "mu_min"] - a.loc[mates, "mu_min"]
    assert gain.sum() > 3.0 and (gain >= -0.05).all(), gain.sort_values().to_dict()
    assert b["adjusted"].to_numpy().tolist() == team.tolist()
    stamps = list(P.STAMPS)
    pd.testing.assert_frame_equal(a.loc[~team].drop(columns=stamps + ["adjusted"]),
                                  b.loc[~team].drop(columns=stamps + ["adjusted"]), check_exact=True)


# ── Slate helpers ────────────────────────────────────────────────────────────
def test_frozen_from_tip_minus_five_minutes(games):
    tip = pd.Timestamp(games["tip_utc"].iloc[0]).to_pydatetime()
    at = tip - P.FREEZE_LEAD
    assert P.FREEZE_LEAD == dt.timedelta(minutes=5)
    assert P.frozen_ids(games, at) == EARLY
    assert P.frozen_ids(games, at - dt.timedelta(seconds=1)) == set()
    assert P.frozen_ids(games, EVENING) == EARLY
    with pytest.raises(ValueError, match="naive"):
        P.frozen_ids(games, dt.datetime(2026, 3, 12, 23, 0))


def test_a_game_whose_time_is_not_set_is_never_frozen(games):
    """cdn's time_tbd: the schedule's tip is a made-up 00:00 ET (05:00Z), so a
    TBD game has no real tip and no freeze point, whatever the clock. The
    same tip with a real time freezes; a frame parsed before the column
    existed falls back to the feed's own 'TBD' text (live.time_tbd)."""
    tip = pd.Timestamp("2026-03-12 05:00", tz="UTC")             # 00:00 ET placeholder
    rows = pd.DataFrame({"game_id": ["tbd", "timed"], "tip_utc": [tip, tip],
                         "time_tbd": [True, False], "status_text": ["TBD", "7:00 pm ET"]})
    assert P.frozen_ids(rows, MORNING) == {"timed"}
    old = rows.drop(columns="time_tbd")
    assert P.frozen_ids(old, MORNING) == {"timed"}
    # The slate fixture's real games (no time_tbd column) freeze as before.
    assert P.frozen_ids(games.assign(time_tbd=False), EVENING) == EARLY
    assert P.frozen_ids(games.assign(time_tbd=True), EVENING) == set()


def test_state_games_keep_python_dates_and_drop_other_games(sched):
    extra = pd.DataFrame([
        {"game_id": "0012500001", "season": SEASON, "game_type": "PRE", "game_date": DATE,
         "tip_utc": pd.Timestamp("2026-03-12 23:00", tz="UTC"), "status": "scheduled",
         "home": "BOS", "away": "NYK"},
        {"game_id": "0022501201", "season": SEASON, "game_type": "RS", "game_date": DATE,
         "tip_utc": pd.Timestamp("2026-03-12 23:00", tz="UTC"), "status": "scheduled",
         "home": None, "away": None},
        {"game_id": "0022500999", "season": SEASON, "game_type": "RS", "game_date": dt.date(2026, 3, 13),
         "tip_utc": pd.Timestamp("2026-03-13 23:00", tz="UTC"), "status": "scheduled",
         "home": "BOS", "away": "NYK"}])
    g = P.state_games(pd.concat([sched, extra], ignore_index=True), DATE)
    assert set(g["game_id"]) == set(sched["game_id"])
    assert all(type(d) is dt.date for d in g["game_date"])
    assert g["tip_utc"].is_monotonic_increasing


def test_a_play_in_game_is_a_state_game_that_is_never_projected(sched, tmp_path, monkeypatch):
    """Every production model trains on regular-season rows only, and the
    play-in measured biased (starters 4.9 minutes short), so a play-in game
    gets no projection anywhere (state_games feeds the bot, the app's live
    frame and store.projected_ids) while it stays a state game: its box score
    is ingested and a missing one is a completeness gap for the next date."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    assert P.PROJECTED_TYPES == {"RS"}
    pi = pd.DataFrame([
        {"game_id": "0052500101", "season": SEASON, "game_type": "PI", "game_date": DATE,
         "tip_utc": pd.Timestamp("2026-03-12 23:00", tz="UTC"), "status": "final",
         "home": "MIA", "away": "CHI"},
        {"game_id": "0022501230", "season": SEASON, "game_type": "RS",
         "game_date": DATE + dt.timedelta(days=1),
         "tip_utc": pd.Timestamp("2026-03-13 23:00", tz="UTC"), "status": "scheduled",
         "home": "MIA", "away": "NYK"}])
    full = pd.concat([sched, pi], ignore_index=True)
    assert set(P.state_games(full, DATE)["game_id"]) == set(sched["game_id"])
    assert "0052500101" in set(V._state_games(full)["game_id"])
    assert "0052500101" in V.completeness(DATE + dt.timedelta(days=1), full, SEASON)
    assert set(P.state_games(full, DATE + dt.timedelta(days=1))["game_id"]) == {"0022501230"}


def test_lines_from_both_odds_shapes(games):
    g = games.iloc[:2]
    (a1, h1), (a2, h2) = zip(g["away"], g["home"])
    as_dict = {(a1, h1): {"home_line": -3.5, "total": 221.5}, (a2, h2): {"home_line": None, "total": 230.0},
               ("XXX", "YYY"): {"home_line": 1.0, "total": 200.0}}
    got = P.lines_from_odds(as_dict, games).set_index("game_id")
    assert got.loc[g["game_id"].iloc[0]].tolist() == [-3.5, 221.5]
    assert np.isnan(got.loc[g["game_id"].iloc[1], "home_line"]) and len(got) == 2
    t0 = pd.Timestamp("2026-03-12 15:00", tz="UTC")
    snaps = pd.DataFrame({"away": [a1, a1, a2], "home": [h1, h1, h2], "home_line": [-3.5, -4.5, 2.0],
                          "total": [221.5, 222.0, 230.0],
                          "captured_at": [t0 + pd.Timedelta(hours=2), t0, t0]})
    got = P.lines_from_odds(snaps, games).set_index("game_id")
    assert got.loc[g["game_id"].iloc[0]].tolist() == [-3.5, 221.5]      # the latest capture
    assert P.lines_from_odds(None, games).empty and P.lines_from_odds({}, games).empty


# ── merge_published: the bot's file ──────────────────────────────────────────
def test_merge_carries_frozen_games_and_keeps_unchanged_rows(base, feats, games, sched, report, mdl,
                                                             names):
    later = P.project(feats, games, sched, report, made_up_lines(games, shift=1.0), mdl,
                      MORNING + dt.timedelta(hours=1), names=names)
    out, _ = P.merge_published(base, later, games, EVENING)
    early = out["game_id"].isin(EARLY)
    pd.testing.assert_frame_equal(out[early].reset_index(drop=True),
                                  base[base["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    pd.testing.assert_frame_equal(out[~early].reset_index(drop=True),
                                  later[~later["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    # A recompute that changes nothing but the stamps keeps the committed rows.
    again = P.project(feats, games, sched, report, made_up_lines(games), mdl,
                      MORNING + dt.timedelta(minutes=1), names=names)
    assert (again["computed_at"] != base["computed_at"]).all()
    pd.testing.assert_frame_equal(P.merge_published(base, again, games, MORNING)[0], base,
                                  check_exact=True)


def test_merge_never_writes_a_game_first_seen_after_its_freeze(base, games):
    out, _ = P.merge_published(None, base, games, EVENING)
    assert set(out["game_id"]) == set(games["game_id"]) - EARLY
    # Withheld this pass: the committed rows stand (stale beats empty).
    late = base[~base["game_id"].isin({games["game_id"].iloc[-1]})]
    kept, _ = P.merge_published(base, late, games, MORNING)
    pd.testing.assert_frame_equal(kept, base, check_exact=True)
    # A game no longer on the date (postponed) leaves the file.
    gone, _ = P.merge_published(base, base, games[games["game_id"] != games["game_id"].iloc[-1]],
                                MORNING)
    assert set(gone["game_id"]) == set(games["game_id"].iloc[:-1])
    assert P.merge_published(None, None, games, MORNING)[0].empty


def test_merge_keeps_a_game_whose_only_change_is_the_clock(base, feats, games, sched, report, mdl,
                                                           names):
    """hours_to_tip and report_age_h move with the clock alone. Same inputs
    and same feature rows: the committed rows stand (no commit every cron);
    a crossed training horizon, a new line or changed feature rows rewrite
    that game only."""
    digests = P.feature_digests(feats)
    at = MORNING + dt.timedelta(minutes=30)
    later = P.project(feats, games, sched, report, made_up_lines(games), mdl, at, names=names)
    moved = later.set_index(["game_id", "person_id"])["mu_min"] \
        - base.set_index(["game_id", "person_id"])["mu_min"]
    assert (moved != 0).any()                                     # the clock alone moves numbers
    out, basis = P.merge_published(base, later, games, at, old_basis=digests, new_basis=digests)
    pd.testing.assert_frame_equal(out, base, check_exact=True)
    assert basis == digests
    # Without the basis the features cannot be vouched for: rewritten.
    assert P.merge_published(base, later, games, at)[0].equals(later)
    # 5:05 PM: t_120 of the 7:00 PM games (5:00 PM) has passed since 12:20.
    five = dt.datetime(2026, 3, 12, 17, 5, tzinfo=ET)
    later = P.project(feats, games, sched, report, made_up_lines(games), mdl, five, names=names)
    out, _ = P.merge_published(base, later, games, five, old_basis=digests, new_basis=digests)
    early = out["game_id"].isin(EARLY)
    pd.testing.assert_frame_equal(out[early].reset_index(drop=True),
                                  later[later["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    pd.testing.assert_frame_equal(out[~early].reset_index(drop=True),
                                  base[~base["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    # One game's line moved, another game's feature rows changed.
    g_line, g_feat = games["game_id"].iloc[4], games["game_id"].iloc[5]
    lines = made_up_lines(games)
    lines.loc[lines["game_id"] == g_line, "home_line"] += 0.5
    later = P.project(feats, games, sched, report, lines, mdl, at, names=names)
    changed = {**digests, g_feat: "another history"}
    out, basis = P.merge_published(base, later, games, at, old_basis=digests, new_basis=changed)
    new = out["game_id"].isin({g_line, g_feat})
    pd.testing.assert_frame_equal(out[new].reset_index(drop=True),
                                  later[later["game_id"].isin({g_line, g_feat})].reset_index(drop=True),
                                  check_exact=True)
    pd.testing.assert_frame_equal(out[~new].reset_index(drop=True),
                                  base[~base["game_id"].isin({g_line, g_feat})].reset_index(drop=True),
                                  check_exact=True)
    assert basis == changed


def test_a_same_content_reissue_of_the_report_leaves_the_file_byte_identical(
        tmp_path, monkeypatch, feats, games, sched, report, reissue, mdl, names):
    """The NBA re-titles an unchanged report every 15 minutes, and a new title
    moves report_age_h, which is the clock. A bot run at 12:05 with the 12:00
    PM report and one at 12:20 with the 12:15 PM report (row for row the
    same) leave the projections file byte-identical; a status that did
    change in the re-issue rewrites that game only (its rows then carry the
    new title)."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    t1, t2 = (dt.datetime(2026, 3, 12, 12, 5, tzinfo=ET), dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET))
    cols = [c for c in report.rows.columns if c != "report_ts_et"]
    assert reissue.meta["report_ts_et"] > report.meta["report_ts_et"]
    assert reissue.rows[cols].reset_index(drop=True).equals(report.rows[cols].reset_index(drop=True))
    assert "report_ts_et" not in P.INPUT_COLS and "report_ts_et" in P.STAMPS
    digests = P.feature_digests(feats)
    first = P.project(feats, games, sched, report, made_up_lines(games), mdl, t1, names=names)
    out, basis = P.merge_published(None, first, games, t1, new_basis=digests)
    assert P.write_published(out, basis, DATE)
    before = P.projections_path(DATE).read_bytes()
    second = P.project(feats, games, sched, reissue, made_up_lines(games), mdl, t2, names=names)
    assert (second["report_ts_et"] == pd.Timestamp(reissue.meta["report_ts_et"])).all()
    assert not np.array_equal(second["mu_min"].to_numpy(), first["mu_min"].to_numpy())
    old, old_basis = P.read_published(DATE)
    out, basis = P.merge_published(old, second, games, t2, old_basis=old_basis, new_basis=digests)
    assert not P.write_published(out, basis, DATE, old_basis)
    assert P.projections_path(DATE).read_bytes() == before
    # The re-issue changes one listed player's status: that game is rewritten.
    rows = reissue.rows.copy()
    i = rows.index[(rows["team"] == "DET") & (rows["status"] != "Available")][0]
    rows.loc[i, "status"] = "Available"
    moved = INJ.ParsedReport(reissue.meta, rows, reissue.team_rows)
    third = P.project(feats, games, sched, moved, made_up_lines(games), mdl, t2, names=names)
    out, basis = P.merge_published(old, third, games, t2, old_basis=old_basis, new_basis=digests)
    det = games.loc[games["home"] == "DET", "game_id"].iloc[0]
    assert set(out.loc[out["computed_at"] == pd.Timestamp(t2).tz_convert(UTC), "game_id"]) == {det}
    assert (out.loc[out["game_id"] == det, "report_ts_et"] == pd.Timestamp(reissue.meta["report_ts_et"])).all()
    pd.testing.assert_frame_equal(out[out["game_id"] != det].reset_index(drop=True),
                                  old[old["game_id"] != det].reset_index(drop=True), check_exact=True)
    assert P.write_published(out, basis, DATE, old_basis)


def test_checkpoints_are_the_training_horizons_floored_at_tip(games):
    tip = pd.Timestamp(games["tip_utc"].iloc[0])           # 7:00 PM ET
    want = [dt.datetime(2026, 3, 11, 17, 30, tzinfo=ET), dt.datetime(2026, 3, 12, 11, 0, tzinfo=ET),
            dt.datetime(2026, 3, 12, 17, 0, tzinfo=ET), dt.datetime(2026, 3, 12, 18, 45, tzinfo=ET)]
    assert P.checkpoints(tip) == [w.astimezone(UTC) for w in want]


# ── live_projections: the app's frame ─────────────────────────────────────────
@pytest.fixture
def committed(tmp_path, monkeypatch, base, feats, names):
    """A state dir as the bot leaves it after its 12:20 run: the date's
    features and projections, and the seed's names file."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "state" / "seed")
    C.write_parquet(names, S.SEED_DIR / S.SEED_FILES["names"])
    P.write_features(feats, DATE, P.history_stamp(SEASON, DATE))
    C.write_parquet(base, P.projections_path(DATE))
    return C.read_parquet(P.projections_path(DATE))


def live_args(games, report, mdl, sched, shift=0.0, **kw):
    odds = {(a, h): {"home_line": hl, "total": t} for a, h, hl, t in
            zip(games["away"], games["home"], made_up_lines(games, shift)["home_line"],
                made_up_lines(games, shift)["total"])}
    return {"report": report, "odds": odds, "models": mdl, "schedule": sched, **kw}


def test_live_frame_recomputes_untipped_games_and_serves_tipped_ones_untouched(
        committed, games, report, mdl, sched):
    frame, info = P.live_projections(DATE, EVENING, **live_args(games, report, mdl, sched, shift=0.5))
    assert info.source == "live" and info.stale is None and info.reason is None
    assert info.frozen == sorted(EARLY) and info.unpublished == [] and info.withheld == []
    assert info.model_version == mdl.version and not info.provisional and info.completeness == []
    assert info.report_ts_et == report.meta["report_ts_et"]
    assert info.report_source == "given" and info.lines_source == "given"
    assert info.notes == []
    assert list(frame.columns) == P.COLUMNS
    tipped = frame[frame["game_id"].isin(EARLY)]
    assert tipped["frozen"].all() and not frame.loc[~frame["game_id"].isin(EARLY), "frozen"].any()
    was = committed[committed["game_id"].isin(EARLY)].reset_index(drop=True)
    pd.testing.assert_frame_equal(tipped.drop(columns="frozen").reset_index(drop=True),
                                  was.drop(columns="frozen"), check_exact=True)
    fresh = frame[~frame["game_id"].isin(EARLY)]
    assert (fresh["computed_at"] == pd.Timestamp(EVENING).tz_convert(UTC)).all()
    old = committed.set_index(["game_id", "person_id"])
    new = fresh.set_index(["game_id", "person_id"])
    assert (new["home_line"] - old.loc[new.index, "home_line"] == 0.5).all()


@pytest.mark.parametrize("failure", ["injury feed", "lines feed", "no report", "null skew",
                                     "models"])
def test_live_frame_falls_back_to_the_committed_projections(failure, committed, games, report, mdl,
                                                            sched, feats, monkeypatch):
    kw = live_args(games, report, mdl, sched)
    if failure == "injury feed":
        def down(_sched=None):
            raise RuntimeError("injury: live fetch failed and nothing is cached")
        monkeypatch.setattr(LV, "injury_report", down)
        kw.pop("report")
    elif failure == "lines feed":
        kw["odds"] = LV.Live({}, None, stale=True, source="none", error="ESPN timed out")
    elif failure == "no report":
        kw["report"] = LV.Live(None, None, source="committed")     # nothing at all: unknown
    elif failure == "null skew":
        # A feature training never had null (team pace), null on every row.
        P.write_features(feats.assign(t_pace=np.nan), DATE, P.history_stamp(SEASON, DATE))
    else:
        def broken(dir=M.MODELS_DIR):
            raise M.ModelMismatch("scikit-learn 9.9 at runtime")
        monkeypatch.setattr(M, "load", broken)
        kw.pop("models")
    frame, info = P.live_projections(DATE, EVENING, marked_out={1}, **kw)
    assert info.source == "committed" and info.stale
    assert {"injury feed": "RuntimeError", "lines feed": "lines feed failed",
            "no report": "NoReport", "null skew": "NullSkew: ",
            "models": "ModelMismatch"}[failure] in info.stale
    assert any("Mark-outs" in n for n in info.notes)
    pd.testing.assert_frame_equal(frame.drop(columns="frozen"), committed.drop(columns="frozen"),
                                  check_exact=True)
    assert frame["frozen"].tolist() == frame["game_id"].isin(EARLY).tolist()


def test_a_report_out_of_reach_serves_the_committed_rows_as_stale(committed, games, report, mdl,
                                                                   sched):
    """10:00 PM: one game (CHI at LAL, 10:30 PM) is still to tip, and the
    report the app holds was titled 31 hours ago. That is no report, so the
    page serves the bot's committed rows for every game and says why."""
    night = dt.datetime(2026, 3, 12, 22, 0, tzinfo=ET)
    kw = live_args(games, retitled(report, night - dt.timedelta(hours=31)), mdl, sched)
    frame, info = P.live_projections(DATE, night, **kw)
    assert info.source == "committed" and info.stale.startswith("NoReport: ")
    assert "out of reach for every game" in info.stale
    pd.testing.assert_frame_equal(frame.drop(columns="frozen"), committed.drop(columns="frozen"),
                                  check_exact=True)
    assert set(frame.loc[~frame["frozen"], "game_id"]) == {"0022500960"}


def test_mark_outs_reach_untipped_games_only(committed, games, report, mdl, sched, base):
    plain, _ = P.live_projections(DATE, EVENING, **live_args(games, report, mdl, sched))
    late = set(games["game_id"]) - EARLY
    star = _healthy_starter(plain, late)
    gone = _healthy_starter(base, EARLY)
    marks = {int(star["person_id"]), int(gone["person_id"])}
    frame, info = P.live_projections(DATE, EVENING, marked_out=marks,
                                     **live_args(games, report, mdl, sched))
    assert info.adjusted and any("past tip" in n for n in info.notes)
    tipped = frame[frame["game_id"].isin(EARLY)]
    pd.testing.assert_frame_equal(tipped.reset_index(drop=True),
                                  plain[plain["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    assert not tipped["adjusted"].any() and (tipped["inj_reason"] != X.MARKED_REASON).all()
    team = (frame["game_id"] == star["game_id"]) & (frame["team"] == star["team"])
    assert frame["adjusted"].tolist() == team.tolist()
    me = frame[frame["person_id"] == star["person_id"]].iloc[0]
    assert me["inj_status"] == "Out" and me["inj_reason"] == X.MARKED_REASON
    mates = team & (frame["person_id"] != star["person_id"])
    before = plain.set_index(["game_id", "person_id"]).loc[
        frame.loc[mates].set_index(["game_id", "person_id"]).index, "mu_min"]
    assert frame.loc[mates, "mu_min"].sum() > before.sum() + 3.0


def test_withheld_games_keep_committed_rows_or_say_the_line_is_not_up(committed, games, report, mdl,
                                                                      sched, tmp_path):
    kw = live_args(games, report, mdl, sched)
    last = games.iloc[-1]
    del kw["odds"][(last["away"], last["home"])]
    frame, info = P.live_projections(DATE, EVENING, **kw)
    assert info.withheld == []                 # the bot published it with a line: still shown
    pd.testing.assert_frame_equal(
        frame[frame["game_id"] == last["game_id"]].reset_index(drop=True),
        committed[committed["game_id"] == last["game_id"]].reset_index(drop=True), check_exact=True)
    C.write_parquet(committed[committed["game_id"] != last["game_id"]], P.projections_path(DATE))
    frame, info = P.live_projections(DATE, EVENING, **kw)
    assert info.withheld == [last["game_id"]] and not (frame["game_id"] == last["game_id"]).any()


def test_nothing_committed_is_not_published_yet(tmp_path, monkeypatch, games, report, mdl, sched):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    frame, info = P.live_projections(DATE, EVENING, **live_args(games, report, mdl, sched))
    assert frame.empty and list(frame.columns) == P.COLUMNS
    assert info.reason == P.NOT_PUBLISHED and info.source == "none"


def test_after_every_tip_the_frame_is_the_committed_file(committed, games, report, mdl, sched):
    night = dt.datetime(2026, 3, 13, 1, 0, tzinfo=ET)
    frame, info = P.live_projections(DATE, night, marked_out={1}, **live_args(games, report, mdl, sched))
    assert info.source == "committed" and info.stale is None and frame["frozen"].all()
    pd.testing.assert_frame_equal(frame.drop(columns="frozen"), committed.drop(columns="frozen"),
                                  check_exact=True)


# ── Provisional rows are never frozen ─────────────────────────────────────────
def test_a_provisional_row_past_its_freeze_point_is_served_untouched_but_never_frozen(
        committed, base, games, report, mdl, sched):
    """Spec 4: stale state gets "no freeze", tomorrow is "never freezable".
    Rows the bot published as provisional (yesterday's "tomorrow", or state
    missing a final) that reach tip are shown as published, frozen False,
    and the Info says the frame is provisional."""
    C.write_parquet(base.assign(provisional=True), P.projections_path(DATE))
    was = C.read_parquet(P.projections_path(DATE))
    frame, info = P.live_projections(DATE, EVENING, **live_args(games, report, mdl, sched))
    tipped = frame[frame["game_id"].isin(EARLY)]
    assert len(tipped) and not tipped["frozen"].any() and tipped["provisional"].all()
    pd.testing.assert_frame_equal(tipped.reset_index(drop=True),
                                  was[was["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)
    assert info.frozen == [] and info.provisional
    assert any("provisional projections" in n for n in info.notes)
    # The committed-only fallback (a live failure) flags them the same way.
    kw = live_args(games, report, mdl, sched)
    kw["report"] = LV.Live(None, None, source="committed")
    frame, info = P.live_projections(DATE, EVENING, **kw)
    assert info.source == "committed" and not frame["frozen"].any() and info.provisional
    # The bot carries them past tip as they are: still provisional, never rewritten.
    out, _ = P.merge_published(was, base, games, EVENING)
    pd.testing.assert_frame_equal(out[out["game_id"].isin(EARLY)].reset_index(drop=True),
                                  was[was["game_id"].isin(EARLY)].reset_index(drop=True),
                                  check_exact=True)


def test_a_live_prior_game_makes_only_its_teams_game_provisional(committed, games, report, mdl,
                                                                 sched):
    """Spec 4, per game: CHI at LAL on the 11th still in progress when the
    page asks (a 10:30 PM tip, not ingested) is a gap for the 12th's CHI at
    LAL only. That game's rows are provisional and the Info names the gap,
    per game and in the union; every other game on the date is not
    provisional (they stay freezable)."""
    gap = "0022500941"
    prior = pd.DataFrame([{"game_id": gap, "season": SEASON, "game_type": "RS",
                           "game_date": DATE - dt.timedelta(days=1),
                           "tip_utc": pd.Timestamp("2026-03-12 02:30", tz="UTC"), "status": "live",
                           "home": "LAL", "away": "CHI"}])
    full = pd.concat([prior, sched], ignore_index=True)
    frame, info = P.live_projections(DATE, MORNING, **live_args(games, report, mdl, full))
    assert info.source == "live" and info.stale is None
    lal = "0022500960"
    assert frame.groupby("game_id")["provisional"].all().to_dict() == {
        g: g == lal for g in games["game_id"]}
    assert not frame.loc[frame["game_id"] != lal, "provisional"].any()
    assert info.provisional and info.completeness == [gap]
    assert info.completeness_by_game == {g: ([gap] if g == lal else []) for g in games["game_id"]}
    assert info.provisional_by_game == {g: g == lal for g in games["game_id"]}
    assert P.STALE_FEATURES not in info.notes
    # A game with no rows to read a flag from (withheld: no line, nothing
    # committed for it) keeps its OWN flag, never the date's.
    held = games[games["game_id"] != lal].iloc[0]
    kw = live_args(games, report, mdl, full)
    del kw["odds"][(held["away"], held["home"])]
    C.write_parquet(committed[committed["game_id"] != held["game_id"]], P.projections_path(DATE))
    frame, info = P.live_projections(DATE, MORNING, **kw)
    assert info.withheld == [held["game_id"]] and info.provisional
    assert info.provisional_by_game == {g: g == lal for g in games["game_id"]}


def test_any_exception_building_the_live_frame_serves_the_committed_rows(
        committed, games, report, mdl, sched, monkeypatch, caplog):
    """A bug in the live path (here project() raising a TypeError, which is
    no expected feed failure) costs the live upgrade, never the page: the
    committed projections are served, Info.stale names the exception's type
    and message, and the traceback is logged (logging.exception), never
    swallowed."""
    def broken(*a, **k):
        raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'float'")
    monkeypatch.setattr(P, "project", broken)
    with caplog.at_level("ERROR", logger="bucketvalue"):
        frame, info = P.live_projections(DATE, EVENING, marked_out={1},
                                         **live_args(games, report, mdl, sched))
    assert info.source == "committed"
    assert info.stale == (f"{P.LIVE_FRAME_FAILED}: TypeError: unsupported operand type(s) for -: "
                          "'NoneType' and 'float'")
    assert P.MARKS_NOT_APPLIED in info.notes
    pd.testing.assert_frame_equal(frame.drop(columns="frozen"), committed.drop(columns="frozen"),
                                  check_exact=True)
    assert frame["frozen"].tolist() == frame["game_id"].isin(EARLY).tolist()
    rec = [r for r in caplog.records if r.levelname == "ERROR" and "live projections" in r.getMessage()]
    assert rec and rec[0].exc_info and rec[0].exc_info[0] is TypeError
    # The public stand-in is the same frame.
    same, why = P.committed_projections(DATE, EVENING, why="x")
    pd.testing.assert_frame_equal(same, frame, check_exact=True)
    assert why.source == "committed" and why.stale == "x" and why.frozen == sorted(EARLY)


def test_committed_projections_with_nothing_committed_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    frame, info = P.committed_projections(DATE, EVENING)
    assert frame.empty and list(frame.columns) == P.COLUMNS
    assert info.source == "none" and info.reason == P.NOT_PUBLISHED


def test_an_unreadable_schedule_serves_the_committed_rows_frozen_by_their_own_tips(
        committed, monkeypatch):
    def down():
        raise RuntimeError("schedule: live fetch failed and nothing is cached")
    monkeypatch.setattr(LV, "schedule", down)
    frame, info = P.live_projections(DATE, EVENING, marked_out={1})
    assert info.source == "committed" and "schedule or state unreadable" in info.stale
    assert frame["frozen"].tolist() == frame["game_id"].isin(EARLY).tolist()
    assert info.frozen == sorted(EARLY)
    assert P.MARKS_NOT_APPLIED in info.notes
    pd.testing.assert_frame_equal(frame.drop(columns="frozen"), committed.drop(columns="frozen"),
                                  check_exact=True)


# ── State: data through, the features stamp, name spellings ──────────────────
GAME = "0021900001"                  # NOP at TOR, 2019-10-22 (tests/fixtures/cdnbox)


def _frames(gid: str = GAME) -> dict[str, pd.DataFrame]:
    """One real final game in the bot's partition shape (step_finals)."""
    payload = C.read_json(FIX / "cdnbox" / f"cdn_{gid}.json.gz")["data"]
    part, meta = box.parse_cdn(payload, gid)
    return {"participation": part, "games": pd.DataFrame([meta]),
            "player_games": cdnbox.player_lines(payload), "team_games": cdnbox.team_lines(payload)}


def _write_partitions(season: str, et_date: dt.date, frames: dict, kinds=None) -> None:
    for key in kinds or S.PARTITIONS:
        df = frames[key].assign(season=season, season_type="RS")
        if key == "games":
            df = df.assign(et_date=et_date.isoformat())
        C.write_parquet(df, S.season_dir(season) / f"{S.PARTITIONS[key]}_{et_date.isoformat()}.parquet")


def test_data_through_is_the_latest_ingested_final(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    season = "2019-20"
    assert P.data_through(season) is None and P.history_stamp(season, "2019-10-30")["games"] == 0
    done, half = dt.date(2019, 10, 22), dt.date(2019, 10, 24)
    _write_partitions(season, done, _frames())
    # A later game whose meta landed but whose stat lines did not: not ingested.
    _write_partitions(season, half, {"games": _frames()["games"].assign(game_id="0021900020")},
                      kinds=["games"])
    assert P.data_through(season) == done
    stamp = P.history_stamp(season, dt.date(2019, 10, 30))
    assert stamp["through"] == done.isoformat() and stamp["games"] == 1
    assert P.history_stamp(season, done)["games"] == 0          # strictly before the date


def test_known_names_carry_the_partitions_spellings_before_the_date(tmp_path, monkeypatch):
    """A spelling only the season's partitions carry (the report may use it)
    is known for dates after that game, never for the game's own date."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "seed")
    season, gasol = "2019-20", 201188
    C.write_parquet(pd.DataFrame({"person_id": [gasol], "player_name": ["Marc Gasol"]}),
                    S.SEED_DIR / S.SEED_FILES["names"])
    fr = _frames()
    for k in ("participation", "player_games"):
        fr[k] = fr[k].assign(player_name=fr[k]["player_name"].mask(fr[k]["person_id"] == gasol,
                                                                   "Big Spain"))
    _write_partitions(season, dt.date(2019, 10, 22), fr)
    after = P.known_names(season, dt.date(2019, 10, 23))
    on = P.known_names(season, dt.date(2019, 10, 22))
    assert {"Marc Gasol", "Big Spain"} <= set(after.loc[after["person_id"] == gasol, "player_name"])
    assert set(on.loc[on["person_id"] == gasol, "player_name"]) == {"Marc Gasol"}
    roster = pd.DataFrame({"game_id": ["g"], "team": ["TOR"], "person_id": [gasol],
                           "player_name": ["Marc Gasol"]})
    inj = pd.DataFrame({"game_id": ["g"], "team": ["TOR"], "player_raw": ["Spain, Big"]})
    assert L.resolve_names(inj, roster, after)["person_id"].tolist() == [gasol]
    assert L.resolve_names(inj, roster, on)["person_id"].isna().all()


def test_features_built_before_the_latest_finals_make_the_frame_provisional(
        committed, games, report, mdl, sched):
    """The bot ingested a final before the date but did not rebuild this
    date's features: the live frame says so instead of passing them off as
    current."""
    frame, info = P.live_projections(DATE, MORNING, **live_args(games, report, mdl, sched))
    assert not info.provisional and P.STALE_FEATURES not in info.notes
    _write_partitions(SEASON, DATE - dt.timedelta(days=1), _frames())
    frame, info = P.live_projections(DATE, MORNING, **live_args(games, report, mdl, sched))
    assert info.source == "live" and info.provisional and frame["provisional"].all()
    assert info.provisional_by_game == {g: True for g in games["game_id"]}
    assert P.STALE_FEATURES in info.notes
    assert info.data_through == DATE - dt.timedelta(days=1)
    # Rebuilt (re-stamped) by the bot: current again.
    feats = C.read_parquet(P.features_path(DATE))
    assert P.write_features(feats, DATE, P.history_stamp(SEASON, DATE))
    frame, info = P.live_projections(DATE, MORNING, **live_args(games, report, mdl, sched))
    assert not info.provisional and not frame["provisional"].any()


def test_write_features_restamps_unchanged_rows_and_skips_an_unchanged_file(tmp_path, monkeypatch,
                                                                           feats):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    one = {"through": None, "games": 0, "digest": "a"}
    assert P.write_features(feats, DATE, one)
    before = P.features_path(DATE).read_bytes()
    assert not P.write_features(feats, DATE, one)
    assert P.features_path(DATE).read_bytes() == before
    two = {**one, "games": 1, "digest": "b"}
    assert P.write_features(feats, DATE, two)
    assert C.read_parquet_meta(P.features_path(DATE))[P.HISTORY] == two


def test_short_games_name_every_side_without_a_whole_roster(feats, games):
    assert P.short_games(games, feats) == {}
    g = games.iloc[0]
    rows = feats[~((feats["game_id"] == g["game_id"]) & (feats["team"] == g["home"]))]
    rows = rows[~((rows["game_id"] == games["game_id"].iloc[1])
                  & (rows["team"] == games["away"].iloc[1])
                  & (rows.groupby(["game_id", "team"]).cumcount() >= P.MIN_TEAM_ROWS - 1))]
    short = P.short_games(games, rows)
    assert set(short) == {g["game_id"], games["game_id"].iloc[1]}
    assert short[g["game_id"]].endswith(f"{g['home']} 0")


def test_write_published_keeps_the_basis_with_the_rows(tmp_path, monkeypatch, base):
    """A new basis is written even over unchanged rows (else the next pass
    could not tell a clock-only change); unchanged rows and basis: no write."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    one, two = {"0022500953": "a"}, {"0022500953": "b"}
    assert P.write_published(base, one, DATE)
    assert not P.write_published(base, one, DATE, one)
    assert P.write_published(base, two, DATE, one)
    frame, basis = P.read_published(DATE)
    assert basis == two
    pd.testing.assert_frame_equal(frame, base, check_exact=True)
