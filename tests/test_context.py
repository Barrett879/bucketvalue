"""Live injury context (bucketlib.context) and the app's injury fetch
(live.injury_report).

THE LOAD-BEARING TEST replays real 2025-26 games through the SERVE path: the
report the selection rule picks at the game's training horizon, then
context.context_for, and compares every livectx.CTX_COLS column to the
training table bit for bit. The refactor test re-runs
horizons.snapshot_games on the cached PDFs and compares to the committed
horizon files (2018-19 included: the only season with ALL PLAYERS AVAILABLE
lines and with no_report horizons). Both call production code end to end and
skip when the local caches are absent. Everything else is offline: one real
fixture PDF (2026-03-12 12:00 PM) and hand-built roster rows with known
answers; where serving and training must agree offline, the training side is
production code too (snapshot_games, build_context on the raw rows).
"""
from __future__ import annotations

import datetime as dt
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import context as X
from bucketlib import horizons as HZ
from bucketlib import http as H
from bucketlib import live
from bucketlib import livectx as L
from bucketlib import state as S
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

FIX = Path(__file__).resolve().parent / "fixtures" / "injury"
REPORT_PDF = FIX / "Injury-Report_2026-03-12_12_00PM.pdf"   # titled 12:00 PM ET
QA_PDF = FIX / "Injury-Report_2026-10-07_10_45AM.pdf"       # a QA file: fake games
NOW = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)


def et(*a) -> dt.datetime:
    return dt.datetime(*a, tzinfo=ET)


@pytest.fixture(scope="module")
def report() -> INJ.ParsedReport:
    return INJ.parse_report(REPORT_PDF.read_bytes())


@pytest.fixture(scope="module")
def sched(report) -> pd.DataFrame:
    """Every game the fixture report names, plus SAC@UTA that night, which it
    does not mention. Ids are made up and stable (sorted by date and away)."""
    cols = ["game_date", "game_time_et", "away", "home"]
    keys = pd.concat([report.rows[cols], report.team_rows[cols]]).drop_duplicates(["game_date", "away", "home"])
    keys = pd.concat([keys, pd.DataFrame([{"game_date": dt.date(2026, 3, 12), "game_time_et": dt.time(21, 0),
                                           "away": "SAC", "home": "UTA"}])])
    keys = keys.sort_values(["game_date", "away"]).reset_index(drop=True)
    tips = [pd.Timestamp(dt.datetime.combine(d, t, tzinfo=ET)).tz_convert(UTC)
            for d, t in zip(keys["game_date"], keys["game_time_et"])]
    return pd.DataFrame({"game_id": [f"00225009{i:02d}" for i in range(len(keys))],
                         "game_date": keys["game_date"], "away": keys["away"], "home": keys["home"],
                         "tip_utc": tips})


def game_of(sched, team, date=dt.date(2026, 3, 12)) -> pd.Series:
    g = sched[(sched["game_date"] == date) & ((sched["away"] == team) | (sched["home"] == team))]
    assert len(g) == 1, (team, date)
    return g.iloc[0]


def team_line(report, away, home, team, kind, date=dt.date(2026, 3, 12)) -> pd.DataFrame:
    return report.team_rows.iloc[[0]].assign(game_date=date, away=away, home=home,
                                             matchup=f"{away}@{home}", team=team, kind=kind)


@pytest.fixture(scope="module")
def tweaked(report) -> INJ.ParsedReport:
    """The fixture with WAS removed (so WAS 'filed nobody': reported_none), a
    QA-style row on a game that does not exist (BOS@DET), Enes Kanter listed
    by Brooklyn under the name his roster row no longer carries, a NOT YET
    SUBMITTED line for Chicago, which also has player rows, and 2018-19
    style ALL PLAYERS AVAILABLE lines for Milwaukee (its rows removed) and
    for Philadelphia, which also has player rows. Every team state a report
    can give is in it."""
    rows = report.rows[~report.rows["team"].isin(["WAS", "MIL"])]
    qa = rows.iloc[[0]].assign(away="BOS", home="DET", matchup="BOS@DET", team="BOS",
                               player_raw="Test, Qa", status="Out")
    kanter = rows[rows["team"] == "BKN"].iloc[[0]].assign(
        player_raw="Kanter, Enes", status="Out", reason_raw="Injury/Illness - Knee", reason_cat="injury_illness")
    lines = [team_line(report, "CHI", "LAL", "CHI", "not_yet_submitted"),
             team_line(report, "MIL", "MIA", "MIL", "all_players_available"),
             team_line(report, "PHI", "DET", "PHI", "all_players_available")]
    return INJ.ParsedReport(report.meta, pd.concat([rows, qa, kanter], ignore_index=True),
                            pd.concat([report.team_rows, *lines], ignore_index=True))


# (person_id, team, roster name, expected minutes); the comment is the answer.
PEOPLE = [
    (1, "PHI", "Joel Embiid", 32.0),          # Out, injury
    (2, "PHI", "Tyrese Maxey", 36.0),         # Out, injury
    (3, "PHI", "Paul George", 30.0),          # Out, suspension -> "Suspended"
    (4, "PHI", "Quentin Grimes", 28.0),       # team filed, not on it: known healthy
    (5, "PHI", "Andre Drummond", 18.0),       # Questionable
    (11, "DET", "Isaac Jones", 5.0),          # Out, G League two-way -> "Two-way"
    (12, "DET", "Cade Cunningham", 35.0),     # known healthy
    (21, "DEN", "Nikola Jokic", 35.0),        # DEN NOT YET SUBMITTED -> unknown
    (22, "SAS", "Victor Wembanyama", 31.0),   # Questionable
    (31, "WAS", "Bub Carrington", 25.0),      # WAS filed nobody (tweaked): known healthy
    (32, "ORL", "Franz Wagner", 30.0),        # Out
    (41, "SAC", "Domantas Sabonis", 34.0),    # game not in the report -> unknown
    (42, "UTA", "Lauri Markkanen", 33.0),     # game not in the report -> unknown
    (51, "MIA", "Terry Rozier", 20.0),        # Out, not with team -> "Not with team"
    (61, "OKC", "Nikola Topic", 3.0),         # Out, G League assignment -> "G League"
    (71, "BKN", "Enes Freedom", 10.0),        # 'Kanter, Enes' via known names
    (81, "MIL", "Giannis Antetokounmpo", 34.0),  # MIL all players available: known healthy
]
NAMES = pd.DataFrame({"person_id": [71, 71, 999], "player_name": ["Enes Freedom", "Enes Kanter", "Enes Kanter"]})


@pytest.fixture(scope="module")
def roster_rows(sched) -> pd.DataFrame:
    out = []
    for pid, team, name, em in PEOPLE:
        g = game_of(sched, team)
        out.append({"person_id": pid, "game_id": g["game_id"], "team": team, "tip_utc": g["tip_utc"],
                    "player_name": name, "min_ewm5_s": em, "miss_streak": 0})
    return pd.DataFrame(out)


def games_for(sched, rows) -> pd.DataFrame:
    return sched[sched["game_id"].isin(set(rows["game_id"]))]


def by_pid(df) -> pd.DataFrame:
    return df.set_index("person_id")


# ── report_states: the live horizon ──────────────────────────────────────────
def test_report_states_columns_horizon_and_cutoff(report, sched):
    p, t = X.report_states(sched, sched, report, NOW)
    assert list(p.columns) == X.PLAYER_COLS and list(t.columns) == X.TEAM_COLS
    assert set(t["horizon"]) == {"live"} and set(p["horizon"]) == {"live"}
    assert len(t) == 2 * len(sched)
    assert (pd.to_datetime(t["cutoff_utc"], utc=True) == pd.Timestamp(NOW).tz_convert(UTC)).all()
    assert (pd.to_datetime(p["report_ts_et"], utc=True) == pd.Timestamp(et(2026, 3, 12, 12, 0))).all()


def test_cutoff_is_now_or_tip_minus_five_minutes(report, sched):
    now = et(2026, 3, 12, 18, 58)                      # PHI@DET tips at 7:00 PM
    _, t = X.report_states(sched, sched, report, now)
    cut = t.set_index(["game_id", "team"])["cutoff_utc"]
    phi, chi = game_of(sched, "PHI"), game_of(sched, "CHI")
    assert cut[(phi["game_id"], "PHI")] == phi["tip_utc"] - HZ.TIP_FLOOR
    assert cut[(chi["game_id"], "CHI")] == pd.Timestamp(now).tz_convert(UTC)


def test_a_report_titled_after_a_games_cutoff_is_never_used_for_it(report, sched):
    late = INJ.ParsedReport({**report.meta, "report_ts_et": et(2026, 3, 12, 18, 58)},
                            report.rows, report.team_rows)
    p, t = X.report_states(sched, sched, late, et(2026, 3, 12, 19, 10))
    phi, chi = game_of(sched, "PHI"), game_of(sched, "CHI")
    st = t.set_index(["game_id", "team"])["state"]
    # PHI@DET tipped at 7:00: the 6:58 report is post-cutoff news for it.
    assert st[(phi["game_id"], "PHI")] == "no_report" and st[(phi["game_id"], "DET")] == "no_report"
    assert not (p["game_id"] == phi["game_id"]).any()
    assert st[(chi["game_id"], "CHI")] == "listed"     # 10:30 PM tip: cutoff is now
    assert (pd.to_datetime(p["report_ts_et"], utc=True)
            <= pd.to_datetime(p["cutoff_utc"], utc=True)).all()


def test_no_report_means_every_team_unknown(sched, roster_rows):
    p, t = X.report_states(sched, sched, None, NOW)
    assert p.empty and list(p.columns) == X.PLAYER_COLS
    assert set(t["state"]) == {"no_report"} and len(t) == 2 * len(sched)
    c = X.context_for(roster_rows, games_for(sched, roster_rows), sched, None, NOW, NAMES)
    assert (c["inj_unknown"] == 1).all() and (c["inj_not_listed"] == 0).all()
    assert (c["tm_unknown_team"] == 1).all()
    assert set(X.display_status(c)["status"]) == {X.NO_REPORT}


def held_by(report, filename: str):
    """A training getter whose archive holds ONE file: `report` at `filename`."""
    return lambda slot: report if slot.filename == filename else None


def retitled(report, ts: dt.datetime) -> INJ.ParsedReport:
    """The same report under another title time (its rows carry it too)."""
    return INJ.ParsedReport({**report.meta, "report_ts_et": ts}, report.rows.assign(report_ts_et=ts),
                            report.team_rows.assign(report_ts_et=ts))


def _agree(games, sched, report, getter, horizons=tuple(HZ.HORIZONS)) -> pd.DataFrame:
    """Training (snapshot_games: the selection rule walks an archive) and
    serving (report_states: `report` in hand, now = the horizon's cutoff)
    give the same rows for every game and horizon. Returns the snapshot's
    team rows."""
    sp, stt = HZ.snapshot_games(games, sched, getter=getter, horizons=horizons)
    cols = ["team", "player_raw", "status", "reason_cat", "reason_raw"]
    for (gid, h), tg in stt.groupby(["game_id", "horizon"], sort=False):
        g = games[games["game_id"] == gid]
        now = HZ.cutoff_for(h, pd.Timestamp(g["tip_utc"].iloc[0]).to_pydatetime())
        lp, lt = X.report_states(g, sched, report, now)
        assert lt["state"].tolist() == tg["state"].tolist(), (gid, h)
        assert lt["team"].tolist() == tg["team"].tolist()
        assert _times_equal(lt["cutoff_utc"], tg["cutoff_utc"])
        assert _times_equal(lt["report_ts_et"], tg["report_ts_et"])
        sg = sp[(sp["game_id"] == gid) & (sp["horizon"] == h)] if len(sp) else lp.iloc[0:0]
        assert lp[cols].reset_index(drop=True).equals(sg[cols].reset_index(drop=True)), (gid, h)
    return stt


def test_report_states_and_snapshot_games_agree_at_every_horizon(tweaked, sched):
    """Every game at every training horizon, the out-of-reach ones included:
    the 3/13 tip horizons and a 3/14 game are more than 30 hours after the
    3/12 12:00 PM report, which training records as no_report."""
    extra = pd.DataFrame([{"game_id": "0022509990", "game_date": dt.date(2026, 3, 14), "away": "BOS",
                           "home": "NYK", "tip_utc": pd.Timestamp(et(2026, 3, 14, 19, 30)).tz_convert(UTC)}])
    sch = pd.concat([sched, extra], ignore_index=True)
    stt = _agree(sch, sch, tweaked, held_by(tweaked, REPORT_PDF.name))
    assert set(stt["state"]) == {"listed", "not_yet_submitted", "all_players_available", "reported_none",
                                 "game_not_in_report", "no_report"}


def test_a_report_titled_exactly_at_the_cutoff_is_not_used(tweaked, sched):
    """The selection rule takes a title STRICTLY before the cutoff
    (injury.report_before) and serving asks that rule: a report titled
    exactly at a game's cutoff is no_report for it in both, one titled a
    minute earlier is used in both. PHI@DET tips at 7:00 PM, t_15 = 6:45."""
    phi = game_of(sched, "PHI")
    games = sched[sched["game_id"] == phi["game_id"]]
    assert HZ.cutoff_for("t_15", phi["tip_utc"].to_pydatetime()) == et(2026, 3, 12, 18, 45)
    for title, filename, state in ((et(2026, 3, 12, 18, 45), "Injury-Report_2026-03-12_06_45PM.pdf", "no_report"),
                                   (et(2026, 3, 12, 18, 44), "Injury-Report_2026-03-12_06_30PM.pdf", "listed")):
        r = retitled(tweaked, title)
        stt = _agree(games, sched, r, held_by(r, filename), horizons=("t_15",))
        assert set(stt["state"]) == {state}, title


def test_a_report_more_than_30_hours_old_is_no_report_as_in_training(tweaked, sched):
    """The stale paths (memory, committed) can hold a report from two days
    ago. Training never encodes one (the selection rule reaches back 30
    hours): no_report with no report age, never 'game_not_in_report' aged
    48 hours. The report is titled 3/12 12:00 PM; the reach ends 3/13 6:00 PM."""
    tip = pd.Timestamp(et(2026, 3, 14, 19, 0)).tz_convert(UTC)
    g = pd.DataFrame([{"game_id": "0022509991", "game_date": dt.date(2026, 3, 14), "away": "BOS",
                       "home": "NYK", "tip_utc": tip}])
    sch = pd.concat([sched, g], ignore_index=True)
    for now, state in ((et(2026, 3, 13, 17, 59), "game_not_in_report"),
                       (et(2026, 3, 13, 18, 1), "no_report"), (et(2026, 3, 14, 12, 0), "no_report")):
        _, t = X.report_states(g, sch, tweaked, now)
        assert set(t["state"]) == {state}, now
        training = HZ.report_at(now.astimezone(UTC), held_by(tweaked, REPORT_PDF.name))
        assert (training is not None) == (state != "no_report"), now
    rows = pd.DataFrame([{"person_id": 1, "game_id": "0022509991", "team": "BOS", "tip_utc": tip,
                          "player_name": "Jayson Tatum", "min_ewm5_s": 30.0}])
    c = X.context_for(rows, g, sch, tweaked, et(2026, 3, 14, 12, 0), None)
    assert c["inj_unknown"].iloc[0] == 1 and np.isnan(c["report_age_h"].iloc[0])


def test_team_state_precedence(tweaked, sched):
    """game_states' order, pinned: NOT YET SUBMITTED outranks player rows
    (CHI), player rows outrank ALL PLAYERS AVAILABLE (PHI), which outranks
    'filed nobody' (MIL against WAS)."""
    p, t = X.report_states(sched, sched, tweaked, NOW)
    st = t.set_index(["game_id", "team"])["state"]

    def state(team):
        return st[(game_of(sched, team)["game_id"], team)]
    assert state("CHI") == "not_yet_submitted" and state("LAL") == "listed"
    assert state("PHI") == "listed" and state("MIL") == "all_players_available"
    assert state("WAS") == "reported_none"
    chi = game_of(sched, "CHI")["game_id"]
    assert not ((p["game_id"] == chi) & (p["team"] == "CHI")).any()


# ── context_for and the status rule ──────────────────────────────────────────
def test_status_rule_unknown_is_never_healthy(tweaked, sched, roster_rows):
    c = by_pid(X.context_for(roster_rows, games_for(sched, roster_rows), sched, tweaked, NOW, NAMES))
    # Not yet submitted, and a game the report does not mention: UNKNOWN.
    for pid in (21, 41, 42):
        assert c.loc[pid, "inj_unknown"] == 1 and c.loc[pid, "inj_not_listed"] == 0, pid
        assert c.loc[pid, "tm_unknown_team"] == 1
    assert c.loc[21, "team_state"] == "not_yet_submitted"
    assert c.loc[41, "team_state"] == "game_not_in_report"
    # The team filed (with players, with nobody, or all available): KNOWN not listed.
    for pid in (4, 12, 31, 81):
        assert c.loc[pid, "inj_not_listed"] == 1 and c.loc[pid, "inj_unknown"] == 0, pid
    assert c.loc[31, "team_state"] == "reported_none"
    assert c.loc[81, "team_state"] == "all_players_available"
    assert c.loc[1, "inj_out"] == 1 and c.loc[5, "inj_questionable"] == 1 and c.loc[22, "inj_questionable"] == 1
    assert (c["inj_pre_archive"] == 0).all()
    # Pre-game teammate minutes: Embiid 32 + Maxey 36 + George 30 out for PHI.
    assert c.loc[4, "tm_out_min"] == 98.0 and c.loc[4, "tm_q_min"] == 18.0
    assert abs(c.loc[4, "report_age_h"] - 20 / 60) < 1e-12


# The projection frame's identity and availability columns that badges read
# (docs/serving_design.md section 3; p_play is a model output, not a badge).
PROJ_COLS = ["person_id", "game_id", "inj_status", "inj_reason", "team_state"]


def test_display_status_badges(tweaked, sched, roster_rows):
    c = X.context_for(roster_rows, games_for(sched, roster_rows), sched, tweaked, NOW, NAMES)
    assert X.DISPLAY_COLS == PROJ_COLS[2:]
    d = by_pid(X.display_status(c[PROJ_COLS]))
    assert list(d.reset_index().columns) == ["person_id", "game_id", "status", "reason"]
    want = {1: "Out", 3: "Suspended", 5: "Questionable", 11: "Two-way", 22: "Questionable",
            32: "Out", 51: "Not with team", 61: "G League", 71: "Out",
            21: X.NOT_FILED, 41: X.NO_REPORT, 42: X.NO_REPORT}
    assert {k: d.loc[k, "status"] for k in want} == want
    for pid in (4, 12, 31, 81):                 # known healthy: no badge
        assert d.loc[pid, "status"] is None and d.loc[pid, "reason"] is None
    assert d.loc[1, "reason"].startswith("Injury/Illness")
    assert d.loc[21, "reason"] is None
    # The display status is the context's own encoding, never a second one.
    for s in L.STATUSES:
        assert ((c["inj_status"] == s) == (c[f"inj_{s.lower()}"] == 1)).all()


def test_display_status_reads_only_the_projection_frame(tweaked, sched, roster_rows, tmp_path):
    """Tipped games and the stale fallback come from the committed
    projections untouched, so badges must come from the projection frame's
    own columns after a parquet round trip, and a missing badge must mean
    exactly what the encoding calls known healthy (inj_not_listed)."""
    games = games_for(sched, roster_rows)
    seen = set()
    for rep, marked in ((tweaked, None), (tweaked, {4, 41}), (None, None), (None, {41})):
        c = X.context_for(roster_rows, games, sched, rep, NOW, NAMES, marked_out=marked)
        path = tmp_path / "projections.parquet"
        C.write_parquet(c[PROJ_COLS], path)
        d = X.display_status(C.read_parquet(path))
        assert d["status"].isna().tolist() == (c["inj_not_listed"] == 1).tolist()
        assert d.equals(X.display_status(c[PROJ_COLS]))
        seen |= set(d["status"].dropna())
    assert {X.NOT_FILED, X.NO_REPORT, "Two-way", "Suspended"} <= seen
    # A state the frame lost is unknown, never healthy.
    c = X.context_for(roster_rows, games, sched, tweaked, NOW, NAMES)[PROJ_COLS]
    gone = X.display_status(c.assign(team_state=None))
    assert set(gone.loc[c["inj_status"].isna().to_numpy(), "status"]) == {X.NO_REPORT}


def test_a_duplicate_listing_reads_as_build_context_reads_it(tweaked, sched, roster_rows):
    """A player listed twice for one game: the serve path must encode the
    listing build_context encodes on the raw rows (the training path) and
    display the reason of that same listing."""
    again = tweaked.rows[tweaked.rows["player_raw"] == "Embiid, Joel"].assign(
        status="Questionable", reason_raw="Personal - Family Matter", reason_cat="personal")
    assert len(again) == 1
    dup = INJ.ParsedReport(tweaked.meta, pd.concat([tweaked.rows, again], ignore_index=True),
                           tweaked.team_rows)
    games = games_for(sched, roster_rows)
    got = X.context_for(roster_rows, games, sched, dup, NOW, NAMES)
    players, teams = X.report_states(games, sched, dup, NOW)
    roster = roster_rows[["game_id", "team", "person_id", "player_name"]]
    training = L.build_context(roster_rows, L.resolve_names(players, roster, NAMES), teams)
    for col in L.CTX_COLS:
        assert np.array_equal(got[col].to_numpy(float), training[col].to_numpy(float), equal_nan=True), col
    e = by_pid(got).loc[1]
    assert e["inj_questionable"] == 1 and e["inj_out"] == 0
    assert e["inj_status"] == "Questionable" and e["inj_reason"] == "Personal - Family Matter"


def test_rows_on_no_game_in_games_raise(tweaked, sched, roster_rows):
    """A row whose game is missing from `games` would read as an unknown team
    without a word; the caller's slip is loud instead."""
    games = games_for(sched, roster_rows)
    phi = game_of(sched, "PHI")["game_id"]
    with pytest.raises(ValueError, match="on no game"):
        X.context_for(roster_rows, games[games["game_id"] != phi], sched, tweaked, NOW, NAMES)
    with pytest.raises(ValueError, match="on no game"):
        X.context_for(roster_rows.assign(team=roster_rows["team"].replace({"PHI": "BOS"})),
                      games, sched, tweaked, NOW, NAMES)


def test_names_resolve_through_every_spelling(tweaked, sched, roster_rows):
    games = games_for(sched, roster_rows)
    without = by_pid(X.context_for(roster_rows, games, sched, tweaked, NOW, None))
    with_ = by_pid(X.context_for(roster_rows, games, sched, tweaked, NOW, NAMES))
    assert without.loc[71, "inj_out"] == 0 and with_.loc[71, "inj_out"] == 1
    assert with_.loc[71, "inj_reason"] == "Injury/Illness - Knee"


def test_rows_off_the_schedule_never_reach_a_game(tweaked, sched):
    p, _ = X.report_states(sched, sched, tweaked, NOW)
    assert "Test, Qa" not in set(p["player_raw"])
    assert set(p["game_id"]) <= set(sched["game_id"])


def test_mark_out_is_status_out_at_age_zero(tweaked, sched, roster_rows):
    games = games_for(sched, roster_rows)
    base = by_pid(X.context_for(roster_rows, games, sched, tweaked, NOW, NAMES))
    raw = X.context_for(roster_rows, games, sched, tweaked, NOW, NAMES, marked_out={4, 41})
    c = by_pid(raw)
    assert c.loc[4, "inj_out"] == 1 and c.loc[4, "report_age_h"] == 0.0
    assert c.loc[5, "tm_out_min"] == base.loc[5, "tm_out_min"] + 28.0
    d = by_pid(X.display_status(raw))
    assert d.loc[4, "status"] == "Out" and d.loc[4, "reason"] == X.MARKED_REASON
    assert d.loc[41, "status"] == "Out"          # marked out on an unknown team: still Out
    assert c.loc[41, "team_state"] == "game_not_in_report"


def test_context_rows_keep_their_order_and_count(tweaked, sched, roster_rows):
    shuffled = roster_rows.sample(frac=1.0, random_state=7).reset_index(drop=True)
    c = X.context_for(shuffled, games_for(sched, shuffled), sched, tweaked, NOW, NAMES)
    assert c["person_id"].tolist() == shuffled["person_id"].tolist()
    assert set(L.CTX_COLS) <= set(c.columns) and set(X.DISPLAY_COLS) <= set(c.columns)


# ── Committed fallback ───────────────────────────────────────────────────────
@pytest.fixture
def committed_paths(tmp_path, monkeypatch):
    """bucketlib.state owns state/: pointing its STATE_DIR at a temp dir moves
    the committed pair with it."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    return X.committed_paths()


def _same_states(a, b) -> None:
    (ap, at), (bp, bt) = a, b
    for x, y in ((ap, bp), (at, bt)):
        assert len(x) == len(y)
        for col in x.columns:
            if col in ("cutoff_utc", "report_ts_et"):
                assert _times_equal(x[col], y[col]), col
            else:
                assert x[col].tolist() == y[col].tolist(), col


# Clocks across the report's life: before every cutoff, after PHI@DET's
# (7:00 PM tip), the next morning, and past the 30-hour reach.
CLOCKS = [NOW, et(2026, 3, 12, 19, 10), et(2026, 3, 13, 9, 0), et(2026, 3, 13, 19, 0),
          et(2026, 3, 14, 12, 0)]


def test_committed_files_round_trip_for_any_games_and_any_clock(tweaked, sched, committed_paths):
    """The pair depends on the report alone, so whatever games and clock the
    app later asks about, the read-back gives the live report's states:
    every game of the slate, not only the ones some earlier call named."""
    assert X.read_committed() == (None, None)
    assert X.write_committed(tweaked, sched) is True
    rep, fetched_at = X.read_committed()
    assert isinstance(rep, X.CommittedReport) and fetched_at is not None
    assert rep.meta["report_ts_et"] == et(2026, 3, 12, 12, 0)
    phi = game_of(sched, "PHI")["game_id"]
    slates = [sched, sched[sched["game_date"] == dt.date(2026, 3, 12)],
              sched[sched["game_date"] == dt.date(2026, 3, 13)], sched[sched["game_id"] == phi]]
    for games in slates:
        for now in CLOCKS:
            _same_states(X.report_states(games, sched, rep, now), X.report_states(games, sched, tweaked, now))
    teams = C.read_parquet(committed_paths[1])
    assert list(teams.columns) == X.COMMITTED_TEAM_COLS
    assert set(teams["state"]) == {"listed", "not_yet_submitted", "all_players_available", "reported_none"}
    assert (pd.to_datetime(teams["report_ts_et"], utc=True) == pd.Timestamp(et(2026, 3, 12, 12, 0))).all()
    assert len(teams) == 2 * teams["game_id"].nunique() == 2 * (len(sched) - 1)   # SAC@UTA is not named
    # No game list and no clock in the files: a later run with the same
    # report is not a commit.
    assert X.write_committed(tweaked, sched) is False


def test_committed_pair_comes_from_one_report(report, sched, committed_paths):
    """Each file is written on its own (if_changed), so the report time lives
    in BOTH files' rows and read_committed refuses a pair from two reports:
    a newer report's rows under an older report's time would be gated and
    aged wrong."""
    a = retitled(report, et(2026, 3, 12, 19, 30))
    rows_b = report.rows.copy()
    rows_b.loc[rows_b.index[0], "status"] = "Available"
    b = retitled(INJ.ParsedReport(report.meta, rows_b, report.team_rows), et(2026, 3, 12, 20, 0))
    assert X.write_committed(a, sched) and X.write_committed(b, sched)
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 12, 20, 0)
    assert (pd.to_datetime(rep.rows["report_ts_et"], utc=True) == pd.Timestamp(et(2026, 3, 12, 20, 0))).all()
    assert rep.rows.iloc[0]["status"] == "Available"
    # A run that died between the two writes: A's players beside B's teams.
    C.write_parquet(X.committed_frames(a, sched)[0], committed_paths[0])
    with pytest.raises(ValueError, match="disagree"):
        X.read_committed()
    # A report with no player rows (every team not yet submitted) leaves an
    # empty players file that has no time to disagree with.
    early = retitled(INJ.ParsedReport(report.meta, report.rows.iloc[0:0], report.team_rows),
                     et(2026, 3, 13, 9, 0))
    assert X.write_committed(early, sched)
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 13, 9, 0) and rep.rows.empty
    _same_states(X.report_states(sched, sched, rep, et(2026, 3, 13, 9, 30)),
                 X.report_states(sched, sched, early, et(2026, 3, 13, 9, 30)))
    committed_paths[1].unlink()
    with pytest.raises(ValueError, match="missing"):
        X.read_committed()


def test_a_same_content_reissue_is_not_a_new_committed_pair(report, sched, committed_paths):
    """The NBA re-titles an unchanged report every 15 minutes, and the title is
    on every row of both files, so writing each re-issue made both files a
    new commit on every bot run. The real 12:15 PM report is row for row the
    12:00 PM one: not written, byte for byte, and the pair keeps the 12:00
    title. Past COMMITTED_RESTAMP the same content is written once under the
    new title (the fallback's report_age_h stays bounded, far inside the
    30-hour reach); a changed row is written at once."""
    reissue = INJ.parse_report((FIX / "Injury-Report_2026-03-12_12_15PM.pdf").read_bytes())
    assert reissue.meta["report_ts_et"] == et(2026, 3, 12, 12, 15)
    assert X.write_committed(report, sched) is True
    before = [p.read_bytes() for p in committed_paths]
    assert X.write_committed(reissue, sched) is False
    assert [p.read_bytes() for p in committed_paths] == before
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 12, 12, 0)
    # Still the same content, the committed title now more than 3 hours behind.
    assert X.COMMITTED_RESTAMP == dt.timedelta(hours=3)
    assert X.write_committed(retitled(reissue, et(2026, 3, 12, 15, 0)), sched) is False
    assert X.write_committed(retitled(reissue, et(2026, 3, 12, 15, 1)), sched) is True
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 12, 15, 1)
    assert (pd.to_datetime(rep.rows["report_ts_et"], utc=True) == pd.Timestamp(et(2026, 3, 12, 15, 1))).all()
    assert X.write_committed(retitled(reissue, et(2026, 3, 12, 15, 15)), sched) is False
    # One status changed: written at once, both files, under the new title.
    rows = reissue.rows.copy()
    rows.loc[rows.index[0], "status"] = "Available"
    changed = retitled(INJ.ParsedReport(reissue.meta, rows, reissue.team_rows), et(2026, 3, 12, 15, 30))
    assert X.write_committed(changed, sched) is True
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 12, 15, 30) and rep.rows.iloc[0]["status"] == "Available"


def test_a_report_with_no_player_rows_restamps_both_files(report, sched, committed_paths):
    """An empty players file hashes the same under any title, so the writer
    rewrites BOTH files whenever it writes: the pair's two titles never
    split, and the next same-content re-issue is again no write."""
    empty = INJ.ParsedReport(report.meta, report.rows.iloc[0:0], report.team_rows)
    assert X.write_committed(retitled(empty, et(2026, 3, 13, 9, 0)), sched) is True
    assert X.write_committed(retitled(empty, et(2026, 3, 13, 12, 30)), sched) is True
    metas = [C.read_parquet_meta(p) for p in committed_paths]
    assert {m[X.TITLE] for m in metas} == {pd.Timestamp(et(2026, 3, 13, 12, 30)).isoformat()}
    assert X.write_committed(retitled(empty, et(2026, 3, 13, 12, 45)), sched) is False
    rep, _ = X.read_committed()
    assert rep.meta["report_ts_et"] == et(2026, 3, 13, 12, 30) and rep.rows.empty


def test_committed_paths_are_the_state_modules_paths(tmp_path, monkeypatch):
    """One writer per path: the pair is where docs/serving_design.md section 1
    puts it, through bucketlib.state (which owns state/), resolved at call time."""
    root = C.REPO_ROOT / "state" / "current"
    assert X.committed_paths() == (root / "injury_players_v1.parquet", root / "injury_teams_v1.parquet")
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    assert X.committed_paths() == (S.current_path("injury_players"), S.current_path("injury_teams"))
    assert X.committed_paths()[0] == tmp_path / "current" / "injury_players_v1.parquet"


def test_committed_files_refuse_none_a_qa_file_and_a_listed_team_without_rows(tweaked, sched, committed_paths):
    """Stale beats empty at the writer: neither None nor a report naming no
    real NBA game (a QA file in the newest slot) replaces a good pair."""
    with pytest.raises(ValueError, match="stale beats empty"):
        X.write_committed(None, sched)
    assert X.write_committed(tweaked, sched)
    good = [C.read_parquet(path) for path in committed_paths]
    qa = INJ.parse_report(QA_PDF.read_bytes())
    assert not HZ.real_report(sched)(qa)
    with pytest.raises(ValueError, match="stale beats empty"):
        X.write_committed(qa, sched)
    # Rows that carry another report time than the title would be a pair
    # read_committed refuses: refused at the writer instead.
    with pytest.raises(ValueError, match="another report time"):
        X.write_committed(INJ.ParsedReport({**tweaked.meta, "report_ts_et": et(2026, 3, 12, 12, 15)},
                                           tweaked.rows, tweaked.team_rows), sched)
    for path, before in zip(committed_paths, good):
        assert C.read_parquet(path).equals(before)
    players = good[0]
    C.write_parquet(players[players["team"] != "PHI"], committed_paths[0])
    with pytest.raises(ValueError, match="listed teams without rows"):
        X.read_committed()


# ── live.injury_report: fresh, memory, committed, circuit breaker ────────────
@pytest.fixture
def live_env(monkeypatch, committed_paths, sched):
    clock = [1_000_000.0]
    monkeypatch.setattr(live, "time", types.SimpleNamespace(time=lambda: clock[0],
                                                            monotonic=lambda: clock[0]))
    monkeypatch.setattr(live, "_good", {})
    monkeypatch.setattr(live, "_fail", {})
    now = [NOW]
    monkeypatch.setattr(live, "now_et", lambda: now[0])
    env = types.SimpleNamespace(clock=clock, now=now, calls=[], timeouts=[], mode="ok", lag=0.0,
                                files={REPORT_PDF.name: REPORT_PDF.read_bytes()}, sched=sched)

    def fetcher(url):
        env.calls.append(url.rsplit("/", 1)[-1])
        clock[0] += env.lag                       # a slow host spends the walk's budget
        if env.mode == "down":
            raise H.Blocked(url, "test: host down")
        name = url.rsplit("/", 1)[-1]
        if env.mode == "ok" and name in env.files:
            return env.files[name]
        raise H.Absent(url, "test: not published")

    def live_store(timeout=8.0):
        env.timeouts.append(timeout)
        return INJ.ReportStore(fetcher=fetcher, persist=False)
    monkeypatch.setattr(INJ, "live_store", live_store)
    return env


def test_injury_report_fresh(live_env):
    v = live.injury_report(live_env.sched)
    assert v.source == "live" and not v.stale
    assert v.value.meta["report_ts_et"] == et(2026, 3, 12, 12, 0)
    assert live_env.calls == ["Injury-Report_2026-03-12_12_15PM.pdf", REPORT_PDF.name]
    assert live_env.timeouts == [live.INJURY_TIMEOUT]
    again = live.injury_report(live_env.sched)              # inside the TTL: no fetch
    assert again.source == "memory" and not again.stale and len(live_env.calls) == 2


def test_injury_report_stale_from_memory_and_the_circuit_breaker(live_env):
    good = live.injury_report(live_env.sched).value
    live_env.clock[0] += live.INJURY_TTL + 1
    live_env.mode = "down"
    v = live.injury_report(live_env.sched)
    assert v.stale and v.source == "memory" and v.value is good and "Blocked" in v.error
    n = len(live_env.calls)
    live_env.clock[0] += live.INJURY_FAIL_TTL - 1           # failure remembered: no refetch
    v = live.injury_report(live_env.sched)
    assert len(live_env.calls) == n and v.stale and v.value is good
    live_env.clock[0] += 2                                  # breaker expired: tries again
    live_env.mode = "ok"
    v = live.injury_report(live_env.sched)
    assert len(live_env.calls) > n and v.source == "live" and not v.stale


def test_injury_report_falls_back_to_the_committed_report(live_env, tweaked):
    X.write_committed(tweaked, live_env.sched)
    live_env.mode = "down"
    v = live.injury_report(live_env.sched)
    assert v.stale and v.source == "committed" and isinstance(v.value, X.CommittedReport)
    s = live_env.sched
    _same_states(X.report_states(s, s, v.value, NOW), X.report_states(s, s, tweaked, NOW))


def test_a_committed_pair_it_cannot_trust_is_a_runtime_error(live_env, tweaked):
    """_cached's documented failure type, so a page that falls back to the
    committed projections on RuntimeError still does; both failures named."""
    X.write_committed(tweaked, live_env.sched)
    players = C.read_parquet(X.committed_paths()[0])
    C.write_parquet(players[players["team"] != "PHI"], X.committed_paths()[0])
    live_env.mode = "down"
    with pytest.raises(RuntimeError, match="Blocked.*committed report is unusable.*listed teams without rows"):
        live.injury_report(live_env.sched)


def test_the_walk_stops_when_its_time_budget_is_spent(live_env):
    """Every slot answers slowly (absent, 0.6 of the budget each): the third
    fetch would start past INJURY_BUDGET, so it is never made, and the
    attempt is a failure, not a walk of all eight slots."""
    live_env.mode, live_env.lag = "empty", 0.6 * live.INJURY_BUDGET
    with pytest.raises(RuntimeError, match="budget"):
        live.injury_report(live_env.sched)
    assert len(live_env.calls) == 2


def test_injury_report_cold_failure_raises(live_env):
    live_env.mode = "down"
    with pytest.raises(RuntimeError, match="nothing is cached"):
        live.injury_report(live_env.sched)


def test_no_report_in_the_walk_is_a_failure_not_an_empty_report(live_env):
    live_env.mode = "empty"
    with pytest.raises(RuntimeError, match="no real injury report"):
        live.injury_report(live_env.sched)
    assert len(set(live_env.calls)) == live.INJURY_SLOTS     # 8 slot times, then stop


def test_injury_report_walks_past_a_qa_file(live_env):
    """A QA file in the newest slot (fake games, title 10:58) is skipped by the
    same real-report check the training snapshots used."""
    live_env.now[0] = et(2026, 10, 7, 11, 5)
    live_env.files = {"Injury-Report_2026-10-07_10_45AM.pdf": QA_PDF.read_bytes(),
                      "Injury-Report_2026-10-07_10_30AM.pdf": REPORT_PDF.read_bytes()}
    v = live.injury_report(live_env.sched)
    assert v.source == "live" and v.value.meta["report_ts_et"] == et(2026, 3, 12, 12, 0)
    assert "Injury-Report_2026-10-07_10_45AM.pdf" in live_env.calls


# ── Real data (local caches; skipped when absent) ────────────────────────────
TRAIN_TABLE = C.CACHE_DIR / "train_table_full_v1.parquet"


def _season_schedule(season: str) -> pd.DataFrame:
    s = C.read_parquet(C.CACHE_DIR / f"schedule_{season}_v1.parquet")
    if s is None:
        pytest.skip(f"schedule_{season}_v1 not cached")
    s = s.copy()
    s["game_date"] = pd.to_datetime(s["game_date"]).dt.date
    return s


# A cached PDF that reads short (an iCloud-evicted Desktop file reads as
# empty until it downloads again) looks like an absent slot offline, and the
# walk would quietly take an older report. Say so instead of a mismatch.
_EVICTED = "cached injury PDFs read short (evicted or truncated): re-download before trusting this"


def _cached_store() -> INJ.ReportStore:
    """The PDFs default_store() reads, opened offline: a test never fetches."""
    root = C.CACHE_DIR / "raw" / "injury"
    if not root.is_dir():
        pytest.skip("injury PDF cache absent")
    return INJ.ReportStore(root, offline=True)


def _times_equal(a: pd.Series, b: pd.Series) -> bool:
    a, b = pd.to_datetime(a, utc=True).reset_index(drop=True), pd.to_datetime(b, utc=True).reset_index(drop=True)
    return bool(((a == b) | (a.isna() & b.isna())).all())


RARE_PER_STATE = 8      # games per rare state (each one costs its date's PDF parses)


@pytest.mark.parametrize("season,min_games", [("2025-26", 150), ("2019-20", 150), ("2018-19", 0)])
def test_refactored_snapshot_games_reproduces_the_cached_horizons(season, min_games):
    """2018-19 is the only season with ALL PLAYERS AVAILABLE lines and with
    no_report horizons, 2019-20 the one with games not in the report; 2018-19
    rides along for those rare states only (its 3-per-day PDFs parse slowly)."""
    ref_t = C.read_parquet(C.CACHE_DIR / f"injury_horizon_teams_{season}_v1.parquet")
    ref_p = C.read_parquet(C.CACHE_DIR / f"injury_horizons_{season}_v1.parquet")
    if ref_t is None or ref_p is None:
        pytest.skip(f"injury horizons for {season} not cached")
    sched = _season_schedule(season)
    store = _cached_store()
    games = sched[sched["game_id"].isin(set(ref_t["game_id"]))]
    dates = sorted(games["game_date"].unique())
    # Whole dates (reports are shared within a date), spread over the season.
    picked, n = [], 0
    for d in dates[::max(1, len(dates) // 30)]:
        if n >= min_games:
            break
        picked.append(d)
        n += int((games["game_date"] == d).sum())
    # Plus games holding each rare state (filed nobody, all available, game
    # not in the report, no report, unparsed), spread over the season: a
    # whole-date sample can miss all of them, and they are where the state
    # precedence and the unknowns a regression would quietly turn healthy live.
    rare = set()
    for _, ids in ref_t.loc[~ref_t["state"].isin(["listed", "not_yet_submitted"])].groupby("state")["game_id"]:
        u = sorted(set(ids))
        rare |= set(u[::max(1, len(u) // RARE_PER_STATE)][:RARE_PER_STATE])
    sample = games[games["game_date"].isin(picked) | games["game_id"].isin(rare)]
    assert len(sample) >= min_games
    p, t = HZ.snapshot_games(sample, sched, getter=store.get)
    assert store.stats["cache_corrupt"] == 0, _EVICTED

    key = ["game_id", "horizon", "team"]
    exp_t = ref_t[ref_t["game_id"].isin(set(sample["game_id"]))]
    m = exp_t.merge(t, on=key, how="outer", suffixes=("_ref", "_new"), indicator=True)
    assert (m["_merge"] == "both").all() and len(t) == len(exp_t) == 8 * len(sample)
    assert (m["state_ref"] == m["state_new"]).all()
    assert _times_equal(m["report_ts_et_ref"], m["report_ts_et_new"])
    assert _times_equal(m["cutoff_utc_ref"], m["cutoff_utc_new"])
    assert set(exp_t["state"]) == set(ref_t["state"])          # every state the season has

    exp_p = ref_p[ref_p["game_id"].isin(set(sample["game_id"]))]
    exp_p = exp_p.sort_values(["game_id", "horizon"], kind="stable").reset_index(drop=True)
    got_p = p.sort_values(["game_id", "horizon"], kind="stable").reset_index(drop=True)
    assert len(got_p) == len(exp_p) > 0
    cols = ["game_id", "horizon", "team", "player_raw", "status", "reason_cat", "reason_raw"]
    assert got_p[cols].equals(exp_p[cols])
    assert _times_equal(got_p["report_ts_et"], exp_p["report_ts_et"])
    assert _times_equal(got_p["cutoff_utc"], exp_p["cutoff_utc"])


def _known_names() -> pd.DataFrame:
    """Every spelling, built from the same inputs as build_train_table's
    `known` (participation roster-games 2018-19..2025-26 plus logs)."""
    from bucketlib import tables
    from bucketlib import train as T
    try:
        rg = pd.concat([tables.roster_games(s) for s in T.seasons("2018-19", "2025-26")], ignore_index=True)
        logs = T.load_table("player_games", T.seasons("2011-12", "2025-26"))
    except (tables.MissingTable, FileNotFoundError) as e:
        pytest.skip(f"name sources not cached: {e}")
    return pd.concat([rg[["person_id", "player_name"]], logs[["person_id", "player_name"]]],
                     ignore_index=True).drop_duplicates()


def test_serve_context_equals_the_training_table_bit_for_bit():
    """THE LOAD-BEARING TEST. For real 2025-26 games trained at t_120 or t_15,
    set now to that horizon's cutoff, pick the report with the injury module's
    selection rule and the snapshot's real-report accept, run the SERVE path
    (context_for) on the table's own rows with the context columns dropped,
    and every CTX_COLS value must equal the training table exactly.

    By two hours before tip every team has filed, so a smaller sample of the
    morning and evening-before horizons rides along: that is where NOT YET
    SUBMITTED (the unknown encoding) lives."""
    if not TRAIN_TABLE.exists():
        pytest.skip("training table not built")
    idx = pd.read_parquet(TRAIN_TABLE, columns=["game_id", "season", "horizon"]).drop_duplicates("game_id")
    idx = idx[idx["season"] == "2025-26"]
    late = sorted(idx.loc[idx["horizon"].isin(["t_120", "t_15"]), "game_id"])[::6]
    early = sorted(idx.loc[idx["horizon"].isin(["d1_1730", "d0_1100"]), "game_id"])[::15]
    sample = late + early
    table = pd.read_parquet(TRAIN_TABLE, filters=[("game_id", "in", sample)])
    sched = _season_schedule("2025-26")
    ref_t = C.read_parquet(C.CACHE_DIR / "injury_horizon_teams_2025-26_v1.parquet")
    names = _known_names()
    store = _cached_store()
    accept = HZ.real_report(sched)

    mismatches, n_rows = [], 0
    for gid in sample:
        rows = table[table["game_id"] == gid]
        g = sched[sched["game_id"] == gid]
        h = rows["horizon"].iloc[0]
        now = HZ.cutoff_for(h, pd.Timestamp(g["tip_utc"].iloc[0]).to_pydatetime())
        sel = INJ.report_before(now, getter=store.get, accept=accept)
        rep = sel.report if sel is not None else None
        got = X.context_for(rows.drop(columns=list(L.CTX_COLS)), g, sched, rep, now, names)
        assert got["person_id"].tolist() == rows["person_id"].tolist()
        for c in L.CTX_COLS:
            if not np.array_equal(got[c].to_numpy(float), rows[c].to_numpy(float), equal_nan=True):
                mismatches.append((gid, h, c))
        # The live states are the snapshot's states at the same cutoff.
        _, lt = X.report_states(g, sched, rep, now)
        rt = ref_t[(ref_t["game_id"] == gid) & (ref_t["horizon"] == h)]
        assert lt["state"].tolist() == rt["state"].tolist(), gid
        n_rows += len(rows)
    assert store.stats["cache_corrupt"] == 0, _EVICTED
    assert not mismatches, mismatches[:10]
    # The sample is not vacuous: it exercises both horizons and every encoding.
    assert len(late) >= 100 and n_rows >= 3000
    assert set(table.loc[table["game_id"].isin(late), "horizon"]) == {"t_120", "t_15"}
    for c in ("inj_out", "inj_questionable", "inj_not_listed", "inj_unknown", "tm_unknown_team"):
        assert (table[c] == 1).any(), c
    assert (table["tm_out_min"] > 0).any() and table["report_age_h"].notna().any()
