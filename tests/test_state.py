"""bucketlib.state and bucketlib.serve on small synthetic state (no cache/ needed).

Every check calls production code: the seed goes through state.build_seed
(with its cache reader swapped for synthetic frames), partitions through
cache.write_parquet into a tmp STATE_DIR, the join through
tables.join_roster_games. The real-data parity proof is tests/test_serve_parity.py.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import features as F
from bucketlib import serve as V
from bucketlib import state as S
from bucketlib import tables
from bucketlib import train as T
from bucketlib.sources import box
from bucketlib.util import season_end_year

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures"
SEASON = "2025-26"


# ── Synthetic games in the bot's partition shape ─────────────────────────────
def _game(gid: str, date: str, home: str = "BOS", away: str = "NYK", pids=((1, 2), (3, 4)),
          season: str = SEASON) -> dict[str, pd.DataFrame]:
    """One final game: per team one player who played and one coach's DNP."""
    tip = pd.Timestamp(f"{date} 19:30", tz="America/New_York").tz_convert("UTC")
    meta = pd.DataFrame([{"game_id": gid, "season": season, "season_type": "RS", "tip_utc": tip,
                          "et_date": date, "home": home, "away": away, "periods": 4, "ot": 0,
                          "source": "cdn"}])
    part, pg, tg = [], [], []
    for team, opp, is_home, (on, off) in ((home, away, True, pids[0]), (away, home, False, pids[1])):
        base = {"season": season, "game_id": gid, "team": team, "is_home": is_home}
        part.append({**base, "person_id": on, "player_name": f"Player {on}", "status": "played",
                     "starter": True, "played_seconds": 1800 + on, "reason_code": "",
                     "absence_kind": ""})
        part.append({**base, "person_id": off, "player_name": f"Player {off}", "status": "dnp",
                     "starter": False, "played_seconds": 0, "reason_code": "DNP_COACH",
                     "absence_kind": "coach"})
        line = {c: on % 7 + 1 for c in tables.COUNT_COLS}
        pg.append({"game_id": gid, "game_date": pd.Timestamp(date), "person_id": on,
                   "player_name": f"Player {on}", "team": team, "opp": opp, "home": is_home,
                   "neutral": False, "min": (1800 + on) / 60, "sec": 1800 + on, **line,
                   "plus_minus": 0, "fantasy": 0.0, "season": season, "season_type": "RS"})
        tg.append({"game_id": gid, "game_date": pd.Timestamp(date), "season": season,
                   "season_type": "RS", "team": team, "opp": opp, "home": is_home,
                   "ot_periods": 0, **{c: 100 + on for c in F.TEAM_ALLOWED}})
    return {"games": meta, "part": pd.DataFrame(part), "pg": pd.DataFrame(pg), "tg": pd.DataFrame(tg)}


def _write(game: dict[str, pd.DataFrame], date: str, kinds=("games", "part", "pg", "tg"),
           season: str = SEASON) -> None:
    """Append one game to the date's partition files (the bot writes one file
    per kind per ET date, every game of that date in it)."""
    for k in kinds:
        p = S.season_dir(season) / f"{k}_{date}.parquet"
        old = C.read_parquet(p)
        df = game[k] if old is None else pd.concat([old, game[k]], ignore_index=True)
        C.write_parquet(df, p)


def _seed_frames(through: str = "2024-25") -> dict[str, pd.DataFrame]:
    """What state.seed_inputs returns, from one synthetic game late in `through`."""
    end = season_end_year(through)
    date = f"{end}-04-13"
    g = _game(f"002{(end - 1) % 100:02d}01200", date, pids=((1, 2), (3, 4)), season=through)
    rg = tables.join_roster_games(g["part"], g["games"], g["pg"], through)
    ev = F.events_from_roster_games(rg)
    priors = pd.DataFrame({"person_id": [1, 1, 3], "season_end": [end, end + 1, end + 1],
                           "first_season_end": [2015, 2015, 2017], "p1_mpg": [20.0, 25.0, 30.0]})
    sd = pd.DataFrame({"team": ["BOS", "NYK"], "game_date": pd.to_datetime([date] * 2)})
    names = rg[["person_id", "player_name"]].drop_duplicates()
    return {"events": ev, "team_games": g["tg"], "priors": priors, "schedule_dates": sd,
            "names": names}


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "seed")
    monkeypatch.setattr(S, "seed_inputs", lambda through: _seed_frames(through))
    return tmp_path


# ── Seed ─────────────────────────────────────────────────────────────────────
def test_seed_keeps_only_the_served_seasons_priors_and_records_a_manifest(state_dir):
    m = S.build_seed(SEASON)
    seed = S.load_seed(SEASON)
    assert m["serves"] == SEASON and m["seed_through"] == "2024-25"
    assert set(seed.priors["season_end"]) == {2026}
    assert m["counts"]["priors"] == 2 and m["counts"]["events"] == len(seed.events)
    for key, fname in S.SEED_FILES.items():
        assert m["content_hashes"][key] == C.read_parquet_meta(S.SEED_DIR / fname)["content_hash"]


def test_seed_rebuild_is_byte_stable(state_dir):
    S.build_seed(SEASON)
    before = {p.name: p.read_bytes() for p in S.SEED_DIR.iterdir()}
    S.build_seed(SEASON)
    after = {p.name: p.read_bytes() for p in S.SEED_DIR.iterdir()}
    assert before == after          # built_at included: nothing to commit


def test_seed_for_another_season_raises_seed_mismatch(state_dir):
    S.build_seed(SEASON)
    with pytest.raises(S.SeedMismatch, match="rebuild the seed"):
        S.load_seed("2026-27")
    seed = S.load_seed(SEASON)
    with pytest.raises(S.SeedMismatch, match="rebuild the seed"):
        S.history("2026-27", "2026-10-21", seed=seed)
    with pytest.raises(S.SeedMismatch, match="rebuild the seed"):
        S.history("2026-27", "2026-10-21")


def test_a_half_updated_seed_raises(state_dir):
    S.build_seed(SEASON)
    pri = C.read_parquet(S.SEED_DIR / "priors_v1.parquet")
    C.write_parquet(pri.assign(p1_mpg=pri["p1_mpg"] + 1), S.SEED_DIR / "priors_v1.parquet")
    with pytest.raises(S.SeedMismatch, match="does not match"):
        S.load_seed(SEASON)


def test_missing_seed_is_a_cold_failure(state_dir):
    with pytest.raises(FileNotFoundError, match="build_serving_seed"):
        S.load_seed(SEASON)


# ── Partitions and history ───────────────────────────────────────────────────
def test_odds_path_is_the_dates_season_partition(state_dir):
    """The one odds path: the bot writes it (step_odds), the app reads it
    (store.committed_odds). The season is the DATE's (util.season_for_date),
    so a January date files under the season that began in October."""
    assert S.odds_path("2025-10-21") == S.season_dir(SEASON) / "odds_2025-10-21.parquet"
    assert S.odds_path(dt.date(2026, 1, 5)) == S.season_dir("2025-26") / "odds_2026-01-05.parquet"
    assert S.odds_path(dt.date(2026, 10, 20)).parent == S.STATE_DIR / "season" / "2026-27"
    with pytest.raises(TypeError, match="ET date"):
        S.odds_path(dt.datetime(2026, 10, 20, 23, 0))
    assert not S.odds_path("2025-10-21").parent.exists()        # a path helper creates nothing


def test_partitions_concatenate_and_ignore_other_files(state_dir):
    _write(_game("0022500001", "2025-10-21"), "2025-10-21")
    _write(_game("0022500002", "2025-10-21", "LAL", "GSW", ((5, 6), (7, 8))), "2025-10-21")
    _write(_game("0022500020", "2025-10-23"), "2025-10-23")
    d = S.season_dir(SEASON)
    (d / ".part_2025-10-24.parquet.tmp123").write_bytes(b"partial")
    C.write_parquet(pd.DataFrame({"game_id": ["x"]}), S.odds_path("2025-10-21"))
    parts = S.season_partitions(SEASON)
    assert set(parts) == {"participation", "games", "player_games", "team_games"}
    assert len(parts["games"]) == 3 and len(parts["participation"]) == 12
    assert len(parts["player_games"]) == 6 and len(parts["team_games"]) == 6
    assert S.ingested_ids(SEASON) == {"0022500001", "0022500002", "0022500020"}
    assert all(df.empty for df in S.season_partitions("2030-31").values())


def test_history_is_strictly_before_the_date(state_dir):
    S.build_seed(SEASON)
    for gid, date in (("0022500001", "2025-10-21"), ("0022500020", "2025-10-23"),
                      ("0022500040", "2025-10-25")):
        _write(_game(gid, date), date)
    h = S.history(SEASON, "2025-10-23")
    cur = h.events[h.events["season"] == SEASON]
    assert set(cur["game_id"]) == {"0022500001"}              # not the 23rd, not the 25th
    assert set(h.team_games.loc[h.team_games["season"] == SEASON, "game_id"]) == {"0022500001"}
    assert h.events["game_date"].max() < pd.Timestamp("2025-10-23")
    assert len(h.events) == len(S.load_seed(SEASON).events) + 4
    assert not h.events["is_target"].any()
    # A DNP has no stat line (NaN, never zero) and canonical minutes.
    dnp = cur[cur["status"] == "dnp"]
    assert dnp[F.RATE_STATS + ["min"]].isna().all().all()
    assert (cur.loc[cur["played"], "min"] == cur.loc[cur["played"], "played_seconds"] / 60).all()
    # On the date itself only earlier games count; the day after includes it.
    nxt = S.history(SEASON, "2025-10-24").events
    assert set(nxt.loc[nxt["season"] == SEASON, "game_id"]) == {"0022500001", "0022500020"}
    assert len(S.history(SEASON, "2025-10-21").events) == len(S.load_seed(SEASON).events)
    assert set(h.priors["season_end"]) == {2026}


def test_a_broken_future_partition_cannot_break_an_earlier_date(state_dir):
    S.build_seed(SEASON)
    _write(_game("0022500001", "2025-10-21"), "2025-10-21")
    bad = _game("0022500020", "2025-10-23")
    bad["pg"] = bad["pg"].assign(person_id=bad["pg"]["person_id"] + 100)   # logs disagree
    _write(bad, "2025-10-23")
    assert len(S.history(SEASON, "2025-10-23").events) > 0
    with pytest.raises(ValueError, match="disagree on who played"):
        S.history(SEASON, "2025-10-24")


def test_a_half_written_ingest_raises(state_dir):
    S.build_seed(SEASON)
    _write(_game("0022500001", "2025-10-21"), "2025-10-21", kinds=("games", "part", "pg"))
    _write(_game("0022500002", "2025-10-21", "LAL", "GSW", ((5, 6), (7, 8))), "2025-10-21")
    with pytest.raises(ValueError, match="miss a side"):
        S.history(SEASON, "2025-10-22")


def test_history_before_the_seed_end_raises(state_dir):
    S.build_seed(SEASON)
    with pytest.raises(ValueError, match="seed would leak"):
        S.history(SEASON, "2025-04-13")


def test_names_carry_every_spelling(state_dir):
    S.build_seed(SEASON)
    g = _game("0022500001", "2025-10-21")
    g["pg"].loc[0, "player_name"] = "P. One"
    _write(g, "2025-10-21")
    names = S.history(SEASON, "2025-10-22").names
    assert set(names.loc[names["person_id"] == 1, "player_name"]) == {"Player 1", "P. One"}


def test_dates_are_et_dates_never_datetimes():
    assert S.as_date("2026-10-20") == dt.date(2026, 10, 20)
    assert S.serve_dir(dt.date(2026, 10, 20)).name == "2026-10-20"
    with pytest.raises(TypeError):
        S.as_date(dt.datetime(2026, 10, 20, 23, 0, tzinfo=dt.timezone.utc))
    with pytest.raises(TypeError):
        S.history(SEASON, pd.Timestamp("2026-01-01"))


# ── Overrides: games the bot cannot ingest ───────────────────────────────────
def _kinds(game: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """A _game in write_override's shape (season_partitions keys)."""
    return {key: game[prefix] for key, prefix in S.PARTITIONS.items()}


def test_an_override_is_read_like_a_partition_and_replaces_the_bot_rows(state_dir):
    S.build_seed(SEASON)
    _write(_game("0022500001", "2025-10-21"), "2025-10-21")
    # The bot could not ingest 0022500002 at all: it is committed locally.
    fix = _game("0022500002", "2025-10-21", "LAL", "GSW", ((5, 6), (7, 8)))
    paths = S.write_override(SEASON, _kinds(fix))
    assert {p.parent for p in paths} == {S.override_dir(SEASON)} and len(paths) == 4
    assert S.ingested_ids(SEASON) == {"0022500001", "0022500002"}
    h = S.history(SEASON, "2025-10-22")
    cur = h.events[h.events["season"] == SEASON]
    assert set(cur["game_id"]) == {"0022500001", "0022500002"} and len(cur) == 8
    assert S.history(SEASON, "2025-10-21").events["season"].ne(SEASON).all()   # still strictly before
    sched = _sched([("0022500001", "RS", "2025-10-21", "final", "BOS", "NYK"),
                    ("0022500002", "RS", "2025-10-21", "final", "LAL", "GSW"),
                    ("0022500030", "RS", "2025-10-23", "scheduled", "LAL", "BOS")])
    assert V.completeness("2025-10-23", sched, SEASON) == []

    # If the bot later ingests the same game too, it is not counted twice:
    # the override's rows win in every kind.
    bot = _game("0022500002", "2025-10-21", "LAL", "GSW", ((5, 6), (7, 8)))
    bot["pg"] = bot["pg"].assign(pts=99)
    _write(bot, "2025-10-21")
    parts = S.season_partitions(SEASON)
    assert (parts["participation"]["game_id"] == "0022500002").sum() == 4
    assert (parts["team_games"]["game_id"] == "0022500002").sum() == 2
    ev = S.history(SEASON, "2025-10-22").events
    got = ev.loc[(ev["game_id"] == "0022500002") & ev["played"]].set_index("person_id")["pts"]
    assert got.sort_index().tolist() == fix["pg"].set_index("person_id")["pts"].sort_index().tolist()
    # Rewriting the same override leaves the files untouched (if_changed).
    before = {p: p.read_bytes() for p in paths}
    S.write_override(SEASON, _kinds(fix))
    assert {p: p.read_bytes() for p in paths} == before


def test_a_bad_override_raises_before_anything_is_written(state_dir):
    g = _kinds(_game("0022500002", "2025-10-21"))
    disagree = dict(g, player_games=g["player_games"].assign(person_id=g["player_games"]["person_id"] + 100))
    with pytest.raises(ValueError, match="disagree on who played"):
        S.write_override(SEASON, disagree)
    two = {k: pd.concat([v, _kinds(_game("0022500003", "2025-10-21", "LAL", "GSW"))[k]])
           for k, v in g.items()}
    with pytest.raises(ValueError, match="one game"):
        S.write_override(SEASON, two)
    with pytest.raises(ValueError, match="exactly the kinds"):
        S.write_override(SEASON, {k: v for k, v in g.items() if k != "team_games"})
    with pytest.raises(ValueError, match="not all season"):
        S.write_override("2026-27", g)
    po = _kinds(_game("0042500101", "2026-04-20"))
    po["games"] = po["games"].assign(season_type="Playoffs")
    with pytest.raises(ValueError, match="never enters state"):
        S.write_override(SEASON, po)
    assert not S.override_dir(SEASON).exists()


def test_half_an_override_raises(state_dir):
    S.write_override(SEASON, _kinds(_game("0022500002", "2025-10-21")))
    (S.override_dir(SEASON) / "tg_0022500002.parquet").unlink()
    with pytest.raises(ValueError, match="override partitions disagree"):
        S.ingested_ids(SEASON)


def test_override_from_backfill_reads_the_local_season_tables(state_dir, monkeypatch):
    """The backfill tables' shapes (participation without season_type, game
    meta with the long season type) come out in the bot's shape."""
    monkeypatch.setattr(C, "CACHE_DIR", state_dir / "cache")
    g = _game("0022500093", "2025-10-24", "HOU", "DET")
    other = _game("0022500094", "2025-10-24", "LAL", "GSW", ((5, 6), (7, 8)))
    for table, prefix in (("participation", "part"), ("participation_games", "games"),
                          ("player_games", "pg"), ("team_games", "tg")):
        df = pd.concat([g[prefix], other[prefix]], ignore_index=True)
        if table == "participation":
            df = df.drop(columns="season_type", errors="ignore")
        if table == "participation_games":
            df = df.assign(season_type="Regular Season")
        C.write_parquet(df, C.CACHE_DIR / f"{table}_{SEASON}_v1.parquet")
    frames = S.override_from_backfill(SEASON, "0022500093")
    assert set(frames) == set(S.PARTITIONS)
    for df in frames.values():
        assert set(df["game_id"]) == {"0022500093"}
        assert (df["season_type"] == "RS").all() and (df["season"] == SEASON).all()
    S.write_override(SEASON, frames)
    assert S.ingested_ids(SEASON) == {"0022500093"}
    with pytest.raises(ValueError, match="backfill that game first"):
        S.override_from_backfill(SEASON, "0022500095")


# ── Completeness ─────────────────────────────────────────────────────────────
def _sched(rows) -> pd.DataFrame:
    """cdn.parse_schedule shape: game_date is a python date, status a word."""
    return pd.DataFrame([{"game_id": g, "season": SEASON, "game_type": t,
                          "game_date": dt.date.fromisoformat(d),
                          "tip_utc": pd.Timestamp(f"{d} 23:30", tz="UTC"),
                          "status": st, "home": h, "away": a} for g, t, d, st, h, a in rows])


def test_completeness_flags_a_missing_final(state_dir):
    _write(_game("0022500001", "2025-11-01"), "2025-11-01")
    half = _game("0022500003", "2025-11-01", "MIA", "PHI", ((5, 6), (7, 8)))
    _write(half, "2025-11-01", kinds=("games", "part"))          # no stat lines yet
    sched = _sched([
        ("0022500001", "RS", "2025-11-01", "final", "BOS", "NYK"),   # ingested
        ("0022500003", "RS", "2025-11-01", "final", "MIA", "PHI"),   # half-ingested: flagged
        ("0022500010", "RS", "2025-11-02", "final", "BOS", "MIA"),   # missing: flagged
        ("0022500011", "RS", "2025-11-02", "final", "LAL", "GSW"),   # teams idle on D
        ("0022500012", "RS", "2025-11-02", "live", "NYK", "PHI"),    # not final yet: flagged too
        ("0012500001", "PRE", "2025-10-10", "final", "BOS", "PHI"),  # preseason
        ("0022500020", "RS", "2025-11-03", "scheduled", "BOS", "NYK"),
        ("0022500021", "RS", "2025-11-03", "scheduled", "MIA", "DAL"),
    ])
    assert V.completeness("2025-11-03", sched, SEASON) == ["0022500003", "0022500010", "0022500012"]
    # Per game: each game's own two teams (BOS-NYK misses BOS's and NYK's
    # games, MIA-DAL only MIA's).
    assert V.completeness_by_game("2025-11-03", sched, SEASON) == {
        "0022500020": ["0022500010", "0022500012"],
        "0022500021": ["0022500003", "0022500010"]}
    _write(_game("0022500010", "2025-11-02", "BOS", "MIA", ((1, 2), (5, 6))), "2025-11-02")
    assert V.completeness("2025-11-03", sched, SEASON) == ["0022500003", "0022500012"]
    assert V.completeness("2025-11-01", sched, SEASON) == []


def test_a_previous_game_still_live_is_a_gap_until_it_is_ingested(state_dir):
    """A 10:30 PM ET first leg is still live when the runs after midnight
    land: the second leg's state lacks it, so that game is provisional (never
    freezable), whatever the schedule's status says."""
    sched = _sched([("0022600215", "RS", "2026-11-10", "live", "ORL", "DEN"),
                    ("0022600224", "RS", "2026-11-11", "scheduled", "ORL", "MIL"),
                    ("0022600225", "RS", "2026-11-11", "scheduled", "BOS", "NYK")]).assign(season="2026-27")
    by_game = V.completeness_by_game("2026-11-11", sched, "2026-27")
    assert by_game == {"0022600224": ["0022600215"], "0022600225": []}
    prov, gaps = V.provisional_by_game("2026-11-11", dt.date(2026, 11, 11), sched, "2026-27")
    assert prov == {"0022600224": True, "0022600225": False} and gaps == by_game
    # The same game final but not ingested: the same answer.
    fin = sched.assign(status=sched["status"].replace({"live": "final"}))
    assert V.completeness_by_game("2026-11-11", fin, "2026-27") == by_game
    # Ingested: no gap.
    _write(_game("0022600215", "2026-11-10", "ORL", "DEN", season="2026-27"), "2026-11-10",
           season="2026-27")
    assert V.provisional_by_game("2026-11-11", dt.date(2026, 11, 11), sched, "2026-27")[0] \
        == {"0022600224": False, "0022600225": False}
    # Tomorrow's slate is provisional whatever its state.
    assert set(V.provisional_by_game("2026-11-11", dt.date(2026, 11, 10), sched, "2026-27")[0].values()) \
        == {True}


def test_a_postponed_game_is_no_state_game_no_gap_and_no_rest_day(state_dir):
    """A postponed game stays on its original date in the live schedule until
    it is rescheduled; training's schedule lists it only where it was played.
    It is no state game (no targets, never a completeness gap, never
    projected), and its date never counts for rest, back-to-back or games in
    the last 7 days (serve.without_postponed)."""
    S.build_seed(SEASON)
    _write(_game("0022500001", "2025-11-01"), "2025-11-01")
    sched = _sched([("0022500001", "RS", "2025-11-01", "final", "BOS", "NYK"),
                    ("0022500005", "RS", "2025-11-03", "scheduled", "BOS", "NYK"),
                    ("0022500010", "RS", "2025-11-04", "scheduled", "BOS", "NYK")]
                   ).assign(postponed=[False, True, False])
    roster = pd.DataFrame({"person_id": [1, 3], "team": ["BOS", "NYK"], "name": ["A", "C"],
                           "two_way": [False, False]})
    assert list(V._state_games(sched)["game_id"]) == ["0022500001", "0022500010"]
    assert V.targets_for(sched.iloc[[1]], roster, SEASON).empty
    assert V.completeness_by_game("2025-11-03", sched, SEASON) == {}
    assert V.completeness_by_game("2025-11-04", sched, SEASON) == {"0022500010": []}
    hist = S.history(SEASON, "2025-11-04")
    f = V.history_features("2025-11-04", sched.iloc[[2]], roster, hist, sched)
    assert len(f) == 2
    assert (f["rest_days"] == 3).all() and (f["b2b"] == 0).all() and (f["games_l7"] == 1).all()
    assert (f["opp_rest_days"] == 3).all()
    # Unflagged (what the flag prevents): the PPD date reads as a game played.
    leak = V.history_features("2025-11-04", sched.iloc[[2]], roster, hist,
                              sched.drop(columns="postponed"))
    assert (leak["rest_days"] == 1).all() and (leak["b2b"] == 1).all() and (leak["games_l7"] == 2).all()
    # A schedule without the column (training's, or one committed before it)
    # is read as is.
    assert V.without_postponed(sched.drop(columns="postponed")).equals(sched.drop(columns="postponed"))


# ── Targets and the serve guards ─────────────────────────────────────────────
def test_targets_for_lists_every_roster_player_of_both_teams():
    sched = _sched([("0022600001", "RS", "2026-10-20", "scheduled", "NYK", "BOS"),
                    ("0012600050", "PRE", "2026-10-20", "scheduled", "LAL", "GSW"),
                    ("0062600001", "CUP_FINAL", "2026-10-20", "scheduled", None, None)])
    roster = pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "team": ["NYK", "NYK", "BOS", "LAL", "DEN"],
                           "name": ["A", "B", "C", "D", "E"], "two_way": [False, True, False, False, False]})
    t = V.targets_for(sched, roster, "2026-27")
    assert list(t.columns) == V.TARGET_COLS
    assert t.set_index("person_id")["team"].to_dict() == {1: "NYK", 2: "NYK", 3: "BOS"}
    assert t.set_index("person_id")["opp"].to_dict() == {1: "BOS", 2: "BOS", 3: "NYK"}
    assert t.set_index("person_id")["home"].to_dict() == {1: True, 2: True, 3: False}
    assert t["two_way"].tolist() == [False, True, False] and (t["season"] == "2026-27").all()
    assert (t["game_date"] == pd.Timestamp("2026-10-20")).all()
    with pytest.raises(ValueError, match="twice"):
        V.targets_for(sched, pd.concat([roster, roster.iloc[[0]]]), "2026-27")


def test_history_features_refuses_a_history_that_reaches_the_date(state_dir):
    S.build_seed(SEASON)
    _write(_game("0022500001", "2025-10-21"), "2025-10-21")
    sched = _sched([("0022500001", "RS", "2025-10-21", "final", "BOS", "NYK"),
                    ("0022500020", "RS", "2025-10-23", "scheduled", "BOS", "NYK")])
    roster = pd.DataFrame({"person_id": [1, 3], "team": ["BOS", "NYK"], "name": ["A", "C"],
                           "two_way": [False, False]})
    leaky = S.history(SEASON, "2025-10-22")
    with pytest.raises(ValueError, match="not strictly before"):
        V.history_features("2025-10-21", sched.iloc[[0]], roster, leaky, sched)
    with pytest.raises(ValueError, match="not on"):
        V.history_features("2025-10-22", sched.iloc[[1]], roster, leaky, sched)


# ── features.apply_lines ─────────────────────────────────────────────────────
def test_apply_lines_matches_the_inline_definition():
    rows = pd.DataFrame({"game_id": ["g1", "g1", "g2"], "home": [True, False, True]})
    lines = pd.DataFrame({"game_id": ["g1"], "home_line": [-4.5], "total": [221.0]})
    out = F.apply_lines(rows, lines)
    assert out["team_line"].tolist()[:2] == [-4.5, 4.5]
    assert out["implied_total"].tolist()[:2] == [221.0 / 2 + 4.5 / 2, 221.0 / 2 - 4.5 / 2]
    assert out["abs_line"].tolist()[:2] == [4.5, 4.5]
    assert out.loc[2, ["home_line", "total", "team_line", "implied_total", "abs_line"]].isna().all()


def test_build_with_lines_equals_apply_lines_on_committed_rows():
    """The bot commits rows built with lines=None and the app adds lines later:
    that must equal what training's build(lines=...) produced."""
    fx = FIX / "features"
    sched = pd.read_parquet(fx / "schedule.parquet")
    tg = pd.read_parquet(fx / "team_games.parquet")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    ev = T.logs_events(pd.read_parquet(fx / "logs_events.parquet"), sched)
    pri = F.prior_seasons(pd.read_parquet(fx / "logs_priors.parquet"))
    sd = T.schedule_dates(sched)
    ids = sorted(ev["game_id"].unique())
    lines = pd.DataFrame({"game_id": ids[::2], "home_line": np.linspace(-12, 12, len(ids[::2])),
                          "total": np.linspace(205, 240, len(ids[::2]))})
    a = F.build(ev, None, tg, sd, pri, lines=lines)
    b = F.apply_lines(F.build(ev, None, tg, sd, pri), lines)
    pd.testing.assert_frame_equal(a, b)
    assert a["team_line"].notna().any() and a["team_line"].isna().any()


# ── The bot's partitions through the same join ───────────────────────────────
BOT_GAMES = ("0021900001", "0021900464")
ZERO_SECOND = ("0021900464", 1629713)   # one turnover, 0 seconds: an appearance in the logs only


def _daily_update():
    spec = importlib.util.spec_from_file_location("daily_update", REPO / "scripts" / "daily_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bot_partitions_join_like_the_backfill(tmp_path, monkeypatch):
    """The bot's real step_finals output (cdnbox lines: no `neutral` column)
    goes through state.history. The full-season version is
    tests/test_serve_parity.py."""
    du = _daily_update()
    payloads = {g: C.read_json(FIX / "cdnbox" / f"cdn_{g}.json.gz")["data"] for g in BOT_GAMES}
    monkeypatch.setattr(du, "cdn_box", payloads.__getitem__)

    def season_dir(season):
        p = tmp_path / "season" / season
        p.mkdir(parents=True, exist_ok=True)
        return p
    monkeypatch.setattr(du, "season_dir", season_dir)
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    tips = {g: pd.Timestamp(p["game"]["gameTimeUTC"]) for g, p in payloads.items()}
    sched = pd.DataFrame([{"game_id": g, "season": "2019-20", "game_type": "RS", "status": "final",
                           "game_date": tips[g].tz_convert("America/New_York").date(),
                           "tip_utc": tips[g], "home": p["game"]["homeTeam"]["teamTricode"],
                           "away": p["game"]["awayTeam"]["teamTricode"]}
                          for g, p in payloads.items()])
    du.RED.clear()
    du.step_finals(sched, dt.date(2019, 12, 31), dry=False)
    assert du.RED == [] and S.ingested_ids("2019-20") == set(BOT_GAMES)

    frames = _seed_frames("2018-19")
    seed = S.Seed(manifest={"serves": "2019-20"}, **frames)
    hist = S.history("2019-20", "2019-12-31", seed=seed)
    got = hist.events[hist.events["season"] == "2019-20"].set_index(["game_id", "person_id"]).sort_index()

    # The backfill's join of the same games: cdn participation + stats logs.
    parsed = [box.parse_cdn(payloads[g], g) for g in BOT_GAMES]
    part = pd.concat([p for p, _ in parsed], ignore_index=True)
    meta = pd.DataFrame([m for _, m in parsed])
    logs = pd.read_parquet(FIX / "cdnbox" / "stats_player_games.parquet")
    rg = tables.join_roster_games(part, meta, logs, "2019-20")
    want = F.events_from_roster_games(rg).set_index(["game_id", "person_id"]).sort_index()
    assert list(got.index) == list(want.index)
    other = got.index != ZERO_SECOND
    for c in F.EVENT_COLS[2:] + ["min"]:
        a, b = got.loc[other, c], want.loc[other, c]
        assert (a.astype(str) == b.astype(str)).all(), c
    # Known and documented (tests/test_cdnbox.py): the logs' zero-second line
    # is an appearance in training; cdn alone says he did not play.
    assert bool(want.loc[ZERO_SECOND, "played"]) and not bool(got.loc[ZERO_SECOND, "played"])
