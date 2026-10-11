"""THE serving parity test: real dates replayed through the SERVE path, on the
partitions the BOT really writes, against the training table.

The 2025-26 season is ingested by the bot's own step, scripts/daily_update.py
step_finals, from the cached cdn box scores (cache/raw/box/2025-26: the very
payloads the bot fetches; nothing is fetched here) into a tmp state dir. Then
the production serve path end to end: a seed built by state.build_seed (serving
2025-26, history through 2024-25), state.history() (seed + the bot's partitions
through tables.join_roster_games, strictly before the date),
serve.targets_for() on the participation roster, serve.history_features()
(features.build with lines=None). Injury context (livectx.CTX_COLS) and line
features are added later in serving and are excluded from the feature checks.

The bot's partitions differ from the stats.nba.com tables training used in
exactly two ways, and each is measured here, never hidden:

  1. ONE game cannot be ingested from cdn. The cdn box of 0022500093 (DET at
     HOU, 2025-10-24) is short 521 HOU player-seconds; box.parse_cdn rejects it
     on every run and the bot goes RED. state.write_override commits it from
     the local backfill (state.override_from_backfill), and history() and
     ingested_ids() pick it up.
  2. STAT CORRECTIONS. The cdn liveData box never receives stats.nba.com's
     corrections: 148 of 26,757 player-games (81 games) and 36 of 2,470
     team-games differ, each by single credits (a block given to a teammate,
     a rebound split differently; at most 2 in any count).

So the proof has three parts:
  CODE PATH: with the logs' counts put into the bot-written pg_/tg_ files
    (the corrections removed, every other byte as the bot wrote it), every
    feature column of the training table that build() produces equals training
    bit for bit (np.array_equal, equal_nan) on six real dates, and on the next
    games of two teams whose game was postponed while the schedule still lists
    it on its original date (cdn's `postponed` flag). Partitions ON and
    AFTER each date sit in the state dir throughout, and scrambling every
    outcome on or after D through the same files leaves D's features unchanged.
  SOURCE: the cdn counts equal the logs' counts outside those corrections
    (bounded per row and in share), which is what makes the substitution above
    honest: a mis-mapped cdn column would fail here, not hide there.
  TOLERANCE (the policy): with the cdn counts as the bot really has them, only
    box-count features move (player rates, team and league pace/allowed), and
    the production models' projections move by at most TOLERANCE below. That
    gap is accepted: the alternative, training on uncorrected cdn lines, would
    throw away the corrections for every past season to match the current one.

Needs the local backfill (cache/train_table_full_v1.parquet, the 2025-26 season
tables and raw cdn boxes); skips without it. One features.build per date and
state (~15s each, about 3 minutes in all).
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import features as F
from bucketlib import livectx as L
from bucketlib import serve as V
from bucketlib import state as S
from bucketlib import tables
from bucketlib import train as T
from bucketlib.sources import box

REPO = Path(__file__).resolve().parent.parent
TRAIN = C.CACHE_DIR / "train_table_full_v1.parquet"
SEASON = "2025-26"
RAW_BOX = C.CACHE_DIR / "raw" / "box" / SEASON
TABLES = ("participation", "participation_games", "player_games", "team_games", "schedule",
          "game_lines")

pytestmark = pytest.mark.skipif(
    not TRAIN.exists() or not RAW_BOX.is_dir()
    or any(not (C.CACHE_DIR / f"{t}_{SEASON}_v1.parquet").exists() for t in TABLES),
    reason="needs the local backfill (train_table_full_v1, 2025-26 tables, raw cdn boxes)")

# Each date is checked below to really be the case it claims to be.
DATES = {
    "season opener": "2025-10-21",
    "first game after the Cup final": "2025-12-18",
    "ordinary date (B2B second legs, a return from a 10+ game absence)": "2026-01-14",
    "right after the trade deadline": "2026-02-07",
    "first game day after the All-Star break": "2026-02-19",
    "postponed-game makeup": "2026-01-29",
}
IN_SEASON = [label for label in DATES if label != "season opener"]
MUTATION_DATE = "2026-01-14"
UNINGESTABLE = "0022500093"            # DET at HOU, 2025-10-24: cdn box short 521 HOU seconds
# MIA at CHI, postponed on 2026-01-08 (ESPN STATUS_POSTPONED) and made up on
# 2026-01-29. Training's schedule lists it on the make-up date only; both
# teams next played on 2026-01-10.
PPD_GAME, PPD_DATE, PPD_NEXT = "0022500529", "2026-01-08", "2026-01-10"
HOU_DATE = "2025-10-26"               # HOU's first game after it
LINE_COLS = ["home_line", "total", "team_line", "implied_total", "abs_line"]
NOT_SERVED_HERE = set(L.CTX_COLS) | set(LINE_COLS)
# Features computed from box COUNTS (player rates, team and league pace and
# allowed rates): the only ones a stat correction can move. Game counters
# (t_games_s, ot_games_s) count games, not box events.
COUNT_FEATURE = re.compile(r"^(r_|t_|ot_|lg_)|^exp_pace$")
NOT_COUNT_FEATURE = re.compile(r"_games_s$")
# Measured 2026-10-09 over the four in-season dates (max |served - training|
# across every row): p_play 0.014, mu_min 0.14, pts 0.24, reb 0.17, stl 0.13,
# pra 0.28, fantasy 0.53 (2026-04-10, outside these dates: mu_min 0.46, pts 0.42).
TOLERANCE = {"p_play": 0.03, "mu_min": 0.75, "stat": 0.6, "composed": 1.0}


# ── Fixtures: real tables, a production seed, the bot's own partitions ───────
@pytest.fixture(scope="module")
def season_tables():
    t = {name: C.read_parquet(C.CACHE_DIR / f"{name}_{SEASON}_v1.parquet") for name in TABLES}
    t["schedule"] = t["schedule"].assign(
        status=lambda s: np.where(s["game_status"] == 3, "final", "scheduled"))
    return t


@pytest.fixture(scope="module")
def seed_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("seed")
    S.build_seed(SEASON, out)
    return out


def _daily_update():
    spec = importlib.util.spec_from_file_location("daily_update", REPO / "scripts" / "daily_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cached_cdn_box(game_id: str) -> dict:
    """The bot's cdn_box, from the raw cache only: a missing payload fails the
    test loudly (it would otherwise be a network call)."""
    p = box.raw_path(game_id, "cdn")
    if not p.exists():
        raise FileNotFoundError(f"no cached cdn box for {game_id}: {p}")
    return C.read_json_stamped(p)[0]


@pytest.fixture(scope="module")
def bot_raw(tmp_path_factory, season_tables):
    """(state root, RED messages): scripts/daily_update.step_finals over every
    final 2025-26 state game, exactly as the bot runs it, into a tmp root."""
    root = tmp_path_factory.mktemp("bot_raw")
    du = _daily_update()

    def season_dir(season):
        p = root / "season" / season
        p.mkdir(parents=True, exist_ok=True)
        return p

    sched = season_tables["schedule"]
    bot_sched = sched.assign(game_date=sched["game_date"].dt.date)     # cdn.parse_schedule shape
    last = bot_sched.loc[bot_sched["game_type"].isin(["RS", "PI"]), "game_date"].max()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(du, "cdn_box", _cached_cdn_box)
        mp.setattr(du, "season_dir", season_dir)
        mp.setattr(S, "STATE_DIR", root)      # state.ingested_ids reads the same tree
        du.RED.clear()
        du.step_finals(bot_sched, last + dt.timedelta(days=1), dry=False)
        red = list(du.RED)
    assert not any(root.joinpath("season", SEASON).glob("*.tmp*"))
    return root, red


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


@pytest.fixture(scope="module")
def bot_fixed(tmp_path_factory, bot_raw):
    """The bot's partitions + the override for the one game it cannot ingest,
    written by the production path (override_from_backfill -> write_override)."""
    root = _copy(bot_raw[0], tmp_path_factory.mktemp("bot_fixed"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "STATE_DIR", root)
        S.write_override(SEASON, S.override_from_backfill(SEASON, UNINGESTABLE))
    return root


def _with_logs_counts(root: Path, t) -> Path:
    """Put the stats logs' box counts into the bot-written pg_ and tg_ files
    (stat corrections removed; every other column as the bot wrote it)."""
    d = root / "season" / SEASON
    for prefix, src, key in (("pg", t["player_games"], ["game_id", "person_id"]),
                             ("tg", t["team_games"], ["game_id", "team"])):
        for f in sorted(d.glob(f"{prefix}_*.parquet")):
            df = pd.read_parquet(f)
            m = df[key].merge(src[key + tables.COUNT_COLS], on=key, how="left", validate="one_to_one")
            assert m[tables.COUNT_COLS].notna().all().all(), f"{f.name}: a bot line has no logs line"
            for c in tables.COUNT_COLS:
                df[c] = m[c].to_numpy()
            C.write_parquet(df, f)
    return root


@pytest.fixture(scope="module")
def bot_corrected(tmp_path_factory, bot_fixed, season_tables):
    return _with_logs_counts(_copy(bot_fixed, tmp_path_factory.mktemp("bot_corrected")), season_tables)


def _scramble_from(root: Path, start: pd.Timestamp, seed: int = 11) -> Path:
    """Scramble every outcome dated on or after `start` in the date files: stat
    lines, team box counts and seconds played."""
    rng = np.random.default_rng(seed)
    for f in sorted((root / "season" / SEASON).glob("*_*.parquet")):
        prefix, date = f.stem.rsplit("_", 1)
        if pd.Timestamp(date) < start or prefix == "games":
            continue
        df = pd.read_parquet(f)
        if prefix in ("pg", "tg"):
            for c in F.RATE_STATS:
                df[c] = rng.permutation(df[c].to_numpy())
        else:
            played = df["status"] == "played"
            df.loc[played, "played_seconds"] = rng.permutation(df.loc[played, "played_seconds"].to_numpy())
        C.write_parquet(df, f)
    return root


@pytest.fixture(scope="module")
def bot_mutated(tmp_path_factory, bot_corrected):
    return _scramble_from(_copy(bot_corrected, tmp_path_factory.mktemp("bot_mut")),
                          pd.Timestamp(MUTATION_DATE))


@pytest.fixture(scope="module")
def train_rows():
    days = [pd.Timestamp(d) for d in DATES.values()]
    return pd.read_parquet(TRAIN, filters=[("game_date", "in", days)])


@pytest.fixture(scope="module")
def served_cdn():
    """Served frames on the cdn counts (bot_fixed), shared by the tolerance tests."""
    return {}


def _at(root: Path, seed_dir: Path):
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "STATE_DIR", root)
    mp.setattr(S, "SEED_DIR", seed_dir)
    return mp


def _serve(root: Path, seed_dir: Path, date: str, t) -> tuple[pd.DataFrame, S.History]:
    """The bot's step for one date: history strictly before it, the date's
    games from the schedule, the roster = the participation roster of each game."""
    sched = t["schedule"]
    day = sched[(sched["game_date"] == pd.Timestamp(date)) & sched["game_type"].isin(["RS", "PI"])]
    part = t["participation"]
    roster = part.loc[part["game_id"].isin(set(day["game_id"])),
                      ["person_id", "team", "player_name"]].assign(two_way=False)
    mp = _at(root, seed_dir)
    try:
        hist = S.history(SEASON, date)
        return V.history_features(date, day, roster, hist, sched), hist
    finally:
        mp.undo()


def _frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.set_index(["game_id", "person_id"]).sort_index()[cols]


def _served_columns(trained: pd.DataFrame) -> list[str]:
    cols = [c for c in T.feature_columns(trained) if c not in NOT_SERVED_HERE]
    assert len(cols) > 150, f"only {len(cols)} feature columns: the comparison is vacuous"
    return cols


def _differing(trained: pd.DataFrame, served: pd.DataFrame, label: str) -> list[str]:
    cols = _served_columns(trained)
    missing = [c for c in cols if c not in served.columns]
    assert not missing, f"{label}: the serve path does not produce {missing[:10]}"
    a, b = _frame(trained, cols), _frame(served, cols)
    assert list(a.index) == list(b.index), f"{label}: served rows differ from training rows"
    return [c for c in cols if not np.array_equal(a[c].to_numpy(dtype=float),
                                                  b[c].to_numpy(dtype=float), equal_nan=True)]


def _assert_bit_identical(trained: pd.DataFrame, served: pd.DataFrame, label: str) -> None:
    bad = _differing(trained, served, label)
    assert not bad, f"{label}: served features differ from training: {bad[:12]}"


# ── Code path: bit for bit on the bot's partitions ───────────────────────────
@pytest.mark.parametrize("label", list(DATES))
def test_serve_path_equals_training_bit_for_bit(label, season_tables, seed_dir, bot_corrected,
                                                train_rows):
    date = DATES[label]
    D = pd.Timestamp(date)
    # Partitions on and after D are present while D is served.
    files = list((bot_corrected / "season" / SEASON).glob("games_*.parquet"))
    assert any(f.stem.endswith(date) for f in files)
    assert any(f.stem[-10:] > date for f in files)

    served, hist = _serve(bot_corrected, seed_dir, date, season_tables)
    trained = train_rows[train_rows["game_date"] == D]
    assert len(trained) > 50
    _assert_bit_identical(trained, served, f"{label} {date}")
    assert hist.events["game_date"].max() < D and hist.team_games["game_date"].max() < D

    # Lines are added to the committed rows later (features.apply_lines, which
    # training's build(lines=...) calls): same result as the training table.
    lined = F.apply_lines(served, season_tables["game_lines"])
    lined = lined.set_index(["game_id", "person_id"]).sort_index()
    t = trained.set_index(["game_id", "person_id"]).sort_index()
    bad = [c for c in LINE_COLS if not np.array_equal(t[c].to_numpy(dtype=float),
                                                      lined[c].to_numpy(dtype=float), equal_nan=True)]
    assert not bad and lined["team_line"].notna().all(), f"{label}: line features differ: {bad}"

    # The state holds every final game before D: completeness has nothing to flag.
    mp = _at(bot_corrected, seed_dir)
    try:
        assert V.completeness(date, season_tables["schedule"], SEASON) == []
    finally:
        mp.undo()

    # The date really is the case it claims to be.
    s = served.set_index(["game_id", "person_id"]).sort_index()
    if label == "season opener":
        assert (hist.events["season"] == SEASON).sum() == 0
        assert (s["gp_s"] == 0).all() and s["p1_mpg"].notna().any()
    elif label == "first game after the Cup final":
        # NYK and SAS played the Cup final (006, in no season log) on 12-16:
        # rest counts it, so their rest is 2 days, not 5.
        r = s[s["team"].isin(["NYK", "SAS"])].groupby("team")["rest_days"].first()
        assert set(r.index) == {"NYK", "SAS"} and (r == 2).all()
    elif label.startswith("ordinary"):
        assert (s["b2b"] == 1).any()
        t = trained.set_index(["game_id", "person_id"])
        assert ((s["miss_streak"] >= 10) & t["played"].reindex(s.index).astype(bool)).any()
    elif label == "right after the trade deadline":
        moved = s[(s["team_changed"] == 1) & (s["last_team"] != s["team"])]
        assert len(moved) >= 10
    elif label.startswith("first game day after the All-Star"):
        prev = hist.events.loc[hist.events["season"] == SEASON, "game_date"].max()
        assert (D - prev).days >= 5
    elif label == "postponed-game makeup":
        # MIA at CHI, postponed on 2026-01-08 and played here.
        assert set(s.loc[PPD_GAME, "team"]) == {"CHI", "MIA"}


def test_future_mutation_through_the_serve_path(season_tables, seed_dir, bot_corrected,
                                                bot_mutated, train_rows):
    """Scramble every outcome on or after D in the bot's partition files: D's
    served features do not move (and later histories do, so the scramble is real)."""
    D = pd.Timestamp(MUTATION_DATE)
    later = (D + pd.Timedelta(days=10)).date()
    served, _ = _serve(bot_mutated, seed_dir, MUTATION_DATE, season_tables)
    _assert_bit_identical(train_rows[train_rows["game_date"] == D], served, "future mutation")

    histories = []
    for root in (bot_corrected, bot_mutated):
        mp = _at(root, seed_dir)
        try:
            ev = S.history(SEASON, later).events
        finally:
            mp.undo()
        histories.append(ev[ev["game_date"] >= D].sort_values(["game_id", "person_id"]))
    a, b = histories
    assert len(a) > 1000 and list(a["game_id"]) == list(b["game_id"])
    for c in ("pts", "min"):
        assert not np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                                  equal_nan=True), f"the scramble left {c} unchanged: the test is vacuous"


# ── A postponed game still on its original date ──────────────────────────────
def test_a_postponed_game_on_its_original_date_leaves_the_next_games_alone(
        season_tables, seed_dir, bot_corrected):
    """Until it is rescheduled the live schedule keeps a postponed game on
    its ORIGINAL date (cdn.parse_schedule flags it `postponed`). Served with
    that schedule, both teams' next games equal training bit for bit: rest,
    back-to-back and games in the last 7 days never count the date of a game
    that was not played. Without the flag they would (checked, so the
    comparison is not vacuous)."""
    sched = season_tables["schedule"]
    ppd = sched.copy()
    moved = ppd["game_id"] == PPD_GAME
    assert moved.sum() == 1
    ppd.loc[moved, "game_date"] = pd.Timestamp(PPD_DATE)
    ppd.loc[moved, "tip_utc"] = pd.Timestamp(f"{PPD_DATE} 20:00", tz="America/New_York").tz_convert("UTC")
    ppd.loc[moved, "status"] = "scheduled"
    ppd["postponed"] = moved
    trained = pd.read_parquet(TRAIN, filters=[("game_date", "==", pd.Timestamp(PPD_NEXT))])
    assert {"CHI", "MIA"} <= set(trained["team"])

    served, _ = _serve(bot_corrected, seed_dir, PPD_NEXT, dict(season_tables, schedule=ppd))
    _assert_bit_identical(trained, served, f"next games after the postponement {PPD_NEXT}")
    assert set(V._state_games(ppd)["game_id"]) == set(V._state_games(sched)["game_id"]) - {PPD_GAME}

    unflagged = ppd.drop(columns="postponed")
    leak, _ = _serve(bot_corrected, seed_dir, PPD_NEXT, dict(season_tables, schedule=unflagged))
    assert {"rest_days", "games_l7"} <= set(_differing(trained, leak, "unflagged"))
    moved_rows = leak[leak["team"].isin(["CHI", "MIA"])].set_index(["game_id", "person_id"])
    want = trained.set_index(["game_id", "person_id"]).loc[moved_rows.index]
    assert (moved_rows["rest_days"] < want["rest_days"]).all()


# ── The one game the bot cannot ingest ───────────────────────────────────────
def test_the_bot_cannot_ingest_one_game_and_the_override_recovers_it(season_tables, seed_dir,
                                                                     bot_raw, bot_fixed):
    root, red = bot_raw
    sched = season_tables["schedule"]
    finals = set(sched.loc[sched["game_type"].isin(["RS", "PI"]) & (sched["status"] == "final"),
                           "game_id"])
    assert red and all(UNINGESTABLE in r for r in red), red
    assert any(r.startswith(f"parse failed for {UNINGESTABLE}") for r in red)

    mp = _at(root, seed_dir)
    try:
        assert finals - S.ingested_ids(SEASON) == {UNINGESTABLE}
        # Stale for every later HOU and DET date: provisional, never freezable.
        assert V.completeness(HOU_DATE, sched, SEASON) == [UNINGESTABLE]
        assert UNINGESTABLE not in set(S.history(SEASON, HOU_DATE).events["game_id"])
    finally:
        mp.undo()

    mp = _at(bot_fixed, seed_dir)
    try:
        assert S.ingested_ids(SEASON) == finals
        assert V.completeness(HOU_DATE, sched, SEASON) == []
        got = S.history(SEASON, HOU_DATE).events
    finally:
        mp.undo()
    # The override's events are the backfill's: tables.roster_games on the season tables.
    want = F.events_from_roster_games(tables.roster_games(SEASON))
    got = got[got["game_id"] == UNINGESTABLE].set_index("person_id").sort_index()
    want = want[want["game_id"] == UNINGESTABLE].set_index("person_id").sort_index()
    assert len(got) >= 26 and list(got.index) == list(want.index)
    assert (got[["team", "status"]] == want[["team", "status"]]).all().all()
    for c in ["played", "played_seconds", "min"] + F.RATE_STATS:
        assert np.array_equal(got[c].to_numpy(dtype=float), want[c].to_numpy(dtype=float),
                              equal_nan=True), c


# ── Source: cdn counts vs the stats logs ─────────────────────────────────────
def test_cdn_counts_equal_the_logs_outside_stat_corrections(season_tables, seed_dir, bot_raw):
    mp = _at(bot_raw[0], seed_dir)
    try:
        parts = S.season_partitions(SEASON)
    finally:
        mp.undo()
    # Measured 2026-10-09: 148 of 26,757 player lines differ (max 2 in a count,
    # 6 summed over a line), 36 of 2,470 team lines (max 1 in a count, 4 summed).
    for kind, src, key, per_cell, per_row, share in (
            ("player_games", season_tables["player_games"], ["game_id", "person_id"], 2, 6, 0.01),
            ("team_games", season_tables["team_games"], ["game_id", "team"], 1, 6, 0.025)):
        bot = parts[kind]
        m = bot.merge(src, on=key, how="left", suffixes=("", "_logs"), validate="one_to_one")
        assert m[f"{tables.COUNT_COLS[0]}_logs"].notna().all(), f"{kind}: a bot line has no logs line"
        d = pd.DataFrame({c: (m[c] - m[f"{c}_logs"]).abs() for c in tables.COUNT_COLS})
        moved = (d > 0).any(axis=1)
        assert d.to_numpy().max() <= per_cell, f"{kind}: a count differs by {d.to_numpy().max()}"
        assert d.sum(axis=1).max() <= per_row
        # Corrections are rare single credits; a mis-mapped cdn column would
        # differ on most lines, and nothing differing would mean no corrections.
        assert 0 < moved.mean() <= share, f"{kind}: {moved.mean():.2%} of lines differ"


# ── Tolerance: the cdn counts as the bot has them ────────────────────────────
@pytest.mark.parametrize("label", IN_SEASON)
def test_stat_corrections_move_only_box_count_features(label, season_tables, seed_dir, bot_fixed,
                                                       train_rows, served_cdn):
    date = DATES[label]
    served, _ = _serve(bot_fixed, seed_dir, date, season_tables)
    served_cdn[date] = served
    trained = train_rows[train_rows["game_date"] == pd.Timestamp(date)]
    bad = _differing(trained, served, f"{label} {date}")
    count = [c for c in _served_columns(trained)
             if COUNT_FEATURE.search(c) and not NOT_COUNT_FEATURE.search(c)]
    assert len(count) > 60
    assert not set(bad) - set(count), \
        f"{label}: non-count features differ on cdn counts: {sorted(set(bad) - set(count))[:12]}"
    assert bad, f"{label}: no count feature moved: the tolerance below measures nothing"


def test_stat_corrections_move_projections_within_tolerance(season_tables, seed_dir, bot_fixed,
                                                            train_rows, served_cdn):
    """The policy number: the production models on served-from-cdn history
    features vs the same rows of the training table (context and lines taken
    from the training table for both, so only the history features differ)."""
    from bucketlib import models as M
    try:
        mods = M.load()
    except (M.ModelMismatch, FileNotFoundError) as e:
        pytest.skip(f"production models not loadable here: {e}")
    worst = {}
    for label in IN_SEASON:
        date = DATES[label]
        if date not in served_cdn:
            served_cdn[date] = _serve(bot_fixed, seed_dir, date, season_tables)[0]
        trained = train_rows[train_rows["game_date"] == pd.Timestamp(date)]
        cols = _served_columns(trained)
        base = trained.set_index(["game_id", "person_id"]).sort_index()
        mixed = base.copy()
        mixed[cols] = _frame(served_cdn[date], cols)
        d = (M.predict(mods, base) - M.predict(mods, mixed)).abs().max()
        worst = {k: max(v, worst.get(k, 0.0)) for k, v in d.items()}
    assert worst["p_play"] <= TOLERANCE["p_play"], worst
    assert worst["mu_min"] <= TOLERANCE["mu_min"], worst
    assert max(worst[s] for s in mods.stats) <= TOLERANCE["stat"], worst
    assert max(worst[c] for c in [*T.COMBOS, "fantasy"]) <= TOLERANCE["composed"], worst
    assert worst["pts"] > 0, "projections did not move: the tolerance measures nothing"


# ── The seed ─────────────────────────────────────────────────────────────────
def test_seed_serves_one_season(seed_dir):
    m = S.load_seed(SEASON, seed_dir).manifest
    assert m["serves"] == SEASON and m["seed_through"] == "2024-25"
    with pytest.raises(S.SeedMismatch, match="rebuild the seed"):
        S.load_seed("2026-27", seed_dir)
    pri = S.load_seed(SEASON, seed_dir).priors
    assert set(pri["season_end"]) == {2026}


def _assert_names_are_every_spelling(seed: S.Seed) -> None:
    """The names seed (injury-name resolution) is every (person_id,
    player_name) pair the participation tables and the stats logs carry over
    the seed window, read straight from those tables, and every person in the
    events has one: an empty or truncated names file cannot pass."""
    w, through = seed.manifest["windows"], seed.manifest["seed_through"]
    pairs = [C.read_parquet(C.CACHE_DIR / f"participation_{s}_v1.parquet")[["person_id", "player_name"]]
             for s in T.seasons(w["participation_from"], through)]
    pairs.append(T.load_table("player_games", T.seasons(w["priors_from"], through))[["person_id", "player_name"]])
    want = set(map(tuple, pd.concat(pairs).drop_duplicates().to_numpy()))
    names = seed.names
    assert not names.duplicated().any() and names["player_name"].fillna("").str.len().gt(0).all()
    got = set(map(tuple, names[["person_id", "player_name"]].to_numpy()))
    assert got == want, (f"names seed: {len(want - got)} spellings missing "
                         f"(e.g. {sorted(want - got)[:3]}), {len(got - want)} not in any source")
    assert set(seed.events["person_id"]) <= set(names["person_id"])


def test_seed_names_are_every_spelling(seed_dir):
    _assert_names_are_every_spelling(S.load_seed(SEASON, seed_dir))


def test_committed_seed_names_are_every_spelling():
    if not (S.SEED_DIR / S.MANIFEST).exists():
        pytest.skip("no committed seed")
    seed = S.load_seed()
    assert len(seed.names) > 1500
    _assert_names_are_every_spelling(seed)
