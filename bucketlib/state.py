"""Committed serving state: every path under state/, the seed, the bot's
per-date partitions, and the history the serving feature build runs on.

WHY THIS EXISTS. Serving must call features.build() exactly as training did
(docs/serving_design.md, principle 0), but the bot on GitHub Actions has no
cache/ backfill and cannot reach stats.nba.com. So the history training saw is
frozen once into a committed SEED (events, team games, priors, schedule dates,
name spellings through the season before the one served), and the season in
progress is the bot's per-date partitions, which go through the SAME join and
reconciliation as the backfill (tables.join_roster_games). history() glues the
two and cuts everything at ONE point: games strictly before the ET date served.

Layout (bot-owned, one writer per path; this module only reads, except the
seed and the overrides, which scripts/build_serving_seed.py writes LOCALLY
through build_seed and write_override):

  state/seed/{events,team_games,priors,schedule_dates,names}_v1.parquet
  state/seed/manifest_v1.json
  state/season/{season}/{pg,tg,part,games,odds}_{YYYY-MM-DD}.parquet
  state/season/{season}/override/{pg,tg,part,games}_{game_id}.parquet
  state/current/{schedule,roster,injury_players,injury_teams}_v1.parquet
  state/serve/{YYYY-MM-DD}/{features,projections}_v1.parquet

OVERRIDES. Some final games the bot can never ingest: the cdn box score of
0022500093 (DET at HOU, 2025-10-24) is short 521 HOU player-seconds, so
sources.box.parse_cdn rejects it on every run, and the bot cannot reach the V3
box score the backfill used instead. Without a way in, that game is missing from
every later HOU and DET history and serve.completeness flags their dates for
the rest of the season. write_override commits such a game LOCALLY (from the
backfill, override_from_backfill) under override/, which the bot never writes;
season_partitions reads it as an ordinary partition, so history() and
ingested_ids() pick it up, and it replaces any bot rows of the same game.

A seed serves ONE season: its priors are the rows for that season only (the
2027-28 priors need the 2026-27 logs), so asking it for another season raises
SeedMismatch instead of silently serving every player without priors.
Path helpers never create directories: the app reads state and writes nothing.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import cache as C
from . import features as F
from . import tables
from . import train as T
from .util import iso_date, now_utc, season_end_year, season_for_date, season_str

STATE_DIR = C.REPO_ROOT / "state"
SEED_DIR = STATE_DIR / "seed"

SEED_FILES = {
    "events": "events_v1.parquet",
    "team_games": "team_games_v1.parquet",
    "priors": "priors_v1.parquet",
    "schedule_dates": "schedule_dates_v1.parquet",
    "names": "names_v1.parquet",
}
MANIFEST = "manifest_v1.json"
# season_partitions() key -> the bot's file prefix.
PARTITIONS = {"participation": "part", "games": "games", "player_games": "pg",
              "team_games": "tg"}

# The training window (scripts/build_train_table.py defaults). The seed is the
# history that script hands to features.build, so it uses the same seasons.
PART_FIRST = "2018-19"     # first season with participation (absence) events
WARMUP_FROM = "2014-15"    # log-only warm-up events and team games start here
PRIORS_FROM = "2011-12"    # box logs pulled only for prior-season summaries


class SeedMismatch(RuntimeError):
    """The committed seed does not serve the season asked for, or its files do
    not match its manifest. The fix is always to rebuild the seed."""


@dataclass
class Seed:
    events: pd.DataFrame
    team_games: pd.DataFrame
    priors: pd.DataFrame
    schedule_dates: pd.DataFrame
    names: pd.DataFrame
    manifest: dict


@dataclass
class History:
    events: pd.DataFrame
    team_games: pd.DataFrame
    schedule_dates: pd.DataFrame
    priors: pd.DataFrame
    names: pd.DataFrame


# ── Paths (resolved at call time, so tests can point STATE_DIR elsewhere) ────
def season_dir(season: str) -> Path:
    return STATE_DIR / "season" / season


def current_path(name: str) -> Path:
    return STATE_DIR / "current" / f"{name}_v1.parquet"


def serve_dir(date) -> Path:
    return STATE_DIR / "serve" / as_date(date).isoformat()


def odds_path(date) -> Path:
    """The bot's pregame odds snapshots for one ET date (ESPN scoreboard
    rows: away, home, home_line, total, captured_at, ...), in the season
    directory of that date: state/season/{season}/odds_{YYYY-MM-DD}.parquet.
    The one place the path is built: the bot writes it (step_odds), the app
    reads it (store.committed_odds)."""
    d = as_date(date)
    return season_dir(season_for_date(d)) / f"odds_{d.isoformat()}.parquet"


def override_dir(season: str) -> Path:
    return season_dir(season) / "override"


def as_date(d) -> dt.date:
    """An ET calendar date from a date or 'YYYY-MM-DD'. A datetime is refused:
    which ET date a timestamp belongs to is the caller's decision (util.to_et),
    never a guess made here."""
    if isinstance(d, dt.datetime):
        raise TypeError(f"pass an ET date, not a datetime: {d!r}")
    return dt.date.fromisoformat(iso_date(d))


# ── Seed: build (local, from cache/) ────────────────────────────────────────
def seed_inputs(through: str, part_first: str = PART_FIRST, warmup_from: str = WARMUP_FROM,
                priors_from: str = PRIORS_FROM,
                schedule_through: str | None = None) -> dict[str, pd.DataFrame]:
    """The history scripts/build_train_table.py passes to features.build, for
    seasons up to `through`: log-only warm-up events, then participation
    events through tables.roster_games; RS + PI team games; prior-season
    summaries over every log season; schedule dates; every name spelling.
    Also `roster_games` (the joined participation rows the events come from,
    which training resolves injury names against); the seed does not store it.

    The window arguments exist so build_train_table.py can call THIS instead of
    keeping its own copy of the assembly (it loads its schedule through
    2026-27, hence `schedule_through`, default `through`). Needs the local
    backfill in cache/ (complete participation tables)."""
    part_seasons = T.seasons(part_first, through)
    warm_seasons = [s for s in T.seasons(warmup_from, through) if s not in part_seasons]
    logs = T.load_table("player_games", T.seasons(priors_from, through))
    sched = T.load_table("schedule", T.seasons(warmup_from, schedule_through or through))
    tg = T.load_table("team_games", T.seasons(warmup_from, through))
    tg = tg[tg["season_type"].isin(["RS", "PI"])].copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])

    rg = pd.concat([tables.roster_games(s, require_complete=True) for s in part_seasons],
                   ignore_index=True)
    ev_part = F.events_from_roster_games(rg)
    ev_part["game_date"] = pd.to_datetime(ev_part["game_date"])
    ev_warm = T.logs_events(logs[logs["season"].isin(warm_seasons)], sched)
    events = pd.concat([ev_warm, ev_part], ignore_index=True)
    names = pd.concat([rg[["person_id", "player_name"]], logs[["person_id", "player_name"]]],
                      ignore_index=True).drop_duplicates()
    return {"events": events, "team_games": tg, "priors": F.prior_seasons(logs),
            "schedule_dates": T.schedule_dates(sched), "names": names, "roster_games": rg}


def build_seed(serves: str, out_dir: Path | None = None) -> dict:
    """Write the seed that serves `serves` (history through the season before)
    and return its manifest. Every file is written with if_changed=True and the
    manifest is rewritten only when something besides built_at changed, so a
    rebuild of the same data leaves the commit empty."""
    out = Path(out_dir) if out_dir is not None else SEED_DIR
    through = season_str(season_end_year(serves) - 1)
    frames = seed_inputs(through)
    pri = frames["priors"]
    frames["priors"] = pri[pri["season_end"] == season_end_year(serves)]
    frames = {k: v.reset_index(drop=True) for k, v in frames.items()}

    counts, hashes = {}, {}
    for key, fname in SEED_FILES.items():
        df = frames[key]
        C.write_parquet(df, out / fname, meta={"seed_through": through, "serves": serves},
                        if_changed=True)
        counts[key], hashes[key] = len(df), C.frame_hash(df)
    manifest = {
        "seed_through": through, "serves": serves,
        "windows": {"participation_from": PART_FIRST, "warmup_from": WARMUP_FROM,
                    "priors_from": PRIORS_FROM},
        "counts": counts, "content_hashes": hashes,
        "built_at": now_utc().isoformat(timespec="seconds"),
    }
    old = C.read_json(out / MANIFEST)
    same = isinstance(old, dict) and {k: v for k, v in old.items() if k != "built_at"} == \
        {k: v for k, v in manifest.items() if k != "built_at"}
    if same:
        return old
    C.write_json(out / MANIFEST, manifest)
    return manifest


# ── Seed: read ───────────────────────────────────────────────────────────────
def load_seed(serves: str | None = None, seed_dir: Path | None = None) -> Seed:
    """The committed seed. Raises FileNotFoundError when it is absent (a cold
    failure, never an empty history) and SeedMismatch when it serves another
    season or a file's content hash differs from the manifest (a half-updated
    seed would pair one season's events with another season's priors)."""
    d = Path(seed_dir) if seed_dir is not None else SEED_DIR
    manifest = C.read_json(d / MANIFEST)
    if manifest is None:
        raise FileNotFoundError(f"no serving seed at {d}: run scripts/build_serving_seed.py")
    if serves is not None and manifest["serves"] != serves:
        raise SeedMismatch(
            f"the seed serves {manifest['serves']} (history through {manifest['seed_through']}), "
            f"not {serves}: rebuild the seed (python scripts/build_serving_seed.py --serves {serves})")
    frames = {}
    for key, fname in SEED_FILES.items():
        p = d / fname
        if not p.exists():
            raise FileNotFoundError(f"seed file {p} is missing: rebuild the seed")
        stored = C.read_parquet_meta(p).get("content_hash")
        if stored != manifest["content_hashes"][key]:
            raise SeedMismatch(f"seed file {fname} does not match {MANIFEST}: rebuild the seed")
        frames[key] = C.read_parquet(p)
    return Seed(manifest=manifest, **frames)


# ── Bot partitions (and local overrides) ────────────────────────────────────
_DATE_FILE = r"\d{4}-\d{2}-\d{2}"
_GAME_FILE = r"\d{10}"


def _partition_files(d: Path, prefix: str, stem: str = _DATE_FILE) -> list[Path]:
    pat = re.compile(rf"^{prefix}_{stem}\.parquet$")
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(f"{prefix}_*.parquet") if pat.match(p.name))


def _read_all(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _overrides(season: str) -> dict[str, pd.DataFrame]:
    """The override partitions per kind. Each override game has all four kinds
    (write_override writes them together), else this raises: half an override
    would silently drop a game's stat lines."""
    o = override_dir(season)
    over = {key: _read_all(_partition_files(o, prefix, _GAME_FILE))
            for key, prefix in PARTITIONS.items()}
    ids = {key: set(df["game_id"]) if len(df) else set() for key, df in over.items()}
    if len({frozenset(v) for v in ids.values()}) > 1:
        raise ValueError(f"{season}: override partitions disagree on their games: {ids}")
    return over


def season_partitions(season: str) -> dict[str, pd.DataFrame]:
    """Every per-date partition of `season`, concatenated per kind: keys
    participation (part_), games (games_), player_games (pg_), team_games (tg_).
    A kind with no files is an empty frame (opening night has none). Override
    games (override_dir) are read with them and REPLACE the bot's rows of the
    same game in every kind, so a game is never counted twice."""
    d = season_dir(season)
    over = _overrides(season)
    fixed = set(over["games"]["game_id"]) if len(over["games"]) else set()
    out = {}
    for key, prefix in PARTITIONS.items():
        bot = _read_all(_partition_files(d, prefix))
        if fixed and len(bot):
            bot = bot[~bot["game_id"].isin(fixed)]
        frames = [f for f in (bot, over[key]) if len(f)]
        out[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def ingested_ids(season: str) -> set[str]:
    """Game ids present in ALL four partition kinds, overrides included. A game
    with its meta but no stat lines (a half-written ingest) is not ingested.
    This is THE definition: the bot's re-fetch and RED logic should use it too."""
    ids = [set(df["game_id"]) if len(df) else set() for df in season_partitions(season).values()]
    return set.intersection(*ids)


def write_override(season: str, frames: dict[str, pd.DataFrame]) -> list[Path]:
    """Commit ONE final state game the bot cannot ingest, as override
    partitions (run locally; the bot never writes override/). `frames` has the
    season_partitions keys, rows of that game only. The game goes through the
    same join and checks history() runs before anything is written, so a bad
    override raises here and never at serve time. Files are written with
    if_changed=True; returns their paths."""
    if set(frames) != set(PARTITIONS):
        raise ValueError(f"an override needs exactly the kinds {sorted(PARTITIONS)}, got {sorted(frames)}")
    ids = {key: set(df["game_id"]) for key, df in frames.items()}
    gids = set().union(*ids.values())
    if len(gids) != 1 or any(v != gids for v in ids.values()):
        raise ValueError(f"an override is one game with rows in every kind, got {ids}")
    gid = next(iter(gids))
    for key, df in frames.items():
        if "season" in df and (df["season"] != season).any():
            raise ValueError(f"{gid}: {key} rows are not all season {season}")
    cut = pd.to_datetime(frames["games"]["et_date"]).max() + pd.Timedelta(days=1)
    _, tg, _ = _partition_history(frames, season, cut)
    if tg.empty:
        raise ValueError(f"{gid} is not a regular-season or play-in game: it never enters state")
    paths = []
    for key, prefix in PARTITIONS.items():
        p = override_dir(season) / f"{prefix}_{gid}.parquet"
        C.write_parquet(frames[key].reset_index(drop=True), p, if_changed=True)
        paths.append(p)
    return paths


def override_from_backfill(season: str, game_id: str) -> dict[str, pd.DataFrame]:
    """One game's frames from the LOCAL backfill tables (cache/ participation,
    participation_games, player_games, team_games), for write_override. The
    backfill reaches stats.nba.com (the V3 box score, the stat logs), which the
    bot cannot, so this is how a game whose cdn box the parser rejects gets into
    state. Every frame carries season and season_type ('RS'/'PI') like the
    bot's partitions."""
    source = {"participation": "participation", "games": "participation_games",
              "player_games": "player_games", "team_games": "team_games"}
    out = {}
    for key, table in source.items():
        df = T.load_table(table, [season])
        df = df[df["game_id"] == game_id].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"the {table}_{season} backfill has no rows for {game_id}: "
                             f"backfill that game first")
        out[key] = df
    kind = out["games"]["season_type"].map(tables._TYPE)
    if kind.isna().any() or kind.iloc[0] not in tables.STATE_TYPES:
        raise ValueError(f"{game_id} is {out['games']['season_type'].iloc[0]!r}, not a state game")
    return {key: df.assign(season=season, season_type=kind.iloc[0]) for key, df in out.items()}


def _partition_history(parts: dict[str, pd.DataFrame], season: str,
                       cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(events, team_games, names) from the partitions of games STRICTLY
    before `cut`. The cut is applied once, to the game list, before the join:
    no row dated on or after the served date reaches the join at all, so a
    future partition's data can neither leak into nor break an earlier date
    (an unreadable file still raises: that is a bug, not a cache miss)."""
    games = parts["games"]
    if games.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    games = games[pd.to_datetime(games["et_date"]) < cut]
    if games.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ids = set(games["game_id"])
    empty = [k for k, v in parts.items() if v.empty]
    if empty:
        raise ValueError(f"{season}: partitions have games but no {empty} files")
    part = parts["participation"][parts["participation"]["game_id"].isin(ids)]
    pg = parts["player_games"][parts["player_games"]["game_id"].isin(ids)]
    tg = parts["team_games"][parts["team_games"]["game_id"].isin(ids)]

    rg = tables.join_roster_games(part, games, pg, season)
    listed = {g for g in ids if not str(g).startswith("006")}
    unlisted = sorted(listed - set(rg["game_id"]))
    if unlisted:
        raise ValueError(f"{season}: {len(unlisted)} ingested games have no participation rows, "
                         f"e.g. {unlisted[:5]}")
    state_ids = set(rg.loc[rg["in_state"], "game_id"])
    tg = tg[tg["game_id"].isin(state_ids)].copy()
    sides = tg.groupby("game_id").size()
    bad = sorted(state_ids - set(sides[sides == 2].index))
    if bad:
        raise ValueError(f"{season}: team_games partitions miss a side for {len(bad)} games, "
                         f"e.g. {bad[:5]}")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    ev = F.events_from_roster_games(rg)
    ev["game_date"] = pd.to_datetime(ev["game_date"])
    names = pd.concat([rg[["person_id", "player_name"]], pg[["person_id", "player_name"]]],
                      ignore_index=True)
    return ev, tg, names


def history(season: str, before, seed: Seed | None = None) -> History:
    """Seed + the season's partitions, every event STRICTLY before the ET date
    `before`: the history the serving build for that date runs on.

    Pass `seed` to reuse one load across dates; it must serve `season`."""
    cut = pd.Timestamp(as_date(before))
    seed = load_seed(season) if seed is None else seed
    if seed.manifest["serves"] != season:
        raise SeedMismatch(f"the seed serves {seed.manifest['serves']}, not {season}: "
                           f"rebuild the seed (python scripts/build_serving_seed.py --serves {season})")
    last = seed.events["game_date"].max()
    if cut <= last:
        raise ValueError(f"before={cut.date()} is not after the seed's last event "
                         f"{last.date()}: the seed would leak into it")
    ev, tg, names = _partition_history(season_partitions(season), season, cut)
    events = pd.concat([seed.events, ev], ignore_index=True) if len(ev) else seed.events
    team_games = pd.concat([seed.team_games, tg], ignore_index=True) if len(tg) else seed.team_games
    if len(names):
        names = pd.concat([seed.names, names], ignore_index=True).drop_duplicates(ignore_index=True)
    else:
        names = seed.names
    return History(events=events, team_games=team_games, schedule_dates=seed.schedule_dates,
                   priors=seed.priors, names=names)
