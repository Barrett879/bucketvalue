"""Joined tables: one row per ROSTERED player per game ("roster-games").

Sources (built by scripts/build_logs.py and scripts/build_participation.py):
  player_games_{season}_v1          stat lines, players who appeared only
  participation_{season}_v1         every listed player: played / dnp / inactive
  participation_games_{season}_v1   per-game meta (tip_utc, ET date, OT)
  team_games_{season}_v1            team box totals, both sides of each game

Rules this module enforces, each a sibling bug:
  - A DNP or inactive player has NO stat line: his stat columns are NaN, never
    0 (a zero-fill is a phantom-actual generator: DiamondValue published
    +4.60% accuracy that was -0.07% because of exactly this).
  - The played set in participation must EQUAL the appeared set in the logs,
    game by game. A mismatch raises with the game ids: two sources disagreeing
    about who played is a data bug to look at, not a row to drop.
  - GAME_UNIVERSE: only regular season (002) and play-in (005) games enter the
    player state; playoffs (004) are kept and flagged; preseason (001),
    All-Star (003) and the Cup final (006, in no season log) never enter state.
  - Historical team = the team on that game's own participation row, never an
    end-of-season roster.
"""
from __future__ import annotations

import pandas as pd

from . import cache as C
from .util import seasons_between

STATE_TYPES = {"RS", "PI"}          # enter player state and training
STORED_TYPES = {"RS", "PI", "PO"}   # stored; PO excluded from v1 training
COUNT_COLS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
              "ast", "stl", "blk", "tov", "pf", "pts"]
STAT_COLS = ["min", "sec"] + COUNT_COLS + ["plus_minus", "fantasy"]
_TYPE = {"Regular Season": "RS", "PlayIn": "PI", "Playoffs": "PO", "RS": "RS", "PI": "PI", "PO": "PO"}


class MissingTable(RuntimeError):
    pass


def _read(name: str, season: str, required: bool = True) -> pd.DataFrame | None:
    p = C.CACHE_DIR / f"{name}_{season}_v1.parquet"
    df = C.read_parquet(p)
    if df is None and required:
        raise MissingTable(f"{p.name} is missing: run the backfill for {season}")
    return df


def season_complete(season: str) -> bool:
    meta = C.read_parquet_meta(C.CACHE_DIR / f"participation_{season}_v1.parquet")
    return bool(meta.get("complete"))


def roster_games(season: str, require_complete: bool = True) -> pd.DataFrame:
    """Every listed player in every stored game of `season`, with his stat line
    when he played (NaN otherwise), from the backfilled season tables."""
    part = _read("participation", season)
    if require_complete and not season_complete(season):
        raise MissingTable(f"participation_{season}_v1 is not complete")
    games = _read("participation_games", season)
    pg = _read("player_games", season)
    return join_roster_games(part, games, pg, season)


def join_roster_games(part: pd.DataFrame, games: pd.DataFrame, pg: pd.DataFrame,
                      season: str) -> pd.DataFrame:
    """The join and reconciliation behind roster_games, as a pure function of
    frames. ONE implementation: the backfilled season tables (roster_games) and
    the bot's per-date partitions (state.history) both go through it, so a
    served history row is built exactly like a training row.

    part   participation rows: game_id, person_id, team, player_name, status,
           starter, played_seconds, reason_code (sources.box COLUMNS)
    games  per-game meta: game_id, season_type, tip_utc, et_date, home, away, ot
    pg     stat lines of players who appeared: game_id, person_id, team, opp,
           home, sec, the COUNT_COLS, plus_minus (stats logs schema, or the
           bot's sources.cdnbox.player_lines; `neutral` optional)
    """
    games = games.copy()
    # The NBA Cup final (id prefix 006) is in participation but in NO season
    # log; it counts only for rest/schedule features (taken from the schedule),
    # never for player state, so it is not a roster-game here.
    games = games[~games["game_id"].astype(str).str.startswith("006")]
    part = part[~part["game_id"].astype(str).str.startswith("006")]
    games["game_type"] = games["season_type"].map(_TYPE)
    if games["game_type"].isna().any():
        raise ValueError(f"{season}: unknown season_type values "
                         f"{sorted(games.loc[games['game_type'].isna(), 'season_type'].unique())}")
    keep_games = set(games.loc[games["game_type"].isin(STORED_TYPES), "game_id"])

    part = part[part["game_id"].isin(keep_games)].copy()
    pg = pg[pg["game_id"].isin(keep_games)].copy()

    # ── The played set must agree exactly ────────────────────────────────────
    played = part.loc[part["status"] == "played", ["game_id", "person_id"]]
    a = set(map(tuple, played.to_numpy()))
    b = set(map(tuple, pg[["game_id", "person_id"]].to_numpy()))
    only_part, only_logs = a - b, b - a
    # ZERO-SECOND LOG ROWS. The logs occasionally carry a 0-second line for a
    # player the box score says did not play. Two kinds, decided by the stats:
    #  - all zero (sec 0, every count 0, plus-minus 0): a phantom. Participation
    #    is the authority and he did not play.
    #  - a stat recorded with 0 seconds (Justin James, SAC, 0021900464: one
    #    turnover, sec 0, cdn played=0): a stat proves he was on the floor, so
    #    it is an appearance. His participation row is flipped to played.
    # Both are counted. A logs-only row WITH seconds is a real conflict and
    # still raises below.
    phantom, zero_sec = set(), set()
    if only_logs:
        lo = pg.set_index(["game_id", "person_id"]).loc[sorted(only_logs)]
        counts = lo[[c for c in COUNT_COLS if c in lo]].fillna(0)
        z = lo["sec"].fillna(0) == 0
        allzero = z & (counts == 0).all(axis=1) & (lo.get("plus_minus", 0) == 0)
        phantom = set(lo.index[allzero.to_numpy()])
        zero_sec = set(lo.index[(z & ~allzero).to_numpy()])
        only_logs = only_logs - phantom - zero_sec
        pg = pg[~pg.set_index(["game_id", "person_id"]).index.isin(phantom)]
        if zero_sec:
            key = part.set_index(["game_id", "person_id"]).index
            flip = key.isin(zero_sec)
            part = part.copy()
            part.loc[flip, "status"] = "played"
            part.loc[flip, "reason_code"] = "ZERO_SECOND_APPEARANCE"
    if only_part or only_logs:
        ex = sorted(only_part)[:3] + sorted(only_logs)[:3]
        raise ValueError(
            f"{season}: participation and logs disagree on who played: "
            f"{len(only_part)} played-only-in-participation, {len(only_logs)} in-logs-only, "
            f"e.g. {ex}")

    # `neutral` rides along when the source has it (the stats logs do; the
    # bot's cdn lines, sources.cdnbox, do not). No feature reads it, so a
    # frame without it joins the same way; it is never invented as False.
    stats = pg[["game_id", "person_id", "team", "opp", "home"] +
               [c for c in ["neutral"] + STAT_COLS if c in pg.columns]]
    out = part.merge(stats, on=["game_id", "person_id"], how="left", suffixes=("", "_pg"))
    if "team_pg" in out:
        bad = out["team_pg"].notna() & (out["team_pg"] != out["team"])
        if bad.any():
            raise ValueError(f"{season}: team differs between participation and logs "
                             f"for {int(bad.sum())} rows, e.g. {out.loc[bad].iloc[0].to_dict()}")
        out = out.drop(columns="team_pg")

    meta = games[["game_id", "game_type", "tip_utc", "et_date", "home", "away", "ot"]].rename(
        columns={"home": "home_team", "away": "away_team"})
    out = out.merge(meta, on="game_id", how="left", validate="many_to_one")
    if out["tip_utc"].isna().any():
        raise ValueError(f"{season}: {int(out['tip_utc'].isna().sum())} rows have no game meta")
    out["game_date"] = pd.to_datetime(out["et_date"])
    out["opp"] = out["opp"].where(out["opp"].notna(),
                                  out["away_team"].where(out["team"] == out["home_team"], out["home_team"]))
    out["home"] = out["team"] == out["home_team"]
    out["played"] = out["status"] == "played"
    # NaN, never zero, for anyone who did not play.
    for c in STAT_COLS:
        if c in out:
            out.loc[~out["played"], c] = float("nan")
    out["in_state"] = out["game_type"].isin(STATE_TYPES)
    out["season"] = season
    out = out.sort_values(["game_date", "tip_utc", "game_id", "team", "person_id"]).reset_index(drop=True)
    out.attrs["phantom_log_rows"] = sorted(phantom)
    out.attrs["zero_second_appearances"] = sorted(zero_sec)
    return out


def roster_games_range(first: str, last: str, require_complete: bool = True) -> pd.DataFrame:
    return pd.concat([roster_games(s, require_complete) for s in seasons_between(first, last)],
                     ignore_index=True)
