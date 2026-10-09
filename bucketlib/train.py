"""Training-table assembly, the column contract, and the baselines.

The training table is features.build(...) over the full history, restricted to
the training rows. Everything a model may see is named by FEATURE_PREFIXES /
feature_columns(); `classify()` RAISES on any column it does not recognise, so a
new column (possibly a same-game outcome) cannot drift into the model silently
(GridironValue's columns.py rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cache as C
from . import features as F
from .util import ET, season_end_year, seasons_between

# ── Column contract ──────────────────────────────────────────────────────────
ID_COLS = {"person_id", "game_id", "season", "season_end", "game_date", "tip_utc", "team",
           "opp", "home", "is_target", "last_played_date", "last_team", "src", "season_type"}
OUTCOME_COLS = {"status", "starter", "played", "played_seconds", "min", "sec"} | set(F.RATE_STATS) | {
    "pra", "pr", "pa", "ra", "stocks", "fantasy", "fg2m", "fg2a"}
FEATURE_PREFIXES = (
    "min_ewm", "min_std_s", "gp_s", "start_share_s", "events_s", "min_last", "started_last",
    "miss_streak", "played_rate_l10", "last_absence_len", "games_since_return",
    "events_with_team", "r_", "career_games", "p1_", "p2_", "p3_", "n_prior_seasons",
    "t_", "ot_", "lg_", "rest_days", "b2b", "games_l7", "opp_rest_days", "opp_b2b",
    "days_since_played", "team_changed", "exp_pace", "home_f", "team_line",
    "implied_total", "abs_line", "home_line", "total")
# Not features even though they match a prefix.
NOT_FEATURES = {"total_line_src"}


def classify(col: str) -> str:
    """'id' | 'outcome' | 'feature'. Raises on anything else."""
    if col in ID_COLS:
        return "id"
    if col in OUTCOME_COLS:
        return "outcome"
    if col not in NOT_FEATURES and col.startswith(FEATURE_PREFIXES):
        return "feature"
    raise KeyError(f"column {col!r} is not in the contract: classify it in train.py")


def feature_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    cols = [c for c in df.columns if classify(c) == "feature"]
    return [c for c in cols if not c.startswith(exclude)]


# ── Inputs ───────────────────────────────────────────────────────────────────
def load_table(name: str, seasons: list[str]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        df = C.read_parquet(C.CACHE_DIR / f"{name}_{s}_v1.parquet")
        if df is not None:
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no {name} tables for {seasons[0]}..{seasons[-1]}")
    return pd.concat(parts, ignore_index=True)


def tips(schedule: pd.DataFrame) -> pd.Series:
    """game_id -> tip_utc; games with an unknown tip get 7pm ET on the game
    date (only ordering within a day depends on it, and a team plays once)."""
    s = schedule.set_index("game_id")["tip_utc"]
    return s


def logs_events(logs: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """State events from the box logs alone (played rows only). Used where the
    participation table is not built yet: every played-game feature (minutes,
    rates, prior seasons) is exact; availability features are NOT (there are
    no absence rows), so models trained on these events exclude them."""
    st = logs[logs["season_type"].isin(["RS", "PI"])].copy()
    ev = pd.DataFrame({
        "person_id": st["person_id"].astype("int64"),
        "game_id": st["game_id"].astype(str),
        "season": st["season"],
        "game_date": pd.to_datetime(st["game_date"]),
        "team": st["team"], "opp": st["opp"], "home": st["home"].astype(bool),
        "status": "played", "starter": np.nan, "played": True,
        "played_seconds": st["sec"].astype(float),
    })
    tip = tips(schedule)
    ev["tip_utc"] = ev["game_id"].map(tip)
    miss = ev["tip_utc"].isna()
    if miss.any():
        ev.loc[miss, "tip_utc"] = (ev.loc[miss, "game_date"] + pd.Timedelta(hours=19)).dt.tz_localize(ET).dt.tz_convert("UTC")
    for c in F.RATE_STATS:
        ev[c] = st[c].astype(float).to_numpy()
    ev["min"] = ev["played_seconds"] / 60.0
    ev["is_target"] = False
    return ev


def schedule_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    """(team, game_date) for every game a team played that counts for rest:
    regular season, play-in, AND the Cup final (in no season log)."""
    s = schedule[schedule["game_type"].isin(["RS", "PI", "CUP_FINAL"])].dropna(subset=["home", "away"])
    d = pd.concat([s[["home", "game_date"]].rename(columns={"home": "team"}),
                   s[["away", "game_date"]].rename(columns={"away": "team"})], ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d.drop_duplicates()


def add_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Board combos and Fantasy Score from the components (PrizePicks weights:
    PTS 1, REB 1.2, AST 1.5, STL 3, BLK 3, TOV -1)."""
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]
    df["stocks"] = df["stl"] + df["blk"]
    df["fantasy"] = df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"] + 3 * df["stl"] + 3 * df["blk"] - df["tov"]
    df["fg2m"] = df["fgm"] - df["fg3m"]
    df["fg2a"] = df["fga"] - df["fg3a"]
    return df


# ── Baselines (strictly prior games) ─────────────────────────────────────────
COMBOS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"),
          "ra": ("reb", "ast"), "stocks": ("stl", "blk")}
FANTASY_W = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "tov": -1.0}


def _b_rate(df: pd.DataFrame, stat: str) -> pd.Series:
    """Baseline 2: minutes (EWMA, then season-to-date, then last season) x
    exposure-weighted rate (EWMA20, then season-to-date, then last season)."""
    mins = df["min_ewm5_s"].fillna(df["min_std_s"]).fillna(df["p1_mpg"])
    rate = df[f"r_{stat}_ewm20"].fillna(df[f"r_{stat}_std"]).fillna(df[f"p1_rate_{stat}"])
    return mins * rate


def baseline_rate(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "min":
        return df["min_ewm5_s"].fillna(df["min_std_s"]).fillna(df["p1_mpg"])
    if target in COMBOS:
        return sum(_b_rate(df, s) for s in COMBOS[target])
    if target == "fantasy":
        return sum(w * _b_rate(df, s) for s, w in FANTASY_W.items())
    if target in ("fg2m", "fg2a"):
        a, b = ("fgm", "fg3m") if target == "fg2m" else ("fga", "fg3a")
        return _b_rate(df, a) - _b_rate(df, b)
    return _b_rate(df, target)


def baseline_std(df: pd.DataFrame, target: str) -> pd.Series:
    """Baseline 1: season-to-date per-game mean from PRIOR games, falling back to
    last season's per-game mean, then the league mean of the training rows
    (applied by the caller). Season-to-date sums = rate_std x minutes_std x gp."""
    if target == "min":
        return df["min_std_s"].fillna(df["p1_mpg"])

    def one(s):
        cur = df[f"r_{s}_std"] * df["min_std_s"]
        prev = df[f"p1_rate_{s}"] * df["p1_mpg"]
        return cur.fillna(prev)
    if target in COMBOS:
        return sum(one(s) for s in COMBOS[target])
    if target == "fantasy":
        return sum(w * one(s) for s, w in FANTASY_W.items())
    if target in ("fg2m", "fg2a"):
        a, b = ("fgm", "fg3m") if target == "fg2m" else ("fga", "fg3a")
        return one(a) - one(b)
    return one(target)


def seasons(first: str, last: str) -> list[str]:
    return seasons_between(first, last)


def season_end(s: str) -> int:
    return season_end_year(s)
