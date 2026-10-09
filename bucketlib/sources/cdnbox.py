"""Stat lines from the cdn.nba.com liveData box score, in the SAME schema as the
stats.nba.com logs the models trained on (statsnba.player_games / team_games).

Why: GitHub Actions cannot reach stats.nba.com (it hangs from Azure, measured
2026-10-09), so the daily bot builds the current season from cdn box scores.
A feature computed from a cdn-built row must equal the one computed from a
stats-built row, so this module is verified field by field against the stats
logs on every cached game (tests/test_cdnbox.py; scripts in the docstring of
check_parity). Seconds come from the ISO duration ('PT25M01.00S'), which is the
precise value; points/rebounds/etc. are integers in both sources.
"""
from __future__ import annotations

import pandas as pd

from .. import teams
from ..util import et_date_of

PLAYER_MAP = {
    "fgm": "fieldGoalsMade", "fga": "fieldGoalsAttempted",
    "fg3m": "threePointersMade", "fg3a": "threePointersAttempted",
    "ftm": "freeThrowsMade", "fta": "freeThrowsAttempted",
    "oreb": "reboundsOffensive", "dreb": "reboundsDefensive", "reb": "reboundsTotal",
    "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
    "pf": "foulsPersonal", "pfd": "foulsDrawn", "pts": "points", "plus_minus": "plusMinusPoints",
}


def iso_seconds(v) -> float:
    """'PT25M01.00S' -> 1501.0; '' or None -> 0.0."""
    s = str(v or "")
    if not s.startswith("PT"):
        return 0.0
    s = s[2:]
    mins = 0.0
    if "M" in s:
        m, s = s.split("M", 1)
        mins = float(m or 0)
    secs = float(s.rstrip("S") or 0) if s else 0.0
    return mins * 60 + secs


def _game(payload: dict) -> dict:
    g = payload.get("game") if isinstance(payload, dict) else None
    if not isinstance(g, dict) or "homeTeam" not in g:
        raise ValueError("not a liveData box score payload")
    return g


def player_lines(payload: dict) -> pd.DataFrame:
    """One row per player who PLAYED (played == '1'), stats.nba logs schema."""
    g = _game(payload)
    gid = str(g["gameId"])
    date = et_date_of(g["gameTimeUTC"])
    rows = []
    for side, other, is_home in (("homeTeam", "awayTeam", True), ("awayTeam", "homeTeam", False)):
        team = teams.from_team_id(g[side]["teamId"])
        opp = teams.from_team_id(g[other]["teamId"])
        if team is None or opp is None:
            raise ValueError(f"{gid}: unknown team id in box score")
        for p in g[side].get("players", []):
            if str(p.get("played")) != "1":
                continue
            st = p.get("statistics") or {}
            sec = iso_seconds(st.get("minutes"))
            row = {"game_id": gid, "game_date": pd.Timestamp(date), "person_id": int(p["personId"]),
                   "player_name": p.get("name", ""), "team": team, "opp": opp, "home": is_home,
                   "min": sec / 60.0, "sec": int(round(sec))}
            for k, src in PLAYER_MAP.items():
                row[k] = int(st.get(src) or 0)
            row["fantasy"] = (row["pts"] + 1.2 * row["reb"] + 1.5 * row["ast"]
                              + 3 * row["stl"] + 3 * row["blk"] - row["tov"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and df["person_id"].duplicated().any():
        raise ValueError(f"{gid}: duplicate players in box score")
    return df


def team_lines(payload: dict) -> pd.DataFrame:
    """Two rows (one per team): counts summed over the team's PLAYERS, which is
    what the stats logs' team totals are (team rebounds and team turnovers are
    not credited to a player and are not in either)."""
    g = _game(payload)
    pl = player_lines(payload)
    gid = str(g["gameId"])
    ot = max(int(g.get("period") or 4) - int(g.get("regulationPeriods") or 4), 0)
    agg = pl.groupby(["team", "opp", "home"])[[k for k in PLAYER_MAP if k not in ("pfd", "plus_minus")]].sum().reset_index()
    out = agg.merge(agg[["team", "pts"]].rename(columns={"team": "opp", "pts": "opp_pts"}), on="opp")
    out["game_id"] = gid
    out["game_date"] = pd.Timestamp(et_date_of(g["gameTimeUTC"]))
    out["margin"] = out["pts"] - out["opp_pts"]
    out["ot_periods"] = ot
    # TEAM turnovers (shot clock, 5-second, etc.) are credited to no player but
    # ARE in the stats logs' team TOV, and possessions depend on them: take the
    # team-level total, measured equal to stats on every cached 2019-20 game.
    tov = {teams.from_team_id(g[s]["teamId"]): int(g[s]["statistics"]["turnoversTotal"])
           for s in ("homeTeam", "awayTeam")}
    out["tov"] = out["team"].map(tov)
    return out
