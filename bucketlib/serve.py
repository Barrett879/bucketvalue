"""Serve-side history features: from committed state to the feature rows of one
date's games, through the SAME builder training used.

THE ONE RULE. history_features() calls features.build(events, targets, team
games, schedule dates, priors, lines=None) exactly as scripts/build_train_table.py
does, with the history from state.history(season, before=date) (every event
strictly before the date) and the date's games appended as synthetic target
rows (features.make_targets: identity only, outcomes unknown). There is no
second implementation of any feature here; tests/test_serve_parity.py replays
real dates through this path and compares to the training table bit for bit.

Lines and injury context are NOT added here: they are volatile, so the bot and
the app add them to these committed rows later (features.apply_lines,
context.context_for). The APP never calls history_features (the full history is
~0.5 GB of columns); the bot does, and commits the result per date.
"""
from __future__ import annotations

import pandas as pd

from . import features as F
from . import tables
from . import train as T
from .sources import cdn
from .state import History, as_date, ingested_ids
from .util import season_for_date

TARGET_COLS = ["person_id", "game_id", "season", "game_date", "tip_utc", "team", "opp",
               "home", "player_name", "two_way"]
FINAL = cdn.STATUS[3]


def without_postponed(schedule: pd.DataFrame) -> pd.DataFrame:
    """The schedule without its postponed rows (cdn.parse_schedule's
    `postponed`, set only on positive evidence). A postponed game stays on
    its ORIGINAL date in the live schedule until it is rescheduled, while
    training's schedules list it only on the date it was played: left in, it
    would give both teams rest, back-to-back and games-in-7 values training
    never saw, and a game that will not be played would be projected and
    frozen. A schedule without the column (training's stats schedules,
    schedules committed before it existed) is returned as is."""
    if "postponed" not in schedule.columns:
        return schedule
    return schedule[~schedule["postponed"].fillna(False).astype(bool)]


def _state_games(games: pd.DataFrame) -> pd.DataFrame:
    """Regular season and play-in games with both teams known, postponed
    games excluded (without_postponed). Preseason, All-Star, playoffs and Cup
    knockout placeholders (TBD) get no projections. The one definition of a
    state game: targets_for, completeness and pipeline.state_games read it."""
    g = without_postponed(games)
    g = g[g["game_type"].isin(tables.STATE_TYPES)].dropna(subset=["home", "away"])
    return g.assign(game_date=pd.to_datetime(g["game_date"]))


def targets_for(games: pd.DataFrame, roster: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per (current roster player, game) for both teams of every state
    game in `games` (schedule rows: game_id, game_type, game_date, tip_utc,
    home, away). `roster` is the cdn playerIndex (person_id, team, name,
    two_way); a `player_name` column stands in for `name`. Two-way and inactive
    players are rows too: training rows are every LISTED player, and the
    availability model decides who plays."""
    if roster["person_id"].duplicated().any():
        dup = roster.loc[roster["person_id"].duplicated(), "person_id"].unique()[:5]
        raise ValueError(f"roster lists a person twice: {list(dup)}")
    g = _state_games(games)
    keep = ["game_id", "game_date", "tip_utc"]
    sides = pd.concat([
        g[keep].assign(team=g["home"], opp=g["away"], home=True),
        g[keep].assign(team=g["away"], opp=g["home"], home=False),
    ], ignore_index=True)
    name = roster["name"] if "name" in roster else roster["player_name"]
    r = pd.DataFrame({"person_id": roster["person_id"].astype("int64"), "team": roster["team"],
                      "player_name": name, "two_way": roster["two_way"].astype(bool)})
    t = sides.merge(r, on="team", how="inner")
    t["season"] = season
    t = t.sort_values(["tip_utc", "game_id", "home", "person_id"], ascending=[True, True, False, True],
                      kind="mergesort")
    return t[TARGET_COLS].reset_index(drop=True)


def history_features(date, games: pd.DataFrame, roster: pd.DataFrame, hist: History,
                     schedule: pd.DataFrame) -> pd.DataFrame:
    """History features for `date`'s games (one row per targets_for row, plus
    player_name and two_way), from `hist` = state.history(season, before=date).

    `schedule` is the season schedule (state/current/schedule_v1): its dates,
    the Cup final included and postponed games excluded (without_postponed),
    feed rest and back-to-back exactly as training's schedule did. Raises when
    a game is not on `date`, when the history reaches `date` (a leak), or when
    the schedule lacks a target game (rest would be silently NaN)."""
    d = pd.Timestamp(as_date(date))
    off = games.loc[pd.to_datetime(games["game_date"]) != d, "game_id"]
    if len(off):
        raise ValueError(f"games not on {d.date()}: {list(off)[:5]}")
    for name, df in (("events", hist.events), ("team_games", hist.team_games)):
        if len(df) and df["game_date"].max() >= d:
            raise ValueError(f"history {name} reach {df['game_date'].max().date()}, not strictly "
                             f"before {d.date()}: build it with state.history(season, before=date)")
    targets = targets_for(games, roster, season_for_date(d.date()))
    if targets.empty:
        return targets
    sd = pd.concat([hist.schedule_dates, T.schedule_dates(without_postponed(schedule))],
                   ignore_index=True).drop_duplicates()
    have = targets[["team", "game_date"]].drop_duplicates().merge(sd, how="left", indicator=True)
    if (have["_merge"] != "both").any():
        raise ValueError(f"schedule lacks target games for {sorted(have.loc[have['_merge'] != 'both', 'team'])}")
    out = F.build(hist.events, F.make_targets(targets), hist.team_games, sd, hist.priors,
                  lines=None)
    out = out.merge(targets[["person_id", "game_id", "player_name", "two_way"]],
                    on=["person_id", "game_id"], how="left", validate="one_to_one")
    return out.sort_values(["tip_utc", "game_id", "team", "person_id"],
                           kind="mergesort").reset_index(drop=True)


def completeness_by_game(date, schedule: pd.DataFrame, season: str) -> dict[str, list[str]]:
    """{game_id: gaps} for every state game of `season` on `date`, where gaps
    are the game ids of the state games before `date` involving THAT game's
    home or away team which the bot has not ingested (state.ingested_ids,
    overrides included), sorted.

    Every status counts, not only final: a previous game still live (a 10:30
    PM ET tip when a run lands after midnight) or not yet flipped to Final is
    missing from the history just the same, and serving on it would publish
    freezable rows built without it. Postponed games are not state games
    (_state_games), so they never stand as a gap.

    Each team's game sequence comes from the ingested partitions, never from
    the schedule (an un-ingested game would read as an absence), so a
    non-empty list means that game's state is stale: its projections are
    provisional and never freezable. The rule is per game (spec 4: "every
    Final game of BOTH TEAMS before D"): one game that cannot be ingested
    holds back only the games of its own two teams."""
    d = pd.Timestamp(as_date(date))
    s = _state_games(schedule[schedule["season"] == season])
    today = s[s["game_date"] == d]
    prior = s[s["game_date"] < d]
    missing = prior[~prior["game_id"].isin(ingested_ids(season))]
    out = {}
    for gid, home, away in zip(today["game_id"], today["home"], today["away"]):
        mine = missing["home"].isin([home, away]) | missing["away"].isin([home, away])
        out[str(gid)] = sorted(map(str, missing.loc[mine, "game_id"]))
    return out


def provisional_by_game(date, today, schedule: pd.DataFrame, season: str
                        ) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """({game_id: provisional}, {game_id: gaps}) for the state games on
    `date`. A game is provisional (never freezable) when the date is after
    `today` (ET: tomorrow's slate), or when its own gaps
    (completeness_by_game) are non-empty."""
    gaps = completeness_by_game(date, schedule, season)
    later = as_date(date) > as_date(today)
    return {g: later or bool(v) for g, v in gaps.items()}, gaps


def completeness(date, schedule: pd.DataFrame, season: str) -> list[str]:
    """The union of completeness_by_game over the date's games: game ids of
    state games of `season` before `date`, whatever their status, involving a
    team that plays on `date`, which the bot has not ingested. Non-empty means
    some of the date's games are built on stale state (completeness_by_game
    says which)."""
    return sorted({g for gaps in completeness_by_game(date, schedule, season).values() for g in gaps})
