"""Injury status AS OF a serve horizon, for training and serving alike.

Design review (2026-10-08): the site serves and freezes its record HOURS before
tip, so a feature defined from "the last report before tip" is a training-only
source. At tip minus 5 minutes nearly every Questionable has resolved to Out or
Available (measured on 2025-26 samples: Out 247 / Available 66 / nothing else),
so a model trained there would never see "Questionable", and a team that has not
filed yet would look healthy.

So status is snapshotted at fixed HORIZONS per game:
  d1_1730  5:30pm ET the day before (evening boards; the record often freezes here)
  d0_1100  11:00am ET on game day (morning bot run)
  t_120    two hours before tip
  t_15     fifteen minutes before tip
each clipped to at most tip minus 5 minutes. The report used is the latest one
whose TITLE time is strictly before the horizon (never the filename time).

Per (game, team, horizon) the team state is one of:
  listed                 the team has player rows in the report
  all_players_available  the 2018-19 team-level line
  reported_none          the game is in the report and this team filed nobody
  not_yet_submitted      the team-level "NOT YET SUBMITTED" line
  game_not_in_report     the report does not mention the game yet
  no_report              no report within 30 hours of the horizon
  report_unparsed        the newest report failed to parse (never fall back to an older one)
  pre_archive            before 2018-12-17
and only the first three say anything about who is healthy. Everything else is UNKNOWN and is encoded as such; a player absent from
a team that has not filed is never "available".
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd

from .sources import injury as INJ
from .util import ET, UTC, to_et

ARCHIVE_START = dt.date(2018, 12, 17)
TIP_FLOOR = dt.timedelta(minutes=5)
MAX_BACK = dt.timedelta(hours=30)

HORIZONS: dict[str, Callable[[dt.datetime], dt.datetime]] = {
    "d1_1730": lambda tip: dt.datetime.combine(to_et(tip).date() - dt.timedelta(days=1),
                                               dt.time(17, 30), tzinfo=ET),
    "d0_1100": lambda tip: dt.datetime.combine(to_et(tip).date(), dt.time(11, 0), tzinfo=ET),
    "t_120": lambda tip: to_et(tip) - dt.timedelta(minutes=120),
    "t_15": lambda tip: to_et(tip) - dt.timedelta(minutes=15),
}
KNOWN_STATES = {"listed", "all_players_available", "reported_none"}


def cutoff_for(horizon: str, tip: dt.datetime) -> dt.datetime:
    """The horizon's clock time, never later than tip minus 5 minutes (UTC
    arithmetic: two ET datetimes subtract as wall clocks across DST)."""
    h = HORIZONS[horizon](tip).astimezone(UTC)
    floor = to_et(tip).astimezone(UTC) - TIP_FLOOR
    return min(h, floor)


def report_at(cutoff_u: dt.datetime, getter, accept=None):
    """Latest report titled strictly before `cutoff_u` (the injury module's ONE
    selection rule, injury.report_before). Returns (slot, report) or None."""
    sel = INJ.report_before(cutoff_u, getter=getter, accept=accept, max_back=MAX_BACK)
    return None if sel is None else (sel.slot, sel.report)


def snapshot_games(games: pd.DataFrame, schedule: pd.DataFrame, getter=None,
                   horizons=tuple(HORIZONS)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Status snapshots for `games` (game_id, game_date, tip_utc, home, away).

    Returns (player_rows, team_rows):
      player_rows: game_id, horizon, cutoff_utc, report_ts_et, team, player_raw,
                   status, reason_cat, reason_raw
      team_rows:   game_id, horizon, cutoff_utc, report_ts_et, team, state
    `schedule` validates every report row (QA and Summer League PDFs share the
    production bucket), so only real NBA games survive.
    """
    getter = getter or INJ.default_store().get
    sched = schedule[["game_id", "game_date", "away", "home", "tip_utc"]].dropna(subset=["home", "away"])

    def is_real(rep) -> bool:
        # Report-level: QA files (fake games) and Summer League files fail; a
        # report holding only NOT YET SUBMITTED lines for real games passes.
        return INJ.lists_nba_game(rep, sched)

    parsed_cache: dict[str, tuple] = {}

    def validated(slot, rep):
        """(player rows kept by schedule validation, team rows) for a report."""
        if slot.url not in parsed_cache:
            rows = rep.rows
            kept = INJ.validate_rows(rows, sched)[0] if rows is not None and len(rows) else rows
            parsed_cache[slot.url] = (kept, rep.team_rows)
        return parsed_cache[slot.url]

    prow, trow = [], []
    for _, g in games.iterrows():
        tip = pd.Timestamp(g["tip_utc"]).to_pydatetime()
        gdate = g["game_date"] if isinstance(g["game_date"], dt.date) else pd.Timestamp(g["game_date"]).date()
        pair = {g["away"], g["home"]}
        for h in horizons:
            cut = cutoff_for(h, tip)
            base = {"game_id": g["game_id"], "horizon": h, "cutoff_utc": cut}
            if to_et(tip).date() < ARCHIVE_START:
                for team in (g["away"], g["home"]):
                    trow.append({**base, "report_ts_et": pd.NaT, "team": team, "state": "pre_archive"})
                continue
            try:
                found = report_at(cut, getter, accept=is_real)
            except ValueError as e:
                # The newest candidate failed to parse. Never fall back to an
                # older report (that would silently age the status): unknown.
                for team in (g["away"], g["home"]):
                    trow.append({**base, "report_ts_et": pd.NaT, "team": team,
                                 "state": "report_unparsed", "detail": str(e)[:120]})
                continue
            if found is None:
                for team in (g["away"], g["home"]):
                    trow.append({**base, "report_ts_et": pd.NaT, "team": team, "state": "no_report"})
                continue
            slot, rep = found
            kept, teams = validated(slot, rep)
            ts = rep.meta["report_ts_et"]
            gi = kept[kept["game_id"] == g["game_id"]] if kept is not None and len(kept) else None
            tr = None
            if teams is not None and len(teams):
                tr = teams[(pd.to_datetime(teams["game_date"]).dt.date == gdate)
                           & teams["away"].isin(pair) & teams["home"].isin(pair)]
            game_in = (gi is not None and len(gi) > 0) or (tr is not None and len(tr) > 0)
            for team in (g["away"], g["home"]):
                kinds = set(tr.loc[tr["team"] == team, "kind"]) if tr is not None and len(tr) else set()
                pl = gi[gi["team"] == team] if gi is not None and len(gi) else None
                if "not_yet_submitted" in kinds:
                    state = "not_yet_submitted"
                elif pl is not None and len(pl):
                    state = "listed"
                elif "all_players_available" in kinds:
                    state = "all_players_available"
                elif game_in:
                    state = "reported_none"
                else:
                    state = "game_not_in_report"
                if state == "listed":
                    for _, r in pl.iterrows():
                        prow.append({**base, "report_ts_et": ts, "team": team,
                                     "player_raw": r["player_raw"], "status": r["status"],
                                     "reason_cat": r["reason_cat"], "reason_raw": r["reason_raw"]})
                trow.append({**base, "report_ts_et": ts, "team": team, "state": state})
    return pd.DataFrame(prow), pd.DataFrame(trow)
