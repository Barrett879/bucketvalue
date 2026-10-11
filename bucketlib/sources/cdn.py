"""cdn.nba.com: the current-season schedule and roster index, the two feeds the
SITE and the BOT read live. The schedule was reachable from GitHub Actions
(probe of 2026-10-09; the playerIndex was not in that probe), unlike
stats.nba.com, which hangs from Azure.

Traps handled here (all observed live):
  - `gameDateTimeEst` is EASTERN WALL CLOCK with a "Z" suffix
    ("2026-10-20T15:00:00Z" for a 3pm ET tip). Parsing it as UTC is off by 4-5
    hours. Tip times come from `gameDateTimeUTC` only, converted with ZoneInfo.
  - `gameDate` is 'MM/DD/YYYY 00:00:00'. String comparison sorts January first.
  - Cup knockout placeholders carry team id 0 (TBD) until the group stage ends;
    the regular season is listed incomplete (1206 of 1230) until then too, so the
    schedule is always refetched, never cached for the season. Their tip TIME
    is unknown too (gameStatusText 'TBD', gameTimeEst '0001-01-01...') while
    gameDateTimeUTC carries a made-up 05:00Z: `time_tbd` flags it, and
    nothing may print or use that tip time.
  - `postponed` is set on positive evidence only ('Y', or the status text
    PPD/Postponed). The stats schedules of 2014-18, 2021-22 and 2023-24 carry
    postponedStatus 'A' on every game, so "anything but N" would drop whole
    seasons; an unrecognised value is listed in
    df.attrs['postponed_status_unknown'] and never excludes a game. Whether
    cdn keeps a postponed game on its original date until it is rescheduled
    was not observed live (no 2026-27 game postponed yet); the serve path
    drops postponed rows either way (serve.without_postponed).
  - Preseason includes non-NBA opponents (London Lions 'LON'). Those games are
    dropped and COUNTED, never mapped onto an NBA team.
  - staticData/playerIndex.json is the same table as stats.nba.com's playerindex
    (615 players, identical teams and two-way flags on 2026-10-08, compared
    from Barrett's Mac), so the bot can refresh rosters without the blocked
    host. SUPPLEMENTAL_STATUS == 1 is the two-way flag. Preseason it holds camp
    rosters (18-21 per team). Its reachability FROM ACTIONS was not in the
    2026-10-09 probe (scripts/probe_hosts.py now lists it); the schedule on
    the same host and path was.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import http as H
from .. import teams
from ..util import UTC, et_date_of, to_et

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
PLAYER_INDEX_URL = "https://cdn.nba.com/static/json/staticData/playerIndex.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

# Game id prefix -> type. One table, shared with GAME_UNIVERSE in the spec.
GAME_TYPES = {"001": "PRE", "002": "RS", "003": "ALLSTAR", "004": "PO", "005": "PI",
              "006": "CUP_FINAL"}
STATUS = {1: "scheduled", 2: "live", 3: "final"}
# A playerIndex with fewer rostered players is a broken pull, never "the
# league emptied" (it held 615 on 2026-10-08). The one threshold: the bot
# refuses to commit such a pull (scripts/daily_update.step_roster) and the
# app's live layer treats it as a failed fetch (live.roster). Both read it at
# call time, so a test patches it here.
MIN_ROSTER = 400


def fetch_schedule() -> dict:
    return H.fetch(SCHEDULE_URL, timeout=30)


def fetch_player_index() -> dict:
    return H.fetch(PLAYER_INDEX_URL, timeout=30)


def parse_schedule(payload: dict) -> pd.DataFrame:
    """One row per game. Returns games for NBA teams (and TBD placeholders);
    non-NBA opponents are dropped and counted in df.attrs['dropped_non_nba']."""
    ls = payload.get("leagueSchedule")
    if not isinstance(ls, dict) or "gameDates" not in ls:
        raise H.BadPayload(SCHEDULE_URL, "no leagueSchedule.gameDates")
    season = ls.get("seasonYear")
    rows, dropped, unknown_ppd = [], [], []
    for day in ls["gameDates"]:
        for g in day.get("games", []):
            gid = str(g["gameId"])
            tip_utc = dt.datetime.fromisoformat(g["gameDateTimeUTC"].replace("Z", "+00:00"))
            if tip_utc.tzinfo is None:
                tip_utc = tip_utc.replace(tzinfo=UTC)
            sides = {}
            tbd = False
            for side in ("homeTeam", "awayTeam"):
                tid = g[side].get("teamId") or 0
                if int(tid) == 0:
                    sides[side] = None
                    tbd = True
                    continue
                code = teams.from_team_id(tid)
                if code is None:
                    sides[side] = "NON_NBA"
                else:
                    sides[side] = code
            if "NON_NBA" in sides.values():
                dropped.append(gid)
                continue
            nat = (g.get("broadcasters") or {}).get("nationalTvBroadcasters") or []
            rows.append({
                "game_id": gid,
                "season": season,
                "game_type": GAME_TYPES.get(gid[:3], "OTHER"),
                # ET date of the tip. gameDateEst's date part is the same thing
                # (asserted below) but it is a wall clock pretending to be UTC.
                "game_date": et_date_of(tip_utc),
                "tip_utc": tip_utc,
                "tip_et": to_et(tip_utc),
                "status": STATUS.get(int(g.get("gameStatus") or 0), "unknown"),
                "status_text": g.get("gameStatusText", ""),
                "home": sides["homeTeam"],
                "away": sides["awayTeam"],
                "is_tbd": tbd,
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
                "arena": g.get("arenaName", ""),
                "city": g.get("arenaCity", ""),
                "is_neutral": bool(g.get("isNeutral")),
                "label": g.get("gameLabel", ""),
                "subtype": g.get("gameSubtype", ""),
                "postponed": _postponed(g, unknown_ppd),
                # The tip TIME is unknown (Cup knockout placeholders): the feed
                # says 'TBD' and gameTimeEst is 0001-01-01, while
                # gameDateTimeUTC holds a made-up 05:00Z (midnight ET). The
                # date stands; the tip time must never be shown or used.
                "time_tbd": (str(g.get("gameStatusText") or "").strip().upper() == "TBD"
                             or str(g.get("gameTimeEst") or "").startswith("0001")),
                "national_tv": ", ".join(b.get("broadcasterDisplay", "") for b in nat
                                         if b.get("broadcasterDisplay")),
                "_date_est": str(g.get("gameDateEst", ""))[:10],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df.attrs["dropped_non_nba"] = dropped
        df.attrs["postponed_status_unknown"] = unknown_ppd
        return df
    # The wall-clock field and the UTC-derived ET date must agree. If they do
    # not, one of the two parses is wrong and every rest/B2B feature would be.
    bad = df[df["_date_est"] != df["game_date"].map(lambda d: d.isoformat())]
    if len(bad):
        raise ValueError(f"ET date mismatch for {len(bad)} games, e.g. {bad.iloc[0].to_dict()}")
    if df["game_id"].duplicated().any():
        raise ValueError("duplicate game ids in schedule")
    df = df.drop(columns="_date_est").sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    df.attrs["dropped_non_nba"] = dropped
    df.attrs["postponed_status_unknown"] = unknown_ppd
    return df


# postponedStatus values that say nothing about a postponement: 'N' on the
# live 2026-27 feed, 'A' on EVERY game of the 2014-18, 2021-22 and 2023-24
# stats schedules (the same scheduleleaguev2 shape), and blank.
NOT_POSTPONED = {"N", "A", ""}
POSTPONED_TEXT = {"PPD", "POSTPONED"}


def _postponed(g: dict, unknown: list) -> bool:
    """True only on positive evidence: postponedStatus 'Y', or the status text
    'PPD' / 'Postponed'. Any other postponedStatus value is recorded in
    `unknown` (game id, value) and is never an exclusion: excluding on an
    unrecognised value would drop whole seasons ('A')."""
    code = str(g.get("postponedStatus") or "").strip().upper()
    text = str(g.get("gameStatusText") or "").strip().upper()
    if code == "Y" or text in POSTPONED_TEXT:
        return True
    if code not in NOT_POSTPONED:
        unknown.append((str(g.get("gameId")), g.get("postponedStatus")))
    return False


def _height_in(h) -> float:
    try:
        f, i = str(h).split("-")
        return int(f) * 12 + int(i)
    except (ValueError, AttributeError):
        return float("nan")


def parse_player_index(payload: dict) -> pd.DataFrame:
    """Current rostered players: one row per person_id."""
    df = H.result_sets(payload)["PlayerIndex"]
    n0 = len(df)
    df = df[(df["ROSTER_STATUS"] == 1) & df["TEAM_ID"].notna()].copy()
    out = pd.DataFrame({
        "person_id": df["PERSON_ID"].astype("int64"),
        "first_name": df["PLAYER_FIRST_NAME"].fillna(""),
        "last_name": df["PLAYER_LAST_NAME"].fillna(""),
        "team": df["TEAM_ID"].map(teams.from_team_id),
        "jersey": df["JERSEY_NUMBER"].fillna("").astype(str),
        "position": df["POSITION"].fillna(""),
        "height_in": df["HEIGHT"].map(_height_in),
        "weight": pd.to_numeric(df["WEIGHT"], errors="coerce"),
        "draft_year": pd.to_numeric(df["DRAFT_YEAR"], errors="coerce"),
        "draft_round": pd.to_numeric(df["DRAFT_ROUND"], errors="coerce"),
        "draft_number": pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce"),
        "from_year": pd.to_numeric(df["FROM_YEAR"], errors="coerce"),
        "two_way": df["SUPPLEMENTAL_STATUS"].fillna(0).astype(int) == 1,
    })
    if out["team"].isna().any():
        raise ValueError(f"unknown TEAM_ID in player index: {df.loc[out['team'].isna().values, 'TEAM_ID'].unique()[:5]}")
    if out["person_id"].duplicated().any():
        raise ValueError("duplicate PERSON_ID in player index")
    # Display name as the NBA prints it: first + last. Family-name-first players
    # (Yang Hansen) are stored first='Yang', last='Hansen', so this is correct.
    out["name"] = (out["first_name"] + " " + out["last_name"]).str.strip()
    out = out.reset_index(drop=True)
    out.attrs["dropped_off_roster"] = n0 - len(df)
    return out
