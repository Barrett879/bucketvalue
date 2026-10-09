"""stats.nba.com bulk endpoints -> tidy per-game tables keyed on PERSON_ID/GAME_ID.

One whole-season call per endpoint, never per-game calls. The research pass
(2026-10-08, live) measured every trap handled below; the short version:

  - THREE sources for one player-game, each wrong in its own way.
    PlayerGameLogs (PGL) Base has the precise float MIN and PFD. LeagueGameLog
    (LGL) P has an int MIN (round(seconds/60)); PGL Advanced rounds MIN to one
    decimal. Stats and minutes come from PGL Base, the advanced rates from PGL
    Advanced, and LGL P is a cross-check only.
    PGL IS THE AUTHORITY ON WHO APPEARED. LGL P sometimes carries an all-zero
    0:00 row PGL lacks, and both checked were players who did NOT play: Dennis
    Schroder, LAL, 0022200140 (cdn box: INACTIVE, injury) and Markieff Morris,
    MIA, 0042100105 (cdn box: played 0). PGL does keep real 0-second
    appearances (Justin James, SAC, 0021900464: one turnover, 0 sec), so an
    LGL-only row is not "an appearance PGL dropped". LGL-only all-zero rows
    are EXCLUDED and counted; one carrying minutes or a stat raises. The LGL P
    call is not even stable: a 19:04 pull carried the Schroder row, a 20:13
    pull the same evening did not (25895 vs 25894), so adding its rows made a
    game gain or lose an "appearance" depending on when it was pulled.
  - PGL and LGL DISAGREE on some box stats after a stat correction (2024-25:
    OREB/REB in 34 rows, PLUS_MINUS in 8; 2014-15: PLUS_MINUS in 280), and each
    source's fantasy column is consistent with its OWN stats. PGL wins: its
    player plus-minus sums to exactly 5x the team margin in every team-game of
    2014-15, 2019-20, 2022-23 and 2024-25, LGL's does not. Disagreements are
    counted, never raised; the fantasy check compares each row with its own
    source.
  - Neutral sites are labelled three ways. From 2024-25 the international and
    Cup Las Vegas games put '@' on BOTH sides (no home team) and the schedule
    says isNeutral. The 2019-20 Orlando bubble kept 'vs.'/'@' with isNeutral
    false. And EVERY earlier neutral site (London 2015-19, Mexico City, Paris
    2020/2023/2024, the 2023-24 Cup semifinals and final in Las Vegas) is
    labelled vs./@ in the logs with isNeutral false. So neutral is derived from
    the schedule's arenaCity against the home team's own cities
    (_HOME_CITIES): that rule reproduces the source's own isNeutral on every
    2024-25..2026-27 game, where the source does label them, including the
    Spurs' Austin games, which are HOME games (same-state second market).
    player_games/team_games read the season's schedule for it.
  - A cancelled game (0021201214, BOS-IND 2013-04-16, after the Boston
    Marathon bombing) sits in LGL T with MIN 0 and no result, and in no other
    log. KNOWN_CANCELLED_GAMES excludes and counts it; an unlisted one raises.
  - PGL's DD2 flag is computed BEFORE corrections (Jalen Suggs 0022401201: REB 10
    and PTS 32 but DD2 0), so dd2/td3 are recomputed from the stats we keep.
  - MIN means something different in every endpoint: LGL T MIN is team
    PLAYER-minutes (240 + 25 per OT period), TeamGameLogs MIN is GAME minutes
    (48 + 5 per OT). Overtime is derived from LGL T only.
  - GAME_DATE is 'YYYY-MM-DD' in LGL and 'YYYY-MM-DDT00:00:00' in PGL. Both are
    the US/Eastern game date; they are parsed with an explicit format each.
  - An EMPTY result (HTTP 200, full headers, zero rows) is the normal state of a
    season type whose first scheduled game has not been played (the schedule
    says when). Any later it is an outage or a format change and raises, and
    an empty answer never replaces a non-empty raw payload. A hang/timeout
    surfaces as http.Blocked and is never read as "no games".
  - The NBA Cup final (GAME_ID prefix 006) is in no Regular Season or Playoffs
    log, but two teams play it: schedule() keeps it so rest/back-to-back
    features can see it.
  - AVAILABLE_FLAG / VIDEO_AVAILABLE is a VIDEO flag (value 2 on whole games in
    2016-17 and 2019-20), not player availability. It is not carried.
  - LeagueDashPlayerStats labels TEAM_ID with the player's LATER team. Nothing
    here uses it; team attribution comes from the per-game TEAM_ID in the logs.
  - playerindex for a PAST season is not that season's roster (it lists the
    players whose last season was then) and CommonTeamRoster for a past season
    is the END-of-season roster; neither gives point-in-time team. A past
    roster's HOW_ACQUIRED is TODAY's record ('Traded from ATL on 07/06/25' on
    Clint Capela's 2018-19 HOU row), so it is withheld for past seasons.

Fetching: every raw payload is cached under cache/raw/statsnba/<season>/ as a
stamped, gzipped JSON envelope the moment it arrives, so an interrupted backfill
resumes without re-calling. A completed season is fetched once; the current
season is refetched (see `_want_fetch`); refresh=False is CACHE-ONLY. Tidy
tables carry the OLDEST raw fetched_at they were built from, never the time
they were written.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
import pandas as pd

from .. import http, teams, util
from ..cache import (CACHE_DIR, read_json_stamped, read_parquet_meta, stamp_now, write_json,
                     write_parquet)

SOURCE = "statsnba"

# ── Season types ─────────────────────────────────────────────────────────────
# API value -> stored code. 'Play In' (with a space) is an HTTP 400, which
# nba_api surfaces as KeyError('resultSet'): the valid value is 'PlayIn'.
SEASON_TYPES = {"Regular Season": "RS", "PlayIn": "PI", "Playoffs": "PO"}
_CODE_TO_API = {v: k for k, v in SEASON_TYPES.items()}
# GAME_ID = '00' + type digit + 2-digit season start year + 5-digit sequence.
GAME_ID_PREFIX = {"RS": "002", "PI": "005", "PO": "004"}
# LGL SEASON_ID = type digit + start year ('22024').
_SEASON_ID_DIGIT = {"RS": "2", "PI": "5", "PO": "4"}

SCHEDULE_GAME_TYPES = {"001": "PRE", "002": "RS", "003": "ALLSTAR", "004": "PO",
                       "005": "PI", "006": "CUP_FINAL"}
# The Cup final got its own 006 game id starting 2023-24.
FIRST_CUP_FINAL_SEASON = "2023-24"
# NBA franchise ids are 1610612737..1610612766. Anything else on a schedule is a
# non-NBA club (London Lions 50015, EuroLeague/G League sides in preseason).
_NBA_TEAM_ID_RANGE = range(1610612737, 1610612767)

# The 2019-20 restart: every game from 2020-07-30 (88 RS seeding games, the
# play-in game and the playoffs) was played in the Orlando bubble, but the
# logs still label one side 'vs.' and the schedule's isNeutral is false for
# all of them. They are flagged neutral here so home-court features see the
# truth; `home` keeps the designated side.
BUBBLE_SEASON, BUBBLE_START = "2019-20", pd.Timestamp("2020-07-30")


def is_bubble(season: str, game_date) -> "pd.Series | bool":
    if season != BUBBLE_SEASON:
        return False if not isinstance(game_date, pd.Series) else pd.Series(False, index=game_date.index)
    return game_date >= BUBBLE_START


# Regular-season games actually played, where it is not 30*82/2. 2011-12 was
# the 66-game lockout season, 2012-13 lost one cancelled game (below), 2019-20
# was cut short (1059 incl. 88 bubble seeding games), 2020-21 was 72 a team.
_RS_GAMES = {"2011-12": 990, "2012-13": 1229, "2019-20": 1059, "2020-21": 1080}

# The play-in began in 2020-21; 2019-20 had one bubble play-in game, filed
# under PlayIn. Earlier PlayIn pulls are empty by definition, so they are not
# made (2017-18 PI cost 5 calls that all returned 0 rows).
FIRST_PLAYIN_SEASON = "2019-20"

# Games that were scheduled and never played. LGL T keeps both sides with MIN
# 0, PTS 0 and no WL, which fails the overtime check and with it the whole
# season. Each is excluded from every log and counted; a listed game that
# carries a result, or player rows, raises (the list would be wrong).
KNOWN_CANCELLED_GAMES: dict[str, str] = {
    "0021201214": "BOS vs IND 2013-04-16, cancelled after the Boston Marathon bombing "
                  "and never made up (absent from the 2012-13 schedule: 1229 RS games)",
}

# Every city a team's HOME games were played in, as the schedule spells
# arenaCity (all 2011-12..2026-27 schedules, read 2026-10-08). A home game in
# any other city is a neutral site. Austin (Spurs) and Tampa (Raptors 2020-21)
# are home cities: the source's own isNeutral, where it exists (2024-25 on),
# is false for the Austin games, and Tampa was the Raptors' home all season.
_HOME_CITIES: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in {
    "ATL": {"Atlanta"}, "BOS": {"Boston"}, "BKN": {"Brooklyn", "Newark"},
    "CHA": {"Charlotte"}, "CHI": {"Chicago"}, "CLE": {"Cleveland"}, "DAL": {"Dallas"},
    "DEN": {"Denver"}, "DET": {"Detroit", "Auburn Hills"},
    "GSW": {"Oakland", "San Francisco"}, "HOU": {"Houston"}, "IND": {"Indianapolis"},
    "LAC": {"Los Angeles", "Inglewood"}, "LAL": {"Los Angeles"}, "MEM": {"Memphis"},
    "MIA": {"Miami"}, "MIL": {"Milwaukee"}, "MIN": {"Minneapolis"},
    "NOP": {"New Orleans"}, "NYK": {"New York"}, "OKC": {"Oklahoma City"},
    "ORL": {"Orlando"}, "PHI": {"Philadelphia"}, "PHX": {"Phoenix"}, "POR": {"Portland"},
    "SAC": {"Sacramento"}, "SAS": {"San Antonio", "Austin"},
    "TOR": {"Toronto", "Tampa"}, "UTA": {"Salt Lake City"}, "WAS": {"Washington"},
}.items()}
# Most competitive home games one team has played outside its own cities in a
# season (bubble aside) is 2 (PHX 2016-17 and BKN 2017-18 in Mexico City, the
# Cup semifinal + final in Las Vegas). Many more means an arena city spelled a
# new way, which would silently mark real home games neutral.
MAX_RELOCATED_HOME_GAMES = 6

# Known team-games whose recorded player minutes sum off by more than the
# tolerance. Empty on purpose: a violation raises until someone has looked at
# the box score and recorded the reason here (game_id -> reason).
KNOWN_MINUTES_EXCEPTIONS: dict[str, str] = {}
MINUTES_TOLERANCE = 2.0  # 2024-25 max was exactly 2.0 (0022400330), 2014-15 0.4

COUNT_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
               "ast", "stl", "blk", "tov", "pf"]
# Raw per-game advanced rates. They EXPLODE at tiny minutes (PACE 14400,
# USG_PCT 1.0 at under 1.6 min), so consumers must minutes-weight them.
# Scales differ: TM_TOV_PCT is 0-100 (3.3), the other *_PCT are 0-1.
ADV_MAP = {"USG_PCT": "usg_pct", "E_USG_PCT": "e_usg_pct", "PACE": "pace",
           "E_PACE": "e_pace", "POSS": "poss", "TS_PCT": "ts_pct", "AST_PCT": "ast_pct",
           "OREB_PCT": "oreb_pct", "DREB_PCT": "dreb_pct", "TM_TOV_PCT": "tm_tov_pct",
           "PIE": "pie", "OFF_RATING": "off_rating", "DEF_RATING": "def_rating"}

PLAYER_GAME_COLUMNS = (
    ["season", "season_type", "game_id", "game_date", "person_id", "player_name",
     "team", "opp", "home", "neutral", "min", "sec"]
    + COUNT_STATS + ["pfd", "pts", "plus_minus", "fantasy", "dd2", "td3"]
    + list(ADV_MAP.values()) + ["src"])

TEAM_GAME_COLUMNS = (
    ["season", "season_type", "game_id", "game_date", "team", "opp", "home", "neutral", "pts",
     "opp_pts", "margin", "ot_periods"] + COUNT_STATS
    + ["pace", "poss", "off_rating", "def_rating", "net_rating"])

SCHEDULE_COLUMNS = ["game_id", "season", "game_type", "game_date", "tip_utc", "home",
                    "away", "is_tbd", "label", "sublabel", "subtype", "is_neutral",
                    "arena_name", "arena_city", "postponed_status", "national_tv",
                    "game_status", "if_necessary"]

PLAYER_INDEX_COLUMNS = ["season", "person_id", "first_name", "last_name", "display_name",
                        "team", "position", "height_in", "weight", "draft_year",
                        "draft_round", "draft_number", "from_year", "two_way"]

ROSTER_COLUMNS = ["season", "team", "person_id", "player_name", "position", "height_in",
                  "weight", "birth_date", "age", "exp", "school", "how_acquired", "two_way"]

_LGL_P_REQ = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
              "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A",
              "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
              "PTS", "PLUS_MINUS", "FANTASY_PTS"]
_LGL_T_REQ = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
              "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB",
              "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]
_PGL_BASE_REQ = ["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
                 "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A",
                 "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF",
                 "PFD", "PTS", "PLUS_MINUS", "NBA_FANTASY_PTS", "DD2", "TD3"]
_PGL_ADV_REQ = ["SEASON_YEAR", "PLAYER_ID", "TEAM_ID", "GAME_ID", "MIN"] + list(ADV_MAP)
_TGL_ADV_REQ = ["SEASON_YEAR", "TEAM_ID", "GAME_ID", "MIN", "PACE", "POSS", "OFF_RATING",
                "DEF_RATING", "NET_RATING"]
_PINDEX_REQ = ["PERSON_ID", "PLAYER_LAST_NAME", "PLAYER_FIRST_NAME", "TEAM_ID",
               "TEAM_ABBREVIATION", "POSITION", "HEIGHT", "WEIGHT", "DRAFT_YEAR",
               "DRAFT_ROUND", "DRAFT_NUMBER", "ROSTER_STATUS", "FROM_YEAR",
               "SUPPLEMENTAL_STATUS"]
_ROSTER_REQ = ["TeamID", "SEASON", "PLAYER", "POSITION", "HEIGHT", "WEIGHT", "BIRTH_DATE",
               "AGE", "EXP", "PLAYER_ID"]


# ── Season-type / season helpers ─────────────────────────────────────────────
def stype_code(season_type: str) -> str:
    """'Regular Season' | 'PlayIn' | 'Playoffs' (or RS/PI/PO) -> 'RS'|'PI'|'PO'."""
    if season_type in SEASON_TYPES:
        return SEASON_TYPES[season_type]
    if season_type in _CODE_TO_API:
        return season_type
    raise ValueError(f"unknown season type {season_type!r}: use one of "
                     f"{list(SEASON_TYPES)} or {list(_CODE_TO_API)}")


def stype_api(season_type: str) -> str:
    return _CODE_TO_API[stype_code(season_type)]


def current_season() -> str:
    return util.season_for_date(util.today_et())


def is_live_season(season: str) -> bool:
    """True for the current season and any later one: their data still grows."""
    return util.season_end_year(season) >= util.season_end_year(current_season())


def has_play_in(season: str) -> bool:
    return util.season_end_year(season) >= util.season_end_year(FIRST_PLAYIN_SEASON)


def first_game_date(season: str, season_type: str, refresh: bool | None = None) -> dt.date | None:
    """ET date of the first scheduled game of this season type (None when the
    schedule lists none yet, e.g. the playoffs of a season in October)."""
    code = stype_code(season_type)
    sch = _schedule_frame(season, refresh)
    d = sch.loc[sch["game_type"] == code, "game_date"]
    return None if d.empty else d.min().date()


def empty_allowed(season: str, season_type: str, refresh: bool | None = None) -> bool:
    """Can a whole-season pull legitimately return zero rows?

    Only while no game of that type can have been played:
      - a season after the current one: yes, nothing is scheduled;
      - the current season: only through the ET date of the first scheduled
        game of that type (opening night itself may not be final yet). The
        schedule is the authority; it is read with the same `refresh`.
        Measured trap: 'empty is fine for the live season' accepted an empty
        pull on any day of it, and wrote 0-row tables in January;
      - a completed season: only PlayIn before 2019-20 (there was none).
    """
    code = stype_code(season_type)
    if code == "PI" and not has_play_in(season):
        return True
    if not is_live_season(season):
        return False
    if util.season_end_year(season) > util.season_end_year(current_season()):
        return True
    first = first_game_date(season, code, refresh)
    return first is None or util.today_et() <= first


# ── Raw cache + fetch ────────────────────────────────────────────────────────
_calls = 0
_fetched_this_run: set[Path] = set()
# fetched_at of every raw payload this process has read or fetched, so a tidy
# table can carry the age of its INPUTS (a --rebuild of week-old payloads is
# week-old data, whatever the write time says).
_stamps: dict[Path, float | None] = {}


_call_budget: int | None = None


class CallBudgetExceeded(RuntimeError):
    """The process hit the live-call budget set by set_call_budget()."""


class CacheMiss(LookupError):
    """refresh=False (cache-only, e.g. --rebuild) and the raw payload is not
    cached. Never a network call, never 'no data'."""


def call_count() -> int:
    """Live stats.nba.com requests made by this process (for budgets/reports)."""
    return _calls


def set_call_budget(n: int | None) -> None:
    """Refuse any live call beyond `n` in this process (None: unlimited).

    A cached payload costs nothing, so a budget only bites on uncached or live
    payloads; it exists because a multi-type run silently fetches every type
    that is not cached yet (30 calls in development, where 0 were expected).
    """
    global _call_budget
    _call_budget = n


LOG_PAYLOADS = ("pgl_Base", "pgl_Advanced", "lgl_P", "lgl_T", "tgl_Advanced")


def planned_log_calls(seasons: list[str], types: list[str], force: bool = False,
                      rebuild: bool = False) -> int:
    """Upper bound on the live calls build_logs will make: one per raw payload
    (the five logs per type, plus the season's schedule, which supplies the
    neutral sites and the first game dates) that is missing, or belongs to a
    live season, or --force. --rebuild is cache-only: 0. PlayIn before
    2019-20 is never called."""
    if rebuild:
        return 0
    n = 0
    for season in seasons:
        names = []
        for t in types:
            code = stype_code(t)
            if code == "PI" and not has_play_in(season):
                continue
            names += [f"{p}_{code}" for p in LOG_PAYLOADS]
        if names:
            names.append("schedule")
        n += sum(1 for name in names
                 if force or is_live_season(season) or not raw_path(season, name).exists())
    return n


def raw_dir(season: str) -> Path:
    util.season_end_year(season)  # validates the label before it names a dir
    return CACHE_DIR / "raw" / SOURCE / season


def raw_path(season: str, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError(f"bad raw payload name {name!r}")
    return raw_dir(season) / f"{name}.json.gz"


def _want_fetch(path: Path, season: str, refresh: bool | None) -> bool:
    """Decide whether to call the network for one raw payload.

    refresh=True  -> fetch, unless this process already fetched it (player_games
                     and team_games both need LGL T; one run calls once).
    refresh=False -> CACHE-ONLY: never fetch; a missing payload raises
                     CacheMiss. (It used to fetch what was missing, so
                     `--rebuild`, documented as "no calls", spent 30 calls.)
    refresh=None  -> fetch when missing; for a LIVE season also when this process
                     has not fetched it yet (the current season's logs and its
                     schedule grow daily, and the 2026-27 schedule lists only
                     1206 of 1230 RS games until the Cup group stage ends).
    """
    if refresh is False:
        if not path.exists():
            raise CacheMiss(f"{path.relative_to(CACHE_DIR)} is not cached and refresh=False "
                            f"(cache-only) forbids fetching it")
        return False
    if not path.exists():
        return True
    if path in _fetched_this_run:
        return False
    if refresh is None:
        return is_live_season(season)
    return bool(refresh)


def _payload_rows(payload: dict) -> int | None:
    """Data rows in a payload (None when its shape is not one we know)."""
    if isinstance(payload.get("leagueSchedule"), dict):
        return sum(len(gd.get("games") or []) for gd in payload["leagueSchedule"].get("gameDates") or [])
    sets = payload.get("resultSets", payload.get("resultSet"))
    if isinstance(sets, dict):
        sets = [sets] if "rowSet" in sets else list(sets.values())
    if isinstance(sets, list) and all(isinstance(x, dict) and "headers" in x for x in sets):
        return sum(len(x.get("rowSet") or []) for x in sets)
    return None


def _read_cached(path: Path) -> dict:
    data, fetched_at = read_json_stamped(path)
    if not isinstance(data, dict) or "payload" not in data:
        raise http.BadPayload(str(path), "cached envelope has no 'payload'")
    _stamps[path] = fetched_at
    return data["payload"]


def oldest_stamp(*stamps) -> float | None:
    """The oldest of several fetched_at stamps (None entries ignored)."""
    vals = [float(s) for s in stamps if s is not None]
    return min(vals) if vals else None


def _stamp_of(season: str, *names: str) -> float | None:
    """Oldest fetched_at among raw payloads this process has read/fetched."""
    return oldest_stamp(*(_stamps.get(raw_path(season, n)) for n in names))


def _get_raw(key_season: str, name: str, endpoint_cls, refresh: bool | None,
             empty_ok: "bool | Callable[[], bool]" = False, **kwargs) -> dict:
    """Raw stats.nba.com JSON for one call, from the cache or the network.

    Failure classes from bucketlib.http propagate with context added. Blocked is
    re-raised as Blocked: a hang is never "no games". A payload of an unknown
    shape, or an EMPTY one where empty is not legitimate (`empty_ok`, a bool
    or a callable evaluated only when the answer is empty, since deciding it
    reads the schedule), raises BadPayload and is NOT cached, so a rerun asks
    again instead of resuming from a bad copy. An empty answer never replaces
    a cached payload that has rows (stale beats empty, at the raw level).
    """
    global _calls
    path = raw_path(key_season, name)
    if not _want_fetch(path, key_season, refresh):
        return _read_cached(path)
    params = http.stats_params(endpoint_cls, **kwargs)
    if _call_budget is not None and _calls >= _call_budget:
        raise CallBudgetExceeded(f"live-call budget {_call_budget} reached before "
                                 f"{endpoint_cls.endpoint} {key_season} {name}")
    _calls += 1
    try:
        payload = http.stats_get(endpoint_cls.endpoint, params)
    except http.FetchError as e:
        raise type(e)(e.url, f"stats.nba.com {endpoint_cls.endpoint} {key_season} {name}: "
                             f"{e.detail}") from e
    if not isinstance(payload, dict):
        raise http.BadPayload(endpoint_cls.endpoint, f"expected a JSON object, got "
                                                     f"{type(payload).__name__}")
    n = _payload_rows(payload)
    if n is None:
        raise http.BadPayload(endpoint_cls.endpoint, f"unrecognised payload shape, keys "
                                                     f"{list(payload)[:6]} (not cached)")
    if n == 0:
        if not (empty_ok() if callable(empty_ok) else empty_ok):
            raise http.BadPayload(endpoint_cls.endpoint, f"{key_season} {name}: 0 rows where "
                                                         f"data must exist (not cached; outage or "
                                                         f"format change, never 'no games')")
        if path.exists():
            old = _payload_rows(_read_cached(path)) or 0
            if old:
                raise http.BadPayload(endpoint_cls.endpoint,
                                      f"{key_season} {name}: an empty answer would replace "
                                      f"{old} cached rows; kept the cache (stale beats empty)")
    payload = _compact(payload)
    write_json(path, {"endpoint": endpoint_cls.endpoint, "params": params,
                      "payload": payload}, stamped=True, gz=True)
    # write_json stamps the envelope itself; this is the same clock read a
    # moment later (re-reading a 13 MB payload for its own stamp is not worth it).
    _stamps[path] = stamp_now()
    _fetched_this_run.add(path)
    return payload


def _compact(payload: dict) -> dict:
    """Drop the *_RANK columns (31 of 72 in PGL Base) before caching.

    They are league ranks of each row within the pull, recomputable and unused,
    and they are the least compressible part of a 13 MB Advanced payload. The
    disk is nearly full; everything else is kept verbatim.
    """
    sets = payload.get("resultSets")
    if not isinstance(sets, list):
        return payload
    out = []
    for s in sets:
        hdr = s.get("headers")
        if not isinstance(hdr, list) or not any(str(h).endswith("_RANK") for h in hdr):
            out.append(s)
            continue
        keep = [i for i, h in enumerate(hdr) if not str(h).endswith("_RANK")]
        out.append(dict(s, headers=[hdr[i] for i in keep],
                        rowSet=[[r[i] for i in keep] for r in s.get("rowSet", [])]))
    return dict(payload, resultSets=out)


def _frame(payload: dict, set_name: str, required: list[str], what: str) -> pd.DataFrame:
    """One named result set as a DataFrame, with required headers asserted.

    Required columns are checked even on an empty set: 'empty with the right
    headers' is legitimate, 'missing headers' is a format change.
    """
    try:
        sets = http.result_sets(payload)
    except KeyError as e:
        raise http.BadPayload(what, f"no result sets: {e}") from None
    if set_name not in sets:
        raise http.BadPayload(what, f"result set {set_name!r} missing; have {list(sets)}")
    df = sets[set_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise http.BadPayload(what, f"{set_name} lacks columns {missing}")
    return df


# Endpoint imports are local to the fetchers: nba_api is only needed to build
# parameter dicts, and the parsers must stay importable without it.
def fetch_lgl(season: str, season_type: str, kind: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import leaguegamelog
    if kind not in ("P", "T"):
        raise ValueError(f"LeagueGameLog kind must be 'P' or 'T', not {kind!r}")
    code = stype_code(season_type)
    return _get_raw(season, f"lgl_{kind}_{code}", leaguegamelog.LeagueGameLog, refresh,
                    empty_ok=lambda: empty_allowed(season, code, refresh), season=season,
                    player_or_team_abbreviation=kind, season_type_all_star=stype_api(code))


def fetch_pgl(season: str, season_type: str, measure: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import playergamelogs
    code = stype_code(season_type)
    # measure_type must be explicit: http.stats_params refuses the None default
    # (curl_cffi would send 'None' -> HTTP 400).
    return _get_raw(season, f"pgl_{measure}_{code}", playergamelogs.PlayerGameLogs, refresh,
                    empty_ok=lambda: empty_allowed(season, code, refresh), season_nullable=season,
                    season_type_nullable=stype_api(code),
                    measure_type_player_game_logs_nullable=measure)


def fetch_tgl(season: str, season_type: str, measure: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import teamgamelogs
    code = stype_code(season_type)
    return _get_raw(season, f"tgl_{measure}_{code}", teamgamelogs.TeamGameLogs, refresh,
                    empty_ok=lambda: empty_allowed(season, code, refresh), season_nullable=season,
                    season_type_nullable=stype_api(code),
                    measure_type_player_game_logs_nullable=measure)


def fetch_schedule(season: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import scheduleleaguev2
    return _get_raw(season, "schedule", scheduleleaguev2.ScheduleLeagueV2, refresh,
                    season=season, league_id="00")


def fetch_player_index(season: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import playerindex
    return _get_raw(season, "playerindex", playerindex.PlayerIndex, refresh,
                    season=season, league_id="00", historical_nullable=0,
                    team_id_nullable=0)


def fetch_team_roster(season: str, team: str, refresh: bool | None = None) -> dict:
    from nba_api.stats.endpoints import commonteamroster
    t = teams.get(team)
    return _get_raw(season, f"roster_{t.code}", commonteamroster.CommonTeamRoster, refresh,
                    team_id=t.team_id, season=season, league_id_nullable="00")


# ── Shared field parsers ─────────────────────────────────────────────────────
def _parse_game_dates(s: pd.Series, fmt: str, what: str) -> pd.Series:
    """Parse with ONE explicit format; anything else raises (never coerces)."""
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    bad = s[out.isna()]
    if len(bad):
        raise ValueError(f"{what}: {len(bad)} GAME_DATE values not in {fmt!r}, "
                         f"e.g. {bad.iloc[:3].tolist()}")
    if (out != out.dt.normalize()).any():
        raise ValueError(f"{what}: GAME_DATE carries a time of day; expected a date")
    return out.astype("datetime64[ns]")


def _check_game_ids(gid: pd.Series, season: str, code: str, what: str) -> None:
    gid = gid.astype(str)
    bad_len = gid[gid.str.len() != 10]
    if len(bad_len):
        raise ValueError(f"{what}: GAME_ID not 10 chars: {bad_len.iloc[:3].tolist()}")
    want = GAME_ID_PREFIX[code] + f"{util.season_start_year(season) % 100:02d}"
    bad = gid[gid.str[:5] != want]
    if len(bad):
        raise ValueError(f"{what}: {len(bad)} GAME_IDs do not start {want!r} for "
                         f"{season} {code}: {sorted(set(bad))[:5]}")


def _team_codes(team_ids: pd.Series, what: str) -> pd.Series:
    codes = team_ids.map(teams.from_team_id)
    bad = team_ids[codes.isna()]
    if len(bad):
        raise ValueError(f"{what}: unknown TEAM_ID {sorted(set(bad.tolist()))[:5]}")
    return codes


def _parse_matchup(matchup: pd.Series, team: pd.Series, what: str) -> tuple[pd.Series, pd.Series]:
    """'LAL vs. GSW' (home) / 'LAL @ GSW' (away) -> (opp tricode, home bool).

    The first token must be the row's own team, or the row's TEAM_ID and its
    MATCHUP disagree, which would silently flip a player onto the wrong side.
    """
    m = matchup.astype(str).str.extract(r"^(\w+) (vs\.|@) (\w+)$")
    bad = matchup[m[0].isna()]
    if len(bad):
        raise ValueError(f"{what}: unparseable MATCHUP {bad.iloc[:3].tolist()}")
    own = m[0].map(teams.canonical)
    opp = m[2].map(teams.canonical)
    if opp.isna().any():
        raise ValueError(f"{what}: unknown opponent code in MATCHUP "
                         f"{sorted(set(m[2][opp.isna()]))[:5]}")
    mismatch = own != team
    if mismatch.any():
        i = mismatch.idxmax()
        raise ValueError(f"{what}: MATCHUP {matchup[i]!r} does not start with the row's "
                         f"team {team[i]!r} ({int(mismatch.sum())} rows)")
    if (opp == team).any():
        raise ValueError(f"{what}: a team listed as its own opponent")
    return opp, m[1] == "vs."


def _check_unique(df: pd.DataFrame, keys: list[str], what: str) -> None:
    dup = df[df.duplicated(keys, keep=False)]
    if len(dup):
        raise ValueError(f"{what}: {len(dup)} rows share a key {keys}, e.g. "
                         f"{dup[keys].head(4).to_dict('records')}")


def _check_season_label(df: pd.DataFrame, col: str, want: str, what: str) -> None:
    bad = df[df[col].astype(str) != want]
    if len(bad):
        raise ValueError(f"{what}: {col} {sorted(set(bad[col].astype(str)))[:3]} "
                         f"!= expected {want!r}")


def fantasy_score(df: pd.DataFrame) -> pd.Series:
    """PrizePicks Fantasy Score, which is also stats.nba.com's NBA_FANTASY_PTS:
    PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV."""
    return (df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"] + 3 * df["stl"]
            + 3 * df["blk"] - df["tov"]).astype(float)


def _dd_td(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    tens = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    return (tens >= 2).astype("int64"), (tens >= 3).astype("int64")


def _check_box_identities(df: pd.DataFrame, what: str) -> None:
    """Impossible values raise. Every identity here held in every row of the
    2014-15 .. 2025-26 logs measured on 2026-10-08."""
    neg = [c for c in COUNT_STATS + ["pts"] if (df[c] < 0).any()]
    if neg:
        raise ValueError(f"{what}: negative counting stats in {neg}")
    checks = {
        "fgm>fga": df.fgm > df.fga, "fg3m>fg3a": df.fg3m > df.fg3a,
        "ftm>fta": df.ftm > df.fta, "fg3a>fga": df.fg3a > df.fga,
        "fg3m>fgm": df.fg3m > df.fgm, "reb!=oreb+dreb": df.reb != df.oreb + df.dreb,
        "pts!=2fgm+fg3m+ftm": df.pts != 2 * df.fgm + df.fg3m + df.ftm,
    }
    for name, mask in checks.items():
        if mask.any():
            raise ValueError(f"{what}: {int(mask.sum())} rows with {name}, e.g. "
                             f"{df.loc[mask].head(3).to_dict('records')}")


# ── Team games ───────────────────────────────────────────────────────────────
def _drop_cancelled(df: pd.DataFrame, what: str, must_be_empty: bool) -> tuple[pd.DataFrame, int]:
    """Remove KNOWN_CANCELLED_GAMES rows. A player log has no row for a game
    that was never played, so any there (`must_be_empty`) raises; a team row
    must look unplayed (no minutes, no points)."""
    gid = df["GAME_ID"].astype(str)
    hit = gid.isin(list(KNOWN_CANCELLED_GAMES))
    if not hit.any():
        return df, 0
    rows = df[hit]
    if must_be_empty:
        raise ValueError(f"{what}: {len(rows)} player rows for a game listed in "
                         f"KNOWN_CANCELLED_GAMES: {sorted(set(gid[hit]))}")
    played = (pd.to_numeric(rows["MIN"], errors="coerce").fillna(0) != 0)
    if "PTS" in rows:
        played |= pd.to_numeric(rows["PTS"], errors="coerce").fillna(0) != 0
    if played.any():
        raise ValueError(f"{what}: {sorted(set(gid[hit][played]))} is listed in "
                         f"KNOWN_CANCELLED_GAMES but has minutes or points")
    return df[~hit].copy(), int(gid[hit].nunique())


def _lgl_team_frame(lgl_t: dict, season: str, code: str,
                    neutral_games: Collection[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """LeagueGameLog T -> (one row per team-game with ot_periods, opp and
    neutral, a dict of counts for the report)."""
    what = f"LGL T {season} {code}"
    t = _frame(lgl_t, "LeagueGameLog", _LGL_T_REQ, what).copy()
    info: dict = {"cancelled_games_excluded": 0,
                  "neutral_source": "logs+schedule" if neutral_games is not None else "logs only"}
    if t.empty:
        return t, info
    _check_season_label(t, "SEASON_ID", _SEASON_ID_DIGIT[code] + str(util.season_start_year(season)), what)
    t["GAME_ID"] = t["GAME_ID"].astype(str)
    _check_game_ids(t["GAME_ID"], season, code, what)
    _check_unique(t, ["GAME_ID", "TEAM_ID"], what)
    t, info["cancelled_games_excluded"] = _drop_cancelled(t, what, must_be_empty=False)
    t["team"] = _team_codes(t["TEAM_ID"], what)
    t["opp"], t["home"] = _parse_matchup(t["MATCHUP"], t["team"], what)
    t["game_date"] = _parse_game_dates(t["GAME_DATE"], "%Y-%m-%d", what)
    # LGL T MIN is team PLAYER-minutes: 240 regulation, +25 per overtime.
    ot = (t["MIN"].astype(float) - 240) / 25
    bad = t[(ot < 0) | (ot != ot.round())]
    if len(bad):
        hint = (" (MIN 0 with no result looks like a cancelled game: check it and list it "
                "in KNOWN_CANCELLED_GAMES)" if (bad["MIN"].astype(float) == 0).any() else "")
        raise ValueError(f"{what}: team MIN not 240+25k: "
                         f"{bad[['GAME_ID', 'team', 'MIN']].head(3).to_dict('records')}{hint}")
    t["ot_periods"] = ot.astype("int64")
    # Neutral sites, three ways (module docstring): '@' on BOTH sides
    # ('MIA @ WAS' and 'WAS @ MIA', 0022400147: home=False for both); the
    # 2019-20 bubble (vs./@ kept, so `home` keeps the designated side); and
    # every pre-2024-25 international or Las Vegas game, which only the
    # schedule's arena city reveals (`neutral_games`).
    homes = t.groupby("GAME_ID")["home"].transform("sum")
    if (homes > 1).any():
        raise ValueError(f"{what}: games with two home sides: "
                         f"{sorted(set(t.loc[homes > 1, 'GAME_ID']))[:5]}")
    t["bubble"] = is_bubble(season, t["game_date"])
    by_logs = homes == 0
    by_sched = t["GAME_ID"].isin(list(neutral_games or ()))
    t["neutral"] = by_logs | t["bubble"] | by_sched
    if neutral_games is not None:
        # Counted, not raised: a game with no home side in the logs that the
        # schedule does not call neutral (0 in every season 2011-12..2025-26).
        info["neutral_logs_not_schedule"] = sorted(set(t.loc[by_logs & ~by_sched & ~t["bubble"], "GAME_ID"]))
        info["neutral_from_schedule_only"] = int(t.loc[by_sched & ~by_logs, "GAME_ID"].nunique())
    return t, info


def parse_team_games(lgl_t: dict, tgl_adv: dict, season: str, season_type: str,
                     neutral_games: Collection[str] | None = None) -> pd.DataFrame:
    """LeagueGameLog T + TeamGameLogs Advanced -> TEAM_GAME_COLUMNS.

    `neutral_games`: game ids the season's schedule calls neutral
    (neutral_site_games); None = the logs' own labelling only, which misses
    every neutral site before 2024-25 (reported as neutral_source).

    Invariants (raise): exactly 2 rows per game_id, at most one home side (none
    = a neutral site listed '@' on both sides), mirrored opp/pts,
    same date and overtime on both sides, team PLUS_MINUS == margin, W iff
    margin > 0, TeamGameLogs MIN == 48 + 5*OT, every LGL row has its advanced
    row and vice versa. KNOWN_CANCELLED_GAMES are excluded and counted.
    """
    code = stype_code(season_type)
    what = f"team_games {season} {code}"
    t, info = _lgl_team_frame(lgl_t, season, code, neutral_games)
    a = _frame(tgl_adv, "TeamGameLogs", _TGL_ADV_REQ, f"TGL Advanced {season} {code}").copy()
    report = {"season": season, "season_type": code, "lgl_rows": len(t), "tgl_rows": len(a),
              **info}
    if t.empty:
        if len(a):
            raise ValueError(f"{what}: LGL T is empty but TeamGameLogs has {len(a)} rows")
        out = _empty(TEAM_GAME_COLUMNS, _TEAM_DTYPES)
        out.attrs["report"] = dict(report, rows=0, games=0)
        return out
    _check_season_label(a, "SEASON_YEAR", season, f"TGL Advanced {season} {code}")
    a["GAME_ID"] = a["GAME_ID"].astype(str)
    _check_unique(a, ["GAME_ID", "TEAM_ID"], f"TGL Advanced {season} {code}")
    a, report["tgl_cancelled_games_excluded"] = _drop_cancelled(a, f"TGL Advanced {season} {code}",
                                                                must_be_empty=False)

    m = t.merge(a[["GAME_ID", "TEAM_ID", "MIN", "PACE", "POSS", "OFF_RATING", "DEF_RATING",
                   "NET_RATING"]].rename(columns={"MIN": "GAME_MIN"}),
                on=["GAME_ID", "TEAM_ID"], how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        lo = m[m["_merge"] != "both"]
        raise ValueError(f"{what}: {len(lo)} team-games in only one of LGL T / TGL "
                         f"Advanced: {lo[['GAME_ID', 'TEAM_ID', '_merge']].head(4).to_dict('records')}")
    # TeamGameLogs MIN is GAME minutes (48 + 5 per OT): a second OT reading.
    gm = m["GAME_MIN"].astype(float)
    if (gm != 48 + 5 * m["ot_periods"]).any():
        bad = m[gm != 48 + 5 * m["ot_periods"]]
        raise ValueError(f"{what}: TeamGameLogs MIN disagrees with LGL T overtime: "
                         f"{bad[['GAME_ID', 'team', 'GAME_MIN', 'ot_periods']].head(3).to_dict('records')}")

    sizes = m.groupby("GAME_ID").size()
    if (sizes != 2).any():
        raise ValueError(f"{what}: game_ids without exactly 2 team rows: "
                         f"{sizes[sizes != 2].head(5).to_dict()}")
    other = m[["GAME_ID", "team", "opp", "PTS", "game_date", "ot_periods"]].rename(
        columns={"team": "o_team", "opp": "o_opp", "PTS": "opp_pts",
                 "game_date": "o_date", "ot_periods": "o_ot"})
    m = m.merge(other, on="GAME_ID")
    m = m[m["team"] != m["o_team"]].copy()
    if len(m) != len(t):
        raise ValueError(f"{what}: mirror join changed the row count {len(t)} -> {len(m)}")
    for cond, msg in [(m["opp"] != m["o_team"], "opp is not the other row's team"),
                      (m["o_opp"] != m["team"], "opponent's opp is not this team"),
                      (m["game_date"] != m["o_date"], "the two sides carry different dates"),
                      (m["ot_periods"] != m["o_ot"], "the two sides carry different OT")]:
        if cond.any():
            raise ValueError(f"{what}: {msg}: {m.loc[cond, 'GAME_ID'].head(3).tolist()}")
    m["margin"] = m["PTS"] - m["opp_pts"]
    if (m["margin"] == 0).any():
        raise ValueError(f"{what}: tied final score in {m.loc[m.margin == 0, 'GAME_ID'].head(3).tolist()}")
    if (m["PLUS_MINUS"] != m["margin"]).any():
        raise ValueError(f"{what}: team PLUS_MINUS != margin in "
                         f"{m.loc[m.PLUS_MINUS != m.margin, 'GAME_ID'].head(3).tolist()}")
    if ((m["WL"] == "W") != (m["margin"] > 0)).any():
        raise ValueError(f"{what}: WL disagrees with the score")

    out = pd.DataFrame({
        "season": season, "season_type": code, "game_id": m["GAME_ID"].astype(str),
        "game_date": m["game_date"], "team": m["team"], "opp": m["opp"],
        "home": m["home"].astype(bool), "neutral": m["neutral"].astype(bool),
        "pts": m["PTS"], "opp_pts": m["opp_pts"],
        "margin": m["margin"], "ot_periods": m["ot_periods"],
    })
    for c in COUNT_STATS:
        out[c] = m[c.upper()]
    out["pace"], out["poss"] = m["PACE"], m["POSS"]
    out["off_rating"], out["def_rating"] = m["OFF_RATING"], m["DEF_RATING"]
    out["net_rating"] = m["NET_RATING"]
    out = _coerce(out, _TEAM_DTYPES)[TEAM_GAME_COLUMNS]
    _check_box_identities(out, what)
    out = out.sort_values(["game_date", "game_id", "home"], ascending=[True, True, False])
    out = out.reset_index(drop=True)
    out.attrs["report"] = dict(report, rows=len(out), games=out["game_id"].nunique(),
                               ot_games=int((out.groupby("game_id").ot_periods.first() > 0).sum()),
                               neutral_games=int(out.loc[out.neutral, "game_id"].nunique()),
                               bubble_games=int(m.loc[m["bubble"], "GAME_ID"].nunique()))
    return out


# ── Player games ─────────────────────────────────────────────────────────────
_INT_STATS = COUNT_STATS + ["pts", "plus_minus", "dd2", "td3", "sec"]
_PLAYER_DTYPES = {**{c: "int64" for c in _INT_STATS}, "person_id": "int64",
                  "min": "float64", "pfd": "float64", "fantasy": "float64",
                  "home": "bool", "neutral": "bool", **{c: "float64" for c in ADV_MAP.values()}}
_TEAM_DTYPES = {**{c: "int64" for c in COUNT_STATS + ["pts", "opp_pts", "margin", "ot_periods"]},
                "home": "bool", "neutral": "bool", "pace": "float64", "poss": "float64",
                "off_rating": "float64", "def_rating": "float64", "net_rating": "float64"}
_STR_COLS = {"season", "season_type", "game_id", "player_name", "team", "opp", "src",
             "home_team", "away_team"}


def _empty(columns: list[str], dtypes: dict) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "object")) for c in columns})
    if "game_date" in df:
        df["game_date"] = pd.Series(dtype="datetime64[ns]")
    return df


def _coerce(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    for c, t in dtypes.items():
        if c in df:
            if t == "int64" and df[c].isna().any():
                raise ValueError(f"column {c} has nulls but must be int64")
            df[c] = df[c].astype(t)
    return df


def parse_player_games(pgl_base: dict, pgl_adv: dict, lgl_p: dict, lgl_t: dict,
                       season: str, season_type: str,
                       neutral_games: Collection[str] | None = None) -> pd.DataFrame:
    """PGL Base + PGL Advanced + LGL P (+ LGL T for overtime) -> PLAYER_GAME_COLUMNS.

    Rows are exactly PGL Base's: PGL is the authority on who appeared (module
    docstring), so `src` is always 'pgl' (kept for the schema). PGL Advanced
    adds the rates; LGL P is a cross-check. `neutral_games` as in
    parse_team_games.

    Raises on: duplicate (person_id, game_id); unknown team; MATCHUP/team
    disagreement; PGL/LGL disagreeing on a player's team, date or (rounded)
    minutes; an LGL-only row that carries minutes or a stat (an appearance PGL
    lacks); an Advanced row with no Base row; a team-game with fewer than 5
    player rows; player minutes summing more than MINUTES_TOLERANCE from
    240 + 25*OT (unless listed in KNOWN_MINUTES_EXCEPTIONS); player points not
    summing to the team score; fantasy != PGL's own fantasy column; impossible
    box values.
    Counted in attrs['report'], not raised: all-zero LGL-only rows (EXCLUDED,
    listed in lgl_only_rows), Base rows without an Advanced row, PGL-vs-LGL
    box-stat disagreements, PGL DD2/TD3 flags that disagree with the stats.
    """
    code = stype_code(season_type)
    what = f"player_games {season} {code}"
    b = _frame(pgl_base, "PlayerGameLogs", _PGL_BASE_REQ, f"PGL Base {season} {code}").copy()
    a = _frame(pgl_adv, "PlayerGameLogs", _PGL_ADV_REQ, f"PGL Advanced {season} {code}").copy()
    lg = _frame(lgl_p, "LeagueGameLog", _LGL_P_REQ, f"LGL P {season} {code}").copy()
    t, info = _lgl_team_frame(lgl_t, season, code, neutral_games)
    report: dict = {"season": season, "season_type": code, "pgl_rows": len(b),
                    "adv_rows": len(a), "lgl_rows": len(lg), "team_games": len(t), **info}

    if b.empty and a.empty and lg.empty:
        if len(t):
            raise ValueError(f"{what}: player logs are empty but LGL T has {len(t)} rows")
        out = _empty(PLAYER_GAME_COLUMNS, _PLAYER_DTYPES)
        out.attrs["report"] = dict(report, rows=0, games=0)
        return out
    if b.empty or lg.empty or t.empty:
        raise ValueError(f"{what}: one source is empty and another is not "
                         f"(PGL {len(b)}, LGL P {len(lg)}, LGL T {len(t)})")

    for df, src in ((b, "PGL Base"), (a, "PGL Advanced"), (lg, "LGL P")):
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        _check_game_ids(df["GAME_ID"], season, code, f"{src} {season} {code}")
        _check_unique(df, ["PLAYER_ID", "GAME_ID"], f"{src} {season} {code}")
        _drop_cancelled(df, f"{src} {season} {code}", must_be_empty=True)
    _check_season_label(b, "SEASON_YEAR", season, f"PGL Base {season} {code}")
    _check_season_label(a, "SEASON_YEAR", season, f"PGL Advanced {season} {code}")
    _check_season_label(lg, "SEASON_ID", _SEASON_ID_DIGIT[code] + str(util.season_start_year(season)),
                        f"LGL P {season} {code}")

    # Base + Advanced: same rows expected. An Advanced row without Base would be
    # dropped by the left join, so it raises; the reverse only loses rates.
    adv = a[["PLAYER_ID", "GAME_ID", "TEAM_ID", "MIN"] + list(ADV_MAP)].rename(
        columns={"TEAM_ID": "ADV_TEAM_ID", "MIN": "ADV_MIN", **ADV_MAP})
    pb = b.merge(adv, on=["PLAYER_ID", "GAME_ID"], how="outer", indicator=True)
    adv_only = pb[pb["_merge"] == "right_only"]
    if len(adv_only):
        raise ValueError(f"{what}: {len(adv_only)} Advanced rows have no Base row: "
                         f"{adv_only[['PLAYER_ID', 'GAME_ID']].head(3).to_dict('records')}")
    has_adv = pb["_merge"] == "both"
    report["base_without_adv"] = int((~has_adv).sum())
    if (has_adv & (pb["ADV_TEAM_ID"] != pb["TEAM_ID"])).any():
        raise ValueError(f"{what}: Base and Advanced disagree on TEAM_ID")
    # Advanced MIN is Base MIN rounded to 0.1 (max gap 0.05 measured 2024-25).
    gap = (pb["ADV_MIN"] - pb["MIN"]).abs()
    if (has_adv & (gap > 0.051)).any():
        bad = pb[has_adv & (gap > 0.051)]
        raise ValueError(f"{what}: Advanced MIN is not Base MIN rounded: "
                         f"{bad[['PLAYER_ID', 'GAME_ID', 'MIN', 'ADV_MIN']].head(3).to_dict('records')}")
    pb = pb.drop(columns=["_merge", "ADV_TEAM_ID", "ADV_MIN"])

    # PGL (the appearance authority) vs LGL P, a cross-check only.
    lcols = ["PLAYER_ID", "GAME_ID", "TEAM_ID", "GAME_DATE", "MIN", "PLAYER_NAME", "MATCHUP",
             "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL",
             "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "FANTASY_PTS"]
    m = pb.merge(lg[lcols].add_prefix("L_").rename(columns={"L_PLAYER_ID": "PLAYER_ID",
                                                            "L_GAME_ID": "GAME_ID"}),
                 on=["PLAYER_ID", "GAME_ID"], how="outer", indicator=True)
    both = m["_merge"] == "both"
    pgl_only = m["_merge"] == "left_only"
    lgl_only = m["_merge"] == "right_only"
    report["pgl_only"] = int(pgl_only.sum())
    report["lgl_only"] = int(lgl_only.sum())
    report["lgl_only_rows"] = m.loc[lgl_only, ["PLAYER_ID", "L_PLAYER_NAME", "GAME_ID", "L_MIN"]].rename(
        columns={"PLAYER_ID": "person_id", "L_PLAYER_NAME": "player", "GAME_ID": "game_id",
                 "L_MIN": "min"}).head(50).to_dict("records")
    # An LGL-only row is excluded only when it is ALL zero: every one seen was
    # a player the box score lists as not playing (Schroder 0022200140
    # INACTIVE, Morris 0042100105 played 0). Minutes or any stat would be an
    # appearance PGL lacks, which the points/minutes sums might not even
    # notice (a rebound), so it raises for a person to look at.
    lstats = ["L_" + c for c in ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB",
                                 "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
                                 "PLUS_MINUS"]]
    lo_vals = m.loc[lgl_only, lstats].apply(pd.to_numeric, errors="coerce")
    live = lo_vals.isna().any(axis=1) | (lo_vals != 0).any(axis=1)
    if live.any():
        bad = m.loc[live[live].index, ["PLAYER_ID", "L_PLAYER_NAME", "GAME_ID", "L_MIN", "L_PTS"]]
        raise ValueError(f"{what}: {len(bad)} LeagueGameLog rows with minutes or stats are "
                         f"missing from PlayerGameLogs (an appearance PGL lacks): "
                         f"{bad.head(3).to_dict('records')}")
    report["lgl_only_excluded"] = int(lgl_only.sum())
    if (both & (m["TEAM_ID"] != m["L_TEAM_ID"])).any():
        bad = m[both & (m["TEAM_ID"] != m["L_TEAM_ID"])]
        raise ValueError(f"{what}: PGL and LGL disagree on a player's team: "
                         f"{bad[['PLAYER_ID', 'GAME_ID', 'TEAM_ID', 'L_TEAM_ID']].head(3).to_dict('records')}")
    if (both & ((m["MIN"].round() - m["L_MIN"]).abs() > 1)).any():
        bad = m[both & ((m["MIN"].round() - m["L_MIN"]).abs() > 1)]
        raise ValueError(f"{what}: LGL MIN is not PGL MIN rounded: "
                         f"{bad[['PLAYER_ID', 'GAME_ID', 'MIN', 'L_MIN']].head(3).to_dict('records')}")
    disagree = {}
    for c in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL",
              "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]:
        n = int((both & (m[c] != m["L_" + c])).sum())
        if n:
            disagree[c.lower()] = n
    report["pgl_vs_lgl_stat_disagreements"] = disagree

    what_dates = f"{what} dates"
    lgl_dates = _parse_game_dates(m.loc[~pgl_only, "L_GAME_DATE"], "%Y-%m-%d", what_dates)
    # The excluded rows go now: every row kept below is a PGL Base row.
    both = both[~lgl_only]
    m = m[~lgl_only].copy()
    m["src"] = "pgl"
    m["src_fantasy"] = m["NBA_FANTASY_PTS"]
    m["game_date"] = _parse_game_dates(m["GAME_DATE"], "%Y-%m-%dT%H:%M:%S", what_dates)
    if (both & (m["game_date"] != lgl_dates.reindex(m.index))).any():
        bad = m[both & (m["game_date"] != lgl_dates.reindex(m.index))]
        raise ValueError(f"{what}: PGL and LGL disagree on a game date: "
                         f"{bad[['PLAYER_ID', 'GAME_ID', 'GAME_DATE', 'L_GAME_DATE']].head(3).to_dict('records')}")

    m["TEAM_ID"] = m["TEAM_ID"].astype("int64")
    team = _team_codes(m["TEAM_ID"], what)
    opp, home = _parse_matchup(m["MATCHUP"], team, what)

    out = pd.DataFrame({
        "season": season, "season_type": code, "game_id": m["GAME_ID"].astype(str),
        "game_date": m["game_date"].astype("datetime64[ns]"),
        "person_id": m["PLAYER_ID"].astype("int64"), "player_name": m["PLAYER_NAME"].astype(str),
        "team": team, "opp": opp, "home": home.astype(bool),
        "min": m["MIN"].astype(float),
    })
    out["sec"] = (out["min"] * 60).round().astype("int64")
    for c in COUNT_STATS:
        out[c] = m[c.upper()]
    out["pfd"] = m["PFD"].astype(float)
    out["pts"], out["plus_minus"] = m["PTS"], m["PLUS_MINUS"]
    out = _coerce(out, {c: "int64" for c in COUNT_STATS + ["pts", "plus_minus"]})
    _check_box_identities(out, what)
    out["fantasy"] = fantasy_score(out)
    err = (out["fantasy"] - m["src_fantasy"].astype(float)).abs()
    has_f = m["src_fantasy"].notna()
    if (has_f & (err > 1e-6)).any():
        bad = out[has_f & (err > 1e-6)]
        raise ValueError(f"{what}: fantasy != source fantasy column in {len(bad)} rows, e.g. "
                         f"{bad[['person_id', 'game_id', 'fantasy']].head(3).to_dict('records')}")
    report["fantasy_checked"] = int(has_f.sum())
    report["fantasy_max_abs_err"] = float(err[has_f].max()) if has_f.any() else None
    out["dd2"], out["td3"] = _dd_td(out)
    report["source_dd2_disagreements"] = int(((m["DD2"] != out["dd2"]) & m["DD2"].notna()).sum())
    report["source_td3_disagreements"] = int(((m["TD3"] != out["td3"]) & m["TD3"].notna()).sum())
    for k, v in ADV_MAP.items():
        out[v] = m[v].astype(float)
    out["src"] = m["src"]
    neutral = t.drop_duplicates("GAME_ID").set_index("GAME_ID")["neutral"]
    out["neutral"] = out["game_id"].map(neutral)
    if out["neutral"].isna().any():
        raise ValueError(f"{what}: player rows for games missing from LGL T: "
                         f"{sorted(set(out.loc[out.neutral.isna(), 'game_id']))[:5]}")
    out["neutral"] = out["neutral"].astype(bool)
    out = _coerce(out, _PLAYER_DTYPES)[PLAYER_GAME_COLUMNS]
    _check_unique(out, ["person_id", "game_id"], what)

    _check_against_team_log(out, t, what, report)
    report["rows"] = len(out)
    report["games"] = int(out["game_id"].nunique())
    report["players"] = int(out["person_id"].nunique())
    report["zero_min_rows"] = int((out["sec"] == 0).sum())
    out = out.sort_values(["game_date", "game_id", "team", "min"],
                          ascending=[True, True, True, False]).reset_index(drop=True)
    out.attrs["report"] = report
    return out


def _check_against_team_log(out: pd.DataFrame, t: pd.DataFrame, what: str, report: dict) -> None:
    """Team-game invariants that tie player rows to the team log."""
    sums = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "ast", "stl", "blk"]
    g = out.groupby(["game_id", "team"]).agg(
        n=("person_id", "size"), min_sum=("min", "sum"), pts=("pts", "sum"),
        pm_sum=("plus_minus", "sum"), date=("game_date", "first"),
        ndates=("game_date", "nunique"), opp=("opp", "first"), nopp=("opp", "nunique"),
        home=("home", "first"), nhome=("home", "nunique"),
        **{c: (c, "sum") for c in sums}).reset_index()
    tt = t[["GAME_ID", "team", "opp", "home", "game_date", "ot_periods", "PTS", "PLUS_MINUS"]
           + [c.upper() for c in sums]].rename(
        columns={"GAME_ID": "game_id", "opp": "t_opp", "home": "t_home",
                 "game_date": "t_date", "PTS": "t_pts", "PLUS_MINUS": "t_margin",
                 **{c.upper(): "t_" + c for c in sums}})
    j = g.merge(tt, on=["game_id", "team"], how="outer", indicator=True)
    if (j["_merge"] != "both").any():
        bad = j[j["_merge"] != "both"]
        raise ValueError(f"{what}: {len(bad)} team-games in only one of the player and team "
                         f"logs: {bad[['game_id', 'team', '_merge']].head(4).to_dict('records')}")
    for cond, msg in [((j["ndates"] != 1) | (j["date"] != j["t_date"]), "game date differs from the team log"),
                      ((j["nopp"] != 1) | (j["opp"] != j["t_opp"]), "opponent differs from the team log"),
                      ((j["nhome"] != 1) | (j["home"] != j["t_home"]), "home flag differs from the team log"),
                      (j["n"] < 5, "fewer than 5 player rows"),
                      (j["pts"] != j["t_pts"], "player points do not sum to the team score")]:
        if cond.any():
            raise ValueError(f"{what}: {msg}: "
                             f"{j.loc[cond, ['game_id', 'team']].head(4).to_dict('records')}")
    expected = 240 + 25 * j["ot_periods"]
    dev = j["min_sum"] - expected
    over = (dev.abs() > MINUTES_TOLERANCE + 1e-9)
    known = j["game_id"].isin(list(KNOWN_MINUTES_EXCEPTIONS))
    if (over & ~known).any():
        bad = j[over & ~known].assign(expected=expected[over & ~known])
        raise ValueError(f"{what}: player minutes sum more than {MINUTES_TOLERANCE} from "
                         f"240+25*OT in {len(bad)} team-games: "
                         f"{bad[['game_id', 'team', 'min_sum', 'expected']].head(5).to_dict('records')}"
                         f" (inspect the box score; add to KNOWN_MINUTES_EXCEPTIONS with a reason)")
    report["players_per_team_game_min"] = int(j["n"].min())
    report["players_per_team_game_max"] = int(j["n"].max())
    report["min_sum_max_abs_dev"] = round(float(dev.abs().max()), 4)
    report["min_sum_dev_over_0_5"] = int((dev.abs() > 0.5).sum())
    report["min_sum_known_exceptions"] = int((over & known).sum())
    # Five players are on the floor throughout, so player plus-minus sums to 5x
    # the margin. PGL satisfied this in every team-game of 2014-15, 2019-20,
    # 2022-23 and 2024-25 while LGL failed it in 4/2/14/0 (it is why PGL wins
    # the PLUS_MINUS disagreements). Counted, not raised: a secondary stat.
    report["plus_minus_sum_violations"] = int((j["pm_sum"] != 5 * j["t_margin"]).sum())
    # Box sums vs the team log (REB/TOV excluded: team rebounds and team
    # turnovers are credited to no player). Counted, not raised.
    report["team_sum_mismatches"] = {c: int((j[c] != j["t_" + c]).sum()) for c in sums
                                     if (j[c] != j["t_" + c]).any()}
    ot = out[["game_id", "team", "min"]].merge(tt[["game_id", "team", "ot_periods"]],
                                                on=["game_id", "team"], how="left")
    if (ot["min"] > 48 + 5 * ot["ot_periods"] + 0.5).any() or (ot["min"] < 0).any():
        bad = ot[(ot["min"] > 48 + 5 * ot["ot_periods"] + 0.5) | (ot["min"] < 0)]
        raise ValueError(f"{what}: a player logged more minutes than the game lasted: "
                         f"{bad.head(3).to_dict('records')}")


# ── Schedule ─────────────────────────────────────────────────────────────────
def _broadcast_names(b: dict | None, key: str) -> str | None:
    lst = (b or {}).get(key) or []
    names = [x.get("broadcasterAbbreviation") or x.get("broadcasterDisplay") for x in lst]
    names = [n for n in names if n]
    return "/".join(names) if names else None


def parse_schedule(payload: dict, season: str) -> pd.DataFrame:
    """scheduleleaguev2 -> SCHEDULE_COLUMNS, one row per game_id.

    Traps handled:
      - gameDate is 'MM/DD/YYYY 00:00:00' (string min() returns '01/01/2027');
        it is parsed, and is the ET game date.
      - gameDateTimeEst/gameDateEst carry a 'Z' suffix but are EASTERN wall
        clock ('2026-10-03T19:00:00Z' for a 23:00 UTC tip). Only
        gameDateTimeUTC is UTC, so tip_utc comes from it and its ET date must
        equal gameDate (raise otherwise).
      - An unknown tip time is gameTimeEst '0001-01-01...' (the Cup knockout
        placeholders sit at 05:00Z = ET midnight): tip_utc is None there.
      - TBD Cup knockout placeholders have teamId 0: home/away None, is_tbd.
      - Non-NBA clubs (London Lions 'LON', id 50015) are dropped and COUNTED;
        a non-NBA team outside preseason raises. An NBA-range id that does not
        map raises (never passed through).
      - isNeutral is false for every neutral site before 2024-25 (London,
        Mexico City, Paris, the 2023-24 Cup games in Las Vegas) and for the
        Orlando bubble. is_neutral = isNeutral OR bubble OR a home game whose
        arenaCity is not one of the home team's _HOME_CITIES. More than
        MAX_RELOCATED_HOME_GAMES such games for one team raises.
    attrs['report'] counts dropped rows, TBD games, RS completeness and lists
    the relocated (city-rule) games.
    """
    what = f"schedule {season}"
    ls = payload.get("leagueSchedule")
    if not isinstance(ls, dict) or "gameDates" not in ls:
        raise http.BadPayload(what, f"no leagueSchedule.gameDates; keys {list(payload)[:5]}")
    if ls.get("seasonYear") != season:
        raise ValueError(f"{what}: payload seasonYear {ls.get('seasonYear')!r} != {season!r}")
    rows, dropped, dropped_allstar, relocated_rows = [], [], [], []
    for gd in ls["gameDates"]:
        try:
            gdate = dt.datetime.strptime(gd["gameDate"], "%m/%d/%Y %H:%M:%S").date()
        except (KeyError, ValueError) as e:
            raise ValueError(f"{what}: bad gameDate {gd.get('gameDate')!r}: {e}") from None
        for g in gd.get("games") or []:
            gid = str(g.get("gameId", ""))
            if len(gid) != 10 or not gid.isdigit():
                raise ValueError(f"{what}: bad gameId {gid!r}")
            gtype = SCHEDULE_GAME_TYPES.get(gid[:3])
            if gtype is None:
                raise ValueError(f"{what}: unknown game id prefix {gid[:3]!r} ({gid})")
            if gid[3:5] != f"{util.season_start_year(season) % 100:02d}":
                raise ValueError(f"{what}: game {gid} is not a {season} id")
            sides, non_nba = {}, []
            for side in ("homeTeam", "awayTeam"):
                tm = g.get(side) or {}
                tid = int(tm.get("teamId") or 0)
                if tid == 0:
                    sides[side] = None
                    continue
                code = teams.from_team_id(tid)
                if code is None:
                    if tid in _NBA_TEAM_ID_RANGE:
                        raise ValueError(f"{what}: NBA-range team id {tid} unknown ({gid})")
                    non_nba.append(f"{tm.get('teamTricode')}:{tid}")
                    continue
                tri = tm.get("teamTricode")
                if tri and teams.canonical(tri) != code:
                    raise ValueError(f"{what}: {gid} tricode {tri!r} != team id {tid} ({code})")
                sides[side] = code
            if non_nba:
                # All-Star weekend teams (EST/WST, Rising Stars CHK/SHQ...) carry
                # ids like 1610616855: not franchises, so not team games.
                # Preseason guests (LON, Maccabi 'MTA' id 41) are not NBA teams.
                # Both are dropped and counted; anywhere else it is a bug.
                if gtype == "ALLSTAR":
                    dropped_allstar.append(gid)
                    continue
                if gtype != "PRE":
                    raise ValueError(f"{what}: non-NBA team {non_nba} in a {gtype} game {gid}")
                dropped.append({"game_id": gid, "game_type": gtype, "non_nba": non_nba})
                continue
            tbd = sides["homeTeam"] is None or sides["awayTeam"] is None
            if tbd and gtype not in ("RS", "CUP_FINAL", "PO", "PI", "ALLSTAR"):
                raise ValueError(f"{what}: TBD team in a {gtype} game {gid}")
            if not tbd and sides["homeTeam"] == sides["awayTeam"]:
                raise ValueError(f"{what}: {gid} has the same team on both sides")
            est_date = str(g.get("gameDateEst", ""))[:10]
            if est_date and est_date != gdate.isoformat():
                raise ValueError(f"{what}: {gid} gameDateEst {est_date} != gameDate {gdate}")
            tip = None
            time_est = str(g.get("gameTimeEst") or "")
            if g.get("gameDateTimeUTC") and not time_est.startswith("0001-") \
                    and str(g.get("gameStatusText", "")).strip().upper() != "TBD":
                tip = dt.datetime.fromisoformat(g["gameDateTimeUTC"].replace("Z", "+00:00"))
                if tip.tzinfo is None:
                    raise ValueError(f"{what}: {gid} gameDateTimeUTC is naive")
                if util.et_date_of(tip) != gdate:
                    raise ValueError(f"{what}: {gid} tip {tip.isoformat()} is ET "
                                     f"{util.et_date_of(tip)}, gameDate says {gdate}")
            b = g.get("broadcasters") or {}
            city = str(g.get("arenaCity") or "").strip() or None
            bubble = bool(gtype in ("RS", "PI", "PO") and is_bubble(season, pd.Timestamp(gdate)))
            # A home game outside the home team's own cities is a neutral site
            # (module docstring); unknown home or city: the source's flag only.
            relocated = bool(sides["homeTeam"] and city
                             and city not in _HOME_CITIES[sides["homeTeam"]])
            if relocated and not bubble:
                relocated_rows.append({"game_id": gid, "game_type": gtype,
                                       "home": sides["homeTeam"], "arena_city": city,
                                       "source_neutral": bool(g.get("isNeutral"))})
            rows.append({
                "game_id": gid, "season": season, "game_type": gtype,
                "game_date": pd.Timestamp(gdate), "tip_utc": tip,
                "home": sides["homeTeam"], "away": sides["awayTeam"], "is_tbd": tbd,
                "label": g.get("gameLabel") or "", "sublabel": g.get("gameSubLabel") or "",
                "subtype": g.get("gameSubtype") or "",
                "is_neutral": bool(g.get("isNeutral")) or bubble or relocated,
                "arena_name": g.get("arenaName") or None, "arena_city": city,
                "postponed_status": g.get("postponedStatus") or None,
                "national_tv": _broadcast_names(b, "nationalBroadcasters"),
                "game_status": int(g.get("gameStatus") or 0),
                "if_necessary": str(g.get("ifNecessary", "")).lower() == "true",
            })
    df = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    _check_unique(df, ["game_id"], what)
    df["game_date"] = df["game_date"].astype("datetime64[ns]")
    df["tip_utc"] = pd.to_datetime(df["tip_utc"], utc=True)
    df = df.sort_values(["game_date", "tip_utc", "game_id"], na_position="last").reset_index(drop=True)

    rs = df[df.game_type == "RS"]
    per_team = pd.concat([rs["home"], rs["away"]]).dropna().value_counts()
    report = {
        "season": season, "rows": len(df), "by_type": df["game_type"].value_counts().to_dict(),
        "dropped_non_nba": len(dropped), "dropped_rows": dropped[:20],
        "dropped_allstar": len(dropped_allstar),
        "tbd_games": int(df["is_tbd"].sum()), "unknown_tip": int(df["tip_utc"].isna().sum()),
        "rs_games": len(rs), "rs_games_expected": _RS_GAMES.get(season, 1230),
        "rs_games_per_team_min": int(per_team.min()) if len(per_team) else 0,
        "rs_games_per_team_max": int(per_team.max()) if len(per_team) else 0,
        "cup_final_games": int((df.game_type == "CUP_FINAL").sum()),
        "neutral_games": int(df["is_neutral"].sum()),
        "relocated_games": [r for r in relocated_rows if r["game_type"] != "PRE"],
        "neutral_by_city_only": sum(1 for r in relocated_rows
                                    if r["game_type"] != "PRE" and not r["source_neutral"]),
    }
    if (util.season_end_year(season) >= util.season_end_year(FIRST_CUP_FINAL_SEASON)
            and report["rs_games"] and report["cup_final_games"] == 0):
        raise ValueError(f"{what}: no Cup final (006) game listed; rest features need it")
    # Exhibitions travel (Quebec City, Macao), so only competitive games count.
    per_team = pd.Series([r["home"] for r in report["relocated_games"]], dtype=object).value_counts()
    if len(per_team) and per_team.max() > MAX_RELOCATED_HOME_GAMES:
        t = per_team.idxmax()
        cities = sorted({r["arena_city"] for r in report["relocated_games"] if r["home"] == t})
        raise ValueError(f"{what}: {per_team.max()} {t} home games outside {sorted(_HOME_CITIES[t])}, "
                         f"in {cities}: a new arena-city spelling? (add it to _HOME_CITIES), "
                         f"else every one of them is marked neutral")
    df.attrs["report"] = report
    return df


# ── Players ──────────────────────────────────────────────────────────────────
_HEIGHT_RE = re.compile(r"^(\d)-(\d{1,2})$")


def parse_height(value) -> float:
    """'6-10' -> 82.0; None/'' -> NaN; anything else raises."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return float("nan")
    m = _HEIGHT_RE.match(str(value).strip())
    if not m or int(m.group(2)) >= 12:
        raise ValueError(f"unparseable height {value!r}")
    inches = int(m.group(1)) * 12 + int(m.group(2))
    if not 60 <= inches <= 96:
        raise ValueError(f"impossible height {value!r}")
    return float(inches)


def _to_num(s: pd.Series, what: str, lo: float | None = None, hi: float | None = None) -> pd.Series:
    """Numeric with None/'' -> NaN; a non-numeric string raises (never coerced
    silently); out-of-range values raise."""
    raw = s.where(s.astype(str).str.strip().ne("") & s.notna())
    num = pd.to_numeric(raw, errors="coerce")
    bad = raw[num.isna() & raw.notna()]
    if len(bad):
        raise ValueError(f"{what}: non-numeric values {sorted(set(bad.astype(str)))[:5]}")
    if lo is not None and (num < lo).any() or hi is not None and (num > hi).any():
        raise ValueError(f"{what}: values outside [{lo}, {hi}]: "
                         f"{num[(num < lo) | (num > hi)].head(3).tolist()}")
    return num.astype(float)


def parse_player_index(payload: dict, season: str, strict: bool = True) -> pd.DataFrame:
    """playerindex (Historical=0, TeamID=0) -> PLAYER_INDEX_COLUMNS.

    Keeps ROSTER_STATUS == 1 with a real TEAM_ID; off-roster rows (ROSTER_STATUS
    None, TEAM_ID 0: waived players, 57 of them in the 2025-26 pull) are
    counted in attrs['report'].

    PAST SEASONS ARE NOT ROSTERS. Measured 2026-10-08: Season=2024-25 returned
    132 rows and Season=2014-15 98 rows, the players whose LAST active season
    (TO_YEAR, a start year) was around then, with their latest team (Lonzo Ball
    'CHI' TO_YEAR 2025 in the 2024-25 pull). Only the live season is a league
    roster (2026-27: 615 rows, TO_YEAR 2026 on every one). strict=True raises
    unless every row has TO_YEAR == the season's start year and all 30 teams
    appear. Past rosters come from CommonTeamRoster; point-in-time team for any
    game comes from the game logs.
    """
    what = f"playerindex {season}"
    p = _frame(payload, "PlayerIndex", _PINDEX_REQ + ["TO_YEAR"], what).copy()
    _check_unique(p, ["PERSON_ID"], what)
    if strict:
        to_year = pd.to_numeric(p["TO_YEAR"], errors="coerce")
        start = util.season_start_year(season)
        off = int((to_year != start).sum())
        if off or p.empty:
            raise ValueError(f"{what}: {off} of {len(p)} rows have TO_YEAR != {start}; a "
                             f"past-season playerindex lists players whose last season "
                             f"was then, not that season's roster (use team_roster)")
    tid = pd.to_numeric(p["TEAM_ID"], errors="coerce").fillna(0).astype("int64")
    on_roster = (pd.to_numeric(p["ROSTER_STATUS"], errors="coerce") == 1) & (tid != 0)
    report = {"season": season, "raw_rows": len(p), "kept": int(on_roster.sum()),
              "excluded_off_roster": int((~on_roster).sum())}
    p, tid = p[on_roster].copy(), tid[on_roster]
    # SUPPLEMENTAL_STATUS is 0 or 1 on every rostered row (2026-27: 532/83).
    # A null or a new code would otherwise read as "not two-way" in silence.
    supp = pd.to_numeric(p["SUPPLEMENTAL_STATUS"], errors="coerce")
    if not supp.isin([0, 1]).all():
        bad = p.loc[~supp.isin([0, 1]), "SUPPLEMENTAL_STATUS"]
        raise ValueError(f"{what}: {len(bad)} rostered rows with SUPPLEMENTAL_STATUS outside "
                         f"{{0, 1}}: {sorted({repr(v) for v in bad})[:5]}")
    first = p["PLAYER_FIRST_NAME"].fillna("").astype(str).str.strip()
    last = p["PLAYER_LAST_NAME"].fillna("").astype(str).str.strip()
    out = pd.DataFrame({
        "season": season, "person_id": p["PERSON_ID"].astype("int64"),
        "first_name": first, "last_name": last,
        # Family-name-first players (Yang Hansen) have FIRST='Yang', so this
        # order reproduces the NBA's own display name.
        "display_name": (first + " " + last).str.strip(),
        "team": _team_codes(tid, what),
        "position": p["POSITION"].where(p["POSITION"].astype(str).str.strip().ne(""), None),
        "height_in": p["HEIGHT"].map(parse_height),
        "weight": _to_num(p["WEIGHT"], f"{what} WEIGHT", 130, 400),
        "draft_year": _to_num(p["DRAFT_YEAR"], f"{what} DRAFT_YEAR", 1940, 2100),
        "draft_round": _to_num(p["DRAFT_ROUND"], f"{what} DRAFT_ROUND", 0, 10),
        "draft_number": _to_num(p["DRAFT_NUMBER"], f"{what} DRAFT_NUMBER", 0, 200),
        "from_year": _to_num(p["FROM_YEAR"], f"{what} FROM_YEAR", 1940, 2100),
        "two_way": supp.eq(1),
    })
    if (out["display_name"] == "").any():
        raise ValueError(f"{what}: player with no name")
    report["two_way"] = int(out["two_way"].sum())
    report["teams"] = int(out["team"].nunique())
    if strict and report["teams"] != 30:
        raise ValueError(f"{what}: rostered players cover {report['teams']} teams, not 30")
    report["per_team_min"] = int(out["team"].value_counts().min()) if len(out) else 0
    report["per_team_max"] = int(out["team"].value_counts().max()) if len(out) else 0
    report["null_position"] = int(out["position"].isna().sum())
    out = out.sort_values(["team", "last_name", "first_name"]).reset_index(drop=True)
    out.attrs["report"] = report
    return out


def _parse_birth(s: pd.Series, what: str) -> pd.Series:
    raw = s.where(s.notna() & s.astype(str).str.strip().ne(""))
    out = pd.to_datetime(raw.astype(str).str.strip().str.title(), format="%b %d, %Y", errors="coerce")
    bad = raw[out.isna() & raw.notna()]
    if len(bad):
        raise ValueError(f"{what}: unparseable BIRTH_DATE {bad.iloc[:3].tolist()}")
    return out.where(raw.notna()).astype("datetime64[ns]")


def parse_team_roster(payload: dict, season: str, team: str) -> pd.DataFrame:
    """CommonTeamRoster -> ROSTER_COLUMNS for one team-season.

    HOW_ACQUIRED and SUPPLEMENTAL_STATUS describe the player's CURRENT record,
    not the season asked for: the live 2018-19 HOU roster has Clint Capela as
    'Traded from ATL on 07/06/25', only for players whose current record links
    to that team (1 of 17 rows), so even a value's presence is later
    information. For a PAST season both are withheld (null, counted in
    attrs['report']), so neither can leak into a feature. For the live season
    they are kept as of the pull: HOW_ACQUIRED stays text, never a date
    ('Signed on 10/15/25' for a player re-signed to a two-way on 9/4/26).
    SUPPLEMENTAL_STATUS exists only in the live season's payload (2025-26 OKC
    has none); absent means unknown (NA), never False. EXP 'R' is a rookie (0).
    """
    code = teams.get(team).code
    what = f"roster {season} {code}"
    r = _frame(payload, "CommonTeamRoster", _ROSTER_REQ, what).copy()
    if r.empty:
        raise ValueError(f"{what}: empty roster")
    tcodes = _team_codes(r["TeamID"], what)
    if (tcodes != code).any():
        raise ValueError(f"{what}: payload is for {sorted(set(tcodes))}")
    want = str(util.season_start_year(season))
    if (r["SEASON"].astype(str) != want).any():
        raise ValueError(f"{what}: SEASON {sorted(set(r['SEASON'].astype(str)))} != {want}")
    _check_unique(r, ["PLAYER_ID"], what)
    live = is_live_season(season)
    exp = r["EXP"].astype(str).str.strip().replace({"R": "0"})
    none = pd.Series(None, index=r.index, dtype=object)
    acq = r["HOW_ACQUIRED"] if "HOW_ACQUIRED" in r else none
    supp = pd.to_numeric(r["SUPPLEMENTAL_STATUS"] if "SUPPLEMENTAL_STATUS" in r else none,
                         errors="coerce")
    report = {"season": season, "team": code, "rows": len(r), "point_in_time_fields": live,
              "how_acquired_withheld": 0 if live else int(acq.notna().sum()),
              "two_way_withheld": 0 if live else int(supp.notna().sum())}
    if not live:
        acq, supp = none, pd.Series(np.nan, index=r.index)
    out = pd.DataFrame({
        "season": season, "team": code, "person_id": r["PLAYER_ID"].astype("int64"),
        "player_name": r["PLAYER"].astype(str), "position": r["POSITION"].where(r["POSITION"].astype(str).str.strip().ne(""), None),
        "height_in": r["HEIGHT"].map(parse_height),
        "weight": _to_num(r["WEIGHT"], f"{what} WEIGHT", 130, 400),
        "birth_date": _parse_birth(r["BIRTH_DATE"], what),
        "age": _to_num(r["AGE"], f"{what} AGE", 16, 50),
        "exp": _to_num(exp, f"{what} EXP", 0, 30),
        "school": r["SCHOOL"] if "SCHOOL" in r else None,
        "how_acquired": acq.where(acq.notna(), None).astype(object),
        "two_way": supp.eq(1).astype("boolean").where(supp.notna(), pd.NA),
    })
    out = out.reset_index(drop=True)
    out.attrs["report"] = report
    return out


# ── Public fetch+parse API ───────────────────────────────────────────────────
def _guard_empty(df: pd.DataFrame, season: str, season_type: str, what: str,
                 refresh: bool | None = None) -> pd.DataFrame:
    """An empty parsed table where empty is not legitimate raises. This is the
    check that catches an empty raw payload cached BEFORE the first game and
    read back after it (refresh=False), which _get_raw never sees."""
    if df.empty and not empty_allowed(season, season_type, refresh):
        raise ValueError(f"{what}: 0 rows after the first scheduled {stype_code(season_type)} "
                         f"game of {season}; that is an outage, a stale cached pull or a "
                         f"format change, not 'no games'")
    return df


def _no_play_in(columns: list[str], dtypes: dict, season: str) -> pd.DataFrame:
    out = _empty(columns, dtypes)
    out.attrs["report"] = {"season": season, "season_type": "PI", "rows": 0, "games": 0,
                           "skipped": f"no play-in before {FIRST_PLAYIN_SEASON} (no call made)"}
    out.attrs["fetched_at"] = None
    return out


_schedule_memo: dict[str, tuple[float | None, pd.DataFrame]] = {}


def _schedule_frame(season: str, refresh: bool | None = None) -> pd.DataFrame:
    """parse_schedule of the season's raw schedule, parsed once per payload
    (player_games and team_games of three season types all ask for it)."""
    raw = fetch_schedule(season, refresh)
    stamp = _stamps.get(raw_path(season, "schedule"))
    hit = _schedule_memo.get(season)
    if hit is not None and hit[0] == stamp and stamp is not None:
        return hit[1]
    df = parse_schedule(raw, season)
    df.attrs["fetched_at"] = stamp
    _schedule_memo[season] = (stamp, df)
    return df


def neutral_site_games(season: str, refresh: bool | None = None) -> frozenset[str]:
    """Game ids the season's schedule marks is_neutral (isNeutral, the bubble,
    or the arena-city rule): 1 call when the schedule is not cached."""
    sch = _schedule_frame(season, refresh)
    return frozenset(sch.loc[sch["is_neutral"], "game_id"])


def player_games(season: str, season_type: str, refresh: bool | None = None) -> pd.DataFrame:
    """Whole-season player-game table (4 log calls + the season schedule when
    uncached; PlayIn before 2019-20 makes none). See parse_player_games.
    attrs['fetched_at'] = the oldest raw fetched_at it was built from."""
    code = stype_code(season_type)
    if code == "PI" and not has_play_in(season):
        return _no_play_in(PLAYER_GAME_COLUMNS, _PLAYER_DTYPES, season)
    df = parse_player_games(fetch_pgl(season, code, "Base", refresh),
                            fetch_pgl(season, code, "Advanced", refresh),
                            fetch_lgl(season, code, "P", refresh),
                            fetch_lgl(season, code, "T", refresh), season, code,
                            neutral_games=neutral_site_games(season, refresh))
    df = _guard_empty(df, season, code, f"player_games {season} {code}", refresh)
    df.attrs["fetched_at"] = _stamp_of(season, f"pgl_Base_{code}", f"pgl_Advanced_{code}",
                                       f"lgl_P_{code}", f"lgl_T_{code}", "schedule")
    return df


def team_games(season: str, season_type: str, refresh: bool | None = None) -> pd.DataFrame:
    """Whole-season team-game table (2 log calls + the schedule when uncached).
    See parse_team_games. attrs['fetched_at'] as in player_games."""
    code = stype_code(season_type)
    if code == "PI" and not has_play_in(season):
        return _no_play_in(TEAM_GAME_COLUMNS, _TEAM_DTYPES, season)
    df = parse_team_games(fetch_lgl(season, code, "T", refresh),
                          fetch_tgl(season, code, "Advanced", refresh), season, code,
                          neutral_games=neutral_site_games(season, refresh))
    df = _guard_empty(df, season, code, f"team_games {season} {code}", refresh)
    df.attrs["fetched_at"] = _stamp_of(season, f"lgl_T_{code}", f"tgl_Advanced_{code}",
                                       "schedule")
    return df


def schedule(season: str, refresh: bool | None = None) -> pd.DataFrame:
    """Full league schedule (1 call). A live season is refetched by default."""
    df = _schedule_frame(season, refresh).copy()
    df.attrs = {"report": dict(_schedule_memo[season][1].attrs["report"]),
                "fetched_at": _stamp_of(season, "schedule")}
    if df.empty:
        raise ValueError(f"schedule {season}: no games listed")
    return df


def player_index(season: str, refresh: bool | None = None) -> pd.DataFrame:
    """League roster for the LIVE season (1 call). Past seasons raise; see
    parse_player_index."""
    df = parse_player_index(fetch_player_index(season, refresh), season)
    if df.empty:
        raise ValueError(f"playerindex {season}: no rostered players")
    df.attrs["fetched_at"] = _stamp_of(season, "playerindex")
    return df


def team_roster(season: str, team: str, refresh: bool | None = None) -> pd.DataFrame:
    df = parse_team_roster(fetch_team_roster(season, team, refresh), season, team)
    df.attrs["fetched_at"] = _stamp_of(season, f"roster_{teams.get(team).code}")
    return df


def tidy_path(table: str, season: str) -> Path:
    """cache/<table>_<season>_v1.parquet."""
    util.season_end_year(season)
    return CACHE_DIR / f"{table}_{season}_v1.parquet"


# ── Script support (build_logs / build_schedule / build_players) ─────────────
def parse_season_range(spec: str) -> list[str]:
    """'2014-15:2025-26' -> every season in between; '2024-25' or
    '2014-15,2024-25' -> those seasons. Labels are validated."""
    out: list[str] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if ":" in part:
            a, b = part.split(":", 1)
            rng = util.seasons_between(a.strip(), b.strip())
            if not rng:
                raise ValueError(f"empty season range {part!r}")
            out += rng
        else:
            util.season_end_year(part)
            out.append(part)
    if not out:
        raise ValueError(f"no seasons in {spec!r}")
    return list(dict.fromkeys(out))


def progress_path(script: str) -> Path:
    return CACHE_DIR / "raw" / SOURCE / f"_progress_{script}.json"


def load_progress(script: str) -> dict:
    from ..cache import read_json
    obj = read_json(progress_path(script))
    return obj if isinstance(obj, dict) else {}


def save_progress(script: str, progress: dict) -> None:
    write_json(progress_path(script), progress)


def parquet_rows(path: Path) -> int | None:
    """Row count of an existing parquet without loading it; None when absent."""
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def shrink_reason(df: pd.DataFrame, path: Path, live: bool, count=None) -> str | None:
    """Why writing `df` over `path` would replace good data with less (None:
    it would not). Stale beats empty: for a LIVE season the tables only grow,
    so a pull with fewer rows than the file on disk (an empty preseason answer
    served after games were played, a truncated edge-cache copy) is refused.
    `count` maps a frame to the number compared (default: rows). A completed
    season is rewritten only on --force/--rebuild, so it is not guarded."""
    count = count or len
    if live and path.exists():
        old = pd.read_parquet(path)
        if count(df) < count(old):
            return (f"kept existing {path.name}: new pull has {count(df)} < "
                    f"{count(old)} on disk (stale beats empty)")
    return None


def write_tidy(df: pd.DataFrame, path: Path, meta: dict, live: bool,
               count=None) -> tuple[bool, str]:
    """Write a tidy table unless shrink_reason refuses it. Returns (written,
    message). meta['fetched_at'] should be the OLDEST raw fetched_at the table
    was built from (the cache rule: freshness comes from the embedded stamp);
    write_parquet stamps the write time only when the key is ABSENT, so an
    unknown age is passed as None (cache.is_fresh: never fresh), and
    `built_at` records the write time separately."""
    if "fetched_at" not in meta:
        raise ValueError(f"{path.name}: meta needs 'fetched_at' (the oldest raw stamp, or "
                         f"None when unknown); the write time would pass for fresh data")
    reason = shrink_reason(df, path, live, count)
    if reason:
        return False, reason
    meta = jsonable(dict(meta, built_at=stamp_now()))
    write_parquet(df, path, meta=meta)
    return True, f"wrote {path.name} ({len(df)} rows)"


def jsonable(obj):
    """Reports carry numpy scalars and Timestamps; parquet metadata is JSON."""
    import json

    def conv(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (pd.Timestamp, dt.date, dt.datetime)):
            return v.isoformat()
        return str(v)
    return json.loads(json.dumps(obj, default=conv))


def tidy_meta(path: Path) -> dict:
    return read_parquet_meta(path)
