"""The Odds API (the-odds-api.com): the licensed player lines (Barrett,
2026-11-09). DraftKings, FanDuel, BetMGM, PrizePicks and Underdog, display
only: feed lines are never a model input.

PURE. This module holds the vendor's facts, the settings reader, the request
builders and the parsers, and does no I/O at all: no network, no file, no
clock and no process environment (m12). bucketlib.live reads the key, adds it
to the params these builders return, makes the call through http.vendor_get
and hands the parsed JSON back here. Every time comes in as an argument, so
every rule below is tested on plain dicts (tests/test_oddsapi.py) and no test
can be a time bomb.

Traps handled here, each from the vendor's own pages (UT = the text copies in
BucketValueDev/understand/text, UL = the schema in understand/live):
  - Props come only from /v4/sports/basketball_nba/events/{id}/odds, one game
    per call. Books named in `bookmakers` bill as ONE region per 10 books, and
    `bookmakers` wins over `regions` (UT/docs_v4.txt:529). An 11th book doubles
    the bill, so at most 10 are ever named. The schema still marks `regions`
    required (UL schema 656-657); the first-key probe of 2026-11-11 found a
    bookmakers-only call accepted and billed the same, so SEND_REGIONS stays
    False.
  - The bill is the unique markets RETURNED (docs 1602, 1639), and an empty
    answer is free (1637-1638). Alternate (`_alternate`), period (`_q1`) and
    Yes/No markets are refused by the settings reader and by the request
    builder, so none is ever billed (M11).
  - /events is free and lists in-play and later games too (docs 1372, 1466).
    Only today's ET games are kept. commence_time is UTC: a 7 PM ET tip is
    23:00Z, and an 8 PM ET tip is 00:00Z on the NEXT UTC day.
  - Team names are full names ("Detroit Pistons", docs 2403). One that
    teams.canonical does not know is counted and skipped, never raised
    (raising would lose the whole pass).
  - A neutral-site game may list home and away the other way round, so events
    match schedule games on the ET date and the UNORDERED pair of teams (m18).
  - The player is only in outcome.description. A missing `point` skips the
    outcome; it is never read as 0. The market-level last_update is when the
    vendor last READ the market (UL schema 809), not when the line moved, and
    the bookmaker-level stamp is deprecated and absent on event odds (docs
    1053, 1501), so it is never read.
  - DFS prices (PrizePicks, Underdog, every us_dfs book) are indicative only
    (UT/bookmakers.txt:115) and are blanked to NaN, so nothing downstream can
    do price math on them. Demons and Goblins sit only in `_alternate` markets
    (bookmakers.txt:123-125), which the feed never buys, so every feed line is
    a standard line.
  - A 401 is a bad key OR an account out of credits; the body's code says
    which (UL schema 869-870; UT/error_codes.txt). Codes are read as whole
    words from the scrubbed snippet, and no vendor text is passed on.

Logging is through the repo's "bucketvalue" logger, numbers and our own codes
only. Every outcome a payload carries is accounted for in a counter.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .. import teams
from ..util import UTC, et_date_of

# The repo's logger (cache.logger), fetched by its name: importing
# bucketlib.cache would read the process environment and the disk at import,
# and this module does neither.
_log = logging.getLogger("bucketvalue")

# -- The vendor's facts -------------------------------------------------------
HOST = "api.the-odds-api.com"
SPORT = "basketball_nba"                                   # UT/docs_v4.txt:301
EVENTS_PATH = f"/v4/sports/{SPORT}/events"                 # free (docs 1372, 1466)
PARTICIPANTS_PATH = f"/v4/sports/{SPORT}/participants"     # 1 credit (docs 1728); probe only
SEND_REGIONS = False      # True only if the probe sees 422 MISSING_REGION and its step 6 bills 1
REGIONS = "us"            # what is sent when SEND_REGIONS is True (the probe's step 6)
MAX_BOOKS = 10            # up to 10 books named in `bookmakers` bill as 1 region (docs 529)

BOOKS = ("draftkings", "fanduel", "betmgm", "prizepicks", "underdog")   # Barrett, 2026-11-09
# Our titles, equal to props.BOOK_TITLE (layout B, amendment A2). The vendor's
# own `title` ("Underdog Fantasy") is never shown.
BOOK_TITLE = {"prizepicks": "PrizePicks", "underdog": "Underdog", "draftkings": "DraftKings",
              "fanduel": "FanDuel", "betmgm": "BetMGM"}
REFERENCE_BOOK = "prizepicks"       # the default pick, and the book the free top-up polls wait for
DFS_BOOKS = frozenset({"prizepicks", "underdog"})
# The vendor's US book keys (UT/bookmakers.txt): the five and others. A fact
# list for the request builder's check. The settings reader accepts ONLY the
# five (build amendment B6): BUCKETVALUE_ODDS_BOOKS may narrow them, never add
# one, since a sixth book would be bought and never shown (the board's pills
# show the five, props.BOOK_ORDER).
KNOWN_BOOKS = frozenset(BOOKS) | {"betrivers", "williamhill_us", "fanatics", "espnbet",
                                  "hardrockbet", "pick6", "dabble_us_dfs"}
# Every us_dfs key above. Their prices are indicative only whichever of them a
# call names (UT/bookmakers.txt:113-127), so the parser blanks them all.
US_DFS_BOOKS = DFS_BOOKS | {"pick6", "dabble_us_dfs"}
_OTHER_TITLE = {"betrivers": "BetRivers", "williamhill_us": "Caesars", "fanatics": "Fanatics",
                "espnbet": "theScore Bet", "hardrockbet": "Hard Rock Bet",
                "pick6": "DraftKings Pick6", "dabble_us_dfs": "Dabble"}

# Market key -> the board's stat label (RC G8). Every label round-trips through
# props.classify_stat to a projected stat (tests/test_oddsapi.py).
# player_field_goals stays out until a real payload shows it means made field
# goals (UNVERIFIED).
MARKET_LABEL = {
    "player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
    "player_threes": "3-PT Made", "player_blocks": "Blocks", "player_steals": "Steals",
    "player_blocks_steals": "Blks+Stls", "player_turnovers": "Turnovers",
    "player_points_rebounds_assists": "Pts+Rebs+Asts", "player_points_rebounds": "Pts+Rebs",
    "player_points_assists": "Pts+Asts", "player_rebounds_assists": "Rebs+Asts",
    "player_frees_made": "FT Made", "player_frees_attempts": "FT Attempted",
    "player_fantasy_points": "Fantasy Score"}     # PrizePicks rows only (FANTASY_BOOKS)
MAIN_MARKETS = tuple(list(MARKET_LABEL)[:12])     # the "12 main markets" of the budget plans
CORE_MARKETS = ("player_points",)
FANTASY_MARKET = "player_fantasy_points"
# PrizePicks' fantasy formula matches props.FANTASY_WEIGHTS (docs/data_notes.md);
# Underdog's is UNVERIFIED, so its fantasy lines are dropped and counted.
FANTASY_BOOKS = frozenset({"prizepicks"})
_ALTERNATE = "_alternate"
_PERIOD = re.compile(r"_(?:q[1-4]|h[12])$")       # period suffixes, read after `_alternate`
_YES_NO = frozenset({"player_first_basket", "player_first_team_basket",
                     "player_double_double", "player_triple_double"})   # UT/markets.txt:487-493
_VARIOUS = frozenset({"player_method_of_first_basket"})                  # markets.txt:495

# -- Frames ---------------------------------------------------------------------
# props.PARSED_COLUMNS, spelled out (this module cannot import props and props
# never imports this one; tests/test_oddsapi.py pins the two together), then
# the feed's own columns.
FEED_COLUMNS = ["name", "team", "position", "stat_type", "line", "start_time",
                "direction", "odds_type", "opp", "is_combo", "league", "source",
                "pick", "alt_pick", "pair", "pair_loose",
                "book", "book_title", "market", "price_over", "price_under", "last_update",
                "fetched_at", "event_id", "event_home", "event_away", "game_id", "game_date",
                "tip_utc"]
SOURCE = "oddsapi"            # the rows' `source` (props.FEED_SOURCE); never a book name
_FLOAT_COLS = ("line", "price_over", "price_under")
_TIME_COLS = ("last_update", "fetched_at")
EVENT_COLUMNS = ["event_id", "commence_utc", "home", "away", "et_date"]
MATCH_COLUMNS = ["event_id", "game_id", "game_date", "commence_utc", "tip_utc", "tbd",
                 "home", "away", "event_home", "event_away", "swapped"]

MAX_NAME_LEN = 80             # props.MAX_NAME_LEN
MAX_LINE = 1000.0             # props.MAX_LINE_VALUE: the paste rule for a line
_SAMPLE = 8                   # names or keys kept for a log line
_SAMPLE_LEN = 40              # characters each
_EVENT_ID = re.compile(r"[A-Za-z0-9]{32}")        # UL schema 640-646, 2016-2019
_LONG_TOKEN = re.compile(r"[A-Za-z0-9]{32,}")     # http.scrub's rule; the key is 40 of them

# -- Settings (design section 4.1) ------------------------------------------------
# Each is a root-level Streamlit secret on Cloud (Streamlit copies root-level
# strings and numbers into the process environment) or a plain environment
# variable locally, read on every call, never at import. The key variables
# (BUCKETVALUE_ODDS_API_KEY, BUCKETVALUE_ODDS_API_KEY_FILE) are read only by
# live._read_key, and BUCKETVALUE_LOG by cache.py; this reader never touches
# them. There is no board-layout setting: the board is layout B (amendment A10).
SETTING_DEFAULTS = {
    "BUCKETVALUE_ODDS_ENABLED": "1",                # "0" stops every vendor call, free ones too
    "BUCKETVALUE_ODDS_MARKETS": "player_points",
    "BUCKETVALUE_ODDS_CORE_MARKETS": "player_points",
    "BUCKETVALUE_ODDS_BOOKS": "draftkings,fanduel,betmgm,prizepicks,underdog",
    "BUCKETVALUE_ODDS_RESERVE": "",                 # unset: oddsbudget's max(25, 5% of the allowance)
    "BUCKETVALUE_ODDS_FIRST_PULL_ET": "08:50",
    "BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN": "30",
    "BUCKETVALUE_ODDS_MIN_REFRESH_MIN": "15",
    "BUCKETVALUE_ODDS_TIP_GUARD_MIN": "5",
    "BUCKETVALUE_ODDS_MAX_CALLS_10MIN": "40",
    "BUCKETVALUE_ODDS_PP_TOPUP": "1",
    "BUCKETVALUE_ODDS_REFRESHER": "1",
}
# Whole-number settings: (lowest, highest) accepted. Anything else is a typo and
# falls back to the default. A refresh floor of 0 would be no floor, and a cap of
# 0 paid calls a silent second kill switch, so both start at 1.
_RANGES = {
    "BUCKETVALUE_ODDS_RESERVE": (0, 100_000_000),
    "BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN": (0, 720),
    "BUCKETVALUE_ODDS_MIN_REFRESH_MIN": (1, 720),
    "BUCKETVALUE_ODDS_TIP_GUARD_MIN": (0, 720),
    "BUCKETVALUE_ODDS_MAX_CALLS_10MIN": (1, 1000),
}
_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"0", "false", "no", "off"})
_WHOLE = re.compile(r"[+-]?\d{1,10}(?:\.0*)?")
_HHMM = re.compile(r"(\d{1,2}):(\d{2})")
_DROPPED = {"alternate": "alternate market dropped", "period": "period market dropped",
            "yes_no": "yes/no market dropped", "various": "first-basket method market dropped",
            "unmapped": "unknown key dropped"}
_UNREADABLE = object()


@dataclass(frozen=True)
class Settings:
    """The feed's settings, parsed from strings; a bad value falls back to its
    default and leaves a fixed phrase in `problems` (the setting's NAME, never
    its value), which live logs as a WARNING."""
    enabled: bool
    books: tuple
    markets: tuple
    core: tuple
    reserve: int | None
    first_pull_et: dt.time
    last_pull_lead: dt.timedelta
    min_refresh: dt.timedelta
    tip_guard: dt.timedelta
    max_calls_10min: int
    pp_topup: bool
    refresher: bool
    problems: tuple[str, ...]


def _raw(env: Mapping, name: str):
    """The setting's text, stripped; None when unset or blank; _UNREADABLE for
    a value that is neither text nor a number (the environment holds text)."""
    v = env.get(name)
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return _UNREADABLE


def _whole(raw) -> int | None:
    """'30', '+30' or '30.0' -> 30; anything else None."""
    if not isinstance(raw, str) or not _WHOLE.fullmatch(raw):
        return None
    return int(raw.split(".")[0])


def _flag(env: Mapping, name: str, problems: list) -> bool:
    default = SETTING_DEFAULTS[name] == "1"
    raw = _raw(env, name)
    if raw is None:
        return default
    low = raw.lower() if isinstance(raw, str) else ""
    if low in _ON:
        return True
    if low in _OFF:
        return False
    problems.append(f"setting {name}: unreadable value; default used")
    return default


def _enabled(env: Mapping, problems: list) -> bool:
    """The kill switch. Unset means on; an unreadable value means OFF, not the
    default: whoever typed it was trying to change the switch, and a feed left
    on by a typo spends credits."""
    name = "BUCKETVALUE_ODDS_ENABLED"
    raw = _raw(env, name)
    if raw is None:
        return True
    low = raw.lower() if isinstance(raw, str) else ""
    if low in _ON:
        return True
    if low not in _OFF:
        problems.append(f"setting {name}: unreadable value; the feed is switched off")
    return False


def _ranged(env: Mapping, name: str, problems: list) -> int | None:
    text = SETTING_DEFAULTS[name]
    default = int(text) if text else None
    raw = _raw(env, name)
    if raw is None:
        return default
    n = _whole(raw)
    if n is None:
        problems.append(f"setting {name}: unreadable value; default used")
        return default
    lo, hi = _RANGES[name]
    if not lo <= n <= hi:
        problems.append(f"setting {name}: out of range; default used")
        return default
    return n


def _clock(env: Mapping, name: str, problems: list) -> dt.time:
    def parse(text):
        m = _HHMM.fullmatch(text) if isinstance(text, str) else None
        if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
            return dt.time(int(m.group(1)), int(m.group(2)))
        return None
    raw = _raw(env, name)
    t = None if raw is None else parse(raw)
    if raw is not None and t is None:
        problems.append(f"setting {name}: unreadable value; default used")
    return t if t is not None else parse(SETTING_DEFAULTS[name])


def _key_list(text: str) -> list[str]:
    return [p.strip().lower() for p in text.split(",") if p.strip()]


def _market_verdict(key) -> tuple[str, str]:
    """(verdict, base) for a market key, suffix first: `_alternate` sits
    outermost, then a period suffix, then the base. verdict is ok,
    alternate, period, yes_no, various or unmapped."""
    k = key.strip().lower() if isinstance(key, str) else ""
    if k.endswith(_ALTERNATE):
        return "alternate", k[:-len(_ALTERNATE)]
    if _PERIOD.search(k):
        return "period", _PERIOD.sub("", k)
    if k in _YES_NO:
        return "yes_no", k
    if k in _VARIOUS:
        return "various", k
    if k not in MARKET_LABEL:
        return "unmapped", k
    return "ok", k


def _markets(env: Mapping, name: str, problems: list) -> tuple[str, ...]:
    raw = _raw(env, name)
    if raw is _UNREADABLE:
        problems.append(f"setting {name}: unreadable value; default used")
        raw = None
    out: list[str] = []
    for key in _key_list(raw if raw is not None else SETTING_DEFAULTS[name]):
        verdict, base = _market_verdict(key)
        if verdict != "ok":
            problems.append(f"setting {name}: {_DROPPED[verdict]}")
        elif base not in out:
            out.append(base)
    if not out:
        problems.append(f"setting {name}: empty list; default used")
        out = _key_list(SETTING_DEFAULTS[name])
    return tuple(out)


def _books(env: Mapping, problems: list) -> tuple[str, ...]:
    """The books to buy: the five (BOOKS) or fewer, in the order named. The
    setting may only NARROW the five (build amendment B6). Any other name is
    dropped with a fixed phrase in `problems` (never the name itself), and an
    empty result means all five."""
    name = "BUCKETVALUE_ODDS_BOOKS"
    raw = _raw(env, name)
    if raw is _UNREADABLE:
        problems.append(f"setting {name}: unreadable value; default used")
        raw = None
    out: list[str] = []
    for key in _key_list(raw if raw is not None else SETTING_DEFAULTS[name]):
        if key not in KNOWN_BOOKS:
            problems.append(f"setting {name}: unknown key dropped")
        elif key not in BOOKS:
            problems.append(f"setting {name}: book outside the five dropped")
        elif key in out:
            continue
        elif len(out) >= MAX_BOOKS:            # an 11th book doubles the bill (docs 529)
            problems.append(f"setting {name}: more than 10 books; the rest dropped")
        else:
            out.append(key)
    if not out:
        problems.append(f"setting {name}: empty list; default used")
        out = _key_list(SETTING_DEFAULTS[name])
    return tuple(out)


def read_settings(environ: Mapping[str, str]) -> Settings:
    """Every feed setting, from `environ` ONLY. live passes the process
    environment, or a test's mapping as it is: an empty mapping is read as
    empty, never swapped for anything else (m11). Unset values take their
    defaults; bad ones take their defaults too and say so in `problems`."""
    if not isinstance(environ, Mapping):
        raise TypeError("read_settings needs a mapping")
    problems: list[str] = []
    enabled = _enabled(environ, problems)
    books = _books(environ, problems)
    markets = _markets(environ, "BUCKETVALUE_ODDS_MARKETS", problems)
    core = _markets(environ, "BUCKETVALUE_ODDS_CORE_MARKETS", problems)
    # What the governor KEEPS when the full list does not fit, so a subset of it.
    kept = tuple(k for k in core if k in markets)
    if not kept:
        problems.append("setting BUCKETVALUE_ODDS_CORE_MARKETS: not in the market list; "
                        "the first market used")
        kept = markets[:1]
    minutes = {n: _ranged(environ, n, problems) for n in (
        "BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN", "BUCKETVALUE_ODDS_MIN_REFRESH_MIN",
        "BUCKETVALUE_ODDS_TIP_GUARD_MIN")}
    return Settings(
        enabled=enabled, books=books, markets=markets, core=kept,
        reserve=_ranged(environ, "BUCKETVALUE_ODDS_RESERVE", problems),
        first_pull_et=_clock(environ, "BUCKETVALUE_ODDS_FIRST_PULL_ET", problems),
        last_pull_lead=dt.timedelta(minutes=minutes["BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN"]),
        min_refresh=dt.timedelta(minutes=minutes["BUCKETVALUE_ODDS_MIN_REFRESH_MIN"]),
        tip_guard=dt.timedelta(minutes=minutes["BUCKETVALUE_ODDS_TIP_GUARD_MIN"]),
        max_calls_10min=_ranged(environ, "BUCKETVALUE_ODDS_MAX_CALLS_10MIN", problems),
        pp_topup=_flag(environ, "BUCKETVALUE_ODDS_PP_TOPUP", problems),
        refresher=_flag(environ, "BUCKETVALUE_ODDS_REFRESHER", problems),
        problems=tuple(dict.fromkeys(problems)))


# -- Requests ---------------------------------------------------------------------
# The key is NOT here: live adds `apiKey` to the params at call time, and
# http.vendor_get keeps it out of every URL string. Never sent:
# includeLinks, includeSids, includeMultipliers, includeBetLimits.
def events_request() -> tuple[str, dict]:
    """The free /events call: today's and later games with their ids."""
    return EVENTS_PATH, {"dateFormat": "iso"}


def _request_keys(values, allowed, most: int) -> tuple[str, ...] | None:
    if isinstance(values, str) or values is None:
        return None
    try:
        keys = tuple(dict.fromkeys(values))
    except TypeError:
        return None
    if not 1 <= len(keys) <= most or not all(isinstance(k, str) and k in allowed for k in keys):
        return None
    return keys


def event_odds_request(event_id, books, markets) -> tuple[str, dict]:
    """(path, params) for one game's lines. The books go in `bookmakers`, and
    `regions` is sent only when SEND_REGIONS is True. Raises ValueError, with
    a fixed text that names nothing, for an event id that is not 32 letters
    and digits, or books or markets the settings reader would refuse."""
    eid = event_id if isinstance(event_id, str) and _EVENT_ID.fullmatch(event_id) else None
    if eid is None:
        raise ValueError("bad event id")
    bks = _request_keys(books, KNOWN_BOOKS, MAX_BOOKS)
    if bks is None:
        raise ValueError("bad books")
    mks = _request_keys(markets, MARKET_LABEL, len(MARKET_LABEL))
    if mks is None:
        raise ValueError("bad markets")
    params = {"bookmakers": ",".join(bks), "markets": ",".join(mks),
              "oddsFormat": "american", "dateFormat": "iso"}
    if SEND_REGIONS:
        params["regions"] = REGIONS
    return f"{EVENTS_PATH}/{eid}/odds", params


# -- Usage and errors -------------------------------------------------------------
@dataclass(frozen=True)
class Usage:
    """The vendor's own credit counters, off one response's headers:
    x-requests-remaining, x-requests-used and x-requests-last (docs
    1630-1634). None where a header is missing or unreadable."""
    remaining: int | None
    used: int | None
    last: int | None

    @property
    def allowance(self) -> int | None:
        """The plan's size, used + remaining, when both are known."""
        if self.remaining is None or self.used is None:
            return None
        return self.used + self.remaining


def _count(value) -> int | None:
    """A header's credit count: a whole number of at least 0, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        n = int(value) if math.isfinite(value) and value == int(value) else None
    else:
        n = _whole(value.strip()) if isinstance(value, str) else None
    return n if n is not None and n >= 0 else None


def usage_from_headers(headers) -> Usage:
    """Usage from a response's headers, whatever the case of their names."""
    try:
        items = list(headers.items())
    except Exception:  # noqa: BLE001 - no headers, no usage
        return Usage(None, None, None)
    low: dict = {}
    for k, v in items:
        if isinstance(k, str):
            low.setdefault(k.strip().lower(), v)
    return Usage(_count(low.get("x-requests-remaining")), _count(low.get("x-requests-used")),
                 _count(low.get("x-requests-last")))


ERROR_CODES = (   # all 34 codes on UT/error_codes.txt:27-265, in page order (m4)
    "MISSING_KEY", "INVALID_KEY", "DEACTIVATED_KEY", "EXCEEDED_FREQ_LIMIT", "OUT_OF_USAGE_CREDITS",
    "MISSING_REGION", "INVALID_REGION", "INVALID_BOOKMAKERS", "MISSING_MARKET", "INVALID_MARKET",
    "INVALID_MARKET_COMBO", "INVALID_DATE_FORMAT", "INVALID_ODDS_FORMAT", "INVALID_ALL_SPORTS_PARAM",
    "INVALID_SPORT", "UNKNOWN_SPORT", "INVALID_SCORES_DAYS_FROM", "INVALID_EVENT_IDS",
    "INVALID_EVENT_ID", "EVENT_NOT_FOUND", "MISSING_HISTORICAL_TIMESTAMP",
    "INVALID_HISTORICAL_TIMESTAMP", "INVALID_COMMENCE_TIME_FROM", "INVALID_COMMENCE_TIME_TO",
    "INVALID_COMMENCE_TIME_RANGE", "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN",
    "INVALID_PARTICIPANT_ID", "INVALID_INCLUDE_LINKS", "INVALID_INCLUDE_SIDS",
    "INVALID_INCLUDE_BET_LIMITS", "INVALID_INCLUDE_MULTIPLIERS", "INVALID_INCLUDE_ROTATION_NUMBERS",
    "HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE", "INVALID_STATUS")
# A code counts only as a WHOLE word: `\b` treats "_" as part of a word, so
# INVALID_EVENT_ID never matches inside INVALID_EVENT_IDS or MISSING_KEY_X.
_CODE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ERROR_CODES), key=len, reverse=True))
                   + r")\b")
FAILURE_KINDS = ("ok", "out_of_credits", "bad_key", "not_found", "bad_request", "rate_limited",
                 "server")


def _status(status) -> int:
    return status if isinstance(status, int) and not isinstance(status, bool) \
        and 0 <= status <= 999 else 0


def error_code(status, snippet) -> str:
    """The documented code in an error body's scrubbed snippet, found as a
    whole word; else "HTTP_<status>". Always one of OUR strings, never the
    vendor's text: the body's field names are UNVERIFIED (RF 2.7)."""
    if isinstance(snippet, str):
        m = _CODE.search(snippet)
        if m:
            return m.group(1)
    return f"HTTP_{_status(status)}"


def failure_kind(status, code) -> str:
    """What a response means for the feed (design 3.5, 3.8): a 401 is out of
    credits only when its code says OUT_OF_USAGE_CREDITS; any other 401
    (INVALID_KEY, DEACTIVATED_KEY, MISSING_KEY, no readable code) is a bad
    key."""
    s = _status(status)
    if s == 200:
        return "ok"
    if s == 401:
        return "out_of_credits" if code == "OUT_OF_USAGE_CREDITS" else "bad_key"
    return {404: "not_found", 422: "bad_request", 429: "rate_limited"}.get(s, "server")


# -- Small readers ------------------------------------------------------------------
# Every instant the vendor sends is a tip or a market stamp of this century. A
# time outside these years is junk (Go's zero time "0001-01-01T00:00:00Z" is
# the usual placeholder), and near either end of the calendar converting it
# to UTC or to ET raises OverflowError (build 4's review, L1-1).
_YEARS = (1900, 2200)


def _utc(value) -> dt.datetime | None:
    """An aware UTC datetime from an ISO string; None for anything else (a
    naive time is refused, never assumed), an instant that overflows the
    calendar once its offset is applied included, and one outside _YEARS:
    junk is dropped and counted by the caller, never raised."""
    if not isinstance(value, str) or not 0 < len(value.strip()) <= 40:
        return None
    try:
        t = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if t.tzinfo is None or t.utcoffset() is None:
            return None
        t = t.astimezone(UTC)
    except (ValueError, OverflowError):
        return None
    return t if _YEARS[0] <= t.year <= _YEARS[1] else None


def _iso_day(value) -> str | None:
    """'YYYY-MM-DD' for a date, a datetime's own calendar date (a schedule's
    game_date read back as a midnight Timestamp) or an ISO date string; None
    otherwise. NaT is a datetime to Python, so it is refused first."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, dt.datetime):            # pd.Timestamp included
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError:
            return None
    return None


def _number(value) -> float | None:
    """A finite number, or None. Booleans are refused (float(True) is 1.0),
    and so are NaN and the infinities json and float() both accept."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value.strip()[:40]) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _point(value) -> float | None:
    """A line, by the paste rule (props._parse_line): finite, 0 to 1000. A
    missing point is None, never 0."""
    x = _number(value)
    return x if x is not None and 0 <= x <= MAX_LINE else None


def _clean(value, maxlen: int) -> str | None:
    if not isinstance(value, str):
        return None
    s = " ".join(value.split())
    return s if s and len(s) <= maxlen else None


def _loggable(value) -> str:
    """A vendor string cut for a log line: printable characters, one line, any
    run of 32 or more letters and digits redacted (http.scrub's long-token
    rule, so nothing key-shaped can reach a log), then 40 characters."""
    if not isinstance(value, str):
        return "(not text)"
    s = " ".join("".join(ch for ch in value if ch.isprintable()).split())
    return _LONG_TOKEN.sub("[redacted]", s)[:_SAMPLE_LEN] or "(blank)"   # the cut comes last


def _sample(into: list, value) -> None:
    if value not in into and len(into) < _SAMPLE:
        into.append(value)


def _implied(price) -> float | None:
    """The probability an American price implies; None for a price that is
    not an American price."""
    if price is None:
        return None
    if price <= -100:
        return -price / (-price + 100.0)
    if price >= 100:
        return 100.0 / (price + 100.0)
    return None


def _records(frame) -> list[dict]:
    if isinstance(frame, pd.DataFrame):
        return frame.to_dict("records")
    if isinstance(frame, list):
        return [r for r in frame if isinstance(r, dict)]
    return []


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """FEED_COLUMNS with one set of dtypes, whatever built the frame: floats
    for the line and prices, UTC datetimes for the two stamps, objects else."""
    out = df.loc[:, FEED_COLUMNS].copy()
    for c in FEED_COLUMNS:
        if c in _FLOAT_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        elif c in _TIME_COLS:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
        else:
            out[c] = out[c].astype(object)
    return out.reset_index(drop=True)


def empty_feed() -> pd.DataFrame:
    """A feed frame with no rows: FEED_COLUMNS and their dtypes."""
    return _typed(pd.DataFrame(columns=FEED_COLUMNS))


# -- Events -----------------------------------------------------------------------------
def parse_events(payload, today: dt.date) -> tuple[pd.DataFrame, dict]:
    """The /events answer as today's games: EVENT_COLUMNS (event_id,
    commence_utc as an aware UTC time, home and away as tricodes, et_date),
    one row per event, in tip order.

    Dropped and counted, never raised: an entry that is not an event or has
    no readable id or time (bad_event), a team teams.canonical does not know
    (unknown_team_events, with up to 8 of the raw names, 40 characters each,
    in unknown_team), an event whose ET date is not `today` (other_date; the
    endpoint lists in-play and later games too), and a repeated id
    (duplicate_id). events_seen is their sum plus `kept`."""
    if isinstance(today, dt.datetime) or not isinstance(today, (dt.date, str)) \
            or _iso_day(today) is None:
        raise TypeError("today must be a date")
    day = _iso_day(today)
    stats = {"events_seen": 0, "kept": 0, "bad_event": 0, "unknown_team_events": 0,
             "unknown_team": [], "other_date": 0, "duplicate_id": 0,
             "bad_payload": not isinstance(payload, list)}
    rows, seen = [], set()
    for ev in payload if isinstance(payload, list) else ():
        stats["events_seen"] += 1
        if not isinstance(ev, dict):
            stats["bad_event"] += 1
            continue
        eid = ev.get("id") if isinstance(ev.get("id"), str) else None
        eid = eid if eid and _EVENT_ID.fullmatch(eid) else None
        tip = _utc(ev.get("commence_time"))
        if eid is None or tip is None:
            stats["bad_event"] += 1
            continue
        raw_home, raw_away = ev.get("home_team"), ev.get("away_team")
        home, away = teams.canonical(raw_home), teams.canonical(raw_away)
        if home is None or away is None:
            stats["unknown_team_events"] += 1
            for raw, code in ((raw_home, home), (raw_away, away)):
                if code is None:
                    _sample(stats["unknown_team"], _loggable(raw))
            continue
        if home == away:
            stats["bad_event"] += 1
            continue
        try:
            d = et_date_of(tip)
        except (OverflowError, ValueError):      # never raised (L1-1): a bad event, counted
            stats["bad_event"] += 1
            continue
        if d.isoformat() != day:
            stats["other_date"] += 1
            continue
        if eid in seen:
            stats["duplicate_id"] += 1
            continue
        seen.add(eid)
        rows.append({"event_id": eid, "commence_utc": tip, "home": home, "away": away,
                     "et_date": d})
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["commence_utc"] = pd.to_datetime(df["commence_utc"], utc=True)
    df = df.sort_values(["commence_utc", "event_id"], kind="mergesort").reset_index(drop=True)
    stats["kept"] = len(df)
    return df, stats


def match_events(events, games) -> tuple[pd.DataFrame, dict]:
    """Vendor events joined to schedule games on the ET date and the UNORDERED
    pair of teams (m18): MATCH_COLUMNS, one row per matched game, home and
    away the SCHEDULE's, event_home and event_away the vendor's.

    `events` is parse_events' frame. `games` holds game_id, game_date (a date
    or an ISO string), home and away, and optionally tip_utc and tbd (or
    time_tbd), which are carried through. A game whose vendor home and away
    are the schedule's reversed (a neutral site) still matches, is counted in
    `swapped` and logged at INFO. Counted, never raised: events with no game,
    games with no event, and a date and pair that is not one to one (two
    events for one game: all refused, refused_event and refused_game), plus
    unreadable rows (bad_event, bad_game). This season's neutral-site games
    include NOP at SAS in Paris on 2027-01-14 and SAS at NOP in Manchester on
    2027-01-17: the same pair three days apart, kept apart by the date."""
    stats = {"events": 0, "games": 0, "matched": 0, "swapped": 0, "event_no_game": 0,
             "game_no_event": 0, "refused_event": 0, "refused_game": 0, "bad_event": 0,
             "bad_game": 0, "swapped_games": []}
    by_event: dict = {}
    for r in _records(events):
        stats["events"] += 1
        eid = r.get("event_id") if isinstance(r.get("event_id"), str) else ""
        home, away = teams.canonical(r.get("home")), teams.canonical(r.get("away"))
        day, tip = _iso_day(r.get("et_date")), r.get("commence_utc")
        if not _EVENT_ID.fullmatch(eid) or home is None or away is None or home == away \
                or day is None:
            stats["bad_event"] += 1
            continue
        by_event.setdefault((day, frozenset((home, away))), []).append(
            {"event_id": eid, "commence_utc": tip, "home": home, "away": away})
    by_game: dict = {}
    for r in _records(games):
        stats["games"] += 1
        gid = r.get("game_id")
        home, away = teams.canonical(r.get("home")), teams.canonical(r.get("away"))
        day = _iso_day(r.get("game_date"))
        if not isinstance(gid, str) or not gid or home is None or away is None \
                or home == away or day is None:
            stats["bad_game"] += 1
            continue
        tbd = r.get("tbd", r.get("time_tbd"))
        by_game.setdefault((day, frozenset((home, away))), []).append(
            {"game_id": gid, "game_date": day, "home": home, "away": away,
             "tip_utc": r.get("tip_utc"), "tbd": tbd is not None and str(tbd) == "True"})
    rows = []
    for key, gl in by_game.items():
        el = by_event.get(key, [])
        if not el:
            stats["game_no_event"] += len(gl)
            continue
        if len(gl) > 1 or len(el) > 1:
            stats["refused_game"] += len(gl)
            stats["refused_event"] += len(el)
            continue
        g, e = gl[0], el[0]
        swapped = e["home"] != g["home"]
        if swapped:
            stats["swapped"] += 1
            _sample(stats["swapped_games"], g["game_id"])
            _log.info("oddsapi: event matched with home and away swapped: %s (%s at %s)",
                      g["game_id"], g["away"], g["home"])
        rows.append({"event_id": e["event_id"], "game_id": g["game_id"],
                     "game_date": g["game_date"], "commence_utc": e["commence_utc"],
                     "tip_utc": g["tip_utc"], "tbd": g["tbd"], "home": g["home"],
                     "away": g["away"], "event_home": e["home"], "event_away": e["away"],
                     "swapped": swapped})
    stats["event_no_game"] = sum(len(el) for key, el in by_event.items() if key not in by_game)
    stats["matched"] = len(rows)
    df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    for c in ("commence_utc", "tip_utc"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.sort_values(["commence_utc", "game_id"], kind="mergesort").reset_index(drop=True)
    return df, stats


# -- Event odds -------------------------------------------------------------------------
# Every outcome a payload carries lands in exactly one place: a kept line, or
# one of these counters (outcomes_seen = kept_outcomes + their sum, in total
# and per book). unknown_book has no book of ours to sit under, so per book the
# sum runs over BOOK_COUNTERS. bad_bookmaker and bad_market count STRUCTURES (an
# entry that is not an object, a `markets` or `outcomes` that is not a list),
# which hold no outcome that could be counted.
COUNTERS = ("unknown_book", "unmapped_market", "refused_market", "alt_dropped",
            "unrequested_market", "fantasy_dropped", "bad_outcome", "duplicate_outcome",
            "extra_points")
BOOK_COUNTERS = COUNTERS[1:]
_SIDES = {"over": "over", "under": "under"}


def _tally(keys) -> dict:
    return {"outcomes_seen": 0, "kept_outcomes": 0, "rows": 0, **{k: 0 for k in keys}}


def _outcome(oc):
    """(side, player, point, price) from one outcome, or None when any part
    is unreadable: the side only "Over" or "Under", the player only from
    `description`, a point by the paste rule, and a finite price."""
    if not isinstance(oc, dict) or not isinstance(oc.get("name"), str):
        return None
    side = _SIDES.get(oc["name"].strip().lower())
    player = _clean(oc.get("description"), MAX_NAME_LEN)
    point, price = _point(oc.get("point")), _number(oc.get("price"))
    if side is None or player is None or point is None or price is None:
        return None
    return side, player, point, price


def _main_point(points: dict, dfs: bool) -> float:
    """The one main line among a player's points in one market: the
    both-sided point; among several, the one whose two implied probabilities
    are closest (DFS prices mean nothing, so there the first); with no
    both-sided point, the first."""
    both = [p for p, sides in points.items() if len(sides) == 2]
    if not both:
        return next(iter(points))
    if len(both) == 1 or dfs:
        return both[0]

    def gap(p):
        a, b = _implied(points[p]["over"]), _implied(points[p]["under"])
        return math.inf if a is None or b is None else abs(a - b)
    return min(both, key=gap)


def parse_event_odds(payload, *, books, markets, fetched_at: dt.datetime) -> pd.DataFrame:
    """One game's event-odds answer as feed lines: FEED_COLUMNS, one row per
    (book, player, market), every row a standard line. Never raises on the
    payload; `fetched_at` must be an aware datetime (the caller's clock, never
    one read here).

    Rules (design 2.2): only books in `books` and markets in `markets` count;
    `_alternate`, period and Yes/No markets are counted and never kept; the
    sides come from "Over"/"Under" pairs grouped by point ("both"), or one
    side alone ("more" for Over, "less" for Under); one main line per book,
    player and market (_main_point), the other points counted in
    extra_points; DFS prices blanked to NaN; fantasy lines kept for
    PrizePicks only; last_update from the MARKET level only.

    df.attrs carries the counters (COUNTERS, outcomes_seen, kept_outcomes and
    rows) in total and per book under by_book, books_answered (the requested
    books that had a bookmaker entry, readable or not), markets_requested
    (the market keys this parse was told to keep: what the call asked for,
    which merge_refresh judges a refresh on), bookmaker_entries (0 on an
    empty, free answer), bad_bookmaker and bad_market, unknown_team (an empty
    frame when a team is not known), bad_payload (not an event object, or no
    bookmakers list), samples of unknown book and market keys, and the
    event's own id and tricodes. game_id, game_date and tip_utc are left for
    live, which fills them from the event match."""
    if not isinstance(fetched_at, dt.datetime) or fetched_at.tzinfo is None \
            or fetched_at.utcoffset() is None:
        raise ValueError("fetched_at must be an aware datetime")
    fetched = pd.Timestamp(fetched_at).tz_convert("UTC")
    want_books = tuple(dict.fromkeys(b for b in (books or ()) if isinstance(b, str)))
    asked_markets = tuple(dict.fromkeys(m for m in (markets or ()) if isinstance(m, str)))
    want_markets = frozenset(asked_markets)
    attrs = {**_tally(COUNTERS), "by_book": {}, "books_answered": (),
             "markets_requested": asked_markets, "bookmaker_entries": 0,
             "bad_bookmaker": 0, "bad_market": 0, "unknown_team": False, "bad_payload": False,
             "unknown_books": [], "unmapped_keys": [], "event_id": None, "event_home": None,
             "event_away": None}
    if not isinstance(payload, dict):
        attrs["bad_payload"] = True
        return _frame([], attrs)
    eid = payload.get("id")
    attrs["event_id"] = eid if isinstance(eid, str) and _EVENT_ID.fullmatch(eid) else None
    home, away = teams.canonical(payload.get("home_team")), teams.canonical(payload.get("away_team"))
    if home is None or away is None or home == away:
        attrs["unknown_team"] = True
        return _frame([], attrs)
    attrs["event_home"], attrs["event_away"] = home, away
    bookmakers = payload.get("bookmakers")
    if not isinstance(bookmakers, list):
        attrs["bad_payload"] = True
        return _frame([], attrs)
    tip = _utc(payload.get("commence_time"))
    start_time = tip.strftime("%Y-%m-%dT%H:%M:%SZ") if tip is not None else None

    by_book = attrs["by_book"]

    def add(book_tally, name, n=1):
        attrs[name] += n
        if book_tally is not None:
            book_tally[name] += n

    groups: dict = {}       # (book, market, player key) -> {"name", "last_update", "points"}
    answered = set()
    for bm in bookmakers:
        attrs["bookmaker_entries"] += 1
        if not isinstance(bm, dict):
            attrs["bad_bookmaker"] += 1
            continue
        book = bm.get("key").strip().lower() if isinstance(bm.get("key"), str) else None
        mlist = bm.get("markets")
        tally = None
        if book in want_books:
            answered.add(book)                    # present, readable or not (merge decides)
            tally = by_book.setdefault(book, _tally(BOOK_COUNTERS))
        if book is None or not isinstance(mlist, list):
            attrs["bad_bookmaker"] += 1
            continue
        if tally is None:
            _sample(attrs["unknown_books"], _loggable(book))
        for mk in mlist:
            outcomes = mk.get("outcomes") if isinstance(mk, dict) else None
            if not isinstance(outcomes, list):
                attrs["bad_market"] += 1
                continue
            n = len(outcomes)
            add(tally, "outcomes_seen", n)
            if tally is None:
                add(None, "unknown_book", n)
                continue
            verdict, base = _market_verdict(mk.get("key"))
            if verdict == "alternate":
                add(tally, "alt_dropped", n)
                continue
            if verdict in ("period", "yes_no", "various"):
                add(tally, "refused_market", n)
                continue
            if verdict == "unmapped":
                add(tally, "unmapped_market", n)
                _sample(attrs["unmapped_keys"], _loggable(mk.get("key")))
                continue
            if base not in want_markets:
                add(tally, "unrequested_market", n)
                continue
            if base == FANTASY_MARKET and book not in FANTASY_BOOKS:
                add(tally, "fantasy_dropped", n)
                continue
            stamp = _utc(mk.get("last_update"))       # the market's own; never the bookmaker's
            for oc in outcomes:
                got = _outcome(oc)
                if got is None:
                    add(tally, "bad_outcome")
                    continue
                side, player, point, price = got
                g = groups.setdefault((book, base, player.casefold()),
                                      {"name": player, "last_update": stamp, "points": {}})
                sides = g["points"].setdefault(point, {})
                if side in sides:
                    add(tally, "duplicate_outcome")
                    continue
                sides[side] = price

    rows = []
    for (book, base, _), g in groups.items():
        tally = by_book[book]
        dfs = book in US_DFS_BOOKS
        main = _main_point(g["points"], dfs)
        for p, sides in g["points"].items():
            if p != main:
                add(tally, "extra_points", len(sides))
        sides = g["points"][main]
        add(tally, "kept_outcomes", len(sides))
        add(tally, "rows")
        direction = "both" if len(sides) == 2 else ("more" if "over" in sides else "less")
        rows.append({
            "name": g["name"], "team": None, "position": None, "stat_type": MARKET_LABEL[base],
            "line": main, "start_time": start_time, "direction": direction,
            "odds_type": "standard", "opp": None, "is_combo": False, "league": "NBA",
            "source": SOURCE, "pick": None, "alt_pick": None, "pair": None, "pair_loose": None,
            "book": book, "book_title": BOOK_TITLE.get(book) or _OTHER_TITLE.get(book, book),
            "market": base,
            "price_over": math.nan if dfs else sides.get("over", math.nan),
            "price_under": math.nan if dfs else sides.get("under", math.nan),
            "last_update": g["last_update"], "fetched_at": fetched,
            "event_id": attrs["event_id"], "event_home": home, "event_away": away,
            "game_id": None, "game_date": None, "tip_utc": None})
    attrs["books_answered"] = tuple(b for b in want_books if b in answered)
    return _frame(rows, attrs)


def _frame(rows: list[dict], attrs: dict) -> pd.DataFrame:
    df = _typed(pd.DataFrame(rows, columns=FEED_COLUMNS))
    df.attrs.update(attrs)
    return df


# -- Merging a refresh --------------------------------------------------------------------
def _feed_or_none(frame) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or not set(FEED_COLUMNS) <= set(frame.columns):
        return None
    return _typed(frame)


def _line_key(row) -> tuple:
    return (str(row["name"]).casefold(), row["market"])


def _plain(v):
    """One cell as a plain comparable value: None for every kind of missing."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (str, bool)):
        return v
    if isinstance(v, (dt.datetime, dt.date)):     # pd.Timestamp included
        return v.isoformat()
    if isinstance(v, float):
        return None if math.isnan(v) else v
    return v if isinstance(v, int) else str(v)


def _same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    def canon(df):
        return sorted((tuple(_plain(v) for v in rec)
                       for rec in df.itertuples(index=False, name=None)), key=repr)
    return len(a) == len(b) and canon(a) == canon(b)


def _asked_markets(explicit, attrs) -> frozenset | None:
    """The markets a call asked for: `explicit` when the caller states them,
    else the markets_requested parse_event_odds recorded; None (judge every
    market) when neither is known."""
    raw = explicit if explicit is not None else attrs.get("markets_requested")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = (raw,)
    try:
        return frozenset(m for m in raw if isinstance(m, str))
    except TypeError:
        return None


def merge_refresh(prev, new, books_requested, *, markets_requested=None) -> tuple[pd.DataFrame, dict]:
    """One game's stored lines after a pull: spec rule 9, "an empty or failed
    fetch never overwrites good data", applied per book, with a book's
    presence decided by its KEPT ROWS, not by its bookmaker entry (M3).

    `prev` is the game's stored frame (None at the first pull), `new` is
    parse_event_odds' frame and `books_requested` the books the call named
    (the five on a full pull, ("prizepicks",) on a top-up). Books not
    requested are never touched, whatever `new` holds.

    Only the markets the call asked for are judged: `markets_requested`
    when given, else the markets_requested parse_event_odds recorded in
    new.attrs (a market a book answered anyway is judged too). A requested
    book's lines in any OTHER market stay exactly as an unrequested book's
    do: a core-markets refresh after a full pull (design 3.3's core mode)
    neither drops the markets it did not ask for nor counts them against the
    book. With neither known, every market is judged. Per requested book,
    over the judged markets:
      - no kept row (no entry, or an entry with nothing readable): absent,
        and its previous rows stay with their own fetched_at; a present but
        unreadable book's parser counters go in report["unreadable"];
      - kept rows under half its previous rows: the previous rows stay and
        report["suspect"][book] = (new, prev) (a book never loses half a game
        between two pulls; a feed change does);
      - otherwise its new rows replace the old, and the previous lines now
        missing are "pulled" (report["pulled"]: 8 at most, as (book, name,
        stat); n_pulled counts them all).
    The report also says "empty" (no bookmaker entry at all: the free answer),
    bad_payload (`new` is not a usable answer: wrong shape, unknown teams, not
    a feed frame), kept_rows (0 on a billed answer is a billed failure, design
    3.5), changed (live bumps the generation only then), bad_prev (`prev` was
    not a feed frame, so it counted as none) and each book's verdict. Nothing
    here raises."""
    report = {"empty": False, "bad_payload": False, "kept_rows": 0, "changed": False,
              "books": {}, "unreadable": {}, "suspect": {}, "pulled": [], "n_pulled": 0,
              "bad_prev": False}
    old_all = _feed_or_none(prev)
    if old_all is None:
        report["bad_prev"] = prev is not None
        old_all = empty_feed()
    fresh_all = _feed_or_none(new)
    unusable = fresh_all is None              # not a feed frame at all: nothing in it counts
    attrs = {} if unusable else new.attrs
    if unusable:
        fresh_all = empty_feed()
    report["bad_payload"] = (unusable or bool(attrs.get("bad_payload"))
                             or bool(attrs.get("unknown_team")))
    report["empty"] = (not report["bad_payload"] and fresh_all.empty
                       and attrs.get("bookmaker_entries") == 0)
    if isinstance(books_requested, str):
        books_requested = (books_requested,)
    requested = tuple(dict.fromkeys(b for b in (books_requested or ()) if isinstance(b, str)))
    asked = _asked_markets(markets_requested, attrs)
    answered = set(attrs.get("books_answered") or ()) | set(fresh_all["book"])
    keep = [old_all[~old_all["book"].isin(requested)]]
    for book in requested:
        old = old_all[old_all["book"] == book]
        fresh = fresh_all[fresh_all["book"] == book]
        if asked is not None:
            judged = old["market"].isin(asked | set(fresh["market"]))
            keep.append(old[~judged])            # markets this call never asked for: untouched
            old = old[judged]
        report["kept_rows"] += len(fresh)
        if fresh.empty:
            verdict = "unreadable" if book in answered else "absent"
            if verdict == "unreadable":
                report["unreadable"][book] = dict((attrs.get("by_book") or {}).get(book, {}))
            keep.append(old)
        elif len(old) and 2 * len(fresh) < len(old):
            verdict = "suspect"
            report["suspect"][book] = (len(fresh), len(old))
            keep.append(old)
        else:
            verdict = "replaced"
            now_there = {_line_key(r) for _, r in fresh.iterrows()}
            for _, r in old.iterrows():
                if _line_key(r) not in now_there:
                    report["n_pulled"] += 1
                    if len(report["pulled"]) < _SAMPLE:
                        report["pulled"].append((book, r["name"], r["stat_type"]))
            keep.append(fresh)
        report["books"][book] = verdict
    parts = [p for p in keep if len(p)]
    out = _typed(pd.concat(parts, ignore_index=True)) if parts else empty_feed()
    out = out.sort_values(["book", "market", "name"], kind="mergesort").reset_index(drop=True)
    out.attrs = {}
    report["changed"] = not _same_rows(old_all, out)
    return out, report
