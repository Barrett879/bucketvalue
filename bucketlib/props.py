"""PrizePicks lines for BucketValue: paste, parse, match, route, compare, grade.

HOW LINES GET IN, AND WHY THERE IS NO SCRAPER
---------------------------------------------
A visitor pastes them. That is the whole ingestion path, and it is deliberate:

  1. api.prizepicks.com returns HTTP 403 to server-side requests behind
     Cloudflare bot management (verified on both siblings), and rejects VPN
     and datacenter IPs at the application level, which is also why a bot on
     GitHub Actions or Streamlit Cloud could never reach it.
  2. Independently, PrizePicks' Terms section 16(l) prohibit "any robot,
     spider, or other automatic device, process, or means to access the Site
     or App". Solving (1) would not solve (2).

So the visitor opens the feed (FEED_URL) or the board in their own browser and
pastes what they see. This module parses it. **Do not add a fetcher here.**
tests/test_props.py parses this module's imports and fails if it, or anything
it imports from bucketlib, imports a network library.

THINGS THAT ARE EASY TO GET WRONG (each was a shipped bug on a sibling)
-----------------------------------------------------------------------
1. Never infer pick direction from `odds_type`. Demons and Goblins were
   More-only until 2026-08-21, when Less picks were announced for MLB and WNBA
   (NFL followed 2026-09-04). No first-party page names the NBA yet. Direction
   is read PER ROW: `allowed_wager_types` in the feed, the Less/More buttons on
   the board, and only then a permissive default.
2. Route each line to the ET date of ITS OWN game, never the date on screen.
   An evening board is tomorrow's games, and a feed start time of 00:30Z is
   the previous evening in New York.
3. Collapse the alt-line ladder to one line per (game date, player, stat),
   standard preferred, newest paste winning.
4. Reject non-person names. A feed pasted without its included[] player list
   puts a TEAM code in `description`.
5. Never let a line leave without a counter. pasted = matched + unmatched +
   ambiguous + refused + unprojected, and matched = shown + hidden + thin +
   void. A board that quietly shrinks reads as broken.
6. The paste box is unauthenticated on one shared public process. Parsers are
   linear time (MEASURED: the regex `(Demon|Goblin)+$` and a slicing loop were
   both quadratic on GridironValue), catch errors per row, reject non-finite
   and absurd numbers, cap sizes, and `parse_any` never raises.

WHAT THIS REFUSES TO PRICE (counted, with a reason, never dropped)
-------------------------------------------------------------------
Period props (1H, 2H, 1Q..4Q, "in First N Minutes"): the model projects full
games. Dunks: not in the box score the model trains on. Double and triple
doubles: joint thresholds over several stats. Two-player combo squares
("A + B"): joint lines. Anything else unknown is counted as UNMAPPED with its
label listed, so a new PrizePicks wording shows up as a visible gap rather than
a silent one.

THE GAP IS A DIFFERENCE, NEVER AN EDGE
--------------------------------------
DiamondValue measured that the posted line is a near-sufficient statistic for
its projection. The board ranks by measured reliability and then by how far
the probability sits from a coin flip, shows the median-minus-line gap, and
never a percentage column or a dollar value.

THE RECORD IS LOCAL ONLY (Barrett, 2026-10-08)
----------------------------------------------
freeze_projections / load_frozen / grade / season_record write and read
`cache/props_record_*.json` (gitignored), refuse under cache.READ_ONLY, and are
never wired into a public page. The board is a view; the record is evidence.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from . import cache as _cache
from . import teams
from .cache import DATA_DIR, logger
from .names import NameIndex
from .names import normalize as normalize_name
from .pmf import Pmf
from .tables import STATE_TYPES
from .util import UTC, now_utc, season_end_year, season_for_date, to_et

VERSION = "v1"

# ── The feed link. A LINK the visitor opens, never a request this code makes ──
# TODO(first real NBA paste): 7 is the NBA league id by REPUTATION only; nobody
# has observed it from this project. DiamondValue's MLB id (2) turned out right
# but was doubted once because the board was empty mid-game, so verify against
# a pre-game paste before trusting it. `game_mode=pickem` is REQUIRED (a bare
# request returns an application error even without a VPN); `single_stat` is
# left out because its effect on combo stats is unverified and combos are
# projected. per_page 1000 because an NBA slate with alt ladders runs past 250.
NBA_LEAGUE_ID = 7
FEED_URL = ("https://api.prizepicks.com/projections?"
            f"league_id={NBA_LEAGUE_ID}&per_page=1000&game_mode=pickem")

# ── Paste-box limits. Each bounds work an anonymous visitor can cause ────────
# MEASURED (audit 2026-10-09): at a 20 MB cap one paste of short lines took
# peak memory from 116 to 536 MB, because every line became a Python string at
# once. The board parser now reads lines lazily and stops at MAX_TEXT_LINES,
# and JSON is refused BEFORE it is built when it holds more objects and arrays
# than any feed could, since `[{},{},...]` costs ~64 bytes per two chars.
# RESIZED 2026-11-04, on the first real NBA paste (Barrett's, refused as too
# large at the old 8M cap). The old cap assumed a feed of ~1-4 MB from the
# constructed fixture's ~300-byte projections; a real one carries ~25
# attributes and 8 relationships, ~1 KB. A full slate of 6,000 lines is then
# 6.4M characters compact and 11.5M as a browser pretty-prints it. MEASURED at
# the new caps: 6,000 realistic lines parse in 0.4 s at a 38 MB peak; a hostile
# 32M-character paste of short lines takes 1.3 s and 33 MB; a hostile flat JSON
# array is refused by MAX_JSON_COMMAS before json builds it (unbounded at 32M
# it took 7.3 s and 143 MB). A realistic line has ~43 commas and ~18 objects.
MAX_PASTE_CHARS = 22_000_000      # under the websocket cap (.streamlit/config.toml)
MAX_JSON_CONTAINERS = 400_000     # ~22,000 realistic lines
MAX_JSON_COMMAS = 2_500_000       # ~58,000 realistic lines; a flat array stays small
MAX_TEXT_LINES = 150_000          # a square is 5-12 lines; 10,000 squares fit
MAX_ROWS_PER_PASTE = 30_000       # a full slate with every alt line is ~6,000-10,000
MAX_LINE_VALUE = 1000.0           # Fantasy and PRA lines are under 100
MAX_NAME_LEN = 80
MAX_LABEL_LEN = 80
MAX_TEXT_LINE_LEN = 300           # board-text lines longer than this are not squares

# ── Leans ────────────────────────────────────────────────────────────────────
# A side's probability with ties excluded inside this distance of a coin flip
# is "Even": no side is leaned and the row is never pickable.
EVEN_BAND = 0.01
# Fewer prior games than this in the projection row = too thin to rank. Ranking
# by anything is an argmax over the model's own error, and that error is
# largest where it has seen least (GridironValue: 2.97 projected carries vs
# 0.08 actual for players with <= 2 prior games). Provisional until the gate's
# cold-start stratum measures the NBA value.
MIN_GAMES = 5
# Break-even per leg from PrizePicks power-play payouts (3x on 2 legs .. 37.5x
# on 6): (1/payout)^(1/legs).
BREAKEVEN_LO, BREAKEVEN_HI = 54.7, 58.5

# ── Stats ────────────────────────────────────────────────────────────────────
FANTASY_WEIGHTS = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0,
                   "tov": -1.0}

# How each stat is composed from box-score counts (the participation table's
# lowercase columns). Used for actuals, for the shipped mean of a combo, and
# for which components a combo's reliability tier inherits from.
STAT_FORMULA: dict[str, dict[str, float]] = {
    "pts": {"pts": 1.0}, "reb": {"reb": 1.0}, "ast": {"ast": 1.0},
    "fg3m": {"fg3m": 1.0}, "blk": {"blk": 1.0}, "stl": {"stl": 1.0},
    "tov": {"tov": 1.0},
    "pr": {"pts": 1.0, "reb": 1.0}, "pa": {"pts": 1.0, "ast": 1.0},
    "ra": {"reb": 1.0, "ast": 1.0}, "pra": {"pts": 1.0, "reb": 1.0, "ast": 1.0},
    "stocks": {"blk": 1.0, "stl": 1.0}, "fantasy": dict(FANTASY_WEIGHTS),
    "ftm": {"ftm": 1.0}, "fta": {"fta": 1.0},
    "fg2m": {"fgm": 1.0, "fg3m": -1.0}, "fg2a": {"fga": 1.0, "fg3a": -1.0},
    "fgm": {"fgm": 1.0}, "fga": {"fga": 1.0}, "fg3a": {"fg3a": 1.0},
    "oreb": {"oreb": 1.0}, "dreb": {"dreb": 1.0}, "pf": {"pf": 1.0},
}
COMBO_STATS = frozenset({"pr", "pa", "ra", "pra", "stocks", "fantasy"})
# Grid step of each stat's outcome. Fantasy lands on tenths.
STAT_STEP = {s: (0.1 if s == "fantasy" else 1.0) for s in STAT_FORMULA}

# ── Fantasy Score: the formula on the numbers shown (Barrett, 2026-11-05) ────
# The pages print each stat's mid-p median to a tenth (util.fmt_num), and
# Fantasy Score is PrizePicks' formula on THOSE numbers, so it always adds up
# from the row a visitor reads. A median of a sum is not the sum of the
# medians (steals and blocks sit low), so it runs about 1-2 points under the
# simulated fantasy median for a starter; a Fantasy Score line is judged
# against it all the same (fantasy_on_shown), and the Game table shows it
# (store.team_view). One definition for both, so they cannot disagree.
SHOWN_PLACES = 1


def shown_value(value) -> float | None:
    """A number as the pages print it (util.fmt_num at SHOWN_PLACES), or None
    when there is none."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return float(f"{v:.{SHOWN_PLACES}f}") if math.isfinite(v) else None


def fantasy_shown(medians) -> float | None:
    """PrizePicks Fantasy Score (FANTASY_WEIGHTS) on the SHOWN medians of its
    six parts, `medians` mapping each part to its median. None when any part
    has no number: never zero-filled, since a missing part is not a zero."""
    total = 0.0
    for stat, w in FANTASY_WEIGHTS.items():
        v = shown_value(medians.get(stat)) if medians is not None else None
        if v is None:
            return None
        total += w * v
    # Every term is a whole number of hundredths: rounding drops float noise,
    # so 39.25 stays 39.25 and a line at it reads as exactly on it.
    return round(total, 2)


def fantasy_on_shown(fantasy_pmf: Pmf | None, medians) -> tuple[float, Pmf] | None:
    """(the shown Fantasy Score, the fantasy distribution slid so its mid-p
    median sits on it, shape kept), or None when either is missing. The
    board prices a Fantasy Score line on this pmf, so its lean agrees with
    the shown number against the line (to within half a 0.1 step)."""
    f = fantasy_shown(medians)
    if f is None or fantasy_pmf is None:
        return None
    return f, fantasy_pmf.shifted(f - fantasy_pmf.median())

STAT_LABEL = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-PT Made",
    "blk": "Blocks", "stl": "Steals", "tov": "Turnovers", "pr": "Pts+Rebs",
    "pa": "Pts+Asts", "ra": "Rebs+Asts", "pra": "Pts+Rebs+Asts",
    "stocks": "Blks+Stls", "fantasy": "Fantasy Score", "ftm": "FT Made",
    "fta": "FT Attempted", "fg2m": "2-PT Made", "fg2a": "2-PT Attempted",
    "fgm": "FG Made", "fga": "FG Attempted", "fg3a": "3-PT Attempted",
    "oreb": "Off Rebounds", "dreb": "Def Rebounds", "pf": "Fouls",
}

# Stat-family pills on the board.
STAT_FAMILY = {
    "pts": "Points", "reb": "Rebounds", "oreb": "Rebounds", "dreb": "Rebounds",
    "ast": "Assists", "fg3m": "Threes", "fg3a": "Threes",
    "pr": "Combos", "pa": "Combos", "ra": "Combos", "pra": "Combos",
    "stl": "Defense", "blk": "Defense", "stocks": "Defense",
    "fantasy": "Fantasy", "ftm": "Shooting", "fta": "Shooting",
    "fgm": "Shooting", "fga": "Shooting", "fg2m": "Shooting", "fg2a": "Shooting",
    "tov": "Other", "pf": "Other",
}
FAMILY_ORDER = ["Points", "Rebounds", "Assists", "Threes", "Combos", "Defense",
                "Fantasy", "Shooting", "Other"]


def label_key(label) -> str:
    """Case-, spacing-, hyphen- and plus-tolerant key for a stat label.
    '3-PT Made', '3PT Made' and '3 pt made' all become '3ptmade';
    'Pts + Rebs + Asts' becomes 'pts+rebs+asts'."""
    s = str(label or "")[:MAX_LABEL_LEN].lower().replace("＋", "+")
    return re.sub(r"[\s\-_.]+", "", s)


_STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "pts": ("points", "pts"),
    "reb": ("rebounds", "rebs", "reb", "totalrebounds"),
    "ast": ("assists", "asts", "ast"),
    "fg3m": ("3ptmade", "3pointersmade", "3pointermade", "threepointersmade",
             "3pm", "3ptsmade", "3pointsmade"),
    "blk": ("blockedshots", "blocks", "blks", "blk"),
    "stl": ("steals", "stls", "stl"),
    "tov": ("turnovers", "tov", "to"),
    "pr": ("pts+rebs", "points+rebounds", "rebs+pts"),
    "pa": ("pts+asts", "points+assists", "asts+pts"),
    "ra": ("rebs+asts", "rebounds+assists", "asts+rebs"),
    # 'Pts+Rebds+Asts' is the spelling in a first-party PrizePicks article.
    "pra": ("pts+rebs+asts", "pts+rebds+asts", "points+rebounds+assists", "pra"),
    "stocks": ("blks+stls", "blocks+steals", "stls+blks", "steals+blocks"),
    "fantasy": ("fantasyscore", "fantasypoints", "fantasy"),
    "ftm": ("freethrowsmade", "ftmade", "ftm", "freethrowmade"),
    "fta": ("ftattempted", "freethrowsattempted", "ftattempts", "fta",
            "freethrowattempts"),
    "fg2m": ("twopointersmade", "2ptmade", "2pointersmade", "twopointermade"),
    "fg2a": ("twopointersattempted", "2ptattempted", "2pointersattempted",
             "twopointerattempts"),
    "fgm": ("fgmade", "fieldgoalsmade", "fgm"),
    "fga": ("fgattempted", "fieldgoalsattempted", "fga", "fgattempts"),
    "fg3a": ("3ptattempted", "3pointersattempted", "3ptattempts",
             "threepointersattempted"),
    "oreb": ("offensiverebounds", "offrebounds", "oreb"),
    "dreb": ("defensiverebounds", "defrebounds", "dreb"),
    "pf": ("personalfouls", "fouls", "pf"),
}
STAT_MAP: dict[str, str] = {alias: stat for stat, aliases in _STAT_ALIASES.items()
                            for alias in aliases}

# Refusals, checked on the lower-cased label with its spacing intact. Labels
# are capped at MAX_LABEL_LEN before any of these run, so even the alternations
# with \s* cannot be driven into long scans.
_PERIOD = re.compile(
    r"\b(?:[1-4]\s*(?:st|nd|rd|th)?\s*(?:h|q|half|qtr|quarter)"
    r"|(?:first|1st)\s+\d+\s+min\w*|in\s+(?:the\s+)?first|first\s+(?:half|quarter)"
    r"|second\s+(?:half|quarter)|third\s+quarter|fourth\s+quarter)\b")
_REFUSALS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdunks?\b"),
     "not in the box score the model is trained on"),
    (re.compile(r"\b(?:double|triple)[\s\-]*doubles?\b"),
     "a joint threshold across several stats; v1 prices single counts and sums"),
    (re.compile(r"\bfirst\s+(?:basket|field\s+goal|fg|to\s+score|points?)\b"),
     "an ordering within the game, not a per-game count"),
]
REFUSE_PERIOD = "a period prop; BucketValue projects full games only"
REFUSE_COMBO = "a combined line for two players; v1 does not price joint lines"
REFUSE_LEAGUE = "from a period, live or non-NBA board ({}); full-game NBA lines only"
REFUSE_PERIOD_BOARD = ("sits far below the full-game projections, as the lines on a quarter or "
                       "half board do; BucketValue projects full games only")


def is_league_refusal(why) -> bool:
    """True for a REFUSE_LEAGUE reason (the projection came from another
    board: NBA1Q, a live board, NHL...). One test, used by compare() and by
    the paste report, so the two never disagree about which rows those are."""
    return isinstance(why, str) and why.startswith(REFUSE_LEAGUE.split("{")[0])


def classify_stat(stat_type, name=None, is_combo=False, league=None
                  ) -> tuple[str | None, str, str | None]:
    """(stat key, kind, reason) with kind 'ok', 'refused' or 'unmapped'."""
    if is_combo or "+" in str(name or ""):
        return None, "refused", REFUSE_COMBO
    lg = str(league or "").strip().upper()
    if lg and lg != "NBA":
        return None, "refused", REFUSE_LEAGUE.format(lg[:20])
    raw = str(stat_type or "")[:MAX_LABEL_LEN].lower()
    spaced = " ".join(raw.split())
    if _PERIOD.search(spaced):
        return None, "refused", REFUSE_PERIOD
    hit = STAT_MAP.get(label_key(raw))
    if hit:
        return hit, "ok", None
    for pat, why in _REFUSALS:
        if pat.search(spaced):
            return None, "refused", why
    return None, "unmapped", "stat not mapped to a projected quantity"


# ── Names that are not people ────────────────────────────────────────────────
_GLUE = {"vs", "vs.", "@", "at", "v", "/", "-"}
# Characters no player name carries but JSON, markup and key=value text do. A
# JSON line like `"name": "Nikola Jokic"` is otherwise "person-like".
_NOT_IN_NAMES = set('"{}[]=:;|\\')


def _person_like(name) -> bool:
    """Reject team codes, matchups and other non-person strings.

    A feed pasted without included[] puts a team code ("LAL") in description.
    Tolerates initials ('P.J. Washington', 'OG Anunoby') and diacritics, which
    the sibling sites' `[A-Za-z]{2,}\\s+[A-Za-z]` test rejected.
    """
    s = str(name or "").strip()
    if len(s) < 4 or len(s) > MAX_NAME_LEN:
        return False
    if any(ch.isdigit() or ch in _NOT_IN_NAMES for ch in s):
        return False
    toks = s.split()
    if len(toks) < 2:
        return False
    if sum(ch.isalpha() for ch in s) < 4:
        return False
    if all(t.lower() in _GLUE or (t.isupper() and len(t) <= 4) for t in toks):
        return False
    return any(sum(ch.isalpha() for ch in t) >= 2 for t in toks)


# ── Which sides a line offers ────────────────────────────────────────────────
# The last-resort default when neither the feed nor the board says. Kept
# permissive on purpose: hiding a real pick is worse than showing one the market
# may not offer. One place, so changing the rule is a one-line edit, and it
# reaches BOTH paste paths: a board square with no buttons and a feed row with
# no allowed_wager_types are each "unknown" ('') and fall through to here
# (audit 2026-10-09: the feed path used to read an absent field as "both", so
# this table was dead code for every feed row).
ODDS_TYPE_SIDES = {"standard": "both", "demon": "both", "goblin": "both"}
_WAGER_TYPE_SIDES = {"over": "more", "under": "less", "more": "more", "less": "less"}


def sides_from_allowed_wager_types(value) -> str:
    """The feed's `allowed_wager_types` as a direction: 'both', 'more',
    'less', or '' for UNKNOWN.

    Absent (None) is unknown, not unrestricted: the row says nothing about its
    sides, so offered_sides() falls back to ODDS_TYPE_SIDES for its odds type,
    exactly as for a board square that shows no Less/More buttons. A field
    that IS present decides: [] and [over, under] are both sides. An
    unrecognised value is logged and treated as two-sided rather than silently
    hiding the line."""
    if value is None:
        return ""
    items = value if isinstance(value, (list, tuple, set)) else [value]
    sides = set()
    for item in list(items)[:10]:
        mapped = _WAGER_TYPE_SIDES.get(str(item).strip().lower()[:20])
        if not mapped:
            logger.warning("unrecognised allowed_wager_types value %r; treating "
                           "the line as two-sided", str(item)[:40])
            return "both"
        sides.add(mapped)
    if not sides or sides == {"more", "less"}:
        return "both"
    return sides.pop()


def offered_sides(direction, odds_type=None) -> str:
    """'both', 'more' or 'less'. An explicit direction (board buttons or the
    feed's allowed_wager_types) always wins; ODDS_TYPE_SIDES is the fallback."""
    d = str(direction or "").strip().lower()
    if d in ("both", "more", "less"):
        return d
    return ODDS_TYPE_SIDES.get(str(odds_type or "").strip().lower(), "both")


def is_pickable(lean, sides) -> bool:
    """Can the lean be played? 'Even' never can: there is no side to take."""
    lean = str(lean or "").strip().lower()
    sides = str(sides or "both").strip().lower()
    if lean not in ("more", "less"):
        return False
    return sides == "both" or sides == lean


# ── Parsing ──────────────────────────────────────────────────────────────────
PARSED_COLUMNS = ["name", "team", "position", "stat_type", "line", "start_time",
                  "direction", "odds_type", "opp", "is_combo", "league", "source"]
_ODDS_TYPES = {"standard", "demon", "goblin"}
# attrs['json_shape'] for a feed whose data[] is an empty list: the projections
# feed itself, with no lines on it right now. Any other json_shape means the
# JSON is not the feed.
EMPTY_FEED = "empty feed"


def _new_attrs() -> dict:
    return {"skipped_noname": 0, "skipped_badline": 0, "skipped_live": 0,
            "skipped_error": 0, "truncated": 0, "too_large": False,
            "json_shape": None, "allowed_wager_types": {}}


def _frame(rows: list[dict], attrs: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=PARSED_COLUMNS)
    df.attrs.update(attrs)
    return df


def _text(value, maxlen: int) -> str | None:
    """A short display string from untrusted input, or None. Only strings and
    plain numbers are accepted; a dict or list in a name field is junk."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or len(s) > maxlen:
        return None
    return s


def _parse_line(value) -> float | None:
    """A posted line, or None. Rejects booleans (float(True) is 1.0), NaN and
    infinities (Python's json accepts the bare NaN and Infinity literals, and
    float() accepts '1e400'), negatives and absurd magnitudes. Each of those
    reached the comparison on a sibling and raised out of the page."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value.strip()[:40]) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(x) or x < 0 or x > MAX_LINE_VALUE:
        return None
    return x


def parse_prizepicks_json(raw) -> pd.DataFrame:
    """Parse a pasted api.prizepicks.com/projections payload.

    JSON:API shape: data[] projections plus included[] player (and league)
    records. Every row is validated on its own and a bad row costs that row
    only: on GridironValue one non-numeric line_score discarded a 59/60-valid
    feed. Counts of what was skipped and why ride on df.attrs.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        raw = json.loads(raw)
    attrs = _new_attrs()
    if not isinstance(raw, dict):
        attrs["json_shape"] = "not a JSON object"
        return _frame([], attrs)
    raw_data = raw.get("data")
    included = raw.get("included")
    # A PASTE BOX. `{"data": ["x"]}` is valid JSON whose data[] holds a string;
    # calling .get on it raised AttributeError out of the page on GridironValue.
    data = [d for d in raw_data if isinstance(d, dict)] if isinstance(raw_data, list) else []
    included = ([i for i in included if isinstance(i, dict)]
                if isinstance(included, list) else [])
    if isinstance(raw_data, list) and not raw_data and "errors" not in raw:
        # The RIGHT feed with nothing on it: before the slate's lines post, and
        # once every game has tipped. Telling the visitor "that is not the
        # feed" here made them doubt the correct link (audit 2026-10-09), the
        # trap NBA_LEAGUE_ID's note recounts from DiamondValue.
        attrs["json_shape"] = EMPTY_FEED
    elif not any(d.get("type") == "projection" for d in data):
        attrs["json_shape"] = "no projections in data[]"

    players: dict[str, dict] = {}
    leagues: dict[str, str] = {}
    for inc in included:
        try:
            a = inc.get("attributes")
            a = a if isinstance(a, dict) else {}
            iid = str(inc.get("id"))[:40]
            typ = inc.get("type")
            if typ in ("new_player", "player"):
                players[iid] = {
                    "name": _text(a.get("name") or a.get("display_name"), MAX_NAME_LEN),
                    "team": _text(a.get("team") or a.get("team_name"), 30),
                    "position": _text(a.get("position"), 10),
                    "combo": a.get("combo") is True,
                    "league": _text(a.get("league"), 20),
                }
            elif typ == "league":
                leagues[iid] = _text(a.get("name"), 20) or ""
        except Exception:  # noqa: BLE001 - one bad record costs one record
            attrs["skipped_error"] += 1

    rows: list[dict] = []
    for p in data:
        if len(rows) >= MAX_ROWS_PER_PASTE:
            attrs["truncated"] += 1
            continue
        try:
            if p.get("type") != "projection":
                continue
            a = p.get("attributes")
            a = a if isinstance(a, dict) else {}
            rel = p.get("relationships")
            rel = rel if isinstance(rel, dict) else {}
            np_rel = rel.get("new_player")
            np_rel = np_rel if isinstance(np_rel, dict) else {}
            np_data = np_rel.get("data")
            pid = str(np_data.get("id"))[:40] if isinstance(np_data, dict) else ""
            player = players.get(pid, {})
            name = player.get("name")
            desc = _text(a.get("description"), MAX_NAME_LEN)
            if not name and _person_like(desc):
                name = desc
            if not name or not _person_like(name.replace("+", " ")):
                attrs["skipped_noname"] += 1
                continue
            if (a.get("is_live") is True or a.get("in_game") is True
                    or str(a.get("status") or "").lower() in ("in_progress", "live", "in_game")):
                attrs["skipped_live"] += 1
                continue
            line = _parse_line(a.get("line_score"))
            if line is None:
                attrs["skipped_badline"] += 1
                continue
            lg_rel = rel.get("league")
            lg_data = lg_rel.get("data") if isinstance(lg_rel, dict) else None
            lg_id = str(lg_data.get("id"))[:40] if isinstance(lg_data, dict) else ""
            # The projection's own league outranks the player record: a 1Q board
            # reuses the same player.
            league = leagues.get(lg_id) or player.get("league") or None
            awt = a.get("allowed_wager_types")
            direction = sides_from_allowed_wager_types(awt)
            key = str(awt)[:60]
            attrs["allowed_wager_types"][key] = attrs["allowed_wager_types"].get(key, 0) + 1
            odds = (_text(a.get("odds_type"), 20) or "standard").lower()
            rows.append({
                "name": name,
                "team": player.get("team"),
                "position": player.get("position"),
                "stat_type": (_text(a.get("stat_type"), MAX_LABEL_LEN)
                              or _text(a.get("stat_display_name"), MAX_LABEL_LEN) or ""),
                "line": line,
                "start_time": _text(a.get("start_time"), 40),
                "direction": direction,
                "odds_type": odds if odds in _ODDS_TYPES else odds[:20],
                "opp": teams.canonical(desc) if desc and len(desc) <= 5 else None,
                "is_combo": bool(player.get("combo")) or "+" in name,
                "league": league,
                "source": "feed",
            })
        except Exception:  # noqa: BLE001 - per row, see docstring
            attrs["skipped_error"] += 1
    if attrs["allowed_wager_types"]:
        logger.info("allowed_wager_types distribution: %s", attrs["allowed_wager_types"])
    return _frame(rows, attrs)


_BOARD_NUM = re.compile(r"^\d{1,6}(?:\.\d{1,3})?$")
# NBA positions include hyphenated pairs: G-F, F-G, F-C, C-F.
_BOARD_TEAMPOS = re.compile(
    r"^([A-Z]{2,4})\s{0,3}-\s{0,3}([A-Za-z0-9]{1,3}(?:-[A-Za-z0-9]{1,3})?)$")
_LIVE_CUR = re.compile(r"^cur\.?\s{0,3}\d{1,4}(?:\.\d{1,3})?$", re.IGNORECASE)
_LIVE_SCORE = re.compile(
    r"^[A-Z]{2,4}\s{0,3}\d{1,3}\s{0,3}(?:vs\.?|@)\s{0,3}[A-Z]{2,4}\s{0,3}\d{1,3}$")
_MATCHUP = re.compile(r"^(?:vs\.?|@)\s{0,3}([A-Z]{2,4})\b")
# A LOOP, not a regex. `(Demon|Goblin)+$` is ambiguously repeatable, so the
# engine restarts from every offset: GridironValue measured 0.076s for 9.8 KB,
# 0.296s for 19.5 KB, 1.177s for 39.1 KB through parse_any, a clean 4x per
# doubling, about eight minutes for an 800 KB paste. `(Demon|Goblin)*$` has the
# same ambiguous repeat and is not a fix.
_BOARD_SUFFIXES = ("Demon", "Goblin")


def _strip_board_suffix(text: str) -> tuple[str, str]:
    """(text without trailing Demon/Goblin markers, odds type they imply).

    Walks an index backwards and copies once. The first replacement for the
    quadratic regex on GridironValue was `while s.endswith(..): s = s[:-5]`,
    which is ALSO quadratic because slicing copies (0.028s / 0.105s / 0.296s
    for 100 / 200 / 400 KB). The marker NEAREST the text wins the odds type.
    """
    out = str(text)
    end = len(out)
    odds = "standard"
    while True:
        for suffix in _BOARD_SUFFIXES:
            n = len(suffix)
            if end >= n and out[end - n:end] == suffix:
                end -= n
                odds = suffix.lower()
                break
        else:
            return out[:end], odds


def _is_stat_label(text: str) -> bool:
    kind = classify_stat(text)[1]
    return kind != "unmapped"


# Every separator str.splitlines() honours. A run of anything else is a line.
_TEXT_LINE = re.compile(r"[^\r\n\x0b\x0c\x1c\x1d\x1e\x85  ]+")


def _board_lines(text: str, attrs: dict) -> list[str]:
    """The paste's non-empty, square-sized lines, read LAZILY and capped.

    `text.splitlines()` built every line of a 20 MB paste as a string at once
    (the audit's 116 -> 536 MB). This walks the text, never materialises a
    line too long to be part of a square, and stops at MAX_TEXT_LINES."""
    out: list[str] = []
    for m in _TEXT_LINE.finditer(text):
        if m.end() - m.start() > 4 * MAX_TEXT_LINE_LEN:
            continue
        ln = m.group().strip()
        if not ln or len(ln) > MAX_TEXT_LINE_LEN:
            continue
        if len(out) >= MAX_TEXT_LINES:
            attrs["truncated"] += 1
            break
        out.append(ln)
    return out


def parse_prizepicks_board(text: str) -> pd.DataFrame:
    """Parse text copied from the PrizePicks board page.

    The real copy (DiamondValue's MLB paste, the only real sample on any
    sibling) looks like this, adapted to the NBA:

        Trending
        12.4K
        LeBron JamesDemon      <- name, payout tag glued on
        LAL - F                <- TEAM - POS, OPTIONAL
        LeBron James           <- the clean name again
        vs GSW Tue 10:00pm     <- matchup (or a live score line + "Cur. 18")
        27.5
        Points
        More                   <- only the buttons the line actually offers

    A state machine, linear in the paste: a person-like line sets the current
    name (and odds from a glued tag), a TEAM - POS line sets the team, a
    matchup line the opponent, a score or "Cur." line marks the square live; a
    bare number followed by a label emits one row and resets the state, so a
    stray number can never be pinned on the previous player. A number right
    after "Trending" is the trending count, not a line. Live squares are
    counted in attrs['skipped_live'] and not emitted.

    SQUARE STATE NEVER OUTLIVES ITS SQUARE. On a real square the name comes
    FIRST and its team, matchup and live lines follow it, so a new player
    starts from a clean slate, a nameless square (a team total, whose TEAM -
    POS reads "TM") takes its state with it when its number goes by, and a "TM"
    line ends any player square left open before it. Audit 2026-10-09: a DEN
    team total followed by Stephen Curry's square (no TEAM - POS line) filed
    Curry under DEN@DAL as an exact match, and a live team total made the next
    pre-game square read as live.
    """
    attrs = _new_attrs()
    rows: list[dict] = []
    lines = _board_lines(str(text), attrs)
    n = len(lines)
    name = team = pos = opp = None
    odds = "standard"
    live = False
    prev = ""
    skip_label = False
    i = 0
    while i < n:
        ln = lines[i]
        if len(rows) >= MAX_ROWS_PER_PASTE:
            attrs["truncated"] += 1
            break
        if skip_label:
            # The label of a square with no player (a team total): never a name.
            skip_label = False
            if ln not in ("Less", "More") and not _BOARD_NUM.match(ln):
                prev = ln
                i += 1
                continue
        m_tp = _BOARD_TEAMPOS.match(ln)
        if m_tp:
            if m_tp.group(2).upper() == "TM" and name is not None:
                # A team square: the player square left open before it was
                # abandoned, and none of its state belongs to this one.
                name = None
                opp = None
                odds, live = "standard", False
            team, pos = m_tp.group(1), m_tp.group(2)
        elif _LIVE_CUR.match(ln) or _LIVE_SCORE.match(ln) or ln.lower() in ("live", "in progress"):
            live = True
        elif _MATCHUP.match(ln):
            opp = teams.canonical(_MATCHUP.match(ln).group(1))
        elif _BOARD_NUM.match(ln):
            if prev.lower() != "trending" and name is not None and i + 1 < n:
                label_raw = lines[i + 1]
                if (label_raw not in ("Less", "More") and not _BOARD_NUM.match(label_raw)
                        and not _BOARD_TEAMPOS.match(label_raw)):
                    label, tag = _strip_board_suffix(label_raw)
                    label = label.strip()
                    j = i + 2
                    btns = set()
                    while j < n and lines[j] in ("Less", "More"):
                        btns.add(lines[j])
                        j += 1
                    if live:
                        attrs["skipped_live"] += 1
                    else:
                        line = _parse_line(ln)
                        if line is None:
                            attrs["skipped_badline"] += 1
                        else:
                            direction = ("both" if {"Less", "More"} <= btns
                                         else "more" if "More" in btns
                                         else "less" if "Less" in btns else "")
                            rows.append({
                                "name": name, "team": team, "position": pos,
                                "stat_type": label[:MAX_LABEL_LEN], "line": line,
                                "start_time": None, "direction": direction,
                                "odds_type": tag if tag != "standard" else odds,
                                "opp": opp, "is_combo": "+" in name,
                                "league": None, "source": "board",
                            })
                    name = team = pos = opp = None
                    odds, live = "standard", False
                    prev = lines[j - 1]
                    i = j
                    continue
            if name is None:
                if prev.lower() != "trending":
                    skip_label = True
                # A nameless square (team total, trending count) ends here,
                # and its team, matchup, odds and live flag end with it.
                team = pos = opp = None
                odds, live = "standard", False
        elif ln in ("Less", "More", "Trending"):
            pass
        else:
            clean, tag = _strip_board_suffix(ln)
            clean = clean.strip()
            if not clean and tag != "standard":
                odds = tag                       # a marker on its own line
            elif _person_like(clean.replace("+", " ")) and not _is_stat_label(clean):
                if name is None:
                    # A new square starts with its name. Team, matchup and
                    # live lines come AFTER it, so any seen before belong to
                    # something else (a team square, junk). A marker on its
                    # own line just before the name is kept.
                    team = pos = opp = None
                    live = False
                elif normalize_name(clean) != normalize_name(name):
                    # A different player before the last square emitted: that
                    # square was abandoned, so none of its state carries over.
                    team = pos = opp = None
                    odds, live = "standard", False
                name = clean
                if tag != "standard":
                    odds = tag
        prev = ln
        i += 1
    return _frame(rows, attrs)


def parse_any(text) -> pd.DataFrame:
    """Feed JSON or board text, whichever the paste is. NEVER raises.

    Deliberately broad excepts: everything here is untrusted paste, the
    contract is "return what you could parse", and it runs before the page
    renders, so anything that escapes replaces the page with a traceback that
    carries server paths. Valid JSON is never re-read as board text, because
    shredding JSON through the text parser produces junk rows (DiamondValue).
    """
    attrs = _new_attrs()
    try:
        t = text.decode("utf-8", "replace") if isinstance(text, (bytes, bytearray)) else str(text or "")
    except Exception:  # noqa: BLE001
        return _frame([], attrs)
    if len(t) > MAX_PASTE_CHARS:
        attrs["too_large"] = True
        return _frame([], attrs)
    t = t.strip()
    if not t:
        return _frame([], attrs)
    if t[0] in "{[":
        if t.count("{") + t.count("[") > MAX_JSON_CONTAINERS or t.count(",") > MAX_JSON_COMMAS:
            # Counted in C before json builds a single object: a paste of
            # `[{},{},...]` or `[1,1,...]` at the size cap would otherwise
            # allocate millions.
            attrs["too_large"] = True
            return _frame([], attrs)
        try:
            # raw_decode: a feed copied with a stray line after it is still
            # the feed, and must not be shredded through the board parser.
            obj, _end = json.JSONDecoder().raw_decode(t)
        except Exception as e:  # noqa: BLE001 - not JSON (or nested to death)
            logger.info("paste is not JSON (%s); reading it as board text", type(e).__name__)
        else:
            try:
                return parse_prizepicks_json(obj)
            except Exception as e:  # noqa: BLE001
                logger.warning("feed parse failed (%s)", type(e).__name__)
                attrs["json_shape"] = "unreadable feed"
                return _frame([], attrs)
    try:
        return parse_prizepicks_board(t)
    except Exception as e:  # noqa: BLE001
        logger.warning("board-text parse failed (%s); returning nothing", type(e).__name__)
        return _frame([], attrs)


# ── Names -> PERSON_ID ───────────────────────────────────────────────────────
def build_name_index(roster: pd.DataFrame) -> NameIndex:
    """NameIndex over the CURRENT roster frame (person_id, name, team)."""
    idx = NameIndex()
    if roster is None or roster.empty:
        return idx
    for pid, name, team in roster[["person_id", "name", "team"]].itertuples(index=False):
        if pid is None or pd.isna(pid) or not name:
            continue
        idx.add(str(name), int(pid), team=teams.canonical(team))
    return idx


def _roster_lookup(roster: pd.DataFrame) -> dict[int, tuple[str, str | None]]:
    if roster is None or roster.empty:
        return {}
    return {int(p): (str(n), teams.canonical(t))
            for p, n, t in roster[["person_id", "name", "team"]].itertuples(index=False)
            if p is not None and not pd.isna(p)}


def resolve_names(lines: pd.DataFrame, roster: pd.DataFrame,
                  index: NameIndex | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(resolved, unresolved). Never drops a row.

    SCOPED BY TEAM when the paste carries a team code PrizePicks spells in a way
    teams.canonical knows. Every non-exact match records its `how` ('alias',
    'prefix', or '+other_team' when the roster puts the player on a different
    team than the paste, e.g. a trade the roster has not caught up with).
    Ambiguous and unmatched names come back in `unresolved` with a `reason`.
    """
    if lines is None or lines.empty:
        cols = list(lines.columns) if lines is not None else []
        return (pd.DataFrame(columns=cols + ["person_id", "how", "roster_name", "roster_team"]),
                pd.DataFrame(columns=cols + ["reason", "how"]))
    idx = index if index is not None else build_name_index(roster)
    look = _roster_lookup(roster)
    res, unres = [], []
    for rec in lines.to_dict(orient="records"):
        try:
            team = teams.canonical(rec.get("team"))
            m = idx.get(str(rec.get("name") or ""), team=team)
            pid = None if m.value is None else int(m.value)
        except Exception as e:  # noqa: BLE001 - one bad row costs one row (audit)
            logger.warning("resolve_names: row failed (%s)", type(e).__name__)
            unres.append({**rec, "reason": "unmatched", "how": "error"})
            continue
        if pid is None:
            reason = "ambiguous" if m.how == "ambiguous" else "unmatched"
            unres.append({**rec, "reason": reason, "how": m.how})
            continue
        rname, rteam = look.get(pid, (None, None))
        how = m.how
        if team is not None and rteam is not None and rteam != team:
            how = f"{how}+other_team"
        res.append({**rec, "person_id": pid, "how": how,
                    "roster_name": rname, "roster_team": rteam})
    resolved = pd.DataFrame(res) if res else pd.DataFrame(
        columns=list(lines.columns) + ["person_id", "how", "roster_name", "roster_team"])
    unresolved = pd.DataFrame(unres) if unres else pd.DataFrame(
        columns=list(lines.columns) + ["reason", "how"])
    return resolved, unresolved


# ── Routing: each line to the ET date of its own game ────────────────────────
# A start time outside this window is junk, not a game. A paste box accepts
# "9999-12-31T23:59:59Z", and subtracting that from a pandas tip raised
# OutOfBoundsDatetime out of the whole paste (audit 2026-10-09).
_TS_YEARS = (2000, 2100)


def _parse_ts(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            value = value.to_pydatetime()
        ts = to_et(value)
    except Exception:  # noqa: BLE001 - naive, malformed or absurd: unrouted
        return None
    return ts if _TS_YEARS[0] <= ts.year <= _TS_YEARS[1] else None


def _utc_iso(ts) -> str | None:
    return None if ts is None else ts.astimezone(UTC).isoformat(timespec="seconds")


# The schedule column a page sets to say which games the site projects. The
# rule is pipeline.state_games (state games, both teams known, not postponed,
# of a pipeline.PROJECTED_TYPES type), which props cannot import: pipeline
# reaches the network, and this module is PASTE ONLY. So the page, which can,
# marks its schedule and props reads the mark.
PROJECTED_COL = "projected"


def _schedule_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    """game_id, home, away, tip (aware ET), date (ISO), final (bool),
    projectable (bool).

    projectable: a game the site projects at all. A line for a game it does
    not (preseason, playoffs, the Cup final, All-Star, and the play-in) is
    still routed to its own date and saved (stale beats empty), but the paste
    report must never promise a comparison that cannot appear (audit
    2026-10-09: a preseason paste said "2 lines loaded ... see Model vs the
    board" and the page promised the comparison "once projections are
    published").

    The schedule's PROJECTED_COL, when the page sets it, decides: it is the
    pipeline's own answer. Without it, a game outside tables.STATE_TYPES or
    without both teams is never projectable, which always holds (a projected
    game is a state game with both teams, by pipeline.state_games's
    construction); a play-in or postponed game is then left projectable,
    since only the pipeline's rule knows it is not. A schedule with neither
    column (constructed tests) counts every game with both teams.
    """
    cols = ["game_id", "home", "away", "tip", "date", "final", "projectable"]
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=cols)
    s = schedule
    tips = [(_parse_ts(t) if t is not None else None) for t in s.get("tip_utc", [None] * len(s))]
    if "game_date" in s.columns:
        dates = [str(pd.Timestamp(d).date()) if d is not None and not pd.isna(d) else None
                 for d in s["game_date"]]
    else:
        dates = [t.date().isoformat() if t is not None else None for t in tips]
    dates = [d or (t.date().isoformat() if t is not None else None) for d, t in zip(dates, tips)]
    home = [teams.canonical(x) for x in s["home"]]
    away = [teams.canonical(x) for x in s["away"]]
    state = (s["game_type"].isin(STATE_TYPES).to_numpy() if "game_type" in s.columns
             else np.ones(len(s), dtype=bool))
    marked = (s[PROJECTED_COL].to_numpy(dtype=object) if PROJECTED_COL in s.columns
              else np.full(len(s), None, dtype=object))
    return pd.DataFrame({
        "game_id": s["game_id"].astype(str).to_numpy(),
        "home": home, "away": away,
        "tip": pd.Series(tips, dtype=object), "date": dates,
        "final": (s["status"].astype(str).str.lower() == "final").to_numpy()
        if "status" in s.columns else np.zeros(len(s), dtype=bool),
        "projectable": [bool(m) if isinstance(m, (bool, np.bool_))
                        else bool(typ) and h is not None and a is not None
                        for m, typ, h, a in zip(marked, state, home, away)],
    }, columns=cols)


def _route_one(rec: dict, games_all: list[dict], by_team: dict, want: set,
               now_et) -> tuple[str | None, str | None, str, str | None, bool | None]:
    """(game_date, game_id, route_how, tip_utc, projectable) for one line.
    projectable is None when the line reached a date but no single game."""
    team = teams.canonical(rec.get("team")) or teams.canonical(rec.get("roster_team"))
    opp = teams.canonical(rec.get("opp"))
    ts = _parse_ts(rec.get("start_time"))
    games = by_team.get(team, []) if team else []
    if ts is not None:
        d = ts.date().isoformat()
        same = [g for g in games if g["date"] == d]
        if not same and not team:
            same = [g for g in games_all
                    if g["tip"] is not None and abs((g["tip"] - ts).total_seconds()) < 3600]
        g = same[0] if len(same) == 1 else None
        tip = g["tip"] if g is not None and g["tip"] is not None else ts
        # The feed's own start time says when the game tips. A line whose game
        # has tipped (or gone Final) is never a pre-game line: the board-text
        # path already refused these, the feed path priced them (audit).
        if ts <= now_et or tip <= now_et or (g is not None and g["final"]):
            return None, None, "tipped", None, None
        return (d, (g["game_id"] if g is not None else None), "start_time", _utc_iso(tip),
                (bool(g["projectable"]) if g is not None else None))
    cands = [g for g in games if g["date"] in want and not g["final"]
             and (g["tip"] is None or g["tip"] > now_et)
             and (opp is None or opp in (g["home"], g["away"]))]
    if cands:
        g = min(cands, key=lambda g: (g["date"], g["tip"] or now_et))
        return g["date"], g["game_id"], "team_schedule", _utc_iso(g["tip"]), bool(g["projectable"])
    return None, None, "unrouted", None, None


_ROUTE_COLUMNS = ("game_date", "game_id", "route_how", "tip_utc", "projectable")


def route_lines(lines: pd.DataFrame, schedule: pd.DataFrame, now=None,
                dates=None) -> pd.DataFrame:
    """Add game_date (ISO), game_id, route_how, tip_utc and projectable to
    every line.

    Feed rows carry start_time: the line goes to the ET date of THAT time
    (00:30Z is the previous evening in New York), and to the team's game on it,
    unless that time has passed ('tipped'). Board text carries no time, so a
    line goes to the player's team's earliest game in `dates` (default: today
    and tomorrow in ET) that has NOT tipped: an evening board is tomorrow's
    games. A line that cannot be placed stays with game_date None and
    route_how 'unrouted' / 'tipped' / 'error', reported rather than filed under
    the date on screen. tip_utc is the schedule's tip (or the feed's start
    time): the record refuses to freeze a line whose tip it does not know.
    projectable is False for a game the site does not project (preseason,
    playoffs, Cup final, and whatever else the page's PROJECTED_COL says: see
    _schedule_frame), None when no single game was found.
    """
    out = lines.copy()
    if out.empty:
        for c in _ROUTE_COLUMNS:
            out[c] = pd.Series(dtype=object)
        return out
    now = now or now_utc()
    now_et = to_et(now)
    if dates is None:
        d0 = now_et.date()
        dates = [d0, d0 + pd.Timedelta(days=1)]
    want = {str(pd.Timestamp(d).date()) for d in dates}
    games_all = _schedule_frame(schedule).to_dict(orient="records")
    by_team: dict[str, list[dict]] = {}
    for g in games_all:
        for side in ("home", "away"):
            if g[side]:
                by_team.setdefault(g[side], []).append(g)

    routed = []
    for rec in out.to_dict(orient="records"):
        try:
            routed.append(_route_one(rec, games_all, by_team, want, now_et))
        except Exception as e:  # noqa: BLE001 - one bad row costs one row (audit)
            logger.warning("route_lines: row failed (%s)", type(e).__name__)
            routed.append((None, None, "error", None, None))
    for i, c in enumerate(_ROUTE_COLUMNS):
        out[c] = pd.Series([r[i] for r in routed], index=out.index, dtype=object)
    return out


# ── The alt-line ladder ──────────────────────────────────────────────────────
_ODDS_RANK = {"standard": 0, "goblin": 1, "demon": 2}


def _collapse_key(df: pd.DataFrame) -> pd.Series:
    date = df["game_date"].astype(str) if "game_date" in df.columns else pd.Series("", index=df.index)
    if "person_id" in df.columns:
        who = df["person_id"].map(lambda p: "" if p is None or pd.isna(p) else f"p{int(p)}")
    else:
        who = pd.Series("", index=df.index)
    name = df["name"].map(lambda n: "n:" + normalize_name(n if isinstance(n, str) else "")) \
        if "name" in df.columns else pd.Series("", index=df.index)
    who = who.where(who != "", name)
    stat = df["stat_type"].map(label_key) if "stat_type" in df.columns else pd.Series("", index=df.index)
    if "stat" in df.columns:
        stat = df["stat"].where(df["stat"].notna(), stat).astype(str)
    return date + "|" + who + "|" + stat


# A standard line re-pasted at under PERIOD_REPASTE_RATIO of the standard line
# already held for the same game date, player and stat, AND at least
# PERIOD_REPASTE_MIN_DROP below it, is held back as a suspected quarter or half
# line instead of replacing it. A period board whose square labels say only
# "Points" (unverified for the NBA board, but the code cannot tell) otherwise
# REPLACED the full-game line, newest paste winning, and priced 9.5 against a
# 33-point projection (audit 2026-10-09). A 1Q line is about a quarter of the
# full game and a 1H line about half; a real market move is never 40%. The
# absolute floor keeps the small-count moves real markets do make (Blocks 1.5
# to 0.5) out of it. Only standard against standard: an alt rung is supposed
# to sit off the standard number.
PERIOD_REPASTE_RATIO = 0.6
PERIOD_REPASTE_MIN_DROP = 2.0


def _period_repastes(out: pd.DataFrame) -> np.ndarray:
    """Rows of `out` (in recency order, with _key and _rank) that are a
    standard line far below the standard line held before them."""
    held = np.zeros(len(out), dtype=bool)
    if "line" not in out.columns:
        return held
    lines = pd.to_numeric(out["line"], errors="coerce").to_numpy(dtype=float)
    std = (np.asarray(out["_rank"]) == 0) if "_rank" in out.columns else np.ones(len(out), bool)
    current: dict[str, float] = {}
    for i, k in enumerate(out["_key"].to_numpy()):
        x = lines[i]
        if not std[i] or not math.isfinite(x):
            continue
        prev = current.get(k)
        if (prev is not None and x < PERIOD_REPASTE_RATIO * prev
                and prev - x >= PERIOD_REPASTE_MIN_DROP):
            held[i] = True                   # keep the full-game line
            continue
        current[k] = x
    return held


def collapse_alt_lines(df: pd.DataFrame) -> pd.DataFrame:
    """One line per (game date, player, stat): standard first, then Goblin,
    then Demon; among equals, the NEWEST paste wins.

    Position in the frame is recency: callers merge as concat([existing,
    batch]), so a later row is a newer paste. Without `_ord` the sort was on
    (key, rank) alone, and pandas' multi-key sort is STABLE, so between two
    standard lines for the same player and stat the previously saved row
    always won. A line PrizePicks moved from 24.5 to 26.5 was discarded on
    every re-paste and the board kept comparing against a number the market no
    longer posts.

    Except a suspected period line (PERIOD_REPASTE_RATIO above): it never
    replaces the full-game line it sits far below. How many were held back is
    `attrs['period_held']` on the result, and per game date
    `attrs['period_held_by_date']`, so the paste report can say so.

    Safe for the record because the record's FREEZE_KEYS are (game_id,
    person_id, stat), WITHOUT the line, and first seen wins: the pick made at
    24.5 stays frozen as the evidence it is, and the board moves to 26.5.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["_key"] = _collapse_key(out)
    out["_rank"] = out["odds_type"].map(lambda o: _ODDS_RANK.get(str(o).lower(), 3)) \
        if "odds_type" in out.columns else 0
    out["_ord"] = np.arange(len(out))
    held = _period_repastes(out)
    held_dates = (out.loc[held, "game_date"].astype(str).value_counts()
                  if "game_date" in out.columns else pd.Series(dtype=int))
    out = (out[~held]
           .sort_values(["_key", "_rank", "_ord"], ascending=[True, True, False],
                        kind="mergesort")
           .drop_duplicates("_key", keep="first")
           .sort_values("_ord", kind="mergesort")
           .drop(columns=["_key", "_rank", "_ord"])
           .reset_index(drop=True))
    out.attrs["period_held"] = int(held.sum())
    out.attrs["period_held_by_date"] = {str(k): int(v) for k, v in held_dates.items()}
    return out


# ── One call from paste text to stored lines ─────────────────────────────────
def _classify_row(stat_type, name, is_combo, league) -> tuple[str | None, str, str | None]:
    try:
        return classify_stat(stat_type, name, bool(is_combo), league)
    except Exception as e:  # noqa: BLE001 - one bad row costs one row (audit)
        logger.warning("classify_stat: row failed (%s)", type(e).__name__)
        return None, "unmapped", "could not be read"


# Why a line that reached a date still cannot be compared, in compare()'s own
# order (a refusal outranks a failed match, a failed match outranks a game the
# site does not project). Keys are shown to the visitor, so props_ui words
# EVERY key here (its _WHY_WORDS); a key it does not know raises there.
#   not priced         period props, Dunks, double-doubles, two-player squares
#   other board        a projection from another board: NBA1Q, a live board, NHL
#   unrecognised stat  a label nothing maps
#   not matched        a name not on a current roster (or ambiguous)
#   not projected      a matched line for a game the site does not project
#                      (preseason, playoffs, Cup final; the play-in too when the
#                      page marks its schedule, see PROJECTED_COL)
NOT_COMPARABLE = ("not priced", "other board", "unrecognised stat", "not matched",
                  "not projected")
# Other boards named in the report, and the share of a paste that makes one
# of them THE board the visitor copied (props_ui then says "that is the NHL
# board, not the NBA board" instead of "not priced").
MAX_BOARD_NAMES = 8
WRONG_BOARD_SHARE = 0.5


def _board_name(league) -> str:
    return " ".join(str(league or "").split()).upper()[:20]


def _classify_and_resolve(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Each line's stat verdict (stat, kind, refusal), its team code, and its
    name resolved against the roster (match, person_id, how, roster_name,
    roster_team), in input order. The paste path's own verdicts: prepare_paste
    runs it on a parse, restore_saved on a board the visitor's browser kept,
    so a restored line is judged exactly as a pasted one."""
    df = df.copy()
    df["_i"] = np.arange(len(df))           # paste order IS recency for collapse
    kinds = [_classify_row(*r) for r in
             df[["stat_type", "name", "is_combo", "league"]].itertuples(index=False)]
    df["stat"] = [k[0] for k in kinds]
    df["kind"] = [k[1] for k in kinds]
    df["refusal"] = [k[2] for k in kinds]
    df["team_code"] = [teams.canonical(t) if isinstance(t, str) else None for t in df["team"]]
    combo = df["kind"].eq("refused") & df["refusal"].eq(REFUSE_COMBO)
    resolved, unresolved = resolve_names(df[~combo], roster)
    parts = []
    if not resolved.empty:
        parts.append(resolved.assign(match="matched"))
    if not unresolved.empty:
        parts.append(unresolved.rename(columns={"reason": "match"}))
    if combo.any():
        parts.append(df[combo].assign(match="refused", how="combo"))
    df = pd.concat(parts, ignore_index=True) if parts else df
    return df.sort_values("_i", kind="mergesort").drop(columns="_i").reset_index(drop=True)


def prepare_paste(text, roster: pd.DataFrame, schedule: pd.DataFrame, now=None,
                  dates=None) -> tuple[pd.DataFrame, dict]:
    """parse -> classify -> resolve names -> route -> collapse.

    Returns (lines, report). `lines` keeps EVERY parsed row (refused,
    unmatched and unrouted included) so each date's ledger can account for it.
    `report` carries the parse diagnostics and where the lines went, which the
    paste panel turns into its confirmation and its pointers:

      kept = comparable + sum(not_comparable.values()) + unrouted

    `comparable` counts the lines that can reach a board at all (a mapped stat,
    a matched player, an upcoming game the site projects). The confirmation
    counts THOSE, never `kept`: "2 lines loaded" for one unknown name and one
    1Q prop promised two rows and showed none (audit 2026-10-09; DiamondValue's
    "6000 lines updated, nothing showing"), and so did a preseason paste
    (not_comparable['not projected']). Each stage catches errors per row.

    Also: `other_boards` {board name: lines} for lines from another board,
    whether or not they routed; `wrong_board`, that board's name when it holds
    at least WRONG_BOARD_SHARE of the paste (the visitor copied the wrong
    feed), else None; `period_held`, standard lines held back as suspected
    quarter or half lines (collapse_alt_lines). Names are capped, never
    escaped here: the page escapes what it renders.
    """
    parsed = parse_any(text)
    report = {"parsed": int(len(parsed)), "parse": dict(parsed.attrs),
              "by_date": {}, "unrouted": 0, "unrouted_names": [], "kept": 0,
              "comparable": 0, "comparable_by_date": {},
              "not_comparable": {k: 0 for k in NOT_COMPARABLE},
              "other_boards": {}, "wrong_board": None, "period_held": 0}
    if parsed.empty:
        return parsed, report
    df = _classify_and_resolve(parsed, roster)
    df = route_lines(df, schedule, now=now, dates=dates)
    df = collapse_alt_lines(df)
    report["period_held"] = int(df.attrs.get("period_held", 0))
    vc = df["game_date"].dropna().value_counts()
    report["by_date"] = {str(k): int(v) for k, v in sorted(vc.items())}
    unr = df[df["game_date"].isna()]
    report["unrouted"] = int(len(unr))
    report["unrouted_names"] = sorted({str(x) for x in unr["name"]})[:8]
    report["kept"] = int(len(df))

    other = (df["kind"].astype(str).eq("refused")
             & df["refusal"].map(is_league_refusal).astype(bool))
    boards = df.loc[other, "league"].map(_board_name).value_counts()
    report["other_boards"] = {str(k): int(v) for k, v in boards.head(MAX_BOARD_NAMES).items()}
    if len(boards) and boards.iloc[0] >= WRONG_BOARD_SHARE * len(df):
        report["wrong_board"] = str(boards.index[0])

    on_date = df["game_date"].notna()
    kind = df["kind"].astype(str)
    matched = df["match"].astype(str).eq("matched")
    ok = kind.eq("ok")
    refused = kind.eq("refused")
    # Explicitly False only: a line that reached a date but no single game
    # (projectable None) is left comparable, as it was before the flag.
    unprojected = df["projectable"].map(lambda v: v is False or v is np.False_).astype(bool)
    comparable = on_date & ok & matched & ~unprojected
    report["not_comparable"] = {
        "not priced": int((on_date & refused & ~other).sum()),
        "other board": int((on_date & other).sum()),
        "unrecognised stat": int((on_date & ~ok & ~refused).sum()),
        "not matched": int((on_date & ok & ~matched).sum()),
        "not projected": int((on_date & ok & matched & unprojected).sum()),
    }
    comp = df[comparable]
    report["comparable"] = int(len(comp))
    cvc = comp["game_date"].value_counts()
    report["comparable_by_date"] = {str(k): int(v) for k, v in sorted(cvc.items())}
    return df, report


def lines_for_date(lines: pd.DataFrame | None, date) -> pd.DataFrame:
    """The stored lines whose own game falls on ET date `date`."""
    if lines is None or lines.empty or "game_date" not in lines.columns:
        return pd.DataFrame(columns=list(lines.columns) if lines is not None else [])
    d = str(pd.Timestamp(date).date())
    return lines[lines["game_date"].astype(str) == d].reset_index(drop=True)


# ── The visitor's board: in-process, token-keyed, never on the server's disk ─
# Every internal link is a full page load (a new Streamlit session), so
# st.session_state cannot carry a paste from the slate to a game page. The
# board lives HERE, in this process, keyed by an unguessable token that rides
# in ?board=. It is never written to disk and never frozen (the visitor's
# browser keeps its own copy: saved_board below). Bounded by an LRU
# with a sliding TTL, a board cap and a total-row cap, because one process
# serves every visitor and a hostile one could otherwise fill memory.
#
# THE TOKEN IS IN EVERY URL (nav, cards, back links, share), so anyone handed a
# link READS that board. Nobody can WRITE to it: a board is immutable once
# stored. add_to_board is copy-on-write (a NEW token holding old + new), and
# the page's Clear all only drops the token from that visitor's own URL. A
# second visitor on a copied link used to merge into, and clear, the first
# visitor's board (audit 2026-10-09).
BOARD_TTL_S = 18 * 3600
BOARD_MAX = 200
BOARD_MAX_ROWS = 12_000         # a full slate, one line per player and stat, is ~5,000
BOARD_TOTAL_ROWS = 100_000
# MEASURED (audit 2026-10-09): ~1.1 KB per stored row with hostile 70-character
# labels, so the old 200,000-row cap let the store hold ~227 MB. The store now
# keeps only the columns compare() reads, measures each board once when it is
# written, and evicts by BYTES as well as by count and rows.
BOARD_MAX_BYTES = 64 * 2**20
# What compare() and collapse_alt_lines() read from a stored line. 'stat' is
# kept because the collapse key prefers it: dropping it would key an old row
# by label and a new one by stat, and a re-paste would stop replacing it.
BOARD_COLUMNS = ("name", "team_code", "stat_type", "stat", "line", "direction",
                 "odds_type", "opp", "is_combo", "league", "match", "person_id", "how",
                 "roster_name", "roster_team", "game_date", "game_id", "tip_utc")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
_clock = time.monotonic
# The wall clock a board's paste time is read from (DiamondValue's "lines
# saved" stamp): the TTL runs on the monotonic _clock, this is only shown.
# A module binding, so a test pins it.
_wall = now_utc
# NOT DEFENDED, deliberately: a visitor who makes BOARD_MAX throwaway boards
# pushes everyone else's out of this LRU. There is no identity on an anonymous
# public process to ration by, and a global rate limit on new boards would let
# the same visitor lock real visitors out of pasting at all, which is worse
# than an eviction they can undo by pasting again (the panel says so).


@dataclass
class _Board:
    lines: pd.DataFrame
    touched: float = field(default=0.0)
    nbytes: int = 0
    dropped: int = 0            # rows the BOARD_MAX_ROWS cap dropped on the last write
    period_held: dict = field(default_factory=dict)   # {game_date: lines} held back on
                                                      # the write that made this board
    pasted_utc: object = None   # wall-clock UTC of that write: the last paste, shown


_BOARDS: "OrderedDict[str, _Board]" = OrderedDict()
_BOARDS_LOCK = threading.Lock()


def valid_token(token) -> bool:
    return isinstance(token, str) and bool(_TOKEN_RE.match(token))


def _evict_locked(now: float) -> None:
    """TTL first, then least recently used, until every cap holds. Never the
    last board standing: one board is bounded by BOARD_MAX_ROWS, and a write
    moves its board to the end first, so a fresh paste is never the one lost."""
    for tok in [t for t, b in _BOARDS.items() if now - b.touched > BOARD_TTL_S]:
        del _BOARDS[tok]
    rows = sum(len(b.lines) for b in _BOARDS.values())
    nbytes = sum(b.nbytes for b in _BOARDS.values())
    while len(_BOARDS) > 1 and (len(_BOARDS) > BOARD_MAX or rows > BOARD_TOTAL_ROWS
                                or nbytes > BOARD_MAX_BYTES):
        _tok, b = _BOARDS.popitem(last=False)
        rows -= len(b.lines)
        nbytes -= b.nbytes


def _cap(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(the stored frame, rows dropped by BOARD_MAX_ROWS). Compact columns only."""
    lines = lines if lines is not None else pd.DataFrame()
    keep = [c for c in BOARD_COLUMNS if c in lines.columns]
    lines = lines[keep]
    dropped = max(0, len(lines) - BOARD_MAX_ROWS)
    if dropped:
        lines = lines.iloc[-BOARD_MAX_ROWS:]      # newest pastes are last
    return lines.reset_index(drop=True).copy(), dropped


def _store_locked(token: str, lines: pd.DataFrame, now: float,
                  period_held: dict | None = None, pasted_utc=None) -> None:
    stored, dropped = _cap(lines)
    _BOARDS[token] = _Board(stored, now, int(stored.memory_usage(deep=True).sum()), dropped,
                            dict(period_held or {}),
                            pasted_utc if pasted_utc is not None else _wall())
    _BOARDS.move_to_end(token)
    _evict_locked(now)


def new_board(lines: pd.DataFrame) -> str:
    """Store `lines` as a new board and return its token."""
    token = secrets.token_urlsafe(24)
    with _BOARDS_LOCK:
        _store_locked(token, lines, _clock())
    return token


def get_board(token) -> pd.DataFrame | None:
    """A COPY of the board for `token`, or None if unknown or expired."""
    if not valid_token(token):
        return None
    with _BOARDS_LOCK:
        now = _clock()
        _evict_locked(now)
        b = _BOARDS.get(token)
        if b is None:
            return None
        b.touched = now
        _BOARDS.move_to_end(token)
        return b.lines.copy()


def board_info(token) -> dict | None:
    """{rows, nbytes, dropped, period_held, period_held_by_date, pasted_utc}
    for `token`'s board, or None. `dropped` is how many rows the BOARD_MAX_ROWS
    cap removed on the write that made this board, `period_held` how many
    pasted lines that write held back as suspected quarter or half lines
    against the board's full-game lines (collapse_alt_lines), per game date in
    `period_held_by_date`: counted, so the paste panel can say so instead of
    changing the board in silence. `pasted_utc` is when that write happened
    (the visitor's last paste, an aware UTC datetime), for the board to show."""
    if not valid_token(token):
        return None
    with _BOARDS_LOCK:
        b = _BOARDS.get(token)
        if b is None:
            return None
        return {"rows": int(len(b.lines)), "nbytes": int(b.nbytes), "dropped": int(b.dropped),
                "period_held": int(sum(b.period_held.values())),
                "period_held_by_date": dict(b.period_held), "pasted_utc": b.pasted_utc}


def add_to_board(token, lines: pd.DataFrame, pasted_utc=None) -> str:
    """COPY ON WRITE: store `token`'s board plus `lines` (collapsing the
    ladder, newest winning) under a NEW token and return it. The board under
    `token` is left exactly as it was, so a visitor who opened someone else's
    link can add lines without changing the board that link shows. An
    unknown, expired or malformed `token` starts from nothing. `pasted_utc`
    (a board the visitor's browser kept, restore_saved) keeps that board's own
    paste time; otherwise the write is stamped now.

    The superseded board moves to the eviction end of the LRU: it is usually
    the same visitor's previous copy, dead weight once their URL carries the
    new token, so under pressure it goes before anyone's live board. Reading
    it again (an old or shared link) moves it back."""
    with _BOARDS_LOCK:
        now = _clock()
        _evict_locked(now)
        b = _BOARDS.get(token) if valid_token(token) else None
        parts = [x for x in ((b.lines if b is not None else None), lines)
                 if x is not None and not x.empty]
        merged = collapse_alt_lines(pd.concat(parts, ignore_index=True)) if parts \
            else pd.DataFrame()
        held = dict(merged.attrs.get("period_held_by_date") or {}) if parts else {}
        if b is not None:
            _BOARDS.move_to_end(token, last=False)
        new = secrets.token_urlsafe(24)
        _store_locked(new, merged, now, period_held=held, pasted_utc=pasted_utc)
    return new


def clear_board(token) -> None:
    """Forget the board under `token` for EVERYONE holding that link (it rides
    in every URL). The page does not call this: Clear all drops the token from
    the visitor's own URL and the board ages out of the LRU. Kept for tests and
    for an operator. Nothing on disk is touched."""
    if not valid_token(token):
        return
    with _BOARDS_LOCK:
        _BOARDS.pop(token, None)


def board_count() -> int:
    with _BOARDS_LOCK:
        return len(_BOARDS)


# ── The visitor's board, kept in THEIR browser (Barrett, 2026-11-05) ────────
# "I have to reenter them every time I refresh": the board above lives in this
# process, so a link without ?board= (a bookmark, a new tab) or a restart (a
# reboot, or freshen's purge on a deploy) loses it. The page keeps a copy in
# the visitor's own browser (props_ui.board_keeper, localStorage) and brings
# it back when the link carries no live board. Nothing is written on the
# server, and the copy never reaches another visitor.
#
# The copy carries what a PASTE carries plus the game each line was routed
# to, never the verdicts compare trusts (person_id, match, how, roster_*):
# restore_saved re-derives those through the paste path's own
# _classify_and_resolve. It never carries a token either: a restore mints a
# fresh one, so a shared link stays read-only. It comes back from a browser,
# so restore_saved is a second paste box: exact shape, every field through
# the paste validators, the board's own row cap, and it never raises.
SAVED_VERSION = 1
SAVED_COLUMNS = ("name", "team_code", "stat_type", "line", "direction", "odds_type",
                 "opp", "is_combo", "league", "game_date", "game_id", "tip_utc")
SAVED_MAX_AGE_DAYS = 7     # a copy pasted longer ago than this is dropped whole
SAVED_AHEAD_DAYS = 3       # PrizePicks posts today and tomorrow; further out is junk
SAVED_MAX_SKEW_S = 300     # a paste time this far in the future is junk too
_DIRECTIONS = frozenset({"both", "more", "less", ""})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_GAME_ID = re.compile(r"^\d{8,12}$")


def _json_value(v):
    """A stored cell as a plain JSON value: None for missing and non-finite,
    never NaN (JSON.parse rejects a bare NaN)."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def saved_board(lines: pd.DataFrame | None, pasted_utc, save_id: str,
                today: dt.date, owner: str | None = None) -> dict | None:
    """The browser's copy of a board: {"v", "id", "owner", "cols",
    "pasted_utc", "rows"}, one row per stored line whose own game falls on
    `today` (ET) or later, in SAVED_COLUMNS order. `save_id` is what the
    browser acknowledges a write with and `owner` names the board (a hash of
    its token, never the token). None when no line qualifies: a board of
    past games has nothing worth keeping."""
    if lines is None or lines.empty or "game_date" not in lines.columns:
        return None
    floor = today.isoformat()
    keep = lines["game_date"].map(lambda d: isinstance(d, str) and d >= floor)
    t = lines[keep]
    if t.empty:
        return None
    rows = [[_json_value(rec.get(c)) for c in SAVED_COLUMNS]
            for rec in t.to_dict(orient="records")]
    return {"v": SAVED_VERSION, "id": str(save_id), "owner": owner,
            "cols": list(SAVED_COLUMNS), "pasted_utc": _utc_iso(_parse_ts(pasted_utc)),
            "rows": rows}


def _saved_row(row, lo: dt.date, hi: dt.date) -> dict | None:
    """One kept line back as a parsed line (the shape _classify_and_resolve
    reads), or None. Every field through the paste validators."""
    if not isinstance(row, list) or len(row) != len(SAVED_COLUMNS):
        return None
    r = dict(zip(SAVED_COLUMNS, row))
    name = _text(r["name"], MAX_NAME_LEN)
    if name is None or not _person_like(name):
        return None
    label = r["stat_type"]
    if not isinstance(label, str) or len(label) > MAX_LABEL_LEN:
        return None
    line = _parse_line(r["line"])
    if line is None:
        return None
    gd = r["game_date"]
    if not isinstance(gd, str) or not _ISO_DATE.match(gd):
        return None
    try:
        day = dt.date.fromisoformat(gd)
    except ValueError:
        return None
    if not lo <= day <= hi:
        return None
    team, opp = r["team_code"], r["opp"]
    gid, tip, side = r["game_id"], r["tip_utc"], r["direction"]
    return {"name": name,
            "team": teams.canonical(team) if isinstance(team, str) and len(team) <= 30 else None,
            "stat_type": label.strip(), "line": line,
            "direction": side if isinstance(side, str) and side in _DIRECTIONS else "",
            "odds_type": (_text(r["odds_type"], 20) or "standard").lower(),
            "opp": teams.canonical(opp) if isinstance(opp, str) and len(opp) <= 30 else None,
            "is_combo": r["is_combo"] is True,
            "league": _text(r["league"], 20),
            "game_date": gd,
            "game_id": gid if isinstance(gid, str) and _GAME_ID.match(gid) else None,
            "tip_utc": _utc_iso(_parse_ts(tip)) if isinstance(tip, str) and len(tip) <= 40 else None}


def _saved_row_or_none(row, lo: dt.date, hi: dt.date) -> dict | None:
    """One bad row costs one row, as at every paste stage."""
    try:
        return _saved_row(row, lo, hi)
    except Exception:  # noqa: BLE001 - untrusted cell types
        return None


def restore_saved(payload, roster: pd.DataFrame, today: dt.date, now: dt.datetime
                  ) -> tuple[pd.DataFrame, dt.datetime, int] | None:
    """(lines, pasted_utc, rows dropped) from the browser's copy of a board,
    ready for add_to_board(None, lines, pasted_utc=...), or None when nothing
    usable comes back. NEVER raises: the payload is whatever a browser sent.

    The whole copy is refused when its shape is wrong, it holds more rows than
    a board can (BOARD_MAX_ROWS), or its paste time is unreadable, in the
    future or older than SAVED_MAX_AGE_DAYS. A row is dropped when any field
    fails the paste validators or its game is not between `today` and
    SAVED_AHEAD_DAYS later. Names are resolved again against `roster`."""
    try:
        if not isinstance(payload, dict) or payload.get("v") != SAVED_VERSION:
            return None
        if payload.get("cols") != list(SAVED_COLUMNS):
            return None
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows or len(rows) > BOARD_MAX_ROWS:
            return None
        stamp = payload.get("pasted_utc")
        pasted = _parse_ts(stamp) if isinstance(stamp, str) and len(stamp) <= 40 else None
        if pasted is None:
            return None
        age = (now - pasted).total_seconds()
        if age < -SAVED_MAX_SKEW_S or age > SAVED_MAX_AGE_DAYS * 86400:
            return None
        hi = today + dt.timedelta(days=SAVED_AHEAD_DAYS)
        recs = [r for r in (_saved_row_or_none(row, today, hi) for row in rows) if r is not None]
        if not recs:
            return None
        lines = _classify_and_resolve(pd.DataFrame(recs), roster)
        return lines, pasted.astimezone(UTC), len(rows) - len(recs)
    except Exception as e:  # noqa: BLE001 - a browser's copy must never raise into the page
        logger.warning("restore_saved: refused (%s)", type(e).__name__)
        return None


# ── Per-stat reliability ─────────────────────────────────────────────────────
# Measured side-picking edge per stat, written by an NBA validate_lean_accuracy
# run on validate/test seasons. Missing file -> every row 'unmeasured', which the
# UI renders quietly. Combos inherit their WEAKEST component.
RELIABILITY_PATH = DATA_DIR / "stat_reliability.csv"
TIER_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "unmeasured": 3, "none": 4,
              "negative": 5}
TIER_LABEL = {"strong": "strong", "moderate": "moderate", "weak": "weak",
              "unmeasured": "unmeasured", "none": "no edge",
              "negative": "worse than a coin flip"}


@lru_cache(maxsize=4)
def _load_reliability(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - absent table degrades, never raises
        logger.info("stat reliability table unavailable (%s)", type(e).__name__)
        return pd.DataFrame(columns=["stat", "tier", "side_edge_pts", "note"])
    for c in ("stat", "tier", "side_edge_pts", "note"):
        if c not in df.columns:
            df[c] = np.nan
    df["stat"] = df["stat"].astype(str)
    df["side_edge_pts"] = pd.to_numeric(df["side_edge_pts"], errors="coerce")
    return df


def reliability_table() -> pd.DataFrame:
    return _load_reliability(str(RELIABILITY_PATH))


def tier_components(stat: str) -> tuple[str, ...]:
    if stat in COMBO_STATS:
        return (stat,) + tuple(STAT_FORMULA[stat])
    return (stat,)


def reliability_for(stat: str) -> dict:
    """{tier, edge, note} for a stat. The weakest of the stat's own row (if
    measured directly) and its components' rows. A tier means a MEASUREMENT: a
    row with no side_edge_pts renders 'unmeasured' whatever tier it claims."""
    tbl = reliability_table()
    rows = tbl[tbl["stat"].isin(tier_components(stat))] if not tbl.empty else tbl
    if rows is None or rows.empty:
        return {"tier": "unmeasured", "edge": None, "note": ""}
    if rows["side_edge_pts"].isna().any():
        return {"tier": "unmeasured", "edge": None,
                "note": str(rows.iloc[0].get("note") or "")}
    worst = max(rows["tier"].astype(str), key=lambda t: TIER_ORDER.get(t, 3))
    # Weakest WITHIN the worst tier too: the lowest measured edge (table order
    # is highest first, so .iloc[0] named Rebounds for PRA when Assists is weaker).
    hit = rows[rows["tier"].astype(str) == worst].sort_values("side_edge_pts", kind="mergesort").iloc[0]
    tier = worst if worst in TIER_ORDER else "unmeasured"
    return {"tier": tier, "edge": float(hit["side_edge_pts"]),
            "note": str(hit.get("note") or "")}


# ── Distributions: what compare() needs from the model side ──────────────────
@runtime_checkable
class DistProvider(Protocol):
    """Hands compare() a distribution for one player, game and stat.

    Optional methods, used when present:
      void_risk(person_id, game_id, stat) -> float   (Reboot hook, see below)
      source(person_id, game_id, stat) -> str        ('model', 'baseline', ...)
    """

    def pmf(self, person_id: int, game_id: str, stat: str) -> Pmf | None: ...


# Variance-to-mean ratios GIVEN MINUTES from docs/data_notes.md (2023-24 ..
# 2025-26). Unmeasured stats take the largest ratio among their components,
# else 1 (Poisson). Unconditional spread is wider (minutes are uncertain), so
# this provider is a labelled BASELINE until the simulation ships pmfs.
DISPERSION = {"pts": 2.2, "ftm": 1.8, "fta": 2.0, "pra": 1.9, "fantasy": 2.4,
              "reb": 1.0, "ast": 1.0, "fg3m": 1.0, "stl": 1.0, "blk": 1.0,
              "tov": 1.0, "pf": 0.8}


def _dispersion(stat: str) -> float:
    if stat in DISPERSION:
        return DISPERSION[stat]
    comps = [DISPERSION[c] for c in STAT_FORMULA.get(stat, {}) if c in DISPERSION]
    return max(comps) if comps else 1.0


def shipped_mean(row, stat: str) -> float | None:
    """E[Y | plays] for `stat` from a projection row: the stat's own column,
    or the exact linear combination of its components' means."""
    if row is None or not hasattr(row, "get"):
        return None

    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None
    own = num(row.get(stat))
    if own is not None:
        return own
    formula = STAT_FORMULA.get(stat)
    if not formula:
        return None
    total = 0.0
    for comp, w in formula.items():
        cv = num(row.get(comp))
        if cv is None:
            return None
        total += w * cv
    return total


class MeanDispersionProvider:
    """BASELINE provider: a negative binomial around each projected mean.

    Until the simulation writes pmfs, this turns the projection frame's means
    into distributions with the measured dispersions above. Fantasy Score is
    built in tenths of a point (mean x10, dispersion x10) and rescaled to the
    0.1 grid. Every row it serves is labelled source='baseline'.
    """

    def __init__(self, projections: pd.DataFrame):
        self._rows: dict[tuple[int, str], dict] = {}
        if projections is not None and not projections.empty:
            for rec in projections.to_dict(orient="records"):
                try:
                    self._rows[(int(rec["person_id"]), str(rec["game_id"]))] = rec
                except Exception:  # noqa: BLE001
                    continue
        self._cache: dict[tuple[int, str, str], Pmf | None] = {}

    def pmf(self, person_id, game_id, stat):
        key = (int(person_id), str(game_id), str(stat))
        if key in self._cache:
            return self._cache[key]
        out = None
        mu = shipped_mean(self._rows.get(key[:2]), stat)
        if mu is not None and mu >= 0 and stat in STAT_FORMULA:
            d = _dispersion(stat)
            if stat == "fantasy":
                out = Pmf.from_negbin(10.0 * mu, 10.0 * d).rescale(0.1)
            else:
                out = Pmf.from_negbin(mu, d)
        self._cache[key] = out
        return out

    def source(self, person_id, game_id, stat) -> str:
        return "baseline"


# ── The lean ─────────────────────────────────────────────────────────────────
# REBOOT HOOK. PrizePicks voids a MORE pick on a full-game line when the player
# leaves in the first half and does not return, unless he already cleared it.
# Pricing that needs an exit model (play-by-play, v1.1). Until then the void
# probability is NOT removed from P(win) and P(loss); a provider may report it
# through `void_risk`, which is carried on the row as 'void_risk' and shown as
# a note on More leans. Flip this and implement the branch below together.
REBOOT_VOIDS_MODELLED = False


def side_probabilities(p_lt: float, p_eq: float, p_gt: float,
                       p_void_more: float | None = None) -> tuple[float, float]:
    """(q_more, q_less): each side's win probability with ties (and, once
    modelled, Reboot voids) excluded. q = P_win / (P_win + P_loss)."""
    if REBOOT_VOIDS_MODELLED and p_void_more is not None:   # pragma: no cover
        raise NotImplementedError("Reboot void pricing is v1.1")
    denom = p_gt + p_lt
    if not denom > 0:
        return 0.5, 0.5
    return p_gt / denom, p_lt / denom


def lean_from(q_more: float, band: float = EVEN_BAND) -> str:
    if q_more > 0.5 + band:
        return "More"
    if q_more < 0.5 - band:
        return "Less"
    return "Even"


# ── Compare ──────────────────────────────────────────────────────────────────
TABLE_COLUMNS = [
    "person_id", "game_id", "game_date", "player", "team", "opp", "stat",
    "stat_label", "line", "model", "mean", "gap", "p_over", "p_push", "p_under",
    "q_more", "q_less", "confidence", "lean", "sides", "odds_type", "tier",
    "tier_edge", "tier_note", "games_prior", "thin", "void_risk", "mu_min",
    "void_note", "source", "how", "pasted_name", "model_version", "as_of",
    "provisional", "adjusted", "tip_utc", "tie_possible",
]
_GAMES_COLS = ("games_prior", "n_games", "games")
_MIN_COLS = ("mu_min", "min")

# ROLE-BASED VOID NOTE (spec 3.5: "More leans show a role-based void risk
# note"). PrizePicks' Reboot voids a More pick on a first-half exit with no
# return, and nothing here measures how often that happens by role (an exit
# model needs play-by-play, v1.1), so there is NO percentage: a More lean for
# a player projected under VOID_NOTE_BELOW_MIN minutes carries `void_note`,
# and the page says in words that a bench role carries more of that risk.
# PROVISIONAL, not measured: 20 minutes is a bench role, not a fitted cut.
VOID_NOTE_BELOW_MIN = 20.0


def _new_meta(n: int) -> dict:
    return {"posted": int(n), "matched": 0, "unmatched": 0, "ambiguous": 0,
            "unprojected": 0, "n_refused": 0, "refused": {}, "refused_why": {},
            "unmapped": {}, "unprojected_why": {}, "unmatched_names": [],
            "ambiguous_names": [], "nonexact": [], "errors": 0,
            "period_board_suspect": False, "period_withheld": 0,
            "games_unknown": False, "unresolved": pd.DataFrame()}


def _finite_or_nan(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _flag(v) -> bool | None:
    """A projection row's bool flag (provisional, adjusted), or None when the
    row does not say. Never guessed: the record refuses None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return None


def projection_as_of(prow) -> str | None:
    """The projection's as-of stamp: its own `as_of`, else `computed_at` (the
    production frame's stamp, pipeline section 3) as ISO UTC. None when it
    carries neither: the record then refuses the row."""
    own = _jsonable(prow.get("as_of")) if hasattr(prow, "get") else None
    if own:
        return str(own)
    return _utc_iso(_parse_ts(prow.get("computed_at"))) if hasattr(prow, "get") else None


def _bump(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def _centred_pmf(dist_provider, pid: int, gid: str, prow: dict, stat: str) -> Pmf | None:
    """The provider's pmf for one player, game and stat, re-centred on the
    projection row's shipped mean when it carries one: how compare() reads
    every line, and every part of a Fantasy Score line."""
    pmf = dist_provider.pmf(pid, gid, stat) if dist_provider is not None else None
    if pmf is None:
        return None
    target = shipped_mean(prow, stat)
    if target is not None and abs(pmf.mean() - target) > 1e-9 * max(1.0, abs(target)):
        try:
            pmf = pmf.recentre(target, lower=None if stat == "fantasy" else 0.0)
        except ValueError as e:
            logger.warning("could not re-centre %s %s on %s: %s", pid, stat, target, e)
    return pmf


def compare(lines: pd.DataFrame, dist_provider, projections: pd.DataFrame,
            roster: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Every pasted line against its distribution. Returns (table, meta).

    Per matched line: P(>), P(=), P(<) from the provider's Pmf (re-centred on
    the shipped mean when the projection row carries one), q per side with ties
    excluded, lean = argmax q with an Even band, display value = the mid-p
    median, gap = median - line. The table is RANKED (rank_rows). `meta`
    accounts for every line: posted = matched + unmatched + ambiguous +
    n_refused + unprojected. Never raises on a bad row; the row is counted as
    unprojected with reason 'internal error'.

    FAILS CLOSED where a missing column would otherwise read as safe (audit
    2026-10-09): a projection frame with no prior-game count marks EVERY row
    thin (meta['games_unknown']), and a row whose projection does not say
    whether it is provisional carries provisional=None, which the record
    refuses. tip_utc comes from the schedule via routing, not the projection.
    Lines that went through prepare_paste keep their paste-time match verdict;
    only lines with no verdict at all are resolved here.
    """
    lines = lines if lines is not None else pd.DataFrame()
    meta = _new_meta(len(lines))
    if lines.empty:
        meta["reason"] = "no lines pasted"
        return pd.DataFrame(columns=TABLE_COLUMNS), meta

    proj_by_key: dict[tuple[int, str], dict] = {}
    proj_by_pid: dict[int, list[dict]] = {}
    if projections is not None and not projections.empty:
        for rec in projections.to_dict(orient="records"):
            try:
                k = (int(rec["person_id"]), str(rec["game_id"]))
            except Exception:  # noqa: BLE001
                continue
            proj_by_key[k] = rec
            proj_by_pid.setdefault(k[0], []).append(rec)
    else:
        meta["reason"] = "no projections for this date"
    games_col = next((c for c in _GAMES_COLS
                      if projections is not None and c in projections.columns), None)
    meta["games_unknown"] = games_col is None

    df = lines.copy()
    if "match" in df.columns:
        # Resolved at paste time. A frame where nothing matched has no
        # person_id column; resolving it AGAIN here duplicated 'match' and the
        # merge raised for the whole date (audit, fuzz input d2_30).
        if "person_id" not in df.columns:
            df["person_id"] = None
    elif "person_id" in df.columns:
        df["match"] = np.where(df["person_id"].notna(), "matched", "unmatched")
    else:
        df = _resolve_inline(df, roster)
    unresolved_rows = []
    rows = []
    for rec in df.to_dict(orient="records"):
        label = str(rec.get("stat_type") or "")
        try:
            stat, kind, why = classify_stat(label, rec.get("name"),
                                            bool(rec.get("is_combo")), rec.get("league"))
            if kind == "refused":
                # Qualified, so two refusals that share a label ("Points" on a
                # two-player square and "Points" from a 1Q board) keep their
                # own reasons instead of one overwriting the other.
                shown_label = label
                if why == REFUSE_COMBO:
                    shown_label = f"{label} (two-player square)"
                elif is_league_refusal(why):
                    shown_label = f"{label} ({str(rec.get('league') or '').upper()[:20]} board)"
                _bump(meta["refused"], shown_label)
                meta["refused_why"][shown_label] = why
                meta["n_refused"] += 1
                continue
            if kind == "unmapped":
                _bump(meta["unmapped"], label)
                meta["n_refused"] += 1
                continue
            match = rec.get("match")
            if match != "matched" or rec.get("person_id") is None or pd.isna(rec.get("person_id")):
                bucket = "ambiguous" if match == "ambiguous" else "unmatched"
                meta[bucket] += 1
                names = meta[f"{bucket}_names"]
                if len(names) < 50 and rec.get("name") not in names:
                    names.append(str(rec.get("name")))
                unresolved_rows.append(rec)
                continue
            pid = int(rec["person_id"])
            gid = rec.get("game_id")
            gid = None if gid is None or (isinstance(gid, float) and math.isnan(gid)) else str(gid)
            prow = proj_by_key.get((pid, gid)) if gid else None
            if prow is None and not gid:
                cands = proj_by_pid.get(pid, [])
                prow = cands[0] if len(cands) == 1 else None
            if prow is None:
                meta["unprojected"] += 1
                _bump(meta["unprojected_why"], "no projection for this player and game")
                continue
            gid = str(prow["game_id"])
            pmf = _centred_pmf(dist_provider, pid, gid, prow, stat)
            if pmf is None:
                meta["unprojected"] += 1
                _bump(meta["unprojected_why"], "no distribution for this stat")
                continue
            shown = None
            if stat == "fantasy":
                # The formula on the medians the page shows for its parts,
                # each read exactly as its own line would be (fantasy_shown).
                parts = {s: _centred_pmf(dist_provider, pid, gid, prow, s) for s in FANTASY_WEIGHTS}
                on = fantasy_on_shown(pmf, {s: (q.median() if q is not None else None)
                                            for s, q in parts.items()})
                if on is None:
                    meta["unprojected"] += 1
                    _bump(meta["unprojected_why"], "no distribution for a Fantasy Score part")
                    continue
                shown, pmf = on
            line = float(rec["line"])
            p_lt, p_eq, p_gt = pmf.split(line)
            vr = np.nan
            if hasattr(dist_provider, "void_risk"):
                try:
                    vr = float(dist_provider.void_risk(pid, gid, stat))
                except Exception:  # noqa: BLE001
                    vr = np.nan
            q_more, q_less = side_probabilities(p_lt, p_eq, p_gt, None)
            lean = lean_from(q_more)
            median = shown if shown is not None else pmf.median()
            rel = reliability_for(stat)
            src = "model"
            if hasattr(dist_provider, "source"):
                try:
                    src = str(dist_provider.source(pid, gid, stat))
                except Exception:  # noqa: BLE001
                    src = "model"
            gp = prow.get(games_col) if games_col else None
            gp = float(gp) if gp is not None and not pd.isna(gp) else np.nan
            # Unknown history is thin: no column, NaN, or below MIN_GAMES.
            thin = not math.isfinite(gp) or gp < MIN_GAMES
            prov = _flag(prow.get("provisional"))
            mu_min = next((x for x in (_finite_or_nan(prow.get(c)) for c in _MIN_COLS)
                           if math.isfinite(x)), float("nan"))
            tip = rec.get("tip_utc")
            if tip is None or (isinstance(tip, float) and math.isnan(tip)):
                tip = prow.get("tip_utc")
            step = STAT_STEP.get(stat, 1.0)
            how = str(rec.get("how") or "exact")
            if how != "exact":
                meta["nonexact"].append((str(rec.get("name")), str(prow.get("name")
                                         or rec.get("roster_name") or ""), how))
            rows.append({
                "person_id": pid, "game_id": gid,
                "game_date": rec.get("game_date") or prow.get("game_date"),
                "player": prow.get("name") or rec.get("roster_name") or rec.get("name"),
                "team": prow.get("team") or rec.get("roster_team") or rec.get("team_code"),
                "opp": prow.get("opp") or rec.get("opp"),
                "stat": stat, "stat_label": label, "line": line,
                "model": median, "mean": pmf.mean(), "gap": median - line,
                "p_over": p_gt, "p_push": p_eq, "p_under": p_lt,
                "q_more": q_more, "q_less": q_less,
                "confidence": max(q_more, q_less) - 0.5, "lean": lean,
                "sides": offered_sides(rec.get("direction"), rec.get("odds_type")),
                "odds_type": str(rec.get("odds_type") or "standard").lower(),
                "tier": rel["tier"], "tier_edge": rel["edge"], "tier_note": rel["note"],
                "games_prior": gp, "thin": thin, "void_risk": vr,
                "mu_min": mu_min,
                "void_note": bool(lean == "More" and math.isfinite(mu_min)
                                  and mu_min < VOID_NOTE_BELOW_MIN),
                "source": src, "how": how, "pasted_name": rec.get("name"),
                "model_version": prow.get("model_version"), "as_of": projection_as_of(prow),
                "provisional": prov,
                # A row recomputed with this visitor's mark-outs (pipeline's
                # `adjusted`): shown on the page, never scored (spec 4).
                "adjusted": _flag(prow.get("adjusted")),
                "tip_utc": tip,
                # On the stat's outcome grid, so an exact landing is possible:
                # whole numbers for counts, tenths for Fantasy (35.5 can tie).
                "tie_possible": abs(line / step - round(line / step)) < 1e-6,
            })
        except Exception as e:  # noqa: BLE001 - a bad row is counted, never raised
            meta["errors"] += 1
            meta["unprojected"] += 1
            _bump(meta["unprojected_why"], "internal error")
            logger.warning("compare: row failed (%s: %s)", type(e).__name__, e)
    meta["unresolved"] = pd.DataFrame(unresolved_rows)
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    if not table.empty:
        # A board whose lines sit far below the full-game projections is a
        # quarter or half board pasted as full-game. Its rows are REFUSED,
        # counted like any refusal, so board_rows, the card pills and the
        # record never see them (audit 2026-10-09: they used to stay shown,
        # ranked first, More at q near 1, under a notice).
        held = _period_suspect_rows(table)
        if held.any():
            for label in table.loc[held, "stat_label"]:
                shown_label = f"{label} (looks like a quarter or half board)"
                _bump(meta["refused"], shown_label)
                meta["refused_why"][shown_label] = REFUSE_PERIOD_BOARD
            meta["n_refused"] += int(held.sum())
            meta["period_board_suspect"] = True
            meta["period_withheld"] = int(held.sum())
            table = table[~held].reset_index(drop=True)
    meta["matched"] = len(table)
    if not table.empty:
        table = rank_rows(table)
        meta["by_tier"] = table["tier"].value_counts().to_dict()
    return table, meta


def _resolve_inline(df: pd.DataFrame, roster) -> pd.DataFrame:
    """Lines with no paste-time verdict at all: resolve names now. Any stale
    resolution columns are dropped first, so nothing is ever duplicated."""
    df = df.drop(columns=[c for c in ("match", "reason", "how", "person_id", "roster_name",
                                      "roster_team") if c in df.columns])
    if roster is None or roster.empty:
        return df.assign(match="unmatched", person_id=None, how="none")
    resolved, unresolved = resolve_names(df, roster)
    parts = []
    if not resolved.empty:
        parts.append(resolved.assign(match="matched"))
    if not unresolved.empty:
        parts.append(unresolved.rename(columns={"reason": "match"}).assign(person_id=None))
    return pd.concat(parts, ignore_index=True) if parts else df


# A 1Q or 1H board pasted as if it were full-game prices Jokic's 6.5
# first-quarter points against a 26-point projection, and every row leans More
# by a mile. A group of lines is that board when, over at least
# PERIOD_MIN_PLAYERS players with a mean above 2, the median player's median
# line/mean ratio is under PERIOD_RATIO (a 1Q line is ~0.25 of the full game,
# a 1H line ~0.5; a real full-game board sits near 1). Per PLAYER, so one star
# on a minutes restriction the model has not heard of (all his lines far
# below) never takes his game down with him. The groups are each game and the
# whole date, so a period board for one game is caught inside a full-game
# slate, and a period slate with a few lines per game is caught as a whole.
# Was: 10 rows, date only, and a notice that left the rows shown.
PERIOD_RATIO = 0.55
PERIOD_MIN_PLAYERS = 3


def _period_group_suspect(t: pd.DataFrame) -> bool:
    mean = pd.to_numeric(t["mean"], errors="coerce")
    line = pd.to_numeric(t["line"], errors="coerce")
    keep = (mean > 2.0) & line.notna()
    if not keep.any():
        return False
    ratio = (line[keep] / mean[keep]).groupby(t.loc[keep, "person_id"]).median()
    return len(ratio) >= PERIOD_MIN_PLAYERS and float(ratio.median()) < PERIOD_RATIO


def _period_suspect_rows(table: pd.DataFrame) -> np.ndarray:
    """Rows of a compare() table that belong to a suspected period board."""
    if table is None or table.empty:
        return np.zeros(0, dtype=bool)
    if _period_group_suspect(table):
        return np.ones(len(table), dtype=bool)
    mask = np.zeros(len(table), dtype=bool)
    for _gid, idx in table.groupby(table["game_id"].astype(str), sort=False).indices.items():
        if _period_group_suspect(table.iloc[idx]):
            mask[idx] = True
    return mask


def rank_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Reliability tier first, then confidence (distance of the winning side's
    q from a coin flip). NEVER the raw gap: it is an argmax over the model's
    own error and is not comparable across stats (two rebounds and two points
    are different things). Ties break on player then stat, for stable output."""
    if table is None or table.empty:
        return table
    out = table.copy()
    out["_t"] = out["tier"].map(lambda t: TIER_ORDER.get(str(t), 3))
    out["_c"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out = (out.sort_values(["_t", "_c", "player", "stat"],
                           ascending=[True, False, True, True], kind="mergesort")
              .drop(columns=["_t", "_c"]))
    return out.reset_index(drop=True)


def filter_pickable(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pickable, hidden). Hidden = the lean is a side the line does not offer,
    or there is no lean (Even). Returned, never dropped, so it is counted."""
    if table is None or table.empty or "lean" not in table.columns:
        return table, (table.iloc[0:0] if table is not None else pd.DataFrame())
    keep = np.array([is_pickable(l, s) for l, s in zip(table["lean"], table["sides"])],
                    dtype=bool)
    return table[keep].reset_index(drop=True), table[~keep].reset_index(drop=True)


def filter_thin(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(informed, thin): fewer than MIN_GAMES of history in the projection row.
    FAILS CLOSED: a table with no 'thin' column, or a row whose flag is
    missing, is thin. An unknown history is not an informed one (audit)."""
    if table is None or table.empty:
        return table, (table.iloc[0:0] if table is not None else pd.DataFrame())
    if "thin" not in table.columns:
        return table.iloc[0:0].reset_index(drop=True), table.reset_index(drop=True)
    thin = np.array([True if v is None or (isinstance(v, float) and math.isnan(v))
                     else bool(v) for v in table["thin"]], dtype=bool)
    return table[~thin].reset_index(drop=True), table[thin].reset_index(drop=True)


MAX_VOID_IDS = 1000


def _as_pid(x) -> int | None:
    """A person id from untrusted input, or None. Void ids arrive from the
    visitor's ?out= query param: `int('abc')` there raised the whole board into
    its error state (audit). Booleans, non-finite and fractional floats, and
    anything that is not plain digits are not ids."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x) if math.isfinite(x) and float(x).is_integer() else None
    t = str(x).strip()[:13]
    return int(t) if t.isdigit() else None


def void_id_set(void_ids) -> set[int]:
    """The usable person ids in `void_ids` (capped; junk ignored)."""
    out: set[int] = set()
    try:
        items = list(void_ids or ())[:MAX_VOID_IDS]
    except TypeError:
        return out
    for x in items:
        pid = _as_pid(x)
        if pid is not None:
            out.add(pid)
    return out


def board_rows(table: pd.DataFrame, void_ids=None
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(shown, hidden, thin, void): THE ONE DEFINITION of what the board shows.

    void = a player listed Out, inactive or marked out (the caller's set of
    person_ids). PrizePicks voids those picks, so nothing else about them
    matters. Then hidden (lean not offered, or Even), then thin. Every row lands
    in exactly one of the four. The card pills, the ledger, the per-player rows
    and anything that counts or grades a line read this, so the site cannot
    disagree with itself (GridironValue's pill promised 533 lines and its game
    pages showed 515).
    """
    empty = pd.DataFrame(columns=TABLE_COLUMNS)
    if table is None or table.empty:
        return empty, empty.copy(), empty.copy(), empty.copy()
    ids = void_id_set(void_ids)
    is_void = np.array([_as_pid(p) in ids for p in table["person_id"]], dtype=bool)
    void = table[is_void].reset_index(drop=True)
    rest = table[~is_void].reset_index(drop=True)
    shown, hidden = filter_pickable(rest)
    shown, thin = filter_thin(shown)
    return shown, hidden, thin, void


def ledger(meta: dict, shown=None, hidden=None, thin=None, void=None) -> dict:
    """The accounting identity, with `balanced` saying whether it holds."""
    out = {k: int(meta.get(k, 0)) for k in ("posted", "matched", "unmatched",
                                            "ambiguous", "n_refused", "unprojected")}
    out["balanced"] = out["posted"] == (out["matched"] + out["unmatched"] + out["ambiguous"]
                                        + out["n_refused"] + out["unprojected"])
    if shown is not None:
        parts = [len(x) if x is not None else 0 for x in (shown, hidden, thin, void)]
        out.update(shown=parts[0], hidden=parts[1], thin=parts[2], void=parts[3])
        out["balanced"] = out["balanced"] and sum(parts) == out["matched"]
    return out


# ── The record: LOCAL ONLY ───────────────────────────────────────────────────
# Without a frozen record a hit rate is worse than useless, because it looks
# authoritative: compare() joins lines to projections computed NOW, so grading
# an old board scores it against a model retrained since, and the record would
# improve every time the models are refit. So the model's number is frozen the
# moment a line is first seen and grading reads the frozen value.
#
# FIRST SEEN WINS, and the key has NO line: one real game outcome is one pick.
# With the line in the key a market move froze a second row for the same
# player and stat and the same result was graded twice (GridironValue). The
# record keeps the pick actually made, at the line it was made against.
#
# Stamped with the model_version and the as-of stamp OF THE PROJECTION ROW (its
# computed_at in the production frame). Never written on a READ_ONLY (shared)
# host, never for a provisional projection, never for a row recomputed with a
# visitor's mark-outs (spec 4: "in-session recomputes are never scored"),
# never after tip. Lives in cache/props_record_*.json, which .gitignore fences
# off.
#
# EVERY GUARD FAILS CLOSED (audit 2026-10-09: compare() output with no tip and
# no provisional flag froze all 9 rows at a time after every game had tipped).
# A row freezes only when its tip is KNOWN and in the future, its projection
# says provisional=False AND adjusted=False EXPLICITLY, it carries an as-of
# stamp, and its own model_version equals the one the caller freezes for (a
# frozen game served from an older model's committed rows is not that model's
# pick; it used to be stamped with the caller's version). And only rows the
# board SHOWS freeze: board_rows() is the one definition, so a hidden side,
# an Even, a thin or a void line is never a pick in the record.
FREEZE_KEYS = ("game_id", "person_id", "stat")
_FREEZE_FIELDS = ("game_id", "person_id", "stat", "game_date", "player", "team", "opp",
                  "stat_label", "line", "model", "mean", "gap", "p_over", "p_push",
                  "p_under", "q_more", "q_less", "lean", "sides", "odds_type", "tier",
                  "tier_edge", "games_prior", "void_risk", "source", "how",
                  "model_version")


def record_path(game_date) -> Path:
    d = str(pd.Timestamp(game_date).date())
    return _cache.dc_path(f"props_record_{d}_{VERSION}.json")


def _freeze_key(row) -> str:
    return "|".join(str(row.get(k)) for k in FREEZE_KEYS)


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (str, int)) else str(v)


def _explicit_false(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and not bool(v)


def freeze_projections(table: pd.DataFrame, model_version: str | None,
                       as_of: str | None = None, now=None, void_ids=None,
                       report: dict | None = None) -> int:
    """Freeze every not-yet-frozen SHOWN line. Returns how many NEW picks froze.

    Refuses (returns 0) under cache.READ_ONLY: a snapshot taken on the shared
    host would carry that container's model, die at restart, and meanwhile
    make the published record disagree with the committed one. Raises
    ValueError without a model_version (a caller bug, not a condition): the
    version this record is for, which every frozen row's own model_version
    must equal. `as_of` stands in only for a row that carries no stamp at all.
    `report`, if given, collects how many rows each guard turned away.
    """
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("freeze_projections needs a model_version to stamp the record")
    rep = report if report is not None else {}
    if _cache.READ_ONLY:
        logger.info("read-only deployment: the record is not written here")
        rep["read_only"] = True
        return 0
    if table is None or table.empty:
        return 0
    shown, hidden, thin, void = board_rows(table, void_ids)
    rep["not_shown"] = len(hidden) + len(thin) + len(void)
    now = now or now_utc()
    now_et = to_et(now)
    stamp = now_et.astimezone(UTC).isoformat(timespec="seconds")
    by_date: dict[str, list[dict]] = {}

    def skip(why: str) -> None:
        rep[why] = rep.get(why, 0) + 1
    for rec in shown.to_dict(orient="records"):
        if any(rec.get(k) is None or (isinstance(rec.get(k), float) and math.isnan(rec.get(k)))
               for k in FREEZE_KEYS + ("game_date",)):
            skip("missing_key")
            continue
        if not _explicit_false(rec.get("provisional")):
            skip("provisional_or_unknown")
            continue
        if not _explicit_false(rec.get("adjusted")):
            skip("adjusted_or_unknown")       # a visitor's mark-out recompute
            continue
        row_version = rec.get("model_version")
        if not isinstance(row_version, str) or not row_version.strip():
            skip("no_model_version")
            continue
        if row_version != model_version:
            skip("version_mismatch")
            continue
        tip = _parse_ts(rec.get("tip_utc"))
        if tip is None:
            skip("tip_unknown")
            continue
        if tip <= now_et:
            skip("tipped")                    # post-tip: never frozen
            continue
        if not (_jsonable(rec.get("as_of")) or as_of):
            skip("no_as_of")
            continue
        by_date.setdefault(str(pd.Timestamp(rec["game_date"]).date()), []).append(rec)
    added = 0
    for d, recs in by_date.items():
        path = record_path(d)
        existing: dict[str, dict] = {}
        try:
            payload = _cache.read_json(path) or {}
            for r in payload.get("rows", []):
                existing.setdefault(_freeze_key(r), r)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not read record %s: %s", path, e)
            continue                      # never overwrite a record we cannot read
        for rec in recs:
            row = {k: _jsonable(rec.get(k)) for k in _FREEZE_FIELDS}
            row["person_id"] = int(rec["person_id"])
            row["game_id"] = str(rec["game_id"])
            k = _freeze_key(row)
            if k in existing:
                continue                  # first seen wins
            row["frozen_at"] = stamp
            row["as_of"] = _jsonable(rec.get("as_of")) or as_of
            row["tip_utc"] = _utc_iso(_parse_ts(rec.get("tip_utc")))
            existing[k] = row
            added += 1
        _cache.write_json(path, {"game_date": d, "version": VERSION, "updated_at": stamp,
                                 "rows": list(existing.values())})
    if rep:
        logger.info("freeze_projections: %d frozen; turned away %s", added, rep)
    return added


def load_frozen(game_date) -> pd.DataFrame | None:
    path = record_path(game_date)
    try:
        payload = _cache.read_json(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read record %s: %s", path, e)
        return None
    if not payload or not payload.get("rows"):
        return None
    df = pd.DataFrame(payload["rows"])
    df.attrs["updated_at"] = payload.get("updated_at")
    return df


def frozen_dates() -> list[str]:
    out = []
    for p in sorted(_cache.CACHE_DIR.glob(f"props_record_*_{VERSION}.json")):
        m = re.match(r"props_record_(\d{4}-\d{2}-\d{2})_", p.name)
        if m:
            out.append(m.group(1))
    return out


def actual_value(stat: str, row) -> float | None:
    """The settled value of `stat` from a box-score row, or None when any
    component is missing. A missing component is NOT zero (phantom actuals)."""
    total = 0.0
    formula = STAT_FORMULA.get(stat)
    if not formula:
        return None
    for comp, w in formula.items():
        v = row.get(comp) if hasattr(row, "get") else None
        if v is None or pd.isna(v):
            return None
        total += w * float(v)
    return round(total, 4)


def _played(row) -> bool:
    if "played" in row and row["played"] is not None and not pd.isna(row["played"]):
        return bool(row["played"])
    for c in ("seconds", "played_seconds", "sec"):
        if c in row and row[c] is not None and not pd.isna(row[c]):
            return float(row[c]) > 0
    return True      # row presence = played, when the frame carries no flag


def grade(frozen: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Score frozen picks against the box score.

    outcome: 'void' (DNP: PrizePicks removes the pick), 'exact' (landed on the
    line: neither a win nor a loss, never a push), 'correct' / 'incorrect'
    (the lean against the side the outcome landed on), 'no_lean' (Even),
    'pending' (no box-score row yet, or a component missing).
    Caveats: PrizePicks settles from partner feeds, so stats.nba.com is a proxy;
    the Reboot void (a More pick on a first-half exit) is not modelled.
    """
    if frozen is None or frozen.empty:
        return pd.DataFrame()
    act: dict[tuple[str, int], dict] = {}
    if actuals is not None and not actuals.empty:
        for rec in actuals.to_dict(orient="records"):
            try:
                act[(str(rec["game_id"]), int(rec["person_id"]))] = rec
            except Exception:  # noqa: BLE001
                continue
    out = []
    for rec in frozen.to_dict(orient="records"):
        try:
            a = act.get((str(rec.get("game_id")), int(rec.get("person_id"))))
        except (TypeError, ValueError):
            a = None
        actual, result, outcome = None, None, "pending"
        if a is not None:
            if not _played(a):
                outcome = "void"
            else:
                actual = actual_value(str(rec.get("stat")), a)
                if actual is not None:
                    line = float(rec["line"])
                    result = "More" if actual > line else "Less" if actual < line else "Exact"
                    lean = rec.get("lean")
                    if result == "Exact":
                        outcome = "exact"
                    elif lean not in ("More", "Less"):
                        outcome = "no_lean"
                    else:
                        outcome = "correct" if lean == result else "incorrect"
        out.append({**rec, "actual": actual, "result": result, "outcome": outcome,
                    "model_correct": (True if outcome == "correct" else
                                      False if outcome == "incorrect" else None)})
    return pd.DataFrame(out)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a hit rate (no impossible bounds, sane
    near 0.5 and at small n where the normal interval is worst)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _season_of(game_date) -> str | None:
    try:
        return season_for_date(str(pd.Timestamp(game_date).date()))
    except Exception:  # noqa: BLE001 - an unreadable date is not in any window
        return None


def season_record(graded: list[pd.DataFrame], registry: dict | None = None) -> dict:
    """Aggregate graded picks with an interval and an honest bar.

    The bar is NOT a coin: always taking the side that comes in more often wins
    that side's base rate for free, so an edge means the interval's LOWER bound
    clears that base rate. Clearing it is still not profit: a leg must hit
    54.7% to 58.5% to break even.

    OUT OF SAMPLE ONLY when a model `registry` is given: picks from a season
    inside its training window are the model marking its own homework, so they
    are dropped and counted in 'in_sample_dropped'. A pick with no readable
    game_date is dropped too, since it cannot be shown to be out of sample. A
    registry without a readable trained_through raises ValueError
    (is_in_training_window): a record over an unknown window is not computed.
    """
    frames = [g for g in graded if g is not None and not g.empty]
    if not frames:
        return {"n": 0, "decided": 0}
    g = pd.concat(frames, ignore_index=True)
    dropped = 0
    if registry is not None:
        dates = g["game_date"] if "game_date" in g.columns else pd.Series([None] * len(g))
        seasons = [_season_of(d) for d in dates]
        inside = np.array([s is None or is_in_training_window(s, registry) for s in seasons],
                          dtype=bool)
        dropped = int(inside.sum())
        g = g[~inside].reset_index(drop=True)
        if g.empty:
            return {"n": 0, "decided": 0, "in_sample_dropped": dropped}
    oc = g["outcome"].value_counts().to_dict()
    hits, misses = int(oc.get("correct", 0)), int(oc.get("incorrect", 0))
    decided = hits + misses
    rec = {"n": int(len(g)), "decided": decided, "hits": hits,
           "exact": int(oc.get("exact", 0)), "voids": int(oc.get("void", 0)),
           "pending": int(oc.get("pending", 0)), "no_lean": int(oc.get("no_lean", 0)),
           "in_sample_dropped": dropped,
           "breakeven_lo": BREAKEVEN_LO, "breakeven_hi": BREAKEVEN_HI}
    if not decided:
        return rec
    lo, hi = wilson_interval(hits, decided)
    res = g[g["outcome"].isin(["correct", "incorrect"])]["result"]
    over = float((res == "More").mean()) if len(res) else 0.5
    base = max(over, 1.0 - over)
    rec.update({"hit_rate": round(100 * hits / decided, 1),
                "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1),
                "baseline": round(100 * base, 1), "beats_coin": bool(lo > base),
                "clears_breakeven": bool(100 * lo > BREAKEVEN_LO)})
    return rec


def is_in_training_window(season, registry: dict | None) -> bool:
    """True when `season` ('2025-26' or its end year 2026) is inside the
    model's training window: grading it is the model marking its own homework.

    Fails CLOSED. A registry without a readable trained_through (missing, None,
    not a season label; models.save always writes one) RAISES ValueError: the
    window is unknown, so no pick can be shown to be out of sample. It used to
    return False, which counted the model's own training seasons as out of
    sample. It raises rather than returning True because True would empty the
    record and file every pick under 'in_sample_dropped', a count that claims
    a window nobody knows. A `season` that cannot be read is inside the window
    (True), as season_record already treats a pick with no readable game_date.
    """
    through = registry.get("trained_through") if isinstance(registry, dict) else None
    try:
        last = season_end_year(through)
    except (TypeError, ValueError) as e:
        raise ValueError(f"model registry has no readable trained_through ({through!r}): the "
                         "training window is unknown, so no pick can be shown to be out of "
                         "sample (read it with models.read_registry)") from e
    try:
        year = season_end_year(season) if isinstance(season, str) and "-" in season else int(season)
    except (TypeError, ValueError):
        return True
    return year <= last
