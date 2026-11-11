"""Player lines for BucketValue: paste, parse, match, route, compare, grade.

HOW LINES GET IN, AND WHY THERE IS NO SCRAPER
---------------------------------------------
Two ways, and neither is a scraper. A licensed odds feed (The Odds API, Barrett
2026-11-09) brings DraftKings, FanDuel, BetMGM, PrizePicks and Underdog lines:
bucketlib.live fetches it and the page hands this module a DataFrame
(prepare_feed, compare_books, feed_board). And a visitor pastes a PrizePicks
board or a typed list, the backup. Nothing ever fetches PrizePicks itself:

  1. api.prizepicks.com returns HTTP 403 to server-side requests behind
     Cloudflare bot management (verified on both siblings), and rejects VPN
     and datacenter IPs at the application level, which is also why a bot on
     GitHub Actions or Streamlit Cloud could never reach it.
  2. Independently, PrizePicks' Terms section 16(l) prohibit "any robot,
     spider, or other automatic device, process, or means to access the Site
     or App". Solving (1) would not solve (2).

So for a paste the visitor opens the feed (FEED_URL) or the board in their own
browser and pastes what they see, or pastes a list of picks typed by hand or
copied from a picks post (parse_typed_lines). This module parses it and never
fetches anything. **Do not add a fetcher here.** tests/test_props.py parses
this module's imports and fails if it, or anything it imports from bucketlib,
imports a network library or the live layer.

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
import functools
import json
import math
import re
import secrets
import threading
import time
import unicodedata
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

# -- The feed link. A LINK the visitor opens, never a request this code makes --
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

# -- Paste-box limits. Each bounds work an anonymous visitor can cause --------
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

# -- Leans --------------------------------------------------------------------
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

# -- Stats --------------------------------------------------------------------
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

# -- Fantasy Score: the formula on the numbers shown (Barrett, 2026-11-05) ----
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
    s = str(label or "")[:MAX_LABEL_LEN].lower().replace("\uff0b", "+")
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
    r"\b(?:[1-4][\s\-]*(?:st|nd|rd|th)?[\s\-]*(?:h|q|half|halves|qtr|qrtr|qrt|quarter|hlf|hf)s?"
    r"|[qh]-?[1-4]"
    r"|(?:first|1st|opening|through)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine"
    r"|ten|eleven|twelve)\s+min(?:s|ute|utes)?"
    r"|in\s+(?:the\s+)?first|(?:first|second)[\s\-]+(?:half|quarter|qtr|qrtr|hlf|frame)"
    r"|(?:third|fourth)[\s\-]+(?:quarter|qtr|qrtr|frame)|opening[\s\-]+(?:quarter|qtr|half)"
    r"|(?:quarter|qtr|half)\s*(?:[1-4](?![\d.])|one|two|three|four)"
    # "First Q", "Q 1", "1st period" (review 7, B23).
    r"|(?:first|second|third|fourth)[\s\-]+q|[qh]\s+[1-4](?![\d.])"
    r"|(?:first|1st|second|2nd|third|3rd|fourth|4th|opening)[\s\-]+period"
    # The sportsbook's "1st 1/2", "1st \u00bd" (review 7, A3).
    r"|(?:first|1st|second|2nd)\s*(?:1/2|\u00bd))(?!\w)")
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
    hit = STAT_MAP.get(label_key(raw)) or typed_stat(raw)
    if hit:
        return hit, "ok", None
    for pat, why in _REFUSALS:
        if pat.search(spaced):
            return None, "refused", why
    return None, "unmapped", "stat not mapped to a projected quantity"


# -- Names that are not people ------------------------------------------------
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


# -- Which sides a line offers ------------------------------------------------
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


def pick_of(value) -> str | None:
    """A visitor's typed pick, 'more' or 'less', or None for anything else."""
    return value if isinstance(value, str) and value in PICKS else None


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


# -- Parsing ------------------------------------------------------------------
# `pick`, `pair` and `pair_loose` come from a typed list only
# (parse_typed_lines): the side the visitor wrote, the game its game heading
# names, and the two teams a loose heading joins ("ORL-PHX"), a game only
# when the schedule pairs them (typed_games). None elsewhere.
PARSED_COLUMNS = ["name", "team", "position", "stat_type", "line", "start_time",
                  "direction", "odds_type", "opp", "is_combo", "league", "source",
                  "pick", "alt_pick", "pair", "pair_loose"]
PICKS = frozenset({"more", "less"})
_ODDS_TYPES = {"standard", "demon", "goblin"}
# attrs['json_shape'] for a feed whose data[] is an empty list: the projections
# feed itself, with no lines on it right now. Any other json_shape means the
# JSON is not the feed.
EMPTY_FEED = "empty feed"


def _new_attrs() -> dict:
    return {"skipped_noname": 0, "skipped_badline": 0, "skipped_live": 0,
            "skipped_error": 0, "truncated": 0, "too_large": False,
            "json_shape": None, "allowed_wager_types": {}, "typed_noname": 0,
            "typed_unread": 0, "typed_recap": 0, "typed_unclear": 0, "typed_period_label": 0}


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
_TEXT_LINE = re.compile(r"[^\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+")


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


# -- A typed list of picks (Barrett, 2026-11-06) ------------------------------
# One pick per line, as a visitor types them or copies them from a picks post.
# The first real one (Magic at Suns, 2026-11-05):
#
#     Suns:
#     Take the OVER on Miles Bridges for a combined 6.5 Rebounds and Assists
#     (Combined Rebounds and Assists will be 7 or more)
#     Oso Ighodaro over 7.5 Points + Rebounds (Combined Points and ...)
#     Magic:
#     Wendell Carter Jr. under 23.5 Points + Rebounds + Assists (...)
#
# Read per line: exactly one number (the line: "7 or more", "7+" and "at least
# 7" are a 6.5 line picked Over, and a "No" after them is the Under), at most
# one side, a stat on either side of the number, and a name of two or more
# words once the words around it ("Take the OVER on", "for a combined") are
# set aside at its edges. Tout decoration is set aside too: emoji and keycap
# numbers, markdown, "POTD:", "NBA:", "#NBA" and "Leg 1:" labels, thread
# numbers ("1/", "[1/3]"), a handle, a date, a tip time, American odds
# ("-120"), unit sizes ("2u") and a units column. A note after the stat
# ("LOCK", ", second half of a back to back") is set aside only on a line
# that names a side, and only when it names no stat and changes none:
# "points, rebounds and assists combined" is PRA, never Points, and
# "rebounds offensive" or "points in the paint" is no Rebounds or Points
# (review 3, R1), nor is "rebounds, only offensive" (review 5, A4). A stat
# run across a comma or a dash with a note after it ("rebounds, assists are
# live too") could be either, so it is not read (review 4, B1); two stats
# with only a comma between them are a combo only when "combined" follows,
# and a letter is a stat only where "+", "&" or "/" joins it ("R+A", "Pts +
# R + A", "Reb + A") or among capitals joined by "and" ("R and A"), so
# "points - a lock" is Points and "points + a lock" is not read (review 5,
# A9, B2; review 6, B2, B14); a slash, pipe, bullet or middle dot between
# them joins them the same way ("Pts / R / A", "Pts | R | A"; review 7, A2).
# A word that changes the stat among a note's first words ("rebounds, but
# only offensive", "points, all from the paint", "rebound chances", "all
# from three") leaves it unread too (review 6, A7; review 7, A8), but the
# matchup does not ("elite offensive rating", "great transition team";
# review 7, B16). A result word before the line or
# first in the note after the stat ("o25.5 pts - cashed", "- that one
# lost"), "cashed" anywhere after it, or a result alone after it or in a
# parenthesis ("- hit", "- easy dub", "(W)", "(cashed it)") makes it a
# recap, counted (typed_recap) and never a pick; later in a note "won" or
# "lost" is context ("Knicks lost last night"), and "cashes" is a hope ("it
# cashes tonight"; review 5, A10, B6; review 6, A8, B13) only beside a
# word of hope: "- this one cashes" is a recap (review 7, B5). "- missed
# shootaround" is about the player and "- push it" is a tout's bet, so both
# are notes (review 7, B19). An average ("Points 27.1 avg") is no pick
# (review 3, R7; review 4, A7).
# A parenthesis is a gloss unless it holds a team, an opponent ("@ATL"), a
# Demon or Goblin tag, a side, a Yes/No, or only words that change the stat
# ("(offensive)", "(3PA)", "(+ assists)", "(and assists)", "(w/ assists)",
# "(assists included)", "(offensive only)"), which join the stat label: the
# line is priced on the stat written or not at all. A parenthesis naming
# another stat outright ("points (rebounds)") or as if added ("(assists
# too)"), or changing it among other words ("(offensive glass)", "(off
# glass)"), leaves it unrecognised, unless the words speak of the matchup
# ("(weak defensive team)", review 4, A2; review 5, A7). A period marker
# anywhere on the line ("(1H)", "Q1", "first 5 minutes", "1st-half", "first
# qtr"), in a section label above it ("1Q props:", "1Q Locks", "## 1Q
# props"), on a name line above it ("Jalen Brunson (1Q)", "Main Slate 1H
# props:", which label every pick under them, named or not, until a label
# naming none) or on its game heading ("Magic @ Suns (1H)", which holds over
# the game's own team headings and sub-labels) makes it a period prop,
# refused and counted as one (review 5, A1, A3, B1, B4; review 6, A1, B4,
# B9), and the report says which picks a label made period props
# (typed_period_label, review 6, B12). A period is read however a book or a
# tout spells it ("1st 1/2", "Q-1", "q 1", "First Q", "1st period", "1st
# qrtr", "1qprops", a keycap 1 before Q; review 7, A3, B23). A "full game"
# on a name line holds for that player's picks and ends only a period a name
# line set; a team label under a period game heading keeps the period unless
# it says "full game" (review 7, B2, B22). Refusing is the safe side, so a
# line naming a period is a label unless the period is cancelled ("No 1Q
# props today", a negation with no mark between it and the period that
# denies the period's picks: "No more waiting, 1Q props:" and "No 1Q
# parlays:" are labels; review 7, A7) or the line reads as a remark: a result
# ("What a 4th quarter!", "1H recap: 3-1") or a remark's verb with a subject
# other than the period ("Q4 Brunson is different"; "1Q is the move" is a
# label; review 4, A6, B2; review 6, A3, B4); "2nd half of a back-to-back"
# is a note, not a period. DiamondValue's "Name, Stat, Line"
# list (comma, pipe, tab or bullet) reads the same way. A name alone on its
# line names the picks on the
# lines after it, with any team, opponent or tip time written beside it. A
# line shaped like a pick that holds more than one number is counted
# (typed_unread), never guessed; so is a pick whose name is a handle
# ("@jalenbrunson1 o25.5 pts", review 4, B3). A tag is a tout's decoration
# only when all of it is ("#NBA", "#PrizePicks", "@hoopspicks:") or it is a
# team's own hashtag ("#DubNation"): a player's handle holding a part of one
# ("@Money23Green", "#Pickett") names nobody, so his line is counted, never
# lent the name above (review 6, A4, B7, A11). A tag that splits wholly into
# a tout's words is decoration only when one of them is a bet's
# ("#LockOfTheDay", "#LetsCash"), so "@King23" and "#Buckets" are a player's
# nickname (review 7, B10, B18). "@Jalen Brunson" is his name (review 5,
# B10, A21).
#
# THE SIDE WRITTEN IS THE VISITOR'S PICK, never the sides the line offers:
# `pick` rides beside the line to the board ("Pick: More") while `direction`
# stays unknown, so a pick the model disagrees with is shown. A square that
# offers one side only lends it to a pick on its number, and a lean on the
# other side is then hidden like any line's; the hidden-lines note names the
# pick and the model's lean (review 3, F19).
# ESPN writes New Orleans "NO": an all-caps "NO" beside the name, or after a
# side word, is read both ways until the roster says whose player it is
# (alt_pick, settled in typed_games). A Pelican keeps his Over; anyone else's
# "NO" is a No. A "NO" after "7+" or a question, or on a line naming no
# player, is the market's No, and so is one with closers after it ("NO -
# fade", "NO at 38c"). A Yes never turns the pick, so the reading weighs a
# No ("No", "Nope", "Nah", "No way", "Not a chance", "(answer: No)",
# _market_answers): it is the answer where it plainly stands as one, right
# after the pick's own question mark ("26+ 3-PT Made? No", "26+ points on
# Saturday? No way"), first after the line or last on it with closers only
# ("7+ R+A No - fade", "26+ points No lol"), before the number of a "7+" line
# ("Miles Bridges No 7+ R+A"), or a parenthesis that is only the answer
# ("(No 2u)"). One that starts a note ("no doubt", "no Towns", "NO TOWNS
# TONIGHT", "with no Towns") or answers a question the note asks ("is Towns
# playing? No") is the note's (review 4, A1; review 5, A2, A5, B3; review
# 6, A2, A6, B1, B3), and "vs NO", "@NO" are New Orleans, the opponent (A4).
# Anywhere else the side cannot be told, above all beside a written Over or
# Under ("No, I like Jalen Brunson over 25.5 points", "over 25.5 points, no
# for real", "(no, not a typo)"), so the line is counted (typed_unclear),
# never guessed (review 6, B5, B6). Right after the pick's own question,
# other ways to answer are read ("Never", "Fat chance", "Yeah, no" are a No;
# "I like it" a Yes), and other words there leave the side unclear ("? Tough
# one"). So do a No that opens a clause that may be a Yes ("no it is a
# lock"), a correction ("No, wait, yes") and a No idiom beside a written
# side ("over 25.5 points - no way"; review 7, A5, A6, B6, B7, B8). A word
# holding a digit closes no answer: "(no B2B)" is a note (review 7, B4).
#
# A TYPED LINE HAS NO GAME TIME, so it routes like board text: the player's
# next game today or tomorrow. A team heading ("Suns:") only GROUPS the list:
# it scopes the name match, and the roster's team wins when they disagree. A
# game is NAMED only by a game heading ("Magic @ Suns", "ORL vs PHX 9:00
# PM", "Magic @ Suns on ESPN": a broadcast, a call sign or "| late game" is
# set aside; review 7, A9, B11), an opponent on the line ("vs Lakers",
# "@ATL"), or two team headings
# the SCHEDULE pairs (typed_games). Two teams joined by "&", "and", "/", a
# dash or "x" are two team headings, a game only when the schedule says so
# ("Knicks & Nets" groups a list; review 3, R6): when their latest game, or
# their next, is against each other, the pair names that game as "@" does,
# so last night's "ORL-PHX" list is late all the next day (review 4, B4),
# unless both teams play again today: that list is today's (review 5, B9).
# A named game that has
# tipped claims the line, so a list sent in after its game is reported as
# late instead of being filed under each team's next game. The first list
# was exactly that: it came in after Magic at Suns had ended.
TYPED_SOURCE = "typed"
MAX_TYPED_LINES = 5_000         # a typed list is dozens of lines; this bounds a hostile paste
MAX_TYPED_WORDS = 40_000        # ...and this one of long lines: a board line is ~2 words
TYPED_LATE_HOURS = 6.0          # a named game that tipped this recently claims a typed line
TYPED_MORNING_AFTER_HOUR = 12   # until noon ET, two headings that met last night name that game
_SIDE_WORDS = {"over": "more", "under": "less", "more": "more", "less": "less",
               "higher": "more", "lower": "less"}
# A letter is a side only right before the number ("o 6.5") or last on the
# line ("6.5 R+A U"), never inside a name.
_SIDE_LETTERS = {"o": "more", "u": "less"}
_OR_SIDE = {"more": "more", "over": "more", "higher": "more",
            "less": "less", "fewer": "less", "under": "less", "lower": "less"}
_YES_NO = {"yes": True, "no": False}
# The words a visitor answers a market with, beyond Yes and No ("? Nope", "?
# Yeah", review 6, A6). Only a No ever turns the pick: a Yes is the line's own
# side.
_ANSWER_WORDS = {**_YES_NO, "nope": False, "nah": False, "naw": False, "yep": True,
                 "yup": True, "yeah": True, "yea": True}
# An amount after an answer ("No 2u", "No at 38c", "No 100%"): a closer. A word
# that merely holds a digit is not one: "(no B2B)", "(no 76ers)" (review 7, B4).
_AMOUNT = re.compile(r"\d{1,3}(?:\.\d{1,2})?(?:u|units?|c|cents?|%)?"
                     r"|[+-]?\d{3,4}|\d{1,2}x|x\d{1,2}")
# Spellings of the same answers: "Nooo", "Nahh", "Nay", "Yesss", "Yeahh"
# (review 7, A6).
_NO_FORM = re.compile(r"no{2,}|na+h+|nay|nope+")
_YES_FORM = re.compile(r"ye+s+|ye+a+h+|yu+p+|ye+p+")


def _answer_value(w: str):
    """True for a Yes, False for a No, None for any other word."""
    if w in _ANSWER_WORDS:
        return _ANSWER_WORDS[w]
    if _NO_FORM.fullmatch(w):
        return False
    return True if _YES_FORM.fullmatch(w) else None


# After the pick's own question mark, the other ways to say No or Yes ("?
# Never", "? Fat chance", "? Doubt it", "? Fade"), closers aside; any other
# words there leave the side unclear (_market_answers; review 7, A6).
_ASKED_NO = frozenset({("never",), ("negative",), ("unlikely",), ("fade",), ("fading",),
                       ("pass",), ("doubt",), ("i", "doubt"), ("fat", "chance"),
                       ("not", "likely"), ("not", "happening"), ("certainly", "not"),
                       ("probably", "not"), ("not",), ("not", "a", "shot"), ("hard", "pass"),
                       ("fade", "him"), ("fade", "that")})
_ASKED_YES = frozenset({("like",), ("i", "like"), ("love",), ("i", "love"), ("think", "so"),
                        ("i", "think", "so"), ("of", "course"), ("certainly",), ("likely",),
                        ("probably",), ("yessir",)})
# After "no", a word that makes the two one No: "? No way" and "? No chance"
# were read as a note, and the board showed the opposite pick (review 6, A2).
_NO_IDIOM_NEXT = frozenset({"way", "chance", "shot", "shots", "dice", "go", "bueno"})
# After an answer, the words a tout closes it with ("No lol", "No - fade", "NO
# 2u", "No at 38c", "(No for me)"). With nothing else after it, a Yes or No
# stands alone (_after_answer).
_ANSWER_CLOSERS = frozenset({
    "for", "sure", "imo", "lol", "lmao", "tbh", "fr", "fade", "fading", "pass", "lock", "locks",
    "locked", "max", "play", "bet", "x2", "x3", "2x", "3x", "at", "@", "sir", "thanks", "thx",
    "please", "pls", "again", "easy", "ez", "money", "cash", "hammer", "smash", "nuke", "tail",
    "unit", "units", "u", "bol", "lfg", "potd", "free", "boys", "baby", "bro", "here", "today",
    "tonight", "me", "all", "day", "jose", "hell", "fam",
    # "Lock it in", "Let's ride", "Absolutely" after a pick's question
    # (review 7, A6).
    "it", "in", "this", "lets", "let's", "ride", "go", "absolutely", "definitely", "obviously",
    "duh", "ofc", "easily"})
# The words a clause opens with. After a No ("No he's cold", "no but
# seriously", "no u can't fade this") they make the No and the clause one
# thought, which beside a written Over or Under may argue either side
# (review 6, B6).
_CLAUSE_STARTS = frozenset({
    "he", "hes", "he's", "she", "shes", "she's", "they", "theyre", "they're", "i", "im", "i'm",
    "id", "i'd", "we", "its", "it's", "it", "this", "that", "thats", "that's", "but", "and",
    "because", "cause", "cuz", "since", "so", "you", "ur", "your", "sadly", "honestly",
    "though", "if", "as", "not", "never", "there", "theres", "there's",
    # "No wait, yes", "No actually": a correction (review 7, B8)
    "wait", "actually", "nvm", "scratch"})
# Before a No that closes a parenthesis, the words that make it the visitor's
# answer: "(answer: No)", "(I say no)", "(taking the No)" (review 6, A6).
_ANSWER_LEADS = frozenset({"answer", "my", "pick", "i", "say", "taking", "take", "going", "im",
                           "i'm", "me", "call", "verdict", "the", "is", "with", "its", "it's"})
# Words that may stand between a line and the question mark closing it, as
# its own question's ("Will Jalen Brunson score 26+ points on Saturday? No",
# "...in his next game? No"): its answer is the visitor's (review 6, B1).
_QUESTION_CONTEXT = frozenset({
    "on", "in", "this", "his", "the", "next", "game", "games", "again", "tonight", "today",
    "tomorrow", "for", "or", "more", "regulation", "road", "home", "night", "matchup", "at",
    "tonight's", "today's", "tomorrow's", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "monday's", "tuesday's", "wednesday's", "thursday's",
    "friday's", "saturday's", "sunday's"})
_ODDS_WORDS = {"demon": "demon", "goblin": "goblin", "standard": "standard"}
_NEUTRAL_SIDES = frozenset({"o/u", "over/under", "u/o", "under/over", "more/less",
                            "less/more", "higher/lower"})
_UNIT_WORDS = frozenset({"u", "unit", "units"})
_TYPED_NUM = re.compile(r"^\d{1,3}(?:\.\d{1,2})?$")
_TYPED_GLUED = re.compile(r"^(o|u|over|under)(\d{1,3}(?:\.\d{1,2})?)$", re.IGNORECASE)
# "25.5pts", "o11.5reb": a stat glued to a decimal line. A decimal only, so
# "2u", "1H" and "3PT" are never split, and the glued word must be a stat.
_TYPED_GLUED_STAT = re.compile(r"^(o|u|over|under)?(\d{1,3}\.\d{1,2})([a-z][a-z+&]{0,14})$",
                               re.IGNORECASE)
_TYPED_PLUS = re.compile(r"^(\d{1,3}(?:\.5)?)\+$")                 # "7+", "6.5+"
_TYPED_PLUS_STAT = re.compile(r"^(\d{1,3}(?:\.5)?\+)([a-z][a-z+&]{0,14})$",   # "9+reb"
                              re.IGNORECASE)
_TYPED_SPREAD = re.compile(r"^-\d{1,2}(?:\.\d{1,2})?$")      # "-3.5": a spread, never a line
_TYPED_ODDS = re.compile(r"^[+-]\d{3,4}$")                      # "-120", "+105": American odds
_TYPED_UNITS = re.compile(r"^\d{1,2}(?:\.\d{1,2})?u(?:nits?)?$", re.IGNORECASE)    # "2u"
_TYPED_PAREN = re.compile(r"\(([^()]{0,200})\)")
_TYPED_BULLET = re.compile(r"^(?:[-*>\u2022\u00b7\u25cf\u2013\u2014\u2192]+\s*"
                           r"|\d{1,3}[.)]\s+|#{1,6}\s+)")
# A keycap number emoji (digit, variation selector, keycap): a bullet, so its
# digit is never a second number on the line.
_KEYCAP = re.compile(r"[0-9#*]\ufe0f?\u20e3")
# ...except a 1 to 4 just before Q or H: "1\ufe0f\u20e3Q props:" is a 1Q label (review 7, A3).
_KEYCAP_PERIOD = re.compile(r"([1-4])\ufe0f?\u20e3(?=[qhQH](?![a-z]))")
# What a chat client leaves around a pick: **bold**, __bold__, _italics_,
# `code`, ~~strike~~. An underscore inside a word ("@hoops_picks") or
# opening a handle ("@_hoopspicks:") stays (review 4, B10).
_MARKDOWN = re.compile(r"\*+|`+|~+|(?<!@)_{2,}|(?<![\w@])_|_(?!\w)")
_TZ_WORDS = ("et|est|edt|ct|cst|cdt|mt|mst|mdt|pt|pst|pdt|az|eastern|central|mountain"
             "|pacific")
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?")
# Before a pick and never part of it, each anchored at the start: a thread
# number or a date ("1/", "2/5", "11/5", "11/05/26", "[1/3]", "Nov 5:"), a
# handle ("@hoopspicks:"), a hashtag ("#NBA"), a tip time ("7:30pm ET"), a
# multiplier ("2x", "x2") and a max-play label ("MAX PLAY:"). A "2pm" or
# "3pm" is a stat as often as a time ("3PM over 4.5"): _read_typed tries both.
_TYPED_PREFIX = re.compile(
    r"^(?:\d{1,2}/\d{0,2}(?:/\d{2,4})?[:.)\-]?\s+|\[\d{1,2}/\d{1,2}\]\s*|@\w{1,30}:\s*"
    r"|#[a-z]\w{0,29}:?\s+"
    r"|" + _MONTH + r"\s*\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?(?:\s*[:.)\-]\s*|\s+)"
    r"|(?![23]pm\b)\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?!\w)"
    r"(?:\s*(?:" + _TZ_WORDS + r")\b)?\s*[:\-]?\s*"
    r"|\d{1,2}x[:\-]?\s+|x\d{1,2}[:\-]?\s+|max\s+(?:play|bet|units?)\b\s*[:\-]?\s*)",
    re.IGNORECASE)
# A tip time that may be the stat ("3pm Jalen Brunson o25.5 pts", against
# "3PM over 4.5 Stephen Curry").
_TIP_23 = re.compile(r"^[23]\s*p\.?m\.?(?!\w)(?:\s*(?:" + _TZ_WORDS + r")\b)?\s*[:\-]?\s*",
                     re.IGNORECASE)
# A handle with no colon ("@hoopspicks Josh Hart ..."), set aside unless it is
# a team: "@ATL" is the opponent.
_HANDLE = re.compile(r"^@(\w{1,30})(?:\s*[:\-]\s*|\s+)")
# "1 Miles Bridges over 6.5 R+A": a list number, set aside only when another
# number is left to be the line, and never the 2 or 3 that begins a stat
# ("3 Pointers Made").
_LEAD_NUMBER = re.compile(r"^\d{1,2}\s+(?=[A-Z])(?!(?i:p(?:oint|t|m)))")
# "Leg 1:", "Pick #2 -", "#3)": a numbered label before the pick, whose number
# is not the line.
_TYPED_LABEL = re.compile(
    r"^(?:(?:leg|pick|play|bet|lock|prop|game|parlay)\s*#?\s*\d{1,2}|#\s*\d{1,2})"
    r"\s*[:.)\-]?\s+", re.IGNORECASE)
_TYPED_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\S*)?$")
# A field separator: a comma, semicolon, pipe, tab, bullet, middle dot or
# arrow, or a slash or ">" with a space either side ("Miles Bridges \u2022 R+A
# \u2022 6.5", "Miles Bridges\u2022Rebs+Asts\u20226.5", "Josh Hart >> o8.5 reb").
_TYPED_SEPS = re.compile(r"\s*[,;|\t\u2022\u00b7\u2192]\s*|\s+(?:/|>+)\s+")
# A separator standing between two stat words or letters ("Pts / R / A",
# "Reb \u2022 A", "Pts | R | A", "Pts\u2022R\u2022A") joins them as "/" does: split into fields, the
# combo was priced as its first stat (review 7, A2).
_STAT_SEP_CHARS = "/|\u2022\u00b7"
_GLUED_SEP = re.compile(r"[|\u2022\u00b7]")
# A units column: a lone small whole number closing a separated row
# ("Miles Bridges, Rebs+Asts, 6.5, Over, 1").
_UNITS_FIELD = re.compile(r",\s*\d{1,2}\s*$")
# What must be left before a units column or a list number is set aside: a
# line with a decimal or a "7+". A whole-number line on a stat with a digit
# in its name ("Stephen Curry, 3PT, 4") keeps its number.
_HAS_LINE = re.compile(r"\d\.\d|\d\+")
# Between the two teams of a game heading: "@", "at", "vs", "v", "versus",
# "x", "and", "&", "/" or a dash ("ORL-PHX", "Magic/Suns").
_TYPED_GAME = re.compile(r"\s*(?:@|\bat\b|\bvs\b\.?|\bv\b\.?|\bversus\b|\bx\b|\band\b"
                         r"|[&/\-\u2013\u2014])\s*", re.IGNORECASE)
# What a heading carries besides its teams, set aside before it is read: a tip
# time ("9:00 PM", "9 ET", "21:00", "9.00pm", "2100", "at 9 PM"), a record
# ("4-4"), a date ("11/5", "Nov 5, 2026", "5 Nov", "2026-11-05"), a weekday
# ("Thursday Night"), a time zone ("ET", "Eastern Time", "Arizona time",
# "local"), the network ("on ESPN", "| ESPN", "NBA League Pass", "on Amazon
# Prime", "FanDuel Sports Network"), the Cup ("NBA Cup group play", "Cup
# game"), "Game 1" and "tonight". "Magic @ Suns 7:00 PM MST" still names its
# game (review 3, F4), and so do "Magic @ Suns 9 PM ET / 7 PM MST" (review 4,
# A5), "Magic at Suns at 9", "Tonight @ 9" and "9p ET" (review 5, A6).
# A regional network may carry its region ("NBC Sports Bay Area", "FanDuel
# Sports Network Southwest"); a national one never does, so the word after it
# is the city of the heading's team: "Tonight on ESPN Washington at Atlanta"
# lost Washington and its game (review 6, B8).
_REGIONAL_NETS = (r"(?:(?:fan\s*duel|bally|nbc|spectrum|monumental|altitude|marquee|root"
                  r"|space\s+city|chicago|gulf\s+coast|rip\s+city|yes)"
                  r"\s+(?:sports(?:\s*net(?:work)?)?"
                  r"|network|home\s+network|television(?:\s+network)?|tv)|fdsn\w*|nbcs\w*)")
_NET_REGIONS = (r"(?:\s+(?:bay\s+area|california|philadelphia|boston|washington|sun|florida"
                r"|southwest|southeast|north|ohio|detroit|wisconsin|indiana|oklahoma"
                r"|new\s+orleans|west|midwest|socal|northwest|plus))?")
_NATIONAL_NETS = (r"(?:msg(?:\d|\s+networks?)?|sportsnet|kjzz"
                  r"|espn(?:2|\s*\+|\s+plus)?|tnt|abc|nbc|nba\s*tv|(?:amazon\s+)?prime(?:\s+video)?"
                  r"|peacock|amazon|trutv|(?:nba\s+)?league\s+pass)")
_NETWORKS = r"(?:" + _REGIONAL_NETS + _NET_REGIONS + r"|" + _NATIONAL_NETS + r")"
# ...and with a region after any network: NBC's own stations brand by city
# ("on NBC Washington"), so a heading that reads as no game is read again with
# this (review 7, B11).
_NETWORKS_WIDE = r"(?:(?:" + _REGIONAL_NETS + r"|" + _NATIONAL_NETS + r")" + _NET_REGIONS + r")"
# A national network named on the line: only then is a heading read twice.
_NATIONAL_NAMED = re.compile(r"\b(?:nbc|abc|tnt|espn|peacock|prime|amazon|msg|nba\s*tv"
                             r"|league\s+pass|trutv)\b", re.IGNORECASE)


def _heading_noise(networks: str) -> re.Pattern:
    """What a heading carries besides its teams (see above), with `networks`
    for the networks it may name. Every piece of it starts a word or starts
    at an "@", so the lookahead in front skips every other place at once: a
    hundred pieces tried at each character of a 300-character line made a
    hostile paste of such lines cost seconds (review P8; the test pins the
    pattern with and without it to the same result)."""
    return re.compile(r"(?=@|\b\w)(?:" + _heading_noise_body(networks) + ")", re.IGNORECASE)


def _heading_noise_body(networks: str) -> str:
    """The pieces of _heading_noise, each starting a word or at an "@"."""
    return (
        r"\b\d{4}-\d{2}-\d{2}\b"
        r"|(?:\bat\s+|@\s*)?\b\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?!\w)"
        r"|(?:\bat\s+|@\s*)?\b\d{1,2}(?::\d{2})?\s*(?:" + _TZ_WORDS + r")\b|\b\d{1,2}:\d{2}\b"
        r"|\b(?:[01]\d|2[0-3])[0-5]\d\b"
        r"|\b\d{1,2}\s*-\s*\d{1,2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
        r"|\b" + _MONTH + r"\s*\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s*\d{4}\b)?"
        r"|\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH
        + r"|\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|sday|nesday|rsday|urday)?\b\.?"
        r"(?:\s+night\b)?"
        r"|(?:\bon\s+)?\b" + networks + r"(?!\w)"
        r"|\b(?:emirates\s+)?nba\s+cup(?:\s+(?:group\s+(?:play|stage)|game|night|knockout))?\b"
        r"|\bin[\s\-]season\s+tournament\b|\bcup\s+(?:night|game|knockout)\b"
        r"|\b(?:cup\s+)?(?:(?:east|west)(?:ern)?\s+)?group\s+(?:play|stage|[a-c])\b"
        r"|\bgame\s+\d{1,2}\b"
        r"|(?:\bat\s+|@\s*)\b\d{1,2}(?:[:.]\d{2})?\b|\b\d{1,2}(?:[:.]\d{2})?\s*[ap]\b"
        r"|\b\d{1,2}h\d{2}\b|\b(?:arizona|phoenix|local|tip|game|start)\s+time\b|\blocal\b"
        # "tip 9 ET", "tipoff 9 PM", "9 o'clock", "9ish", "21h", "7:00 Arizona",
        # a date as 11.5.26 (review 6, A9).
        r"|\btip(?:[\s\-]*off)?\b|\b\d{1,2}\s*o.?\s*clock\b|\b\d{1,2}(?::\d{2})?\s*ish\b"
        r"|\b(?:1[2-9]|2[0-3])h\b|\barizona\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"
        # "on Max", "on Bally", "on Suns Live", "| late game" (review 7, A9).
        r"|\bon\s+(?:hbo\s+)?max\b|\bon\s+bally\b|\bon\s+[a-z]+\s+live\b"
        r"|\b(?:late|early|night|main)\s+(?:game|slate|window)\b"
        r"|\b(?:" + _TZ_WORDS + r")(?:\s+time)?\b|\b(?:tonight|today|tomorrow)\b")


_HEADING_NOISE = _heading_noise(_NETWORKS)
_HEADING_NOISE_WIDE = _heading_noise(_NETWORKS_WIDE)
# Context a tout writes that is not a period ("2nd half of a back-to-back",
# "2nd half of B2B", "in the first meeting", "in the first 5 games", "in first
# place", "2nd half of the season"), set aside before a line is searched for
# one (review 3, F12).
_PERIOD_NOTE = re.compile(
    r"\b(?:1st|2nd|first|second)[\s\-]+(?:(?:half|leg|night|end|game)\s+)?"
    r"of\s+(?:(?:a|the|this)\s+)?(?:back[\s\-]*to[\s\-]*backs?|b2b)\b"
    r"|\b(?:1st|2nd|first|second)[\s\-]+half\s+of\s+(?:(?:the|this|last)\s+)?season\b"
    r"|\bin\s+(?:(?:the|their|his)\s+)?first\s+(?:\w+\s+)?"
    r"(?:meetings?|matchups?|games?|times?|starts?|outings?|weeks?|months?|rounds?|place)\b"
    r"|\bfirst\s+(?:\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:games|meetings|starts|outings|matchups|weeks)\b",
    re.IGNORECASE)
# A word that is only a period marker ("1Q", "Q1", "1st", "half"), left out of
# the name and the stat once the line is known to be a period prop.
_PERIOD_TOKEN = re.compile(r"^(?:[1-4](?:st|nd|rd|th)?-?[qh]s?|[qh]-?[1-4]|[1-4](?:st|nd|rd|th)"
                           r"|(?:[1-4](?:st|nd|rd|th)?|first|second|third|fourth)-"
                           r"(?:half|quarter|qtr)"
                           r"|first|second|third|fourth|half|halves|quarter|quarters|qtr|qrtr"
                           r"|qrt|hlf|hf|frame|period|q|1/2|\u00bd)$",
                           re.IGNORECASE)
# A period glued to the word after it ("#1QProps", "1Q_props", "1Qprops:",
# "#FirstQuarter", "FIRSTQUARTER:"), spaced before a period is looked for
# (review 7, A3).
_GLUED_PERIOD = re.compile(
    r"(?<![a-z0-9])([1-4](?:st|nd|rd|th)?[qh]|[qh][1-4])_?(?=(?:props?|picks?|plays?|bets?|locks?"
    r"|lines?|slips?|unders?|overs?)\b)|\b(first|second|third|fourth)_?(quarter|half|qtr)\b",
    re.IGNORECASE)


def _unglue_periods(text: str) -> str:
    return _GLUED_PERIOD.sub(lambda m: f"{m.group(1)} " if m.group(1)
                             else f"{m.group(2)} {m.group(3)}", str(text))
_POSSESSIVE = re.compile(r"(?:'s|\u2019s|'|\u2019)$", re.IGNORECASE)   # "BRUNSON'S" too
_EDGE_PUNCT = ",:;!?\"()[]{}|-\u2013\u2014"
_SIGN_PUNCT = ",:;!?\"()[]{}|"           # edges stripped before reading a sign ("-120")
_SYMBOL_CATEGORIES = frozenset({"So", "Sk", "Cs", "Co", "Cf", "Cn"})
_TYPED_MAX_TOKENS = 40
# Words at the EDGES of a typed name that are not part of it, set aside at its
# edges only: a word inside a name is never removed. None is a first name on
# the roster ("max" was: Max Strus, Max Christie; "MAX PLAY:" is a prefix).
_NAME_LEAD = frozenset({"take", "taking", "bet", "bets", "betting", "play", "playing",
                        "hammer", "smash", "ride", "back", "tail", "grab", "go", "pick",
                        "picks", "lock", "lean", "like", "love", "i", "im", "i'm", "the",
                        "on", "for", "with", "my", "best", "a", "potd", "lotd", "potn",
                        "bonus", "parlay", "nuke", "leg", "prop", "props", "nba", "give",
                        "me", "gimme",
                        # "NO DOUBT Karl-Anthony Towns over 11.5" (review 5)
                        "no", "doubt", "brainer", "question"})
_NAME_TAIL = frozenset({"for", "a", "an", "combined", "total", "of", "at", "with", "on",
                        "to", "record", "get", "gets", "score", "scores", "have", "has",
                        "hit", "hits", "the", "will", "be", "should", "goes", "go", "than",
                        "in", "his", "is", "tonight", "today", "tomorrow",
                        # "Will Josh Hart grab 9+ rebounds?" (review 6)
                        "records", "grab", "grabs", "make", "makes", "dish", "dishes",
                        "drop", "drops", "tally", "notch", "register", "collect"})
# ("post" was one too, and cut Quinten Post's surname: review 7, A10.)
# Before the number the words are cut back the same way, except the two that
# are also stat letters: "P + R + A" and "TO" (turnovers) keep their last word.
_PRE_TAIL = _NAME_TAIL - {"a", "to"}
# After the line, the words a tout closes a pick with ("tonight", "LOCK", "BOL").
_POST_TAIL = frozenset({"tonight", "today", "tomorrow", "lock", "locks", "bol", "lfg", "boom",
                        "hammer", "smash", "nuke", "easy", "cash", "money", "potd", "free"})
_TOTAL_WORDS = frozenset({"total", "team", "teams", "game"})
_NAME_PARTICLES = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "de", "da", "del", "di",
                             "du", "la", "le", "van", "von", "der", "dos", "el", "st"})
_STAT_LEAD = frozenset({"combined", "total", "of", "the", "in", "his"})
# A parenthesis word that changes the stat ("rebounds (offensive)", "threes
# (attempts)", "points (fantasy)"): it joins the label, so the line is priced
# on the stat written or not at all, never on the stat without it.
_STAT_QUALIFIERS = frozenset({"offensive", "defensive", "off", "def", "oreb", "dreb",
                              "attempted", "attempts", "attempt", "att", "atts", "fga", "fta",
                              "3pa", "fantasy", "fpts", "combo", "combos"})
# "off" and "def" change the stat beside stat words ("(off.)", "(def reb)")
# and beside most others ("(off glass)", "(off, not total)", review 5, A7);
# only before a word of an everyday phrase (_WEAK_GLOSS_NEXT) are they no
# qualifier: "(off the bench)", "(def a lock)", "(off night)".
_WEAK_QUALIFIERS = frozenset({"off", "def"})
_WEAK_GLOSS_NEXT = frozenset({"the", "a", "an", "bench", "ball", "rest", "night", "nights",
                              "day", "days", "week", "weeks", "season", "injury", "court",
                              "floor", "script", "his", "their", "my", "it", "lock", "play",
                              "this", "that", "to", "of", "and", "or"})
# Words a parenthesis may hold beside a qualifier without changing what it
# says: "(offensive only)", "(just attempts)" (review 4, A2).
_PAREN_FILLER = frozenset({"only", "just", "ones", "the"})
# After a qualifier, a word that makes the parenthesis about the matchup, not
# the stat: "(weak defensive team)" (review 3, R10).
_MATCHUP_NOUNS = frozenset({"team", "teams", "matchup", "matchups", "opponent", "opponents",
                            "opp", "unit", "rating", "ratings", "lineup", "lineups", "scheme",
                            "frontcourt", "backcourt", "ranking", "efficiency", "pace"})
# Before a stat in a parenthesis, an addition: "(+ assists)", "(and assists)",
# "(w/ assists)", "(including rebounds and assists)"; after it, the same:
# "(assists included)" (review 5, A7).
_CONNECT = frozenset({"+", "&", "and", "plus", "with", "w", "w/", "incl", "including",
                      "includes", "counting", "inc", "alongside"})
# Two words that add a stat as one does: "(along with assists)", "(together
# with assists)", "(combined with assists)" (review 6, A7).
_CONNECT_PAIRS = (["combined", "with"], ["along", "with"], ["together", "with"])
# Roots of a word that makes a stat another one: "rebounds, exclusively
# offensive", "rebounds (OREBs)", "(o-boards)", "points, all from the paint",
# "threes, as attempts", "points, in transition" were priced as the stat
# without it (review 6, A7). In a note's first words, or in a parenthesis,
# the line is then not read as that stat.
_QUAL_ROOTS = ("attempt", "paint", "fastbreak", "oreb", "dreb", "oboard", "dboard", "oglass",
               "dglass", "putback")
# Whole words that do it unless a matchup noun follows ("weak defensive team",
# "elite offensive rating", "great transition team" are the matchup; review 7,
# B16), and words that do it only as all of a short note ("threes, shots
# taken", but not "shots are falling").
_QUAL_FORMS = frozenset({"offensive", "defensive", "transition"})
_QUAL_SHORT = frozenset({"shots", "taken", "volume"})
# Whole words that do the same: "threes (tries)", "rebs, ORB", "assists
# (potential)", "rebound chances", "steals (deflections)" (review 7, A8).
_QUAL_WORDS = frozenset({"fga", "fta", "3pa", "tries", "potential",
                         "chances", "contested", "deflections", "deflection",
                         "secondary", "volume", "fts", "ftm", "orb", "drb", "orbs", "drbs"})
# After "from", a range that makes points another stat: "from three", "from
# deep", "from the line" (review 7, A8).
_FROM_RANGE = frozenset({"three", "threes", "3", "deep", "downtown", "distance", "line",
                         "beyond", "outside"})
_CONNECT_AFTER = frozenset({"included", "incl", "added"})
# In a parenthesis, a word that makes the stat another one ("(in the paint)",
# "(fastbreak)"): the line is not read as the stat outside it.
_PAREN_STAT_NOTES = frozenset({"paint", "fastbreak"})
# Beside another stat in a parenthesis, a word that may add it to the line's:
# "(assists too)", "(his assists count as well)" (review 5, A7).
_ADDING_WORDS = frozenset({"too", "also", "well", "count", "counts", "counted", "added", "adding",
                           "combined", "together", "both", "including", "included"})
# In a note right after the stat, a word that makes it another stat ("points
# in the paint", "points per game"): the line is not read as the stat before
# it. Later in a note it is context ("coming off a big game", review 4, B8).
_NOTE_STAT_WORDS = frozenset({"paint", "bench", "fastbreak", "per", "avg", "average", "ppg",
                              "rpg", "apg"})
# Words a note may open with before the word that changes the stat: "rebounds,
# only offensive", "threes, counting attempts", "points, mostly in the paint"
# were priced as the stat without it (review 5, A4, B5).
_NOTE_FILLER = _PAREN_FILLER | frozenset({"all", "strictly", "purely", "mostly", "mainly",
                                          "counting", "including", "includes", "incl", "on",
                                          "in", "of", "kind", "just", "only", "a", "an"})
# A result, not a pick ("Josh Hart o8.5 reb - cashed"), when one of these
# comes before the line, or first in the note after the stat (_read_line).
_RECAP_WORDS = frozenset({"cashed", "missed", "pushed", "graded", "won", "lost"})
# Results that are never a note's context, wherever they sit after the stat:
# "- easy, cashed", "- he cashed it", "Knicks won and it cashed" (review 5,
# B6). "won", "lost", "missed" and "pushed" also tell of the team ("Knicks
# lost last night", "Knicks pushed the pace"), so they count only first.
_RECAP_ALWAYS = frozenset({"cashed", "graded"})
# Words that make "cashes" a hope, not a result: "it cashes tonight", "this
# line cashes often". Without one, "... CASHES" is last night's (review 7,
# B5); a hope beside a result word is a pick's note: "(W tonight)" (B19).
_HOPE_WORDS = frozenset({"tonight", "today", "tomorrow", "will", "gonna", "should", "often",
                         "always", "usually", "every"})
# Results that make a note or a parenthesis a recap when they are all it says
# beside a subject: "- hit", "- easy dub", ": HIT", "(W)", "(push)", "- loss"
# (review 6, A8). Beside other words "hit" is a tout's "bet" ("hit this") and
# "W" a hope ("easy W tonight" is still read as a result, the safe side).
_RESULT_ALONE = frozenset({"w", "l", "hit", "push", "loss", "win", "dub", "winner", "cashed",
                           "won", "lost", "missed", "pushed", "graded"})
# Words a recap's note may open with before its result: "- that one lost",
# "- it pushed", "- barely missed", "= cashed", "LOCK cashed" (review 5, A10,
# B6).
_RECAP_SUBJECTS = frozenset({"it", "that", "this", "one", "barely", "just", "finally",
                             "easily", "easy", "tough", "close", "almost", "nearly", "sadly",
                             "unfortunately", "w", "l", "and", "but", "so"})
# What may follow a stat run that crossed a comma: "points, rebounds and
# assists combined", "Points, Rebounds, and Assists for sure" and "...
# assists - lets ride" are PRA. Anything else after it ("assists are live
# too") leaves the line unread (_stat_then_note).
_COMBO_CLOSERS = frozenset({"combined", "total", "together", "overall", "for", "sure", "imo",
                            "lets", "let's", "go", "baby", "ride", "riding", "it", "lock",
                            "locked", "in", "book", "tail", "fr", "boys", "love", "like", "i",
                            # "all day", "max play", "trust me", "let's get it", "no
                            # brainer", "my fav" (review 5, B11)
                            "all", "day", "max", "play", "ez", "trust", "me", "bet", "get",
                            "no", "brainer", "doubt", "fav", "fave", "favorite", "my", "units",
                            "unit", "x2", "dub", "w", "here", "we"})
# After two stats with only a comma between them, a word that makes them one
# combo: "rebounds, assists combined" (review 5, A9).
_COMBINERS = frozenset({"combined", "total", "together", "overall"})
# A name typed all in lower case is taken only when none of its words is one
# of these: "some words about the matchup" is a sentence, not a player.
_PLAIN_WORDS = frozenset({
    "a", "about", "all", "an", "and", "any", "are", "at", "be", "been", "but", "by", "did",
    "do", "does", "for", "from", "game", "had", "has", "have", "he", "her", "here", "his",
    "how", "i", "if", "in", "is", "it", "its", "just", "last", "like", "line", "lines",
    "love", "matchup", "my", "night", "no", "not", "of", "on", "or", "our", "parlay",
    "pick", "picks", "really", "she", "so", "some", "that", "the", "their", "there", "these",
    "they", "think", "this", "those", "to", "today", "tonight", "very", "was", "we", "were",
    "what", "when", "why", "with", "words", "yes", "you", "your"})
_OPP_MARKS = frozenset({"vs", "vs.", "v", "v.", "@", "at", "against", "versus"})
# After "no", a word that makes the two a note's words wherever they sit: "no
# doubt", "NO BRAINER", "no question" (review 4, A1; review 5, A2). They mean
# Yes, which never turns the pick. "no way" and "no chance" mean No
# (_NO_IDIOM_NEXT, review 6, A2).
_NO_NOTE_WORDS = frozenset({"doubt", "doubts", "brainer", "brainers", "question", "questions",
                            "cap", "fear", "worries", "worry", "problem", "problems", "sweat",
                            "hesitation", "issue", "issues", "lie", "joke", "one"})
# Before "no", a word that makes it a note's: "with no Towns", "there is no
# stopping him".
_NOTE_LEADS = frozenset({"with", "w/", "w", "has", "have", "had", "there's", "theres", "is",
                         "are", "zero"})
_HEADING_TAIL = frozenset({"picks", "pick", "props", "prop", "plays", "play", "bets", "bet",
                           "lines", "parlay", "leans", "lean"})
# What a tout calls the picks a section holds ("1Q props", "1Q locks", "1H
# slips"). Inside a board paste a section label needs a colon or one of these,
# since a square's own label ("1Q Points") and a board's period tabs ("1Q")
# hold neither (review 5, B1). Set aside at the end of a heading or a name
# line ("Knicks 1Q locks", "Jalen Brunson 1Q props").
_TOUT_NOUNS = _HEADING_TAIL | frozenset({
    "locks", "lock", "slips", "slip", "card", "cards", "targets", "target", "specials",
    "special", "potd", "sweats", "fliers", "flyers", "bangers", "sprinkles", "overs", "unders",
    "parlays", "tickets", "ticket", "spots", "selections"})
# The most a player's line can plausibly be, per stat. Only used before a name
# from the line above is lent to a line that has none: "Over 112.5 points"
# under a player's name is a team total, not his.
_TYPED_PLAUSIBLE = {"pts": 75, "reb": 30, "ast": 25, "fg3m": 15, "blk": 12, "stl": 10,
                    "tov": 12, "pr": 95, "pa": 95, "ra": 50, "pra": 110, "stocks": 15,
                    "fantasy": 130, "ftm": 30, "fta": 35, "fgm": 30, "fga": 50, "fg2m": 30,
                    "fg2a": 45, "fg3a": 25, "oreb": 15, "dreb": 25, "pf": 6.5}

# Stat words a person types, for typed_stat. Single letters count only inside
# a combination ("R+A"), never alone.
_TYPED_PART = {
    "points": "pts", "point": "pts", "pts": "pts", "pt": "pts", "p": "pts",
    "rebounds": "reb", "rebound": "reb", "rebs": "reb", "reb": "reb", "rebds": "reb",
    "boards": "reb", "board": "reb", "rb": "reb", "rbs": "reb", "r": "reb",
    "assists": "ast", "assist": "ast", "asts": "ast", "ast": "ast", "dimes": "ast",
    "a": "ast",
    "blocks": "blk", "block": "blk", "blks": "blk", "blk": "blk", "b": "blk",
    "steals": "stl", "steal": "stl", "stls": "stl", "stl": "stl", "s": "stl",
}
_TYPED_COMBO = {frozenset({"pts", "reb"}): "pr", frozenset({"pts", "ast"}): "pa",
                frozenset({"reb", "ast"}): "ra", frozenset({"pts", "reb", "ast"}): "pra",
                frozenset({"blk", "stl"}): "stocks"}
# Whole typed phrases, in label_key form, that STAT_MAP does not carry.
_TYPED_ALIASES = {
    "threes": "fg3m", "3s": "fg3m", "3pt": "fg3m", "3pointer": "fg3m", "3pointers": "fg3m",
    "threepointer": "fg3m", "threepointers": "fg3m", "madethrees": "fg3m",
    "threesmade": "fg3m", "3smade": "fg3m", "3ptm": "fg3m", "3pointfieldgoals": "fg3m",
    "3pointfieldgoalsmade": "fg3m", "pr": "pr", "pa": "pa", "ra": "ra",
    "stocks": "stocks", "fpts": "fantasy", "fantasypts": "fantasy", "tos": "tov",
    "threesattempted": "fg3a", "attemptedthrees": "fg3a", "threesattempts": "fg3a",
    "3sattempted": "fg3a", "3pa": "fg3a",
    # "rebounds (OREBs)" is the offensive line (review 6, A7).
    "orebs": "oreb", "drebs": "dreb",
    # Basketball-Reference's ORB and DRB (review 7, A8).
    "orb": "oreb", "orbs": "oreb", "drb": "dreb", "drbs": "dreb",
}
_TYPED_SPLIT = re.compile(r"\s*(?:\+|&|,|/|\band\b|\bplus\b)\s*")
# A joiner standing alone between the words of a stat ("Pts + R + A").
_JOINERS = frozenset({"+", "&", "/"})


def typed_stat(phrase) -> str | None:
    """The stat a typed phrase names, or None: 'Rebounds and Assists' is 'ra',
    'Points, Rebounds, and Assists' 'pra', 'R + A' 'ra', anything STAT_MAP
    reads itself. Every part must be a stat word, so 'first half points' is
    None and the paste path hands that label to classify_stat, which refuses
    it as a period prop."""
    words = str(phrase or "")[:MAX_LABEL_LEN].lower().split()
    while words and words[0].strip(_EDGE_PUNCT) in _STAT_LEAD:
        words.pop(0)
    raw = " ".join(words).strip(_EDGE_PUNCT + " .")
    if not raw:
        return None
    key = label_key(raw)
    hit = STAT_MAP.get(key) or _TYPED_ALIASES.get(key)
    if hit:
        return hit
    parts = [p for p in _TYPED_SPLIT.split(raw) if p]
    if len(parts) == 1:
        parts = raw.split()            # "Points Rebounds Assists": whole words only
        if len(parts) == 1:
            k = label_key(parts[0])
            return _TYPED_PART.get(k) if len(k) >= 4 else None
        if any(len(p) < 3 for p in parts) or any(p[-1] in ";:.!?" for p in parts[:-1]):
            return None                # "rebounds; assists": a break, as a comma is
    else:
        # A letter is a stat only where "+", "&" or "/" joins it ("R+A",
        # "P/R/A", "Pts + R + A", "Reb + A", which were priced as their first
        # stat alone: review 6, B2), or among capitals joined by "and" ("R and
        # A", "P, R and A", review 6, B14). "points, a lock" and "points - a lock" are no
        # Pts+Asts (review 5, B2), and "points + a lock" is not read
        # (_stat_then_note). Two stats with nothing but a comma between them
        # are a combo or a stat and a remark ("assists, rebounds no"), so
        # neither is read here (review 5, A9; _stat_then_note reads them when
        # "combined" follows).
        single = [len(label_key(p)) == 1 for p in parts]
        joined = (re.search(r"[+&/]", raw) is not None
                  and re.search(r",|\band\b|\bplus\b", raw) is None)
        capitals = all(single) and all(
            c.isupper() for c in re.findall(r"\b[A-Za-z]\b", str(phrase)[:MAX_LABEL_LEN]))
        if any(single) and not (joined or capitals):
            return None
        if len(parts) == 2 and re.search(r"[+&/]|\band\b|\bplus\b", raw) is None:
            return None
    if len(parts) > 3:
        return None
    keys = [_TYPED_PART.get(label_key(p)) for p in parts]
    if None in keys or len(set(keys)) != len(keys):
        return None
    return _TYPED_COMBO.get(frozenset(keys))


def _stat_piece(tok: str, last: bool) -> bool:
    """The last (or the first) "/"-part of a token is a stat word or a stat
    letter ("Pts", "R", "Reb/A")."""
    parts = [p for p in tok.strip(_EDGE_PUNCT + ".").split("/") if p]
    if not parts:
        return False
    k = label_key(parts[-1] if last else parts[0])
    return (len(k) == 1 and k in _TYPED_PART) or _stat_word(k)


def _join_stat_seps(s: str) -> str:
    """s with each separator between two stat pieces turned into "/" (see
    _STAT_SEP_CHARS). Linear: one pass over the words."""
    words = s.split()
    if len(words) < 2 or not any(set(w) & set(_STAT_SEP_CHARS) for w in words):
        return s
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if (len(w) == 1 and w in _STAT_SEP_CHARS and out and i + 1 < len(words)
                and out[-1][-1:].isalnum() and _stat_piece(out[-1], True)
                and _stat_piece(words[i + 1], False)):
            out[-1] += "/" + words[i + 1]
            i += 2
            continue
        if _GLUED_SEP.search(w):
            parts = _GLUED_SEP.split(w)
            if len(parts) > 1 and all(p and _stat_piece(p, False) for p in parts):
                w = "/".join(parts)
        out.append(w)
        i += 1
    return " ".join(out)


def _strip_symbols(text: str) -> str:
    """Emoji and other symbols out, a space each, so a bullet emoji or a trailing
    flame sticks to neither a name nor a stat. A keycap number goes whole."""
    if text.isascii():
        return text
    text = _KEYCAP_PERIOD.sub(r"\1", text)      # "1\ufe0f\u20e3Q props": the label's own digit
    text = _KEYCAP.sub(" ", text)
    return "".join(" " if (unicodedata.category(ch) in _SYMBOL_CATEGORIES
                           or 0xFE00 <= ord(ch) <= 0xFE0F or ch == "\u20e3") else ch
                   for ch in text)


def _clean(ln: str, tags: list | None = None) -> str:
    """A typed line without its decoration: symbols, markdown, then a bullet,
    a _TYPED_PREFIX, a _TYPED_LABEL and a handle (_HANDLE) at its start.
    Each handle or hashtag set aside is added to `tags`. Bounded: at most four
    passes, each anchored at the start."""
    s, found = _clean_tags(str(ln))
    if tags is not None:
        tags.extend(found)
    return s


@functools.lru_cache(maxsize=512)
def _clean_tags(ln: str) -> tuple[str, tuple[str, ...]]:
    """_clean's line and tags, kept: the heading, name-line, section-label and
    pick readers each clean the same line (review P8)."""
    tags: list[str] = []
    s = _MARKDOWN.sub(" ", _strip_symbols(ln)).strip()
    for _ in range(4):
        b = _TYPED_BULLET.sub("", s)
        p = _TYPED_PREFIX.match(b)
        if p is not None and p.group(0).lstrip()[:1] in ("@", "#"):
            tags.append(p.group(0).strip())
        t = _TYPED_LABEL.sub("", _TYPED_PREFIX.sub("", b)).strip()
        h = _HANDLE.match(t)
        if h and _team_word(h.group(1)) is None:
            tags.append(h.group(0).strip())
            t = t[h.end():].strip()
        if t == s:
            break
        s = t
    return s, tuple(tags)


def _team_word(text: str) -> str | None:
    """A team written in a heading or a parenthesis: 'Suns', 'Phoenix Suns',
    'Phoenix', 'PHX'. Codes count only as three capitals ('PHX', not 'Phx'),
    so a stray 'No' or 'NO' is not the Pelicans; '@ATL' is an opponent, never
    a team."""
    s = str(text or "").strip(_EDGE_PUNCT + " ")
    if not s or len(s) > 40 or "@" in s:
        return None
    key = teams._key(s)
    if len(key) < 3 or (len(key) == 3 and not s.isupper()):
        return None
    return teams.from_words(s)


# The words that name a matchup outright. "&", "and", "/", a dash or "x"
# between two teams may only group them ("Knicks & Nets"), so the schedule
# decides those (_heading_parts, typed_games).
_GAME_MARKS = frozenset({"@", "at", "vs", "vs.", "v", "v.", "versus"})


# A local station's call sign after "on" ("Magic @ Suns on KTVK"), in
# capitals: noise in a heading, unless it is a team's code (review 7, A9).
_CALL_SIGN = re.compile(r"\b[Oo]n\s+([KW][A-Z]{2,3})\b")


def _heading_parts(ln: str) -> tuple[tuple[str, ...], bool] | None:
    """(teams, loose) for a heading: (('PHX',), False) for a team heading
    ('Suns:', 'Phoenix Suns picks', 'Suns 5-3:'), (('ORL', 'PHX'), False) for
    a game heading ('Magic @ Suns', 'ORL vs PHX 9:00 PM ET', 'Thu 11/5: Magic
    at Suns'), with loose True when the joiner may only group the two
    ('ORL-PHX', 'Magic/Suns', 'Knicks & Nets'); None for anything else. A
    digit left once the time, date and record are set aside means a pick,
    never a heading. A period it names ('Knicks @ Cavs First Half', 'Magic
    @ Warriors 1H:') is set aside here and read by _section_period (review 5,
    A19, B4)."""
    got = _heading_read(ln, _HEADING_NOISE)
    if got is None and _NATIONAL_NAMED.search(str(ln)):
        got = _heading_read(ln, _HEADING_NOISE_WIDE)     # "Wizards at Hawks on NBC Washington"
    return got


@functools.lru_cache(maxsize=256)
def _noise_sub(noise: re.Pattern, s: str) -> str:
    """noise.sub(" ", s), kept: the heading reader and the name-line reader
    each set the same line's heading noise aside (review P8)."""
    return noise.sub(" ", s)


def _has_digit(s: str) -> bool:
    """Any digit in s, as str.isdigit counts them."""
    return any(map(str.isdigit, s))


def _heading_read(ln: str, noise: re.Pattern) -> tuple[tuple[str, ...], bool] | None:
    """_heading_parts with `noise` for what a heading carries."""
    s = _TYPED_PAREN.sub(" ", _clean(ln))
    s = _CALL_SIGN.sub(lambda m: m.group(0) if _team_word(m.group(1)) else " ", s)
    s = _drop_period(_noise_sub(noise, s))
    if _has_digit(s.replace("76ers", "")):
        return None
    words = s.replace(":", " ").split()
    # A tail word, or a separator the noise left behind ("Magic @ Suns /" once
    # "9 PM ET / 7 PM MST" is out, review 4, A5).
    while words and (words[-1].lower().strip(_EDGE_PUNCT + "/.,") in _TOUT_NOUNS
                     or not words[-1].strip(_EDGE_PUNCT + "/.,")):
        words.pop()
    while words and not words[0].strip(_EDGE_PUNCT + "/.,"):
        words.pop(0)
    s = " ".join(words).strip(_EDGE_PUNCT + " ")         # "Magic vs Suns -" once the time is out
    one = _team_word(s)
    if one:
        return (one,), False
    joint = _TYPED_GAME.search(s)
    parts = _TYPED_GAME.split(s)
    if joint is not None and len(parts) == 2:
        a, b = _team_or_no(parts[0]), _team_or_no(parts[1])
        if a and b and a != b:
            return (a, b), joint.group(0).strip().lower() not in _GAME_MARKS
    return None


def _typed_heading(ln: str) -> tuple[str, ...] | None:
    """The teams of a heading (_heading_parts), or None."""
    got = _heading_parts(ln)
    return got[0] if got else None


def _edge_team(tok: str) -> str | None:
    """A team code or nickname written at the edge of a typed name ('PHX',
    "Suns'", or ESPN's two-letter 'GS', 'NY', 'SA', 'NO'). Never a city
    (Orlando Robinson and P.J. Washington keep theirs). Codes count only in
    capitals, and 'NO' is also read as a No until the roster decides
    (alt_pick)."""
    core = _POSSESSIVE.sub("", tok.strip(_EDGE_PUNCT))
    if core.isupper() and len(core) in (2, 3):
        return teams.canonical(core)
    return teams.from_words(core, cities=False) if len(core) >= 4 else None


def _capitalised(tok: str) -> bool:
    t = tok.strip(_EDGE_PUNCT)
    return bool(t) and (t[0].isupper() or t.lower().rstrip(".") in _NAME_PARTICLES)


# What the words around a typed pick turned out to be (_typed_name's kind).
NAME_OK, NAME_NONE, NAME_BAD, NAME_TEAM = "name", "none", "bad", "team"


def _typed_name(tokens, strict: bool = False, question: bool = False
                ) -> tuple[str | None, str | None, str]:
    """(name, team, kind) from the words around a typed pick. Lead-in and
    trailing words are set aside at the edges, and a team code or nickname
    at an edge is read as his team. kind is NAME_OK, NAME_NONE (no words: a
    name on a line above may name it), NAME_TEAM (a team: a team total, not
    a player's line) or NAME_BAD (words that are not a full name, such as a
    surname alone). A name is written in capitals ('Miles Bridges', 'ZACH
    EDEY') or, unless `strict` (a name alone on its line, with no pick to
    vouch for it), all in lower case with no everyday word in it ('miles
    bridges'). Never a mix, which keeps 'Last night we hit' from reading as a
    player. `question` ('Will Miles Bridges record 7+ ...?') sets a leading
    'Will' aside when two words are left after it, so Will Riley keeps his
    name."""
    toks = [t for t in tokens if t.strip(_EDGE_PUNCT)]
    while toks and toks[0].strip(_EDGE_PUNCT).lower() in _NAME_LEAD:
        toks.pop(0)
    while toks and toks[-1].strip(_EDGE_PUNCT).lower() in _NAME_TAIL:
        toks.pop()
    if question and len(toks) >= 3 and toks[0].strip(_EDGE_PUNCT).lower() == "will":
        toks.pop(0)
    if not toks:
        return None, None, NAME_NONE
    whole = _team_word(" ".join(toks))
    if whole is not None:
        return None, whole, NAME_TEAM
    team = None
    for edge in (0, -1):
        if len(toks) > 2:
            code = _edge_team(toks[edge])
            if code:
                team = code
                toks.pop(edge)
    toks[-1] = _POSSESSIVE.sub("", toks[-1].rstrip(_EDGE_PUNCT))
    name = " ".join(toks).strip(_EDGE_PUNCT + " ")
    parts = name.split()
    if name.isupper() and all(teams.from_words(p.strip(_EDGE_PUNCT)) for p in parts):
        return None, team, NAME_TEAM                 # "LAL BOS": a matchup in codes
    lower = (not strict and name == name.lower()
             and not any(t.strip(_EDGE_PUNCT) in _PLAIN_WORDS for t in parts))
    # Capitals throughout are a tout's style ('ZACH EDEY'), which the guard
    # against team codes in _person_like would otherwise refuse.
    looks = name.title() if name.isupper() else name
    if (not _person_like(looks) or len(parts) > 5
            or not (all(_capitalised(t) for t in parts) or lower)
            or _team_word(name) is not None or typed_stat(name) is not None):
        return None, team, NAME_BAD
    return name, team, NAME_OK


def _strip_opp(tokens: list[str]) -> tuple[list[str], str | None]:
    """Tokens without a 'vs Magic' / '@ ORL' / 'against the Suns' clause, and
    the opponent it names. Looks at the last two such marks only: the work per
    line stays a few lookups however a hostile line repeats them."""
    low = [t.strip(_EDGE_PUNCT).lower() if t != "@" else "@" for t in tokens]
    seen = 0
    for i in range(len(tokens) - 1, -1, -1):
        if low[i] not in _OPP_MARKS:
            continue
        seen += 1
        j = i + 1 + (1 if i + 1 < len(tokens) and low[i + 1] == "the" else 0)
        if j < len(tokens) and tokens[j].strip(_EDGE_PUNCT + ".") == "NO":
            return tokens[:i] + tokens[j + 1:], "NOP"     # ESPN's "vs NO" (review 4, A4)
        for span in (3, 2, 1):
            code = _team_word(" ".join(tokens[j:j + span])) if j + span <= len(tokens) else None
            if code:
                return tokens[:i] + tokens[j + span:], code
        if seen >= 2:
            break
    return tokens, None


def _team_or_no(text: str) -> str | None:
    """_team_word, or ESPN's "NO" (New Orleans) where only a team can stand:
    after "vs" or "@", or as one side of a matchup ("MIN @ NO")."""
    return "NOP" if str(text or "").strip(_EDGE_PUNCT + ". ") == "NO" else _team_word(text)


def _stat_digit(bare: list[str], i: int) -> bool:
    """A 2 or 3 that begins a stat ('3 pointers made', '3 pt made'), not a line."""
    if bare[i] not in ("2", "3") or i + 1 >= len(bare):
        return False
    nxt = bare[i + 1]
    return nxt.startswith(("pointer", "pt", "point-", "pm")) or nxt == "point"


def _glued_stat(word: str) -> bool:
    return typed_stat(word) is not None or label_key(word) in STAT_MAP


def _typed_tokens(s: str) -> list[str]:
    """Whitespace tokens, with a glued side split off ('o6.5'), a glued stat
    split off ('25.5pts', '9+reb', 'o25.5pts.'), a glued '@' split off
    ('@ATL'), a sentence's full stop taken off a number ('25.5.'), and dates,
    American odds and unit sizes dropped."""
    out: list[str] = []
    for t in s.split():
        core = t.strip(_EDGE_PUNCT)
        signed = t.strip(_SIGN_PUNCT)
        plain = core[:-1] if core.endswith(".") and len(core) > 1 else core
        m = _TYPED_GLUED.match(plain)
        if m:
            out += [m.group(1), m.group(2)]
            continue
        g = _TYPED_GLUED_STAT.match(plain) or _TYPED_PLUS_STAT.match(plain)
        if g and _glued_stat(g.groups()[-1]):
            out += [x for x in g.groups() if x]
        elif core.endswith(".") and _TYPED_NUM.match(core[:-1]):
            out.append(core[:-1])
        elif core.startswith("@") and len(core) > 1:
            out += ["@", core[1:]]
        elif (_TYPED_DATE.match(core) or _TYPED_ODDS.match(signed)
              or _TYPED_UNITS.match(core)):
            continue
        else:
            out.append(t)
    return out


def _kept(tok: str, period) -> bool:
    """A word that stays for the name and the stat: not bare punctuation
    ('-', '!!'), and not a period marker once the line is a period prop."""
    core = tok.strip(_EDGE_PUNCT + ".")
    return bool(core) and not (period is not None and _PERIOD_TOKEN.match(core))


_DASHES = frozenset("-\u2013\u2014")


def _breaks(tokens: list[str]) -> list[str]:
    """Tokens with a lone dash folded into a comma on the word before it, so
    "rebounds - assists too" breaks where "rebounds, assists too" does
    (review 4, B1)."""
    out: list[str] = []
    for t in tokens:
        if t and set(t) <= _DASHES:
            if out and not out[-1].endswith((",", ";")):
                out[-1] += ","
            continue
        out.append(t)
    return out


def _name_then_stat(words: list[str], question: bool):
    """(stat, name, team, kind, phrase) for 'NAME STAT' words, the stat being
    the last one to six of them, or None. A stat whose words before it are
    not a full name comes back as NAME_BAD, with no name."""
    first = None
    for j in range(min(len(words), 6), 0, -1):
        phrase = " ".join(words[-j:])
        st_j = typed_stat(phrase)
        if st_j is None:
            continue
        nm, nt, kd = _typed_name(words[:-j], question=question)
        if kd in (NAME_OK, NAME_NONE, NAME_TEAM):
            return st_j, nm, nt, kd, phrase
        first = first or (st_j, None, nt, NAME_BAD, phrase)
    return first


# _read_typed's answer for a line shaped like a pick (a side or a "7+") that
# holds more than one number: counted, never guessed.
_UNREAD = "unread"
# ...and for a pick that reports its result ("o25.5 pts - cashed"): counted
# as typed_recap, never filed (review 4, A7, B7). _RECAP_NAMED when the recap
# names a player: it ends the name line above, as a pick would (review 5,
# A11, B7).
_RECAP = "recap"
_RECAP_NAMED = "recap named"
# ...and for a pick whose side cannot be told: a No that may be the answer
# or a note's word (_market_answers), counted as typed_unclear (review 6), and
# _UNCLEAR_NAMED when the line names a player: it ends the name line above.
_UNCLEAR = "unclear"
_UNCLEAR_NAMED = "unclear named"


def _flip(side: str) -> str:
    return "less" if side == "more" else "more"


def _pickish(tok: str) -> bool:
    """A word that may sit between a line and its answer without being a
    note: the stat, a joiner, a closer, an opponent or a mark ('26+ points?
    No', '7+ R+A: No', 'over 25.5 pts vs MIA No')."""
    core = tok.strip(_EDGE_PUNCT + ".?")
    low = core.lower()
    return bool(not low or low in _CONNECT or low in _STAT_LEAD or low in _POST_TAIL
                or low in _OPP_MARKS or _stat_word(label_key(low))
                or typed_stat(low) is not None or _team_word(core) is not None)


def _answer_spans(bare: list[str]) -> list[tuple[int, int, bool, str]]:
    """(first, last, value, kind) for each Yes or No among the words: kind
    'plain' ("No", "Nope", "Yeah"), 'idiom' for a No in more words than one
    ("No way", "No chance", "Not a chance", "Absolutely not"), or 'note' for
    a "no" that a word after it makes a note's ("no doubt", "no brainer").
    Linear: one pass over the words."""
    out: list[tuple[int, int, bool, str]] = []
    i = 0
    while i < len(bare):
        w = bare[i].rstrip(".")
        nxt = bare[i + 1].rstrip(".") if i + 1 < len(bare) else ""
        if w == "no" and nxt in _NO_NOTE_WORDS:
            out.append((i, i + 1, True, "note"))
            i += 2
        elif w == "no" and nxt in _NO_IDIOM_NEXT:
            out.append((i, i + 1, False, "idiom"))
            i += 2
        elif w == "not" and [x.rstrip(".") for x in bare[i + 1:i + 3]] == ["a", "chance"]:
            out.append((i, i + 2, False, "idiom"))
            i += 3
        elif w == "not" and i > 0 and bare[i - 1] in ("absolutely", "definitely"):
            out.append((i - 1, i, False, "idiom"))
            i += 1
        elif w in ("yeah", "yea", "yep", "yup") and nxt in ("no", "nah", "nope") and (
                bare[i + 2].rstrip(".") if i + 2 < len(bare) else "") not in _NO_NOTE_WORDS:
            # "Yeah, no": a No, and "Yeah no way" one too (review 7, A5).
            more = i + 2 < len(bare) and bare[i + 2].rstrip(".") in _NO_IDIOM_NEXT
            out.append((i, i + 1 + more, False, "idiom"))
            i += 2 + more
        else:
            v = _answer_value(w)
            if v is not None and not (w == "no" and bare[i].endswith(".") and nxt.isdigit()):
                out.append((i, i, v, "plain"))           # never "No. 1 option" (review 7, B4)
            i += 1
    return out


def _after_answer(toks: list[str], bare: list[str], e: int, val: bool,
                  used=frozenset()) -> str:
    """What follows a Yes or No whose last word is token e: 'alone' (nothing,
    or closers only: "No", "No lol", "No - fade", "NO 2u", "No at 38c"),
    'two' (the other answer, last on the line: "Yes No", a copied market's
    two buttons), 'clause' (a mark, a clause or a closer with more words
    after it: "No, he's cold", "No - Towns out", "No he's cold", "no for
    real") or 'word' (a note's own word: "no Towns", "NO MATTER WHAT", "no
    defense")."""
    rest = [j for j in range(e + 1, len(bare)) if j not in used and bare[j].rstrip(".")]
    if not rest:
        return "alone"
    words = [bare[j].rstrip(".") for j in rest]
    if len(words) == 1 and _answer_value(words[0]) is not None:
        return "two" if _answer_value(words[0]) != val else "alone"
    if all(w in _ANSWER_CLOSERS or _AMOUNT.fullmatch(w) for w in words):
        return "alone"
    marked = toks[e][-1:] in ",;:!.)" or any(not bare[j].rstrip(".") for j in range(e + 1, rest[0]))
    if marked or words[0] in _ANSWER_CLOSERS or words[0] in _CLAUSE_STARTS:
        return "clause"
    return "word"


# In the clause after a No, words that may make it a Yes: a second negation
# ("no, there's no way he misses", "no you can't fade him") or a word of the
# Yes ("no it is a lock", "no your boy is cooking", review 7, B6).
_TURNS = frozenset({
    "no", "not", "never", "nobody", "nothing", "none", "cant", "can't", "wont", "won't", "dont",
    "don't", "doesnt", "doesn't", "isnt", "isn't", "aint", "ain't", "wasnt", "wasn't", "without",
    "lock", "locks", "locked", "free", "money", "easy", "ez", "cooking", "cook", "cooks", "hammer",
    "smash", "love", "like", "sure", "definitely", "absolutely", "guaranteed", "gets", "get",
    "bank", "cash", "lfg", "gimme", "hit", "hits", "doubt", "fade", "fading"})


# Closers that speak for the side written ("lock it", "hammer", "easy money"):
# after a No beside a written Over or Under they make it unclear.
_YES_CLOSERS = frozenset({
    "lock", "locks", "locked", "max", "play", "bet", "easy", "ez", "money", "cash", "hammer",
    "smash", "nuke", "tail", "bol", "lfg", "potd", "free", "it", "in", "this", "lets", "let's",
    "ride", "go", "absolutely", "definitely", "obviously", "duh", "ofc", "easily", "boys", "baby"})


def _clause_turns(bare: list[str], e: int, used=frozenset()) -> bool:
    """The clause after the answer at token e may make it a Yes (_TURNS). A
    clause that is another answer ("No. Yes on Hart though") is not."""
    rest = [bare[j].rstrip(".") for j in range(e + 1, min(len(bare), e + 10))
            if j not in used and bare[j].rstrip(".")]
    if not rest or _answer_value(rest[0]) is not None:
        return False
    return any(w in _TURNS for w in rest)


def _stat_end(toks: list[str], k: int) -> int:
    """The last token of the stat named right after the number at token k
    ("26+ 3-PT Made? No", "11+ R + A? No"), or k when none is. Six tokens
    at most are tried."""
    end = k
    for j in range(k + 1, min(len(toks), k + 7)):
        if typed_stat(" ".join(t.rstrip("?") for t in toks[k + 1:j + 1])) is not None:
            end = j
    return end


def _result_only(words) -> bool:
    """Words that say a result and nothing else, beside a subject: "hit",
    "it hit", "easy dub", "W", "that one lost" (review 6, A8). "hit" with a
    word after it is a tout's "bet": "hit this", "hit it"."""
    ws = [w for w in (str(t).strip(_EDGE_PUNCT + ".=<>~").lower() for t in words)
          if w and w not in _POST_TAIL]
    if any(w in ("hit", "push") for w in ws[:-1]):
        return False                                     # "hit it", "push it": a bet
    core = [w for w in ws if w not in _RECAP_SUBJECTS or w in _RESULT_ALONE]
    return bool(core) and all(w in _RESULT_ALONE for w in core)


def _missed_context(words) -> bool:
    """'missed' about the player's game, not the pick's: '(missed last game)',
    '- missed shootaround', '(just missed last time)' (review 7, B19)."""
    ws = [str(t).strip(_EDGE_PUNCT + ".").lower() for t in words]
    return bool(ws) and ws[0] == "missed" and bool(set(ws[1:3]) & {
        "game", "games", "shootaround", "practice", "time", "last", "workout", "workouts"})


def _paren_recap(low: str) -> bool:
    """A parenthesis that tells a result: "(cashed)", "(won)", "(cashed it)",
    "(it cashed)", "(won easily)", "(W)", "(hit)", "(push)", "(cashes)"
    (review 5, A10; review 6, A8; review 7, B5). One with a number is context
    ("(won 3 straight)"), and so is one with a hope ("(W tonight)") or about
    the player's games ("(missed last game)"), and "(hit it)" is a bet
    (review 7, B19)."""
    words = [w for w in (x.strip(_EDGE_PUNCT + ".!") for x in low.split()) if w]
    if not words or len(words) > 4 or _has_digit(low):
        return False
    if any(w in _RECAP_ALWAYS for w in words):
        return True
    if any(w in _HOPE_WORDS for w in words):
        return False
    if "cashes" in words:
        return True
    core = [w for w in words if w not in _RECAP_SUBJECTS or w in _RESULT_ALONE]
    return bool(core) and ((core[0] in _RECAP_WORDS and not _missed_context(core))
                           or _result_only(words))


def _paren_answer(inner: str) -> tuple[bool, str, str] | None:
    """(value, form, kind) for a parenthesis that answers the market. "(No)",
    "(No.)", "(No, fade)", "(No 2u)", "(No for me)", "(answer: No)" and "(I
    say no)" are 'alone'; "(no, not a typo)" and "(No - Towns out)" a
    'clause', which may be the answer or a note on the number. None for a
    gloss: "(no doubt)", "(no assists line today)", "(Yes/No)" (review 6,
    A6, B6)."""
    ptoks = inner.split()
    if not ptoks or len(ptoks) > 8:
        return None
    pbare = [t.strip(_EDGE_PUNCT).lower() for t in ptoks]
    spans = _answer_spans(pbare)
    if len(spans) != 1 or spans[0][3] == "note":
        return None
    s, e, val, _kind = spans[0]
    if s == 0:
        form = _after_answer(ptoks, pbare, e, val)
        if form == "word":
            return None
        return val, ("alone" if form == "alone" else "clause"), _kind
    if e == len(pbare) - 1 and all(w.rstrip(".") in _ANSWER_LEADS for w in pbare[:s]):
        return val, "alone", _kind
    return None


def _market_answers(toks: list[str], bare: list[str], k: int, used: set, end: int,
                    explicit: bool):
    """The visitor's answers to a market on a line whose number is token k:
    (answers, no_at), answers a set of True (Yes) and False (No), no_at the
    all-caps NOs whose reading the roster settles (True when after the
    number); the tokens they take are added to `used`. _UNCLEAR when a No may
    be the answer or a note's word; None for a copied market's two buttons
    ("7+ R+A Yes No"). A Yes never turns the pick, so a No is what is
    weighed: the answer where it plainly stands as one (after the pick's own
    question mark, first after the line, last on it, before the number of a
    "7+" line), the note's where a note's word stands ("no Towns", "no
    doubt", "with no Towns", the answer to the note's own question), and
    unclear anywhere else, above all beside a written Over or Under, where
    "No, he's cold" and "No, I'm not kidding" argue opposite sides (review 6,
    A2, A6, B1, B3, B5, B6)."""
    spans = _answer_spans(bare)
    answers: set = set()
    no_at: list[bool] = []
    if not spans and not any("?" in t for t in toks):
        return answers, no_at
    if len(spans) > 4:
        return None                # no market line says Yes or No five times
    qmarks = [i for i, t in enumerate(toks) if "?" in t]
    stat_end = _stat_end(toks, k)

    def plain(j: int) -> bool:
        # A word between the line and its answer that is no note's: the stat
        # ("3-PT Made", "R + A"), a joiner, a closer, a mark, a team, the
        # opponent, or the game ("on Saturday", "in his next game").
        w = bare[j].rstrip(".")
        return (j in used or not w or k < j <= stat_end or _pickish(toks[j])
                or w in _QUESTION_CONTEXT)

    # The question mark that closes the pick ("26+ points? No", "26+ 3-PT
    # Made? No"): plain words between it and the line, and after a mark only
    # the options it offers ("7+ R+A: Yes or No? No"). Any other is a
    # question the note asks ("is Towns playing? No", review 5, A2).
    pick_q = None
    if qmarks:
        q, ok, marked = qmarks[0], True, False
        for j in range(k + 1, q + 1):
            w = bare[j].rstrip(".")
            option = _answer_value(w) is not None or w == "or"
            if not w:
                marked = marked or j < q
                continue
            if (marked and not option) or not (plain(j) or option):
                ok = False
                break
            marked = marked or (j < q and toks[j].rstrip("?")[-1:] in ",;:.!")
        pick_q = q if ok else None
    # The options a question offers, never its answer: "Yes or No", and a Yes
    # or No inside the pick's question when an answer follows it ("7+ R+A -
    # yes? no.").
    options = {j for o in range(1, len(bare) - 1) if bare[o] == "or"
               and _answer_value(bare[o - 1].rstrip(".")) is not None
               and _answer_value(bare[o + 1].rstrip(".")) is not None for j in (o - 1, o, o + 1)}
    if pick_q is not None and any(s > pick_q for s, _e, _v, _k in spans):
        options |= {s for s, _e, _v, _k in spans if s <= pick_q}
    used |= {j for j in options if bare[j] == "or"}
    taken = False
    noted: set = set()             # the words of a note's Yes ("no doubt") and the opponent's NO
    for s, e, val, kind in spans:
        if s in used:
            continue                                 # "no more than 23": the line's own words
        caps = kind == "plain" and toks[s].strip(_EDGE_PUNCT + ".") == "NO"
        if caps and s > 0 and bare[s - 1] in _OPP_MARKS:
            noted.add(s)
            continue                                 # "vs NO": the opponent (_strip_opp)
        if s in options:
            used.add(s)
            continue                                 # "Yes or No? No": the question's options
        first_val = next(iter(answers), False if no_at else None)
        if (taken and kind != "note" and first_val is not None and val != first_val
                and not (s > 0 and bare[s - 1] in _NOTE_LEADS)
                and _after_answer(toks, bare, e, val, used) != "word"
                and not (e + 1 < len(bare) and bare[e + 1] in ("on", "for"))):
            # "No... actually yes", "Yes. No wait, no": a corrected answer,
            # which may be read either way (review 7, B8). "No. Yes on Hart
            # though" answers another pick, and in "Yes. No Towns, more
            # shots" the No starts a note.
            return _UNCLEAR
        if kind == "note" or taken:
            # "no doubt", and once an answer is given, the note's ("7+ R+A No.
            # Yes on Hart though").
            noted |= set(range(s, e + 1)) if kind == "note" else set()
            continue
        lead_note = s > 0 and bare[s - 1] in _NOTE_LEADS     # "with no Towns", "there's no way"
        span = set(range(s, e + 1))
        if lead_note and e < k:
            continue
        if e < k:
            # Before the number: "Miles Bridges No 7+ R+A", "No - Miles
            # Bridges 7+ R+A", "Miles Bridges: No on 7+ R+A", ESPN's "NO Zion
            # Williamson ...". Beside a written side, a No opening the line is
            # a reply or a pass ("No, I like Jalen Brunson over 25.5 points",
            # "No play: ...", review 6, B5): not read.
            on = e + 2 == k and bare[e + 1] == "on"
            if val:
                answers.add(True)
            elif caps and not toks[e].endswith(","):
                no_at.append(False)
            elif explicit:
                return _UNCLEAR
            elif (toks[e][-1:] in ":|" or e + 1 == k or on
                  or (e + 1 < k and set(toks[e + 1]) <= _DASHES)):
                answers.add(False)
                span |= {e + 1} if on else set()
            else:
                return _UNCLEAR
            used |= span
            taken = True
            continue
        form = _after_answer(toks, bare, e, val, used)
        if form == "two":
            # "7+ R+A Yes No": a copied market's two buttons, counted, never
            # dropped without a word (review 7, A5).
            return _UNCLEAR
        q_before = max((q for q in qmarks if q < s), default=None)
        if q_before is not None and q_before != pick_q:
            slot = "note"                            # "is Towns playing? No"
        elif q_before is not None:
            between = [j for j in range(q_before + 1, s) if j not in used and bare[j].rstrip(".")]
            slot = "asked" if len(between) <= 2 else "later"     # "26+ points? I say no"
        elif all(plain(j) for j in range(k + 1, s)):
            slot = "first"
        else:
            slot = "later"
        if lead_note and not (slot == "asked" and kind == "idiom"):
            # The note's ("with no Towns", "Knicks have no defense"), except
            # "There's no way" right after the pick's own question (review 7,
            # A6).
            noted |= span
            continue
        if kind == "idiom" and explicit and slot != "asked":
            # "Josh Hart under 8.5 rebounds no shot": "no shot he gets there"
            # or "no shot he misses", beside a side already written (review 7,
            # B7).
            return _UNCLEAR
        if val:
            if slot in ("asked", "first") or form == "alone":
                answers.add(True)
                used |= span
                taken = True
            continue
        if caps and slot != "asked" and e == end:
            # ESPN's NO last on the line: New Orleans, or a No, settled by the
            # roster (alt_pick).
            no_at.append(True)
            used |= span
            taken = True
            continue
        if slot == "note":
            if explicit:
                continue                             # the note's answer: the written side stands
            return _UNCLEAR
        if kind == "idiom" and form != "alone":
            return _UNCLEAR                          # "No way he misses": which side?
        if form == "word":
            if slot == "asked":
                return _UNCLEAR                      # "26+ points? No Towns though"
            continue                                 # "no Towns", "NO MATTER WHAT": the note's
        if (form == "clause" and (slot == "later" or (slot == "first" and explicit))) or (
                slot == "later" and explicit):
            # "over 25.5 points, no but seriously", "Over 8.5 rebounds No, he's
            # been cold": the answer, or a note on the Over, cannot be told.
            return _UNCLEAR
        if form == "clause" and _clause_turns(bare, e, used):
            # "26+ points no there's no way he misses", "9+ rebounds no it is a
            # lock": a No that opens a Yes (review 7, B6).
            return _UNCLEAR
        if explicit and form == "alone" and any(
                bare[j].rstrip(".") in _YES_CLOSERS
                for j in range(e + 1, len(bare)) if j not in used):
            # "over 25.5 points nope, lock it": a Yes's closer beside a written
            # side (review 7).
            return _UNCLEAR
        answers.add(False)
        used |= span
        taken = True
    if pick_q is not None and pick_q > k and not taken:
        # After the pick's own question mark, words that are no answer the
        # reader knows: "? Fat chance", "? Never", "? Doubt it" were read as
        # the implied side (review 7, A6). Another way to say No is read; any
        # other words leave the side unclear.
        idx = [j for j in range(pick_q + 1, len(bare)) if j not in used and j not in noted
               and bare[j].rstrip(".") and not _AMOUNT.fullmatch(bare[j].rstrip("."))
               and not _pickish(toks[j])
               and (bare[j].rstrip(".") not in _ANSWER_CLOSERS
                    or bare[j].rstrip(".") in ("fade", "fading", "pass"))]
        said = tuple(bare[j].rstrip(".") for j in idx)
        if said in _ASKED_NO or said in _ASKED_YES:
            answers.add(said in _ASKED_YES)
            used |= set(idx)
        elif said:
            return _UNCLEAR
    return answers, no_at


# The single stats a combination is made of, for a parenthesis that names
# one (_named_stats).
_STAT_PARTS = {"pr": {"pts", "reb"}, "pa": {"pts", "ast"}, "ra": {"reb", "ast"},
               "pra": {"pts", "reb", "ast"}, "stocks": {"blk", "stl"}}


def _named_stats(text: str) -> set:
    """The single stats the words of `text` name ('his assists are up' is
    {'ast'})."""
    out: set = set()
    for w in str(text).split():
        k = label_key(w.strip(_EDGE_PUNCT + "."))
        st = (STAT_MAP.get(k) or _TYPED_ALIASES.get(k) or _TYPED_PART.get(k)) if _stat_word(k) \
            else None
        if st:
            out |= _STAT_PARTS.get(st, {st})
    return out


def _stat_word(key: str) -> bool:
    """A label_key naming a stat by itself ('rebounds', 'pts', '3pt', '3s'),
    never a single letter or 'to'."""
    if key in _TYPED_ALIASES:
        return key not in ("pr", "pa", "ra")
    return len(key) >= 3 and (key in STAT_MAP or key in _TYPED_PART)


def _strong_qual(keys: list[str], i: int) -> bool:
    """keys[i], a label_key word, changes the stat beside it wherever it sits
    among a few words: a _QUAL_ROOTS word, 'o' or 'd' before 'boards' or
    'glass', 'offense' or 'defense' after 'on' or beside 'only' or 'just' (or
    alone: 'on defense', 'defense only', but never 'no defense on the other
    side'), FGA, FTA or 3PA (review 6, A7), _QUAL_WORDS, _QUAL_FORMS and
    _QUAL_SHORT (review 7, A8, B16). Never
    'off' or 'def' alone, which are an everyday word as often ('coming off a
    big game', 'bad def tonight')."""
    k = keys[i]
    if any(r in k for r in _QUAL_ROOTS) or k in _QUAL_WORDS:
        return True
    nxt = keys[i + 1] if i + 1 < len(keys) else ""
    prev = keys[i - 1] if i > 0 else ""
    if k in _QUAL_FORMS:
        return nxt not in _MATCHUP_NOUNS
    if k in _QUAL_SHORT:
        return len([x for x in keys if x]) <= 2
    if k == "rim":
        # "points, at the rim" (the paint), never "he attacks the rim".
        return keys[max(0, i - 2):i] == ["at", "the"] and len([x for x in keys if x]) <= 3
    if k == "from" and (nxt in _FROM_RANGE or (nxt == "the" and keys[i + 2:i + 3] == ["line"])):
        return True                                      # "points, from three", "(from 3)"
    if k in ("o", "d") and nxt in ("board", "boards", "glass", "reb", "rebs", "rebounds"):
        return True
    return k in ("offense", "defense") and (
        len(keys) == 1 or prev == "on" or bool({prev, nxt} & {"only", "just"}))


def _names_stat(words) -> bool:
    """Words after a stat that name or change one. A stat anywhere
    ('rebounds', 'R+A'): 'points, rebounds and assists combined' is PRA. A
    word that changes the stat in the first four (_strong_qual: 'rebounds,
    but only offensive', 'points, all from the paint'), or a qualifier or a
    note word first ('offensive', 'per game'): 'rebounds offensive' is not
    Rebounds. Later in a note those are context: 'coming off a big game',
    'bad def tonight' (review 4, B8)."""
    keys = [label_key(str(w).strip(_EDGE_PUNCT + ".")) for w in words]
    for w, k in zip(words, keys):
        if _stat_word(k) or (len(k) >= 3 and typed_stat(w) is not None):
            return True
    if any(_strong_qual(keys, i) for i in range(min(4, len(keys)))):
        return True
    rest = keys
    for _ in range(4):                   # "only offensive", "on the offensive glass"
        if rest and (not rest[0] or rest[0] in _NOTE_FILLER):
            rest = rest[1:]
    if rest and (rest[0] in _STAT_QUALIFIERS or rest[0] in _NOTE_STAT_WORDS):
        return True
    return "paint" in keys[:3]


def _paren_stat(inner: str) -> tuple[str | None, str | None]:
    """How a parenthesis bears on the stat: ('add', '+ assists') for an
    addition ('(+ assists)', '(and assists)', '(+ reb + ast)', '(w/ assists)',
    '(assists included)'); ('mention', ...) for another stat named as if it
    were added ('(assists too)'), which leaves a line on another stat
    unrecognised (review 5, A7); ('qual', ...)
    when every word changes or names the stat and one changes it
    ('(offensive)', '(3PA)', '(off.)', '(3pt attempts)', '(offensive
    only)'), and also when a qualifier sits among other words ('(offensive
    boards are his thing)'), whose join then names no stat, so the line is
    unrecognised, never priced on the stat without it (review 4, A2);
    ('stat', ...) for a stat named outright ('(R+A)'); else (None, None), a
    gloss whose words leave the stat alone ('(weak defensive team)', '(off
    the bench)', review 3, R10)."""
    words = [w for w in re.split(r"[\s,/]+", inner) if w]
    if not words or len(words) > 8:
        return None, None
    if [w.lower() for w in words[:2]] in _CONNECT_PAIRS:
        words = words[1:]                                # "(combined with assists)"
    head = words[0].lower().rstrip(".")
    if head in _CONNECT or head[0] in "+&":
        rest = " ".join(words[1:]) if head in _CONNECT else inner.lstrip("+& ")
        return ("add", f"+ {rest}") if rest and typed_stat(rest) is not None else (None, None)
    if len(words) >= 2 and words[-1].lower().rstrip(".") in _CONNECT_AFTER \
            and typed_stat(" ".join(words[:-1])) is not None:
        return "add", "+ " + " ".join(words[:-1])       # "(assists included)"
    keys = [label_key(w.strip(_EDGE_PUNCT + ".")) for w in words]
    quals = [i for i, k in enumerate(keys) if k in _STAT_QUALIFIERS or _strong_qual(keys, i)]
    if quals:
        if all(i in quals or k in _CONNECT or k in _PAREN_FILLER or _stat_word(k)
               for i, k in enumerate(keys)):
            return "qual", " ".join(w for w, k in zip(words, keys) if k not in _PAREN_FILLER)
        strong = [i for i in quals if keys[i] not in _WEAK_QUALIFIERS
                  or not (i + 1 < len(keys) and keys[i + 1] in _WEAK_GLOSS_NEXT)]
        if strong and not any(i + 1 < len(keys) and keys[i + 1] in _MATCHUP_NOUNS
                              for i in strong):
            return "qual", inner
    if typed_stat(inner) is not None:
        return "stat", inner
    core = " ".join(w for w, k in zip(words, keys) if k not in _PAREN_FILLER)
    if core and typed_stat(core) is not None:
        return "stat", core                              # "points (3s only)" (review 7, A8)
    if any(k in _PAREN_STAT_NOTES for k in keys):
        return "qual", inner                             # "(in the paint)": another stat
    if not _has_digit(inner) and any(_stat_word(k) for k in keys) \
            and any(k in _ADDING_WORDS for k in keys):
        # "(assists too)", "(his assists count as well)": another stat named
        # as if it were added. Whether it is the line's own is told where that
        # is known (_read_line). A note that only mentions one ("(no assists
        # line today)", "(UNDER on his assists)") is a gloss.
        return "mention", inner
    return None, None


def _ends_in_a(tokens: list[str]) -> bool:
    """The words end in a lone letter "a" joined to the stat before it
    ("points + a", "PTS & A", "points + A+", "points/a"), which may be the
    article of the words after it (review 6, B2; review 7, A4, B3)."""
    last = tokens[-1].lower().rstrip("+") if tokens else ""
    if last == "a":
        # After a stat word only: in "R + A" the "A" is Assists.
        return (len(tokens) >= 3 and tokens[-2] in _JOINERS
                and len(label_key(tokens[-3])) >= 3)
    glued = re.search(r"([a-z0-9]+)[+&/]a$", last)
    return glued is not None and len(glued.group(1)) >= 3      # "points/a", not "R+A"


def _stat_then_note(post: list[str]) -> tuple[str, str] | None:
    """(stat, phrase) for the words after a line: the longest run of up to
    six of them that names a stat, with a note after it that names none.
    None when no run does, when the note names a stat, or when the run
    crosses a comma and more than a closer ('combined') follows it:
    'rebounds, assists are a sneaky play too' is R+A or Rebounds and a note,
    and neither is read (review 4, B1)."""
    for j in range(min(len(post) - 1, 6), 0, -1):
        phrase = " ".join(post[:j])
        note = post[j:]
        if post[j - 1] in _JOINERS:
            continue                                     # "Pts +": an unfinished phrase
        if note[0][:1] in _JOINERS and _stat_piece(
                note[0].lstrip("+&/") or (note[1] if len(note) > 1 else ""), False):
            continue                                     # "Pts", then "+ R + A": never Points alone
        if _ends_in_a(post[:j]):
            # "points + a lock", "POINTS + A LOCK", "points + A+ spot",
            # "points/a lock": "+ a" is Assists, or "a" the article of the note
            # after it (review 6, B2; review 7, A4, B3).
            return None
        lead = note[0].strip(_EDGE_PUNCT + ".")
        if len(lead) == 1 and lead.lower() in _TYPED_PART and (lead.lower() != "a" or lead == "A"):
            # "Pts, R, A", "Reb, A": a combo written with commas, or a stat
            # and a note? Never the first stat alone (review 7, A2). A
            # lower-case "a" is the article ("points, a lock").
            return None
        st = typed_stat(phrase)
        if st is None and "," in phrase and note[0].strip(_EDGE_PUNCT + ".").lower() in _COMBINERS:
            # "rebounds, assists combined": two stats with a comma between them
            # are a combo when a word says so (review 5, A9).
            st = typed_stat(phrase.replace(",", " and "))
        if st is None:
            continue
        if _names_stat(note):
            return None
        if any(w.endswith((",", ";")) for w in post[:j - 1]) and any(
                w.strip(_EDGE_PUNCT + ".").lower() not in _COMBO_CLOSERS | _POST_TAIL
                for w in note):
            return None
        return st, phrase
    return None


# The words of a tout's decoration in a handle or a hashtag ("#NBA",
# "#PrizePicks", "#LOCK", "@hoopspicks:", "@hoops_daily"), which never stand
# for a player. A tag is decoration only when it splits wholly into these
# (digits aside): one that merely holds one is a player's as often
# ("@Money23Green", "@HarrisonBarnes", "#Pickett" were read as decoration and
# their lines priced as the name line's player, review 6, A4, B7). A pick
# under its name line with decoration keeps the name above (review 5, B10);
# "#KAT" and "@jalenbrunson1" stand for a player (review 4, B3). None of these
# is a word of a name on the roster.
_DECOR_PARTS = frozenset({
    "nba", "picks", "pick", "props", "prop", "bets", "bet", "betting", "lock", "locks", "potd",
    "bol", "parlay", "parlays", "dfs", "prize", "play", "plays", "fire", "lfg", "gambling",
    "gamble", "sports", "sport", "tips", "tip", "free", "money", "cash", "units", "unit",
    "slip", "slips", "ticket", "tickets", "hoops", "hoop", "fantasy", "odds", "twitter",
    "daily", "the", "best", "top", "king", "guru", "edge", "value", "official", "hq", "news",
    "club", "talk", "nation", "insider", "live", "bettor", "bettors", "capper", "cappers",
    "winners", "winner", "league", "bank", "bankroll", "degen", "degens", "stats", "analytics",
    "model", "models", "my", "our", "bucket", "buckets",
    # "#PlayerProps", "#LockOfTheDay", "#MoneyMaker", "#LetsCash" (review 7, B18)
    "player", "players", "of", "day", "it", "maker", "underdog", "lets", "let"})
# Among a decoration tag's parts, at least one of the tout's own words: "#TheKing",
# "@King23" and "#Buckets" are a player's nickname (review 7, B10).
_BET_PARTS = frozenset({
    "nba", "picks", "pick", "props", "prop", "bets", "bet", "betting", "lock", "locks", "potd",
    "bol", "parlay", "parlays", "dfs", "prize", "play", "plays", "fire", "lfg", "gambling",
    "gamble", "sports", "sport", "tips", "free", "money", "cash", "units", "unit", "slip", "slips",
    "ticket", "tickets", "hoops", "hoop", "fantasy", "odds", "bettor", "bettors", "capper",
    "cappers", "degen", "degens", "bankroll", "underdog", "twitter"})
_DECOR_TAGS = frozenset({"pp", "ud", "w", "gl", "ez", "goat", "fyp"})
# Teams' own hashtags ("#DubNation", "#LakeShow", "#BleedGreen"): a fan's
# decoration, never a player (review 6, A11).
_TEAM_SLOGANS = frozenset({
    "truetoatlanta", "bleedgreen", "differenthere", "brooklyngrit", "netsworld", "allfly",
    "buzzcity", "bullsnation", "letemknow", "bethefight", "mffl", "milehighbasketball",
    "detroitbasketball", "dubnation", "onemission", "rocketsnation", "boombaby", "goldblooded",
    "clippernation", "lakeshow", "grindcity", "bigmemphis", "heatculture", "heattwitter",
    "fearthedeer", "raisedbywolves", "wontbowdown", "newyorkforever", "thunderup",
    "magictogether", "philaunite", "valleyproud", "wearethevalley", "ripcity", "lightthebeam",
    "sacramentoproud", "porvida", "gospursgo", "wethenorth", "takenote", "dcaboveall"})
# Beside a team's nickname in a fan's tag ("#GoSpursGo", "#KnicksTape").
_FAN_WORDS = frozenset({"go", "lets", "let", "up", "tape", "fan", "fans", "gang", "szn",
                        "season", "forever", "pride", "family", "nation", "in", "for", "we",
                        "are", "all", "day", "win", "time"})
# "hoopspicks: Jalen Brunson ...": a source's name and a colon before a pick.
_SOURCE_PREFIX = re.compile(r"^_*[a-z][a-z0-9_]{1,29}_*:\s+")


def _splits_into(word: str, parts) -> list[str] | None:
    """`word` as a run of `parts` and nothing else ("hoopspicks" is hoops +
    picks; "pickett" is no run), or None. Bounded: a tag is at most 30
    letters and a part at most 12."""
    best: list = [[]] + [None] * len(word)
    for i in range(1, len(word) + 1):
        for j in range(max(0, i - 12), i):
            if best[j] is not None and word[j:i] in parts:
                best[i] = best[j] + [word[j:i]]
                break
    return best[-1]


def _could_name(tag: str) -> bool:
    """A handle or hashtag that may stand for a player ('@jalenbrunson1',
    '#KAT', '@Money23Green'), not a tout's decoration ('#NBA', '#LOCK',
    '@hoopspicks:'), a team ('#Knicks') or a team's own hashtag ('#DubNation',
    '#GoSpursGo')."""
    core = str(tag).strip(_EDGE_PUNCT + "@#:. ")[:30]
    low = core.lower()
    if not low or low in _DECOR_TAGS or low in _TEAM_SLOGANS:
        return False
    pieces = [p for p in re.split(r"[_\d]+", low) if p]
    runs = [_splits_into(p, _DECOR_PARTS) for p in pieces]
    if pieces and all(r is not None for r in runs) and any(
            w in _BET_PARTS for r in runs for w in r):
        return False
    if _team_word(core) is not None or _team_word(core.upper()) is not None:
        return False
    # A fan's tag made of a team's nickname and a fan's words ("#GoSpursGo",
    # "#KnicksTape"). Never a city, which names players too ("@PJWashington"),
    # and never a tag with another word in it ("@SunsDevinBooker").
    camel = re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|\d+", core)
    words = [w.lower() for w in camel if not w.isdigit()]
    nick = [w for w in words if len(w) >= 4 and teams.from_words(w, cities=False)]
    return not (nick and "".join(camel).lower() == low.replace("_", "")
                and all(w in nick or w in _DECOR_PARTS or w in _FAN_WORDS for w in words))


def _read_typed(ln: str) -> dict | str | None:
    """One typed line as {name, team, opp, stat_type, line, pick, alt_pick,
    odds, kind, total, stat}; _UNREAD for a line shaped like a pick that holds
    more than one number ('Bridges o6.5 R+A, hit in 8 of 10'); _RECAP for a
    pick that reports its result ('Jalen Brunson o25.5 pts - cashed'), and
    _RECAP_NAMED when that recap names a player; _UNCLEAR for a pick whose
    side cannot be told (a No that may be the answer or a note's word,
    _market_answers); None
    when it is not a player's pick (a team total is not, nor an average).
    A line whose name was a handle or a hashtag ('@jalenbrunson1 o25.5 pts')
    names nobody (NAME_BAD), so it never borrows the name above (review 4,
    B3).
    `kind` is _typed_name's: NAME_NONE when the line has no name words at all
    ('over 6.5 Rebs+Asts' under a name on its own line), NAME_BAD when its
    words are not a full name. `total` marks a nameless line that speaks of a
    team or game total, which never borrows a name. `alt_pick` is the pick if
    an ESPN-style 'NO' on the line was the market's No, not New Orleans.
    Bounded: a line is at most MAX_TEXT_LINE_LEN characters and
    _TYPED_MAX_TOKENS words, every loop is over those words, and a line is
    read at most twice (a leading '3pm', _TIP_23)."""
    tags: list[str] = []
    s = _clean(ln, tags)
    got = None
    tip = _TIP_23.match(s)
    if tip:
        # "3pm Jalen Brunson o25.5 pts" is a weekend tip time; "3PM over 4.5
        # Stephen Curry" reads only with it as the stat (review 3, F11).
        got = _read_line(_clean(s[tip.end():]))
        if not (isinstance(got, dict) and got["stat"] is not None):
            got = None
    if got is None:
        got = _read_line(s)
    if isinstance(got, dict) and got["kind"] == NAME_NONE and any(map(_could_name, tags)):
        got["kind"] = NAME_BAD          # a handle or hashtag stood where the name goes
    if isinstance(got, dict) and got["kind"] == NAME_BAD:
        # "@Jalen Brunson over 25.5 points": the at sign was a mention and the
        # handle his first name; "hoopspicks: Jalen Brunson ...": a source and
        # a colon. Read again without them, and kept only when that reading
        # names a player in full (review 5, A21).
        raw = _strip_symbols(str(ln)).strip()
        for again in (re.sub(r"^@\s*(?=[A-Z])", "", raw), _SOURCE_PREFIX.sub("", raw)):
            if again != raw:
                alt = _read_line(_clean(again))
                if isinstance(alt, dict) and alt["kind"] == NAME_OK:
                    return alt
    return got


def _read_line(s: str) -> dict | str | None:
    """_read_typed on a line _clean has cleaned."""
    lead = _LEAD_NUMBER.match(s)
    if lead and _HAS_LINE.search(s, lead.end()):
        s = s[lead.end():]                       # "1 Miles Bridges over 6.5": a list number
    if not _has_digit(s):
        return None
    question = "?" in s                          # "Will X record 7+ ...? No"
    period = _PERIOD.search(" ".join(_PERIOD_NOTE.sub(" ", _unglue_periods(s)).lower().split()))
    s = _TYPED_SEPS.sub(", ", _join_stat_seps(s))
    units = _UNITS_FIELD.search(s)
    if units and _HAS_LINE.search(s, 0, units.start()):
        s = s[:units.start()]                    # a units column closing the row
    team = opp = None
    odds = "standard"
    extra: list[str] = []          # a side a parenthesis holds
    numeric: list[str] = []        # a parenthesis holding a number
    qual: list[str] = []           # a parenthesis that changes the stat
    adding = False                 # ...by adding one: "(+ assists)", "(and assists)"
    restate: list[str] = []        # a parenthesis naming a stat outright: "(R+A)"
    mention: list[str] = []        # ...or among other words: "(his assists are up)"
    paren_no: list[bool] = []      # an all-caps "(NO)": True when a number comes before it
    paren_ans: list = []           # a parenthesis answering the market: (value, form)
    paren_recap = False            # "(cashed)", "(won)": a result (review 5, A10)
    q_at = s.find("?")
    for m in _TYPED_PAREN.finditer(s):
        inner = m.group(1).strip(_EDGE_PUNCT + " ")
        low = inner.lower()
        words = low.split()
        if not words:
            continue
        if inner.startswith("@") or (words[0] in _OPP_MARKS and len(words) > 1):
            code = _team_or_no(inner[1:] if inner.startswith("@") else inner.split(None, 1)[1])
            opp = code or opp
        elif _team_word(inner):
            team = _team_word(inner)
        elif low in _ODDS_WORDS:
            odds = _ODDS_WORDS[low]
        elif inner.rstrip(".!") == "NO" and not (question and 0 <= q_at < m.start()):
            paren_no.append(_has_digit(s[:m.start()]))
        elif low.rstrip(".!") in _SIDE_WORDS or low.rstrip(".!") in _SIDE_LETTERS:
            # "(Over)", "(U)": a letter as its word, so "25.5 (U)" is never
            # read as "25.5 u", a unit size (review 5, A12).
            yl = low.rstrip(".!")
            extra.append({"o": "over", "u": "under"}.get(yl, yl))
        elif _paren_recap(low):
            paren_recap = True
        elif _paren_answer(inner) is not None:
            # "(No)", "(No.)", "(No, fade)", "(No 2u)", "(answer: No)": the
            # visitor's answer, weighed once the side is known (review 5, A12;
            # review 6, A6, B6).
            paren_ans.append(_paren_answer(inner))
        else:
            how, said = _paren_stat(inner)
            if how in ("add", "qual"):
                qual.append(said)
                adding = adding or how == "add"
            elif how == "stat":
                restate.append(said)
            elif how == "mention":
                mention.append(said)
            elif _has_digit(inner) and not (
                    _TYPED_ODDS.match(inner.strip(_SIGN_PUNCT)) or _TYPED_UNITS.match(inner)):
                numeric.append(inner)
    s = _TYPED_PAREN.sub(" ", s)
    if "(" in s:
        s = s[:s.index("(")]
    s = s.replace(")", " ")
    if not _has_digit(s):
        # "Miles Bridges Rebounds + Assists (Over 6.5)": the line sits in a
        # parenthesis. Read from there only when it is the one number given.
        if len(numeric) != 1:
            return None
        s = s + " " + numeric[0]
    toks = _typed_tokens(s + " " + " ".join(extra))
    if not toks or len(toks) > _TYPED_MAX_TOKENS:
        return None
    bare = [t.strip(_EDGE_PUNCT).lower() if t != "@" else "@" for t in toks]
    used: set[int] = set()
    nums = []
    shaped = False                 # a side word or a side letter before a number
    for i, b in enumerate(bare):
        if _TYPED_SPREAD.match(toks[i].strip(_SIGN_PUNCT)):
            return None                                  # "-3.5": a spread, never a line
        if i + 1 < len(bare) and bare[i + 1] in _UNIT_WORDS and _TYPED_NUM.match(b):
            used |= {i, i + 1}                           # "2 units"
            continue
        plus = _TYPED_PLUS.match(b)
        if plus:
            nums.append((i, float(plus.group(1)), "more"))
        elif _TYPED_NUM.match(b) and not _stat_digit(bare, i):
            nums.append((i, float(b), None))
        elif b in _SIDE_WORDS or (b in _SIDE_LETTERS and i + 1 < len(bare)
                                  and _TYPED_NUM.match(bare[i + 1])):
            shaped = True
    if len(nums) != 1:
        return _UNREAD if len(nums) > 1 and (shaped or any(x[2] for x in nums)) else None
    k, value, implied = nums[0]
    used.add(k)
    if bare[k + 1:k + 2] == ["or"] and k + 2 < len(bare) and bare[k + 2] in _OR_SIDE:
        implied = _OR_SIDE[bare[k + 2]]                  # "7 or more"
        used |= {k + 1, k + 2}
    lead_words = bare[max(0, k - 3):k]
    if lead_words[-2:] in (["at", "least"], ["at", "most"]):
        implied = "more" if lead_words[-1] == "least" else "less"
        used |= {k - 2, k - 1}
    elif lead_words[-3:] in (["no", "more", "than"], ["no", "less", "than"],
                             ["no", "fewer", "than"]):
        implied = "less" if lead_words[-2] == "more" else "more"
        used |= {k - 3, k - 2, k - 1}
    sides = {implied} if implied else set()
    last = len(bare) - 1
    # A side written as a word or a letter, not implied: beside one, a No is
    # read only where it plainly answers (_market_answers; review 6, B5, B6).
    explicit = any(i not in used and (b in _SIDE_WORDS or (b in _SIDE_LETTERS and (
        i == k - 1 or (i == last and i > k)))) for i, b in enumerate(bare))
    # The last word that says something, a closer ("LOCK"), a unit size or a
    # mark aside.
    end = last
    while end > k and (end in used or not bare[end].rstrip(".")
                       or bare[end].rstrip(".") in _POST_TAIL):
        end -= 1
    got_answers = _market_answers(toks, bare, k, used, end, explicit)
    if got_answers is None:
        return None
    # A side that cannot be told is said once the line is known to be a pick:
    # "No, Brunson had 40 last night" is chatter, not an unclear pick (review 7,
    # B12).
    unclear = isinstance(got_answers, str)
    answers, no_at = (set(), []) if unclear else got_answers
    for val, form, pkind in paren_ans:
        if val or (form == "alone" and not (pkind == "idiom" and explicit)):
            answers.add(val)
        else:
            unclear = True         # "(no, not a typo)", "(no way)" beside an Over: which side?
    for i, b in enumerate(bare):
        if i in used:
            continue
        if b in _SIDE_WORDS:
            sides.add(_SIDE_WORDS[b])
        elif b in _SIDE_LETTERS and (i == k - 1 or (i == last and i > k)):
            sides.add(_SIDE_LETTERS[b])
        elif b in _ODDS_WORDS:
            odds = _ODDS_WORDS[b]
        elif b in _NEUTRAL_SIDES:
            pass                                         # "O/U 6.5": the side, if any, is elsewhere
        elif "/" in b and set(b.split("/")) & {"over", "under", "o", "u", "more", "less"}:
            return None                                  # "over/no": no side to read
        else:
            continue
        used.add(i)
    maybe_no = False               # an ESPN "NO" that may be New Orleans, settled by the roster
    for after in no_at + paren_no:
        if after and implied is not None and not explicit:
            answers.add(False)                           # "7+ R+A (NO)": the market's No
        else:
            maybe_no = True                              # "Zion Williamson o22.5 NO" (review 3, F1)
    if len(sides) > 1 or len(answers) > 1:
        # "over 25.5 points, no chance he goes under": two sides beside a No
        # that may be the note's, counted; any other line with two is none.
        return _UNCLEAR_NAMED if unclear else None
    pick = next(iter(sides), None)
    answered = bool(answers)
    if answers:
        # A prediction market's Yes or No on the line's own side: "7+ ... No"
        # is the Under.
        if pick is None:
            return None
        if not answers.pop():
            pick = _flip(pick)
    line = value - 0.5 if implied == "more" and value.is_integer() else \
        value + 0.5 if implied == "less" and value.is_integer() else value
    if _parse_line(line) is None:
        return None
    pre, opp_pre = _strip_opp([toks[i] for i in range(k)
                               if i not in used and _kept(toks[i], period)])
    post, opp_post = _strip_opp([t for t in _breaks([toks[i] for i in range(k + 1, len(toks))
                                                     if i not in used])
                                 if _kept(t, period)])
    while pre and pre[-1].strip(_EDGE_PUNCT).lower() in _PRE_TAIL:
        pre.pop()                                        # "will be", "for a combined"
    if pick is not None or line % 1 == 0.5:
        # A closer: "tonight", "LOCK". With no side, only after a line's .5:
        # "31 points tonight" is a box score (review 4, B9). Never after "+
        # a", whose "a" may be its article: "points + a lock" (review 6, B2).
        while post and post[-1].strip(_EDGE_PUNCT).lower() in _POST_TAIL and not (
                len(post) >= 2 and _ends_in_a(post[:-1])):
            post.pop()
    total = any(b in _TOTAL_WORDS for i, b in enumerate(bare) if i not in used)
    opp = opp_post or opp_pre or opp
    stat = name = name_team = label = phrase = None
    kind = NAME_NONE
    note: list[str] = []           # the words set aside after the stat
    if post:
        st = typed_stat(" ".join(post))
        if st is not None:                               # NAME [side] 6.5 STAT
            stat, phrase = st, " ".join(post)
            name, name_team, kind = _typed_name(pre, question=question)
        elif typed_stat(" ".join(pre)) is not None:      # STAT 6.5 [side] NAME
            stat, phrase = typed_stat(" ".join(pre)), " ".join(pre)
            name, name_team, kind = _typed_name(post, question=question)
        else:
            if _typed_name(pre, question=question)[2] == NAME_NONE:
                for j in range(min(len(post) - 1, 6), 0, -1):    # [side] 6.5 STAT for NAME
                    st_j = typed_stat(" ".join(post[:j]))
                    if st_j is None:
                        continue
                    nm, nt, kd = _typed_name(post[j:], question=question)
                    if kd == NAME_OK:
                        stat, phrase, name, name_team, kind = (st_j, " ".join(post[:j]),
                                                               nm, nt, kd)
                        break
            if stat is None and pick is not None:
                # NAME [side] 6.5 STAT, then a note ("points, second half of a
                # back to back", "Points lets go"). On a line with a side only:
                # without one, "12 rebounds, cashed" is a result and "Points
                # 27.1 avg" an average (review 3, R7).
                got = _stat_then_note(post)
                nm, nt, kd = _typed_name(pre, question=question)
                if got is not None and kd == NAME_OK:
                    (stat, phrase), name, name_team, kind = got, nm, nt, kd
                    note = post[len(phrase.split()):]
            if stat is None and pick is not None and not _names_stat(post):
                # NAME STAT 6.5 [side], then a note ("- Max play").
                got = _name_then_stat(pre, question)
                if got is not None and got[3] == NAME_OK:
                    stat, name, name_team, kind, phrase = got
                    note = list(post)
            if stat is None and (pick is not None or period is not None):
                # A pick on a stat nothing maps: kept under its own words, so
                # classify_stat refuses or counts it, never drops it.
                name, name_team, kind = _typed_name(pre, question=question)
                label = " ".join(post)
                note = list(post)
    else:                                                # NAME STAT [side] 6.5
        got = _name_then_stat(pre, question)
        if got is not None:
            stat, name, name_team, kind, phrase = got
    if kind == NAME_TEAM or (stat is None and label is None):
        return None

    def result_word(t: str) -> bool:
        return t.strip(_EDGE_PUNCT + ".").lower() in _RECAP_WORDS

    def opens_with_result(words: list[str]) -> bool:
        for n, t in enumerate(words):
            w = t.strip(_EDGE_PUNCT + ".=<>~").lower()
            if not w or w in _POST_TAIL or w in _RECAP_SUBJECTS:
                continue                                 # "=", "LOCK", "that one"
            return w in _RECAP_WORDS and not _missed_context(words[n:])
        return False

    hope = any(b.rstrip(".") in _HOPE_WORDS for b in bare)
    cashes = any(t.strip(_EDGE_PUNCT + ".").lower() == "cashes" for t in pre + note)
    if (any(result_word(t) for t in pre) or opens_with_result(note) or paren_recap
            or any(t.strip(_EDGE_PUNCT + ".").lower() in _RECAP_ALWAYS for t in note)
            or (label is not None and any(result_word(t) for t in note))
            or (cashes and not hope) or (_result_only(note) and not hope)):
        # "Jalen Brunson o25.5 pts - cashed", "Cashed: ...", "- that one lost",
        # "- he cashed it": a result before the line, first in the note or
        # (cashed) anywhere in it is a recap, counted, never a pick (review 3,
        # R7; review 4, A7, B7; review 5, A10, B6). Later in a note "won" and
        # "lost" are context: "over 25.5 points, Knicks lost last night".
        return _RECAP_NAMED if kind in (NAME_OK, NAME_BAD) else _RECAP
    if unclear:
        # Counted, never guessed; one naming nobody belongs to the name line
        # above, which it keeps (review 7, B13).
        return _UNCLEAR if kind == NAME_NONE else _UNCLEAR_NAMED
    if maybe_no and pick is not None and not answered and kind == NAME_NONE:
        # "NO 7+ assists" under a player's name names no team: the market's
        # No (review 3, R4).
        pick, maybe_no = _flip(pick), False
    alt_pick = _flip(pick) if maybe_no and pick is not None and not answered else None
    if qual and stat is not None:
        # "rebounds (offensive)", "threes (3PA)", "points (+ assists)": the
        # stat written, or none (an unrecognised label), never the stat
        # without its word.
        q = " ".join(qual)
        better = None
        for cand in ((f"{phrase} {q}",) if adding else (f"{q} {phrase}", f"{phrase} {q}")):
            better = STAT_MAP.get(label_key(cand)) or typed_stat(cand)
            if better:
                break
        if better is None and not adding:
            better = typed_stat(q)                       # "(OREB)", "(3PA)" name it outright
        stat, label = (better, None) if better else (None, f"{phrase} {q}")
    if restate and stat is not None and any(typed_stat(r) != stat for r in restate):
        stat, label = None, f"{phrase} ({restate[0]})"   # "points (rebounds)": which is it?
    if mention and stat is not None and not all(
            _named_stats(r) <= _STAT_PARTS.get(stat, {stat}) for r in mention):
        # "points (his assists are up too)": priced as Points it was a lean on
        # a stat the visitor may not mean (review 5, A7).
        stat, label = None, f"{phrase} ({mention[0]})"
    label = STAT_LABEL[stat] if stat is not None else label
    if period is not None:
        # A half, quarter or first-minutes line, wherever the period was
        # written: classify_stat refuses the label as a period prop.
        label = f"{label} {period.group(0)}"
    # A NO beside the name scopes the name match to the Pelicans, below any
    # team written out; resolve_names retries without it.
    hint = "NOP" if maybe_no and name else None
    return {"name": name, "team": team or name_team or hint, "opp": opp,
            "stat_type": label[:MAX_LABEL_LEN], "line": line, "pick": pick,
            "alt_pick": alt_pick, "odds": odds, "kind": kind,
            "total": total and name is None, "stat": stat}


def _name_only(ln: str) -> tuple[str, str | None, str | None, str | None] | None:
    """(name, team, opponent, period) for a player's name alone on its line,
    which names the picks after it: 'Jalen Brunson', 'Jalen Brunson (NYK)',
    'Jalen Brunson vs Heat', 'Jalen Brunson - NYK @ MIA', 'Jalen Brunson
    props:', 'Jalen Brunson @ Heat 7:30 PM' (a tip time or a date is set
    aside, as in a heading: review 3, F10). A period on the line ('Jalen
    Brunson (1Q)', 'Jalen Brunson 1Q:', '1Q: Jalen Brunson') is the period of
    the picks under it, which were priced as full-game lines (review 5,
    A3)."""
    s = _clean(ln)
    team = per = None
    for m in _TYPED_PAREN.finditer(s):
        inner = m.group(1)
        team = team or _team_word(inner) or next(
            (t for t in map(_team_word, re.split(r"[,;/]", inner)) if t), None)
        per = per or _named_period(_PERIOD_NOTE.sub(" ", inner))
    s = _noise_sub(_HEADING_NOISE, _TYPED_PAREN.sub(" ", s))
    per = per or _named_period(_PERIOD_NOTE.sub(" ", s))
    s = _drop_period(s).strip(_EDGE_PUNCT + " ")
    if not s or _has_digit(s):
        return None
    toks, opp = _strip_opp(_typed_tokens(s))
    while toks and toks[-1].strip(_EDGE_PUNCT).lower() in _TOUT_NOUNS:
        toks.pop()
    name, name_team, _kind = _typed_name(toks, strict=True)
    return (name, team or name_team, opp, per) if name else None


_FULL_GAME = re.compile(r"\bfull[\s\-]*game\b", re.IGNORECASE)
# A word that cancels the period right after it: "No 1Q props today", "not
# betting the 1H", "My full-game picks (no 1Q or 1H)" (review 4, B2). Only
# right before it: "1Q props, no parlays:" and "No doubt 1Q locks:" are
# labels, and so is "Not your usual 1Q props:" (review 5, A1, B1).
_NEGATIONS = frozenset({"no", "not", "non", "without"})
# Words a negation reaches the period across: "not betting the 1H", "no more
# 1Q props", "no 1Q or 1H". Of the words ending in "ing", only the verbs
# whose object is the period: "No more waiting, 1Q props:" and "No fading
# the 1Q:" are labels (review 6, A3).
_NEG_REACH = frozenset({"the", "a", "an", "any", "more", "of", "or", "and", "&", "nor", "/"})
_NEG_VERBS = frozenset({"betting", "playing", "taking", "doing", "touching", "chasing",
                        "posting"})
# After a negated period, the words that make it the period's picks it denies.
_CANCEL_NEXT = frozenset({"", "props", "prop", "picks", "pick", "plays", "play", "bets", "bet",
                          "lines", "line", "locks", "lock", "action", "today", "tonight",
                          "tomorrow", "for", "or", "and", "nor", "&", "/", "either", "this",
                          "here", "yet", "anymore", "please", "only", "games", "game", "at",
                          "all"})
# After a period, on a line that also says "full game": the period comes
# later, and the section is the full game ("Full game picks, 1H later:").
_DEFERRED = frozenset({"later", "tomorrow", "soon"})
# Words that make a line naming a period a remark about a game, not the label
# of a section: "Q4 Brunson is different", "the Suns have been awful in the
# first half", "1H recap: 3-1" (review 4, A6, B2). Never on a line that ends
# with a colon or names the picks ("1Q props no one is talking about:").
_REMARK_WORDS = frozenset({
    "is", "are", "was", "were", "been", "has", "have", "had", "went", "goes", "blew", "blown",
    "hit", "lost", "won", "cashed", "missed", "scored", "dropped", "did", "does", "got",
    "looked", "looks", "struggled", "struggles", "dominated", "averaged", "averages", "recap",
    "recaps", "results", "yesterday", "isn't", "wasn't", "aren't", "weren't", "doesn't",
    "didn't", "owns", "loves", "kills", "torches", "cooks", "dominates"})
# Of those, the ones that make a remark wherever they sit: a result or a
# recap ("1H recap: 3-1", "Brunson went off in the 4th quarter"). The rest
# ("is", "has", "looks") make one only with a subject other than the period:
# "Q4 Brunson is different" and "the Suns have been awful in the first half"
# are remarks, while "1Q is the move", "1Q looks juicy tonight" and "What I
# have for the 1Q" label quarter picks, which were priced as full-game lines
# (review 6, B4).
_REMARK_STRONG = frozenset({
    "went", "blew", "blown", "hit", "lost", "won", "cashed", "missed", "scored", "dropped",
    "struggled", "dominated", "averaged", "recap", "recaps", "results", "yesterday"})
# A line's first word that opens a sentence and names no one ("What I have for
# the 1Q", "Tonight is a 1Q night").
_REMARK_OPENERS = frozenset({
    "what", "how", "here", "here's", "heres", "tonight", "today", "tomorrow", "my", "the", "this",
    "these", "those", "some", "our", "all", "top", "best", "free", "big", "late", "early",
    "main", "full", "quick", "more", "any", "every", "why", "when", "where", "who", "which", "if",
    "so", "and", "but", "also", "just", "only", "no", "not", "yes", "ok", "okay", "lets", "let's",
    "i", "i'm", "im", "we", "we're", "it", "it's", "its", "there", "there's", "that", "that's"})
# Words that stand for a player or a team as a remark's subject.
_REMARK_PRONOUNS = frozenset({"he", "he's", "hes", "she", "they", "they're", "his", "their"})
_REMARK_PHRASE = re.compile(r"\blast\s+(?:night|game|time|week)\b"
                            r"|^\W*(?:what|how)\s+(?:a|an|about)\b", re.IGNORECASE)
# A line saying the period's picks are posted elsewhere ("2H plays coming later
# in the discord", "Live 2H bets in my discord"): a remark even beside "plays"
# or "bets", since those picks are not the ones under it (review 4, B2).
_ELSEWHERE = re.compile(r"\b(?:in|on)\s+(?:the|my|our)\s+(?:discord|telegram|patreon|group\s+chat"
                        r"|chat|channel)\b", re.IGNORECASE)
# A pick line that names no player in full is counted in a board paste only
# when it is this short: a board square run together onto one line (name,
# TEAM - POS, name again, matchup, line, label, buttons) is always longer.
_SHORT_PICK_WORDS = 10


def _drop_period(s: str) -> str:
    """s without the period markers in it ('Magic @ Suns 1H' is 'Magic @
    Suns'), nor the notes that only look like one ('2nd half of a
    back-to-back'), its case kept. Linear: one _PERIOD pass over a line whose
    spaces are already single."""
    s = " ".join(_PERIOD_NOTE.sub(" ", str(s)).split())
    low = s.lower()
    if len(low) != len(s):
        return s                         # a letter whose lower case is longer: left as is
    keep, at = [], 0
    for m in _PERIOD.finditer(low):
        keep.append(s[at:m.start()])
        at = m.end()
    keep.append(s[at:])
    return " ".join(" ".join(keep).split())


def _negated(before: list[str]) -> bool:
    """A negation right before a period, across at most three _NEG_REACH
    words, _NEG_VERBS or other period words ('no 1Q', 'not betting the 1H',
    'no 1Q or 1H'), and never across a mark: 'no doubt 1Q', 'not your usual
    1Q', 'No more waiting, 1Q props:' and 'No 2H, 1H props:' name their period
    (review 6, A3, B4). `before` keeps its punctuation."""
    for n, raw in enumerate(reversed(before)):
        if raw[-1:] in ",;:.!?" or set(raw) <= _DASHES:
            return False                             # a mark between it and the period
        w = raw.strip(_EDGE_PUNCT + ".")
        if w in _NEGATIONS:
            return True
        if n >= 3 or not (w in _NEG_REACH or w in _NEG_VERBS or _PERIOD_TOKEN.match(w)):
            return False
    return False


def _named_period(text: str) -> str | None:
    """The first period the text names that no negation cancels ('No 1Q props
    today, 1H only' names 1H) and that, beside 'full game', is not put off
    for later; None when there is none. At most four periods are looked at."""
    low = _unglue_periods(" ".join(str(text).lower().split()))[:MAX_LABEL_LEN]
    full = _FULL_GAME.search(low) is not None
    for n, m in enumerate(_PERIOD.finditer(low)):
        if n >= 4:
            break
        before = low[:m.start()].split()[-4:]
        nxt = low[m.end():].split()[:2]
        after = {w.strip(_EDGE_PUNCT + ".") for w in nxt}
        head = nxt[0].strip(_EDGE_PUNCT + ".") if nxt else ""
        # A negation cancels the period only when it denies the period's
        # picks ("No 1Q props today", "no 1Q or 1H", "Not betting the 1H
        # tonight"), never another word ("No 1Q unders, only overs:", "No 1Q
        # parlays:", review 7, A7).
        if (_negated(before) and (head in _CANCEL_NEXT or _PERIOD_TOKEN.match(head))) or (
                full and after & _DEFERRED):
            continue
        return m.group(0)
    return None


def _remark(text: str) -> bool:
    """Words of a remark about a game, not of a section's label: a result or
    a recap word (_REMARK_STRONG), 'last night', 'What a ...', or another of
    _REMARK_WORDS with a subject that is not the period: a team, a pronoun,
    or a capitalised word that is not the line's sentence opener ('Q4
    Brunson is different', 'Brunson owns the fourth quarter')."""
    raw = [w.strip(_EDGE_PUNCT + ".") for w in str(text).split()][:_TYPED_MAX_TOKENS]
    raw = [w for w in raw if w]
    words = {w.lower() for w in raw}
    if words & _REMARK_STRONG or _REMARK_PHRASE.search(str(text)) is not None:
        return True
    if not words & _REMARK_WORDS:
        return False
    # In Title Case ("1Q Is The Move") a capital says nothing of a name.
    alpha = [w for w in raw if w[:1].isalpha()]
    title = len(alpha) >= 3 and sum(w[:1].isupper() for w in alpha) >= 0.6 * len(alpha)
    for w in raw:
        low = w.lower()
        if _PERIOD_TOKEN.match(w) or _PERIOD.search(low) or low in _TOUT_NOUNS:
            continue
        if low in _REMARK_PRONOUNS or (w[:1].isupper() and _team_word(w) is not None):
            return True
        if not title and w[:1].isupper() and not (
                low in _PLAIN_WORDS or low in _REMARK_OPENERS or low in _REMARK_WORDS):
            return True
    return False


def _label_paren(m: re.Match) -> str:
    """A parenthesis on a label line, as _section_period reads it. One whose
    first item names a period stays, as that item alone: "Tonight's picks
    (1Q, been hitting)", "Card (1H, cashed yesterday)" were remarks, and
    their quarter picks were priced as full-game lines (review 6, B4). One
    that is a remark or cancels its period is set aside ("Suns (blew a 2H
    lead last night):", "(no 1Q or 1H)"); any other stays ("(1Q, 7 PM)")."""
    inner = m.group(1)
    if _PERIOD.search(" ".join(inner.lower().split())) and not _named_period(inner):
        # "(no 1Q or 1H)", "(1H later, full game now)": the parenthesis puts
        # its period off as a whole, before its first item is read (review 7,
        # B20).
        return " "
    first: list[str] = []
    for w in inner.split()[:_TYPED_MAX_TOKENS]:
        if set(w) <= _DASHES:
            break
        first.append(w)
        if w[-1:] in ",;":
            break
    head = " ".join(first).strip(",; ")
    if len(first) <= 3 and _named_period(head):
        return f"({head})"
    if _remark(inner) or (_PERIOD.search(" ".join(inner.lower().split()))
                          and not _named_period(inner)):
        return " "
    return m.group(0)


def _section_period(ln: str, strict: bool = False) -> str | None:
    """The period a section label or a heading names ('1Q props:', 'First
    quarter picks', '1Q Locks', '## 1Q props', 'Magic @ Suns (1H)'), or None.
    Refusing is the safe side, so a line that names a period is the label of
    the section under it unless the period is cancelled ('No 1Q props today',
    'Full game picks, 1H later:') or the line reads as a remark (_remark:
    'What a 4th quarter!', 'Q4 Brunson is different'). A line ending with a
    colon or naming the picks (_TOUT_NOUNS) is never a remark: round 4's
    shape test missed '1Q Locks' and '1Q props I like', and their quarter
    picks were priced as full-game lines (review 4, A6, B2; review 5, A1,
    B1). A parenthesis that is a remark or cancels its period is set aside
    first ('Suns (blew a 2H lead last night):'), one holding a tip time or a
    network is not ('(1Q, 7 PM)', review 5, B4), and one whose first item is
    the period is kept as that item ('Card (1H, cashed yesterday)', review 6,
    B4; _label_paren). `strict` (lines inside
    a board paste): only a label ending with a colon or naming the picks,
    since a square's own label ('1Q Points') and a board's period tabs ('1Q')
    do neither."""
    s = _unglue_periods(_TYPED_PAREN.sub(_label_paren, _clean(ln)))
    s = " ".join(_PERIOD_NOTE.sub(" ", s).split())[:MAX_LABEL_LEN]
    colon = s.endswith(":")
    tout = any(w.strip(_EDGE_PUNCT + ".#").lower() in _TOUT_NOUNS
               for w in s.replace(":", " ").split())
    if strict and not (colon or tout):
        return None
    found = _named_period(s)
    if found is None or (not colon and (_ELSEWHERE.search(s) or (not tout and _remark(s)))):
        return None
    return found


def _says_name(ln: str, name) -> bool:
    """The line names the player by his surname ("Brunson o25.5 pts - cashed"
    under "Jalen Brunson"), a suffix such as "Jr." aside, outside any
    parenthesis, and with no other first name before it: "Jaylin Williams"
    under "Jalen Williams" is another player (review 7, B9)."""
    parts = [p.strip(".").lower() for p in str(name or "").split()]
    parts = [p for p in parts if p and p not in _NAME_PARTICLES]
    if not parts:
        return False
    words = [_POSSESSIVE.sub("", w.strip(_EDGE_PUNCT + ".")) for w in
             _TYPED_PAREN.sub(" ", _clean(ln)).split()[:_TYPED_MAX_TOKENS]]
    for i, w in enumerate(words):
        if w.lower() == parts[-1]:
            prev = words[i - 1] if i else ""
            return not (prev[:1].isupper() and prev.isalpha() and prev.lower() != parts[0])
    return False


def _names_team(ln: str, pair) -> bool:
    """The line names one of the teams in `pair`, a word or two at a time
    ("Magic players:", "Orlando side:", "Golden State starters:")."""
    words = _clean(ln).replace(":", " ").split()[:_TYPED_MAX_TOKENS]
    return any(_team_word(" ".join(words[i:i + n])) in pair
               for n in (1, 2) for i in range(len(words) - n + 1))


def parse_typed_lines(text: str, board: bool = False) -> pd.DataFrame:
    """Parse a typed list of picks (see the block comment above). Linear in
    the paste and bounded: lines come through _board_lines (capped, never
    materialised whole), at most MAX_TYPED_LINES of them and MAX_TYPED_WORDS
    words (5,000 lines of 40 junk words cost seconds, review P8), and each is
    read once. A line that is not a pick is skipped; one that reads as a pick but
    names no player in full is counted in attrs['typed_noname'] (and, when
    it has a side and at most _SHORT_PICK_WORDS words, in
    attrs['typed_noname_pick'], which a board paste reports), and one
    shaped like a pick that holds more than one number in
    attrs['typed_unread'], and a pick reporting its result in
    attrs['typed_recap']. Errors cost the line (attrs['skipped_error']).
    attrs['typed_headings'] lists the team headings, which typed_games reads
    to name a game. A period a section label names ('1Q props:') is written
    onto the picks under it, across the headings under it, until a section
    label that names none; a heading's own ('Magic @ Suns (1H)') holds over
    the game's own team headings under it ('Magic:', 'Suns:', 'ORL-PHX') and
    its sub-labels naming one of them ('Magic players:'), and ends at any
    other (review 4, B2; review 5, B4; review 6, A13, B9). A name line's
    period ('Jalen Brunson (1Q)', 'Main Slate 1H props:') is a label's:
    every pick under it takes it, named or not (review 5, A3; review 6, A1).
    Picks given a period from above are counted in attrs['typed_period_label']
    (and those with a side in attrs['typed_period_label_pick']), a pick whose
    side cannot be told in attrs['typed_unclear'], and a nameless pick with
    no name line above it in attrs['typed_noname'] (review 6, B10, B12).
    `board` (the lines of a board paste, _with_typed_picks): a square's own
    label ('1Q Points') is no section label (_section_period's strict)."""
    attrs = _new_attrs()
    attrs["typed_headings"] = []
    attrs["typed_noname_pick"] = 0
    attrs["typed_period_label_pick"] = 0
    rows: list[dict] = []
    heading: str | None = None            # the team of the last team heading
    game: tuple[str, ...] | None = None   # the pair of the last game heading
    loose: tuple[str, ...] | None = None  # the pair of the last loosely joined heading
    pending: tuple | None = None          # (name, team, opp, period) alone on a line above
    period: str | None = None             # the period of the section label or heading above
    period_head = False                   # ...named by a heading: it ends at the next one
    period_pair: tuple | None = None      # ...a game heading's, which its own teams keep
    period_name = False                   # ...a name line's, which its own "full game" ends
    pending_full = False                  # the name line above says "full game"
    words_read = 0
    for n_read, ln in enumerate(_board_lines(str(text), attrs)):
        if (n_read >= MAX_TYPED_LINES or words_read >= MAX_TYPED_WORDS
                or len(rows) >= MAX_ROWS_PER_PASTE):
            attrs["truncated"] += 1
            break
        words_read += len(ln.split())
        try:
            parts = _heading_parts(ln)
            if parts is not None:
                head, is_loose = parts
                if len(head) == 2 and not is_loose:
                    heading, game, loose = None, head, None
                else:
                    # A team heading, or two teams a joiner may only group
                    # ("Knicks & Nets", "ORL-PHX"): a team heading each, and
                    # the schedule says whether the two are one game
                    # (typed_games; review 3, R6; review 4, B4).
                    heading = head[0] if len(head) == 1 else None
                    game = game if game and set(head) <= set(game) else None
                    loose = head if len(head) == 2 else (
                        loose if loose and heading in loose else None)
                    for h in head:
                        if h not in attrs["typed_headings"]:
                            attrs["typed_headings"].append(h)
                named = _section_period(ln)
                if named is not None:
                    period, period_head, period_name = named, True, False
                    period_pair = head if len(head) == 2 else None
                elif period_head and not ((len(head) == 1 or is_loose) and period_pair
                                          and set(head) <= set(period_pair)):
                    # "Knicks @ Cavs (1Q)", then "Knicks:" and "Cavs:": the
                    # game's own teams keep its period (review 5, B4).
                    period, period_head, period_pair = None, False, None
                pending = None
                continue
            got = _read_typed(ln)
            if got in (_RECAP, _RECAP_NAMED):
                named = _section_period(ln, strict=board)
                if named is not None:
                    # "Q1 props, cashed all 3 last night": a brag that labels
                    # the quarter picks under it (review 6, B4).
                    period, period_head, heading, game, loose = named, False, None, None, None
                    period_pair = pending = None
                    period_name = False
                    continue
                # "Jalen Brunson o25.5 pts - cashed": a result, counted (review 4,
                # A7). One naming another player ends the name line above, as a
                # pick would, so the next nameless line is never lent the name
                # across it (review 5, A11, B7); "Over 6.5 assists - cashed", or
                # "Brunson o25.5 pts - cashed" under his own name line, keeps it
                # (review 6, B10).
                attrs["typed_recap"] += 1
                if got == _RECAP_NAMED and not (pending and _says_name(ln, pending[0])):
                    pending = None
                continue
            if got in (_UNCLEAR, _UNCLEAR_NAMED):
                # A pick whose Yes or No may be the answer or a note's word:
                # counted, never guessed (review 6). One naming nobody keeps
                # the name line above, as a nameless pick would (review 7,
                # B13).
                attrs["typed_unclear"] += 1
                if got == _UNCLEAR_NAMED:
                    pending = None
                continue
            if isinstance(got, str):
                attrs["typed_unread"] += 1
                pending = None
                continue
            if got is None or (got["pick"] is None and got["stat"] is None
                               and _PERIOD.search(str(got["stat_type"]).lower())):
                # Not a pick: a name line, a section label, or a period's
                # reading with no side or stat ("First 5 minutes props:",
                # "Tonight's First 5 Minutes Props", whose digit made it
                # pick-shaped: review 5, A8), even beside words that read as a
                # name: "What I Like 1Q 7 PM:", "Jalen Brunson (1Q, 7 PM)" and
                # "Main Slate 1Q 3 picks:" were filed as refused rows, and the
                # quarter picks under them priced as full game (review 7, A1).
                pending = _name_only(ln)
                pending_full = pending is not None and bool(_FULL_GAME.search(ln))
                if pending is not None:
                    if pending[3]:
                        # "Jalen Brunson (1Q)", "Jalen Brunson 1Q props", "Main
                        # Slate 1H props:": a name line naming a period labels
                        # every pick under it, named or not, as a section label
                        # does; only its nameless picks were refused, and the
                        # named ones were priced as full-game lines (review 6,
                        # A1, B1).
                        period, period_head, period_pair = pending[3], False, None
                        period_name = True
                    elif pending_full and period_name:
                        # "Josh Hart (full game)" ends a name line's period. A
                        # label's or a heading's goes on past his block, and
                        # only his own picks are full game (pending_full):
                        # ending it put the later quarter picks at full game
                        # (review 7, B2).
                        period, period_head, period_pair = None, False, None
                        period_name = False
                else:
                    named = _section_period(ln, strict=board)
                    if named is not None:
                        # "1Q props:": the picks under it are that period's,
                        # refused as if it were written on each (review 3, R2).
                        period, period_head, heading, game, loose = named, False, None, None, None
                        period_pair, period_name = None, False
                    elif ln.rstrip().endswith(":"):
                        # "Bonus play:", "Parlay:": a section of its own, so the
                        # team heading above no longer groups what follows. Under
                        # "Magic @ Warriors (1H)", "Magic players:" is still that
                        # game's, and its half (review 6, B9).
                        # "Magic full game:" says otherwise (review 7, B22).
                        own = bool(period_head and period_pair and _names_team(ln, period_pair)
                                   and not _FULL_GAME.search(ln))
                        heading = None
                        if not own:
                            game = loose = period = period_pair = None
                            period_head = False
                    elif _FULL_GAME.search(ln):
                        period, period_head, period_pair = None, False, None
                continue
            name, team, opp = got["name"], got["team"], got["opp"]
            full_here = False             # lent by a name line that says "full game"
            if name is None and got["kind"] == NAME_NONE and pending and not got["total"] \
                    and (got["stat"] is None
                         or got["line"] <= _TYPED_PLAUSIBLE.get(got["stat"], MAX_LINE_VALUE)):
                # Lent from the name line above, and it stays lent to the next.
                name, team, opp = pending[0], team or pending[1], opp or pending[2]
                full_here = pending_full
            else:
                pending = None
            if name is None:
                # Counted when it named someone, just not in full ("Bridges
                # over 6.5"), and when it is a player's pick with no name line
                # above it ("Over 2.5 threes" after a recap of another player:
                # it vanished without a word, review 6, B10). A line with no
                # name and no stat is a game total or a stray, as on the board,
                # and a board paste's squares name nobody either.
                bad = got["kind"] == NAME_BAD
                orphan = (not board and got["kind"] == NAME_NONE and not got["total"]
                          and got["pick"] is not None and got["stat"] is not None
                          and got["line"] <= _TYPED_PLAUSIBLE.get(got["stat"], MAX_LINE_VALUE))
                attrs["typed_noname"] += int(bad or orphan)
                attrs["typed_noname_pick"] += int(bad and got["pick"] is not None
                                                  and len(ln.split()) <= _SHORT_PICK_WORDS)
                continue
            label = got["stat_type"]
            if period and not full_here and not _PERIOD.search(label.lower()):
                # The period of the label, heading or name line above, which
                # the report names (typed_period_label, review 6, B12).
                label = f"{label} {period}"[:MAX_LABEL_LEN]
                attrs["typed_period_label"] += 1
                attrs["typed_period_label_pick"] += int(got["pick"] is not None)
            rows.append({
                "name": name, "team": team or heading, "position": None,
                "stat_type": label, "line": got["line"], "start_time": None,
                "direction": "", "odds_type": got["odds"], "opp": opp,
                "is_combo": "+" in name, "league": None, "source": TYPED_SOURCE,
                "pick": got["pick"], "alt_pick": got["alt_pick"],
                "pair": " ".join(game) if game else None,
                "pair_loose": " ".join(loose) if loose and not game else None,
            })
        except Exception as e:  # noqa: BLE001 - one bad line costs one line
            logger.warning("typed-list line failed (%s)", type(e).__name__)
            attrs["skipped_error"] += 1
    return _frame(rows, attrs)


def _tipped(g: dict, now_et) -> bool:
    return bool(g["final"]) or (g["tip"] is not None and g["tip"] <= now_et)


def _tipped_within(g: dict, now_et, hours: float) -> bool:
    return (g["tip"] is not None and g["tip"] <= now_et
            and (now_et - g["tip"]).total_seconds() <= hours * 3600)


def _is_true(v) -> bool:
    return v is True or v is np.True_


def _heading_games(heads: set, games: list[dict], now_et, want: set, pairs=()
                   ) -> tuple[dict[str, str], dict[str, str], set]:
    """({team: opponent} for heading teams whose list is a game that has
    tipped, {team: opponent} for heading teams paired on a game to come,
    the loosely joined `pairs` the schedule makes one game), by typed_games's
    rules."""
    if len(heads) < 2:
        return {}, {}, set()
    today = now_et.date().isoformat()
    yday = (now_et.date() - dt.timedelta(days=1)).isoformat()
    last: dict[str, dict] = {}                # each team's latest recent game that has tipped
    nxt: dict[str, dict] = {}                 # each team's next game still to come in `want`
    plays_today: set[str] = set()
    mine = [g for g in games if g["home"] in heads or g["away"] in heads]
    for g in sorted(mine, key=lambda g: (g["tip"] is None, g["tip"] or now_et)):
        for t in (g["home"], g["away"]):
            if t not in heads:
                continue
            if _tipped(g, now_et):
                if g["date"] in (today, yday) or _tipped_within(g, now_et, TYPED_LATE_HOURS):
                    last[t] = g
            else:
                if g["date"] == today:
                    plays_today.add(t)
                if g["date"] in want and t not in nxt:
                    nxt[t] = g

    def other(g: dict, t: str) -> str:
        return g["away"] if g["home"] == t else g["home"]

    # "ORL-PHX" on one line names their game when the latest game of both,
    # or the next of both, is against each other: routed as "@" is, so last
    # night's list stays late after noon too (review 4, B4). Not by the
    # latest when both play again today: that list is today's, as it is for
    # two team headings (review 5, B9).
    pinned = {p for p in pairs if len(p) == 2 and (
        (last.get(p[0]) is not None and last.get(p[0]) is last.get(p[1])
         and not set(p) <= plays_today)
        or (nxt.get(p[0]) is not None and nxt.get(p[0]) is nxt.get(p[1])))}

    upcoming = {t: other(g, t) for t, g in nxt.items()
                if other(g, t) in heads and nxt.get(other(g, t)) is g}
    late: dict[str, str] = {}
    for t, g in last.items():
        o = other(g, t)
        if o not in heads or last.get(o) is not g:
            continue
        if upcoming.get(t, o) != o or upcoming.get(o, t) != t:
            continue                 # a game to come pairs one of them with ANOTHER heading
        recent = _tipped_within(g, now_et, TYPED_LATE_HOURS) or g["date"] == today
        # Last night's game, the morning after: that game's list sent late,
        # even when one of the two plays again today (a back-to-back, review
        # 3, F3), unless both do (the list is today's). From noon it is a
        # list for their next games (review 3, R5).
        morning = (g["date"] == yday and now_et.hour < TYPED_MORNING_AFTER_HOUR
                   and not {t, o} <= plays_today)
        if recent or morning:
            late[t] = o
    return late, {t: o for t, o in upcoming.items() if t not in late}, pinned


def typed_games(df: pd.DataFrame, schedule: pd.DataFrame, now=None, dates=None,
                headings=()) -> pd.DataFrame:
    """A typed line's team, opponent and pick, after its name is matched.

    The roster's team wins over a heading's: a heading only groups the list,
    so a player listed under the wrong team (a "Bonus" section, a forgotten
    heading) is filed under his own game, and his match is not reported as
    'another team'. An ESPN 'NO' before the number (alt_pick) was New
    Orleans for a Pelican and the market's No for anyone else.

    Then the game, when the line did not name one: the other team of its
    game heading, or a game two TEAM HEADINGS pair (_heading_games):
      - their latest game, when it was against each other and tipped today
        or within TYPED_LATE_HOURS, or tipped yesterday and it is not yet
        TYPED_MORNING_AFTER_HOUR (noon ET), unless both teams play again
        today. The list is that game's, sent late, and its lines are
        'tipped' (column `late`): the first list, pasted at 1:13 AM or the
        next morning, is reported as late, never filed under each team's
        next game. From noon the next day it is a list for their next
        games (review 3, R5).
      - else their next game in `dates`, when it is the NEXT game of both.
    A late pairing gives way when either team's next game pairs it with
    ANOTHER heading team (a list for tomorrow, sent the night two of its
    teams played), and never to a rematch of the same two. Two headings
    whose next games are against other teams name nothing, so tonight's list
    is never moved to a game the two play later in the week. Two teams one
    loose heading joins ("ORL-PHX", `pair_loose`) are a game heading when
    their latest game, or their next, is against each other, routed as "@"
    is, noon or not (review 4, B4). Rows from the feed or the board are
    returned as they are."""
    if df is None or df.empty or "source" not in df.columns:
        return df
    typed = df["source"].astype(str).eq(TYPED_SOURCE).to_numpy()
    if not typed.any():
        return df
    out = df.copy()
    now_et = to_et(now or now_utc())
    if dates is None:
        d0 = now_et.date()
        dates = [d0, d0 + pd.Timedelta(days=1)]
    want = {str(pd.Timestamp(d).date()) for d in dates}
    heads = {h for h in (teams.canonical(x) for x in headings or ()) if h}
    loose_col = out["pair_loose"] if "pair_loose" in out.columns else pd.Series(dtype=object)
    pairs = {tuple(sorted(p.split())) for p in loose_col.dropna()
             if isinstance(p, str) and len(p.split()) == 2}
    late, upcoming, pinned = _heading_games(
        heads, _schedule_frame(schedule).to_dict(orient="records"), now_et, want, pairs)
    for c in ("team", "team_code", "opp", "how", "pick", "late"):
        if c not in out.columns:
            out[c] = None
        out[c] = out[c].astype(object)
    for pos in np.flatnonzero(typed):
        i = out.index[pos]
        rec = out.loc[i]
        rt = teams.canonical(rec.get("roster_team"))
        tc = teams.canonical(rec.get("team_code"))
        matched = str(rec.get("match")) == "matched"
        alt = pick_of(rec.get("alt_pick"))
        if matched and alt is not None and rt != "NOP":
            out.at[i, "pick"] = alt                  # "NO" was the market's No
        if matched and rt and tc and rt != tc:
            out.at[i, "team"] = rt
            out.at[i, "team_code"] = rt
            how = str(rec.get("how") or "")
            if how.endswith("+other_team"):
                out.at[i, "how"] = how[:-len("+other_team")]
            tc = rt
        team = tc or rt
        opp = rec.get("opp")
        if not team or (isinstance(opp, str) and opp):
            continue
        pair = rec.get("pair")
        sides = pair.split() if isinstance(pair, str) else []
        joined = rec.get("pair_loose")
        if len(sides) != 2 and isinstance(joined, str) \
                and tuple(sorted(joined.split())) in pinned:
            sides = joined.split()          # "ORL-PHX": the game the schedule pairs them on
        if len(sides) == 2:
            out.at[i, "opp"] = (sides[1] if team == sides[0] else sides[0]) \
                if team in sides else None
            continue
        if team in late:
            out.at[i, "opp"] = late[team]
            out.at[i, "late"] = True
        elif team in upcoming:
            out.at[i, "opp"] = upcoming[team]
    return out


def _useful_rows(df: pd.DataFrame) -> int:
    """Rows whose label names a stat (priced or refused), not junk."""
    if df is None or df.empty:
        return 0
    return sum(_classify_row(s, n, c, lg)[1] != "unmapped" for s, n, c, lg in
               df[["stat_type", "name", "is_combo", "league"]].itertuples(index=False))


def _picked(df: pd.DataFrame) -> np.ndarray:
    """Rows that carry the visitor's own pick (a typed line's written side)."""
    if df is None or "pick" not in df.columns:
        return np.zeros(0 if df is None else len(df), dtype=bool)
    return df["pick"].map(pick_of).notna().to_numpy(dtype=bool)


def _with_typed_picks(board: pd.DataFrame, t: str) -> pd.DataFrame:
    """A board paste with the visitor's own picks typed into it keeps both:
    the squares as board rows, and each typed line that names a side as a
    pick (review 2, P10 and I2: the picks were dropped without a word). A
    typed reading with no side is left out: on board text it is a square the
    board rows already hold."""
    try:
        typed = parse_typed_lines(t, board=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("typed-list parse failed (%s); keeping the board", type(e).__name__)
        return board
    mine = typed[typed["pick"].map(pick_of).notna().to_numpy()] if not typed.empty \
        else typed
    out = pd.concat([board, mine], ignore_index=True) if not mine.empty else board.copy()
    out.attrs = dict(board.attrs)
    # The typed lines it could not read are counted, as when the list is
    # pasted alone (review 3, F13): only lines with a side, since a square's
    # words are no pick and name no player in full.
    out.attrs["typed_noname"] = int(typed.attrs.get("typed_noname_pick", 0) or 0)
    out.attrs["typed_unread"] = int(typed.attrs.get("typed_unread", 0) or 0)
    out.attrs["typed_recap"] = int(typed.attrs.get("typed_recap", 0) or 0)
    out.attrs["typed_unclear"] = int(typed.attrs.get("typed_unclear", 0) or 0)
    out.attrs["typed_period_label"] = int(typed.attrs.get("typed_period_label_pick", 0) or 0)
    out.attrs["typed_headings"] = list(typed.attrs.get("typed_headings") or [])
    return out


def parse_any(text) -> pd.DataFrame:
    """Feed JSON, board text or a typed list, whichever the paste is. NEVER raises.
    Board text is read as board text first; the typed reader then adds the
    picks typed into it (_with_typed_picks), or reads the whole paste when
    the board parser finds no square naming a stat.

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
        board = parse_prizepicks_board(t)
    except Exception as e:  # noqa: BLE001
        logger.warning("board-text parse failed (%s); returning nothing", type(e).__name__)
        board = _frame([], attrs)
    if _useful_rows(board):
        return _with_typed_picks(board, t)
    # Board text puts the number and the label on lines of their own, so a
    # paste with no square naming a stat may be a typed list ("Name over 6.5
    # Stat"), even when a stray number line in a copied post ("12", then
    # "Reposts") gave the board parser a junk square.
    try:
        typed = parse_typed_lines(t)
    except Exception as e:  # noqa: BLE001
        logger.warning("typed-list parse failed (%s); returning nothing", type(e).__name__)
        return board
    if _useful_rows(typed) or (board.empty and not typed.empty):
        return typed
    for k in ("typed_noname", "typed_unread", "typed_recap", "typed_unclear"):
        board.attrs[k] = int(typed.attrs.get(k, 0) or 0)
    return board


# -- Names -> PERSON_ID -------------------------------------------------------
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
            if (m.value is None and m.how == "none" and team is not None
                    and str(rec.get("source") or "") == TYPED_SOURCE):
                # A heading only groups a typed list, so 'Steph Curry' under
                # 'Suns:' is still Stephen Curry, and typed_games files him
                # under his own team (review 2, P14).
                m = idx.get(str(rec.get("name") or ""), team=None)
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


# -- Routing: each line to the ET date of its own game ------------------------
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
    if str(rec.get("source") or "") == TYPED_SOURCE and (
            _is_true(rec.get("late")) or (opp is not None and any(
                opp in (g["home"], g["away"]) and _tipped_within(g, now_et, TYPED_LATE_HOURS)
                for g in games))):
        # A typed list naming a game that tipped within the last few hours,
        # or whose two headings typed_games found to be a game that has
        # tipped (`late`), is that game's list, sent late, even when the two
        # teams meet again tomorrow. A board square for a started game is
        # live, never pasted.
        return None, None, "tipped", None, None
    cands = [g for g in games if g["date"] in want and not g["final"]
             and (g["tip"] is None or g["tip"] > now_et)
             and (opp is None or opp in (g["home"], g["away"]))]
    if cands:
        g = min(cands, key=lambda g: (g["date"], g["tip"] or now_et))
        return g["date"], g["game_id"], "team_schedule", _utc_iso(g["tip"]), bool(g["projectable"])
    if opp is not None:
        # The line names its game (a matchup line, a typed list's headings),
        # and that game, today's or last night's, has already started.
        recent = {now_et.date().isoformat(), (now_et.date() - dt.timedelta(days=1)).isoformat()}
        if any(opp in (g["home"], g["away"]) and g["date"] in recent
               and (g["final"] or (g["tip"] is not None and g["tip"] <= now_et))
               for g in games):
            return None, None, "tipped", None, None
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


# -- The alt-line ladder ------------------------------------------------------
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


def _period_repastes(out: pd.DataFrame, exempt=None) -> tuple[np.ndarray, np.ndarray]:
    """(held, below_pick): rows of `out` (in recency order, with _key and
    _rank) that are a standard line far below the standard line held before
    them, and of those the ones held for a typed pick's number alone.
    `exempt` rows (the visitor's own picks) are never held: a typed pick
    names its own number, and a period written on it was refused at the
    paste (review 2, I3). The board's own line is the standard; with none, a
    typed pick's number is, so a quarter square pasted below a typed
    full-game pick is held (review 3, R3). A pick never moves the board's
    standard: the full-game board re-pasted after a Demon typed above it is
    the same board, set aside for the pick, never held (review 4, A8, B5)."""
    held = np.zeros(len(out), dtype=bool)
    below_pick = np.zeros(len(out), dtype=bool)
    if "line" not in out.columns:
        return held, below_pick
    lines = pd.to_numeric(out["line"], errors="coerce").to_numpy(dtype=float)
    std = (np.asarray(out["_rank"]) == 0) if "_rank" in out.columns else np.ones(len(out), bool)
    skip = np.zeros(len(out), dtype=bool) if exempt is None else np.asarray(exempt, dtype=bool)
    board: dict[str, float] = {}
    picks: dict[str, float] = {}
    for i, k in enumerate(out["_key"].to_numpy()):
        x = lines[i]
        if not std[i] or not math.isfinite(x):
            continue
        if skip[i]:
            picks[k] = x                     # a pick is never held, and sets no board standard
            continue
        ref = board.get(k)
        mine = ref is None
        if mine:
            ref = picks.get(k)
        if (ref is not None and x < PERIOD_REPASTE_RATIO * ref
                and ref - x >= PERIOD_REPASTE_MIN_DROP):
            held[i], below_pick[i] = True, mine      # keep the full-game line (or the pick)
            continue
        board[k] = x
    return held, below_pick


def _carry_same_line(out: pd.DataFrame, skip=None) -> None:
    """In place, oldest to newest: a row of the SAME line (game date, player,
    stat, number and odds type) as an older one takes what the older one knew
    and it does not. A typed pick survives a board re-paste of its square, and
    a square's offered sides ('More only') survive a typed pick on it. Sides
    are carried only into a row that carries a pick, since a typed pick has
    none of its own: between board and feed rows the newest paste says what
    it offers, as before (review 2, I9). `skip` rows (held back as period
    lines) neither give nor take. A moved line carries neither: a pick made
    at 6.5 is not a pick at 7.5."""
    if "pick" not in out.columns or "line" not in out.columns:
        return
    has_dir = "direction" in out.columns
    lines = pd.to_numeric(out["line"], errors="coerce").to_numpy(dtype=float)
    odds = out["odds_type"].astype(str).str.lower().to_numpy() if "odds_type" in out.columns \
        else np.full(len(out), "standard", dtype=object)
    skip = np.zeros(len(out), dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    picks = out["pick"].to_numpy(dtype=object).copy()
    dirs = out["direction"].to_numpy(dtype=object).copy() if has_dir else None
    last_pick: dict[tuple, str] = {}
    last_dir: dict[tuple, str] = {}
    for i, key in enumerate(out["_key"].to_numpy()):
        if skip[i] or not math.isfinite(lines[i]):
            continue
        k = (key, round(float(lines[i]), 3), odds[i])
        if pick_of(picks[i]) is None and k in last_pick:
            picks[i] = last_pick[k]
        elif pick_of(picks[i]) is not None:
            last_pick[k] = picks[i]
        if has_dir:
            d = dirs[i] if isinstance(dirs[i], str) else ""
            if d.strip() == "" and k in last_dir and pick_of(picks[i]) is not None:
                dirs[i] = last_dir[k]
            elif d.strip().lower() in ("both", "more", "less"):
                last_dir[k] = d
    out["pick"] = pd.Series(picks, index=out.index, dtype=object)
    if has_dir:
        out["direction"] = pd.Series(dirs, index=out.index, dtype=object)


def _set_aside_for_picks(ranked: pd.DataFrame, kept: pd.DataFrame, new_from) -> pd.DataFrame:
    """Of collapse_alt_lines's sorted rows, per (date, player, stat) whose
    kept row is a typed pick: the board or feed line that would have won
    without it, when it came in at row `new_from` or later and sits at
    another number. The same number merged into the pick instead."""
    picks = kept[kept["_mine"] == 0]
    if picks.empty or "line" not in ranked.columns:
        return ranked.iloc[0:0]
    rivals = ranked[ranked["_mine"] == 1].drop_duplicates("_key", keep="first")
    rivals = rivals[rivals["_key"].isin(picks["_key"])]
    if "game_date" in rivals.columns:
        # A line placed on no game never showed: it is reported as unrouted,
        # never as set aside for a pick on the same empty date (review 4, B12).
        d = rivals["game_date"]
        rivals = rivals[d.notna() & ~d.astype(str).isin(["", "None", "nan", "NaT"])]
    if new_from is not None:
        rivals = rivals[rivals["_ord"] >= new_from]
    if rivals.empty:
        return rivals
    mine = pd.Series(pd.to_numeric(picks["line"], errors="coerce").to_numpy(dtype=float),
                     index=picks["_key"].to_numpy())
    theirs = pd.to_numeric(rivals["line"], errors="coerce").to_numpy(dtype=float)
    ours = rivals["_key"].map(mine).to_numpy(dtype=float)
    return rivals[~np.isclose(theirs, ours, equal_nan=True)]


def _line_name(rec: dict) -> str:
    stat = rec.get("stat")
    label = STAT_LABEL.get(stat) if isinstance(stat, str) else None
    return f"{rec.get('name') or ''} {label or rec.get('stat_type') or ''}".strip()[:60]


def collapse_alt_lines(df: pd.DataFrame, new_from: int | None = None) -> pd.DataFrame:
    """One line per (game date, player, stat): the visitor's own pick first,
    then standard, then Goblin, then Demon; among equals, the NEWEST paste
    wins.

    A typed pick outranks the board's line for the same player and stat
    because it is the bet the visitor asked about: a Goblin typed after the
    board, or a board pasted after the list at another number, no longer
    makes the pick vanish while the paste says "loaded" (review 2, I3). The
    same number merges into one row (_carry_same_line).

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
    `attrs['period_held_by_date']`, so the paste report can say so; those
    held below a typed pick's number alone are `period_held_pick` (named in
    `period_held_pick_names`), whose report names the pick (review 4, A8).

    Safe for the record because the record's FREEZE_KEYS are (game_id,
    person_id, stat), WITHOUT the line, and first seen wins: the pick made at
    24.5 stays frozen as the evidence it is, and the board moves to 26.5.

    Before the newest row wins, _carry_same_line hands it what an older row
    of the same number knew (a typed pick, a square's offered sides). The
    visitor's own picks are never held back as period lines.

    A board or feed line a typed pick outranks at ANOTHER number is set
    aside and counted, `attrs['pick_kept']` (per date in
    `pick_kept_by_date`, named in `pick_kept_names`), so the paste report
    never calls it loaded (review 3, R8). Only lines from row `new_from` on
    count (the paste being merged; every row when None): a pick typed over
    an older board line replaces it, as any newer paste does.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["_key"] = _collapse_key(out)
    out["_rank"] = out["odds_type"].map(lambda o: _ODDS_RANK.get(str(o).lower(), 3)) \
        if "odds_type" in out.columns else 0
    out["_ord"] = np.arange(len(out))
    held, below_pick = _period_repastes(out, exempt=_picked(out))
    _carry_same_line(out, skip=held)
    out["_mine"] = np.where(_picked(out), 0, 1)
    held_dates = (out.loc[held, "game_date"].astype(str).value_counts()
                  if "game_date" in out.columns else pd.Series(dtype=int))
    below_names = [_line_name(r) for r in out[below_pick].head(8).to_dict(orient="records")]
    ranked = out[~held].sort_values(["_key", "_mine", "_rank", "_ord"],
                                    ascending=[True, True, True, False], kind="mergesort")
    kept = ranked.drop_duplicates("_key", keep="first")
    aside = _set_aside_for_picks(ranked, kept, new_from)
    out = (kept.sort_values("_ord", kind="mergesort")
           .drop(columns=["_key", "_mine", "_rank", "_ord"])
           .reset_index(drop=True))
    out.attrs["period_held"] = int(held.sum())
    out.attrs["period_held_by_date"] = {str(k): int(v) for k, v in held_dates.items()}
    out.attrs["period_held_pick"] = int(below_pick.sum())
    out.attrs["period_held_pick_names"] = below_names
    aside_dates = (aside["game_date"].astype(str).value_counts()
                   if "game_date" in aside.columns else pd.Series(dtype=int))
    out.attrs["pick_kept"] = int(len(aside))
    out.attrs["pick_kept_by_date"] = {str(k): int(v) for k, v in aside_dates.items()}
    out.attrs["pick_kept_names"] = [_line_name(r) for r in
                                    aside.head(8).to_dict(orient="records")]
    return out


# -- One call from paste text to stored lines ---------------------------------
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

    and `tipped` (with `tipped_names`) counts the unrouted lines whose own
    game has already started.

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
    quarter or half lines (collapse_alt_lines); `pick_kept` (with
    `pick_kept_names`), board and feed lines set aside for a typed pick at
    another number in the same paste. Names are capped, never escaped here:
    the page escapes what it renders.
    """
    parsed = parse_any(text)
    report = {"parsed": int(len(parsed)), "parse": dict(parsed.attrs),
              "by_date": {}, "unrouted": 0, "unrouted_names": [], "kept": 0,
              "comparable": 0, "comparable_by_date": {},
              "not_comparable": {k: 0 for k in NOT_COMPARABLE},
              "other_boards": {}, "wrong_board": None, "period_held": 0,
              "pick_kept": 0, "pick_kept_names": []}
    if parsed.empty:
        return parsed, report
    df = _classify_and_resolve(parsed, roster)
    df = typed_games(df, schedule, now=now, dates=dates,
                     headings=parsed.attrs.get("typed_headings") or ())
    df = route_lines(df, schedule, now=now, dates=dates)
    df = collapse_alt_lines(df)
    report["period_held"] = int(df.attrs.get("period_held", 0))
    report["pick_kept"] = int(df.attrs.get("pick_kept", 0))
    report["pick_kept_names"] = list(df.attrs.get("pick_kept_names") or [])
    vc = df["game_date"].dropna().value_counts()
    report["by_date"] = {str(k): int(v) for k, v in sorted(vc.items())}
    unr = df[df["game_date"].isna()]
    report["unrouted"] = int(len(unr))
    report["unrouted_names"] = sorted({str(x) for x in unr["name"]})[:8]
    # Of those, the lines whose own game has already started ('tipped'):
    # said apart from the rest, so a list sent in after the game reads as
    # late, not as unknown names.
    tipped = unr[unr["route_how"].astype(str).eq("tipped")]
    report["tipped"] = int(len(tipped))
    report["tipped_names"] = sorted({str(x) for x in tipped["name"]})[:8]
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


# -- The visitor's board: in-process, token-keyed, never on the server's disk -
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
                 "roster_name", "roster_team", "game_date", "game_id", "tip_utc", "pick")
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
    pick_kept: dict = field(default_factory=dict)     # {game_date: lines} that write set
                                                      # aside for a typed pick
    pick_kept_names: list = field(default_factory=list)
    period_held_pick: int = 0   # of period_held, the lines held below a typed pick alone
    period_held_pick_names: list = field(default_factory=list)


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


def _without_feed_rows(lines):
    """`lines` less every row of the licensed feed (build amendment B14): a
    line whose line-level `source` is FEED_SOURCE, or a compared row whose
    `origin` is "feed" (compare's mark of the same thing). The token board
    and its browser copy hold only the visitor's paste (design 2.5, RC R5).
    The feed path never hands them a feed row; this makes a later mistake
    that does harmless. Anything else, a non-frame included, passes as it
    is."""
    if not isinstance(lines, pd.DataFrame) or lines.empty:
        return lines
    feed = np.zeros(len(lines), dtype=bool)
    for col, mark in (("source", FEED_SOURCE), ("origin", "feed")):
        if col in lines.columns:
            feed |= np.array([isinstance(v, str) and v == mark for v in lines[col]], dtype=bool)
    return lines[~feed] if feed.any() else lines


def _cap(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(the stored frame, rows dropped by BOARD_MAX_ROWS). Compact columns
    only, and never a feed row (B14): every write to the store comes
    through here."""
    lines = _without_feed_rows(lines if lines is not None else pd.DataFrame())
    keep = [c for c in BOARD_COLUMNS if c in lines.columns]
    lines = lines[keep]
    dropped = max(0, len(lines) - BOARD_MAX_ROWS)
    if dropped:
        lines = lines.iloc[-BOARD_MAX_ROWS:]      # newest pastes are last
    return lines.reset_index(drop=True).copy(), dropped


def _store_locked(token: str, lines: pd.DataFrame, now: float,
                  period_held: dict | None = None, pasted_utc=None,
                  pick_kept: dict | None = None, pick_kept_names=None,
                  period_held_pick: int = 0, period_held_pick_names=None) -> None:
    stored, dropped = _cap(lines)
    _BOARDS[token] = _Board(stored, now, int(stored.memory_usage(deep=True).sum()), dropped,
                            dict(period_held or {}),
                            pasted_utc if pasted_utc is not None else _wall(),
                            dict(pick_kept or {}), list(pick_kept_names or [])[:8],
                            int(period_held_pick or 0), list(period_held_pick_names or [])[:8])
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
    changing the board in silence. `pick_kept` is how many pasted lines that
    write set aside for a typed pick at another number (per date in
    `pick_kept_by_date`, named in `pick_kept_names`), and `period_held_pick`
    how many of the held lines sat below a typed pick alone (named in
    `period_held_pick_names`). `pasted_utc` is when that write happened (the
    visitor's last paste, an aware UTC datetime), for the board to show."""
    if not valid_token(token):
        return None
    with _BOARDS_LOCK:
        b = _BOARDS.get(token)
        if b is None:
            return None
        return {"rows": int(len(b.lines)), "nbytes": int(b.nbytes), "dropped": int(b.dropped),
                "period_held": int(sum(b.period_held.values())),
                "period_held_by_date": dict(b.period_held), "pasted_utc": b.pasted_utc,
                "pick_kept": int(sum(b.pick_kept.values())),
                "pick_kept_by_date": dict(b.pick_kept),
                "pick_kept_names": list(b.pick_kept_names),
                "period_held_pick": int(b.period_held_pick),
                "period_held_pick_names": list(b.period_held_pick_names)}


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
    it again (an old or shared link) moves it back.

    A feed row in `lines` is dropped before the merge (build amendment B14;
    _cap drops one on every write too), so it can never displace the
    board's own pasted line for the same player and stat."""
    lines = _without_feed_rows(lines)
    with _BOARDS_LOCK:
        now = _clock()
        _evict_locked(now)
        b = _BOARDS.get(token) if valid_token(token) else None
        old = b.lines if b is not None and not b.lines.empty else None
        parts = [x for x in (old, lines) if x is not None and not x.empty]
        # The board's rows come first, so the pasted ones start at len(old).
        merged = collapse_alt_lines(pd.concat(parts, ignore_index=True),
                                    new_from=len(old) if old is not None else 0) if parts \
            else pd.DataFrame()
        held = dict(merged.attrs.get("period_held_by_date") or {}) if parts else {}
        kept = dict(merged.attrs.get("pick_kept_by_date") or {}) if parts else {}
        names = list(merged.attrs.get("pick_kept_names") or []) if parts else []
        below = int(merged.attrs.get("period_held_pick") or 0) if parts else 0
        below_names = list(merged.attrs.get("period_held_pick_names") or []) if parts else []
        if b is not None:
            _BOARDS.move_to_end(token, last=False)
        new = secrets.token_urlsafe(24)
        _store_locked(new, merged, now, period_held=held, pasted_utc=pasted_utc,
                      pick_kept=kept, pick_kept_names=names, period_held_pick=below,
                      period_held_pick_names=below_names)
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


# -- The visitor's board, kept in THEIR browser (Barrett, 2026-11-05) --------
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
# Version 2 (2026-11-06) adds `pick`, the side a typed list named. A version 1
# copy, saved before it, still restores: its lines come back with no pick.
SAVED_VERSION = 2
SAVED_COLUMNS = ("name", "team_code", "stat_type", "line", "direction", "odds_type",
                 "opp", "is_combo", "league", "game_date", "game_id", "tip_utc", "pick")
SAVED_LAYOUTS = {1: SAVED_COLUMNS[:-1], 2: SAVED_COLUMNS}
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
    `today` (ET) or later, in the layout of its version (SAVED_LAYOUTS). `save_id` is what the
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
    recs = t.to_dict(orient="records")
    # A board with no typed pick is saved in version 1, which the code before
    # picks also reads: a rollback, or an older process, keeps such a copy
    # instead of refusing it and clearing it from the browser.
    v = SAVED_VERSION if any(pick_of(r.get("pick")) for r in recs) else 1
    cols = SAVED_LAYOUTS[v]
    rows = [[_json_value(rec.get(c)) for c in cols] for rec in recs]
    return {"v": v, "id": str(save_id), "owner": owner,
            "cols": list(cols), "pasted_utc": _utc_iso(_parse_ts(pasted_utc)),
            "rows": rows}


def _saved_row(row, lo: dt.date, hi: dt.date, cols=SAVED_COLUMNS) -> dict | None:
    """One kept line back as a parsed line (the shape _classify_and_resolve
    reads), or None. Every field through the paste validators. `cols` is the
    copy's own layout (SAVED_LAYOUTS)."""
    if not isinstance(row, list) or len(row) != len(cols):
        return None
    r = dict(zip(cols, row))
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
            "tip_utc": _utc_iso(_parse_ts(tip)) if isinstance(tip, str) and len(tip) <= 40 else None,
            "pick": pick_of(r.get("pick"))}


def _saved_row_or_none(row, lo: dt.date, hi: dt.date, cols=SAVED_COLUMNS) -> dict | None:
    """One bad row costs one row, as at every paste stage."""
    try:
        return _saved_row(row, lo, hi, cols)
    except Exception:  # noqa: BLE001 - untrusted cell types
        return None


def restore_saved(payload, roster: pd.DataFrame, today: dt.date, now: dt.datetime
                  ) -> tuple[pd.DataFrame, dt.datetime, int] | None:
    """(lines, pasted_utc, rows dropped) from the browser's copy of a board,
    ready for add_to_board(None, lines, pasted_utc=...), or None when nothing
    usable comes back. NEVER raises: the payload is whatever a browser sent.

    A copy in any SAVED_LAYOUTS version is read in its own layout. The whole
    copy is refused when its shape is wrong, it holds more rows than
    a board can (BOARD_MAX_ROWS), or its paste time is unreadable, in the
    future or older than SAVED_MAX_AGE_DAYS. A row is dropped when any field
    fails the paste validators or its game is not between `today` and
    SAVED_AHEAD_DAYS later. Names are resolved again against `roster`."""
    try:
        if not isinstance(payload, dict):
            return None
        v = payload.get("v")
        cols = SAVED_LAYOUTS.get(v) if type(v) is int else None
        if cols is None or payload.get("cols") != list(cols):
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
        recs = [r for r in (_saved_row_or_none(row, today, hi, cols) for row in rows)
                if r is not None]
        if not recs:
            return None
        lines = _classify_and_resolve(pd.DataFrame(recs), roster)
        return lines, pasted.astimezone(UTC), len(rows) - len(recs)
    except Exception as e:  # noqa: BLE001 - a browser's copy must never raise into the page
        logger.warning("restore_saved: refused (%s)", type(e).__name__)
        return None


# -- Per-stat reliability -----------------------------------------------------
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


# -- Distributions: what compare() needs from the model side ------------------
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


# -- The lean -----------------------------------------------------------------
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


# -- The licensed odds feed: its books and columns (design 2.5, layout B) -----
# The page hands this module the feed's lines as a DataFrame, exactly as it
# hands it the schedule mark: nothing here fetches, and this module imports
# nothing of the feed (tests/test_props.py's guards). The rows come from The
# Odds API under licence (Barrett, 2026-11-09), for five books. The board shows
# ONE book at a time, picked with a row of pills (layout B, amendment A1), and
# lists the other four books' lines beside each row.
FEED_SOURCE = "oddsapi"      # the line-level `source` of a feed row; TYPED_SOURCE is untouched
PASTE_BOOK = "prizepicks"    # every pasted line is a PrizePicks line, as before the feed
# The pill order and the Other books order (amendment A2). sources.oddsapi
# keeps the same five as BOOKS and the same titles (tests/test_props_feed.py
# pins both), and this module never imports it.
BOOK_ORDER = ("prizepicks", "underdog", "draftkings", "fanduel", "betmgm")
BOOK_TITLE = {"prizepicks": "PrizePicks", "underdog": "Underdog", "draftkings": "DraftKings",
              "fanduel": "FanDuel", "betmgm": "BetMGM"}
BOOK_CODE = {"prizepicks": "PP", "underdog": "UD", "draftkings": "DK", "fanduel": "FD",
             "betmgm": "MGM"}
# Pick'em books. Their prices are indicative only, so a DFS row never carries a
# price, and their side words are More and Less (side_words).
DFS_BOOKS = frozenset({"prizepicks", "underdog"})
# The columns prepare_feed needs from the feed frame (sources.oddsapi's
# FEED_COLUMNS hold them all; a contract test pins it).
FEED_REQUIRED = ("name", "stat_type", "line", "start_time", "direction", "odds_type", "league",
                 "source", "book", "book_title", "price_over", "price_under", "last_update",
                 "fetched_at", "event_id", "event_home", "event_away", "game_id", "game_date",
                 "tip_utc")
# What compare() adds to every row for the feed: the book (PASTE_BOOK for a
# pasted line), its title, the origin ("feed" or "pasted"), the two prices (NaN
# on DFS rows and pasted rows) and the feed's two stamps (None on pasted rows).
TABLE_EXTRA = ("book", "book_title", "origin", "price_over", "price_under", "fetched_at",
               "last_update")


# -- Compare ------------------------------------------------------------------
TABLE_COLUMNS = [
    "person_id", "game_id", "game_date", "player", "team", "opp", "stat",
    "stat_label", "line", "model", "mean", "gap", "p_over", "p_push", "p_under",
    "q_more", "q_less", "confidence", "lean", "sides", "odds_type", "tier",
    "tier_edge", "tier_note", "games_prior", "thin", "void_risk", "mu_min",
    "void_note", "source", "how", "pasted_name", "model_version", "as_of",
    "provisional", "adjusted", "tip_utc", "tie_possible", "pick",
    *TABLE_EXTRA,
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


def _line_book(rec) -> str:
    """The book a line came from: a feed row's own key, else PASTE_BOOK (a
    pasted line, and every line before the feed, is a PrizePicks line)."""
    b = rec.get("book") if hasattr(rec, "get") else None
    return b.strip().lower() if isinstance(b, str) and b.strip() else PASTE_BOOK


def _utc_stamp(v) -> pd.Timestamp | None:
    """An aware UTC Timestamp, or None for a stamp that is missing, naive or
    unreadable (_parse_ts's rules)."""
    ts = _parse_ts(v)
    return None if ts is None else pd.Timestamp(ts).tz_convert("UTC")


def _line_extra(rec) -> dict:
    """compare()'s TABLE_EXTRA cells for one line (design 2.5). The title is
    this module's own for the five books, so a vendor's spelling never
    reaches the page. A DFS row carries no price: its prices are indicative
    only, so nothing may do price math on them."""
    book = _line_book(rec)
    title = BOOK_TITLE.get(book)
    if title is None:
        own = rec.get("book_title")
        title = own if isinstance(own, str) and own else book
    dfs = book in DFS_BOOKS
    return {"book": book, "book_title": title,
            "origin": "feed" if str(rec.get("source") or "") == FEED_SOURCE else "pasted",
            "price_over": float("nan") if dfs else _finite_or_nan(rec.get("price_over")),
            "price_under": float("nan") if dfs else _finite_or_nan(rec.get("price_under")),
            "fetched_at": _utc_stamp(rec.get("fetched_at")),
            "last_update": _utc_stamp(rec.get("last_update"))}


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
    """Every line, pasted or prepared from the odds feed (prepare_feed),
    against its distribution. Returns (table, meta).

    Each row also says which book's line it is and where it came from
    (TABLE_EXTRA): `book` (PASTE_BOOK for a pasted line), `book_title`,
    `origin` ("feed" for a row whose `source` is FEED_SOURCE, else
    "pasted"), the prices (NaN on a DFS row), and the feed's `fetched_at`
    and `last_update` (None on a pasted row).

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
                # The side a typed list named: shown beside the lean, never
                # used to price, rank or hide the line.
                "pick": pick_of(rec.get("pick")),
                # Which book's line this is, where it came from and, for a
                # feed row, its prices and stamps (TABLE_EXTRA).
                **_line_extra(rec),
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
        # The visitor's own picks are never withheld, and lend the suspicion
        # their players but never their lines: a typed Goblin list sits far
        # below the medians by design (review 2, I3), while two quarter
        # squares beside a typed pick are still a quarter board (review 3,
        # R3). A period written on a pick, or on its section label, was
        # refused at the paste.
        held = _period_suspect_rows(table, _picked(table))
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


def _period_group_suspect(t: pd.DataFrame, picked=None) -> bool:
    """PERIOD_MIN_PLAYERS players, and the median player's line/mean ratio
    under PERIOD_RATIO. The visitor's own picks (`picked`) count as players
    but never set the ratio, and at least two board players must show it
    themselves: a pick completes a period board, never makes one. Goblin and
    Demon rungs sit off the standard number by design, so they count as
    players but set the ratio only when fewer than two standard players do,
    and then against a bar twice as strict: two Goblins and a pick are no
    quarter board (review 4, B6), while a quarter board copied as text with
    its Goblin rungs and one standard square still is (review 5, B8)."""
    mean = pd.to_numeric(t["mean"], errors="coerce").to_numpy(dtype=float)
    line = pd.to_numeric(t["line"], errors="coerce").to_numpy(dtype=float)
    pk = np.zeros(len(t), dtype=bool) if picked is None else np.asarray(picked, dtype=bool)
    alt = (t["odds_type"].astype(str).str.lower().isin(["goblin", "demon"]).to_numpy()
           if "odds_type" in t.columns else np.zeros(len(t), dtype=bool))
    keep = (mean > 2.0) & np.isfinite(line)
    pid = pd.Series(t["person_id"].to_numpy(dtype=object))

    def player_ratios(rows: np.ndarray) -> pd.Series:
        if not rows.any():
            return pd.Series(dtype=float)
        return pd.Series(line[rows] / mean[rows]).groupby(pid[rows].to_numpy()).median()

    ratio, bar = player_ratios(keep & ~pk & ~alt), PERIOD_RATIO
    if len(ratio) < 2:
        ratio, bar = player_ratios(keep & ~pk), PERIOD_RATIO / 2
    if len(ratio) < 2 or float(ratio.median()) >= bar:
        return False
    return int(pid[keep].dropna().nunique()) >= PERIOD_MIN_PLAYERS


def _period_suspect_rows(table: pd.DataFrame, picked=None) -> np.ndarray:
    """Rows of a compare() table that belong to a suspected period board,
    never the visitor's own picks (`picked`)."""
    if table is None or table.empty:
        return np.zeros(0, dtype=bool)
    pk = np.zeros(len(table), dtype=bool) if picked is None else np.asarray(picked, dtype=bool)
    if _period_group_suspect(table, pk):
        return ~pk
    mask = np.zeros(len(table), dtype=bool)
    for _gid, idx in table.groupby(table["game_id"].astype(str), sort=False).indices.items():
        if _period_group_suspect(table.iloc[idx], pk[idx]):
            mask[idx] = ~pk[idx]
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


# -- The licensed odds feed: lines, tables and the board (design 2.5) --------
# The page hands this module the feed's frame (oddsapi's FEED_COLUMNS, one
# row per book, player and market, every row a standard full-game line).
#   prepare_feed   gives it the shape prepare_paste gives a paste: stat
#                  verdicts, names resolved inside the event's two teams, ISO
#                  game dates, one row per game date, player, stat and book.
#                  It never routes by the clock: the event match placed every
#                  row on its game, and route_lines drops started games (T2).
#   compare_books  runs compare() once per (book, game).
#   feed_board     takes the SELECTED book's rows for every game (layout B,
#                  amendment A1), merges the visitor's paste into PrizePicks'
#                  rows (newest wins, M5), and lists the other books' rows
#                  for the same keys as context. Its meta balances under
#                  ledger(meta, *board_rows(reference)) for every pick.
# Storage stays apart (RC R5): the visitor's token board and its browser copy
# hold only the paste. Nothing here calls new_board, add_to_board,
# saved_board or restore_saved, and feed rows never reach _cap.
FEED_REPORT_NAMES = 12        # unmatched and ambiguous names listed per book
FEED_META_NAMES = 50          # compare()'s own cap on a meta's name lists
_FALLBACK_HOWS = ("exact", "alias")    # what the other-team fallback accepts (m20, B13)
_PREPARED_COLUMNS = ("stat", "kind", "refusal", "team", "team_code", "match", "person_id",
                     "how", "roster_name", "roster_team", "route_how", "projectable")
_SUMMED = ("posted", "matched", "unmatched", "ambiguous", "unprojected", "n_refused",
           "errors", "period_withheld")
_TALLIED = ("refused", "unmapped", "unprojected_why")


def _iso_game_date(v) -> str | None:
    """A game date as the ISO string a pasted row carries (route_lines), from
    a python date (the schedule's, pipeline.state_games), a Timestamp or an
    ISO string; None when it is not a date (m3)."""
    if isinstance(v, str):
        s = v.strip()[:10]
        if not _ISO_DATE.match(s):
            return None
        try:
            return dt.date.fromisoformat(s).isoformat()
        except ValueError:
            return None
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(v, dt.datetime):            # pd.Timestamp included
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return None


def _game_id_of(v) -> str | None:
    if isinstance(v, str):
        return v.strip() or None
    return None


def _in_game_pairs(projections) -> set:
    """{(person_id, game_id)} for every projection row: who plays in which game."""
    if projections is None or not isinstance(projections, pd.DataFrame) or projections.empty \
            or not {"person_id", "game_id"} <= set(projections.columns):
        return set()
    out = set()
    for pid, gid in projections[["person_id", "game_id"]].itertuples(index=False):
        p = _as_pid(pid)
        if p is not None and gid is not None:
            out.add((p, str(gid)))
    return out


def resolve_event_names(lines: pd.DataFrame, roster: pd.DataFrame,
                        projections: pd.DataFrame | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(resolved, unresolved) for feed lines, in resolve_names' shape. Never
    drops a row.

    A feed line names its event's two teams (event_away, event_home), never
    the player's own, so resolve_names' one-team scope cannot apply, and with
    no team an exact name two players share is ambiguous (RC G9). So, per
    event:
      1. one build_name_index over the two teams' roster rows, looked up
         with team=None: alias, exact, then the unique given-name prefix,
         all inside those two teams (NameIndex.get);
      2. if that finds nobody, an index over the roster rows of the players
         with a projection row in THIS line's game only, and only an exact or
         alias match there (never a prefix): a roster lagging a trade while
         the projections already have him on his new team (m20). Looking
         only at this game's players (build amendment B13), a namesake
         anywhere else in the league never makes such a line ambiguous or
         unmatched. Such rows carry `how` + "+other_team", as resolve_names
         marks a player the roster puts on another team.
    Everything else is unresolved, with reason "ambiguous" or "unmatched".
    Resolved rows get team = team_code = the roster's team."""
    cols = list(lines.columns) if isinstance(lines, pd.DataFrame) else []
    res_cols = cols + [c for c in ("person_id", "how", "roster_name", "roster_team",
                                   "team", "team_code") if c not in cols]
    unres_cols = cols + [c for c in ("reason", "how") if c not in cols]
    if not isinstance(lines, pd.DataFrame) or lines.empty:
        return pd.DataFrame(columns=res_cols), pd.DataFrame(columns=unres_cols)
    have_roster = isinstance(roster, pd.DataFrame) and not roster.empty
    look = _roster_lookup(roster) if have_roster else {}
    codes = roster["team"].map(teams.canonical) if have_roster else pd.Series(dtype=object)
    in_game = _in_game_pairs(projections)
    roster_pids = roster["person_id"].map(_as_pid) if have_roster else pd.Series(dtype=object)
    by_pair: dict = {}
    by_game: dict = {}            # game_id -> index of the roster rows projected in that game
    res, unres = [], []
    for rec in lines.to_dict(orient="records"):
        try:
            away = teams.canonical(rec.get("event_away"))
            home = teams.canonical(rec.get("event_home"))
            name = str(rec.get("name") or "")
            if away is None or home is None or away == home or not have_roster:
                unres.append({**rec, "reason": "unmatched", "how": "none"})
                continue
            idx = by_pair.get((away, home))
            if idx is None:
                idx = by_pair[(away, home)] = build_name_index(roster[codes.isin([away, home])])
            m = idx.get(name, team=None)
            pid, how, last = (None if m.value is None else int(m.value)), m.how, m.how
            if pid is None and m.how != "ambiguous":
                gid = str(rec.get("game_id"))
                gidx = by_game.get(gid)
                if gidx is None:                  # B13: this game's projected players only
                    here = {p for p, g in in_game if g == gid}
                    gidx = by_game[gid] = build_name_index(roster[roster_pids.isin(here)])
                f = gidx.get(name, team=None)
                last = f.how
                if (f.value is not None and f.how in _FALLBACK_HOWS
                        and (int(f.value), gid) in in_game):
                    pid, how = int(f.value), f"{f.how}+other_team"
        except Exception as e:  # noqa: BLE001 - one bad row costs one row (audit)
            logger.warning("resolve_event_names: row failed (%s)", type(e).__name__)
            unres.append({**rec, "reason": "unmatched", "how": "error"})
            continue
        if pid is None:
            reason = "ambiguous" if last == "ambiguous" else "unmatched"
            unres.append({**rec, "reason": reason, "how": "ambiguous" if reason == "ambiguous"
                          else "none"})
            continue
        rname, rteam = look.get(pid, (None, None))
        res.append({**rec, "person_id": pid, "how": how, "roster_name": rname,
                    "roster_team": rteam, "team": rteam, "team_code": rteam})
    resolved = pd.DataFrame(res, columns=res_cols) if res else pd.DataFrame(columns=res_cols)
    unresolved = pd.DataFrame(unres, columns=unres_cols) if unres \
        else pd.DataFrame(columns=unres_cols)
    return resolved, unresolved


def _empty_prepared() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FEED_REQUIRED)
                        + [c for c in _PREPARED_COLUMNS if c not in FEED_REQUIRED])


def _new_feed_report() -> dict:
    return {"error": None, "rows": 0, "kept": 0, "comparable": 0, "not_a_person": 0,
            "bad_rows": 0, "duplicates": 0, "books": {}, "games": {}, "parser": {}}


def _book_tally() -> dict:
    return {"rows": 0, "kept": 0, "comparable": 0, "unmatched": 0, "ambiguous": 0,
            "refused": 0, "not_a_person": 0, "bad_rows": 0, "duplicates": 0,
            "unmatched_names": [], "ambiguous_names": [], "refused_labels": {}}


def _feed_keys(df: pd.DataFrame) -> pd.Series:
    """One key per (game date, player, stat, book): the person_id when the
    name resolved, else the normalized name; the stat when it has one, else
    the label's key."""
    who = [f"p{int(p)}" if _as_pid(p) is not None else
           "n:" + normalize_name(n if isinstance(n, str) else "")
           for p, n in zip(df["person_id"], df["name"])]
    stat = [s if isinstance(s, str) and s else label_key(t)
            for s, t in zip(df["stat"], df["stat_type"])]
    return pd.Series([f"{d}|{w}|{s}|{b}" for d, w, s, b in
                      zip(df["game_date"], who, stat, df["book"])], index=df.index)


def prepare_feed(feed, roster, projections) -> tuple[pd.DataFrame, dict]:
    """The feed's lines, in the shape prepare_paste leaves a paste in, ready
    for compare_books: (lines, report). NEVER raises.

    1. None, a frame with no rows, or one missing a FEED_REQUIRED column: an
       empty frame and report["error"] ("no feed", "no lines" or "missing
       columns", with the missing names in report["missing"]).
    2. game_date becomes the ISO string a pasted row carries (m3), game_id
       a string, tip_utc an ISO UTC string (a pasted row's form), and
       fetched_at and last_update aware UTC Timestamps (None when
       unreadable). A row with no game, no date, no book, or a line outside
       the paste rule (finite, 0 to MAX_LINE_VALUE) is dropped and counted
       (bad_rows); so is a name that is not a person (not_a_person), as the
       paste parsers drop one.
    3. Each line's stat verdict (stat, kind, refusal), as _classify_row
       gives a paste.
    4. Names through resolve_event_names, filling what _classify_and_resolve
       fills for a paste (match, person_id, how, roster_name, roster_team,
       team_code).
    5. No routing by the clock: game_id, game_date and tip_utc are the event
       match's, route_how is "event" and projectable is True (only projected
       games are bought).
    6. One row per (game_date, person_id, stat, book), the newest fetched_at
       winning; duplicates are counted (the parser keeps one line per book,
       player and market, so they are expected to be 0).
    7. The report: per book under "books" (rows, kept, comparable,
       unmatched, ambiguous, refused, the first FEED_REPORT_NAMES unmatched
       and ambiguous names, refused labels, not_a_person, bad_rows,
       duplicates), the same totals at the top, each game's date, tip and
       event teams under "games", and the parser's counters under "parser"
       when the frame still carries them in its attrs.
    The report also rides on the lines as attrs["feed_report"], which is
    where compare_books finds it. Names are capped, never escaped here: the
    page escapes what it renders."""
    report = _new_feed_report()
    try:
        lines = _prepare_feed(feed, roster, projections, report)
    except Exception as e:  # noqa: BLE001 - whatever a page hands in, never a traceback
        logger.warning("prepare_feed failed (%s)", type(e).__name__)
        report["error"] = "internal error"
        lines = _empty_prepared()
    lines.attrs = {"feed_report": report}
    return lines, report


def _prepare_feed(feed, roster, projections, report: dict) -> pd.DataFrame:
    if not isinstance(feed, pd.DataFrame):
        report["error"] = "no feed"
        return _empty_prepared()
    missing = [c for c in FEED_REQUIRED if c not in feed.columns]
    if missing:
        report["error"], report["missing"] = "missing columns", missing
        return _empty_prepared()
    if feed.empty:
        report["error"] = "no lines"
        return _empty_prepared()
    by_book = feed.attrs.get("by_book") if isinstance(feed.attrs, dict) else None
    if isinstance(by_book, dict):
        report["parser"] = {str(b): dict(t) for b, t in by_book.items() if isinstance(t, dict)}
    df = feed.reset_index(drop=True).copy()
    df.attrs = {}
    df["book"] = [b.strip().lower() if isinstance(b, str) and b.strip() else None
                  for b in df["book"]]
    df["game_id"] = [_game_id_of(g) for g in df["game_id"]]
    df["game_date"] = [_iso_game_date(d) for d in df["game_date"]]
    df["tip_utc"] = [_utc_iso(_parse_ts(t)) for t in df["tip_utc"]]
    df["line"] = [_parse_line(x) for x in df["line"]]
    for c in ("fetched_at", "last_update"):
        df[c] = pd.Series([_utc_stamp(t) for t in df[c]], index=df.index, dtype=object)
    books = report["books"]
    report["rows"] = len(df)
    for b in df["book"]:
        if (b or "unknown") not in books:
            books[b or "unknown"] = _book_tally()
        books[b or "unknown"]["rows"] += 1
    bad = (df["book"].isna() | df["game_id"].isna() | df["game_date"].isna()
           | df["line"].isna()).to_numpy()
    person = np.array([_person_like(n) for n in df["name"]], dtype=bool)
    for b in df.loc[bad, "book"]:
        books[b or "unknown"]["bad_rows"] += 1
    for b in df.loc[~bad & ~person, "book"]:
        books[b or "unknown"]["not_a_person"] += 1
    report["bad_rows"] = int(bad.sum())
    report["not_a_person"] = int((~bad & ~person).sum())
    df = df[~bad & person].reset_index(drop=True)
    if df.empty:
        return _empty_prepared()
    df["line"] = df["line"].astype(float)

    df["_i"] = np.arange(len(df))
    is_combo = df["is_combo"] if "is_combo" in df.columns else pd.Series(False, index=df.index)
    kinds = [_classify_row(t, n, c, lg) for t, n, c, lg in
             zip(df["stat_type"], df["name"], is_combo, df["league"])]
    df["stat"] = [k[0] for k in kinds]
    df["kind"] = [k[1] for k in kinds]
    df["refusal"] = [k[2] for k in kinds]
    combo = df["kind"].eq("refused") & df["refusal"].eq(REFUSE_COMBO)
    resolved, unresolved = resolve_event_names(df[~combo], roster, projections)
    parts = []
    if not resolved.empty:
        parts.append(resolved.assign(match="matched"))
    if not unresolved.empty:
        parts.append(unresolved.rename(columns={"reason": "match"}))
    if combo.any():
        parts.append(df[combo].assign(match="refused", how="combo"))
    df = _stack(parts)
    for c in ("person_id", "roster_name", "roster_team", "team", "team_code"):
        if c not in df.columns:
            df[c] = None
    df["person_id"] = pd.Series([_as_pid(p) for p in df["person_id"]], index=df.index,
                                dtype=object)
    df["route_how"] = "event"
    df["projectable"] = True

    # One row per (game date, player, stat, book): the newest fetched_at wins.
    stamps = [_utc_stamp(t) for t in df["fetched_at"]]
    df["_t"] = [s.value if s is not None else -(2 ** 62) for s in stamps]
    df["_k"] = _feed_keys(df)
    ranked = df.sort_values(["_k", "_t", "_i"], ascending=[True, False, True], kind="mergesort")
    dup = ranked.duplicated("_k", keep="first")
    for b in ranked.loc[dup, "book"]:
        books[b]["duplicates"] += 1
    report["duplicates"] = int(dup.sum())
    df = (ranked[~dup].sort_values("_i", kind="mergesort")
          .drop(columns=["_i", "_t", "_k"]).reset_index(drop=True))

    in_game = _in_game_pairs(projections)
    for rec in df.to_dict(orient="records"):
        t = books[rec["book"]]
        t["kept"] += 1
        match, kind = str(rec.get("match")), str(rec.get("kind"))
        if kind != "ok":
            t["refused"] += 1
            label = str(rec.get("stat_type") or "")[:MAX_LABEL_LEN]
            _bump(t["refused_labels"], label)
        elif match in ("unmatched", "ambiguous"):
            t[match] += 1
            names = t[f"{match}_names"]
            if len(names) < FEED_REPORT_NAMES and rec.get("name") not in names:
                names.append(str(rec.get("name")))
        elif match == "matched" and (not in_game or (rec["person_id"], rec["game_id"]) in in_game):
            t["comparable"] += 1
        gid = rec["game_id"]
        if gid not in report["games"]:
            report["games"][gid] = {"game_date": rec["game_date"], "tip_utc": rec["tip_utc"],
                                    "event_away": teams.canonical(rec.get("event_away")),
                                    "event_home": teams.canonical(rec.get("event_home"))}
    report["kept"] = len(df)
    report["comparable"] = sum(t["comparable"] for t in books.values())
    return df


@dataclass
class FeedTables:
    """compare() per (book, game_id): `tables` and `metas` keyed alike, and
    prepare_feed's report. Read-only for callers (the page caches it).

    `failed` marks a date whose feed holds lines that could not be compared
    (store.feed_tables: the simulator parameters missing, a step that raised,
    or no line left to compare). It has no table and no meta, so no book has
    a board to pick, and the page's board explains the failure (props_ui's
    failed view) instead of promising lines the feed already holds."""
    tables: dict = field(default_factory=dict)
    metas: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    failed: bool = False


def _game_projections(projections, gid: str) -> pd.DataFrame | None:
    if not isinstance(projections, pd.DataFrame) or projections.empty \
            or "game_id" not in projections.columns:
        return projections if isinstance(projections, pd.DataFrame) else None
    return projections[projections["game_id"].astype(str) == str(gid)]


def _failed_meta(n: int) -> dict:
    """A (book, game) whose compare() raised: every line counted as
    unprojected for an internal error, so the ledger still balances."""
    meta = _new_meta(n)
    meta.update(unprojected=int(n), errors=int(n), unprojected_why={"internal error": int(n)})
    return meta


def _book_rank(book) -> int:
    return BOOK_ORDER.index(book) if book in BOOK_ORDER else len(BOOK_ORDER)


def compare_books(lines, dist_provider, projections, roster=None) -> FeedTables:
    """compare() once per (book, game) of prepare_feed's lines:
    compare(rows, dist_provider, projections[projections.game_id == game],
    roster). Per game, so every row is priced against its own game's
    projection and the period-board check keeps its per-game group; per book,
    so that check stays inside one book, and every row's lean is against its
    own book's line (design 7). A (book, game) whose compare() raises gets an
    empty table and a meta counting its lines as unprojected, so its ledger
    still balances. The report is the one prepare_feed left on the lines."""
    report = {}
    if isinstance(lines, pd.DataFrame):
        report = dict(lines.attrs.get("feed_report") or {})
    if not isinstance(lines, pd.DataFrame) or lines.empty \
            or not {"book", "game_id"} <= set(lines.columns):
        return FeedTables({}, {}, report)
    tips = {}
    if "tip_utc" in lines.columns:
        for gid, tip in zip(lines["game_id"], lines["tip_utc"]):
            if gid is not None and gid not in tips:
                tips[gid] = _parse_ts(tip)
    groups = lines.groupby(["book", "game_id"], sort=False).indices
    order = sorted(groups, key=lambda k: (_book_rank(k[0]), str(k[0]),
                                          tips.get(k[1]) is None,
                                          tips.get(k[1]) or dt.datetime.max.replace(tzinfo=UTC),
                                          str(k[1])))
    tables, metas = {}, {}
    for book, gid in order:
        rows = lines.iloc[groups[(book, gid)]].reset_index(drop=True)
        try:
            table, meta = compare(rows, dist_provider, _game_projections(projections, gid), roster)
        except Exception as e:  # noqa: BLE001 - one (book, game) costs only itself
            logger.warning("compare_books: %s %s failed (%s)", book, gid, type(e).__name__)
            table, meta = pd.DataFrame(columns=TABLE_COLUMNS), _failed_meta(len(rows))
        tables[(book, gid)] = table
        metas[(book, gid)] = meta
    return FeedTables(tables, metas, report)


@dataclass
class FeedBoard:
    """What the board shows for one date and one selected book (layout B).

    reference     the selected book's rows for every game, one per
                  (game_date, person_id, stat), ranked (rank_rows): what
                  ranks, counts and colours. PrizePicks' rows are the
                  feed's merged with the visitor's paste.
    context       every OTHER book's row for the reference keys (PrizePicks'
                  merged rows when another book is selected), in the
                  reference's order and then BOOK_ORDER: the Other books cell.
    meta          the merged meta, balanced under
                  ledger(meta, *board_rows(reference)).
    ref_by_game   game_id -> the selected book, for each game it has lines
                  for, in (tip, game_id) order.
    missing_games the games, in (tip, game_id) order, where another book (or
                  the paste) has lines and the selected book has none.
    paste_won, feed_won   the PrizePicks keys both the feed and the paste
                  hold, by whose line the merge kept (whichever book is
                  selected, since PrizePicks' rows are context otherwise).
    paste_times   {"pasted_utc": the paste's time, "feed_oldest" and
                  "feed_newest": the range of fetch times of the feed lines
                  that beat the paste}, each an aware UTC Timestamp or None.
    """
    reference: pd.DataFrame
    context: pd.DataFrame
    meta: dict
    ref_by_game: dict = field(default_factory=dict)
    missing_games: list = field(default_factory=list)
    paste_won: int = 0
    feed_won: int = 0
    paste_times: dict = field(default_factory=dict)


def _row_keys(df: pd.DataFrame) -> list[tuple]:
    """(game date ISO, person_id, stat) per table row: the merge key (M5)."""
    if df is None or df.empty:
        return []
    return [(_iso_game_date(d), _as_pid(p), s if isinstance(s, str) else None)
            for d, p, s in zip(df["game_date"], df["person_id"], df["stat"])]


def _one_per_key(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(rows, rows removed): the first row of each key. compare() leaves one
    row per key already, so the count is expected to be 0; a removed row
    leaves the ledger along with its key, so the meta still balances."""
    if df is None or df.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS), 0
    dup = pd.Series(_row_keys(df), index=df.index).duplicated(keep="first").to_numpy()
    return df[~dup].reset_index(drop=True), int(dup.sum())


def _stack(frames, columns=()) -> pd.DataFrame:
    """The frames' rows in one frame: `columns` first, then every other column
    in the order it first appears. Built from records, so pandas never infers
    a dtype across an empty or all-missing column (its concat warns there)."""
    parts = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    cols = list(dict.fromkeys([*columns, *(c for f in parts for c in f.columns)]))
    recs = [r for f in parts for r in f.to_dict(orient="records")]
    return pd.DataFrame(recs, columns=cols) if recs else pd.DataFrame(columns=cols)


def _concat_rows(frames: list) -> pd.DataFrame:
    return _stack(frames, TABLE_COLUMNS)


def _paste_wins(paste_row: dict, feed_row: dict, pasted) -> bool:
    """The merge rule for one key (M5): the visitor's pasted row wins when it
    is a typed pick, a Demon or a Goblin, or when the board's paste time is
    later than the feed row's fetch time. A board with no paste time counts
    as older; so does a feed row with no fetch time."""
    if pick_of(paste_row.get("pick")) is not None:
        return True
    if str(paste_row.get("odds_type") or "").lower() in ("demon", "goblin"):
        return True
    if pasted is None:
        return False
    fetched = _utc_stamp(feed_row.get("fetched_at"))
    return fetched is None or pasted > fetched


def _merge_paste(feed_pp: pd.DataFrame, paste: pd.DataFrame, pasted) -> tuple[pd.DataFrame, dict]:
    """PrizePicks' rows for the date: the feed's and the paste's, one row per
    key, by _paste_wins where both hold the key. Returns (rows, stats):
    removed (rows the merge left out, overlaps and any duplicate), paste_won,
    feed_won and the fetch times of the feed rows that won."""
    feed_rows, dup_f = _one_per_key(feed_pp)
    paste_rows, dup_p = _one_per_key(paste)
    fkeys, pkeys = _row_keys(feed_rows), _row_keys(paste_rows)
    paste_at = {k: i for i, k in enumerate(pkeys)}
    feed_recs = feed_rows.to_dict(orient="records")
    paste_recs = paste_rows.to_dict(orient="records")
    paste_won, feed_won, won_at = set(), set(), []
    for i, k in enumerate(fkeys):
        j = paste_at.get(k)
        if j is None:
            continue
        if _paste_wins(paste_recs[j], feed_recs[i], pasted):
            paste_won.add(k)
        else:
            feed_won.add(k)
            at = _utc_stamp(feed_recs[i].get("fetched_at"))
            if at is not None:
                won_at.append(at)
    keep_feed = np.array([k not in paste_won for k in fkeys], dtype=bool)
    keep_paste = np.array([k not in feed_won for k in pkeys], dtype=bool)
    rows = _concat_rows([feed_rows[keep_feed] if len(feed_rows) else feed_rows,
                         paste_rows[keep_paste] if len(paste_rows) else paste_rows])
    removed = dup_f + dup_p + len(paste_won) + len(feed_won)
    return rows, {"removed": removed, "paste_won": len(paste_won), "feed_won": len(feed_won),
                  "feed_oldest": min(won_at) if won_at else None,
                  "feed_newest": max(won_at) if won_at else None}


def _sum_metas(parts: list, removed: int) -> dict:
    """compare() metas summed into one (design 2.5): every count summed,
    then posted and matched each less the rows the merge left out (every one
    of them was matched, so the identity holds); tallies summed, names and
    non-exact matches deduplicated and capped, the period and history flags
    ORed, and the unresolved lines stacked."""
    out = _new_meta(0)
    unresolved = []
    for m in parts:
        if not isinstance(m, dict):
            continue
        for k in _SUMMED:
            out[k] += int(m.get(k, 0) or 0)
        for k in _TALLIED:
            for label, n in (m.get(k) or {}).items():
                _bump(out[k], label, int(n))
        out["refused_why"].update(m.get("refused_why") or {})
        for k in ("unmatched_names", "ambiguous_names", "nonexact"):
            for item in m.get(k) or []:
                if item not in out[k] and len(out[k]) < FEED_META_NAMES:
                    out[k].append(item)
        for k in ("period_board_suspect", "games_unknown"):
            out[k] = bool(out[k] or m.get(k))
        u = m.get("unresolved")
        if isinstance(u, pd.DataFrame) and not u.empty:
            unresolved.append(u)
    out["posted"] -= int(removed)
    out["matched"] -= int(removed)
    out["unresolved"] = _stack(unresolved)
    return out


def _games_of(df) -> set:
    if not isinstance(df, pd.DataFrame) or df.empty or "game_id" not in df.columns:
        return set()
    return {str(g) for g in df["game_id"] if isinstance(g, str) and g}


def _game_tips(ft: FeedTables, frames: list) -> dict:
    """game_id -> its earliest known tip (an aware datetime), from prepare_feed's
    report and every frame's tip_utc."""
    tips: dict = {}

    def note(gid, tip):
        t = _parse_ts(tip)
        if isinstance(gid, str) and t is not None and (gid not in tips or t < tips[gid]):
            tips[gid] = t
    for gid, g in ((ft.report or {}).get("games") or {}).items():
        if isinstance(g, dict):
            note(gid, g.get("tip_utc"))
    for df in frames:
        if isinstance(df, pd.DataFrame) and not df.empty and {"game_id", "tip_utc"} <= set(df.columns):
            for gid, tip in zip(df["game_id"], df["tip_utc"]):
                note(gid, tip)
    return tips


def feed_board(ft, paste_table, paste_meta, pasted_utc, *, selected: str) -> FeedBoard:
    """The board for one date with the feed on (layout B, amendment A1).

    `ft` is compare_books' FeedTables for the date; `paste_table` and
    `paste_meta` are compare() of the visitor's pasted lines for the date,
    as the board compares them today (None or empty with no paste), and
    `pasted_utc` the board's paste time (board_info). `selected` is a
    BOOK_ORDER key (the page's pills; "prizepicks" by default there);
    anything else raises ValueError.

    - reference: the selected book's rows for every game, and only that
      book's: a game it has not posted has no rows (amendment A6), never
      another book's. For PrizePicks they are the feed's rows merged with the
      paste by the M5 rule (_paste_wins: newest wins; a typed pick, a Demon or
      a Goblin always wins; a board with no paste time counts as older).
      Pasted rows the paste path withheld are not in the paste table, so the
      feed's line stays for those keys.
    - context: every other book's rows for the reference keys; PrizePicks'
      are the same merged rows.
    - meta: the sum of the selected book's per-game metas, plus the paste's
      meta when PrizePicks is selected, less the rows the merge left out
      (_sum_metas), with by_tier counted on the reference and pasted_utc the
      paste's when PrizePicks is selected (None otherwise). It balances
      under ledger(meta, *board_rows(reference)).
    - ref_by_game, missing_games, paste_won, feed_won and paste_times as
      FeedBoard says. A book "has lines" for a game when compare_books holds
      a meta for it (every line it posted, compared or not), and the paste
      when its table or its unresolved lines name the game."""
    if not isinstance(selected, str) or selected not in BOOK_ORDER:
        raise ValueError("selected must be a BOOK_ORDER key")
    ft = ft if isinstance(ft, FeedTables) else FeedTables()
    paste = paste_table if isinstance(paste_table, pd.DataFrame) and not paste_table.empty \
        else pd.DataFrame(columns=TABLE_COLUMNS)
    pmeta = paste_meta if isinstance(paste_meta, dict) else {}
    pasted = _utc_stamp(pasted_utc)

    tables: dict = {b: [] for b in BOOK_ORDER}
    metas: dict = {b: [] for b in BOOK_ORDER}
    posted: dict = {b: set() for b in BOOK_ORDER}
    for (book, gid), meta in ft.metas.items():
        if book in BOOK_ORDER and isinstance(meta, dict):
            metas[book].append(meta)
            if int(meta.get("posted", 0) or 0) > 0:
                posted[book].add(str(gid))
    for (book, _gid), table in ft.tables.items():
        if book in BOOK_ORDER:
            tables[book].append(table)
    rows: dict = {}
    removed: dict = {}
    for b in BOOK_ORDER:
        rows[b], removed[b] = _one_per_key(_concat_rows(tables[b]))
    pp_rows, merge = _merge_paste(_concat_rows(tables[PASTE_BOOK]), paste, pasted)
    rows[PASTE_BOOK], removed[PASTE_BOOK] = pp_rows, merge["removed"]
    unresolved = pmeta.get("unresolved")
    paste_games = _games_of(paste) | _games_of(unresolved)
    posted[PASTE_BOOK] |= paste_games

    reference = rank_rows(rows[selected]) if len(rows[selected]) else rows[selected]
    reference = reference.reset_index(drop=True)
    parts = metas[selected] + ([pmeta] if selected == PASTE_BOOK and pmeta else [])
    meta = _sum_metas(parts, removed[selected])
    if not reference.empty:
        meta["by_tier"] = reference["tier"].value_counts().to_dict()
    meta["pasted_utc"] = pasted_utc if selected == PASTE_BOOK and pasted is not None else None

    order = {k: i for i, k in enumerate(_row_keys(reference))}
    context_parts = []
    for b in BOOK_ORDER:
        if b == selected or rows[b].empty:
            continue
        keys = _row_keys(rows[b])
        hit = np.array([k in order for k in keys], dtype=bool)
        if hit.any():
            context_parts.append(rows[b][hit])
    context = _concat_rows(context_parts)
    if not context.empty:
        pos = [order[k] for k in _row_keys(context)]
        rank = [_book_rank(b) for b in context["book"]]
        context = (context.assign(_o=pos, _b=rank)
                   .sort_values(["_o", "_b"], kind="mergesort")
                   .drop(columns=["_o", "_b"]).reset_index(drop=True))

    tips = _game_tips(ft, [*ft.tables.values(), paste, unresolved,
                           *[m.get("unresolved") for m in ft.metas.values()
                             if isinstance(m, dict)]])
    far = dt.datetime.max.replace(tzinfo=UTC)

    def by_tip(gid):
        return (gid not in tips, tips.get(gid, far), gid)
    mine = posted[selected]
    every = set().union(*posted.values())
    return FeedBoard(
        reference=reference, context=context, meta=meta,
        ref_by_game={g: selected for g in sorted(mine, key=by_tip)},
        missing_games=sorted(every - mine, key=by_tip),
        paste_won=merge["paste_won"], feed_won=merge["feed_won"],
        paste_times={"pasted_utc": pasted, "feed_oldest": merge["feed_oldest"],
                     "feed_newest": merge["feed_newest"]})


def american_breakeven(price) -> float | None:
    """The chance a side at an American price needs to break even: the
    probability the price implies, -p / (-p + 100) for p <= -100 and
    100 / (p + 100) for p >= 100 (-110 gives 0.5238). None for anything that
    is not an American price: missing, NaN, a boolean, text, or a number
    between -100 and +100. Every DFS row's price is NaN (its prices are
    indicative only), so a DFS row never gets a figure. A display note only,
    never compared with a model probability on the page (design 7)."""
    if price is None or isinstance(price, (bool, np.bool_)) \
            or not isinstance(price, (int, float, np.integer, np.floating)):
        return None
    p = float(price)
    if not math.isfinite(p):
        return None
    if p <= -100:
        return -p / (-p + 100.0)
    if p >= 100:
        return 100.0 / (p + 100.0)
    return None


def side_words(book) -> tuple[str, str]:
    """The words for the two sides on a row of `book` (design 2.6 m8,
    amendment A4): ("More", "Less") on a pick'em book (DFS_BOOKS),
    ("Over", "Under") on a sportsbook. Words only: the stored lean stays
    More, Less or Even, which the colour classes read. A row with no book is
    a pasted PrizePicks row."""
    b = book.strip().lower() if isinstance(book, str) and book.strip() else PASTE_BOOK
    return ("More", "Less") if b in DFS_BOOKS else ("Over", "Under")


# -- The record: LOCAL ONLY ---------------------------------------------------
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
