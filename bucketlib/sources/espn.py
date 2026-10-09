"""ESPN's public JSON: historical closing game lines for training, and the
small live adapters (scoreboard odds, injuries, rosters) the site reads at
serve time.

Every trap below was observed live (research pass and this module's own
probes, 2026-10-08). Each one fails SILENTLY if ignored:

SCHEDULE (site.api scoreboard)
  - `dates=YYYYMM` is capped at 100 events unless `limit` is passed, with no
    error (202610 returned 100 of 155). We pass limit=500 and, if a month
    ever returns exactly the limit, re-query it day by day.
  - The range form `dates=A-B` is an HTTP 400 "Failed to get events
    endpoint", which a careless caller reads as "no games". It is refused
    here before any request is sent.
  - Event dates are UTC. A 7:30pm ET tip is the next UTC day, so a join on
    the UTC date prefix fails for ~78% of games. et_date = ET date of tip.
  - The header `leagues[0].season.type` says "Preseason" for every date
    queried in October 2026; only each event's own `season.slug` is right.
  - Non-NBA opponents ride in the same feed: London Lions (134478) in the
    2026 preseason, the Rising Stars "WORLD VS USA" (type STD, not ALLSTAR)
    and the All-Star game (type ALLSTAR) with fake team ids. They are dropped
    and COUNTED.
  - Postponed games keep their original event id forever as
    STATUS_POSTPONED, sometimes WITH a full set of book odds (401267343,
    14 providers). The rescheduled game gets a new id. Only STATUS_FINAL
    events are lines for training.
  - The NBA Cup final is type "CC" with slug regular-season. It is a real
    game but its NBA GAME_ID has prefix 006 and it is NOT in the Regular
    Season game log, so it shows up unmatched unless the games table has it.
  - Four Feb-2020 events carry corrupted timestamps and venues (401161490
    DET@POR "2020-02-24T12:00Z" at the United Center). The crosswalk falls
    back to +/-1 day with the same teams and season.
  - The NBA side has its own trap: the 2025-26 LeagueGameLog prints BOTH
    rows of five neutral-site games as '@' (Mexico City, Berlin, London, the
    Las Vegas Cup semifinals), so it names no home team. Those games are
    matched in either orientation and ESPN's designation is kept.

CLOSING LINES (sports.core.api .../events/{id}/competitions/{id}/odds)
  - Provider sets change by era: offshore books + an "Opening" provider
    (id 23) + initialSpread/initialOverUnder in 2014-17; Vegas and state
    books with one value each in 2017-23; open/close/current objects from
    2023-24; ESPN BET (58) alone in 2024-25; DraftKings (100, not the old
    id 40) from 2025-11-25, alone from 2025-12-01.
  - In-play feeds sit beside pregame books: "ESPN Bet - Live Odds" (59),
    "Caesars ... - Live Odds" (44/46), "DraftKings - Live Odds" (200, new
    in 2026 and not in the research deny-list). They carry numbers like
    -40.5 that survive a median. We ALLOW-LIST pregame books by id and also
    drop any provider whose name contains "live", case-insensitively.
  - Aggregators and models (accuscore 1001, teamrankings 1002, numberfire
    1003, consensus 1004) and the Opening provider (23) are never in the
    close median: in 2015-17 "consensus" equals the OPENER in 25 of 25
    games, and numberfire's 0.0 means missing, not pick'em.
  - Item-level `spread`/`overUnder` are the CURRENT values. Before 2023-24
    that is the only value and it behaves as the close. From 2023-24 the
    current can be in-play or stale: 401654757 (2024 playoffs) has three
    books at totals 181-184 against a 206.5 close, and 401584954 has ESPN
    BET current 217.5 against its own close 234.5. So from 2023-24 the
    close{} object is used, falling back to current per value (flagged
    current_fallback: DK 100 sometimes has no close{} in late Nov 2025).
  - ESPN BET (58) close{} objects in 2022-23 hold PRICES ('-110') in the
    line slot. close{} is never read before 2023-24, and every close spread
    must equal minus the away close spread or it is rejected.
  - Lines are read from `.american` strings, never `.value` (a decimal
    price, e.g. 1.952 next to american '+6.5'). Totals carry 'o'/'u'
    prefixes ('o220.5').
  - Caesars Sportsbook (31) reports quarter/half markets as if full-game
    (total 59.0 with spread -2 against a -6 consensus). A book whose total
    is impossible for a full game loses its spread too: both came from the
    same partial-game market.
  - Book-level garbage in the multi-book eras: a flipped favourite (Wynn
    CLE -6 vs BOS -6 everywhere else), stale lines (DK 0.5/212 vs 5.5/222),
    in-play currents. Bounds, then a cross-book median, then drop any book
    more than 10 points from it on either market, then re-median.
  - `details` ('BKN -1') describes the CURRENT value. It checks the sign of
    a spread read from current, never one read from close{}: near pick'em
    the two legitimately disagree (401585288).
  - Opening lines exist only in 2014-17 (provider 23 / initialSpread) and
    from 2023-24 (open{}). 2017-18..2022-23 have NO opener: NaN, never a
    copy of the close.

LIVE
  - Scoreboard odds disappear at TIP (not at the final), and before tip the
    scoreboard's "close" is the CURRENT line. The caller must persist
    pre-tip snapshots; a missed one can be recovered next morning from the
    core odds (which keep the pregame close after the final).
  - Injuries: the item `id` is NOT the athlete id (parse /id/(\\d+)/ from the
    athlete links); `athlete.team` is stale after a trade (Aldama sits in the
    DAL group with athlete.team MEM); a team with no injuries is absent.

Every request goes through bucketlib.http, so Blocked is never read as
"no games" or "no odds".
"""
from __future__ import annotations

import calendar
import datetime as dt
import math
import re
import statistics
import time
from pathlib import Path

import pandas as pd

from .. import cache, http, teams, util
from ..names import NameIndex

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
             "events/{e}/competitions/{e}/odds")

MONTH_LIMIT = 500  # 239 events is the busiest month seen; the default cap is 100.

# ── Providers ────────────────────────────────────────────────────────────────
# Pregame sportsbooks seen in the core odds, 2014-15..2025-26 (research +
# this module's samples). An ALLOW-list: a provider that is not here is
# excluded and reported (flag unknown_provider), never silently trusted,
# because the deny-list approach already missed "DraftKings - Live Odds" (200).
PREGAME_BOOKS: dict[int, str] = {
    # 1 and 3 price every sampled 2016-17 game (8 of 8, verifier 2026-10-08)
    # in line with the other books; leaving them out cost 2 of 9 books.
    1: "SportsInteraction.com", 3: "Sportsbook.com",
    2: "Pinnacle", 4: "BetUS", 15: "SportsBetting.ag", 18: "5Dimes.eu",
    19: "BETONLINE.ag", 20: "BOVADA.lv", 21: "Fantasy911.com", 24: "BOVADA.lv",
    25: "Westgate", 26: "BETNOW", 28: "Caesar's", 31: "Caesars Sportsbook",
    32: "Wynn", 35: "CG Technology", 36: "Unibet", 38: "Caesars",
    40: "DraftKings", 41: "SugarHouse", 43: "Caesars Sportsbook (Pennsylvania)",
    45: "Caesars Sportsbook (New Jersey)", 47: "MGM", 48: "PointsBet",
    50: "BetfairSportsbook", 52: "Caesars Sportsbook (Colorado)", 53: "Titanbets",
    55: "Holland Casino", 57: "Caesars Sportsbook (Tennessee)", 58: "ESPN BET",
    100: "DraftKings", 2000: "Bet365",
}
OPENING_PROVIDER = 23
AGGREGATORS = {1001: "accuscore", 1002: "teamrankings", 1003: "numberfire", 1004: "consensus"}
KNOWN_LIVE = {44, 46, 59, 200}

SPREAD_MAX = 25.0
TOTAL_MIN, TOTAL_MAX = 170.0, 280.0
OUTLIER_PTS = 10.0
# A median within this of zero is too close to pick'em for the favourite
# named in `details` to contradict it meaningfully.
SIGN_CHECK_MIN = 1.0

# close{}/open{} objects carry real lines from 2023-24 on. Before that they
# are absent, empty, or (ESPN BET 2022-23) hold prices in the line slot.
FIRST_CLOSE_OBJECT_SEASON = "2023-24"
# Opening lines: provider 23 / initialSpread in 2014-17 only.
LAST_INITIAL_OPEN_SEASON = "2016-17"

_SCOREBOARD_TTL = 6 * 3600  # an unsettled month (current season) is refetched after this
# A month is settled once a copy was FETCHED this many days after it ended:
# every game in it is final (or postponed for good) by then.
SETTLE_DAYS = 3
# A cached 404 marker is trusted only this long. A real 404 for a finished
# game is rare (0 of 1,311 in the 2014-15 backfill), so re-asking costs a
# call; trusting a transient 404 forever costs the line for good.
ABSENT_TTL = 6 * 3600


def _is_live_name(name) -> bool:
    return "live" in str(name or "").lower()


# ── Small parsers ────────────────────────────────────────────────────────────
_LINE_RE = re.compile(r"^[ou]?\s*([+-]?\d+(?:\.\d+)?)$", re.I)


def parse_line(v) -> float | None:
    """A line from any ESPN field: 9.5, '+9.5', 'o220.5', 'u220.5', 'EVEN',
    'PK'. None for missing or unparseable. Bools are rejected (json false is
    not a line)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if math.isnan(f) else f
    s = str(v).strip()
    if not s:
        return None
    if s.upper() in ("EVEN", "PK", "PICK", "PICKEM", "PICK'EM"):
        return 0.0
    m = _LINE_RE.match(s)
    return float(m.group(1)) if m else None


def parse_american(v) -> int | None:
    """American odds '+235' / '-110' / 'EVEN' -> int. 0 is BOVADA's 2015-16
    missing sentinel and is returned as None."""
    if v is None or isinstance(v, bool):
        return None
    s = str(v).strip().upper()
    if s == "EVEN":
        return 100
    try:
        n = int(float(s))
    except ValueError:
        return None
    return None if n == 0 else n


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _spread_ok(x) -> bool:
    return x is not None and abs(x) <= SPREAD_MAX


def _total_ok(x) -> bool:
    return x is not None and TOTAL_MIN <= x <= TOTAL_MAX


_DETAILS_RE = re.compile(r"^([A-Za-z]{2,5})\s+([+-]?\d+(?:\.\d+)?)$")


def details_favourite(details, home: str, away: str) -> str | None:
    """'home' | 'away' | 'even' from a details string like 'PHI -8.5', or
    None when it cannot be read. The favourite is the named team when its
    number is negative (always, in every payload seen)."""
    if not details:
        return None
    s = str(details).strip()
    if s.upper() in ("EVEN", "PK", "PICK"):
        return "even"
    m = _DETAILS_RE.match(s)
    if not m:
        return None
    code, num = teams.canonical(m.group(1)), float(m.group(2))
    if code not in (home, away):
        return None
    if num == 0:
        return "even"
    named = "home" if code == home else "away"
    if num < 0:
        return named
    return "away" if named == "home" else "home"


def _sign_side(home_line: float) -> str:
    return "home" if home_line < 0 else ("away" if home_line > 0 else "even")


def season_of_event(ev: dict) -> str:
    year = _dig(ev, "season", "year")
    if not isinstance(year, int):
        raise ValueError(f"event {ev.get('id')}: no integer season.year ({year!r})")
    return util.season_str(year)


# ── Schedule: monthly scoreboard ─────────────────────────────────────────────
def season_months(season: str) -> list[str]:
    """'YYYYMM' months that can hold games of `season` (preseason through
    Finals). 2019-20 resumed Jul-Oct 2020 (bubble); 2020-21 ran Dec-Jul;
    2011-12 started on Christmas."""
    start = util.season_start_year(season)

    def span(y0, m0, y1, m1):
        out, y, m = [], y0, m0
        while (y, m) <= (y1, m1):
            out.append(f"{y}{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    if season == "2019-20":
        return span(2019, 10, 2020, 3) + span(2020, 7, 2020, 10)
    if season == "2020-21":
        return span(2020, 12, 2021, 7)
    if season == "2011-12":
        return span(2011, 12, 2012, 6)
    return span(start, 10, start + 1, 6)


def _check_dates_param(dates: str) -> None:
    # The range form 'A-B' is an HTTP 400 that looks like "no games".
    if not re.fullmatch(r"\d{6}|\d{8}", str(dates)):
        raise ValueError(f"scoreboard dates must be YYYYMM or YYYYMMDD, got {dates!r} "
                         "(the range form A-B is an HTTP 400 on this API)")


def fetch_scoreboard(dates: str, limit: int = MONTH_LIMIT) -> dict:
    """One raw scoreboard request. `dates` is YYYYMM or YYYYMMDD (US-local)."""
    _check_dates_param(dates)
    payload = http.fetch(SITE + "scoreboard", params={"dates": str(dates), "limit": str(limit)})
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise http.BadPayload(SITE + "scoreboard", f"dates={dates}: no events list in payload")
    return payload


def _trim_competition(c: dict, keep_odds: bool) -> dict:
    out = {k: c.get(k) for k in ("id", "date", "startDate", "type", "neutralSite",
                                  "timeValid", "notes", "status") if k in c}
    v = c.get("venue") or {}
    out["venue"] = {"id": v.get("id"), "fullName": v.get("fullName"), "address": v.get("address")}
    out["competitors"] = [{
        "id": x.get("id"), "homeAway": x.get("homeAway"), "winner": x.get("winner"),
        "score": x.get("score"),
        "team": {k: (x.get("team") or {}).get(k) for k in ("id", "abbreviation", "displayName")},
    } for x in c.get("competitors") or []]
    if keep_odds and c.get("odds"):
        out["odds"] = _strip_links(c["odds"])
    return out


def _strip_links(obj):
    """Drop sportsbook deep links (most of the odds bytes, none of the data)."""
    if isinstance(obj, dict):
        return {k: _strip_links(v) for k, v in obj.items() if k not in ("link", "links", "logos")}
    if isinstance(obj, list):
        return [_strip_links(v) for v in obj]
    return obj


def trim_scoreboard(payload: dict, keep_odds: bool = True) -> dict:
    """Same shape as the raw payload, minus team logos, leaders, broadcasts
    and links (about 90% of the bytes). Disk is nearly full and a season of
    raw months would otherwise be ~2 MB gzipped."""
    events = []
    for e in payload.get("events") or []:
        comps = e.get("competitions") or []
        events.append({
            "id": e.get("id"), "date": e.get("date"), "name": e.get("name"),
            "shortName": e.get("shortName"), "season": e.get("season"),
            "status": e.get("status"),
            "competitions": [_trim_competition(c, keep_odds) for c in comps[:1]],
        })
    return {"events": events}


def fetched_after(fetched_at: float | None, day: dt.date, days: int = 0) -> bool:
    """True when the copy stamped `fetched_at` was fetched on an ET date more
    than `days` after `day`. A missing stamp is never late enough."""
    if fetched_at is None:
        return False
    fetched_et = util.et_date_of(dt.datetime.fromtimestamp(float(fetched_at), util.UTC))
    return (fetched_et - day).days > days


def _month_settled(ym: str, fetched_at: float | None) -> bool:
    # Judged by WHEN THE COPY WAS FETCHED, never by today's date: a copy
    # saved mid-month holds that month's games as STATUS_SCHEDULED forever,
    # and a calendar test would call it settled three days after month end
    # (verifier 2026-10-08: 202610 cached on Oct 8, read "in December", gave
    # 88 of 88 regular-season events still scheduled and no lines).
    y, m = int(ym[:4]), int(ym[4:])
    last = dt.date(y, m, calendar.monthrange(y, m)[1])
    return fetched_after(fetched_at, last, SETTLE_DAYS)


def scoreboard_month(ym: str, *, refresh: bool = False, use_cache: bool = True) -> dict:
    """Trimmed scoreboard payload for one month, cached at
    raw/espn_scoreboard/YYYYMM.json.gz. A cached copy FETCHED more than
    SETTLE_DAYS after the month ended is settled and never refetched (unless
    refresh); any other copy is refetched once its stamp is older than 6
    hours. An empty copy is never settled: a 200 with no events for a month
    the caller expects games in is retried, not trusted forever."""
    _check_dates_param(ym)
    if len(ym) != 6:
        raise ValueError(f"month must be YYYYMM, got {ym!r}")
    path = cache.dc_path(f"raw/espn_scoreboard/{ym}.json.gz")
    if use_cache and not refresh:
        data, fetched_at = cache.read_json_stamped(path)
        if data is not None:
            settled = _month_settled(ym, fetched_at) and bool(data.get("events"))
            if settled or cache.is_fresh(fetched_at, _SCOREBOARD_TTL):
                return data
    payload = fetch_scoreboard(ym)
    events = payload["events"]
    if len(events) >= MONTH_LIMIT:
        # Possibly truncated: never trust a count equal to the cap.
        y, m = int(ym[:4]), int(ym[4:])
        merged, seen = [], set()
        for day in range(1, calendar.monthrange(y, m)[1] + 1):
            dp = fetch_scoreboard(f"{ym}{day:02d}")
            if len(dp["events"]) >= MONTH_LIMIT:
                raise ValueError(f"{ym}{day:02d}: {len(dp['events'])} events hit the limit")
            for e in dp["events"]:
                if e.get("id") not in seen:
                    seen.add(e.get("id"))
                    merged.append(e)
        payload = {"events": merged}
    trimmed = trim_scoreboard(payload)
    if use_cache:
        cache.write_json(path, trimmed, stamped=True, gz=True)
    return trimmed


EVENT_COLUMNS = ["event_id", "season", "start_utc", "et_date", "status", "season_slug",
                 "comp_type", "home", "away", "neutral_site", "venue", "home_score",
                 "away_score", "notes"]


def parse_events(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Scoreboard payload -> (events frame, dropped counts).

    Rows whose teams are not both NBA teams (London Lions, Rising Stars,
    All-Star) are dropped and counted by reason, with their ids listed under
    'dropped_ids'. A FINAL event whose score cannot be read is KEPT (its
    line is still good) with NA scores, and counted under
    'final_score_missing' / 'final_score_missing_ids'. Structural surprises
    raise."""
    rows, dropped = [], {"non_nba_team": 0, "allstar": 0, "dropped_ids": [],
                         "final_score_missing": 0, "final_score_missing_ids": []}
    seen = set()
    for e in payload.get("events") or []:
        eid = str(e.get("id") or "")
        if not eid.isdigit():
            raise ValueError(f"event without a numeric id: {str(e)[:120]}")
        if eid in seen:
            raise ValueError(f"event {eid} appears twice in one scoreboard payload")
        seen.add(eid)
        comps = e.get("competitions") or []
        if len(comps) < 1:
            raise ValueError(f"event {eid}: no competitions")
        c = comps[0]
        comp_type = _dig(c, "type", "abbreviation")
        sides = {x.get("homeAway"): x for x in c.get("competitors") or []}
        if set(sides) != {"home", "away"} or len(c.get("competitors") or []) != 2:
            raise ValueError(f"event {eid}: competitors are not one home + one away: "
                             f"{[x.get('homeAway') for x in c.get('competitors') or []]}")
        home = teams.from_espn_id(_dig(sides["home"], "team", "id"))
        away = teams.from_espn_id(_dig(sides["away"], "team", "id"))
        if comp_type == "ALLSTAR":
            dropped["allstar"] += 1
            dropped["dropped_ids"].append(eid)
            continue
        if home is None or away is None:
            dropped["non_nba_team"] += 1
            dropped["dropped_ids"].append(eid)
            continue
        if home == away:
            raise ValueError(f"event {eid}: {home} plays itself")
        start = e.get("date") or c.get("date")
        if not start:
            raise ValueError(f"event {eid}: no start date")
        start_utc = util.to_et(start).astimezone(util.UTC)
        status = _dig(e, "status", "type", "name") or _dig(c, "status", "type", "name")
        if not status:
            raise ValueError(f"event {eid}: no status")
        slug = _dig(e, "season", "slug")

        def score(x):
            s = x.get("score")
            s = s.get("value") if isinstance(s, dict) else s
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None

        notes = "; ".join(n.get("headline", "") for n in c.get("notes") or [] if n.get("headline"))
        hs = score(sides["home"]) if status == "STATUS_FINAL" else None
        aw = score(sides["away"]) if status == "STATUS_FINAL" else None
        if status == "STATUS_FINAL" and (hs is None or aw is None):
            # A finished game always has a score; count it rather than let
            # an NA pass as normal.
            dropped["final_score_missing"] += 1
            dropped["final_score_missing_ids"].append(eid)
        rows.append({
            "event_id": eid, "season": season_of_event(e), "start_utc": start_utc,
            "et_date": util.et_date_of(start_utc), "status": status, "season_slug": slug,
            "comp_type": comp_type, "home": home, "away": away,
            "neutral_site": bool(c.get("neutralSite")),
            "venue": _dig(c, "venue", "fullName"),
            "home_score": hs, "away_score": aw,
            "notes": notes or None,
        })
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    for col in ("home_score", "away_score"):
        df[col] = df[col].astype("Int64")
    return df, dropped


def events(season: str, *, months: list[str] | None = None, refresh: bool = False,
           use_cache: bool = True) -> pd.DataFrame:
    """Every ESPN event of `season` with two NBA teams, one call per month.

    Months after the current ET month are skipped (nothing to fetch yet).
    Events belonging to another season (none seen, but months can straddle)
    are filtered out. df.attrs carries 'dropped' (counts by reason),
    'months' (fetched) and 'month_counts' ({YYYYMM: raw events in the
    payload}), so a caller can refuse a season with an empty month."""
    util.season_end_year(season)
    want = months or season_months(season)
    this_month = util.today_et().strftime("%Y%m")
    frames, dropped = [], {"non_nba_team": 0, "allstar": 0, "other_season": 0,
                           "duplicate_across_months": 0, "dropped_ids": [],
                           "final_score_missing": 0, "final_score_missing_ids": []}
    fetched, month_counts = [], {}
    for ym in want:
        if ym > this_month:
            continue
        payload = scoreboard_month(ym, refresh=refresh, use_cache=use_cache)
        month_counts[ym] = len(payload.get("events") or [])
        df, d = parse_events(payload)
        for k in ("non_nba_team", "allstar", "final_score_missing"):
            dropped[k] += d.get(k, 0)
        dropped["dropped_ids"] += d["dropped_ids"]
        dropped["final_score_missing_ids"] += d.get("final_score_missing_ids", [])
        frames.append(df)
        fetched.append(ym)
    if not frames:
        out = pd.DataFrame(columns=EVENT_COLUMNS)
        out.attrs.update(dropped=dropped, months=fetched, month_counts=month_counts)
        return out
    allev = pd.concat(frames, ignore_index=True)
    other = allev["season"] != season
    dropped["other_season"] = int(other.sum())
    allev = allev[~other]
    dup = allev.duplicated("event_id", keep="first")
    if dup.any():
        # A tip near midnight can appear in two month queries. Identical
        # rows are fine; conflicting ones mean a parsing error somewhere.
        firsts = allev.drop_duplicates("event_id", keep="first").set_index("event_id")
        for _, r in allev[dup].iterrows():
            f = firsts.loc[r["event_id"]]
            if (f["home"], f["away"], f["start_utc"]) != (r["home"], r["away"], r["start_utc"]):
                raise ValueError(f"event {r['event_id']} differs between month payloads")
        dropped["duplicate_across_months"] = int(dup.sum())
        allev = allev[~dup]
    allev = allev.sort_values(["start_utc", "event_id"]).reset_index(drop=True)
    allev.attrs.update(dropped=dropped, months=fetched, month_counts=month_counts)
    return allev


# ── Closing lines: core odds ─────────────────────────────────────────────────
def odds_cache_path(season: str, event_id) -> Path:
    return cache.dc_path(f"raw/espn_odds/{season}/{event_id}.json.gz")


def fetch_odds_pages(event_id) -> list[dict]:
    """All pages of the core odds collection for one event. Raises the
    http failure classes untouched (Absent = no odds resource)."""
    url = CORE_ODDS.format(e=event_id)
    pages = [http.fetch(url, params={"limit": "100"})]
    while True:
        p = pages[-1]
        if not isinstance(p, dict) or not isinstance(p.get("items"), list):
            raise http.BadPayload(url, f"no items list (page {len(pages)})")
        idx, count = int(p.get("pageIndex", 1)), int(p.get("pageCount", 1))
        if idx >= count:
            break
        if len(pages) > 20:
            raise http.BadPayload(url, f"pagination runaway at page {idx}/{count}")
        pages.append(http.fetch(url, params={"limit": "100", "page": str(idx + 1)}))
    n_items = sum(len(p["items"]) for p in pages)
    declared = pages[0].get("count")
    if declared is not None and int(declared) != n_items:
        raise http.BadPayload(url, f"count={declared} but {n_items} items across {len(pages)} pages")
    return pages


def _usable_odds_cache(path: Path, retry_absent: bool) -> dict | None:
    """The cached payload if it may be reused, else None. A real payload is
    final for a finished game and never expires; an absent (404) marker is
    reused only while younger than ABSENT_TTL (and never with retry_absent),
    so a burst of transient 404s cannot poison every later rerun."""
    data, fetched_at = cache.read_json_stamped(path)
    if data is None:
        return None
    if data.get("absent") and (retry_absent or not cache.is_fresh(fetched_at, ABSENT_TTL)):
        return None
    return data


def odds_is_cached(event_id, season: str, *, retry_absent: bool = False) -> bool:
    """True when odds_payload would answer from the cache (no request)."""
    return _usable_odds_cache(odds_cache_path(season, event_id), retry_absent) is not None


def odds_payload(event_id, season: str, *, use_cache: bool = True, refresh: bool = False,
                 retry_absent: bool = False) -> dict:
    """{'event_id', 'pages': [...]} or {'event_id', 'absent': True}, cached.

    Only call this for events whose odds are final (finished games): a real
    payload never expires. Absent (404) is cached too, as a marker, so a
    resumed backfill does not re-ask within ABSENT_TTL; after that (or with
    retry_absent) it is asked again. Blocked and BadRequest are raised."""
    path = odds_cache_path(season, event_id)
    if use_cache and not refresh:
        data = _usable_odds_cache(path, retry_absent)
        if data is not None:
            return data
    try:
        pages = fetch_odds_pages(event_id)
        data = {"event_id": str(event_id), "pages": [_strip_links(p) for p in pages]}
    except http.Absent:
        data = {"event_id": str(event_id), "absent": True}
    if use_cache:
        cache.write_json(path, data, stamped=True, gz=True)
    return data


def _team_from_ref(side: dict | None) -> str | None:
    ref = _dig(side, "team", "$ref") or ""
    m = re.search(r"/teams/(\d+)", ref)
    return teams.from_espn_id(m.group(1)) if m else None


def _close_spread(item: dict, which: str) -> float | None:
    """Home line from the {which} (open/close) objects, validated against
    the away side: a real line has away == -home. ESPN BET 2022-23 put
    prices here ('-110' / '-105'), which fail this check."""
    h = parse_line(_dig(item, "homeTeamOdds", which, "pointSpread", "american"))
    a = parse_line(_dig(item, "awayTeamOdds", which, "pointSpread", "american"))
    if h is None:
        return None
    if a is not None and abs(h + a) > 1e-9:
        return None
    return h


def _close_total(item: dict, which: str) -> float | None:
    return parse_line(_dig(item, which, "total", "american"))


def _robust_median(vals: dict[str, tuple[float | None, float | None]]):
    """vals: {book: (spread, total)} already bounds-checked. Returns
    (spread_median, total_median, kept_spread_books, kept_total_books,
    outlier_books). A book more than OUTLIER_PTS from either median loses
    BOTH values: in-play and stale snapshots move both markets together."""
    sp = [s for s, _ in vals.values() if s is not None]
    to = [t for _, t in vals.values() if t is not None]
    if not sp and not to:
        return math.nan, math.nan, [], [], []
    ms = statistics.median(sp) if sp else None
    mt = statistics.median(to) if to else None
    outliers = [b for b, (s, t) in vals.items()
                if (s is not None and ms is not None and abs(s - ms) > OUTLIER_PTS)
                or (t is not None and mt is not None and abs(t - mt) > OUTLIER_PTS)]
    kept = {b: v for b, v in vals.items() if b not in outliers}
    ks = [b for b, (s, _) in kept.items() if s is not None]
    kt = [b for b, (_, t) in kept.items() if t is not None]
    s_med = statistics.median([kept[b][0] for b in ks]) if ks else math.nan
    t_med = statistics.median([kept[b][1] for b in kt]) if kt else math.nan
    return float(s_med), float(t_med), ks, kt, outliers


ODDS_COLUMNS = ["event_id", "home", "away", "home_line", "total", "n_books", "n_books_total",
                "books", "open_home_line", "open_total", "has_open", "line_field", "qc_flags",
                "source"]


def parse_odds(data: dict, season: str, *, home: str | None = None,
               away: str | None = None) -> dict:
    """Core odds payload -> one consensus row (see ODDS_COLUMNS).

    home_line is the HOME team's line, negative = home favoured (ESPN's own
    convention, kept everywhere in this repo). `qc_flags` is a ';'-joined list
    of everything excluded or suspicious, so a backfill can count them.
    home/away, when given, must agree with the payload's team refs."""
    util.season_end_year(season)
    eid = str(data.get("event_id"))
    flags: list[str] = []
    row = {"event_id": eid, "home": home, "away": away, "home_line": math.nan,
           "total": math.nan, "n_books": 0, "n_books_total": 0, "books": "",
           "open_home_line": math.nan, "open_total": math.nan, "has_open": False,
           "line_field": None, "qc_flags": "", "source": "espn_core"}
    if data.get("absent"):
        row["qc_flags"] = "odds_absent"
        return row
    items = [it for p in data.get("pages") or [] for it in p.get("items") or []]
    if not items:
        row["qc_flags"] = "no_items"
        return row

    # Teams from the payload's own refs; they must be one consistent pair.
    pairs = {(_team_from_ref(it.get("homeTeamOdds")), _team_from_ref(it.get("awayTeamOdds")))
             for it in items}
    pairs.discard((None, None))
    if len(pairs) > 1:
        raise ValueError(f"event {eid}: odds items disagree on home/away {sorted(pairs, key=str)}")
    p_home, p_away = next(iter(pairs)) if pairs else (None, None)
    if (home or away) and pairs and (home, away) != (p_home, p_away):
        raise ValueError(f"event {eid}: odds refs say {p_away}@{p_home}, caller says {away}@{home}")
    home, away = home or p_home, away or p_away
    row["home"], row["away"] = home, away

    use_close = season >= FIRST_CLOSE_OBJECT_SEASON
    books: dict[str, tuple[float | None, float | None]] = {}
    details_by_book: dict[str, str | None] = {}
    # Where each kept book's SPREAD came from. `details` describes the
    # current value, so it can only check a spread read from current: from
    # 2023-24 the close and the current legitimately disagree on the
    # favourite near pick'em (401585288: Betfair close +1, details 'BKN -1'
    # while the other nine books closed -0.5..+1.5).
    spread_src: dict[str, str] = {}
    fields: set[str] = set()
    opening_item = None
    initial = []
    open_vals: dict[str, tuple[float | None, float | None]] = {}
    seen_ids = set()
    for it in items:
        prov = it.get("provider") or {}
        try:
            pid = int(prov.get("id"))
        except (TypeError, ValueError):
            raise ValueError(f"event {eid}: odds item without a provider id: {str(it)[:120]}") from None
        name = prov.get("name") or str(pid)
        if pid in seen_ids:
            raise ValueError(f"event {eid}: provider {pid} listed twice")
        seen_ids.add(pid)
        if season <= LAST_INITIAL_OPEN_SEASON and (it.get("initialSpread") is not None
                                                   or it.get("initialOverUnder") is not None):
            initial.append((parse_line(it.get("initialSpread")), parse_line(it.get("initialOverUnder"))))
        if pid == OPENING_PROVIDER:
            opening_item = it
            continue
        if pid in KNOWN_LIVE or _is_live_name(name):
            flags.append(f"live:{pid}")
            continue
        if pid in AGGREGATORS:
            continue
        if pid not in PREGAME_BOOKS:
            flags.append(f"unknown_provider:{pid}:{name}")
            continue
        key = f"{name}#{pid}"
        cur_s, cur_t = parse_line(it.get("spread")), parse_line(it.get("overUnder"))
        s, t, src, s_src = cur_s, cur_t, "current", "current"
        if use_close:
            cs, ct = _close_spread(it, "close"), _close_total(it, "close")
            if cs is not None or ct is not None:
                s = cs if cs is not None else cur_s
                t = ct if ct is not None else cur_t
                s_src = "close" if cs is not None else "current"
                src = "close" if (cs is not None and ct is not None) else "mixed"
            else:
                flags.append(f"current_fallback:{pid}")
            os_, ot = _close_spread(it, "open"), _close_total(it, "open")
            os_ = os_ if _spread_ok(os_) else None
            ot = ot if _total_ok(ot) else None
            if os_ is not None or ot is not None:
                open_vals[key] = (os_, ot)
        # A total no full game can have means this row is a quarter/half
        # market (Caesars 31: 59.0); its spread is from the same market.
        if t is not None and not _total_ok(t):
            flags.append(f"partial_or_bad_total:{pid}")
            s = t = None
        if s is not None and not _spread_ok(s):
            flags.append(f"spread_out_of_bounds:{pid}")
            s = None
        # Per-book favourite check, only where the spread is the value
        # `details` describes (the current one).
        if s is not None and home and away:
            fav = details_favourite(it.get("details"), home, away)
            if fav in ("home", "away") and s != 0 and _sign_side(s) != fav:
                if s_src == "current":
                    flags.append(f"book_sign_mismatch:{pid}")
                    s = None
                elif abs(s) >= SIGN_CHECK_MIN:
                    # The close is kept: the current moved across zero (or
                    # went in-play) after the close. Reported, not fatal.
                    flags.append(f"details_vs_close:{pid}")
        if s is None and t is None:
            continue
        books[key] = (s, t)
        details_by_book[key] = it.get("details")
        spread_src[key] = s_src
        fields.add(src)

    s_med, t_med, ks, kt, outliers = _robust_median(books)
    flags += [f"outlier:{b.split('#')[-1]}" for b in outliers]
    # Final sign check against the favourite most kept books name, counting
    # only books whose spread is the value their `details` describes.
    if not math.isnan(s_med) and abs(s_med) >= SIGN_CHECK_MIN and home and away:
        votes = [details_favourite(details_by_book[b], home, away)
                 for b in ks if spread_src[b] == "current"]
        nh, na = votes.count("home"), votes.count("away")
        majority = "home" if nh > na else ("away" if na > nh else None)
        if majority and majority != _sign_side(s_med):
            flags.append("sign_conflict")
            s_med = math.nan
            ks = []
    row.update(home_line=s_med, total=t_med, n_books=len(ks), n_books_total=len(kt),
               books="|".join(sorted({b.split("#")[0] for b in set(ks) | set(kt)})))
    if fields:
        row["line_field"] = fields.pop() if len(fields) == 1 else "mixed"

    # Opening line, era-gated.
    o_s = o_t = math.nan
    if season <= LAST_INITIAL_OPEN_SEASON:
        if opening_item is not None:
            a, b = parse_line(opening_item.get("spread")), parse_line(opening_item.get("overUnder"))
            o_s = a if _spread_ok(a) else math.nan
            o_t = b if _total_ok(b) else math.nan
        if initial and (math.isnan(o_s) or math.isnan(o_t)):
            # initialSpread is one game-level value repeated on every item.
            si = [x for x, _ in initial if _spread_ok(x)]
            ti = [y for _, y in initial if _total_ok(y)]
            if math.isnan(o_s) and si:
                o_s = statistics.median(si)
            if math.isnan(o_t) and ti:
                o_t = statistics.median(ti)
    elif use_close and open_vals:
        os_med, ot_med, _, _, o_out = _robust_median(open_vals)
        o_s, o_t = os_med, ot_med
        flags += [f"open_outlier:{b.split('#')[-1]}" for b in o_out]
    row.update(open_home_line=float(o_s), open_total=float(o_t),
               has_open=bool(not math.isnan(o_s) or not math.isnan(o_t)))
    if not books:
        flags.append("no_pregame_books")
    row["qc_flags"] = ";".join(flags)
    return row


def game_odds(event_id, season: str, *, home: str | None = None, away: str | None = None,
              use_cache: bool = True, refresh: bool = False, retry_absent: bool = False) -> dict:
    """Consensus closing (and opening) line for one FINISHED event.

    `season` is required: it decides which fields are trustworthy (close{}
    only from 2023-24, openers only in 2014-17 and 2023-24+). Raises
    Blocked/BadRequest/BadPayload; a 404 becomes a row flagged odds_absent."""
    return parse_odds(odds_payload(event_id, season, use_cache=use_cache, refresh=refresh,
                                   retry_absent=retry_absent),
                      season, home=home, away=away)


# ── Crosswalk: ESPN event -> NBA GAME_ID ─────────────────────────────────────
def season_from_game_id(game_id) -> str:
    """'0021900002' -> '2019-20' (digits 4-5 are the season start year)."""
    gid = str(game_id).zfill(10)
    if not re.fullmatch(r"00\d{8}", gid):
        raise ValueError(f"not an NBA GAME_ID: {game_id!r}")
    return util.season_str(2000 + int(gid[3:5]) + 1)


GAMES_COLUMNS = ["game_id", "game_date", "home", "away", "neutral", "season"]


def _games_from_sides(d: pd.DataFrame, what: str) -> pd.DataFrame:
    """d: one row per (GAME_ID, code) with side in {'home', 'away'} and a
    GAME_DATE. Two 'away' rows = a neutral site the log gives no home team
    (see nba_games_from_gamelog): orientation becomes alphabetical and
    neutral=True tells the crosswalk to accept either."""
    counts = d.groupby("GAME_ID")["side"].agg(lambda s: tuple(sorted(s)))
    ok = counts.isin([("away", "home"), ("away", "away")])
    if not ok.all():
        raise ValueError(f"{what}: {int((~ok).sum())} games without one home + one away row "
                         f"(or two '@' rows), e.g. {counts[~ok].head(3).to_dict()}")
    dates = d.groupby("GAME_ID")["GAME_DATE"].nunique()
    if (dates > 1).any():
        raise ValueError(f"{what}: the two sides carry different dates for {dates[dates > 1].index[:3].tolist()}")
    rows = []
    for gid, g in d.groupby("GAME_ID", sort=False):
        neutral = counts[gid] == ("away", "away")
        if neutral:
            home, away = sorted(g["code"])
        else:
            home = g.loc[g["side"] == "home", "code"].iloc[0]
            away = g.loc[g["side"] == "away", "code"].iloc[0]
        rows.append({"game_id": gid, "game_date": pd.to_datetime(g["GAME_DATE"].iloc[0]).date(),
                     "home": home, "away": away, "neutral": bool(neutral)})
    out = pd.DataFrame(rows, columns=GAMES_COLUMNS[:-1])
    out["season"] = out["game_id"].map(season_from_game_id)
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def nba_games_from_gamelog(df: pd.DataFrame) -> pd.DataFrame:
    """LeagueGameLog (PlayerOrTeam=T) rows, two per game -> one row per game
    (GAMES_COLUMNS). 'X vs. Y' is the home row, 'X @ Y' the away row.

    Trap (2025-26, five games): at neutral sites abroad and the Las Vegas
    Cup semifinals the log prints BOTH rows as '@' ('DAL @ DET' and
    'DET @ DAL', 0022500147). There is no home team to read, so those games
    are marked neutral=True. The 2019-20 bubble is the opposite case: a
    neutral site still labelled vs./@."""
    need = {"GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"game log is missing columns {sorted(missing)}")
    if df.empty:
        return pd.DataFrame(columns=GAMES_COLUMNS)
    d = df.copy()
    d["code"] = [teams.from_team_id(t) or teams.canonical(a)
                 for t, a in zip(d["TEAM_ID"], d["TEAM_ABBREVIATION"])]
    bad = d[d["code"].isna()]
    if len(bad):
        raise ValueError(f"unknown teams in game log: {sorted(set(bad['TEAM_ABBREVIATION']))}")
    d["side"] = ["home" if " vs. " in m else ("away" if " @ " in m else None) for m in d["MATCHUP"]]
    if d["side"].isna().any():
        raise ValueError(f"unparseable MATCHUP values: {sorted(set(d.loc[d['side'].isna(), 'MATCHUP']))[:5]}")
    d["GAME_ID"] = d["GAME_ID"].astype(str).str.zfill(10)
    if d.duplicated(["GAME_ID", "code"]).any():
        raise ValueError("game log lists a team twice in one game")
    return _games_from_sides(d, "game log")


def nba_games_from_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team game rows (game_id, game_date, team, home: bool, neutral:
    bool), the shape of the stats.nba.com team-games table, -> GAMES_COLUMNS."""
    need = {"game_id", "game_date", "team", "home"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"team rows are missing columns {sorted(missing)}")
    d = pd.DataFrame({
        "GAME_ID": df["game_id"].astype(str).str.zfill(10),
        "GAME_DATE": pd.to_datetime(df["game_date"]).dt.date,
        "code": df["team"].map(teams.canonical),
        "side": ["home" if bool(h) else "away" for h in df["home"]],
    })
    if d["code"].isna().any():
        raise ValueError(f"unknown team codes {sorted(set(df.loc[d['code'].isna(), 'team']))[:5]}")
    if d.duplicated(["GAME_ID", "code"]).any():
        raise ValueError("team rows list a team twice in one game")
    return _games_from_sides(d, "team rows")


XWALK_COLUMNS = ["event_id", "game_id", "how", "season", "et_date", "nba_game_date", "home",
                 "away", "status", "season_slug", "comp_type", "hint"]


def _prep_games(nba_games_df: pd.DataFrame) -> pd.DataFrame:
    need = {"game_id", "game_date", "home", "away"}
    missing = need - set(nba_games_df.columns)
    if missing:
        raise ValueError(f"nba_games_df is missing columns {sorted(missing)}")
    g = nba_games_df.copy()
    g["game_id"] = g["game_id"].astype(str).str.zfill(10)
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    if "season" not in g.columns:
        g["season"] = g["game_id"].map(season_from_game_id)
    g["neutral"] = (g["neutral"].map(lambda v: bool(v) if pd.notna(v) else False)
                    if "neutral" in g.columns else False)
    bad = sorted({c for c in pd.concat([g["home"], g["away"]]) if c not in teams.TEAMS})
    if bad:
        raise ValueError(f"nba_games_df has non-canonical team codes {bad[:10]}")
    if (g["home"] == g["away"]).any():
        raise ValueError("nba_games_df has a team playing itself")
    if g["game_id"].duplicated().any():
        raise ValueError(f"duplicate game_id in nba_games_df: "
                         f"{g.loc[g['game_id'].duplicated(), 'game_id'].tolist()[:5]}")
    # Unordered key: two games between the same pair on one date would be
    # ambiguous for the neutral-site rule, and never happen in the NBA.
    pair = g.apply(lambda r: (r["game_date"], *sorted((r["home"], r["away"]))), axis=1)
    if pair.duplicated(keep=False).any():
        raise ValueError(f"duplicate (date, team pair) in nba_games_df: "
                         f"{g.loc[pair.duplicated(keep=False), ['game_id', 'game_date', 'home', 'away']].values.tolist()[:4]}")
    return g


def _orientations(r) -> list[tuple[str, str]]:
    # A neutral-site game whose log names no home team matches either way.
    return [(r.home, r.away), (r.away, r.home)] if r.neutral else [(r.home, r.away)]


def crosswalk(events_df: pd.DataFrame, nba_games_df: pd.DataFrame) -> pd.DataFrame:
    """ESPN event_id -> NBA game_id, one row per input event.

    nba_games_df: game_id, game_date (ET), home, away, optional season and
    neutral (True = the source names no home team; either orientation
    matches, the ESPN orientation is kept).
    how: exact     same ET date and teams
         date_pm1  no exact match; the only unclaimed game with the same
                   teams and season one day either side (the corrupted
                   Feb-2020 timestamps)
         unmatched / ambiguous   FINAL NBA-vs-NBA event with no single game
         not_final  postponed / scheduled / in progress: never matched, the
                    rescheduled game has its own event id
         preseason  never matched (not modelled)
    Matching is one-to-one; a game claimed exactly is never reused by the
    fallback. df.attrs['unclaimed_games'] lists NBA games inside the
    events' date range (same seasons) that no event matched."""
    if events_df["event_id"].duplicated().any():
        raise ValueError("duplicate event_id in events_df")
    g = _prep_games(nba_games_df)
    ev = events_df.copy()
    ev["et_date"] = pd.to_datetime(ev["et_date"]).dt.date
    exact_ix: dict[tuple, str] = {}
    by_teams: dict[tuple, list] = {}
    for r in g.itertuples():
        for h, a in _orientations(r):
            exact_ix[(r.game_date, h, a)] = r.game_id
            by_teams.setdefault((r.season, h, a), []).append((r.game_date, r.game_id))
    gdate = dict(zip(g["game_id"], g["game_date"]))

    out = []
    claimed: dict[str, str] = {}
    eligible = []
    for r in ev.itertuples():
        base = {"event_id": r.event_id, "game_id": None, "season": r.season, "et_date": r.et_date,
                "nba_game_date": None, "home": r.home, "away": r.away, "status": r.status,
                "season_slug": r.season_slug, "comp_type": r.comp_type, "hint": None}
        if r.season_slug == "preseason":
            out.append({**base, "how": "preseason"})
        elif r.status != "STATUS_FINAL":
            out.append({**base, "how": "not_final"})
        else:
            gid = exact_ix.get((r.et_date, r.home, r.away))
            if gid is not None:
                if gid in claimed:
                    raise ValueError(f"game {gid} matched exactly by events {claimed[gid]} and {r.event_id}")
                claimed[gid] = r.event_id
                out.append({**base, "how": "exact", "game_id": gid, "nba_game_date": gdate[gid]})
            else:
                eligible.append((len(out), base))
                out.append(None)
    # Fallback after every exact match is claimed, so it cannot steal one.
    for pos, base in eligible:
        cands = sorted({gid for d, gid in by_teams.get((base["season"], base["home"], base["away"]), [])
                        if abs((d - base["et_date"]).days) == 1 and gid not in claimed})
        if len(cands) == 1:
            claimed[cands[0]] = base["event_id"]
            out[pos] = {**base, "how": "date_pm1", "game_id": cands[0],
                        "nba_game_date": gdate[cands[0]]}
        else:
            hint = None
            if base["comp_type"] == "CC":
                hint = "NBA Cup final: GAME_ID prefix 006, absent from Regular Season logs"
            elif base["season_slug"] == "play-in-season":
                hint = "play-in: GAME_ID prefix 005, needs the PlayIn game log"
            elif base["season_slug"] == "post-season":
                hint = "playoffs: GAME_ID prefix 004, needs the Playoffs game log"
            out[pos] = {**base, "how": "ambiguous" if len(cands) > 1 else "unmatched", "hint": hint}
    res = pd.DataFrame(out, columns=XWALK_COLUMNS)
    m = res["game_id"].notna()
    if res.loc[m, "game_id"].duplicated().any():
        raise ValueError("crosswalk is not one-to-one")
    # NBA games inside the events' span that nothing claimed.
    fin = res[res["how"].isin(["exact", "date_pm1", "unmatched", "ambiguous"])]
    if len(fin):
        lo, hi = min(fin["et_date"]), max(fin["et_date"])
        span = g[(g["game_date"] >= lo) & (g["game_date"] <= hi) & g["season"].isin(set(fin["season"]))]
        unclaimed = span[~span["game_id"].isin(set(claimed))]
    else:
        unclaimed = g.iloc[0:0]
    res.attrs["unclaimed_games"] = unclaimed.reset_index(drop=True)
    return res


# ── Live adapters (serve time) ───────────────────────────────────────────────
SCOREBOARD_COLUMNS = ["event_id", "start_utc", "et_date", "status", "season", "season_slug",
                      "comp_type", "home", "away", "neutral_site", "provider_id",
                      "provider_name", "details", "home_line", "total", "open_home_line",
                      "open_total", "home_ml", "away_ml", "qc_flags", "captured_at", "source"]


def _pick_scoreboard_odds(odds_list) -> tuple[dict | None, list[str]]:
    flags, cands = [], []
    for o in odds_list or []:
        name = _dig(o, "provider", "name")
        pid = _dig(o, "provider", "id")
        if _is_live_name(name) or str(pid) in {str(x) for x in KNOWN_LIVE}:
            flags.append(f"live:{pid}")
            continue
        cands.append(o)
    # DraftKings (100) is the serve-time book; anything else is a fallback.
    cands.sort(key=lambda o: 0 if str(_dig(o, "provider", "id")) == "100" else 1)
    return (cands[0] if cands else None), flags


def parse_scoreboard(payload: dict, captured_at: dt.datetime) -> pd.DataFrame:
    """Scoreboard payload -> one row per NBA-vs-NBA game with its odds
    snapshot (NaN when the game is not priced yet or has tipped).

    home_line/total are the item's current values (before tip the
    scoreboard's 'close' IS the current line); open values come from the
    pointSpread/total open objects. df.attrs['dropped'] counts non-NBA and
    All-Star events."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    ev, dropped = parse_events(payload)
    by_id = {str(e.get("id")): e for e in payload.get("events") or []}
    rows = []
    for r in ev.itertuples():
        c = (by_id[r.event_id].get("competitions") or [{}])[0]
        o, flags = _pick_scoreboard_odds(c.get("odds"))
        row = {"event_id": r.event_id, "start_utc": r.start_utc, "et_date": r.et_date,
               "status": r.status, "season": r.season, "season_slug": r.season_slug,
               "comp_type": r.comp_type, "home": r.home, "away": r.away,
               "neutral_site": r.neutral_site, "provider_id": None, "provider_name": None,
               "details": None, "home_line": math.nan, "total": math.nan,
               "open_home_line": math.nan, "open_total": math.nan, "home_ml": None,
               "away_ml": None, "captured_at": captured_at.astimezone(util.UTC),
               "source": "espn_scoreboard"}
        if o is not None:
            s = parse_line(o.get("spread"))
            t = parse_line(o.get("overUnder"))
            cs = parse_line(_dig(o, "pointSpread", "home", "close", "line"))
            ct = parse_line(_dig(o, "total", "over", "close", "line"))
            if s is None:
                s = cs
            elif cs is not None and cs != s:
                flags.append("spread_vs_close_mismatch")
            if t is None:
                t = ct
            os_ = parse_line(_dig(o, "pointSpread", "home", "open", "line"))
            ot = parse_line(_dig(o, "total", "over", "open", "line"))
            if s is not None and not _spread_ok(s):
                flags.append("spread_out_of_bounds")
                s = None
            if t is not None and not _total_ok(t):
                flags.append("total_out_of_bounds")
                t = None
            fav = details_favourite(o.get("details"), r.home, r.away)
            if s is not None and abs(s) >= SIGN_CHECK_MIN and fav in ("home", "away") \
                    and _sign_side(s) != fav:
                flags.append("sign_conflict")
                s = None
            row.update(
                provider_id=str(_dig(o, "provider", "id")), provider_name=_dig(o, "provider", "name"),
                details=o.get("details"),
                home_line=math.nan if s is None else s, total=math.nan if t is None else t,
                open_home_line=os_ if _spread_ok(os_) else math.nan,
                open_total=ot if _total_ok(ot) else math.nan,
                home_ml=parse_american(_dig(o, "moneyline", "home", "close", "odds")),
                away_ml=parse_american(_dig(o, "moneyline", "away", "close", "odds")),
            )
        row["qc_flags"] = ";".join(flags)
        rows.append(row)
    df = pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    for col in ("home_ml", "away_ml"):
        df[col] = df[col].astype("Int64")
    df.attrs["dropped"] = dropped
    return df


def scoreboard(date_et) -> pd.DataFrame:
    """Live: today's (or any ET date's) games with the current odds
    snapshot. captured_at is the fetch time; persist it, because the odds
    vanish from this surface at tip.

    date_et: a date, an ISO date string, or an AWARE datetime (converted to
    its ET date: 02:00 UTC on the 21st is the 20th's slate). A naive
    datetime raises."""
    if isinstance(date_et, dt.datetime):  # before date: datetime is a date subclass
        d = util.et_date_of(date_et)
    elif isinstance(date_et, dt.date):
        d = date_et
    else:
        d = dt.date.fromisoformat(str(date_et))
    captured_at = util.now_utc()
    payload = fetch_scoreboard(d.strftime("%Y%m%d"))
    return parse_scoreboard(payload, captured_at)


INJURY_COLUMNS = ["team", "espn_team_id", "espn_athlete_id", "name", "status", "status_type",
                  "fantasy_status", "return_date", "short_comment", "date", "feed_timestamp",
                  "captured_at"]
_ATHLETE_ID_RE = re.compile(r"/id/(\d+)(?:/|$)")


def _athlete_id_from_links(athlete: dict) -> str:
    ids = {m.group(1) for link in athlete.get("links") or []
           for m in [_ATHLETE_ID_RE.search(str(link.get("href") or ""))] if m}
    if len(ids) != 1:
        raise ValueError(f"athlete {athlete.get('displayName')!r}: expected one id in links, got {sorted(ids)}")
    return ids.pop()


def parse_injuries(payload: dict, captured_at: dt.datetime) -> pd.DataFrame:
    """Injuries payload -> one row per (team, athlete).

    team is the ENCLOSING group's team (athlete.team is stale after a
    trade); espn_athlete_id comes from the athlete links (the item id is the
    injury's id). Unknown team groups and athletes without an id raise. A
    duplicate athlete keeps its most recent entry; the drop is counted in
    attrs['duplicates_dropped']. attrs also reports stale athlete.team rows
    and the teams with no entries (absent from the feed = no injuries)."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    groups = payload.get("injuries")
    if not isinstance(groups, list):
        raise ValueError("injuries payload has no 'injuries' list")
    rows, stale = [], 0
    for grp in groups:
        team = teams.from_espn_id(grp.get("id"))
        if team is None:
            raise ValueError(f"injury group for unknown team id {grp.get('id')!r} "
                             f"({grp.get('displayName')!r})")
        for it in grp.get("injuries") or []:
            a = it.get("athlete") or {}
            aid = _athlete_id_from_links(a)
            if str(_dig(a, "team", "id") or grp.get("id")) != str(grp.get("id")):
                stale += 1
            ret = _dig(it, "details", "returnDate")
            when = it.get("date")
            rows.append({
                "team": team, "espn_team_id": int(grp["id"]), "espn_athlete_id": aid,
                "name": a.get("displayName"), "status": it.get("status"),
                "status_type": _dig(it, "type", "name"),
                "fantasy_status": _dig(it, "details", "fantasyStatus", "abbreviation"),
                "return_date": dt.date.fromisoformat(ret[:10]) if ret else None,
                "short_comment": it.get("shortComment"),
                "date": util.to_et(when).astimezone(util.UTC) if when else None,
                "feed_timestamp": payload.get("timestamp"),
                "captured_at": captured_at.astimezone(util.UTC),
            })
    df = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    dup = df.sort_values("date").duplicated("espn_athlete_id", keep="last")
    n_dup = int(dup.sum())
    df = df.sort_values("date")[~dup].sort_values(["team", "name"]).reset_index(drop=True)
    df.attrs["duplicates_dropped"] = n_dup
    df.attrs["stale_athlete_team"] = stale
    df.attrs["teams_without_entries"] = sorted(set(teams.CODES) - set(df["team"]))
    return df


def injuries() -> pd.DataFrame:
    """Live: the current ESPN injury list. ESPN keeps no history (a 2015
    game's summary shows TODAY's injuries), so the bot must snapshot this."""
    captured_at = util.now_utc()
    return parse_injuries(http.fetch(SITE + "injuries"), captured_at)


ROSTER_COLUMNS = ["team", "espn_athlete_id", "full_name", "date_of_birth", "jersey",
                  "position", "captured_at"]


def parse_roster(payload: dict, team: str, captured_at: dt.datetime) -> pd.DataFrame:
    code = teams.get(team).code
    got = teams.from_espn_id(_dig(payload, "team", "id"))
    if got != code:
        raise ValueError(f"roster payload is for {got!r}, asked for {code}")
    raw = payload.get("athletes")
    if not isinstance(raw, list):
        raise ValueError("roster payload has no athletes list")
    # Some ESPN sports group athletes by position ({items: [...]}); NBA is flat.
    flat = []
    for a in raw:
        flat.extend(a["items"] if isinstance(a, dict) and "items" in a else [a])
    rows = []
    for a in flat:
        aid = str(a.get("id") or "")
        if not aid.isdigit():
            raise ValueError(f"roster athlete without a numeric id: {a.get('fullName')!r}")
        dob = a.get("dateOfBirth")
        rows.append({
            "team": code, "espn_athlete_id": aid,
            "full_name": a.get("fullName") or a.get("displayName"),
            # ESPN writes a birth date as midnight Pacific in UTC
            # ('2001-01-10T08:00Z'); the calendar date is the prefix.
            "date_of_birth": dt.date.fromisoformat(dob[:10]) if dob else None,
            "jersey": a.get("jersey"),
            "position": _dig(a, "position", "abbreviation"),
            "captured_at": captured_at.astimezone(util.UTC),
        })
    df = pd.DataFrame(rows, columns=ROSTER_COLUMNS)
    if df["espn_athlete_id"].duplicated().any():
        raise ValueError(f"{code} roster lists an athlete twice")
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


def team_roster(team: str) -> pd.DataFrame:
    """Live: the current ESPN roster of one team (any alias)."""
    t = teams.get(team)
    captured_at = util.now_utc()
    return parse_roster(http.fetch(SITE + f"teams/{t.espn_id}/roster"), t.code, captured_at)


def match_athletes(espn_df: pd.DataFrame, nba_df: pd.DataFrame) -> pd.DataFrame:
    """ESPN athlete -> NBA PERSON_ID. No payload carries a crosswalk, so:
    name (names.NameIndex, team-scoped) confirmed by date of birth, else a
    unique same-DOB same-team player. A name match whose DOB disagrees is
    REJECTED (two players can share a name; they rarely share a birthday).

    espn_df: espn_athlete_id, full_name, date_of_birth, team
    nba_df:  person_id, name, birthdate, team
    -> espn_athlete_id, person_id (None when unmatched), how."""
    for need, df, nm in ((("espn_athlete_id", "full_name", "date_of_birth", "team"), espn_df, "espn_df"),
                         (("person_id", "name", "birthdate", "team"), nba_df, "nba_df")):
        miss = set(need) - set(df.columns)
        if miss:
            raise ValueError(f"{nm} is missing {sorted(miss)}")
    def as_date(v):
        # Timestamps are dates too (subclass), so normalise everything.
        if v is None or (not isinstance(v, (dt.date, str)) and pd.isna(v)):
            return None
        return pd.Timestamp(v).date()

    idx = NameIndex()
    dob_of, team_of = {}, {}
    for r in nba_df.itertuples():
        idx.add(r.name, r.person_id, team=r.team)
        dob_of[r.person_id] = as_date(r.birthdate)
        team_of[r.person_id] = r.team
    out = []
    for r in espn_df.itertuples():
        dob = as_date(r.date_of_birth)
        m = idx.get(r.full_name, team=r.team)
        pid, how = None, "none"
        if m.value is not None:
            nd = dob_of.get(m.value)
            if dob is None or nd is None:
                pid, how = m.value, f"name_{m.how}"
            elif nd == dob:
                pid, how = m.value, f"name_{m.how}+dob"
            else:
                how = "dob_conflict"
        if pid is None and how != "dob_conflict" and dob is not None:
            same = [p for p, d in dob_of.items() if d == dob and team_of[p] == r.team]
            if len(same) == 1:
                pid, how = same[0], "dob+team"
        out.append({"espn_athlete_id": r.espn_athlete_id, "person_id": pid, "how": how})
    res = pd.DataFrame(out, columns=["espn_athlete_id", "person_id", "how"])
    res["person_id"] = res["person_id"].astype("Int64")
    hit = res["person_id"].notna()
    if res.loc[hit, "person_id"].duplicated().any():
        dups = res.loc[hit & res["person_id"].duplicated(keep=False)]
        raise ValueError(f"two ESPN athletes matched one PERSON_ID: {dups.values.tolist()[:4]}")
    return res


# ── Misc ─────────────────────────────────────────────────────────────────────
def timed(fn, *a, **kw):
    """(result, seconds). Used by the backfill's latency report."""
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t
