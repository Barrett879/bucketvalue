"""Per-game participation: every listed player per team-game, whether he
played, sat (DNP), or was inactive, whether he started, and why he sat.

WHY THIS EXISTS. The bulk game logs (LeagueGameLog / PlayerGameLogs) list only
players who APPEARED, so anything trained on them alone estimates
E[stat | played]. Availability (P(plays)) and minutes need the denominator:
every rostered player per team-game, including coach's-decision DNPs and
inactives, plus who started. One row per (game_id, person_id).

TWO SOURCES BY ERA (live-verified 2026-10-08):

  2019-20 on   cdn.nba.com liveData boxscore_{game_id}.json, one call per game.
               Each team's players[] carries status ACTIVE/INACTIVE, starter
               '1'/'0', played '1'/'0', notPlayingReason (INACTIVE_INJURY,
               INACTIVE_GLEAGUE_TWOWAY, DND_INJURY, INACTIVE_NOT_WITH_TEAM ...)
               and notPlayingDescription. Older ids answer with the S3 XML 403
               (Absent); an Akamai HTML 403 is Blocked and must stop a run.
  2014-15 to   stats.nba.com BoxScoreTraditionalV3 (players incl. DNPs, reason
  2018-19      in `comment`) + BoxScoreSummaryV3 (inactives, no reason), two
               calls per game. NEVER the V2 endpoints: they return HTTP 200 with
               zero rows for 2025-26 games (a silent empty, not an error).

TRAPS, each one seen in a real payload:
  - PLAYED is a flag, not "minutes > 0". A 1-second cameo ('PT00M01.00S',
    LeagueGameLog MIN 0) played. cdn: played == '1'. V3: minutes non-empty.
  - V3 has no starter flag. Starters are the FIRST FIVE players per team in
    list order; `position` is filled for bench players before ~2018-19
    (0021400500: DEN bench tagged G/C/F). The first-five rule is VERIFIED per
    team against V3's own `starters` aggregate (seconds and points must match),
    so a payload that breaks the ordering raises instead of mislabelling.
  - V3 `gameEt` is Eastern wall-clock time with a fake 'Z' suffix
    ('2024-01-07T18:00:00Z' for a 6pm ET tip). Only `gameTimeUTC` is UTC.
  - cdn 2019-20 payloads say gameStatusText 'finalbox', not 'Final'; status is
    read from gameStatus (3 = final). Some 2019-20 payloads put the STRING
    'finalbox' in gameStatus itself and drop regulationPeriods (0021900718:
    a complete box, team minutes 240, points = score). That string is final
    too (cdn only); the first full 2019-20 run filed 89 such games as not
    final. A missing or null regulationPeriods means the NBA's 4.
  - Bubble-era cdn payloads (0021901260, 0041900161 ...) give players who did
    not play statistics.minutes = null instead of 'PT00M00.00S'. That is 0
    seconds for a played='0' row; a null on a played='1' row still raises.
  - cdn 2019-20 DNPs often carry NO notPlayingReason at all, and its reason
    codes are noisy (Andre Iguodala, holding out, is INACTIVE_GLEAGUE_TWOWAY;
    INACTIVE_COACH rows carry injury text). reason_code is kept as-is ('' when
    absent); status comes from the status/played flags, never from the reason.
  - cdn lists players who are no longer with the team (INACTIVE_NOT_WITH_TEAM,
    description 'Waived'). They are inactive rows, not rostered players; the
    coverage report counts them so a denominator can drop them deliberately.
  - Names and jerseys carry stray whitespace ('  Nene', V3 jersey '41  ').
  - A box score fetched before the final buzzer is partial. Only gameStatus 3
    payloads are cached or parsed (NotFinal otherwise).
  - status 'dnp' is NOT "the coach chose not to play him". It holds every
    listed, non-inactive player who did not play: coach's decisions, and also
    players who dressed (or did not dress) while hurt: cdn DND_INJURY /
    DNP_INJURY / NWT_* and V3 'DND - ...' / 'NWT - ...'. An availability model
    reads `absence_kind` (coach / unavailable / unknown), never `status` alone.
    INACTIVE_COACH is a coach's decision too, except in 2019-20, which files
    injuries under it ('Left Ankle; Sprain'): the description wins there.
  - V3 comments usually read 'NWT - Personal', but some 2019-20 payloads carry
    the cdn code instead ('NWT_LEAGUE_SUSPENSION', 0021900723, the Grizzlies
    suspended for leaving the bench). Both spellings parse to the prefix.
  - V3 answers an id it does not know with HTTP 200 and an all-null
    boxScoreSummary (gameId None, gameStatus 0; live 0021601300). That is
    Absent, not a malformed payload: it must not stop a run.
  - THE 2026-27 cdn BOX LISTS ACTIVE PLAYERS ONLY. Through 2025-26 every
    listed player per team (median 18 in 2025-26) came with status ACTIVE or
    INACTIVE, the inactive ones with a notPlayingReason; from the 2026-27
    opener a box lists the dressed players only (11-16 a team, median 13, in
    the first 92 games) and no INACTIVE row anywhere. Training's availability
    history has a row for every listed player in every game, so a box that
    lists no INACTIVE player on either side (lists_inactives) gets its
    inactive rows REBUILT (rebuild_inactives): the team's roster at game time
    minus the box, each with the reason the last injury report before tip
    gives him (REPORT_REASON_CODES, the cdn vocabulary) or UNLISTED, source
    ROSTER_MINUS_BOX. A box that lists an inactive player on either side is
    the old format and is never touched (three 2025-26 team-games listed none,
    BOS twice in the regular season and CHA in the play-in, 15 dressed each,
    while the other side listed some). In a play-in or playoff game the
    roster's two-way players get no row (TWO_WAY_INELIGIBLE).

THE GAME LIST (the denominator of every season table), by season state:

  live season  The cdn schedule (scheduleLeagueV2_1.json), final games only.
               It is the one game list reachable from GitHub Actions: the
               2026-10-09 probe saw stats.nba.com LeagueGameLog hang from Azure.
  past season  LeagueGameLog T (Regular Season, PlayIn, Playoffs), CHECKED
               against ScheduleLeagueV2: the same final games, dates and teams
               (they agreed exactly in 2014-15, 2019-20, 2020-21, 2024-25 and
               2025-26). The schedule also contributes the NBA Cup Final (006
               ids, 2023-24 on), which no LeagueGameLog lists although both
               teams play it.
  A LeagueGameLog payload fetched while its season was still live is
  provisional and is refetched; one fetched after the season ended is reused
  forever. An empty game list is legitimate only for a live season (or the
  play-in before 2019-20); anywhere else it is a silent empty and is refused
  without being cached.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .. import cache, http, teams
from ..util import UTC, et_date_of, season_end_year, season_for_date, season_str, today_et
from . import statsnba as _stats
from .cdn import SCHEDULE_URL as CDN_SCHEDULE_URL

# ── Constants ────────────────────────────────────────────────────────────────
CDN_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
CDN_FIRST_SEASON = "2019-20"     # older ids are S3-absent on the CDN
RAW_SUBDIR = "raw/box"

STATUSES = ("played", "dnp", "inactive")
# A rebuilt inactive row (rebuild_inactives): a rostered player the box does
# not list. Its reason codes are cdn's vocabulary; its game's source is cdn.
ROSTER_MINUS_BOX = "roster_minus_box"
SOURCES = ("cdn", "v3", ROSTER_MINUS_BOX)
# The reason of a rebuilt inactive row the last report before tip does not name.
UNLISTED = "INACTIVE_UNLISTED"
# Why a listed player did not play: '' (he played), 'coach' (a healthy scratch:
# the reason names the coach, or a cdn ACTIVE row with no reason, see
# absence_kind), 'unavailable' (injury, illness, rest, personal, suspension,
# not with the team, G League, two-way, trade), 'unknown' (no reason given: a
# cdn INACTIVE row with an empty reason, and every V3 inactive).
ABSENCE_KINDS = ("", "coach", "unavailable", "unknown")
COLUMNS = ["season", "game_id", "team", "is_home", "person_id", "player_name",
           "jersey", "status", "starter", "played_seconds", "reason_code",
           "reason_text", "absence_kind", "source"]
META_COLUMNS = ["game_id", "season", "season_type", "tip_utc", "et_date", "arena",
                "arena_city", "home", "away", "periods", "ot", "source"]

# game id = '00' + type digit + 2-digit season start year + 5-digit sequence.
SEASON_TYPES = {"1": "Pre Season", "2": "Regular Season", "3": "All Star",
                "4": "Playoffs", "5": "PlayIn", "6": "NBA Cup Final"}
_GID_RE = re.compile(r"^00([1-6])(\d{2})(\d{5})$")

_ISO_MIN = re.compile(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")
_MMSS = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")
# V3 comment prefixes: Did Not Play, Did Not Dress, Not With Team. Parse on the
# prefix: the remainder mixes generic and specific text ('DND - Injury/Illness'
# vs 'DND - Left Lower Leg Contusion'). The separator is ' - ' or, in some
# 2019-20 payloads, the cdn code's underscore ('NWT_LEAGUE_SUSPENSION'): a
# regex word boundary does not fire between 'NWT' and '_', which is how the
# first full 2019-20 run failed two games.
_COMMENT_RE = re.compile(r"^(DNP|DND|NWT)(?=$|[\s_-])[\s_-]*(.*)$", re.S)
_CDN_REASON_PREFIXES = ("DNP", "DND", "NWT", "INACTIVE")

# Regulation is 4 x 12 minutes, each overtime 5 minutes, five players on court:
# a team's player-seconds are 14,400 + 1,500 per OT (LeagueGameLog T MIN
# 240/265/290/315). A parse that loses or double-counts a player breaks this
# by whole minutes; real feeds agree to the second in every sampled game.
REG_TEAM_SECONDS = 4 * 12 * 60 * 5
OT_TEAM_SECONDS = 5 * 60 * 5
TEAM_SECONDS_TOL = 60

# ── Call accounting (live calls only; cache hits are not counted) ────────────
CALL_LOG: dict[str, list[float]] = defaultdict(list)


def _timed(kind: str, fn, *a, **k):
    t = time.monotonic()
    try:
        return fn(*a, **k)
    finally:
        CALL_LOG[kind].append(time.monotonic() - t)


class NotFinal(Exception):
    """The game exists but is not final (or not started): its box score is
    partial. Never cached, never parsed into the participation table."""


# ── Game ids ─────────────────────────────────────────────────────────────────
def _gid_parts(game_id: str) -> tuple[str, int]:
    m = _GID_RE.match(str(game_id))
    if not m:
        raise ValueError(f"not an NBA game id: {game_id!r} (want 10 digits, '00' + type + yy + seq)")
    return m.group(1), int(m.group(2))


def game_season(game_id: str) -> str:
    """'0021400500' -> '2014-15'."""
    _, yy = _gid_parts(game_id)
    return season_str(2000 + yy + 1)


def season_type_of(game_id: str) -> str:
    return SEASON_TYPES[_gid_parts(game_id)[0]]


def default_source(game_id: str) -> str:
    """'cdn' from 2019-20 on (one call, reasons included), else 'v3'."""
    return "cdn" if game_season(game_id) >= CDN_FIRST_SEASON else "v3"


# ── Raw payload cache ────────────────────────────────────────────────────────
RAW_KINDS = ("cdn", "cdnabsent", "v3box", "v3sum")


def raw_path(game_id: str, kind: str) -> Path:
    """cache/raw/box/{season}/{kind}_{game_id}.json.gz. kind: cdn | v3box |
    v3sum, or cdnabsent (a marker: the CDN answered S3-absent for this id)."""
    if kind not in RAW_KINDS:
        raise ValueError(f"raw kind must be one of {RAW_KINDS}, got {kind!r}")
    return cache.dc_path(f"{RAW_SUBDIR}/{game_season(game_id)}/{kind}_{game_id}.json.gz")


def _read_raw(path: Path):
    data, _ = cache.read_json_stamped(path)
    return data


def _final_status(payload_game: dict, game_id: str, where: str) -> None:
    st = payload_game.get("gameStatus")
    if st != 3 and not (where == "cdn" and st == "finalbox"):
        raise NotFinal(f"{game_id}: {where} gameStatus={st!r} "
                       f"({payload_game.get('gameStatusText')!r}), not final")


def is_final(payload_game: dict, game_id: str, where: str = "cdn") -> bool:
    """Is this box-score game object final, by the ONE rule the parsers
    refuse a partial box under (_final_status: gameStatus 3, or the cdn
    payload's "finalbox")? `payload_game` is the payload's game object (the
    cdn liveData `game`, or the BoxScoreSummaryV3 summary with
    where="BoxScoreSummaryV3"). For readers that show a box in progress
    (live.box on the Game page after tip) instead of refusing it."""
    try:
        _final_status(payload_game, game_id, where)
    except NotFinal:
        return False
    return True


def fetch_cdn(game_id: str, refresh: bool = False) -> dict:
    """Raw cdn box score, cached gz under cache/raw/box/{season}/.

    Raises http.Absent (no such object: pre-2019-20 ids), http.Blocked (Akamai
    HTML 403 / hang: stop the run), NotFinal (not cached), BadPayload.
    """
    p = raw_path(game_id, "cdn")
    if p.exists() and not refresh:
        return _read_raw(p)
    try:
        payload = _timed("cdn", http.fetch, CDN_URL.format(gid=game_id), timeout=20, retries=2)
    except http.Absent as e:
        # Remember it, so a rerun or --reparse goes straight to V3 instead of
        # re-probing. Only the S3 XML 403 lands here; an Akamai block is
        # Blocked and is never recorded as absence.
        cache.write_json(raw_path(game_id, "cdnabsent"), {"url": e.url, "detail": e.detail},
                         stamped=True, gz=True)
        raise
    game = payload.get("game") if isinstance(payload, dict) else None
    if not isinstance(game, dict) or game.get("gameId") != game_id:
        raise http.BadPayload(CDN_URL.format(gid=game_id),
                              f"no game object for {game_id} (keys {list(payload)[:5]})")
    _final_status(game, game_id, "cdn")
    cache.write_json(p, payload, stamped=True, gz=True)
    return payload


def _v3_null_summary(body) -> bool:
    """V3's answer for a game id it does not know: HTTP 200, a boxScoreSummary
    whose gameId, teams and arena are all null and gameStatus is 0."""
    return (isinstance(body, dict) and body.get("gameId") is None
            and body.get("homeTeam") is None and body.get("awayTeam") is None)


def fetch_v3(game_id: str, refresh: bool = False) -> tuple[dict, dict]:
    """(BoxScoreTraditionalV3, BoxScoreSummaryV3) raw JSON, each cached gz.
    The summary is fetched first: it carries gameStatus, and a non-final game
    caches neither file. An all-null summary raises http.Absent (V3 has no such
    game); any other summary or box without this game id is BadPayload."""
    from nba_api.stats.endpoints import boxscoresummaryv3, boxscoretraditionalv3

    out = {}
    for kind, cls, root in (("v3sum", boxscoresummaryv3.BoxScoreSummaryV3, "boxScoreSummary"),
                            ("v3box", boxscoretraditionalv3.BoxScoreTraditionalV3,
                             "boxScoreTraditional")):
        p = raw_path(game_id, kind)
        if p.exists() and not refresh:
            out[kind] = _read_raw(p)
            continue
        payload = _timed(kind, http.stats_call, cls, timeout=30, retries=2, game_id=game_id)
        body = payload.get(root) if isinstance(payload, dict) else None
        url = f"{http.STATS_BASE}{cls.endpoint}?GameID={game_id}"
        if kind == "v3sum" and _v3_null_summary(body):
            # Before this, the message read "no boxScoreSummary object (keys
            # ['meta', 'boxScoreSummary'])" and, as BadPayload, ended a whole
            # --keep-going run with exit 3.
            raise http.Absent(url, f"{game_id}: V3 has no such game (all-null boxScoreSummary)")
        if not isinstance(body, dict) or body.get("gameId") != game_id:
            raise http.BadPayload(url, f"{game_id}: no {root} object for this game "
                                       f"(keys {list(payload)[:5]})")
        if kind == "v3sum":
            _final_status(body, game_id, "BoxScoreSummaryV3")
        cache.write_json(p, payload, stamped=True, gz=True)
        out[kind] = payload
    return out["v3box"], out["v3sum"]


# ── Field parsers ────────────────────────────────────────────────────────────
def iso_seconds(value: str) -> float:
    """'PT25M01.00S' -> 1501.0. Anything else raises."""
    m = _ISO_MIN.match(str(value or "").strip())
    if not m:
        raise ValueError(f"unparseable ISO minutes {value!r}")
    return int(m.group(1)) * 60 + float(m.group(2))


def mmss_seconds(value: str) -> float:
    """'35:20' -> 2120.0. Anything else raises (callers handle '' = DNP)."""
    m = _MMSS.match(str(value or "").strip())
    if not m:
        raise ValueError(f"unparseable MM:SS minutes {value!r}")
    return int(m.group(1)) * 60 + float(m.group(2))


def _clean(s) -> str:
    return " ".join(str(s or "").split())


def _team_code(tricode, team_id, game_id: str) -> str:
    code = teams.canonical(tricode)
    if code is None or code != str(tricode).strip():
        raise ValueError(f"{game_id}: unknown or non-canonical team code {tricode!r}")
    by_id = teams.from_team_id(team_id)
    if by_id is not None and by_id != code:
        raise ValueError(f"{game_id}: team id {team_id} is {by_id}, payload says {code}")
    return code


def parse_comment(comment: str) -> tuple[str, str]:
    """V3 DNP comment -> (prefix code, remainder). '' -> ('', '').
    'DNP - Coach's Decision' -> ('DNP', "Coach's Decision");
    'NWT_LEAGUE_SUSPENSION' -> ('NWT', 'LEAGUE_SUSPENSION').
    An unrecognised non-empty comment raises: a new prefix must be added on
    purpose, not filed as a reasonless DNP."""
    c = _clean(comment)
    if not c:
        return "", ""
    m = _COMMENT_RE.match(c)
    if not m:
        raise ValueError(f"unrecognised V3 comment {comment!r}")
    return m.group(1), m.group(2).strip()


def absence_kind(status: str, reason_code: str, reason_text: str, source: str) -> str:
    """Why a listed player did not play, from the source's own reason fields.
    See ABSENCE_KINDS. An unknown cdn reason prefix raises (classify it on
    purpose), as does a V3 code other than DNP/DND/NWT/INACTIVE."""
    if status == "played":
        return ""
    if status not in ("dnp", "inactive"):
        raise ValueError(f"unknown status {status!r}")
    code, text = reason_code or "", (reason_text or "").lower()
    if source == "cdn":
        if code == UNLISTED:
            # A rebuilt inactive row no report names: no reason is known,
            # exactly like a cdn INACTIVE row with an empty reason.
            return "unknown"
        if not code:
            # The cdn never says DNP_COACH outside 2019-20: an ACTIVE player
            # who did not play with no reason is its coach's decision. On the
            # four games cached from both sources (0021901232 in the bubble,
            # 0022300061, 0022300500, 0022500001) all 12 such rows are V3
            # "DNP - Coach's Decision". An INACTIVE row with no reason stays
            # unknown.
            return "coach" if status == "dnp" else "unknown"
        prefix, _, rest = code.partition("_")
        if prefix not in _CDN_REASON_PREFIXES or not rest:
            raise ValueError(f"unrecognised cdn reason code {code!r}")
        if rest == "COACH":
            # 2019-20 files injuries under INACTIVE_COACH ('Left Ankle; Sprain',
            # 'Illness'): when the description names something other than the
            # coach, the description wins.
            return "coach" if not text or "coach" in text else "unavailable"
        return "unavailable"
    if source == "v3":
        if code in ("", "INACTIVE"):   # V3 inactives never carry a reason
            return "unknown"
        if code not in ("DNP", "DND", "NWT"):
            raise ValueError(f"unrecognised V3 reason code {code!r}")
        if "coach" in text:            # 'DNP - Coach's Decision', 'DND - Coach's Decision'
            return "coach"
        return "unavailable" if text else "unknown"
    raise ValueError(f"unknown source {source!r}")


# ── Parsers ──────────────────────────────────────────────────────────────────
def _expected_team_seconds(ot: int) -> int:
    return REG_TEAM_SECONDS + OT_TEAM_SECONDS * ot


def parse_cdn(payload: dict, game_id: str) -> tuple[pd.DataFrame, dict]:
    """cdn box score -> (participation rows, game meta)."""
    g = payload.get("game") if isinstance(payload, dict) else None
    if not isinstance(g, dict) or g.get("gameId") != game_id:
        raise ValueError(f"{game_id}: cdn payload has no matching game object")
    _final_status(g, game_id, "cdn")
    season = game_season(game_id)
    meta = _meta(game_id, season, tip_utc=g.get("gameTimeUTC"), game_code=g.get("gameCode"),
                 arena=g.get("arena") or {}, home=g.get("homeTeam"), away=g.get("awayTeam"),
                 period=g.get("period"), reg=g.get("regulationPeriods") or 4, source="cdn")
    rows = []
    for side, is_home in (("homeTeam", True), ("awayTeam", False)):
        t = g[side]
        code = _team_code(t.get("teamTricode"), t.get("teamId"), game_id)
        players = t.get("players") or []
        if not players:
            raise ValueError(f"{game_id}: cdn {code} has no players")
        for p in players:
            st, played_flag = p.get("status"), p.get("played")
            if st not in ("ACTIVE", "INACTIVE") or played_flag not in ("0", "1"):
                raise ValueError(f"{game_id}: {code} {p.get('personId')} status={st!r} "
                                 f"played={played_flag!r}")
            played = played_flag == "1"
            minutes = (p.get("statistics") or {}).get("minutes")
            try:
                secs = 0.0 if (not played and minutes in (None, "")) else iso_seconds(minutes)
            except ValueError as e:
                raise ValueError(f"{game_id}: {code} {p.get('personId')}: {e}") from None
            if played and st == "INACTIVE":
                raise ValueError(f"{game_id}: {code} {p.get('personId')} INACTIVE but played")
            if not played and secs > 0:
                raise ValueError(f"{game_id}: {code} {p.get('personId')} played=0 with {secs}s")
            status = "played" if played else ("inactive" if st == "INACTIVE" else "dnp")
            reason = "" if played else _clean(p.get("notPlayingReason"))
            text = "" if played else _clean(p.get("notPlayingDescription"))
            name = _clean(p.get("name")) or _clean(f"{p.get('firstName', '')} {p.get('familyName', '')}")
            try:
                kind = absence_kind(status, reason, text, "cdn")
            except ValueError as e:
                raise ValueError(f"{game_id}: {code} {p.get('personId')}: {e}") from None
            rows.append((season, game_id, code, is_home, int(p["personId"]), name,
                         _clean(p.get("jerseyNum")), status, p.get("starter") == "1",
                         int(round(secs)) if played else 0, reason, text, kind, "cdn"))
    df = _frame(rows)
    check_participation(df, ot=meta["ot"])
    return df, meta


def _v3_starters_check(t: dict, code: str, game_id: str) -> None:
    """The first-five rule, verified against V3's own `starters` aggregate."""
    agg = t.get("starters")
    if not isinstance(agg, dict) or "minutes" not in agg or "points" not in agg:
        raise ValueError(f"{game_id}: V3 {code} has no starters aggregate to verify first-5")
    first5 = t["players"][:5]
    secs = sum(mmss_seconds(p["statistics"]["minutes"]) for p in first5
               if p["statistics"].get("minutes"))
    pts = sum(int(p["statistics"].get("points") or 0) for p in first5)
    agg_secs = mmss_seconds(agg["minutes"]) if agg["minutes"] else 0.0
    if abs(secs - agg_secs) > 1 or pts != int(agg["points"] or 0):
        raise ValueError(f"{game_id}: V3 {code} first five ({secs:.0f}s, {pts} pts) do not match "
                         f"the starters aggregate ({agg_secs:.0f}s, {agg['points']} pts)")


def parse_v3(box: dict, summary: dict, game_id: str) -> tuple[pd.DataFrame, dict]:
    """BoxScoreTraditionalV3 + BoxScoreSummaryV3 -> (participation rows, meta)."""
    b = (box or {}).get("boxScoreTraditional")
    s = (summary or {}).get("boxScoreSummary")
    if not isinstance(b, dict) or b.get("gameId") != game_id:
        raise ValueError(f"{game_id}: V3 box payload has no matching boxScoreTraditional")
    if not isinstance(s, dict) or s.get("gameId") != game_id:
        raise ValueError(f"{game_id}: V3 summary payload has no matching boxScoreSummary")
    _final_status(s, game_id, "BoxScoreSummaryV3")
    season = game_season(game_id)
    meta = _meta(game_id, season, tip_utc=s.get("gameTimeUTC"), game_code=s.get("gameCode"),
                 arena=s.get("arena") or {}, home=s.get("homeTeam"), away=s.get("awayTeam"),
                 period=s.get("period"), reg=4, source="v3")
    rows = []
    for side, is_home in (("homeTeam", True), ("awayTeam", False)):
        t = b[side]
        code = _team_code(t.get("teamTricode"), t.get("teamId"), game_id)
        if code != meta["home" if is_home else "away"]:
            raise ValueError(f"{game_id}: V3 box {side} is {code}, summary says "
                             f"{meta['home' if is_home else 'away']}")
        players = t.get("players") or []
        if len(players) < 5:
            raise ValueError(f"{game_id}: V3 {code} lists {len(players)} players")
        _v3_starters_check(t, code, game_id)
        for i, p in enumerate(players):
            st = p.get("statistics") or {}
            minutes = _clean(st.get("minutes"))
            played = minutes != ""
            try:
                code_, text = parse_comment(p.get("comment"))
            except ValueError as e:
                # The game id must be in the message: the first full 2019-20
                # run logged "FAILED unrecognised V3 comment ..." with no id.
                raise ValueError(f"{game_id}: {code} {p.get('personId')}: {e}") from None
            if played and code_:
                raise ValueError(f"{game_id}: {code} {p.get('personId')} played {minutes} "
                                 f"but has comment {p.get('comment')!r}")
            secs = mmss_seconds(minutes) if played else 0.0
            name = _clean(f"{p.get('firstName', '')} {p.get('familyName', '')}")
            status = "played" if played else "dnp"
            rows.append((season, game_id, code, is_home, int(p["personId"]), name,
                         _clean(p.get("jerseyNum")), status, i < 5,
                         int(round(secs)), code_, text, absence_kind(status, code_, text, "v3"),
                         "v3"))
        # Inactives: summary only, and no reason is published for them.
        inact = s[side].get("inactives")
        if inact is None:
            raise ValueError(f"{game_id}: V3 summary {code} has no inactives key")
        for p in inact:
            rows.append((season, game_id, code, is_home, int(p["personId"]),
                         _clean(f"{p.get('firstName', '')} {p.get('familyName', '')}"),
                         _clean(p.get("jerseyNum")), "inactive", False, 0, "INACTIVE", "",
                         "unknown", "v3"))
    df = _frame(rows)
    check_participation(df, ot=meta["ot"])
    return df, meta


def _meta(game_id, season, *, tip_utc, game_code, arena, home, away, period, reg, source) -> dict:
    if not isinstance(home, dict) or not isinstance(away, dict):
        raise ValueError(f"{game_id}: payload is missing a team (home={type(home).__name__}, "
                         f"away={type(away).__name__})")
    h = _team_code(home.get("teamTricode"), home.get("teamId"), game_id)
    a = _team_code(away.get("teamTricode"), away.get("teamId"), game_id)
    if h == a:
        raise ValueError(f"{game_id}: home and away are both {h}")
    try:
        period, reg = int(period), int(reg)
    except (TypeError, ValueError):
        raise ValueError(f"{game_id}: bad period {period!r}/{reg!r}") from None
    if period < reg:
        raise ValueError(f"{game_id}: final game ended in period {period} < {reg}")
    # The league's own game date is the gameCode prefix ('20240107/NOPSAC').
    # tip_utc must land on the same ET date: this is the check that catches a
    # tip time stored in the wrong zone (V3's gameEt is ET with a fake 'Z').
    m = re.match(r"^(\d{4})(\d{2})(\d{2})/([A-Z]{3})([A-Z]{3})$", str(game_code or ""))
    if not m:
        raise ValueError(f"{game_id}: unparseable gameCode {game_code!r}")
    code_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    tip = pd.Timestamp(str(tip_utc).replace("Z", "+00:00"))
    if tip.tzinfo is None:
        raise ValueError(f"{game_id}: tip time {tip_utc!r} has no zone")
    tip = tip.tz_convert("UTC")
    et = et_date_of(tip.to_pydatetime()).isoformat()
    if et != code_date:
        raise ValueError(f"{game_id}: tip {tip_utc} is ET date {et}, gameCode says {code_date}")
    return {"game_id": game_id, "season": season, "season_type": season_type_of(game_id),
            "tip_utc": tip, "et_date": code_date, "arena": _clean(arena.get("arenaName")),
            "arena_city": _clean(arena.get("arenaCity")), "home": h, "away": a,
            "periods": period, "ot": period - reg, "source": source}


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype({"person_id": "int64", "played_seconds": "int64", "starter": bool,
                      "is_home": bool})


# ── Inactive rows the box no longer lists (the 2026-27 cdn format) ───────────
# The cdn notPlayingReason of an inactive row whose player the injury report
# names, by the report's reason category (injury.REASON_CATS, every one of
# them: a new category raises in report_reason until it is mapped here). Read
# off 2025-26, where both exist: the t_15 report's category of every INACTIVE
# row it names against the box's own code (INACTIVE_INJURY 6,393 of 6,563
# from injury_illness, INACTIVE_GLEAGUE_TWOWAY 2,667 of 3,184 from
# g_league_two_way, and so on). 'g_league' (2018-19's bare "G League Team")
# has no cdn twin and reads as an assignment. A listing with no reason, or one
# no rule classifies, gets cdn's empty reason (an INACTIVE row without one).
REPORT_REASON_CODES = {
    "injury_illness": "INACTIVE_INJURY",
    "concussion_protocol": "INACTIVE_CONCUSSION_PROTOCOL",
    "reconditioning": "INACTIVE_RETURN_TO_COMPETITION_RECONDITIONING",
    "health_protocols": "INACTIVE_HEALTH_AND_SAFETY_PROTOCOLS",
    "g_league_two_way": "INACTIVE_GLEAGUE_TWOWAY",
    "g_league_assignment": "INACTIVE_GLEAGUE_ON_ASSIGNMENT",
    "g_league": "INACTIVE_GLEAGUE_ON_ASSIGNMENT",
    "not_with_team": "INACTIVE_NOT_WITH_TEAM",
    "personal": "INACTIVE_PERSONAL",
    "suspension": "INACTIVE_LEAGUE_SUSPENSION",
    "coach_decision": "INACTIVE_COACH",
    "rest": "INACTIVE_REST",
    "trade_pending": "INACTIVE_TRADE",
    "unspecified": "",
    "other": "",
}
# The report rows rebuild_inactives reads: horizons.game_states' player rows
# (one game, both teams).
REPORT_COLUMNS = ["team", "player_raw", "status", "reason_cat", "reason_raw"]
# The games (season_type_of) a two-way contract cannot play in: after the
# regular season. A rebuilt inactive list of such a game leaves the roster's
# two-way players out (rebuild_inactives), the way the training listing
# mostly does: in 2021-22..2025-26, 38 of 60 play-in team-games and 824 of
# 844 playoff team-games listed 15 players or fewer (in 2025-26's play-in
# GSW, LAC, PHX, POR, CHA and ORL listed 15 and left out the two or three
# two-way players their last regular-season box had listed). The other
# play-in boxes listed theirs: inactive with INACTIVE_GLEAGUE_TWOWAY,
# INACTIVE_INELIGIBLE_TO_PLAY, INACTIVE_COACH or no reason, or even active.
# (2019-20 and 2020-21 let two-way players into the postseason: 16-17 a team.)
TWO_WAY_INELIGIBLE = ("PlayIn", "Playoffs")


def lists_inactives(payload: dict) -> bool:
    """Does this cdn box list an INACTIVE player on either side? Every cdn box
    of a regular-season or play-in game through 2025-26 does; no 2026-27 box
    does (it lists the dressed players only). The test is per GAME: an
    old-format side with nobody inactive (three 2025-26 team-games: BOS twice
    in the regular season and CHA in the play-in, 15 dressed each, the
    opponent listing its inactives as usual) is not the new format, and its
    game is parsed exactly as before."""
    g = payload.get("game") if isinstance(payload, dict) else None
    if not isinstance(g, dict):
        raise ValueError("cdn payload has no game object")
    return any(p.get("status") == "INACTIVE"
               for side in ("homeTeam", "awayTeam")
               for p in ((g.get(side) or {}).get("players") or []))


def report_reason(status, reason_cat, reason_raw) -> tuple[str, str]:
    """(reason_code, reason_text) of a rebuilt inactive row whose player the
    report names: the cdn code of his report category (REPORT_REASON_CODES)
    and the report's own words after his status ('Out: Injury/Illness - Left
    Ankle; Sprain'). A category the table does not map raises."""
    if reason_cat not in REPORT_REASON_CODES:
        raise ValueError(f"report reason category {reason_cat!r} has no cdn code: "
                         f"map it in box.REPORT_REASON_CODES")
    reason = _clean(reason_raw)
    text = _clean(status) + (f": {reason}" if reason and reason != "-" else "")
    return REPORT_REASON_CODES[reason_cat], text


def _first_last(last_first: str) -> str:
    """'Butler III, Jimmy' -> 'Jimmy Butler III' (a name for a player the
    report names whom no roster or history spells)."""
    last, sep, first = str(last_first).partition(",")
    return _clean(f"{first} {last}") if sep else _clean(last_first)


def _league_index(frames: list[pd.DataFrame]):
    """An index of every (person_id, player_name) in `frames`, with no team
    scoping: the second chance of a report row no roster of its game holds."""
    from ..names import NameIndex
    ix = NameIndex()
    for f in frames:
        for pid, name in zip(f["person_id"], f["player_name"]):
            if isinstance(name, str) and name:
                ix.add(name, int(pid))
    return ix


def _report_people(gid: str, listed: pd.DataFrame | None, rost: pd.DataFrame,
                   part: pd.DataFrame, known: pd.DataFrame | None,
                   roster_after: bool) -> pd.DataFrame:
    """The game's report rows (REPORT_COLUMNS) with person_id (Int64, NA when
    unresolved) and how ('team': livectx.resolve_names against the team's
    roster and box players under every spelling, the serving rule; 'league':
    an exact or alias match among every person `known`, the rosters and the
    box, for an Out row of a roster taken after the game only)."""
    cols = REPORT_COLUMNS + ["person_id", "how"]
    if listed is None or len(listed) == 0:
        return pd.DataFrame({c: pd.Series(dtype="Int64" if c == "person_id" else object)
                             for c in cols})
    missing = [c for c in REPORT_COLUMNS if c not in listed.columns]
    if missing:
        raise ValueError(f"{gid}: report rows lack {missing}")
    from .. import livectx
    teams_ = set(part["team"])
    inj = listed.loc[listed["team"].isin(teams_), REPORT_COLUMNS].reset_index(drop=True)
    cand = pd.concat([rost.loc[rost["team"].isin(teams_)], part[["person_id", "team", "player_name"]]],
                     ignore_index=True).drop_duplicates(["person_id", "team"])
    res = livectx.resolve_names(inj.assign(game_id=gid), cand.assign(game_id=gid), known)
    res["how"] = [("team" if p == p else None) for p in res["person_id"]]
    todo = res["person_id"].isna() & (res["status"] == "Out")
    if roster_after and todo.any():
        frames = [rost, part[["person_id", "player_name"]]]
        if known is not None and len(known):
            frames.append(known)
        ix = _league_index(frames)
        for i in res.index[todo]:
            m = ix.resolve_last_first(res.at[i, "player_raw"])
            if m.value is not None and m.how in ("exact", "alias"):
                res.at[i, "person_id"], res.at[i, "how"] = float(m.value), "league"
    res["person_id"] = res["person_id"].astype("Int64")
    return res[cols]


def rebuild_inactives(part: pd.DataFrame, roster: pd.DataFrame, listed: pd.DataFrame | None = None,
                      known: pd.DataFrame | None = None, roster_after: bool = False,
                      elsewhere: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """A box that lists no inactive player (lists_inactives is False) with its
    inactive rows rebuilt: the ONE definition the bot ingests by.

    part          parse_cdn rows of one such game
    roster        the roster at game time (state.roster_for_game: the cdn
                  playerIndex the bot pulled; person_id, team, name or
                  player_name, and two_way, which only a play-in or playoff
                  game reads and then requires), every team
    listed        the game's rows of the last injury report before tip
                  (horizons.game_states player rows, REPORT_COLUMNS), None
                  when there is no report
    known         (person_id, player_name): every spelling of every person
                  the report might name (seed names, the season's
                  partitions, the roster snapshots)
    roster_after  the roster was taken after the game's ET date (the
                  backlog: a first run ingests every game since the opener
                  with that day's roster)
    elsewhere     (person_id, team): every box listing dated between the
                  roster's date and the game's date, both included (any game
                  the state or the run holds; rebuilt rows are no evidence)

    Every player on a team's roster who is not in the box (either side) gets a
    row: status 'inactive', no seconds, no start, the cdn reason code of his
    report listing for that team (report_reason) or UNLISTED, source
    ROSTER_MINUS_BOX, jersey '' (the snapshot carries none). Unless a box in
    `elsewhere` lists him for another team and the team's report does not
    list him: then the roster is stale for him (he joined after the game, or
    left before it) and he gets no row. A box is the stronger evidence: no
    2025-26 player was listed by two teams on one date, and with a season-end
    roster this is what put a traded player down as absent for his future
    team on nights he was playing for his old one (p_play 0.96 -> 0.18). With a roster
    taken AFTER the game, a player the team's report lists Out who is neither
    in the box nor on its roster was on the team at tip and has left since:
    he gets a row too (resolved team-scoped first, then among every known
    person by an exact or alias match only), and a player the report places on
    one team is never rebuilt for the other. With a roster taken on or before
    the game's date the roster is the authority: a report name it does not
    hold (2025-26 boxes left out a few injured players their report listed,
    e.g. Finney-Smith and Tate for HOU on opening night) adds nothing. In a
    play-in or playoff game (TWO_WAY_INELIGIBLE) a two-way player on the
    team's roster cannot play and gets no row, whatever the report says of
    him; one the box lists keeps his box row like everyone else.

    Returns (rows, info): the game's rows, each team's box rows followed by its
    rebuilt rows (check_participation holds), and per team the counts
    rebuilt / listed / unlisted / left_since / elsewhere (roster players not
    rebuilt because a box put them on another team) / ineligible (roster
    two-way players a play-in or playoff game leaves out) plus the report rows
    that resolved to nobody ('unresolved', player_raw)."""
    ids = part["game_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"rebuild_inactives takes one game's rows, got {list(ids)[:3]}")
    gid = str(ids[0])
    if (part["status"] == "inactive").any():
        raise ValueError(f"{gid}: the box lists inactive players: it is the old format, "
                         f"never rebuilt (lists_inactives)")
    sides = part.drop_duplicates("team").set_index("team")["is_home"]
    if len(sides) != 2:
        raise ValueError(f"{gid}: expected two teams, got {sorted(sides.index)}")
    names = roster["name"] if "name" in roster.columns else roster["player_name"]
    rost = pd.DataFrame({"person_id": roster["person_id"].astype("int64").to_numpy(),
                         "team": roster["team"].astype(str).to_numpy(),
                         "player_name": names.astype(str).to_numpy()})
    two_ways: set = set()
    if season_type_of(gid) in TWO_WAY_INELIGIBLE:
        # A null would read as "not two-way" in silence, and that player
        # would come back as an inactive row the training listing lacks.
        if "two_way" not in roster.columns or roster["two_way"].isna().any():
            raise ValueError(f"{gid}: a {season_type_of(gid)} game needs every roster row's two_way "
                             f"flag (two-way players cannot play in it)")
        two_ways = set(rost.loc[roster["two_way"].astype(bool).to_numpy(), "person_id"])
    dup = rost.loc[rost["person_id"].duplicated(), "person_id"]
    if len(dup):
        raise ValueError(f"the roster lists {sorted(set(dup))[:5]} twice")
    absent = sorted(set(sides.index) - set(rost["team"]))
    if absent:
        raise ValueError(f"{gid}: the roster has nobody for {absent}: its inactive list cannot "
                         f"be rebuilt")
    season = part["season"].iloc[0]
    in_box = set(part["person_id"].astype("int64"))
    rep = _report_people(gid, listed, rost, part, known, roster_after)
    listing: dict[tuple[str, int], object] = {}
    for r in rep[rep["person_id"].notna()].itertuples():
        listing.setdefault((r.team, int(r.person_id)), r)
    name_of = dict(zip(rost["person_id"], rost["player_name"]))
    if known is not None and len(known):
        for pid, n in zip(known["person_id"], known["player_name"]):
            if isinstance(n, str) and n:
                name_of.setdefault(int(pid), n)

    claimed: dict[int, set] = {}
    for team, pid in listing:
        claimed.setdefault(pid, set()).add(team)
    out, info = [], {"unresolved": sorted(set(rep.loc[rep["person_id"].isna(), "player_raw"]))}
    for team in sorted(sides.index, key=lambda t: not sides[t]):       # home side first, like parse_cdn
        on_roster = set(rost.loc[rost["team"] == team, "person_id"])
        mine = on_roster - in_box
        ineligible = mine & two_ways                  # none in a regular-season game
        mine -= ineligible
        if elsewhere is not None and len(elsewhere):
            away = set(elsewhere.loc[elsewhere["team"] != team, "person_id"].astype("int64"))
            stale = {p for p in mine if p in away and (team, p) not in listing}
            mine -= stale
        else:
            stale = set()
        left = set()
        if roster_after:
            # The report (titled before tip) places him on the other team of
            # this game: he was theirs then, whatever the later roster says.
            mine = {p for p in mine if claimed.get(p, {team}) == {team}}
            left = {p for (t, p), r in listing.items()
                    if t == team and r.status == "Out" and claimed[p] == {team}
                    and p not in in_box and p not in on_roster}
        rows = []
        for pid in sorted(mine | left):
            hit = listing.get((team, pid))
            code, text = (report_reason(hit.status, hit.reason_cat, hit.reason_raw)
                          if hit is not None else (UNLISTED, ""))
            name = name_of.get(pid) or (_first_last(hit.player_raw) if hit is not None else "")
            rows.append((season, gid, team, bool(sides[team]), int(pid), _clean(name), "",
                         "inactive", False, 0, code, text,
                         absence_kind("inactive", code, text, "cdn"), ROSTER_MINUS_BOX))
        info[team] = {"rebuilt": len(rows), "listed": sum(r[10] != UNLISTED for r in rows),
                      "unlisted": sum(r[10] == UNLISTED for r in rows), "left_since": len(left),
                      "elsewhere": len(stale), "ineligible": len(ineligible)}
        out.append(part[part["team"] == team])
        if rows:
            out.append(_frame(rows))
    df = pd.concat(out, ignore_index=True)
    check_participation(df)
    return df, info


# ── Invariants ───────────────────────────────────────────────────────────────
def check_participation(df: pd.DataFrame, ot: int | None = None) -> None:
    """Raise ValueError (with the game id) on any broken invariant:
    two distinct teams, unique person_id per game, exactly 5 starters per
    team-game, starters played with seconds > 0, seconds only for players,
    reason_code '' for players, absence_kind '' exactly for players, known
    statuses/sources/kinds, and (when `ot` is given) team player-seconds equal
    to regulation + OT within TEAM_SECONDS_TOL.
    """
    if df.empty:
        raise ValueError("empty participation frame")
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns {list(df.columns)} != {COLUMNS}")
    for gid, g in df.groupby("game_id", sort=False):
        bad = set(g["status"]) - set(STATUSES)
        if bad:
            raise ValueError(f"{gid}: unknown status {sorted(bad)}")
        if set(g["source"]) - set(SOURCES):
            raise ValueError(f"{gid}: unknown source {sorted(set(g['source']))}")
        if set(g["absence_kind"]) - set(ABSENCE_KINDS):
            raise ValueError(f"{gid}: unknown absence_kind {sorted(set(g['absence_kind']))}")
        if ((g["status"] == "played") != (g["absence_kind"] == "")).any():
            raise ValueError(f"{gid}: absence_kind must be '' exactly for players who played")
        tms = sorted(g["team"].unique())
        if len(tms) != 2:
            raise ValueError(f"{gid}: expected 2 teams, got {tms}")
        if g.groupby("team")["is_home"].nunique().max() != 1 or g["is_home"].nunique() != 2:
            raise ValueError(f"{gid}: home/away flag inconsistent per team")
        dup = g["person_id"][g["person_id"].duplicated()]
        if len(dup):
            raise ValueError(f"{gid}: person_id listed twice: {sorted(set(dup))}")
        if (g["played_seconds"] < 0).any():
            raise ValueError(f"{gid}: negative seconds")
        np_ = g[g["status"] != "played"]
        if (np_["played_seconds"] != 0).any() or np_["starter"].any():
            raise ValueError(f"{gid}: a non-player has seconds or a starter flag")
        if ((g["source"] == ROSTER_MINUS_BOX) & (g["status"] != "inactive")).any():
            raise ValueError(f"{gid}: a rebuilt row ({ROSTER_MINUS_BOX}) that is not inactive")
        if (g.loc[g["status"] == "played", "reason_code"] != "").any():
            raise ValueError(f"{gid}: a player who played carries a reason code")
        for team, t in g.groupby("team"):
            st = t[t["starter"]]
            if len(st) != 5:
                raise ValueError(f"{gid}: {team} has {len(st)} starters, expected 5")
            if (st["status"] != "played").any() or (st["played_seconds"] <= 0).any():
                raise ValueError(f"{gid}: {team} starter with no playing time: "
                                 f"{st.loc[st['played_seconds'] <= 0, 'person_id'].tolist()}")
            if ot is not None:
                diff = int(t["played_seconds"].sum()) - _expected_team_seconds(ot)
                if abs(diff) > TEAM_SECONDS_TOL:
                    raise ValueError(f"{gid}: {team} player-seconds off by {diff}s "
                                     f"(OT={ot}): a player is missing or double-counted")


def team_seconds_diff(df: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Per (game_id, team): sum(played_seconds) - expected for the OT count."""
    tot = df.groupby(["game_id", "team"])["played_seconds"].sum().rename("secs").reset_index()
    tot = tot.merge(meta[["game_id", "ot"]], on="game_id", how="left")
    exp = REG_TEAM_SECONDS + OT_TEAM_SECONDS * tot["ot"]
    return (tot["secs"] - exp).set_axis(pd.MultiIndex.from_frame(tot[["game_id", "team"]]))


def check_table(part: pd.DataFrame, games: pd.DataFrame, season: str | None = None) -> None:
    """Season-table invariants, vectorised: what check_participation asserts
    per game, plus what only the whole table can show. (game_id, person_id) is
    unique, each game appears once in `games`, both tables cover the same games,
    every row's team / is_home / source / season agree with its game row.

    Run on every checkpoint: the per-game parse checks cannot see a game that
    was parsed twice (`--games X,X` wrote 35 duplicate rows that way)."""
    if list(part.columns) != COLUMNS:
        raise ValueError(f"participation columns {list(part.columns)} != {COLUMNS} "
                         f"(a table from an older schema: re-parse it from raw)")
    if list(games.columns) != META_COLUMNS:
        raise ValueError(f"games columns {list(games.columns)} != {META_COLUMNS}")
    if part.empty or games.empty:
        raise ValueError("empty participation or games table")

    def ids(mask) -> list:
        return sorted(set(part.loc[mask, "game_id"]))[:5]

    dup = games["game_id"][games["game_id"].duplicated()]
    if len(dup):
        raise ValueError(f"games table lists {sorted(set(dup))[:5]} more than once")
    key = part.duplicated(["game_id", "person_id"], keep=False)
    if key.any():
        raise ValueError(f"(game_id, person_id) is not unique: {key.sum()} rows in {ids(key)}")
    a, b = set(part["game_id"]), set(games["game_id"])
    if a != b:
        raise ValueError(f"participation and games tables cover different games: only "
                         f"participation {sorted(a - b)[:5]}, only games {sorted(b - a)[:5]}")
    if season is not None and ((part["season"] != season).any() or (games["season"] != season).any()):
        raise ValueError(f"rows from another season in the {season} table")
    g = games.set_index("game_id")
    # A rebuilt inactive row (ROSTER_MINUS_BOX) belongs to a cdn game.
    row_source = part["source"].where(part["source"] != ROSTER_MINUS_BOX, "cdn")
    checks = [
        (~part["status"].isin(STATUSES), "unknown status"),
        (~part["source"].isin(SOURCES), "unknown source"),
        (~part["absence_kind"].isin(ABSENCE_KINDS), "unknown absence_kind"),
        (part["season"] != part["game_id"].map(g["season"]), "row season differs from its game"),
        (row_source != part["game_id"].map(g["source"]), "row source differs from its game"),
        (part["played_seconds"] < 0, "negative seconds"),
    ]
    played = part["status"] == "played"
    home, away = part["game_id"].map(g["home"]), part["game_id"].map(g["away"])
    checks += [
        (~((part["is_home"] & (part["team"] == home)) | (~part["is_home"] & (part["team"] == away))),
         "team / is_home disagree with the game's home and away"),
        (~played & ((part["played_seconds"] != 0) | part["starter"]),
         "a non-player has seconds or a starter flag"),
        (played & (part["reason_code"] != ""), "a player who played carries a reason code"),
        (played != (part["absence_kind"] == ""), "absence_kind must be '' exactly for players"),
        ((part["source"] == ROSTER_MINUS_BOX) & (part["status"] != "inactive"),
         f"a rebuilt row ({ROSTER_MINUS_BOX}) that is not inactive"),
        (part["starter"] & (part["played_seconds"] <= 0), "a starter with no playing time"),
    ]
    for mask, what in checks:
        if mask.any():
            raise ValueError(f"{what}: {int(mask.sum())} rows in {ids(mask)}")
    n_teams = part.groupby("game_id")["team"].nunique()
    if (n_teams != 2).any():
        raise ValueError(f"games without exactly two teams: {list(n_teams[n_teams != 2].index[:5])}")
    team_games = pd.MultiIndex.from_frame(part[["game_id", "team"]].drop_duplicates())
    starters = part[part["starter"]].groupby(["game_id", "team"]).size().reindex(team_games, fill_value=0)
    if (starters != 5).any():
        raise ValueError(f"team-games without exactly 5 starters: "
                         f"{list(starters[starters != 5].items())[:5]}")
    diff = team_seconds_diff(part, games)
    off = diff[diff.abs() > TEAM_SECONDS_TOL]
    if len(off):
        raise ValueError(f"team player-seconds off by more than {TEAM_SECONDS_TOL}s: "
                         f"{list(off.items())[:5]}")


# ── Public entry points ──────────────────────────────────────────────────────
def load_game(game_id: str, source: str = "auto", refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """(participation rows, game meta) for one FINAL game.

    source='auto': cdn from 2019-20, V3 before. A cdn Absent for a 2019-20+ id
    falls back to V3 (the returned rows say source='v3'): the CDN does have
    holes inside its era (the 2019-20 bubble play-in game 0051900111 is
    S3-absent). Blocked always propagates: a blocked host is never "no game".
    """
    if source not in ("auto", "cdn", "v3"):
        raise ValueError(f"source must be auto|cdn|v3, got {source!r}")
    use = default_source(game_id) if source == "auto" else source
    if use == "cdn" and source == "auto" and not refresh and raw_path(game_id, "cdnabsent").exists():
        use = "v3"  # known CDN gap (e.g. the 2019-20 bubble play-in 0051900111)
    if use == "cdn":
        try:
            return parse_cdn(fetch_cdn(game_id, refresh=refresh), game_id)
        except http.Absent:
            if source == "cdn":
                raise
        except NotFinal:
            # The CDN ARCHIVE holds some long-finished games frozen mid-game
            # (17 in 2019-20, e.g. 0021900877 'inprogress', 0021901238
            # gameStatus 2): a partial box score. For a PAST season the game is
            # over, so the V3 box score is the final one. For the live season
            # NotFinal means what it says and propagates.
            if source == "cdn" or _is_live_game(game_id):
                raise
        except ValueError:
            # A cdn box that breaks an invariant (0022100124: BKN player-seconds
            # 301s short; 0022400061: BOS 260s over) while the stats logs for the
            # same game reconcile exactly. Past season: the V3 box is the
            # authority, and it still has to pass the same invariants. Live
            # season: raise, so the bot goes red instead of guessing.
            if source == "cdn" or _is_live_game(game_id):
                raise
    box, summary = fetch_v3(game_id, refresh=refresh)
    return parse_v3(box, summary, game_id)


def _is_live_game(game_id: str) -> bool:
    """True when the game's season is the current one (ET date, Aug 1 rule)."""
    from ..util import season_for_date, season_end_year, today_et
    return season_end_year(game_season(game_id)) >= season_end_year(season_for_date(today_et()))


def participation(game_id: str, source: str = "auto") -> pd.DataFrame:
    """One row per (game_id, person_id); see COLUMNS and the module docstring."""
    return load_game(game_id, source)[0]


def game_meta(game_id: str, source: str = "auto") -> dict:
    """{game_id, season, season_type, tip_utc (UTC Timestamp), et_date
    ('YYYY-MM-DD', the league's game date), arena, arena_city, home, away,
    periods (periods played, 4 + OT), ot (overtime count), source}."""
    return load_game(game_id, source)[1]


# ── Season tables ────────────────────────────────────────────────────────────
TABLE_VERSION = "v1"


def participation_path(season: str) -> Path:
    """cache/participation_{season}_v1.parquet: one row per (game_id, person_id)."""
    return cache.dc_path(f"participation_{season}_{TABLE_VERSION}.parquet")


def games_path(season: str) -> Path:
    """cache/participation_games_{season}_v1.parquet: one game_meta row per game."""
    return cache.dc_path(f"participation_games_{season}_{TABLE_VERSION}.parquet")


def read_participation(season: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """(participation, games, info) as built by scripts/build_participation.py.

    `info` carries games_total / games_done / complete plus list_source,
    list_final and list_fetched_at (when the game list was fetched). complete
    is True only for a season that is OVER with every listed game parsed: a
    --sample build and a live season are complete=False and must not be read
    as a full season. A table written under an older schema raises: rebuild it
    with the script (it re-parses from raw payloads, no live calls)."""
    part, games = cache.read_parquet(participation_path(season)), cache.read_parquet(games_path(season))
    if part is not None and list(part.columns) != COLUMNS:
        raise ValueError(f"{participation_path(season).name} has columns {list(part.columns)}, "
                         f"not {COLUMNS}: rebuild with scripts/build_participation.py")
    return part, games, cache.read_parquet_meta(games_path(season))


def check_meta_against_lgl(meta: dict, lgl_row) -> None:
    """A parsed game must match its game-list entry (LeagueGameLog T, or the
    schedule for a live season and the Cup Final): same two teams, same game
    date, and the same home team whenever the list is not neutral."""
    gid = meta["game_id"]
    box_teams = sorted((meta["home"], meta["away"]))
    if box_teams != [lgl_row["team_a"], lgl_row["team_b"]] or meta["et_date"] != lgl_row["game_date"]:
        raise ValueError(f"{gid}: box says {meta['away']}@{meta['home']} on {meta['et_date']}, "
                         f"game list says {lgl_row['team_a']}/{lgl_row['team_b']} on "
                         f"{lgl_row['game_date']}")
    if lgl_row["home"] and lgl_row["home"] != meta["home"]:
        raise ValueError(f"{gid}: box home {meta['home']}, game list home {lgl_row['home']}")


# ── Season state: live vs over, provisional vs settled payloads ──────────────
PLAYIN_FIRST_SEASON = "2019-20"      # one bubble play-in game; six a year from 2020-21
CUP_FINAL_FIRST_SEASON = "2023-24"   # the Cup Final got its own 006 id that season


def is_live_season(season: str) -> bool:
    """The current season (by the ET date and util's August 1 cutover) and any
    later one: their game lists still grow."""
    return season_end_year(season) >= season_end_year(season_for_date(today_et()))


def settled(fetched_at, season: str) -> bool:
    """True when a payload about `season` was fetched after the season stopped
    being live, so it can never change again. A payload fetched mid-season
    (say mid-playoffs, the last run before August 1) is provisional: reusing it
    would mark a season complete without its last playoff games."""
    if fetched_at is None:
        return False
    d = et_date_of(dt.datetime.fromtimestamp(float(fetched_at), UTC))
    return season_end_year(season_for_date(d)) > season_end_year(season)


def lgl_empty_ok(season: str, season_type: str) -> bool:
    """Can a LeagueGameLog game list legitimately have zero rows? Only for a
    live season, or the play-in before it existed. A past Regular Season or
    Playoffs with zero rows is a silent empty (HTTP 200, headers, no rows),
    never "no games"."""
    if is_live_season(season):
        return True
    return season_type == "PlayIn" and season_end_year(season) < season_end_year(PLAYIN_FIRST_SEASON)


# ── Season enumeration ───────────────────────────────────────────────────────
LGL_TYPES = ("Regular Season", "PlayIn", "Playoffs")
_TYPE_DIGIT = {"Regular Season": "2", "PlayIn": "5", "Playoffs": "4"}
_LGL_SLUG = {"Regular Season": "rs", "PlayIn": "playin", "Playoffs": "po"}
# Schedule game_type (bucketlib.sources.statsnba) -> season_type in our tables.
SCHEDULE_TYPES = {"RS": "Regular Season", "PI": "PlayIn", "PO": "Playoffs",
                  "CUP_FINAL": "NBA Cup Final"}


LGL_GAME_COLUMNS = ["game_id", "season", "season_type", "game_date", "team_a", "team_b",
                    "home", "away"]


def parse_lgl_games(payload: dict, season: str, season_type: str) -> pd.DataFrame:
    """LeagueGameLog(T) -> one row per game: game_id, season, season_type,
    game_date ('YYYY-MM-DD'), team_a/team_b (the two codes, sorted), home, away.

    Each game must have exactly two team rows with canonical codes, one date,
    and a game id whose type digit matches the requested season type. home and
    away come from MATCHUP ('vs.' = home, '@' = away) and are '' when the log
    marks the game neutral: in 2025-26 BOTH rows read '@' for the Mexico City,
    Berlin and London games and the Cup semifinals in Las Vegas ('DAL @ DET' and
    'DET @ DAL' for 0022500147), while 2023-24 labelled its Vegas semis
    home/away. The box score's homeTeam is the authority for is_home."""
    df = http.result_sets(payload)["LeagueGameLog"]
    out = []
    for gid, g in df.groupby("GAME_ID", sort=True):
        gid = str(gid)
        if game_season(gid) != season or _gid_parts(gid)[0] != _TYPE_DIGIT[season_type]:
            raise ValueError(f"{gid}: in {season} {season_type} LeagueGameLog but id says "
                             f"{game_season(gid)} {season_type_of(gid)}")
        if len(g) != 2:
            raise ValueError(f"{gid}: {len(g)} team rows in LeagueGameLog T")
        if g["GAME_DATE"].nunique() != 1:
            raise ValueError(f"{gid}: team rows disagree on the date")
        codes, home, away = [], [], []
        for _, r in g.iterrows():
            code = teams.canonical(r["TEAM_ABBREVIATION"])
            if code is None or teams.from_team_id(r["TEAM_ID"]) != code:
                raise ValueError(f"{gid}: unknown team {r['TEAM_ABBREVIATION']!r}/{r['TEAM_ID']}")
            codes.append(code)
            if " vs. " in r["MATCHUP"]:
                home.append(code)
            elif " @ " in r["MATCHUP"]:
                away.append(code)
            else:
                raise ValueError(f"{gid}: unparseable MATCHUP {r['MATCHUP']!r}")
        if codes[0] == codes[1]:
            raise ValueError(f"{gid}: both team rows are {codes[0]}")
        h, a = (home[0], away[0]) if len(home) == 1 and len(away) == 1 else ("", "")
        ta, tb = sorted(codes)
        out.append((gid, season, season_type, str(g["GAME_DATE"].iloc[0])[:10], ta, tb, h, a))
    return pd.DataFrame(out, columns=LGL_GAME_COLUMNS)


def lgl_raw_path(season: str, season_type: str, kind: str = "T", date: str | None = None) -> Path:
    if season_type not in _LGL_SLUG:
        raise ValueError(f"no LeagueGameLog for season type {season_type!r} "
                         f"(the NBA Cup Final is in none)")
    name = f"lgl{kind}_{_LGL_SLUG[season_type]}" + (f"_{date}" if date else "")
    return cache.dc_path(f"{RAW_SUBDIR}/{season}/{name}.json.gz")


def fetch_lgl(season: str, season_type: str, kind: str = "T", date: str | None = None,
              refresh: bool = False) -> dict:
    """Raw LeagueGameLog, cached. kind 'T' (team rows, the game list) or 'P'
    (player appearances); `date` ('YYYY-MM-DD') filters to one game date.

    A cached copy is reused only when it is settled (fetched after its season
    stopped being live); a provisional one is refetched, so a per-date P log
    fetched before a late game went final is not reused the next day. An empty
    rowSet is returned only where lgl_empty_ok allows it; anywhere else it
    raises BadPayload and is NOT cached, so a rerun asks again."""
    from nba_api.stats.endpoints import leaguegamelog

    p = lgl_raw_path(season, season_type, kind, date)
    if p.exists() and not refresh:
        data, fetched_at = cache.read_json_stamped(p)
        if settled(fetched_at, season):
            return data
    dfmt = "" if date is None else f"{date[5:7]}/{date[8:10]}/{date[:4]}"
    payload = _timed("lgl", http.stats_call, leaguegamelog.LeagueGameLog, timeout=45, retries=3,
                     season=season, season_type_all_star=season_type,
                     player_or_team_abbreviation=kind, date_from_nullable=dfmt,
                     date_to_nullable=dfmt)
    what = f"LeagueGameLog {kind} {season} {season_type}" + (f" {date}" if date else "")
    try:
        df = http.result_sets(payload)["LeagueGameLog"]
    except (KeyError, AttributeError, TypeError) as e:
        raise http.BadPayload(http.STATS_BASE + "leaguegamelog", f"{what}: {e!r}") from None
    if df.empty and not lgl_empty_ok(season, season_type):
        raise http.BadPayload(http.STATS_BASE + "leaguegamelog",
                              f"{what}: 0 rows for a season that is over (a silent empty, "
                              f"not 'no games'); not cached")
    cache.write_json(p, payload, stamped=True, gz=True)
    return payload


def fetch_cdn_schedule() -> dict:
    """The cdn league schedule (current season only), refetched every call:
    it is the in-season game list, and the one reachable from GitHub Actions.
    The raw payload is kept under cache/raw/box/{seasonYear}/cdnsched.json.gz
    (rewritten only when it changes)."""
    payload = _timed("sched", http.fetch, CDN_SCHEDULE_URL, timeout=30, retries=2)
    ls = payload.get("leagueSchedule") if isinstance(payload, dict) else None
    if not isinstance(ls, dict) or "gameDates" not in ls:
        raise http.BadPayload(CDN_SCHEDULE_URL, "no leagueSchedule.gameDates")
    year = str(ls.get("seasonYear"))
    try:
        season_end_year(year)
    except ValueError:
        raise http.BadPayload(CDN_SCHEDULE_URL, f"seasonYear {year!r} is not a season") from None
    cache.write_json(cache.dc_path(f"{RAW_SUBDIR}/{year}/cdnsched.json.gz"), payload,
                     stamped=True, gz=True, if_changed=True)
    return payload


def league_schedule(season: str, refresh: bool = False) -> pd.DataFrame:
    """ScheduleLeagueV2 for a season via bucketlib.sources.statsnba (its raw
    cache is shared with build_schedule). A copy fetched while the season was
    live is provisional and refetched, like the LeagueGameLog lists."""
    p = _stats.raw_path(season, "schedule")
    force = refresh
    if p.exists() and not force:
        _, fetched_at = cache.read_json_stamped(p)
        force = not settled(fetched_at, season)
    if force or not p.exists():
        return _timed("sched", _stats.schedule, season, refresh=True)
    return _stats.schedule(season, refresh=False)


def schedule_final_games(sched: pd.DataFrame, season: str) -> pd.DataFrame:
    """Final RS / PlayIn / Playoffs / Cup Final games of a parsed schedule
    (statsnba.parse_schedule shape) as game-list rows (LGL_GAME_COLUMNS).
    Neutral sites get home/away '' like LeagueGameLog's neutral rows, so a
    parsed box is held to teams + date there."""
    s = sched[sched["game_type"].isin(list(SCHEDULE_TYPES)) & (sched["game_status"] == 3)]
    tbd = s[s["is_tbd"]]
    if len(tbd):
        raise ValueError(f"schedule {season}: final games without both teams {tbd['game_id'].tolist()[:5]}")
    out = []
    for r in s.itertuples():
        stype = SCHEDULE_TYPES[r.game_type]
        if season_type_of(r.game_id) != stype or game_season(r.game_id) != season:
            raise ValueError(f"schedule {season}: {r.game_id} listed as {season} {stype}")
        ta, tb = sorted((r.home, r.away))
        h, a = ("", "") if r.is_neutral else (r.home, r.away)
        out.append((r.game_id, season, stype, pd.Timestamp(r.game_date).date().isoformat(),
                    ta, tb, h, a))
    df = pd.DataFrame(out, columns=LGL_GAME_COLUMNS)
    if df["game_id"].duplicated().any():
        raise ValueError(f"schedule {season}: duplicate game ids")
    return df


def check_list_against_schedule(lgl: pd.DataFrame, sched_games: pd.DataFrame, season: str) -> None:
    """LeagueGameLog T and the schedule must list the same final RS / PlayIn /
    Playoffs games with the same dates and teams (and home team, where both
    name one). This is what catches a silent-empty, truncated or stale
    LeagueGameLog list, which nothing downstream of the list can see."""
    a = lgl.set_index("game_id")
    b = sched_games[sched_games["season_type"].isin(LGL_TYPES)].set_index("game_id")
    only_l, only_s = sorted(set(a.index) - set(b.index)), sorted(set(b.index) - set(a.index))
    if only_l or only_s:
        raise ValueError(f"{season}: LeagueGameLog T and the schedule list different final games: "
                         f"{len(only_l)} only in LeagueGameLog {only_l[:8]}, "
                         f"{len(only_s)} only in the schedule {only_s[:8]}")
    b = b.loc[a.index]
    bad = a.index[(a["game_date"] != b["game_date"]) | (a["team_a"] != b["team_a"])
                  | (a["team_b"] != b["team_b"]) | (a["season_type"] != b["season_type"])
                  | ((a["home"] != "") & (b["home"] != "") & (a["home"] != b["home"]))]
    if len(bad):
        g = bad[0]
        raise ValueError(f"{season}: {len(bad)} games differ between LeagueGameLog T and the "
                         f"schedule, e.g. {g}: {a.loc[g].to_dict()} vs {b.loc[g].to_dict()}")


def _empty_list() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=str) for c in LGL_GAME_COLUMNS})


def season_game_list(season: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """(games, list_info) for a season: every FINAL game, one row each
    (LGL_GAME_COLUMNS, sorted by date), and where the list came from.

    list_info: list_source (str), list_final (True when the season is over and
    the list can no longer grow), list_fetched_at (epoch seconds).
    Live season: the cdn schedule, no stats.nba.com call. If the cdn schedule
    still shows the previous season, the live one has no final game yet and
    the list is legitimately empty. Past season: LeagueGameLog T checked
    against ScheduleLeagueV2, plus the Cup Final from the schedule."""
    if is_live_season(season):
        payload = fetch_cdn_schedule()
        year = str(payload["leagueSchedule"]["seasonYear"])
        info = {"list_source": "cdn schedule", "list_final": False, "list_fetched_at": cache.stamp_now()}
        if season_end_year(year) < season_end_year(season):
            info["list_source"] = f"cdn schedule (still {year}: no {season} game is final yet)"
            return _empty_list(), info
        if year == season:
            games = schedule_final_games(_stats.parse_schedule(payload, season), season)
            return games.sort_values(["game_date", "game_id"]).reset_index(drop=True), info
        # The cdn schedule has already moved past this season (it can, in the
        # weeks before util's August 1 cutover): only LeagueGameLog lists it now.
    frames, stamps = [], []
    for st in LGL_TYPES:
        frames.append(parse_lgl_games(fetch_lgl(season, st, "T", refresh=refresh), season, st))
        stamps.append(cache.read_json_stamped(lgl_raw_path(season, st))[1])
    lgl = pd.concat(frames, ignore_index=True)
    if lgl["game_id"].duplicated().any():
        raise ValueError(f"{season}: a game id appears under two season types")
    sched = schedule_final_games(league_schedule(season, refresh=refresh), season)
    check_list_against_schedule(lgl, sched, season)
    cup = sched[sched["season_type"] == "NBA Cup Final"]
    live = is_live_season(season)
    if season_end_year(season) < season_end_year(CUP_FINAL_FIRST_SEASON):
        if len(cup):
            raise ValueError(f"{season}: a Cup Final id before {CUP_FINAL_FIRST_SEASON}: {cup['game_id'].tolist()}")
    elif not live and len(cup) != 1:
        raise ValueError(f"{season}: {len(cup)} final NBA Cup Final games in the schedule, expected 1")
    games = pd.concat([lgl, cup], ignore_index=True).sort_values(["game_date", "game_id"])
    info = {"list_source": "LeagueGameLog T + schedule", "list_final": not live,
            "list_fetched_at": min((s for s in stamps if s is not None), default=None)}
    return games.reset_index(drop=True), info


# ── Cross-checks ─────────────────────────────────────────────────────────────
def lgl_appearances(payload: dict) -> pd.DataFrame:
    """LeagueGameLog(P) -> game_id, person_id, team, lgl_min (rounded MIN)."""
    df = http.result_sets(payload)["LeagueGameLog"]
    out = pd.DataFrame({"game_id": df["GAME_ID"].astype(str),
                        "person_id": df["PLAYER_ID"].astype("int64"),
                        "team": df["TEAM_ABBREVIATION"].map(teams.canonical),
                        "lgl_min": df["MIN"]})
    if out["team"].isna().any():
        raise ValueError(f"unknown LGL team codes: {sorted(set(df.loc[out['team'].isna(), 'TEAM_ABBREVIATION']))}")
    return out


def crosscheck_appearances(part: pd.DataFrame, lgl: pd.DataFrame) -> pd.DataFrame:
    """Per game in `part`: players who played per the box vs LeagueGameLog P
    appearances (0-minute cameo rows included on both sides). Returns one row
    per game: n_box, n_lgl, only_box, only_lgl, team_mismatch (lists of
    person_ids). Agreement = all three lists empty."""
    rows = []
    for gid, g in part.groupby("game_id", sort=True):
        box = g[g["status"] == "played"].set_index("person_id")["team"]
        lg = lgl[lgl["game_id"] == gid].set_index("person_id")["team"]
        both = box.index.intersection(lg.index)
        rows.append({"game_id": gid, "n_box": len(box), "n_lgl": len(lg),
                     "only_box": sorted(set(box.index) - set(lg.index)),
                     "only_lgl": sorted(set(lg.index) - set(box.index)),
                     "team_mismatch": sorted(int(p) for p in both if box[p] != lg[p])})
    return pd.DataFrame(rows)


def crosscheck_starters(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Compare starter sets per (game_id, team) between two parses of the same
    games (e.g. cdn flags vs the V3 first-five rule). One row per team-game
    with only_a / only_b lists; agreement = both empty."""
    rows = []
    for (gid, team), g in a[a["starter"]].groupby(["game_id", "team"]):
        h = b[(b["game_id"] == gid) & (b["team"] == team) & b["starter"]]
        sa, sb = set(g["person_id"]), set(h["person_id"])
        rows.append({"game_id": gid, "team": team, "only_a": sorted(sa - sb),
                     "only_b": sorted(sb - sa)})
    return pd.DataFrame(rows)
