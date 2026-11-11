"""The site's live layer: every network read the APP makes goes through here.

Rules (design review, 2026-10-08):
  - The app never calls stats.nba.com (it hangs from cloud hosts).
  - Fetches get a short budget and no retries; a hang costs seconds, not the
    ~96s of the default client's retry loop.
  - A FAILURE is cached too, with its own short TTL (a circuit breaker):
    st.cache_data does not cache exceptions, so without this every rerun
    (every click, every paste) would re-wait on the same dead host.
  - Stale beats empty: when a fetch fails, the last good value is served with
    `stale=True` and its age, and the page says so. With no good value at all,
    the committed snapshot the bot wrote is used; with neither, it raises and
    the page renders an outage notice instead of an empty slate. When both
    exist, the NEWER of the two is served (a long-lived process must not keep
    a morning copy while the bot commits fresher ones).
  - A failure is ANY exception from the fetch or its parser: these are
    third-party payloads, and a null where a field is always set
    (AttributeError, TypeError) is an outage of that feed, not a reason to
    take the page down. An EMPTY payload is a failure too (an error body, a
    schedule with no games, a roster index with a handful of players): it
    never replaces a good copy (spec rule 9).
  - Nothing here writes to disk, so READ_ONLY deployments stay clean: the
    checkout is shared by every visitor and owned by the bot. Besides the
    bot's committed copies (read, never written), it reads one local file:
    the odds feed's key file, and only when BUCKETVALUE_ODDS_API_KEY_FILE
    names one (a local run or the probe; never on Cloud, where the key is a
    secret in the environment). It refuses anything but a regular file
    without blocking, so a FIFO or /dev/stdin there never hangs a page.

The licensed odds feed (The Odds API, design 2.4) is the one feed that does
not go through `_cached`: its memory is the FEED ANCHOR, a module object in
sys.modules under a name outside the bucketlib package, which a code push
(freshen's purge) and a Streamlit cache clear both miss and a restart loses.
One walk at a time buys today's games on a thread pool, each game's call
guarded by the credit governor (bucketlib.oddsbudget); pages read memory and
never wait more than PAGE_WAIT_S. A background refresher, at most one daemon
thread per process, runs the same walk without a page view when a game is
due (design 2.4.6). With no key nothing of it runs.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import functools
import logging
import os
import stat
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import cache as C
from . import context as CTX
from . import horizons as HZ
from . import http as H
from . import oddsbudget as OB
from . import state as S
from .sources import box as BX
from .sources import cdn, cdnbox, espn
from .sources import injury as INJ
from .sources import oddsapi as O
from .util import ET, UTC, now_et, now_utc, to_et

# Per fetch, no retries (spec 4: a ~5 s budget). A dead host costs this once
# per failure TTL, not once per rerun.
TIMEOUT = 4.0


@dataclass
class Live:
    value: Any
    fetched_at: float | None          # epoch seconds of the data actually shown
    stale: bool = False               # True when the live fetch failed
    source: str = "live"              # "live" | "memory" | "committed"
    error: str = ""                   # why it is stale, for the notice
    extra: dict = field(default_factory=dict)

    @property
    def age_min(self) -> float | None:
        if self.fetched_at is None:
            return None
        return (time.time() - self.fetched_at) / 60.0


_lock = threading.Lock()
_good: dict[str, tuple[float, Any]] = {}
_fail: dict[str, tuple[float, str]] = {}


def _newer_than_memory(cval, cts, good: tuple[float, Any], newer: Callable[[Any], Any] | None) -> bool:
    """Is the committed value newer than the one in memory? By `newer` (the
    data's own as-of, e.g. a report's title time) when given, else by the two
    fetch stamps. Undecidable counts as not newer: memory is kept."""
    try:
        if newer is not None:
            return bool(newer(cval) > newer(good[1]))
        return cts is not None and float(cts) > float(good[0])
    except Exception:  # noqa: BLE001 - an unreadable stamp never displaces a good copy
        return False


def _cached(key: str, ttl: float, fail_ttl: float, fn: Callable[[], Any],
            committed: Callable[[], tuple[Any, float | None]] | None = None,
            newer: Callable[[Any], Any] | None = None) -> Live:
    now = time.time()
    with _lock:
        good = _good.get(key)
        fail = _fail.get(key)
    if good and now - good[0] < ttl:
        return Live(good[1], good[0], source="memory")
    if fail and now - fail[0] < fail_ttl:
        err = fail[1]
    else:
        try:
            val = fn()
            with _lock:
                _good[key] = (now, val)
                _fail.pop(key, None)
            return Live(val, now, source="live")
        except Exception as e:  # noqa: BLE001 - third-party payloads: any failure is a feed failure
            C.logger.warning("live %s failed", key, exc_info=True)
            err = f"{type(e).__name__}: {str(e)[:160]}"
            with _lock:
                _fail[key] = (now, err)
    if committed is not None:
        try:
            cval, cts = committed()
        except RuntimeError:
            if not good:
                raise
            cval, cts = None, None
        except Exception as e:  # noqa: BLE001 - an unreadable committed copy is no copy
            if not good:
                raise RuntimeError(f"{key}: live fetch failed ({err}) and the committed copy is "
                                   f"unreadable ({type(e).__name__}: {str(e)[:160]})") from e
            cval, cts = None, None
        if cval is not None and (not good or _newer_than_memory(cval, cts, good, newer)):
            # Not written into _good: the next successful fetch replaces it.
            return Live(cval, cts, stale=True, source="committed", error=err)
    if good:
        return Live(good[1], good[0], stale=True, source="memory", error=err)
    raise RuntimeError(f"{key}: live fetch failed and nothing is cached ({err})")


def _committed_parquet(name: str):
    """The bot's committed copy, state/current/{name}_v1.parquet. The path is
    resolved through bucketlib.state at call time (state owns every path
    under state/, and a test that points STATE_DIR elsewhere moves it too).
    An empty file is no copy (the bot never writes one)."""
    def load():
        p = S.current_path(name)
        df = C.read_parquet(p)
        if df is None or df.empty:
            return None, None
        return df, C.read_parquet_meta(p).get("fetched_at")
    return load


# -- Schedule and rosters (cdn.nba.com) ---------------------------------------
def _schedule_fetch() -> pd.DataFrame:
    df = cdn.parse_schedule(H.fetch(cdn.SCHEDULE_URL, timeout=TIMEOUT, retries=0))
    if df.empty:
        # The bot refuses the same payload (step_schedule: "schedule parsed
        # empty"). cdn.parse_schedule's empty return stays: the bot reads it.
        raise ValueError("schedule parsed empty")
    return df


def schedule() -> Live:
    """The current season's schedule. TTL 10 minutes: game status and scores
    change through the night, and the post-Cup games get added mid-season."""
    return _cached("schedule", 600, 90, _schedule_fetch, _committed_parquet("schedule"))


def _roster_fetch() -> pd.DataFrame:
    df = cdn.parse_player_index(H.fetch(cdn.PLAYER_INDEX_URL, timeout=TIMEOUT, retries=0))
    # cdn.MIN_ROSTER, read at call time: the one threshold the bot refuses under too.
    if len(df) < cdn.MIN_ROSTER:
        raise ValueError(f"player index parsed {len(df)} rostered players (< {cdn.MIN_ROSTER})")
    return df


def roster() -> Live:
    """Current rostered players with the two-way flag. TTL 1 hour."""
    return _cached("roster", 3600, 120, _roster_fetch, _committed_parquet("roster"))


def games_on(sched: pd.DataFrame, date) -> pd.DataFrame:
    """Games whose ET date is `date`, NBA games only, in tip order."""
    if sched is None or sched.empty:
        return sched
    g = sched[sched["game_date"] == date]
    return g.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)


def default_view_date():
    """The slate a visitor most likely wants: the ET date of (now - 4h), so a
    10:30pm ET West Coast tip stays on screen after midnight."""
    return (now_et() - pd.Timedelta(hours=4)).date()


def _flag(v) -> bool:
    """A schedule boolean that may be missing (None, NaN) in an older frame."""
    try:
        return bool(v) and not pd.isna(v)
    except (TypeError, ValueError):
        return False


def is_postponed(row) -> bool:
    """The schedule's own postponed flag (cdn.parse_schedule decides it). A
    postponed game stays on its original date with status 'scheduled', so the
    clock alone would call it live once its old tip passed."""
    return _flag(row.get("postponed"))


def time_tbd(row) -> bool:
    """A placeholder whose tip TIME is not set (Cup knockout games before the
    group stage settles): the schedule's tip is a made-up 00:00 ET that must
    never be printed or read as a tip. cdn.parse_schedule's `time_tbd` column
    decides; a frame parsed before that column existed falls back to the
    feed's own 'TBD' status text."""
    if "time_tbd" in row:
        return _flag(row.get("time_tbd"))
    return str(row.get("status_text") or "").strip().upper() == "TBD"


def game_state(row, now=None) -> str:
    """'final' | 'live' | 'upcoming' | 'postponed' from the schedule row and
    the clock. A postponed game, or one whose time is not set, is never
    called live by the clock."""
    st = row.get("status")
    if st == "final":
        return "final"
    if st == "live":
        return "live"
    if is_postponed(row):
        return "postponed"
    if time_tbd(row):
        return "upcoming"
    t = now or now_et()
    return "live" if to_et(row["tip_utc"]) <= t else "upcoming"


# -- Lines (ESPN scoreboard, DraftKings) --------------------------------------
def odds_on(date) -> Live:
    """{game key (away, home): odds dict} for one ET date. TTL 5 minutes.

    Lines post gradually (a missing line is "not posted yet", not "no game").
    ESPN drops a game's odds at tip, so a game this process saw priced keeps
    that pre-tip line once it has started (its entry carries "pre_tip":
    True); a process that never saw it gets it from the served frame, or
    the bot's committed snapshots (store.posted_lines). A payload with no NBA game at all (an error body,
    an empty scoreboard) is a failure, never "no lines": the pages only ask
    for a date that has games.
    """
    key = f"odds:{date.isoformat()}"

    def fetch():
        # The one scoreboard request (espn.fetch_scoreboard: its date check,
        # its limit, its events-list check), on the app's budget.
        payload = espn.fetch_scoreboard(date.strftime("%Y%m%d"), timeout=TIMEOUT, retries=0)
        now = now_utc()
        df = espn.parse_scoreboard(payload, now)
        if df.empty:
            raise ValueError(f"ESPN scoreboard for {date.isoformat()} parsed no NBA game")
        out = {}
        for _, r in df.iterrows():
            if pd.notna(r.get("home_line")) or pd.notna(r.get("total")):
                out[(r["away"], r["home"])] = {
                    "home_line": r.get("home_line"), "total": r.get("total"),
                    "open_home_line": r.get("open_home_line"), "open_total": r.get("open_total"),
                    "provider": r.get("provider_name"), "captured_at": r.get("captured_at"),
                }
        started = {(r["away"], r["home"]) for _, r in df.iterrows()
                   if r.get("status") != "STATUS_SCHEDULED"
                   or (pd.notna(r.get("start_utc")) and pd.Timestamp(r["start_utc"]) <= pd.Timestamp(now))}
        with _lock:
            prev = (_good.get(key) or (None, {}))[1] or {}
        for k, v in prev.items():
            if k not in out and k in started:
                out[k] = {**v, "pre_tip": True}
        return out
    try:
        return _cached(key, 300, 90, fetch)
    except RuntimeError as e:
        return Live({}, None, stale=True, source="none", error=str(e))


# -- Box scores after tip (cdn.nba.com liveData) ------------------------------
BOX_TTL, BOX_FAIL_TTL = 120, 60


@dataclass(frozen=True)
class Box:
    """One game's box score as the page reads it: `lines` holds one row per
    player who PLAYED (cdnbox.player_lines: the logs' schema, so
    props.actual_value composes any board stat from it); `final` is the cdn
    payload's own final flag (box's one definition). A player with no row
    has not played: in a final game, he did not play at all."""
    game_id: str
    final: bool
    lines: pd.DataFrame


def box(game_id) -> Live:
    """The cdn liveData box score of one game, for the Game page after tip.
    TTL 120 s (spec 5), a failure remembered for 60 s; no committed copy
    (the page says the box score is unavailable). cdn only, never
    stats.nba.com."""
    gid = str(game_id)

    def fetch():
        payload = H.fetch(cdn.BOX_URL.format(gid=gid), timeout=TIMEOUT, retries=0)
        g = payload.get("game") if isinstance(payload, dict) else None
        if not isinstance(g, dict) or str(g.get("gameId")) != gid:
            raise ValueError(f"box {gid}: no matching game object")
        final = BX.is_final(g, gid, "cdn")
        lines = cdnbox.player_lines(payload)
        if final and lines.empty:
            raise ValueError(f"box {gid}: final with nobody who played")
        return Box(gid, final, lines)
    return _cached(f"box:{gid}", BOX_TTL, BOX_FAIL_TTL, fetch)


# -- Injury report (ak-static PDFs) -------------------------------------------
INJURY_KEY = "injury"
INJURY_TTL, INJURY_FAIL_TTL = 600, 120
INJURY_SLOTS = 8          # floor(now ET, 15 min) and at most 8 slot times back (spec 4)
INJURY_TIMEOUT = 3.0      # per PDF, no retries
INJURY_BUDGET = 5.0       # seconds for the whole walk
INJURY_MIN_FETCH = 0.5    # a fetch with less than this left in the budget is not started


def _report_time(rep):
    return pd.Timestamp(rep.meta["report_ts_et"])


def injury_report(sched: pd.DataFrame | None = None) -> Live:
    """The latest REAL injury report as the object context.report_states takes
    (a ParsedReport live, a context.CommittedReport from the bot's files).

    Real = horizons.real_report against `sched` (default: schedule()), the
    same accept the training snapshots used: QA and Summer League files land
    in the production bucket and one in the newest slot would otherwise
    become the slate's report. TTL 10 minutes; a failure is remembered for 2
    (circuit breaker). Finding no report in the walk is a failure too, never
    "nobody is injured": the last good report is served stale, then the
    committed one, and with neither (or a committed pair read_committed
    cannot trust) it raises RuntimeError (the page falls back to the
    committed projections, which carry the bot's context). On a failure with
    both a remembered and a committed report, the one with the later TITLE
    time is served.
    """
    if sched is None:
        sched = schedule().value
    accept = HZ.real_report(sched)

    def fetch():
        store, end = INJ.live_store(timeout=INJURY_TIMEOUT), time.monotonic() + INJURY_BUDGET

        def getter(slot):
            # No new fetch once the budget is spent, and none that could
            # outlast it: with less than one timeout left, the slot is fetched
            # with what is left (a slow host costs one budget per TTL, not
            # eight timeouts, and never more than the budget).
            left = end - time.monotonic()
            if left < INJURY_MIN_FETCH:
                raise H.FetchError(slot.url, f"injury walk over its {INJURY_BUDGET:.0f}s budget")
            return (store if left >= INJURY_TIMEOUT else INJ.live_store(timeout=left)).get(slot)
        sel = INJ.latest_report(now_et(), getter=getter, max_slots=INJURY_SLOTS, accept=accept)
        if sel is None:
            raise ValueError(f"no real injury report in the last {INJURY_SLOTS} slots")
        return sel.report

    def committed():
        # read_committed raises on a pair it cannot trust. Reported in
        # _cached's own failure type (RuntimeError), so a page that falls back
        # to the committed projections on RuntimeError still does; the
        # message names both failures.
        try:
            return CTX.read_committed()
        except (ValueError, KeyError, OSError) as e:
            with _lock:
                live_err = _fail.get(INJURY_KEY, (None, "unknown"))[1]
            raise RuntimeError(f"{INJURY_KEY}: live fetch failed ({live_err}) and the committed "
                               f"report is unusable ({type(e).__name__}: {str(e)[:160]})") from e
    return _cached(INJURY_KEY, INJURY_TTL, INJURY_FAIL_TTL, fetch, committed, newer=_report_time)


# -- The licensed odds feed (The Odds API): the key, read per call -----------
# Design 2.4.3 (m11, m12; build amendment B7). The key is read on EVERY call,
# never into a module constant at import: on Cloud it is a root-level secret
# Streamlit copies into the environment, and locally (the site or the probe) a
# private file outside every repo. With neither set there is no key, and no
# vendor call is made. The key lives only in a local variable and in the
# params handed to http.vendor_get: never a log line, a cache argument, the
# session, the feed anchor or a page.
KEY_ENV, KEY_FILE_ENV = "BUCKETVALUE_ODDS_API_KEY", "BUCKETVALUE_ODDS_API_KEY_FILE"
KEY_READ_BYTES = 200                       # the most of the key file ever read
_NOT_REGULAR = "not a regular file"        # the fixed code for a FIFO, a device, a folder


def _key_unreadable(code: str) -> None:
    C.logger.warning("oddsapi: key file unreadable (%s)", code)


def _read_key(environ=None) -> str | None:
    """The key: BUCKETVALUE_ODDS_API_KEY, else the first line of the file that
    BUCKETVALUE_ODDS_API_KEY_FILE names ("~" expanded, m12); None when neither
    is set. The file is opened without blocking and refused unless fstat says
    it is a regular file (B7), so a FIFO or /dev/stdin named there returns
    None at once and never hangs a page view; at most KEY_READ_BYTES are
    read. Anything unreadable gives None and one WARNING with a fixed code
    (the exception's type, or "not a regular file"): never the path, a
    message or the key."""
    env = os.environ if environ is None else environ        # an empty test dict means NO key (m11)
    v = (env.get(KEY_ENV) or "").strip()
    if v:
        return v
    path = (env.get(KEY_FILE_ENV) or "").strip()
    if not path:
        return None                                          # no default path, ever
    try:
        fd = os.open(os.path.expanduser(path), os.O_RDONLY | os.O_NONBLOCK)
    except Exception as e:                                   # noqa: BLE001
        _key_unreadable(type(e).__name__)
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            _key_unreadable(_NOT_REGULAR)
            return None
        v = os.read(fd, KEY_READ_BYTES).decode("ascii").strip().splitlines()[0].strip()
        return v or None
    except Exception as e:                                   # noqa: BLE001
        _key_unreadable(type(e).__name__)
        return None
    finally:
        os.close(fd)


# -- The licensed odds feed: the anchor, the walk and the view (design 2.4) ---
# The build amendments apply: a first pull never waits for another game's
# first pull (B1); the governor counts the work order's horizon, the period's
# games from its opening day, tipped ones too (B2); merge_refresh is told what
# each call asked for (B3); and the anchor holds plain data and stdlib
# objects only, never a bucketlib instance or function (B4). A ledger that
# did not see a game's morning, in a period that was spending, calls its
# first pull a re-buy, which the floor guards (B11); a game whose pulls were
# all billed failures is retried once its backoff allows (B12); and the
# refresher recovers today's order written before today's files were (B15).
# A game whose latest paid call the reserve, the floor or the ceiling
# refused is held, and today's view then reads "paused" (design 1.3 and 6,
# N1; build 3's review, F4); the first such hold of an ET day per guard is
# logged once (B19). A re-buy asks the core markets only (B20). A game with
# no lines after tip is "none_posted" only when its books posted none (B18).
# A clean empty answer ends a run of billed failures, and a timeout of a
# call expected to be free (a PrizePicks-only top-up, a pull after an empty
# answer) still costs its worst case but is no pull (build 4's review, L2-1).
# A game the memory-loss brake counted as pulled, with no lines in this
# ledger, keeps one re-buy at once while P >= 2 and before its last refresh
# time (B21). Only an answer that came back, billed and unusable, grows the
# day breaker's streak; a transport failure never does (B22). A first pull by
# a ledger born at or before the game's first-pull time skips the spending
# ceiling (B23).
ODDS_TIMEOUT = TIMEOUT                    # 4 s per call (spec 4); fixed, never a remainder (M8)
PAGE_WAIT_S = 3.0                         # the most a page waits for today's tasks (M6)
ODDS_WORKERS = 4                          # paid calls at once; the pool has one more, for the walk
ODDS_EVENTS_TTL = 600                     # the free /events answer, 10 minutes (UT/ratelimit.txt:60)
FEED_MEMORY_VERSION = 2                   # the ledger's shape (2.4.2): a change replaces the ledger
ANCHOR_NAME = "_bucketvalue_feed_anchor"  # a sys.modules entry OUTSIDE the bucketlib package
REFRESHER_TICK_S = 60                     # the refresher's tick (design 2.4.6)
REFRESHER_VERSION = 1                     # a change replaces the running loop (2.4.6)
FEED_STATES = ("off", "idle", "ok", "stale", "paused", "down")
FEED_REASONS = ("not_yet", "none_posted", "paused", "not_published", "lost_in_restart")
# The budget guards of design 3.6 (items 4 to 6) whose refusal stops a game's
# paid calls until the balance or the date allows: "the reserve, floor or
# ceiling reached" (design 1.3 and 6), which visitors read as paused. The
# rate cap only spaces calls (the rest wait their turn); the tip guard ends
# a game's buying anyway; the auth state and a paused plan pause the view by
# themselves; an unknown balance is learned by the next free events call.
HOLDS = ("reserve", "floor", "ceiling")
AUTH_WAIT = dt.timedelta(minutes=30)      # 3.8: a bad key stops every call this long, the free one too
WARN_EVERY = dt.timedelta(minutes=10)     # one WARNING per log-line code per 10 minutes
_ONCE_A_DAY = dt.timedelta(days=1)
_NO_USAGE = O.Usage(None, None, None)
_LOCK_TYPE = type(threading.Lock())


@dataclass
class FeedLive:
    """What a page reads of the feed for one date (design 2.4). `value` holds
    oddsapi.FEED_COLUMNS rows; `asof` is the OLDEST latest-pull time among
    the games in `value`; `per_game` maps a game_id to {"asof", "books":
    {book: fetched_at}, "pretip", "pp_missing", "reason"}, the reason (one of
    FEED_REASONS) only when the game has no lines; `loading` counts the
    date's games whose call is queued or running, and `loading_games` names
    them (game ids: the Game page counts its own game only, build 4's
    review, L3-4); `generation` is the ledger's change counter (it keys the
    feed tables, m19); `report` holds the unknown team names, the unmatched
    events and the lines taken down (`n_pulled` counts every one, `pulled`
    names up to 8). Plain values: no key, URL or vendor message ever."""
    state: str                 # one of FEED_STATES
    value: pd.DataFrame
    asof: dt.datetime | None
    per_game: dict
    loading: int = 0
    generation: int = 0
    report: dict = field(default_factory=dict)
    loading_games: tuple = ()

    @classmethod
    def off(cls) -> "FeedLive":
        """No key, or the kill switch: today's site, unchanged (M9)."""
        return cls("off", O.empty_feed(), None, {})

    @classmethod
    def down(cls) -> "FeedLive":
        """The feed path failed before it could read memory."""
        return cls("down", O.empty_feed(), None, {})


# -- Small readers ------------------------------------------------------------
def _as_utc(value) -> dt.datetime | None:
    """An aware instant as a plain UTC datetime; None for None, NaT, an ISO
    string that does not read, or a naive time (never guessed)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, dt.datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(UTC)


def _iso_day(value) -> str | None:
    """'YYYY-MM-DD' for a date, a datetime's own date or an ISO date string."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError:
            return None
    return None


def _et_day(t: dt.datetime) -> dt.date:
    return t.astimezone(ET).date()


def _next_et_midnight(t: dt.datetime) -> dt.datetime:
    return dt.datetime.combine(_et_day(t) + dt.timedelta(days=1), dt.time(0), tzinfo=ET).astimezone(UTC)


def _words(kind: str, code: str = "") -> str:
    return " ".join(p for p in (kind, code) if p)


def _warn(mem: dict, code: str, now: dt.datetime, msg: str, *args,
          every: dt.timedelta = WARN_EVERY, level: int = logging.WARNING) -> None:
    """One log line per `code` per `every` (the ledger's "warned"; called with
    mem["lock"] held). Every part of a line is fixed, a number or one of our
    own codes: never exception text, a URL, a vendor message or the key."""
    last = mem["warned"].get(code)
    if isinstance(last, dt.datetime) and abs(now - last) < every:
        return
    mem["warned"][code] = now
    C.logger.log(level, msg, *args)


# -- The feed anchor (2.4.1) and the ledger (2.4.2) ----------------------------
def _anchor_module() -> types.ModuleType:
    """The anchor itself, made once per process: the first setdefault wins.
    A types.ModuleType has no __file__ and a None __spec__, so Streamlit's
    file watcher finds no file for it, and its name is no bucketlib name, so
    freshen's purge never pops it."""
    mod = sys.modules.get(ANCHOR_NAME)
    if mod is None:
        fresh = types.ModuleType(ANCHOR_NAME)
        fresh.lock, fresh.state, fresh.births = threading.Lock(), None, 0
        fresh.pool = None
        fresh.refresher = None
        fresh.refresher_stop = None
        fresh.refresher_version = None
        mod = sys.modules.setdefault(ANCHOR_NAME, fresh)    # one dict operation: the first one wins
    return mod


def feed_anchor() -> dict:
    """The ledger, kept on the feed anchor (2.4.1, B1). A code push (freshen
    pops only bucketlib names), a Streamlit cache clear (it touches only
    Streamlit's caches and the sender's session) and the file watcher all
    miss it; a restart loses it. A ledger that is missing, of another
    FEED_MEMORY_VERSION or of a bad shape (m25) is replaced, and that is
    logged: "process start" the first time, "replaced: version change"
    after. The pool (ODDS_WORKERS + 1 threads: the walk takes one) is made
    here too."""
    mod = _anchor_module()
    with mod.lock:
        if not _valid_ledger(mod.state):
            mod.births += 1
            mod.state = new_feed_memory(now_utc(), had_ledger=mod.births > 1)
            C.logger.info("oddsapi: feed memory created (%s)",
                          "process start" if mod.births == 1 else "replaced: version change")
        if getattr(mod, "pool", None) is None:
            mod.pool = cf.ThreadPoolExecutor(max_workers=ODDS_WORKERS + 1,
                                             thread_name_prefix="oddsapi")
        return mod.state


def _anchor_pool() -> cf.Executor:
    feed_anchor()
    return _anchor_module().pool


def new_feed_memory(now, had_ledger: bool) -> dict:
    """A fresh ledger (2.4.2), born at `now`. Plain data and stdlib objects
    only (B4): no bucketlib instance or function is ever stored, so no code
    of a purged module runs from it, and a task of old code still running
    after a version change writes into the OLD dict it captured. The brake
    reads born_at and had_ledger (3.4); `braked` says it was decided (once
    per ledger), `brake_day` is the ET day it applies to (None when it
    does not) and `brake_used` the account's `used` it was decided on (None
    until then): what the period had spent before this ledger's first paid
    call, which oddsbudget.first_kind reads too (B11). budget["mode"] is the
    last plan's mode (its change is logged)."""
    born = _as_utc(now)
    if born is None:
        raise ValueError("now must be an aware instant")
    return {"v": FEED_MEMORY_VERSION, "born_at": born, "had_ledger": bool(had_ledger),
            "lock": threading.Lock(),            # guards every field below (RC C1)
            "event_locks": {},                   # event_id -> threading.Lock (single flight)
            "queued": set(), "futures": {},      # event_id -> Future, while queued or running (M8)
            "walk": None,                        # the running walk's Future, one at a time
            "events": None,                      # (fetched_at, today's events, report)
            "events_fail": None,                 # (until, kind, fails)
            "games": {},                         # event_id -> entry (_new_entry)
            "budget": {"remaining": None, "used": None, "last": None, "seen_at": None,
                       "auth": "ok", "auth_until": None, "pending": 0, "mode": None},
            "calls": [],                         # UTC times of paid calls in the last 10 minutes
            "streak": 0, "breaker_day": None,    # billed failures in a row; the ET day stopped (B2)
            "generation": 0,                     # bumped on every change to a stored frame
            "work": None,                        # the latest work order (2.4.5)
            "warned": {},                        # log-line code -> last UTC
            "braked": False,                     # the memory-loss brake was decided (3.4)
            "brake_day": None,                   # the ET day it applies to, when it does (review R5)
            "brake_used": None}                  # the `used` it was decided on (B11's usage test)


_LEDGER_FIELDS = {"born_at": dt.datetime, "had_ledger": bool, "lock": _LOCK_TYPE,
                  "event_locks": dict, "queued": set, "futures": dict, "games": dict,
                  "budget": dict, "calls": list, "streak": int, "generation": int,
                  "warned": dict, "braked": bool}
_LEDGER_OPTIONAL = {"walk": cf.Future, "events": tuple, "events_fail": tuple,
                    "breaker_day": dt.date, "work": dict, "brake_day": dt.date,
                    "brake_used": int}
_LEDGER_KEYS = frozenset({"v", *_LEDGER_FIELDS, *_LEDGER_OPTIONAL})
_BUDGET_KEYS = frozenset({"remaining", "used", "last", "seen_at", "auth", "auth_until",
                          "pending", "mode"})


def _valid_ledger(state) -> bool:
    """The ledger's version and the type of every top-level field (m25)."""
    if not isinstance(state, dict) or state.get("v") != FEED_MEMORY_VERSION:
        return False
    if set(state) != _LEDGER_KEYS:
        return False
    for name, kind in _LEDGER_FIELDS.items():
        v = state[name]
        if not isinstance(v, kind) or (kind is int and isinstance(v, bool)):
            return False
    for name, kind in _LEDGER_OPTIONAL.items():
        v = state[name]
        if v is not None and (not isinstance(v, kind) or (kind is int and isinstance(v, bool))):
            return False
    b = state["budget"]
    return (set(b) == _BUDGET_KEYS and type(b["pending"]) is int
            and b["auth"] in OB.AUTH_STATES)


def _new_entry(game_id: str, game_date: str, tip, vendor_tip) -> dict:
    """One game's ledger entry, keyed by its event id (2.4.2). `held` is the
    budget guard of HOLDS that refused the game's latest paid call (a fixed
    code), None once a call goes ahead or another guard decides it; the view
    reads it, the governor never does. `braked` says the memory-loss brake
    counted the game's first pull at the ledger's birth (oddsbudget.Slot's
    `braked`: the game keeps one re-buy, build amendment B21). `free_failed`
    says a call expected to be free (L2-1) failed with no answer, may have
    been billed and was not counted as a pull, and no clean answer (nothing
    posted, or lines that landed) has come for the game since: until one
    does, the game's next call is neither expected to be free nor a true
    first pull (build 6's review, S12-R1). An entry written before either
    flag existed (a ledger kept across a code push, whose shape the version
    check does not see) reads it as False, so it stays valid and nothing
    more is bought for it."""
    return {"game_id": game_id, "game_date": game_date, "tip_utc": tip, "vendor_tip": vendor_tip,
            "frame": None, "pulls": 0, "first_at": None, "last_at": None, "empty_next": None,
            "pp_missing": False, "pp_next": None, "fail_until": None, "fails": 0,
            "books_at": {}, "pulled": [], "n_pulled": 0, "held": None, "braked": False,
            "free_failed": False}


_ENTRY_TIMES = ("tip_utc", "vendor_tip", "first_at", "last_at", "empty_next", "pp_next",
                "fail_until")


def _instant(v) -> bool:
    return v is None or (isinstance(v, dt.datetime) and v.tzinfo is not None
                         and v.utcoffset() is not None)


def _slot_of(entry, event_id: str = "") -> OB.Slot | None:
    """The governor's Slot for one ledger entry, built when read and never
    stored; None for a malformed entry (m25), which then counts as absent.
    has_lines is whether the game has a stored frame with lines in it (B12:
    due_kind retries a game with none after a failure); braked is the
    entry's own flag (B21: a braked game keeps one re-buy), False for an
    entry from before the flag existed, and anything but a bool is
    malformed. So is a `free_failed` (S12-R1) that is not a bool; the Slot
    does not carry it (live's own ticket reads it), and an entry from before
    it existed reads it as False."""
    try:
        gid, day, pulls, fails = entry["game_id"], entry["game_date"], entry["pulls"], entry["fails"]
        frame, braked = entry["frame"], entry.get("braked", False)
        if not (isinstance(gid, str) and gid and isinstance(day, str)
                and type(pulls) is int and pulls >= 0 and type(fails) is int and fails >= 0
                and type(entry["pp_missing"]) is bool and isinstance(entry["books_at"], dict)
                and isinstance(entry["pulled"], list) and type(braked) is bool
                and type(entry.get("free_failed", False)) is bool
                and (frame is None or isinstance(frame, pd.DataFrame))
                and all(_instant(entry[k]) for k in _ENTRY_TIMES)):
            return None
        dt.date.fromisoformat(day)
        return OB.Slot(event_id=str(event_id), game_id=gid, tip_utc=entry["tip_utc"], pulls=pulls,
                       first_at=entry["first_at"], last_at=entry["last_at"],
                       empty_next=entry["empty_next"], pp_missing=entry["pp_missing"],
                       pp_next=entry["pp_next"], fail_until=entry["fail_until"], fails=fails,
                       has_lines=frame is not None and len(frame) > 0, braked=braked)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def reset_feed_anchor_for_tests() -> None:
    """Drop the anchor and stop what it runs: the refresher's stop Event is
    set and its thread joined, and the pool is shut down and waited for, so
    no task outlives a test (M10). The refresher's memory of committed files
    is emptied too. Tests only."""
    with _ids_lock:
        _ids_seen.clear()
    mod = sys.modules.pop(ANCHOR_NAME, None)
    if mod is None:
        return
    stop = getattr(mod, "refresher_stop", None)
    if isinstance(stop, threading.Event):
        stop.set()
    thread = getattr(mod, "refresher", None)
    if isinstance(thread, threading.Thread) and thread.is_alive():
        thread.join(timeout=2.0)
    pool = getattr(mod, "pool", None)
    if isinstance(pool, cf.Executor):
        pool.shutdown(wait=True, cancel_futures=True)


# -- The work order (2.4.5) -------------------------------------------------------
def _work_rows(work, name: str) -> list[dict]:
    rows = work.get(name) if isinstance(work, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _tbd(value) -> bool:
    """A work order's tbd flag: True only when it says so (match_events' rule)."""
    return value is True or str(value) == "True"


def _work_tip(row) -> dt.datetime | None:
    if not isinstance(row, dict) or _tbd(row.get("tbd")):
        return None
    return _as_utc(row.get("tip_utc"))


def _todays_games(work, today: str) -> list[dict]:
    """The work order's buyable games dated `today` (an ISO date), in the
    shape match_events reads."""
    out = []
    for r in _work_rows(work, "games"):
        gid = r.get("game_id")
        if isinstance(gid, str) and gid and _iso_day(r.get("game_date")) == today:
            out.append({"game_id": gid, "game_date": today, "home": r.get("home"),
                        "away": r.get("away"), "tip_utc": _work_tip(r), "tbd": _tbd(r.get("tbd"))})
    return out


def _horizon_of(work) -> list:
    """The governor's games (B2): the work order's horizon, every projected
    game from the period's opening day to the horizon end, tipped ones too.
    A TBD tip is None (oddsbudget.Game's rule)."""
    out = []
    for r in _work_rows(work, "horizon"):
        gid, day = r.get("game_id"), _iso_day(r.get("game_date"))
        if isinstance(gid, str) and gid and day is not None:
            out.append(OB.Game(gid, dt.date.fromisoformat(day), _work_tip(r)))
    return out


def _busiest(work) -> int | None:
    b = work.get("busiest") if isinstance(work, dict) else None
    return b if type(b) is int and b > 0 else None


@dataclass(frozen=True)
class _Budget:
    """One walk's or one pull's reading of the ledger for the governor.
    Built under mem["lock"] each time it is needed, and never stored (B4)."""
    horizon: list
    period: OB.Period
    plan: OB.Plan
    slots: dict
    busiest: int | None


def _slots(mem: dict) -> dict:
    out = {}
    for eid, entry in mem["games"].items():
        slot = _slot_of(entry, eid)
        if slot is not None:
            out[eid] = slot
    return out


def _budget(mem: dict, s, now: dt.datetime) -> _Budget:
    """The period and the plan from the ledger's counters and the work order
    (design 3.2, 3.3): the horizon goes in as period_games and as horizon
    (B2), with the ledger's breaker_day (B4)."""
    work = mem["work"]
    horizon, busiest = _horizon_of(work), _busiest(work)
    b = mem["budget"]
    remaining, used = b["remaining"], b["used"]
    allowance = None if remaining is None or used is None else remaining + used
    period = OB.period_of(now, used, allowance, s.reserve, len(s.markets), horizon, busiest=busiest)
    slots = _slots(mem)
    plan = OB.plan(remaining, used, b["pending"], s, horizon, slots, period, b["auth"], busiest,
                   now=now, period_games=horizon, breaker_day=mem["breaker_day"])
    return _Budget(horizon, period, plan, slots, busiest)


# -- The kick (2.4.4) ---------------------------------------------------------------
def feed_kick(work: dict, *, mem: dict | None = None, environ=None, pool=None, now=None) -> None:
    """Start a walk for `work` (2.4.4), from a page view or the refresher. It
    never blocks and never raises. The key and the settings are read on this
    call (RC K3, G1): with no key, or the switch off, it returns before it
    touches memory. Otherwise the work order goes into the ledger and, unless
    a walk is already running (it reads the new order), one is submitted to
    `pool`: the anchor's when None, a synchronous one in tests (M10). `mem`
    is the ledger (the anchor's when None), and `now` pins the walk's clock."""
    try:
        env = os.environ if environ is None else environ
        if _read_key(env) is None or not O.read_settings(env).enabled:
            return
        if mem is None:
            mem = feed_anchor()
        ticket = cf.Future()                 # holds the walk's place until its own Future exists
        with mem["lock"]:
            if isinstance(work, dict):
                mem["work"] = work
            running = mem["walk"]
            if running is not None and not running.done():
                return                       # one walk at a time, never waited for
            mem["walk"] = ticket
        target = pool if pool is not None else _anchor_pool()
        try:
            fut = target.submit(_walk, mem, environ, target, now)
        except Exception:
            with mem["lock"]:
                if mem["walk"] is ticket:
                    mem["walk"] = None
            ticket.set_result(None)
            raise
        with mem["lock"]:
            if mem["walk"] is ticket:
                mem["walk"] = fut
        fut.add_done_callback(functools.partial(_walk_done, ticket))
    except Exception as e:  # noqa: BLE001 - the type only: never its text
        C.logger.warning("oddsapi: kick failed (%s)", type(e).__name__)


def _walk_done(ticket: cf.Future, fut: cf.Future) -> None:
    if not fut.cancelled() and fut.exception() is not None:
        C.logger.warning("oddsapi: walk failed (%s)", type(fut.exception()).__name__)
    if not ticket.done():
        ticket.set_result(None)


# -- The walk (2.4.4) -----------------------------------------------------------------
def _walk(mem: dict, environ=None, pool=None, now=None) -> None:
    """One walk: the free events call when due, the period and the plan, the
    brake (once per ledger), then each of today's matched games with a due
    kind, in tip order, queued once and sent to the pool. Never raises: a
    failure logs its type only."""
    try:
        _walk_once(mem, environ, pool, now)
    except Exception as e:  # noqa: BLE001 - the type only: never its text
        C.logger.warning("oddsapi: walk failed (%s)", type(e).__name__)


def _walk_once(mem: dict, environ, pool, pinned) -> None:
    now = _as_utc(now_utc() if pinned is None else pinned)     # live's own binding (T1)
    env = os.environ if environ is None else environ
    key = _read_key(env)
    s = O.read_settings(env)
    if key is None or not s.enabled:
        return
    today = _et_day(now)
    with mem["lock"]:
        for problem in s.problems:                             # fixed phrases: names, never values
            _warn(mem, "setting:" + problem, now, "oddsapi: %s", problem)
        _prune(mem, now)
        games = _todays_games(mem["work"], today.isoformat())
    if not games:
        return                                 # nothing buyable today: no call at all (m26)
    events = _events(mem, key, today, now, s)
    del key                                    # the pulls read their own (3.6 item 8)
    if events is None:
        return
    with mem["lock"]:
        todays = _match(mem, events, games, now)
        view = _budget(mem, s, now)
        if _brake(mem, s, view, todays, today, now):
            view = _budget(mem, s, now)
        _note_mode(mem, view, now)
        _watch_reset(mem, s, view, now)
        due = _due(mem, todays, view, s, now, cap=len(games))
    if due:
        _send(mem, due, environ, pool, pinned)


def _prune(mem: dict, now: dt.datetime) -> None:
    """2.4.2: games dated before yesterday go, with their locks, so the
    ledger keeps one compact frame per game (RC C5); a malformed entry goes
    too (it counts as absent, m25, logged once). Old rate-cap times and old
    log marks go. A game whose call is queued or running stays."""
    cutoff = (_et_day(now) - dt.timedelta(days=1)).isoformat()
    for eid, entry in list(mem["games"].items()):
        if eid in mem["queued"]:
            continue
        if _slot_of(entry, eid) is None:
            _warn(mem, "malformed_entry", now, "oddsapi: a malformed game entry counted as absent")
            del mem["games"][eid]
        elif entry["game_date"] < cutoff:
            del mem["games"][eid]
    for eid, lock in list(mem["event_locks"].items()):
        if eid not in mem["games"] and not lock.locked():
            del mem["event_locks"][eid]
    mem["calls"] = [c for c in mem["calls"] if abs(now - c) < OB.RATE_WINDOW]
    for code, at in list(mem["warned"].items()):
        if not isinstance(at, dt.datetime) or abs(now - at) > 2 * _ONCE_A_DAY:
            del mem["warned"][code]


def _events(mem: dict, key: str, today: dt.date, now: dt.datetime, s) -> pd.DataFrame | None:
    """Today's vendor events (2.4.4 step 2): the free call when the copy is
    older than ODDS_EVENTS_TTL and nothing holds it back, else the copy. A
    failure keeps the old copy and backs off (90 s doubling to 60 min); a
    bad key stops it for 30 minutes (3.8). None when there is no copy."""
    with mem["lock"]:
        due = _events_due(mem, today, now)
    if due:
        _events_call(mem, key, today, now, s)
    with mem["lock"]:
        ev = mem["events"]
        return ev[1] if ev is not None and _et_day(ev[0]) == today else None


def _events_due(mem: dict, today: dt.date, now: dt.datetime) -> bool:
    fail = mem["events_fail"]
    if fail is not None and now < fail[0]:
        return False                           # backing off, whatever the auth state
    b = mem["budget"]
    if b["auth"] in ("bad_key", "out_of_credits") and b["auth_until"] is not None:
        return now >= b["auth_until"]          # after 30 minutes, one free call checks (3.8)
    ev = mem["events"]
    fresh = (ev is not None and _et_day(ev[0]) == today
             and now - ev[0] < dt.timedelta(seconds=ODDS_EVENTS_TTL))
    return not fresh


def _events_call(mem: dict, key: str, today: dt.date, now: dt.datetime, s) -> None:
    path, params = O.events_request()
    params = {**params, "apiKey": key}
    resp, err = None, None
    try:
        resp = H.vendor_get(O.HOST, path, params, timeout=ODDS_TIMEOUT, secrets=(key,))
    except H.VendorTransportError as e:
        err = (e.kind, e.received)             # its kind and byte count only: never its text
    frame = stats = None
    if resp is not None and resp.status == 200:
        try:
            frame, stats = O.parse_events(resp.body, today)
        except Exception:  # noqa: BLE001 - the parser never raises by contract; one that does
            # is a bad payload, settled as one: the balance learned and the call backed off
            # (build 4's review, L1-1), never a walk that dies before it settles.
            frame, stats = None, {"bad_payload": True, "unknown_team": []}
    with mem["lock"]:
        _settle_events(mem, resp, err, frame, stats, now, s)


def _settle_events(mem, resp, err, frame, stats, now, s) -> None:
    b = mem["budget"]
    if resp is not None:
        _take_usage(mem, O.usage_from_headers(resp.headers), billed=False, cost=0, now=now, s=s)
    if resp is not None and resp.status == 200 and not stats["bad_payload"]:
        mem["events"] = (now, frame, {"unknown_team": list(stats["unknown_team"]),
                                      "unmatched_events": 0, "games_no_event": 0, "swapped": 0})
        mem["events_fail"] = None
        if stats["unknown_team"]:              # oddsapi cut them: printable, long tokens redacted
            _warn(mem, "unknown_team", now, "oddsapi: unknown team name(s): %s",
                  ", ".join(stats["unknown_team"]))
        if b["auth"] == "bad_key" or (b["auth"] == "out_of_credits" and b["auth_until"] is not None):
            b["auth"], b["auth_until"] = "ok", None     # 3.8: the first 200 after the wait
            C.logger.info("oddsapi: the events call answered again; paid calls may resume")
        return
    if resp is not None and resp.status == 401:
        _auth_refused(mem, O.error_code(401, resp.snippet), now, events=True)
        return
    kind, code = _failure_words(resp, err)
    fails = (mem["events_fail"][2] if mem["events_fail"] else 0) + 1
    mem["events_fail"] = (now + OB.backoff(fails), kind, fails)
    _warn(mem, f"events:{kind}:{code}", now, "oddsapi events failed: %s", _words(kind, code))


def _failure_words(resp, err) -> tuple[str, str]:
    """(kind, code) of a failed call: our own words, never vendor text."""
    if err is not None:
        kind = err[0]
        return ("server", kind) if kind in ("connect", "transport") else (kind, "")
    if resp.status == 200:
        return "bad_payload", ""
    code = O.error_code(resp.status, resp.snippet)
    return O.failure_kind(resp.status, code), code


def _auth_refused(mem: dict, code: str, now: dt.datetime, *, events: bool) -> None:
    """3.8: a 401 is out of credits only when its code says so; any other
    (INVALID_KEY, DEACTIVATED_KEY, MISSING_KEY, no readable code) is a bad
    key, which stops every call for AUTH_WAIT. Out of credits stops the paid
    calls; the free events call goes on watching the balance, or, when it
    was refused too, tries again every AUTH_WAIT."""
    b = mem["budget"]
    if code == "OUT_OF_USAGE_CREDITS":
        b["auth"], b["auth_until"] = "out_of_credits", (now + AUTH_WAIT if events else None)
        _warn(mem, "out_of_credits", now,
              "oddsapi: 401 OUT_OF_USAGE_CREDITS; paid calls stopped until the credits reset")
    else:
        b["auth"], b["auth_until"] = "bad_key", now + AUTH_WAIT
        _warn(mem, "bad_key:" + code, now, "oddsapi: 401 %s; all calls stopped for 30 min", code)


def _take_usage(mem: dict, usage, *, billed: bool, cost: int, now: dt.datetime, s) -> int:
    """The vendor's counters off one response (3.1, 3.5): `remaining` from
    x-requests-remaining, else less x-requests-last, else less the cost when
    the call may have been billed; `used` alike; `last` as given. Returns
    the credits counted for the call. Answers can land out of order while
    calls overlap, so within one period `remaining` only falls and `used`
    only rises, unless the counters move by more than the calls that can be
    in flight: a reset (used falling, 3.8 and 3.2) or new credits."""
    b = mem["budget"]
    prev_r, prev_u = b["remaining"], b["used"]
    spent = usage.last if usage.last is not None else (int(cost) if billed else 0)
    slack = (ODDS_WORKERS + 1) * max(1, len(s.markets))
    reset = OB.reset_seen(prev_u, None if usage.used is None else usage.used + slack)
    rose = prev_r is not None and usage.remaining is not None and usage.remaining > prev_r + slack
    jump = reset or rose
    if usage.remaining is not None:
        b["remaining"] = usage.remaining if jump or prev_r is None else min(prev_r, usage.remaining)
    elif prev_r is not None:
        b["remaining"] = max(0, prev_r - spent)
    if usage.used is not None:
        b["used"] = usage.used if jump or prev_u is None else max(prev_u, usage.used)
    elif prev_u is not None:
        b["used"] = prev_u + spent
    if usage.last is not None:
        b["last"] = usage.last
    if (usage.remaining, usage.used, usage.last) != (None, None, None):
        b["seen_at"] = now
    if reset and not OB.in_reset_window(now):
        _warn(mem, "reset_elsewhere", now,
              "oddsapi: credits reset outside the 1st; the period rule needs a change",
              every=_ONCE_A_DAY)
    if jump and b["auth"] == "out_of_credits":
        b["auth"], b["auth_until"] = "ok", None
        C.logger.info("oddsapi: the credits came back; paid calls resume")
    return spent


def _match(mem: dict, events: pd.DataFrame, games: list, now: dt.datetime) -> list:
    """Today's matched games as (event_id, ledger entry), in tip order. A new
    match gets a fresh entry, and the tip is THE tip (oddsbudget.tip_for: the
    earlier of the vendor's and the schedule's, a TBD schedule tip ignored)."""
    matched, stats = O.match_events(events, games)
    ev = mem["events"]
    if ev is not None:
        ev[2].update(unmatched_events=stats["event_no_game"], games_no_event=stats["game_no_event"],
                     swapped=stats["swapped"])
    out = []
    for r in matched.to_dict("records"):
        eid, gid, day = r["event_id"], r["game_id"], r["game_date"]
        vendor = _as_utc(r["commence_utc"])
        tip = _as_utc(OB.tip_for(r["commence_utc"], r["tip_utc"], bool(r["tbd"])))
        entry = mem["games"].get(eid)
        if entry is not None and _slot_of(entry, eid) is None:
            _warn(mem, "malformed_entry", now, "oddsapi: a malformed game entry counted as absent")
            entry = None
        if entry is None:
            entry = _carried(mem, gid, eid) or _new_entry(gid, day, tip, vendor)
            mem["games"][eid] = entry
        entry.update(game_id=gid, game_date=day, tip_utc=tip, vendor_tip=vendor)
        out.append((eid, entry))
    out.sort(key=lambda p: (p[1]["tip_utc"] is None, p[1]["tip_utc"] or now, p[1]["game_id"]))
    return out


def _carried(mem: dict, game_id: str, event_id: str) -> dict | None:
    """A game the vendor now lists under another event id keeps its entry."""
    for other, entry in list(mem["games"].items()):
        if (other != event_id and other not in mem["queued"] and isinstance(entry, dict)
                and entry.get("game_id") == game_id and _slot_of(entry, other) is not None):
            del mem["games"][other]
            return entry
    return None


def _brake(mem: dict, s, view: _Budget, todays: list, today: dt.date, now: dt.datetime) -> int:
    """The memory-loss brake (3.4; B1 item 2, M1; B4): decided once per
    ledger, at the first walk that knows the balance (an unknown `used` would
    read as not spending), and the decision is kept: `brake_day` is the ET
    day it applies to, or None. On that day, each of the day's untipped
    games with no pull counts as pulled once, at the ledger's birth, so its
    next pull follows the normal spacing from then: the games matched at the
    walk that decided it, and those matched at any later walk (an events
    answer that listed only some of them, review R5). It never reaches
    another day's games. The `used` it was decided on is kept as
    `brake_used`: no paid call is made before this decision (none goes
    ahead without a balance), so it is what the period had spent before
    this ledger, and oddsbudget.first_kind's usage test reads it (B11).
    Each game it counts is marked `braked`: its one pull is assumed, never
    bought by this ledger, so with no lines here the game keeps one re-buy
    at once while P >= 2 and before its last refresh time
    (oddsbudget.due_kind's "rebuy", build amendment B21). Returns the games
    braked by this walk."""
    b = mem["budget"]
    if not mem["braked"]:
        if b["used"] is None:
            return 0
        mem["braked"], mem["brake_used"] = True, int(b["used"])
        first = OB.first_pull_time(today, s.first_pull_et)
        before = OB.games_before(view.horizon, view.period, today)
        if OB.brake_applies(mem["born_at"], mem["had_ledger"], b["used"], len(s.core), before, first):
            mem["brake_day"] = today
    if mem["brake_day"] != today:
        return 0
    born, n = mem["born_at"], 0
    for _eid, entry in todays:
        tip = entry["tip_utc"]
        if entry["pulls"] == 0 and tip is not None and now < tip:
            entry["pulls"], entry["first_at"], entry["last_at"] = 1, born, born
            entry["braked"] = True                  # B21: its pull is assumed, never bought here
            n += 1
    if n:
        # A WARNING (the default log level shows it; build 4's review, L2-2):
        # a braked game is re-bought at once only while P >= 2 (B21), else it
        # gets no line until its next planned refresh, if it has one.
        C.logger.warning("oddsapi: memory-loss brake: %d game(s) of today counted as pulled at the "
                         "ledger's birth", n)
    return n


_MODE_WORDS = {("full", "ok"): "full", ("core", "below_full"): "core markets only",
               ("short", "below_core"): "short, morning pulls only",
               ("paused", "at_reserve"): "paused at the reserve",
               ("paused", "out_of_credits"): "paused, out of credits",
               ("paused", "bad_key"): "paused, the key was refused",
               ("paused", "breaker"): "paused by the day breaker",
               ("unknown", "no_balance"): "unknown, no balance yet"}


def _note_mode(mem: dict, view: _Budget, now: dt.datetime) -> None:
    """A mode change logs one WARNING with the mode, remaining, reserve and
    games (design 2.4.4 step 3, section 6); a first plan that is "full" is
    only INFO."""
    b, pl = mem["budget"], view.plan
    if pl.mode == b["mode"]:
        return
    first, b["mode"] = b["mode"] is None, pl.mode
    games = len(OB.horizon_games(view.horizon, now, view.period))
    C.logger.log(logging.INFO if first and pl.mode == "full" else logging.WARNING,
                 "oddsapi: budget %s (remaining %s, reserve %d, %d games to the reset)",
                 _MODE_WORDS.get((pl.mode, pl.reason), pl.mode), b["remaining"], pl.reserve, games)


def _watch_reset(mem: dict, s, view: _Budget, now: dt.datetime) -> None:
    """3.2: with no reset seen by 12:00 ET on the 1st (`used` still fails the
    window's own test), one WARNING that day."""
    t, b = now.astimezone(ET), mem["budget"]
    if t.day != 1 or t.time() < dt.time(12, 0) or b["used"] is None or b["remaining"] is None:
        return
    close = OB.reset_window(t.year, t.month)[1]
    probe = OB.period_of(close - dt.timedelta(microseconds=1), b["used"], b["used"] + b["remaining"],
                         s.reserve, len(s.markets), view.horizon, busiest=view.busiest)
    if probe.basis == "used_old":
        _warn(mem, "no_reset:" + t.date().isoformat(), now,
              "oddsapi: credits did not reset by midday on the 1st; check the vendor dashboard",
              every=_ONCE_A_DAY)


def _due(mem: dict, todays: list, view: _Budget, s, now: dt.datetime, *, cap: int) -> list[str]:
    """Each matched game of today, in tip order, with a due kind and not
    already queued, is queued now (M8): the queue never holds more tasks
    than today's games."""
    due = []
    for eid, entry in todays:
        if len(mem["queued"]) >= cap:
            break
        slot = view.slots.get(eid)
        if eid in mem["queued"] or slot is None:
            continue
        if OB.due_kind(slot, view.plan, s, now, entry["game_date"],
                       breaker_day=mem["breaker_day"]) is None:
            continue
        mem["queued"].add(eid)
        due.append(eid)
    return due


def _send(mem: dict, due: list, environ, pool, pinned) -> None:
    """Each queued game's pull goes to the pool. Its Future's done-callback
    takes it off the queue and logs any exception's type (M7); the walk's
    one INFO line is logged when its last pull is done."""
    tally = {"lock": threading.Lock(), "left": len(due), "games": 0, "cost": 0}
    for eid in due:
        try:
            fut = pool.submit(_pull_game, mem, eid, environ, pinned)
        except Exception as e:  # noqa: BLE001 - a pool that takes no work (shut down)
            with mem["lock"]:
                mem["queued"].discard(eid)
            C.logger.warning("oddsapi: worker failed (%s)", type(e).__name__)
            _tallied(mem, tally, None)
            continue
        with mem["lock"]:
            mem["futures"][eid] = fut
        fut.add_done_callback(functools.partial(_pull_done, mem, eid, tally))


def _pull_done(mem: dict, event_id: str, tally: dict, fut: cf.Future) -> None:
    with mem["lock"]:
        if mem["futures"].get(event_id) in (None, fut):
            mem["futures"].pop(event_id, None)
            mem["queued"].discard(event_id)
    result = None
    if not fut.cancelled():
        exc = fut.exception()
        if exc is not None:
            C.logger.warning("oddsapi: worker failed (%s)", type(exc).__name__)
        else:
            result = fut.result()
    _tallied(mem, tally, result)


def _tallied(mem: dict, tally: dict, result) -> None:
    with tally["lock"]:
        tally["left"] -= 1
        if isinstance(result, dict) and result.get("billed"):
            tally["games"] += 1
            tally["cost"] += int(result.get("cost") or 0)
        last, games, cost = tally["left"] == 0, tally["games"], tally["cost"]
    if last and games:
        with mem["lock"]:
            remaining = mem["budget"]["remaining"]
        C.logger.info("oddsapi: walk pulled %d game(s), cost %d, remaining %s", games, cost, remaining)


# -- One game's paid call (2.4.4, 3.5, 3.6) -----------------------------------------
def _event_lock(mem: dict, event_id: str) -> threading.Lock:
    with mem["lock"]:
        return mem["event_locks"].setdefault(event_id, threading.Lock())


def _pull_game(mem: dict, event_id: str, environ=None, now=None) -> dict | None:
    """One game's paid call, on the pool. Design 3.6's guards, in order: a
    non-blocking event lock (M8: another task on this game means this one
    returns at once, never waits), then under the ledger lock a fresh
    due_kind (fail_until and the day breaker, B2 item 2) and
    oddsbudget.may_spend (the tip guard, the auth state, the reserve, the
    floor, the ceiling and the rate cap), and the key and the kill switch
    read again (item 8). No lock is held across the call but the event's
    own, which nobody ever waits on. `pending` comes back in a finally
    (M7), and a raise anywhere after the ticket, the settling included,
    settles the call once as a failed one (review R1). Returns {"billed",
    "cost"} after a call, else None; it never raises (an exception logs its
    type only)."""
    try:
        env = os.environ if environ is None else environ
        key = _read_key(env)
        s = O.read_settings(env)
        if key is None or not s.enabled:
            return None
        lock = _event_lock(mem, event_id)
        if not lock.acquire(blocking=False):
            return None
        try:
            return _pull_locked(mem, event_id, key, s, now)
        finally:
            lock.release()
    except Exception as e:  # noqa: BLE001 - the type only: never its text
        C.logger.warning("oddsapi: worker failed (%s)", type(e).__name__)
        return None


def _ticket(mem: dict, event_id: str, s, now: dt.datetime) -> dict | None:
    """Under mem["lock"]: the fresh due kind and every budget guard of 3.6.
    A game with no pull in this ledger (due_kind's "first") goes to the
    governor as oddsbudget.first_kind says: "first", or "rebuy" when the
    ledger has not seen the game's morning in a period that was spending,
    which the morning-pull floor then guards (B11). A "first" by a ledger
    born at or before the game's first-pull time (oddsbudget.true_first_pull,
    first_kind's first clause) is a true first pull, which skips the
    spending ceiling too (B23), except while the game's entry says
    `free_failed` (build 6's review, S12-R1): since its last clean answer a
    call of it expected to be free failed with no answer and may have been
    billed, so this retry may buy again what that call bought, and the
    ceiling holds it as it holds the retry of a first pull whose failure
    counted (B12's "refresh"). A failure that cannot have been billed (no
    connection, nothing received) leaves the exemption alone. A braked
    game's one re-buy comes as due_kind's own "rebuy" (B21) and is guarded
    as B11's. On a yes,
    `pending += cost` and the call's time for the rate cap, and the call's
    books (the five, or PrizePicks alone for a top-up) and markets (B1: this
    ledger's first pull of a game asks the plan's markets only when the
    floor holds at that cost, else the core markets; B20: a re-buy asks the
    core markets only). A refusal by a budget guard (HOLDS) holds the game
    and is logged once a day per guard (B19)."""
    entry = mem["games"].get(event_id)
    slot = _slot_of(entry, event_id)
    if slot is None:
        return None
    view = _budget(mem, s, now)
    kind = OB.due_kind(slot, view.plan, s, now, entry["game_date"], breaker_day=mem["breaker_day"])
    if kind is None:
        return None
    true_first = False
    free_failed = entry.get("free_failed", False) is True       # S12-R1 (_failed sets it)
    if kind == "first":
        day = entry["game_date"]
        first_at = OB.first_pull_time(day, s.first_pull_et)
        kind = OB.first_kind(mem["born_at"], mem["had_ledger"], mem["brake_used"], len(s.core),
                             OB.games_before(view.horizon, view.period, day), first_at)
        true_first = (kind == "first" and not free_failed
                      and OB.true_first_pull(mem["born_at"], first_at))
    b = mem["budget"]
    waiting = OB.waiting_first(view.horizon, view.slots, now, view.period, slot.game_id)
    if kind == OB.REBUY:
        # B20: a re-buy is likely a duplicate purchase, so it asks the core
        # markets only (cost c_core), never first_pull_markets' full list.
        markets = tuple(s.core) or tuple(view.plan.markets)[:1]
    elif kind == "first":
        markets = OB.first_pull_markets(view.plan, s, b["remaining"], b["pending"], waiting)
    else:
        markets = tuple(view.plan.markets)
    mem["calls"] = [c for c in mem["calls"] if abs(now - c) < OB.RATE_WINDOW]
    refused = OB.may_spend(b["remaining"], b["used"], b["pending"], len(markets), view.plan,
                           waiting, len(s.core), now=now, tip=slot.tip_utc, guard=s.tip_guard,
                           auth=b["auth"], calls=mem["calls"], max_calls=s.max_calls_10min,
                           kind=kind, true_first=true_first)
    # The latest budget verdict on this game, for the view (HOLDS): held by
    # the reserve, the floor or the ceiling, or not (a call that goes ahead,
    # or a guard that is no budget hold, such as the rate cap).
    entry["held"] = refused if refused in HOLDS else None
    if refused is not None:
        _refused(mem, refused, now)
        return None
    b["pending"] += len(markets)
    mem["calls"].append(now)
    # "free": a call expected to be free, whose game's last answer had
    # nothing (a PrizePicks-only top-up, or a pull after an empty answer). A
    # timeout of one still takes its worst case off the balance, but is not
    # one of the game's P pulls (build 4's review, L2-1). Once per clean
    # answer: after such a failure (`free_failed`) the game's next call is
    # not taken as free until the vendor answers for the game with nothing
    # posted or with lines (build 6's review, S12-R1), so a vendor that bills
    # and times out every call meets P again, as B22 leaves no streak to
    # stop it.
    return {"kind": kind, "markets": markets, "cost": len(markets),
            "books": (O.REFERENCE_BOOK,) if kind == "pp" else tuple(s.books),
            "game_id": entry["game_id"], "game_date": entry["game_date"], "tip": entry["tip_utc"],
            "free": not free_failed and (kind == "pp" or entry["empty_next"] is not None)}


def _refused(mem: dict, code: str, now: dt.datetime) -> None:
    """The log lines of a refused paid call. A budget hold (HOLDS: the
    reserve, the floor or the ceiling) logs one fixed WARNING per guard per
    ET day (build amendment B19), through the ledger's once-per-period
    marks; the ceiling keeps design 6's own line beside it."""
    if code in HOLDS:
        day = _et_day(now).isoformat()
        _warn(mem, f"held:{code}:{day}", now, "oddsapi: budget held a pull today (%s)", code,
              every=_ONCE_A_DAY)
    if code == "ceiling":
        _warn(mem, "ceiling:" + _et_day(now).isoformat(), now,
              "oddsapi: spending ceiling reached for today", every=_ONCE_A_DAY)
    elif code == "rate_cap":
        _warn(mem, "rate_cap", now, "oddsapi: paid-call rate cap reached; the rest wait their turn")


def _release(mem: dict, t: dict) -> None:
    b = mem["budget"]
    b["pending"] = max(0, b["pending"] - t["cost"])


def _pull_locked(mem: dict, event_id: str, key: str, s, pinned) -> dict | None:
    """The ticket, the call and its settling (3.5, 3.6). Whatever raises
    after the ticket, the call is settled once, as a broken one (M7): a
    raise before the settling, in the finally, with `pending` given back;
    a raise inside the settling itself (review R1), under the same hold of
    mem["lock"]: what the settling had done is undone first, so the call
    counts once, never twice and never not at all."""
    now = _as_utc(now_utc() if pinned is None else pinned)
    with mem["lock"]:
        t = _ticket(mem, event_id, s, now)
    if t is None:
        return None
    released, resp, err, at = False, None, None, now
    try:
        path, params = O.event_odds_request(event_id, t["books"], t["markets"])
        params["apiKey"] = key
        try:
            resp = H.vendor_get(O.HOST, path, params, timeout=ODDS_TIMEOUT, secrets=(key,))
        except H.VendorTransportError as e:
            err = (e.kind, e.received)         # its kind and byte count only: never its text
        del params
        at = _as_utc(now_utc() if pinned is None else pinned)
        frame = None
        if resp is not None and resp.status == 200 and isinstance(resp.body, dict):
            frame = O.parse_event_odds(resp.body, books=t["books"], markets=t["markets"], fetched_at=at)
            _fill(frame, t)
        with mem["lock"]:
            _release(mem, t)
            released = True
            point, settled = _undo_point(mem, event_id), False
            try:
                result = _settle_pull(mem, event_id, t, resp, err, frame, at, s)
                settled = True
            finally:
                if not settled:                # R1: the settling raised, the lock still held
                    _undo(mem, point)
                    _settle_broken(mem, event_id, t, resp, err, at, s)
            return result
    finally:
        if not released:
            with mem["lock"]:
                _release(mem, t)
                _settle_broken(mem, event_id, t, resp, err, at, s)


def _undo_point(mem: dict, event_id: str) -> tuple:
    """What a settling may change, copied under mem["lock"] so that one that
    raises half way can be undone (review R1): the budget, the game's entry,
    the streak, the day breaker and the generation. Copies of the dicts are
    enough, since a settling only ever replaces their values (merge_refresh
    copies the stored frame, never changes it). The log marks ("warned")
    are not undone: a line once logged stays logged."""
    entry = mem["games"].get(event_id)
    fields = dict(entry) if isinstance(entry, dict) else None
    return (dict(mem["budget"]), entry, fields, mem["streak"], mem["breaker_day"], mem["generation"])


def _undo(mem: dict, point: tuple) -> None:
    budget, entry, fields, streak, breaker_day, generation = point
    mem["budget"].clear()
    mem["budget"].update(budget)
    if fields is not None:
        entry.clear()
        entry.update(fields)
    mem["streak"], mem["breaker_day"], mem["generation"] = streak, breaker_day, generation


def _fill(frame: pd.DataFrame, t: dict) -> None:
    """The game's id, ISO date and tip on the parsed rows (parse rule 7),
    in place, keeping the parser's attrs for merge_refresh."""
    attrs = dict(frame.attrs)
    frame["game_id"] = t["game_id"]
    frame["game_date"] = t["game_date"]
    frame["tip_utc"] = pd.Timestamp(t["tip"]) if t["tip"] is not None else None
    frame.attrs.update(attrs)


def _outcome(resp, err, frame, usage) -> tuple[str, bool, str, str]:
    """(outcome, billed, kind, code) of a paid call by design 3.5's table:
    "data" (bookmaker entries to merge), "empty" (none: free, not a pull) or
    "fail". Every call that may have been billed counts as billed."""
    if err is not None:
        tkind = err[0]
        kind, code = ("server", tkind) if tkind in ("connect", "transport") else (tkind, "")
        return "fail", _transport_billed(err), kind, code
    if resp.status != 200:
        code = O.error_code(resp.status, resp.snippet)
        return "fail", (usage.last or 0) > 0, O.failure_kind(resp.status, code), code
    if frame is None or frame.attrs.get("bad_payload"):
        return "fail", True, "bad_payload", ""
    if frame.attrs.get("unknown_team"):
        return "fail", True, "bad_payload", "unknown_team"
    if not frame.attrs.get("bookmaker_entries"):
        return "empty", False, "", ""
    return "data", True, "", ""


def _transport_billed(err: tuple) -> bool:
    """3.5: a call with no complete answer may have been billed, unless it
    could not connect and received nothing (DNS, refused)."""
    return not (err[0] == "connect" and err[1] == 0)


def _answered(resp, err) -> bool:
    """Whether the vendor's answer came back (build amendment B22): a
    complete response, whatever its status, or an oversize one. http's
    vendor_get raises "oversize" only from its body callback, so the status
    line and the headers had come and the body was arriving when the client
    stopped it at VENDOR_MAX_BYTES: an answer, assumed billed (3.5), of no
    use, the billed junk the day breaker is for (design 6 lists "a body over
    4 MB" beside an unusable 200; build 6's review, S12-R4). A timeout, a
    failed connection or a download the network cut off (the kinds timeout,
    connect and transport, whatever bytes arrived) is no answer. The cut-off
    download is design 3.5's "connection failure after body bytes arrived",
    http's kind transport (build amendment B31)."""
    return resp is not None or (err is not None and err[0] == "oversize")


_COUNTER_NAMES = ("outcomes_seen", "kept_outcomes", "bad_bookmaker", "bad_market") + O.COUNTERS


def _counters(frame) -> str:
    """The parser's counters for a log line: our names and numbers only."""
    attrs = frame.attrs if isinstance(frame, pd.DataFrame) else {}
    parts = [f"{flag} 1" for flag in ("bad_payload", "unknown_team") if attrs.get(flag)]
    for name in _COUNTER_NAMES:
        n = attrs.get(name)
        if type(n) is int and n:
            parts.append(f"{name} {n}")
    return ", ".join(parts)


def _settle_pull(mem, event_id, t, resp, err, frame, now, s) -> dict:
    """Design 3.5's table for one paid call, under mem["lock"]: the vendor's
    counters, then the game's slot, the streak and the day breaker, the auth
    state, and the generation when the stored frame changed. A clean answer
    ends the streak, an empty one included; a timeout of a call expected to
    be free counts no pull (build 4's review, L2-1), once per clean answer
    for the game (`free_failed`, build 6's review, S12-R1); and only an
    answer that came back (_answered: an oversize one too,
    S12-R4) can grow the streak: a transport failure (a timeout, no
    connection, a download the network cut off) never does (build amendment
    B22)."""
    usage = O.usage_from_headers(resp.headers) if resp is not None else _NO_USAGE
    outcome, billed, kind, code = _outcome(resp, err, frame, usage)
    answered = _answered(resp, err)
    spent = _take_usage(mem, usage, billed=billed, cost=t["cost"], now=now, s=s)
    entry = mem["games"].get(event_id)
    if _slot_of(entry, event_id) is None:      # pruned or replaced while the call ran
        return {"billed": billed, "cost": spent}
    if outcome == "data":
        merged, report = O.merge_refresh(entry["frame"], frame, t["books"],
                                         markets_requested=t["markets"])          # B3
        _note_books(mem, report, now)
        if report["kept_rows"] > 0 and not report["bad_payload"]:
            _pulled(mem, entry, merged, report, s, now)
            return {"billed": True, "cost": spent}
        outcome, kind, code = "fail", "bad_payload", "no_lines_kept"
    if outcome == "empty":
        # A clean answer, if an empty one: the vendor is answering, so a run of
        # billed failures is over (build 4's review, L2-1). The day breaker is
        # for a vendor that bills junk on every call (RP case E). And the game's
        # next call is expected to be free again (S12-R1).
        mem["streak"] = 0
        entry["free_failed"] = False
        if t["kind"] == "pp":
            entry["pp_next"] = now + OB.PP_RETRY            # PrizePicks still has nothing
        else:
            entry["empty_next"] = now + OB.EMPTY_RETRY
        return {"billed": False, "cost": spent}
    # A transport failure of a call expected to be free (t["free"]) is no
    # pull (L2-1): its worst case came off the balance above, but it does not
    # use one of the game's P pulls, so a timed-out top-up never costs the
    # game its PrizePicks lines for the day. Once per clean answer: it sets
    # the entry's `free_failed`, so the game's next call is not free and its
    # failure counts (S12-R1). And no transport failure grows the streak
    # (B22): only an answer that came back can be the billed junk the day
    # breaker is for (RP case E), so a burst of timeouts at 08:50 never
    # stops the day. An oversize answer came back (S12-R4): it is billed
    # junk, a pull and a step of the streak, never L2-1's free timeout.
    _failed(mem, entry, billed, kind, code, now, _counters(frame) or _words(kind, code),
            counted=answered or not t.get("free"), answered=answered)
    return {"billed": billed, "cost": spent}


def _settle_broken(mem, event_id, t, resp, err, now, s) -> None:
    """The task raised after its ticket, before its settling ended (M7,
    review R1): the call counts as a failed one, billed when it may have
    been (3.5's safe side): an answer that came as a 200 or with a non-zero
    x-requests-last, or a transport failure that may have been billed. The
    vendor's counters are taken once, the game backs off and, billed, the
    call counts as a pull; it grows the streak (and so the day breaker) only
    when an answer came back (build amendment B22: a transport failure, or
    a call that raised before any answer, never does; an oversize answer
    came back, _answered, S12-R4)."""
    usage = O.usage_from_headers(resp.headers) if resp is not None else _NO_USAGE
    if resp is not None:
        billed = resp.status == 200 or (usage.last or 0) > 0
    else:
        billed = err is not None and _transport_billed(err)
    _take_usage(mem, usage, billed=billed, cost=t["cost"], now=now, s=s)
    entry = mem["games"].get(event_id)
    if _slot_of(entry, event_id) is not None:
        _failed(mem, entry, billed, "server", "worker", now, "worker", log=False,
                answered=_answered(resp, err))


def _note_books(mem: dict, report: dict, now: dt.datetime) -> None:
    """A book present but unreadable, or under half its lines: its previous
    lines were kept (2.2, M3). Numbers and our book keys only."""
    for book, tally in sorted(report["unreadable"].items()):
        detail = ", ".join(f"{n} {tally[n]}" for n in O.BOOK_COUNTERS if tally.get(n))
        _warn(mem, "unreadable:" + book, now,
              "oddsapi: %s answered 0 of %d lines for a game; kept the previous lines (%s)",
              book, int(tally.get("outcomes_seen") or 0), detail or "nothing readable")
    for book, (new, prev) in sorted(report["suspect"].items()):
        _warn(mem, "suspect:" + book, now,
              "oddsapi: %s answered %d of %d lines for a game; kept the previous lines", book,
              new, prev)


def _books_at(frame: pd.DataFrame) -> dict:
    """Each book's own latest fetched_at in a stored frame (m23)."""
    out = {}
    if frame is None or frame.empty:
        return out
    for book, stamps in frame.groupby("book", sort=False)["fetched_at"]:
        t = _as_utc(stamps.max())
        if t is not None:
            out[str(book)] = t
    return out


def _pulled(mem: dict, entry: dict, merged: pd.DataFrame, report: dict, s, now) -> None:
    """A success (3.5): the merged frame stored, the pull counted, the
    failures and the streak cleared, and PrizePicks' absence marked for the
    free PrizePicks-only polls (3.4, M4)."""
    entry["frame"] = merged
    entry["pulls"] += 1
    entry["first_at"] = entry["first_at"] or now
    entry["last_at"] = now
    entry["fails"], entry["fail_until"], entry["empty_next"] = 0, None, None
    entry["free_failed"] = False                                     # a clean answer (S12-R1)
    mem["streak"] = 0
    missing = O.REFERENCE_BOOK in s.books and not bool((merged["book"] == O.REFERENCE_BOOK).any())
    entry["pp_missing"] = missing
    entry["pp_next"] = now + OB.PP_RETRY if missing else None
    entry["books_at"] = _books_at(merged)
    entry["pulled"] = [tuple(p) for p in report["pulled"]]           # a sample, 8 at most
    entry["n_pulled"] = int(report.get("n_pulled", 0) or 0)          # all of them (L3-10)
    if report["changed"]:
        mem["generation"] += 1


def _failed(mem: dict, entry: dict, billed: bool, kind: str, code: str, now: dt.datetime,
            counters: str, *, log: bool = True, counted: bool = True, answered: bool) -> None:
    """A failed call (3.5): the game backs off (90 s doubling to 60 min; a 404
    ends its pulls for the day). A billed one counts as a pull (unless not
    `counted`: a timeout of a call expected to be free, L2-1, which then
    sets the entry's `free_failed`, so the game's next call is not taken as
    free, and its next first pull is no true first pull, until a clean
    answer, nothing posted or lines that landed, clears it: build 6's
    review, S12-R1; a failure, answered or not, never does). A billed one
    that `answered` (an answer came back and its body was no use, an
    oversize one too: the billed junk RP case E is about) grows the streak,
    and BREAKER_STREAK in a row, any games, stop paid calls until the next
    ET day (B2), with one WARNING. A transport failure (no answer: a
    timeout, no connection, a download the network cut off) still takes its
    worst case off the balance and backs the game off, but never grows the
    streak (build amendment B22). `answered` has no default, so no caller
    leaves it to chance. A 401 sets the auth state (3.8)."""
    entry["fails"] += 1
    until = now + OB.backoff(entry["fails"])
    if kind == "not_found":
        until = max(until, _next_et_midnight(now))
    entry["fail_until"] = until
    if billed:
        if counted:
            entry["pulls"] += 1
            entry["first_at"] = entry["first_at"] or now
            entry["last_at"] = now
        else:
            entry["free_failed"] = True        # L2-1's free failure: once per clean answer (S12-R1)
        if answered:
            mem["streak"] += 1
            if OB.breaker_trips(mem["streak"]) and not OB.breaker_on(mem["breaker_day"], now):
                mem["breaker_day"] = _et_day(now)
                C.logger.warning("oddsapi: %d billed failures in a row; paid calls stopped until "
                                 "tomorrow (parser counters: %s)", mem["streak"], counters)
    if kind in ("out_of_credits", "bad_key"):
        _auth_refused(mem, code, now, events=False)
    elif kind == "not_found":
        _warn(mem, "odds:not_found", now, "oddsapi event odds: %s; the game's pulls end for today",
              code, level=logging.INFO)
    elif log:
        _warn(mem, f"odds:{kind}:{code}", now, "oddsapi event odds failed: %s", _words(kind, code))


# -- The view (2.4.4) ---------------------------------------------------------------------
def feed_view(date, *, mem: dict | None = None, wait_s: float = PAGE_WAIT_S, now=None) -> FeedLive:
    """What memory holds for `date`, as a FeedLive. It never fetches: it
    waits up to `wait_s` (timed by live's own time.monotonic, which tests
    fake) for the running walk and that date's queued or running calls, then
    reads the ledger: the current frames, and after tip the pre-tip copies,
    which nothing ever buys again. `mem` is the ledger (the anchor's when
    None)."""
    if mem is None:
        mem = feed_anchor()
    day = _iso_day(date)
    wait = max(0.0, float(wait_s or 0))
    started = time.monotonic()
    with mem["lock"]:
        walk = mem["walk"]
    if walk is not None and wait > 0:
        cf.wait([walk], timeout=wait)
    left = wait - (time.monotonic() - started)
    with mem["lock"]:
        futs = [f for eid, f in mem["futures"].items()
                if isinstance(mem["games"].get(eid), dict) and mem["games"][eid].get("game_date") == day]
    if futs and left > 0:
        cf.wait(futs, timeout=left)
    at = _as_utc(now_utc() if now is None else now)
    guard = O.read_settings(os.environ).tip_guard
    with mem["lock"]:
        return _assemble(mem, day, at, guard)


def _assemble(mem: dict, day: str | None, now: dt.datetime, guard: dt.timedelta) -> FeedLive:
    """The FeedLive of `day`, under mem["lock"]. Its state: "paused" when
    today's paid calls are stopped (the auth state, the day breaker, a
    paused plan, or a game of today held by a budget guard: HOLDS), with or
    without lines (N1); else "stale" with lines, or "down" without, when a
    call or the events failed; else "ok" with lines, or "idle" without (no
    lines and nothing wrong). Another day's memory is never paused, stale
    or down: nothing is bought for it."""
    work, b = mem["work"], mem["budget"]

    def dated(name):
        return {r["game_id"]: r for r in _work_rows(work, name)
                if isinstance(r.get("game_id"), str) and _iso_day(r.get("game_date")) == day}
    buyable, projected = dated("games"), dated("horizon")
    entries: dict = {}
    for eid, entry in mem["games"].items():
        if _slot_of(entry, eid) is None or entry["game_date"] != day:
            continue
        held = entries.get(entry["game_id"])
        if held is None or (held[1]["frame"] is None and entry["frame"] is not None):
            entries[entry["game_id"]] = (eid, entry)
    today = day == _et_day(now).isoformat()
    # Paid calls stopped (N1): the auth state, the day breaker, a paused plan,
    # or a game of today whose latest paid call a budget guard refused (HOLDS:
    # design 1.3 and 6, "the reserve, floor or ceiling reached"; build 3's
    # review, F4), so a morning re-buy the floor refuses is never read as a
    # board still to fill.
    holding = any(entry.get("held") in HOLDS for _eid, entry in entries.values())
    paused = today and (b["auth"] != "ok" or OB.breaker_on(mem["breaker_day"], now)
                        or b["mode"] == "paused" or holding)
    failed = today and mem["events_fail"] is not None
    frames, per_game, asofs, pulled, n_pulled, loading = [], {}, [], [], 0, 0
    loading_games: list = []
    for gid in dict.fromkeys([*entries, *buyable, *projected]):
        eid, entry = entries.get(gid, (None, None))
        frame = entry["frame"] if entry is not None else None
        has = frame is not None and len(frame) > 0
        tip = entry["tip_utc"] if entry is not None else _work_tip(buyable.get(gid) or projected.get(gid))
        books = dict(entry["books_at"]) if has else {}
        asof = max(books.values()) if books else None
        per_game[gid] = {"asof": asof, "books": books,
                         "pretip": bool(has and tip is not None and now >= tip),
                         "pp_missing": bool(entry is not None and entry["pp_missing"]),
                         "reason": None if has else _reason(entry, gid in buyable, tip, guard,
                                                             paused, mem["born_at"], now)}
        if has:
            frames.append(frame)
            pulled.extend(entry["pulled"])
            # The true count (L3-10); an entry from before it was kept counts its sample.
            n = entry.get("n_pulled")
            n_pulled += max(n if type(n) is int else 0, len(entry["pulled"]))
            if asof is not None:
                asofs.append(asof)
        if entry is not None:
            failed = failed or (today and entry["fails"] > 0)
            if _in_flight(mem, eid):
                loading += 1
                loading_games.append(gid)
    if paused:
        state = "paused"
    elif failed:
        state = "stale" if frames else "down"
    else:
        state = "ok" if frames else "idle"
    ev = mem["events"]
    report = ({k: (list(v) if isinstance(v, list) else v) for k, v in ev[2].items()}
              if ev is not None and _et_day(ev[0]).isoformat() == day else
              {"unknown_team": [], "unmatched_events": 0, "games_no_event": 0, "swapped": 0})
    report.update(pulled=pulled[:8], n_pulled=max(n_pulled, len(pulled)))
    value = pd.concat(frames, ignore_index=True) if frames else O.empty_feed()
    return FeedLive(state=state, value=value, asof=min(asofs) if asofs else None, per_game=per_game,
                    loading=loading, generation=mem["generation"], report=report,
                    loading_games=tuple(loading_games))


def _in_flight(mem: dict, event_id: str) -> bool:
    """Queued or running. A finished task can still sit in the queue for a
    moment: concurrent.futures wakes its waiters before it runs the
    done-callback that takes the game off the queue."""
    if event_id not in mem["queued"]:
        return False
    fut = mem["futures"].get(event_id)
    return fut is None or not fut.done()


def _reason(entry, buyable: bool, tip, guard: dt.timedelta, paused: bool, born: dt.datetime,
            now: dt.datetime) -> str:
    """Why a game of the date has no lines (2.4.4; m22 for lost_in_restart:
    the ledger was born after the game's tip guard and holds no frame).
    "none_posted" only when its books really posted none: its last answer
    had no bookmaker entry (build amendment B18). A game the budget held, or
    any game on a paused day, reads "paused"; one never pulled, or pulled
    with no answer, "not_yet", after its tip guard too, where the Game page
    says the odds feed did not load its lines (B18)."""
    if entry is None and not buyable:
        return "not_published"
    if tip is not None and born >= tip - guard:
        return "lost_in_restart"
    if paused:
        return "paused"
    if entry is not None and entry["empty_next"] is not None:
        return "none_posted"
    return "not_yet"


def prop_lines(work: dict, *, mem: dict | None = None, environ=None, pool=None,
               wait_s: float = PAGE_WAIT_S, now=None) -> FeedLive:
    """feed_kick, then feed_view of the work order's date: the feed in one
    call, for tests with a synchronous pool. With no key, or the switch off,
    FeedLive.off(), and memory is not touched."""
    try:
        env = os.environ if environ is None else environ
        if _read_key(env) is None or not O.read_settings(env).enabled:
            return FeedLive.off()
        if mem is None:
            mem = feed_anchor()
        feed_kick(work, mem=mem, environ=environ, pool=pool, now=now)
        at = _as_utc(now_utc() if now is None else now)
        date = (work.get("date") if isinstance(work, dict) else None) or _et_day(at).isoformat()
        return feed_view(date, mem=mem, wait_s=wait_s, now=now)
    except Exception as e:  # noqa: BLE001 - the type only
        C.logger.warning("oddsapi: feed view failed (%s)", type(e).__name__)
        return FeedLive.down()


def prop_lines_memory(date, *, mem: dict | None = None, now=None) -> FeedLive:
    """What memory holds for any date (today's games, or yesterday's late
    games after midnight ET, since the default view date is now minus 4
    hours). It never calls the vendor and never waits."""
    return feed_view(date, mem=mem, wait_s=0, now=now)


# -- The refresher (2.4.6, M6) ------------------------------------------------------------
# A daemon thread, at most one per process, that runs the page view's walk
# without a view: every REFRESHER_TICK_S it asks refresher_tick whether a game
# of today is due, and kicks the walk when one is. So the 08:50 pulls and the
# planned refreshes happen with or without visitors, and Barrett's 9am view
# normally reads lines already in memory. store.feed_start starts it through
# ensure_refresher, only with the feed on and BUCKETVALUE_ODDS_REFRESHER not
# off; conftest switches it off for every test. The tick never calls st.*,
# never builds the feed tables and never reads Streamlit's caches: only live
# functions run off the script thread.
REFRESHER_THREAD = "oddsapi-refresher"    # the thread's name
ORDER_MAX_AGE = dt.timedelta(hours=36)    # 2.4.5: an older work order is ignored; a view brings a new one
TICK_WORDS = ("off", "no_work", "idle", "kicked")
_IDS_KEEP = 8                             # committed files whose game ids are remembered
_ids_lock = threading.Lock()
_ids_seen: dict = {}                      # path -> ((mtime_ns, size), frozenset of game ids)


def ensure_refresher(environ=None) -> None:
    """Start the refresher (2.4.6): a daemon thread running _refresher_loop,
    at most one per process. Only with the feed on (a key, and the kill
    switch on) and BUCKETVALUE_ODDS_REFRESHER on, all read on this call;
    otherwise it returns before it touches the anchor, so with no key nothing
    changes. The thread, its stop Event and its version live in the anchor
    (stdlib objects and an int, B4). A live loop of this REFRESHER_VERSION is
    left alone; a dead one is restarted; one of another version has its stop
    Event set, so it ends, and a new loop takes over. It never raises."""
    try:
        env = os.environ if environ is None else environ
        s = O.read_settings(env)
        if not (s.enabled and s.refresher) or _read_key(env) is None:
            return
        mod = _anchor_module()
        with mod.lock:
            thread = getattr(mod, "refresher", None)
            version = getattr(mod, "refresher_version", None)
            if (isinstance(thread, threading.Thread) and thread.is_alive()
                    and version == REFRESHER_VERSION):
                return                                  # at most one per process
            old = getattr(mod, "refresher_stop", None)
            if isinstance(old, threading.Event):
                old.set()                               # an old loop ends at its next wake
            if not isinstance(thread, threading.Thread):
                how, level = "started", logging.INFO
            elif version != REFRESHER_VERSION:
                how, level = "replaced (version change)", logging.INFO
            else:
                how, level = "restarted (the last loop had stopped)", logging.WARNING
            stop = threading.Event()
            fresh = threading.Thread(target=_refresher_loop, args=(stop, REFRESHER_VERSION),
                                     name=REFRESHER_THREAD, daemon=True)
            mod.refresher, mod.refresher_stop, mod.refresher_version = fresh, stop, REFRESHER_VERSION
            fresh.start()
        C.logger.log(level, "oddsapi: refresher %s", how)
    except Exception as e:  # noqa: BLE001 - the type only: never its text
        C.logger.warning("oddsapi: refresher start failed (%s)", type(e).__name__)


def _refresher_loop(stop: threading.Event, version: int) -> None:
    """The refresher's loop (2.4.6), deliberately tiny: after a code push it
    is OLD code. So on every tick it looks up the CURRENT bucketlib.live in
    sys.modules, and it returns once the anchor names another version (a
    newer loop took over) or is gone. A failed tick logs its type only."""
    while not stop.wait(REFRESHER_TICK_S):
        anchor = sys.modules.get(ANCHOR_NAME)
        mod = sys.modules.get("bucketlib.live")    # the CURRENT module, after any purge
        if anchor is None or getattr(anchor, "refresher_version", None) != version:
            return                                 # a newer loop took over
        if mod is None:
            continue                               # purged, not re-imported yet: next tick
        try:
            mod.refresher_tick(mod.feed_anchor())
        except Exception as e:                     # noqa: BLE001
            logging.getLogger("bucketvalue").warning(
                "oddsapi: refresher tick failed (%s)", type(e).__name__)


def refresher_tick(mem: dict, *, environ=None, pool=None, now=None) -> str:
    """One tick of the refresher (2.4.6): one of TICK_WORDS.

    The key and the settings are read on this call (RC K3, G1): no key, the
    kill switch off or BUCKETVALUE_ODDS_REFRESHER off gives "off". Then the
    work order (2.4.5, _tick_order): "no_work" when there is none, it is
    older than ORDER_MAX_AGE, or it holds no buyable game of today (an
    order dated today with no game of today is read like one from the
    evening before: its horizon rows and today's serve_files, B15). The tick
    kicks the walk (feed_kick, as a page view does) when a game of today is
    past its first-pull time, before its tip guard, not already queued, and
    either the balance is not known yet (a new ledger: the walk's free
    events call learns it) or oddsbudget.due_kind says it is due: "kicked".
    Otherwise "idle", and an idle tick makes no network call: the free
    events call happens only inside a walk. `mem` is the ledger; `pool` the
    walk's pool (the anchor's when None) and `now` the walk's clock (live's
    own now_utc when None), both handed to feed_kick."""
    env = os.environ if environ is None else environ
    s = O.read_settings(env)
    if not (s.enabled and s.refresher) or _read_key(env) is None:
        return "off"
    t = _as_utc(now_utc() if now is None else now)         # live's own binding (T1)
    if t is None:
        raise ValueError("now must be an aware instant")
    today = _et_day(t).isoformat()
    with mem["lock"]:
        work = mem["work"]
    order, games = _tick_order(mem, work, today, t)
    if not games:
        return "no_work"
    with mem["lock"]:
        due = _tick_due(mem, games, s, t, today)
    if not due:
        return "idle"
    feed_kick(order, mem=mem, environ=environ, pool=pool, now=now)
    return "kicked"


def _dated(rows: list, day: str) -> list[dict]:
    return [r for r in rows if isinstance(r.get("game_id"), str) and r["game_id"]
            and _iso_day(r.get("game_date")) == day]


def _tick_order(mem: dict, work, today: str, now: dt.datetime) -> tuple[dict | None, list]:
    """The order the tick kicks with, and its buyable games of `today`
    (design 2.4.5). When the order's date is today and it holds games of
    today: the order and those games. Otherwise (an order from the evening
    before, or today's order written before the bot published today's
    files, which holds no game of today: build amendment B15) the horizon
    entries dated today whose game ids are in the first of today's
    serve_files that exists (the date's projections file, else its features
    file: m26), so the refresher buys at 08:50 with nobody visiting. They go
    into a copy of the order, beside its games of other dates, so the walk
    finds them; the order's date and written_at stay its own. An entry
    without both teams cannot be matched to a vendor event and is left out,
    logged once a day. (None, []) when there is no order, its written_at is
    unreadable or older than ORDER_MAX_AGE (the refresher waits for a view),
    or no game of today is buyable in it."""
    if not isinstance(work, dict):
        return None, []
    written = _as_utc(work.get("written_at"))
    if written is None or now - written > ORDER_MAX_AGE:
        return None, []
    if _iso_day(work.get("date")) == today:
        games = _dated(_work_rows(work, "games"), today)
        if games:
            return work, games
    rows = _dated(_work_rows(work, "horizon"), today)
    if not rows:
        return None, []
    files = work.get("serve_files")
    paths = files.get(today) if isinstance(files, dict) else None
    ids, problem = _published_ids(paths if isinstance(paths, list) else [])
    if problem:
        with mem["lock"]:
            _warn(mem, "serve_file:" + today, now, "oddsapi: today's published file is unreadable "
                  "(%s); the refresher waits for a view", problem, every=_ONCE_A_DAY)
    if not ids:
        return None, []
    kept = [dict(r) for r in rows if r["game_id"] in ids]
    teamless = sum(1 for r in kept if not (r.get("away") and r.get("home")))
    if teamless:
        with mem["lock"]:
            _warn(mem, "horizon_teams:" + today, now, "oddsapi: %d game(s) of today have no teams in "
                  "the work order's horizon; the refresher waits for a view", teamless,
                  every=_ONCE_A_DAY)
    kept = [r for r in kept if r.get("away") and r.get("home")]
    if not kept:
        return None, []
    others = [dict(r) for r in _work_rows(work, "games") if _iso_day(r.get("game_date")) != today]
    return {**work, "games": others + kept}, kept


def _tick_due(mem: dict, games: list, s, now: dt.datetime, today: str) -> bool:
    """Under mem["lock"]: is one of today's `games` past its first-pull time,
    before its tip guard (the ledger's tip once matched, else the order's)
    and not queued, with the balance unknown or a due kind (2.4.6)? A game
    the ledger has not matched yet counts as never pulled, with no lines."""
    if now < OB.first_pull_time(today, s.first_pull_et):
        return False
    b = mem["budget"]
    unknown = b["remaining"] is None or b["used"] is None
    plan = None if unknown else _budget(mem, s, now).plan
    held: dict = {}
    for eid, entry in mem["games"].items():
        slot = _slot_of(entry, eid)
        if slot is not None:
            held.setdefault(slot.game_id, []).append(slot)
    for r in games:
        gid = r["game_id"]
        for slot in held.get(gid) or [OB.Slot(event_id="", game_id=gid, tip_utc=_work_tip(r),
                                              has_lines=False)]:
            if slot.event_id in mem["queued"] or not OB.before_tip(now, slot.tip_utc, s.tip_guard):
                continue
            if unknown or OB.due_kind(slot, plan, s, now, today,
                                      breaker_day=mem["breaker_day"]) is not None:
                return True
    return False


def _read_game_ids(path: str) -> frozenset:
    """The game ids of one committed serve file: its game_id column only."""
    df = pd.read_parquet(path, columns=["game_id"])
    return frozenset(str(g) for g in df["game_id"].dropna())


def _published_ids(paths: list) -> tuple[frozenset | None, str]:
    """The game ids of the first of `paths` that exists (2.4.5: the date's
    projections file, else its features file), remembered on the file's
    (path, mtime_ns, size), so an unchanged file is read once. (None, "")
    when none exists; (None, <the exception's type>) when the first that
    exists cannot be read (a folder or a FIFO counts as unreadable, and is
    never opened)."""
    for p in paths:
        if not isinstance(p, str) or not p:
            continue
        try:
            info = os.stat(p)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            return None, _NOT_REGULAR
        version = (info.st_mtime_ns, info.st_size)
        with _ids_lock:
            hit = _ids_seen.get(p)
        if hit is not None and hit[0] == version:
            return hit[1], ""
        try:
            ids = _read_game_ids(p)
        except Exception as e:  # noqa: BLE001 - the type only
            return None, type(e).__name__
        with _ids_lock:
            _ids_seen.pop(p, None)
            _ids_seen[p] = (version, ids)
            while len(_ids_seen) > _IDS_KEEP:
                _ids_seen.pop(next(iter(_ids_seen)))
        return ids, ""
    return None, ""
