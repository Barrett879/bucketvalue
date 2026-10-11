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
  - Nothing here touches disk, so READ_ONLY deployments stay clean: the
    checkout is shared by every visitor and owned by the bot.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import cache as C
from . import context as CTX
from . import horizons as HZ
from . import http as H
from . import state as S
from .sources import box as BX
from .sources import cdn, cdnbox, espn
from .sources import injury as INJ
from .util import now_et, now_utc, to_et

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


# ── Schedule and rosters (cdn.nba.com) ───────────────────────────────────────
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


# ── Lines (ESPN scoreboard, DraftKings) ──────────────────────────────────────
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


# ── Box scores after tip (cdn.nba.com liveData) ──────────────────────────────
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


# ── Injury report (ak-static PDFs) ───────────────────────────────────────────
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
