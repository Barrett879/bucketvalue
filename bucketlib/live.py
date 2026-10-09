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
    the page renders an outage notice instead of an empty slate.
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
from .sources import cdn, espn
from .sources import injury as INJ
from .util import now_et, now_utc, to_et

TIMEOUT = 8.0


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


def _cached(key: str, ttl: float, fail_ttl: float, fn: Callable[[], Any],
            committed: Callable[[], tuple[Any, float | None]] | None = None) -> Live:
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
        except (H.FetchError, ValueError, KeyError) as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
            with _lock:
                _fail[key] = (now, err)
    if good:
        return Live(good[1], good[0], stale=True, source="memory", error=err)
    if committed is not None:
        val, ts = committed()
        if val is not None:
            return Live(val, ts, stale=True, source="committed", error=err)
    raise RuntimeError(f"{key}: live fetch failed and nothing is cached ({err})")


def _committed_parquet(name: str):
    """The bot's committed copy, state/current/{name}_v1.parquet. The path is
    resolved through bucketlib.state at call time (state owns every path
    under state/, and a test that points STATE_DIR elsewhere moves it too)."""
    def load():
        p = S.current_path(name)
        df = C.read_parquet(p)
        return (df, C.read_parquet_meta(p).get("fetched_at")) if df is not None else (None, None)
    return load


# ── Schedule and rosters (cdn.nba.com) ───────────────────────────────────────
def schedule() -> Live:
    """The current season's schedule. TTL 10 minutes: game status and scores
    change through the night, and the post-Cup games get added mid-season."""
    return _cached("schedule", 600, 90,
                   lambda: cdn.parse_schedule(H.fetch(cdn.SCHEDULE_URL, timeout=TIMEOUT, retries=0)),
                   _committed_parquet("schedule"))


def roster() -> Live:
    """Current rostered players with the two-way flag. TTL 1 hour."""
    return _cached("roster", 3600, 120,
                   lambda: cdn.parse_player_index(H.fetch(cdn.PLAYER_INDEX_URL, timeout=TIMEOUT, retries=0)),
                   _committed_parquet("roster"))


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


def game_state(row, now=None) -> str:
    """'final' | 'live' | 'upcoming' from the schedule row and the clock."""
    st = row.get("status")
    if st == "final":
        return "final"
    if st == "live":
        return "live"
    t = now or now_et()
    return "live" if to_et(row["tip_utc"]) <= t else "upcoming"


# ── Lines (ESPN scoreboard, DraftKings) ──────────────────────────────────────
def odds_on(date) -> Live:
    """{game key (away, home): odds dict} for one ET date. TTL 5 minutes.

    Lines post gradually (a missing line is "not posted yet", not "no game")
    and VANISH from the scoreboard at tip, so a started game shows whatever
    this process last saw; the bot persists pre-tip snapshots separately.
    """
    def fetch():
        payload = H.fetch(espn.SITE + "scoreboard",
                          params={"dates": date.strftime("%Y%m%d"), "limit": "500"},
                          timeout=TIMEOUT, retries=0)
        df = espn.parse_scoreboard(payload, now_utc())
        out = {}
        for _, r in df.iterrows():
            if pd.notna(r.get("home_line")) or pd.notna(r.get("total")):
                out[(r["away"], r["home"])] = {
                    "home_line": r.get("home_line"), "total": r.get("total"),
                    "open_home_line": r.get("open_home_line"), "open_total": r.get("open_total"),
                    "provider": r.get("provider_name"), "captured_at": r.get("captured_at"),
                }
        return out
    key = f"odds:{date.isoformat()}"
    try:
        return _cached(key, 300, 90, fetch)
    except RuntimeError as e:
        return Live({}, None, stale=True, source="none", error=str(e))


# ── Injury report (ak-static PDFs) ───────────────────────────────────────────
INJURY_KEY = "injury"
INJURY_TTL, INJURY_FAIL_TTL = 600, 120
INJURY_SLOTS = 8          # floor(now ET, 15 min) and at most 8 slot times back (spec 4)
INJURY_TIMEOUT = 3.0      # per PDF, no retries
INJURY_BUDGET = 5.0       # seconds for the whole walk


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
    committed projections, which carry the bot's context).
    """
    if sched is None:
        sched = schedule().value
    accept = HZ.real_report(sched)

    def fetch():
        store, end = INJ.live_store(timeout=INJURY_TIMEOUT), time.monotonic() + INJURY_BUDGET

        def getter(slot):
            # No new fetch once the budget is spent: a slow host costs one
            # budget per TTL, not eight timeouts.
            if time.monotonic() > end:
                raise H.FetchError(slot.url, f"injury walk over its {INJURY_BUDGET:.0f}s budget")
            return store.get(slot)
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
    return _cached(INJURY_KEY, INJURY_TTL, INJURY_FAIL_TTL, fetch, committed)
