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
from . import http as H
from .sources import cdn
from .util import now_et, to_et

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
    def load():
        p = C.CACHE_DIR / name
        df = C.read_parquet(p)
        return (df, C.read_parquet_meta(p).get("fetched_at")) if df is not None else (None, None)
    return load


# ── Schedule and rosters (cdn.nba.com) ───────────────────────────────────────
def schedule() -> Live:
    """The current season's schedule. TTL 10 minutes: game status and scores
    change through the night, and the post-Cup games get added mid-season."""
    return _cached("schedule", 600, 90,
                   lambda: cdn.parse_schedule(H.fetch(cdn.SCHEDULE_URL, timeout=TIMEOUT, retries=0)),
                   _committed_parquet("schedule_current_v1.parquet"))


def roster() -> Live:
    """Current rostered players with the two-way flag. TTL 1 hour."""
    return _cached("roster", 3600, 120,
                   lambda: cdn.parse_player_index(H.fetch(cdn.PLAYER_INDEX_URL, timeout=TIMEOUT, retries=0)),
                   _committed_parquet("roster_current_v1.parquet"))


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
