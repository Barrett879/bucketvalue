"""Small shared helpers: Eastern-time dates, NBA season labels, safe display.

THE DATE RULE, and the reason this module exists before anything else:
every "today" on this site is the NBA's game date, which is the US/Eastern
calendar date of tip-off. The bot (GitHub Actions) and the host (Streamlit
Community Cloud) both run in UTC, and a 7pm ET tip is midnight UTC from
November to mid-March (and an 8pm ET tip is past UTC midnight even under
daylight time). DiamondValue's `dt.date.today()` silently moved its evening
refresh onto the wrong date for exactly this reason. Nothing in this repo may
call `date.today()`, `datetime.now()` without a zone, or `time.localtime()`;
tests/test_util.py greps for it.
"""
from __future__ import annotations

import datetime as dt
import html
import re
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# The one em dash allowed in UI text: a missing value. Everything else on the
# site avoids em dashes (house style).
SENTINEL = "—"


# ── Clocks ───────────────────────────────────────────────────────────────────
def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def today_et() -> dt.date:
    """The NBA game date right now. Use this, never date.today()."""
    return now_et().date()


def to_et(ts) -> dt.datetime:
    """Any aware datetime or ISO string -> aware ET datetime.

    A naive input is REJECTED rather than assumed: guessing the zone of a
    timestamp is how a 7:30pm tip becomes the next day's game.
    """
    if isinstance(ts, str):
        s = ts.strip().replace("Z", "+00:00")
        ts = dt.datetime.fromisoformat(s)
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime {ts!r}: pass an aware timestamp")
    return ts.astimezone(ET)


def et_date_of(ts) -> dt.date:
    """ET calendar date of an aware timestamp (ESPN event dates are UTC)."""
    return to_et(ts).date()


def iso_date(d: dt.date | str) -> str:
    if isinstance(d, str):
        return dt.date.fromisoformat(d).isoformat()
    return d.isoformat()


# ── Seasons ──────────────────────────────────────────────────────────────────
# An NBA season is labelled by both calendar years ("2026-27"). Training camp
# opens late September and the Finals end in June; the offseason belongs to the
# season just finished until August, when the next one becomes "current" for
# roster purposes. August 1 is the cutover.
_SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")


def season_for_date(d: dt.date | str) -> str:
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    start = d.year if d.month >= 8 else d.year - 1
    return season_str(start + 1)


def season_str(end_year: int) -> str:
    """2027 -> '2026-27'."""
    return f"{end_year - 1}-{end_year % 100:02d}"


def season_end_year(season: str) -> int:
    """'2026-27' -> 2027. Raises on anything that is not a season label."""
    m = _SEASON_RE.match(season or "")
    if not m:
        raise ValueError(f"not a season label: {season!r}")
    start = int(m.group(1))
    if (start + 1) % 100 != int(m.group(2)):
        raise ValueError(f"inconsistent season label: {season!r}")
    return start + 1


def season_start_year(season: str) -> int:
    return season_end_year(season) - 1


def seasons_between(first: str, last: str) -> list[str]:
    a, b = season_end_year(first), season_end_year(last)
    return [season_str(y) for y in range(a, b + 1)]


# ── Display ──────────────────────────────────────────────────────────────────
def esc(value) -> str:
    """HTML-escape a value for interpolation into a rendered fragment.

    Every string that reaches an unsafe_allow_html block goes through this,
    including query params and third-party feed text. GridironValue shipped a
    reflected injection because one page skipped it.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _isnan(v) -> bool:
    try:
        return v != v  # NaN is the only value not equal to itself
    except Exception:  # noqa: BLE001
        return False


def fmt_num(value, places: int = 1) -> str:
    if value is None or _isnan(value):
        return SENTINEL
    return f"{float(value):.{places}f}"


def fmt_signed(value, places: int = 1) -> str:
    """'+3.5' / '-2.5'. For spreads, where the sign carries the meaning."""
    if value is None or _isnan(value):
        return SENTINEL
    return f"{float(value):+.{places}f}"


def fmt_tip(ts) -> str:
    """'7:30 PM ET' from an aware timestamp; the sentinel when unknown."""
    if ts is None or _isnan(ts):
        return SENTINEL
    t = to_et(ts)
    hour = t.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{t.strftime('%M %p')} ET"
