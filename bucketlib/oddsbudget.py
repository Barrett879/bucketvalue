"""The credit governor of the licensed odds feed (The Odds API): how many
pulls each game may have in the credit period, when each one is due, and
whether a paid call may spend (design section 3; this is the surface of
design 2.3).

PURE. No I/O, no clock, no process environment and no network: every function
takes `now` and plain values, so every rule is tested on literal numbers
(tests/test_oddsbudget.py) and no test can be a time bomb. bucketlib.live
reads the vendor's counters off each response, keeps the ledger in the feed
anchor, and asks this module what to do. Nothing of this module's (no class
instance, no function) is ever stored in the anchor: only plain values.

What the governor reads (3.1):
  - `remaining` and `used`: the vendor's own counters (x-requests-remaining,
    x-requests-used) on every response, the free /events call included, so
    the balance is known before any credit is spent. Unknown means no paid
    call at all. The allowance A = used + remaining is the plan's size, read
    from the headers, so an upgrade is seen without a setting. `used` is the
    account's spend since its last reset: the one durable record of spending,
    which survives every restart of the app.
  - The games, with each game's ET date and tip (None while the tip is
    TBD), in two roles. The PERIOD'S games are every projected game from the
    period's opening day (the reset window's first day, the last day of the
    month before) to the horizon end, tipped ones too: the work order's
    `horizon` of 2.4.5. They give the counts of the period: the spending
    ceiling's n_period and n_through, the `used` test's n and the brake's
    earlier games. G, the games still to plan, is only their untipped part
    (3.3). A list that starts at `now`, the horizon as design 3.1 words it,
    would shrink those counts to what is left and stop the feed buying by
    about the 3rd of each month (review I1, F5), so plan() takes the
    period's games as their own required argument, `period_games`.
  - The reserve S: BUCKETVALUE_ODDS_RESERVE, else max(25, ceil(5% of A)).
    It is never planned and never crossed.

The credit period (3.2, B3, M2). The vendor resets credits "on the first of
every month" for every plan (UT/faqs.txt:53-55) at an instant it does not
state, and it is in Sydney. So a reset falls in the window W(m): from 05:00 ET
on the last day of month m-1 (midnight on the 1st at UTC+14) to 08:00 ET on
the 1st (after midnight on the 1st at UTC-12). Outside every window the period
is the calendar's; inside one, `used` decides. The horizon ends at the close
of the next window. There is no reset-day setting (M2).

The rules: one share per GAME over the period (3.3, m9, m10); when the pulls
happen (3.4); what counts as a pull, and billed failures (3.5, B2); and the
guards on every paid call, in order (3.6). The bounds on spending (the size
of the per-game cap, the morning-pull floor and the spending ceiling) read
only the vendor's counters, so they hold across restarts and crash loops
(M1), and a second process's refreshes, top-ups and re-buys meet the floor
and the ceiling too. The floor guards refreshes, top-ups and re-buys, never a
first pull (build amendment B1): today's games are bought first, in tip
order, and in a shortage the month's LAST games go without lines until the
reset, never today's. A ledger that has not seen a game's whole morning (born
after its first-pull time, in a period that was already spending: a restart
before the brake's window, 3.4) cannot tell that game's first pull from a
re-buy of what the process before it bought, so first_kind calls the pull a
"rebuy" and the floor guards it as design 3.6 wrote it (build amendment
B11): a morning crash loop is bounded by the floor again. A game whose pulls
so far were all billed failures is due again as soon as its backoff allows,
not at its spaced refresh (build amendment B12). A game the memory-loss brake
counted as pulled, for which the ledger holds no lines, keeps ONE re-buy at
once while P >= 2 and before its last refresh time, so a cold start after the
brake's window still shows today's lines (build amendment B21): a "rebuy",
which the floor and the ceiling guard. A TRUE first pull, by a ledger born at
or before the game's first-pull time (it can never re-buy what an EARLIER
process bought), skips the spending ceiling as every first pull skips the
floor (build amendment B23), so an outside drain early in the month never
holds today's morning lines while credits stay above the reserve. No ledger
sees what another process running at the same time bought, so a second
process whose ledger was born at or before that time (a local run started
before 08:50, or left open overnight) buys the game again, bounded only by
the reserve (build 7's review, R7-1; docs/runbook.md section 8).

Every instant is an AWARE datetime; a naive one is refused, never guessed
(util.to_et's rule). Results are in UTC.
"""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from .util import ET, UTC

# -- Constants (design 2.3) ------------------------------------------------------
MAX_PULLS_PER_GAME = 48                    # the hard cap on P (3.3)
CEILING_FACTOR = 1.5                       # B3 item 3, made durable (3.6)
BRAKE = dt.timedelta(minutes=60)           # after the day's first-pull time (3.4)
EMPTY_RETRY = dt.timedelta(minutes=10)     # an empty answer is free; retry no sooner
PP_RETRY = EMPTY_RETRY                     # the PrizePicks-only poll, every 10 minutes (3.4, M4)
BACKOFF_FIRST = dt.timedelta(seconds=90)   # then doubling (B2 item 3)
BACKOFF_MAX = dt.timedelta(minutes=60)
BREAKER_STREAK = 3                         # billed failures in a row, any games (B2 item 4)
WINDOW_OPEN_ET = dt.time(5, 0)             # on the last day of a month: midnight on the 1st at UTC+14
WINDOW_CLOSE_ET = dt.time(8, 0)            # on the 1st: after midnight on the 1st at UTC-12
RESERVE_MIN, RESERVE_DIVISOR = 25, 20      # S = max(25, ceil(A / 20)), 5% of the allowance (3.1)
MARGIN_MIN, MARGIN_DIVISOR = 50, 100       # MARGIN = max(50, ceil(A / 100)), 1% (3.2)
RATE_WINDOW = dt.timedelta(minutes=10)     # the window of BUCKETVALUE_ODDS_MAX_CALLS_10MIN (3.6)

MODES = ("full", "core", "short", "paused", "unknown")
SPENDING_MODES = frozenset({"full", "core", "short"})
PLAN_REASONS = ("ok", "below_full", "below_core", "at_reserve", "no_balance",
                "out_of_credits", "bad_key", "breaker")
AUTH_STATES = ("ok", "out_of_credits", "bad_key")      # from the last 401 (3.8)
BASES = ("calendar", "used_fresh", "used_old")
KINDS = ("first", "refresh", "pp")                     # due_kind's words; "pp": the PrizePicks-only top-up (3.4)
REBUY = "rebuy"                                        # first_kind's word for a first pull that may re-buy (B11),
                                                       # and due_kind's for a braked game's one re-buy (B21)
SPEND_KINDS = KINDS + (REBUY,)                         # the kinds may_spend reads
# What may_spend answers when a call may not spend, in the order of 3.6.
REFUSALS = ("tip_guard", "out_of_credits", "bad_key", "no_balance", "paused",
            "reserve", "floor", "ceiling", "rate_cap")


# -- The values (design 2.3) ---------------------------------------------------------
@dataclass(frozen=True)
class Game:
    """One projected game, from the work order (2.4.5)."""
    game_id: str
    game_date: dt.date                     # the game's ET date
    tip_utc: dt.datetime | None            # None while the time is TBD


@dataclass(frozen=True)
class Period:
    """The credit period a plan spends from (3.2)."""
    month: tuple[int, int]                 # the calendar month the period is named for
    opens: dt.datetime                     # the earliest instant its reset could have happened
    horizon_end: dt.datetime               # the close of the next reset window
    basis: str                             # "calendar" | "used_fresh" | "used_old"


@dataclass(frozen=True)
class Plan:
    """One walk's plan (3.3)."""
    mode: str            # "full" | "core" | "short" | "paused" | "unknown"
    markets: tuple       # what to request in this mode
    per_pull: int        # worst-case credits for one game pull: len(markets) x 1 region
    pulls_per_game: int  # P, capped
    p_max: int           # the per-game cap (3.3)
    reserve: int         # S
    ceiling: int         # the spending ceiling for today (3.6)
    reason: str          # a fixed code of PLAN_REASONS


@dataclass(frozen=True)
class Slot:
    """One game, from the ledger. `pulls` counts every call that may have
    been billed (3.5), the brake's assumed pull included (3.4). `has_lines`
    says the game has stored lines: live sets it from whether the game has a
    stored frame (build amendment B12). It is required, keyword only and a
    bool: there is no default, so a Slot built without it fails at once
    instead of reading as either answer. `braked` says the memory-loss brake
    counted the game's first pull at the ledger's birth (3.4): that pull was
    assumed, never bought by this ledger, so due_kind keeps the game one
    re-buy (build amendment B21). Keyword only and a bool; its default,
    False, is the brake's own behaviour before B21, so a Slot that does not
    say it was braked never buys more."""
    event_id: str
    game_id: str
    tip_utc: dt.datetime | None            # tip_for(...): None while unknown
    pulls: int = 0
    first_at: dt.datetime | None = None
    last_at: dt.datetime | None = None
    empty_next: dt.datetime | None = None  # an empty answer: not before this
    pp_missing: bool = False               # the last pull had no PrizePicks line
    pp_next: dt.datetime | None = None     # the next PrizePicks-only poll
    fail_until: dt.datetime | None = None  # B2 item 2: read by due_kind
    fails: int = 0                         # failures in a row on this game
    has_lines: bool = field(kw_only=True)  # B12: the game has a stored frame with lines
    braked: bool = field(default=False, kw_only=True)   # B21: its first pull is the brake's

    def __post_init__(self):
        if type(self.has_lines) is not bool:
            raise TypeError("Slot.has_lines must be True or False (build amendment B12)")
        if type(self.braked) is not bool:
            raise TypeError("Slot.braked must be True or False (build amendment B21)")


# -- Instants and days ---------------------------------------------------------------
def _utc(value) -> dt.datetime | None:
    """An aware instant in UTC; None for None or NaT; an ISO string is read.
    A naive datetime raises: guessing its zone is how a 7:30 PM tip becomes
    the next day's game."""
    if value is None:
        return None
    if type(value) is dt.datetime and value.tzinfo is UTC:
        return value                        # the common case, already in UTC
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if not isinstance(value, dt.datetime):
        raise TypeError("an instant must be an aware datetime")
    if value != value:                      # NaT is the only one not equal to itself
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("naive datetime: pass an aware instant")
    return value.astimezone(UTC)


def _day(value) -> dt.date | None:
    """An ET calendar date from a date, an ISO date string or an aware instant."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        t = _utc(value)
        return None if t is None else t.astimezone(ET).date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value.strip()[:10])
    raise TypeError("a day must be a date")


def _et_day(now) -> dt.date:
    return _utc(now).astimezone(ET).date()


def _at_et(day: dt.date, clock: dt.time) -> dt.datetime:
    """`clock` ET on `day`, as a UTC instant."""
    return dt.datetime.combine(day, clock, tzinfo=ET).astimezone(UTC)


def _flag(value) -> bool:
    """A schedule boolean that may be missing (None, NaN) in an older frame."""
    try:
        return bool(value) and value == value
    except (TypeError, ValueError):
        return False


def _next_month(month: tuple[int, int]) -> tuple[int, int]:
    y, m = month
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _prev_month(month: tuple[int, int]) -> tuple[int, int]:
    y, m = month
    return (y - 1, 12) if m == 1 else (y, m - 1)


def _unique(games) -> list:
    """Each game once (by game_id, the first entry kept)."""
    seen, out = set(), []
    for g in games:
        if g.game_id not in seen:
            seen.add(g.game_id)
            out.append(g)
    return out


def _slot_list(slots) -> list:
    """The ledger's slots, given as a mapping (any key) or an iterable."""
    if slots is None:
        return []
    return list(slots.values()) if isinstance(slots, Mapping) else list(slots)


def _pulls_by_game(slots) -> dict:
    out: dict = {}
    for s in _slot_list(slots):
        out[s.game_id] = out.get(s.game_id, 0) + max(0, int(s.pulls))
    return out


# -- The credit period and the horizon (3.2) -------------------------------------
def reset_window(year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
    """W(m), the instants a reset for the period that begins in month m could
    happen: (05:00 ET on the last day of month m-1, 08:00 ET on day 1 of m),
    in UTC. It covers midnight on the 1st in every zone from UTC-12 to
    UTC+14; the real instant is UNVERIFIED (RF part 4, fact 11)."""
    first = dt.date(year, month, 1)
    return _at_et(first - dt.timedelta(days=1), WINDOW_OPEN_ET), _at_et(first, WINDOW_CLOSE_ET)


def in_reset_window(now) -> bool:
    """`now` falls inside a reset window W(m). A reset seen OUTSIDE every
    window (reset_seen, which a paid plan resetting on its sign-up day would
    cause) is what live logs once: the period rule then needs a change (3.2)."""
    t = _utc(now)
    et_now = t.astimezone(ET)
    this = (et_now.year, et_now.month)
    return any(lo <= t < hi for lo, hi in (reset_window(*this), reset_window(*_next_month(this))))


def _period(month: tuple[int, int], basis: str) -> Period:
    return Period(month=month, opens=reset_window(*month)[0],
                  horizon_end=reset_window(*_next_month(month))[1], basis=basis)


def margin(allowance: int) -> int:
    """MARGIN of the `used` test: max(50, ceil(1% of the allowance)). It covers
    use from outside the site; the probe spends at most 16 (3.2)."""
    return max(MARGIN_MIN, -(-int(allowance) // MARGIN_DIVISOR))


def default_reserve(allowance: int) -> int:
    """max(25, ceil(5% of the allowance)): 25 on free, 1,000 on $30 (20K),
    5,000 on $59 (100K) (3.1)."""
    return max(RESERVE_MIN, -(-int(allowance) // RESERVE_DIVISOR))


def _reserve(setting, allowance) -> int:
    if setting is not None:
        return int(setting)
    return default_reserve(allowance) if allowance is not None else RESERVE_MIN


def busiest_month(games) -> int:
    """The most projected games in one calendar month (by ET date): the
    season's busiest month when `games` is the season (January 2027: 236)."""
    counts: dict = {}
    for g in _unique(games):
        d = _day(g.game_date)
        counts[(d.year, d.month)] = counts.get((d.year, d.month), 0) + 1
    return max(counts.values(), default=0)


def p_max(allowance: int, reserve: int, c: int, busiest: int) -> int:
    """P_max(c) = min(48, 2 x max(1, floor((A - S) / (c x busiest)))): twice the
    normal share, so no planning error, horizon mistake or lost ledger puts
    more pulls than this on one game (3.3). Free with points: 4; $30 with 12
    markets: 12; $59 with 12 markets: 48."""
    per = max(1, int(c)) * max(1, int(busiest))
    return min(MAX_PULLS_PER_GAME, 2 * max(1, (int(allowance) - int(reserve)) // per))


def period_of(now, used, allowance, reserve, c_max, period_games, *, busiest=None) -> Period:
    """The credit period at `now` (3.2).

    Outside every window: the month whose window last closed ("calendar"); no
    `used` test is needed, since by 08:00 ET on the 1st the 1st has begun
    everywhere. Inside W(m) the reset may or may not have happened, and the
    period is m ("used_fresh") when

        used <= c_max x P_max(c_max) x n + MARGIN

    else m-1 ("used_old"), where n counts the games dated from the window's
    first day through today (ET). An unknown `used` keeps m-1: no paid call
    is made without a balance anyway (3.1).

    `period_games` are the projected games from the window's first day on,
    TIPPED ONES TOO (the work order's horizon, 2.4.5): n counts games that
    may have tipped already, so a list starting at `now` undercounts it.
    `reserve` None means default_reserve(allowance), and `busiest` None
    means busiest_month(period_games); live passes the work order's
    `busiest`, the season's busiest month, as RP does."""
    t = _utc(now)
    if t is None:
        raise ValueError("now is required")
    et_now = t.astimezone(ET)
    this = (et_now.year, et_now.month)
    cand = _next_month(this) if t >= reset_window(*_next_month(this))[0] else this
    w_open, w_close = reset_window(*cand)
    if t >= w_close:
        return _period(cand, "calendar")
    if used is None or allowance is None:
        return _period(_prev_month(cand), "used_old")
    games = _unique(period_games)
    first_day, today = w_open.astimezone(ET).date(), et_now.date()
    n = sum(1 for g in games if first_day <= _day(g.game_date) <= today)
    s = _reserve(reserve, allowance)
    c = max(1, int(c_max))
    b = busiest_month(games) if busiest is None else busiest
    limit = c * p_max(allowance, s, c, b) * n + margin(allowance)
    if used <= limit:
        return _period(cand, "used_fresh")
    return _period(_prev_month(cand), "used_old")


def reset_seen(prev_used, used) -> bool:
    """`used` fell: the credits reset (3.8: paid calls resume)."""
    return prev_used is not None and used is not None and used < prev_used


# -- The games a plan counts -------------------------------------------------------
def _tip_key(g: Game):
    tip = _utc(g.tip_utc)
    return (tip is None, tip or _at_et(_day(g.game_date), dt.time(23, 59)), str(g.game_id))


def _horizon(games: list, t: dt.datetime, period: Period) -> list[Game]:
    today = t.astimezone(ET).date()
    end = period.horizon_end
    end_day = end.astimezone(ET).date()
    out = []
    for g in games:
        tip = _utc(g.tip_utc)
        if tip is not None:
            if t < tip < end:
                out.append(g)
        elif today <= _day(g.game_date) < end_day:
            out.append(g)
    return sorted(out, key=_tip_key)


def _in_period(games: list, period: Period) -> list[Game]:
    first_day = period.opens.astimezone(ET).date()
    end_day = period.horizon_end.astimezone(ET).date()
    out = []
    for g in games:
        tip = _utc(g.tip_utc)
        if tip is not None:
            if period.opens <= tip < period.horizon_end:
                out.append(g)
        elif first_day <= _day(g.game_date) < end_day:
            out.append(g)
    return out


def horizon_games(games, now, period: Period) -> list[Game]:
    """G of 3.3: the untipped games of the horizon, in tip order. Untipped is
    now < tip < horizon end; a game whose tip is TBD counts while its date is
    today or later and before the horizon end's date."""
    return _horizon(_unique(games), _utc(now), period)


def period_games(games, period: Period) -> list[Game]:
    """The period's projected games, tipped ones too: tip in [opens, horizon
    end), or for a TBD tip, dated from the opening day to the day before the
    end. With the window rule a period counts the last day of the month
    before it (its window opens that morning) and never the 1st after it."""
    return _in_period(_unique(games), period)


def games_before(games, period: Period, today) -> int:
    """The period's games dated before `today`: the brake's usage test (3.4).
    `games` are the period's games, tipped ones too (plan's and period_of's
    `period_games`): every game this counts has tipped by now, so a list
    that starts at `now` counts none, and the brake would never apply."""
    day = _day(today)
    return sum(1 for g in period_games(games, period) if _day(g.game_date) < day)


def waiting_first(games, slots, now, period: Period, game_id=None) -> int:
    """F of 3.6: the horizon's OTHER untipped games still waiting for their
    first pull (no pull at all, the brake's assumed one counting as a pull)."""
    pulls = _pulls_by_game(slots)
    return sum(1 for g in horizon_games(games, now, period)
               if g.game_id != game_id and pulls.get(g.game_id, 0) == 0)


# -- The plan (3.3) and the ceiling (3.6) ------------------------------------------
def spending_ceiling(allowance, reserve, period_games, today) -> int:
    """ceiling(d) = min(A - S, ceil(1.5 x (A - S) x n_through(d) / n_period)),
    where n_period counts the period's games and n_through(d) those dated on
    or before d (3.6). It reads `used`, which the vendor keeps, so no restart
    or crash loop can push the period's spending more than half again above
    its pro-rata pace: it holds every refresh, top-up and re-buy, and every
    first pull of a ledger born after the game's first-pull time. A true
    first pull skips it (may_spend's `true_first`, build amendment B23), so a
    second process whose ledger was born at or before that time can buy the
    morning's games again past it, bounded only by the reserve (build 7's
    review, R7-1). Exact in integers (no float)."""
    return _ceiling(int(allowance) - int(reserve), _unique(period_games), _day(today))


def _ceiling(budget: int, games: list, day: dt.date) -> int:
    n_through = sum(1 for g in games if _day(g.game_date) <= day)
    n_period = max(1, len(games))
    return min(budget, math.ceil(Fraction(CEILING_FACTOR) * budget * n_through / n_period))


def plan(remaining, used, pending, settings, horizon, slots, period: Period, auth, busiest,
         *, now, period_games, breaker_day=None) -> Plan:
    """The plan for one walk (3.3):

        S = reserve;  A = used + remaining;  B = remaining - S - pending
        G = untipped games in the horizon (at least 1)
        made = pulls already made by those games (this ledger)
        auth "bad_key" / "out_of_credits"        -> paused
        the day breaker on today                 -> paused (reason breaker)
        remaining (or used) unknown              -> unknown: no paid call
        P_full = floor((B / c_full + made) / G) >= 1 -> full: P = min(P_full, P_max(c_full))
        P_core = floor((B / c_core + made) / G) >= 1 -> core: P = min(P_core, P_max(c_core))
        B >= c_core                              -> short: core markets, P = 1
        otherwise                                -> paused (reason at_reserve)

    P counts every pull a game may have, including those already made (m9),
    and is recomputed from the live balance on every walk, so credits a game
    did not use flow to the games after it.

    `horizon` holds the games to plan: only its untipped games before the
    horizon end count (G), so a list from `now` (design 3.1's wording) and
    the whole period both do. `period_games` (required) are the PERIOD'S
    projected games from its opening day, TIPPED ONES TOO: the work order's
    horizon (2.4.5). Today's spending ceiling counts them (n_period, and
    n_through today), so it is never taken from a list that starts at
    `now`, which would shrink it to what is left of the period. The two may
    be the same list; every game of either counts once, and games outside
    the period are not counted. `slots` are the ledger's Slots. `busiest` is
    the work order's (the season's busiest month); None counts the games
    given. `now` is the walk's instant: G and today's ceiling depend on it.
    `breaker_day` is the ledger's day breaker (an ET date or None)."""
    full = tuple(settings.markets)
    core = tuple(settings.core) or full[:1]
    known = remaining is not None and used is not None
    allowance = int(used) + int(remaining) if known else None
    s = _reserve(settings.reserve, allowance)
    games = _unique(list(period_games) + list(horizon))
    busiest = busiest_month(games) if busiest is None else busiest
    t = _utc(now)
    if t is None:
        raise ValueError("now is required")
    ceiling = _ceiling(allowance - s, _in_period(games, period), _et_day(t)) if known else 0
    cap_core = p_max(allowance, s, len(core), busiest) if known else 0

    def made_plan(mode, markets, pulls, cap, reason):
        return Plan(mode=mode, markets=markets, per_pull=len(markets), pulls_per_game=pulls,
                    p_max=cap, reserve=s, ceiling=ceiling, reason=reason)

    if auth != "ok":
        return made_plan("paused", core, 0, cap_core, auth if auth in AUTH_STATES else "bad_key")
    if breaker_on(breaker_day, t):
        return made_plan("paused", core, 0, cap_core, "breaker")
    if not known:
        return made_plan("unknown", core, 0, 0, "no_balance")
    b = int(remaining) - s - int(pending or 0)
    ids = {g.game_id for g in _horizon(games, t, period)}
    n = max(1, len(ids))
    made = sum(max(0, int(x.pulls)) for x in _slot_list(slots) if x.game_id in ids)
    for mode, markets, reason in (("full", full, "ok"), ("core", core, "below_full")):
        c = len(markets)
        pulls = (b + made * c) // (c * n)            # floor((B / c + made) / G), exactly
        if pulls >= 1:
            cap = p_max(allowance, s, c, busiest)
            return made_plan(mode, markets, min(pulls, cap), cap, reason)
    if b >= len(core):
        return made_plan("short", core, 1, cap_core, "below_core")
    return made_plan("paused", core, 0, cap_core, "at_reserve")


# -- When pulls happen (3.4) -------------------------------------------------------
def tip_for(vendor_commence, schedule_tip, schedule_tbd) -> dt.datetime | None:
    """THE tip (m2): the earlier of the vendor's commence_time and the
    schedule's tip. The schedule's tip is ignored while its time is TBD, since
    it is then a made-up 00:00 ET that would read as already tipped
    (live.time_tbd). None when neither is known."""
    tips = [t for t in (_utc(vendor_commence), None if _flag(schedule_tbd) else _utc(schedule_tip))
            if t is not None]
    return min(tips) if tips else None


def before_tip(now, tip, guard: dt.timedelta) -> bool:
    """now < tip - guard: THE tip guard (RC G6). No pull starts once now >=
    tip - TIP_GUARD (5 minutes, equal to pipeline.FREEZE_LEAD), and an
    unknown tip is never before it."""
    tp = _utc(tip)
    return tp is not None and _utc(now) < tp - guard


def first_pull_time(game_date, first_pull_et: dt.time) -> dt.datetime:
    """The earliest first pull of a game: BUCKETVALUE_ODDS_FIRST_PULL_ET
    (08:50 by default) on the game's own ET date, as a UTC instant."""
    return _at_et(_day(game_date), first_pull_et)


def breaker_on(breaker_day, now) -> bool:
    """The day breaker stops paid calls until the next ET day (3.5)."""
    day = _day(breaker_day)
    return day is not None and day >= _et_day(now)


def breaker_trips(streak: int) -> bool:
    """Billed failures in a row, across all games, that trip the day breaker."""
    return int(streak) >= BREAKER_STREAK


def due_kind(slot: Slot, plan: Plan, settings, now, game_date, *, breaker_day=None) -> str | None:
    """"first", "refresh", "pp" (the PrizePicks-only top-up), "rebuy" (REBUY:
    a braked game's one re-buy, build amendment B21) or None (3.4).

    Nothing is due while the plan does not spend, the day breaker is on, the
    tip guard has passed, the game backs off after a failure (`fail_until`,
    B2 item 2) or an empty answer is not 10 minutes old (`empty_next`). Then:
      - rebuy: the memory-loss brake counted the game as pulled at the
        ledger's birth (`braked`) and that assumed pull is still its only
        one, the ledger holds no lines for it, P >= 2 and it is before the
        game's last refresh time, t_last = tip - LAST_PULL_LEAD: ONE re-buy,
        at once (build amendment B21), so a cold start after the brake's
        window still shows today's lines. may_spend guards it as B11's
        re-buy (the floor and the ceiling checked) and live asks the core
        markets for it (B20). P >= 2 is its own condition: with P = 1 the
        brake's assumed pull is the game's only one, and nothing more is
        bought. A re-buy that counts as a pull (3.5: one that may have been
        billed) is the game's second, so one ledger never re-buys a game
        twice. An empty answer is free and no pull, so the re-buy is due
        again 10 minutes on; and live counts no pull for the timeout of such
        a free retry only once per clean answer for the game (its
        `free_failed`, build 6's review, S12-R1), so the next billed failure
        is that second pull. From t_last the planned refresh is due anyway;
    and nothing else once the game has had its P pulls (P counts every
    pull, a top-up's too, and never exceeds P_max):
      - first: at or after the first-pull time on the game's own ET date;
      - refresh, retried: a game whose pulls so far brought no stored lines
        after a failure (has_lines False and fails > 0: every pull a billed
        failure) is due again as soon as its backoff allows, while it has
        pulls left, instead of at the spaced time (build amendment B12), so a
        timeout at 08:50 is retried in minutes, not at the evening refresh.
        An empty answer keeps its own 10-minute wait (empty_next, above),
        and the brake's assumed pull is no failure: a braked game with no
        lines has the rebuy above, once, and otherwise keeps the spacing.
        The day breaker still stops the day;
      - refresh: pull k (k = pulls so far, 1 .. P-1) is due at
            t_first + (t_last - t_first) x k / (P - 1), t_last = tip - LAST_PULL_LEAD,
        and no sooner than last_at + MIN_REFRESH, only while that time is
        <= t_last. A late first pull spaces the rest from itself;
      - pp: a game whose pull lacked PrizePicks, every 10 minutes (pp_next),
        unless BUCKETVALUE_ODDS_PP_TOPUP is off. A due refresh comes first:
        it costs the same and asks every book, PrizePicks included.
    `game_date` is the game's ET date; `breaker_day` the ledger's breaker."""
    if slot is None or plan is None or plan.mode not in SPENDING_MODES:
        return None
    t = _utc(now)
    if breaker_on(breaker_day, t):
        return None
    tip = _utc(slot.tip_utc)
    if not before_tip(t, tip, settings.tip_guard):
        return None
    for until in (slot.fail_until, slot.empty_next):
        u = _utc(until)
        if u is not None and t < u:
            return None
    k, pulls = max(0, int(slot.pulls)), int(plan.pulls_per_game)
    if (slot.braked and k == 1 and not slot.has_lines and pulls >= 2
            and t < tip - settings.last_pull_lead):
        return REBUY                           # B21: the braked game's one re-buy, now
    if k >= pulls:
        return None
    if k == 0:
        day = _day(game_date) if game_date is not None else tip.astimezone(ET).date()
        return "first" if t >= first_pull_time(day, settings.first_pull_et) else None
    if not slot.has_lines and int(slot.fails) > 0:
        return "refresh"                       # B12: its backoff (fail_until, above) is over
    first = _utc(slot.first_at)
    if first is not None and pulls >= 2:
        t_last = tip - settings.last_pull_lead
        if t_last > first:
            last = _utc(slot.last_at) or first
            due = max(first + (t_last - first) * k / (pulls - 1), last + settings.min_refresh)
            if due <= t and due <= t_last:
                return "refresh"
    if settings.pp_topup and slot.pp_missing:
        nxt = _utc(slot.pp_next)
        if nxt is None or t >= nxt:
            return "pp"
    return None


# -- Guards on every paid call (3.6) ----------------------------------------------
def calls_in_window(calls, now) -> int:
    """Paid calls within the rate cap's 10 minutes before `now`."""
    t = _utc(now)
    return sum(1 for c in (_utc(x) for x in calls) if c is not None and t - c < RATE_WINDOW)


def may_spend(remaining, used, pending, cost, plan: Plan, waiting_first, c_core, *,
              now, tip, guard, auth, calls, max_calls, kind, true_first: bool = False) -> str | None:
    """None when a paid call may go ahead, else the first guard of 3.6 that
    refuses it, in order (a fixed code of REFUSALS):

      tip_guard   now >= tip - guard (2; RC G6, m2), or an unknown tip
      out_of_credits, bad_key   the auth state from the last 401 (3; 3.8)
      no_balance  remaining or used unknown (3.1)
      paused      the plan does not spend (its reason says why)
      reserve     remaining - pending - cost >= S fails (4)
      floor       remaining - pending - cost - S >= c_core x F fails, F being
                  `waiting_first`, the horizon's other games still waiting
                  for their first pull (5): no refresh, top-up or re-buy ever
                  spends a credit a later game needs for its morning pull.
                  Never checked for a FIRST pull (build amendment B1): the
                  floor protects first pulls from everything else, and never
                  makes one game's first pull wait to save credits for
                  another game's first pull. In a shortage the walk's tip
                  order then gives the earliest tips the credits first, and
                  first_pull_markets keeps a first pull's extra markets from
                  eating another game's morning points line. A first pull
                  that may re-buy what the process before this ledger bought
                  (first_kind's "rebuy", build amendment B11) is checked,
                  exactly as a refresh is, and so is a braked game's one
                  re-buy (due_kind's "rebuy", build amendment B21).
      ceiling     used + pending + cost <= ceiling(today) fails (6). Never
                  checked for a TRUE first pull (`true_first`, build
                  amendment B23): a first pull by a ledger born at or before
                  the game's first-pull time (true_first_pull, first_kind's
                  first clause), which can never re-buy what an earlier
                  process bought, so it is never what the ceiling bounds (a
                  restart or a crash loop buying again). It skips the
                  ceiling as every first pull skips the floor, and the
                  reserve still stops it. It cannot see what another
                  process running at the same time bought: a second process
                  whose ledger was born at or before that time buys the game
                  again past the ceiling (build 7's review, R7-1, as built).
                  The ceiling still holds every refresh, top-up and re-buy,
                  and a first pull whose ledger was born later (first_kind's
                  usage-test clause: go-live day, a period's first day).
                  live passes it False for the retry of a first pull whose
                  call expected to be free failed with no answer and may
                  have been billed, until a clean answer (build 6's review,
                  S12-R1): that retry may buy again what the call bought,
                  like the retry of a first pull whose failure counted,
                  which is a refresh
      rate_cap    `max_calls` paid calls already in the last 10 minutes (7)

    `kind` is the call's word, one of SPEND_KINDS: due_kind's "first",
    "refresh" or "pp" (the PrizePicks-only top-up), or "rebuy" (first_kind's
    for a first pull the ledger cannot tell from a re-buy, B11, and
    due_kind's for a braked game's one re-buy, B21: the floor checked, every
    other guard as for "first"); anything else raises ValueError.
    `true_first` (B23) says a "first" is a true first pull: only True or
    False is taken (anything else raises TypeError), and True with any other
    kind raises ValueError (a re-buy, a refresh or a top-up is never one).
    Its default, False, keeps the ceiling: an exemption nobody passes is no
    exemption. Item 1 (single flight, a fresh due_kind with fail_until and
    the day breaker) and item 8 (the kill switch and the key, read again) are
    live's. Every guard's argument is required: a guard nobody passes is no
    guard."""
    if kind not in SPEND_KINDS:
        raise ValueError("kind must be one of oddsbudget.SPEND_KINDS")
    if type(true_first) is not bool:
        raise TypeError("true_first must be True or False (build amendment B23)")
    if true_first and kind != "first":
        raise ValueError("only a first pull can be a true first pull (build amendment B23)")
    if not before_tip(now, tip, guard):
        return "tip_guard"
    if auth != "ok":
        return auth if auth in AUTH_STATES else "bad_key"
    if remaining is None or used is None:
        return "no_balance"
    if plan is None or plan.mode not in SPENDING_MODES:
        return "paused"
    left = int(remaining) - int(pending or 0) - int(cost)
    if left < plan.reserve:
        return "reserve"
    if kind != "first" and left - plan.reserve < int(c_core) * int(waiting_first):
        return "floor"
    if not true_first and int(used) + int(pending or 0) + int(cost) > plan.ceiling:
        return "ceiling"
    if calls_in_window(calls, now) >= int(max_calls):
        return "rate_cap"
    return None


def first_pull_markets(plan: Plan, settings, remaining, pending, waiting_first) -> tuple[str, ...]:
    """The markets a FIRST pull asks for (build amendment B1), a re-buy's
    too (B11: it is this ledger's first pull of the game): the plan's own
    when the morning-pull floor holds at their cost,

        remaining - pending - c_plan - S >= c_core x waiting_first

    (c_plan = plan.per_pull, c_core = the number of core markets, S = the
    plan's reserve, waiting_first = the horizon's OTHER games still waiting
    for their first pull, as may_spend counts them), else the core markets.
    So a first pull's extra markets never eat another game's morning points
    line, and a first pull is never starved while the plan stays "full".
    With the shipped default (points only) the two are the same. An unknown
    balance gives the core markets (no paid call goes ahead without one)."""
    markets = tuple(plan.markets)
    core = tuple(settings.core) or markets[:1]
    if remaining is None:
        return core
    left = int(remaining) - int(pending or 0) - int(plan.per_pull) - int(plan.reserve)
    return markets if left >= len(core) * max(0, int(waiting_first)) else core


# -- Billed failures (3.5) and the memory-loss brake (3.4) -----------------------
def backoff(fails: int) -> dt.timedelta:
    """The wait after the n-th failure in a row on a game: 90 s, 3, 6, 12, 24
    and 48 minutes, then 60 minutes each time (B2 item 3). None yet: none."""
    n = int(fails)
    if n < 1:
        return dt.timedelta(0)
    wait = BACKOFF_FIRST
    for _ in range(n - 1):
        wait *= 2
        if wait >= BACKOFF_MAX:
            return BACKOFF_MAX
    return wait


def _spending(used, c_core, games_before_today) -> bool:
    """The brake's usage test (3.4): the account already spent at least half
    a core pull per earlier game of the period (2 x used >= c_core x the
    earlier games), which says a feed was running. Never on the period's
    first day (no earlier game: nothing to go on), and an unknown `used`
    never passes it. brake_applies and first_kind both read it, so the two
    can never disagree on whether a period was spending."""
    n = int(games_before_today or 0)
    return used is not None and n > 0 and 2 * int(used) >= int(c_core) * n


def brake_applies(born_at, had_ledger, used, c_core, games_before_today, first_pull_at) -> bool:
    """The memory-loss brake (B1 item 2, M1). A ledger born more than BRAKE
    (60 minutes) after the day's first-pull time applies it when this process
    had a ledger before (a version change), or when the account already spent
    at least half a core pull per earlier game of the period (used >= 0.5 x
    c_core x games_before_today, which says a feed was running). It then
    treats each of today's untipped games as pulled once, at the ledger's
    birth, so the next pull follows the normal spacing from that time (live
    applies it, once per ledger, and marks each such Slot `braked`). A braked
    game the ledger holds no lines for keeps one re-buy at once, while P >= 2
    and before its last refresh time (due_kind's "rebuy", build amendment
    B21). On the first day of a period (games_before_today 0) the usage test
    cannot pass, and an unknown `used` never passes it."""
    born, first = _utc(born_at), _utc(first_pull_at)
    if born is None or first is None or born <= first + BRAKE:
        return False
    if had_ledger:
        return True
    return _spending(used, c_core, games_before_today)


def true_first_pull(born_at, first_pull_at) -> bool:
    """first_kind's first clause: the ledger was born at or before the game's
    first-pull time (`first_pull_at`, first_pull_time on its own ET date), so
    it has seen the game's whole morning and its first pull of the game can
    never re-buy what an earlier process bought (build amendment B11). Such
    a TRUE first pull skips the spending ceiling, as every first pull skips
    the floor (may_spend's `true_first`, build amendment B23). An unknown
    instant is never one."""
    born, first = _utc(born_at), _utc(first_pull_at)
    return born is not None and first is not None and born <= first


def first_kind(born_at, had_ledger, used, c_core, games_before_today, first_pull_at) -> str:
    """"first" or "rebuy" (REBUY): the kind may_spend reads for a pull of a
    game with no pull in this ledger, which due_kind calls "first" (build
    amendment B11). The arguments are brake_applies', whose usage test this
    is: `first_pull_at` is first_pull_time on the game's own ET date, and
    `games_before_today` counts the period's games dated before that day.

    B1's floor exemption holds only when the ledger has seen the game's whole
    morning, so that what it buys cannot be what an earlier process bought:
      - it was born at or before the game's first-pull time, or
      - the period was not spending before it was born: no earlier ledger in
        this process (`had_ledger` False) and the usage test fails (no
        earlier game this period, or 2 x used below c_core x the earlier
        games).
    Otherwise the pull may re-buy what the process before this ledger bought
    (a restart in the morning, before the brake's window): a "rebuy", which
    the morning-pull floor guards as design 3.6 wrote it. So for a ledger
    born after the brake's window, "rebuy" is exactly brake_applies (whose
    braked games have their pull, so first_kind never sees them). `used` is
    the account's spend when the ledger's brake was decided (live keeps it:
    what the period had spent before this ledger made any paid call), so the
    ledger's own pulls never turn its later first pulls into re-buys. As in
    the brake, an unknown `used` never passes the usage test; no paid call
    goes ahead without a balance anyway (3.1). The first clause is
    true_first_pull, which also says whether a "first" skips the spending
    ceiling (build amendment B23: only that clause's does)."""
    if true_first_pull(born_at, first_pull_at):
        return "first"
    if not had_ledger and not _spending(used, c_core, games_before_today):
        return "first"
    return REBUY
