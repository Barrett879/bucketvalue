"""bucketlib.oddsbudget: the credit governor of the licensed odds feed (The
Odds API), design section 3 (its surface is design 2.3; the checks are 8.2's).

Pure, and so are these tests: every `now` is an argument, nothing reads a
clock, the environment, a file or a network, and the settings come from
oddsapi.read_settings on a literal mapping (the production reader).

RP is design/replay_v2.py, the budget replay behind design 3.9. Its cases
enter here as LITERAL numbers with literal inputs (design 8.4 T4): TIPS below
holds the games per day and their ET tips, copied by hand from the committed
2026-27 schedule at ad1826f (state/current/schedule_v1.parquet) with RP's
filter: regular season, both teams known, not postponed, a set tip. No test
reads the schedule.

Every date is a fixed PAST date (rule T4). The 2026-27 games are replayed ONE
YEAR EARLIER, on the same month, day and ET clock: RP's 2027-01-07 is
2026-01-07 here, and RP's 2026-11-30 is 2025-11-30. The governor has no
weekday rule, and every replayed day (Nov 30 to Jan 31) falls in Eastern
Standard Time in both years, so each reset window, horizon, tip guard and
ceiling lands on the same ET clock time, and RP's numbers hold unchanged.
(Nov 1, still daylight time in 2025, is used once, by the m10 check, where
only the count of untipped games matters.)

Replay is RP's Sim with every DECISION a call into the module (period_of,
plan, brake_applies, due_kind, first_kind, waiting_first, first_pull_markets,
may_spend, backoff and breaker_trips). Only design 3.5's bookkeeping is the
test's own, kept as live keeps it: a pull adds one to `pulls`; a pull that
lands lines sets `has_lines` (B12); a pull without PrizePicks marks it
missing and a pull with it clears that; a free PrizePicks-only answer moves
`pp_next`; billed junk backs the game off and grows the streak. A ledger's
first pull of a game is asked of first_kind, as live asks it (B11), with the
`used` its brake was decided on; the first ledger of a replay was born before
the replay starts, so it has seen every morning.

The MUTATION checks (8.2): the reserve, the morning-pull floor and the tip
guard (part 1; part 2 is live's). The day breaker's read in due_kind is
pinned by RP case E: without it the nine other games of that morning's walk
are bought. Each was seen to fail with its guard broken (the S4 mutation run
broke 45 rules in all, and every one failed a test here).
"""
from __future__ import annotations

import datetime as dt
import inspect
from dataclasses import replace

import pandas as pd
import pytest

from bucketlib import horizons, pipeline
from bucketlib import oddsbudget as OB
from bucketlib.sources import oddsapi as O
from bucketlib.util import ET, UTC

FREE = 500                         # the Starter plan's 500 credits a month
BUSIEST = 236                      # January's games: the season's busiest month (RP)
POINTS = O.read_settings({})       # the defaults: points only, the reserve unset
TWELVE_ENV = {"BUCKETVALUE_ODDS_MARKETS": ",".join(O.MAIN_MARKETS)}
TWELVE = O.read_settings(TWELVE_ENV)
PTS = ("player_points",)
STEP = dt.timedelta(minutes=5)     # RP: a view every 5 minutes, 08:00 to 23:55 ET
SECOND = dt.timedelta(seconds=1)


def year(month: int) -> int:
    """RP's 2026-27 season, one year earlier: August to December 2025, then 2026."""
    return 2025 if month >= 8 else 2026


def day(month: int, d: int) -> dt.date:
    return dt.date(year(month), month, d)


def et(month: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> dt.datetime:
    return dt.datetime(year(month), month, d, hh, mm, ss, tzinfo=ET)


# The 2026-27 regular season, November to January, as RP reads it: "MM-DD" and
# each game's ET tip ("HHMM"), in tip order (ties in game-id order).
TIPS = """
    11-01 1530 1700 2030 2100
    11-02 1900 1930 1930 1930 1930 1930 2000 2000 2000 2030 2030 2100 2200 2200 2200
    11-04 1900 1900 1930 1930 1930 2000 2000 2000 2030 2100 2100 2130 2200 2230
    11-05 2100 2200
    11-06 1900 1900 1930 2000 2000 2000 2000 2200 2200
    11-07 1700 1930 2000 2000 2030 2100 2200
    11-08 1530 1800 1900 1930 2000 2100 2100
    11-09 1900 1900 1900 1930 2000 2030 2100 2200
    11-10 1930 2000 2100 2100 2200 2300
    11-11 1900 1900 1900 2000 2000 2000 2000 2100 2130 2200
    11-12 1900 1900 2000 2100
    11-13 1900 1900 2000 2000 2000 2030 2130 2200 2230
    11-14 1900 1930 2000 2000 2000
    11-15 1300 1600 1630 1700 1800 1800 1900 2000 2100
    11-16 1900 1900 1900 1930 1930 2000 2000 2100
    11-17 1930 2000 2000 2230 2300
    11-18 1900 1900 1900 1900 1930 1930 2000 2000 2100 2130 2200 2200
    11-19 2000 2000 2000 2000
    11-20 1900 1930 1930 1930 2000 2000 2130 2200 2200
    11-21 1300 1930 2000 2000 2000 2200
    11-22 1530 1800 1900 1900 1900 2000 2030
    11-23 1900 1900 1930 1930 1930 2000 2000 2030 2200 2200
    11-24 1900 2000 2200
    11-25 1700 1900 1930 1930 2000 2000 2030 2200 2200
    11-27 1900 1930 1930 1930 2000 2030 2100 2130 2130 2130 2130
    11-28 1700 1800 2000 2000 2030
    11-29 1600 1600 1700 1800 1800 1800 1800 1800 2030 2100
    11-30 1900 1900 1930 1930 2000 2000 2200
    12-01 1900 1900 1930 2000 2030 2100 2200 2300
    12-02 1900 1900 1930 2000 2000 2000 2000 2100 2230
    12-03 1900 1930 2000 2000 2200
    12-12 1700 1900 1930 2000 2030 2100 2130 2200
    12-13 1500 1530 1700 1800 1900 2100
    12-14 1900 1900 1900 1930 2000 2000 2000 2030 2100 2100 2200
    12-15 2000 2300
    12-16 1900 1900 1900 1900 1900 1930 1930 2000 2000 2000 2000 2030 2130
    12-17 1900 1900 2100
    12-18 1900 1900 1900 1930 1930 1930 1930 2000 2000 2000 2000 2000
    12-19 1800 1900 2000 2100
    12-20 1500 1530 1600 1800 1800 1900 1900 2000 2000 2200
    12-21 1900 1900 1900 1930 1930 2000 2000 2000 2200 2230
    12-22 1900 2000 2230
    12-23 1900 1900 1900 1930 1930 1930 1930 1930 2000 2000 2000 2100 2200 2200 2230
    12-25 1200 1430 1700 2000 2230
    12-26 1300 1530 1700 1800 1900 2000 2000 2230
    12-27 1530 1800 1800 1800 1800 1800 1900 1900 2000 2030 2100 2100
    12-28 1930 2000 2100 2230
    12-29 1900 1900 1900 2000 2000 2000 2200 2200 2230
    12-30 1900 1930 2000 2100 2100 2200
    12-31 1300 1500 1500 1500 1600 1800 2000 2000 2100
    01-01 1700 1930 1930 2000 2030 2130
    01-02 1300 1700 1900 1900 1930 2000 2000 2030 2100 2200
    01-03 1700 1730 1800 1900 1900 1900 1900 2000
    01-04 1900 1900 2000 2100 2100 2200
    01-05 1900 1900 1900 1900 1930 2000 2000 2200 2200
    01-06 1930 1930 2000 2000 2200
    01-07 1900 1900 1900 1900 1930 1930 2000 2000 2030 2200 2200 2200
    01-08 1900 1900 1930 2000 2000 2130 2200 2200
    01-09 1700 1800 1900 2030 2200 2230
    01-10 1500 1800 1800 1800 1800 1800 1900 2100
    01-11 1900 1900 1900 1930 2130 2230
    01-12 1930 1930 2000 2000 2000 2100 2200 2300
    01-13 1900 1900 1900 1930 2130 2200 2200
    01-14 1400 1900 1900 1930 2000 2100 2100 2200 2200
    01-15 1900 1900 1930 2000 2000 2030 2200 2230
    01-16 1230 1900 1930 2000 2030 2100
    01-17 1030 1300 1530 1530 1700 1800 2000 2100
    01-18 1200 1430 1500 1700 1930 2000 2230
    01-19 1900 1900 1900 1930 2000 2200 2200
    01-20 1900 1900 1930 1930 2000 2000 2100 2100 2130 2200
    01-21 1930 1930 2100 2200
    01-22 1900 1900 1900 1930 2000 2000 2100 2130 2200 2200 2230
    01-23 1700 1800 1900 2000 2000 2100
    01-24 1530 1600 1600 1800 1800 1900 2030
    01-25 1900 1900 1900 1930 2000 2000 2000 2130
    01-26 1900 1900 1930 2000 2100 2200 2300
    01-27 1900 1900 1930 1930 1930 2000 2030 2130 2200
    01-28 1900 1900 1900 1900 2000 2000 2100 2130
    01-29 1900 1930 1930 2000 2030 2130
    01-30 1230 1500 1730 1900 1930 2000 2030 2130
    01-31 1500 1800 1800 1800 1800 1900 2000 2030
"""


def _days(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        mmdd, *tips = line.split()
        m, d = int(mmdd[:2]), int(mmdd[3:])
        out[day(m, d)] = tuple(
            OB.Game(f"{mmdd}-{i:02d}", day(m, d), et(m, d, int(t[:2]), int(t[2:])).astimezone(UTC))
            for i, t in enumerate(tips, 1))
    return out


DAYS = _days(TIPS)


def games(first: dt.date, last: dt.date) -> list:
    return [g for when in sorted(DAYS) if first <= when <= last for g in DAYS[when]]


JANUARY = games(day(1, 1), day(1, 31))
PERIOD_JAN = games(day(12, 31), day(1, 31))     # January's period: its window opens on Dec 31


def plan_with(pulls: int, mode: str = "full", per_pull: int = 1) -> OB.Plan:
    """A plan made by hand, for the due-time checks (the plan is an input)."""
    return OB.Plan(mode=mode, markets=PTS, per_pull=per_pull, pulls_per_game=pulls, p_max=4,
                   reserve=25, ceiling=475, reason="ok")


def slot_of(g: OB.Game, *, has_lines=None, **kw) -> OB.Slot:
    """A ledger slot made by hand. A Slot must say whether the game has
    lines (B12); here a pull is a successful one unless the test says
    otherwise, so a slot with a pull has lines unless `has_lines` is given."""
    lines = int(kw.get("pulls", 0)) > 0 if has_lines is None else has_lines
    return OB.Slot(g.game_id, g.game_id, g.tip_utc, has_lines=lines, **kw)


def spend(remaining, used, plan, *, now, tip, pending=0, cost=1, waiting=0, auth="ok", calls=(),
          kind="refresh"):
    """may_spend with the defaults' guard and cap. The kind defaults to a
    refresh, which every guard applies to (a first pull skips the floor, B1)."""
    return OB.may_spend(remaining, used, pending, cost, plan, waiting, 1, now=now, tip=tip,
                        guard=POINTS.tip_guard, auth=auth, calls=list(calls), max_calls=40,
                        kind=kind)


# -- The replay -------------------------------------------------------------------------
class Replay:
    """RP's Sim (design/replay_v2.py), every decision made by the module."""

    def __init__(self, gs, *, allowance=FREE, used=0, env=None, pp_post=None, junk=False,
                 born=None):
        self.games = list(gs)
        self.by_day: dict = {}
        for g in self.games:
            self.by_day.setdefault(g.game_date, []).append(g)
        for v in self.by_day.values():
            v.sort(key=lambda g: (g.tip_utc, g.game_id))
        self.st = O.read_settings(dict(env or {}))
        self.allowance = allowance
        self.used, self.remaining = used, allowance - used
        self.pp_post, self.junk = pp_post, junk
        self.calls: list = []              # (ET time, game id, cost, kind)
        self.new_ledger(born)

    def new_ledger(self, born, had_ledger=False):
        """A ledger born at `born` (None: before the replay starts, so it
        has seen every morning). `born` stays set until the brake is decided
        at the ledger's first walk; `birth` and the `used` the brake was
        decided on are what first_kind reads (B11), as live keeps them."""
        self.slots, self.born, self.had_ledger = {}, born, had_ledger
        self.birth, self.brake_used = born, None
        self.rate, self.streak, self.breaker_day = [], 0, None

    def slot(self, g):
        return self.slots.get(g.game_id) or slot_of(g)

    def _brake(self, now, per):
        """Once per ledger, at its first walk (design 2.4.4 step 4; 3.4). A
        game it counts is marked braked, as live marks it (B21: with no
        lines it keeps one re-buy)."""
        today = now.date()
        first = OB.first_pull_time(today, self.st.first_pull_et)
        self.brake_used = self.used
        if OB.brake_applies(self.born, self.had_ledger, self.used, len(self.st.core),
                            OB.games_before(self.games, per, today), first):
            for g in OB.horizon_games(self.games, now, per):
                s = self.slot(g)
                if g.game_date == today and s.pulls == 0:
                    self.slots[g.game_id] = replace(s, pulls=1, first_at=self.born, last_at=self.born,
                                                    braked=True)
        self.born = None

    def view(self, now, *, plan=None):
        st = self.st
        per = OB.period_of(now, self.used, self.used + self.remaining, st.reserve,
                           len(st.markets), self.games, busiest=BUSIEST)
        if self.born is not None:
            self._brake(now, per)
        pl = plan or OB.plan(self.remaining, self.used, 0, st, self.games, self.slots, per, "ok",
                             BUSIEST, now=now, breaker_day=self.breaker_day, period_games=self.games)
        today = now.date()
        for g in self.by_day.get(today, ()):
            s = self.slot(g)
            kind = OB.due_kind(s, pl, st, now, g.game_date, breaker_day=self.breaker_day)
            if kind is None:
                continue
            posted = self.pp_post is None or now.time() >= self.pp_post
            if kind == "pp" and not posted:        # free while PrizePicks has nothing: not a pull
                self.slots[g.game_id] = replace(s, pp_next=now + OB.PP_RETRY)
                continue
            self.rate = [t for t in self.rate if now - t < OB.RATE_WINDOW]
            true_first = False
            if kind == "first":
                first_at = OB.first_pull_time(g.game_date, st.first_pull_et)
                if self.birth is not None:                    # B11, as live asks it
                    kind = OB.first_kind(self.birth, self.had_ledger, self.brake_used, len(st.core),
                                         OB.games_before(self.games, per, today), first_at)
                # B23, as live passes it: a ledger that saw the game's morning (the
                # replay's first ledger saw every morning) makes a TRUE first pull.
                true_first = kind == "first" and (self.birth is None
                                                  or OB.true_first_pull(self.birth, first_at))
            waiting = OB.waiting_first(self.games, self.slots, now, per, g.game_id)
            # B1: a ledger's first pull asks the plan's markets only when the floor holds at that
            # cost; B20, as live asks it: a re-buy (B11's, or a braked game's, B21) the core ones.
            markets = (tuple(st.core) if kind == OB.REBUY
                       else OB.first_pull_markets(pl, st, self.remaining, 0, waiting)
                       if kind == "first" else pl.markets)
            cost = len(markets)
            refused = OB.may_spend(
                self.remaining, self.used, 0, cost, pl, waiting, len(st.core),
                now=now, tip=s.tip_utc, guard=st.tip_guard, auth="ok", calls=self.rate,
                max_calls=st.max_calls_10min, kind=kind, true_first=true_first)
            if refused is not None:
                continue
            assert now < s.tip_utc - st.tip_guard          # the replay's own witness
            self.rate.append(now)
            self.used += cost
            self.remaining -= cost
            assert self.remaining >= pl.reserve            # never across the reserve
            self.calls.append((now, g.game_id, cost, kind))
            s = replace(s, pulls=s.pulls + 1, first_at=s.first_at or now, last_at=now)
            if self.junk:                                  # billed junk: a billed failure
                fails = s.fails + 1
                self.slots[g.game_id] = replace(s, fails=fails, fail_until=now + OB.backoff(fails))
                self.streak += 1
                if OB.breaker_trips(self.streak):
                    self.breaker_day = today
                continue
            self.streak = 0
            self.slots[g.game_id] = replace(s, fails=0, fail_until=None, pp_missing=not posted,
                                            pp_next=None if posted else now + OB.PP_RETRY,
                                            has_lines=True)

    def run(self, start, end, *, restarts=(), reset_at=None):
        t = start
        while t < end:
            if reset_at is not None and reset_at <= t < reset_at + STEP:
                self.used, self.remaining = 0, self.allowance       # the vendor's reset
            if any(r <= t < r + STEP for r in restarts):
                self.new_ledger(t)                                  # a process restart
            if dt.time(8, 0) <= t.time() <= dt.time(23, 55):
                self.view(t)
            t += STEP
        return self


def spend_on(rp: Replay, when: dt.date) -> int:
    return sum(k for t, _, k, _ in rp.calls if t.date() == when)


def pulls_per_game(rp: Replay, gs) -> dict:
    n = {g.game_id: 0 for g in gs}
    for _, gid, _, _ in rp.calls:
        if gid in n:
            n[gid] += 1
    out: dict = {}
    for v in n.values():
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def by_kind(rp: Replay) -> dict:
    out: dict = {}
    for *_, kind in rp.calls:
        out[kind] = out.get(kind, 0) + 1
    return out


# -- The literal inputs and the constants -------------------------------------------
def test_the_literal_schedule_has_rps_counts():
    """design 3.9: 215 games in November, 172 in December (before the Cup-week
    additions) and 236 in January, the busiest month; Dec 31 has 9, Jan 1 6,
    Jan 7 12 and Jan 12 8; two of the three 15-game days are in here."""
    months: dict = {}
    for g in games(day(11, 1), day(1, 31)):
        months[g.game_date.month] = months.get(g.game_date.month, 0) + 1
    assert months == {11: 215, 12: 172, 1: 236}
    assert OB.busiest_month(games(day(11, 1), day(1, 31))) == BUSIEST
    assert [len(DAYS[day(m, d)]) for m, d in ((12, 31), (1, 1), (1, 7), (1, 12))] == [9, 6, 12, 8]
    assert len(DAYS[day(11, 2)]) == len(DAYS[day(12, 23)]) == 15
    assert len(PERIOD_JAN) == 245


def test_the_constants_and_the_settings_the_governor_reads_are_the_designs():
    assert OB.MAX_PULLS_PER_GAME == 48 and OB.CEILING_FACTOR == 1.5 and OB.BREAKER_STREAK == 3
    assert OB.BRAKE == dt.timedelta(minutes=60)
    assert OB.EMPTY_RETRY == OB.PP_RETRY == OB.RATE_WINDOW == dt.timedelta(minutes=10)
    assert (OB.BACKOFF_FIRST, OB.BACKOFF_MAX) == (dt.timedelta(seconds=90), dt.timedelta(minutes=60))
    assert (OB.WINDOW_OPEN_ET, OB.WINDOW_CLOSE_ET) == (dt.time(5, 0), dt.time(8, 0))
    s = POINTS                                         # design 4.1's defaults
    assert (s.markets, s.core, s.reserve) == (PTS, PTS, None)
    assert s.first_pull_et == dt.time(8, 50)
    assert (s.last_pull_lead, s.min_refresh, s.tip_guard) == (
        dt.timedelta(minutes=30), dt.timedelta(minutes=15), dt.timedelta(minutes=5))
    assert s.max_calls_10min == 40 and s.pp_topup is True
    assert TWELVE.markets == O.MAIN_MARKETS and len(TWELVE.markets) == 12 and TWELVE.core == PTS


def test_the_default_reserve_is_5_percent_and_never_under_25_and_the_margin_1_percent():
    assert [OB.default_reserve(a) for a in (500, 20_000, 100_000)] == [25, 1000, 5000]
    assert (OB.default_reserve(100), OB.default_reserve(501)) == (25, 26)      # ceil(25.05)
    assert [OB.margin(a) for a in (500, 20_000, 100_000)] == [50, 200, 1000]


# -- The credit period (3.2; B3, M2) --------------------------------------------------
@pytest.mark.parametrize("month", [(2025, 11), (2025, 12), (2026, 1), (2026, 4)])
def test_the_reset_window_covers_midnight_on_the_1st_in_every_zone(month):
    """Daylight time (Nov 1 2025, Apr 1 2026) and standard time alike."""
    opens, closes = OB.reset_window(*month)
    first = dt.date(*month, 1)
    assert opens == dt.datetime.combine(first - dt.timedelta(days=1), dt.time(5, 0), tzinfo=ET)
    assert closes == dt.datetime.combine(first, dt.time(8, 0), tzinfo=ET)
    for minutes in range(-12 * 60, 14 * 60 + 1, 15):
        zone = dt.timezone(dt.timedelta(minutes=minutes))
        assert opens <= dt.datetime.combine(first, dt.time(0, 0), tzinfo=zone) <= closes, minutes


def test_outside_every_window_the_period_is_the_calendar_month_whatever_used_says():
    for used in (0, 470, 20_000):
        p = OB.period_of(et(1, 7, 12), used, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
        assert (p.month, p.basis) == ((2026, 1), "calendar")
        assert (p.opens, p.horizon_end) == (et(12, 31, 5), et(2, 1, 8))
    before = OB.period_of(et(12, 31, 4, 59), 0, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert (before.month, before.basis, before.horizon_end) == ((2025, 12), "calendar", et(1, 1, 8))
    closed = OB.period_of(et(1, 1, 8), 470, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert (closed.month, closed.basis) == ((2026, 1), "calendar")       # 08:00 on the 1st
    inside = OB.period_of(et(1, 1, 7, 59), 470, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert (inside.month, inside.basis) == ((2025, 12), "used_old")


def test_a_fresh_process_at_0850_on_jan_1_with_0_used_plans_236_games_at_two_pulls():
    """B3: v1 spent 248 credits on Jan 1's 6 games, 47 pulls each."""
    now = et(1, 1, 8, 50)
    per = OB.period_of(now, 0, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert (per.month, per.basis) == ((2026, 1), "calendar")
    assert len(OB.horizon_games(PERIOD_JAN, now, per)) == 236
    pl = OB.plan(500, 0, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    assert (pl.mode, pl.pulls_per_game, pl.p_max, pl.reserve, pl.per_pull, pl.reason) == (
        "full", 2, 4, 25, 1, "ok")                                       # floor(475 / 236)
    assert pl.ceiling == 44                    # ceil(1.5 x 475 x 15 / 245): Dec 31's 9 and Jan 1's 6


def test_a_reset_at_0800_on_nov_30_is_believed_from_used():
    """RP case B. Inside W(December) n = Nov 30's 7 games, so the line is
    c_max x P_max x n + MARGIN = 1 x 4 x 7 + 50 = 78."""
    now = et(11, 30, 10)
    gs = games(day(11, 1), day(12, 31))
    fresh = OB.period_of(now, 0, FREE, None, 1, gs, busiest=BUSIEST)
    assert (fresh.month, fresh.basis, fresh.horizon_end) == ((2025, 12), "used_fresh", et(1, 1, 8))
    assert OB.period_of(now, 78, FREE, None, 1, gs, busiest=BUSIEST).basis == "used_fresh"
    old = OB.period_of(now, 79, FREE, None, 1, gs, busiest=BUSIEST)
    assert (old.month, old.basis, old.horizon_end) == ((2025, 11), "used_old", et(12, 1, 8))
    assert len(OB.horizon_games(gs, now, fresh)) == 179                  # Nov 30's 7 and December's 172
    assert OB.plan(500, 0, 0, POINTS, gs, {}, fresh, "ok", BUSIEST, now=now,
                   period_games=gs).pulls_per_game == 2
    # had the reset not come, the old period's last day plans the 30 left over its own 7 games
    pl = OB.plan(30, 470, 0, POINTS, gs, {}, old, "ok", BUSIEST, now=now, period_games=gs)
    assert len(OB.horizon_games(gs, now, old)) == 7
    assert (pl.mode, pl.pulls_per_game, pl.reason) == ("short", 1, "below_core")    # B = 5 < 7


def test_at_1000_on_dec_31_with_470_used_the_old_period_stays_and_used_0_at_1900_flips_it():
    """RP case C. Inside W(January) n = Dec 31's 9: the line is 1 x 4 x 9 + 50 = 86."""
    gs = games(day(11, 30), day(1, 31))
    old = OB.period_of(et(12, 31, 10), 470, FREE, None, 1, gs, busiest=BUSIEST)
    assert (old.month, old.basis, old.horizon_end) == ((2025, 12), "used_old", et(1, 1, 8))
    assert OB.period_of(et(12, 31, 10), 86, FREE, None, 1, gs, busiest=BUSIEST).basis == "used_fresh"
    assert OB.period_of(et(12, 31, 10), 87, FREE, None, 1, gs, busiest=BUSIEST).basis == "used_old"
    new = OB.period_of(et(12, 31, 19), 0, FREE, None, 1, gs, busiest=BUSIEST)
    assert (new.month, new.basis, new.horizon_end) == ((2026, 1), "used_fresh", et(2, 1, 8))
    pl = OB.plan(30, 470, 0, POINTS, gs, {}, old, "ok", BUSIEST, now=et(12, 31, 10), period_games=gs)
    assert (pl.mode, pl.pulls_per_game) == ("short", 1)                  # B = 5: little, whatever the horizon
    unknown = OB.period_of(et(12, 31, 10), None, None, None, 1, gs, busiest=BUSIEST)
    assert (unknown.month, unknown.basis) == ((2025, 12), "used_old")    # no balance, no new period


def test_a_restart_on_the_2nd_plans_the_month_from_the_calendar():
    """B3 item 4. A fresh process on Jan 2 plans January's balance over
    January's games, whatever `used` says: never over one day."""
    now = et(1, 2, 8, 50)
    pers = {OB.period_of(now, u, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST) for u in (0, 12, 470)}
    assert len(pers) == 1
    (per,) = pers
    assert (per.month, per.basis) == ((2026, 1), "calendar")
    assert len(OB.horizon_games(PERIOD_JAN, now, per)) == 230
    pl = OB.plan(488, 12, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    assert (pl.mode, pl.pulls_per_game) == ("full", 2)                   # floor(463 / 230)


def test_paid_plans_use_the_same_window_and_no_reset_day_input_exists():
    """M2: the sign-up day is a paid plan's BILLING day; credits reset on the
    1st for every plan (UT/faqs.txt:53-55). $30 (20K) with 12 markets, at 10:00
    on Dec 31: 12 x P_max 12 x 9 + MARGIN 200 = 1496."""
    gs = games(day(11, 30), day(1, 31))
    assert OB.p_max(20_000, 1000, 12, BUSIEST) == 12 and OB.margin(20_000) == 200
    assert OB.period_of(et(12, 31, 10), 1496, 20_000, None, 12, gs, busiest=BUSIEST).basis == "used_fresh"
    assert OB.period_of(et(12, 31, 10), 1497, 20_000, None, 12, gs, busiest=BUSIEST).basis == "used_old"
    for d in (5, 12, 15, 20, 28):
        assert OB.period_of(et(1, d, 3), 0, 20_000, None, 12, gs, busiest=BUSIEST).basis == "calendar"
    for fn in (OB.period_of, OB.plan, OB.reset_window, OB.spending_ceiling):
        assert not [p for p in inspect.signature(fn).parameters if "reset" in p or "signup" in p]
    assert not [k for k in O.SETTING_DEFAULTS if "RESET" in k]


def test_reset_seen_is_used_falling():
    assert OB.reset_seen(470, 0) and OB.reset_seen(475, 474)
    assert not OB.reset_seen(0, 0) and not OB.reset_seen(5, 7)
    assert not OB.reset_seen(None, 0) and not OB.reset_seen(470, None)


def test_in_reset_window_tells_a_reset_on_the_1st_from_one_on_another_day():
    """3.2: a reset seen outside every window (a paid plan resetting on its
    sign-up day would do that) is what live logs once."""
    assert OB.in_reset_window(et(12, 31, 5)) and OB.in_reset_window(et(12, 31, 19))
    assert OB.in_reset_window(et(1, 1, 7, 59, 59))
    assert not OB.in_reset_window(et(12, 31, 4, 59, 59)) and not OB.in_reset_window(et(1, 1, 8))
    assert not OB.in_reset_window(et(1, 15, 3)) and not OB.in_reset_window(et(1, 30, 23, 59))
    assert OB.in_reset_window(et(1, 31, 5)) and OB.in_reset_window(et(4, 1, 7))    # Feb's; April's (EDT)


def test_a_game_listed_twice_counts_once():
    now = et(1, 7, 9)
    per = OB.period_of(now, 88, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    twice = PERIOD_JAN + list(DAYS[day(1, 7)]) + list(DAYS[day(12, 31)])
    assert len(OB.horizon_games(twice, now, per)) == 192
    assert len(OB.period_games(twice, per)) == 245
    assert OB.spending_ceiling(500, 25, twice, day(1, 7)) == 190
    assert OB.busiest_month(twice) == BUSIEST
    assert OB.plan(412, 88, 0, POINTS, twice, {}, per, "ok", BUSIEST, now=now,
                   period_games=twice) == OB.plan(
        412, 88, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)


# -- The plan (3.3; m9, m10) and the ceiling (3.6) ---------------------------------------
def test_p_counts_the_pulls_already_made_m9():
    """Noon on Jan 7: the 12 games have had their morning pull. With the pulls
    already made added back P stays 2; v1's B // (G x c) sagged to 1 and lost
    the evening refresh (CR m9)."""
    now = et(1, 7, 12)
    per = OB.period_of(now, 100, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    morning = {g.game_id: slot_of(g, pulls=1, first_at=et(1, 7, 8, 50), last_at=et(1, 7, 8, 50))
               for g in DAYS[day(1, 7)]}
    assert len(OB.horizon_games(PERIOD_JAN, now, per)) == 192
    assert OB.plan(400, 100, 0, POINTS, PERIOD_JAN, morning, per, "ok", BUSIEST,
                   now=now, period_games=PERIOD_JAN).pulls_per_game == 2   # floor((375 + 12) / 192)
    assert OB.plan(400, 100, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST,
                   now=now, period_games=PERIOD_JAN).pulls_per_game == 1   # floor(375 / 192)
    tipped = {g.game_id: slot_of(g, pulls=2) for g in DAYS[day(1, 6)]}   # yesterday's: not made by G
    assert OB.plan(400, 100, 0, POINTS, PERIOD_JAN, tipped, per, "ok", BUSIEST,
                   now=now, period_games=PERIOD_JAN).pulls_per_game == 1
    as_list = OB.plan(400, 100, 0, POINTS, PERIOD_JAN, list(morning.values()), per, "ok", BUSIEST, now=now,
                      period_games=PERIOD_JAN)
    assert as_list.pulls_per_game == 2                                   # the slots as a list do too


def test_p_max_is_4_on_free_12_on_30_and_48_on_59_and_it_binds():
    assert OB.p_max(500, 25, 1, BUSIEST) == 4
    assert OB.p_max(20_000, 1000, 12, BUSIEST) == 12
    assert OB.p_max(100_000, 5000, 12, BUSIEST) == 48                    # 2 x 33, capped at 48
    # The last day of a period (inside February's window, 400 used: still January):
    # 75 credits over 8 games would be 9 pulls each; the cap keeps them to 4.
    now = et(1, 31, 8, 50)
    per = OB.period_of(now, 400, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert (per.month, per.basis) == ((2026, 1), "used_old")
    pl = OB.plan(100, 400, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    assert (pl.pulls_per_game, pl.p_max) == (4, 4)
    # $59 (100K), 12 markets, on Jan 1: floor(95000 / (12 x 236)) = 33, under its cap of 48
    paid = OB.period_of(et(1, 1, 8, 50), 0, 100_000, None, 12, PERIOD_JAN, busiest=BUSIEST)
    pl = OB.plan(100_000, 0, 0, TWELVE, PERIOD_JAN, {}, paid, "ok", BUSIEST, now=et(1, 1, 8, 50),
                 period_games=PERIOD_JAN)
    assert (pl.mode, pl.pulls_per_game, pl.p_max, pl.per_pull, pl.reserve) == ("full", 33, 48, 12, 5000)


def test_every_mode_of_the_plan():
    now = et(1, 7, 8)
    per = OB.period_of(now, 88, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)

    def mode(remaining, used, settings=POINTS, auth="ok", breaker_day=None, pending=0):
        pl = OB.plan(remaining, used, pending, settings, PERIOD_JAN, {}, per, auth, BUSIEST, now=now,
                     breaker_day=breaker_day, period_games=PERIOD_JAN)
        return pl.mode, pl.pulls_per_game, pl.reason, pl.markets, pl.per_pull

    assert mode(412, 88) == ("full", 2, "ok", PTS, 1)                    # floor(387 / 192)
    assert mode(412, 88, TWELVE) == ("core", 2, "below_full", PTS, 1)    # 12 markets do not fit
    assert mode(200, 300) == ("short", 1, "below_core", PTS, 1)          # B = 175 < G = 192
    assert mode(26, 474) == ("short", 1, "below_core", PTS, 1)           # B = 1 = c_core
    assert mode(25, 475) == ("paused", 0, "at_reserve", PTS, 1)          # B = 0
    assert mode(30, 470, pending=5) == ("paused", 0, "at_reserve", PTS, 1)   # pending is not there
    assert mode(412, 88, auth="out_of_credits") == ("paused", 0, "out_of_credits", PTS, 1)
    assert mode(412, 88, auth="bad_key") == ("paused", 0, "bad_key", PTS, 1)
    assert mode(412, 88, auth="something else") == ("paused", 0, "bad_key", PTS, 1)
    assert mode(412, 88, breaker_day=day(1, 7)) == ("paused", 0, "breaker", PTS, 1)
    assert mode(412, 88, breaker_day=day(1, 6))[:3] == ("full", 2, "ok")     # yesterday's is off
    assert mode(None, 88) == ("unknown", 0, "no_balance", PTS, 1)
    assert mode(412, None) == ("unknown", 0, "no_balance", PTS, 1)
    big = OB.plan(20_000, 0, 0, TWELVE, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    assert (big.mode, big.pulls_per_game, big.markets, big.per_pull) == (
        "full", 8, O.MAIN_MARKETS, 12)                                   # floor(19000 / (12 x 192))
    assert set(OB.MODES) == {"full", "core", "short", "paused", "unknown"}


def test_an_unknown_balance_spends_nothing():
    """3.1: before any successful response, no paid call at all."""
    g = DAYS[day(1, 7)][0]
    now = et(1, 7, 9)
    per = OB.period_of(now, None, None, None, 1, PERIOD_JAN, busiest=BUSIEST)
    blind = OB.plan(None, None, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    assert (blind.mode, blind.pulls_per_game, blind.ceiling) == ("unknown", 0, 0)
    assert OB.due_kind(slot_of(g), blind, POINTS, now, g.game_date) is None
    full = plan_with(2)
    assert OB.due_kind(slot_of(g), full, POINTS, now, g.game_date) == "first"
    assert spend(None, 88, full, now=now, tip=g.tip_utc) == "no_balance"
    assert spend(412, None, full, now=now, tip=g.tip_utc) == "no_balance"
    assert spend(412, 88, blind, now=now, tip=g.tip_utc) == "paused"


def test_the_ceiling_is_190_credits_on_jan_7():
    """3.6: the period counts 245 games, Dec 31's nine included because the
    window opens that morning, and 65 of them are dated through Jan 7. Normal
    spending by that morning is 100: Jan 1-6's 44 games at two pulls, then Jan
    7's 12 morning pulls."""
    assert OB.spending_ceiling(500, 25, PERIOD_JAN, day(1, 7)) == 190    # ceil(1.5 x 475 x 65 / 245)
    assert sum(1 for g in PERIOD_JAN if g.game_date <= day(1, 7)) == 65
    now = et(1, 7, 8, 50)
    per = OB.period_of(now, 88, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    feb_first = OB.Game("0201-01", day(2, 1), et(2, 1, 19).astimezone(UTC))
    wider = games(day(12, 30), day(1, 31)) + [feb_first]
    assert len(OB.period_games(wider, per)) == 245                     # not Dec 30, not Feb 1
    pl = OB.plan(412, 88, 0, POINTS, wider, {}, per, "ok", BUSIEST, now=now, period_games=wider)
    assert pl.ceiling == 190 and 44 * 2 + 12 == 100 < pl.ceiling
    tip = DAYS[day(1, 7)][0].tip_utc
    assert spend(311, 189, pl, now=now, tip=tip) is None
    assert spend(310, 190, pl, now=now, tip=tip) == "ceiling"
    assert spend(312, 188, pl, now=now, tip=tip, pending=2) == "ceiling"
    # it never passes A - S, and with no games at all it lets nothing through
    assert OB.spending_ceiling(500, 25, PERIOD_JAN, day(1, 31)) == 475
    assert OB.spending_ceiling(500, 25, [], day(1, 7)) == 0


def test_the_ceiling_counts_the_periods_games_never_a_horizon_from_now():
    """Review I1, F5. Design 3.1 words the horizon as "every projected game
    from now to the horizon end", but the ceiling counts the PERIOD'S games
    (3.6), tipped ones too. Jan 3 at 08:50 with 32 used: 33 of the period's
    245 games are dated through Jan 3, so the ceiling is ceil(1.5 x 475 x 33 /
    245) = 96 and the morning pull goes ahead. Counted from a list that
    starts now (220 games, 8 of them through Jan 3) it would be 26: that pull
    refused while the plan still said "full", and the feed buying nothing
    from about the 3rd of every month. So plan() takes the period's games as
    their own required argument, and a horizon from now is harmless."""
    now = et(1, 3, 8, 50)
    per = OB.period_of(now, 32, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    from_now = [g for g in PERIOD_JAN if g.tip_utc > now]
    assert (len(from_now), sum(1 for g in from_now if g.game_date <= day(1, 3))) == (220, 8)
    assert OB.spending_ceiling(500, 25, PERIOD_JAN, day(1, 3)) == 96
    assert OB.spending_ceiling(500, 25, from_now, day(1, 3)) == 26     # what a list from now would give
    whole = OB.plan(468, 32, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now,
                    period_games=PERIOD_JAN)
    assert (whole.mode, whole.pulls_per_game, whole.ceiling) == ("full", 2, 96)    # floor(443 / 220)
    for horizon, period_games in ((from_now, PERIOD_JAN), (PERIOD_JAN, PERIOD_JAN + from_now),
                                  (PERIOD_JAN, from_now)):           # either list may hold the period
        assert OB.plan(468, 32, 0, POINTS, horizon, {}, per, "ok", BUSIEST, now=now,
                       period_games=period_games) == whole
    assert spend(468, 32, whole, now=now, tip=DAYS[day(1, 3)][0].tip_utc) is None
    param = inspect.signature(OB.plan).parameters["period_games"]
    assert param.kind is param.KEYWORD_ONLY and param.default is param.empty
    with pytest.raises(TypeError):
        OB.plan(468, 32, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now)   # never left out


def test_per_game_not_per_day_at_the_same_cost_m10():
    """3.10, at c = 1 on the free plan: a per-day split gives every November
    day 475 / 30 = 15.8 credits, so the 15-game Nov 2 gets its 15 morning pulls
    and no refresh while a 2-game day gets the same 15.8. Per game, every game
    gets floor(475 / 215) = 2 pulls: 30 credits for Nov 2, 4 for Nov 5."""
    now = et(11, 1, 8, 50)
    november = games(day(11, 1), day(11, 30))
    per = OB.period_of(now, 0, FREE, None, 1, november, busiest=BUSIEST)
    assert (per.month, per.basis) == ((2025, 11), "calendar")
    assert len(OB.horizon_games(november, now, per)) == 215
    pl = OB.plan(500, 0, 0, POINTS, november, {}, per, "ok", BUSIEST, now=now, period_games=november)
    assert pl.pulls_per_game == 2 and pl.per_pull == 1
    per_day = (FREE - 25) / 30
    assert round(per_day, 1) == 15.8 and int(per_day) == len(DAYS[day(11, 2)]) == 15
    assert len(DAYS[day(11, 2)]) * pl.pulls_per_game * pl.per_pull == 30
    assert len(DAYS[day(11, 5)]) * pl.pulls_per_game * pl.per_pull == 4


# -- MUTATIONS (8.2) ------------------------------------------------------------------
def test_the_reserve_is_never_crossed():
    """MUTATION. Reserve 30 (BUCKETVALUE_ODDS_RESERVE) on an account with one
    game left: 30 remaining spends nothing, 31 spends once. The plan (B = 0)
    and the floor (at F = 0 it is the same sum) would stop the call too, so
    the refusal must name the reserve itself: a broken reserve check fails
    here by its reason."""
    g = DAYS[day(1, 20)][0]
    env = {"BUCKETVALUE_ODDS_RESERVE": "30"}
    assert O.read_settings(env).reserve == 30
    assert Replay([g], used=470, env=env).run(et(1, 20, 8), et(1, 21)).calls == []
    once = Replay([g], used=469, env=env).run(et(1, 20, 8), et(1, 21))
    assert [(t, k, kind) for t, _, k, kind in once.calls] == [(et(1, 20, 8, 50), 1, "first")]
    assert (once.remaining, once.used) == (30, 470)
    now = et(1, 20, 8, 50)
    st = O.read_settings(env)
    per = OB.period_of(now, 469, FREE, 30, 1, [g], busiest=BUSIEST)
    pl = OB.plan(31, 469, 0, st, [g], {}, per, "ok", BUSIEST, now=now, period_games=[g])
    assert (pl.mode, pl.pulls_per_game, pl.reserve) == ("full", 1, 30)
    assert spend(31, 469, pl, now=now, tip=g.tip_utc) is None
    assert spend(30, 470, pl, now=now, tip=g.tip_utc) == "reserve"               # one credit later
    assert spend(31, 469, pl, now=now, tip=g.tip_utc, pending=1) == "reserve"    # one in flight
    assert spend(35, 465, pl, now=now, tip=g.tip_utc, cost=6) == "reserve"       # a bigger pull
    assert OB.plan(30, 470, 0, st, [g], {}, per, "ok", BUSIEST, now=now,
                   period_games=[g]).reason == "at_reserve"


def test_the_morning_pull_floor_keeps_later_games_first_pulls():
    """MUTATION. 18:30 on Jan 30, the period's last evening. The walk's plan,
    made with 50 credits left, has the 19:00 game due its evening refresh.
    Before the pull, another program on a shared key (RC B1) spends
    17: 33 left, S = 25, and tomorrow's 8 games wait for their first pull. The
    floor reads the vendor's remaining and refuses the refresh; tomorrow every
    game gets its morning pull. Without the floor the refresh takes the
    credit and one of tomorrow's games gets no line."""
    now = et(1, 30, 18, 30)
    rp = Replay(PERIOD_JAN, used=450)
    for g in DAYS[day(1, 30)]:                   # the morning's pulls, and the early games' refreshes
        tipped = g.tip_utc <= now
        last = g.tip_utc - POINTS.last_pull_lead if tipped else et(1, 30, 8, 50)
        rp.slots[g.game_id] = slot_of(g, pulls=2 if tipped else 1, first_at=et(1, 30, 8, 50), last_at=last)
    per = OB.period_of(now, 450, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    stale = OB.plan(50, 450, 0, POINTS, PERIOD_JAN, rp.slots, per, "ok", BUSIEST, now=now,
                    period_games=PERIOD_JAN)
    assert stale.pulls_per_game == 2                                     # floor((25 + 5) / 13)
    seven = DAYS[day(1, 30)][3]
    assert seven.tip_utc == et(1, 30, 19).astimezone(UTC)
    assert OB.due_kind(rp.slots[seven.game_id], stale, POINTS, now, seven.game_date) == "refresh"
    rp.used, rp.remaining = 467, 33                                      # the other process's 17
    assert OB.waiting_first(PERIOD_JAN, rp.slots, now, per, seven.game_id) == 8
    assert spend(33, 467, stale, now=now, tip=seven.tip_utc, waiting=8) == "floor"
    assert spend(34, 466, stale, now=now, tip=seven.tip_utc, waiting=8) is None    # one more is enough
    rp.view(now, plan=stale)
    rp.run(now + STEP, et(2, 1))
    assert [c for c in rp.calls if c[0].date() == day(1, 30)] == []      # no refresh tonight
    tomorrow = [c for c in rp.calls if c[0].date() == day(1, 31)]
    assert [kind for *_, kind in tomorrow] == ["first"] * 8
    assert {gid for _, gid, _, _ in tomorrow} == {g.game_id for g in DAYS[day(1, 31)]}
    assert rp.remaining == 25


@pytest.mark.parametrize("kind", ["first", "refresh", "pp"])
def test_no_game_is_due_inside_the_tip_guard(kind):
    """MUTATION, part 1 (part 2, live's own guard, is tests/test_live_feed.py).
    A 7 PM game set up to be due as `kind` is due to the last instant before
    6:55 PM and never from 6:55 on; may_spend refuses from the same instant."""
    g = DAYS[day(1, 7)][0]
    tip = et(1, 7, 19)
    assert g.tip_utc == tip.astimezone(UTC)
    morning = et(1, 7, 8, 50)
    setups = {
        "first": (slot_of(g), plan_with(2)),
        "refresh": (slot_of(g, pulls=1, first_at=morning, last_at=morning), plan_with(2)),
        "pp": (slot_of(g, pulls=2, first_at=morning, last_at=et(1, 7, 18, 45), pp_missing=True,
                       pp_next=et(1, 7, 18, 46)), plan_with(3)),
    }
    slot, pl = setups[kind]
    guard = tip - POINTS.tip_guard
    assert guard == et(1, 7, 18, 55)
    just_before = guard - dt.timedelta(microseconds=1)
    assert OB.due_kind(slot, pl, POINTS, just_before, g.game_date) == kind
    assert spend(412, 88, pl, now=just_before, tip=g.tip_utc, kind=kind) is None
    t = guard
    while t <= tip + dt.timedelta(minutes=15):
        assert OB.due_kind(slot, pl, POINTS, t, g.game_date) is None, t
        assert spend(412, 88, pl, now=t, tip=g.tip_utc, kind=kind) == "tip_guard", t
        t += dt.timedelta(seconds=20)


def test_one_tip_rule_everywhere():
    """The governor's guard is the projections' freeze lead (pipeline.FREEZE_LEAD
    = horizons.TIP_FLOOR): the feed stops when the projections freeze. due_kind
    and may_spend both read before_tip, so they agree at every instant."""
    assert POINTS.tip_guard == pipeline.FREEZE_LEAD == horizons.TIP_FLOOR == dt.timedelta(minutes=5)
    g = DAYS[day(1, 7)][0]
    t = et(1, 7, 18, 50)
    while t <= et(1, 7, 19, 1):
        before = OB.before_tip(t, g.tip_utc, POINTS.tip_guard)
        assert before == (t < et(1, 7, 18, 55))
        assert (OB.due_kind(slot_of(g), plan_with(2), POINTS, t, g.game_date) == "first") == before
        assert (spend(412, 88, plan_with(2), now=t, tip=g.tip_utc) == "tip_guard") == (not before)
        t += dt.timedelta(seconds=15)
    assert not OB.before_tip(et(1, 7, 9), None, POINTS.tip_guard)       # an unknown tip never is


# -- Due times (3.4) --------------------------------------------------------------------
def test_refreshes_are_spaced_to_the_last_pull_lead():
    """3.9: a 7:30 PM tip, first pull 08:50, last 7:00 PM. Free, P = 2: 08:50
    and 7:00 PM. $30, P = 7: every 1 hour 41 minutes 40 seconds."""
    g = OB.Game("x", day(1, 7), et(1, 7, 19, 30).astimezone(UTC))
    first = et(1, 7, 8, 50)
    s = slot_of(g, pulls=1, first_at=first, last_at=first)
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 19) - SECOND, g.game_date) is None
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 19), g.game_date) == "refresh"
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 19, 24), g.game_date) == "refresh"
    step = dt.timedelta(hours=1, minutes=41, seconds=40)
    assert first + 6 * step == et(1, 7, 19)
    for k in range(1, 7):
        due = first + k * step
        assert OB.due_kind(s, plan_with(7), POINTS, due - SECOND, g.game_date) is None, k
        assert OB.due_kind(s, plan_with(7), POINTS, due, g.game_date) == "refresh", k
        s = replace(s, pulls=k + 1, last_at=due)
    assert OB.due_kind(s, plan_with(7), POINTS, et(1, 7, 19, 20), g.game_date) is None   # all 7 made


def test_a_late_first_pull_spaces_the_rest_from_itself_and_15_minutes_is_the_floor():
    g = OB.Game("x", day(1, 12), et(1, 12, 19, 30).astimezone(UTC))
    late = et(1, 12, 14)
    s = slot_of(g, pulls=1, first_at=late, last_at=late)                # RP case H: P = 3 from 2 PM
    assert OB.due_kind(s, plan_with(3), POINTS, et(1, 12, 16, 30) - SECOND, g.game_date) is None
    assert OB.due_kind(s, plan_with(3), POINTS, et(1, 12, 16, 30), g.game_date) == "refresh"
    s = replace(s, pulls=2, last_at=et(1, 12, 16, 30))
    assert OB.due_kind(s, plan_with(3), POINTS, et(1, 12, 19) - SECOND, g.game_date) is None
    assert OB.due_kind(s, plan_with(3), POINTS, et(1, 12, 19), g.game_date) == "refresh"
    # P = 48 from a 6 PM first pull would space pulls 77 seconds apart: 15 minutes is the floor
    six = et(1, 12, 18)
    s = slot_of(g, pulls=1, first_at=six, last_at=six)
    assert OB.due_kind(s, plan_with(48), POINTS, et(1, 12, 18, 15) - SECOND, g.game_date) is None
    assert OB.due_kind(s, plan_with(48), POINTS, et(1, 12, 18, 15), g.game_date) == "refresh"
    # and never past t_last: after a pull at 6:50 PM the next would be 7:05, past 7:00
    s = replace(s, pulls=5, last_at=et(1, 12, 18, 50))
    for m in range(0, 70):
        now = et(1, 12, 18, 50) + dt.timedelta(seconds=30 * m)
        assert OB.due_kind(s, plan_with(48), POINTS, now, g.game_date) is None, now
    # a first pull after t_last leaves nothing to refresh (t_first when t_last <= t_first)
    s = slot_of(g, pulls=1, first_at=et(1, 12, 19, 10), last_at=et(1, 12, 19, 10))
    assert all(OB.due_kind(s, plan_with(4), POINTS, et(1, 12, 19, 10 + m), g.game_date) is None
               for m in range(15))


def test_the_first_pull_is_at_0850_on_the_games_own_date():
    g = DAYS[day(1, 7)][0]
    assert OB.first_pull_time(g.game_date, POINTS.first_pull_et) == et(1, 7, 8, 50)
    assert OB.due_kind(slot_of(g), plan_with(2), POINTS, et(1, 7, 8, 50) - SECOND, g.game_date) is None
    assert OB.due_kind(slot_of(g), plan_with(2), POINTS, et(1, 7, 8, 50), g.game_date) == "first"
    assert OB.due_kind(slot_of(g), plan_with(2), POINTS, et(1, 6, 21), g.game_date) is None   # the eve
    early = O.read_settings({"BUCKETVALUE_ODDS_FIRST_PULL_ET": "07:30"})
    assert OB.due_kind(slot_of(g), plan_with(2), early, et(1, 7, 7, 30), g.game_date) == "first"
    assert OB.due_kind(slot_of(g), plan_with(2), POINTS, et(1, 7, 9), None) == "first"   # date from the tip


def test_a_billed_failure_backs_the_game_off_from_90_seconds_to_60_minutes():
    """B2 items 2 to 4: 90 s, 3, 6, 12, 24 and 48 minutes, then 60 each time;
    due_kind reads fail_until for every kind; 3 billed failures in a row trip
    the day breaker, which due_kind reads too."""
    minutes = [OB.backoff(n).total_seconds() / 60 for n in range(0, 10)]
    assert minutes == [0, 1.5, 3, 6, 12, 24, 48, 60, 60, 60]
    assert OB.backoff(10_000) == OB.BACKOFF_MAX
    g = DAYS[day(1, 7)][0]
    morning = et(1, 7, 8, 50)
    cases = [(slot_of(g), plan_with(2), et(1, 7, 9), "first"),
             (slot_of(g, pulls=1, first_at=morning, last_at=morning), plan_with(2), et(1, 7, 18, 30),
              "refresh"),
             (slot_of(g, pulls=1, first_at=morning, last_at=morning, pp_missing=True, pp_next=et(1, 7, 9)),
              plan_with(2), et(1, 7, 9), "pp")]
    for s, pl, now, kind in cases:
        backing = replace(s, fails=1, fail_until=now + OB.backoff(1))
        assert OB.due_kind(backing, pl, POINTS, now, g.game_date) is None
        assert OB.due_kind(backing, pl, POINTS, now + OB.backoff(1) - SECOND, g.game_date) is None
        assert OB.due_kind(backing, pl, POINTS, now + OB.backoff(1), g.game_date) == kind
        assert OB.due_kind(s, pl, POINTS, now, g.game_date, breaker_day=day(1, 7)) is None
        assert OB.due_kind(s, pl, POINTS, now, g.game_date, breaker_day=day(1, 6)) == kind
    assert [OB.breaker_trips(n) for n in range(5)] == [False, False, False, True, True]
    assert OB.breaker_on(day(1, 7), et(1, 7, 23, 59)) and not OB.breaker_on(day(1, 7), et(1, 8))


def test_an_empty_answer_is_retried_in_10_minutes_and_is_not_a_pull():
    """An answer with no bookmakers is free (UT/docs_v4.txt:1637-1638): the
    game keeps pulls 0, so it still waits for its first pull (the floor's F)
    and adds nothing to the pulls made (P's made)."""
    g = DAYS[day(1, 7)][0]
    empty = slot_of(g, empty_next=et(1, 7, 8, 50) + OB.EMPTY_RETRY)
    assert OB.due_kind(empty, plan_with(2), POINTS, et(1, 7, 9) - SECOND, g.game_date) is None
    assert OB.due_kind(empty, plan_with(2), POINTS, et(1, 7, 9), g.game_date) == "first"
    now = et(1, 7, 9)
    per = OB.period_of(now, 88, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    assert OB.waiting_first(PERIOD_JAN, {g.game_id: empty}, now, per) == 192
    assert OB.waiting_first(PERIOD_JAN, {g.game_id: empty}, now, per, g.game_id) == 191
    pulled = slot_of(g, pulls=1, first_at=et(1, 7, 8, 50), last_at=et(1, 7, 8, 50))
    assert OB.waiting_first(PERIOD_JAN, {g.game_id: pulled}, now, per, "another") == 191
    with_empty = OB.plan(412, 88, 0, POINTS, PERIOD_JAN, {g.game_id: empty}, per, "ok", BUSIEST, now=now,
                         period_games=PERIOD_JAN)
    without = OB.plan(412, 88, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now,
                      period_games=PERIOD_JAN)
    assert with_empty == without
    # an empty refresh waits its 10 minutes too
    morning = et(1, 7, 8, 50)
    s = slot_of(g, pulls=1, first_at=morning, last_at=morning, empty_next=et(1, 7, 18, 40))
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 18, 35), g.game_date) is None
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 18, 40), g.game_date) == "refresh"


def test_the_prizepicks_only_poll_comes_every_10_minutes_and_counts_as_a_pull():
    """M4. A pull without PrizePicks marks the game; the PrizePicks-only poll
    is due every 10 minutes after (free while PrizePicks has nothing). The
    first non-empty answer counts as a pull, so on the free plan (P = 2) it
    takes the place of the evening refresh (RP case F)."""
    g = DAYS[day(1, 7)][0]                                             # a 7 PM tip
    morning = et(1, 7, 8, 50)
    s = slot_of(g, pulls=1, first_at=morning, last_at=morning, pp_missing=True,
                pp_next=morning + OB.PP_RETRY)
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 9) - SECOND, g.game_date) is None
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 9), g.game_date) == "pp"
    assert OB.due_kind(replace(s, pp_next=et(1, 7, 9, 10)), plan_with(2), POINTS, et(1, 7, 9, 5),
                       g.game_date) is None
    off = O.read_settings({"BUCKETVALUE_ODDS_PP_TOPUP": "0"})
    assert OB.due_kind(s, plan_with(2), off, et(1, 7, 9), g.game_date) is None
    assert OB.due_kind(s, plan_with(2), off, et(1, 7, 18, 30), g.game_date) == "refresh"
    # a due refresh comes first: the same cost, and it asks PrizePicks too
    assert OB.due_kind(s, plan_with(2), POINTS, et(1, 7, 18, 30), g.game_date) == "refresh"
    topped = replace(s, pulls=2, last_at=et(1, 7, 11), pp_missing=False, pp_next=None)
    t = et(1, 7, 11)
    while t < et(1, 7, 19, 10):
        assert OB.due_kind(topped, plan_with(2), POINTS, t, g.game_date) is None, t
        t += STEP
    assert OB.due_kind(s, plan_with(1, mode="short"), POINTS, et(1, 7, 9), g.game_date) is None


def test_a_tbd_schedule_tip_is_ignored_m2():
    """The schedule's tip for a TBD row is a made-up 00:00 ET that reads as
    tipped all day (live.time_tbd); the vendor's commence_time is the tip."""
    made_up, vendor = et(1, 7, 0, 0), et(1, 7, 19, 30)
    assert OB.tip_for(None, made_up, True) is None
    assert OB.tip_for(vendor, made_up, True) == vendor
    assert OB.tip_for(vendor, et(1, 7, 19), False) == et(1, 7, 19)       # the earlier of the two
    assert OB.tip_for(et(1, 7, 19), vendor, False) == et(1, 7, 19)
    assert OB.tip_for(None, et(1, 7, 19), float("nan")) == et(1, 7, 19)  # a missing flag is not TBD
    assert OB.tip_for(pd.NaT, pd.Timestamp("2026-01-08T00:00:00Z"), False) == et(1, 7, 19)
    assert OB.tip_for("2026-01-08T00:30:00Z", None, None) == et(1, 7, 19, 30)
    assert OB.tip_for(None, None, False) is None
    tbd = OB.Game("tbd", day(1, 7), None)
    right = OB.Slot("e", "tbd", OB.tip_for(vendor, made_up, True), has_lines=False)
    wrong = OB.Slot("e", "tbd", made_up.astimezone(UTC), has_lines=False)
    assert OB.due_kind(right, plan_with(2), POINTS, et(1, 7, 8, 50), day(1, 7)) == "first"
    assert OB.due_kind(wrong, plan_with(2), POINTS, et(1, 7, 8, 50), day(1, 7)) is None
    # a game whose time is still TBD counts as untipped in G until its day is over
    now = et(1, 7, 9)
    per = OB.period_of(now, 88, FREE, None, 1, [tbd], busiest=BUSIEST)
    assert OB.horizon_games([tbd], now, per) == [tbd]
    assert OB.horizon_games([tbd], et(1, 8, 9), per) == []


def test_naive_instants_are_refused_and_iso_strings_read():
    with pytest.raises(ValueError):
        OB.period_of(dt.datetime(2026, 1, 7, 12), 0, FREE, None, 1, PERIOD_JAN)
    with pytest.raises(ValueError):
        OB.before_tip(et(1, 7, 9), dt.datetime(2026, 1, 8, 0), POINTS.tip_guard)
    assert OB.before_tip("2026-01-07T14:00:00Z", "2026-01-08T00:00:00+00:00", POINTS.tip_guard)
    with pytest.raises(TypeError):
        OB.plan(412, 88, 0, POINTS, PERIOD_JAN, {}, OB.period_of(et(1, 7, 9), 88, FREE, None, 1,
                PERIOD_JAN), "ok", BUSIEST, period_games=PERIOD_JAN)   # `now` is required


# -- The guards on every paid call (3.6) ---------------------------------------------
def test_may_spend_names_the_first_guard_that_refuses_in_the_designs_order():
    g = DAYS[day(1, 7)][0]
    now, late = et(1, 7, 9), et(1, 7, 18, 56)
    pl = OB.plan(412, 88, 0, POINTS, PERIOD_JAN, {}, OB.period_of(now, 88, FREE, None, 1, PERIOD_JAN),
                 "ok", BUSIEST, now=now, period_games=PERIOD_JAN)
    full_minute = [now - dt.timedelta(seconds=10 * i) for i in range(40)]
    assert spend(None, None, None, now=late, tip=g.tip_utc, auth="bad_key", calls=full_minute) == "tip_guard"
    assert spend(None, None, None, now=now, tip=g.tip_utc, auth="bad_key") == "bad_key"
    assert spend(None, None, None, now=now, tip=g.tip_utc, auth="out_of_credits") == "out_of_credits"
    assert spend(None, None, pl, now=now, tip=g.tip_utc) == "no_balance"
    assert spend(412, 88, plan_with(0, mode="paused"), now=now, tip=g.tip_utc) == "paused"
    assert spend(25, 475, pl, now=now, tip=g.tip_utc, waiting=9, calls=full_minute) == "reserve"
    assert spend(30, 470, pl, now=now, tip=g.tip_utc, waiting=9, calls=full_minute) == "floor"
    assert spend(300, 200, pl, now=now, tip=g.tip_utc, calls=full_minute) == "ceiling"
    assert spend(412, 88, pl, now=now, tip=g.tip_utc, calls=full_minute) == "rate_cap"
    assert spend(412, 88, pl, now=now, tip=g.tip_utc, calls=full_minute[:39]) is None
    assert set(OB.REFUSALS) == {"tip_guard", "out_of_credits", "bad_key", "no_balance", "paused",
                                "reserve", "floor", "ceiling", "rate_cap"}


def test_the_rate_cap_counts_paid_calls_in_the_last_10_minutes():
    """BUCKETVALUE_ODDS_MAX_CALLS_10MIN (40): fewer than 40 paid calls in any
    10 minutes. A call exactly 10 minutes old no longer counts."""
    g = DAYS[day(1, 7)][0]
    now = et(1, 7, 9)
    pl = plan_with(2)
    old = [now - OB.RATE_WINDOW] * 5
    recent = [now - dt.timedelta(minutes=9, seconds=59)] * 39
    assert OB.calls_in_window(old + recent, now) == 39
    assert spend(412, 88, pl, now=now, tip=g.tip_utc, calls=old + recent) is None
    assert spend(412, 88, pl, now=now, tip=g.tip_utc, calls=old + recent + [now]) == "rate_cap"
    three = O.read_settings({"BUCKETVALUE_ODDS_MAX_CALLS_10MIN": "3"})
    assert OB.may_spend(412, 88, 0, 1, pl, 0, 1, now=now, tip=g.tip_utc, guard=three.tip_guard,
                        auth="ok", calls=[now] * 3, max_calls=three.max_calls_10min,
                        kind="refresh") == "rate_cap"


def test_may_spend_takes_every_guard_as_a_required_argument():
    """A guard nobody passes is no guard: called with design 2.3's seven
    positional values only, may_spend refuses to run at all. B1 adds the
    call's kind, required too, and a kind that is not due_kind's word is
    refused rather than read as one."""
    params = inspect.signature(OB.may_spend).parameters
    required = [n for n, p in params.items() if p.kind is p.KEYWORD_ONLY and p.default is p.empty]
    assert required == ["now", "tip", "guard", "auth", "calls", "max_calls", "kind"]
    with pytest.raises(TypeError):
        OB.may_spend(412, 88, 0, 1, plan_with(2), 0, 1)
    g = DAYS[day(1, 7)][0]
    with pytest.raises(TypeError):
        OB.may_spend(412, 88, 0, 1, plan_with(2), 0, 1, now=et(1, 7, 9), tip=g.tip_utc,
                     guard=POINTS.tip_guard, auth="ok", calls=[], max_calls=40)
    for bad in ("First", "topup", None, ""):
        with pytest.raises(ValueError):
            spend(412, 88, plan_with(2), now=et(1, 7, 9), tip=g.tip_utc, kind=bad)
    assert set(OB.KINDS) == {"first", "refresh", "pp"}


# -- The memory-loss brake (3.4; B1 item 2, M1) ---------------------------------------
def test_the_brake_counts_todays_games_as_pulled_after_0950_when_the_period_was_spending():
    """On Jan 7 the period's 53 earlier games (Dec 31's 9, Jan 1-6's 44) and
    100 credits used say a feed was running. A ledger born after 09:50 treats
    the day's games as pulled once at its birth. A slot with lines, or one
    the brake did not mark, keeps the normal spacing: its next pull is the
    evening refresh. A braked game with no lines in the ledger (every braked
    game, at its birth) keeps one re-buy at once (build amendment B21, which
    replaced the deferral to the evening refresh; the next test).

    RP case D's three afternoon restarts, as live keeps the ledger since
    B21: the 13:00 ledger re-buys Jan 7's twelve games at its birth; that
    spent the evening refreshes' credits, so the 14:00 and 15:00 ledgers
    plan P = 1 and buy nothing more until 20:30, when the tips that passed
    lift P to 2 for the 15:00 ledger's three late games, which are re-bought
    then. 27 credits on Jan 7, where the brake alone, before B21, spent 24."""
    first = OB.first_pull_time(day(1, 7), POINTS.first_pull_et)
    per = OB.period_of(et(1, 7, 13), 100, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    before = OB.games_before(PERIOD_JAN, per, day(1, 7))
    assert (first, before) == (et(1, 7, 8, 50), 53)
    assert OB.brake_applies(et(1, 7, 13), False, 100, 1, before, first)
    assert not OB.brake_applies(et(1, 7, 9, 50), False, 100, 1, before, first)    # not MORE than 60 min
    assert OB.brake_applies(et(1, 7, 9, 50, 1), False, 100, 1, before, first)
    assert not OB.brake_applies(et(1, 7, 13), False, 26, 1, before, first)        # 26 < 0.5 x 53
    assert OB.brake_applies(et(1, 7, 13), False, 27, 1, before, first)
    born = et(1, 7, 13)
    braked = {g.game_id: slot_of(g, pulls=1, first_at=born, last_at=born) for g in DAYS[day(1, 7)]}
    pl = OB.plan(400, 100, 0, POINTS, PERIOD_JAN, braked, per, "ok", BUSIEST, now=born,
                 period_games=PERIOD_JAN)
    assert pl.pulls_per_game == 2
    seven = DAYS[day(1, 7)][0]
    assert OB.due_kind(braked[seven.game_id], pl, POINTS, born, seven.game_date) is None
    assert OB.due_kind(braked[seven.game_id], pl, POINTS, et(1, 7, 18, 30), seven.game_date) == "refresh"
    marked = replace(braked[seven.game_id], braked=True, has_lines=False)       # as live marks it
    assert OB.due_kind(marked, pl, POINTS, born, seven.game_date) == OB.REBUY
    # RP case D: three restarts in the afternoon
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 8),
                                restarts=[et(1, 7, 13), et(1, 7, 14), et(1, 7, 15)])
    assert spend_on(rp, day(1, 7)) == 27
    afternoon = [(t, gid, kind) for t, gid, _, kind in rp.calls if t.date() == day(1, 7) and t.hour >= 12]
    assert [(t, kind) for t, _, kind in afternoon] == [(et(1, 7, 13), "rebuy")] * 12 + [
        (et(1, 7, 20, 30), "rebuy")] * 3
    assert [gid for t, gid, _ in afternoon[12:]] == [g.game_id for g in DAYS[day(1, 7)][-3:]]
    assert pulls_per_game(rp, DAYS[day(1, 7)]) == {2: 9, 3: 3}


def test_the_brake_stays_off_on_go_live_day_rp_case_h():
    """A key added at 2 PM on Jan 12 to an account with 15 credits used: the
    month was unused (15 < 0.5 x 93), so the day is bought at once, P = 3."""
    first = et(1, 12, 8, 50)
    per = OB.period_of(et(1, 12, 14), 15, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    before = OB.games_before(PERIOD_JAN, per, day(1, 12))
    assert before == 93
    assert not OB.brake_applies(et(1, 12, 14), False, 15, 1, before, first)
    pl = OB.plan(485, 15, 0, POINTS, PERIOD_JAN, {}, per, "ok", BUSIEST, now=et(1, 12, 14),
                 period_games=PERIOD_JAN)
    assert pl.pulls_per_game == 3                                        # floor(460 / 152)
    rp = Replay(PERIOD_JAN, used=15, born=et(1, 12, 14)).run(et(1, 12, 14), et(1, 13))
    assert len(rp.calls) == 24
    assert pulls_per_game(rp, DAYS[day(1, 12)]) == {3: 8}
    assert [t for t, *_ in rp.calls[:8]] == [et(1, 12, 14)] * 8


def test_the_brake_always_applies_after_a_version_change():
    """had_ledger: this process had a ledger before. On the first day of a
    period, with nothing used, a late-born ledger still brakes; inside the
    hour after the first-pull time it never does."""
    first = et(1, 1, 8, 50)
    assert OB.brake_applies(et(1, 1, 10), True, 0, 1, 0, first)
    assert OB.brake_applies(et(1, 1, 10), True, None, 1, 0, first)
    assert not OB.brake_applies(et(1, 1, 9, 50), True, 0, 1, 0, first)
    assert not OB.brake_applies(et(1, 1, 10), False, 500, 1, 0, first)   # no earlier game: no usage test
    assert not OB.brake_applies(et(1, 7, 13), False, None, 1, 53, et(1, 7, 8, 50))   # used unknown
    assert not OB.brake_applies(None, True, 0, 1, 0, first)


# -- RP's cases (design 3.9), as literal numbers --------------------------------------
def test_rp_case_a_free_points_january_from_a_fresh_process_at_0800_on_jan_1():
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(2, 1))
    assert pulls_per_game(rp, JANUARY) == {2: 233, 3: 3}
    assert (rp.used, rp.remaining - 25) == (475, 0)
    assert (spend_on(rp, day(1, 1)), spend_on(rp, day(1, 7))) == (12, 24)


def test_rp_case_b_a_reset_at_0800_on_nov_30_and_a_fresh_process_at_1000():
    december = games(day(12, 1), day(12, 31))
    rp = Replay(games(day(11, 30), day(12, 31)), born=et(11, 30, 10)).run(et(11, 30, 10), et(1, 1))
    assert (spend_on(rp, day(11, 30)), spend_on(rp, day(12, 1))) == (14, 16)
    assert pulls_per_game(rp, december) == {2: 60, 3: 107, 4: 5}
    assert max(t for t, *_ in rp.calls) == et(12, 31, 20, 30)


def test_rp_case_c_470_used_at_1000_on_dec_31_and_the_reset_at_1900():
    """Dec 31 costs 5 and Jan 1 12. The 470 used say a feed was running, so
    the 10:00 ledger brakes Dec 31's games as pulled at its birth. With 5
    above the reserve for 9 games the plan is P = 1 until the 3 PM tips:
    then P = 2 for the five games still ahead, and as braked games with no
    lines each gets its one re-buy at once (build amendment B21), from the
    old balance's 5 above the reserve, before the reset. (Before B21 the
    same 5 credits bought their evening refreshes, the three late games'
    after the reset.) The reset at 7 PM leaves them their two pulls."""
    rp = Replay(games(day(11, 30), day(1, 31)), used=470, born=et(12, 31, 10)).run(
        et(12, 31, 10), et(1, 2), reset_at=et(12, 31, 19))
    assert (spend_on(rp, day(12, 31)), spend_on(rp, day(1, 1))) == (5, 12)
    dec31 = [(t, gid, kind) for t, gid, _, kind in rp.calls if t.date() == day(12, 31)]
    assert dec31 == [(et(12, 31, 15), g.game_id, "rebuy") for g in DAYS[day(12, 31)][4:]]
    assert [g.tip_utc for g in DAYS[day(12, 31)][4:]] == [
        et(12, 31, h).astimezone(UTC) for h in (16, 18, 20, 20, 21)]


@pytest.mark.parametrize("label, times, cost", [
    ("three in the afternoon", ((13, 0), (14, 0), (15, 0)), 27),
    ("five between 09:00 and 09:40", ((9, 0), (9, 10), (9, 20), (9, 30), (9, 40)), 72),
    ("one at 15:00", ((15, 0),), 24),
])
def test_rp_case_d_restarts_on_jan_7(label, times, cost):
    """3.7, as live keeps the ledger since build amendment B21: an afternoon
    restart re-buys today's games once, at its birth, in place of their
    evening refreshes (one at 15:00 costs 24, the same as no restart; three
    cost 27, see the test of the brake above), where the brake alone made it
    free; five inside the brake's hour cost 48 more, all in the morning,
    while the ceiling and the floor hold and every later game keeps its
    morning pull (Jan 8's eight here)."""
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 9), restarts=[et(1, 7, h, m) for h, m in times])
    assert spend_on(rp, day(1, 7)) == cost, label
    if label == "one at 15:00":                                   # B21: the re-buys, no evening pull
        assert [(t, kind) for t, _, _, kind in rp.calls if t.date() == day(1, 7) and t.hour >= 12] == [
            (et(1, 7, 15), "rebuy")] * 12
    if cost == 72:
        assert all(t.hour < 12 for t, *_ in rp.calls if t.date() == day(1, 7))
        assert sum(spend_on(rp, day(1, d)) for d in range(1, 8)) <= OB.spending_ceiling(
            500, 25, PERIOD_JAN, day(1, 7))
    firsts = {gid for t, gid, _, kind in rp.calls if t == et(1, 8, 8, 50) and kind == "first"}
    assert firsts == {g.game_id for g in DAYS[day(1, 8)]}


def test_rp_case_d_with_no_restart_jan_7_costs_24():
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 8))
    assert spend_on(rp, day(1, 7)) == 24
    assert pulls_per_game(rp, DAYS[day(1, 7)]) == {2: 12}


def test_rp_case_e_a_vendor_that_bills_junk_all_day_gets_3_paid_calls():
    """B2: three billed failures in a row trip the day breaker. Inside the
    08:50 walk, due_kind reads the breaker: the 9 other games are not bought."""
    rp = Replay(PERIOD_JAN, junk=True).run(et(1, 7, 8), et(1, 8))
    assert len(rp.calls) == 3
    assert [t for t, *_ in rp.calls] == [et(1, 7, 8, 50)] * 3
    assert rp.breaker_day == day(1, 7)


def test_rp_case_f_prizepicks_posts_at_1100_every_day():
    """Every game gets its 08:50 pull and one PrizePicks top-up: 236 first
    pulls, 235 top-ups, 4 evening refreshes in the month; 475 spent."""
    rp = Replay(PERIOD_JAN, pp_post=dt.time(11, 0)).run(et(1, 1, 8), et(2, 1))
    assert by_kind(rp) == {"first": 236, "pp": 235, "refresh": 4}
    assert pulls_per_game(rp, JANUARY) == {2: 233, 3: 3}
    assert rp.used == 475
    assert {t.time() for t, *_, kind in rp.calls if kind == "first"} == {dt.time(8, 50)}


def test_rp_case_g_30_dollars_twelve_markets_january():
    rp = Replay(PERIOD_JAN, allowance=20_000, env=TWELVE_ENV).run(et(1, 1, 8), et(2, 1))
    assert pulls_per_game(rp, JANUARY) == {5: 1, 6: 70, 7: 162, 8: 3}
    assert rp.used == 18_996
    assert spend_on(rp, day(1, 7)) == 864


def test_a_mid_month_drain_buys_each_days_games_first_in_tip_order_b1():
    """Build amendment B1 (review I1, F6). It replaces build 1's test of the
    rule as design 3.3 and 3.6 first worded it, which gave Jan 20 and Jan 21
    no lines at all.

    Another user of the key drains it to 100 left at 08:00 on Jan 20: B = 75
    for G = 92 games, so the plan is "short" (P = 1). The floor no longer
    makes one game's first pull wait to save credits for another game's
    first pull, so each morning's walk buys that day's games at 08:50, in
    tip order: Jan 20's ten and Jan 21's four come first, then every later
    day's, until the reserve stops the 76th. Every game is bought at most
    once, the balance never goes below the reserve (the Replay asserts it on
    every call), the month ends on the reserve, and the games left without
    lines are the month's LAST seventeen: Jan 29's 9:30 PM game, then every
    game of Jan 30 and Jan 31."""
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 20, 8))
    rp.used, rp.remaining = 400, 100                                     # the outside drain
    now = et(1, 20, 8)
    per = OB.period_of(now, 400, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    pl = OB.plan(100, 400, 0, POINTS, PERIOD_JAN, rp.slots, per, "ok", BUSIEST, now=now,
                 period_games=PERIOD_JAN)
    assert (pl.mode, pl.pulls_per_game, pl.reason) == ("short", 1, "below_core")
    waiting = OB.horizon_games(PERIOD_JAN, now, per)                    # in tip order
    assert len(waiting) == 92
    before = len(rp.calls)
    rp.run(now, et(2, 1))
    late = rp.calls[before:]
    bought = [gid for _, gid, _, _ in late]
    assert (len(late), len(set(bought)), rp.remaining) == (75, 75, 25)  # each game at most once
    assert {kind for *_, kind in late} == {"first"} and {k for _, _, k, _ in late} == {1}
    assert bought == [g.game_id for g in waiting[:75]]                   # in tip order, every one
    assert [t for t, *_ in late[:14]] == [et(1, 20, 8, 50)] * 10 + [et(1, 21, 8, 50)] * 4
    assert {t.time() for t, *_ in late} == {dt.time(8, 50)}             # each on its own morning
    per_day = {d: sum(1 for t, *_ in late if t.date() == day(1, d)) for d in range(20, 32)}
    assert per_day == {20: 10, 21: 4, 22: 11, 23: 6, 24: 7, 25: 8, 26: 7, 27: 9, 28: 8, 29: 5,
                       30: 0, 31: 0}
    unbought = [g for g in waiting if g.game_id not in set(bought)]
    assert unbought == waiting[75:] and len(unbought) == 17
    assert unbought[0] == DAYS[day(1, 29)][-1] and unbought[0].tip_utc == et(1, 29, 21, 30)
    assert rp.used == 475


def test_dec_31_at_1000_with_30_left_buys_first_pulls_at_once_in_tip_order_b1():
    """Build amendment B1, build 1's Dec 31 case: 470 used at 10:00 ET on Dec
    31, before the reset and with no brake (no ledger born after the morning
    with earlier spending to brake). The old period's last day plans B = 5
    over its own nine games: "short", P = 1. The first pulls go ahead at
    once, in tip order, until the reserve stops them: the 1 PM game, the
    three 3 PM games and the 4 PM one. Under the floor as design 3.6 first
    worded it, none went until enough of the day's games had tipped unbought
    (30 - 1 - 25 = 4 credits against 8 other waiting games)."""
    gs = games(day(11, 30), day(1, 31))
    now = et(12, 31, 10)
    per = OB.period_of(now, 470, FREE, None, 1, gs, busiest=BUSIEST)
    assert (per.month, per.basis) == ((2025, 12), "used_old")
    pl = OB.plan(30, 470, 0, POINTS, gs, {}, per, "ok", BUSIEST, now=now, period_games=gs)
    assert (pl.mode, pl.pulls_per_game, pl.reason) == ("short", 1, "below_core")
    rp = Replay(gs, used=470).run(now, et(12, 31, 18, 55))
    dec31 = DAYS[day(12, 31)]
    assert [(t, gid, k, kind) for t, gid, k, kind in rp.calls] == [
        (now, g.game_id, 1, "first") for g in dec31[:5]]
    assert [g.tip_utc for g in dec31[:5]] == [et(12, 31, h).astimezone(UTC) for h in (13, 15, 15, 15, 16)]
    assert (rp.used, rp.remaining) == (475, 25)


def test_the_floor_still_refuses_a_refresh_and_a_top_up_but_never_a_first_pull_b1():
    """B1: the floor protects first pulls from refreshes, PrizePicks top-ups
    and re-buys, exactly as before; it never makes a first pull wait. 33 left
    with 8 games waiting for their first pull: 33 - 1 - 25 = 7 < 8. Every
    other guard still applies to a first pull, in design 3.6's order."""
    g = DAYS[day(1, 30)][3]
    now = et(1, 30, 18, 30)
    pl = plan_with(2)                                      # reserve 25, ceiling 475
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="refresh") == "floor"
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="pp") == "floor"
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="first") is None
    for kind in OB.KINDS:                                  # one more credit: the floor holds for all
        assert spend(34, 466, pl, now=now, tip=g.tip_utc, waiting=8, kind=kind) is None, kind
    full = [now - dt.timedelta(seconds=10 * i) for i in range(40)]
    assert spend(25, 475, pl, now=now, tip=g.tip_utc, waiting=8, kind="first") == "reserve"
    assert spend(30, 475, pl, now=now, tip=g.tip_utc, waiting=8, kind="first") == "ceiling"
    assert spend(33, 400, pl, now=now, tip=g.tip_utc, waiting=8, kind="first", calls=full) == "rate_cap"
    assert spend(33, 467, pl, now=et(1, 30, 18, 55), tip=g.tip_utc, waiting=8,
                 kind="first") == "tip_guard"
    assert spend(33, 467, plan_with(0, mode="paused"), now=now, tip=g.tip_utc, waiting=8,
                 kind="first") == "paused"
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="first",
                 auth="out_of_credits") == "out_of_credits"


def test_a_first_pull_asks_its_full_markets_only_when_the_floor_holds_at_that_cost_b1():
    """B1's first_pull_markets: the plan's markets when remaining - pending -
    c_plan - S >= c_core x (the other waiting games), else the core markets;
    the two are equal when the plan's markets are the core (the default)."""
    twelve = OB.Plan(mode="full", markets=O.MAIN_MARKETS, per_pull=12, pulls_per_game=2, p_max=12,
                     reserve=25, ceiling=19_000, reason="ok")
    assert TWELVE.core == PTS and len(O.MAIN_MARKETS) == 12
    assert OB.first_pull_markets(twelve, TWELVE, 42, 0, 5) == O.MAIN_MARKETS    # 42 - 12 - 25 = 5
    assert OB.first_pull_markets(twelve, TWELVE, 41, 0, 5) == PTS
    assert OB.first_pull_markets(twelve, TWELVE, 43, 1, 5) == O.MAIN_MARKETS    # pending counts
    assert OB.first_pull_markets(twelve, TWELVE, 42, 1, 5) == PTS
    assert OB.first_pull_markets(twelve, TWELVE, 37, 0, 0) == O.MAIN_MARKETS    # nobody else waits
    assert OB.first_pull_markets(twelve, TWELVE, 36, 0, 0) == PTS               # the reserve itself
    assert OB.first_pull_markets(twelve, TWELVE, None, 0, 0) == PTS             # no balance known
    two_core = O.read_settings({**TWELVE_ENV, "BUCKETVALUE_ODDS_CORE_MARKETS":
                                "player_points,player_rebounds"})
    assert two_core.core == ("player_points", "player_rebounds")
    assert OB.first_pull_markets(twelve, two_core, 47, 0, 5) == O.MAIN_MARKETS   # 2 x 5 = 10
    assert OB.first_pull_markets(twelve, two_core, 46, 0, 5) == two_core.core
    for remaining in (0, 26, 30, 500):                     # points only: the same either way
        for waiting in (0, 8, 300):
            assert OB.first_pull_markets(plan_with(2), POINTS, remaining, 0, waiting) == PTS


# -- Build amendment B11: a first pull that may be a re-buy (stage S10) -----------------
def test_first_kind_is_first_only_when_the_ledger_saw_the_games_whole_morning_b11():
    """B11. On Jan 7 (the period's 53 earlier games; the first-pull time
    08:50) a ledger born at or before 08:50 has seen the game's whole
    morning, whatever was spent. One born later has seen it only when the
    period was not spending before it: no earlier ledger in this process,
    and 2 x used below c_core x the earlier games, or no earlier game at
    all. Otherwise its first pull may re-buy what the process before it
    bought: "rebuy". The usage test is the brake's own, so for a ledger born
    after the brake's window a re-buy is exactly a braked ledger."""
    first = et(1, 7, 8, 50)

    def kind(born, used, *, had=False, before=53, core=1):
        return OB.first_kind(born, had, used, core, before, first)
    for born in (et(1, 6, 21), et(1, 7, 6), first):
        assert kind(born, 100) == kind(born, 100, had=True) == kind(born, None) == "first", born
    assert kind(first + SECOND, 100) == "rebuy"                      # a second late: may re-buy
    assert (kind(et(1, 7, 9), 26), kind(et(1, 7, 9), 27)) == ("first", "rebuy")   # 52 < 53 <= 54
    assert kind(et(1, 7, 9), 0, had=True) == "rebuy"                 # an earlier ledger in this process
    assert kind(et(1, 7, 9), 500, before=0) == "first"               # the period's first day
    assert kind(et(1, 7, 9), None) == "first"                        # as in the brake: unknown never passes
    assert (kind(et(1, 7, 9), 52, core=2), kind(et(1, 7, 9), 53, core=2)) == ("first", "rebuy")
    late = et(1, 7, 13)                                              # after the brake's window
    for used in [None, *range(0, 60)]:
        for had in (False, True):
            assert OB.brake_applies(late, had, used, 1, 53, first) == (
                OB.first_kind(late, had, used, 1, 53, first) == "rebuy"), (used, had)
    assert OB.REBUY == "rebuy" and OB.SPEND_KINDS == ("first", "refresh", "pp", "rebuy")
    params = list(inspect.signature(OB.first_kind).parameters)
    assert params == list(inspect.signature(OB.brake_applies).parameters)


def test_may_spend_checks_the_floor_for_a_rebuy_and_every_other_guard_as_for_a_first_pull_b11():
    """B11: may_spend reads first_kind's "rebuy" too. 33 left with 8 games
    waiting for their first pull: 33 - 1 - 25 = 7 < 8, so a first pull goes
    ahead (B1) and a re-buy is refused by the floor, exactly as a refresh is.
    One more credit and the floor holds. Every other guard applies to a
    re-buy in design 3.6's order, as to a first pull."""
    g = DAYS[day(1, 30)][3]
    now = et(1, 30, 18, 30)
    pl = plan_with(2)                                      # reserve 25, ceiling 475
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="first") is None
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="rebuy") == "floor"
    assert spend(33, 467, pl, now=now, tip=g.tip_utc, waiting=8, kind="refresh") == "floor"
    assert spend(34, 466, pl, now=now, tip=g.tip_utc, waiting=8, kind="rebuy") is None
    full = [now - dt.timedelta(seconds=10 * i) for i in range(40)]
    assert spend(25, 475, pl, now=now, tip=g.tip_utc, waiting=0, kind="rebuy") == "reserve"
    assert spend(30, 475, pl, now=now, tip=g.tip_utc, waiting=0, kind="rebuy") == "ceiling"
    assert spend(34, 400, pl, now=now, tip=g.tip_utc, waiting=8, kind="rebuy", calls=full) == "rate_cap"
    assert spend(34, 466, pl, now=et(1, 30, 18, 55), tip=g.tip_utc, waiting=8,
                 kind="rebuy") == "tip_guard"
    assert spend(34, 466, plan_with(0, mode="paused"), now=now, tip=g.tip_utc, waiting=8,
                 kind="rebuy") == "paused"
    assert spend(34, 466, pl, now=now, tip=g.tip_utc, waiting=8, kind="rebuy",
                 auth="bad_key") == "bad_key"
    assert spend(None, 466, pl, now=now, tip=g.tip_utc, waiting=8, kind="rebuy") == "no_balance"
    for bad in ("Rebuy", "re-buy", "first "):
        with pytest.raises(ValueError):
            spend(34, 466, pl, now=now, tip=g.tip_utc, kind=bad)


def test_rp_case_ds_morning_restarts_are_re_buys_that_pass_the_floor_b11():
    """B11 in the replay, design 3.7 case D ("the ceiling and the floor
    held"). The five restarts between 09:00 and 09:40 on Jan 7 make ledgers
    born after the 08:50 first pull in a spending period (100 used against
    53 earlier games), so each of their pulls of Jan 7's games is a "rebuy",
    and the floor checks it. With about 400 left for the month's other games
    every one passes: 60 re-buys, and Jan 7 costs 72 as 3.7 says. The ledger
    born at 09:40 saw Jan 8's whole morning, so Jan 8's eight are bought at
    08:50 as first pulls."""
    restarts = [et(1, 7, 9, m) for m in (0, 10, 20, 30, 40)]
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 9), restarts=restarts)
    jan7 = [(t, kind) for t, _, _, kind in rp.calls if t.date() == day(1, 7)]
    assert spend_on(rp, day(1, 7)) == 72
    assert [kind for t, kind in jan7 if t == et(1, 7, 8, 50)] == ["first"] * 12
    assert [t for t, kind in jan7 if kind == "rebuy"] == [t for t in restarts for _ in range(12)]
    assert {kind for _, kind in jan7} == {"first", "rebuy"}
    jan8 = [(t, gid, kind) for t, gid, _, kind in rp.calls if t.date() == day(1, 8)]
    assert {gid for t, gid, kind in jan8 if t == et(1, 8, 8, 50) and kind == "first"} == {
        g.game_id for g in DAYS[day(1, 8)]}


# -- Build amendment B12: a morning failure is retried in minutes (stage S10) ------------
def test_a_game_whose_pulls_were_all_billed_failures_is_due_once_its_backoff_allows_b12():
    """B12. A 7 PM game whose 08:50 pull was a billed failure (a timeout:
    one pull, no lines, a 90 s backoff) is due again at 08:51:30, while it
    has pulls left, not at its spaced refresh. The same failure on a game
    with lines (a refresh that failed) keeps the spacing, as does the brake's
    assumed pull, which is no failure. An empty answer keeps its 10-minute
    wait, the day breaker still stops the day, the tip guard still holds and
    P is never exceeded."""
    g = DAYS[day(1, 7)][0]
    t0 = et(1, 7, 8, 50)
    back = t0 + OB.backoff(1)
    assert back == et(1, 7, 8, 51, 30)
    failed = slot_of(g, pulls=1, first_at=t0, last_at=t0, fails=1, fail_until=back, has_lines=False)
    assert OB.due_kind(failed, plan_with(2), POINTS, back - SECOND, g.game_date) is None
    assert OB.due_kind(failed, plan_with(2), POINTS, back, g.game_date) == "refresh"
    assert OB.due_kind(failed, plan_with(2), POINTS, et(1, 7, 12), g.game_date) == "refresh"
    t_last = et(1, 7, 18, 30)
    lined = replace(failed, has_lines=True)                    # lines from before: the spacing holds
    braked = slot_of(g, pulls=1, first_at=et(1, 7, 13), last_at=et(1, 7, 13), has_lines=False)
    for s, start in ((lined, back), (braked, et(1, 7, 13))):
        t = start
        while t < t_last:
            assert OB.due_kind(s, plan_with(2), POINTS, t, g.game_date) is None, (s, t)
            t += dt.timedelta(minutes=10)
        assert OB.due_kind(s, plan_with(2), POINTS, t_last, g.game_date) == "refresh"
    empty = replace(failed, empty_next=back + OB.EMPTY_RETRY)
    assert OB.due_kind(empty, plan_with(2), POINTS, back + OB.EMPTY_RETRY - SECOND, g.game_date) is None
    assert OB.due_kind(empty, plan_with(2), POINTS, back + OB.EMPTY_RETRY, g.game_date) == "refresh"
    assert OB.due_kind(failed, plan_with(2), POINTS, back, g.game_date, breaker_day=day(1, 7)) is None
    assert OB.due_kind(failed, plan_with(2), POINTS, et(1, 7, 18, 55), g.game_date) is None
    assert OB.due_kind(failed, plan_with(2, mode="paused"), POINTS, back, g.game_date) is None
    twice = replace(failed, pulls=2, fails=2, last_at=back, fail_until=back + OB.backoff(2))
    t = twice.fail_until
    while t < et(1, 7, 19, 10):                                # P = 2 made: due no more all day
        assert OB.due_kind(twice, plan_with(2), POINTS, t, g.game_date) is None, t
        t += dt.timedelta(minutes=5)
    assert OB.due_kind(twice, plan_with(3), POINTS, twice.fail_until, g.game_date) == "refresh"
    assert OB.due_kind(replace(twice, pulls=3), plan_with(3), POINTS, et(1, 7, 12), g.game_date) is None


def test_a_slot_has_no_default_for_has_lines_b12():
    """B12: has_lines has no default that could hide a missing value. It is
    required and keyword only, and only True or False is taken."""
    param = inspect.signature(OB.Slot).parameters["has_lines"]
    assert param.kind is param.KEYWORD_ONLY and param.default is param.empty
    tip = et(1, 7, 19).astimezone(UTC)
    with pytest.raises(TypeError):
        OB.Slot("e", "g", tip)
    for bad in (None, 0, 1, "False", float("nan")):
        with pytest.raises(TypeError):
            OB.Slot("e", "g", tip, has_lines=bad)
    assert OB.Slot("e", "g", tip, has_lines=False).has_lines is False
    assert replace(OB.Slot("e", "g", tip, has_lines=False), has_lines=True).has_lines is True


# -- Build 4's fix pass: the campaign's survivors OB09 and OB17, and review L2-3 and L2-5,
# pinned as built (both are open questions for Barrett) ---------------------------------------
def test_p_max_is_never_under_two_pulls_whatever_the_markets_cost_ob09():
    """Survivor OB09 (3.3): P_max = min(48, 2 x max(1, ...)): an allowance
    under one share of the busiest month for the markets still gives each
    game two pulls, never zero (which would make nothing ever due)."""
    assert OB.p_max(FREE, 25, 12, BUSIEST) == OB.p_max(FREE, 25, 3, BUSIEST) == 2
    three = O.read_settings({"BUCKETVALUE_ODDS_MARKETS": ",".join(PTS + ("player_rebounds",
                                                                          "player_assists")),
                             "BUCKETVALUE_ODDS_CORE_MARKETS": ",".join(PTS + ("player_rebounds",
                                                                               "player_assists"))})
    assert len(three.markets) == len(three.core) == 3
    now = et(1, 31, 8, 50)
    per = OB.period_of(now, 200, FREE, None, 3, PERIOD_JAN, busiest=BUSIEST)
    pl = OB.plan(300, 200, 0, three, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now,
                 period_games=PERIOD_JAN)
    assert (pl.mode, pl.p_max, pl.pulls_per_game) == ("full", 2, 2)


def test_the_reset_window_across_the_november_2026_clock_change_ob17():
    """Survivor OB17: W(November 2026) opens at 05:00 EDT on Oct 31 and closes
    at 08:00 EST on Nov 1, the morning the clocks went back: 28 hours, never
    a fixed 27 back from the close. Midnight on Nov 1 in every zone from
    UTC-12 to UTC+14 is inside it (a fixed past date)."""
    opens, closes = OB.reset_window(2026, 11)
    assert opens == dt.datetime(2026, 10, 31, 5, 0, tzinfo=ET) == dt.datetime(2026, 10, 31, 9, 0,
                                                                              tzinfo=UTC)
    assert closes == dt.datetime(2026, 11, 1, 8, 0, tzinfo=ET) == dt.datetime(2026, 11, 1, 13, 0,
                                                                              tzinfo=UTC)
    assert closes - opens == dt.timedelta(hours=28)
    for minutes in range(-12 * 60, 14 * 60 + 1, 15):
        zone = dt.timezone(dt.timedelta(minutes=minutes))
        assert opens <= dt.datetime(2026, 11, 1, 0, 0, tzinfo=zone) <= closes, minutes
    assert OB.in_reset_window(dt.datetime(2026, 10, 31, 5, 30, tzinfo=ET))
    assert not OB.in_reset_window(dt.datetime(2026, 10, 31, 4, 59, tzinfo=ET))


def test_an_outside_drain_early_in_the_month_still_buys_each_days_first_pulls_b23():
    """Build amendment B23 (build 4's review, L2-3, which pinned the ceiling
    holding these first pulls). B1's outside drain moved to Jan 7: another
    user of the key takes the free account to 100 left (400 used) at 08:00.
    The plan is short (P = 1) and B1 skips the floor for first pulls; the
    spending ceiling, which reads the account's `used`, is 190 on Jan 7,
    and it refused every first pull until it reached 405 on Jan 17, while
    75 credits sat above the reserve. A first pull by a ledger born at or
    before the game's first-pull time (here the replay's ledger, born before
    it started, which has seen every morning) can never re-buy, so it skips
    the ceiling now: the 08:50 walk buys Jan 7's twelve games in tip order,
    and each later morning its own games, until the reserve stops the 76th.
    Every game is bought at most once, the balance never goes below the
    reserve (the Replay asserts it on every call), and the month ends on
    it. The ceiling still holds a first pull of first_kind's usage-test
    clause, and every refresh, top-up and re-buy."""
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 7, 8))
    rp.used, rp.remaining = 400, 100                                     # the outside drain
    now = et(1, 7, 8, 50)
    per = OB.period_of(now, 400, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
    pl = OB.plan(100, 400, 0, POINTS, PERIOD_JAN, rp.slots, per, "ok", BUSIEST, now=now,
                 period_games=PERIOD_JAN)
    assert (pl.mode, pl.pulls_per_game, pl.ceiling, pl.reserve) == ("short", 1, 190, 25)
    g = DAYS[day(1, 7)][0]
    waiting = OB.waiting_first(PERIOD_JAN, rp.slots, now, per, g.game_id)

    def ask(kind, true_first=False):
        return OB.may_spend(100, 400, 0, 1, pl, waiting, 1, now=now, tip=g.tip_utc,
                            guard=POINTS.tip_guard, auth="ok", calls=[], max_calls=40, kind=kind,
                            true_first=true_first)
    assert ask("first", true_first=True) is None                         # B23
    assert ask("first") == "ceiling"                                    # the usage-test clause's
    assert ask("refresh") == ask("pp") == ask("rebuy") == "floor"       # the floor comes first
    born = et(1, 7, 9)                                                   # a key added at 09:00
    assert OB.first_kind(born, False, 0, 1, 53, OB.first_pull_time(day(1, 7), POINTS.first_pull_et)) \
        == "first" and not OB.true_first_pull(born, OB.first_pull_time(day(1, 7), POINTS.first_pull_et))
    queue = OB.horizon_games(PERIOD_JAN, et(1, 7, 8), per)                # in tip order
    before = len(rp.calls)
    rp.run(et(1, 7, 8), et(2, 1))
    late = rp.calls[before:]
    bought = [gid for _, gid, _, _ in late]
    assert (len(late), len(set(bought)), rp.remaining, rp.used) == (75, 75, 25, 475)
    assert {kind for *_, kind in late} == {"first"} and {k for _, _, k, _ in late} == {1}
    assert bought == [x.game_id for x in queue[:75]]                     # in tip order, every one
    assert [(t, gid) for t, gid, _, _ in late[:12]] == [(now, x.game_id) for x in DAYS[day(1, 7)]]
    assert {t.time() for t, *_ in late} == {dt.time(8, 50)}             # each on its own morning
    per_day = {d: sum(1 for t, *_ in late if t.date() == day(1, d)) for d in range(7, 18)}
    assert per_day == {7: 12, 8: 8, 9: 6, 10: 8, 11: 6, 12: 8, 13: 7, 14: 9, 15: 8, 16: 3, 17: 0}


def test_true_first_pull_is_first_kinds_first_clause_and_may_spend_reads_it_b23():
    """B23's surface. true_first_pull: the ledger was born at or before the
    game's first-pull time; an unknown instant is never one. first_kind
    says "first" for every such ledger. may_spend skips the ceiling only for
    a "first" passed as a true first pull, takes only True or False, and
    refuses a true first pull of any other kind; its default keeps the
    ceiling. Every other guard still holds a true first pull."""
    first = et(1, 7, 8, 50)
    for born in (et(1, 6, 21), et(1, 7, 6), first):
        assert OB.true_first_pull(born, first) and OB.first_kind(born, True, 500, 1, 53, first) == "first"
    assert not OB.true_first_pull(first + SECOND, first)
    assert not OB.true_first_pull(None, first) and not OB.true_first_pull(first, None)
    g = DAYS[day(1, 30)][3]
    now = et(1, 30, 18, 30)
    pl = plan_with(2)                                      # reserve 25, ceiling 475

    def ask(remaining, used, kind="first", **kw):
        return OB.may_spend(remaining, used, 0, 1, pl, 0, 1, now=now, tip=g.tip_utc,
                            guard=POINTS.tip_guard, auth=kw.pop("auth", "ok"),
                            calls=kw.pop("calls", []), max_calls=40, kind=kind, **kw)
    assert ask(30, 475) == "ceiling" and ask(30, 475, true_first=False) == "ceiling"
    assert ask(30, 475, true_first=True) is None
    assert ask(25, 475, true_first=True) == "reserve"
    assert ask(30, 475, true_first=True, auth="bad_key") == "bad_key"
    full = [now - dt.timedelta(seconds=10 * i) for i in range(40)]
    assert ask(30, 475, true_first=True, calls=full) == "rate_cap"
    for kind in ("refresh", "pp", "rebuy"):
        assert ask(30, 475, kind) == "ceiling", kind
        with pytest.raises(ValueError):
            ask(30, 475, kind, true_first=True)
    for bad in (None, 0, 1, "True"):
        with pytest.raises(TypeError):
            ask(30, 475, true_first=bad)
    param = inspect.signature(OB.may_spend).parameters["true_first"]
    assert param.kind is param.KEYWORD_ONLY and param.default is False


# -- Build amendment B21: a cold start after 09:50 still buys today's lines --------------
def test_a_braked_game_with_no_lines_keeps_one_re_buy_at_once_while_p_is_two_b21():
    """B21. A 7 PM game on Jan 7 that the brake counted as pulled at a 13:00
    birth (braked, one pull, no lines in this ledger) is due as a "rebuy" at
    once, while P >= 2 and before its last refresh time (6:30 PM, tip minus
    30): its one re-buy. With P = 1 nothing is due all day (the brake's
    pull is its only one). After the re-buy (two pulls, lines) it is never
    re-bought; from 6:30 PM the planned refresh is due anyway, and a re-buy
    that came back empty waits its 10 minutes. A slot with lines, one not
    braked, or one with a second pull already keeps the normal rules. The
    day breaker, a backoff and a plan that does not spend still stop it."""
    g = DAYS[day(1, 7)][0]
    born, t_last = et(1, 7, 13), et(1, 7, 18, 30)
    braked = slot_of(g, pulls=1, first_at=born, last_at=born, has_lines=False, braked=True)
    assert OB.due_kind(braked, plan_with(2), POINTS, born, g.game_date) == OB.REBUY
    assert OB.due_kind(braked, plan_with(5), POINTS, t_last - SECOND, g.game_date) == OB.REBUY
    t = born
    while t < et(1, 7, 19):
        assert OB.due_kind(braked, plan_with(1), POINTS, t, g.game_date) is None, t   # P = 1: never
        t += dt.timedelta(minutes=5)
    assert OB.due_kind(braked, plan_with(2), POINTS, t_last, g.game_date) == "refresh"   # t_last: planned
    rebought = replace(braked, pulls=2, last_at=born, has_lines=True)
    t = born
    while t < et(1, 7, 19):
        assert OB.due_kind(rebought, plan_with(2), POINTS, t, g.game_date) is None, t
        t += dt.timedelta(minutes=5)
    for s in (replace(braked, has_lines=True), replace(braked, braked=False),
              replace(braked, pulls=2)):
        assert OB.due_kind(s, plan_with(3), POINTS, born, g.game_date) is None, s
    empty = replace(braked, empty_next=born + OB.EMPTY_RETRY)
    assert OB.due_kind(empty, plan_with(2), POINTS, born + OB.EMPTY_RETRY - SECOND, g.game_date) is None
    assert OB.due_kind(empty, plan_with(2), POINTS, born + OB.EMPTY_RETRY, g.game_date) == OB.REBUY
    failed = replace(braked, fails=1, fail_until=born + OB.backoff(1))    # nothing received: no pull
    assert OB.due_kind(failed, plan_with(2), POINTS, born, g.game_date) is None
    assert OB.due_kind(failed, plan_with(2), POINTS, born + OB.backoff(1), g.game_date) == OB.REBUY
    assert OB.due_kind(braked, plan_with(2), POINTS, born, g.game_date, breaker_day=day(1, 7)) is None
    assert OB.due_kind(braked, plan_with(2, mode="paused"), POINTS, born, g.game_date) is None
    assert OB.due_kind(braked, plan_with(2), POINTS, et(1, 7, 18, 55), g.game_date) is None   # tip guard
    assert OB.REBUY in OB.SPEND_KINDS and OB.REBUY not in OB.KINDS


def test_a_slot_says_braked_only_as_a_bool_and_defaults_to_not_braked_b21():
    tip = et(1, 7, 19).astimezone(UTC)
    assert OB.Slot("e", "g", tip, has_lines=False).braked is False
    assert OB.Slot("e", "g", tip, has_lines=False, braked=True).braked is True
    for bad in (None, 0, 1, "True"):
        with pytest.raises(TypeError):
            OB.Slot("e", "g", tip, has_lines=False, braked=bad)
    param = inspect.signature(OB.Slot).parameters["braked"]
    assert param.kind is param.KEYWORD_ONLY and param.default is False


def test_a_crash_loop_after_0950_re_buys_todays_games_once_and_keeps_the_floors_promise_b21():
    """B21's bound in the replay. Restarts every 10 minutes from 10:00 to
    18:00 on Jan 7 (49 of them: a crash loop after the brake's window; the
    period was spending). The 10:00 ledger re-buys Jan 7's twelve games once
    each, at its birth, as "rebuy" (the floor and the ceiling checked: both
    hold). Those twelve credits are the evening refreshes' own, so every
    later ledger plans P = 1 and its braked games keep their one assumed
    pull, until the last ledger's evening: the tips that passed by 20:30
    lift P to 2, and its three games still ahead are re-bought then. The
    loop costs 15 credits, not 12 a restart, and no game is bought more
    than three times on the day (its morning pull and at most one re-buy
    per ledger that may have one). Throughout, after every call the balance
    still covers the morning pull of every game waiting for one (the
    floor's promise: remaining - S >= c_core x the waiting games; the floor
    itself never has to refuse here, since P under 2 stops the re-buys
    first), the ceiling holds, and Jan 8's eight are bought at 08:50 as
    first pulls by the ledger born at 18:00, which saw Jan 8's morning."""
    restarts = [et(1, 7, 10) + k * dt.timedelta(minutes=10) for k in range(49)]
    assert restarts[-1] == et(1, 7, 18)
    rp = Replay(PERIOD_JAN).run(et(1, 1, 8), et(1, 7, 10))
    calls_before = len(rp.calls)
    seen = []                       # after each view that bought: the floor's promise, the ceiling
    real_view = rp.view

    def view(now, **kw):
        n = len(rp.calls)
        real_view(now, **kw)
        if len(rp.calls) > n:
            per = OB.period_of(now, rp.used, FREE, None, 1, PERIOD_JAN, busiest=BUSIEST)
            waiting = OB.waiting_first(PERIOD_JAN, rp.slots, now, per)
            seen.append((now, rp.remaining - 25 >= waiting,
                         rp.used <= OB.spending_ceiling(FREE, 25, PERIOD_JAN, now.date())))
    rp.view = view
    rp.run(et(1, 7, 10), et(1, 9), restarts=restarts)
    jan7 = [(t, gid, kind) for t, gid, _, kind in rp.calls[calls_before:] if t.date() == day(1, 7)]
    assert jan7 == [(et(1, 7, 10), x.game_id, "rebuy") for x in DAYS[day(1, 7)]] + [
        (et(1, 7, 20, 30), x.game_id, "rebuy") for x in DAYS[day(1, 7)][-3:]]
    assert spend_on(rp, day(1, 7)) == 27                     # 12 first pulls, 15 re-buys
    assert pulls_per_game(rp, DAYS[day(1, 7)]) == {2: 9, 3: 3}
    assert seen and all(floor and ceiling for _t, floor, ceiling in seen)
    jan8 = [(t, gid, kind) for t, gid, _, kind in rp.calls if t.date() == day(1, 8)]
    assert [(t, gid, kind) for t, gid, kind in jan8 if t == et(1, 8, 8, 50)] == [
        (et(1, 8, 8, 50), x.game_id, "first") for x in DAYS[day(1, 8)]]


def test_the_last_days_plan_flips_to_full_for_its_last_first_pull_l2_5_pinned():
    """Build 4's review, L2-5, pinned as built: plan() adds the pulls made to
    B / c whatever they cost (design 3.3's formula). The free plan with the
    12 main markets runs as core (points) all month; on Jan 31 at 08:50
    (44 left, 8 games, core P = 2) seven first pulls of points leave 37,
    and the eighth game's ticket then reads the plan as full (P_full =
    floor((12 / 12 + 7) / 8) = 1): first_pull_markets gives it all twelve
    markets, the credits the core plan kept for the evening's refreshes.
    Whether made pulls should count at their own cost is an open question."""
    now = et(1, 31, 8, 50)
    per = OB.period_of(now, 456, FREE, None, len(TWELVE.markets), PERIOD_JAN, busiest=BUSIEST)
    assert (per.month, per.basis) == ((2026, 1), "used_old")         # inside W(Feb): January
    start = OB.plan(44, 456, 0, TWELVE, PERIOD_JAN, {}, per, "ok", BUSIEST, now=now,
                    period_games=PERIOD_JAN)
    assert (start.mode, start.pulls_per_game, start.markets) == ("core", 2, PTS)
    jan31 = sorted(DAYS[day(1, 31)], key=lambda g: g.tip_utc)
    assert len(jan31) == 8
    made = {g.game_id: slot_of(g, pulls=1, first_at=now, last_at=now) for g in jan31[:7]}
    later = OB.plan(37, 463, 0, TWELVE, PERIOD_JAN, made, per, "ok", BUSIEST, now=now,
                    period_games=PERIOD_JAN)
    waiting = OB.waiting_first(PERIOD_JAN, made, now, per, jan31[7].game_id)
    assert (later.mode, later.pulls_per_game, waiting) == ("full", 1, 0)
    assert OB.first_pull_markets(later, TWELVE, 37, 0, waiting) == TWELVE.markets
