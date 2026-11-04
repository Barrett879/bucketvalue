"""The bot: refresh the current season from sources reachable from GitHub
Actions, and publish projections.

    python scripts/daily_update.py [--date YYYY-MM-DD] [--dry-run]

--date replays a PAST (or today's) ET date; a date after today is refused
(exit 2): the clock-driven parts (the injury walk, the odds capture, the
freeze) read the real clock, so a future "today" would publish freezable rows
from inputs days old. A plain run on the evening before already projects
tomorrow (provisional).

Sources: cdn.nba.com (schedule, liveData box scores, playerIndex.json), the
ak-static injury PDFs, and ESPN, every one reached from Actions by the probes
of 2026-10-09 and 2026-10-11 (docs/probes/; playerIndex in the second) and
read by the scheduled bot since 2026-11-02. stats.nba.com is never called (it
hangs from Azure).

Every write goes under state/, through bucketlib.state's paths (one writer
per path; docs/serving_design.md section 1). Nothing under cache/ is written:
the injury PDFs are read through an in-memory store.

Steps, each independent (one failing step does not skip the others; an
exception in a step is caught, recorded and turns the run RED):
  1. schedule      cdn schedule            -> state/current/schedule_v1.parquet
  2. roster        cdn playerIndex          -> state/current/roster_v1.parquet
                   and, dated by the run's own ET date, the snapshot
                   state/season/{season}/roster_{date}.parquet
                   (state.roster_snapshot_path; a failed pull writes neither)
  3. finals        cdn box scores of every FINAL state game not yet ingested
                   (state.ingested_ids: overrides count, so a game committed as
                   an override is never re-fetched)
                   -> state/season/{season}/{pg,tg,part,games}_{date}.parquet
                   (per-ET-date partitions, immutable once all that date's games
                   are final; the stat lines equal the stats logs outside stat
                   corrections, tests/test_serve_parity.py). Each game stands
                   alone: a box that fails to fetch or parse is reported and
                   skipped, and every other game of the run is still written.
                   A state game before today that is not ingested is a WARN
                   while its tip is under STUCK_AFTER old (a late West Coast
                   game still live after midnight) and RED after that.
                   A box listing no INACTIVE player on either side (the
                   2026-27 format: actives only) gets its inactive rows rebuilt
                   (sources.box.rebuild_inactives): the roster snapshot of the
                   game's date (state.roster_for_game: the latest on or before
                   it, else the earliest after it, the backlog) minus the box,
                   minus anyone another team's box lists between the snapshot
                   and the game (every box of the run is parsed first, so the
                   backlog's later boxes count), minus a play-in game's
                   two-way players (ineligible), reasons from the last real
                   report before the t_15 horizon
                   (horizons, one fetch per slot per run; no report means
                   INACTIVE_UNLISTED, never a failure). One line per such game
                   says which snapshot and which report. No usable snapshot:
                   the game is not ingested (RED, retried next run), because
                   a history written without its inactive rows would never be
                   rewritten.
  4. injuries      latest REAL report (horizons.real_report)
                   -> state/current/injury_{players,teams}_v1.parquet
                   (context.write_committed); with no new report the committed
                   one stands in for the projections, as it does in the app
  5. odds          ESPN pregame snapshot   -> state/season/{season}/odds_{date}.parquet
                   (regular season and play-in only: preseason and playoffs
                   never enter state). A capture that differs from the stored
                   latest one only by its captured_at stamp is not a new
                   capture, so an unchanged board leaves the file byte-identical.
  6. projections   for D in (today ET, tomorrow ET), regular-season games
                   only (pipeline.PROJECTED_TYPES through pipeline.state_games:
                   play-in games are ingested as state, never projected):
                   state.history(season, before=D) -> serve.history_features
                   -> state/serve/{D}/features_v1.parquet (stamped with the
                   ingested games its history holds, pipeline.history_stamp);
                   pipeline.project with the report of step 4, the odds of
                   step 5 and the history's own name spellings
                   -> state/serve/{D}/projections_v1.parquet, where every row of
                   a game past tip minus 5 minutes is carried over from the file
                   byte for byte (pipeline.merge_published: a game is never
                   written for the first time after that point) and an untipped
                   game is rewritten only when an input changed or the clock
                   crossed one of its training horizons (pipeline: COMMIT
                   CHURN). Tomorrow is provisional, and so is a game whose own
                   two teams have a game before D missing from the state
                   (serve.completeness_by_game: per game, whatever that
                   game's status). Games without a line are withheld
                   (pipeline's documented policy), never RED. A team with more
                   than MAX_TEAM_ROWS roster rows (camp rosters not cut) is
                   RED today and a warning tomorrow until the day after the
                   regular season's first game day; from then on it is
                   checked against ESPN's roster (espn.team_roster): the same
                   players is an info line, any difference a WARN naming the
                   players only one source lists, and nothing is dropped
                   either way (an in-season 19 is a hardship signing, not a
                   camp roster: LAC on 2026-11-02). A game carried past its
                   freeze point on rows computed with a report older than the
                   newest committed report titled before that point is a
                   warning naming both report times (frozen_on_old_report).

Rules (ops review):
  - The ET date is derived HERE, every run (cron starts were measured 2-7 hours
    late on DiamondValue; nothing assumes which run this is).
  - Every committed write uses if_changed=True, so an unchanged file stays
    byte-identical and the bot does not commit, push and redeploy for nothing.
  - Stale beats empty: an empty or failed pull never overwrites good data,
    and the committed copy stands in for ANY failed pull (Blocked, a bad
    payload, a parser refusing it), not only for a Blocked host.
  - The run goes RED (exit 1) on any Blocked source, on any source that
    could not be read, when a date with final games ingested zero of them,
    when a box listing only its active players has no roster snapshot to
    rebuild its inactive list from (that game is not ingested),
    when ingestion trails a state game by more than STUCK_AFTER after its tip,
    when today's untipped games could not be projected for a reason other than
    the missing-line policy (no injury report at all is such a reason: unknown
    statuses are a null skew the models never saw; so is a game with a side
    under pipeline.MIN_TEAM_ROWS rows, none at all included), and on any
    exception in a step. The same failures on tomorrow's slate are warnings
    (it is provisional and every later run retries it). A green run with stale
    data is the failure mode this exists to prevent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import context as CTX  # noqa: E402
from bucketlib import horizons as HZ  # noqa: E402
from bucketlib import http as H  # noqa: E402
from bucketlib import models as M  # noqa: E402
from bucketlib import pipeline as P  # noqa: E402
from bucketlib import serve as V  # noqa: E402
from bucketlib import state as S  # noqa: E402
from bucketlib.sources import box, cdn, cdnbox  # noqa: E402
from bucketlib.sources import injury as INJ  # noqa: E402
from bucketlib.util import now_et, now_utc, season_for_date, to_et  # noqa: E402

INJURY_SLOTS = 12         # report slot times walked back from now (3 hours of quarter-hours)
# 15 standard contracts + 3 two-way. More on a team playing today means the
# playerIndex still holds camp rosters (18-21 a team until the cut-down the
# day before the opener): waived players would be projected and shift their
# teammates' team features. A hardship exception is the rare legitimate excess.
MAX_TEAM_ROWS = 18
# A state game not ingested this long after its tip is stuck (a feed frozen
# mid-game, a box that never turns final), not late: a 10:30 PM ET tip is
# still live when the runs after midnight land.
STUCK_AFTER = dt.timedelta(hours=6)
# A failed pull: the host refused (Blocked), the object is missing (Absent),
# a bad payload or request, or a payload the parser cannot read. The
# committed copy stands in for every one of them (stale beats empty).
UNUSABLE = (H.FetchError, ValueError, KeyError, TypeError, AttributeError)
STATE_SLUGS = {"regular-season", "play-in-season"}    # ESPN season slugs that enter state
# The report a rebuilt inactive list takes its reasons from: the last real
# report before this horizon (horizons.HORIZONS: tip minus 15 minutes, the one
# training calls "the last report before tip"; 89.5% of 2023-24..2025-26
# inactive players are listed Out on it).
REBUILD_HORIZON = "t_15"
RED: list[str] = []
# Run-scoped caches, emptied by main(): the in-memory report store every step
# shares (report_store) and ESPN rosters fetched for the roster-size check.
REPORTS: INJ.ReportStore | None = None
ESPN_ROSTERS: dict[str, object] = {}


def red(msg: str) -> None:
    RED.append(msg)
    print(f"RED: {msg}")


def season_dir(season: str) -> Path:
    """Where step_finals writes `season`'s partitions: state.season_dir,
    created. (Tests point it at a tmp root.)"""
    p = S.season_dir(season)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 1-2. Schedule and roster ─────────────────────────────────────────────────
def _unusable(what: str, e: Exception) -> None:
    """RED for a failed pull whose committed copy stands in."""
    if isinstance(e, H.Blocked):
        red(f"{what} blocked: {e}")
    else:
        red(f"{what} unusable ({type(e).__name__}): {e}; keeping the last good copy")


def step_schedule(dry: bool) -> pd.DataFrame | None:
    path = S.current_path("schedule")
    try:
        sched = cdn.parse_schedule(cdn.fetch_schedule())
    except UNUSABLE as e:
        _unusable("schedule", e)
        return C.read_parquet(path)
    if sched.empty:
        red("schedule parsed empty; keeping the last good copy")
        return C.read_parquet(path)
    odd = sched.attrs.get("postponed_status_unknown") or []
    if odd:
        print(f"WARN schedule: unrecognised postponedStatus on {len(odd)} game(s), "
              f"e.g. {odd[:3]}: NOT treated as postponed (check the feed)")
    if dry:
        print(f"schedule (dry run): {len(sched)} games")
        return sched
    wrote = C.write_parquet(sched, path, if_changed=True)
    print(f"schedule: {len(sched)} games ({'updated' if wrote else 'unchanged'})")
    return sched


def _per_team(r: pd.DataFrame) -> str:
    """'15-18 a team (max BOS 18)': the opening-night check on camp cuts."""
    n = r.groupby("team").size()
    return f"{n.min()}-{n.max()} a team (max {n.idxmax()} {n.max()})" if len(n) else "no teams"


def step_roster(dry: bool) -> pd.DataFrame | None:
    path = S.current_path("roster")
    try:
        r = cdn.parse_player_index(cdn.fetch_player_index())
    except UNUSABLE as e:
        _unusable("roster", e)
        return C.read_parquet(path)
    if len(r) < cdn.MIN_ROSTER:            # the one threshold, read at call time
        red(f"roster has only {len(r)} players (under {cdn.MIN_ROSTER}); keeping the last good copy")
        return C.read_parquet(path)
    if dry:
        print(f"roster (dry run): {len(r)} players, {_per_team(r)}, {int(r['two_way'].sum())} two-way")
        return r
    wrote = C.write_parquet(r, path, if_changed=True)
    # The dated snapshot: who was on each team, which a 2026-27 box no longer
    # says (step 3 rebuilds a box's inactive rows from it). Dated by the run's
    # own ET date, never --date: it is the roster as pulled now.
    snap = S.roster_snapshot_path(now_et().date())
    snapped = C.write_parquet(r[S.ROSTER_SNAPSHOT_COLS].reset_index(drop=True), snap, if_changed=True)
    print(f"roster: {len(r)} players, {_per_team(r)}, {int(r['two_way'].sum())} two-way "
          f"({'updated' if wrote else 'unchanged'}); snapshot {snap.name} "
          f"({'updated' if snapped else 'unchanged'})")
    return r


# ── 3. Finals ────────────────────────────────────────────────────────────────
def _hours(td: dt.timedelta) -> str:
    return f"{td.total_seconds() / 3600:.1f}h"


def _tip_age(g, now: dt.datetime) -> dt.timedelta:
    """How long ago the game tipped (a missing tip counts as long ago: the
    conservative side of STUCK_AFTER)."""
    tip = pd.to_datetime(g.get("tip_utc"), utc=True) if "tip_utc" in g else pd.NaT
    return dt.timedelta(days=365) if pd.isna(tip) else now - tip.to_pydatetime()


def report_store() -> INJ.ReportStore:
    """The run's one in-memory report store (nothing under cache/ is written),
    shared by steps 3 and 4 so a slot fetched for one is not fetched again."""
    global REPORTS
    if REPORTS is None:
        REPORTS = INJ.ReportStore(persist=False)
    return REPORTS


class NoRoster(Exception):
    """No roster snapshot can say who else was on a team: a box listing only
    its active players cannot be ingested yet."""


class OncePerSlot:
    """A report getter that asks the store about each slot ONCE per run, its
    answer (a report, None for absent, or the parse error) remembered: the
    backlog's games share slots, and ReportStore alone re-probes a recent
    absent slot every time."""

    def __init__(self, store: INJ.ReportStore):
        self.store, self.seen = store, {}

    def __call__(self, slot):
        if slot.filename not in self.seen:
            try:
                self.seen[slot.filename] = self.store.get(slot)
            except ValueError as e:              # a PDF that does not parse stays that way
                self.seen[slot.filename] = e
        got = self.seen[slot.filename]
        if isinstance(got, Exception):
            raise got
        return got


class Rebuild:
    """Step 3's rebuild of the inactive rows of a box that lists none
    (sources.box.rebuild_inactives). Snapshots, names and reports are read on
    first use only, so a run of old-format boxes reads none of them."""

    def __init__(self, season: str, sched: pd.DataFrame, first_day: dt.date):
        self.season, self.sched, self.first_day = season, sched, first_day
        self._known = None
        self._getter = None
        self._tables: dict[str, tuple] = {}
        self._blocked: str | None = None
        self._run: list[pd.DataFrame] = []
        self._listings = None

    def known(self) -> pd.DataFrame:
        if self._known is None:
            self._known = S.known_names(self.season)
        return self._known

    def boxes(self, parsed: list[tuple[pd.DataFrame, str]]) -> None:
        """This run's parsed boxes, (participation rows, ET date) each: with
        the committed ones (state.box_listings) they are the evidence that a
        snapshot is stale for a player (box.rebuild_inactives' elsewhere)."""
        self._run = [p[["person_id", "team"]].assign(et_date=d) for p, d in parsed]
        self._listings = None

    def listings(self) -> pd.DataFrame:
        if self._listings is None:
            frames = [f for f in [S.box_listings(self.season), *self._run] if len(f)]
            self._listings = (pd.concat(frames, ignore_index=True) if frames
                              else pd.DataFrame(columns=S.LISTING_COLS))
        return self._listings

    def elsewhere(self, roster_date: dt.date, game_date: dt.date) -> pd.DataFrame:
        """(person_id, team) of every box listing dated between the snapshot's
        date and the game's, both included."""
        lo, hi = sorted((roster_date, game_date))
        ev = self.listings()
        on = (ev["et_date"] >= lo.isoformat()) & (ev["et_date"] <= hi.isoformat())
        return ev.loc[on, ["person_id", "team"]]

    def listed(self, g, tip) -> tuple[pd.DataFrame | None, str]:
        """(the game's rows of the last real report before REBUILD_HORIZON,
        where it came from) through horizons, the training snapshots' own
        functions. No report, a blocked host or a PDF that does not parse is
        (None, why): reasons are then INACTIVE_UNLISTED, never a failure."""
        if self._blocked:
            return None, f"no report (the report host is blocked: {self._blocked})"
        if self._getter is None:
            self._getter = OncePerSlot(report_store())
            self._frame = HZ.schedule_frame(self.sched)
            self._real = HZ.real_report(self._frame)
        cut = HZ.cutoff_for(REBUILD_HORIZON, tip)
        try:
            found = HZ.report_at(cut, self._getter, accept=self._real)
        except H.Blocked as e:
            self._blocked = str(e)
            print(f"WARN finals: the injury report host is blocked ({e}): the rest of this run's "
                  f"rebuilt inactive rows carry {box.UNLISTED}")
            return None, f"no report (the report host is blocked: {e})"
        except ValueError as e:
            # The newest candidate does not parse: never an older report
            # instead (horizons.snapshot_games' rule).
            return None, f"no report (the newest before {_et(cut)} ET does not parse: {e})"
        except H.FetchError as e:
            return None, f"no report ({type(e).__name__}: {e})"
        if found is None:
            return None, f"no report titled before {_et(cut)} ET"
        slot, rep = found
        if slot.url not in self._tables:
            self._tables[slot.url] = HZ.report_tables(rep, self._frame)
        kept, teams = self._tables[slot.url]
        rows, _ = HZ.game_states(g, {"game_id": g["game_id"], "horizon": REBUILD_HORIZON,
                                     "cutoff_utc": cut}, kept, teams, rep.meta["report_ts_et"])
        return pd.DataFrame(rows), f"the {_et(rep.meta['report_ts_et'])} ET report"

    def __call__(self, part: pd.DataFrame, g, tip) -> tuple[pd.DataFrame, str]:
        """(the game's participation rows with its inactive rows rebuilt, the
        log line). Raises NoRoster when no snapshot qualifies."""
        at = S.roster_for_game(self.season, g["game_date"], self.first_day)
        if at is None:
            raise NoRoster(f"no roster snapshot of {self.season} on or after its first game day "
                           f"{self.first_day} ({S.roster_snapshot_path(g['game_date']).parent})")
        listed, whence = self.listed(g, tip)
        rows, info = box.rebuild_inactives(part, at.roster, listed, known=self.known(),
                                           roster_after=at.after,
                                           elsewhere=self.elsewhere(at.date, g["game_date"]))
        which = ("the earliest snapshot after the game: the backlog" if at.after
                 else "the latest snapshot on or before the game")
        per = []
        for team in (g["away"], g["home"]):
            c = info[team]
            bits = [f"{c['listed']} listed", f"{c['unlisted']} unlisted"]
            if c["left_since"]:
                bits.append(f"{c['left_since']} left since")
            if c["elsewhere"]:
                bits.append(f"{c['elsewhere']} on the roster kept off: another team's box "
                            f"lists them between the snapshot and the game")
            if c["ineligible"]:
                bits.append(f"{c['ineligible']} two-way left out (ineligible after the regular season)")
            per.append(f"{team} {c['rebuilt']} ({', '.join(bits)})")
        n = sum(info[t]["rebuilt"] for t in (g["away"], g["home"]))
        line = (f"finals: {g['game_id']} {g['away']} at {g['home']} ({g['game_date']}): the box lists "
                f"no inactive player; {n} rebuilt from the {at.date} roster ({which}) minus the box, "
                f"reasons from {whence}: {'; '.join(per)}")
        if info["unresolved"]:
            line += (f"; {len(info['unresolved'])} report name(s) resolved to nobody: "
                     f"{', '.join(info['unresolved'][:4])}")
        return rows, line


def step_finals(sched: pd.DataFrame, today: dt.date, dry: bool) -> None:
    # The state games (serve._state_games: RS and PI with both teams, postponed
    # excluded), as schedule rows (game_date stays a python date).
    state = sched[sched["game_id"].isin(set(V._state_games(sched)["game_id"]))
                  & (sched["game_date"] <= today)]
    finals = state[state["status"] == V.FINAL]
    if state.empty:
        print("finals: none yet this season")
        return
    season = state["season"].iloc[0]
    have = S.ingested_ids(season)
    todo = finals[~finals["game_id"].isin(have)]
    now = now_utc()
    print(f"finals: {len(finals)} final state games, {len(have)} ingested, {len(todo)} to fetch")
    rebuild = Rebuild(season, sched, min(state["game_date"]))
    by_date: dict[dt.date, dict[str, list]] = {}
    parsed = []
    # Each game stands alone: whatever one box does, every other game of the
    # run is still parsed and written (one stuck box must never stop the
    # league's ingestion). Only a Blocked host ends the loop.
    for _, g in todo.iterrows():
        gid = g["game_id"]
        try:
            payload = cdn_box(gid)
        except H.Blocked as e:
            red(f"box score blocked at {gid}: {e}")
            break
        except H.Absent:
            red(f"box score absent for final game {gid}")
            continue
        except Exception as e:  # noqa: BLE001 - BadPayload, BadRequest, anything else
            red(f"box score fetch failed for {gid} ({type(e).__name__}): {e}")
            continue
        try:
            part, meta = box.parse_cdn(payload, gid)
            pl, tl = cdnbox.player_lines(payload), cdnbox.team_lines(payload)
        except box.NotFinal as e:
            # The schedule can say Final before the box does; a box frozen
            # mid-game (17 in 2019-20) never turns final at all.
            age = _tip_age(g, now)
            if age < STUCK_AFTER:
                print(f"WARN finals: {e} (the schedule says Final; retried next run)")
            else:
                red(f"box score of final game {gid} is still not final {_hours(age)} after tip: {e}. "
                    f"If it stays stuck, read 'A stuck game' in docs/runbook.md section 4 before "
                    f"any scripts/build_serving_seed.py --override {gid}: a 2026-27 override "
                    f"must carry the game's inactive rows")
            continue
        except Exception as e:  # noqa: BLE001 - one strange payload must never throw away the night
            red(f"parse failed for {gid}: {type(e).__name__}: {e}")
            continue
        parsed.append((g, payload, part, meta, pl, tl))
    # Every box of the run is parsed before any inactive list is rebuilt: a
    # later box of the backlog is evidence about an earlier game (Rebuild.boxes).
    rebuild.boxes([(part, meta["et_date"]) for _, _, part, meta, _, _ in parsed])
    for g, payload, part, meta, pl, tl in parsed:
        gid = g["game_id"]
        if not box.lists_inactives(payload):
            # The 2026-27 format: actives only. Ingested without its inactive
            # rows, the game would read as "not on the team" for every injured
            # player, for good (a final game is never re-fetched).
            try:
                part, line = rebuild(part, g, meta["tip_utc"].to_pydatetime())
            except NoRoster as e:
                why = (f"box score of {gid} lists no inactive player and there is {e}: "
                       f"not ingested (retried next run)")
                if dry:
                    print(f"WARN finals (dry run): {why}")
                else:
                    red(why)
                continue
            except Exception as e:  # noqa: BLE001 - one game's rebuild never stops the others
                red(f"inactive rows of {gid} could not be rebuilt ({type(e).__name__}): {e}; "
                    f"not ingested (retried next run)")
                continue
            print(line)
        for frame, df in (("pg", pl), ("tg", tl), ("part", part), ("games", pd.DataFrame([meta]))):
            df = df.copy()
            df["season"] = season
            df["season_type"] = g["game_type"]
            by_date.setdefault(g["game_date"], {}).setdefault(frame, []).append(df)
    if dry:
        print(f"finals (dry run): would write {sum(len(v.get('games', [])) for v in by_date.values())} games")
        return
    d = season_dir(season)
    for date, frames in by_date.items():
        for frame, parts in frames.items():
            p = d / f"{frame}_{date.isoformat()}.parquet"
            old = C.read_parquet(p)
            new = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
            key = ["game_id", "person_id"] if "person_id" in new else ["game_id", "team"] if "team" in new else ["game_id"]
            new = new.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
            C.write_parquet(new, p, if_changed=True)
    # Completeness: every state game before today must be ingested, by the
    # one definition of ingested (overrides included), WHATEVER its status
    # (serve.completeness counts a live or not-yet-final game as a gap too).
    # A late game still live after midnight is normal: a warning until
    # STUCK_AFTER past its tip, RED after that (a stuck feed or box).
    have = S.ingested_ids(season)
    behind = state[(state["game_date"] < today) & ~state["game_id"].isin(have)]
    stuck, late = [], []
    for _, g in behind.iterrows():
        (stuck if _tip_age(g, now) >= STUCK_AFTER else late).append(f"{g['game_id']} ({g['status']})")
    if late:
        print(f"WARN finals: {len(late)} state game(s) before today not ingested yet, tipped under "
              f"{_hours(STUCK_AFTER)} ago (still live or just final; their teams' next games stay "
              f"provisional until ingested): {', '.join(late[:5])}")
    if stuck:
        red(f"{len(stuck)} state game(s) before today are not ingested {_hours(STUCK_AFTER)} after tip, "
            f"e.g. {', '.join(sorted(stuck)[:5])}")


def cdn_box(game_id: str) -> dict:
    return H.fetch(cdn.BOX_URL.format(gid=game_id), timeout=20, retries=2)


# ── 4. Injury report ─────────────────────────────────────────────────────────
def _committed_report(why: str):
    """The committed report (context.read_committed), or None. A pair it
    cannot trust is RED, and the projections then have no report."""
    try:
        rep, _ = CTX.read_committed()
    except (ValueError, KeyError, OSError) as e:
        red(f"injuries: {why}, and the committed report is unusable: {e}")
        return None
    if rep is None:
        print(f"injuries: {why}, and none is committed: statuses are unknown")
        return None
    print(f"injuries: {why}; the committed report of {rep.meta['report_ts_et']} stands in")
    return rep


def step_injuries(sched: pd.DataFrame, dry: bool):
    """The report the projections use: the newest REAL report in the last
    INJURY_SLOTS slot times (written as the committed pair), else the
    committed one, else None. PDFs go through an in-memory store: nothing
    under cache/ is written."""
    store = report_store()
    try:
        sel = INJ.latest_report(now_et(), getter=store.get, max_slots=INJURY_SLOTS,
                                accept=HZ.real_report(sched))
    except H.Blocked as e:
        red(f"injury reports blocked: {e}")
        return _committed_report("the report host is blocked")
    except H.FetchError as e:
        # BadPayload (a truncated PDF), BadRequest: the committed report stands in.
        red(f"the newest injury report could not be fetched ({type(e).__name__}): {e}")
        return _committed_report("the newest report could not be fetched")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        red(f"the newest injury report does not parse ({type(e).__name__}): {e}")
        return _committed_report("the newest report does not parse")
    if sel is None:
        return _committed_report(f"no real report in the last {INJURY_SLOTS} slots "
                                 "(preseason, or between reports)")
    rep = sel.report
    if dry:
        print(f"injuries (dry run): report {rep.meta['report_ts_et']}, {len(rep.rows)} rows")
        return rep
    wrote = CTX.write_committed(rep, sched)
    print(f"injuries: report {rep.meta['report_ts_et']}, {len(rep.rows)} rows "
          f"({'updated' if wrote else 'unchanged'})")
    return rep


# ── 5. Odds ──────────────────────────────────────────────────────────────────
def step_odds(today: dt.date, dry: bool) -> dict[dt.date, pd.DataFrame | None]:
    """Snapshot pregame lines for today and tomorrow; returns per date the
    snapshots the projections read (committed plus this run's)."""
    from bucketlib.sources import espn
    out: dict[dt.date, pd.DataFrame | None] = {}
    for d in (today, today + dt.timedelta(days=1)):
        p = S.odds_path(d)
        old = C.read_parquet(p)
        out[d] = old
        try:
            sb = espn.scoreboard(d)
            pre = sb[sb["status"].isin(["STATUS_SCHEDULED"]) & sb["home_line"].notna()
                     & sb["season_slug"].isin(STATE_SLUGS)]
        except UNUSABLE as e:
            # Blocked, a bad payload, a parser refusing it: the committed
            # snapshots still feed the projections (stale beats empty), the
            # other date is still tried, and the run is RED.
            red(f"odds {d}: {type(e).__name__}: {e}")
            continue
        if pre.empty:
            print(f"odds {d}: no priced regular-season or play-in game")
            continue
        fresh = new_captures(old, pre)
        if fresh.empty:
            print(f"odds {d}: {len(pre)} priced games, no line moved since the last capture")
            continue
        new = pd.concat(([old] if old is not None else []) + [fresh], ignore_index=True)
        # Keep the FIRST and the LATEST pre-tip capture per game.
        new = new.sort_values("captured_at", kind="mergesort")
        keep = pd.concat([new.groupby("event_id").head(1), new.groupby("event_id").tail(1)]).drop_duplicates()
        keep = keep.reset_index(drop=True)
        out[d] = keep
        if dry:
            print(f"odds {d} (dry run): {len(pre)} priced games, {len(fresh)} new captures")
            continue
        C.write_parquet(keep, p, if_changed=True)
        print(f"odds {d}: {len(pre)} priced games, {len(fresh)} new captures")
    return out


def new_captures(old: pd.DataFrame | None, pre: pd.DataFrame) -> pd.DataFrame:
    """The rows of `pre` (this run's scoreboard captures) that say something
    the stored LATEST capture of the same event does not: every column but
    captured_at (the fetch time, new on every run) is compared, NaN equal to
    NaN. An unchanged board is then no new capture at all, so the file stays
    byte-identical and the run does not commit for nothing."""
    if old is None or old.empty:
        return pre
    same_cols = [c for c in pre.columns if c != "captured_at"]
    latest = old.sort_values("captured_at", kind="mergesort").drop_duplicates("event_id", keep="last")
    both = pd.concat([latest.reindex(columns=same_cols), pre[same_cols]], ignore_index=True)
    seen = both.duplicated(keep="first").to_numpy()[len(latest):]
    return pre[~seen]


# ── 6. Projections ───────────────────────────────────────────────────────────
def history_features(d: dt.date, games: pd.DataFrame, roster: pd.DataFrame,
                     sched: pd.DataFrame, seeds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """state.history strictly before `d` -> serve.history_features: the ONE
    features.build of the date. Returns (feature rows, History.names: every
    spelling the history carries, which the report's names resolve against).
    `seeds` caches the seed per season."""
    season = season_for_date(d)
    if season not in seeds:
        seeds[season] = S.load_seed(season)
    hist = S.history(season, d, seed=seeds[season])
    return V.history_features(d, games, roster, hist, sched), hist.names


def flag(d: dt.date, today: dt.date, msg: str) -> None:
    """RED for today's slate; a warning for tomorrow's (provisional, and every
    later run retries it)."""
    if d == today:
        red(msg)
    else:
        print(f"WARN {msg}")


def oversized(games: pd.DataFrame, roster: pd.DataFrame, cap: int = MAX_TEAM_ROWS) -> dict[str, int]:
    """{team: roster rows} for every team playing in `games` with more than
    `cap` rows in the roster pull (camp rosters not cut yet)."""
    playing = set(games["home"]) | set(games["away"])
    n = roster[roster["team"].isin(playing)].groupby("team").size()
    return {str(t): int(k) for t, k in n.items() if k > cap}


def first_game_day(sched: pd.DataFrame, season: str) -> dt.date | None:
    """The ET date of `season`'s first regular-season game in the schedule
    (serve._state_games: postponed games are not games), or None."""
    g = V._state_games(sched)
    g = g[(g["game_type"] == "RS") & ((g["season"] == season) if "season" in g else True)]
    return None if g.empty else g["game_date"].min().date()


def espn_check(d: dt.date, team: str, n: int, roster: pd.DataFrame) -> str:
    """One line on an in-season team with more than MAX_TEAM_ROWS roster rows:
    its playerIndex rows against ESPN's roster (espn.team_roster, matched by
    espn.match_athletes, team-scoped names). The same players is an info line;
    any difference, or no ESPN roster to compare, a WARN naming the players
    only one source lists. Nothing is dropped either way. One ESPN fetch per
    team per run (ESPN_ROSTERS)."""
    from bucketlib.sources import espn
    if team not in ESPN_ROSTERS:
        try:
            ESPN_ROSTERS[team] = espn.team_roster(team)
        except UNUSABLE as e:
            ESPN_ROSTERS[team] = e
    theirs = ESPN_ROSTERS[team]
    head = f"projections {d}: {team} has {n} players in the roster pull (more than {MAX_TEAM_ROWS})"
    if isinstance(theirs, Exception):
        return (f"WARN {head}, and ESPN's roster could not be read to check them "
                f"({type(theirs).__name__}): {theirs}; projected as listed")
    name = roster["name"] if "name" in roster.columns else roster["player_name"]
    ours = pd.DataFrame({"person_id": roster["person_id"], "name": name, "team": roster["team"],
                         "birthdate": None})
    ours = ours[ours["team"] == team]
    try:
        m = espn.match_athletes(theirs, ours)
    except ValueError as e:
        return f"WARN {head}, and its names do not match ESPN's one to one: {e}; projected as listed"
    matched = set(m["person_id"].dropna().astype("int64"))
    only_ours = sorted(ours.loc[~ours["person_id"].isin(matched), "name"])
    unmatched = set(m.loc[m["person_id"].isna(), "espn_athlete_id"])
    only_theirs = sorted(theirs.loc[theirs["espn_athlete_id"].isin(unmatched), "full_name"])
    if not only_ours and not only_theirs:
        return f"{head}; ESPN's roster lists the same {len(theirs)}: projected as listed"
    return (f"WARN {head}, ESPN's {len(theirs)}: only the playerIndex lists "
            f"{', '.join(only_ours) or 'nobody'}; only ESPN lists {', '.join(only_theirs) or 'nobody'}. "
            f"Projected as the playerIndex lists them (nothing dropped)")


def check_big_rosters(d: dt.date, today: dt.date, sched: pd.DataFrame, roster: pd.DataFrame,
                      big: dict[str, int]) -> None:
    """The roster-size guard for `d`'s teams with more than MAX_TEAM_ROWS rows
    (oversized). Until the day after the regular season's first game day
    such a pull is camp rosters not cut yet: RED today, a warning tomorrow
    (flag). From then on it is an in-season roster over the cap (LAC listed
    19 on 2026-11-02, and so did ESPN: a hardship signing), told apart from a
    feed error by ESPN's roster (espn_check): never RED, nothing dropped."""
    first = first_game_day(sched, season_for_date(d))
    if first is None or today < first + dt.timedelta(days=1):
        flag(d, today, f"projections {d}: the roster pull lists more than {MAX_TEAM_ROWS} players for "
                       + ", ".join(f"{t} {n}" for t, n in sorted(big.items()))
                       + " (camp rosters not cut yet?): waived players are projected and shift "
                         "their teammates' team features")
        return
    for team, n in sorted(big.items()):
        print(espn_check(d, team, n, roster))


def _title(report) -> pd.Timestamp | None:
    """A report's title time (report_ts_et), or None for no report."""
    if report is None:
        return None
    ts = pd.Timestamp(report.meta["report_ts_et"])
    return None if pd.isna(ts) else ts


def committed_title() -> pd.Timestamp | None:
    """The title of the committed injury report (context.read_committed) as
    it stands, or None (none committed, or a pair it cannot trust: step 4
    says so). main() reads it BEFORE step 4 replaces it, so a report the
    previous run committed is still known when a game freezes."""
    try:
        rep, _ = CTX.read_committed()
    except (ValueError, KeyError, OSError):
        return None
    return _title(rep)


def frozen_on_old_report(old: pd.DataFrame | None, games: pd.DataFrame, frozen: set[str],
                         titles) -> list[str]:
    """One line per game carried past its freeze point (tip minus
    pipeline.FREEZE_LEAD) on rows computed with a report OLDER than the
    newest committed report titled strictly before that freeze point (the
    training selection rule's strictness). `titles` are the committed
    report times this run knows: the pair as the run found it, and the
    report it uses. The carried rows stand (a frozen game is never
    rewritten); this only says the frozen projection missed a report the bot
    held before the freeze (its projections step failed or did not run in
    time, or the game's rows were kept as clock-only). Provisional rows are
    skipped: they are never the frozen projection."""
    if old is None or old.empty or not frozen:
        return []
    known = sorted({pd.Timestamp(t) for t in titles if t is not None and not pd.isna(t)})
    out = []
    for _, g in games[games["game_id"].isin(frozen)].iterrows():
        rows = old[(old["game_id"] == g["game_id"]) & ~old["provisional"]]
        if rows.empty:
            continue
        point = pd.Timestamp(g["tip_utc"]) - P.FREEZE_LEAD
        before = [t for t in known if t < point]
        if not before:
            continue
        used = pd.to_datetime(rows["report_ts_et"], utc=True).dropna()
        used = used.max() if len(used) else None
        if used is None or used < before[-1]:
            was = "no report in reach" if used is None else f"the {_et(used)} ET report"
            out.append(f"{g['game_id']} ({g['away']} at {g['home']}) froze on rows computed with "
                       f"{was}, but the {_et(before[-1])} ET report was committed before its "
                       f"freeze point ({_et(point)} ET)")
    return out


def _et(t) -> str:
    return f"{to_et(pd.Timestamp(t).to_pydatetime()):%Y-%m-%d %H:%M}"


def project_date(d: dt.date, games: pd.DataFrame, sched: pd.DataFrame, roster: pd.DataFrame,
                 report, odds: pd.DataFrame | None, models: M.Models, seeds: dict,
                 today: dt.date, dry: bool, held=None) -> None:
    """Step 6 for one date. `held` is the committed report title as the run
    found it (committed_title, read before step 4)."""
    season = season_for_date(d)
    old, old_basis = P.read_published(d)
    frozen = P.frozen_ids(games, now_utc())
    live = games[~games["game_id"].isin(frozen)]
    kept = 0 if old is None else int(old["game_id"].isin(frozen).sum())
    # A WARN, never RED: the carried rows are the published record either way.
    for line in frozen_on_old_report(old, games, frozen, (held, _title(report))):
        print(f"WARN projections {d}: {line}")
    if live.empty:
        print(f"projections {d}: all {len(games)} games past tip - 5 min; "
              f"{kept} committed rows stand frozen")
        return
    big = oversized(live, roster)
    if big:
        check_big_rosters(d, today, sched, roster, big)
    feats, names = history_features(d, games, roster, sched, seeds)
    stamp = P.history_stamp(season, d)
    if not dry:
        wrote = P.write_features(feats, d, stamp)
        print(f"features {d}: {len(feats)} rows, history through {stamp['through']} "
              f"({'updated' if wrote else 'unchanged'})")
    rows = feats[feats["game_id"].isin(set(live["game_id"]))]
    # Only the missing-line policy may leave an untipped game unprojected: a
    # game whose side has no rows (or a broken few) is RED today.
    short = P.short_games(live, rows)
    if short:
        flag(d, today, f"projections {d}: {len(short)} untipped game(s) without a whole roster "
                       f"(under {P.MIN_TEAM_ROWS} rows a side): "
                       + "; ".join(f"{g} {s}" for g, s in short.items()))
    # Provisional per game (pipeline.provisional_for); project() sets each
    # row from the map, and a game missing from it is provisional.
    prov, gaps = P.provisional_for(d, today, sched, season)
    now = now_utc()
    try:
        new = P.project(rows, games, sched, report, P.lines_from_odds(odds, games), models, now,
                        provisional=prov, names=names)
    except M.NullSkew as e:
        # NoReport says itself why: no injury report at all, or one out of
        # reach for every game.
        why = "" if isinstance(e, P.NoReport) else " (a null skew)"
        flag(d, today, f"projections {d}: nothing new written{why}: {e}")
        return
    out, basis = P.merge_published(old, new, games, now_utc(), old_basis=old_basis,
                                   new_basis=P.feature_digests(rows))
    withheld = new.attrs.get("withheld", [])
    projected = sorted(set(new["game_id"]))
    rewritten = set(out.loc[out["computed_at"] == pd.Timestamp(now), "game_id"])
    stale = {g: v for g, v in gaps.items() if v and g in set(projected)}
    note = (f"{len(projected)} games projected ({len(rewritten)} rewritten, "
            f"{len(set(projected) - rewritten)} kept: no input changed), {len(frozen)} frozen "
            f"({kept} rows carried), {len(withheld)} withheld (no line yet)"
            + (", provisional (tomorrow)" if d > today else "")
            + (f", {len(stale)} provisional on stale state ("
               + "; ".join(f"{g} misses {', '.join(v[:3])}" for g, v in sorted(stale.items())[:5]) + ")"
               if stale else ""))
    if dry:
        print(f"projections {d} (dry run): would hold {len(out)} rows; {note}")
        return
    wrote = P.write_published(out, basis, d, old_basis)
    print(f"projections {d}: {len(out)} rows ({'updated' if wrote else 'unchanged'}); {note}")


def step_projections(sched: pd.DataFrame | None, roster: pd.DataFrame | None, report,
                     odds: dict | None, today: dt.date, dry: bool, held=None) -> None:
    """Step 6: features and projections for today and tomorrow (ET), of
    regular-season games only (pipeline.PROJECTED_TYPES, read through
    pipeline.state_games): play-in games are ingested as state (step 3) but
    never projected, and preseason never enters state. `report` is step 4's,
    `odds` step 5's snapshots per date, `held` the committed report title as
    the run found it (committed_title, read before step 4)."""
    if sched is None or sched.empty:
        print("projections: no schedule")
        return
    days = [(d, P.state_games(sched, d)) for d in (today, today + dt.timedelta(days=1))]
    if all(g.empty for _, g in days):
        print("projections: no regular-season games today or tomorrow "
              "(preseason and play-in are never projected)")
        return
    if roster is None or roster.empty:
        red("projections: no roster (live or committed); nothing projected")
        return
    models = M.load()
    print(f"projections: models {models.version}; data through "
          f"{P.data_through(season_for_date(today))}")
    seeds: dict = {}
    for d, games in days:
        if games.empty:
            print(f"projections {d}: no regular-season games")
            continue
        try:
            project_date(d, games, sched, roster, report, (odds or {}).get(d), models, seeds,
                         today, dry, held)
        except Exception as e:  # noqa: BLE001 - one date's failure must not skip the other
            red(f"projections {d}: {type(e).__name__}: {e}")
            traceback.print_exc()


# ── Run ──────────────────────────────────────────────────────────────────────
def run(name: str, fn, *args):
    """One step: an exception is recorded (RED) and the run goes on."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - every step is independent
        red(f"{name} step raised {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="replay a PAST ET date as today (default: now, ET); "
                                   "a date after today is refused")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    RED.clear()
    global REPORTS
    REPORTS = None
    ESPN_ROSTERS.clear()
    real = now_et().date()        # util.today_et, through the one clock the tests pin
    today = dt.date.fromisoformat(args.date) if args.date else real
    if today > real:
        # The injury walk, the odds capture and the freeze read the real clock:
        # "today" in the future would publish NON-provisional, freezable rows
        # from a report and lines days old. A plain run already projects
        # tomorrow (provisional).
        print(f"refusing --date {today}: it is after today ({real} ET). --date replays past "
              f"dates only; leave it empty for a live run (tomorrow is projected anyway).")
        return 2
    print(f"BucketValue daily update: ET date {today}, run at {now_utc().isoformat(timespec='seconds')}"
          + (" (dry run: nothing is written)" if args.dry_run else ""))
    sched = run("schedule", step_schedule, args.dry_run)
    roster = run("roster", step_roster, args.dry_run)
    report = None
    # The committed report as this run found it, before step 4 replaces it:
    # step 6 warns when a game freezes on rows older than it.
    held = run("committed report", committed_title)
    if sched is not None and len(sched):
        run("finals", step_finals, sched, today, args.dry_run)
        report = run("injuries", step_injuries, sched, args.dry_run)
    odds = run("odds", step_odds, today, args.dry_run)
    run("projections", step_projections, sched, roster, report, odds, today, args.dry_run, held)
    if RED:
        print(f"\nRUN IS RED ({len(RED)}):")
        for r in RED:
            print("  -", r)
        return 1
    print("\nrun is green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
