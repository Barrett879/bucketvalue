"""The bot: refresh the current season from sources reachable from GitHub
Actions, and publish projections.

    python scripts/daily_update.py [--date YYYY-MM-DD] [--dry-run]

--date replays a PAST (or today's) ET date; a date after today is refused
(exit 2): the clock-driven parts (the injury walk, the odds capture, the
freeze) read the real clock, so a future "today" would publish freezable rows
from inputs days old. A plain run on the evening before already projects
tomorrow (provisional).

Sources: cdn.nba.com (schedule, liveData box scores, playerIndex.json), the
ak-static injury PDFs, and ESPN. The 2026-10-09 Actions probe (docs/probes/)
reached the cdn schedule and box scores, the PDFs and ESPN; playerIndex.json
sits on the same host and path as the schedule but was not in that probe
(scripts/probe_hosts.py lists it now: rerun the probe before going live).
stats.nba.com is never called (it hangs from Azure).

Every write goes under state/, through bucketlib.state's paths (one writer
per path; docs/serving_design.md section 1). Nothing under cache/ is written:
the injury PDFs are read through an in-memory store.

Steps, each independent (one failing step does not skip the others; an
exception in a step is caught, recorded and turns the run RED):
  1. schedule      cdn schedule            -> state/current/schedule_v1.parquet
  2. roster        cdn playerIndex          -> state/current/roster_v1.parquet
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
                   RED today and a warning tomorrow. A game carried past its
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
from bucketlib.sources import cdn, cdnbox  # noqa: E402
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
RED: list[str] = []


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
    print(f"roster: {len(r)} players, {_per_team(r)}, {int(r['two_way'].sum())} two-way "
          f"({'updated' if wrote else 'unchanged'})")
    return r


# ── 3. Finals ────────────────────────────────────────────────────────────────
def _hours(td: dt.timedelta) -> str:
    return f"{td.total_seconds() / 3600:.1f}h"


def _tip_age(g, now: dt.datetime) -> dt.timedelta:
    """How long ago the game tipped (a missing tip counts as long ago: the
    conservative side of STUCK_AFTER)."""
    tip = pd.to_datetime(g.get("tip_utc"), utc=True) if "tip_utc" in g else pd.NaT
    return dt.timedelta(days=365) if pd.isna(tip) else now - tip.to_pydatetime()


def step_finals(sched: pd.DataFrame, today: dt.date, dry: bool) -> None:
    from bucketlib.sources import box  # participation parser (status, starters, reasons)
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
    by_date: dict[dt.date, dict[str, list]] = {}
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
                    f"If it stays stuck, commit an override locally: "
                    f"scripts/build_serving_seed.py --override {gid}")
            continue
        except Exception as e:  # noqa: BLE001 - one strange payload must never throw away the night
            red(f"parse failed for {gid}: {type(e).__name__}: {e}")
            continue
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
    store = INJ.ReportStore(persist=False)
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
        flag(d, today, f"projections {d}: the roster pull lists more than {MAX_TEAM_ROWS} players for "
                       + ", ".join(f"{t} {n}" for t, n in sorted(big.items()))
                       + " (camp rosters not cut yet?): waived players are projected and shift "
                         "their teammates' team features")
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
