"""The bot: refresh the current season from sources reachable from GitHub
Actions, and publish projections.

    python scripts/daily_update.py [--date YYYY-MM-DD] [--dry-run]

Every source here was PROVEN reachable from Actions (docs/probes/): cdn.nba.com
(schedule, liveData box scores, playerIndex.json), the ak-static injury PDFs, and
ESPN. stats.nba.com is never called (it hangs from Azure).

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
                   corrections, tests/test_serve_parity.py)
  4. injuries      latest REAL report (horizons.real_report)
                   -> state/current/injury_{players,teams}_v1.parquet
                   (context.write_committed); with no new report the committed
                   one stands in for the projections, as it does in the app
  5. odds          ESPN pregame snapshot   -> state/season/{season}/odds_{date}.parquet
                   (regular season and play-in only: preseason never enters state)
  6. projections   for D in (today ET, tomorrow ET), state games only:
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
                   CHURN). Tomorrow is provisional, and so is a date whose state
                   misses a final (serve.completeness). Games without a line are
                   withheld (pipeline's documented policy), never RED.

Rules (ops review):
  - The ET date is derived HERE, every run (cron starts were measured 2-7 hours
    late on DiamondValue; nothing assumes which run this is).
  - Every committed write uses if_changed=True, so an unchanged file stays
    byte-identical and the bot does not commit, push and redeploy for nothing.
  - Stale beats empty: an empty or failed pull never overwrites good data.
  - The run goes RED (exit 1) on any Blocked source, when a date with final
    games ingested zero of them, when ingestion trails the last final game,
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
from bucketlib.util import now_et, now_utc, season_for_date, today_et  # noqa: E402

STATE_TYPES = {"RS", "PI"}
MIN_ROSTER = 400          # a playerIndex with fewer rostered players is a broken pull
INJURY_SLOTS = 12         # report slot times walked back from now (3 hours of quarter-hours)
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
def step_schedule(dry: bool) -> pd.DataFrame | None:
    path = S.current_path("schedule")
    try:
        sched = cdn.parse_schedule(cdn.fetch_schedule())
    except H.Blocked as e:
        red(f"schedule blocked: {e}")
        return C.read_parquet(path)
    if sched.empty:
        red("schedule parsed empty; keeping the last good copy")
        return C.read_parquet(path)
    if dry:
        print(f"schedule (dry run): {len(sched)} games")
        return sched
    wrote = C.write_parquet(sched, path, if_changed=True)
    print(f"schedule: {len(sched)} games ({'updated' if wrote else 'unchanged'})")
    return sched


def step_roster(dry: bool) -> pd.DataFrame | None:
    path = S.current_path("roster")
    try:
        r = cdn.parse_player_index(cdn.fetch_player_index())
    except H.Blocked as e:
        red(f"roster blocked: {e}")
        return C.read_parquet(path)
    if len(r) < MIN_ROSTER:
        red(f"roster has only {len(r)} players; keeping the last good copy")
        return C.read_parquet(path)
    if dry:
        print(f"roster (dry run): {len(r)} players, {int(r['two_way'].sum())} two-way")
        return r
    wrote = C.write_parquet(r, path, if_changed=True)
    print(f"roster: {len(r)} players, {int(r['two_way'].sum())} two-way ({'updated' if wrote else 'unchanged'})")
    return r


# ── 3. Finals ────────────────────────────────────────────────────────────────
def step_finals(sched: pd.DataFrame, today: dt.date, dry: bool) -> None:
    from bucketlib.sources import box  # participation parser (status, starters, reasons)
    finals = sched[(sched["status"] == "final") & sched["game_type"].isin(STATE_TYPES)
                   & (sched["game_date"] <= today)]
    if finals.empty:
        print("finals: none yet this season")
        return
    season = finals["season"].iloc[0]
    have = S.ingested_ids(season)
    todo = finals[~finals["game_id"].isin(have)]
    print(f"finals: {len(finals)} final state games, {len(have)} ingested, {len(todo)} to fetch")
    by_date: dict[dt.date, dict[str, list]] = {}
    for _, g in todo.iterrows():
        try:
            payload = cdn_box(g["game_id"])
        except H.Blocked as e:
            red(f"box score blocked at {g['game_id']}: {e}")
            break
        except H.Absent:
            red(f"box score absent for final game {g['game_id']}")
            continue
        try:
            part, meta = box.parse_cdn(payload, g["game_id"])
            pl, tl = cdnbox.player_lines(payload), cdnbox.team_lines(payload)
        except (ValueError, KeyError) as e:
            red(f"parse failed for {g['game_id']}: {e}")
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
    # Completeness: every final state game up to yesterday must be ingested,
    # by the one definition of ingested (overrides included).
    missing = set(finals.loc[finals["game_date"] < today, "game_id"]) - S.ingested_ids(season)
    if missing:
        red(f"{len(missing)} final games before today are not ingested, e.g. {sorted(missing)[:5]}")


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
    except ValueError as e:
        red(f"the newest injury report does not parse: {e}")
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
        p = S.season_dir(season_for_date(d)) / f"odds_{d.isoformat()}.parquet"
        old = C.read_parquet(p)
        out[d] = old
        try:
            sb = espn.scoreboard(d)
        except H.FetchError as e:
            # Blocked or a bad payload: the committed snapshots still feed the
            # projections (stale beats empty), and the run is RED.
            red(f"odds {d}: {type(e).__name__}: {e}")
            continue
        pre = sb[sb["status"].isin(["STATUS_SCHEDULED"]) & sb["home_line"].notna()
                 & (sb["season_slug"] != "preseason")]
        if pre.empty:
            print(f"odds {d}: no priced regular-season or play-in game")
            continue
        new = pd.concat(([old] if old is not None else []) + [pre], ignore_index=True)
        # Keep the FIRST and the LATEST pre-tip snapshot per game.
        new = new.sort_values("captured_at")
        keep = pd.concat([new.groupby("event_id").head(1), new.groupby("event_id").tail(1)]).drop_duplicates()
        keep = keep.reset_index(drop=True)
        out[d] = keep
        if dry:
            print(f"odds {d} (dry run): {len(pre)} priced games")
            continue
        C.write_parquet(keep, p, if_changed=True)
        print(f"odds {d}: {len(pre)} priced games snapshotted")
    return out


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


def project_date(d: dt.date, games: pd.DataFrame, sched: pd.DataFrame, roster: pd.DataFrame,
                 report, odds: pd.DataFrame | None, models: M.Models, seeds: dict,
                 today: dt.date, dry: bool) -> None:
    season = season_for_date(d)
    old, old_basis = P.read_published(d)
    frozen = P.frozen_ids(games, now_utc())
    live = games[~games["game_id"].isin(frozen)]
    kept = 0 if old is None else int(old["game_id"].isin(frozen).sum())
    if live.empty:
        print(f"projections {d}: all {len(games)} games past tip - 5 min; "
              f"{kept} committed rows stand frozen")
        return
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
    provisional, gaps = P.provisional_for(d, today, sched, season)
    now = now_utc()
    try:
        new = P.project(rows, games, sched, report, P.lines_from_odds(odds, games), models, now,
                        provisional=provisional, names=names)
    except M.NullSkew as e:
        # NoReport says "no injury report at all" itself.
        why = "" if isinstance(e, P.NoReport) else " (a null skew)"
        flag(d, today, f"projections {d}: nothing new written{why}: {e}")
        return
    out, basis = P.merge_published(old, new, games, now_utc(), old_basis=old_basis,
                                   new_basis=P.feature_digests(rows))
    withheld = new.attrs.get("withheld", [])
    projected = sorted(set(new["game_id"]))
    rewritten = set(out.loc[out["computed_at"] == pd.Timestamp(now), "game_id"])
    note = (f"{len(projected)} games projected ({len(rewritten)} rewritten, "
            f"{len(set(projected) - rewritten)} kept: no input changed), {len(frozen)} frozen "
            f"({kept} rows carried), {len(withheld)} withheld (no line yet)"
            + (f", provisional ({'tomorrow' if d > today else 'state misses ' + ', '.join(gaps[:5])})"
               if provisional else ""))
    if dry:
        print(f"projections {d} (dry run): would hold {len(out)} rows; {note}")
        return
    wrote = P.write_published(out, basis, d, old_basis)
    print(f"projections {d}: {len(out)} rows ({'updated' if wrote else 'unchanged'}); {note}")


def step_projections(sched: pd.DataFrame | None, roster: pd.DataFrame | None, report,
                     odds: dict | None, today: dt.date, dry: bool) -> None:
    if sched is None or sched.empty:
        print("projections: no schedule")
        return
    days = [(d, P.state_games(sched, d)) for d in (today, today + dt.timedelta(days=1))]
    if all(g.empty for _, g in days):
        print("projections: no regular-season or play-in games today or tomorrow "
              "(preseason is never projected)")
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
            print(f"projections {d}: no regular-season or play-in games")
            continue
        try:
            project_date(d, games, sched, roster, report, (odds or {}).get(d), models, seeds,
                         today, dry)
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
    ap.add_argument("--date", help="ET date to treat as today (default: now, ET)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    RED.clear()
    today = dt.date.fromisoformat(args.date) if args.date else today_et()
    print(f"BucketValue daily update: ET date {today}, run at {now_utc().isoformat(timespec='seconds')}"
          + (" (dry run: nothing is written)" if args.dry_run else ""))
    sched = run("schedule", step_schedule, args.dry_run)
    roster = run("roster", step_roster, args.dry_run)
    report = None
    if sched is not None and len(sched):
        run("finals", step_finals, sched, today, args.dry_run)
        report = run("injuries", step_injuries, sched, args.dry_run)
    odds = run("odds", step_odds, today, args.dry_run)
    run("projections", step_projections, sched, roster, report, odds, today, args.dry_run)
    if RED:
        print(f"\nRUN IS RED ({len(RED)}):")
        for r in RED:
            print("  -", r)
        return 1
    print("\nrun is green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
