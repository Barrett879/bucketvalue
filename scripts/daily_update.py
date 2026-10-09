"""The bot: refresh the current season from sources reachable from GitHub Actions.

    python scripts/daily_update.py [--date YYYY-MM-DD] [--dry-run]

Every source here was PROVEN reachable from Actions (docs/probes/): cdn.nba.com
(schedule, liveData box scores, playerIndex.json), the ak-static injury PDFs, and
ESPN. stats.nba.com is never called (it hangs from Azure).

Steps, each independent (one failing step does not skip the others):
  1. schedule      cdn schedule            -> cache/schedule_current_v1.parquet
  2. roster        cdn playerIndex          -> cache/roster_current_v1.parquet
  3. finals        cdn box scores of every FINAL state game not yet ingested
                   -> cache/season/{season}/{pg,tg,part,games}_{date}.parquet
                   (per-ET-date partitions, immutable once all that date's games
                   are final; the stat lines equal the stats logs exactly,
                   tests/test_cdnbox.py)
  4. injuries      latest real report      -> cache/injury_latest_v1.parquet
  5. odds          ESPN pregame snapshot   -> cache/season/{season}/odds_{date}.parquet
  (6. projections  wired once models ship)

Rules (ops review):
  - The ET date is derived HERE, every run (cron starts were measured 2-7 hours
    late on DiamondValue; nothing assumes which run this is).
  - Every committed write uses if_changed=True, so an unchanged file stays
    byte-identical and the bot does not commit, push and redeploy for nothing.
  - Stale beats empty: an empty or failed pull never overwrites good data.
  - The run goes RED (exit 1) on any Blocked source, or when a date with final
    games ingested zero of them, or when ingestion trails the last final game.
    A green run with stale data is the failure mode this exists to prevent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import http as H  # noqa: E402
from bucketlib.sources import cdn, cdnbox  # noqa: E402
from bucketlib.util import now_et, now_utc, season_for_date, today_et  # noqa: E402

STATE_TYPES = {"RS", "PI"}
RED: list[str] = []


def red(msg: str) -> None:
    RED.append(msg)
    print(f"RED: {msg}")


def season_dir(season: str) -> Path:
    p = C.CACHE_DIR / "season" / season
    p.mkdir(parents=True, exist_ok=True)
    return p


def step_schedule(dry: bool) -> pd.DataFrame | None:
    try:
        sched = cdn.parse_schedule(cdn.fetch_schedule())
    except H.Blocked as e:
        red(f"schedule blocked: {e}")
        return C.read_parquet(C.CACHE_DIR / "schedule_current_v1.parquet")
    if sched.empty:
        red("schedule parsed empty; keeping the last good copy")
        return C.read_parquet(C.CACHE_DIR / "schedule_current_v1.parquet")
    if not dry:
        wrote = C.write_parquet(sched, C.CACHE_DIR / "schedule_current_v1.parquet", if_changed=True)
        print(f"schedule: {len(sched)} games ({'updated' if wrote else 'unchanged'})")
    return sched


def step_roster(dry: bool) -> None:
    try:
        r = cdn.parse_player_index(cdn.fetch_player_index())
    except H.Blocked as e:
        red(f"roster blocked: {e}")
        return
    if len(r) < 400:
        red(f"roster has only {len(r)} players; keeping the last good copy")
        return
    if not dry:
        wrote = C.write_parquet(r, C.CACHE_DIR / "roster_current_v1.parquet", if_changed=True)
        print(f"roster: {len(r)} players, {int(r['two_way'].sum())} two-way ({'updated' if wrote else 'unchanged'})")


def ingested_ids(season: str) -> set[str]:
    ids = set()
    for f in season_dir(season).glob("games_*.parquet"):
        ids |= set(pd.read_parquet(f)["game_id"])
    return ids


def step_finals(sched: pd.DataFrame, today: dt.date, dry: bool) -> None:
    from bucketlib.sources import box  # participation parser (status, starters, reasons)
    finals = sched[(sched["status"] == "final") & sched["game_type"].isin(STATE_TYPES)
                   & (sched["game_date"] <= today)]
    if finals.empty:
        print("finals: none yet this season")
        return
    season = finals["season"].iloc[0]
    have = ingested_ids(season)
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
    # Completeness: every final state game up to yesterday must be ingested.
    missing = set(finals.loc[finals["game_date"] < today, "game_id"]) - ingested_ids(season)
    if missing:
        red(f"{len(missing)} final games before today are not ingested, e.g. {sorted(missing)[:5]}")


def cdn_box(game_id: str) -> dict:
    return H.fetch(cdn.BOX_URL.format(gid=game_id), timeout=20, retries=2)


def step_injuries(sched: pd.DataFrame, dry: bool) -> None:
    from bucketlib.sources import injury as INJ
    sel = INJ.latest_report(now_et(), max_slots=12)
    if sel is None:
        print("injuries: no report found (preseason or between reports): keeping the last good table")
        return
    rep = sel.report
    sched_v = sched.dropna(subset=["home", "away"])[["game_id", "game_date", "away", "home", "tip_utc"]]
    kept, dropped = INJ.validate_rows(rep.rows, sched_v) if rep.rows is not None and len(rep.rows) else (rep.rows, None)
    if kept is None or kept.empty:
        print(f"injuries: report {rep.meta['report_ts_et']} lists no real NBA games (QA/test file?); keeping last good")
        return
    kept = kept.copy()
    kept["captured_at"] = now_utc()
    if not dry:
        wrote = C.write_parquet(kept, C.CACHE_DIR / "injury_latest_v1.parquet",
                                meta={"report_ts_et": str(rep.meta["report_ts_et"])}, if_changed=True)
        print(f"injuries: {len(kept)} rows from {rep.meta['report_ts_et']} ({'updated' if wrote else 'unchanged'})")


def step_odds(today: dt.date, dry: bool) -> None:
    from bucketlib.sources import espn
    for d in (today, today + dt.timedelta(days=1)):
        try:
            sb = espn.scoreboard(d)
        except H.Blocked as e:
            red(f"odds blocked: {e}")
            return
        pre = sb[sb["status"].isin(["STATUS_SCHEDULED"]) & sb["home_line"].notna()]
        if pre.empty or dry:
            continue
        p = season_dir(season_for_date(d)) / f"odds_{d.isoformat()}.parquet"
        old = C.read_parquet(p)
        new = pd.concat(([old] if old is not None else []) + [pre], ignore_index=True)
        # Keep the FIRST and the LATEST pre-tip snapshot per game.
        new = new.sort_values("captured_at")
        keep = pd.concat([new.groupby("event_id").head(1), new.groupby("event_id").tail(1)]).drop_duplicates()
        C.write_parquet(keep.reset_index(drop=True), p, if_changed=True)
        print(f"odds {d}: {len(pre)} priced games snapshotted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="ET date to treat as today (default: now, ET)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.date) if args.date else today_et()
    print(f"BucketValue daily update: ET date {today}, run at {now_utc().isoformat(timespec='seconds')}")
    sched = step_schedule(args.dry_run)
    step_roster(args.dry_run)
    if sched is not None and len(sched):
        step_finals(sched, today, args.dry_run)
        step_injuries(sched, args.dry_run)
    step_odds(today, args.dry_run)
    if RED:
        print(f"\nRUN IS RED ({len(RED)}):")
        for r in RED:
            print("  -", r)
        return 1
    print("\nrun is green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
