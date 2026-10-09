"""Build the league schedule per season from stats.nba.com scheduleleaguev2.

    python scripts/build_schedule.py --seasons 2014-15:2026-27
    python scripts/build_schedule.py --seasons 2026-27          # the daily refresh
    python scripts/build_schedule.py --sample 2
    python scripts/build_schedule.py --rebuild                  # cache only, 0 calls

Writes cache/schedule_<season>_v1.parquet (one call per season; old seasons
back to 2014-15 answer in the same format, verified 2026-10-08).

Why the schedule matters beyond the slate: the NBA Cup final (game id 006...)
is in no Regular Season or Playoffs log, but two teams play it, so rest and
back-to-back features must take game dates from here, not from the logs.

The CURRENT season is always refetched: it lists only 1206 of 1230 regular-
season games until the Cup group stage ends, and Cup knockout games carry TBD
teams until they are decided. A refetch that lists fewer regular-season games
than the file on disk is refused (stale beats empty). Completed seasons are
skipped unless --force. --rebuild re-parses the raw cache only: a season
whose raw schedule is missing is reported "missing_raw", never fetched. The
table carries the raw payload's fetched_at, not the write time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from bucketlib import http, util  # noqa: E402
from bucketlib.sources import statsnba as S  # noqa: E402

SCRIPT = "build_schedule"
DEFAULT_SEASONS = f"2014-15:{S.current_season()}"


def _rs_games(df: pd.DataFrame) -> int:
    return int((df["game_type"] == "RS").sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--force", action="store_true", help="refetch completed seasons")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-parse and rewrite from the raw cache, no refetch (after a parser change)")
    ap.add_argument("--max-calls", type=int, default=None, metavar="N",
                    help="stop before making more than N live stats.nba.com calls")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="only the first N seasons of the range")
    args = ap.parse_args(argv)
    if args.force and args.rebuild:
        ap.error("--force refetches, --rebuild never fetches: pick one")
    S.set_call_budget(args.max_calls)
    seasons = S.parse_season_range(args.seasons)
    if args.sample:
        seasons = seasons[: args.sample]
    progress = S.load_progress(SCRIPT)
    failures, t_all = [], time.time()
    print(f"build_schedule: {len(seasons)} seasons; current season {S.current_season()}", flush=True)
    for i, season in enumerate(seasons, 1):
        path = S.tidy_path("schedule", season)
        live = S.is_live_season(season)
        if path.exists() and not live and not (args.force or args.rebuild):
            print(f"[{i}/{len(seasons)}] {season}: done, skipped", flush=True)
            continue
        t0, c0 = time.time(), S.call_count()
        try:
            df = S.schedule(season, refresh=True if args.force else (False if args.rebuild else None))
        except (http.Blocked, S.CallBudgetExceeded) as e:
            # A budget stop is not a block: the host never refused anything.
            status = "budget" if isinstance(e, S.CallBudgetExceeded) else "blocked"
            progress[season] = {"status": status, "error": str(e)[:400]}
            S.save_progress(SCRIPT, progress)
            failures.append((season, str(e)))
            print(f"[{i}/{len(seasons)}] {season}: {status.upper()} {e}; stopping", flush=True)
            break
        except S.CacheMiss as e:
            progress[season] = {"status": "missing_raw", "error": str(e)[:400]}
            S.save_progress(SCRIPT, progress)
            failures.append((season, str(e)))
            print(f"[{i}/{len(seasons)}] {season}: MISSING RAW (no call made) {e}", flush=True)
            continue
        except (ValueError, http.FetchError) as e:
            progress[season] = {"status": "failed", "error": str(e)[:400]}
            S.save_progress(SCRIPT, progress)
            failures.append((season, str(e)))
            print(f"[{i}/{len(seasons)}] {season}: FAILED (not written) {e}", flush=True)
            continue
        rep = df.attrs["report"]
        ok, msg = S.write_tidy(df, path, {"source": "stats.nba.com scheduleleaguev2",
                                          "season": season, "report": rep,
                                          "fetched_at": df.attrs.get("fetched_at")},
                               live=live, count=_rs_games)
        progress[season] = {"status": "done" if ok else "kept", "note": msg,
                            "built_at": util.now_et().isoformat()}
        S.save_progress(SCRIPT, progress)
        print(f"[{i}/{len(seasons)}] {season}: {len(df)} games, {S.call_count() - c0} calls, "
              f"{time.time() - t0:.1f}s; {msg}", flush=True)

    lines = []
    for season in seasons:
        path = S.tidy_path("schedule", season)
        if not path.exists():
            lines.append({"season": season, "status": "missing"})
            continue
        r = S.tidy_meta(path).get("report", {})
        bt = r.get("by_type", {})
        lines.append({
            "season": season, "status": "ok", "rows": r.get("rows"),
            "RS": bt.get("RS", 0), "RS_expected": r.get("rs_games_expected"),
            "per_team": f"{r.get('rs_games_per_team_min')}-{r.get('rs_games_per_team_max')}",
            "PI": bt.get("PI", 0), "PO": bt.get("PO", 0), "PRE": bt.get("PRE", 0),
            "cup_final": r.get("cup_final_games"), "tbd": r.get("tbd_games"),
            "no_tip": r.get("unknown_tip"), "neutral": r.get("neutral_games"),
            "neutral_by_city": r.get("neutral_by_city_only"),
            "drop_non_nba": r.get("dropped_non_nba"), "drop_allstar": r.get("dropped_allstar"),
        })
    print("\nCOVERAGE (schedule)")
    print(pd.DataFrame(lines).fillna("").to_string(index=False))
    print(f"\nlive calls this run: {S.call_count()}, elapsed {time.time() - t_all:.1f}s")
    if failures:
        print(f"\n{len(failures)} season(s) failed:")
        for s, e in failures:
            print(f"  {s}: {e[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
