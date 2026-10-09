"""Backfill injury-report status snapshots at serve horizons (bucketlib.horizons).

    python scripts/build_injury_horizons.py --seasons 2018-19:2025-26 [--sample N]

Per season it walks every regular-season and play-in game from the archive start
(2018-12-17), snapshots each game at the four horizons, and writes
  cache/injury_horizons_{season}_v1.parquet       player rows (player_raw, status, reason)
  cache/injury_horizon_teams_{season}_v1.parquet  team states (listed / unknown kinds)
Names are resolved to PERSON_IDs later, at feature build, against the game's own
participation rows (the most precise point-in-time roster there is).

PDFs are cached by the injury module's ReportStore, so a rerun re-parses from disk
and only fetches what is missing. A Blocked host stops the run (never "no report");
a single unparseable PDF is logged with its date and the run continues; the date
is listed at the end and the exit status is non-zero.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import horizons as HZ  # noqa: E402
from bucketlib import http as H  # noqa: E402
from bucketlib.util import seasons_between  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2018-19:2025-26")
    ap.add_argument("--sample", type=int, default=0, help="only N dates per season (dev)")
    args = ap.parse_args()
    a, _, b = args.seasons.partition(":")
    failures = []
    for season in seasons_between(a, b or a):
        sched = C.read_parquet(C.CACHE_DIR / f"schedule_{season}_v1.parquet")
        if sched is None:
            print(f"{season}: no schedule table; run scripts/build_schedule.py first")
            return 2
        sched = sched.copy()
        sched["game_date"] = pd.to_datetime(sched["game_date"]).dt.date
        games = sched[sched["game_type"].isin(["RS", "PI"]) & sched["tip_utc"].notna()
                      & (sched["game_date"] >= HZ.ARCHIVE_START)
                      & sched["home"].notna() & sched["away"].notna()]
        if games.empty:
            print(f"{season}: no games in the archive window")
            continue
        dates = sorted(games["game_date"].unique())
        if args.sample:
            step = max(1, len(dates) // args.sample)
            dates = dates[::step][: args.sample]
        out_p = C.CACHE_DIR / f"injury_horizons_{season}_v1.parquet"
        out_t = C.CACHE_DIR / f"injury_horizon_teams_{season}_v1.parquet"
        prior_p = C.read_parquet(out_p) if not args.sample else None
        prior_t = C.read_parquet(out_t) if not args.sample else None
        done = set(prior_t["game_id"]) if prior_t is not None else set()
        P = [prior_p] if prior_p is not None else []
        T = [prior_t] if prior_t is not None else []
        t0 = time.time()
        for i, d in enumerate(dates, 1):
            gd = games[(games["game_date"] == d) & ~games["game_id"].isin(done)]
            if gd.empty:
                continue
            try:
                p, t = HZ.snapshot_games(gd, sched)
            except H.Blocked as e:
                print(f"BLOCKED, stopping (not 'no report'): {e}")
                _write(P, T, out_p, out_t, season, args.sample, complete=False)
                return 3
            except (ValueError, KeyError) as e:
                failures.append((season, str(d), f"{type(e).__name__}: {str(e)[:160]}"))
                print(f"  {d}: FAILED {type(e).__name__}: {str(e)[:160]}")
                continue
            P.append(p)
            T.append(t)
            if i % 25 == 0:
                print(f"  [{season}] {i}/{len(dates)} dates, {time.time() - t0:.0f}s")
                _write(P, T, out_p, out_t, season, args.sample, complete=False)
        complete = not args.sample and not any(f[0] == season for f in failures)
        _write(P, T, out_p, out_t, season, args.sample, complete=complete)
        tt = pd.concat(T, ignore_index=True) if T else pd.DataFrame()
        pp = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
        if len(tt):
            print(f"{season}: {tt['game_id'].nunique()} games; team states by horizon:")
            print(tt.pivot_table(index="state", columns="horizon", values="game_id",
                                 aggfunc="count", fill_value=0).to_string())
        if len(pp):
            print(pp.pivot_table(index="status", columns="horizon", values="game_id",
                                 aggfunc="count", fill_value=0).to_string())
    if failures:
        print(f"\n{len(failures)} date(s) failed:")
        for f in failures:
            print("  ", *f)
        return 1
    return 0


def _write(P, T, out_p, out_t, season, sample, complete):
    if sample:
        return
    pp = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    tt = pd.concat(T, ignore_index=True) if T else pd.DataFrame()
    if len(tt):
        C.write_parquet(tt, out_t, meta={"season": season, "complete": complete})
        C.write_parquet(pp, out_p, meta={"season": season, "complete": complete})


if __name__ == "__main__":
    sys.exit(main())
