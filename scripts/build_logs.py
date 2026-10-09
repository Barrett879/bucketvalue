"""Build the per-season player-game and team-game tables from stats.nba.com.

    python scripts/build_logs.py --seasons 2014-15:2025-26 --types RS,PI,PO
    python scripts/build_logs.py --seasons 2024-25 --types RS --force
    python scripts/build_logs.py --seasons 2014-15:2025-26 --sample 2
    python scripts/build_logs.py --seasons 2014-15:2025-26 --rebuild   # cache only, 0 calls

Writes cache/player_games_<season>_v1.parquet and cache/team_games_<season>_v1.parquet,
one file per season holding every requested season type (column season_type).
Each season type costs 5 whole-season calls (PGL Base, PGL Advanced, LGL P,
LGL T, TeamGameLogs Advanced), plus one schedule call per season (neutral
sites and first game dates come from it); PlayIn before 2019-20 costs none.
Every raw payload is cached under cache/raw/statsnba/<season>/ the moment it
arrives, so a rerun after an interruption makes no call it already made.
--rebuild is CACHE-ONLY: a missing raw payload fails that season (reported
"missing_raw"), it is never fetched.

Resumable: a completed season whose parquet already holds the requested types
is skipped unless --force. The CURRENT season is always refetched (its logs
grow daily) and neither of its tables is overwritten by a smaller pull: if
either would shrink, BOTH are kept, so player_games and team_games never come
from different pulls (stale beats empty). The tables carry the oldest raw
fetched_at they were built from, not the write time.
A season that fails an invariant is reported and NOT written; the run goes on
to the next season and exits non-zero at the end.
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

SCRIPT = "build_logs"


def parse_types(spec: str) -> list[str]:
    codes = [S.stype_code(t.strip()) for t in spec.split(",") if t.strip()]
    if not codes:
        raise ValueError(f"no season types in {spec!r}")
    return list(dict.fromkeys(codes))


def season_done(season: str, types: list[str]) -> bool:
    """A completed season is done when both parquets exist and hold every
    requested type. A live season is never done."""
    if S.is_live_season(season):
        return False
    for table in ("player_games", "team_games"):
        p = S.tidy_path(table, season)
        if not p.exists() or not set(types) <= set(S.tidy_meta(p).get("types", [])):
            return False
    return True


def build_season(season: str, types: list[str], force: bool, rebuild: bool = False) -> dict:
    """Fetch, parse and write one season. Returns its progress record.

    force: refetch every payload. rebuild: re-parse from the raw cache only
    (after a parser change); a missing payload raises S.CacheMiss.
    None (neither): live seasons refetch once, completed seasons use the cache.
    """
    refresh = True if force else (False if rebuild else None)
    pgs, tgs, reports, stamps = [], [], {}, []
    for code in types:
        t0, c0 = time.time(), S.call_count()
        p = S.player_games(season, code, refresh=refresh)
        t = S.team_games(season, code, refresh=refresh)
        pr, tr = dict(p.attrs["report"]), dict(t.attrs["report"])
        reports[code] = {"player": pr, "team": tr}
        stamps += [p.attrs.get("fetched_at"), t.attrs.get("fetched_at")]
        pgs.append(p)
        tgs.append(t)
        note = f" ({pr['skipped']})" if pr.get("skipped") else ""
        print(f"  {season} {code}: {len(p):>6} player rows, {tr.get('games', 0):>4} games, "
              f"{S.call_count() - c0} calls, {time.time() - t0:.1f}s{note}", flush=True)
    live = S.is_live_season(season)
    built = {}
    for new, table in ((pgs, "player_games"), (tgs, "team_games")):
        path = S.tidy_path(table, season)
        built[table] = (path,) + merge_with_existing(path, new, types, reports, season,
                                                     S.oldest_stamp(*stamps))
    # Both tables or neither: keeping one and replacing the other would pair
    # a player table and a team table from different pulls.
    refused = [r for path, df, _ in built.values() if (r := S.shrink_reason(df, path, live))]
    msgs, sizes = [], {}
    for table, (path, df, meta) in built.items():
        sizes[table] = len(df)
        if refused:
            msg = f"kept existing {path.name}: " + "; ".join(refused)
        else:
            _, msg = S.write_tidy(df, path, meta, live=live)
        msgs.append(msg)
        print(f"  {msg}", flush=True)
    return {"status": "kept" if refused else "done", "types": types,
            "rows": sizes["player_games"], "team_rows": sizes["team_games"], "notes": msgs,
            "built_at": util.now_et().isoformat()}


def merge_with_existing(path: Path, new: list[pd.DataFrame], types: list[str],
                        reports: dict, season: str,
                        fetched_at: float | None = None) -> tuple[pd.DataFrame, dict]:
    """New rows for `types` + the existing file's rows for every OTHER type.

    One file holds all of a season's types, so `--types RS --rebuild` must not
    silently drop the season's playoffs (it did, once, in development). The
    result's fetched_at is the oldest of the new rows' and the kept rows'.
    """
    meta_old = S.tidy_meta(path) if path.exists() else {}
    keep_types = [t for t in meta_old.get("types", []) if t not in types]
    if keep_types:
        fetched_at = S.oldest_stamp(fetched_at, meta_old.get("fetched_at"))
    parts = list(new)
    all_reports = {k: v for k, v in (meta_old.get("reports") or {}).items() if k in keep_types}
    if keep_types:
        old = pd.read_parquet(path)
        parts.insert(0, old[old["season_type"].isin(keep_types)])
    all_reports.update(reports)
    order = [t for t in ("RS", "PI", "PO") if t in set(keep_types) | set(types)]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("season_type", key=lambda s: s.map(order.index), kind="stable")
    meta = {"source": "stats.nba.com", "season": season, "types": order, "reports": all_reports,
            "fetched_at": fetched_at}
    return df.reset_index(drop=True), meta


def coverage(seasons: list[str]) -> list[dict]:
    """One line per (season, type) from the written parquets' metadata."""
    out = []
    for season in seasons:
        p = S.tidy_path("player_games", season)
        if not p.exists():
            out.append({"season": season, "type": "-", "status": "missing"})
            continue
        meta = S.tidy_meta(p)
        reps = meta.get("reports") or {}
        for code in [t for t in meta.get("types", []) if t in reps]:
            pr, tr = reps[code]["player"], reps[code]["team"]
            dis = pr.get("pgl_vs_lgl_stat_disagreements") or {}
            out.append({
                "season": season, "type": code, "status": "ok",
                "rows": pr.get("rows", 0), "games": tr.get("games", 0),
                "players": pr.get("players", 0), "lgl_only": pr.get("lgl_only", 0),
                "pgl_only": pr.get("pgl_only", 0), "no_adv": pr.get("base_without_adv", 0),
                "min_dev_max": pr.get("min_sum_max_abs_dev"),
                "min_dev_gt_0_5": pr.get("min_sum_dev_over_0_5", 0),
                "min_known_exc": pr.get("min_sum_known_exceptions", 0),
                "fantasy_n": pr.get("fantasy_checked", 0),
                "fantasy_err": pr.get("fantasy_max_abs_err"),
                "stat_disagree": sum(dis.values()), "dd2_src_diff": pr.get("source_dd2_disagreements", 0),
                "pm_sum_viol": pr.get("plus_minus_sum_violations", 0),
                "team_sum_diff": sum((pr.get("team_sum_mismatches") or {}).values()),
                "ot_games": tr.get("ot_games", 0), "neutral": tr.get("neutral_games", 0),
            })
    return out


def print_coverage(lines: list[dict]) -> None:
    cols = ["season", "type", "status", "rows", "games", "players", "lgl_only", "pgl_only",
            "no_adv", "min_dev_max", "min_dev_gt_0_5", "min_known_exc", "fantasy_n",
            "fantasy_err", "stat_disagree", "dd2_src_diff", "pm_sum_viol", "team_sum_diff",
            "ot_games", "neutral"]
    df = pd.DataFrame(lines, columns=cols)
    if "fantasy_err" in df:
        df["fantasy_err"] = df["fantasy_err"].map(lambda v: "" if v is None or v != v else f"{v:.0e}")
    print("\nCOVERAGE (player_games / team_games)")
    print(df.fillna("").to_string(index=False))
    ok = df[df.status == "ok"]
    if len(ok):
        print(f"\ntotal rows {int(ok.rows.sum())}, games {int(ok.games.sum())}, "
              f"min-sum max deviation {ok.min_dev_max.max()} min "
              f"(tolerance {S.MINUTES_TOLERANCE}), fantasy rows checked {int(ok.fantasy_n.sum())}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2014-15:2025-26")
    ap.add_argument("--types", default="RS,PI,PO")
    ap.add_argument("--force", action="store_true", help="refetch and rewrite completed seasons")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-parse and rewrite from the raw cache, no refetch (after a parser change)")
    ap.add_argument("--max-calls", type=int, default=None, metavar="N",
                    help="stop before making more than N live stats.nba.com calls")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="only the first N seasons of the range")
    args = ap.parse_args(argv)
    S.set_call_budget(args.max_calls)
    seasons = S.parse_season_range(args.seasons)
    types = parse_types(args.types)
    if args.sample:
        seasons = seasons[: args.sample]
    progress = S.load_progress(SCRIPT)
    failures = []
    t_all = time.time()
    if args.force and args.rebuild:
        ap.error("--force refetches, --rebuild never fetches: pick one")
    todo = [s for s in seasons if args.force or args.rebuild or not season_done(s, types)]
    print(f"build_logs: {len(seasons)} seasons x {types}; current season {S.current_season()}; "
          f"at most {S.planned_log_calls(todo, types, args.force, args.rebuild)} live calls planned",
          flush=True)
    for i, season in enumerate(seasons, 1):
        if not (args.force or args.rebuild) and season_done(season, types):
            print(f"[{i}/{len(seasons)}] {season}: done, skipped", flush=True)
            continue
        print(f"[{i}/{len(seasons)}] {season}: building", flush=True)
        try:
            progress[season] = build_season(season, types, args.force, args.rebuild)
        except S.CallBudgetExceeded as e:
            progress[season] = {"status": "budget", "error": str(e)[:400]}
            S.save_progress(SCRIPT, progress)
            failures.append((season, str(e)))
            print(f"  STOPPED: {e}; rerun resumes from the raw cache", flush=True)
            break
        except S.CacheMiss as e:
            # --rebuild is cache-only: this season lacks a raw payload.
            progress[season] = {"status": "missing_raw", "error": str(e)[:400]}
            failures.append((season, str(e)))
            print(f"  MISSING RAW (not written, no call made): {e}", flush=True)
        except http.Blocked as e:
            # A hang/refusal is never "no games": stop, the rest would fail too.
            progress[season] = {"status": "blocked", "error": str(e)[:400]}
            S.save_progress(SCRIPT, progress)
            failures.append((season, str(e)))
            print(f"  BLOCKED: {e}\n  stopping; rerun resumes from the raw cache", flush=True)
            break
        except (ValueError, http.FetchError) as e:
            progress[season] = {"status": "failed", "error": str(e)[:400]}
            failures.append((season, str(e)))
            print(f"  FAILED (not written): {e}", flush=True)
        S.save_progress(SCRIPT, progress)
    print_coverage(coverage(seasons))
    print(f"\nlive calls this run: {S.call_count()}, elapsed {time.time() - t_all:.1f}s")
    if failures:
        print(f"\n{len(failures)} season(s) failed:")
        for s, e in failures:
            print(f"  {s}: {e[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
