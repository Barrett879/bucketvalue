#!/usr/bin/env python
"""Build the per-game participation tables (who played, sat, was inactive,
started) for a range of seasons.

    .venv/bin/python scripts/build_participation.py --seasons 2014-15:2025-26
    .venv/bin/python scripts/build_participation.py --seasons 2023-24 --sample 5 --crosscheck
    .venv/bin/python scripts/build_participation.py --games 0052400101,0042400401 --crosscheck
    .venv/bin/python scripts/build_participation.py --seasons 2014-15:2025-26 --report-only

Writes, per season:
  cache/participation_{season}_v1.parquet        one row per (game_id, person_id)
  cache/participation_games_{season}_v1.parquet  one row per game (game_meta)
Raw payloads: cache/raw/box/{season}/ (gz JSON), reused on every rerun.

RESUMABLE: games already in the season parquet are skipped, raw payloads
already on disk are never refetched, and the parquet is rewritten every
CHECKPOINT_EVERY games, so a killed run loses at most that much parsing (and
no downloads). The parquet metadata says whether the season is complete; a
--sample run leaves complete=False.

STOPS LOUDLY on Blocked (an Akamai HTML 403 or a stats.nba.com hang): a
blocked host is never "no games". A broken invariant (ValueError) stops the
run too, after a checkpoint; --keep-going records it, carries on, and makes
the exit status non-zero.

THE GAME LIST (bucketlib.sources.box.season_game_list):
  live season  the cdn schedule's final games: no stats.nba.com call, so the
               in-season run works from GitHub Actions (stats.nba.com hangs
               from Azure). A cdn hole still falls back to V3, which is
               stats.nba.com: from Actions that run ends Blocked (exit 2).
  past season  LeagueGameLog T (Regular Season, PlayIn, Playoffs) checked
               against ScheduleLeagueV2, plus the NBA Cup Final (006 ids,
               2023-24 on), which no LeagueGameLog lists.
Every parsed game is checked against its list row: same teams, same game date,
same home team unless the list marks the site neutral.

complete=True only when the season is over and every listed game is in the
table; a live season is never complete. A table written by an older schema is
re-parsed from its raw payloads (no live calls) on the next run.

EXIT STATUS
  0  every target game is in the table and nothing disagreed
  1  an invariant failed (--keep-going) or --crosscheck found a mismatch
  2  Blocked: a host refused us (never read as "no games")
  3  BadPayload / BadRequest: wrong content or our bug
  4  INCOMPLETE: some target games were absent from every source, or not
     final yet; nothing broke (an in-season box can lag its schedule entry)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache, http  # noqa: E402
from bucketlib.sources import box as B  # noqa: E402
from bucketlib.util import season_for_date, seasons_between, today_et  # noqa: E402

CHECKPOINT_EVERY = 200
part_path = B.participation_path
games_path = B.games_path
# Metadata keys that change on every run without the data changing. A
# checkpoint that differs from the file on disk only in these is not written
# (the old stamps are kept, as cache.write_parquet(if_changed=True) does), so a
# no-op or Blocked run never churns a file that may be committed.
VOLATILE_META = ("fetched_at", "content_hash", "list_fetched_at")


def season_games(season: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """(final games, list_info) for a season; see box.season_game_list."""
    return B.season_game_list(season, refresh=refresh)


def pick_sample(games: pd.DataFrame, n: int) -> list[str]:
    """n games spread evenly over the season (date order), deterministic."""
    if n >= len(games):
        return games["game_id"].tolist()
    idx = sorted(set(np.linspace(0, len(games) - 1, n).round().astype(int).tolist()))
    return games["game_id"].iloc[idx].tolist()


def _write_if_changed(df: pd.DataFrame, path: Path, info: dict) -> bool:
    old = cache.read_parquet_meta(path)
    stable = {k: v for k, v in info.items() if k not in VOLATILE_META}
    if (old.get("content_hash") == cache.frame_hash(df)
            and {k: v for k, v in old.items() if k not in VOLATILE_META} == stable):
        return False
    cache.write_parquet(df, path, meta=info)
    return True


def checkpoint(season: str, games: pd.DataFrame, list_info: dict, old_part, old_meta,
               new_part, new_meta) -> tuple[int, bool]:
    """Merge newly parsed games into the season tables, assert the season-level
    invariants (box.check_table), and write whatever changed.
    Returns (games in the table, whether anything was written)."""
    if old_meta is None and not new_meta:
        return 0, False
    # A re-parsed game replaces its old rows wholesale (never a mix of both).
    replaced = {m["game_id"] for m in new_meta}
    metas = ([] if old_meta is None else [old_meta[~old_meta["game_id"].isin(replaced)]])
    if new_meta:
        metas.append(pd.DataFrame(new_meta, columns=B.META_COLUMNS))
    meta_df = pd.concat(metas, ignore_index=True).sort_values(["et_date", "game_id"], kind="stable")
    parts = ([] if old_part is None else [old_part[~old_part["game_id"].isin(replaced)]]) + new_part
    part_df = pd.concat(parts, ignore_index=True).sort_values(
        ["game_id", "is_home", "team"], ascending=[True, False, True], kind="stable")
    part_df, meta_df = part_df.reset_index(drop=True), meta_df.reset_index(drop=True)
    B.check_table(part_df, meta_df, season)
    done = set(meta_df["game_id"])
    listed = set(games["game_id"])
    info = {"season": season, "games_total": len(listed), "games_done": len(done & listed),
            "complete": bool(list_info.get("list_final")) and bool(listed) and done >= listed,
            "sources": dict(Counter(meta_df["source"])),
            "list_source": list_info.get("list_source"), "list_final": bool(list_info.get("list_final")),
            "list_fetched_at": list_info.get("list_fetched_at"),
            "builder": "scripts/build_participation.py"}
    wrote = _write_if_changed(part_df, part_path(season), info)
    wrote = _write_if_changed(meta_df, games_path(season), info) or wrote
    return len(done), wrote


def _stale_schema(old_part, old_meta) -> bool:
    return old_part is not None and (list(old_part.columns) != B.COLUMNS or old_meta is None
                                     or list(old_meta.columns) != B.META_COLUMNS)


def run_season(season: str, args, targets: list[str] | None, problems: dict) -> None:
    games, list_info = season_games(season, refresh=args.refresh_games)
    by_id = games.set_index("game_id")
    if targets is None:
        targets = pick_sample(games, args.sample) if args.sample else games["game_id"].tolist()
    # Parsing a game twice in one run wrote duplicate rows (--games X,X).
    targets = list(dict.fromkeys(targets))
    unknown = [g for g in targets if g not in by_id.index]
    if unknown:
        raise SystemExit(f"{season}: not in the game list (not final, or wrong season): {unknown}")
    old_part, old_meta = cache.read_parquet(part_path(season)), cache.read_parquet(games_path(season))
    if _stale_schema(old_part, old_meta):
        # Raw payloads are on disk: re-parse every game of the old table (no
        # live calls) rather than mix two schemas in one file.
        old_ids = [] if old_meta is None else [g for g in old_meta["game_id"] if g in by_id.index]
        print(f"[{season}] table schema changed: re-parsing its {len(old_ids)} games from raw payloads",
              flush=True)
        targets = list(dict.fromkeys(targets + old_ids))
        old_part = old_meta = None
    done = set() if (old_meta is None or args.reparse) else set(old_meta["game_id"])
    todo = [g for g in targets if g not in done]
    print(f"[{season}] {len(games)} final games listed ({list_info.get('list_source')}; "
          f"{dict(Counter(games['season_type']))}); targets {len(targets)}, "
          f"already done {len(targets) - len(todo)}, to do {len(todo)}", flush=True)
    new_part, new_meta = [], []
    failure: BaseException | None = None
    try:
        for i, gid in enumerate(todo, 1):
            try:
                df, meta = B.load_game(gid, args.source)
                B.check_meta_against_lgl(meta, by_id.loc[gid])
            except B.NotFinal as e:
                problems["not_final"].append(str(e))
                continue
            except http.Absent as e:
                problems["absent"].append(f"{gid}: {e}")
                print(f"  ABSENT {gid}: {e}", flush=True)
                continue
            except ValueError as e:
                if not args.keep_going:
                    raise
                problems["failed"].append(f"{gid}: {e}")
                print(f"  FAILED {gid}: {e}", flush=True)
                continue
            if args.source == "auto" and meta["source"] != B.default_source(gid):
                print(f"  {gid}: CDN has no box score, used V3 fallback", flush=True)
            new_part.append(df)
            new_meta.append(meta)
            if len(new_meta) % CHECKPOINT_EVERY == 0:
                n, _ = checkpoint(season, games, list_info, old_part, old_meta, new_part, new_meta)
                print(f"  checkpoint: {n} games on disk", flush=True)
            if i % 25 == 0 or i == len(todo):
                calls = {k: len(v) for k, v in B.CALL_LOG.items()}
                print(f"  [{season}] {i}/{len(todo)} {gid} src={meta['source']} live calls {calls}",
                      flush=True)
    except BaseException as e:
        failure = e
        raise
    finally:
        # Always persist what was parsed, including on Blocked / Ctrl-C / a
        # ValueError stop: raw payloads are on disk, parsing is the checkpoint.
        # A checkpoint that fails while another error is propagating must not
        # replace it (a Blocked run has to exit 2, not with the write error).
        try:
            n, wrote = checkpoint(season, games, list_info, old_part, old_meta, new_part, new_meta)
        except Exception as ce:  # noqa: BLE001
            if failure is None:
                raise
            print(f"[{season}] checkpoint after {type(failure).__name__} failed too: {ce}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[{season}] " + (f"{n} games in {part_path(season).name}"
                                    + ("" if wrote else " (unchanged, not rewritten)") if n
                                    else "nothing to save (no final games yet)"), flush=True)
    if args.crosscheck and new_meta:
        crosscheck(season, pd.concat(new_part, ignore_index=True), pd.DataFrame(new_meta), problems)


def crosscheck(season: str, part: pd.DataFrame, meta: pd.DataFrame, problems: dict) -> None:
    """Played set per game == LeagueGameLog P appearances (one date-filtered
    call per game date and season type), 0-minute cameo rows included. The
    NBA Cup Final is in no LeagueGameLog, so it is skipped and counted."""
    cup = meta[~meta["season_type"].isin(B.LGL_TYPES)]
    if len(cup):
        print(f"  crosscheck: {len(cup)} game(s) skipped, in no LeagueGameLog: "
              f"{cup['game_id'].tolist()}")
    meta = meta[meta["season_type"].isin(B.LGL_TYPES)]
    part = part[part["game_id"].isin(meta["game_id"])]
    if meta.empty:
        return
    lgl = []
    for (d, st), _ in meta.groupby(["et_date", "season_type"]):
        lgl.append(B.lgl_appearances(B.fetch_lgl(season, st, "P", date=d)))
    lgl = pd.concat(lgl, ignore_index=True)
    res = B.crosscheck_appearances(part, lgl)
    bad = res[(res["only_box"].str.len() > 0) | (res["only_lgl"].str.len() > 0)
              | (res["team_mismatch"].str.len() > 0)]
    zero = lgl[(lgl["lgl_min"] == 0) & lgl["game_id"].isin(part["game_id"])]
    zstat = part.merge(zero, on=["game_id", "person_id"])[["game_id", "player_name", "status",
                                                            "played_seconds", "lgl_min"]]
    print(f"  crosscheck vs LeagueGameLog P: {len(res) - len(bad)}/{len(res)} games agree; "
          f"LGL 0-minute rows: {len(zero)}, all played in box: {bool((zstat['status'] == 'played').all())}")
    for r in zstat.itertuples():
        print(f"    0-min cameo {r.game_id} {r.player_name}: {r.status} {r.played_seconds}s")
    for r in bad.itertuples():
        problems["crosscheck"].append(f"{r.game_id}: only_box={r.only_box} only_lgl={r.only_lgl} "
                                      f"team_mismatch={r.team_mismatch}")


# ── Coverage report ──────────────────────────────────────────────────────────
def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def report(seasons: list[str], problems: dict) -> None:
    print("\n=== COVERAGE ===")
    tot_part, tot_meta = [], []
    for s in seasons:
        part, meta = cache.read_parquet(part_path(s)), cache.read_parquet(games_path(s))
        info = cache.read_parquet_meta(games_path(s))
        if part is None:
            print(f"{s}: no table")
            continue
        if _stale_schema(part, meta):
            print(f"{s}: table from an older schema ({len(part.columns)} columns); "
                  f"rerun without --report-only to re-parse it from raw")
            continue
        tot_part.append(part)
        tot_meta.append(meta)
        ct = pd.crosstab([part["game_id"], part["team"]], part["status"]).reindex(
            columns=list(B.STATUSES), fill_value=0)
        per = ct.sum(axis=1)
        print(f"{s}: games {info.get('games_done')}/{info.get('games_total')} "
              f"complete={info.get('complete')} sources={info.get('sources')} "
              f"by type={dict(Counter(meta['season_type']))}; list: {info.get('list_source')} "
              f"(final={info.get('list_final')})")
        print(f"   players/team-game min {per.min()} median {per.median():.0f} max {per.max()}; "
              f"mean played {ct['played'].mean():.1f} dnp {ct['dnp'].mean():.1f} "
              f"inactive {ct['inactive'].mean():.1f}")
    if not tot_part:
        return
    part, meta = pd.concat(tot_part, ignore_index=True), pd.concat(tot_meta, ignore_index=True)
    starters = part[part["starter"]].groupby(["game_id", "team"]).size()
    diff = B.team_seconds_diff(part, meta)
    # Playoff boxes legitimately list no inactives (two-ways are ineligible and
    # the whole 15 can dress: 2024 Finals G5 has none on either side), so the
    # silent-empty canary counts regular-season team-games only.
    rs = part[part["game_id"].str[2] == "2"]
    zero_inact = rs.groupby(["game_id", "team"])["status"].apply(lambda s: (s == "inactive").sum())
    # A game parsed from its non-default source is a CDN hole only when the
    # CDN said so (the cdnabsent marker); --source v3 is a request, not a hole.
    off = meta[meta["source"] != meta["game_id"].map(B.default_source)]
    hole = off["game_id"].map(lambda g: B.raw_path(g, "cdnabsent").exists()).astype(bool)
    fb, forced = off[hole], off[~hole]
    print(f"TOTAL games {meta['game_id'].nunique()}, team-games {len(starters)}, rows {len(part)}")
    print(f"cdn-era games served by the V3 fallback (CDN absent): {len(fb)} {fb['game_id'].tolist()[:10]}")
    print(f"games parsed from a non-default source by request (--source): {len(forced)} "
          f"{forced['game_id'].tolist()[:10]}")
    print(f"status counts {dict(Counter(part['status']))}")
    print(f"starters: {int((starters == 5).sum())}/{len(starters)} team-games have exactly 5; "
          f"played starters with 0s: {int(((part['starter']) & (part['played_seconds'] <= 0)).sum())}")
    print(f"team player-seconds vs 240+25*OT min: exact {int((diff == 0).sum())}/{len(diff)}, "
          f"max |diff| {int(diff.abs().max())}s")
    print(f"played with 0 seconds (sub-second cameo): {int(((part.status == 'played') & (part.played_seconds == 0)).sum())}; "
          f"played under 60s: {int(((part.status == 'played') & (part.played_seconds < 60)).sum())}")
    print(f"regular-season team-games with zero inactives: {int((zero_inact == 0).sum())}/"
          f"{len(zero_inact)} (a silent-empty summary would show up here)")
    print("absence_kind of non-players (coach = healthy scratch; status 'dnp' alone mixes them):")
    np_ = part[part.status != "played"]
    for (src, st, kind), n in sorted(Counter(zip(np_["source"], np_["status"], np_["absence_kind"])).items()):
        print(f"   {src:4s} {st:9s} {kind:12s} {n}")
    nr = part[(part.status != "played") & (part.reason_code == "")]
    print(f"non-players with no reason code: {len(nr)} {dict(Counter(nr['status']))}")
    waived = part[part["reason_text"].str.contains("Waived", case=False, na=False)]
    print(f"inactive rows described 'Waived' (not on the roster any more): {len(waived)}")
    print("reason_code counts:")
    for (src, code), n in Counter(zip(part["source"], part["reason_code"])).most_common():
        if code:
            print(f"   {src:4s} {code:42s} {n}")
    print("latency (live calls this run):")
    for kind, xs in sorted(B.CALL_LOG.items()):
        print(f"   {kind:6s} n={len(xs):4d} mean {statistics.mean(xs):.3f}s p50 {_q(xs, .5):.3f}s "
              f"p90 {_q(xs, .9):.3f}s max {max(xs):.3f}s")
    for k, v in problems.items():
        print(f"{k}: {len(v)}")
        for line in v[:20]:
            print(f"   {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", help="A:B or one season, e.g. 2014-15:2025-26")
    ap.add_argument("--games", help="comma-separated game ids (their seasons are inferred)")
    ap.add_argument("--sample", type=int, default=0, help="N games per season, spread evenly")
    ap.add_argument("--source", default="auto", choices=["auto", "cdn", "v3"])
    ap.add_argument("--crosscheck", action="store_true",
                    help="compare played sets with LeagueGameLog P (one call per game date)")
    ap.add_argument("--keep-going", action="store_true",
                    help="record invariant failures and continue (exit status 1)")
    ap.add_argument("--reparse", action="store_true", help="re-parse targets from raw payloads")
    ap.add_argument("--refresh-games", action="store_true", help="refetch the past-season game lists (LeagueGameLog T + schedule)")
    ap.add_argument("--report-only", action="store_true", help="coverage report from disk, no network")
    args = ap.parse_args(argv)

    targets: dict[str, list[str] | None] = {}
    if args.games:
        for gid in [g.strip() for g in args.games.split(",") if g.strip()]:
            targets.setdefault(B.game_season(gid), []).append(gid)
    if args.seasons:
        a, _, b = args.seasons.partition(":")
        for s in seasons_between(a, b or a):
            targets.setdefault(s, None)
    if not targets:
        ap.error("give --seasons and/or --games")
    seasons = sorted(targets)
    problems = {"not_final": [], "absent": [], "failed": [], "crosscheck": []}
    if not args.report_only:
        for s in seasons:
            try:
                run_season(s, args, targets[s], problems)
            except http.Blocked as e:
                print(f"\nBLOCKED, stopping (not 'no games'): {e}", file=sys.stderr)
                report(seasons, problems)
                return 2
            except http.FetchError as e:
                # BadRequest = our bug; BadPayload = wrong content. Neither is data.
                print(f"\n{type(e).__name__}, stopping: {e}", file=sys.stderr)
                report(seasons, problems)
                return 3
    report(seasons, problems)
    code = exit_code(problems)
    print(json.dumps({"live_calls": {k: len(v) for k, v in B.CALL_LOG.items()},
                      "problems": {k: len(v) for k, v in problems.items()}, "exit": code}))
    return code


def exit_code(problems: dict) -> int:
    """1 when something disagreed or broke; 4 when target games are missing
    (absent everywhere, or not final yet) and nothing broke; else 0. A table
    with missing games is marked complete=False too, but an orchestrator that
    reads only the exit status must see the gap."""
    if problems["failed"] or problems["crosscheck"]:
        return 1
    if problems["absent"] or problems["not_final"]:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
