"""Build player identity/bio tables from stats.nba.com.

    python scripts/build_players.py --seasons 2014-15:2026-27      # 390 + 1 calls
    python scripts/build_players.py --seasons 2026-27               # daily: 30 + 1
    python scripts/build_players.py --sample 3                      # 3 team-seasons

Writes:
  cache/player_index_<live season>_v1.parquet   playerindex: the league roster
      with two-way flags. LIVE SEASON ONLY: for a past season playerindex returns
      the players whose last season was then, not that season's roster.
  cache/team_rosters_<season>_v1.parquet        CommonTeamRoster, all 30 teams;
      written only when every team of the season parsed.
  cache/player_bio_v1.parquet                   one row per person_id: birth date,
      latest height/weight/position, seasons seen. Written only when every
      requested team-season is present; a partial run (--sample, failures)
      writes cache/sample/ copies instead so nothing partial sits at a canonical
      path.

Past-season CommonTeamRoster is the END-OF-SEASON roster: a player waived in
January is on no 2014-15 roster at all, and AGE is computed at a later
reference date (Kawhi Leonard, born 1991-06-29, shows AGE 24 for 2014-15). Use
birth_date and the game date, never AGE; take point-in-time team from the game
logs. The coverage report measures how many player-game person_ids lack a bio.
HOW_ACQUIRED and the two-way flag describe the player's CURRENT record (a
2018-19 HOU row reads 'Traded from ATL on 07/06/25'), so they are withheld
(null) for past seasons and kept, as of the pull, for the live season only.

No row-count guard on these tables: rosters SHRINK legitimately (camp
rosters of 20+ are cut to 15 + 3 two-way before opening night), and the
parsers already refuse an empty or incomplete pull (an empty roster raises,
playerindex must cover all 30 teams, team_rosters is written to its canonical
path only with 30/30 teams). Every table carries the oldest raw fetched_at
it was built from, not the write time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from bucketlib import http, teams, util  # noqa: E402
from bucketlib.cache import CACHE_DIR  # noqa: E402
from bucketlib.sources import statsnba as S  # noqa: E402

SCRIPT = "build_players"
DEFAULT_SEASONS = f"2014-15:{S.current_season()}"
BIO_PATH = CACHE_DIR / "player_bio_v1.parquet"
SAMPLE_DIR = CACHE_DIR / "sample"


def spread(pairs: list, n: int) -> list:
    """n pairs spread evenly over the list, so a sample touches old and new
    seasons (the roster format differs: SUPPLEMENTAL_STATUS only in the live
    season's payload)."""
    if n <= 0 or n >= len(pairs):
        return pairs
    step = (len(pairs) - 1) / max(n - 1, 1)
    return [pairs[round(i * step)] for i in range(n)]


def build_bio(rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Person-level bio from team-season roster rows.

    Birth date must agree across seasons; a disagreement is a source error, so
    the modal value is kept and every conflict is COUNTED and listed.
    """
    r = rosters.sort_values(["person_id", "season"])
    births = r.dropna(subset=["birth_date"]).groupby("person_id")["birth_date"]
    n_birth = births.nunique()
    conflicts = n_birth[n_birth > 1].index.tolist()

    def latest(col):
        return r.dropna(subset=[col]).groupby("person_id")[col].last()

    bio = pd.DataFrame({
        "player_name": r.groupby("person_id")["player_name"].last(),
        "birth_date": births.agg(lambda s: s.mode().iloc[0]),
        "height_in": latest("height_in"), "weight": latest("weight"),
        "position": latest("position"),
        "first_season": r.groupby("person_id")["season"].min(),
        "last_season": r.groupby("person_id")["season"].max(),
        "team_seasons": r.groupby("person_id").size(),
    }).reset_index()
    bio["birth_date"] = bio["birth_date"].astype("datetime64[ns]")
    report = {"persons": len(bio), "birth_date_missing": int(bio.birth_date.isna().sum()),
              "birth_date_conflicts": len(conflicts),
              "conflict_rows": r[r.person_id.isin(conflicts)][
                  ["person_id", "player_name", "season", "birth_date"]].astype(str)
                  .head(30).to_dict("records")}
    return bio, report


def games_coverage(bio: pd.DataFrame, seasons: list[str]) -> list[dict]:
    """Share of player-game person_ids (from already-built logs) with a birth date."""
    have = set(bio.loc[bio.birth_date.notna(), "person_id"])
    out = []
    for season in seasons:
        p = S.tidy_path("player_games", season)
        if not p.exists():
            continue
        ids = set(pd.read_parquet(p, columns=["person_id"])["person_id"])
        miss = ids - have
        out.append({"season": season, "logged_players": len(ids), "without_bio": len(miss)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--teams", default="", help="comma list of tricodes (default: all 30)")
    ap.add_argument("--force", action="store_true", help="refetch cached completed rosters")
    ap.add_argument("--max-calls", type=int, default=None, metavar="N",
                    help="stop before making more than N live stats.nba.com calls")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="only N team-seasons, spread over the range")
    ap.add_argument("--no-index", action="store_true", help="skip the live playerindex call")
    args = ap.parse_args(argv)
    S.set_call_budget(args.max_calls)
    seasons = S.parse_season_range(args.seasons)
    codes = [teams.get(t.strip()).code for t in args.teams.split(",") if t.strip()] or list(teams.CODES)
    pairs = [(s, t) for s in seasons for t in codes]
    run_pairs = spread(pairs, args.sample) if args.sample else pairs
    partial = bool(args.sample) or len(codes) < 30
    refresh = True if args.force else None
    progress = S.load_progress(SCRIPT)
    failures, t_all = [], time.time()
    print(f"build_players: {len(run_pairs)} team-seasons over {len(seasons)} seasons; "
          f"current season {S.current_season()}", flush=True)

    # 1. Live-season league index (one call).
    index_report = None
    live_seasons = [s for s in seasons if S.is_live_season(s)]
    if live_seasons and not args.no_index:
        s = live_seasons[0]
        try:
            idx = S.player_index(s, refresh=refresh)
            index_report = idx.attrs["report"]
            out = S.tidy_path("player_index", s)
            # live=False: no shrink guard (see the module docstring).
            S.write_tidy(idx, out, {"source": "stats.nba.com playerindex", "season": s,
                                    "report": index_report,
                                    "fetched_at": idx.attrs.get("fetched_at")}, live=False)
            print(f"playerindex {s}: {len(idx)} rostered players, {index_report['two_way']} "
                  f"two-way, {index_report['excluded_off_roster']} off-roster excluded", flush=True)
        except (http.Blocked, S.CallBudgetExceeded) as e:
            failures.append((f"playerindex {s}", str(e)))
            print(f"playerindex {s}: BLOCKED {e}", flush=True)
        except (ValueError, http.FetchError) as e:
            failures.append((f"playerindex {s}", str(e)))
            print(f"playerindex {s}: FAILED {e}", flush=True)

    # 2. Team rosters. Raw payloads are the checkpoint: a rerun skips them.
    blocked = False
    for i, (season, team) in enumerate(run_pairs, 1):
        key = f"{season}:{team}"
        c0, t0 = S.call_count(), time.time()
        try:
            r = S.team_roster(season, team, refresh=refresh)
            progress[key] = {"status": "done", "rows": len(r)}
            if S.call_count() > c0 or i % 30 == 0 or i == len(run_pairs):
                print(f"[{i}/{len(run_pairs)}] {season} {team}: {len(r)} players "
                      f"({S.call_count() - c0} calls, {time.time() - t0:.1f}s)", flush=True)
        except (http.Blocked, S.CallBudgetExceeded) as e:
            progress[key] = {"status": "blocked", "error": str(e)[:300]}
            failures.append((key, str(e)))
            print(f"[{i}/{len(run_pairs)}] {season} {team}: BLOCKED {e}; stopping", flush=True)
            blocked = True
        except (ValueError, http.FetchError) as e:
            progress[key] = {"status": "failed", "error": str(e)[:300]}
            failures.append((key, str(e)))
            print(f"[{i}/{len(run_pairs)}] {season} {team}: FAILED {e}", flush=True)
        if i % 10 == 0 or blocked:
            S.save_progress(SCRIPT, progress)
        if blocked:
            break
    S.save_progress(SCRIPT, progress)

    # 3. Assemble from every cached roster (this run's and earlier runs'),
    #    reading the cache only: no call is made here.
    per_season, lines = {}, []
    for season in seasons:
        frames = []
        for team in codes:
            if S.raw_path(season, f"roster_{team}").exists():
                try:
                    frames.append(S.team_roster(season, team, refresh=False))
                except (ValueError, http.FetchError, S.CacheMiss) as e:
                    # A bad cached payload is one failed team-season, reported
                    # below; it must not crash the run before the coverage report.
                    failures.append((f"{season}:{team}", str(e)))
        df = pd.concat(frames, ignore_index=True) if frames else None
        n_teams = len(frames)
        complete = n_teams == 30
        if df is not None:
            df.attrs["fetched_at"] = S.oldest_stamp(*(f.attrs.get("fetched_at") for f in frames))
            per_season[season] = df
            dest = S.tidy_path("team_rosters", season) if complete else \
                SAMPLE_DIR / f"team_rosters_{season}_partial_v1.parquet"
            S.write_tidy(df, dest, {"source": "stats.nba.com commonteamroster",
                                    "season": season, "teams": n_teams, "complete": complete,
                                    "report": [f.attrs.get("report") for f in frames],
                                    "fetched_at": S.oldest_stamp(*(f.attrs.get("fetched_at")
                                                                   for f in frames))},
                         live=False)
        lines.append({
            "season": season, "teams": f"{n_teams}/30", "players": 0 if df is None else len(df),
            "dup_person": 0 if df is None else int(df.person_id.duplicated().sum()),
            "two_way_known": 0 if df is None else int(df.two_way.notna().sum()),
            "two_way": 0 if df is None else int(df.two_way.fillna(False).sum()),
            "how_acq_known": 0 if df is None else int(df.how_acquired.notna().sum()),
            "how_acq_withheld": sum(f.attrs["report"]["how_acquired_withheld"] for f in frames),
            "birth_missing": 0 if df is None else int(df.birth_date.isna().sum()),
            "written": "canonical" if complete else ("partial->sample/" if df is not None else "-"),
        })

    bio_report, cov = None, []
    if per_season:
        rosters = pd.concat(per_season.values(), ignore_index=True)
        bio, bio_report = build_bio(rosters)
        all_present = all(r["teams"] == "30/30" for r in lines)
        dest = BIO_PATH if all_present and not partial else SAMPLE_DIR / "player_bio_partial_v1.parquet"
        S.write_tidy(bio, dest, {"source": "stats.nba.com commonteamroster", "seasons": seasons,
                                 "complete": all_present, "report": bio_report,
                                 "fetched_at": S.oldest_stamp(*(df.attrs.get("fetched_at")
                                                                for df in per_season.values()))},
                     live=False)
        cov = games_coverage(bio, seasons)
        print(f"\nbio: {bio_report['persons']} persons -> {dest.relative_to(CACHE_DIR.parent)}")

    print("\nCOVERAGE (team rosters)")
    print(pd.DataFrame(lines).to_string(index=False))
    if index_report:
        print(f"\nplayerindex: {index_report}")
    if bio_report:
        print(f"bio: persons {bio_report['persons']}, birth_date missing "
              f"{bio_report['birth_date_missing']}, birth_date conflicts "
              f"{bio_report['birth_date_conflicts']}")
        for c in bio_report["conflict_rows"][:10]:
            print(f"  conflict: {c}")
    if cov:
        print("player-game ids without a bio (end-of-season rosters miss mid-season departures):")
        print(pd.DataFrame(cov).to_string(index=False))
    print(f"\nlive calls this run: {S.call_count()}, elapsed {time.time() - t_all:.1f}s")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for k, e in failures[:40]:
            print(f"  {k}: {e[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
