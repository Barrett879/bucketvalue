"""Backfill historical closing game lines from ESPN's core odds API.

    python scripts/build_lines.py --seasons 2015-16:2025-26
    python scripts/build_lines.py --seasons 2019-20 --sample 20
    python scripts/build_lines.py --seasons 2023-24 --months 202401 --kaggle-check

Per season:
  1. ESPN events from the monthly scoreboard (cached raw/espn_scoreboard/).
  2. NBA games for the crosswalk: --nba-games TEMPLATE (a parquet per
     season), else cache/schedule_{season}_v1.parquet (stats.nba.com
     schedule: has the play-in, playoffs and the 006 Cup final), else the
     LeagueGameLog (T) for the season types the events need, cached under
     raw/espn_xwalk/ (no Cup final in those logs).
  3. event -> GAME_ID crosswalk; every unmatched FINAL event is listed.
  4. Core odds per FINAL NBA-vs-NBA event (raw/espn_odds/{season}/{id}.json.gz).
     Resumable: a cached event is never re-asked. A Blocked response stops
     the run (the cache is the checkpoint; rerun to continue). It is never
     read as "no odds".
  5. cache/game_lines_{season}_v1.parquet keyed by GAME_ID, plus a coverage
     report. --sample / --months write a PREVIEW under
     raw/espn_lines_preview/ instead, so a partial table can never be read
     as the season's table.
  6. COMPLETENESS GATE before a season table is written: no empty month,
     every NBA game the games table calls final claimed by a FINAL ESPN
     event (the whole table, not just the events' date span), no unkeyable
     FINAL event, and line coverage >= MIN_LINE_PCT. A failing season gets a
     preview carrying `gate_failures`, the season table is left alone, and
     the run exits 1. --allow-incomplete writes it anyway with
     complete=False.

--refresh refetches the scoreboard months and game logs (never the odds of
finished games). --retry-absent re-asks every cached 404 marker now (they
are otherwise re-asked once older than espn.ABSENT_TTL).

--kaggle-check compares the lines built in this run with the Kaggle CC0
archive, LOCAL ONLY: it refuses to run unless git ignores cache/local_only/,
and nothing from the Kaggle file is written anywhere else.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bucketlib import cache, http, teams, util  # noqa: E402
from bucketlib.sources import espn  # noqa: E402

TABLE_VERSION = "v1"
# The stats.nba.com schedule table (built by scripts/build_schedule.py).
SCHEDULE_TABLE = "schedule_{season}_v1.parquet"
SEASON_TYPES = {"regular-season": "Regular Season", "play-in-season": "PlayIn",
                "post-season": "Playoffs"}
TRAINING_SLUGS = set(SEASON_TYPES)
_GAMELOG_TTL = 6 * 3600

# et_date is the NBA schedule's ET date of the game. espn_et_date is ESPN's,
# which differs only on date_pm1 rows (corrupted Feb-2020 timestamps: ESPN
# says 2020-02-29 for 0021900886, played 2020-02-28), so any join on the
# date uses et_date.
LINE_COLUMNS = ["game_id", "event_id", "season", "et_date", "espn_et_date", "home", "away",
                "season_slug", "comp_type", "home_line", "total", "n_books", "n_books_total",
                "books", "open_home_line", "open_total", "has_open", "line_field", "qc_flags",
                "match", "source"]

# Completeness gate. Measured 2026-10-08: the crosswalk claimed 15,479 of
# 15,479 final games over 12 seasons against the schedule tables, and the
# worst season's line coverage was 2014-15 at 96.95% (aggregator-only
# events). So one missing game is a real hole, and coverage under 90% means
# something systematic (a 404 burst, a parser regression), not an era.
MAX_MISSING_GAMES = 0
MIN_LINE_PCT = 90.0

KAGGLE_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
              "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024")
# Pinned 2026-10-08 (dataset v4). Kaggle serves only the newest version and
# renames the file each season, so a mismatch is reported, not fatal: this
# is a local cross-check, not a training input.
KAGGLE_ZIP_SHA256 = "c144d6609bcddd687ba6bc4ddeee7e5d26738f09eb62d7cbf91b039f8611dc74"
KAGGLE_CSV_SHA256 = "729eb2ead85c14affbb81ae9a4c611fee9790b8dd3d1d7824e1ddba14410a8c4"
LOCAL_ONLY = REPO / "cache" / "local_only"


def parse_seasons(spec: str) -> list[str]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return util.seasons_between(a, b)
    util.season_end_year(spec)
    return [spec]


# ── NBA games for the crosswalk ──────────────────────────────────────────────
def _season_is_settled(season: str) -> bool:
    # The season's games are all played once its July (Finals end in June;
    # 2020-21 ended July 20) is over.
    return util.today_et() > pd.Timestamp(f"{util.season_end_year(season)}-08-01").date()


def _gamelog_settled(season: str, fetched_at: float | None) -> bool:
    # Judged by when the copy was FETCHED: a log saved mid-season lacks every
    # later game, and a test on today's date would reuse it forever once
    # August passed. Fetched on or after Aug 1 of the end year = complete.
    return espn.fetched_after(fetched_at, dt.date(util.season_end_year(season), 7, 31))


def load_gamelog(season: str, season_type: str, *, refresh: bool = False) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    path = cache.dc_path(f"raw/espn_xwalk/leaguegamelog_{season}_{season_type.replace(' ', '')}.json.gz")
    data, fetched_at = (None, None) if refresh else cache.read_json_stamped(path)
    if data is None or (not _gamelog_settled(season, fetched_at)
                        and not cache.is_fresh(fetched_at, _GAMELOG_TTL)):
        data = http.stats_call(leaguegamelog.LeagueGameLog, season=season,
                               season_type_all_star=season_type, player_or_team_abbreviation="T")
        cache.write_json(path, data, stamped=True, gz=True)
    return http.result_sets(data)["LeagueGameLog"]


# GAME_ID prefixes that can carry a closing line: regular season, playoffs,
# play-in, Cup final. Preseason (001) and All-Star (003) never do.
_LINE_GAME_PREFIXES = ("002", "004", "005", "006")


def games_from_table(g: pd.DataFrame, what: str) -> pd.DataFrame:
    """A games parquet in any of the shapes this repo produces -> the
    crosswalk's GAMES_COLUMNS: game-level (home/away, neutral or is_neutral,
    e.g. schedule_{season}_v1) or per-team rows (team/home bool, e.g. the
    team-games table). Preseason/All-Star ids and TBD rows (no teams yet)
    are left out and counted.

    Adds `final` (bool): the games the completeness gate expects ESPN to
    have finished. From the schedule's game_status (3 = final) when present,
    else every game dated before today (ET)."""
    g = g.rename(columns={"is_neutral": "neutral"}) if "neutral" not in g.columns else g
    status = None
    if "game_status" in g.columns:
        status = dict(zip(g["game_id"].astype(str).str.zfill(10),
                          pd.to_numeric(g["game_status"], errors="coerce")))
    if {"team", "home"} <= set(g.columns) and "away" not in g.columns:
        g = espn.nba_games_from_team_rows(g)
    g = g.copy()
    g["game_id"] = g["game_id"].astype(str).str.zfill(10)
    keep = g["game_id"].str[:3].isin(_LINE_GAME_PREFIXES)
    tbd = keep & (g["home"].isna() | g["away"].isna())
    if tbd.any():
        print(f"  {what}: {int(tbd.sum())} TBD games (teams not set yet) left out")
    g = g[keep & ~tbd]
    out = g[[c for c in espn.GAMES_COLUMNS if c in g.columns]].reset_index(drop=True)
    if status is not None:
        out["final"] = out["game_id"].map(status).eq(3)
    else:
        out["final"] = pd.to_datetime(out["game_date"]).dt.date < util.today_et()
    return out


def load_nba_games(season: str, slugs: set[str], template: str | None,
                   *, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """(games, source). Preference: --nba-games TEMPLATE, else the
    stats.nba.com schedule table (cache/schedule_{season}_v1.parquet, which
    carries the play-in, playoffs AND the 006 Cup final), else LeagueGameLog
    calls for the season types the events need (no Cup final there). Every
    frame carries `final` (see games_from_table; a game-log row is a played
    game, so always final)."""
    if template:
        p = Path(template.format(season=season))
        p = p if p.is_absolute() else REPO / p
        if not p.exists():
            raise SystemExit(f"--nba-games: {p} does not exist")
        return games_from_table(pd.read_parquet(p), str(p)), str(p)
    sched = cache.dc_path(SCHEDULE_TABLE.format(season=season))
    if sched.exists():
        return games_from_table(pd.read_parquet(sched), sched.name), sched.name
    frames = [espn.nba_games_from_gamelog(load_gamelog(season, SEASON_TYPES[s], refresh=refresh))
              for s in sorted(slugs) if s in SEASON_TYPES]
    if not frames:
        return pd.DataFrame(columns=espn.GAMES_COLUMNS + ["final"]), "none"
    return pd.concat(frames, ignore_index=True).assign(final=True), "LeagueGameLog(" + ",".join(
        SEASON_TYPES[s] for s in sorted(slugs) if s in SEASON_TYPES) + ")"


# ── Odds ─────────────────────────────────────────────────────────────────────
def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    pos = np.linspace(0, len(df) - 1, n).round().astype(int)
    return df.iloc[sorted(set(pos))]


def fetch_lines(targets: pd.DataFrame, season: str, workers: int, *,
                retry_absent: bool = False) -> tuple[pd.DataFrame, dict]:
    """game_odds for every target event, in parallel (pacing is per host in
    bucketlib.http, so workers only overlap latency).

    The FIRST failure of any kind cancels every queued request: Blocked
    stops with "rerun to resume"; anything else (BadPayload, BadRequest, a
    parser ValueError) is re-raised with the event id attached. Before
    this, only Blocked cancelled, and a BadPayload surfaced after all 300 of
    300 queued requests had gone to a host already serving garbage."""
    stats = {"cached": 0, "network": 0, "absent": 0, "net_secs": []}
    rows: dict[str, dict] = {}

    def one(r):
        was_cached = espn.odds_is_cached(r.event_id, season, retry_absent=retry_absent)
        row, secs = espn.timed(espn.game_odds, r.event_id, season, home=r.home, away=r.away,
                               retry_absent=retry_absent)
        return r.event_id, row, was_cached, secs

    todo = list(targets.itertuples())
    done = 0
    blocked: Exception | None = None
    failed: Exception | None = None
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, r): r.event_id for r in todo}
        for fut in cf.as_completed(futs):
            try:
                eid, row, was_cached, secs = fut.result()
            except cf.CancelledError:  # before Exception: it is one
                continue
            except Exception as e:  # noqa: BLE001 - re-raised below
                # Stop asking: queued requests to a host that just blocked us
                # or served the wrong content would only fail too.
                if isinstance(e, http.Blocked):
                    blocked = blocked or e
                elif failed is None:
                    e.add_note(f"while fetching odds for event {futs[fut]} ({season})")
                    failed = e
                for f in futs:
                    f.cancel()
                continue
            rows[eid] = row
            stats["cached" if was_cached else "network"] += 1
            if not was_cached:
                stats["net_secs"].append(secs)
            if "odds_absent" in row["qc_flags"]:
                stats["absent"] += 1
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  [{season}] odds {done}/{len(todo)} "
                      f"(network {stats['network']}, cached {stats['cached']})", flush=True)
    if blocked is not None:
        raise SystemExit(f"STOPPED: ESPN blocked us ({blocked}). {len(rows)} events are cached; "
                         "rerun the same command to resume. Nothing was treated as 'no odds'.")
    if failed is not None:
        print(f"  STOPPED [{season}] at the first error after {len(rows)} events; queued requests "
              f"were cancelled. Cached events are kept; rerun to resume once fixed.", flush=True)
        raise failed
    out = pd.DataFrame([rows[e] for e in targets["event_id"] if e in rows])
    return out, stats


# ── Per season ───────────────────────────────────────────────────────────────
def missing_games(nba: pd.DataFrame, xw: pd.DataFrame, months: list[str] | None) -> pd.DataFrame:
    """Games the games table calls final that no FINAL ESPN event claimed,
    over the WHOLE table (the crosswalk's unclaimed_games only covers the
    events' date span, so a missing first or last month never showed up
    there). `months` restricts to games dated in those YYYYMM months (a
    --months preview); None means every game."""
    exp = nba[nba["final"].astype(bool)] if "final" in nba.columns else nba
    if months is not None:
        exp = exp[pd.to_datetime(exp["game_date"]).dt.strftime("%Y%m").isin(months)]
    claimed = set(xw["game_id"].dropna())
    miss = exp[~exp["game_id"].astype(str).str.zfill(10).isin(claimed)]
    out = miss[["game_id", "game_date", "away", "home"]].reset_index(drop=True)
    out.attrs["expected"] = int(len(exp))
    return out


def completeness_gate(cov: dict) -> list[str]:
    """Reasons a full-season build must not be written as the season table
    (empty = pass). Each one is a hole a reader would otherwise take for a
    complete season (verifier 2026-10-08: 9 empty months of 10 still wrote
    6 rows as complete=True)."""
    fails = []
    if cov["empty_months"]:
        fails.append(f"empty months {cov['empty_months']}: ESPN returned no events")
    if cov["missing_games"] > MAX_MISSING_GAMES:
        fails.append(f"{cov['missing_games']} of {cov['expected_final_games']} final NBA games "
                     f"have no FINAL ESPN event, e.g. {cov['missing_examples'][:3]}")
    if cov["unmatched_blocking"]:
        fails.append(f"{cov['unmatched_blocking']} FINAL ESPN events match no NBA game "
                     "(other than an NBA Cup final under the game-log fallback)")
    if cov["lines_rows"] and cov["with_line_pct"] < MIN_LINE_PCT:
        fails.append(f"line coverage {cov['with_line_pct']}% < {MIN_LINE_PCT}% "
                     f"({cov['odds_absent']} odds 404s; --retry-absent re-asks them)")
    return fails


def build_season(season: str, *, sample: int | None, months: list[str] | None,
                 template: str | None, workers: int, write: bool, refresh: bool = False,
                 retry_absent: bool = False,
                 allow_incomplete: bool = False) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    season_mo = espn.season_months(season)
    use_months = [m for m in season_mo if m in months] if months else None
    if months and not use_months:
        print(f"== {season}: none of --months {months} belong to this season, skipped")
        return pd.DataFrame(columns=LINE_COLUMNS), {}
    ev = espn.events(season, months=use_months, refresh=refresh)
    drop = ev.attrs.get("dropped", {})
    final_mask = (ev["status"] == "STATUS_FINAL") & ev["season_slug"].isin(TRAINING_SLUGS)
    slugs = set(ev.loc[final_mask, "season_slug"])
    nba, nba_source = load_nba_games(season, slugs, template, refresh=refresh)
    xw = espn.crosswalk(ev, nba)
    targets = xw[xw["how"].isin(["exact", "date_pm1", "unmatched", "ambiguous"])].copy()
    if sample:
        targets = _sample(targets, sample)
    odds, ostats = fetch_lines(targets, season, workers, retry_absent=retry_absent)
    if len(odds):
        merged = targets.merge(odds.drop(columns=["home", "away"]), on="event_id", how="left")
    else:
        merged = targets.assign(**{c: np.nan for c in espn.ODDS_COLUMNS if c not in targets})
    merged = merged.rename(columns={"how": "match"})
    lines = merged[merged["game_id"].notna()].copy()
    if lines["nba_game_date"].isna().any():
        raise ValueError(f"{season}: matched rows without an NBA game date "
                         f"{lines.loc[lines['nba_game_date'].isna(), 'event_id'].tolist()[:5]}")
    # The NBA's date is the game's date; ESPN's is wrong on date_pm1 rows.
    lines["espn_et_date"] = pd.to_datetime(lines["et_date"]).dt.date
    lines["et_date"] = pd.to_datetime(lines["nba_game_date"]).dt.date
    for col in ("n_books", "n_books_total"):
        lines[col] = lines[col].fillna(0).astype(int)
    lines["has_open"] = lines["has_open"].fillna(False).astype(bool)
    lines = lines[LINE_COLUMNS].sort_values(["et_date", "game_id"]).reset_index(drop=True)
    if lines["game_id"].duplicated().any():
        raise ValueError(f"{season}: duplicate game_id in lines")

    full_run = not sample and not months
    miss = missing_games(nba, xw, None if full_run else ev.attrs.get("months", []))
    cov = coverage(season, ev, drop, xw, targets, merged, lines, ostats, full_run, miss)
    cov["nba_games_source"] = nba_source
    gate = completeness_gate(cov) if full_run else []
    # complete = every month fetched, no sampling, AND the gate passed.
    complete = full_run and not gate
    cov["complete"] = complete
    cov["gate_failures"] = gate
    cov["secs"] = round(time.perf_counter() - t0, 1)
    if write:
        # season_settled says whether more games can still arrive (the
        # current season); `through` is the last game date in the table.
        meta = {"source": "espn_core", "season": season, "complete": complete,
                "gate_failures": gate, "season_settled": _season_is_settled(season),
                "through": str(max(lines["et_date"])) if len(lines) else None,
                "sample": sample, "months": use_months, "coverage": cov,
                "built_at": util.now_utc().isoformat()}
        season_path = cache.dc_path(f"game_lines_{season}_{TABLE_VERSION}.parquet")
        if complete or (full_run and allow_incomplete):
            path = season_path
        else:
            path = cache.dc_path(f"raw/espn_lines_preview/game_lines_{season}.parquet")
        cache.write_parquet(lines, path, meta=meta)
        kind = ("" if complete else ", INCOMPLETE: gate failed, written by --allow-incomplete"
                if path == season_path else ", PREVIEW: not the season table")
        print(f"  wrote {path} ({len(lines)} rows{kind})")
    if gate:
        print(f"  GATE FAILED for {season}: " + " | ".join(gate))
        if not allow_incomplete:
            old = cache.dc_path(f"game_lines_{season}_{TABLE_VERSION}.parquet")
            if old.exists():
                m = cache.read_parquet_meta(old)
                print(f"  season table NOT written; the existing {old.name} (built_at "
                      f"{m.get('built_at')}, complete={m.get('complete')}) is left as it was")
            else:
                print("  season table NOT written")
    print_coverage(cov)
    return lines, cov


def coverage(season, ev, drop, xw, targets, merged, lines, ostats, complete, miss=None) -> dict:
    hows = xw["how"].value_counts().to_dict()
    unm = xw[xw["how"].isin(["unmatched", "ambiguous"])]
    # The Cup final (006) is the one FINAL event the game-log fallback cannot
    # key; anything else unmatched is a hole.
    blocking = unm[~((unm["how"] == "unmatched") & (unm["comp_type"] == "CC"))]
    unclaimed = xw.attrs.get("unclaimed_games", pd.DataFrame())
    month_counts = ev.attrs.get("month_counts", {})
    if miss is None:
        miss = pd.DataFrame(columns=["game_id", "game_date", "away", "home"])
    expected = int(miss.attrs.get("expected", 0))
    flag_counts = collections.Counter()
    for f in merged.get("qc_flags", pd.Series(dtype=object)).fillna(""):
        for part in filter(None, str(f).split(";")):
            kind = part.split(":")[0]
            flag_counts[kind if kind != "unknown_provider" else part] += 1
    nb = lines["n_books"].value_counts().sort_index().to_dict() if len(lines) else {}
    net = ostats.get("net_secs", [])
    n = len(lines)
    return {
        "season": season, "complete": complete,
        "events_nba": int(len(ev)),
        "dropped": {k: v for k, v in drop.items() if not k.endswith("_ids")},
        "final_score_missing_ids": list(drop.get("final_score_missing_ids", []))[:20],
        "month_counts": dict(month_counts),
        "empty_months": sorted(ym for ym, k in month_counts.items() if k == 0),
        "status_counts": ev["status"].value_counts().to_dict(),
        "final_nba_events": int(sum(hows.get(k, 0) for k in ("exact", "date_pm1", "unmatched", "ambiguous"))),
        "match_counts": hows,
        "unmatched": unm[["event_id", "et_date", "away", "home", "comp_type", "season_slug", "hint"]]
        .astype(str).values.tolist(),
        "unmatched_blocking": int(len(blocking)),
        "expected_final_games": expected,
        "missing_games": int(len(miss)),
        "missing_examples": miss.head(10).astype(str).values.tolist() if len(miss) else [],
        "unclaimed_nba_games": int(len(unclaimed)),
        "unclaimed_examples": unclaimed.head(10).astype(str).values.tolist() if len(unclaimed) else [],
        "targets": int(len(targets)),
        "lines_rows": n,
        "with_line_pct": round(100 * lines["home_line"].notna().mean(), 2) if n else None,
        "with_total_pct": round(100 * lines["total"].notna().mean(), 2) if n else None,
        "has_open_pct": round(100 * lines["has_open"].mean(), 2) if n else None,
        "n_books_dist": {int(k): int(v) for k, v in nb.items()},
        "line_field": lines["line_field"].value_counts(dropna=False).astype(int)
        .rename(index=str).to_dict() if n else {},
        "qc_flags": dict(flag_counts.most_common()),
        "odds_network_calls": ostats.get("network", 0), "odds_cached": ostats.get("cached", 0),
        "odds_absent": ostats.get("absent", 0),
        "odds_latency_mean_s": round(float(np.mean(net)), 3) if net else None,
        "odds_latency_p95_s": round(float(np.percentile(net, 95)), 3) if net else None,
    }


def _secs(v) -> str:
    return "n/a" if v is None else f"{v}s"


def print_coverage(c: dict) -> None:
    tag = "" if c["complete"] else (" (GATE FAILED)" if c.get("gate_failures")
                                    else " (PARTIAL: sample/months)")
    print(f"== {c['season']} coverage{tag}")
    print(f"  ESPN NBA-vs-NBA events {c['events_nba']}  dropped {c['dropped']}  "
          f"statuses {c['status_counts']}")
    print(f"  raw events per month {c['month_counts']}  empty months {c['empty_months']}")
    print(f"  FINAL training events {c['final_nba_events']}  match {c['match_counts']}  "
          f"(NBA games from {c.get('nba_games_source')})")
    for u in c["unmatched"]:
        print(f"    UNMATCHED {u}")
    print(f"  final NBA games with no FINAL ESPN event: {c['missing_games']} of "
          f"{c['expected_final_games']} {c['missing_examples'][:5]}  "
          f"(in the events' date span: {c['unclaimed_nba_games']})")
    print(f"  odds targets {c['targets']} (network {c['odds_network_calls']}, cached {c['odds_cached']}, "
          f"404 {c['odds_absent']}, latency mean {_secs(c['odds_latency_mean_s'])} "
          f"p95 {_secs(c['odds_latency_p95_s'])})")
    print(f"  lines rows {c['lines_rows']}  with line {c['with_line_pct']}%  with total "
          f"{c['with_total_pct']}%  has_open {c['has_open_pct']}%")
    print(f"  n_books {c['n_books_dist']}  line_field {c['line_field']}")
    print(f"  qc_flags {c['qc_flags']}")
    for g in c.get("gate_failures") or []:
        print(f"  GATE: {g}")


# ── Kaggle cross-check (LOCAL ONLY) ──────────────────────────────────────────
def ensure_local_only_ignored() -> None:
    """The Kaggle archive's provenance rules out committing it. Refuse unless
    git agrees cache/local_only/ is ignored."""
    r = subprocess.run(["git", "check-ignore", "-q", "cache/local_only/probe"], cwd=REPO)
    if r.returncode != 0:
        raise SystemExit("refusing --kaggle-check: `git check-ignore -q cache/local_only/probe` "
                         f"returned {r.returncode}; cache/local_only/ must be gitignored first")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def kaggle_frame() -> pd.DataFrame:
    ensure_local_only_ignored()
    zpath = LOCAL_ONLY / "kaggle" / "nba-betting-data.zip"
    if not zpath.exists():
        print(f"  downloading Kaggle archive -> {zpath.relative_to(REPO)}")
        # The research pass downloaded this unauthenticated (302 to a signed
        # storage URL) with plain curl; send a plain, honest User-Agent.
        try:
            blob = http.fetch(KAGGLE_URL, expect="bytes", timeout=120, retries=0,
                              headers={"User-Agent": "BucketValue-build_lines/1.0 (local cross-check)",
                                       "Accept": "*/*"})
        except (http.Absent, http.Blocked) as e:
            # 2026-10-08: the view API answered this client (v4, CC0) but the
            # download endpoint returned 404 twice. Kaggle may now want a
            # logged-in client. Never fall back to another mirror.
            raise SystemExit(
                f"Kaggle download failed ({e}). Download the dataset zip yourself (logged in) "
                f"from https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024 "
                f"and save it as {zpath}, then rerun with --kaggle-check.") from None
        if blob[:2] != b"PK":
            raise http.BadPayload(KAGGLE_URL, f"expected a zip, got {blob[:40]!r}")
        cache._atomic_write_bytes(zpath, blob)  # noqa: SLF001 - no public bytes writer
    blob = zpath.read_bytes()
    zsha = _sha256(blob)
    print(f"  kaggle zip sha256 {zsha} {'(pinned)' if zsha == KAGGLE_ZIP_SHA256 else 'WARNING: differs from the 2026-10-08 pin'}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"expected one CSV in the Kaggle zip, found {z.namelist()}")
        raw = z.read(csvs[0])  # read in memory: never extract member paths to disk
    csha = _sha256(raw)
    print(f"  kaggle csv {csvs[0]} sha256 {csha} "
          f"{'(pinned)' if csha == KAGGLE_CSV_SHA256 else 'WARNING: differs from the 2026-10-08 pin'}")
    return pd.read_csv(io.BytesIO(raw))


def kaggle_lines(k: pd.DataFrame) -> pd.DataFrame:
    """Kaggle conventions -> ours. season is the END year; spread is always
    positive with whos_favored naming the side; codes are lowercase
    ESPN-style; play-in and Cup finals are regular=False AND playoffs=False,
    so no row is filtered on those flags."""
    need = {"season", "date", "home", "away", "spread", "total", "whos_favored"}
    miss = need - set(k.columns)
    if miss:
        raise ValueError(f"Kaggle CSV is missing columns {sorted(miss)} (has {list(k.columns)})")
    k = k.copy()
    home = k["home"].map(teams.canonical)
    away = k["away"].map(teams.canonical)
    bad = sorted(set(k.loc[home.isna(), "home"]) | set(k.loc[away.isna(), "away"]))
    if bad:
        raise ValueError(f"unknown Kaggle team codes {bad[:10]}")
    fav = k["whos_favored"].astype(str).str.lower()
    odd = ~fav.isin(["home", "away"]) & k["spread"].fillna(0).ne(0)
    if odd.any():
        raise ValueError(f"whos_favored outside home/away on {int(odd.sum())} non-pick'em rows")
    if (k["spread"].dropna() < 0).any():
        raise ValueError("Kaggle spread is supposed to be non-negative")
    out = pd.DataFrame({
        "et_date": pd.to_datetime(k["date"]).dt.date,
        "home": home, "away": away,
        "season": k["season"].astype(int).map(util.season_str),
        "k_home_line": np.where(fav == "home", -k["spread"], k["spread"]).astype(float),
        "k_total": k["total"].astype(float),
    })
    if out.duplicated(["et_date", "home", "away"]).any():
        raise ValueError("duplicate (date, home, away) in Kaggle")
    return out


def agreement(lines: pd.DataFrame, kag: pd.DataFrame) -> pd.DataFrame:
    j = lines.merge(kag, on=["et_date", "home", "away"], how="inner", suffixes=("", "_k"))
    out = []
    for season, g in list(j.groupby("season")) + [("ALL", j)]:
        rec = {"season": season, "overlap": len(g)}
        for ours, theirs, nm in (("home_line", "k_home_line", "line"), ("total", "k_total", "total")):
            m = g[ours].notna() & g[theirs].notna()
            d = (g.loc[m, ours] - g.loc[m, theirs]).abs()
            rec[f"{nm}_n"] = int(m.sum())
            for lab, tol in (("exact", 0.0), ("le0.5", 0.5), ("le1.0", 1.0)):
                rec[f"{nm}_{lab}"] = round(100 * float((d <= tol + 1e-9).mean()), 1) if len(d) else None
        m = g["home_line"].notna() & g["k_home_line"].notna()
        flips = (np.sign(g.loc[m, "home_line"]) * np.sign(g.loc[m, "k_home_line"]) < 0).sum()
        rec["fav_flips"] = int(flips)
        out.append(rec)
    res = pd.DataFrame(out)
    worst = j.assign(d=(j["home_line"] - j["k_home_line"]).abs()).nlargest(5, "d")
    res.attrs["worst"] = worst[["et_date", "away", "home", "home_line", "k_home_line",
                                "total", "k_total", "n_books", "books"]]
    res.attrs["unjoined"] = int(len(lines) - len(j))
    return res


def kaggle_check(lines: pd.DataFrame) -> pd.DataFrame:
    print("== Kaggle cross-check (local only; nothing from it is written outside cache/local_only/)")
    kag = kaggle_lines(kaggle_frame())
    res = agreement(lines, kag)
    print(res.to_string(index=False))
    print(f"  ESPN lines with no Kaggle row: {res.attrs['unjoined']}")
    print("  largest spread disagreements:")
    print(res.attrs["worst"].to_string(index=False))
    return res


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", required=True, help="A:B or a single season, e.g. 2015-16:2025-26")
    ap.add_argument("--sample", type=int, default=None, help="only N events per season, evenly spaced")
    ap.add_argument("--months", default=None, help="dev only: comma list of YYYYMM to restrict to")
    ap.add_argument("--nba-games", default=None,
                    help="parquet path template with {season}: game_id, game_date, home, away")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kaggle-check", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch scoreboard months and game logs (never finished games' odds)")
    ap.add_argument("--retry-absent", action="store_true",
                    help="re-ask every cached odds 404 now, whatever its age")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the season table even when the completeness gate fails "
                         "(meta complete=False, gate_failures listed)")
    a = ap.parse_args(argv)
    months = [m.strip() for m in a.months.split(",")] if a.months else None
    all_lines, covs = [], []
    for season in parse_seasons(a.seasons):
        lines, cov = build_season(season, sample=a.sample, months=months, template=a.nba_games,
                                  workers=a.workers, write=not a.no_write, refresh=a.refresh,
                                  retry_absent=a.retry_absent,
                                  allow_incomplete=a.allow_incomplete)
        if len(lines):
            all_lines.append(lines)
        if cov:
            covs.append(cov)
    print("== summary")
    for c in covs:
        print(f"  {c['season']}: final {c['final_nba_events']}  lines {c['lines_rows']}  "
              f"line {c['with_line_pct']}%  open {c['has_open_pct']}%  "
              f"unmatched {len(c['unmatched'])}  missing {c['missing_games']}  "
              f"empty months {len(c['empty_months'])}  {c['secs']}s"
              f"{'  GATE FAILED' if c.get('gate_failures') else ''}")
    if a.kaggle_check:
        if not all_lines:
            print("kaggle check: no lines built in this run")
        else:
            kaggle_check(pd.concat(all_lines, ignore_index=True))
    failed = [c["season"] for c in covs if c.get("gate_failures")]
    if failed and not a.allow_incomplete:
        print(f"GATE FAILED for {failed}: their season tables were NOT written (see GATE lines above)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
