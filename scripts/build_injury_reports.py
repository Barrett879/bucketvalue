#!/usr/bin/env python
"""Backfill the NBA's official injury reports into per-season training tables.

For every game in a season's schedule, find the latest REAL report (one that
lists any NBA game on the schedule; QA and Summer League files share the
bucket) whose TITLE time is before tip minus 5 minutes
(bucketlib.sources.injury.select_reports_for_date; the filename time is never
used, it leaks post-tip news in the hourly era), keep that report's rows for
the game, and record per team whether the team was listed, not yet
submitted, or absent from the report. Every output row is asserted to use a
report titled before tip - lead before anything is written.

    python scripts/build_injury_reports.py --seasons 2018-19:2025-26
    python scripts/build_injury_reports.py --seasons 2024-25 --sample 5 --budget 40
    python scripts/build_injury_reports.py --seasons 2024-25 --schedule-from-logs

Inputs: cache/schedule_{season}_v1.parquet (game_id, game_date, away, home,
tip_utc). Until that exists, --schedule-from-logs derives games from
stats.nba.com LeagueGameLog (Regular Season + PlayIn + Playoffs: playoff
reports are real reports, and a regular-season-only schedule would count
every playoff row as a validation drop) and tips from the ESPN scoreboard by
month (--tips espn, the default), or puts every tip at 7:00 PM ET (--tips
default) with a warning: a wrong tip moves the report cutoff.

--lead-minutes N (default 5) selects an earlier snapshot instead; the
outputs then carry a _leadN tag. Measured on samples: at tip-15 min (2025-26)
the statuses are only Out/Available, at tip-120 min (2018-19) Questionable
and Probable still appear. A model served from a morning refresh must be
trained on a matching lead, or it never sees Questionable at all.

Outputs (complete runs only; --sample/--dates write to cache/raw/injury/
<season>/_sample_*.parquet and an --offline run that found slots missing from
disk to _offline_partial_*.parquet, so a partial table never sits under the
canonical name):
  cache/injury_reports_{season}_v1.parquet        one row per listed player
  cache/injury_report_teams_{season}_v1.parquet   one row per (game, team)
Raw PDFs go to cache/raw/injury/{season}/; known-absent slots are
checkpointed in _absent.json.gz, so a rerun makes no network calls for work
already done. Blocked (an Akamai refusal) always stops the run: it is not
"no report". With --keep-going a PDF that does not parse is reported once and
its games get state 'report_unparsed'; the walk never skips past it to an
older report. Exit codes: see main().
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from bucketlib import http, teams, util  # noqa: E402
from bucketlib.cache import CACHE_DIR, read_json, read_parquet, write_json, write_parquet  # noqa: E402
from bucketlib.sources import injury as I  # noqa: E402

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
# ESPN caps a month query at `limit` events WITHOUT an error (the default is
# 100); a month at the cap is refused rather than silently truncated.
ESPN_LIMIT = 500
DEFAULT_TIP_ET = dt.time(19, 0)
LOG_TYPES = ("Regular Season", "PlayIn", "Playoffs")

TEAM_STATES = ("listed", "none_listed", "not_yet_submitted", "all_players_available",
               "game_not_in_report", "report_unparsed", "no_report")


class ReportUnparsed(Exception):
    """The report a game should use exists but does not parse (--keep-going).
    Deliberately NOT a ValueError: it must stop the selection walk, because
    walking past it would hand the game an older report (hours staler) with
    nothing in the output saying so."""

    def __init__(self, filename: str, error: ValueError):
        super().__init__(f"{filename}: {error}")
        self.filename = filename
        self.error = error


# ── Schedule ─────────────────────────────────────────────────────────────────
def raw_dir(season: str) -> Path:
    return CACHE_DIR / "raw" / "injury" / season


def schedule_path(season: str) -> Path:
    return CACHE_DIR / f"schedule_{season}_v1.parquet"


def _league_game_log(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    path = raw_dir(season) / f"_leaguegamelog_{season_type.replace(' ', '')}.json.gz"
    payload = read_json(path)
    if payload is None:
        print(f"  [{season}] stats.nba.com LeagueGameLog T {season_type!r} (1 call)")
        payload = http.stats_call(leaguegamelog.LeagueGameLog, season=season,
                                  season_type_all_star=season_type,
                                  player_or_team_abbreviation="T")
        # A season still in progress must be re-pulled next time, so only a
        # finished season's log is cached.
        if util.season_end_year(season) < util.season_end_year(util.season_for_date(util.today_et())):
            write_json(path, payload, gz=True)
    return http.result_sets(payload)["LeagueGameLog"]


def games_from_logs(season: str) -> pd.DataFrame:
    """One row per game: game_id, game_date, away, home, season_type."""
    out = []
    for stype in LOG_TYPES:
        df = _league_game_log(season, stype)
        if df.empty:
            continue
        for gid, g in df.groupby("GAME_ID"):
            if len(g) != 2:
                raise ValueError(f"{season} {stype} game {gid}: {len(g)} team rows")
            home = [r for r in g.itertuples() if " vs. " in r.MATCHUP]
            away = [r for r in g.itertuples() if " @ " in r.MATCHUP]
            if len(home) != 1 or len(away) != 1:
                raise ValueError(f"{season} game {gid}: cannot tell home/away from {g.MATCHUP.tolist()}")
            h, a = teams.canonical(home[0].TEAM_ABBREVIATION), teams.canonical(away[0].TEAM_ABBREVIATION)
            if h is None or a is None:
                raise ValueError(f"{season} game {gid}: unknown team code")
            if home[0].GAME_DATE != away[0].GAME_DATE:
                raise ValueError(f"{season} game {gid}: two different dates")
            out.append(dict(game_id=str(gid), game_date=dt.date.fromisoformat(home[0].GAME_DATE[:10]),
                            away=a, home=h, season_type=stype))
    games = pd.DataFrame(out)
    if games.empty:
        raise ValueError(f"{season}: LeagueGameLog returned no games")
    if games.game_id.duplicated().any():
        raise ValueError(f"{season}: duplicate game ids across season types")
    return games


def _espn_month(ym: str) -> list[list]:
    payload = http.fetch(ESPN_SCOREBOARD, params={"dates": ym, "limit": ESPN_LIMIT}, expect="json")
    events = payload.get("events") or []
    if len(events) >= ESPN_LIMIT:
        raise ValueError(f"ESPN month {ym} returned {len(events)} events = the limit: truncated")
    out = []
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): teams.canonical((c.get("team") or {}).get("abbreviation"))
                 for c in comp.get("competitors", [])}
        if not sides.get("home") or not sides.get("away") or not ev.get("date"):
            continue  # All-Star, Rising Stars, international exhibition opponents
        tip = util.to_et(ev["date"])
        out.append([tip.date().isoformat(), sides["away"], sides["home"],
                    tip.astimezone(util.UTC).isoformat()])
    return out


def espn_tips(season: str, game_dates) -> dict[tuple, str]:
    """{(et_date, away, home): tip_utc_iso} from ESPN month scoreboards."""
    path = raw_dir(season) / "_espn_tips.json.gz"
    cache = read_json(path) or {}
    months = sorted({d.strftime("%Y%m") for d in game_dates})
    this_month = util.today_et().strftime("%Y%m")
    for ym in months:
        if ym in cache:
            continue
        print(f"  [{season}] ESPN scoreboard {ym} (1 call)")
        cache[ym] = _espn_month(ym)
        if ym < this_month:  # a month still in progress is re-pulled next run
            write_json(path, cache, gz=True)
    tips = {}
    for ym in months:
        for d, a, h, tip in cache[ym]:
            tips[(dt.date.fromisoformat(d), a, h)] = tip
    return tips


def load_schedule(season: str, from_logs: bool, tips_mode: str,
                  dates_wanted: Callable[[list], list] | None = None) -> tuple[pd.DataFrame, dict]:
    """The season's games with tips. With `dates_wanted` (a --sample/--dates
    run), ESPN is asked only for the months those dates fall in; other games
    get a placeholder tip that no selection uses."""
    notes: dict = {}
    p = schedule_path(season)
    sched = read_parquet(p)
    if sched is not None:
        notes["schedule_source"] = str(p.name)
    else:
        if not from_logs:
            raise SystemExit(f"{p} does not exist; pass --schedule-from-logs to derive one")
        sched = games_from_logs(season)
        notes["schedule_source"] = "LeagueGameLog"
        needed = set(sched.game_date)
        if dates_wanted is not None:
            in_window = sorted(d for d in needed if d >= I.FIRST_REPORT_DATE)
            needed = set(dates_wanted(in_window))
        tip_map = espn_tips(season, sorted(needed)) if tips_mode == "espn" else {}
        tips, n_default, n_swapped = [], 0, 0
        for d, a, h in zip(sched.game_date, sched.away, sched.home):
            t = tip_map.get((d, a, h))
            if t is None and (d, h, a) in tip_map:
                t, n_swapped = tip_map[(d, h, a)], n_swapped + 1
            if t is None:
                n_default += d in needed
                t = dt.datetime.combine(d, DEFAULT_TIP_ET, tzinfo=util.ET).astimezone(util.UTC).isoformat()
            tips.append(t)
        sched["tip_utc"] = pd.to_datetime(tips, utc=True)
        notes["tips_defaulted_19et"] = n_default
        notes["tips_espn_swapped_home_away"] = n_swapped
        if n_default:
            print(f"  WARNING [{season}] {n_default} of {len(needed)} needed tips unknown, set to "
                  f"7:00 PM ET: their report cutoff may be wrong")
    missing = [c for c in I.SCHEDULE_COLUMNS if c not in sched.columns]
    if missing:
        raise SystemExit(f"{season} schedule is missing columns {missing}")
    sched = sched.copy()
    sched["game_date"] = sched["game_date"].map(I._as_date)
    sched["tip_utc"] = pd.to_datetime(sched["tip_utc"], utc=True)
    if "is_tbd" in sched.columns and sched["is_tbd"].fillna(False).astype(bool).any():
        # An unannounced tip (if-necessary playoff games) carries a placeholder
        # time that can sit on the wrong ET date; 7 PM ET is the honest guess.
        tbd = sched["is_tbd"].fillna(False).astype(bool)
        sched.loc[tbd, "tip_utc"] = [pd.Timestamp(dt.datetime.combine(d, DEFAULT_TIP_ET, tzinfo=util.ET))
                                     .tz_convert("UTC") for d in sched.loc[tbd, "game_date"]]
        notes["tips_tbd_defaulted_19et"] = int(tbd.sum())
    bad = sched.tip_utc.map(lambda t: util.et_date_of(t.to_pydatetime())) != sched.game_date
    if bad.any():
        raise ValueError(f"{season}: {int(bad.sum())} tips are not on their ET game date, e.g. "
                         f"{sched.loc[bad, ['game_id', 'game_date', 'tip_utc']].head(3).values.tolist()}")
    return sched, notes


# ── Per-season build ─────────────────────────────────────────────────────────
def choose_dates(all_dates: list, args) -> list:
    dates = list(all_dates)
    if args.dates:
        want = {dt.date.fromisoformat(d) for d in args.dates.split(",")}
        dates = [d for d in dates if d in want]
    if args.sample:
        dates = sample_dates(dates, args.sample)
    return dates


def sample_dates(dates: list, n: int) -> list:
    if n >= len(dates):
        return list(dates)
    step = (len(dates) - 1) / max(n - 1, 1)
    return sorted({dates[round(i * step)] for i in range(n)})


# GAME_ID prefixes: 001 preseason, 002 regular season, 003 All-Star, 004
# playoffs, 005 play-in, 006 NBA Cup final. Preseason reports are sparse to
# absent (0/24 hourly files on 2025-10-03/10/17) and a date without reports
# costs a full 30-hour walk of 403 probes, so preseason is skipped unless
# asked for. All-Star games have no injury report.
MODELLED_PREFIXES = ("002", "004", "005", "006")


def modelled_mask(sched: pd.DataFrame, include_preseason: bool) -> pd.Series:
    prefixes = MODELLED_PREFIXES + (("001",) if include_preseason else ())
    return sched["game_id"].astype(str).str[:3].isin(prefixes)


def check_before_tip(rows: pd.DataFrame, tdf: pd.DataFrame, lead: dt.timedelta) -> None:
    """Spec §2 production assertion: every report used is titled strictly
    before tip - lead. Selection builds this in today; this check is what
    makes a regression there (a filename-time pick, a wall-clock cutoff)
    fail the run instead of leaking post-tip news into training."""
    for name, df in (("player rows", rows), ("team rows", tdf)):
        has = df["report_ts_et"].notna()
        if not has.any():
            continue
        ts = pd.to_datetime(df.loc[has, "report_ts_et"], utc=True)
        cut = pd.to_datetime(df.loc[has, "tip_utc"], utc=True) - lead
        bad = ~(ts < cut)
        if bad.any():
            ex = df.loc[has][bad.values][["game_id", "report_ts_et", "tip_utc"]].head(3).values.tolist()
            raise ValueError(f"{int(bad.sum())} {name} use a report not titled before tip - {lead}: {ex}")


def build_season(season: str, sched: pd.DataFrame, store: I.ReportStore, args) -> dict:
    t_season = time.monotonic()
    cov = Counter()
    s0, f0, p0 = Counter(store.stats), store.fetch_seconds, store.parse_seconds
    lead_min = int(getattr(args, "lead_minutes", 5))
    if lead_min < 0:
        raise ValueError(f"--lead-minutes {lead_min}: a negative lead selects post-tip reports")
    lead = dt.timedelta(minutes=lead_min)
    # Any lead other than the default 5 is a different snapshot and gets its
    # own filename, so it can never overwrite the canonical table.
    tag = "" if lead_min == 5 else f"_lead{lead_min}"
    sched = sched.assign(game_date=sched.game_date.map(I._as_date),
                         tip_utc=pd.to_datetime(sched.tip_utc, utc=True))
    sched = sched.sort_values(["game_date", "tip_utc", "game_id"]).reset_index(drop=True)
    pre = sched.game_date < I.FIRST_REPORT_DATE
    keep = modelled_mask(sched, getattr(args, "include_preseason", False))
    cov["games_pre_archive"] = int(pre.sum())
    cov["games_skipped_type"] = int((~keep & ~pre).sum())
    sched_in = sched[~pre & keep]   # games to select reports for; `sched` validates
    all_dates = sorted(sched_in.game_date.unique())
    dates = choose_dates(all_dates, args)
    partial = len(dates) != len(all_dates)

    player_rows, team_rows = [], []
    drops: dict[str, tuple[int, list]] = {}          # per distinct report
    excluded: dict[str, int] = {}                    # per distinct report: non-NBA rows
    vetoed: Counter = Counter()
    failures: dict[str, str] = {}                    # filename -> error, each file once

    def accept(rep: I.ParsedReport) -> bool:
        # REPORT-level: any real NBA game after schedule validation. A QA or
        # Summer League file is vetoed and the walk goes on; a real report
        # that does not list a game yet gives 'game_not_in_report' (see
        # select_reports_for_date for why a game-level test is wrong).
        ok = I.lists_nba_game(rep, sched)
        if not ok:
            vetoed[rep.meta["title"]] += 1
        return ok

    def getter(slot):
        try:
            return store.get(slot)
        except ValueError as e:   # a parse failure; Blocked / budget / BadPayload propagate
            if not args.keep_going:
                raise
            failures[slot.filename] = str(e)[:300]
            raise ReportUnparsed(slot.filename, e) from None

    def team_row(g, team, **kw):
        team_rows.append(dict(dict(game_id=g.game_id, game_date=g.game_date, tip_utc=g.tip_utc,
                                   team=team, report_file=None, report_ts_et=pd.NaT,
                                   n_rows=0, n_out=0), **kw))

    reports_per_date: list[int] = []
    lags: list[float] = []
    for di, day in enumerate(dates, 1):
        day_games = sched_in[sched_in.game_date == day]
        tip_groups = {util.to_et(t.to_pydatetime()): g for t, g in day_games.groupby("tip_utc")}
        chosen_files = set()
        for tip, games in tip_groups.items():
            try:
                sel = I.select_reports_for_date(day, [tip], getter=getter, lead=lead,
                                                accept=accept)[tip]
            except ReportUnparsed as e:
                cov["games_report_unparsed"] += len(games)
                for g in games.itertuples():
                    for team in (g.away, g.home):
                        team_row(g, team, report_file=e.filename, state="report_unparsed")
                continue
            gids = set(games.game_id)
            if sel is None:
                cov["games_no_report"] += len(games)
                for g in games.itertuples():
                    for team in (g.away, g.home):
                        team_row(g, team, state="no_report")
                continue
            rep = sel.report
            chosen_files.add(sel.slot.filename)
            lags.append((tip.astimezone(util.UTC) - rep.meta["report_ts_et"].astimezone(util.UTC))
                        .total_seconds() / 60)
            kept, dropped = I.validate_rows(rep.rows, sched)
            tkept, tdropped = I.validate_rows(rep.team_rows, sched)
            if sel.slot.filename not in drops:
                drops[sel.slot.filename] = (len(dropped) + len(tdropped),
                                            sorted(set(dropped.matchup) | set(tdropped.matchup)))
                excluded[sel.slot.filename] = (rep.meta.get("n_excluded_rows", 0)
                                               + rep.meta.get("n_excluded_team_rows", 0))
            kept = kept[kept.game_id.isin(gids)]
            tkept = tkept[tkept.game_id.isin(gids)]
            kept = kept.assign(report_file=sel.slot.filename)
            player_rows.append(kept)
            listed_games = set(kept.game_id) | set(tkept.game_id)
            n_by_team = kept.groupby(["game_id", "team"]).size().to_dict()
            out_by_team = kept[kept.status == "Out"].groupby(["game_id", "team"]).size().to_dict()
            kind_by_team = dict(zip(zip(tkept.game_id, tkept.team), tkept.kind))
            for g in games.itertuples():
                if g.game_id not in listed_games:
                    cov["games_not_in_report"] += 1
                for team in (g.away, g.home):
                    k = (g.game_id, team)
                    if k in kind_by_team:
                        state = kind_by_team[k]
                    elif n_by_team.get(k):
                        state = "listed"
                    elif g.game_id in listed_games:
                        state = "none_listed"
                    else:
                        state = "game_not_in_report"
                    team_row(g, team, report_file=sel.slot.filename,
                             report_ts_et=rep.meta["report_ts_et"], state=state,
                             n_rows=int(n_by_team.get(k, 0)), n_out=int(out_by_team.get(k, 0)))
        reports_per_date.append(len(chosen_files))
        if di % 10 == 0 or di == len(dates):
            s = store.stats
            print(f"  [{season}] {di}/{len(dates)} {day} tips={len(tip_groups)} "
                  f"reports={len(chosen_files)} live={s['live']} absent={s['absent_live']} "
                  f"cached={s['cache_hit'] + s['absent_cached']} unparsed={len(failures)}", flush=True)
        if di % 25 == 0:
            store.flush()
    store.flush()

    rows = (pd.concat(player_rows, ignore_index=True) if player_rows
            else pd.DataFrame(columns=I.ROW_COLUMNS + ["game_id", "tip_utc", "swapped",
                                                       "tip_mismatch", "report_file"]))
    keep = ["report_ts_et", "report_file", "game_id", "game_date", "tip_utc", "team",
            "player_raw", "status", "reason_cat", "reason_raw", "prev_status",
            "game_time_raw", "swapped", "tip_mismatch"]
    rows = rows.loc[:, keep].reset_index(drop=True)
    rows["report_ts_et"] = pd.to_datetime(rows["report_ts_et"]).dt.tz_convert(util.ET) \
        if len(rows) else pd.Series(dtype="datetime64[ns, America/New_York]")
    rows["tip_utc"] = pd.to_datetime(rows["tip_utc"], utc=True)
    rows["game_id"] = rows["game_id"].astype(str)
    if rows.duplicated(["game_id", "team", "player_raw"]).any():
        raise ValueError(f"{season}: duplicate (game_id, team, player_raw) in the output")
    tdf = pd.DataFrame(team_rows, columns=["game_id", "game_date", "tip_utc", "team", "report_file",
                                           "report_ts_et", "state", "n_rows", "n_out"])
    tdf["tip_utc"] = pd.to_datetime(tdf["tip_utc"], utc=True)
    tdf["report_ts_et"] = pd.to_datetime(tdf["report_ts_et"], utc=True).dt.tz_convert(util.ET)
    tdf["game_id"] = tdf["game_id"].astype(str)
    if tdf.duplicated(["game_id", "team"]).any():
        raise ValueError(f"{season}: duplicate (game_id, team) in the team table")
    check_before_tip(rows, tdf, lead)

    # ── Coverage ──
    other = Counter(I.reason_lead(r) for r, c in zip(rows.reason_raw, rows.reason_cat) if c == "other")
    n_drop_rows = sum(n for n, _ in drops.values())
    drop_matchups = Counter(m for _, ms in drops.values() for m in ms)
    s = Counter(store.stats)
    s.subtract(s0)
    n_live = max(s["live"], 1)
    n_parse = max(s["parsed"] + s["parse_failed"], 1)
    offline_miss = int(s["offline_miss"])
    if offline_miss:
        # Offline, a slot missing from disk is UNKNOWN (never fetched, or
        # absent but not yet checkpointed), and the walk then picks an older
        # report or none. Such a table must never sit under the canonical name.
        partial = True
        print(f"  WARNING [{season}] {offline_miss} slots not on disk in --offline mode: "
              f"output is PARTIAL; rerun online to complete it", flush=True)
    coverage = dict(
        season=season, partial=partial,
        partial_reason=("offline misses" if offline_miss else "sample/dates") if partial else None,
        dates_total=len(all_dates), dates_processed=len(dates),
        games_in_window=int(sched_in.game_date.isin(dates).sum()),
        games_pre_archive=cov["games_pre_archive"], games_skipped_type=cov["games_skipped_type"],
        games_no_report=cov["games_no_report"], games_not_in_report=cov["games_not_in_report"],
        games_report_unparsed=cov["games_report_unparsed"],
        reports_selected=len({f for f in tdf.loc[tdf.state != "report_unparsed", "report_file"].dropna()}),
        reports_per_date=dict(mean=round(statistics.mean(reports_per_date), 2) if reports_per_date else 0,
                              min=min(reports_per_date, default=0), max=max(reports_per_date, default=0)),
        tip_minus_report_minutes=dict(median=round(statistics.median(lags), 1) if lags else None,
                                      max=round(max(lags), 1) if lags else None),
        rows=len(rows), status_counts=rows.status.value_counts().to_dict(),
        reason_cat_counts=rows.reason_cat.value_counts().to_dict(),
        other_reason_leads=dict(other.most_common(15)),
        team_states=tdf.state.value_counts().to_dict(),
        not_yet_submitted=int((tdf.state == "not_yet_submitted").sum()),
        schedule_drops=dict(rows=n_drop_rows, reports_with_drops=sum(1 for n, _ in drops.values() if n),
                            matchups=dict(drop_matchups.most_common(12))),
        non_nba_rows_excluded=dict(rows=sum(excluded.values()),
                                   reports=sum(1 for n in excluded.values() if n)),
        reports_vetoed_not_nba=dict(vetoed.most_common(10)),
        swapped_home_away_rows=int(rows.swapped.sum()) if len(rows) else 0,
        tip_mismatch_rows=int(rows.tip_mismatch.sum()) if len(rows) else 0,
        parse_failures=failures,
        fetch=dict(live=s["live"], fetched=s["fetched"], absent_live=s["absent_live"],
                   absent_cached=s["absent_cached"], cache_hit=s["cache_hit"],
                   cache_corrupt=s["cache_corrupt"], offline_miss=offline_miss,
                   parsed=s["parsed"], parse_failed=s["parse_failed"],
                   mean_fetch_ms=round(1000 * (store.fetch_seconds - f0) / n_live, 1),
                   mean_parse_ms=round(1000 * (store.parse_seconds - p0) / n_parse, 1)),
        seconds=round(time.monotonic() - t_season, 1),
    )
    meta = dict(source="ak-static.cms.nba.com/referee/injury", coverage=coverage,
                lead_minutes=lead_min,
                selection=f"latest report with title time < tip - {lead_min} min that lists "
                          f"any real NBA game (report-level)")
    if partial:
        pfx = "_sample" if not offline_miss else "_offline_partial"
        out_rows = raw_dir(season) / f"{pfx}_injury_reports{tag}.parquet"
        out_teams = raw_dir(season) / f"{pfx}_injury_report_teams{tag}.parquet"
    else:
        out_rows = CACHE_DIR / f"injury_reports{tag}_{season}_v1.parquet"
        out_teams = CACHE_DIR / f"injury_report_teams{tag}_{season}_v1.parquet"
    write_parquet(rows, out_rows, meta=meta)
    write_parquet(tdf, out_teams, meta=meta)
    coverage["outputs"] = [str(out_rows), str(out_teams)]
    return coverage


def print_coverage(cov: dict) -> None:
    tag = f" (PARTIAL: {cov.get('partial_reason')})" if cov["partial"] else ""
    print(f"\n== coverage {cov['season']}{tag} ==")
    for k in ("partial_reason", "dates_total", "dates_processed", "games_in_window",
              "games_pre_archive", "games_skipped_type", "games_no_report",
              "games_not_in_report", "games_report_unparsed", "reports_selected",
              "reports_per_date", "tip_minus_report_minutes", "rows", "status_counts",
              "reason_cat_counts", "other_reason_leads", "team_states", "not_yet_submitted",
              "schedule_drops", "non_nba_rows_excluded", "reports_vetoed_not_nba",
              "swapped_home_away_rows", "tip_mismatch_rows", "parse_failures", "fetch",
              "seconds", "outputs"):
        print(f"  {k}: {cov.get(k)}")


def parse_seasons(spec: str) -> list[str]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return util.seasons_between(a, b)
    out = [s.strip() for s in spec.split(",") if s.strip()]
    for s in out:
        util.season_end_year(s)
    return out


def _lead_minutes(v: str) -> int:
    n = int(v)
    if n < 0:
        # A negative lead is a cutoff AFTER tip: post-tip news in training.
        raise argparse.ArgumentTypeError(f"{n}: the lead must be >= 0 minutes")
    return n


def main(argv=None) -> int:
    """Exit codes: 0 ok; 2 PDFs failed to parse (--keep-going; their games are
    'report_unparsed'); 3 Blocked (nothing written for the season in progress);
    4 --budget exhausted (resumable); 5 --offline run with slots missing from
    disk (outputs written under _offline_partial_*, never the canonical names)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", required=True, help="2018-19:2025-26, or a comma list")
    ap.add_argument("--sample", type=int, default=0, help="only N evenly spaced game dates per season")
    ap.add_argument("--dates", default="", help="comma list of ET game dates (YYYY-MM-DD)")
    ap.add_argument("--schedule-from-logs", action="store_true",
                    help="derive the schedule from LeagueGameLog when cache/schedule_* is missing")
    ap.add_argument("--tips", choices=("espn", "default"), default="espn",
                    help="tip times for --schedule-from-logs: ESPN scoreboard or 7:00 PM ET")
    ap.add_argument("--budget", type=int, default=None, help="max live PDF fetches (all seasons)")
    ap.add_argument("--offline", action="store_true", help="cache only; absent from disk = absent")
    ap.add_argument("--lead-minutes", type=_lead_minutes, default=5,
                    help="report must be titled before tip minus this (default 5). The last "
                         "pre-tip report has mostly resolved Q/P/D into Out/Available; to train "
                         "on what a scheduled refresh sees, match its lead (e.g. 180)")
    ap.add_argument("--include-preseason", action="store_true",
                    help="also select reports for preseason (001) games")
    ap.add_argument("--keep-going", action="store_true",
                    help="record PDFs that fail to parse and continue: their games get state "
                         "'report_unparsed', never an older report (exit code 2)")
    args = ap.parse_args(argv)

    store = I.ReportStore(budget=args.budget, offline=args.offline)
    seasons = parse_seasons(args.seasons)
    covs = []
    try:
        for season in seasons:
            if util.season_end_year(season) < util.season_end_year("2018-19"):
                print(f"[{season}] skipped: the injury-report archive starts 2018-12-17")
                continue
            print(f"[{season}] loading schedule", flush=True)
            partial = bool(args.sample or args.dates)
            sched, notes = load_schedule(season, args.schedule_from_logs, args.tips,
                                         (lambda ds: choose_dates(ds, args)) if partial else None)
            print(f"[{season}] {len(sched)} games, {sched.game_date.nunique()} dates ({notes})", flush=True)
            cov = build_season(season, sched, store, args)
            cov["schedule"] = notes
            print_coverage(cov)
            covs.append(cov)
    except http.Blocked as e:
        store.flush()
        print(f"\nBLOCKED, stopping (this is NOT 'no report'): {e}", file=sys.stderr)
        return 3
    except RuntimeError as e:  # --budget exhausted: resumable, nothing lost
        print(f"\nSTOPPED: {e}. Rerun to resume; fetched PDFs are cached.", file=sys.stderr)
        return 4
    finally:
        store.flush()
    failed = sum(len(c["parse_failures"]) for c in covs)
    missing = sum(c["fetch"]["offline_miss"] for c in covs)
    print(f"\ndone: {len(covs)} season(s), live fetches {store.stats['live']}, "
          f"parse failures {failed}, offline misses {missing}")
    if missing:
        return 5
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
