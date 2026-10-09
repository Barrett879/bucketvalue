"""The projection pipeline: committed history features plus the volatile
context (injury report, lines, mark-outs) in, the projection frame out.

Streamlit-free: the bot (scripts/daily_update.py) and the app both import it,
so a number on the site and a number in the bot's committed file come from ONE
function (docs/serving_design.md sections 2-4 and the section 6 amendments).

THE PATH. project() runs, in order: features.apply_lines (the lines block of
the training builder) -> context.context_for (the training context function
on one live report) -> models.check_nulls (the stratified null check; a raise
is a train/serve skew, never an exemption) -> models.predict -> the section 3
columns. Nothing here computes a feature.

THE FREEZE. A game is frozen from tip minus 5 minutes (horizons.TIP_FLOOR:
from then on the live report is titled after the context cutoff). Its rows are
never computed again by anyone: the bot carries the committed rows over byte
for byte (merge_published) and never writes a game for the first time after
that point; the app serves the committed rows untouched (live_projections),
mark-outs included, which never reach a frozen game. A PROVISIONAL row
(tomorrow's, or built on state missing a final) is never the frozen
projection (spec 4: "no freeze" on stale state): past the freeze point it is
still served and carried untouched, but with frozen False, and the local
record must skip it.

MISSING LINES (measured 2026-10-09, scripts/measure_missing_lines.py). Every
2018-19..2025-26 training row has a line but 36, so a missing line is a value
the models never learned. Fitting the gated configuration on 2018-19..2023-24
and predicting the 2024-25 regular season with the five line columns blanked:
  aggregate   minutes RMSE +0.99%, PTS +0.16%, PRA +0.24%, Fantasy +0.20%,
              P(play) Brier^0.5 +0.82%; mean shifts at most 0.45% of the mean.
              By horizon the minutes cost is +1.75% the evening before
              (d1_1730) and +1.51% in the morning (d0_1100).
  by spread   the aggregate nets favourites against underdogs: minutes bias
              goes from +0.6% to +5.9% for 10+ point favourites (PTS -1.2% to
              +2.6%) and from +1.2% to -2.1% for 10+ point underdogs (PTS +1.5%
              to -2.0%); per player the minutes move 0.57 on average, 1.95 at
              the 95th percentile, 6.7 at most.
The aggregate passes the bar set for projecting without lines (under 2% RMSE,
under 1% bias) but the spread strata fail it by five times, exactly where a
prop line sits (a star on a big favourite), and models.check_nulls has no way
to accept a missing column on purpose (an `allow` argument would be needed in
models.py, which this module does not extend). So the policy is WITHHOLD: a
game without both a spread and a total gets no projection rows
(NO_LINE: "Projections post when the line does"); project() reports it in
frame.attrs["withheld"], live_projections in Info.withheld. A game whose line
goes away after the bot projected it keeps its committed rows (stale beats
empty).

NO REPORT. report=None (nothing fetched and no committed fallback) makes every
team no_report: statuses unknown and report_age_h null on every row, a slate
training never had. project() refuses it itself (NoReport, a NullSkew) rather
than leaving it to models.check_nulls, which skips strata under
models.NULL_MIN_ROWS rows and so let a small slate (opening night, one game)
through. The app then serves the committed projections; the bot writes nothing
new for that date and says so.

COMMIT CHURN (the bot's file, merge_published). Two model features move with
the clock alone (hours_to_tip, report_age_h), so recomputing an untipped game
every bot run changes its numbers every run (measured on the 2026-03-12 slate:
every game, at +15 minutes and later, up to 0.5 minutes of mu_min) and the bot
would commit both files on every cron. So a game's committed rows are
recomputed only when an INPUT changed: a column in INPUT_COLS (the row set,
statuses, report time, lines, provisional, model version) or its history
feature rows (their content hash, kept per game in the file's metadata under
BASIS), or when the clock crossed one of the game's training horizons
(horizons.HORIZONS: d1_1730, d0_1100, t_120, t_15, each floored at tip minus
5 minutes) since the rows were computed, so the published numbers are at most
one horizon behind the clock and are refreshed at the horizons training
measured. The app recomputes untipped games live in any case.

STATE STAMP. The committed features file carries, in its metadata under
HISTORY, which ingested games its history held (history_stamp). The app
compares it with the state now: a final ingested since (the bot rebuilt the
finals but not this date's features) makes the live frame provisional, with a
note, instead of silently serving features that miss a game.

THE FRAME (section 3, plus `adjusted`): identity, availability (p_play,
inj_status, inj_reason, team_state: what context.display_status reads), the
means E[stat | plays] (min = mu_min), the simulator inputs, games_prior
(= career_games), provisional (never missing), model_version, and provenance
(report_ts_et, home_line, total, computed_at, frozen). `adjusted` says which
rows a visitor's mark-out changed (every row of a team with a marked-out
player); the bot's file has it False everywhere. `frozen` is False in the
bot's file (a frozen row is never rewritten, so it cannot be flagged there);
the app's frame sets it True on rows served from the committed file because
their game is past its freeze point and they are not provisional, and
touches nothing else.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import cache as C
from . import context as CTX
from . import features as F
from . import horizons as HZ
from . import http as H
from . import live as LV
from . import models as M
from . import serve as V
from . import state as S
from .util import ET, UTC, season_for_date, to_et

FEATURES_FILE = "features_v1.parquet"
PROJECTIONS_FILE = "projections_v1.parquet"
FREEZE_LEAD = HZ.TIP_FLOOR
LINE_COLS = ("home_line", "total", "team_line", "implied_total", "abs_line")
NO_LINE = "Projections post when the line does"
NOT_PUBLISHED = "not published yet"
MARKS_NOT_APPLIED = "Mark-outs need the live layer and are not applied."
STALE_FEATURES = ("The history features were built before the latest finals were ingested: "
                  "provisional until the bot rebuilds them.")
# Metadata keys: the features file's state stamp, the projections file's
# per-game feature hashes.
HISTORY = "history"
BASIS = "basis"
# The fewest rows any team had in a training team-game (19,112 of them,
# 2018-19..2025-26). A side with fewer is a broken roster pull for that team,
# never a slate to publish quietly.
MIN_TEAM_ROWS = 12

# The projection frame (docs/serving_design.md section 3), one dtype per column.
IDENTITY = {"person_id": "int64", "game_id": "object", "game_date": "datetime64[ns]",
            "tip_utc": "datetime64[ns, UTC]", "team": "object", "opp": "object", "home": "bool",
            "player_name": "object", "two_way": "bool", "season": "object"}
AVAILABILITY = {"p_play": "float64", "inj_status": "object", "inj_reason": "object",
                "team_state": "object"}
MEANS = {c: "float64" for c in ["min", *M.STATS, *M.COMPOSED]}
SIM_INPUTS = {c: "float64" for c in ["mu_min", *(f"rate_{s}" for s in M.STATS)]}
PROPS = {"games_prior": "float64", "provisional": "bool", "model_version": "object"}
PROVENANCE = {"report_ts_et": f"datetime64[ns, {ET.key}]", "home_line": "float64",
              "total": "float64", "computed_at": "datetime64[ns, UTC]", "frozen": "bool",
              "adjusted": "bool"}
DTYPES = {**IDENTITY, **AVAILABILITY, **MEANS, **SIM_INPUTS, **PROPS, **PROVENANCE}
COLUMNS = list(DTYPES)
ORDER = ["tip_utc", "game_id", "team", "person_id"]
# Stamps that alone never make a published row "changed": a projection whose
# numbers, statuses and lines are what the file already holds keeps its rows
# (and their stamps), so an unchanged slate is not a commit every bot run.
STAMPS = ("computed_at", "report_ts_et")
# The inputs of a published row that the frame itself shows (the rest is the
# feature rows, BASIS). Equal inputs and equal features: only the clock moved.
INPUT_COLS = [*IDENTITY, "inj_status", "inj_reason", "team_state", "provisional",
              "model_version", "report_ts_et", "home_line", "total", "adjusted"]

# What the app's live layer falls back on: every way a fetch, a committed file,
# the models or the null check can fail (NullSkew is a ValueError,
# MissingFeatures a KeyError, ModelMismatch a RuntimeError). A TypeError or
# AttributeError is a bug and propagates.
LIVE_FAILURES = (RuntimeError, ValueError, KeyError, OSError, H.FetchError)


class NoReport(M.NullSkew):
    """No injury report at all: every status unknown and report_age_h null on
    every row, which training never had. Refused whatever the slate size."""


# ── Paths and slate helpers ──────────────────────────────────────────────────
def features_path(date) -> Path:
    return S.serve_dir(date) / FEATURES_FILE


def projections_path(date) -> Path:
    return S.serve_dir(date) / PROJECTIONS_FILE


def _utc(now) -> dt.datetime:
    """An aware clock in UTC (util.to_et refuses a naive one)."""
    return to_et(now).astimezone(UTC)


def state_games(schedule: pd.DataFrame, date) -> pd.DataFrame:
    """The date's regular-season and play-in games with both teams known, as
    schedule rows (game_date stays a python date, which the report validation
    compares against), in tip order. Which games are state games is
    serve._state_games, the one definition."""
    d = pd.Timestamp(S.as_date(date))
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=["game_id", "game_date", "tip_utc", "away", "home"])
    ids = set(V._state_games(schedule)["game_id"])
    g = schedule[schedule["game_id"].isin(ids) & (pd.to_datetime(schedule["game_date"]) == d)]
    return g.sort_values(["tip_utc", "game_id"], kind="mergesort").reset_index(drop=True)


def frozen_ids(games: pd.DataFrame, now) -> set[str]:
    """Games past their freeze point (tip minus FREEZE_LEAD) at `now`. Any
    frame with game_id and tip_utc will do (schedule rows, or committed
    projection rows, which carry their own tip)."""
    if games is None or games.empty:
        return set()
    tips = pd.to_datetime(games["tip_utc"], utc=True)
    return set(games.loc[(tips - FREEZE_LEAD) <= pd.Timestamp(_utc(now)), "game_id"])


def provisional_for(date, today: dt.date, schedule: pd.DataFrame, season: str
                    ) -> tuple[bool, list[str]]:
    """(provisional, completeness gaps). Provisional: a date after `today`
    (ET), or one whose state misses a final game of a team playing that day
    (serve.completeness), so it is never freezable."""
    gaps = V.completeness(date, schedule, season)
    return S.as_date(date) > today or bool(gaps), gaps


def short_games(games: pd.DataFrame, rows: pd.DataFrame, min_rows: int = MIN_TEAM_ROWS
                ) -> dict[str, str]:
    """{game_id: "AWY n / HOM n"} for every game of `games` where a side has
    fewer than `min_rows` rows in `rows` (none at all included): the game
    cannot be projected whole, and only the missing-line policy may leave a
    game unprojected quietly."""
    n = rows.groupby(["game_id", "team"]).size() if len(rows) else pd.Series(dtype="int64")
    out = {}
    for gid, away, home in zip(games["game_id"], games["away"], games["home"]):
        a, h = int(n.get((gid, away), 0)), int(n.get((gid, home), 0))
        if min(a, h) < min_rows:
            out[gid] = f"{away} {a} / {home} {h}"
    return out


def lines_from_odds(odds, games: pd.DataFrame) -> pd.DataFrame:
    """(game_id, home_line, total) for `games`, from either odds shape:
    live.odds_on's {(away, home): {...}} or a frame of snapshots (the ESPN
    scoreboard rows the bot fetched or committed: away, home, home_line,
    total, captured_at; the latest capture per pairing wins). A game with no
    match has no row: no line yet."""
    cols = ["game_id", "home_line", "total"]
    empty = pd.DataFrame({"game_id": pd.Series(dtype=object),
                          "home_line": pd.Series(dtype="float64"), "total": pd.Series(dtype="float64")})
    if odds is None or games is None or games.empty or len(odds) == 0:
        return empty
    if isinstance(odds, dict):
        df = pd.DataFrame([{"away": a, "home": h, "home_line": (v or {}).get("home_line"),
                            "total": (v or {}).get("total")} for (a, h), v in odds.items()])
    else:
        df = odds.sort_values("captured_at", kind="mergesort") if "captured_at" in odds else odds
        df = df.drop_duplicates(["away", "home"], keep="last")
    m = games[["game_id", "away", "home"]].merge(df[["away", "home", "home_line", "total"]],
                                                 on=["away", "home"], how="inner",
                                                 validate="one_to_one")
    return m[cols].astype({"home_line": "float64", "total": "float64"}).reset_index(drop=True)


def known_names(season: str, before) -> pd.DataFrame:
    """(person_id, player_name): every spelling a person id has carried in the
    history of `before` (ET date), for the APP, which cannot afford the
    seed's events: the seed's names file plus the names state.history takes
    from the season's partitions (state._partition_history, the one join and
    cut), united exactly as state.history unites them. The bot uses
    History.names itself. Injury reports may use any spelling (Kanter /
    Freedom)."""
    seed = C.read_parquet(S.SEED_DIR / S.SEED_FILES["names"])
    if seed is None:
        raise FileNotFoundError(f"no seed names at {S.SEED_DIR}: run scripts/build_serving_seed.py")
    _, _, names = S._partition_history(S.season_partitions(season), season,
                                       pd.Timestamp(S.as_date(before)))
    if len(names):
        return pd.concat([seed, names], ignore_index=True).drop_duplicates(ignore_index=True)
    return seed


def ingested_games(season: str) -> pd.DataFrame:
    """(game_id, et_date) of every ingested game of `season`, by state's one
    definition of ingested (state.ingested_ids: overrides count, a game
    missing a partition kind does not)."""
    games = S.season_partitions(season)["games"]
    if games.empty:
        return pd.DataFrame({"game_id": pd.Series(dtype=object),
                             "et_date": pd.Series(dtype="datetime64[ns]")})
    g = games[games["game_id"].isin(S.ingested_ids(season))]
    return pd.DataFrame({"game_id": g["game_id"].to_numpy(),
                         "et_date": pd.to_datetime(g["et_date"]).to_numpy()}
                        ).drop_duplicates("game_id", ignore_index=True)


def data_through(season: str, ingested: pd.DataFrame | None = None) -> dt.date | None:
    """The ET date of the latest ingested final game of `season`; None before
    the first one. Pass `ingested` (ingested_games) to skip the read."""
    g = ingested_games(season) if ingested is None else ingested
    return None if g.empty else pd.Timestamp(g["et_date"].max()).date()


def history_stamp(season: str, date, ingested: pd.DataFrame | None = None) -> dict:
    """Which ingested games the history of `date` holds (state.history takes
    every partition game strictly before the date): through (ET date), games
    (count), digest (of the sorted ids). The bot stamps the features file
    with it; the app compares it with the state now."""
    g = ingested_games(season) if ingested is None else ingested
    g = g[g["et_date"] < pd.Timestamp(S.as_date(date))]
    ids = sorted(map(str, g["game_id"]))
    return {"through": pd.Timestamp(g["et_date"].max()).date().isoformat() if len(g) else None,
            "games": len(ids), "digest": C.json_hash(ids)}


def write_features(feats: pd.DataFrame, date, stamp: dict) -> bool:
    """The bot's features file for `date`, stamped with `stamp`
    (history_stamp at build time). An unchanged file stays byte-identical; a
    new stamp is written even over unchanged rows, so the stamp always says
    what the rows were built from. Returns True when it wrote."""
    p = features_path(date)
    same = C.read_parquet_meta(p).get(HISTORY) == stamp
    return C.write_parquet(feats, p, meta={HISTORY: stamp}, if_changed=same)


def feature_digests(feature_rows: pd.DataFrame) -> dict[str, str]:
    """Per game, the content hash of the history-feature rows a projection is
    computed from (the BASIS the bot keeps with its file)."""
    if feature_rows is None or not len(feature_rows):
        return {}
    return {str(g): C.frame_hash(df.reset_index(drop=True))
            for g, df in feature_rows.groupby("game_id", sort=True)}


# ── The frame ────────────────────────────────────────────────────────────────
def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in DTYPES.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """The section 3 columns, in order, in their dtypes. A missing column
    raises (a contract break, never filled); so does a missing value in a
    bool column (astype(bool) would read NaN as True)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"projection frame lacks {missing}")
    out = df[COLUMNS].copy()
    for c, t in DTYPES.items():
        s = out[c]
        if t == "bool":
            if s.isna().any():
                raise ValueError(f"projection column {c} has missing values")
            out[c] = s.astype(bool)
        elif t.startswith("datetime64[ns, "):
            out[c] = pd.to_datetime(s, utc=True).dt.tz_convert(t[len("datetime64[ns, "):-1]).astype(t)
        elif t == "datetime64[ns]":
            out[c] = pd.to_datetime(s).astype(t)
        else:
            out[c] = s.astype(t)
    return out


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(ORDER, kind="mergesort").reset_index(drop=True)


def project(feature_rows: pd.DataFrame, games: pd.DataFrame, schedule: pd.DataFrame, report,
            lines: pd.DataFrame | None, models: M.Models, now, marked_out=None,
            provisional: bool = False, *, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """The projection frame for `feature_rows` (committed history features:
    serve.history_features rows, no lines, no context).

    games       schedule rows of every game the rows are on (game_id,
                game_date, tip_utc, away, home)
    schedule    the season schedule (validates the report's rows)
    report      a ParsedReport / CommittedReport; None raises NoReport
    lines       (game_id, home_line, total); a game with no row, or without
                both values, is WITHHELD (module docstring)
    now         aware clock; the context cutoff is min(now, tip - 5 min)
    marked_out  person ids a visitor marked out (status Out at age 0)
    names       every spelling per person id (History.names, known_names)

    Raises NoReport and NullSkew (models.check_nulls) and MissingFeatures
    (predict) rather than projecting a slate training never saw. attrs:
    withheld (game ids with no line), unresolved (report rows naming nobody
    on the roster).
    """
    now_u = _utc(now)
    if report is None:
        raise NoReport("no injury report at all: every status would be unknown and report_age_h "
                       "null on every row, which training never had")
    clash = [c for c in LINE_COLS if c in feature_rows.columns]
    if clash:
        raise ValueError(f"feature rows already carry line columns {clash}: pass the committed "
                         f"history features, the lines go in `lines`")
    if lines is None:
        lines = lines_from_odds(None, games)
    rows = F.apply_lines(feature_rows, lines)
    lined = rows["home_line"].notna() & rows["total"].notna()
    withheld = sorted(set(rows.loc[~lined, "game_id"]))
    rows = rows[lined].reset_index(drop=True)
    if rows.empty:
        out = empty_frame()
        out.attrs.update(withheld=withheld, unresolved=0)
        return out
    games = games[games["game_id"].isin(set(rows["game_id"]))]
    marks = {int(p) for p in marked_out} if marked_out else set()
    ctx = CTX.context_for(rows, games, schedule, report, now_u, names, marked_out=marks or None)
    M.check_nulls(models, ctx)
    pred = M.predict(models, ctx)

    out = ctx[list(IDENTITY)].copy()
    out["p_play"] = pred["p_play"]
    for c in CTX.DISPLAY_COLS:
        out[c] = ctx[c]
    out["min"] = pred["mu_min"]
    for c in [*M.STATS, *M.COMPOSED, "mu_min", *(f"rate_{s}" for s in M.STATS)]:
        out[c] = pred[c]
    out["games_prior"] = ctx["career_games"]
    out["provisional"] = bool(provisional)
    out["model_version"] = models.version
    # One report per call: a row carries its time wherever the report was in
    # reach for that game (report_states leaves exactly those teams off
    # no_report), NaT where it was not.
    ts = pd.Timestamp(to_et(report.meta["report_ts_et"]))
    out["report_ts_et"] = pd.Series(ts, index=out.index).where(ctx["team_state"] != "no_report", pd.NaT)
    out["home_line"], out["total"] = ctx["home_line"], ctx["total"]
    out["computed_at"] = pd.Timestamp(now_u)
    out["frozen"] = False
    marked_teams = set(zip(ctx.loc[ctx["person_id"].isin(marks), "game_id"],
                           ctx.loc[ctx["person_id"].isin(marks), "team"]))
    out["adjusted"] = [k in marked_teams for k in zip(ctx["game_id"], ctx["team"])]
    out = _sorted(conform(out))
    out.attrs.update(withheld=withheld, unresolved=int(ctx.attrs.get("unresolved", 0)))
    return out


# ── The bot's file: freeze, carry-over, churn, stale beats empty ──────────────
def _by_person(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.sort_values("person_id", kind="mergesort")[cols].reset_index(drop=True)


def _same(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Same rows on every column but the stamps (exact: floats bit for bit)."""
    cols = [c for c in COLUMNS if c not in STAMPS]
    return _by_person(old, cols).equals(_by_person(new, cols))


def checkpoints(tip) -> list[dt.datetime]:
    """A game's training horizons as UTC clock times (horizons.cutoff_for:
    each floored at tip minus 5 minutes): when a published projection is
    refreshed even though no input changed."""
    t = pd.Timestamp(tip).to_pydatetime()
    return sorted(HZ.cutoff_for(h, t) for h in HZ.HORIZONS)


def _clock_only(old: pd.DataFrame, new: pd.DataFrame, old_basis: str | None,
                new_basis: str | None, now_u: dt.datetime) -> bool:
    """True when the new rows differ from the committed ones by the clock
    alone: the same feature rows (basis), the same INPUT_COLS, and no
    training horizon crossed since the committed rows were computed."""
    if old_basis is None or old_basis != new_basis:
        return False
    last = pd.Timestamp(old["computed_at"].min()).to_pydatetime()
    if any(last < c <= now_u for c in checkpoints(old["tip_utc"].iloc[0])):
        return False
    return _by_person(old, INPUT_COLS).equals(_by_person(new, INPUT_COLS))


def merge_published(old: pd.DataFrame | None, new: pd.DataFrame | None, games: pd.DataFrame,
                    now, *, old_basis: dict | None = None, new_basis: dict | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """The date's projections file after one bot pass, and its BASIS.

    old        the committed file (None when absent); old_basis its BASIS
    new        project()'s frame for the pass; new_basis feature_digests of
               the rows it was computed from
    games      the date's state games (state_games); `now` the clock at WRITE
               time

    Per game, in tip order:
      frozen (now >= tip - 5 min)  the committed rows, byte for byte; new rows
                                   are ignored, so a game is never written for
                                   the first time after its freeze point (a
                                   provisional row is carried too, and stays
                                   provisional: never the frozen projection)
      no new rows (withheld, or    the committed rows when there are any:
      the pass failed)             stale beats empty
      new rows equal to the file   the committed rows (stamps included)
      only the clock moved         the committed rows (module docstring:
                                   COMMIT CHURN)
      otherwise                    the new rows
    Rows of games no longer on the date (postponed) are dropped. The basis
    holds, per game written, the hash of the feature rows its rows come from.
    """
    now_u = _utc(now)
    old_basis, new_basis = old_basis or {}, new_basis or {}
    frozen = frozen_ids(games, now_u)
    old = conform(old) if old is not None and len(old) else None
    new = conform(new) if new is not None and len(new) else None
    parts, basis = [], {}
    for gid in games["game_id"]:
        o = old[old["game_id"] == gid] if old is not None else None
        n = new[new["game_id"] == gid] if new is not None else None
        o = o if o is not None and len(o) else None
        n = n if n is not None and len(n) else None
        ob, nb = old_basis.get(gid), new_basis.get(gid)
        if gid in frozen or n is None:
            rows, b = o, ob
        elif o is not None and _same(o, n):
            rows, b = o, nb
        elif o is not None and _clock_only(o, n, ob, nb, now_u):
            rows, b = o, ob
        else:
            rows, b = n, nb
        if rows is not None:
            parts.append(rows)
            if b is not None:
                basis[gid] = b
    out = _sorted(pd.concat(parts, ignore_index=True)) if parts else empty_frame()
    return out, basis


def read_published(date) -> tuple[pd.DataFrame | None, dict]:
    """The committed projections file of `date` (conformed) and its BASIS."""
    p = projections_path(date)
    df = C.read_parquet(p)
    return (conform(df) if df is not None else None), (C.read_parquet_meta(p).get(BASIS) or {})


def write_published(frame: pd.DataFrame, basis: dict, date, old_basis: dict | None = None) -> bool:
    """Write the bot's projections file with its BASIS. Unchanged rows and an
    unchanged basis leave the file byte-identical. Returns True when it
    wrote."""
    return C.write_parquet(frame, projections_path(date), meta={BASIS: basis},
                           if_changed=basis == (old_basis or {}))


# ── The app's frame ──────────────────────────────────────────────────────────
@dataclass
class Info:
    """What the page says about the frame live_projections returned."""
    date: dt.date
    source: str = "none"              # "live" (recomputed now) | "committed" | "none"
    reason: str | None = None         # why there is no frame at all
    stale: str | None = None          # why the committed projections stand in for a recompute
    report_ts_et: dt.datetime | None = None
    report_source: str | None = None  # live.Live source of the report ("live", "memory", "committed")
    lines_source: str | None = None
    # Any served row provisional (tomorrow, state missing a final, features
    # built before the latest finals): the stale-state badge.
    provisional: bool = False
    data_through: dt.date | None = None
    completeness: list[str] = field(default_factory=list)
    model_version: str | None = None
    withheld: list[str] = field(default_factory=list)       # no line yet: NO_LINE
    # past tip - 5 min and served from committed, non-provisional rows
    frozen: list[str] = field(default_factory=list)
    # frozen, or never built by the bot, and with no committed rows to serve
    unpublished: list[str] = field(default_factory=list)
    adjusted: bool = False                                   # a mark-out changed some rows
    notes: list[str] = field(default_factory=list)


def _source_of(x) -> tuple[object, str]:
    """(value, source label) from a live.Live or a plain value."""
    if hasattr(x, "value") and hasattr(x, "source"):
        label = x.source + (f" (stale: {x.error})" if getattr(x, "stale", False) else "")
        return x.value, label
    return x, "given"


def _mark_frozen(df: pd.DataFrame, past: set[str]) -> pd.DataFrame:
    """frozen = the game is past its freeze point AND the row is not
    provisional (a provisional row is never the frozen projection)."""
    df = df.copy()
    df["frozen"] = df["game_id"].isin(past) & ~df["provisional"].astype(bool)
    return df


def _finish(frame: pd.DataFrame, past: set[str], info: Info) -> tuple[pd.DataFrame, Info]:
    """The served frame (frozen flagged, sorted) and the Info fields read off
    it: which games are frozen, whether any row is provisional."""
    out = _sorted(_mark_frozen(frame, past))
    info.frozen = sorted(set(out.loc[out["frozen"], "game_id"]))
    unfrozen = sorted(set(out.loc[out["game_id"].isin(past) & ~out["frozen"], "game_id"]))
    if unfrozen:
        info.notes.append(f"{len(unfrozen)} game(s) reached tip on provisional projections: "
                          f"shown as published, never frozen.")
    info.provisional = info.provisional or bool(out["provisional"].any())
    return out, info


def _committed_only(committed: pd.DataFrame | None, games: pd.DataFrame | None, now_u: dt.datetime,
                    info: Info, marks: set[int], why: str | None) -> tuple[pd.DataFrame, Info]:
    """The committed projections for the date's games as the bot wrote them
    (frozen flagged), with `why` as the stale notice. Without `games` (the
    schedule could not be read) every committed row is served, and its own
    tip_utc decides whether its game is past the freeze point."""
    info.stale = why
    if marks:
        info.notes.append(MARKS_NOT_APPLIED)
    if committed is None or committed.empty:
        info.source = "none"
        info.reason = NOT_PUBLISHED
        return empty_frame(), info
    if games is None:
        rows = committed
        past = frozen_ids(committed.drop_duplicates("game_id"), now_u)
    else:
        rows = committed[committed["game_id"].isin(set(games["game_id"]))]
        past = frozen_ids(games, now_u)
    info.source = "committed"
    info.model_version = info.model_version or (rows["model_version"].iloc[0] if len(rows) else None)
    return _finish(rows, past, info)


def live_projections(date, now, marked_out=None, *, report=None, odds=None,
                     models: M.Models | None = None, schedule: pd.DataFrame | None = None
                     ) -> tuple[pd.DataFrame, Info]:
    """The app's frame for one ET date and the Info the page shows with it.

    Committed features (state/serve/{date}/features_v1) + the live report
    (live.injury_report) + live lines (live.odds_on, mapped to game ids
    through the schedule) -> project(). Games past their freeze point come
    from the committed projections UNTOUCHED (frozen=True unless provisional;
    mark-outs never reach them). Any live failure, a NullSkew or no report
    at all serves the committed projections with Info.stale saying why.
    Features built before the latest ingested finals make the frame
    provisional. With neither file committed for the date: an empty frame
    and Info.reason NOT_PUBLISHED.

    report, odds, models, schedule: pass them to skip the live fetch / load
    (a live.Live or a plain value); the app passes its cached models.
    """
    d = S.as_date(date)
    now_u = _utc(now)
    info = Info(date=d)
    marks = {int(p) for p in marked_out} if marked_out else set()
    feats = C.read_parquet(features_path(d))
    committed = C.read_parquet(projections_path(d))
    committed = conform(committed) if committed is not None else None
    if feats is None and committed is None:
        info.reason = NOT_PUBLISHED
        return empty_frame(), info
    season = season_for_date(d)
    try:
        sched = schedule if schedule is not None else LV.schedule().value
        games = state_games(sched, d)
        frozen = frozen_ids(games, now_u)
        info.provisional, info.completeness = provisional_for(d, to_et(now_u).date(), sched, season)
        ingested = ingested_games(season)
        info.data_through = data_through(season, ingested)
    except LIVE_FAILURES as e:
        return _committed_only(committed, None, now_u, info, marks,
                               f"schedule or state unreadable: {type(e).__name__}: {e}")
    have = set(committed["game_id"]) if committed is not None else set()
    info.unpublished = sorted(frozen - have)
    tipped = committed[committed["game_id"].isin(frozen)] if committed is not None else empty_frame()
    todo = [g for g in games["game_id"] if g not in frozen]
    if not todo:
        info.source = "committed" if len(tipped) else "none"
        info.reason = None if len(tipped) else NOT_PUBLISHED
        info.model_version = tipped["model_version"].iloc[0] if len(tipped) else None
        if marks:
            info.notes.append("Every game has tipped: mark-outs no longer apply.")
        return _finish(tipped, frozen, info)
    if feats is None:
        return _committed_only(committed, games, now_u, info, marks,
                               "the history features for this date are not committed")
    if C.read_parquet_meta(features_path(d)).get(HISTORY) != history_stamp(season, d, ingested):
        info.provisional = True
        info.notes.append(STALE_FEATURES)
    # A game the bot has not built features for (added to the schedule since)
    # can only be served from committed rows.
    unbuilt = [g for g in todo if g not in set(feats["game_id"])]
    try:
        rep, info.report_source = _source_of(report if report is not None
                                             else LV.injury_report(sched))
        od = odds if odds is not None else LV.odds_on(d)
        if getattr(od, "stale", False) and getattr(od, "source", "") == "none":
            raise RuntimeError(f"lines feed failed ({od.error})")
        od, info.lines_source = _source_of(od)
        mdl = models if models is not None else M.load()
        rows = feats[feats["game_id"].isin(todo)]
        fresh = project(rows, games, sched, rep, lines_from_odds(od, games), mdl, now_u,
                        marked_out=marks, provisional=info.provisional,
                        names=known_names(season, d))
    except LIVE_FAILURES as e:
        return _committed_only(committed, games, now_u, info, marks, f"{type(e).__name__}: {e}")
    info.source = "live"
    info.model_version = mdl.version
    info.report_ts_et = to_et(rep.meta["report_ts_et"])
    # A withheld game keeps its committed rows when the bot published it with a
    # line (stale beats empty); otherwise the page says the line is not up yet.
    withheld = fresh.attrs.get("withheld", [])
    stand_in = set(withheld) | set(unbuilt)
    held = committed[committed["game_id"].isin(stand_in)] if committed is not None else empty_frame()
    info.withheld = sorted(set(withheld) - set(held["game_id"]))
    info.unpublished = sorted(set(info.unpublished) | (set(unbuilt) - set(held["game_id"])))
    info.adjusted = bool(fresh["adjusted"].any())
    if marks & set(tipped["person_id"]):
        info.notes.append("Mark-outs on games past tip are not applied.")
    parts = [p for p in (tipped, fresh, held) if len(p)]
    return _finish(conform(pd.concat(parts, ignore_index=True)) if parts else empty_frame(),
                   frozen, info)
