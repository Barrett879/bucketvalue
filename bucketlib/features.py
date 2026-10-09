"""Point-in-time features. ONE builder for training and serving.

HOW PARITY IS GUARANTEED
------------------------
Training and serving both call `build(events, targets, ...)`. `events` is the
roster-game history (tables.roster_games, state games only). `targets` are the
rows to featurize:
  - training: the events themselves;
  - serving: synthetic rows for games not yet played (outcomes unknown),
    APPENDED to the same history.
Every history feature is computed on the combined, time-ordered frame and then
SHIFTED ONE EVENT within its group, so a row's own outcome can never reach its
features, and a synthetic row's features depend only on the history before it.
tests/test_features.py proves it two ways on real data: a DATE-REPLAY test
(truncate history before D, append D's games as synthetic rows, the features
equal the training features bit for bit) and a FUTURE-MUTATION test (scramble
every outcome on or after D; D's features do not move).

WHAT IS DELIBERATELY NOT A FEATURE
----------------------------------
  - Anything grouped or ranked within the current game's rows ("who is here"
    is postgame information in an appeared-only table).
  - The current game's status, starter flag, minutes, or the final margin.
  - DNP-coach's-decision as a teammate-absence signal (an in-game decision).
  - League norms computed over a whole season (they include future games);
    norms here are as-of the game date, blended with the prior season.

SEASON BOUNDARIES (design review): short-memory MINUTES state resets each
season (an EWMA carried over the summer had opener RMSE 8.98 vs 7.64 for
prior-season MPG); long-memory RATE state carries across seasons; the prior
season is summarized explicitly. Everything keys on PERSON_ID; the team on a
row is the team of that game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .util import season_end_year, season_str

RATE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fgm", "fga", "fg3a",
              "ftm", "fta", "oreb", "dreb", "pf"]
MIN_HALF_LIVES = (2, 5)
RATE_HALF_LIVES = (10, 20, 40)
TEAM_HALF_LIFE = 10
# League-norm ballast: prior-season mean counts as this many team-games.
LEAGUE_BALLAST = 300

EVENT_COLS = ["person_id", "game_id", "season", "game_date", "tip_utc", "team", "opp",
              "home", "status", "starter", "played", "played_seconds"] + RATE_STATS


# ── Events ───────────────────────────────────────────────────────────────────
def events_from_roster_games(rg: pd.DataFrame) -> pd.DataFrame:
    """State events: regular season + play-in rows, with canonical minutes
    (played_seconds / 60, the same unit every source provides) and stats NaN
    for anyone who did not play."""
    ev = rg.loc[rg["in_state"], [c for c in EVENT_COLS if c in rg.columns]].copy()
    ev["min"] = np.where(ev["played"], ev["played_seconds"] / 60.0, np.nan)
    for c in RATE_STATS:
        ev.loc[~ev["played"], c] = np.nan
    ev["is_target"] = False
    return ev


def make_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Synthetic event rows for games not yet played: identity columns only,
    outcomes unknown. rows needs person_id, game_id, season, game_date,
    tip_utc, team, opp, home."""
    t = rows[["person_id", "game_id", "season", "game_date", "tip_utc", "team", "opp", "home"]].copy()
    t["status"] = None
    t["starter"] = np.nan
    t["played"] = np.nan
    t["played_seconds"] = np.nan
    t["min"] = np.nan
    for c in RATE_STATS:
        t[c] = np.nan
    t["is_target"] = True
    return t


def _as_float(s: pd.Series) -> pd.Series:
    """bool/0-1/NaN -> float, NaN preserved (targets have unknown outcomes)."""
    return s.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                 else float(bool(v)))


def _order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["person_id", "tip_utc", "game_id"], kind="mergesort").reset_index(drop=True)


# ── Group helpers (all return values aligned to df's index, PRIOR-only) ───────
def _shift(df, s: pd.Series, keys) -> pd.Series:
    return s.groupby([df[k] for k in keys]).shift(1)


def _ewm_prior(df, x: pd.Series, keys, halflife) -> pd.Series:
    """EWM over non-NaN values (each played game is one step), carried across
    NaN rows, then shifted one event: the state BEFORE each row."""
    g = x.groupby([df[k] for k in keys])
    m = g.transform(lambda v: v.ewm(halflife=halflife, ignore_na=True).mean())
    return m.groupby([df[k] for k in keys]).shift(1)


def _ewm_sum_prior(df, x: pd.Series, keys, halflife) -> pd.Series:
    """Exponentially weighted SUM (decay per event step over non-NaN values),
    prior-only. Ratios of these give exposure-weighted rates (ratio of sums,
    which under-predicts bench players far less than a mean of per-game rates)."""
    alpha = 1 - 0.5 ** (1.0 / halflife)

    def run(v: pd.Series) -> pd.Series:
        out = np.empty(len(v))
        acc = 0.0
        for i, val in enumerate(v.to_numpy(dtype=float)):
            if not np.isnan(val):
                acc = acc * (1 - alpha) + val
            out[i] = acc
        return pd.Series(out, index=v.index)

    g = x.groupby([df[k] for k in keys])
    s = g.transform(run)
    return s.groupby([df[k] for k in keys]).shift(1)


def _cum_prior(df, x: pd.Series, keys) -> pd.Series:
    """Sum of x over PRIOR rows in the group (NaN counted as 0).

    A cumulative sum SHIFTED one row, never "cumsum minus the current value":
    (a + b + x) - x is not bit-identical to (a + b) in floating point, and the
    replay test (training rows vs served rows) caught exactly that drift."""
    g = [df[k] for k in keys]
    c = x.fillna(0).groupby(g).cumsum()
    return c.groupby(g).shift(1).fillna(0.0)


# ── Player history ───────────────────────────────────────────────────────────
def player_history(ev: pd.DataFrame) -> pd.DataFrame:
    """Prior-only player features for every row of `ev` (already ordered)."""
    P, PS = ["person_id"], ["person_id", "season"]
    played = _as_float(ev["played"])               # NaN for targets
    pf = played.fillna(0.0)
    m = ev["min"]
    f = pd.DataFrame(index=ev.index)

    # Minutes: short memory, reset each season.
    for h in MIN_HALF_LIVES:
        f[f"min_ewm{h}_s"] = _ewm_prior(ev, m, PS, h)
    gp = _cum_prior(ev, pf, PS)
    f["gp_s"] = gp
    f["min_std_s"] = (_cum_prior(ev, m, PS) / gp).where(gp > 0)
    starts = _as_float(ev["starter"]).where(played == 1)
    f["start_share_s"] = (_cum_prior(ev, starts, PS) / gp).where(gp > 0)
    f["events_s"] = ev.groupby(PS).cumcount()
    f["min_last"] = _shift(ev, m.groupby([ev[k] for k in PS]).ffill(), PS)
    f["started_last"] = _shift(ev, starts.groupby([ev[k] for k in PS]).ffill(), PS)

    # Availability history within the season: streak of events not played,
    # recent played rate, the last completed absence and games since it ended.
    notp = (1 - pf)
    grp = (pf.groupby([ev[k] for k in PS]).cumsum())          # increments on played rows
    streak_incl = notp.groupby([ev[k] for k in PS] + [grp]).cumsum()
    f["miss_streak"] = _shift(ev, streak_incl, PS).fillna(0)
    pr = pf.groupby([ev[k] for k in PS]).transform(lambda v: v.rolling(10, min_periods=1).mean())
    f["played_rate_l10"] = _shift(ev, pr, PS)
    # Length of the absence that ended just before the latest played game.
    prev_streak = _shift(ev, streak_incl, PS).fillna(0)
    ended = (pf == 1) & (prev_streak > 0)
    last_abs = prev_streak.where(ended)
    last_abs_ff = last_abs.groupby([ev[k] for k in PS]).ffill().fillna(0)
    f["last_absence_len"] = _shift(ev, last_abs_ff, PS).fillna(0)
    played_since = pf.groupby([ev[k] for k in PS] + [ended.astype(int).groupby([ev[k] for k in PS]).cumsum()]).cumsum()
    f["games_since_return"] = _shift(ev, played_since, PS)

    # Dates for context (days since last played; carried across seasons).
    lp = ev["game_date"].where(pf == 1)
    f["last_played_date"] = _shift(ev, lp.groupby(ev["person_id"]).ffill(), P)
    f["last_team"] = _shift(ev, ev["team"], P)
    f["events_with_team"] = ev.groupby(["person_id", "team"]).cumcount()

    # Rates: long memory, carried across seasons, exposure-weighted.
    for h in RATE_HALF_LIVES:
        wmin = _ewm_sum_prior(ev, m, P, h)
        for s in RATE_STATS:
            ws = _ewm_sum_prior(ev, ev[s], P, h)
            f[f"r_{s}_ewm{h}"] = (ws / wmin).where(wmin > 1.0)
    smin = _cum_prior(ev, m, PS)
    for s in RATE_STATS:
        f[f"r_{s}_std"] = (_cum_prior(ev, ev[s], PS) / smin).where(smin > 1.0)
    f["career_games"] = _cum_prior(ev, pf, P)
    return f


def prior_seasons(logs: pd.DataFrame, starters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (person_id, season) summaries of the PREVIOUS seasons, joinable on the
    current season. `logs` = player_games rows (RS only, any seasons, including
    priors-only seasons with no participation table).

    Columns: p1_* (last season), p2_* (two back), p3_gp; n_prior_seasons.
    """
    rs = logs[logs["season_type"] == "RS"].copy()
    rs["season_end"] = rs["season"].map(season_end_year)
    agg = rs.groupby(["person_id", "season_end"]).agg(
        gp=("game_id", "size"), min_sum=("min", "sum"),
        **{f"{s}_sum": (s, "sum") for s in RATE_STATS})
    agg["mpg"] = agg["min_sum"] / agg["gp"]
    for s in RATE_STATS:
        agg[f"rate_{s}"] = (agg[f"{s}_sum"] / agg["min_sum"]).where(agg["min_sum"] > 1)
    agg = agg.reset_index()
    out = []
    for lag, prefix, cols in ((1, "p1_", ["gp", "mpg"] + [f"rate_{s}" for s in RATE_STATS]),
                              (2, "p2_", ["gp", "mpg"] + [f"rate_{s}" for s in RATE_STATS]),
                              (3, "p3_", ["gp", "mpg"])):
        a = agg[["person_id", "season_end"] + cols].copy()
        a["season_end"] = a["season_end"] + lag
        out.append(a.rename(columns={c: prefix + c for c in cols}).set_index(["person_id", "season_end"]))
    res = pd.concat(out, axis=1)
    first = agg.groupby("person_id")["season_end"].min()
    res = res.reset_index()
    res["n_prior_seasons"] = [
        int(((agg["person_id"] == p) & (agg["season_end"] < se)).sum()) if False else np.nan
        for p, se in zip(res["person_id"], res["season_end"])]
    # Vectorized count of distinct prior seasons with an appearance.
    seasons_by_p = agg.groupby("person_id")["season_end"].apply(lambda v: np.sort(v.to_numpy()))
    res["n_prior_seasons"] = [
        int(np.searchsorted(seasons_by_p[p], se)) for p, se in zip(res["person_id"], res["season_end"])]
    res["first_season_end"] = res["person_id"].map(first)
    return res


# ── Team history and league norms ────────────────────────────────────────────
def team_possessions(tg: pd.DataFrame) -> pd.Series:
    """Possessions from box counts (the serve-time box score has no POSS
    field): FGA - OREB + TOV + 0.44 FTA."""
    return tg["fga"] - tg["oreb"] + tg["tov"] + 0.44 * tg["fta"]


TEAM_ALLOWED = ["pts", "reb", "oreb", "dreb", "ast", "fg3m", "fg3a", "fga", "fta", "stl", "blk", "tov"]


def team_history(tg: pd.DataFrame, team_targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prior-only per (game_id, team) features from team box rows (both sides of
    each state game). Pace and per-100 rates are possession-based from counts.

    `team_targets` (game_id, team, opp, game_date, season) are games not yet
    played: appended with NaN stats, exactly like player targets, so a target
    game gets the team state after the team's previous game."""
    base = ["game_id", "game_date", "season", "team", "ot_periods"] + TEAM_ALLOWED
    t = tg[base].copy()
    if team_targets is not None and len(team_targets):
        new = team_targets[~team_targets["game_id"].isin(set(t["game_id"]))]
        new = new[["game_id", "game_date", "season", "team"]].drop_duplicates()
        t = pd.concat([t, new], ignore_index=True)
    t["poss"] = team_possessions(t)
    # The opponent's side of the same game, prefixed x_ (team_games already
    # carries opp/opp_pts columns; a bare opp_ prefix would collide).
    other = t[["game_id", "team", "poss"] + TEAM_ALLOWED].rename(
        columns={c: f"x_{c}" for c in ["team", "poss"] + TEAM_ALLOWED})
    t = t.merge(other, on="game_id")
    t = t[t["team"] != t["x_team"]].copy()
    if t.groupby(["game_id", "team"]).size().max() != 1:
        raise ValueError("team_games must have exactly two sides per game")
    mins = 48 + 5 * t["ot_periods"]
    t["pace"] = (t["poss"] + t["x_poss"]) / 2 * 48 / mins
    t["ortg"] = 100 * t["pts"] / t["poss"]
    t["drtg"] = 100 * t["x_pts"] / t["x_poss"]
    for c in TEAM_ALLOWED:
        t[f"allow_{c}"] = 100 * t[f"x_{c}"] / t["x_poss"]
    t = t.sort_values(["team", "game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    f = t[["game_id", "team", "game_date", "season"]].copy()
    for c in ["pace", "ortg", "drtg"] + [f"allow_{c}" for c in TEAM_ALLOWED]:
        f[f"t_{c}"] = _ewm_prior(t, t[c], ["team"], TEAM_HALF_LIFE)
    f["t_games_s"] = t.groupby(["team", "season"]).cumcount()
    return f, t


def league_norms(t: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """League means AS OF each requested (season, game_date): over every
    team-game of that season strictly BEFORE the date, blended with the previous
    season's full mean counted as LEAGUE_BALLAST team-games. The first season in
    the window has no prior, so it uses its own as-of mean only (never its own
    full-season mean, which would include future games).

    The same function serves training and serving, so a date with no completed
    games yet gets exactly what training saw for it."""
    cols = ["pace", "ortg"] + [f"allow_{c}" for c in TEAM_ALLOWED]
    done = t.dropna(subset=["pace"])
    daily = done.groupby(["season", "game_date"])[cols].agg(["sum", "count"])
    full = done.groupby("season")[cols].mean()
    # The previous season is found by its LABEL, never by position among the
    # seasons that happen to have completed games: at a season opener the
    # current season has none, and a positional lookup made serving miss the
    # prior-season blend that training had (caught by the replay test).
    def prev_of(s):
        pv = season_str(season_end_year(s) - 1)
        return pv if pv in full.index else None
    out = []
    for s, q in dates.drop_duplicates().groupby("season"):
        qd = np.sort(q["game_date"].unique())
        if s in daily.index.get_level_values(0):
            d = daily.loc[s]
            sums = d.xs("sum", axis=1, level=1).cumsum()
            cnts = d.xs("count", axis=1, level=1).cumsum()
            idx = np.searchsorted(d.index.to_numpy(), qd, side="left") - 1   # strictly before
            S = np.where(idx[:, None] >= 0, sums.to_numpy()[np.clip(idx, 0, None)], 0.0)
            N = np.where(idx[:, None] >= 0, cnts.to_numpy()[np.clip(idx, 0, None)], 0.0)
        else:
            S = np.zeros((len(qd), len(cols)))
            N = np.zeros((len(qd), len(cols)))
        pv = prev_of(s)
        if pv is not None:
            prior = full.loc[pv].to_numpy()
            val = (S + prior * LEAGUE_BALLAST) / (N + LEAGUE_BALLAST)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)
        r = pd.DataFrame(val, columns=[f"lg_{c}" for c in cols])
        r["season"], r["game_date"] = s, qd
        out.append(r)
    return pd.concat(out, ignore_index=True)


# ── Context ──────────────────────────────────────────────────────────────────
def team_rest(schedule_dates: pd.DataFrame) -> pd.DataFrame:
    """Per (team, game_date): days since that team's previous game and games in
    the prior 7 days. `schedule_dates` holds (team, game_date) for EVERY game
    the team played or will play, including the Cup final (in no season log)."""
    d = schedule_dates.drop_duplicates().sort_values(["team", "game_date"]).reset_index(drop=True)
    prev = d.groupby("team")["game_date"].shift(1)
    d["rest_days"] = (d["game_date"] - prev).dt.days
    d["b2b"] = (d["rest_days"] == 1).astype(float)
    g7 = []
    for _, grp in d.groupby("team"):
        dates = grp["game_date"].to_numpy()
        lo = np.searchsorted(dates, dates - np.timedelta64(7, "D"), side="left")
        g7.extend((np.arange(len(dates)) - lo).tolist())
    d["games_l7"] = pd.Series(g7, index=d.sort_values(["team", "game_date"]).index).sort_index()
    return d


def build(events: pd.DataFrame, targets: pd.DataFrame | None, tg: pd.DataFrame,
          schedule_dates: pd.DataFrame, priors: pd.DataFrame,
          lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for `targets` (or for every event when targets is None).

    events          events_from_roster_games(...) over the history
    targets         make_targets(...) rows to serve, or None for training
    tg              team_games rows (state games) for team history
    schedule_dates  (team, game_date) incl. the target games and the Cup final
    priors          prior_seasons(logs)
    lines           optional (game_id, home_line, total)
    """
    frame = events if targets is None else pd.concat([events, targets], ignore_index=True)
    frame = _order(frame)
    ph = player_history(frame)
    out = pd.concat([frame, ph], axis=1)
    if targets is not None:
        out = out[out["is_target"]].copy()

    out["season_end"] = out["season"].map(season_end_year)
    out = out.merge(priors.drop(columns=["first_season_end"]), on=["person_id", "season_end"], how="left")

    team_targets = None
    if targets is not None:
        tt = out[["game_id", "team", "opp", "game_date", "season"]].drop_duplicates()
        opp_side = tt.rename(columns={"team": "opp", "opp": "team"})
        team_targets = pd.concat([tt, opp_side], ignore_index=True).drop_duplicates(["game_id", "team"])
    th, t = team_history(tg, team_targets)
    out = out.merge(th.drop(columns=["game_date", "season"]), on=["game_id", "team"], how="left")
    opp = th.drop(columns=["game_date", "season"]).rename(
        columns={c: "o" + c for c in th.columns if c.startswith("t_")})
    opp = opp.rename(columns={"team": "opp"})
    out = out.merge(opp, on=["game_id", "opp"], how="left")
    lg = league_norms(t, out[["season", "game_date"]])
    out = out.merge(lg, on=["season", "game_date"], how="left")

    rest = team_rest(schedule_dates)
    out = out.merge(rest[["team", "game_date", "rest_days", "b2b", "games_l7"]],
                    on=["team", "game_date"], how="left")
    orest = rest[["team", "game_date", "rest_days", "b2b"]].rename(
        columns={"team": "opp", "rest_days": "opp_rest_days", "b2b": "opp_b2b"})
    out = out.merge(orest, on=["opp", "game_date"], how="left")

    out["days_since_played"] = (out["game_date"] - pd.to_datetime(out["last_played_date"])).dt.days
    out["team_changed"] = (out["last_team"].notna() & (out["last_team"] != out["team"])).astype(float)
    out["exp_pace"] = out["t_pace"] * out["ot_pace"] / out["lg_pace"]
    out["home_f"] = out["home"].astype(float)

    if lines is not None:
        out = apply_lines(out, lines)
    return out


def apply_lines(out: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Line features for feature rows. `lines` is (game_id, home_line, total).

    Split out of build() so the bot and the app can add LIVE lines to committed
    history-feature rows (built with lines=None) through the same code training
    used; a game without a line keeps NaN line features."""
    out = out.merge(lines[["game_id", "home_line", "total"]], on="game_id", how="left")
    out["team_line"] = np.where(out["home"].astype(bool), out["home_line"], -out["home_line"])
    out["implied_total"] = out["total"] / 2 - out["team_line"] / 2
    out["abs_line"] = out["team_line"].abs()
    return out
