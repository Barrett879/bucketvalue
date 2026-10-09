"""The volatile part of a game's context: injury status at a horizon, and the
teammate absences it implies. ONE function for training and serving.

Training: horizon snapshots from the PDF archive (bucketlib.horizons), joined to
every rostered player-game. Serving: the latest report the app or bot can fetch,
plus the user's "mark out" overrides (status Out at age 0, which the trained
definition already covers). Same function, same columns.

Leak discipline (design review): every teammate quantity is built from the
teammates' PRE-GAME state (their history features) and the REPORT at the
horizon. Nothing here looks at who actually appeared in the game, at DNPs, or at
the inactive list: those are targets, not features.

Status encoding is three-way (spec 3.2): a status from the report; KNOWN
not listed (the team filed and he is not on it); UNKNOWN (team not yet
submitted, game not in the report yet, no report) ; and the pre-archive era
(before 2018-12-17), which serving must never emit (tests assert it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .horizons import KNOWN_STATES
from .names import NameIndex

STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
REASON_GROUPS = {
    "injury": {"injury_illness", "concussion_protocol", "reconditioning", "health_protocols"},
    "gleague": {"g_league_two_way", "g_league_assignment", "g_league"},
    "rest": {"rest"},
    "other": {"not_with_team", "personal", "suspension", "coach_decision", "trade_pending",
              "other", "unspecified"},
}
# Expected-minutes proxy for a teammate: his own pre-game state.
EXP_MIN_COLS = ("min_ewm5_s", "min_std_s", "p1_mpg")
ROTATION_MIN = 10.0

CTX_COLS = (
    [f"inj_{s.lower()}" for s in STATUSES]
    + ["inj_not_listed", "inj_unknown", "inj_pre_archive"]
    + [f"inj_reason_{g}" for g in REASON_GROUPS]
    + ["hours_to_tip", "report_age_h",
       "tm_out_min", "tm_doubt_min", "tm_q_min", "tm_out_newly_min", "tm_out_max_min",
       "tm_unknown_team", "rotation_depth", "own_exp_min_share"]
)


def expected_minutes(rows: pd.DataFrame) -> pd.Series:
    e = pd.Series(np.nan, index=rows.index)
    for c in EXP_MIN_COLS:
        if c in rows:
            e = e.fillna(rows[c])
    return e.fillna(0.0)


def resolve_names(inj: pd.DataFrame, roster: pd.DataFrame,
                  known_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach person_id to injury rows by name, scoped to the game's own team
    roster (roster: game_id, team, person_id, player_name).

    `known_names` (person_id, player_name): EVERY spelling a person id has ever
    carried. The sources disagree by era and source ("Nene" / "Nene Hilario",
    "Enes Kanter" / "Enes Freedom", "Charles" / "Charlie Brown Jr."), and the
    report may use any of them, so each rostered player is indexed under all of
    his names, still scoped to that game's team. Unresolved rows keep
    person_id NaN and are counted in attrs['unresolved'] (mostly traded or
    G League players listed by a team whose game roster no longer has them,
    who therefore cannot affect that game)."""
    if inj is None or inj.empty:
        out = inj.copy() if inj is not None else pd.DataFrame()
        out["person_id"] = pd.Series(dtype="float")
        out.attrs["unresolved"] = 0
        return out
    names_by_pid: dict[int, set] = {}
    for src in (roster, known_names):
        if src is not None and len(src):
            for pid_, n in zip(src["person_id"], src["player_name"]):
                if isinstance(n, str) and n:
                    names_by_pid.setdefault(int(pid_), set()).add(n)
    out = inj.copy()
    pid = np.full(len(out), np.nan)
    idx_cache: dict[tuple, NameIndex] = {}
    for i, (gid, team, raw) in enumerate(zip(out["game_id"], out["team"], out["player_raw"])):
        key = (gid, team)
        if key not in idx_cache:
            ix = NameIndex()
            r = roster[(roster["game_id"] == gid) & (roster["team"] == team)]
            for p_ in set(r["person_id"]):
                for n in names_by_pid.get(int(p_), ()):
                    ix.add(n, int(p_), team=team)
            idx_cache[key] = ix
        m = idx_cache[key].resolve_last_first(raw, team=team)
        if m.value is not None:
            pid[i] = m.value
    out["person_id"] = pid
    out.attrs["unresolved"] = int(np.isnan(pid).sum())
    return out


def build_context(rows: pd.DataFrame, inj_players: pd.DataFrame, inj_teams: pd.DataFrame,
                  marked_out: set | None = None) -> pd.DataFrame:
    """Context columns for `rows` (one horizon at a time).

    rows         game_id, person_id, team, tip_utc + history features
                 (min_ewm5_s / min_std_s / p1_mpg, miss_streak)
    inj_players  game_id, team, person_id (resolved), status, reason_cat,
                 report_ts_et, cutoff_utc   (may be empty)
    inj_teams    game_id, team, state, report_ts_et, cutoff_utc
    marked_out   person_ids the user marked out (serving only): status Out,
                 report age 0, reason injury.
    """
    out = rows.copy()
    ip = inj_players.dropna(subset=["person_id"]).copy() if len(inj_players) else inj_players.copy()
    if len(ip):
        ip["person_id"] = ip["person_id"].astype("int64")
        ip = ip.drop_duplicates(["game_id", "person_id"], keep="last")
        out = out.merge(ip[["game_id", "person_id", "status", "reason_cat"]].rename(
            columns={"status": "_st", "reason_cat": "_rc"}), on=["game_id", "person_id"], how="left")
    else:
        # Object dtype: a mark-out below writes strings into them, which a
        # float NaN column refuses (FutureWarning in pandas 2, an error in 3).
        out["_st"] = pd.Series(np.nan, index=out.index, dtype=object)
        out["_rc"] = pd.Series(np.nan, index=out.index, dtype=object)
    tstate = inj_teams[["game_id", "team", "state", "report_ts_et", "cutoff_utc"]].drop_duplicates(
        ["game_id", "team"], keep="last")
    out = out.merge(tstate, on=["game_id", "team"], how="left")

    if marked_out:
        out["state"] = out["state"].astype(object)      # all-NaN (float) when no team rows matched
        mk = out["person_id"].isin(marked_out)
        out.loc[mk, "_st"] = "Out"
        out.loc[mk, "_rc"] = "injury_illness"
        out.loc[mk, "state"] = "listed"

    st = out["_st"]
    state = out["state"].fillna("no_report")
    known = state.isin(KNOWN_STATES)
    for s in STATUSES:
        out[f"inj_{s.lower()}"] = (st == s).astype(float)
    out["inj_not_listed"] = (st.isna() & known).astype(float)
    out["inj_pre_archive"] = (state == "pre_archive").astype(float)
    out["inj_unknown"] = (st.isna() & ~known & (state != "pre_archive")).astype(float)
    for g, cats in REASON_GROUPS.items():
        out[f"inj_reason_{g}"] = out["_rc"].isin(cats).astype(float)

    tip = pd.to_datetime(out["tip_utc"], utc=True)
    cut = pd.to_datetime(out["cutoff_utc"], utc=True)
    rep = pd.to_datetime(out["report_ts_et"], utc=True)
    out["hours_to_tip"] = (tip - cut).dt.total_seconds() / 3600
    out["report_age_h"] = (cut - rep).dt.total_seconds() / 3600
    if marked_out:
        out.loc[out["person_id"].isin(marked_out), "report_age_h"] = 0.0

    # Teammate quantities from PRE-GAME state only.
    em = expected_minutes(out)
    out["_em"] = em
    out["_out"] = (st == "Out").astype(float)
    out["_dbt"] = (st == "Doubtful").astype(float)
    out["_q"] = (st == "Questionable").astype(float)
    # "Newly" out: listed Out now but played his team's previous game.
    played_last = (out.get("miss_streak", pd.Series(0.0, index=out.index)).fillna(0) == 0) & (em > 0)
    out["_newly"] = out["_out"] * played_last.astype(float)
    g = out.groupby(["game_id", "team"])
    for col, src in (("tm_out_min", "_out"), ("tm_doubt_min", "_dbt"), ("tm_q_min", "_q"),
                     ("tm_out_newly_min", "_newly")):
        tot = g[src].transform(lambda s, e=out["_em"]: (s * e.loc[s.index]).sum())
        out[col] = tot - out[src] * out["_em"]          # teammates only, not himself
    # Largest absent teammate's expected minutes (the "star out" signal). A
    # player who is himself Out gets NaN: he is not redistributed to.
    outmin = out["_out"] * out["_em"]
    out["tm_out_max_min"] = outmin.groupby([out["game_id"], out["team"]]).transform("max")
    out.loc[out["_out"] == 1, "tm_out_max_min"] = np.nan
    out["tm_unknown_team"] = (~known & (state != "pre_archive")).astype(float)
    rot = ((out["_em"] >= ROTATION_MIN) & (out["_out"] == 0)).astype(float)
    out["rotation_depth"] = rot.groupby([out["game_id"], out["team"]]).transform("sum") - rot
    team_em = (out["_em"] * (1 - out["_out"])).groupby([out["game_id"], out["team"]]).transform("sum")
    out["own_exp_min_share"] = (out["_em"] / team_em).where(team_em > 0)
    return out.drop(columns=["_st", "_rc", "_em", "_out", "_dbt", "_q", "_newly", "state",
                             "report_ts_et", "cutoff_utc"])
