"""Live injury context: the report the app or the bot holds right now, turned
into the SAME team states and context columns the models were trained on.

Training snapshots each game at fixed horizons (bucketlib.horizons); serving
has one already-chosen report and a clock. Both go through
horizons.game_states for the per-game team state and through
livectx.build_context for the columns, so a state or a feature cannot mean one
thing in training and another on the site (GridironValue shipped four silent
train/serve skews). tests/test_context.py replays real 2025-26 games through
this module and compares every context column to the training table bit for bit.

The serving horizon is "live": cutoff = horizons.tip_floored(now, tip), the
same tip floor the training horizons use. Whether the report in hand counts
for a game is decided by the training selection rule itself
(horizons.report_at, asked about this one report): titled STRICTLY before the
cutoff and no more than 30 hours behind it. Otherwise the game is no_report,
which is what a training snapshot records when no report is in reach: a
report titled after a game's cutoff is post-tip news for it, and a report a
day and a half old is not a status the models ever saw. A team that has not
filed, a game the report does not mention yet, and every game when there is
no report at all are UNKNOWN too: unknown is never healthy.

Committed fallback: the bot writes the report it holds to
state/current/injury_{players,teams}_v1.parquet (write_committed; the paths
come from bucketlib.state, which owns state/) and the app's live layer reads
it back as a CommittedReport when its own fetch fails (live.injury_report).
Both files are a function of the report alone, never of a game list or a
clock (a file that depended on either would be clobbered by the next call for
other games, or rewritten every run), and every row carries the report time:
  players  the report's player rows that pass schedule validation
  teams    per (game_id, team) of every game the report names: the state the
           report gives it, with no cutoff applied (report_states applies the
           cutoff on read-back, exactly as it does for a live report)
A re-issue that says exactly what the committed pair says (only the title
moved: the NBA re-titles an unchanged report every 15 minutes) is not
written until the committed title is COMMITTED_RESTAMP behind, so the pair
is not a new commit on every bot run.
read_committed raises when the pair cannot be trusted, and a round-trip test
proves the states survive for any games and any clock.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache as C
from . import horizons as HZ
from . import livectx as L
from . import state as S
from .sources import injury as INJ
from .util import UTC, to_et

LIVE = "live"

# The bot's committed report (one writer per path), read by the app's
# fallback. bucketlib.state owns every path under state/; these are names.
INJURY_PLAYERS = "injury_players"
INJURY_TEAMS = "injury_teams"
# write_committed: a report saying what the committed pair says is not
# written until the committed title is this far behind (the NBA re-issues an
# unchanged report every 15 minutes with a new title). The bound keeps the
# fallback's title, and so report_age_h in the app's committed fallback,
# within 3 hours of the newest report the bot saw, and well inside the
# 30-hour reach (horizons.MAX_BACK) beyond which a held report is no report.
# It costs at most 8 restamp commits a day while nothing changes.
COMMITTED_RESTAMP = dt.timedelta(hours=3)
# Metadata keys of the committed pair: the content digest (_content_digest)
# and the title it was written under.
CONTENT = "report_content"
TITLE = "report_title"

PLAYER_COLS = ["game_id", "horizon", "cutoff_utc", "report_ts_et", "team", "player_raw",
               "status", "reason_cat", "reason_raw"]
TEAM_COLS = ["game_id", "horizon", "cutoff_utc", "report_ts_et", "team", "state"]
COMMITTED_TEAM_COLS = ["game_id", "game_date", "away", "home", "team", "state", "report_ts_et"]

# Display columns context_for adds next to livectx.CTX_COLS (never model
# inputs): exactly the projection frame's availability columns
# (docs/serving_design.md section 3), so display_status can draw badges for
# the committed projections as well as for the live frame.
DISPLAY_COLS = ["inj_status", "inj_reason", "team_state"]
MARKED_REASON = "Marked out (your adjustment)"

# Badge labels (spec 5). A player with no badge is KNOWN not listed; every
# unknown state gets a label, so unknown can never render as healthy.
NOT_FILED = "Report not filed"
NO_REPORT = "No report yet"
_OUT_BADGES = {"g_league_two_way": "Two-way", "g_league_assignment": "G League",
               "g_league": "G League", "not_with_team": "Not with team",
               "suspension": "Suspended"}

# The committed teams file's states that become team-level lines on read-back
# ("listed" needs none: the player rows say it). A "reported_none" line is
# redundant today (game_states only reaches that state when another line or
# row puts the game in the report, and that one is kept too); it is kept so
# the read-back does not lean on that.
_REPORT_KINDS = ("not_yet_submitted", "all_players_available", "reported_none")


@dataclass(frozen=True)
class CommittedReport:
    """A report rebuilt from the bot's committed files: the three fields
    report_states reads from a ParsedReport (meta["report_ts_et"], rows,
    team_rows), so the fallback goes through the one state function."""
    meta: dict
    rows: pd.DataFrame
    team_rows: pd.DataFrame


# ── States ───────────────────────────────────────────────────────────────────
def report_states(games: pd.DataFrame, schedule: pd.DataFrame, report, now
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-game team states and listed players from ONE chosen report.

    games     game_id, game_date, tip_utc, away, home
    schedule  validates every report row (QA and Summer League files share
              the production bucket), exactly as horizons.snapshot_games does
    report    a ParsedReport or CommittedReport, or None (every team no_report)
    now       aware datetime

    Returns (player_rows, team_rows) with snapshot_games' columns (PLAYER_COLS,
    TEAM_COLS), horizon "live" and cutoff_utc = horizons.tip_floored(now, tip).
    """
    sched = HZ.schedule_frame(schedule)
    kept = teams = ts = in_reach = None
    if report is not None:
        kept, teams = HZ.report_tables(report, sched)
        ts = report.meta["report_ts_et"]
        in_reach = _selectable(report)
    prow, trow = [], []
    for _, g in games.iterrows():
        cut = HZ.tip_floored(now, pd.Timestamp(g["tip_utc"]).to_pydatetime())
        base = {"game_id": g["game_id"], "horizon": LIVE, "cutoff_utc": cut}
        if in_reach is None or not in_reach(cut):
            for team in (g["away"], g["home"]):
                trow.append({**base, "report_ts_et": pd.NaT, "team": team, "state": "no_report"})
            continue
        p, t = HZ.game_states(g, base, kept, teams, ts)
        prow.extend(p)
        trow.extend(t)
    return pd.DataFrame(prow, columns=PLAYER_COLS), pd.DataFrame(trow, columns=TEAM_COLS)


def _selectable(report) -> Callable[[dt.datetime], bool]:
    """cutoff -> would horizons.report_at (the training selection rule) pick
    this report at that cutoff, were it the only report in the archive?

    That is the rule's own strictness (title strictly before the cutoff) and
    its own 30-hour reach, not a restatement of them. A held report has no
    filename, so it stands in the newest slot at or before its title time
    (the hourly era's 07PM file holds the 7:30 report; a 10:58 title sits in
    the 10:45 slot). Memoized per cutoff: a slate mostly shares one clock."""
    own = next(INJ.slots_descending(to_et(report.meta["report_ts_et"]), inclusive=True), None)
    memo: dict[dt.datetime, bool] = {}

    def held(slot):
        return report if own is not None and slot.nominal_et == own.nominal_et else None

    def at(cut: dt.datetime) -> bool:
        if cut not in memo:
            memo[cut] = HZ.report_at(cut, held) is not None
        return memo[cut]
    return at


# ── Context columns ──────────────────────────────────────────────────────────
def context_for(rows: pd.DataFrame, games: pd.DataFrame, schedule: pd.DataFrame, report, now,
                names: pd.DataFrame | None, marked_out=None) -> pd.DataFrame:
    """livectx.build_context for `rows` against the live report.

    rows        game_id, person_id, team, tip_utc + history features
                (min_ewm5_s / min_std_s / p1_mpg, miss_streak); player_name
                when present
    games       every game the rows belong to (a row on no game in `games`
                raises: it would otherwise read as an unknown team silently)
    names       (person_id, player_name): every spelling a person id has
                carried. The report may use any of them (Kanter / Freedom).
    marked_out  person ids the visitor marked out: status Out at age 0.

    Names resolve team-scoped against the game's own rows, as in training.
    Returns rows + livectx.CTX_COLS + DISPLAY_COLS, in the order of `rows`.
    """
    games = games.drop_duplicates("game_id")
    _check_rows_on_games(rows, games)
    players, teams = report_states(games, schedule, report, now)
    roster = rows[["game_id", "team", "person_id"]].assign(
        player_name=rows["player_name"] if "player_name" in rows else None)
    ip = L.resolve_names(players, roster, names)
    unresolved = ip.attrs.get("unresolved", 0)
    ip = _one_listing_per_player(ip)
    out = L.build_context(rows, ip, teams, marked_out=marked_out)
    out = _with_display(out, ip, teams, marked_out)
    out.attrs["unresolved"] = unresolved
    return out


def _check_rows_on_games(rows: pd.DataFrame, games: pd.DataFrame) -> None:
    """Every row's (game_id, team) is one side of a game in `games`."""
    sides = set(zip(games["game_id"], games["away"])) | set(zip(games["game_id"], games["home"]))
    off = sorted({k for k in zip(rows["game_id"], rows["team"]) if k not in sides})
    if off:
        raise ValueError(f"context_for: {len(off)} (game_id, team) of rows are on no game in "
                         f"`games`: {off[:5]}")


def _one_listing_per_player(ip: pd.DataFrame) -> pd.DataFrame:
    """A player listed twice for one game keeps his LAST listing, the one
    build_context encodes. Selecting it here, once, before build_context
    (whose own selection is then a no-op) means the encoded status and the
    displayed reason come from the same row; tests prove this equals
    build_context on the raw rows (the training path). Unresolved rows stay."""
    if not len(ip):
        return ip
    again = ip["person_id"].notna() & ip.duplicated(["game_id", "person_id"], keep="last")
    return ip[~again]


def _with_display(out: pd.DataFrame, ip: pd.DataFrame, teams: pd.DataFrame,
                  marked_out) -> pd.DataFrame:
    """inj_status from the context's own one-hot columns (one definition of
    who is listed as what); inj_reason from that same listing (ip holds one
    per player); team_state from the team rows build_context read (one per
    (game, team): games are unique and every row is on one of them)."""
    status = pd.Series(None, index=out.index, dtype=object)
    for s in L.STATUSES:
        status = status.mask(out[f"inj_{s.lower()}"] == 1, s)
    out["inj_status"] = status
    listed = ip.dropna(subset=["person_id"]) if len(ip) else ip
    reason = dict(zip(zip(listed["game_id"], listed["person_id"].astype("int64")), listed["reason_raw"])) \
        if len(listed) else {}
    if len(reason) != len(listed):
        raise RuntimeError("context: more than one listing per player reached the display merge")
    out["inj_reason"] = pd.Series([reason.get(k) for k in zip(out["game_id"], out["person_id"])],
                                  index=out.index, dtype=object)
    if marked_out:
        out.loc[out["person_id"].isin(marked_out), "inj_reason"] = MARKED_REASON
    # A reason only travels with a status.
    out["inj_reason"] = out["inj_reason"].where(status.notna(), None)
    state = dict(zip(zip(teams["game_id"], teams["team"]), teams["state"]))
    out["team_state"] = [state[k] for k in zip(out["game_id"], out["team"])]
    return out


def display_status(rows: pd.DataFrame) -> pd.DataFrame:
    """(person_id, game_id, status, reason) for the status badges.

    Reads only the projection frame's columns (person_id, game_id,
    inj_status, inj_reason, team_state; docs/serving_design.md section 3), so
    a tipped game served from the committed projections gets the same badges.

    status: the report status (Out, Doubtful, Questionable, Probable,
    Available), Out refined by its reason category (Two-way, G League, Not
    with team, Suspended; injury.reason_category, the parser's own mapping).
    With no status: None ONLY where the team's state is a known one
    (horizons.KNOWN_STATES: the team filed and he is not on it, the
    encoding's inj_not_listed); "Report not filed" for a team that has not
    submitted; "No report yet" for every other unknown, a missing state
    included, so unknown is never left blank. reason: the report's text (or
    the mark-out note); raw PDF text, so util.esc it when rendering.
    """
    st = rows["inj_status"].astype(object).where(rows["inj_status"].notna(), None)
    reason = rows["inj_reason"].astype(object).where(st.notna() & rows["inj_reason"].notna(), None)
    badge = reason.map(lambda r: None if r is None else _OUT_BADGES.get(INJ.reason_category(r)))
    label = st.where(~((st == "Out") & badge.notna()), badge)
    state = rows["team_state"]
    unknown = pd.Series(np.where(state == "not_yet_submitted", NOT_FILED, NO_REPORT),
                        index=rows.index, dtype=object)
    label = label.where(label.notna(), unknown.where(~state.isin(HZ.KNOWN_STATES), None))
    return pd.DataFrame({"person_id": rows["person_id"], "game_id": rows["game_id"],
                         "status": label, "reason": reason}, index=rows.index)


# ── Committed fallback (bot writes, app reads) ───────────────────────────────
def committed_paths() -> tuple[Path, Path]:
    """(players, teams) paths, resolved through bucketlib.state at call time
    (a test that points state.STATE_DIR elsewhere moves them too)."""
    return S.current_path(INJURY_PLAYERS), S.current_path(INJURY_TEAMS)


def committed_frames(report, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(players, teams) the bot commits for `report` (module docstring).

    Refuses what must never replace a good committed report: None, and a
    report naming no real NBA game (a QA or Summer League file; the
    horizons.real_report accept every report choice uses). Stale beats empty.
    """
    if report is None:
        raise ValueError("no report: keep the committed files as they are (stale beats empty)")
    ts = report.meta["report_ts_et"]
    if not HZ.real_report(schedule)(report):
        raise ValueError(f"report {ts} names no real NBA game (QA or Summer League file?): "
                         "keep the committed files as they are (stale beats empty)")
    sched = HZ.schedule_frame(schedule)
    kept, lines = HZ.report_tables(report, sched)
    players = kept if kept is not None and len(kept) else report.rows.iloc[0:0]
    if len(players) and (pd.to_datetime(players["report_ts_et"], utc=True) != pd.Timestamp(ts)).any():
        raise ValueError(f"report {ts}: player rows carry another report time; refusing to commit")
    named = set(players["game_id"]) if len(players) else set()
    if lines is not None and len(lines):
        named |= set(INJ.validate_rows(lines, sched)[0]["game_id"])
    games = sched[sched["game_id"].isin(named)].sort_values(["game_date", "game_id"])
    trow = []
    for _, g in games.iterrows():
        trow.extend(HZ.game_states(g, {"game_id": g["game_id"]}, kept, lines, ts)[1])
    teams = pd.DataFrame(trow, columns=["game_id", "report_ts_et", "team", "state"]).merge(
        games[["game_id", "game_date", "away", "home"]], on="game_id", how="left")
    return players.reset_index(drop=True), teams[COMMITTED_TEAM_COLS].reset_index(drop=True)


def _content_digest(players: pd.DataFrame, teams: pd.DataFrame) -> str:
    """What a report SAYS, without its title: both committed frames hashed
    with report_ts_et dropped (every row carries it)."""
    return C.json_hash([C.frame_hash(players.drop(columns="report_ts_et")),
                        C.frame_hash(teams.drop(columns="report_ts_et"))])


def _restamp_due(paths: tuple[Path, Path], digest: str, ts: dt.datetime) -> bool:
    """True unless the committed pair already says what this report says
    (both files carry `digest`) under one title no more than
    COMMITTED_RESTAMP behind `ts`. A pair from an older writer (no digest),
    a half-written pair (two titles) or a title gone stale is rewritten."""
    metas = [C.read_parquet_meta(p) for p in paths]
    if any(m.get(CONTENT) != digest for m in metas):
        return True
    titles = {m.get(TITLE) for m in metas}
    if len(titles) != 1 or None in titles:
        return True
    held = pd.Timestamp(titles.pop())
    return held < pd.Timestamp(ts) - COMMITTED_RESTAMP


def write_committed(report, schedule: pd.DataFrame) -> bool:
    """The bot's writer for the two committed files. Raises on any failure and
    refuses what committed_frames refuses. Returns True when either file was
    written.

    A report whose content (every row and team state, the title aside) equals
    the committed pair's is not written, unless the committed title is more
    than COMMITTED_RESTAMP behind this report's: the NBA re-issues the report
    every 15 minutes around the clock with a new title even when nothing
    changed, and the title is on every row, so a write per re-issue made both
    files a new commit on every bot run (2026-03-12: 96 titles, the row
    count changed about 11 times). Both files are written or neither, so the pair
    always carries one title (read_committed refuses two)."""
    players, teams = committed_frames(report, schedule)
    paths = committed_paths()
    digest = _content_digest(players, teams)
    ts = report.meta["report_ts_et"]
    if not _restamp_due(paths, digest, ts):
        return False
    # Both, unconditionally: an empty players file hashes the same under any
    # title, and skipping it would leave the pair's metadata on two titles.
    meta = {CONTENT: digest, TITLE: pd.Timestamp(ts).isoformat()}
    C.write_parquet(players, paths[0], meta=meta)
    C.write_parquet(teams, paths[1], meta=meta)
    return True


def read_committed() -> tuple[CommittedReport | None, float | None]:
    """(CommittedReport, fetched_at) from the committed pair, or (None, None)
    when neither file exists.

    Raises ValueError when the pair cannot be trusted: one file without the
    other; files from two different reports (each is written on its own, so
    a run that dies between the two leaves a mixed pair, and a report time
    paired with another report's rows would mis-gate and mis-age them); or a
    listed team without its player rows (read back, it would turn into
    'filed nobody', a false healthy)."""
    pp, tp = committed_paths()
    players, teams = C.read_parquet(pp), C.read_parquet(tp)
    if players is None and teams is None:
        return None, None
    if players is None or teams is None:
        raise ValueError(f"committed injury files: {(pp if players is None else tp).name} is missing")
    ts = _one_report_time(players, teams)
    listed = teams.loc[teams["state"] == "listed", ["game_id", "team"]]
    have = set(zip(players.get("game_id", []), players.get("team", [])))
    lost = [k for k in zip(listed["game_id"], listed["team"]) if k not in have]
    if lost:
        raise ValueError(f"committed injury files disagree: listed teams without rows {lost[:3]}")
    said = teams[teams["state"].isin(_REPORT_KINDS)]
    team_rows = pd.DataFrame({"report_ts_et": ts, "game_date": said["game_date"],
                              "away": said["away"], "home": said["home"],
                              "team": said["team"], "kind": said["state"]}).reset_index(drop=True)
    rep = CommittedReport({"report_ts_et": ts, "source": "committed"}, players, team_rows)
    stamps = [C.read_parquet_meta(p).get("fetched_at") for p in (pp, tp)]
    return rep, max((s for s in stamps if s is not None), default=None)


def _one_report_time(players: pd.DataFrame, teams: pd.DataFrame) -> dt.datetime:
    """The one report time every row of both committed files carries."""
    tt = pd.to_datetime(teams["report_ts_et"], utc=True)
    if not len(tt) or tt.isna().any() or tt.nunique() != 1:
        raise ValueError(f"committed teams file holds {tt.nunique()} report times "
                         f"({int(tt.isna().sum())} missing) for {len(tt)} rows")
    t0 = tt.iloc[0]
    tp = set(pd.to_datetime(players["report_ts_et"], utc=True)) if len(players) else set()
    if tp - {t0}:
        raise ValueError(f"committed injury files disagree: teams from the {t0} report, "
                         f"players from {sorted(str(t) for t in tp - {t0})[:2]}")
    return to_et(t0.to_pydatetime())
