"""One game: header, context, and both teams' projected players.

URL: /Game?date=YYYY-MM-DD&away=BOS&home=DET[&out=pid,pid][&board=token].
Team params go through teams.canonical, which returns None for anything
unknown (never a pass-through into HTML), and every rendered string goes
through util.esc.

The numbers come from store.load_projections, the same cached frame the slate
reads (pipeline.live_projections: committed history features plus the live
report and lines; tipped games served frozen as committed). MIN is the
projected average; the other columns are mid-p medians "if he plays" from the
date's shared simulator provider. A visitor's mark-outs live in ?out=, written
by per-team pickers whose options never reorder between reruns (a keyed
widget's id is hashed from its options); the pickers are seeded from the URL
only when the URL itself changes, never because the two disagree. After tip,
each player's box-score line (live.box, 120 s) sits under his projected one,
and his pasted lines show what happened to them.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Before any bucketlib import: load the code on disk, never a stale package
# left in this process by an earlier deploy (freshen.py).
sys.modules.pop("freshen", None)
import freshen  # noqa: E402

freshen.ensure()

from bucketlib import live, props_ui, store, teams  # noqa: E402
from bucketlib.theme import (SITE_NAME, render_footer, render_nav, render_page_chrome,  # noqa: E402
                             render_share)
from bucketlib.util import esc, fmt_tip, to_et, today_et  # noqa: E402

st.set_page_config(page_title=f"Game | {SITE_NAME}", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

OUT_SEEN = "_url_out_seen"

qp = st.query_params
away, home = teams.canonical(qp.get("away")), teams.canonical(qp.get("home"))
try:
    date = dt.date.fromisoformat(str(qp.get("date")))
except ValueError:
    date = None
TOKEN = props_ui.board_token()
MARKS = store.parse_out(qp.get("out"))
keep = {"theme": "dark" if st.session_state.get("theme_dark") else "light",
        **store.view_params(TOKEN, MARKS)}
FOOT_KEEP = {"date": date.isoformat() if date else "", **store.view_params(TOKEN, MARKS)}
render_nav("Game", keep=FOOT_KEEP)
back = store.page_url("/", keep, date=date.isoformat() if date else None)

if not (away and home and date):
    st.markdown(store.render_notice(
        "<b>That game link is incomplete.</b> It needs a date and two NBA teams, "
        "for example <code>?date=2026-10-20&away=BOS&home=DET</code>."), unsafe_allow_html=True)
    st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
                unsafe_allow_html=True)
    render_footer(FOOT_KEEP)
    st.stop()

try:
    sched = live.schedule().value
except RuntimeError as e:
    st.markdown(store.render_notice(f"<b>The schedule feed is unreachable.</b> {esc(e)}"),
                unsafe_allow_html=True)
    render_footer(FOOT_KEEP)
    st.stop()

# The paste popover and Share live in the nav on this page too. A paste here
# can mint a NEW token, so every link below is rebuilt from it (the back link
# and the footer included).
try:
    ROSTER = live.roster().value
except RuntimeError:
    ROSTER = None
with st.container(key="bv_nav_actions"):
    if ROSTER is not None:
        _route_dates = sorted({today_et(), today_et() + dt.timedelta(days=1), date})
        TOKEN = props_ui.render_update_lines_popover(date, store.mark_projected(sched, _route_dates),
                                                     ROSTER, dates=_route_dates,
                                                     in_nav_container=False) or TOKEN
        keep.update(store.view_params(TOKEN, MARKS))
        back = store.page_url("/", keep, date=date.isoformat())
        FOOT_KEEP = {"date": date.isoformat(), **store.view_params(TOKEN, MARKS)}
    render_share(store.game_url(date, away, home, keep))

g = sched[(sched["game_date"] == date) & (sched["away"] == away) & (sched["home"] == home)]
if g.empty:
    st.markdown(store.render_notice(
        f"<b>No {esc(away)} at {esc(home)} game on {esc(date.strftime('%B %-d, %Y'))}.</b> "
        "The schedule has no such game; the date or the home team may be swapped."),
        unsafe_allow_html=True)
    st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
                unsafe_allow_html=True)
    render_footer(FOOT_KEEP)
    st.stop()
row = g.iloc[0]
gid = str(row["game_id"])
state = live.game_state(row)

# ── Projections: the date's one cached frame ─────────────────────────────────
projectable = gid in store.projected_ids(sched, date)
served = store.load_projections(date, MARKS) if projectable else None
rows = served.game(gid) if served is not None else pd.DataFrame()
past = served is not None and gid in served.past
FEATS = store.display_features(date) if projectable else None

st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
            unsafe_allow_html=True)
st.markdown(store.render_matchup_header(row, state), unsafe_allow_html=True)
odds = live.odds_on(date)
# ESPN drops the line at tip: a started game shows the line its frozen
# projection was made with, or without projection rows the bot's last
# pre-tip capture (store.committed_odds), labelled (display only).
_started = past or state in ("live", "final")
posted = store.posted_lines(odds.value, g, served.frame if served is not None else None,
                            {gid} if _started else (),
                            snapshots=store.committed_odds(date) if _started else None)
facts = store.market_facts(posted.get((away, home)), away, home,
                           unavailable=odds.stale and odds.source == "none")
_rest = store.rest_fact(store.rest_by_game(FEATS).get(gid), away, home)
if _rest:
    facts.append(_rest)
_inj = store.injury_fact(rows, served.info) if projectable else None
if _inj:
    facts.append(store.fact("Injury report", *_inj))
facts += [
    store.fact("National TV", esc(row.get("national_tv") or "None")),
    store.fact("Game", esc({"RS": "Regular season", "PRE": "Preseason", "PI": "Play-in",
                            "PO": "Playoffs", "CUP_FINAL": "NBA Cup final"}.get(row["game_type"], row["game_type"]))),
]
st.markdown(store.render_facts(facts), unsafe_allow_html=True)

if state == "postponed":
    st.markdown(store.render_notice(
        "<b>Postponed.</b> This game will not be played on this date. Projections are made "
        "for the makeup game once it is rescheduled."), unsafe_allow_html=True)
    render_footer(FOOT_KEEP)
    st.stop()


def _roster_tables(notice_html: str, statuses: pd.DataFrame | None = None) -> None:
    """No projections for this game: the rosters, every number the sentinel.
    `statuses` (store.roster_statuses: the live report through the
    projections' own context path) badges them; without it the caption says
    statuses are not shown (unknown is never drawn as healthy)."""
    st.markdown(notice_html, unsafe_allow_html=True)
    try:
        rost = live.roster()
    except RuntimeError as e:
        st.markdown(store.render_notice(f"<b>Rosters are unavailable.</b> {esc(e)}"),
                    unsafe_allow_html=True)
        return
    asof = (f"Rosters as of {pd.Timestamp(rost.fetched_at, unit='s', tz='UTC').tz_convert('America/New_York').strftime('%b %-d, %-I:%M %p')} ET"
            if rost.fetched_at else "")
    st_by = ({int(p): (s, r, o) for p, s, r, o in zip(statuses["person_id"], statuses["status"],
                                                      statuses["reason"], statuses["out"])}
             if statuses is not None else {})
    # Stacked, not side by side: ten stat columns per team need the full width
    # (side by side, the headers wrapped letter by letter and names truncated).
    for team in (away, home):
        p = rost.value[rost.value["team"] == team].copy()
        listed = [st_by.get(int(x), (None, None, False)) for x in p["person_id"]]
        p["status"] = [s if s else ("Two-way" if tw else "")
                       for (s, _, _), tw in zip(listed, p["two_way"])]
        p["reason"] = [r for _, r, _ in listed]
        p["out"] = [bool(o) for _, _, o in listed]
        p["group"] = ["Out" if o else ("Two-way contracts" if tw else "")
                      for o, tw in zip(p["out"], p["two_way"])]
        p = p.sort_values(["out", "two_way", "last_name"], kind="mergesort")
        if statuses is None:
            note = "Injury status not shown"
        else:
            labels = set(statuses.loc[statuses["team"] == team, "status"].dropna())
            note = next((u for u in store.UNKNOWN if u in labels), "")
        st.markdown(store.render_team_table(team, p, asof_note=asof, team_note=note),
                    unsafe_allow_html=True)


if not projectable:
    if row["game_type"] == "PRE":
        _roster_tables(store.render_notice(
            "<b>Preseason: no projections.</b> Preseason minutes are not regular-season "
            "minutes, so the site does not project them. The regular season opens October 20."))
    elif row["game_type"] == "PI":
        _roster_tables(store.render_notice(
            "<b>Play-in: no projections.</b> The models were trained on regular-season games "
            "only, and play-in rotations run tighter: on past play-in games starters played "
            "about 5 minutes more than projected."))
    else:
        _roster_tables(store.render_notice(
            "<b>No projections for this game.</b> Version 1 projects regular-season games only."))
    render_footer(FOOT_KEEP)
    st.stop()

info = served.info
for block in store.projection_notes(served, gid, today=today_et()):
    st.markdown(block, unsafe_allow_html=True)

if rows.empty:
    # Withheld for its line, or not published yet: the roster, badged from the
    # live report while the game is still ahead (a tipped game's report is
    # past its cutoff, so its statuses are not shown).
    _roster_tables(store.empty_notice(served, gid),
                   None if past else store.roster_statuses(row, sched, ROSTER))
    render_footer(FOOT_KEEP)
    st.stop()

# ── Mark-outs: per team, ?out=, stable options ───────────────────────────────
TEAMS = (away, home)
KEYS = {t: f"bv_out_{t}" for t in TEAMS}
names = dict(zip(rows["person_id"].astype("int64"), rows["player_name"].astype(str)))
# Options: every player of the team on this game, by name then id. Never by
# minutes (a mark-out moves minutes, which would reorder the options and
# orphan the selection) and never by status (a new report would).
OPTIONS = {t: [int(p) for p in rows[rows["team"] == t]
               .sort_values(["player_name", "person_id"])["person_id"]] for t in TEAMS}
game_pids = {p for opts in OPTIONS.values() for p in opts}
raw_out = qp.get("out")
if raw_out != st.session_state.get(OUT_SEEN):
    for t in TEAMS:
        st.session_state[KEYS[t]] = [p for p in OPTIONS[t] if p in MARKS]
    st.session_state[OUT_SEEN] = raw_out


def _apply_marks() -> None:
    chosen = {int(p) for t in TEAMS for p in (st.session_state.get(KEYS[t]) or [])}
    others = {p for p in store.parse_out(st.query_params.get("out")) if p not in game_pids}
    raw = store.out_param(others | chosen)
    if raw:
        st.query_params["out"] = raw
    elif "out" in st.query_params:
        del st.query_params["out"]
    st.session_state[OUT_SEEN] = raw


marked_here = sorted(p for p in MARKS if p in game_pids)      # what the pickers show
applied_here = [p for p in served.applied if p in game_pids]  # what moved the numbers
if bool(rows["adjusted"].any()) and applied_here:
    _rest_marks = store.out_param(set(MARKS) - game_pids)
    _clear = store.game_url(date, away, home, {**{k: v for k, v in keep.items() if k != "out"},
                                               **({"out": _rest_marks} if _rest_marks else {})})
    st.markdown(store.adjusted_banner(len(applied_here), _clear), unsafe_allow_html=True)
if not past:
    with st.expander("Mark players out" + (f" ({len(marked_here)} marked)" if marked_here else ""),
                     expanded=bool(marked_here)):
        st.caption("Mark a player out to see the rest of his team without him. Remove the "
                   "name to mark him back in. This changes only what you see.")
        # A player the report already lists Out stays in the options (they never
        # change between reruns) but marking him moves nothing: say so.
        _listed = [names.get(p, str(p)) for p in marked_here if p not in served.marks]
        if _listed:
            st.caption("Already listed Out on the injury report, so marking "
                       + ("him" if len(_listed) == 1 else "them") + " changes nothing: "
                       + props_ui.md_escape(", ".join(_listed)) + ".")
        cols = st.columns(2)
        for t, col in zip(TEAMS, cols):
            with col:
                st.multiselect(f"{t}: marked out", OPTIONS[t], key=KEYS[t],
                               format_func=lambda p: names.get(int(p), str(p)),
                               on_change=_apply_marks, placeholder="Nobody marked out")

# ── Tables ───────────────────────────────────────────────────────────────────
try:
    MED = store.table_medians(served.frame)
except Exception as e:  # noqa: BLE001 - minutes still show; the medians read the sentinel
    MED = None
    st.markdown(store.render_notice(
        "<b>The stat medians could not be drawn just now.</b> Minutes are shown; the other "
        f"columns show the sentinel. ({esc(type(e).__name__)})"), unsafe_allow_html=True)

# After tip: the box score so far, or final (cdn, 120 s cache).
ACT = None
if state in ("live", "final"):
    try:
        _box = live.box(gid)
        ACT = store.actuals_from(_box.value)
        if _box.stale:
            st.markdown(store.render_notice(
                f"<b>Showing the box score from {_box.age_min:.0f} minutes ago.</b> The live box "
                "score did not answer just now."), unsafe_allow_html=True)
    except RuntimeError:
        st.markdown(store.render_notice(
            "<b>The box score is unavailable right now.</b> Actual lines appear here once the "
            "NBA's live box score answers."), unsafe_allow_html=True)

board = None
if TOKEN:
    try:
        params = store.sim_params()
    except Exception:  # noqa: BLE001 - date_board explains a board it cannot build
        params = None
    board = props_ui.date_board(TOKEN, date, served.frame, params, ROSTER)
by_person = {}
if board is not None:
    mine = [x[x["game_id"].astype(str) == gid] if len(x) else x
            for x in (board.shown, board.hidden, board.thin, board.void)]
    by_person = props_ui.rows_by_person(mine[0])
    if board.meta.get("error"):
        st.markdown(store.render_notice(
            "<b>The comparison with your lines could not be built just now.</b> Your lines are "
            "still saved; try reloading the page."), unsafe_allow_html=True)
    elif sum(len(x) for x in mine):
        when = props_ui.paste_time_text(board.meta)
        st.markdown(
            f'<div class="bv-note"><b>Your PrizePicks lines on this game:</b> '
            f"{esc(props_ui.count_header(*mine))}.{' ' + esc(when) if when else ''} A teal "
            "count beside a name is how many lines he has: open him under his team's table "
            "to compare each line with his median.</div>",
            unsafe_allow_html=True)

positions = (dict(zip(ROSTER["person_id"].astype("int64"), ROSTER["position"].astype(str)))
             if ROSTER is not None and "position" in ROSTER.columns else {})
STARTERS = store.projected_starters(served.frame, FEATS)
stamp = rows["computed_at"].dropna()
asof = ""
if gid in info.frozen:
    asof = store.frozen_caption(rows, date)
elif len(stamp):
    t = to_et(pd.Timestamp(stamp.max()).to_pydatetime())
    asof = f"Projected {fmt_tip(t)}"
for team in TEAMS:
    trows = rows[rows["team"] == team]
    view = store.team_view(trows, MED, positions, STARTERS)
    view = view.assign(n_lines=[len(by_person.get(int(p), ())) for p in view["person_id"]])
    st.markdown(store.render_projection_table(team, view, asof_note=asof,
                                              team_note=store.team_report_label(trows) or "",
                                              actuals=ACT),
                unsafe_allow_html=True)
    for pid, name in zip(view["person_id"], view["name"]):
        recs = by_person.get(int(pid))
        if recs:
            with st.expander(f"{props_ui.md_escape(name)}: {len(recs)} "
                             f"line{'s' if len(recs) != 1 else ''}"):
                props_ui.render_player_lines_block(store.line_actuals(recs, ACT))

st.markdown(
    '<div class="bv-note">MIN is the projected average for a player who plays. The other '
    "columns are medians if he plays: half of his outcomes in this spot land above the number "
    "and half below, so a column of medians does not add up to a team total. FS is PrizePicks' "
    "Fantasy Score formula (PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV) on the numbers "
    "shown in its row, and a Fantasy Score line is judged against it. P(play) appears "
    "only for a player on the injury report or below 90% to play. A dot marks a projected "
    "starter: the five likely players who have started most this season (projected minutes "
    "decide before anyone has started).</div>",
    unsafe_allow_html=True)

# DiamondValue's biggest-gaps strip for this game, at the bottom.
if board is not None:
    props_ui.render_discrepancies(mine[0], f"{away} at {home}")
elif ROSTER is not None and state == "upcoming":
    # DiamondValue's empty-board prompt, while a paste can still land here.
    st.markdown(f'<div class="bv-note">{props_ui.NO_LINES_GAME}</div>', unsafe_allow_html=True)

render_footer(FOOT_KEEP)
