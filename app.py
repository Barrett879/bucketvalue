"""BucketValue home: the slate for one ET date.

Pick a date, see every game grouped by tip time, open one for its page. The
schedule and rosters come from cdn.nba.com through bucketlib.live (short budget,
circuit breaker, stale-beats-empty with a visible notice).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import live, store  # noqa: E402
from bucketlib.theme import SITE_NAME, render_footer, render_nav, render_page_chrome  # noqa: E402
from bucketlib.util import esc, today_et  # noqa: E402

st.set_page_config(page_title=SITE_NAME, page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

# ── Schedule ─────────────────────────────────────────────────────────────────
try:
    SCHED_LIVE = live.schedule()
except RuntimeError as e:
    render_nav("Home")
    st.markdown(store.render_notice(
        "<b>The NBA schedule feed is unreachable and nothing is cached.</b> "
        f"{SITE_NAME} cannot draw a slate without it. This is a data-source "
        f"outage, not an empty day. Details: {esc(e)}"), unsafe_allow_html=True)
    render_footer()
    st.stop()
SCHED = SCHED_LIVE.value
SEASON_DATES = sorted(SCHED["game_date"].unique())

# ── View date: seeded from ?date= only when the URL CHANGES ──────────────────
# A keyed widget ignores its default after the first render, and a sync that
# fires whenever URL and widget DISAGREE throws away the user's pick on the
# rerun before the URL catches up. So the last reconciled URL value is kept in
# its own key and the sync fires only when the URL itself changed.
def _parse_date(raw):
    try:
        return dt.date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


url_date = st.query_params.get("date")
if url_date != st.session_state.get("_url_date_seen"):
    parsed = _parse_date(url_date)
    st.session_state["view_date"] = parsed or st.session_state.get("view_date") or live.default_view_date()
    st.session_state["_url_date_seen"] = url_date
st.session_state.setdefault("view_date", live.default_view_date())


def _step(delta: int):
    st.session_state["view_date"] = st.session_state["view_date"] + dt.timedelta(days=delta)


def _jump(d: dt.date):
    st.session_state["view_date"] = d


view = st.session_state["view_date"]
if st.query_params.get("date") != view.isoformat():
    st.query_params["date"] = view.isoformat()
    st.session_state["_url_date_seen"] = view.isoformat()

keep = {"theme": "dark" if st.session_state.get("theme_dark") else "light"}
render_nav("Home", keep={"date": view.isoformat()})

# ── Date bar ─────────────────────────────────────────────────────────────────
today = today_et()
with st.container(key="bv_datebar"):
    c_prev, c_next, c_date, c_y, c_t, c_tm, _ = st.columns([0.05, 0.05, 0.17, 0.11, 0.1, 0.12, 0.4])
    with c_prev:
        st.button("‹", key="bv_day_prev", on_click=_step, args=(-1,), help="Previous day")
    with c_next:
        st.button("›", key="bv_day_next", on_click=_step, args=(1,), help="Next day")
    with c_date:
        st.date_input("Date", key="view_date", format="YYYY-MM-DD")
    with c_y:
        st.button("Yesterday", key="bv_day_y", on_click=_jump, args=(today - dt.timedelta(days=1),))
    with c_t:
        st.button("Today", key="bv_day_today", on_click=_jump, args=(today,), disabled=(view == today))
    with c_tm:
        st.button("Tomorrow", key="bv_day_tm", on_click=_jump, args=(today + dt.timedelta(days=1),))
st.markdown('<div class="bv-bar-rule"></div>', unsafe_allow_html=True)

if SCHED_LIVE.stale:
    st.markdown(store.render_notice(
        f"<b>Showing a saved schedule from {SCHED_LIVE.age_min:.0f} minutes ago.</b> "
        "The live NBA feed did not answer just now, so scores and statuses may lag."),
        unsafe_allow_html=True)

# ── Slate ────────────────────────────────────────────────────────────────────
games = live.games_on(SCHED, view)
nba = games[games["game_type"].isin(["RS", "PI", "PO", "CUP_FINAL"])] if len(games) else games
pre = games[games["game_type"] == "PRE"] if len(games) else games

_odds_for_mast = live.odds_on(view).value if len(nba) else {}
st.markdown(store.render_masthead(view, nba, len(_odds_for_mast)), unsafe_allow_html=True)

if len(nba):
    states = {r["game_id"]: live.game_state(r) for _, r in nba.iterrows()}
    odds = live.odds_on(view)
    by_game = {r["game_id"]: odds.value.get((r["away"], r["home"])) for _, r in nba.iterrows()}
    st.markdown(store.render_slate(nba, view, keep, odds_by_game=by_game, states=states),
                unsafe_allow_html=True)
    if odds.stale and odds.source == "none":
        st.markdown(store.render_notice(
            "<b>Game lines are unavailable right now.</b> The ESPN feed did not answer, "
            "so cards show no spread or total. This is a feed outage, not missing games."),
            unsafe_allow_html=True)
elif len(pre):
    st.markdown(store.render_notice(
        f"<b>Preseason: {len(pre)} game{'s' if len(pre) != 1 else ''}, no projections.</b> "
        "Preseason minutes are not regular-season minutes (2025-26 starters averaged "
        "half their usual minutes in this preseason), so the site does not project them. "
        "The regular season opens October 20."), unsafe_allow_html=True)
    states = {r["game_id"]: live.game_state(r) for _, r in pre.iterrows()}
    st.markdown(store.render_slate(pre, view, keep, states=states), unsafe_allow_html=True)
else:
    upcoming = [d for d in SEASON_DATES if d > view]
    nxt = (f" The next slate is <b>{upcoming[0].strftime('%A, %B %-d')}</b>."
           if upcoming else "")
    st.markdown(store.render_notice(f"<b>No NBA games on {view.strftime('%A, %B %-d')}.</b>{nxt}"),
                unsafe_allow_html=True)

render_footer()
