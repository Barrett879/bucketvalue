"""One game: header, context, and both teams' players.

URL: /Game?date=YYYY-MM-DD&away=BOS&home=DET. Team params go through
teams.canonical, which returns None for anything unknown (never a pass-through
into HTML), and every rendered string goes through util.esc.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import live, store, teams  # noqa: E402
from bucketlib.theme import SITE_NAME, render_footer, render_nav, render_page_chrome  # noqa: E402
from bucketlib.util import esc  # noqa: E402

st.set_page_config(page_title=f"Game | {SITE_NAME}", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

qp = st.query_params
away, home = teams.canonical(qp.get("away")), teams.canonical(qp.get("home"))
try:
    date = dt.date.fromisoformat(str(qp.get("date")))
except ValueError:
    date = None
keep = {"theme": "dark" if st.session_state.get("theme_dark") else "light"}
render_nav("Game", keep={"date": date.isoformat() if date else ""})
back = store.page_url("/", keep, date=date.isoformat() if date else None)

if not (away and home and date):
    st.markdown(store.render_notice(
        "<b>That game link is incomplete.</b> It needs a date and two NBA teams, "
        "for example <code>?date=2026-10-20&away=BOS&home=DET</code>."), unsafe_allow_html=True)
    st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
                unsafe_allow_html=True)
    render_footer()
    st.stop()

try:
    sched = live.schedule().value
except RuntimeError as e:
    st.markdown(store.render_notice(f"<b>The schedule feed is unreachable.</b> {esc(e)}"),
                unsafe_allow_html=True)
    render_footer()
    st.stop()

g = sched[(sched["game_date"] == date) & (sched["away"] == away) & (sched["home"] == home)]
if g.empty:
    st.markdown(store.render_notice(
        f"<b>No {esc(away)} at {esc(home)} game on {esc(date.strftime('%B %-d, %Y'))}.</b> "
        "The schedule has no such game; the date or the home team may be swapped."),
        unsafe_allow_html=True)
    st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
                unsafe_allow_html=True)
    render_footer()
    st.stop()
row = g.iloc[0]
state = live.game_state(row)

st.markdown(f'<a class="bv-back" href="{esc(back)}" target="_self">Back to the slate</a>',
            unsafe_allow_html=True)
st.markdown(store.render_matchup_header(row, state), unsafe_allow_html=True)
odds = live.odds_on(date).value.get((away, home))
if odds and pd.notna(odds.get("home_line")):
    hl = float(odds["home_line"])
    spread_txt = "PK" if hl == 0 else f"{home if hl < 0 else away} {-abs(hl):.1f}"
    if pd.notna(odds.get("total")):
        imp_h = float(odds["total"]) / 2 - hl / 2
        imp_a = float(odds["total"]) / 2 + hl / 2
        implied = f"{esc(away)} {imp_a:.1f}<span class='sep'> / </span>{esc(home)} {imp_h:.1f}"
    else:
        implied = "Not posted"
    total_txt = f"{float(odds['total']):.1f}" if pd.notna(odds.get("total")) else "Not posted"
else:
    spread_txt, total_txt, implied = "Not posted", "Not posted", "Not posted"
st.markdown(store.render_facts([
    store.fact("Spread", esc(spread_txt), (odds or {}).get("provider") or ""),
    store.fact("Total", esc(total_txt)),
    store.fact("Implied points", implied),
    store.fact("National TV", esc(row.get("national_tv") or "None")),
    store.fact("Game", esc({"RS": "Regular season", "PRE": "Preseason", "PI": "Play-in",
                            "PO": "Playoffs", "CUP_FINAL": "NBA Cup final"}.get(row["game_type"], row["game_type"]))),
]), unsafe_allow_html=True)

try:
    rost = live.roster()
except RuntimeError as e:
    rost = None
    st.markdown(store.render_notice(f"<b>Rosters are unavailable.</b> {esc(e)}"), unsafe_allow_html=True)

if rost is not None:
    st.markdown(store.render_notice(
        "<b>Rosters and availability only, for now.</b> Projected stat lines arrive with "
        "the models; every number column shows the sentinel until then."), unsafe_allow_html=True)
    asof = f"Rosters as of {pd.Timestamp(rost.fetched_at, unit='s', tz='UTC').tz_convert('America/New_York').strftime('%b %-d, %-I:%M %p')} ET" if rost.fetched_at else ""
    # Stacked, not side by side: ten stat columns per team need the full width
    # (side by side, the headers wrapped letter by letter and names truncated).
    for team in (away, home):
        p = rost.value[rost.value["team"] == team].copy()
        p["status"] = p["two_way"].map(lambda tw: "Two-way" if tw else "")
        p["out"] = False
        p["group"] = p["two_way"].map(lambda tw: "Two-way contracts" if tw else "")
        p = p.sort_values(["two_way", "last_name"])
        st.markdown(store.render_team_table(team, p, asof_note=asof), unsafe_allow_html=True)

render_footer()
