"""What BucketValue is, where its numbers come from, and what they are not."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib.theme import SITE_NAME, render_footer, render_nav, render_page_chrome  # noqa: E402

st.set_page_config(page_title=f"About | {SITE_NAME}", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
render_nav("About")

st.markdown(f"""
<div class="bv-masthead"><div class="bv-kicker"><span class="bv-pip"></span>About</div>
<div class="bv-mast-rule"></div><div class="bv-brand">{SITE_NAME}</div>
<div class="bv-deck">A modelled stat line for every NBA player in every game, built the same
way as its siblings DiamondValue (MLB) and GridironValue (NFL): models trained on twelve
seasons of box scores, validated season by season on games they never saw, and shown as
the median outcome for a player who plays.</div></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bv-note"><b>Where the numbers come from.</b> Box scores, schedules, rosters and
participation from NBA.com. Game lines from ESPN (DraftKings). Availability from the NBA's
official injury report, read as of the time a projection is made, not as of tip-off.</div>
<div class="bv-note"><b>What they are not.</b> A forecast of what will happen, or betting
advice. The posted line already prices most of what any model knows, so a gap between the
two is information to weigh, not an edge to bet.</div>
""", unsafe_allow_html=True)

render_footer()
