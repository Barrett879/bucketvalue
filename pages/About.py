"""What BucketValue is, where its numbers come from, and what they are not.

The trust table is GENERATED from data/stat_reliability.csv through the
board's own functions (store.trust_table -> props.reliability_for), so the tier
a stat shows here is the tier its rows show on the board. No accuracy record
appears here: that is local only.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Before any bucketlib import: load the code on disk, never a stale package
# left in this process by an earlier deploy (freshen.py).
sys.modules.pop("freshen", None)
import freshen  # noqa: E402

freshen.ensure()

from bucketlib import dist, models, pipeline, props, props_ui, store  # noqa: E402
from bucketlib.theme import SITE_NAME, render_footer, render_nav, render_page_chrome  # noqa: E402
from bucketlib.util import esc  # noqa: E402

st.set_page_config(page_title=f"About | {SITE_NAME}", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()


def _date_param() -> str:
    """The ?date= the visitor came with, validated (the nav writes it into
    hrefs unescaped), so Home returns to the slate he left."""
    try:
        return dt.date.fromisoformat(str(st.query_params.get("date"))).isoformat()
    except ValueError:
        return ""


# The nav and the footer carry the board and the mark-outs like every other
# page's (a link that drops them loses the visitor's paste on the way back to
# the slate).
KEEP = {"date": _date_param(),
        **store.view_params(props_ui.board_token(), store.parse_out(st.query_params.get("out")))}
render_nav("About", keep=KEEP)

# Read from the code, so the page cannot drift from what is served.
FIRST, LAST = models.TRAIN_SEASONS[0], models.TRAIN_SEASONS[-1]
st.markdown(f"""
<div class="bv-masthead"><div class="bv-kicker"><span class="bv-pip"></span>About</div>
<div class="bv-mast-rule"></div><div class="bv-brand">{SITE_NAME}</div>
<div class="bv-deck">A modelled stat line for every NBA player in every game, built the same
way as its siblings DiamondValue (MLB) and GridironValue (NFL): models trained on the
{FIRST} to {LAST} seasons, with box-score history back to 2014-15, validated season by
season on games they never saw, and shown as the median outcome for a player who plays.</div></div>
""", unsafe_allow_html=True)

FREEZE_MIN = int(pipeline.FREEZE_LEAD.total_seconds() // 60)
st.markdown(f"""
<div class="bv-note"><b>Where the numbers come from.</b> Box scores, schedules, rosters and
participation from NBA.com. Game lines from ESPN (DraftKings). Availability from the NBA's
official injury report, read as of the time a projection is made, not as of tip-off.</div>
<div class="bv-note"><b>What they are not.</b> A forecast of what will happen, or betting
advice. The posted line already prices most of what any model knows, so a gap between the
two is information to weigh, not an edge to bet.</div>
<div class="bv-note"><b>How to read a game page.</b> MIN is the projected average for a player
who plays. Every other column is a median if he plays: half of his outcomes land above it and
half below. FS is the exception: PrizePicks' Fantasy Score formula (PTS + 1.2 REB + 1.5 AST +
3 STL + 3 BLK - TOV) on the numbers shown in its row, which is also the number a Fantasy Score
line is judged against. P(play), the chance he plays at all, appears only for a player on the injury
report or below 90%. Projections freeze {FREEZE_MIN} minutes before tip-off and never change
after that. A provisional projection (made for a later date, or before the latest box scores
were in) is shown as published but is never frozen.</div>
<div class="bv-note"><b>How it is built.</b> Three kinds of gradient-boosted model per player
and game: the chance he plays at all, his minutes if he plays, and a per-minute rate for each
stat, so a stat's projection is minutes times rate. Each is trained only on what was known
before the game: earlier box scores, the game's spread and total, and the injury report as
filed at the time. Each team's projected minutes are then pulled part of the way toward the
{models.TEAM_MINUTES:g} an average game holds (240 plus overtime), moving bench minutes more
than starters' and never a player listed Out. Last, every player's game is simulated
{dist.N_DRAWS:,} times, with minutes errors measured by role and shots drawn so the points
always add up, and the medians shown come from those simulations, centred on the model's own
numbers.</div>
<div class="bv-note"><b>What version 1 does not do.</b> It projects regular-season games only:
no preseason, play-in, playoff or NBA Cup final. Nothing changes after the freeze, so a
scratch announced after the last injury report is not in the numbers until you mark him out
on the game page. The board does not price period lines (halves, quarters, first minutes),
Dunks, double-doubles or two-player squares, and the model does not price PrizePicks' Reboot
rule (a More pick voided when a player leaves in the first half and does not return).</div>
""", unsafe_allow_html=True)

# ── The trust table ──────────────────────────────────────────────────────────
st.markdown('<div class="bv-window"><span class="lab">How far to trust each stat</span>'
            '<span class="meta">Generated from the measured reliability table</span></div>',
            unsafe_allow_html=True)
st.markdown(store.render_trust_table(store.trust_table()), unsafe_allow_html=True)
notes = store.trust_caveats() + [
    "A combination line (Pts+Rebs+Asts, Blks+Stls, Fantasy Score and the rest) inherits the "
    "tier of its weakest component, so the board never trusts a combo more than its least "
    "reliable part. Its own measurement is shown beside it.",
    "The tier is the order the board ranks rows in: a strong stat's lean outranks a weak "
    "stat's, whatever the size of either gap.",
    "Biggest discrepancies, under the board, sorts the same shown lines by the size of the gap "
    "instead: standard lines only, in each stat's own units, with a "
    "stat-family filter for comparing like with like. It is a display, not a ranking of picks: "
    "a large gap is as often the model's miss as the line's.",
]
st.markdown("".join(f'<div class="bv-note">{esc(n)}</div>' for n in notes), unsafe_allow_html=True)

# ── Break-even and the board's ceiling ───────────────────────────────────────
be2 = store.breakeven(2)
st.markdown(f"""
<div class="bv-note"><b>What it takes to break even.</b> A two-pick Power Play pays
{store.POWER_PLAY_2_PAYOUT:g} times the entry, so the pair has to hit at least one time in
three: each pick needs about {be2:.1f}% just to break even. Across PrizePicks play sizes the
per-pick break-even runs from {props.BREAKEVEN_LO:.1f}% to {props.BREAKEVEN_HI:.1f}%. An edge
of a few points over the more common side is a long way from that.</div>
<div class="bv-note"><b>The board is at its ceiling.</b> A posted line is set by people who see
the same box scores, injury reports and lines the models see, and it moves with the money. It
already prices most of what a model knows, so the comparison on this site shows where the model
disagrees with the line, not an edge over it. Read a large gap as a question to look into.</div>
""", unsafe_allow_html=True)

render_footer(KEEP)
