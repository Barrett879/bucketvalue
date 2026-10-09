"""Theme tokens, page chrome, nav, and footer for BucketValue.

The visual system is HoopsValue's by way of DiamondValue and GridironValue: the
same surface, text-ramp and accent tokens in dark and light, the same Space
Grotesk + Manrope self-hosted variable webfonts (one @font-face per family), the
fixed pill nav with the red active state, and the pinned theme toggle. Ported
from gridlib/theme.py with the class prefix `bv-`; the football components
(week bar, broadcast windows) became a date bar and tip-time groups.

SITE_NAME is the ONE place the product name lives. util.SENTINEL is the only em
dash permitted in a string literal; it renders in cells for missing values.
"""
from __future__ import annotations

from .util import today_et

import streamlit as st

# ── The one-line rename ──────────────────────────────────────────────────────
SITE_NAME = "BucketValue"
SITE_TAGLINE = "NBA stat lines, modelled game by game"

# The single permitted em dash lives in util.SENTINEL (re-exported here).
from .util import SENTINEL  # noqa: E402,F401

# Ship light by default (config.toml base="light" matches); dark is opt-in via
# the pinned toggle, persisted across page navigations via ?theme=.
THEME_DEFAULT_DARK = False

# Nav pages: (label, url). Home is rendered separately.
# Only pages that EXIST IN GIT. A nav link to a page with no file renders the
# app shell and then a client-side "page not found", which reads as a broken
# site. Accuracy is LOCAL ONLY (Barrett, 2026-10-08): pages/Accuracy.py is
# gitignored, so it is never listed here; tests/test_nav.py enforces both.
_NAV_PAGES = [
    ("About", "/About"),
]

# ── Tokens: HoopsValue's exact palette ───────────────────────────────────────
THEME_BASE_CSS = """
<style>
    :root {
        /* surfaces */
        --app-bg:      #0a0a14;
        --bg-base:     #0a0a14;
        --bg-nav:      #0a0a0a;
        --panel:       rgba(20, 20, 42, 0.55);
        --panel-solid: #15171d;
        --panel-2:     #1a1a2e;
        --panel-hover: rgba(30, 30, 56, 0.85);
        --panel-line:  rgba(80, 80, 110, 0.35);
        --hairline:    rgba(255, 255, 255, 0.08);
        --hairline-soft: rgba(255, 255, 255, 0.04);
        --nav-border:  #222;
        --nav-divider: #333;
        /* tinted surfaces */
        --tint-good:   #1a2e1a;
        --tint-bad:    #2e1a1a;
        --tint-even:   #1a1a2e;
        /* text ramp */
        --fg-1: #ffffff;
        --fg-2: #cdcdd5;
        --fg-3: #aaaaaa;
        --fg-4: #8a8a93;
        --fg-5: #777777;
        --fg-6: #666666;
        /* brand accents */
        --accent-red:  #e63946;
        --accent-teal: #16d4c1;
        --value-good:  #2ecc71;
        --value-bad:   #e74c3c;
        --gold:        #f1c40f;
        --blue:        #3498db;
        --orange:      #f39c12;
        --purple:      #9b59b6;
        --sky:         #7ec8e8;
        --amber:       #f0b35b;
        /* elevation + table polish */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.35);
        --shadow-hover: 0 10px 30px -8px rgba(0, 0, 0, 0.55);
        --rail-lift: 22%;   /* lifts dark team navies off the near-black bg */
        --row-tint: rgba(255, 255, 255, 0.025);
        --bar-tint: rgba(22, 212, 193, 0.16);
    }
    html, body, .stApp { background: var(--app-bg) !important; }
    .stApp, body { color: var(--fg-2); }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 { color: var(--fg-1) !important; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label,
    [data-baseweb="form-control-label"] { color: var(--fg-2) !important; }
    /* ── Widget surfaces ─────────────────────────────────────────────────
       STREAMLIT 1.59 REPLACED BASEWEB WITH REACT ARIA. On 1.59.1 there are
       ZERO `data-baseweb` elements in the DOM, so every `data-baseweb`
       selector (the ones DiamondValue and HoopsValue still use) silently
       matches nothing and selectboxes render in Streamlit's own config.toml
       theme. In dark mode that means a white dropdown in a dark app.

       The selectbox is now `.react-aria-ComboBox` wrapping a `[role="group"]`,
       and its dropdown is PORTALED TO document.body as
       `[data-testid="stSelectboxVirtualDropdown"]` with virtualized
       `[role="option"]` rows. Tokens live on :root, so the portal inherits
       them fine.

       Selectors are anchored on ARIA roles and Streamlit testids, never on the
       emotion hash classes (st-emotion-cache-*), which change between builds.
       The data-baseweb rules are kept below as a fallback so this styling
       survives a downgrade to a pre-1.59 Streamlit. */
    .react-aria-ComboBox [role="group"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-1) !important;
    }
    .react-aria-ComboBox input {
        background: transparent !important;
        color: var(--fg-1) !important;
    }
    .react-aria-ComboBox [role="group"]:focus-within {
        border-color: var(--accent-teal) !important;
    }
    .react-aria-ComboBox button { background: transparent !important; }
    .react-aria-ComboBox svg,
    div[data-baseweb="select"] svg { fill: var(--fg-3) !important; color: var(--fg-3) !important; }

    [data-testid="stSelectboxVirtualDropdown"],
    ul[data-baseweb="menu"], div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [role="listbox"] {
        background: var(--panel-solid) !important;
        border: 1px solid var(--panel-line) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] *,
    ul[data-baseweb="menu"] li { color: var(--fg-2) !important; }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
    [data-baseweb="popover"] [role="option"]:hover,
    ul[data-baseweb="menu"] li:hover { background: var(--panel-hover) !important; }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] * {
        background: var(--bar-tint) !important;
        color: var(--fg-1) !important;
        font-weight: 700 !important;
    }

    .react-aria-TextField input,
    [data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-baseweb="input"] {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] input { color: var(--fg-1) !important; }
    [data-testid="stExpander"] details {
        background: var(--panel) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stButton"] button, .stButton button {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-2) !important;
    }
    [data-testid="stButton"] button:hover, .stButton button:hover {
        background: var(--panel-hover) !important;
        border-color: var(--fg-5) !important;
        color: var(--fg-1) !important;
    }
</style>
"""

THEME_LIGHT_CSS = """
<style>
    :root {
        --app-bg:      linear-gradient(180deg, #fbfcfd 0%, #eef1f4 100%);
        --bg-base:     #f4f6f8;
        --bg-nav:      #ffffff;
        --panel:       #ffffff;
        --panel-solid: #ffffff;
        --panel-2:     #eef1f4;
        --panel-hover: #f1f3f6;
        --panel-line:  #e3e6eb;
        --hairline:    rgba(20, 22, 40, 0.10);
        --hairline-soft: rgba(20, 22, 40, 0.05);
        --nav-border:  #e3e6eb;
        --nav-divider: #c9ccd3;
        --tint-good:   #eafaf1;
        --tint-bad:    #fdeceb;
        --tint-even:   #eef3f8;
        --fg-1: #14142a;
        --fg-2: #3a3d48;
        --fg-3: #585c68;
        --fg-4: #71757f;
        --fg-5: #9aa0ab;
        --fg-6: #b3b8c2;
        --accent-teal: #0fae9d;
        --value-good:  #16a34a;
        --value-bad:   #dc3a2c;
        --gold:        #9a6a00;
        --amber:       #a8730a;
        --orange:      #b45f06;
        --purple:      #7d3fa8;
        --blue:        #2471a3;
        --sky:         #146c94;
        --shadow-card: 0 1px 2px rgba(20,22,40,.05);
        --shadow-hover: 0 8px 24px rgba(20,22,40,.12);
        --rail-lift: 0%;   /* true team hue on white */
        --row-tint: rgba(20, 22, 40, 0.028);
        --bar-tint: rgba(15, 174, 157, 0.15);
    }
</style>
"""

# ── Fonts + shared chrome ────────────────────────────────────────────────────
COMMON_CSS = """
<style>
    /* One declaration per family. Both files are VARIABLE fonts, so a weight
       range covers every weight the site uses from a single download. */
    @font-face{font-family:'Space Grotesk';font-style:normal;font-weight:300 700;
      font-display:swap;src:url('/app/static/fonts/space-grotesk-var.woff2') format('woff2');}
    @font-face{font-family:'Manrope';font-style:normal;font-weight:200 800;
      font-display:swap;src:url('/app/static/fonts/manrope-var.woff2') format('woff2');}

    html, body, .stApp, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"], button, input {
        font-family: 'Manrope', -apple-system, sans-serif;
    }
    [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3, .bv-brand, .bv-slate-away, .bv-slate-home {
        font-family: 'Space Grotesk', 'Manrope', sans-serif;
    }

    /* Clear the fixed nav bar */
    .block-container { padding-top: 3.6rem; max-width: 1360px;
        padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
    /* Collapse the empty spacer rows Streamlit leaves around injected <style>
       blocks (the CSS still applies from a display:none subtree). */
    .stElementContainer:has([data-testid="stMarkdownContainer"] > style:only-child) {
        display: none !important;
    }
    #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stSidebarNav"], [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Fixed top nav bar: the pill nav, red active state */
    .top-nav {
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        display: flex; align-items: center; gap: 0.25rem;
        padding: 0 1.5rem; padding-right: 3.5rem; height: 3rem;
        background: var(--bg-nav); border-bottom: 1px solid var(--nav-border);
        flex-wrap: nowrap;
    }
    .top-nav a {
        text-decoration: none; padding: 0.3rem 0.85rem; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; color: var(--fg-3);
        border: 1px solid transparent; transition: all 0.15s; white-space: nowrap;
    }
    .top-nav a:hover { border-color: var(--accent-red); color: var(--fg-1); text-decoration: none; }
    .top-nav a.active { background: var(--accent-red); border-color: var(--accent-red); color: #fff; }
    .top-nav .home-link {
        color: var(--fg-6); font-size: 0.82rem; font-weight: 500;
        padding: 0.3rem 0.7rem; margin-right: 0.25rem; border: none;
    }
    .top-nav .home-link:hover { color: var(--fg-1); border: none; }
    .top-nav .divider { color: var(--nav-divider); font-size: 0.75rem;
        margin: 0 0.1rem; user-select: none; }
    @media (max-width: 760px) {
        .top-nav { overflow-x: auto; overflow-y: hidden; scrollbar-width: none; padding-right: 4rem; }
        .top-nav::-webkit-scrollbar { display: none; }
        .top-nav::after { content: ""; flex: 0 0 8rem; }
    }

    /* Header action cluster: "Update lines", pinned into the nav bar just left
       of the theme toggle, same pill look. The keyed container IS the
       stVerticalBlock, so flex-direction goes straight on it; a
       `> [data-testid="stVerticalBlock"]` child selector matches nothing.
       On narrow screens it falls back into the page flow so it cannot overlap
       the nav links. */
    .st-key-bv_nav_actions {
        position: fixed;
        top: 1.5rem; right: 5.4rem;
        transform: translateY(-50%);
        z-index: 10000; width: auto !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.4rem !important;
    }
    .st-key-bv_nav_actions [data-testid="stElementContainer"],
    .st-key-bv_nav_actions [data-testid="stLayoutWrapper"] { width: auto !important; }
    .st-key-bv_nav_actions button {
        padding: 0.24rem 0.85rem !important; font-size: 0.82rem !important;
        font-weight: 600 !important; font-family: inherit !important;
        border-radius: 20px !important; background: transparent !important;
        border: 1px solid var(--panel-line) !important; color: var(--fg-2) !important;
        white-space: nowrap !important; min-height: 0 !important;
        line-height: 1.25 !important; transition: all 0.15s !important;
    }
    .st-key-bv_nav_actions button:hover {
        border-color: var(--accent-teal) !important;
        color: var(--accent-teal) !important; background: transparent !important;
    }
    .st-key-bv_nav_actions button svg { color: inherit !important; }
    @media (max-width: 900px) {
        .st-key-bv_nav_actions { position: static; transform: none;
                                 margin-top: 0.3rem; }
    }

    /* Pinned theme toggle (top-right, inside the nav bar row) */
    .st-key-bv_theme_toggle {
        position: fixed;
        top: 1.5rem; right: 0.9rem;      /* bar is 3rem tall: pin to its */
        transform: translateY(-50%);     /* midline, truly centered      */
        z-index: 10000; width: auto !important;
    }
    .st-key-bv_theme_toggle button {
        padding: 0.24rem 0.85rem !important; font-size: 0.82rem !important;
        font-weight: 600 !important; font-family: inherit !important;
        border-radius: 20px !important; background: transparent !important;
        border: 1px solid var(--panel-line) !important; color: var(--fg-2) !important;
        white-space: nowrap !important; min-height: 0 !important;
        line-height: 1.25 !important; transition: all 0.15s !important;
    }
    .st-key-bv_theme_toggle button:hover {
        border-color: var(--accent-teal) !important;
        color: var(--accent-teal) !important; background: transparent !important;
    }

    /* Brand + masthead */
    .bv-brand {
        font-size: clamp(2rem, 5vw, 2.7rem); font-weight: 700;
        letter-spacing: -0.03em; line-height: 1.0;
        color: var(--fg-1); margin: 0 0 0.1rem;
    }
    .bv-brand .accent { color: var(--accent-teal); }
    .bv-masthead { margin: 0.2rem 0 0.2rem; }
    .bv-mast-rule { position: relative; height: 1px; background: var(--panel-line);
        margin-bottom: 0.9rem; }
    .bv-mast-rule::before { content: ""; position: absolute; left: 0; top: -1px;
        width: 44px; height: 3px; background: var(--accent-teal); }
    .bv-mast-row { display: flex; justify-content: space-between;
        align-items: flex-end; gap: 1.25rem; flex-wrap: wrap; }
    .bv-kicker { display: flex; align-items: center;
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--fg-4); margin-bottom: 0.45rem; }
    /* The ball: a small circle, same visual weight as DiamondValue's rotated
       square and GridironValue's squashed football. */
    .bv-pip { width: 8px; height: 8px; background: var(--accent-teal);
        border-radius: 50%; margin-right: 0.5rem; flex: none; }
    .bv-mast-summary { display: flex; align-items: flex-end; margin-left: auto; }
    .bv-sum { padding: 0 0.95rem; text-align: right; }
    .bv-sum + .bv-sum { border-left: 1px solid var(--panel-line); }
    .bv-sum:last-child { padding-right: 0; }
    .bv-sum-num { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.2rem; color: var(--fg-1); font-variant-numeric: tabular-nums;
        line-height: 1.1; white-space: nowrap; }
    .bv-sum-lab { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-top: 0.2rem; }
    .bv-deck { color: var(--fg-3); font-size: 0.95rem; line-height: 1.5;
        max-width: 62ch; margin: 0.75rem 0 0; }
    @media (max-width: 640px) { .bv-mast-summary { margin: 0.8rem 0 0; }
        .bv-sum:first-child { padding-left: 0; } }

    /* Date bar (keyed container): chevrons + date picker + Yesterday/Today/Tomorrow */
    .st-key-bv_datebar [data-testid="stHorizontalBlock"] { align-items: flex-end; }
    .st-key-bv_datebar [data-testid="stWidgetLabel"] p {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem;
        font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--fg-5); }
    .st-key-bv_datebar [data-testid="stButton"] button {
        height: 2.35rem; font-size: 0.82rem; font-weight: 600;
        border-radius: 10px !important; }
    .st-key-bv_day_prev button, .st-key-bv_day_next button {
        padding: 0 0.5rem !important; font-size: 1.1rem !important;
        font-weight: 700 !important; line-height: 1 !important; }
    .st-key-bv_day_today button {
        border-color: var(--accent-teal) !important; color: var(--accent-teal) !important; }
    .bv-bar-rule { height: 1px; background: var(--panel-line); margin: 0.5rem 0 0.2rem; }

    /* Tip-time group header: "7:00 PM ET - 5 games" */
    .bv-window {
        display: flex; align-items: baseline; gap: 0.6rem;
        margin: 1.15rem 0 0.55rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--hairline);
    }
    .bv-window .lab {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--fg-3);
    }
    .bv-window .meta { font-size: 0.7rem; color: var(--fg-5);
        font-variant-numeric: tabular-nums; }
    .bv-window .lock { margin-left: auto; font-size: 0.66rem; color: var(--fg-5); }

    /* Slate grid: square cards, 4 across, team-color split stripe + duotone */
    .bv-slate-grid {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem; margin: 0.2rem 0 1.2rem;
    }
    @media (max-width: 1080px) { .bv-slate-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 760px)  { .bv-slate-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 460px)  { .bv-slate-grid { grid-template-columns: 1fr; } }
    a.bv-slate-card {
        position: relative; overflow: hidden;
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 9.5rem;
        padding: 1.15rem 0.6rem 0.75rem;
        background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; text-decoration: none; text-align: center;
        box-shadow: var(--shadow-card);
        transition: border-color .14s ease, transform .14s ease, box-shadow .14s ease;
    }
    /* Top split stripe: away | 1px seam | home. Plain fallback, lifted when color-mix. */
    a.bv-slate-card::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
        transition: height .14s ease;
        background: linear-gradient(90deg,
            var(--away, var(--fg-5)) 0 calc(50% - .5px),
            var(--panel-line) calc(50% - .5px) calc(50% + .5px),
            var(--home, var(--fg-5)) calc(50% + .5px) 100%);
    }
    @supports (color: color-mix(in srgb, red, white)) {
        a.bv-slate-card::before {
            background: linear-gradient(90deg,
                color-mix(in srgb, var(--away, var(--fg-5)), #fff var(--rail-lift)) 0 calc(50% - .5px),
                var(--panel-line) calc(50% - .5px) calc(50% + .5px),
                color-mix(in srgb, var(--home, var(--fg-5)), #fff var(--rail-lift)) calc(50% + .5px) 100%);
        }
        a.bv-slate-card {
            background: linear-gradient(150deg,
                color-mix(in srgb, var(--away, transparent) 6%, var(--panel)),
                color-mix(in srgb, var(--home, transparent) 6%, var(--panel)));
        }
    }
    a.bv-slate-card:hover {
        border-color: var(--accent-teal); transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }
    a.bv-slate-card:hover::before { height: 5px; }
    a.bv-slate-card:focus-visible { outline: 2px solid var(--accent-teal); outline-offset: 2px; }
    @media (prefers-reduced-motion: reduce) {
        a.bv-slate-card { transition: none; }
        a.bv-slate-card:hover { transform: none; }
    }
    .bv-slate-teams { display: flex; align-items: center; justify-content: center;
        gap: 0.55rem; }
    .bv-slate-away, .bv-slate-home {
        font-size: 1.45rem; font-weight: 700; color: var(--fg-1);
        line-height: 1.1; letter-spacing: -0.01em;
    }
    .bv-slate-at { color: var(--fg-5); font-weight: 600; font-size: 0.7rem; }
    .bv-slate-time {
        color: var(--fg-3); font-size: 0.72rem; font-weight: 600;
        margin-top: 0.55rem; letter-spacing: 0.01em; white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }
    /* The market readout: spread on the favorite, then the total. Informational
       context for the projection, never a wager prompt. */
    .bv-slate-mkt {
        display: inline-flex; align-items: baseline; gap: 0.45rem;
        margin-top: 0.4rem; font-size: 0.7rem; color: var(--fg-4);
        font-variant-numeric: tabular-nums; white-space: nowrap;
    }
    .bv-slate-mkt b { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        color: var(--fg-2); }
    .bv-slate-mkt .sep { color: var(--fg-6); }
    .bv-slate-final { margin-top: 0.4rem; font-size: 0.78rem; font-weight: 700;
        color: var(--fg-2); font-variant-numeric: tabular-nums; }
    .bv-slate-final .w { color: var(--fg-1); }
    /* Top-left micro-pill: the game's status word (Final, Live, or the day). */
    .bv-slate-tag {
        position: absolute; top: 7px; left: 7px; z-index: 1; pointer-events: none;
        padding: 0.1rem 0.4rem; border-radius: 999px;
        background: var(--hairline); border: 1px solid var(--panel-line);
        font-family: 'Space Grotesk', sans-serif; font-size: 0.52rem;
        font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
        color: var(--fg-5); line-height: 1.35; white-space: nowrap;
    }
    .bv-slate-tag.live { color: var(--value-bad); border-color: var(--value-bad); }
    .bv-slate-tag.final { color: var(--fg-4); }

    /* ── Game page ───────────────────────────────────────────────────────── */
    .bv-mh { margin: 0.2rem 0 1.1rem; }
    .bv-mh-tag {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--fg-4); margin-bottom: 0.5rem;
    }
    .bv-mh-row { display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap; }
    .bv-mh-team { display: flex; flex-direction: column; gap: 0.1rem; }
    .bv-mh-abbr {
        font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem;
        font-weight: 700; line-height: 1; letter-spacing: -0.02em;
        color: var(--c);
    }
    /* On white the true team hue reads; on near-black the dark navies need a
       lift, the same --rail-lift trick the slate cards use. */
    @supports (color: color-mix(in srgb, red, white)) {
        .bv-mh-abbr { color: color-mix(in srgb, var(--c), #fff var(--rail-lift)); }
    }
    .bv-mh-name { font-size: 0.76rem; color: var(--fg-4); font-weight: 600; }
    .bv-mh-at { color: var(--fg-5); font-size: 0.85rem; font-weight: 600; }
    .bv-mh-score {
        display: flex; align-items: baseline; gap: 0.5rem;
        font-family: 'Space Grotesk', sans-serif; font-size: 2rem;
        font-weight: 700; color: var(--fg-4); font-variant-numeric: tabular-nums;
    }
    .bv-mh-score .w { color: var(--fg-1); }
    .bv-mh-score .dash { color: var(--fg-6); font-size: 1.2rem; }
    .bv-mh-when {
        margin-top: 0.7rem; color: var(--fg-3); font-size: 0.85rem;
        font-variant-numeric: tabular-nums;
    }
    .bv-mh-when .sep { color: var(--fg-6); margin: 0 0.5rem; }

    /* Fact strip: the exogenous inputs to a game script */
    .bv-facts {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.1rem 0; background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; margin-bottom: 1.4rem;
    }
    .bv-fact { padding: 0.75rem 1rem; border-right: 1px solid var(--hairline); }
    .bv-fact:last-child { border-right: none; }
    .bv-fact-lab {
        font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-bottom: 0.28rem;
    }
    .bv-fact-val {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.98rem; color: var(--fg-1); font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }
    .bv-fact-val .sep { color: var(--fg-5); font-weight: 500; }
    .bv-fact-hint {
        font-size: 0.62rem; color: var(--fg-5); margin-top: 0.28rem;
        line-height: 1.35;
    }

    /* Depth charts, one panel per team */
    .bv-depth {
        background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; height: 100%;
    }
    .bv-depth-head {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.9rem; padding: 0.7rem 1rem; color: var(--on);
        background: var(--c); letter-spacing: -0.01em;
    }
    .bv-depth-grp { padding: 0.6rem 1rem 0.7rem; border-top: 1px solid var(--hairline); }
    .bv-depth-grp:first-of-type { border-top: none; }
    .bv-depth-pos {
        font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-bottom: 0.35rem;
    }
    .bv-depth-row {
        display: grid; grid-template-columns: 1.4rem 1fr; align-items: baseline;
        gap: 0.5rem; padding: 0.16rem 0; font-size: 0.86rem;
    }
    .bv-depth-row .rk {
        color: var(--fg-5); font-size: 0.7rem; font-weight: 700;
        font-variant-numeric: tabular-nums; text-align: right;
    }
    .bv-depth-row .nm { color: var(--fg-2); }
    .bv-depth-row:first-of-type .nm { color: var(--fg-1); font-weight: 600; }
    .bv-depth-empty { padding: 1rem; color: var(--fg-4); font-size: 0.85rem; }
    .bv-inj {
        display: inline-block; margin-left: 0.4rem; padding: 0.02rem 0.34rem;
        border-radius: 999px; font-size: 0.56rem; font-weight: 700;
        letter-spacing: 0.05em; text-transform: uppercase; vertical-align: middle;
    }
    .bv-inj-out { background: var(--tint-bad); color: var(--value-bad); }
    .bv-inj-doubtful { background: var(--tint-bad); color: var(--value-bad); }
    .bv-inj-questionable { background: var(--tint-even); color: var(--amber); }

    /* Per-game count of posted PrizePicks lines that map to a stat we project,
       summed across BOTH teams. Informational model-vs-market readout, never a
       wager prompt. A neutral top-LEFT micro-pill: top-right would read as a
       promotional notification badge. Absolutely positioned so it cannot shift
       the centred abbreviations, pointer-events:none so the card stays one
       click target, and shown only when the count is above zero. */
    .bv-slate-lines {
        position: absolute; top: 7px; left: 7px; z-index: 1;
        pointer-events: none;
        display: inline-flex; align-items: baseline; gap: 0.2rem;
        padding: 0.1rem 0.38rem; border-radius: 999px;
        background: var(--hairline); border: 1px solid var(--panel-line);
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1; white-space: nowrap; font-variant-numeric: tabular-nums;
        transition: color .14s ease, border-color .14s ease, background .14s ease;
    }
    .bv-slate-lines b { font-weight: 700; font-size: 0.74rem; color: var(--fg-1); }
    .bv-slate-lines .u {
        font-size: 0.5rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: var(--fg-5);
    }
    a.bv-slate-card:hover .bv-slate-lines { border-color: var(--accent-teal); }
    a.bv-slate-card:hover .bv-slate-lines b,
    a.bv-slate-card:hover .bv-slate-lines .u { color: var(--accent-teal); }
    @supports (color: color-mix(in srgb, red, white)) {
        a.bv-slate-card:hover .bv-slate-lines {
            border-color: color-mix(in srgb, var(--accent-teal) 45%, var(--panel-line));
            background: color-mix(in srgb, var(--accent-teal) 12%, var(--panel));
        }
    }
    @media (prefers-reduced-motion: reduce) { .bv-slate-lines { transition: none; } }

    /* The model-vs-market ledger */
    .bv-ledger { background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; margin: 0.2rem 0 0.6rem;
        font-variant-numeric: tabular-nums; }
    .bv-ledger-head, .bv-ledger-row {
        display: grid;
        grid-template-columns: minmax(9rem,1.6fr) minmax(7rem,1.2fr)
                               4.2rem 4.2rem 4.4rem 3.4rem;
        gap: 0.5rem; align-items: center; padding: 0.5rem 0.9rem;
    }
    .bv-ledger-head {
        font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--fg-4); background: var(--panel-2);
        border-bottom: 1px solid var(--panel-line);
    }
    .bv-ledger-row { border-bottom: 1px solid var(--hairline-soft);
        font-size: 0.85rem; }
    .bv-ledger-row:last-child { border-bottom: none; }
    .bv-ledger-row:nth-child(even) { background: var(--row-tint); }
    .bv-ledger .nm { color: var(--fg-1); font-weight: 600; display: flex;
        flex-direction: column; line-height: 1.25; overflow: hidden; }
    .bv-ledger .nm i { font-style: normal; font-size: 0.66rem;
        color: var(--fg-5); font-weight: 500; }
    .bv-ledger .st { color: var(--fg-3); font-size: 0.78rem; }
    .bv-ledger .num, .bv-ledger-head .num { text-align: right; }
    .bv-ledger .num { color: var(--fg-2); }
    .bv-ledger .num.line { color: var(--fg-4); }
    .bv-ledger .lean, .bv-ledger-head .lean { text-align: right;
        font-size: 0.72rem; font-weight: 700; }
    .bv-ledger .diff.over, .bv-ledger .lean.over { color: var(--value-good); }
    .bv-ledger .diff.under, .bv-ledger .lean.under { color: var(--value-bad); }

    /* Provenance badge: which numbers came from a model that passed its gate */
    .bv-src { display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.32rem; border-radius: 999px; font-size: 0.54rem;
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        vertical-align: middle; }
    .bv-src.base { background: var(--tint-bad); color: var(--value-bad); }
    .bv-src.low  { background: var(--tint-even); color: var(--amber); }
    .bv-src.comp { background: var(--hairline); color: var(--fg-4); }

    /* Demon and Goblin markers. These lines are deliberately shifted off the
       standard number, so a big gap against one is manufactured rather than an
       edge, and it has to be visible at a glance. Coloured to match how
       PrizePicks itself signals them: the Demon red, the Goblin green. Outlined
       rather than filled so they read as an annotation on the stat and not as a
       verdict on the row, which is what the Gap and Lean columns are for. */
    .bv-odds {
        display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.34rem; border-radius: 999px;
        font-size: 0.54rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; vertical-align: middle;
        border: 1px solid currentColor; background: transparent;
        cursor: help;
    }
    .bv-odds.demon  { color: var(--value-bad); }
    .bv-odds.goblin { color: var(--value-good); }

    /* Measured per-stat reliability. Filled rather than outlined, because
       unlike the Demon marker this IS a verdict on the row: it says how much
       the lean beside it is worth. A pass-attempts gap and a receiving-yards
       gap of the same size are opposite in sign as evidence. */
    .bv-tier {
        display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.34rem; border-radius: 999px;
        font-size: 0.52rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; vertical-align: middle;
        white-space: nowrap; cursor: help;
    }
    .bv-tier.t-strong { background: var(--tint-good); color: var(--value-good); }
    .bv-tier.t-mod    { background: var(--bar-tint);  color: var(--accent-teal); }
    .bv-tier.t-weak   { background: var(--hairline);  color: var(--fg-4); }
    .bv-tier.t-none   { background: var(--hairline);  color: var(--fg-5); }
    .bv-tier.t-neg    { background: var(--tint-bad);  color: var(--value-bad); }
    .bv-tier.t-unk    { background: transparent; color: var(--fg-6);
                        border: 1px solid var(--panel-line); }

    /* A single player's lines, opened under his own roster row */
    .bv-pl { margin: 0.25rem 0 0.5rem 1.9rem; border-left: 2px solid var(--panel-line);
        padding-left: 0.7rem; font-variant-numeric: tabular-nums; }
    .bv-pl-head, .bv-pl-row {
        display: grid; grid-template-columns: minmax(6rem,1fr) 3.6rem 3.6rem 3.8rem 3rem;
        gap: 0.4rem; align-items: center; padding: 0.16rem 0;
    }
    .bv-pl-head { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        font-size: 0.56rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--fg-5); }
    .bv-pl-row { font-size: 0.78rem; }
    .bv-pl .st { color: var(--fg-3); }
    .bv-pl .num { text-align: right; color: var(--fg-2); }
    .bv-pl .num.line { color: var(--fg-4); }
    .bv-pl .lean { text-align: right; font-weight: 700; font-size: 0.68rem; }
    .bv-pl .diff.over, .bv-pl .lean.over { color: var(--value-good); }
    .bv-pl .diff.under, .bv-pl .lean.under { color: var(--value-bad); }
    /* Depth rows that carry lines get a quiet marker so a reader can see which
       players are worth opening without expanding every one. */
    .bv-depth-row.has-lines .nm::after {
        content: ""; display: inline-block; width: 5px; height: 5px;
        border-radius: 50%; background: var(--accent-teal);
        margin-left: 0.4rem; vertical-align: middle;
    }


    /* ── NBA additions ───────────────────────────────────────────────────── */
    /* Tip-time group header on the slate: "7:00 PM ET  5 games" */
    .bv-tipgrp { display: flex; align-items: baseline; gap: 0.6rem;
        margin: 1.15rem 0 0.55rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--hairline); }
    .bv-tipgrp .lab { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--fg-3); }
    .bv-tipgrp .meta { font-size: 0.7rem; color: var(--fg-5); }
    /* Card extras: national TV, back-to-back chip, key absences */
    .bv-slate-sub { margin-top: 0.3rem; font-size: 0.66rem; color: var(--fg-5);
        text-align: center; line-height: 1.35; }
    .bv-chip { display: inline-block; padding: 0.02rem 0.36rem; border-radius: 999px;
        font-size: 0.56rem; font-weight: 700; letter-spacing: 0.05em;
        text-transform: uppercase; border: 1px solid var(--panel-line); color: var(--fg-4);
        margin: 0 0.12rem; }
    .bv-chip.warn { color: var(--amber); border-color: var(--amber); }
    .bv-out { color: var(--fg-4); }
    .bv-out b { color: var(--value-bad); font-weight: 700; }

    /* Player stat table: one team. Rounded, shadowed container; tabular
       numerals; MIN is the lead column; Out rows greyed and sunk. */
    .bv-pt { border: 1px solid var(--panel-line); border-radius: 12px;
        overflow: hidden; background: var(--panel); box-shadow: var(--shadow-card);
        margin: 0.4rem 0 1.2rem; font-variant-numeric: tabular-nums; }
    .bv-pt-cap { display: flex; align-items: center; gap: 0.55rem;
        padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--panel-line);
        background: linear-gradient(90deg,
            color-mix(in srgb, var(--team) 16%, transparent), transparent 70%); }
    .bv-pt-cap .ab { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1rem; color: var(--fg-1); }
    .bv-pt-cap .nm { font-size: 0.78rem; color: var(--fg-4); font-weight: 600; }
    .bv-pt-cap .asof { margin-left: auto; font-size: 0.66rem; color: var(--fg-5); }
    .bv-pt table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .bv-pt th { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        font-size: 0.6rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--fg-5); text-align: right; padding: 0.42rem 0.45rem;
        border-bottom: 1px solid var(--hairline); }
    .bv-pt th.l, .bv-pt td.l { text-align: left; }
    .bv-pt td { padding: 0.36rem 0.45rem; font-size: 0.82rem; color: var(--fg-2);
        text-align: right; border-bottom: 1px solid var(--hairline-soft);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bv-pt tr:nth-child(even) td { background: var(--row-tint); }
    .bv-pt tr:hover td { background: var(--panel-hover); }
    .bv-pt td.lead { color: var(--fg-1); font-weight: 700; }
    .bv-pt td.pl { color: var(--fg-1); font-weight: 600; }
    .bv-pt td.pl .pos { color: var(--fg-5); font-weight: 500; font-size: 0.68rem;
        margin-left: 0.3rem; }
    .bv-pt tr.out td { color: var(--fg-6); }
    .bv-pt tr.out td.pl { color: var(--fg-4); font-weight: 500; }
    .bv-pt .grp td { font-family: 'Space Grotesk', sans-serif; font-size: 0.58rem;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--fg-5);
        background: transparent !important; padding-top: 0.6rem; }
    img.bv-hs { width: 1.55rem; height: 1.55rem; border-radius: 50%;
        object-fit: cover; vertical-align: middle; margin-right: 0.45rem;
        background: var(--panel-2); }
    /* Injury status badge. Unknown is never drawn as healthy. */
    .bv-st { display: inline-block; min-width: 1.5rem; padding: 0.04rem 0.32rem;
        border-radius: 6px; font-size: 0.56rem; font-weight: 700;
        letter-spacing: 0.04em; text-transform: uppercase; text-align: center; }
    .bv-st.out { background: var(--tint-bad); color: var(--value-bad); }
    .bv-st.dbt { background: var(--tint-bad); color: var(--orange); }
    .bv-st.q   { background: var(--tint-even); color: var(--amber); }
    .bv-st.prob { background: var(--tint-good); color: var(--value-good); }
    .bv-st.gl  { background: var(--hairline); color: var(--fg-4); }
    .bv-st.tw  { border: 1px solid var(--panel-line); color: var(--fg-5); }

    /* Notices: a quiet panel for "no games" / "data not published yet" states.
       Never let an empty result render as silence; say why it is empty. */
    .bv-notice {
        background: var(--panel); border: 1px solid var(--panel-line);
        border-left: 3px solid var(--amber); border-radius: 10px;
        padding: 0.85rem 1.1rem; margin: 0.9rem 0;
        color: var(--fg-3); font-size: 0.88rem; line-height: 1.55;
    }
    .bv-notice b { color: var(--fg-1); font-weight: 700; }

    .bv-note { color: var(--fg-5); font-size: 0.78rem; line-height: 1.55;
        max-width: 72ch; margin-top: 0.4rem; }
    a.bv-back {
        display: inline-block; margin: 1.4rem 0 0.2rem;
        color: var(--fg-3); font-size: 0.84rem; font-weight: 600;
        text-decoration: none; padding: 0.35rem 0.9rem; border-radius: 20px;
        border: 1px solid var(--panel-line); transition: all 0.15s;
    }
    a.bv-back:hover { color: var(--accent-teal); border-color: var(--accent-teal); }

    /* Footer */
    .bv-footer { margin-top: 3rem; padding-top: 1.1rem; font-size: 0.76rem; }
    .bv-foot-disc { color: var(--fg-5); line-height: 1.6; max-width: 78ch; }
    .bv-foot-disc a { color: var(--fg-4); text-decoration: underline; }
    .bv-foot-disc a:hover { color: var(--fg-2); }
    .bv-foot-rule { height: 1px; background: var(--panel-line); margin: 0.9rem 0 0.7rem; }
    .bv-foot-bottom { display: flex; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem 1.1rem; color: var(--fg-5); }
    .bv-foot-bottom a { color: var(--fg-3); text-decoration: none; }
    .bv-foot-bottom a:hover { color: var(--fg-1); }
</style>
"""


def inject_theme() -> None:
    """Emit theme tokens (dark base + light override when active). Call once per
    page before any token-referencing CSS. Theme persists across navigations via
    ?theme=.
    """
    if "theme_dark" not in st.session_state:
        qp = st.query_params.get("theme")
        st.session_state["theme_dark"] = (
            (qp == "dark") if qp in ("dark", "light") else THEME_DEFAULT_DARK
        )
    st.markdown(THEME_BASE_CSS, unsafe_allow_html=True)
    if not st.session_state.get("theme_dark", THEME_DEFAULT_DARK):
        st.markdown(THEME_LIGHT_CSS, unsafe_allow_html=True)


def render_theme_toggle() -> bool:
    """Light/dark toggle backed by st.session_state['theme_dark']. Mirrors the
    choice into ?theme= so a full-reload navigation carries it. Returns dark.
    """
    st.session_state.setdefault("theme_dark", THEME_DEFAULT_DARK)

    def _flip():
        new_dark = not st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
        st.session_state["theme_dark"] = new_dark
        st.query_params["theme"] = "dark" if new_dark else "light"

    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    st.button("Light" if dark else "Dark", key="bv_theme_toggle_btn",
              on_click=_flip, help="Toggle light/dark")
    return st.session_state.get("theme_dark", THEME_DEFAULT_DARK)


def render_page_chrome() -> None:
    """One-call chrome: theme tokens + shared CSS. Call once after
    st.set_page_config on every page.
    """
    inject_theme()
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def render_nav(current: str, keep: dict[str, str] | None = None) -> None:
    """Fixed top nav bar plus the pinned theme toggle. `current` matches a label
    in _NAV_PAGES (or "Home").

    Links use target="_self", NOT "_top": Streamlit Community Cloud serves the
    app inside an iframe, and "_top" would navigate the outer wrapper (wrong
    origin) instead of the app frame, so the links appear dead. "_self"
    navigates the app's own frame, which works iframed or served directly.

    `keep` carries view state (season, week) into every destination, because
    each Streamlit page navigation is a fresh session and a bare href would
    reset it.
    """
    parts = [f"{k}={v}" for k, v in (keep or {}).items() if v not in (None, "")]
    parts.append("theme=" + ("dark" if st.session_state.get("theme_dark") else "light"))
    qs = "?" + "&".join(parts)

    home_cls = "active" if current == "Home" else ""
    links = (f'<a class="home-link {home_cls}" href="/{qs}" '
             f'target="_self">{SITE_NAME}</a>')
    links += '<span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        cls = "active" if label == current else ""
        links += (f'<a class="{cls}" href="{url}{qs}" target="_self">{label}</a>')
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)

    with st.container(key="bv_theme_toggle"):
        render_theme_toggle()


def render_footer() -> None:
    """Site-wide footer: sources and the informational-only disclaimer."""
    year = today_et().year
    html = (
        '<div class="bv-footer">'
        '<div class="bv-foot-disc">'
        "Box scores, schedules and rosters from NBA.com; game lines from ESPN "
        "(DraftKings); availability from the NBA's official injury report. "
        f"{SITE_NAME} is an informational analytics site, unaffiliated with the "
        "NBA, ESPN or PrizePicks. It is not a sportsbook, it accepts no wagers, "
        "and it offers no betting advice. Player numbers are modelled medians "
        "for a player who plays: half of his outcomes in that spot land above "
        "the number and half below."
        "</div>"
        '<div class="bv-foot-rule"></div>'
        '<div class="bv-foot-bottom">'
        f"<div>&copy; {year} {SITE_NAME}. Every number is a modelled outcome, not a forecast.</div>"
        '<div><a href="/About" target="_self">About</a></div>'
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def theme_fig(fig):
    """Make a Plotly figure follow the active theme (charts cannot read CSS
    vars). Call inline at the plot site.
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    axis = "#cdcdd5" if dark else "#3a3d48"
    grid = "rgba(255,255,255,0.08)" if dark else "rgba(20,22,40,0.10)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=axis, family="Manrope, sans-serif"),
        hoverlabel=dict(
            bgcolor="#1a1a2e" if dark else "#ffffff",
            font=dict(color="#e8e8f0" if dark else "#14142a"),
        ),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig
