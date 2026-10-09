"""HTML renderers for BucketValue's pages. Pure functions: data in, markup out.

Every interpolated string goes through util.esc (team names, arenas and player
names come from third-party feeds and are not trusted input). URLs are MINTED
with readable params and carry view state, because every internal link is a full
page load and therefore a fresh Streamlit session.
"""
from __future__ import annotations

import datetime as dt
from urllib.parse import urlencode

import pandas as pd

from . import teams
from .util import SENTINEL, esc, fmt_num, fmt_tip, to_et

# ── Links ────────────────────────────────────────────────────────────────────
def page_url(path: str, keep: dict | None = None, **params) -> str:
    """A minted internal link: `params` first, then carried view state."""
    q = {k: v for k, v in params.items() if v not in (None, "")}
    for k, v in (keep or {}).items():
        if v not in (None, "") and k not in q:
            q[k] = v
    return f"{path}?{urlencode(q)}" if q else path


def game_url(date: dt.date, away: str, home: str, keep: dict | None = None) -> str:
    return page_url("/Game", keep, date=date.isoformat(), away=away, home=home)


# ── Small pieces ─────────────────────────────────────────────────────────────
def render_notice(body_html: str) -> str:
    """Never let an empty or degraded result render as silence."""
    return f'<div class="bv-notice">{body_html}</div>'


def _colors(away: str, home: str) -> tuple[str, str]:
    """Two stripe colours that do not collide (several teams share red)."""
    a, h = teams.get(away), teams.get(home)
    ac, hc = a.hex1, h.hex1
    if ac.lower() == hc.lower():
        hc = h.hex2
    return ac, hc


def market_html(odds: dict | None, away: str, home: str) -> str:
    """The spread quoted on the favourite, then the total.

    ESPN's line is the HOME team's (negative = home favoured), the opposite sign
    convention to nflverse's spread_line; quoting the favourite from the sign is
    exactly where a port would go wrong, so it is done here once.
    """
    if not odds:
        return '<div class="bv-slate-mkt"><span>Line not posted</span></div>'
    line, total = odds.get("home_line"), odds.get("total")
    bits = []
    if line is not None and not pd.isna(line):
        if float(line) == 0.0:
            bits.append("<b>PK</b>")
        else:
            fav = home if float(line) < 0 else away
            bits.append(f"<b>{esc(fav)} {-abs(float(line)):.1f}</b>")
    if total is not None and not pd.isna(total):
        if bits:
            bits.append('<span class="sep">/</span>')
        bits.append(f"<span>O/U {float(total):.1f}</span>")
    if not bits:
        return '<div class="bv-slate-mkt"><span>Line not posted</span></div>'
    return f'<div class="bv-slate-mkt">{"".join(bits)}</div>'


def _final_html(row) -> str:
    a, h = row.get("away_score"), row.get("home_score")
    try:
        a, h = int(a), int(h)
    except (TypeError, ValueError):
        return ""
    return (f'<div class="bv-slate-final"><span{" class=w" if a > h else ""}>{a}</span>'
            f' - <span{" class=w" if h > a else ""}>{h}</span></div>')


# ── Slate ────────────────────────────────────────────────────────────────────
def render_masthead(date: dt.date, games: pd.DataFrame, n_lines_posted: int,
                    injury_note: str = "") -> str:
    n = 0 if games is None else len(games)
    first = fmt_tip(games["tip_utc"].min()) if n else SENTINEL
    summary = (
        f'<div class="bv-sum"><div class="bv-sum-num">{n}</div>'
        f'<div class="bv-sum-lab">Games</div></div>'
        f'<div class="bv-sum"><div class="bv-sum-num">{esc(first)}</div>'
        f'<div class="bv-sum-lab">First tip</div></div>'
        f'<div class="bv-sum"><div class="bv-sum-num">{n_lines_posted}/{n}</div>'
        f'<div class="bv-sum-lab">Lines posted</div></div>'
    )
    day = date.strftime("%A")
    pretty = date.strftime("%B %-d")
    deck = f'<div class="bv-deck">{injury_note}</div>' if injury_note else ""
    return (
        f'<div class="bv-masthead">'
        f'<div class="bv-kicker"><span class="bv-pip"></span>NBA projections</div>'
        f'<div class="bv-mast-rule"></div>'
        f'<div class="bv-mast-row">'
        f'<div><div class="bv-brand">{esc(day)} <span class="accent">{esc(pretty)}</span></div></div>'
        f'<div class="bv-mast-summary">{summary}</div>'
        f"</div>{deck}</div>"
    )


def render_slate_card(row, date: dt.date, keep: dict, odds: dict | None = None,
                      state: str = "upcoming", n_lines: int = 0, sub_html: str = "") -> str:
    away, home = row["away"], row["home"]
    if away is None or home is None:
        return (f'<div class="bv-slate-card" style="--away:#888;--home:#888">'
                f'<div class="bv-slate-teams"><span class="bv-slate-away">TBD</span>'
                f'<span class="bv-slate-at">at</span><span class="bv-slate-home">TBD</span></div>'
                f'<div class="bv-slate-time">{esc(fmt_tip(row["tip_utc"]))}</div></div>')
    ac, hc = _colors(away, home)
    if state == "final":
        tag, bottom = '<span class="bv-slate-tag final">Final</span>', _final_html(row)
    elif state == "live":
        tag, bottom = '<span class="bv-slate-tag live">Live</span>', market_html(odds, away, home)
    else:
        tag, bottom = "", market_html(odds, away, home)
    lines_html = ""
    if n_lines > 0:
        unit = "line" if n_lines == 1 else "lines"
        lines_html = f'<span class="bv-slate-lines"><b>{n_lines}</b><span class="u">{unit}</span></span>'
    title = f"{teams.get(away).full_name} at {teams.get(home).full_name}"
    return (
        f'<a class="bv-slate-card" href="{esc(game_url(date, away, home, keep))}" target="_self" '
        f'title="{esc(title)}" style="--away:{ac};--home:{hc}">'
        f"{tag}{lines_html}"
        f'<div class="bv-slate-teams"><span class="bv-slate-away">{esc(away)}</span>'
        f'<span class="bv-slate-at">at</span><span class="bv-slate-home">{esc(home)}</span></div>'
        f'<div class="bv-slate-time">{esc(fmt_tip(row["tip_utc"]))}</div>'
        f"{bottom}{sub_html}</a>"
    )


def render_slate(games: pd.DataFrame, date: dt.date, keep: dict,
                 odds_by_game: dict | None = None, states: dict | None = None,
                 line_counts: dict | None = None, subs: dict | None = None) -> str:
    """Cards grouped by tip time: with 10-15 games a night the useful grouping
    is when each one starts (and so when its information stops changing)."""
    if games is None or games.empty:
        return ""
    odds_by_game, states = odds_by_game or {}, states or {}
    line_counts, subs = line_counts or {}, subs or {}
    parts = []
    tips = games.assign(_t=games["tip_utc"].map(lambda t: to_et(t).strftime("%H:%M")))
    for _, block in tips.groupby("_t", sort=True):
        label = fmt_tip(block["tip_utc"].iloc[0])
        n = len(block)
        parts.append(f'<div class="bv-tipgrp"><span class="lab">{esc(label)}</span>'
                     f'<span class="meta">{n} game{"s" if n != 1 else ""}</span></div>')
        cards = "".join(
            render_slate_card(r, date, keep, odds_by_game.get(r["game_id"]),
                              states.get(r["game_id"], "upcoming"),
                              int(line_counts.get(r["game_id"], 0)), subs.get(r["game_id"], ""))
            for _, r in block.iterrows())
        parts.append(f'<div class="bv-slate-grid">{cards}</div>')
    return "".join(parts)


# ── Game page ────────────────────────────────────────────────────────────────
def render_matchup_header(row, state: str) -> str:
    away, home = row["away"], row["home"]
    ac, hc = _colors(away, home)
    if state == "final":
        a, h = int(row["away_score"]), int(row["home_score"])
        mid = (f'<div class="bv-mh-score"><span class="{"w" if a > h else ""}">{a}</span>'
               f'<span class="dash">-</span><span class="{"w" if h > a else ""}">{h}</span></div>')
        tag = "Final"
    else:
        mid, tag = '<div class="bv-mh-at">at</div>', ("Live" if state == "live" else "Upcoming")
    when = to_et(row["tip_utc"]).strftime("%a %b %-d")
    tv = f'<span class="sep">/</span>{esc(row["national_tv"])}' if row.get("national_tv") else ""
    return (
        f'<div class="bv-mh"><div class="bv-mh-tag">{tag}</div><div class="bv-mh-row">'
        f'<div class="bv-mh-team" style="--c:{ac}"><div class="bv-mh-abbr">{esc(away)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(away).full_name)}</div></div>{mid}'
        f'<div class="bv-mh-team" style="--c:{hc}"><div class="bv-mh-abbr">{esc(home)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(home).full_name)}</div></div></div>'
        f'<div class="bv-mh-when">{esc(when)}, {esc(fmt_tip(row["tip_utc"]))}'
        f'<span class="sep">/</span>{esc(row.get("arena", ""))}{tv}</div></div>'
    )


def fact(label: str, value_html: str, hint: str = "") -> str:
    h = f'<div class="bv-fact-hint">{esc(hint)}</div>' if hint else ""
    return (f'<div class="bv-fact"><div class="bv-fact-lab">{esc(label)}</div>'
            f'<div class="bv-fact-val">{value_html}</div>{h}</div>')


def render_facts(facts: list[str]) -> str:
    return f'<div class="bv-facts">{"".join(facts)}</div>'


_BADGE = {"Out": ("out", "Out"), "Doubtful": ("dbt", "Doubt"), "Questionable": ("q", "Q"),
          "Probable": ("prob", "Prob"), "G League": ("gl", "G Lg"), "Two-way": ("tw", "2-way")}


def status_badge(status: str | None) -> str:
    if not status:
        return ""
    cls, lab = _BADGE.get(status, ("q", status[:5]))
    return f'<span class="bv-st {cls}" title="{esc(status)}">{esc(lab)}</span>'


STAT_COLS = [("min", "MIN"), ("pts", "PTS"), ("reb", "REB"), ("ast", "AST"), ("fg3m", "3PM"),
             ("stl", "STL"), ("blk", "BLK"), ("tov", "TOV"), ("pra", "PRA"), ("fantasy", "FS")]


def render_team_table(team: str, players: pd.DataFrame, asof_note: str = "",
                      lead: str = "min") -> str:
    """One team's players. `players` columns: person_id, name, position,
    status (badge text or ''), out (bool), group ('' or a group label), and any
    of the STAT_COLS keys (missing -> the sentinel)."""
    t = teams.get(team)
    head = "".join(f'<th>{esc(lab)}</th>' for _, lab in STAT_COLS)
    rows, last_grp = [], None
    for _, p in players.iterrows():
        grp = p.get("group") or ""
        if grp and grp != last_grp:
            rows.append(f'<tr class="grp"><td class="l" colspan="{len(STAT_COLS) + 2}">{esc(grp)}</td></tr>')
            last_grp = grp
        img = (f'<img class="bv-hs" src="{esc(teams.headshot_url(p["person_id"]))}" alt="" '
               f'loading="lazy">')
        cells = "".join(
            f'<td class="{"lead" if k == lead else ""}">'
            f'{"" if p.get("out") else fmt_num(p.get(k), 1)}</td>' for k, _ in STAT_COLS)
        pos = f'<span class="pos">{esc(p.get("position", ""))}</span>' if p.get("position") else ""
        rows.append(f'<tr class="{"out" if p.get("out") else ""}">'
                    f'<td class="l">{status_badge(p.get("status"))}</td>'
                    f'<td class="l pl">{img}{esc(p["name"])}{pos}</td>{cells}</tr>')
    asof = f'<span class="asof">{esc(asof_note)}</span>' if asof_note else ""
    return (
        f'<div class="bv-pt" style="--team:{t.hex1}">'
        f'<div class="bv-pt-cap"><span class="ab">{esc(team)}</span>'
        f'<span class="nm">{esc(t.full_name)}</span>{asof}</div>'
        f'<table><colgroup><col style="width:3.4rem"><col style="width:16rem">'
        f'{"".join("<col>" for _ in STAT_COLS)}</colgroup>'
        f'<thead><tr><th class="l"></th><th class="l">Player</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )
