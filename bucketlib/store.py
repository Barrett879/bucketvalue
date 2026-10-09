"""HTML renderers for BucketValue's pages, and the page layer's one loader.

The renderers are pure functions: data in, markup out. Every interpolated
string goes through util.esc (team names, arenas, player names and injury
reasons come from third-party feeds and are not trusted input). URLs are MINTED
with readable params and carry view state, because every internal link is a full
page load and therefore a fresh Streamlit session.

THE LOADER (last section) is the only way a page reaches projections:
load_projections(date, marks) wraps pipeline.live_projections in st.cache_data
keyed by CONTENT: the ET date, the report (kind, title time, rows), the lines
(spreads and totals), the sorted effective mark-outs (effective_marks) and the
games past their freeze point (so a game freezes at tip minus 5 minutes even
inside the TTL); never by how a feed answered (live, memory, stale), which
would split one input into two frames. The models come from st.cache_resource
through models.load(). The slate and the Game page call it with the same
arguments for the same view, so the card pill, the board and the tables read
ONE frame, and their medians come from ONE shared simulator provider
(simprov.provider_for). The app never runs features.build and never writes
to disk.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

from . import context as CTX
from . import dist as D
from . import live as LV
from . import models as M
from . import pipeline as P
from . import props, props_ui, simprov, teams
from .cache import MODELS_DIR
from .util import SENTINEL, esc, fmt_num, fmt_tip, now_utc, to_et

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


def view_params(token: str | None = None, marks=()) -> dict:
    """The view state every minted internal URL carries besides the date and
    the theme (spec 5): the pasted board (?board=, props_ui.keep_params) and
    the visitor's mark-outs (?out=). One definition for the slate, the Game
    page and About, so no page's links drop either."""
    out = out_param(marks)
    return {**props_ui.keep_params(token), **({"out": out} if out else {})}


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
                    injury_note: str = "", reports: tuple[int, int] | None = None) -> str:
    """`reports` = (teams filed, teams playing) from reports_filed(): the
    injury-report tile, drawn only when there are projections to read it from."""
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
    if reports is not None:
        summary += (f'<div class="bv-sum"><div class="bv-sum-num">{int(reports[0])}/{int(reports[1])}</div>'
                    f'<div class="bv-sum-lab">Injury reports filed</div></div>')
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
        f"{lines_html}{tag}"
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
    arena = f'<span class="sep">/</span>{esc(row["arena"])}' if row.get("arena") else ""
    return (
        f'<div class="bv-mh"><div class="bv-mh-tag">{tag}</div><div class="bv-mh-row">'
        f'<div class="bv-mh-team" style="--c:{ac}"><div class="bv-mh-abbr">{esc(away)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(away).full_name)}</div></div>{mid}'
        f'<div class="bv-mh-team" style="--c:{hc}"><div class="bv-mh-abbr">{esc(home)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(home).full_name)}</div></div></div>'
        f'<div class="bv-mh-when">{esc(when)}, {esc(fmt_tip(row["tip_utc"]))}'
        f'{arena}{tv}</div></div>'
    )


def fact(label: str, value_html: str, hint: str = "") -> str:
    h = f'<div class="bv-fact-hint">{esc(hint)}</div>' if hint else ""
    return (f'<div class="bv-fact"><div class="bv-fact-lab">{esc(label)}</div>'
            f'<div class="bv-fact-val">{value_html}</div>{h}</div>')


def render_facts(facts: list[str]) -> str:
    return f'<div class="bv-facts">{"".join(facts)}</div>'




# ── Status badges ────────────────────────────────────────────────────────────
# Every label context.display_status can produce (spec 5). The two team-level
# unknowns are drawn once in the table caption AND as "?" on every row, so an
# unknown status can never read as a blank, healthy badge.
UNKNOWN = (CTX.NOT_FILED, CTX.NO_REPORT)
_BADGE = {"Out": ("out", "Out"), "Doubtful": ("dbt", "Doubt"), "Questionable": ("q", "Q"),
          "Probable": ("prob", "Prob"), "Available": ("avail", "Avail"),
          "G League": ("gl", "G Lg"), "Two-way": ("tw", "2-way"),
          "Not with team": ("gl", "NWT"), "Suspended": ("out", "Susp"),
          CTX.NOT_FILED: ("unk", "?"), CTX.NO_REPORT: ("unk", "?")}


def status_badge(status: str | None) -> str:
    if not status:
        return ""
    cls, lab = _BADGE.get(status, ("q", status[:5]))
    return f'<span class="bv-st {cls}" title="{esc(status)}">{esc(lab)}</span>'


STAT_COLS = [("min", "MIN"), ("pts", "PTS"), ("reb", "REB"), ("ast", "AST"), ("fg3m", "3PM"),
             ("stl", "STL"), ("blk", "BLK"), ("tov", "TOV"), ("pra", "PRA"), ("fantasy", "FS")]


def _player_cell(p) -> str:
    img = (f'<img class="bv-hs" src="{esc(teams.headshot_url(p["person_id"]))}" alt="" '
           f'loading="lazy">')
    pos = f'<span class="pos">{esc(p.get("position", ""))}</span>' if p.get("position") else ""
    return f'<td class="l pl">{img}{esc(p["name"])}{pos}</td>'


def _caption(team: str, asof_note: str = "", team_note: str = "") -> str:
    t = teams.get(team)
    chip = (f'<span class="bv-chip warn" title="Injury statuses for this team are unknown">'
            f'{esc(team_note)}</span>') if team_note else ""
    asof = f'<span class="asof">{esc(asof_note)}</span>' if asof_note else ""
    return (f'<div class="bv-pt-cap"><span class="ab">{esc(team)}</span>'
            f'<span class="nm">{esc(t.full_name)}</span>{chip}{asof}</div>')


def render_team_table(team: str, players: pd.DataFrame, asof_note: str = "",
                      lead: str = "min", team_note: str = "") -> str:
    """One team's ROSTER, no projections (preseason, or a game not published
    yet). `players` columns: person_id, name, position, status (badge text or
    ''), out (bool), group ('' or a group label), and any of the STAT_COLS
    keys (missing -> the sentinel). `team_note` is a caption chip (no injury
    report is joined here, so the page says so rather than leave every badge
    blank)."""
    t = teams.get(team)
    head = "".join(f'<th>{esc(lab)}</th>' for _, lab in STAT_COLS)
    rows, last_grp = [], None
    for _, p in players.iterrows():
        grp = p.get("group") or ""
        if grp and grp != last_grp:
            rows.append(f'<tr class="grp"><td class="l" colspan="{len(STAT_COLS) + 2}">{esc(grp)}</td></tr>')
            last_grp = grp
        cells = "".join(
            f'<td class="{"lead" if k == lead else ""}">'
            f'{"" if p.get("out") else fmt_num(p.get(k), 1)}</td>' for k, _ in STAT_COLS)
        rows.append(f'<tr class="{"out" if p.get("out") else ""}">'
                    f'<td class="l">{status_badge(p.get("status"))}</td>{_player_cell(p)}{cells}</tr>')
    return (
        f'<div class="bv-pt" style="--team:{t.hex1}">{_caption(team, asof_note, team_note)}'
        f'<table><colgroup><col style="width:3.4rem"><col style="width:16rem">'
        f'{"".join("<col>" for _ in STAT_COLS)}</colgroup>'
        f'<thead><tr><th class="l"></th><th class="l">Player</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


# ── Game page: one team's projected players ──────────────────────────────────
def projected_ids(schedule: pd.DataFrame | None, date) -> set[str]:
    """The date's games the bot projects: pipeline.state_games (serve's
    _state_games and tables.STATE_TYPES, the one definition), so a page
    never decides "projectable" from a copy of the game types."""
    if schedule is None or schedule.empty:
        return set()
    return {str(g) for g in P.state_games(schedule, date)["game_id"]}


PPLAY_SHOW_BELOW = 0.90          # spec 5: P(play) only when listed or below 90%
DEEP_BENCH_MIN = 10.0            # projected minutes under this: the collapsed deep bench
KEY_ABSENCE_MIN = 15.0           # the card's "Out:" names players projected at 15+ minutes
# The Game table's medians: every STAT_COLS column but MIN, which is the mean.
MEDIAN_STATS = tuple(k for k, _ in STAT_COLS if k != "min")
VIEW_COLS = ["person_id", "game_id", "name", "position", "status", "reason", "out", "pplay",
             "section", "p_play", *(k for k, _ in STAT_COLS)]
_SECTIONS = ("main", "deep", "out")


def _pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{100 * p:.0f}%"


def _badges(rows: pd.DataFrame, shown: pd.DataFrame) -> pd.Series:
    """The badge per row: context.display_status's label, else the roster's
    two-way flag (a player the report does not list)."""
    label = shown["status"].astype(object)
    tw = rows["two_way"].fillna(False).astype(bool).to_numpy()
    return label.where(label.notna() | ~tw, "Two-way")


def team_view(rows: pd.DataFrame, medians: pd.DataFrame | None = None,
              positions: dict | None = None) -> pd.DataFrame:
    """One team's projection rows as the Game table shows them (VIEW_COLS).

    rows       projection-frame rows of ONE team in ONE game
    medians    (person_id, game_id, med_<stat>) from the date's shared
               simulator provider (table_medians); the stat columns are those
               mid-p medians, MIN stays the mean (rows["min"])
    positions  {person_id: position} from the live roster (label only)

    status: context.display_status (Out refined to Two-way, G League, Not with
    team, Suspended; unknown labelled, never blank), else "Two-way" from the
    roster flag. out: listed Out or marked out; no numbers. pplay: '' unless
    the player is listed or below 90% to play. section: main, deep (under 10
    projected minutes) or out. Sorted by section, then projected minutes.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(columns=VIEW_COLS)
    r = rows.reset_index(drop=True)
    shown = CTX.display_status(r)
    out = (r["inj_status"] == "Out").to_numpy(dtype=bool)
    listed = r["inj_status"].notna().to_numpy()
    p = pd.to_numeric(r["p_play"], errors="coerce")
    show_p = ~out & (listed | (p < PPLAY_SHOW_BELOW).to_numpy()) & p.notna().to_numpy()
    mins = pd.to_numeric(r["min"], errors="coerce")
    v = pd.DataFrame({
        "person_id": r["person_id"].astype("int64"), "game_id": r["game_id"].astype(str),
        "name": r["player_name"].astype(str),
        "position": r["person_id"].map(positions or {}).fillna("").astype(str),
        "status": _badges(r, shown), "reason": shown["reason"], "out": out,
        "pplay": [_pct(x) if s else "" for x, s in zip(p, show_p)],
        "section": np.where(out, "out", np.where(mins >= DEEP_BENCH_MIN, "main", "deep")),
        "p_play": p, "min": mins})
    keys = pd.MultiIndex.from_arrays([v["person_id"], v["game_id"]])
    med = (medians.drop_duplicates(["person_id", "game_id"], keep="last")
           .assign(game_id=lambda m: m["game_id"].astype(str))
           .set_index(["person_id", "game_id"]) if medians is not None and len(medians) else None)
    for s in MEDIAN_STATS:
        col = f"med_{s}"
        v[s] = med[col].reindex(keys).to_numpy() if med is not None and col in med else np.nan
    v["_s"] = v["section"].map({s: i for i, s in enumerate(_SECTIONS)})
    v = v.sort_values(["_s", "min", "name"], ascending=[True, False, True], kind="mergesort",
                      na_position="last").drop(columns="_s").reset_index(drop=True)
    v.loc[v["out"], [k for k, _ in STAT_COLS]] = np.nan
    return v[VIEW_COLS]


def team_report_label(rows: pd.DataFrame) -> str | None:
    """'Report not filed' / 'No report yet' when the team's report state is
    unknown (context.display_status labels every row of such a team), else
    None."""
    if rows is None or rows.empty:
        return None
    labels = CTX.display_status(rows)["status"]
    return next((u for u in UNKNOWN if (labels == u).any()), None)


_COLGROUP = ('<colgroup><col style="width:3.4rem"><col style="width:15rem">'
             '<col style="width:4.2rem">' + "".join("<col>" for _ in STAT_COLS) + "</colgroup>")


def _proj_row(p) -> str:
    pid = int(p["person_id"])
    badge = f'<td class="l">{status_badge(p.get("status"))}</td>'
    if p.get("out"):
        reason = p.get("reason")
        if not isinstance(reason, str) or reason.strip() in ("", "-"):
            reason = p.get("status") or "Out"
        return (f'<tr class="out" data-pid="{pid}">{badge}{_player_cell(p)}'
                f'<td class="l reason" colspan="{len(STAT_COLS) + 1}">{esc(reason)}</td></tr>')
    cells = "".join(f'<td class="{"lead" if k == "min" else ""}">{fmt_num(p.get(k), 1)}</td>'
                    for k, _ in STAT_COLS)
    return (f'<tr data-pid="{pid}">{badge}{_player_cell(p)}'
            f'<td class="pp">{esc(p.get("pplay") or "")}</td>{cells}</tr>')


def render_projection_table(team: str, players: pd.DataFrame, asof_note: str = "",
                            team_note: str = "") -> str:
    """One team's projected players (team_view rows). MIN, the projected
    average, is the bold lead column; the nine stats after it are mid-p
    medians under an "if he plays" header; P(play) only where team_view
    filled it; the deep bench collapsed under its own summary; Out rows
    greyed at the bottom with the reason and no numbers."""
    t = teams.get(team)
    sec = players["section"] if len(players) else pd.Series(dtype=object)
    main, deep, out = (players[sec == s] for s in _SECTIONS)
    head = ('<thead><tr class="hgrp"><th colspan="3"></th><th class="avg">Avg</th>'
            f'<th class="ifp" colspan="{len(STAT_COLS) - 1}">Median if he plays</th></tr>'
            '<tr><th class="l"></th><th class="l">Player</th><th>P(play)</th>'
            + "".join(f"<th>{esc(lab)}</th>" for _, lab in STAT_COLS) + "</tr></thead>")
    parts = [f'<table>{_COLGROUP}{head}<tbody>{"".join(_proj_row(p) for _, p in main.iterrows())}'
             f'</tbody></table>']
    if len(deep):
        n = len(deep)
        parts.append(
            f'<details class="bv-deep"><summary>Deep bench: {n} player{"s" if n != 1 else ""} '
            f'under {DEEP_BENCH_MIN:g} projected minutes</summary>'
            f'<table>{_COLGROUP}<tbody>{"".join(_proj_row(p) for _, p in deep.iterrows())}'
            f'</tbody></table></details>')
    if len(out):
        parts.append(
            f'<table class="bv-pt-out">{_COLGROUP}<tbody><tr class="grp">'
            f'<td class="l" colspan="{len(STAT_COLS) + 3}">Out ({len(out)})</td></tr>'
            f'{"".join(_proj_row(p) for _, p in out.iterrows())}</tbody></table>')
    return (f'<div class="bv-pt" style="--team:{t.hex1}">{_caption(team, asof_note, team_note)}'
            f'{"".join(parts)}</div>')


def _report_time(t) -> tuple[str, str]:
    t = to_et(pd.Timestamp(t).to_pydatetime())
    return esc(fmt_tip(t)), f"{t.strftime('%b')} {t.day} report"


def injury_fact(rows: pd.DataFrame | None, info: "P.Info | None" = None
                ) -> tuple[str, str] | None:
    """(value html, hint) for the "Injury report" fact, or None when there is
    nothing checked to say.

    With projection rows: the title time of the report they were projected
    with (report_ts_et), or "Not filed yet" where every row has none (no
    report in reach for the game: team state no_report). Without rows (a
    withheld game, a date not published): the report the live layer read
    (info.report_ts_et) if it read one, else None, never a "Not filed yet"
    claim about a report nobody looked at."""
    if rows is None or rows.empty:
        ts = getattr(info, "report_ts_et", None)
        return _report_time(ts) if ts is not None and not pd.isna(ts) else None
    ts = pd.to_datetime(rows["report_ts_et"], utc=True).dropna()
    if ts.empty:
        return "Not filed yet", ""
    return _report_time(ts.max())


def adjusted_banner(n_marked: int, clear_url: str | None = None) -> str:
    """The banner over numbers a visitor's mark-outs changed."""
    who = "1 player" if n_marked == 1 else f"{n_marked} players"
    clear = (f' <a href="{esc(clear_url)}" target="_self">Clear the marks</a>.'
             if clear_url else "")
    return ('<div class="bv-adjust"><b>Your adjustments, not the published projection.</b> '
            f"{who} marked out. Teammates' minutes and stat lines are recomputed without "
            f"{'him' if n_marked == 1 else 'them'}. Nothing you mark is saved or shown to "
            f"anyone else.{clear}</div>")


# ── Slate cards ──────────────────────────────────────────────────────────────
def short_name(name) -> str:
    """'Maxey' from 'Tyrese Maxey', 'Oubre Jr.' from 'Kelly Oubre Jr.'."""
    parts = str(name or "").strip().split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[1] else parts[0]


def key_absences(frame: pd.DataFrame | None, min_minutes: float = KEY_ABSENCE_MIN,
                 show: int = 2) -> dict[str, str]:
    """{game_id: '<div>Out: X, Y +2</div>'}: players the REPORT lists Out (a
    visitor's mark-out is not a listing) projected at `min_minutes`+ if he
    plays, two-way and G League excluded, most minutes first."""
    if frame is None or frame.empty:
        return {}
    label = CTX.display_status(frame)["status"]
    mins = pd.to_numeric(frame["min"], errors="coerce")
    keep = ((frame["inj_status"] == "Out") & (frame["inj_reason"] != CTX.MARKED_REASON)
            & ~frame["two_way"].fillna(False).astype(bool) & ~label.isin(["Two-way", "G League"])
            & (mins >= min_minutes))
    out = frame[keep.to_numpy()].assign(_m=mins[keep.to_numpy()]).sort_values(
        ["game_id", "_m", "player_name"], ascending=[True, False, True], kind="mergesort")
    res = {}
    for gid, g in out.groupby("game_id", sort=False):
        names = [short_name(n) for n in g["player_name"]]
        more = len(names) - show
        txt = ", ".join(esc(n) for n in names[:show]) + (f" +{more}" if more > 0 else "")
        res[str(gid)] = f'<div class="bv-slate-sub bv-out"><b>Out:</b> {txt}</div>'
    return res


def card_subs(served: "Served | None") -> dict[str, str]:
    """The extra line under each slate card: the key absences, or why a game
    has no projection yet."""
    if served is None:
        return {}
    subs = key_absences(served.frame)
    for gid in served.info.withheld:
        subs.setdefault(str(gid), f'<div class="bv-slate-sub">{esc(P.NO_LINE)}</div>')
    return subs


def reports_filed(frame: pd.DataFrame | None) -> tuple[int, int] | None:
    """(teams whose injury-report state is known, teams in the frame); a team
    is filed when no row of it carries an unknown label."""
    if frame is None or frame.empty:
        return None
    unknown = CTX.display_status(frame)["status"].isin(UNKNOWN).to_numpy()
    per = frame[["game_id", "team"]].assign(_u=unknown).groupby(["game_id", "team"])["_u"].any()
    return int((~per).sum()), int(len(per))


# ── Notices about the frame ──────────────────────────────────────────────────
# Why the committed projections stand in for a live recompute, in words. The
# raw exception text is never rendered: it can carry server paths.
_STALE_WHY = (("NoReport", "no injury report could be read"),
              ("lines feed failed", "the lines feed did not answer"),
              ("schedule or state unreadable", "the schedule could not be read"),
              ("not committed", "this date's history features are not published"),
              ("NullSkew", "the live inputs look unlike anything the models were trained on"),
              ("ModelMismatch", "the models could not be loaded"),
              ("MissingFeatures", "the published features lack a column the models need"),
              ("injury", "the injury report could not be fetched"))


def stale_reason(text) -> str:
    t = str(text or "")
    return next((why for key, why in _STALE_WHY if key in t), "a live source did not answer")


def projection_notes(served: "Served", game_id: str | None = None,
                     today: dt.date | None = None) -> list[str]:
    """The notices a page owes the reader about the frame it shows (html):
    stale (the committed stand-in), frozen at tip, provisional and why, the
    pipeline's own notes, and how far the box scores go."""
    info = served.info
    rows = served.frame if game_id is None else served.game(game_id)
    out: list[str] = []
    if info.stale:
        when = ""
        if len(rows) and rows["computed_at"].notna().any():
            t = to_et(pd.Timestamp(rows["computed_at"].max()).to_pydatetime())
            when = f" at {fmt_tip(t)} on {t.strftime('%b')} {t.day}"
        out.append(render_notice(
            f"<b>Showing the published projections.</b> The live update could not run "
            f"({esc(stale_reason(info.stale))}), so these are the numbers the bot "
            f"published{esc(when)}."))
    if game_id is not None and game_id in info.frozen:
        out.append(render_notice(
            "<b>Frozen at tip.</b> These are the projections as published before tip-off. "
            "They no longer change, and mark-outs no longer apply."))
    elif game_id is not None and game_id in served.past and len(rows):
        out.append(render_notice(
            "<b>Tipped on provisional projections.</b> Shown as published; a provisional "
            "projection is never frozen."))
    provisional = bool(rows["provisional"].any()) if game_id is not None and len(rows) \
        else info.provisional
    if provisional:
        why = []
        if today is not None and info.date > today:
            why.append("Projections for a future date update with the morning run, the injury "
                       "report and the lines.")
        n = len(info.completeness)
        if n:
            why.append(f"{n} final game{'s' if n != 1 else ''} involving these teams "
                       f"{'are' if n != 1 else 'is'} not in the box scores yet, so recent "
                       "games are missing from the history.")
        if P.STALE_FEATURES in info.notes:
            why.append("The history features were built before the latest finals were ingested.")
        out.append(render_notice(
            "<b>Provisional.</b> " + esc(" ".join(why) or "Built on incomplete data.")
            + " Provisional numbers are never frozen."))
    extra = [n for n in info.notes if n != P.STALE_FEATURES and "provisional projections" not in n]
    for n in extra:
        out.append(f'<div class="bv-note">{esc(n)}</div>')
    if info.data_through is not None:
        d = info.data_through
        out.append(f'<div class="bv-note">Box scores through {esc(d.strftime("%B"))} {d.day}.</div>')
    return out


# ── About: the trust table, generated from data/stat_reliability.csv ─────────
POWER_PLAY_2_PAYOUT = 3.0        # a two-pick PrizePicks Power Play pays 3x


def breakeven(legs: int = 2, payout: float = POWER_PLAY_2_PAYOUT) -> float:
    """Per-pick hit rate (percent) at which a `legs`-pick play paying
    `payout` breaks even: (1/payout)^(1/legs)."""
    return 100.0 * (1.0 / payout) ** (1.0 / legs)


def trust_table() -> pd.DataFrame:
    """One row per board stat: the tier and edge the board itself shows
    (props.reliability_for: a combo inherits its weakest component), the
    stat's own measurement, the component it inherits from, and whether the
    simulator ('model') or the labelled baseline serves it."""
    tbl = props.reliability_table()
    own = {str(s): (str(t), e) for s, t, e in zip(tbl["stat"], tbl["tier"], tbl["side_edge_pts"])}
    recs = []
    for s in simprov.GATED:
        rel = props.reliability_for(s)
        # The component whose measurement the board shows (None: the stat's own).
        via = None
        for c in props.tier_components(s)[1:]:
            rc = props.reliability_for(c)
            if (rc["tier"], rc["edge"]) == (rel["tier"], rel["edge"]):
                via = c
                break
        o = own.get(s)
        recs.append({"stat": s, "label": props.STAT_LABEL[s], "tier": rel["tier"],
                     "edge": rel["edge"], "own_tier": o[0] if o else None,
                     "own_edge": float(o[1]) if o and pd.notna(o[1]) else None,
                     "via": via, "source": "model"})
    for s in simprov.BASELINE:
        recs.append({"stat": s, "label": props.STAT_LABEL.get(s, s), "tier": "unmeasured",
                     "edge": None, "own_tier": None, "own_edge": None, "via": None,
                     "source": "baseline"})
    return pd.DataFrame(recs)


def trust_caveats() -> list[str]:
    """What the reliability file says about how it was measured: the seasons
    and whether the lines were proxies (plain text, from its notes)."""
    notes = " ".join(str(n) for n in props.reliability_table().get("note", pd.Series(dtype=object))
                     .dropna())
    out = []
    m = re.search(r"worst of the (.+?) out-of-sample seasons", notes)
    if m:
        out.append(f"Each edge is the worst of the {m.group(1)} seasons, every one of them out "
                   "of sample: percentage points of sides called correctly above always taking "
                   "the more common side.")
    if "PROXY" in notes.upper():
        out.append("The lines were PROXY half-point lines built from a baseline projection, not "
                   "real PrizePicks lines. A posted line already prices most of what a model "
                   "knows, so against a real board expect less than these edges.")
    return out


def render_trust_table(table: pd.DataFrame) -> str:
    """The About page's trust table (trust_table rows); tier badges are the
    board's own (props_ui._tier_badge), so the two cannot draw a tier
    differently."""
    body = []
    baseline_started = False
    for r in table.to_dict(orient="records"):
        if r["source"] == "baseline" and not baseline_started:
            baseline_started = True
            body.append('<tr class="grp"><td class="l" colspan="4">Served from a labelled '
                        "baseline until the gate measures it</td></tr>")
        edge = f"{float(r['edge']):+.1f}" if r["edge"] is not None and pd.notna(r["edge"]) else SENTINEL
        if r["source"] == "baseline":
            basis = "Not measured yet: a mean and a dispersion, not the simulator"
        elif r["via"]:
            alone = (f" (alone: {r['own_tier']}, {float(r['own_edge']):+.1f})"
                     if r["own_tier"] and r["own_edge"] is not None else "")
            basis = (f"Inherits {props.STAT_LABEL.get(r['via'], r['via'])}, its weakest "
                     f"component{alone}")
        elif r["stat"] in props.COMBO_STATS:
            basis = "Measured directly; no component sits in a lower tier"
        else:
            basis = "Measured directly"
        body.append(f'<tr><td class="l pl">{esc(r["label"])}</td>'
                    f'<td class="l">{props_ui._tier_badge(r["tier"], r["edge"])}</td>'
                    f'<td>{esc(edge)}</td><td class="l basis">{esc(basis)}</td></tr>')
    return ('<div class="bv-trust"><table><colgroup><col style="width:11rem">'
            '<col style="width:8rem"><col style="width:5rem"><col></colgroup>'
            '<thead><tr><th class="l">Stat</th><th class="l">Tier</th><th>Edge (pts)</th>'
            '<th class="l">Basis</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# ── The loader ───────────────────────────────────────────────────────────────
LOADER_TTL_S = 120          # the clock moves hours_to_tip and report_age_h; reports, 10 min
LOADER_MAX_ENTRIES = 64     # mark-out variants per date stay bounded
SIM_PARAMS_FILE = "sim_params_v1.json"
MAX_MARKS = 40


@dataclass
class Served:
    """What a page shows for one ET date: the frame and its Info
    (pipeline.live_projections), the clock they were computed at, the
    mark-outs applied, and the games past their freeze point at that clock."""
    frame: pd.DataFrame
    info: P.Info
    now: dt.datetime
    marks: tuple[int, ...] = ()
    past: frozenset = field(default_factory=frozenset)

    def game(self, game_id) -> pd.DataFrame:
        f = self.frame
        return f[f["game_id"].astype(str) == str(game_id)] if len(f) else f

    @property
    def applied(self) -> tuple[int, ...]:
        """The marks this frame applied: rows the live layer marked out
        (context's mark-out reason). A mark on a frozen game or on the
        committed stand-in is in `marks` but not here, and moved nothing."""
        f = self.frame
        if not len(f) or "inj_reason" not in f.columns:
            return ()
        hit = f.loc[(f["inj_status"] == "Out") & (f["inj_reason"] == CTX.MARKED_REASON), "person_id"]
        return tuple(sorted({int(p) for p in hit}))


def parse_out(raw) -> tuple[int, ...]:
    """?out=pid,pid -> sorted unique person ids. Junk is ignored through
    props.void_id_set, the board's own hardened parser for this param."""
    if not raw:
        return ()
    return tuple(sorted(props.void_id_set(str(raw)[:600].split(",")[:MAX_MARKS])))


def out_param(marks) -> str | None:
    """The ?out= value for `marks`, or None for no marks."""
    return ",".join(str(p) for p in sorted({int(m) for m in marks})) or None


@st.cache_resource(show_spinner=False)
def models() -> M.Models:
    """The production models, verified (models.load), once per process."""
    return M.load()


@st.cache_resource(show_spinner=False)
def sim_params() -> D.SimParams:
    return D.SimParams.load(MODELS_DIR / SIM_PARAMS_FILE)


@st.cache_data(ttl=LOADER_TTL_S, max_entries=LOADER_MAX_ENTRIES, show_spinner=False)
def _live_frame(date_iso: str, report_key: tuple, odds_key: tuple, marks: tuple, past: tuple,
                _now: dt.datetime, _report, _odds, _sched) -> tuple[pd.DataFrame, P.Info]:
    try:
        mdl = models()
    except P.LIVE_FAILURES:
        mdl = None      # live_projections loads again, fails the same way, serves the committed rows
    return P.live_projections(dt.date.fromisoformat(date_iso), _now, marked_out=set(marks),
                              report=_report, odds=_odds, models=mdl, schedule=_sched)


# The two keys below are CONTENT only. live.Live's source ("live" on the fetch,
# "memory" on every hit inside the feed's TTL) and its stale flag describe the
# fetch, not what was fetched: keyed on them, every report or lines refresh
# computed the date twice for the same inputs, at two clocks, and the slate
# and the Game page opened from it could read two different frames.
def _digest(df) -> int | None:
    """A content hash of a report table (about a millisecond for a full
    slate's report), so a slot re-issued under the same title time still
    changes the key."""
    if not isinstance(df, pd.DataFrame):
        return None
    try:
        return int(pd.util.hash_pandas_object(df, index=False).sum())
    except (TypeError, ValueError):
        return len(df)


def _report_key(sched) -> tuple[object, tuple]:
    """(the live report, its cache key: its kind, title time and content).
    A failed fetch passes None, so live_projections asks again (the circuit
    breaker answers at once) and serves the committed projections with the
    reason."""
    try:
        rep = LV.injury_report(sched)
    except RuntimeError:
        return None, ("unavailable",)
    val = rep.value
    meta = getattr(val, "meta", None) or {}
    return rep, (type(val).__name__, str(meta.get("report_ts_et")),
                 _digest(getattr(val, "rows", None)), _digest(getattr(val, "team_rows", None)))


def _odds_key(odds) -> tuple:
    """The lines as live_projections reads them: a failed feed (stale with
    source "none": it serves the committed rows), else every game's spread
    and total."""
    if odds is None:
        return ("none",)
    if getattr(odds, "stale", False) and getattr(odds, "source", "") == "none":
        return ("failed",)
    items = sorted((str(a), str(h), repr((v or {}).get("home_line")), repr((v or {}).get("total")))
                   for (a, h), v in (odds.value or {}).items())
    return ("lines", tuple(items))


def effective_marks(frame: pd.DataFrame | None, marks) -> tuple[int, ...]:
    """The mark-outs that can change something: players with a row in the
    date's UNMARKED frame whom the report does not already list Out.

    A mark naming nobody on the date (junk in ?out=) would cost a full
    recompute per distinct value and push real entries out of the cache. A
    mark on a player already listed Out moves no number, but would swap his
    report reason for the mark-out note (dropping him from the card's Out
    line) and flag his whole team adjusted. Both are dropped before the
    cache key. Marks on a frozen game's players stay: live_projections
    ignores them there and says so."""
    if not marks or frame is None or frame.empty:
        return ()
    ok = frame.loc[frame["inj_status"] != "Out", "person_id"].astype("int64")
    return tuple(sorted({int(m) for m in marks} & set(ok)))


def load_projections(date: dt.date, marks=()) -> Served:
    """The projection frame for one ET date, with the visitor's mark-outs.

    A date with neither committed file returns at once (Info.reason
    pipeline.NOT_PUBLISHED, no fetch). Otherwise the live report and lines
    are read first (each through live.py's own TTL and circuit breaker) and
    their CONTENT keys the cached recompute; a date whose games have all
    reached their freeze point fetches nothing and serves the committed
    rows. The unmarked frame is loaded first (the slate's own, normally
    cached already); only effective_marks reach the key, and Served.marks
    holds those."""
    now = now_utc()
    if not (P.features_path(date).exists() or P.projections_path(date).exists()):
        frame, info = P.live_projections(date, now)
        return Served(frame, info, now)
    try:
        sched = LV.schedule().value
    except RuntimeError:
        sched = None
    games = P.state_games(sched, date) if sched is not None else None
    past = frozenset(P.frozen_ids(games, now)) if games is not None else frozenset()
    todo = games is None or bool(set(games["game_id"]) - past)
    report, rkey = _report_key(sched) if todo else (None, ("not needed",))
    odds = LV.odds_on(date) if todo else None

    def frame_for(m: tuple) -> tuple[pd.DataFrame, P.Info]:
        return _live_frame(date.isoformat(), rkey, _odds_key(odds), m, tuple(sorted(past)),
                           _now=now, _report=report, _odds=odds, _sched=sched)
    frame, info = frame_for(())
    eff = effective_marks(frame, marks)
    if eff:
        frame, info = frame_for(eff)
    return Served(frame, info, now, eff, past)


def table_medians(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Mid-p medians (MEDIAN_STATS) for every row of the date's frame, from
    the shared provider (simprov.provider_for), so the Game tables and the
    board read one simulation."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["person_id", "game_id", *(f"med_{s}" for s in MEDIAN_STATS)])
    return simprov.provider_for(frame, sim_params()).medians(MEDIAN_STATS)
