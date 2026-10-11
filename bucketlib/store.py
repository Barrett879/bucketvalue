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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

from . import cache as C
from . import context as CTX
from . import dist as D
from . import live as LV
from . import models as M
from . import pipeline as P
from . import props, props_ui, simprov, teams
from . import state as S
from .cache import MODELS_DIR
from .util import SENTINEL, esc, fmt_num, fmt_tip, now_utc, season_for_date, to_et

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
def active_games(games: pd.DataFrame | None) -> pd.DataFrame | None:
    """The games that will be played on the date: postponed ones dropped
    (live.is_postponed). The masthead counts these and their lines."""
    if games is None or games.empty:
        return games
    return games[[not LV.is_postponed(r) for _, r in games.iterrows()]]


def first_tip(games: pd.DataFrame | None) -> str:
    """'7:00 PM ET', from games whose tip time is SET (a Cup knockout
    placeholder's 00:00 ET is made up: live.time_tbd); 'TBD' when no game's
    time is set, the sentinel when there are no games."""
    if games is None or games.empty:
        return SENTINEL
    timed = [r["tip_utc"] for _, r in games.iterrows() if not LV.time_tbd(r)]
    return fmt_tip(min(timed)) if timed else "TBD"


def render_masthead(date: dt.date, games: pd.DataFrame, n_lines_posted: int | None,
                    injury_note: str = "", reports: tuple[int, int] | None = None,
                    games_label: str = "Games") -> str:
    """`games`: the games the tiles count (active_games: postponed ones out).
    `n_lines_posted` None leaves the lines tile out (preseason: no lines are
    read; a date with no games: there is nothing to post). `reports` =
    (teams filed, teams playing) from reports_filed(): the injury-report
    tile, drawn only when there are projections to read it from."""
    n = 0 if games is None else len(games)
    summary = (
        f'<div class="bv-sum"><div class="bv-sum-num">{n}</div>'
        f'<div class="bv-sum-lab">{esc(games_label)}</div></div>'
        f'<div class="bv-sum"><div class="bv-sum-num">{esc(first_tip(games))}</div>'
        f'<div class="bv-sum-lab">First tip</div></div>'
    )
    if n_lines_posted is not None and n:
        summary += (f'<div class="bv-sum"><div class="bv-sum-num">{int(n_lines_posted)}/{n}</div>'
                    f'<div class="bv-sum-lab">Lines posted</div></div>')
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


TBD_TIME = "Time TBD"
_TAGS = {"final": ("final", "Final"), "live": ("live", "Live"), "upcoming": ("upcoming", "Upcoming"),
         "postponed": ("ppd", "Postponed")}


def _tip_text(row) -> str:
    """The card's and the header's tip time, never a placeholder's made-up
    00:00 ET (live.time_tbd)."""
    return TBD_TIME if LV.time_tbd(row) else fmt_tip(row["tip_utc"])


def _placeholder_label(row) -> str:
    """What a Cup knockout placeholder is ('Emirates NBA Cup', 'Quarterfinal'),
    from the schedule's own label and subtype."""
    bits = [str(row.get(k) or "").strip() for k in ("label", "subtype")]
    return " / ".join(b for b in bits if b and b.lower() != "nan")


def b2b_chips(teams_b2b) -> str:
    """'DET B2B' chips for the teams on the second night of a back-to-back."""
    if not teams_b2b:
        return ""
    chips = "".join(f'<span class="bv-chip" title="Played yesterday: second night of a '
                    f'back-to-back">{esc(t)} B2B</span>' for t in teams_b2b)
    return f'<div class="bv-slate-chips">{chips}</div>'


def render_slate_card(row, date: dt.date, keep: dict, odds: dict | None = None,
                      state: str = "upcoming", n_lines: int = 0, sub_html: str = "",
                      b2b: tuple = ()) -> str:
    """One game card: status tag (Upcoming, Live, Final, Postponed), the
    matchup, the tip time ('Time TBD' when it is not set), the line or the
    score, back-to-back chips and the card's extra line. A placeholder whose
    teams are not known yet is an unlinked card in the same panel."""
    away, home = row["away"], row["home"]
    if not (isinstance(away, str) and isinstance(home, str)):
        label = _placeholder_label(row)
        label_html = f'<div class="bv-slate-label">{esc(label)}</div>' if label else ""
        return (f'<div class="bv-slate-card bv-slate-tbd" style="--away:#888;--home:#888">'
                f'<div class="bv-slate-teams"><span class="bv-slate-away">TBD</span>'
                f'<span class="bv-slate-at">at</span><span class="bv-slate-home">TBD</span></div>'
                f'<div class="bv-slate-time">{esc(_tip_text(row))}</div>{label_html}</div>')
    ac, hc = _colors(away, home)
    cls, word = _TAGS.get(state, _TAGS["upcoming"])
    tag = f'<span class="bv-slate-tag {cls}">{word}</span>'
    if state == "final":
        bottom = _final_html(row)
    elif state == "postponed":
        bottom = '<div class="bv-slate-mkt"><span>Postponed</span></div>'
    else:
        bottom = market_html(odds, away, home)
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
        f'<div class="bv-slate-time">{esc(_tip_text(row))}</div>'
        f"{bottom}{b2b_chips(b2b if state != 'postponed' else ())}{sub_html}</a>"
    )


def render_slate(games: pd.DataFrame, date: dt.date, keep: dict,
                 odds_by_game: dict | None = None, states: dict | None = None,
                 line_counts: dict | None = None, subs: dict | None = None,
                 b2b: dict | None = None) -> str:
    """Cards grouped by tip time: with 10-15 games a night the useful grouping
    is when each one starts (and so when its information stops changing).
    Games whose time is not set (live.time_tbd) are grouped last under
    'Time TBD', never under their made-up 00:00 ET. `b2b`: {game_id: teams
    on a back-to-back} (rest_by_game)."""
    if games is None or games.empty:
        return ""
    odds_by_game, states = odds_by_game or {}, states or {}
    line_counts, subs, b2b = line_counts or {}, subs or {}, b2b or {}
    parts = []
    tbd = [LV.time_tbd(r) for _, r in games.iterrows()]
    tips = games.assign(_t=[("~" if t else to_et(r["tip_utc"]).strftime("%H:%M"))
                            for t, (_, r) in zip(tbd, games.iterrows())])
    for key, block in tips.groupby("_t", sort=True):
        label = TBD_TIME if key == "~" else fmt_tip(block["tip_utc"].iloc[0])
        n = len(block)
        parts.append(f'<div class="bv-tipgrp"><span class="lab">{esc(label)}</span>'
                     f'<span class="meta">{n} game{"s" if n != 1 else ""}</span></div>')
        cards = "".join(
            render_slate_card(r, date, keep, odds_by_game.get(r["game_id"]),
                              states.get(r["game_id"], "upcoming"),
                              int(line_counts.get(r["game_id"], 0)), subs.get(r["game_id"], ""),
                              tuple(b2b.get(str(r["game_id"]), ())))
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
        mid, tag = '<div class="bv-mh-at">at</div>', _TAGS.get(state, _TAGS["upcoming"])[1]
    when = to_et(row["tip_utc"]).strftime("%a %b %-d")
    tv = f'<span class="sep">/</span>{esc(row["national_tv"])}' if row.get("national_tv") else ""
    arena = f'<span class="sep">/</span>{esc(row["arena"])}' if row.get("arena") else ""
    return (
        f'<div class="bv-mh"><div class="bv-mh-tag">{esc(tag)}</div><div class="bv-mh-row">'
        f'<div class="bv-mh-team" style="--c:{ac}"><div class="bv-mh-abbr">{esc(away)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(away).full_name)}</div></div>{mid}'
        f'<div class="bv-mh-team" style="--c:{hc}"><div class="bv-mh-abbr">{esc(home)}</div>'
        f'<div class="bv-mh-name">{esc(teams.get(home).full_name)}</div></div></div>'
        f'<div class="bv-mh-when">{esc(when)}, {esc(_tip_text(row))}'
        f'{arena}{tv}</div></div>'
    )


def fact(label: str, value_html: str, hint: str = "") -> str:
    h = f'<div class="bv-fact-hint">{esc(hint)}</div>' if hint else ""
    return (f'<div class="bv-fact"><div class="bv-fact-lab">{esc(label)}</div>'
            f'<div class="bv-fact-val">{value_html}</div>{h}</div>')


def render_facts(facts: list[str]) -> str:
    return f'<div class="bv-facts">{"".join(facts)}</div>'


# ── Lines as the pages show them ─────────────────────────────────────────────
PRE_TIP = "Pre-tip line"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def posted_lines(odds: dict | None, games: pd.DataFrame | None, frame: pd.DataFrame | None,
                 started=(), snapshots: pd.DataFrame | None = None) -> dict:
    """{(away, home): odds dict} for the date's games as the pages SHOW them.

    The live ESPN line where it has one. ESPN drops a game's odds at tip, so
    a game in `started` (past its freeze point, or live or final) with no
    live line shows, marked pre_tip:
      the line its projection rows were computed with (home_line, total: the
      frozen projection's own inputs), else
      the latest pre-tip capture in the bot's committed odds snapshots
      (`snapshots`: committed_odds, mapped to the game by
      pipeline.lines_from_odds), so a tipped game with no projection rows
      (withheld, never published, or not projected) still shows its line.
    Display only: this never reaches pipeline.live_projections, where an
    untipped game without a line stays withheld."""
    out = {k: v for k, v in (odds or {}).items() if v}
    if games is None or games.empty:
        return out
    has_rows = frame is not None and not frame.empty
    if not has_rows and (snapshots is None or snapshots.empty):
        return out
    snap = (P.lines_from_odds(snapshots, games).set_index("game_id")
            if snapshots is not None and not snapshots.empty else None)
    started = {str(g) for g in started}
    for _, g in games.iterrows():
        k, gid = (g["away"], g["home"]), str(g["game_id"])
        have = out.get(k) or {}
        if gid not in started or _num(have.get("home_line")) is not None \
                or _num(have.get("total")) is not None:
            continue
        rows = frame[frame["game_id"].astype(str) == gid] if has_rows else frame
        if rows is None or rows.empty:
            rows = snap.loc[[gid]] if snap is not None and gid in snap.index else None
        if rows is None:
            continue
        hl = pd.to_numeric(rows["home_line"], errors="coerce").dropna()
        tot = pd.to_numeric(rows["total"], errors="coerce").dropna()
        if len(hl) or len(tot):
            out[k] = {"home_line": float(hl.iloc[0]) if len(hl) else None,
                      "total": float(tot.iloc[0]) if len(tot) else None,
                      "provider": None, "pre_tip": True}
    return out


@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def _committed_odds(path: str, content_hash) -> pd.DataFrame | None:
    return C.read_parquet(Path(path))


def committed_odds(date) -> pd.DataFrame | None:
    """The bot's committed pregame odds snapshots for `date`
    (state.odds_path: every capture is pre-tip, regular season and play-in),
    or None. Cached by the file's content hash. Read for display only
    (posted_lines), never written: the app writes nothing to disk."""
    p = S.odds_path(date)
    if not p.exists():
        return None
    try:
        return _committed_odds(str(p), C.read_parquet_meta(p).get("content_hash"))
    except Exception as e:  # noqa: BLE001 - a display extra never takes a page down
        C.logger.warning("committed odds for %s unreadable (%s: %s)", date, type(e).__name__, e)
        return None


def lines_posted(posted: dict, games: pd.DataFrame | None) -> int:
    """How many of `games` have a spread or a total in `posted` (the
    masthead's Lines posted tile)."""
    if games is None or games.empty:
        return 0
    return sum(1 for _, g in games.iterrows()
               if _num((posted.get((g["away"], g["home"])) or {}).get("home_line")) is not None
               or _num((posted.get((g["away"], g["home"])) or {}).get("total")) is not None)


def market_facts(odds: dict | None, away: str, home: str, unavailable: bool = False) -> list[str]:
    """The Game header's Spread, Total and Implied points facts. ESPN's line
    is the HOME team's (negative = home favoured); the favourite is quoted.
    A pre-tip line (posted_lines) says so in its hint. With no line:
    'Not posted', or 'Unavailable' when the lines feed itself failed."""
    none = "Unavailable" if unavailable else "Not posted"
    o = odds or {}
    hl, tot = _num(o.get("home_line")), _num(o.get("total"))
    if hl is not None:
        spread_txt = "PK" if hl == 0 else f"{home if hl < 0 else away} {-abs(hl):.1f}"
    else:
        spread_txt = none
    total_txt = f"{tot:.1f}" if tot is not None else none
    if hl is not None and tot is not None:
        implied = (f"{esc(away)} {tot / 2 + hl / 2:.1f}<span class='sep'> / </span>"
                   f"{esc(home)} {tot / 2 - hl / 2:.1f}")
    else:
        implied = none
    hint = PRE_TIP if o.get("pre_tip") else ""
    provider = str(o.get("provider") or "")
    if provider and provider.lower() != "nan":
        hint = f"{hint}, {provider}" if hint else provider
    return [fact("Spread", esc(spread_txt), hint), fact("Total", esc(total_txt)),
            fact("Implied points", implied)]


# ── Rest, starters and season minutes: the committed features ────────────────
# Display columns read from the date's committed history features (the rows
# the models were fed), so a B2B chip or a starter dot says what the model
# saw. A date without committed features shows none of them.
DISPLAY_FEATURES = ("rest_days", "b2b", "start_share_s", "min_std_s", "p1_mpg")


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def _display_features(path: str, content_hash) -> pd.DataFrame | None:
    df = C.read_parquet(Path(path))
    if df is None or df.empty:
        return None
    cols = ["person_id", "game_id", "team", *[c for c in DISPLAY_FEATURES if c in df.columns]]
    return df[cols].assign(game_id=df["game_id"].astype(str)).reset_index(drop=True)


def display_features(date) -> pd.DataFrame | None:
    """(person_id, game_id, team, rest_days, b2b, start_share_s, min_std_s,
    p1_mpg) from the date's committed features file, or None. Cached by the
    file's content hash, never its mtime."""
    p = P.features_path(date)
    if not p.exists():
        return None
    try:
        return _display_features(str(p), C.read_parquet_meta(p).get("content_hash"))
    except Exception as e:  # noqa: BLE001 - display extras never take a page down
        C.logger.warning("display features for %s unreadable (%s: %s)", date, type(e).__name__, e)
        return None


def rest_by_game(feats: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """{game_id: {team: rest_days}} (days since the team's previous game;
    1 = the second night of a back-to-back; NaN = no earlier game this
    season), as features.team_rest computed it for the model."""
    if feats is None or feats.empty or "rest_days" not in feats.columns:
        return {}
    first = feats.drop_duplicates(["game_id", "team"])
    out: dict[str, dict[str, float]] = {}
    for gid, team, r in zip(first["game_id"], first["team"], first["rest_days"]):
        out.setdefault(str(gid), {})[str(team)] = r
    return out


def b2b_by_game(rest: dict) -> dict[str, list[str]]:
    """{game_id: [teams on the second night of a back-to-back]}."""
    return {gid: [t for t, r in per.items() if _num(r) == 1.0] for gid, per in rest.items()
            if any(_num(r) == 1.0 for r in per.values())}


def rest_text(days) -> str:
    d = _num(days)
    if d is None:
        return "First game"
    if d <= 1:
        return "Back-to-back"
    off = int(d) - 1
    return f"{off} day{'s' if off != 1 else ''} off"


def rest_fact(rest: dict | None, away: str, home: str) -> str | None:
    """The Game header's Rest fact ('DEN Back-to-back / SAS 1 day off'), or
    None without committed features for the game."""
    if not rest or (away not in rest and home not in rest):
        return None
    val = (f"{esc(away)} {esc(rest_text(rest.get(away)))}<span class='sep'> / </span>"
           f"{esc(home)} {esc(rest_text(rest.get(home)))}")
    return fact("Rest", val)


def season_mpg(feats: pd.DataFrame | None) -> dict[tuple[int, str], float]:
    """{(person_id, game_id): minutes per game this season, else last
    season's} from the committed features (min_std_s, then p1_mpg)."""
    if feats is None or feats.empty or "min_std_s" not in feats.columns:
        return {}
    m = pd.to_numeric(feats["min_std_s"], errors="coerce")
    if "p1_mpg" in feats.columns:
        m = m.fillna(pd.to_numeric(feats["p1_mpg"], errors="coerce"))
    keep = m.notna().to_numpy()
    return dict(zip(zip(feats["person_id"].astype("int64")[keep], feats["game_id"].astype(str)[keep]),
                    m[keep].astype(float)))


STARTERS = 5
STARTER_MIN_PPLAY = 0.5          # a player more likely out than in is never a projected starter


def projected_starters(frame: pd.DataFrame | None, feats: pd.DataFrame | None
                       ) -> set[tuple[int, str]]:
    """{(person_id, game_id)} of each team's five projected starters: among
    players not listed Out and at least STARTER_MIN_PPLAY to play, the five
    with the highest share of starts this season (start_share_s, from the
    committed features), projected minutes breaking ties and ranking players
    with no starts history (opening night: projected minutes alone)."""
    if frame is None or frame.empty:
        return set()
    f = frame[["person_id", "game_id", "team", "inj_status", "p_play", "min"]].copy()
    f["game_id"] = f["game_id"].astype(str)
    f["person_id"] = f["person_id"].astype("int64")
    share = {}
    if feats is not None and len(feats) and "start_share_s" in feats.columns:
        share = dict(zip(zip(feats["person_id"].astype("int64"), feats["game_id"].astype(str)),
                         pd.to_numeric(feats["start_share_s"], errors="coerce")))
    f["_s"] = [share.get(k, np.nan) for k in zip(f["person_id"], f["game_id"])]
    ok = (f["inj_status"] != "Out") & (pd.to_numeric(f["p_play"], errors="coerce") >= STARTER_MIN_PPLAY)
    f = f[ok.to_numpy()]
    f = f.assign(_s=f["_s"].fillna(-1.0), _m=pd.to_numeric(f["min"], errors="coerce").fillna(0.0))
    top = (f.sort_values(["game_id", "team", "_s", "_m", "person_id"],
                         ascending=[True, True, False, False, True], kind="mergesort")
           .groupby(["game_id", "team"], sort=False).head(STARTERS))
    return set(zip(top["person_id"].astype(int), top["game_id"]))




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
    dot = ('<span class="bv-sd" title="Projected starter"></span>'
           if p.get("starter") is True or p.get("starter") == 1 else "")
    return f'<td class="l pl">{img}{esc(p["name"])}{dot}{pos}</td>'


def _caption(team: str, asof_note: str = "", team_note: str = "") -> str:
    t = teams.get(team)
    chip = (f'<span class="bv-chip warn" title="Injury statuses for this team are unknown">'
            f'{esc(team_note)}</span>') if team_note else ""
    asof = f'<span class="asof">{esc(asof_note)}</span>' if asof_note else ""
    return (f'<div class="bv-pt-cap"><span class="ab">{esc(team)}</span>'
            f'<span class="nm">{esc(t.full_name)}</span>{chip}{asof}</div>')


# Column widths are CLASSES (theme.py sizes them, and narrows them on phones);
# an inline style would beat the stylesheet.
ROSTER_COLGROUP = ('<colgroup><col class="c-badge"><col class="c-player">'
                   + "".join("<col>" for _ in STAT_COLS) + "</colgroup>")


def render_team_table(team: str, players: pd.DataFrame, asof_note: str = "",
                      lead: str = "min", team_note: str = "") -> str:
    """One team's ROSTER, no projections (preseason, or a game not published
    yet). `players` columns: person_id, name, position, status (badge text or
    ''), out (bool), group ('' or a group label), and any of the STAT_COLS
    keys (missing -> the sentinel); `reason` greys an Out row with it.
    `team_note` is a caption chip (where no injury report is joined, the
    page says so rather than leave every badge blank)."""
    t = teams.get(team)
    head = "".join(f'<th>{esc(lab)}</th>' for _, lab in STAT_COLS)
    rows, last_grp = [], None
    for _, p in players.iterrows():
        grp = p.get("group") or ""
        if grp and grp != last_grp:
            rows.append(f'<tr class="grp"><td class="l" colspan="{len(STAT_COLS) + 2}">{esc(grp)}</td></tr>')
            last_grp = grp
        reason = p.get("reason")
        if p.get("out") and isinstance(reason, str) and reason.strip() not in ("", "-"):
            cells = f'<td class="l reason" colspan="{len(STAT_COLS)}">{esc(reason)}</td>'
        else:
            cells = "".join(
                f'<td class="{"lead" if k == lead else ""}">'
                f'{"" if p.get("out") else fmt_num(p.get(k), 1)}</td>' for k, _ in STAT_COLS)
        rows.append(f'<tr class="{"out" if p.get("out") else ""}">'
                    f'<td class="l">{status_badge(p.get("status"))}</td>{_player_cell(p)}{cells}</tr>')
    return (
        f'<div class="bv-pt" style="--team:{t.hex1}">{_caption(team, asof_note, team_note)}'
        f'<table>{ROSTER_COLGROUP}'
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


def mark_projected(schedule: pd.DataFrame | None, dates=None) -> pd.DataFrame | None:
    """The schedule with props.PROJECTED_COL set to the pipeline's own answer
    (projected_ids, per date), which is how a page hands it to the paste:
    props cannot import pipeline (it reaches the network), and without the
    mark a line for a game the site does not project (a play-in or a
    postponed game) would be announced as loaded. `dates` (the paste's
    routing dates) limits the marking to those dates; rows on other dates
    stay unmarked (None), where props keeps its own type rule."""
    if schedule is None or schedule.empty:
        return schedule
    gd = pd.to_datetime(schedule["game_date"]).dt.date
    want = sorted(set(gd)) if dates is None else sorted({pd.Timestamp(str(d)).date() for d in dates})
    ids: set[str] = set()
    for d in want:
        ids |= projected_ids(schedule, d)
    on = gd.isin(want).to_numpy()
    mark = [bool(g in ids) if o else None
            for g, o in zip(schedule["game_id"].astype(str), on)]
    return schedule.assign(**{props.PROJECTED_COL: pd.Series(mark, index=schedule.index, dtype=object)})


PPLAY_SHOW_BELOW = 0.90          # spec 5: P(play) only when listed or below 90%
DEEP_BENCH_MIN = 10.0            # projected minutes under this: the collapsed deep bench
KEY_ABSENCE_MIN = 15.0           # the card's "Out:" names 15+ MPG players (spec 5)
# The Game table's medians: every STAT_COLS column but MIN, which is the mean.
MEDIAN_STATS = tuple(k for k, _ in STAT_COLS if k != "min")
VIEW_COLS = ["person_id", "game_id", "name", "position", "status", "reason", "out", "pplay",
             "section", "starter", "p_play", *(k for k, _ in STAT_COLS)]
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
              positions: dict | None = None, starters=None) -> pd.DataFrame:
    """One team's projection rows as the Game table shows them (VIEW_COLS).

    rows       projection-frame rows of ONE team in ONE game
    medians    (person_id, game_id, med_<stat>) from the date's shared
               simulator provider (table_medians); the stat columns are those
               mid-p medians, MIN stays the mean (rows["min"])
    positions  {person_id: position} from the live roster (label only)
    starters   {(person_id, game_id)} projected starters (projected_starters)

    status: context.display_status (Out refined to Two-way, G League, Not with
    team, Suspended; unknown labelled, never blank), else "Two-way" from the
    roster flag. out: listed Out or marked out; no numbers. pplay: '' unless
    the player is listed or below 90% to play. section: main (the projected
    starters, and anyone at 10+ projected minutes), deep (under 10) or out.
    Sorted by section; within main the starters first, each group by
    projected minutes (spec 5: starters by minutes, then the bench).
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
    starters = starters or set()
    start = np.array([(int(a), str(b)) in starters for a, b in zip(r["person_id"], r["game_id"])],
                     dtype=bool) & ~out
    v = pd.DataFrame({
        "person_id": r["person_id"].astype("int64"), "game_id": r["game_id"].astype(str),
        "name": r["player_name"].astype(str),
        "position": r["person_id"].map(positions or {}).fillna("").astype(str),
        "status": _badges(r, shown), "reason": shown["reason"], "out": out,
        "pplay": [_pct(x) if s else "" for x, s in zip(p, show_p)],
        "section": np.where(out, "out", np.where((mins >= DEEP_BENCH_MIN) | start, "main", "deep")),
        "starter": start, "p_play": p, "min": mins})
    keys = pd.MultiIndex.from_arrays([v["person_id"], v["game_id"]])
    med = (medians.drop_duplicates(["person_id", "game_id"], keep="last")
           .assign(game_id=lambda m: m["game_id"].astype(str))
           .set_index(["person_id", "game_id"]) if medians is not None and len(medians) else None)
    for s in MEDIAN_STATS:
        col = f"med_{s}"
        v[s] = med[col].reindex(keys).to_numpy() if med is not None and col in med else np.nan
    v["_s"] = v["section"].map({s: i for i, s in enumerate(_SECTIONS)})
    v["_b"] = ~v["starter"]
    v = v.sort_values(["_s", "_b", "min", "name"], ascending=[True, True, False, True],
                      kind="mergesort", na_position="last").drop(columns=["_s", "_b"]).reset_index(drop=True)
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


COLGROUP = ('<colgroup><col class="c-badge"><col class="c-player"><col class="c-pp">'
            + "".join("<col>" for _ in STAT_COLS) + "</colgroup>")


# ── After tip: the box score ─────────────────────────────────────────────────
DNP_TEXT = "Did not play"


@dataclass(frozen=True)
class Actuals:
    """A game's box score for the page: `by_person` {person_id: box row}
    for every player who PLAYED (live.Box.lines, the logs' schema); `final`
    the box score's own final flag. Without a row, a player has not played
    (yet, or at all once final)."""
    final: bool
    by_person: dict


def actuals_from(box) -> Actuals | None:
    """Actuals from a live.Box (or None)."""
    if box is None:
        return None
    lines = box.lines
    rows = {} if lines is None or lines.empty else {
        int(r["person_id"]): r for r in lines.to_dict(orient="records")}
    return Actuals(bool(box.final), rows)


def actual_of(stat: str, row: dict | None) -> float | None:
    """A board stat's value from a box row (props.actual_value, the record's
    own composition: a missing component is no value, never zero). MIN is
    the precise seconds over 60."""
    if row is None:
        return None
    if stat == "min":
        sec = _num(row.get("sec"))
        return None if sec is None else sec / 60.0
    return props.actual_value(stat, row)


def line_actuals(recs: list[dict], actuals: Actuals | None) -> list[dict]:
    """Each pasted-line row with what happened: `actual` (the stat so far,
    or final), `settle` 'void' (final and he did not play: PrizePicks voids
    the pick) or 'exact' (final and the value equals the line: neither a win
    nor a loss, never called a push), `final`."""
    if actuals is None:
        return recs
    out = []
    for r in recs:
        row = actuals.by_person.get(int(r["person_id"]))
        val = actual_of(str(r.get("stat")), row)
        settle = None
        if actuals.final and row is None:
            settle = "void"
        elif actuals.final and val is not None and _num(r.get("line")) is not None \
                and round(val, 4) == round(float(r["line"]), 4):
            settle = "exact"
        out.append({**r, "actual": val, "settle": settle, "final": actuals.final})
    return out


def _actual_row(p, actuals: Actuals | None) -> str:
    """The row under a player's projected line after tip: his box-score line
    so far (or final), or 'Did not play' once final. A listed-Out player who
    did not play gets none (his Out row already says it); a player not in
    yet during a live game gets none."""
    if actuals is None:
        return ""
    row = actuals.by_person.get(int(p["person_id"]))
    lab = f'<td class="l"></td><td class="l lab">{"Final" if actuals.final else "So far"}</td>'
    if row is None:
        if not actuals.final or p.get("out"):
            return ""
        return f'<tr class="act">{lab}<td class="dnp" colspan="{len(STAT_COLS) + 1}">{DNP_TEXT}</td></tr>'
    cells = "".join(f'<td>{esc(props_ui.actual_text(k, actual_of(k, row)))}</td>'
                    for k, _ in STAT_COLS)
    return f'<tr class="act">{lab}<td></td>{cells}</tr>'


def _proj_row(p, actuals: Actuals | None = None) -> str:
    pid = int(p["person_id"])
    badge = f'<td class="l">{status_badge(p.get("status"))}</td>'
    if p.get("out"):
        reason = p.get("reason")
        if not isinstance(reason, str) or reason.strip() in ("", "-"):
            reason = p.get("status") or "Out"
        return (f'<tr class="out" data-pid="{pid}">{badge}{_player_cell(p)}'
                f'<td class="l reason" colspan="{len(STAT_COLS) + 1}">{esc(reason)}</td></tr>'
                + _actual_row(p, actuals))
    cells = "".join(f'<td class="{"lead" if k == "min" else ""}">{fmt_num(p.get(k), 1)}</td>'
                    for k, _ in STAT_COLS)
    return (f'<tr data-pid="{pid}">{badge}{_player_cell(p)}'
            f'<td class="pp">{esc(p.get("pplay") or "")}</td>{cells}</tr>' + _actual_row(p, actuals))


def render_projection_table(team: str, players: pd.DataFrame, asof_note: str = "",
                            team_note: str = "", actuals: Actuals | None = None) -> str:
    """One team's projected players (team_view rows). MIN, the projected
    average, is the bold lead column; the nine stats after it are mid-p
    medians under an "if he plays" header; P(play) only where team_view
    filled it; the projected starters marked with a dot; the deep bench
    collapsed under its own summary; Out rows greyed at the bottom with the
    reason and no numbers. After tip (`actuals`), each player's box-score
    line sits under his projected one."""
    t = teams.get(team)
    sec = players["section"] if len(players) else pd.Series(dtype=object)
    main, deep, out = (players[sec == s] for s in _SECTIONS)
    head = ('<thead><tr class="hgrp"><th colspan="3"></th><th class="avg">Avg</th>'
            f'<th class="ifp" colspan="{len(STAT_COLS) - 1}">Median if he plays</th></tr>'
            '<tr><th class="l"></th><th class="l">Player</th><th>P(play)</th>'
            + "".join(f"<th>{esc(lab)}</th>" for _, lab in STAT_COLS) + "</tr></thead>")

    def body(part):
        return "".join(_proj_row(p, actuals) for _, p in part.iterrows())
    parts = [f'<table>{COLGROUP}{head}<tbody>{body(main)}</tbody></table>']
    if len(deep):
        n = len(deep)
        parts.append(
            f'<details class="bv-deep"><summary>Deep bench: {n} player{"s" if n != 1 else ""} '
            f'under {DEEP_BENCH_MIN:g} projected minutes</summary>'
            f'<table>{COLGROUP}<tbody>{body(deep)}</tbody></table></details>')
    if len(out):
        parts.append(
            f'<table class="bv-pt-out">{COLGROUP}<tbody><tr class="grp">'
            f'<td class="l" colspan="{len(STAT_COLS) + 3}">Out ({len(out)})</td></tr>'
            f'{body(out)}</tbody></table>')
    return (f'<div class="bv-pt" style="--team:{t.hex1}">{_caption(team, asof_note, team_note)}'
            f'{"".join(parts)}</div>')


# ── The slate's player search ────────────────────────────────────────────────
def player_options(frame: pd.DataFrame | None) -> list[tuple[int, str]]:
    """[(person_id, 'Name, TEAM')] of every projected player on the date, by
    name then id: a fixed order, so the keyed search box's options never
    reorder between reruns (its id is hashed from them)."""
    if frame is None or frame.empty:
        return []
    f = frame.drop_duplicates("person_id").sort_values(["player_name", "person_id"], kind="mergesort")
    return [(int(p), f"{n}, {t}") for p, n, t in zip(f["person_id"], f["player_name"], f["team"])]


def player_panel(frame: pd.DataFrame, person_id: int, date: dt.date, keep: dict,
                 feats: pd.DataFrame | None = None) -> str:
    """One searched player's projected line, drawn exactly as his row on the
    Game page (team_view, render_projection_table; his medians from his own
    rows, whose simulation is seeded per player and game, so they equal the
    Game page's), with a link to his game."""
    rows = frame[frame["person_id"].astype("int64") == int(person_id)]
    if rows.empty:
        return ""
    r = rows.iloc[0]
    try:
        med = table_medians(rows)
    except Exception as e:  # noqa: BLE001 - minutes still show; the medians read the sentinel
        C.logger.warning("player panel medians failed (%s: %s)", type(e).__name__, e)
        med = None
    view = team_view(rows, med, starters=projected_starters(frame, feats))
    away, home = (r["opp"], r["team"]) if bool(r["home"]) else (r["team"], r["opp"])
    note = f"{away} at {home}, {fmt_tip(r['tip_utc'])}"
    link = (f'<a class="bv-back" href="{esc(game_url(date, away, home, keep))}" target="_self">'
            f"Open {esc(away)} at {esc(home)}</a>")
    return render_projection_table(str(r["team"]), view, asof_note=note) + link


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
                 show: int = 2, mpg: dict | None = None) -> dict[str, str]:
    """{game_id: '<div>Out: X, Y +2</div>'}: players the REPORT lists Out (a
    visitor's mark-out is not a listing) who average `min_minutes`+ minutes
    a game (spec 5: 15+ MPG), two-way and G League excluded, most minutes
    first.

    `mpg` {(person_id, game_id): minutes per game} is season_mpg (this
    season, else last season, from the committed features). The projection's
    own minutes are E[MIN | plays] computed WITH the Out status, which reads
    a long absence down: about half of the 15-20 MPG players listed Out in
    2025-26 projected under 15. They stand in only where the features give
    no MPG (a rookie, or no committed features)."""
    if frame is None or frame.empty:
        return {}
    label = CTX.display_status(frame)["status"]
    mins = pd.to_numeric(frame["min"], errors="coerce")
    if mpg:
        season = pd.Series([mpg.get((int(p), str(g)), np.nan)
                            for p, g in zip(frame["person_id"], frame["game_id"])],
                           index=frame.index, dtype="float64")
        mins = season.fillna(mins)
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


def card_subs(served: "Served | None", mpg: dict | None = None) -> dict[str, str]:
    """The extra line under each slate card: the key absences, or why a game
    has no projection yet."""
    if served is None:
        return {}
    subs = key_absences(served.frame, mpg=mpg)
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
_STALE_WHY = ((P.LIVE_FRAME_FAILED, "the live update hit an error"),
              ("NoReport", "no injury report could be read"),
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


def _et_time(t) -> dt.datetime | None:
    try:
        ts = pd.Timestamp(t)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return to_et(ts.to_pydatetime())


def _stamp(t: dt.datetime | None, today: dt.date | None = None) -> str:
    """'11:17 AM ET', with ', Feb 10' when not the date shown."""
    if t is None:
        return ""
    day = "" if today is not None and t.date() == today else f", {t.strftime('%b')} {t.day}"
    return f"{fmt_tip(t)}{day}"


def frozen_basis(rows: pd.DataFrame) -> tuple[dt.datetime | None, dt.datetime | None]:
    """(when the frozen rows were computed, the title time of the report they
    were computed with), ET; either None when the rows do not say."""
    if rows is None or rows.empty:
        return None, None
    comp = pd.to_datetime(rows["computed_at"], utc=True).dropna()
    rep = pd.to_datetime(rows["report_ts_et"], utc=True).dropna()
    return (_et_time(comp.max()) if len(comp) else None,
            _et_time(rep.max()) if len(rep) else None)


def frozen_caption(rows: pd.DataFrame, date: dt.date | None = None) -> str:
    """The table caption of a frozen game: 'Frozen at tip: projected 11:17
    AM ET with the 10:45 AM ET report', so a reader can see how old the
    numbers are (the freeze shows the bot's LAST pre-tip file, however old)."""
    comp, rep = frozen_basis(rows)
    out = "Frozen at tip"
    if comp is not None:
        out += f": projected {_stamp(comp, date)}"
        if rep is not None:
            out += f" with the {_stamp(rep, date)} report"
    return out


def projection_notes(served: "Served", game_id: str | None = None,
                     today: dt.date | None = None) -> list[str]:
    """The notices a page owes the reader about the frame it shows (html):
    stale (the committed stand-in), frozen at tip (when and with which report
    the frozen numbers were made, and whether a later report has been read
    since), provisional and why, the pipeline's own notes, and how far the
    box scores go. With no rows to show there is no stale notice here:
    empty_notice says why the page is empty, once."""
    info = served.info
    rows = served.frame if game_id is None else served.game(game_id)
    out: list[str] = []
    if info.stale and len(rows):
        when = ""
        if rows["computed_at"].notna().any():
            t = to_et(pd.Timestamp(rows["computed_at"].max()).to_pydatetime())
            when = f" at {fmt_tip(t)} on {t.strftime('%b')} {t.day}"
        out.append(render_notice(
            f"<b>Showing the published projections.</b> The live update could not run "
            f"({esc(stale_reason(info.stale))}), so these are the numbers the bot "
            f"published{esc(when)}."))
    if game_id is not None and game_id in info.frozen:
        comp, rep = frozen_basis(rows)
        made = ""
        if comp is not None:
            made = f" published at {_stamp(comp, info.date)}"
            if rep is not None:
                made += f" with the {_stamp(rep, info.date)} injury report"
        later = ""
        seen = _et_time(info.report_ts_et) if info.report_ts_et is not None else None
        if seen is not None and rep is not None and seen > rep:
            later = (f" Injury reports after that one (the latest read is {_stamp(seen, info.date)}) "
                     "are not reflected.")
        out.append(render_notice(
            f"<b>Frozen at tip.</b> These are the projections{esc(made)}, the last version "
            f"published before tip-off. They no longer change, and mark-outs no longer apply."
            f"{esc(later)}"))
    elif game_id is not None and game_id in served.past and len(rows):
        out.append(render_notice(
            "<b>Tipped on provisional projections.</b> Shown as published; a provisional "
            "projection is never frozen."))
    if game_id is None:
        provisional = info.provisional
    elif len(rows):
        provisional = bool(rows["provisional"].any())
    else:
        # No rows (withheld, unbuilt, unpublished): the game's OWN flag, never
        # the date's, which another game's gap can set. A game with no flag
        # of its own (past its freeze point, or the committed stand-in) has
        # no numbers to call provisional; empty_notice says why it is empty.
        provisional = info.provisional_by_game.get(str(game_id), False)
    if provisional:
        why = []
        if today is not None and info.date > today:
            why.append("Projections for a future date update with the morning run, the injury "
                       "report and the lines.")
        # A game's OWN gaps on /Game (provisional is per game); the date's
        # union on the slate. A gap can be a game still in progress, so it is
        # never called final here.
        if game_id is not None:
            if info.completeness_by_game.get(str(game_id)):
                why.append("Earlier game(s) of these teams are not in the box scores yet, so "
                           "the history misses them.")
        elif info.completeness:
            why.append("Earlier game(s) of teams playing on this date are not in the box scores "
                       "yet, so the history misses them.")
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


_PUBLISHES = "The bot publishes them once the history features, the injury report and the lines are in."


def empty_notice(served: "Served", game_id: str | None = None, game_ids=()) -> str:
    """The ONE notice for a game (game_id) or a date (game_ids: its projected
    games) with no projection rows to show, saying why:
      no line yet (withheld)        "Projections post when the line does"
      past its freeze point         none was published before tip, and none
                                    ever will be (the bot never writes a
                                    tipped game): no promise
      the live update failed        not up yet, and why the live update
                                    could not run
      otherwise                     not published yet, and what it waits for
    """
    info = served.info
    past = {str(g) for g in served.past}
    if game_id is not None:
        gid = str(game_id)
        if gid in info.withheld:
            return render_notice(f"<b>{esc(P.NO_LINE)}.</b> The models were trained with the "
                                 "spread and total, so a game without them is not projected.")
        if gid in past:
            return render_notice("<b>No projection was published before tip.</b> The bot never "
                                 "projects a game after tip-off, so this game has none.")
        if info.stale:
            return render_notice(
                "<b>Projections are not up yet, and the live update could not run</b> "
                f"({esc(stale_reason(info.stale))}). The bot publishes them once the game's "
                "history features and the injury report are in.")
        return render_notice("<b>Projections not published yet for this game.</b> The bot "
                             "publishes them once the game's history features and the injury "
                             "report are in.")
    ids = {str(g) for g in game_ids}
    if ids and ids <= past:
        return render_notice("<b>No projections were published for this date before tip.</b> "
                             "The bot never projects a game after tip-off.")
    if info.withheld and info.reason is None:
        return render_notice(f"<b>{esc(P.NO_LINE)}.</b> The models were trained with the spread "
                             "and total, so a game is projected once both are posted.")
    if info.stale:
        return render_notice(
            "<b>Projections are not up yet, and the live update could not run</b> "
            f"({esc(stale_reason(info.stale))}). {_PUBLISHES}")
    return render_notice(f"<b>Projections not published yet for this date.</b> {_PUBLISHES}")


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
    return ('<div class="bv-trust"><table><colgroup><col class="c-stat">'
            '<col class="c-tier"><col class="c-edge"><col></colgroup>'
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
    reason. ANY exception is that failure: live.injury_report parses
    third-party PDFs, and a parser error on a novel file must cost the live
    upgrade, never the page. A failure that escaped the live layer is handed
    on as a report-less Live, so live_projections does not fetch again (and
    raise again): it serves the committed projections (NoReport)."""
    try:
        rep = LV.injury_report(sched)
    except RuntimeError:
        return None, ("unavailable",)
    except Exception as e:  # noqa: BLE001 - see the docstring
        C.logger.warning("injury report failed outside the live layer", exc_info=True)
        return (LV.Live(None, None, stale=True, source="none",
                        error=f"{type(e).__name__}: {str(e)[:160]}"), ("unavailable",))
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
    try:
        sched = LV.schedule().value
    except RuntimeError:
        sched = None
    games = P.state_games(sched, date) if sched is not None else None
    past = frozenset(P.frozen_ids(games, now)) if games is not None else frozenset()
    if not (P.features_path(date).exists() or P.projections_path(date).exists()):
        # Nothing committed: no fetch. `past` still says which games tipped,
        # so the page never promises a projection that can no longer come.
        frame, info = P.live_projections(date, now)
        return Served(frame, info, now, past=past)
    todo = games is None or bool(set(games["game_id"]) - past)
    report, rkey, odds = None, ("not needed",), None
    if todo:
        # The two live reads at once: a cold page waits for the slower one
        # (each has its own budget and circuit breaker), never their sum.
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_report, f_odds = pool.submit(_report_key, sched), pool.submit(LV.odds_on, date)
            report, rkey = f_report.result()
            odds = f_odds.result()

    def frame_for(m: tuple) -> tuple[pd.DataFrame, P.Info]:
        return _live_frame(date.isoformat(), rkey, _odds_key(odds), m, tuple(sorted(past)),
                           _now=now, _report=report, _odds=odds, _sched=sched)
    frame, info = frame_for(())
    eff = effective_marks(frame, marks)
    if eff:
        frame, info = frame_for(eff)
    return Served(frame, info, now, eff, past)


@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def _known_names(season: str, date_iso: str) -> pd.DataFrame | None:
    try:
        return P.known_names(season, dt.date.fromisoformat(date_iso))
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        C.logger.warning("known names unreadable (%s: %s)", type(e).__name__, e)
        return None


def roster_statuses(game_row, sched: pd.DataFrame, roster: pd.DataFrame | None,
                    now: dt.datetime | None = None) -> pd.DataFrame | None:
    """Injury statuses for the ROSTER of one game that has no projection rows
    (withheld for its line, or not published yet), from the live report, so
    the page draws the statuses it has rather than a blank badge column.

    The report goes through context.context_for, the projections' own path
    (team-scoped name resolution with every spelling a player has carried,
    the report's cutoff and 30-hour reach), and context.display_status
    labels it: (person_id, team, status, reason, out). Unknown stays
    labelled ('Report not filed', 'No report yet'). None when the report
    could not be read (the page then says statuses are not shown)."""
    if roster is None or roster.empty or sched is None:
        return None
    rep, _ = _report_key(sched)
    report = getattr(rep, "value", None)
    if report is None:
        return None
    away, home, gid = game_row["away"], game_row["home"], str(game_row["game_id"])
    r = roster[roster["team"].isin([away, home])]
    if r.empty:
        return None
    games = pd.DataFrame([{"game_id": gid, "game_date": game_row["game_date"],
                           "tip_utc": game_row["tip_utc"], "away": away, "home": home}])
    rows = pd.DataFrame({"game_id": gid, "team": r["team"].to_numpy(),
                         "person_id": r["person_id"].astype("int64").to_numpy(),
                         "tip_utc": game_row["tip_utc"],
                         "player_name": r["name"].astype(str).to_numpy()})
    d = pd.Timestamp(game_row["game_date"]).date()
    try:
        ctx = CTX.context_for(rows, games, sched, report, now or now_utc(),
                              _known_names(season_for_date(d), d.isoformat()))
    except Exception as e:  # noqa: BLE001 - statuses are an extra on a page that has none
        C.logger.warning("roster statuses failed (%s: %s)", type(e).__name__, e)
        return None
    shown = CTX.display_status(ctx)
    return pd.DataFrame({"person_id": ctx["person_id"].astype("int64").to_numpy(),
                         "team": ctx["team"].to_numpy(), "status": shown["status"].to_numpy(),
                         "reason": shown["reason"].to_numpy(),
                         "out": (ctx["inj_status"] == "Out").to_numpy(dtype=bool)})


def table_medians(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Mid-p medians (MEDIAN_STATS) for every row of the date's frame, from
    the shared provider (simprov.provider_for), so the Game tables and the
    board read one simulation."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["person_id", "game_id", *(f"med_{s}" for s in MEDIAN_STATS)])
    return simprov.provider_for(frame, sim_params()).medians(MEDIAN_STATS)
