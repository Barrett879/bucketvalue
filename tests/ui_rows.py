"""Hand-built board rows for the props_ui HTML tests (not a test module).

Every row has props.TABLE_COLUMNS, the shape compare() gives a line. The
pasted rows are PrizePicks rows as a paste leaves them (book "prizepicks",
origin "pasted", no prices, no feed stamps); the feed rows carry their book,
origin "feed" and the time their book's line was fetched. Players, lines and
numbers are invented; the dates are the serve fixture's 2026-03-12, a fixed
past date (EDT, UTC-4). Nothing here reads a clock.

feed_off_outputs(props_ui) renders a fixed set of these rows through the
props_ui functions as they were called before the feed (no feed argument).
tests/fixtures/props_ui/feed_off_golden.json holds what the pre-feed
props_ui (build 2's tree, before stage S8) gave for them, so the feed-off HTML
is checked byte for byte (tests/test_props_ui_feed.py).
"""
from __future__ import annotations

import pandas as pd

from bucketlib import props

DAY = "2026-03-12"
EARLY, LATE = "0022500953", "0022500958"        # PHI at DET 7 PM ET, DEN at SAS 9 PM ET
NINE = pd.Timestamp("2026-03-12T13:02:00Z")      # 9:02 AM ET: the morning pull
NOON = pd.Timestamp("2026-03-12T16:00:00Z")      # 12:00 PM ET: a later refresh
LAST_NIGHT = pd.Timestamp("2026-03-12T01:41:00Z")   # 9:41 PM ET on Mar 11: a paste
NAN = float("nan")


def row(**kw) -> dict:
    """One compare() row, a pasted PrizePicks Points line unless `kw` says
    otherwise."""
    base = {c: None for c in props.TABLE_COLUMNS}
    base.update(
        person_id=1001, game_id=EARLY, game_date=DAY, player="Ann Able", team="DET", opp="PHI",
        stat="pts", stat_label="Points", line=24.5, model=26.3, mean=26.0, gap=1.8,
        p_over=0.62, p_push=0.0, p_under=0.38, q_more=0.62, q_less=0.38, confidence=0.12,
        lean="More", sides="both", odds_type="standard", tier="strong", tier_edge=3.1,
        tier_note=None, games_prior=50, thin=False, void_risk=NAN, mu_min=34.0,
        void_note=False, source="model", how="exact", pasted_name="Ann Able",
        model_version="v1", as_of="2026-03-12T16:20:00+00:00", provisional=False,
        adjusted=False, tip_utc="2026-03-12T23:00:00+00:00", tie_possible=False, pick=None,
        book="prizepicks", book_title="PrizePicks", origin="pasted", price_over=NAN,
        price_under=NAN, fetched_at=None, last_update=None)
    base.update(kw)
    return base


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=props.TABLE_COLUMNS)


def pasted_rows() -> pd.DataFrame:
    """Five pasted rows that reach every per-row branch of the ledger: a More
    lean on a bench role (the void-risk tag), a Less-only Goblin from the
    baseline, an Even typed pick that can land exactly on its line, a Demon
    combo with a measured void risk, and a row with no model value. A name
    carries HTML, which must come out escaped."""
    return frame([
        row(void_note=True, mu_min=18.0),
        row(person_id=1002, player='Bo "B" <Baker> & Co', team="PHI", opp="DET", stat="reb",
            stat_label="Rebounds", line=8.5, model=7.9, mean=8.0, gap=-0.6, q_more=0.41,
            q_less=0.59, lean="Less", sides="less", odds_type="goblin", tier="moderate",
            tier_edge=1.2, source="baseline"),
        row(person_id=1003, player="Cal Cole", stat="ast", stat_label="Assists", line=6.0,
            model=6.0, mean=6.1, gap=0.0, p_push=0.12, q_more=0.5, q_less=0.5, lean="Even",
            tier="weak", tier_edge=0.4, tie_possible=True, pick="more", source="composite"),
        row(person_id=1004, player="Dee Dunn", game_id=LATE, team="SAS", opp="DEN",
            stat="pra", stat_label="Pts+Rebs+Asts", line=35.5, model=38.2, mean=37.9, gap=2.7,
            odds_type="demon", tier="none", tier_edge=None, source="model_low_confidence",
            void_risk=0.08, tip_utc="2026-03-13T01:00:00+00:00"),
        row(person_id=1005, player="Eve Eads", game_id=LATE, team="DEN", opp="SAS",
            stat="fg3m", stat_label="3-PT Made", line=2.5, model=NAN, mean=NAN, gap=NAN,
            lean=None, tier="unmeasured", tier_edge=None, tip_utc="2026-03-13T01:00:00+00:00"),
    ])


def after_tip(rows: pd.DataFrame) -> list[dict]:
    """The first three rows as store.line_actuals leaves them once final:
    a value, a player who did not play, and a value on the line."""
    recs = rows.head(3).to_dict(orient="records")
    recs[0].update(actual=27.0, settle=None, final=True)
    recs[1].update(actual=None, settle="void", final=True)
    recs[2].update(actual=6.0, settle="exact", final=True)
    return recs


def meta_full() -> dict:
    """A paste's meta that says something in every note."""
    return {"posted": 9, "matched": 5, "unmatched": 2, "ambiguous": 1, "n_refused": 1,
            "unprojected": 0, "unmatched_names": ["Zed Zane", "Yan <b>Yu</b>"],
            "ambiguous_names": ["Jalen Williams"], "refused": {"Dunks": 1},
            "refused_why": {"Dunks": "not priced"}, "unmapped": {"Fouls": 2},
            "unprojected_why": {"no projection for this player and game": 1},
            "nonexact": [("Cade", "Cade Cunningham", "prefix")],
            "pasted_utc": LAST_NIGHT}


def split(rows: pd.DataFrame) -> tuple:
    """(shown, hidden, thin, void) out of pasted_rows(): two shown, the Even
    row hidden, the row without a model value thin, the Demon void."""
    return (rows.iloc[[0, 1]].reset_index(drop=True), rows.iloc[[2]].reset_index(drop=True),
            rows.iloc[[4]].reset_index(drop=True), rows.iloc[[3]].reset_index(drop=True))


def gap_rows() -> pd.DataFrame:
    """Nine standard lines and a Goblin with gaps, for the discrepancy strip
    and its '+N more' table."""
    out = []
    for i in range(9):
        lean = "More" if i % 2 == 0 else "Less"
        gap = (0.5 + i) * (1 if lean == "More" else -1)
        out.append(row(person_id=2000 + i, player=f"Player {i}", line=20.5, model=20.5 + gap,
                       gap=gap, lean=lean))
    out.append(row(person_id=2100, player="Gob Lin", line=10.5, model=19.5, gap=9.0,
                   odds_type="goblin"))
    return frame(out)


def feed_off_outputs(props_ui) -> dict:
    """{name: html}: the props_ui functions on these rows, called as the
    pages called them before the feed (no feed argument anywhere)."""
    rows = pasted_rows()
    shown, hidden, thin, void = split(rows)
    meta = meta_full()
    gaps = gap_rows()
    out = {
        "ledger": props_ui.ledger_html(rows, {1001: 2}),
        "ledger_empty": props_ui.ledger_html(rows.iloc[:0]),
        "player_lines": props_ui.render_player_lines(rows),
        "player_lines_after_tip": props_ui.render_player_lines(after_tip(rows)),
        "player_lines_none": props_ui.render_player_lines([]),
        "notes": "\n".join(props_ui.notes_html(meta, shown, hidden, thin, void)),
        "notes_plain": "\n".join(props_ui.notes_html({"posted": 2, "matched": 2},
                                                     shown, None, None, None)),
        "window": props_ui.window_html(shown, hidden, thin, void, "March 12"),
        "none_matched": props_ui.none_matched_html(3, "March 12"),
        "none_matched_one": props_ui.none_matched_html(1, "PHI at DET"),
        "count_header": props_ui.count_header(shown, hidden, thin, void),
        "odds_footnote": props_ui.odds_footnote(rows),
        "reboot": props_ui.reboot_note(rows),
        "discrepancies_window": props_ui.discrepancies_window(gaps, "March 12"),
        "discrepancies": props_ui.discrepancies_html(gaps, "March 12"),
        "discrepancy_strip": props_ui.discrepancy_strip_html(props_ui.discrepancies(gaps)),
        "stat_cells": "\n".join(props_ui._stat_cell(r) for r in rows.to_dict(orient="records")),
        "paste_time": props_ui.paste_time_text(meta),
        "no_lines": props_ui.NO_LINES_SLATE + "\n" + props_ui.NO_LINES_GAME,
        "props_css": props_ui.PROPS_CSS,
    }
    return out
