"""The serve fixture installed as the app's world, for the page tests.

install() points bucketlib.state at a temp state dir holding what the bot
would have committed for 2026-03-12 (the two games' history features, stamped
as pipeline.write_features stamps them, and their projections), makes the
live layer answer from the fixture (schedule, rosters, the real 12:00 PM ET
injury report, the made-up lines) and pins every clock the pages read. No
network, no repo state, no writes outside the temp dir.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bucketlib import cache as C
from bucketlib import live as LV
from bucketlib import pipeline as P
from bucketlib import props_ui, store, util
from bucketlib import state as S
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATE = dt.date(2026, 3, 12)
SEASON = "2025-26"
EARLY = "0022500953"     # PHI at DET, tip 7:00 PM ET: past its freeze point from 6:55
LATE = "0022500958"      # DEN at SAS, tip 9:00 PM ET; Denver had not filed at noon
MORNING = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)   # the fixture's own compute time
EVENING = dt.datetime(2026, 3, 12, 19, 0, tzinfo=ET)    # PHI at DET frozen, DEN at SAS live


@dataclass(frozen=True)
class World:
    feats: pd.DataFrame
    sched: pd.DataFrame
    proj: pd.DataFrame
    report: object
    odds: dict
    roster: pd.DataFrame
    board_text: str


def roster_from(feats: pd.DataFrame) -> pd.DataFrame:
    """The fixture's players in cdn.parse_player_index's columns (the live
    roster the pages read for positions and paste resolution)."""
    r = feats.drop_duplicates("person_id")
    first = r["player_name"].str.split(" ", n=1).str[0]
    last = r["player_name"].str.split(" ", n=1).str[1].fillna("")
    return pd.DataFrame({"person_id": r["person_id"].astype("int64").to_numpy(),
                         "first_name": first.to_numpy(), "last_name": last.to_numpy(),
                         "team": r["team"].to_numpy(), "position": "",
                         "two_way": r["two_way"].astype(bool).to_numpy(),
                         "name": r["player_name"].to_numpy()})


@lru_cache(maxsize=1)
def load() -> World:
    feats = pd.read_parquet(HERE / "features.parquet")
    sched = pd.read_parquet(HERE / "schedule.parquet")
    proj = pd.read_parquet(HERE / "projections.parquet")
    report = INJ.parse_report((REPO / "tests" / "fixtures" / "injury"
                               / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())
    lines = json.loads((HERE / "lines.json").read_text())
    odds = {(a, h): {**lines[g], "provider": "DraftKings"}
            for g, a, h in zip(sched["game_id"], sched["away"], sched["home"])}
    return World(feats, sched, proj, report, odds, roster_from(feats),
                 (HERE / "board.txt").read_text())


def pin(monkeypatch, at: dt.datetime) -> None:
    """Every clock a page reads, at `at` (util's, and the names live and
    store imported from it)."""
    u, e = at.astimezone(UTC), at.astimezone(ET)
    for mod in (util, LV):
        monkeypatch.setattr(mod, "now_utc", lambda: u)
        monkeypatch.setattr(mod, "now_et", lambda: e)
    monkeypatch.setattr(store, "now_utc", lambda: u)


def install(monkeypatch, tmp_path: Path, at: dt.datetime = MORNING, report=None) -> World:
    w = load()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "state" / "seed")
    C.write_parquet(w.feats[["person_id", "player_name"]].drop_duplicates(),
                    S.SEED_DIR / S.SEED_FILES["names"])
    P.write_features(w.feats, DATE, P.history_stamp(SEASON, DATE))
    C.write_parquet(w.proj, P.projections_path(DATE))
    rep = w.report if report is None else report
    monkeypatch.setattr(LV, "schedule", lambda: LV.Live(w.sched, 0.0))
    monkeypatch.setattr(LV, "roster", lambda: LV.Live(w.roster, 0.0))
    monkeypatch.setattr(LV, "odds_on", lambda d: LV.Live(w.odds if d == DATE else {}, 0.0))
    monkeypatch.setattr(LV, "injury_report", lambda sched=None: LV.Live(rep, 0.0))
    pin(monkeypatch, at)
    store._live_frame.clear()
    return w


def paste(w: World, at: dt.datetime = MORNING) -> str:
    """The board text through the real paste path (props_ui.ingest): the
    token a visitor's ?board= would carry."""
    token, report = props_ui.ingest(w.board_text, None, w.roster, w.sched, now=at, dates=[DATE])
    assert token and report["comparable"] == 11, report
    return token
