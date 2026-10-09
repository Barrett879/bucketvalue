"""Cut the real-data fixture the page tests serve (tests/test_pages_smoke.py,
tests/test_app_projections.py).

    .venv/bin/python tests/fixtures/serve/make_fixture.py

Two games of 2026-03-12, taken from tests/fixtures/pipeline (cut from the
training table by scripts/make_pipeline_fixture.py):
  0022500953  PHI at DET, 7:00 PM ET. Four PHI regulars and two DET rotation
              players listed Out at noon (one suspension, three two-way
              assignments): the Out section, the card's "Out:" line, badges.
  0022500958  DEN at SAS, 9:00 PM ET. Denver had not submitted at noon: the
              team-level "Report not filed" state, unknown on every row.

  features.parquet     the two games' history-feature rows exactly as the bot
                       commits them (state/serve/{date}/features_v1)
  schedule.parquet     their two schedule rows (status scheduled)
  projections.parquet  pipeline.project on those rows with the REAL 12:00 PM
                       ET report (tests/fixtures/injury), the production
                       models (models/, models.load) at 12:20 PM ET, the lines
                       below: what the bot would have committed at a 12:20
                       run. Nothing in it is typed by hand.
  lines.json           the spreads and totals given to project() and served
                       as the live odds by the tests. MADE UP: historical
                       lines are never committed (provenance), as in
                       tests/test_pipeline.py.
  board.txt            a PrizePicks board-text paste (written by hand) for
                       players of these games, in the format of
                       tests/fixtures/props/board_constructed.txt. The line
                       values are made up: no real NBA paste exists yet.

Rerun after retraining the models: projections.parquet carries the
model_version it was computed with.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from bucketlib import models as M  # noqa: E402
from bucketlib import pipeline as P  # noqa: E402
from bucketlib.sources import injury as INJ  # noqa: E402
from bucketlib.util import ET  # noqa: E402

DATE = dt.date(2026, 3, 12)
GAMES = ("0022500953", "0022500958")
AT = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)
REPORT = REPO / "tests" / "fixtures" / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf"
LINES = {"0022500953": {"home_line": -6.5, "total": 219.5},
         "0022500958": {"home_line": -3.5, "total": 236.5}}


def main() -> int:
    src = REPO / "tests" / "fixtures" / "pipeline"
    feats = pd.read_parquet(src / "features.parquet")
    feats = feats[feats["game_id"].isin(GAMES)].reset_index(drop=True)
    sched = pd.read_parquet(src / "schedule.parquet")
    sched = sched[sched["game_id"].isin(GAMES)].reset_index(drop=True)
    games = P.state_games(sched, DATE)
    report = INJ.parse_report(REPORT.read_bytes())
    lines = pd.DataFrame([{"game_id": g, **v} for g, v in LINES.items()])
    names = feats[["person_id", "player_name"]].drop_duplicates()
    proj = P.project(feats, games, sched, report, lines, M.load(), AT, names=names)
    if set(proj["game_id"]) != set(GAMES):
        raise ValueError(f"projected {sorted(set(proj['game_id']))}, wanted {GAMES}")
    feats.to_parquet(HERE / "features.parquet", index=False)
    sched.to_parquet(HERE / "schedule.parquet", index=False)
    proj.to_parquet(HERE / "projections.parquet", index=False)
    (HERE / "lines.json").write_text(json.dumps(LINES, indent=1) + "\n")
    for f in sorted(HERE.glob("*.parquet")):
        print(f.name, f.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
