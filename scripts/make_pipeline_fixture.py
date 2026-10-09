"""Cut the real-data fixture used by tests/test_pipeline.py and
tests/test_daily_update.py.

    python scripts/make_pipeline_fixture.py

The slate is 2026-03-12, the date of the committed injury fixture
tests/fixtures/injury/Injury-Report_2026-03-12_12_00PM.pdf, so names, teams
and report statuses are real and resolve against each other.

  features.parquet  the nine games' history-feature rows exactly as the bot
                    commits them (state/serve/{date}/features_v1): the training
                    table's rows restricted to features.build's own output
                    columns (no lines, no injury context, no derived targets, no
                    horizon; tests/test_serve_parity.py proves the serve path
                    builds these bit for bit), outcomes blanked the way
                    features.make_targets blanks them, plus player_name (the
                    participation table) and two_way (False for everyone: the
                    backfill keeps no as-of-date two-way flag).
  schedule.parquet  every game of 2026-03-12 in cdn.parse_schedule's shape, as
                    the morning schedule had them (status scheduled).

No line is included: historical lines are never committed (provenance), so
the tests pass made-up lines of their own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import features as F  # noqa: E402
from bucketlib import livectx as L  # noqa: E402
from bucketlib import train as T  # noqa: E402

DATE = pd.Timestamp("2026-03-12")
SEASON = "2025-26"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pipeline"
LINE_COLS = ["home_line", "total", "team_line", "implied_total", "abs_line"]
NOT_BUILT = set(LINE_COLS) | set(L.CTX_COLS) | {"horizon", *T.COMBOS, "fantasy", "fg2m", "fg2a"}


def main() -> int:
    table = C.CACHE_DIR / "train_table_full_v1.parquet"
    cols = [c for c in pq.ParquetFile(table).schema_arrow.names if c not in NOT_BUILT]
    rows = pd.read_parquet(table, columns=cols, filters=[("game_date", "==", DATE)])
    rows = rows[rows["game_id"].str.startswith("002")].copy()
    # Outcomes unknown, exactly as features.make_targets leaves a target row.
    rows["status"] = None
    for c in ["starter", "played", "played_seconds", "min", *F.RATE_STATS]:
        rows[c] = np.nan
    rows["is_target"] = True
    part = C.read_parquet(C.CACHE_DIR / f"participation_{SEASON}_v1.parquet")
    names = part.loc[part["game_id"].isin(set(rows["game_id"])), ["game_id", "person_id", "player_name"]]
    rows = rows.merge(names, on=["game_id", "person_id"], how="left", validate="one_to_one")
    if rows["player_name"].isna().any():
        raise ValueError("a fixture row has no participation name")
    rows["two_way"] = False
    rows = rows.sort_values(["tip_utc", "game_id", "team", "person_id"], kind="mergesort")
    OUT.mkdir(parents=True, exist_ok=True)
    rows.reset_index(drop=True).to_parquet(OUT / "features.parquet", index=False)

    s = C.read_parquet(C.CACHE_DIR / f"schedule_{SEASON}_v1.parquet")
    s = s[s["game_date"] == DATE]
    sched = pd.DataFrame({"game_id": s["game_id"], "season": SEASON, "game_type": "RS",
                          "game_date": s["game_date"].dt.date, "tip_utc": s["tip_utc"],
                          "status": "scheduled", "home": s["home"], "away": s["away"]})
    if set(sched["game_id"]) != set(rows["game_id"]):
        raise ValueError("the schedule and the training rows disagree on the slate")
    sched.sort_values(["tip_utc", "game_id"]).reset_index(drop=True).to_parquet(
        OUT / "schedule.parquet", index=False)
    for f in sorted(OUT.glob("*.parquet")):
        print(f.name, f.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
