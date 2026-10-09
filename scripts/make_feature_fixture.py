"""Cut the small real-data fixture used by tests/test_features.py.

Four teams (BOS, LAL, DEN, DAL) over 2023-24 and 2024-25 from the box logs,
with both sides of every game they played, so team history and opponent
features are real. Includes Luka Doncic's DAL -> LAL trade (first LAL game
2025-02-10) for the replay test's first-game-after-a-trade case.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bucketlib import train as T  # noqa: E402

TEAMS = {"BOS", "LAL", "DEN", "DAL"}
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "features"

logs = T.load_table("player_games", T.seasons("2021-22", "2024-25"))
sched = T.load_table("schedule", T.seasons("2023-24", "2024-25"))
tg = T.load_table("team_games", T.seasons("2023-24", "2024-25"))
games = set(tg.loc[tg["team"].isin(TEAMS), "game_id"])
ev_logs = logs[logs["season"].isin(["2023-24", "2024-25"]) & logs["team"].isin(TEAMS)
               & logs["season_type"].isin(["RS", "PI"])]
keep = ["season", "season_type", "game_id", "game_date", "person_id", "team", "opp", "home",
        "sec", "min"] + ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast",
                         "stl", "blk", "tov", "pf", "pts"]
ev_logs[keep].to_parquet(OUT / "logs_events.parquet", index=False)
prior_logs = logs[logs["person_id"].isin(set(ev_logs["person_id"])) & (logs["season_type"] == "RS")]
prior_logs[keep].to_parquet(OUT / "logs_priors.parquet", index=False)
tg[tg["game_id"].isin(games) & tg["season_type"].isin(["RS", "PI"])].to_parquet(OUT / "team_games.parquet", index=False)
sched[["game_id", "game_type", "game_date", "tip_utc", "home", "away"]].to_parquet(OUT / "schedule.parquet", index=False)
for f in sorted(OUT.glob("*.parquet")):
    print(f.name, f.stat().st_size)
