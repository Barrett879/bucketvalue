"""The bot's cdn-built stat lines must equal the stats.nba.com logs the models
trained on (measured on all 647 cached 2019-20 cdn games / 1,294 team-games;
these two real games pin it offline)."""
from pathlib import Path

import pandas as pd

from bucketlib import cache as C
from bucketlib.sources import cdnbox

FIX = Path(__file__).parent / "fixtures" / "cdnbox"
COUNTS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
          "blk", "tov", "pf", "pts"]


def payload(gid):
    obj = C.read_json(FIX / f"cdn_{gid}.json.gz")
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


def test_player_lines_match_stats_logs():
    stats = pd.read_parquet(FIX / "stats_player_games.parquet").set_index(["game_id", "person_id"])
    for gid in ("0021900001", "0021900464"):
        pl = cdnbox.player_lines(payload(gid)).set_index(["game_id", "person_id"])
        s = stats.loc[[gid]]
        common = pl.index.intersection(s.index)
        assert (pl.loc[common, COUNTS].astype(int) == s.loc[common, COUNTS].astype(int)).all().all()
        assert (pl.loc[common, "sec"] - s.loc[common, "sec"]).abs().max() <= 2
        assert pl.loc[common, "fantasy"].round(6).equals(s.loc[common, "fantasy"].round(6))
        # The only allowed difference: a zero-second line with a stat (Justin
        # James, 0021900464), which cdn marks played=0 and tables.roster_games
        # reconciles as an appearance.
        assert set(s.index) - set(pl.index) <= {("0021900464", 1629713)}


def test_team_lines_match_stats_team_totals_including_team_turnovers():
    stats = pd.read_parquet(FIX / "stats_team_games.parquet").set_index(["game_id", "team"])
    for gid in ("0021900001", "0021900464"):
        tl = cdnbox.team_lines(payload(gid)).set_index(["game_id", "team"])
        for key in tl.index:
            for c in COUNTS + ["opp_pts", "ot_periods", "margin"]:
                assert int(tl.loc[key, c]) == int(stats.loc[key, c]), (key, c)


def test_iso_seconds():
    assert cdnbox.iso_seconds("PT25M01.00S") == 1501.0
    assert cdnbox.iso_seconds("PT00M01.00S") == 1.0
    assert cdnbox.iso_seconds("") == 0.0
