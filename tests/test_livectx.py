"""Injury/teammate context: hand-built teams with known answers."""
import numpy as np
import pandas as pd

from bucketlib import livectx as L

TIP = pd.Timestamp("2026-01-15T00:00Z")
CUT = pd.Timestamp("2026-01-14T16:00Z")


def rows():
    return pd.DataFrame({
        "game_id": ["g"] * 4 + ["g"] * 2, "team": ["AAA"] * 4 + ["BBB"] * 2,
        "person_id": [1, 2, 3, 4, 5, 6], "tip_utc": [TIP] * 6,
        "min_ewm5_s": [30.0, 20.0, 10.0, 4.0, 25.0, 25.0], "miss_streak": [0, 0, 0, 5, 0, 0]})


def inj():
    p = pd.DataFrame({"game_id": ["g", "g"], "team": ["AAA", "AAA"], "person_id": [1, 2],
                      "status": ["Out", "Questionable"], "reason_cat": ["injury_illness", "injury_illness"]})
    t = pd.DataFrame({"game_id": ["g", "g"], "team": ["AAA", "BBB"], "state": ["listed", "not_yet_submitted"],
                      "report_ts_et": [pd.Timestamp("2026-01-14T15:30Z")] * 2, "cutoff_utc": [CUT] * 2})
    return p, t


def test_teammate_minutes_from_pregame_state_only():
    p, t = inj()
    c = L.build_context(rows(), p, t).set_index("person_id")
    assert c.loc[3, "tm_out_min"] == 30.0 and c.loc[3, "tm_q_min"] == 20.0
    assert c.loc[2, "tm_out_min"] == 30.0 and c.loc[2, "tm_q_min"] == 0.0   # not himself
    assert c.loc[1, "tm_out_min"] == 0.0 and np.isnan(c.loc[1, "tm_out_max_min"])
    assert c.loc[3, "tm_out_newly_min"] == 30.0                              # played last game
    assert c.loc[3, "tm_out_max_min"] == 30.0
    assert c.loc[3, "rotation_depth"] == 1.0                                 # player 2 (20 min), not 1 (Out) or 4 (4 min)


def test_three_way_status_encoding():
    p, t = inj()
    c = L.build_context(rows(), p, t).set_index("person_id")
    assert c.loc[1, "inj_out"] == 1 and c.loc[2, "inj_questionable"] == 1
    assert c.loc[3, "inj_not_listed"] == 1 and c.loc[3, "inj_unknown"] == 0  # team filed: known healthy
    assert c.loc[5, "inj_unknown"] == 1 and c.loc[5, "inj_not_listed"] == 0  # team not submitted: unknown
    assert c.loc[5, "tm_unknown_team"] == 1
    assert (c["inj_pre_archive"] == 0).all()
    assert abs(c.loc[3, "hours_to_tip"] - 8.0) < 1e-9 and abs(c.loc[3, "report_age_h"] - 0.5) < 1e-9


def test_mark_out_is_status_out_at_age_zero():
    p, t = inj()
    c = L.build_context(rows(), p, t, marked_out={3}).set_index("person_id")
    assert c.loc[3, "inj_out"] == 1 and c.loc[3, "report_age_h"] == 0.0
    assert c.loc[4, "tm_out_min"] == 40.0          # 1 (30) + 3 (10)
    # marking out conserves the team's expected minutes among the rest
    assert c.loc[2, "tm_out_min"] == 40.0


def test_resolve_names_scoped_to_game_roster():
    roster = pd.DataFrame({"game_id": ["g", "g"], "team": ["AAA", "AAA"], "person_id": [11, 12],
                           "player_name": ["Dereck Lively II", "Yang Hansen"]})
    inj = pd.DataFrame({"game_id": ["g", "g", "g"], "team": ["AAA"] * 3,
                        "player_raw": ["Lively II, Dereck", "Yang, Hansen", "Nobody, Some"]})
    r = L.resolve_names(inj, roster)
    assert r["person_id"].tolist()[:2] == [11, 12] and np.isnan(r["person_id"].iloc[2])
    assert r.attrs["unresolved"] == 1
