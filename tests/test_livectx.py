"""Injury/teammate context: hand-built teams with known answers."""
import numpy as np
import pandas as pd
import pytest

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


NO_ROWS = pd.DataFrame({"game_id": pd.Series(dtype=object), "team": pd.Series(dtype=object),
                        "person_id": pd.Series(dtype=float), "status": pd.Series(dtype=object),
                        "reason_cat": pd.Series(dtype=object)})


@pytest.mark.filterwarnings("error")
def test_mark_out_with_no_injury_rows_writes_strings_into_object_columns():
    """No report filed (every team no_report, no player rows) and a visitor
    mark-out: the status and reason columns started as float NaN and took
    'Out' / 'injury_illness' (pandas FutureWarning now, an error in pandas 3).
    Run with warnings as errors; the encoding must equal a reported Out."""
    t = pd.DataFrame({"game_id": ["g", "g"], "team": ["AAA", "BBB"], "state": ["no_report"] * 2,
                      "report_ts_et": [pd.NaT] * 2, "cutoff_utc": [CUT] * 2})
    c = L.build_context(rows(), NO_ROWS, t, marked_out={3}).set_index("person_id")
    assert c.loc[3, "inj_out"] == 1 and c.loc[3, "inj_reason_injury"] == 1
    assert c.loc[3, "report_age_h"] == 0.0 and c.loc[3, "inj_unknown"] == 0
    assert c.loc[1, "tm_out_min"] == 10.0 and c.loc[1, "inj_unknown"] == 1
    assert (c[[f"inj_{s.lower()}" for s in L.STATUSES]].drop(index=3) == 0).all().all()
    # No team rows at all (state merges in all-NaN): the mark-out still encodes.
    none = t.iloc[:0].astype({"state": float})
    c = L.build_context(rows(), NO_ROWS, none, marked_out={3}).set_index("person_id")
    assert c.loc[3, "inj_out"] == 1 and c.loc[5, "inj_unknown"] == 1


def test_resolve_names_scoped_to_game_roster():
    roster = pd.DataFrame({"game_id": ["g", "g"], "team": ["AAA", "AAA"], "person_id": [11, 12],
                           "player_name": ["Dereck Lively II", "Yang Hansen"]})
    inj = pd.DataFrame({"game_id": ["g", "g", "g"], "team": ["AAA"] * 3,
                        "player_raw": ["Lively II, Dereck", "Yang, Hansen", "Nobody, Some"]})
    r = L.resolve_names(inj, roster)
    assert r["person_id"].tolist()[:2] == [11, 12] and np.isnan(r["person_id"].iloc[2])
    assert r.attrs["unresolved"] == 1


def test_resolve_uses_every_spelling_the_person_ever_had():
    """The roster for this game spells him 'Enes Freedom' (cdn, current name);
    the report says 'Kanter, Enes'. Known names bridge it, still team-scoped."""
    roster = pd.DataFrame({"game_id": ["g"], "team": ["BOS"], "person_id": [202683],
                           "player_name": ["Enes Freedom"]})
    known = pd.DataFrame({"person_id": [202683, 202683, 999],
                          "player_name": ["Enes Freedom", "Enes Kanter", "Enes Kanter"]})
    inj = pd.DataFrame({"game_id": ["g"], "team": ["BOS"], "player_raw": ["Kanter, Enes"]})
    assert np.isnan(L.resolve_names(inj, roster)["person_id"].iloc[0])
    assert L.resolve_names(inj, roster, known)["person_id"].iloc[0] == 202683
