"""Train/serve parity and point-in-time correctness of bucketlib.features,
proven on REAL data (tests/fixtures/features, cut by scripts/make_feature_fixture.py).

The two load-bearing tests call the production builder only (a test that
reimplements the filter it checks tests the copy):
  - replay: features for date D built from history truncated before D, with
    D's games appended as synthetic target rows, equal the training features
    for D BIT FOR BIT, on boundary dates;
  - future mutation: scrambling every outcome on or after D leaves D's features
    unchanged (and does move later dates, so the scramble is real).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import features as F
from bucketlib import train as T

FIX = Path(__file__).parent / "fixtures" / "features"


@pytest.fixture(scope="module")
def inputs():
    logs = pd.read_parquet(FIX / "logs_events.parquet")
    priors_logs = pd.read_parquet(FIX / "logs_priors.parquet")
    sched = pd.read_parquet(FIX / "schedule.parquet")
    tg = pd.read_parquet(FIX / "team_games.parquet")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    ev = T.logs_events(logs, sched)
    pri = F.prior_seasons(priors_logs)
    sd = T.schedule_dates(sched)
    return ev, tg, sd, pri


@pytest.fixture(scope="module")
def full(inputs):
    ev, tg, sd, pri = inputs
    return F.build(ev, None, tg, sd, pri)


def _feature_frame(df):
    cols = T.feature_columns(df)
    return df.set_index(["game_id", "person_id"]).sort_index()[cols]


def _assert_identical(a: pd.DataFrame, b: pd.DataFrame, label: str):
    assert list(a.index) == list(b.index), f"{label}: different rows"
    bad = [c for c in a.columns
           if not np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True)]
    assert not bad, f"{label}: features differ between training and serving: {bad[:12]}"


# Boundary dates the design review named. Each is checked to really be the
# case it claims to be, so the list cannot silently drift into easy dates.
REPLAY_DATES = {
    "season opener": "2024-10-22",
    "first game after a trade (Doncic DAL->LAL)": "2025-02-10",
    "day after the All-Star break": "2025-02-20",
    "B2B second leg": None,          # found from the data below
    "mid-season ordinary": "2024-12-03",
}


def _b2b_date(full):
    r = full[(full["b2b"] == 1) & (full["season"] == "2024-25")]
    return str(r["game_date"].min().date())


def test_replay_boundary_dates_bit_for_bit(inputs, full):
    ev, tg, sd, pri = inputs
    dates = dict(REPLAY_DATES)
    dates["B2B second leg"] = _b2b_date(full)
    doncic = full[(full["person_id"] == 1629029) & (full["game_date"] == pd.Timestamp("2025-02-10"))]
    assert len(doncic) == 1 and doncic["team"].iloc[0] == "LAL" and doncic["team_changed"].iloc[0] == 1
    for label, d in dates.items():
        D = pd.Timestamp(d)
        day = ev[ev["game_date"] == D]
        assert len(day), f"{label}: no games on {d}"
        served = F.build(ev[ev["game_date"] < D], F.make_targets(day), tg[tg["game_date"] < D], sd, pri)
        trained = full[full["game_date"] == D]
        _assert_identical(_feature_frame(trained), _feature_frame(served), f"{label} {d}")


def test_future_mutation_leaves_features_unchanged(inputs, full):
    ev, tg, sd, pri = inputs
    rng = np.random.default_rng(7)
    D = pd.Timestamp("2025-01-15")
    ev2, tg2 = ev.copy(), tg.copy()
    m = ev2["game_date"] >= D
    for c in F.RATE_STATS + ["played_seconds"]:
        ev2.loc[m, c] = rng.permutation(ev2.loc[m, c].to_numpy())
    ev2["min"] = ev2["played_seconds"] / 60
    mt = tg2["game_date"] >= D
    for c in ["pts", "fga", "fta", "tov", "oreb", "dreb", "reb", "ast", "fg3m", "fg3a", "stl", "blk"]:
        tg2.loc[mt, c] = rng.permutation(tg2.loc[mt, c].to_numpy())
    mut = F.build(ev2, None, tg2, sd, pri)
    _assert_identical(_feature_frame(full[full["game_date"] == D]),
                      _feature_frame(mut[mut["game_date"] == D]), "future mutation")
    later = full["game_date"] > D + pd.Timedelta(days=5)
    a, b = _feature_frame(full[later]), _feature_frame(mut[mut["game_date"] > D + pd.Timedelta(days=5)])
    assert not np.array_equal(a["min_ewm2_s"].to_numpy(dtype=float), b["min_ewm2_s"].to_numpy(dtype=float),
                              equal_nan=True), "the scramble did not change anything: the test is vacuous"


def test_first_game_of_season_has_no_current_season_state(full):
    first = full.sort_values("tip_utc").groupby(["person_id", "season"]).head(1)
    assert first["gp_s"].eq(0).all()
    assert first["min_ewm2_s"].isna().all() and first["min_std_s"].isna().all()


def test_short_memory_resets_long_memory_carries(full):
    """Minutes EWMA resets each season; rate EWMA carries across seasons."""
    p = full[full["person_id"] == 1628369].sort_values("tip_utc")  # Jayson Tatum
    s2 = p[p["season"] == "2024-25"].iloc[0]
    assert np.isnan(s2["min_ewm2_s"])
    assert not np.isnan(s2["r_pts_ewm20"])
    assert s2["p1_mpg"] > 30


# ── Primitives ───────────────────────────────────────────────────────────────
def test_cum_prior_is_a_shifted_sum():
    df = pd.DataFrame({"k": [1, 1, 1, 2, 2]})
    x = pd.Series([1.0, np.nan, 3.0, 5.0, 7.0])
    assert F._cum_prior(df, x, ["k"]).tolist() == [0.0, 1.0, 1.0, 0.0, 5.0]


def test_ewm_prior_excludes_current_row():
    df = pd.DataFrame({"k": [1, 1, 1]})
    x = pd.Series([10.0, 20.0, 30.0])
    out = F._ewm_prior(df, x, ["k"], 2)
    assert np.isnan(out[0]) and out[1] == 10.0 and 10.0 < out[2] < 20.0


def test_league_norms_use_only_earlier_dates():
    t = pd.DataFrame({
        "season": ["2024-25"] * 4, "game_date": pd.to_datetime(["2024-10-22"] * 2 + ["2024-10-23"] * 2),
        "pace": [100.0, 100.0, 120.0, 120.0], "ortg": [110.0] * 4,
        **{f"allow_{c}": [1.0] * 4 for c in F.TEAM_ALLOWED}})
    q = pd.DataFrame({"season": ["2024-25"] * 3,
                      "game_date": pd.to_datetime(["2024-10-22", "2024-10-23", "2024-10-24"])})
    lg = F.league_norms(t, q).set_index("game_date")["lg_pace"]
    assert np.isnan(lg[pd.Timestamp("2024-10-22")])       # nothing before opening night
    assert lg[pd.Timestamp("2024-10-23")] == 100.0          # only the 22nd
    assert lg[pd.Timestamp("2024-10-24")] == 110.0          # both days


def test_rest_counts_the_cup_final():
    sd = pd.DataFrame({"team": ["MIL", "MIL", "MIL"],
                       "game_date": pd.to_datetime(["2024-12-14", "2024-12-17", "2024-12-18"])})
    r = F.team_rest(sd).set_index("game_date")
    assert r.loc[pd.Timestamp("2024-12-18"), "b2b"] == 1.0
    assert r.loc[pd.Timestamp("2024-12-17"), "rest_days"] == 3


def test_targets_carry_no_outcomes():
    rows = pd.DataFrame({"person_id": [1], "game_id": ["0022600001"], "season": ["2026-27"],
                         "game_date": [pd.Timestamp("2026-10-20")],
                         "tip_utc": [pd.Timestamp("2026-10-20T19:00Z")], "team": ["BOS"],
                         "opp": ["DET"], "home": [False]})
    t = F.make_targets(rows)
    assert t[F.RATE_STATS + ["played_seconds", "min", "played"]].isna().all().all()
    assert t["is_target"].all()


def test_contract_raises_on_unknown_column():
    with pytest.raises(KeyError):
        T.classify("final_margin")
