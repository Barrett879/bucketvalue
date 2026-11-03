"""bucketlib.models: the production model layer (docs/serving_design.md section 2).

Every check calls PRODUCTION code (models.train / save / load / predict /
check_nulls) on slices of the REAL training table; nothing here reimplements
what it checks. The gated configuration (GATED_HGB and the losses, the rate
target, the regular-season filter), the combos and Fantasy are hard-coded on
purpose: comparing models.py with its own constants would pass any retrain.

Three groups:
  - slice tests train small models with models.train on the first rows of
    cache/train_table_full_v1.parquet (local only; skipped when absent or
    unreadable);
  - the training-wiring test records what models.train hands each estimator's
    fit (without fitting) on real rows that include play-in games;
  - shipped-artifact tests load models/registry_v1.json and its joblib files
    and FAIL when they are missing, because the site cannot start without them.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from bucketlib import features as F
from bucketlib import models as M
from bucketlib import props as P
from bucketlib import train as T

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import validate_full as VF  # noqa: E402

TABLE = VF.TABLE
SLICE_SEASON = "2018-19"   # the table is season-ordered: its first rows are 2018-19 RS

# The contract (docs/serving_design.md, bucketlib/models.py section), spelled out
# here so a dropped or renamed column fails.
CONTRACT_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta", "fgm", "fga",
                  "fg3a", "oreb", "dreb", "pf"]
CONTRACT_COLS = (["p_play", "mu_min"] + [f"rate_{s}" for s in CONTRACT_STATS] + CONTRACT_STATS
                 + ["pra", "pr", "pa", "ra", "stocks", "fantasy"])
# Three full 2025-26 slates, the unit serving checks and projects: opening
# night, night two, and a mid-season date.
SLATES = ["2025-10-21", "2025-10-22", "2026-01-14"]
TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
                 "2025-26"]

# The gated estimators exactly as scripts/train_eval.py constructed them (the
# predictions docs/model_report.md scored). Full get_params() is compared, so
# an added argument (class_weight, a loss, a seed) fails too.
GATED_HGB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                 random_state=0)
GATED_EXTRA = {"max_min": 53.0, "rate_cap": 5.0, "min_exposure": 1 / 60}


def gated_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**GATED_HGB)


def gated_regressor(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, **GATED_HGB)


def assert_gated_estimators(m: M.Models) -> None:
    assert m.availability.get_params() == gated_classifier().get_params()
    assert m.minutes.loss == "squared_error"
    assert m.minutes.get_params() == gated_regressor("squared_error").get_params()
    assert list(m.rates) == CONTRACT_STATS
    for s, est in m.rates.items():
        assert est.loss == "poisson", s
        assert est.get_params() == gated_regressor("poisson").get_params(), s


def table_or_skip() -> None:
    """Skip unless the local-only table is present AND readable. iCloud can
    evict it to a 'dataless' placeholder that exists() but has no bytes, which
    used to fail these tests with ArrowInvalid; reading the footer tells the
    two apart."""
    if not TABLE.exists():
        pytest.skip(f"{TABLE.name} is local only and absent here")
    try:
        pq.read_schema(TABLE)
    except Exception as e:  # ArrowInvalid / OSError on an evicted placeholder
        pytest.skip(f"{TABLE.name} is present but unreadable ({type(e).__name__}): "
                    "evicted by iCloud? Keep the repo downloaded")


@pytest.fixture(scope="module")
def table():
    """(train_part, serve_part): two disjoint 4,000-row slices of the real table."""
    table_or_skip()
    it = pq.ParquetFile(TABLE).iter_batches(batch_size=4000)
    train_part, serve_part = next(it).to_pandas(), next(it).to_pandas()
    assert set(train_part["season"]) == {SLICE_SEASON}, "slice assumption broke: table order changed"
    return train_part, serve_part


@pytest.fixture(scope="module")
def real():
    """One read of the table. `tail`: its last 3,000 rows (2025-26 including
    play-in rows; the table is person-ordered within a season, so these are
    the newest players, NOT a representative slate). `slates`: every
    regular-season row of each SLATES date, keyed by date."""
    table_or_skip()
    t = pq.read_table(TABLE)
    tail = t.slice(t.num_rows - 3000).to_pandas()
    dates = t.column("game_date").to_pandas()
    rs = t.column("game_id").to_pandas().str.startswith("002")
    picked = t.take(np.flatnonzero(dates.isin(pd.to_datetime(SLATES)) & rs)).to_pandas()
    del t
    assert set(tail["season"]) == {"2025-26"}, "slice assumption broke: table order changed"
    assert tail["game_id"].str.startswith("005").any(), "slice assumption broke: no play-in rows"
    slates = {d: picked[picked["game_date"] == pd.Timestamp(d)] for d in SLATES}
    assert all(len(v) >= 50 for v in slates.values())
    return SimpleNamespace(tail=tail, slates=slates)


@pytest.fixture(scope="module")
def fitted(table):
    return M.train(table[0], seasons=[SLICE_SEASON])


@pytest.fixture
def saved(fitted, tmp_path):
    M.save(fitted, tmp_path)
    return tmp_path


@pytest.fixture
def fit_calls(monkeypatch):
    """Record every estimator fit WITHOUT fitting, so a test sees exactly what
    models.train trains each model on (rows, target, weight, estimator)."""
    calls = []

    def record(self, X, y, sample_weight=None):
        calls.append(SimpleNamespace(est=self, X=X, y=y, w=sample_weight))
        self._predictors, self.n_features_in_ = [], X.shape[1]  # n_iter_ reads as 0
        return self
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", record)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", record)
    return calls


def rewrite_registry(dir: Path, edit) -> None:
    path = dir / M.REGISTRY_NAME
    reg = json.loads(path.read_text())
    edit(reg)
    path.write_text(json.dumps(reg))


# ── The gated configuration ──────────────────────────────────────────────────
def test_hyperparameters_are_the_gated_ones():
    """models.HGB is the one production definition; the report scripts keep
    literal copies. Pin all three to the gate's literal configuration."""
    assert M.availability_model().get_params() == gated_classifier().get_params()
    for loss in ("squared_error", "poisson"):
        assert M.regressor(loss).get_params() == gated_regressor(loss).get_params()
        assert VF.hgb_reg(loss).get_params() == gated_regressor(loss).get_params()
    assert (M.MAX_MIN, M.RATE_CAP, M.MIN_EXPOSURE) == (53.0, 5.0, 1 / 60)
    assert M.STATS == VF.STATS == CONTRACT_STATS
    assert M.COMPOSED == VF.COMPOSED
    assert sorted(M.STATS) == sorted(F.RATE_STATS)
    assert M.TRAIN_SEASONS == TRAIN_SEASONS


def test_rate_target_is_the_gated_exposure_poisson_target():
    """(stat / max(min, 1 second)) capped at 5 per minute, weighted by
    max(min, 1 second): the gate's target, worked by hand."""
    played = pd.DataFrame({"min": [20.0, 2.0, 0.5, 0.0, 1 / 120],
                           "pts": [10.0, 9.0, 3.0, 0.0, 1.0]})
    y, w = M.rate_target(played, "pts")
    np.testing.assert_allclose(w, [20.0, 2.0, 0.5, 1 / 60, 1 / 60], rtol=0, atol=1e-15)
    np.testing.assert_allclose(y, [0.5, 4.5, 5.0, 0.0, 5.0], rtol=0, atol=1e-12)


def test_training_rows_keep_only_regular_season_games_of_the_seasons():
    t = pd.DataFrame({"game_id": ["0021800001", "0051900001", "0041800001", "0011800001",
                                  "0021900002", "0022000003", "0061900001"],
                      "season": ["2018-19", "2019-20", "2018-19", "2018-19", "2019-20",
                                 "2020-21", "2019-20"]})
    out = M.training_rows(t, ["2018-19", "2019-20"])
    assert out["game_id"].tolist() == ["0021800001", "0021900002"]


def test_train_raises_on_an_absent_season(table):
    with pytest.raises(ValueError, match="2019-20"):
        M.train(table[0], seasons=[SLICE_SEASON, "2019-20"])


def test_train_fits_each_model_on_the_gated_rows_targets_and_weights(real, fit_calls):
    """What models.train hands every fit, on real 2025-26 rows that include
    play-in games: availability on every regular-season row with target played;
    minutes (squared error) and the 15 rates (Poisson, rate target, minutes
    weight) on played regular-season rows. Play-in rows train nothing."""
    tail = real.tail
    m = M.train(tail, seasons=["2025-26"])
    rs = tail[tail["game_id"].str.startswith("002")]
    played = rs[rs["played"] == True]  # noqa: E712  (the gate's filter, train_eval.py)
    feats = T.feature_columns(tail)
    clf, mins, *rates = fit_calls
    assert len(rates) == len(CONTRACT_STATS)

    assert isinstance(clf.est, HistGradientBoostingClassifier)
    assert list(clf.X.columns) == feats
    pd.testing.assert_index_equal(clf.X.index, rs.index)
    np.testing.assert_array_equal(clf.y, rs["played"].astype(int))
    assert clf.w is None

    assert isinstance(mins.est, HistGradientBoostingRegressor) and mins.est.loss == "squared_error"
    assert list(mins.X.columns) == feats
    pd.testing.assert_index_equal(mins.X.index, played.index)
    np.testing.assert_array_equal(mins.y, played["min"])
    assert mins.w is None

    for s, call in zip(CONTRACT_STATS, rates):
        assert call.est.loss == "poisson", s
        assert list(call.X.columns) == feats
        pd.testing.assert_index_equal(call.X.index, played.index)
        y, w = M.rate_target(played, s)
        np.testing.assert_array_equal(call.y, y)
        assert call.w is not None, f"rate_{s} fitted without the minutes weight"
        np.testing.assert_array_equal(call.w, w)

    assert m.registry["n_rows"]["availability"] == len(rs) < len(tail)
    assert m.registry["n_rows"]["minutes"] == m.registry["n_rows"]["rate_pts"] == len(played)
    assert sum(m.registry["null_reference"]["n"].values()) == len(rs)


def test_trained_through_is_the_latest_season_in_any_order(table, real, fit_calls):
    """The props in-sample guard keys on it; seasons passed newest-first must
    still record the newest (fits are recorded, not run)."""
    two = pd.concat([real.tail, table[0]], ignore_index=True)
    m = M.train(two, seasons=["2025-26", SLICE_SEASON])
    assert m.registry["trained_through"] == "2025-26"
    assert P.is_in_training_window("2024-25", m.registry)
    assert not P.is_in_training_window("2026-27", m.registry)


def test_trained_estimators_are_the_gated_configuration(fitted):
    assert_gated_estimators(fitted)


# ── Round trip and in-memory parity ──────────────────────────────────────────
def test_loaded_models_predict_exactly_what_the_fitted_ones_do(fitted, saved, table):
    rows = table[1]
    loaded = M.load(saved)
    pd.testing.assert_frame_equal(M.predict(loaded, rows), M.predict(fitted, rows), check_exact=True)
    assert loaded.version == fitted.version and loaded.features == fitted.features


def test_predict_is_the_estimators_own_output(fitted, table):
    """predict() adds nothing to the estimators but the gated clip, the
    made <= attempts caps (FG3A and FGM at FGA, FG3M at the capped FG3A and
    FGM, FTM at FTA) and the E[stat | plays] = mu_min x rate composition
    (p_play is NOT multiplied in)."""
    rows = table[1]
    X = rows[fitted.features]
    out = M.predict(fitted, rows)
    assert list(out.columns) == CONTRACT_COLS
    np.testing.assert_array_equal(out["p_play"], fitted.availability.predict_proba(X)[:, 1])
    np.testing.assert_array_equal(out["mu_min"], np.clip(fitted.minutes.predict(X), 0, 53))
    raw = {s: fitted.rates[s].predict(X) for s in CONTRACT_STATS}
    raw["fg3a"] = np.minimum(raw["fg3a"], raw["fga"])
    raw["fgm"] = np.minimum(raw["fgm"], raw["fga"])
    raw["fg3m"] = np.minimum(np.minimum(raw["fg3m"], raw["fg3a"]), raw["fgm"])
    raw["ftm"] = np.minimum(raw["ftm"], raw["fta"])
    for s in CONTRACT_STATS:
        np.testing.assert_array_equal(out[f"rate_{s}"], raw[s])
        np.testing.assert_array_equal(out[s], out["mu_min"] * out[f"rate_{s}"])
    assert out["p_play"].between(0, 1).all() and out["p_play"].nunique() > 10


def test_mu_min_is_clipped_to_the_gated_range(fitted, table):
    class Fixed:
        def predict(self, X):
            return np.resize([-3.0, 61.0, 20.0], len(X))
    m = dataclasses.replace(fitted, minutes=Fixed())
    out = M.predict(m, table[1].head(3))
    assert out["mu_min"].tolist() == [0.0, 53.0, 20.0]
    assert out["pts"].iloc[0] == 0.0


def test_fg3a_rate_is_capped_at_the_fga_rate(fitted, table):
    """A three is a field-goal attempt. The two rate models are fit apart, so a
    raw fg3a rate above the fga rate made FG2A negative and the simulator's
    FG3A share clip to 1 (14 played rows of the real table). predict() caps
    the rate and builds the fg3a mean from the capped rate; rows at or below
    the fga rate are untouched."""
    rows = table[1].head(300)
    X = rows[fitted.features]
    fga = fitted.rates["fga"].predict(X)
    factor = np.resize([0.5, 1.0, 1.5, 3.0], len(X))

    class Fixed:
        def predict(self, X_):
            return fga * factor
    m = dataclasses.replace(fitted, rates={**fitted.rates, "fg3a": Fixed()})
    out = M.predict(m, rows)
    np.testing.assert_array_equal(out["rate_fg3a"], np.minimum(fga * factor, fga))
    np.testing.assert_array_equal(out["fg3a"], out["mu_min"] * out["rate_fg3a"])
    assert (out["fg3a"] <= out["fga"]).all() and (out["fga"] - out["fg3a"] >= 0).all()
    np.testing.assert_array_equal(out["rate_fg3a"].to_numpy()[factor < 1], (fga * factor)[factor < 1])
    np.testing.assert_array_equal(out["rate_fga"], fga)          # the cap moves fg3a only


def test_every_made_rate_is_capped_at_its_attempts(fitted, table):
    """Fixed fake rate models put each part above and below its whole, one
    case per row: predict() serves the cap (FTM at FTA, FGM at FGA, FG3A at
    FGA, FG3M at FG3A and at FGM, each cap itself capped first), builds the
    mean from it, and leaves rows below the cap and the wholes alone."""
    rows = table[1].head(6)

    class Fixed:
        def __init__(self, *v):
            self.v = np.array(v, dtype=float)

        def predict(self, X_):
            return self.v
    #                fga   fg3a   fgm    fg3m   fta   ftm
    case = {"fga":  Fixed(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "fg3a": Fixed(0.6, 0.6, 0.6, 1.4, 0.6, 0.6),   # row 3: threes above shots
            "fgm":  Fixed(0.4, 0.4, 1.2, 0.4, 0.4, 0.4),   # row 2: makes above shots
            "fg3m": Fixed(0.3, 0.5, 0.5, 0.9, 0.7, 0.3),   # row 1: above fgm; 4: above both
            "fta":  Fixed(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            "ftm":  Fixed(0.4, 0.6, 0.4, 0.4, 0.4, 0.5)}   # row 1 above, row 5 at
    out = M.predict(dataclasses.replace(fitted, rates={**fitted.rates, **case}), rows)
    want = {"fga": [1.0] * 6, "fg3a": [0.6, 0.6, 0.6, 1.0, 0.6, 0.6],
            "fgm": [0.4, 0.4, 1.0, 0.4, 0.4, 0.4], "fg3m": [0.3, 0.4, 0.5, 0.4, 0.4, 0.3],
            "fta": [0.5] * 6, "ftm": [0.4, 0.5, 0.4, 0.4, 0.4, 0.5]}
    for s, v in want.items():
        np.testing.assert_array_equal(out[f"rate_{s}"], v, err_msg=s)
        np.testing.assert_array_equal(out[s], out["mu_min"] * out[f"rate_{s}"], err_msg=s)
    for part, whole in PART_OF:
        assert (out[part] <= out[whole]).all(), (part, whole)


def test_predict_accepts_extra_columns_any_order_and_keeps_the_index(fitted, table):
    rows = table[1].head(300)
    messy = rows[rows.columns[::-1]].assign(not_a_feature="x", also_not=1.5)
    messy.index = pd.Index([f"row{i}" for i in range(len(messy))])
    out = M.predict(fitted, messy)
    assert list(out.index) == list(messy.index)
    np.testing.assert_array_equal(out.to_numpy(), M.predict(fitted, rows).to_numpy())


def test_all_nan_current_season_features_still_predict(fitted, table):
    """Opening night: every current-season column is legitimately NaN. That is
    present-but-empty, not missing, and must not raise."""
    rows = table[1].head(50).copy()
    for c in fitted.features:
        if c.startswith(("r_", "min_ewm", "min_std_s", "gp_s")):
            rows[c] = np.nan
    out = M.predict(fitted, rows)
    assert np.isfinite(out.to_numpy()).all()


def test_empty_rows_give_an_empty_frame_with_the_contract_columns(fitted, table):
    out = M.predict(fitted, table[1].head(0))
    assert list(out.columns) == CONTRACT_COLS and len(out) == 0


# ── Missing features: raise, name every one, never fill ──────────────────────
def test_missing_features_raise_and_name_every_column(fitted, table):
    feats = fitted.features
    drop = [feats[0], feats[60], feats[-1]]
    with pytest.raises(M.MissingFeatures) as e:
        M.predict(fitted, table[1].drop(columns=drop))
    assert e.value.missing == drop
    for c in drop:
        assert c in str(e.value)
    assert isinstance(e.value, KeyError)
    with pytest.raises(M.MissingFeatures):
        M.check_nulls(fitted, table[1].drop(columns=drop))


def test_a_non_numeric_feature_raises_instead_of_becoming_nan(fitted, table):
    rows = table[1].head(5).copy()
    rows[fitted.features[3]] = "n/a"
    with pytest.raises(ValueError):
        M.predict(fitted, rows)


# ── Null fractions: stratified train/serve parity (spec lesson 5) ────────────
def test_null_reference_is_the_training_rows_null_fractions(fitted, table):
    ref = fitted.registry["null_reference"]
    rs = M.training_rows(table[0], [SLICE_SEASON])
    assert sum(ref["n"].values()) == len(rs) == fitted.registry["n_rows"]["availability"]
    assert set(ref["null_frac"]) == set(ref["n"])
    first = rs[(rs["events_s"] == 0) & (rs["team_changed"] == 0)]
    assert ref["n"]["0|0"] == len(first) > 0
    # A player's first event of a season has no current-season minutes; the
    # event count itself is never NaN.
    assert ref["null_frac"]["0|0"]["min_ewm2_s"] == 1.0
    assert all(fr["events_s"] == 0.0 for fr in ref["null_frac"].values())
    assert ref["null_frac"]["0|0"]["p1_gp"] == round(float(first["p1_gp"].isna().mean()), 6)


def test_check_nulls_accepts_real_rows(fitted, table):
    M.check_nulls(fitted, M.training_rows(table[0], [SLICE_SEASON]))
    M.check_nulls(fitted, table[1])          # the next 4,000 rows of the same season
    M.check_nulls(fitted, table[1].head(0))  # nothing to check


def test_check_nulls_raises_on_a_column_nan_where_training_had_values(fitted, table):
    rows = table[1].copy()
    rows["t_pace"] = np.nan
    with pytest.raises(M.NullSkew, match=r"10\+\|0 t_pace serve 1\.00 vs train 0\.00"):
        M.check_nulls(fitted, rows)


def test_check_nulls_raises_on_a_column_filled_where_training_had_nan(fitted, table):
    """The classic fill skew: NaN turned into 0 at serve. The slice is
    2018-19 veterans, so the column is hours_to_tip (NaN before the injury
    report archive starts); the shipped test does the same with priors."""
    rows = table[1].assign(hours_to_tip=table[1]["hours_to_tip"].fillna(0.0))
    bad = M.null_skew(fitted, rows)
    assert set(bad["feature"]) == {"hours_to_tip"} and (bad["serve"] == 0.0).all()
    assert {"3-9|0", "10+|0"} <= set(bad["stratum"])
    with pytest.raises(M.NullSkew, match="hours_to_tip"):
        M.check_nulls(fitted, rows)


def test_opening_night_all_nan_current_season_columns_pass_but_filled_ones_raise(fitted, table):
    """Bucket 0 (no event yet this season) is all-NaN in its current-season
    columns in training too, so opening night passes without an exemption;
    filling those columns is the skew."""
    rs = M.training_rows(table[0], [SLICE_SEASON])
    first = rs[rs["events_s"] == 0]
    assert len(first) >= M.NULL_MIN_ROWS and first["min_ewm2_s"].isna().all()
    M.check_nulls(fitted, first)
    with pytest.raises(M.NullSkew, match=r"0\|0 min_ewm2_s serve 0\.00 vs train 1\.00"):
        M.check_nulls(fitted, first.assign(min_ewm2_s=0.0))


def test_null_check_skips_small_strata_and_raises_on_rows_without_a_stratum(fitted, table):
    few = table[1].head(M.NULL_MIN_ROWS - 1).assign(t_pace=np.nan)
    assert M.null_skew(fitted, few).empty
    keyless = table[1].head(3).copy()
    keyless.loc[keyless.index[0], "events_s"] = np.nan
    with pytest.raises(M.NullSkew, match=r"\(stratum keys\)"):
        M.check_nulls(fitted, keyless)


# ── load() verification ──────────────────────────────────────────────────────
def test_load_refuses_a_different_sklearn_before_unpickling(saved, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unpickled before the version check")
    monkeypatch.setattr(M.joblib, "load", boom)
    monkeypatch.setattr(sklearn, "__version__", "0.0.0")
    with pytest.raises(M.ModelMismatch, match=r"scikit-learn 0\.0\.0"):
        M.load(saved)


def test_load_refuses_a_corrupted_file(saved):
    p = saved / "rate_blk_v1.joblib"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(M.ModelMismatch, match=r"rate_blk_v1\.joblib: sha256"):
        M.load(saved)


def test_load_refuses_a_registry_sha_that_does_not_match(saved):
    rewrite_registry(saved, lambda r: r["files"].__setitem__("availability_v1.joblib", "0" * 64))
    with pytest.raises(M.ModelMismatch, match=r"availability_v1\.joblib: sha256"):
        M.load(saved)


def test_load_refuses_a_missing_file_and_a_missing_registry(saved):
    (saved / "minutes_v1.joblib").unlink()
    with pytest.raises(M.ModelMismatch, match=r"minutes_v1\.joblib: missing"):
        M.load(saved)
    (saved / M.REGISTRY_NAME).unlink()
    with pytest.raises(M.ModelMismatch, match="no model registry"):
        M.load(saved)


def test_load_refuses_registry_files_that_are_not_the_served_models(saved):
    rewrite_registry(saved, lambda r: r["files"].__setitem__("rate_xyz_v1.joblib", "0" * 64))
    with pytest.raises(M.ModelMismatch,
                       match=r"rate_xyz_v1\.joblib: listed in the registry but not a model this "
                             r"version serves"):
        M.load(saved)
    rewrite_registry(saved, lambda r: (r["files"].pop("rate_xyz_v1.joblib"),
                                       r["files"].pop("rate_pf_v1.joblib")))
    with pytest.raises(M.ModelMismatch, match=r"rate_pf_v1\.joblib: not in the registry"):
        M.load(saved)


def test_load_refuses_models_fit_on_other_columns_than_the_registry(saved):
    def swap(r):
        f = r["features"]
        f[0], f[1] = f[1], f[0]
    rewrite_registry(saved, swap)
    with pytest.raises(M.ModelMismatch, match="columns other than the registry features"):
        M.load(saved)


def _without(key):
    return lambda reg: json.dumps({k: v for k, v in reg.items() if k != key})


def _short_null_frac(reg):
    nf = {s: dict(list(fr.items())[1:]) for s, fr in reg["null_reference"]["null_frac"].items()}
    return json.dumps({**reg, "null_reference": {**reg["null_reference"], "null_frac": nf}})


UNREADABLE = {
    "truncated": lambda reg: json.dumps(reg)[:200],
    "not an object": lambda reg: "[]",
    "flat layout only (serving_design.md)": _without("versions"),
    "top level disagrees with versions": lambda reg: json.dumps({**reg, "sklearn": "9.9.9"}),
    "no trained_through": _without("trained_through"),
    "trained_through not the last season": lambda reg: json.dumps(
        {**reg, "trained_through": "2017-18"}),
    "no null_reference": _without("null_reference"),
    "null_reference missing a feature": _short_null_frac,
    "features not names": lambda reg: json.dumps({**reg, "features": [1, 2]}),
}


@pytest.mark.parametrize("case", list(UNREADABLE))
def test_an_unreadable_registry_is_a_model_mismatch(saved, case):
    """Callers catch ModelMismatch to fall back to the committed projections;
    a malformed registry must not escape as KeyError or JSONDecodeError."""
    path = saved / M.REGISTRY_NAME
    path.write_text(UNREADABLE[case](json.loads(path.read_text())))
    with pytest.raises(M.ModelMismatch, match="registry unreadable"):
        M.read_registry(saved)
    with pytest.raises(M.ModelMismatch, match="registry unreadable"):
        M.load(saved)


# ── Registry ─────────────────────────────────────────────────────────────────
def test_registry_records_what_was_trained(saved, table):
    reg = json.loads((saved / M.REGISTRY_NAME).read_text())
    part = table[0]
    assert reg["features"] == T.feature_columns(part)
    assert reg["stats"] == CONTRACT_STATS
    assert reg["train_seasons"] == [SLICE_SEASON] and reg["trained_through"] == SLICE_SEASON
    rs = part[part["game_id"].str.startswith("002")]
    assert reg["n_rows"]["availability"] == len(rs)
    assert reg["n_rows"]["minutes"] == reg["n_rows"]["rate_pts"] == int(rs["played"].sum())
    assert set(reg["versions"]) == {"sklearn", "joblib", "numpy", "pandas", "python"}
    assert reg["versions"]["sklearn"] == sklearn.__version__
    # serving_design.md section 2 layout: the versions at the top level too.
    for k in ("sklearn", "joblib", "numpy", "python"):
        assert reg[k] == reg["versions"][k]
    assert reg["hyperparameters"] == {**GATED_HGB, **GATED_EXTRA}
    assert re.fullmatch(r"v1-\d{4}-\d{2}-\d{2}", reg["model_version"])
    assert dt.datetime.fromisoformat(reg["trained_at"]).utcoffset() == dt.timedelta(0)
    want = {f"{n}_v1.joblib" for n in ["availability", "minutes", *(f"rate_{s}" for s in CONTRACT_STATS)]}
    assert set(reg["files"]) == want
    for f, sha in reg["files"].items():
        assert hashlib.sha256((saved / f).read_bytes()).hexdigest() == sha


def test_registry_window_drives_the_props_in_sample_guard(saved):
    """props.season_record drops picks from seasons the model trained on; it
    reads registry['trained_through']. Without the key it failed OPEN."""
    reg = M.read_registry(saved)
    assert P.is_in_training_window(SLICE_SEASON, reg)
    assert not P.is_in_training_window("2019-20", reg)


# ── Composition ──────────────────────────────────────────────────────────────
def test_combos_are_exact_sums_of_their_components(fitted, table):
    out = M.predict(fitted, table[1])
    np.testing.assert_array_equal(out["pra"], out["pts"] + out["reb"] + out["ast"])
    np.testing.assert_array_equal(out["pr"], out["pts"] + out["reb"])
    np.testing.assert_array_equal(out["pa"], out["pts"] + out["ast"])
    np.testing.assert_array_equal(out["ra"], out["reb"] + out["ast"])
    np.testing.assert_array_equal(out["stocks"], out["stl"] + out["blk"])


def test_fantasy_uses_the_prizepicks_weights(fitted, table):
    """PrizePicks NBA Fantasy Score: PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV."""
    out = M.predict(fitted, table[1])
    want = (out["pts"] + 1.2 * out["reb"] + 1.5 * out["ast"] + 3 * out["stl"] + 3 * out["blk"]
            - out["tov"])
    np.testing.assert_allclose(out["fantasy"], want, rtol=1e-12, atol=1e-12)


# ── The shipped artifacts (models/) ──────────────────────────────────────────
def _pins() -> dict:
    text = (REPO / "requirements.txt").read_text()
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, flags=re.M))


def test_shipped_registry_matches_the_requirements_pins():
    """The deploy installs requirements.txt; load() refuses any other sklearn.
    A registry trained on an unpinned version would take the site down."""
    reg = M.read_registry()
    pins = _pins()
    assert reg["versions"]["sklearn"] == pins["scikit-learn"]
    assert reg["versions"]["joblib"] == pins["joblib"]


def test_shipped_models_are_the_gated_configuration():
    """No table needed: a retrain with another loss, weight-free fit settings,
    an extra estimator argument or another window fails here."""
    m = M.load()
    assert_gated_estimators(m)
    reg = m.registry
    assert reg["hyperparameters"] == {**GATED_HGB, **GATED_EXTRA}
    assert reg["train_seasons"] == TRAIN_SEASONS and reg["trained_through"] == "2025-26"
    assert reg["season_type"] == "regular season (game_id 002)"


def test_shipped_registry_keeps_the_training_seasons_out_of_the_record():
    reg = M.read_registry()
    assert all(P.is_in_training_window(s, reg) for s in TRAIN_SEASONS)
    assert not P.is_in_training_window("2026-27", reg)


def test_shipped_null_reference_covers_every_training_row():
    reg = M.read_registry()
    ref = reg["null_reference"]
    assert sum(ref["n"].values()) == reg["n_rows"]["availability"]
    assert set(ref["n"]) == {f"{b}|{t}" for b in M.NULL_LABELS for t in (0, 1)}
    assert ref["null_frac"]["0|0"]["min_ewm2_s"] == 1.0
    assert all(fr["events_s"] == 0.0 and fr["team_changed"] == 0.0
               for fr in ref["null_frac"].values())


def test_shipped_models_load_and_predict_cold_rows():
    """No table needed: an opening-night row with every feature NaN."""
    m = M.load()
    rows = pd.DataFrame(np.nan, index=[10, 11], columns=m.features)
    out = M.predict(m, rows)
    assert list(out.columns) == CONTRACT_COLS and list(out.index) == [10, 11]
    assert np.isfinite(out.to_numpy()).all() and out["p_play"].between(0, 1).all()


def test_shipped_registry_matches_the_training_table():
    table_or_skip()
    reg = M.read_registry()
    cols = pq.read_schema(TABLE).names
    assert reg["features"] == T.feature_columns(pd.DataFrame(columns=cols))
    assert reg["train_seasons"] == M.TRAIN_SEASONS
    t = pd.read_parquet(TABLE, columns=["game_id", "season", "played"])
    rs = t[t["game_id"].str.startswith("002") & t["season"].isin(M.TRAIN_SEASONS)]
    assert reg["n_rows"]["availability"] == len(rs)
    assert reg["n_rows"]["minutes"] == int(rs["played"].sum())


def test_shipped_models_predict_real_rows_sanely(real):
    rs = real.tail[real.tail["game_id"].str.startswith("002")]
    m = M.load()
    out = M.predict(m, rs)
    played = rs["played"].astype(bool)
    assert abs(out["p_play"].mean() - played.mean()) < 0.05
    assert abs(out.loc[played, "mu_min"].mean() - rs.loc[played, "min"].mean()) < 2.0
    assert abs(out.loc[played, "pts"].mean() - rs.loc[played, "pts"].mean()) < 1.5


# (part, whole): a made count is part of its attempts, a three part of the field
# goals. Spelled out here, not read from models.RATE_CEILINGS.
PART_OF = [("fg3a", "fga"), ("fgm", "fga"), ("fg3m", "fg3a"), ("fg3m", "fgm"), ("ftm", "fta")]


def test_shipped_models_never_project_more_makes_than_attempts():
    """Every row of the real table: the shipped rate models are fit apart and
    disagree on real rows (measured 2026-10-09: ftm above fta on 837, fg3m
    above fg3a on 87, fg3a above fga on 21, fg3m above fgm on 8), and
    predict() still never serves a part's rate or mean above its whole's.
    A row the raw models put above the cap is served AT the cap."""
    table_or_skip()
    m = M.load()
    rows = pd.read_parquet(TABLE, columns=m.features)
    X = M.feature_matrix(m, rows)
    raw = {s: m.rates[s].predict(X) for s in {s for pair in PART_OF for s in pair}}
    out = M.predict(m, rows)
    for part, whole in PART_OF:
        assert (out[f"rate_{part}"] <= out[f"rate_{whole}"]).all(), (part, whole)
        assert (out[part] <= out[whole]).all(), (part, whole)
    for part, whole in [("fg3a", "fga"), ("fg3m", "fg3a"), ("fg3m", "fgm"), ("ftm", "fta")]:
        over = raw[part] > raw[whole]
        assert over.sum() > 0, f"no raw {part} > {whole} left: this check measures nothing"
        assert (out[f"rate_{part}"].to_numpy()[over] <= raw[whole][over]).all(), (part, whole)
    over = raw["ftm"] > raw["fta"]
    np.testing.assert_array_equal(out["rate_ftm"].to_numpy()[over], raw["fta"][over])
    np.testing.assert_array_equal(out["ftm"].to_numpy()[over], out["fta"].to_numpy()[over])


def test_shipped_null_check_passes_real_slates_and_catches_serve_skews(real):
    """Each real slate (one serving call) passes, opening night included; the
    skews a serving path makes (NaN priors filled with 0, lines missing,
    current-season columns filled on opening night) raise."""
    m = M.load()
    for d, rows in real.slates.items():
        M.check_nulls(m, rows)
    mid, opener = real.slates["2026-01-14"], real.slates["2025-10-21"]
    priors = [c for c in m.features if c.startswith("p2_")]
    with pytest.raises(M.NullSkew, match="p2_gp"):
        M.check_nulls(m, mid.assign(**{c: mid[c].fillna(0.0) for c in priors}))
    with pytest.raises(M.NullSkew, match="home_line"):
        M.check_nulls(m, mid.assign(home_line=np.nan))
    with pytest.raises(M.NullSkew, match=r"0\|0 min_std_s serve 0\.00 vs train 1\.00"):
        M.check_nulls(m, opener.assign(min_std_s=0.0))


def _listed_out(rows: pd.DataFrame, share: float) -> tuple[pd.DataFrame, np.ndarray]:
    """`rows` with the first `share` of their 10+|0 rows listed Out the way
    livectx.build_context encodes it (inj_out 1, the other status one-hots
    0, tm_out_max_min NaN: an Out player is not redistributed to), and the
    mask of the rows changed."""
    rows = rows.copy()
    ten = np.flatnonzero((M.null_strata(rows) == "10+|0").to_numpy())
    mask = np.zeros(len(rows), bool)
    mask[ten[: int(np.ceil(share * len(ten)))]] = True
    for c in ("inj_doubtful", "inj_questionable", "inj_probable", "inj_available",
              "inj_not_listed", "inj_unknown"):
        rows.loc[mask, c] = 0.0
    rows.loc[mask, "inj_out"] = 1.0
    rows.loc[mask, "tm_out_max_min"] = np.nan
    return rows, mask


def test_a_slate_with_many_players_out_passes_the_null_check(real):
    """tm_out_max_min is NaN exactly on the rows listed Out (livectx; exact on
    every training row), so its null fraction is the slate's Out share. The
    registry's 10+|0 reference (0.16) pools the four horizons, while serving
    checks the games still to tip: 2025-26 rebuilt at t_15 had 0.37 on the
    15-game final day and 0.40 on its 8-game remainder, and the fraction test
    withheld every untipped game of those dates. A real slate with 40% of
    its 10+|0 rows Out projects; models.STRUCTURAL_NAN checks the
    invariant instead."""
    m = M.load()
    out, mask = _listed_out(real.slates["2026-01-14"], 0.40)
    ten = (M.null_strata(out) == "10+|0").to_numpy()
    assert out.loc[ten, "tm_out_max_min"].isna().mean() > 0.40
    assert (out["tm_out_max_min"].isna() == out["inj_out"].eq(1.0)).all()
    M.check_nulls(m, out)
    assert M.null_skew(m, out).empty


def test_the_out_invariant_catches_both_skews_on_any_slate_size(real):
    """What the fraction test on tm_out_max_min stood for, checked exactly:
    NaN on a row not listed Out (a serving path that lost the value), or a
    value on a row listed Out (a fill), raises on a slate of any size, the
    small strata check_nulls skips included."""
    m = M.load()
    out, mask = _listed_out(real.slates["2026-01-14"], 0.40)
    healthy = np.flatnonzero(~out["inj_out"].eq(1.0).to_numpy())[:5]
    lost = out.copy()
    lost.iloc[healthy, lost.columns.get_loc("tm_out_max_min")] = np.nan
    with pytest.raises(M.NullSkew, match=r"tm_out_max_min is NaN exactly where inj_out == 1 in "
                                         r"training, broken on 5 of"):
        M.check_nulls(m, lost)
    filled = out.copy()
    filled.iloc[np.flatnonzero(mask)[:1], filled.columns.get_loc("tm_out_max_min")] = 0.0
    with pytest.raises(M.NullSkew, match="broken on 1 of"):
        M.check_nulls(m, filled)
    few = lost.iloc[healthy[:2]]
    assert len(few) < M.NULL_MIN_ROWS
    bad = M.null_skew(m, few)
    assert bad["feature"].tolist() == ["tm_out_max_min"] and bad["serve"].tolist() == [1.0]


# ── Team minutes coherence (models.cohere) ───────────────────────────────────
# A real game: ATL at IND on 2026-11-02, the bot's committed rows at 6:13 PM ET
# (tests/fixtures/cohere, cut from state/serve/2026-11-02/projections_v1).
# Indiana listed ten players Out; its seven others summed to 188 projected
# team minutes. It played six of them, 34.5 to 44.7 minutes each.
COHERE_FIX = REPO / "tests" / "fixtures" / "cohere" / "atl_ind_2026-11-02.parquet"
IND_PLAYED = {"Jarace Walker": 44.7, "Braden Smith": 42.2, "Jalen Slawson": 40.6, "Jay Huff": 40.1,
              "Kobe Brown": 37.8, "Quenton Jackson": 34.5}


def _served(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predict()-shaped output, the rows it scored) from a fixture frame of
    p_play, mu_min, rate_<s>, inj_status (or inj_out) per (game_id, team)."""
    stats = [c[len("rate_"):] for c in frame.columns if c.startswith("rate_")]
    pred = frame[["p_play", "mu_min", *(f"rate_{s}" for s in stats)]].copy()
    M._compose(pred, stats)
    out = frame["inj_out"] if "inj_out" in frame else (frame["inj_status"] == "Out").astype(float)
    return pred, frame[["game_id", "team", "person_id"]].assign(inj_out=out.to_numpy())


PAD = 14   # a constructed team fills up to this many rows with players listed Out


def _team(mu, p=None, out=None, team="AAA", game="g1", pad=PAD) -> pd.DataFrame:
    """A constructed team: the players given, then rows listed Out (p_play 0,
    10 minutes) up to `pad` rows, as a real roster carries its absences (a
    side under COHERE_MIN_ROWS rows is left alone). Rates of 0.5 points and
    0.2 of everything else a minute, so the means follow mu_min alone. The
    `pad` column marks the filler."""
    n = len(mu)
    k = max(pad - n, 0)
    f = pd.DataFrame({"game_id": game, "team": team, "person_id": np.arange(n + k) + 1000 * len(team),
                      "p_play": np.r_[np.ones(n) if p is None else np.asarray(p, float), np.zeros(k)],
                      "mu_min": np.r_[np.asarray(mu, float), np.full(k, 10.0)],
                      "inj_out": np.r_[np.zeros(n) if out is None else np.asarray(out, float), np.ones(k)],
                      "pad": np.r_[np.zeros(n, bool), np.ones(k, bool)]})
    for s in CONTRACT_STATS:
        f[f"rate_{s}"] = 0.5 if s == "pts" else 0.2
    return f


def _team_sum(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    return (pred["p_play"] * pred["mu_min"]).groupby([rows["game_id"], rows["team"]]).sum()


def test_cohere_gives_indiana_its_missing_minutes():
    f = pd.read_parquet(COHERE_FIX)
    pred, rows = _served(f)
    after = M.cohere(pred, rows)
    before_sum, after_sum = _team_sum(pred, rows), _team_sum(after, rows)
    ind = ("0022600160", "IND")
    assert 187 < before_sum[ind] < 189 and 233 < after_sum[ind] < M.TEAM_MINUTES
    healthy = (rows["team"] == "IND") & rows["inj_out"].eq(0)
    assert (after.loc[healthy, "mu_min"] > pred.loc[healthy, "mu_min"] + 5).all()
    # Closer to what they played, every one of the six who did.
    played = f["player_name"].map(IND_PLAYED)
    err0 = (played - pred["mu_min"]).abs()[played.notna()]
    err1 = (played - after["mu_min"]).abs()[played.notna()]
    assert len(err0) == 6 and (err1 < err0).all() and err1.mean() < 0.6 * err0.mean()


def test_cohere_serves_the_validated_numbers_on_a_real_game():
    """The step docs/model_report.md validated, pinned on the fixture: any
    change to its constants or its rule moves these (the relative tests below
    would let a refit through; this one says it happened)."""
    f = pd.read_parquet(COHERE_FIX)
    pred, rows = _served(f)
    out = M.cohere(pred, rows)
    sums = (out["p_play"] * out["mu_min"]).groupby(rows["team"]).sum()
    np.testing.assert_allclose(sums[["ATL", "IND"]].to_numpy(), [239.92939987059418, 236.3949023503905],
                               rtol=0, atol=1e-9)
    got = out.assign(name=f["player_name"].to_numpy()).set_index("name")
    for name, mins, pts in (("Jarace Walker", 34.888212170869366, 17.599585812292542),
                            ("Braden Smith", 30.619864100555382, 9.515628823896908),
                            ("CJ McCollum", 29.71921895876544, 18.099614723316364),
                            ("Corey Kispert", 14.870946394300628, 7.694465813015802),
                            ("RayJ Dennis", 12.722884306271443, 5.07103180845935)):
        np.testing.assert_allclose([got.loc[name, "mu_min"], got.loc[name, "pts"]], [mins, pts],
                                   rtol=0, atol=1e-9, err_msg=name)


def test_the_shipped_scales_are_the_validated_ones():
    """docs/model_report.md and docs/cohere_tm1.csv validated exactly these. A
    change is a refit (scripts/fit_cohere.py, a new report section), never an
    edit that the relative tests would let through."""
    assert (M.TEAM_MINUTES, M.COHERE_S0, M.COHERE_FULL, M.COHERE_CAP, M.COHERE_FLOOR,
            M.COHERE_MIN_ROWS) == (241.5, 30.0, 20.0, 44.0, 1.0, 12)


def test_out_rows_keep_their_minutes_whichever_way_the_team_moves():
    """Short (the real fixture: ATL and IND both under) and over (a constructed
    team 262 minutes deep, with two Out rows at p_play 0.004 and its filler)."""
    over = _team([36, 34, 32, 30, 30, 28, 26, 24, 22, 30, 25], p=[1.0] * 9 + [0.004] * 2,
                 out=[0.0] * 9 + [1.0] * 2, team="OVR")
    for frame, n_out in ((pd.read_parquet(COHERE_FIX), 17), (over, 5)):
        pred, rows = _served(frame)
        after = M.cohere(pred, rows)
        out = rows["inj_out"].eq(1.0).to_numpy()
        moved = (after["mu_min"] - pred["mu_min"]).to_numpy()
        assert out.sum() == n_out and (np.abs(moved[~out]) > 0.1).all()
        np.testing.assert_array_equal(after.loc[out, "mu_min"], pred.loc[out, "mu_min"])
        for c in ["p_play", *(c for c in pred.columns if c.startswith("rate_"))]:
            np.testing.assert_array_equal(after[c], pred[c], err_msg=c)


def test_cohere_recomposes_every_mean_from_the_new_minutes():
    pred, rows = _served(pd.read_parquet(COHERE_FIX))
    out = M.cohere(pred, rows)
    for s in CONTRACT_STATS:
        np.testing.assert_allclose(out[s], out["mu_min"] * out[f"rate_{s}"], rtol=1e-12, err_msg=s)
    np.testing.assert_array_equal(out["pra"], out["pts"] + out["reb"] + out["ast"])
    np.testing.assert_array_equal(out["stocks"], out["stl"] + out["blk"])
    want = (out["pts"] + 1.2 * out["reb"] + 1.5 * out["ast"] + 3 * out["stl"] + 3 * out["blk"]
            - out["tov"])
    np.testing.assert_allclose(out["fantasy"], want, rtol=1e-12, atol=1e-12)
    assert list(out.columns) == list(pred.columns) and out.index.equals(pred.index)


def test_each_team_game_is_cohered_on_its_own_rows_in_any_order():
    pred, rows = _served(pd.read_parquet(COHERE_FIX))
    both = M.cohere(pred, rows)
    ind = (rows["team"] == "IND").to_numpy()
    alone = M.cohere(pred[ind], rows[ind])
    pd.testing.assert_frame_equal(both[ind], alone, check_exact=True)
    order = np.random.default_rng(0).permutation(len(pred))
    shuffled = M.cohere(pred.iloc[order], rows.iloc[order])      # sums in another order: to rounding
    pd.testing.assert_frame_equal(shuffled.sort_index(), both.sort_index(), rtol=1e-12, atol=1e-12)


def test_a_team_in_two_games_is_cohered_per_game():
    """The key is (game_id, team): the same club on two dates, as
    scripts/fit_cohere.py and the mark-out gate pass whole seasons, is two
    team-games, one short and one over here."""
    g1 = _team([30.0] * 7, team="BOS", game="g1")          # 210 minutes
    g2 = _team([36.0] * 8, team="BOS", game="g2")          # 288 minutes
    pred, rows = _served(pd.concat([g1, g2], ignore_index=True))
    both = M.cohere(pred, rows)
    for g in ("g1", "g2"):
        m = (rows["game_id"] == g).to_numpy()
        pd.testing.assert_frame_equal(both[m], M.cohere(pred[m], rows[m]), check_exact=True)
    s = _team_sum(both, rows)
    assert abs(s[("g1", "BOS")] - M.TEAM_MINUTES) < 1e-9 and abs(s[("g2", "BOS")] - M.TEAM_MINUTES) < 1e-9


def test_a_certain_lineup_adds_up_and_every_player_gains_the_same():
    """Nobody's availability in doubt (p 1 or 0): the whole gap is minutes,
    shared equally while no one is past COHERE_FULL."""
    f = _team([18, 18, 17, 16, 16, 15, 14, 14, 12, 10, 9, 20], p=[1.0] * 9 + [0.0] * 3,
              out=[0.0] * 9 + [1.0] * 3)
    pred, rows = _served(f)
    after = M.cohere(pred, rows)
    gain = (after["mu_min"] - pred["mu_min"]).to_numpy()
    assert abs(_team_sum(after, rows).iloc[0] - M.TEAM_MINUTES) < 1e-9
    np.testing.assert_allclose(gain[:9], (M.TEAM_MINUTES - 140) / 9, rtol=1e-12)
    np.testing.assert_array_equal(gain[9:], 0.0)


def test_doubt_about_who_plays_moves_a_team_part_of_the_way_both_ways():
    """Sure of everyone, a short team goes all the way to TEAM_MINUTES; in
    much doubt it goes part of the way, short or over. Every player not Out
    moves by the one shift (all here at one mu, so one weight), a doubtful
    one exactly as far as a sure one."""
    sure = _team([25.0] * 8, team="SUR")                               # 200 minutes, no doubt
    unsure = _team([25.0] * 12, p=[1.0] * 4 + [0.5] * 8, team="UNS")   # 200 minutes, much doubt
    deep = _team([35.0] * 12, p=[1.0] * 4 + [0.6] * 8, team="DEP")     # 308 minutes, much doubt
    pred, rows = _served(pd.concat([sure, unsure, deep], ignore_index=True))
    after = M.cohere(pred, rows)
    s0, s1 = _team_sum(pred, rows), _team_sum(after, rows)
    assert s0[("g1", "SUR")] == s0[("g1", "UNS")] == 200.0 and abs(s0[("g1", "DEP")] - 308.0) < 1e-9
    assert abs(s1[("g1", "SUR")] - M.TEAM_MINUTES) < 1e-9
    assert 205 < s1[("g1", "UNS")] < 230 and 255 < s1[("g1", "DEP")] < 295
    move = (after["mu_min"] - pred["mu_min"]).to_numpy()
    for team in ("UNS", "DEP"):
        m = move[((rows["team"] == team) & rows["inj_out"].eq(0)).to_numpy()]
        assert len(m) == 12 and abs(m[0]) > 0.5
        np.testing.assert_allclose(m, m[0], rtol=1e-12)


def test_heavier_players_move_less_both_ways():
    short = _team([40, 36, 30, 20, 12, 8], team="SHT")                          # 146 minutes
    over = _team([40, 36, 30, 30, 30, 30, 30, 30, 20, 12], team="OVR")          # 288 minutes
    pred, rows = _served(pd.concat([short, over], ignore_index=True))
    after = M.cohere(pred, rows)
    move = (after["mu_min"] - pred["mu_min"]).to_numpy()
    real = ~pd.concat([short, over], ignore_index=True)["pad"].to_numpy()
    sht, ovr = real & (rows["team"] == "SHT").to_numpy(), real & (rows["team"] == "OVR").to_numpy()
    # Rows run heaviest first; the three at or under COHERE_FULL move alike.
    assert (np.diff(move[sht]) >= -1e-12).all() and move[sht][0] > 0
    np.testing.assert_allclose(move[sht][3:], move[sht][-1], rtol=1e-12)
    assert (np.diff(move[ovr]) <= 1e-12).all() and move[ovr][0] < 0
    assert abs(_team_sum(after, rows)[("g1", "SHT")] - M.TEAM_MINUTES) < 1e-9


def test_the_shift_tapers_for_heavier_players_before_any_cap():
    """A modest gap and nobody near 44, short and over: a 36-minute player
    moves a third as much as one at or under 20 (the documented weight
    (44 - mu) / (44 - 20)), so the taper, not the cap, is what spares him."""
    short = _team([36, 30, 24, 16, 16, 14, 12, 10, 10, 10, 6, 6], team="SHT")    # 190 minutes
    over = _team([36, 30, 30, 30, 30, 28, 26, 24, 16, 12], team="OVR")          # 262 minutes
    pred, rows = _served(pd.concat([short, over], ignore_index=True))
    after = M.cohere(pred, rows)
    assert after["mu_min"].max() < 39                                          # no cap anywhere
    move = (after["mu_min"] - pred["mu_min"]).to_numpy()
    for team, star, light in (("SHT", 0, 3), ("OVR", 0, 8)):
        m = move[(rows["team"] == team).to_numpy()]
        assert abs(m[star]) < 0.5 * abs(m[light]) and np.sign(m[star]) == np.sign(m[light]) != 0
        np.testing.assert_allclose(m[star] / m[light], 1 / 3, rtol=1e-9)


def test_the_floor_and_the_cap_bind_where_a_shift_would_cross_them():
    """Far over and sure of everyone: the deepest bench would go below zero
    and stops at min(mu, 1), the team still reaching TEAM_MINUTES. Five healthy
    players far short: every one stops at 44, the team short of the target
    rather than anyone pushed past it."""
    deep = _team([36, 35, 34, 33, 32, 30, 30, 28, 26, 24, 22, 20, 18, 5, 3.5, 2, 0.6],
                 team="DEP", pad=0)                                            # 379.1 minutes
    five = _team([30, 28, 26, 24, 22], team="FIV")                             # 130, nine Out
    pred, rows = _served(pd.concat([deep, five], ignore_index=True))
    after = M.cohere(pred, rows)
    d = (rows["team"] == "DEP").to_numpy()
    np.testing.assert_array_equal(after.loc[d, "mu_min"].to_numpy()[-4:], [1.0, 1.0, 1.0, 0.6])
    assert (after.loc[d, "mu_min"].to_numpy()[:-4] > 1.0).all()
    assert abs(_team_sum(after, rows)[("g1", "DEP")] - M.TEAM_MINUTES) < 1e-9
    healthy = ((rows["team"] == "FIV") & rows["inj_out"].eq(0)).to_numpy()
    np.testing.assert_array_equal(after.loc[healthy, "mu_min"], 44.0)
    assert _team_sum(after, rows)[("g1", "FIV")] == 220.0


def test_a_side_under_the_training_minimum_is_left_as_predicted():
    """Eleven rows is fewer than any training team-game had: a broken roster
    pull, whose sum says nothing about minutes, so cohere leaves it alone; at
    twelve it moves. The threshold is pipeline's MIN_TEAM_ROWS."""
    from bucketlib import pipeline as P
    assert M.COHERE_MIN_ROWS == P.MIN_TEAM_ROWS == 12
    eleven = _team([30.0] * 6, team="ELV", pad=11)
    twelve = _team([30.0] * 6, team="TWL", pad=12)
    pred, rows = _served(pd.concat([eleven, twelve], ignore_index=True))
    after = M.cohere(pred, rows)
    e = (rows["team"] == "ELV").to_numpy()
    pd.testing.assert_frame_equal(after[e], pred[e], check_exact=True)
    assert (after.loc[~e & rows["inj_out"].eq(0).to_numpy(), "mu_min"] > 30.5).all()


def test_a_team_that_already_adds_up_is_untouched():
    f = _team([30, 30, 30, 30, 30, 30, 30, 31.5])          # exactly TEAM_MINUTES
    pred, rows = _served(f)
    assert _team_sum(pred, rows).iloc[0] == M.TEAM_MINUTES
    pd.testing.assert_frame_equal(M.cohere(pred, rows), pred, check_exact=True)


def test_cohere_refuses_rows_it_cannot_group_or_align():
    pred, rows = _served(_team([30.0] * 8))
    with pytest.raises(KeyError, match="inj_out"):
        M.cohere(pred, rows.drop(columns=["inj_out"]))
    with pytest.raises(ValueError, match="share one index"):
        M.cohere(pred, rows.set_axis(rows.index + 100))
    assert M.cohere(pred.iloc[:0], rows.iloc[:0]).empty


def test_the_served_version_names_the_team_minutes_step():
    m = M.Models(None, None, {}, {"model_version": "v1-2026-10-09"})
    assert m.version == f"v1-2026-10-09+{M.COHERE_TAG}" == "v1-2026-10-09+tm1"
