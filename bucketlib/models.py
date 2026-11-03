"""Production models: the gated configuration, trained once on every season,
saved beside a registry, and loaded only when this runtime can reproduce them.

WHAT IS SERVED. Exactly the structure that passed the ship gate
(docs/model_report.md; scripts/train_eval.py produced the gated predictions):
  availability   P(plays at least one second), a classifier over every rostered row
  minutes        E[MIN | plays], squared-error regressor on played rows
  rate_<stat>    per-minute rate, exposure Poisson (target stat/min capped at
                 RATE_CAP, weighted by the minutes) on played rows
Every stat mean is E[stat | plays] = mu_min x rate (spec lesson 1: two
estimands, never mixed; p_play lives in its own column and is NOT multiplied
in). Combos and Fantasy are SUMS of the component means, so a combo can never
disagree with its parts and the simulator re-centres on numbers that add up.

WHY A REGISTRY. A pickled model is the model it claims to be only when (a) the
bytes on disk are the bytes that were trained, (b) the scikit-learn unpickling
it is the one that pickled it (version drift breaks joblib.load, loudly or
silently depending on the release), and (c) it is fed the columns it was fit
on, in order. load() checks (a) and (b) and raises ModelMismatch; predict()
checks (c) and raises MissingFeatures rather than filling. A filled column is a
train/serve skew no metric ever shows (spec lesson 5: GridironValue had four,
DiamondValue's reindex still fails open).

WHY NULL FRACTIONS IN THE REGISTRY. A column that is present but NaN where
training had values (or filled where training had NaN) passes MissingFeatures.
Spec lesson 5 checks it STRATIFIED, and the training table is local only, so
the training null fractions per stratum ship inside the registry and
check_nulls() compares serve rows against them. A feature that is NaN by
construction where another feature says so (STRUCTURAL_NAN) is checked by
that invariant instead of by fraction.

Nothing is cached at module level: the app wraps load() in its own resource
cache, and every test gets a fresh load it can throw away.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import cache as C
from . import train as T
from .util import now_utc, season_end_year, today_et

MODELS_DIR = C.MODELS_DIR
VERSION = "v1"
REGISTRY_NAME = f"registry_{VERSION}.json"

# The 15 rate models, in the gated order (scripts/validate_full.STATS; a test
# pins the two together). Combos and Fantasy come from train.COMBOS and
# train.FANTASY_W, never a second copy.
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta", "fgm", "fga", "fg3a",
         "oreb", "dreb", "pf"]
COMPOSED = [*T.COMBOS, "fantasy"]

# Every regular season with participation, injury horizons and lines. The gate
# is spent (validate, test and the one-time 2025-26 holdout all passed), so the
# shipped model trains on all of them; nothing here is tuned on any season.
TRAIN_SEASONS = T.seasons("2018-19", "2025-26")

# THE gated hyperparameters, one definition for every production model.
# train_eval.py, validate_full.py and fit_dist.py keep literal copies so the
# reports they wrote stay reproducible; tests/test_models.py pins them equal.
HGB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0)
MAX_MIN = 53.0          # mu_min is clipped to [0, MAX_MIN], as the gate scored it
RATE_CAP = 5.0          # per-minute target cap: a 2-second cameo with a rebound is not a rate
MIN_EXPOSURE = 1 / 60   # one second: the floor on minutes in the rate target and its weight
# A made count is part of its attempts, a three is part of the field goals:
# (rate, the rates it may not exceed), applied in order so each ceiling is
# itself already capped (predict).
RATE_CEILINGS = (("fg3a", ("fga",)), ("fgm", ("fga",)), ("fg3m", ("fg3a", "fgm")),
                 ("ftm", ("fta",)))

# Null-fraction strata (spec lesson 5): the player's games into the season
# (events_s: every roster event, played or not) bucketed 0 / 1-2 / 3-9 / 10+,
# crossed with team_changed. Opening night is all-NaN in its current-season
# columns, and so are training's bucket-0 rows, so neither needs an exemption.
NULL_KEYS = ("events_s", "team_changed")
NULL_BINS = (-np.inf, 0, 2, 9, np.inf)
NULL_LABELS = ("0", "1-2", "3-9", "10+")
# A (stratum, feature) cell is skewed when |serve - train| null fraction exceeds
# max(NULL_TOL, NULL_Z x the binomial sd of the difference). Calibrated
# leave-one-season-out on the training table with null_skew() itself, one call
# per game date (2026-10-09): over the 1,106 dates of 2019-20..2025-26 it fired
# on 6, each a real data event rather than noise (4 Omicron hardship-signing
# dates, Dec 2021 to Jan 2022; the 2023-24 opener with no report for 63% of rows; one
# 2020-21 date of tm_out_max_min), and 0.25 fired on the same 6. Held-out
# 2018-19 fires on its 63 dates before the injury-report archive starts, the
# era difference it exists to see. Strata under NULL_MIN_ROWS rows, serve or
# training, are too noisy to judge and are skipped.
NULL_TOL = 0.20
NULL_Z = 4.0
NULL_MIN_ROWS = 20
# Features that are NaN BY CONSTRUCTION exactly where another feature says so,
# mapped to that feature (its rows are NaN iff it is 1). livectx.build_context:
# a player listed Out is not redistributed to, so his tm_out_max_min is NaN
# (exact on all 325,371 training rows). Its null fraction is therefore the
# slate's Out share, a property of the slate, not of the serving path, and the
# registry's reference for it pools the four horizons while serving checks the
# games still to tip, where the Out share runs higher (2025-26 rebuilt at t_15:
# 0.37 on the 15-game final day, 0.40 on its 8-game remainder, against 0.16
# pooled). The fraction test fired on such slates and withheld every untipped
# game of the date. null_skew checks these features by their invariant
# instead, exactly and on every row whatever the slate size: NaN on a row the
# key does not flag, or a value on a row it does, is the skew. Their
# null_reference entries stay (the registry must cover every feature) and are
# not read. The NULL_TOL calibration above pooled mixed horizons per date, so
# it never saw a remaining-slate subset; this is why it did not fire there.
STRUCTURAL_NAN = {"tm_out_max_min": "inj_out"}


class ModelMismatch(RuntimeError):
    """The files on disk are not the models the registry describes, or this
    runtime cannot reproduce them. Callers never fall back to a guess: a site
    without loadable models shows the committed projections, nothing new."""


class MissingFeatures(KeyError):
    """Rows lack registry features. Names EVERY absent column (fixing them one
    error at a time hides how far a serving path has drifted)."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"{len(self.missing)} trained-on feature(s) absent from the rows "
                         f"(never filled): {', '.join(self.missing)}")

    def __str__(self) -> str:  # KeyError would print the message as a quoted repr
        return self.args[0]


class NullSkew(ValueError):
    """Serve rows' null fractions differ from training's in the same stratum:
    a serving path is NaN where training had values, or filled where training
    had NaN. Names EVERY skewed (stratum, feature) cell."""


@dataclass
class Models:
    availability: HistGradientBoostingClassifier
    minutes: HistGradientBoostingRegressor
    rates: dict[str, HistGradientBoostingRegressor]
    registry: dict

    @property
    def features(self) -> list[str]:
        return self.registry["features"]

    @property
    def stats(self) -> list[str]:
        return self.registry["stats"]

    @property
    def version(self) -> str:
        """Stamped on every projection row (props contract: model_version):
        the registry's model_version plus the serving step's tag (COHERE_TAG),
        so a row says which team-minutes step made its numbers too."""
        return f"{self.registry['model_version']}+{COHERE_TAG}"

    def estimators(self) -> dict:
        return {"availability": self.availability, "minutes": self.minutes,
                **{f"rate_{s}": self.rates[s] for s in self.stats}}


def model_names(stats: list[str]) -> list[str]:
    return ["availability", "minutes", *(f"rate_{s}" for s in stats)]


def file_name(name: str) -> str:
    return f"{name}_{VERSION}.joblib"


def output_columns(stats: list[str]) -> list[str]:
    return ["p_play", "mu_min", *(f"rate_{s}" for s in stats), *stats, *COMPOSED]


def runtime_versions() -> dict:
    """Read at call time, so a test can monkeypatch sklearn.__version__."""
    return {"sklearn": sklearn.__version__, "joblib": joblib.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": platform.python_version()}


# ── Training ─────────────────────────────────────────────────────────────────
def availability_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB)


def regressor(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, **HGB)


def training_rows(table: pd.DataFrame, seasons: list[str] = TRAIN_SEASONS) -> pd.DataFrame:
    """Regular-season rows (game_id 002...) of `seasons`. Play-in rows stay out,
    as they did in the gate. Raises when a requested season has no rows: a
    silently absent season trains a different model than the registry claims."""
    rows = table[table["game_id"].str.startswith("002") & table["season"].isin(seasons)]
    absent = [s for s in seasons if not (rows["season"] == s).any()]
    if absent:
        raise ValueError(f"no regular-season rows for {absent} in the training table")
    return rows


def rate_target(played: pd.DataFrame, stat: str) -> tuple[pd.Series, pd.Series]:
    """(stat per minute capped at RATE_CAP, minutes weight). The exposure
    Poisson: weighting the per-minute rate by minutes makes the fit a Poisson
    on the count with log(minutes) as offset."""
    w = played["min"].clip(lower=MIN_EXPOSURE)
    return (played[stat] / w).clip(upper=RATE_CAP), w


def null_strata(rows: pd.DataFrame) -> pd.Series:
    """Stratum label per row ('0|0' .. '10+|1'), the ONE definition training's
    reference and the serve check share. None where a key is NaN or
    team_changed is not 0/1: training never has that, so the check raises."""
    bucket = pd.cut(rows[NULL_KEYS[0]].astype("float64"), list(NULL_BINS),
                    labels=list(NULL_LABELS)).astype(object)
    tc = rows[NULL_KEYS[1]].astype("float64")
    ok = bucket.notna() & tc.isin([0.0, 1.0])
    out = pd.Series(None, index=rows.index, dtype=object)
    out[ok] = bucket[ok].astype(str) + "|" + tc[ok].astype(int).astype(str)
    return out


def null_reference(rows: pd.DataFrame, features: list[str]) -> dict:
    """Training null fraction of every feature in every stratum (rounded to
    6 places), plus the stratum sizes. Built on the availability rows, the same
    population serving projects: every rostered player for every game."""
    st = null_strata(rows)
    if st.isna().any():
        raise ValueError(f"{int(st.isna().sum())} training rows have no null stratum "
                         f"({'/'.join(NULL_KEYS)} NaN or not 0/1)")
    frac = rows[features].isna().groupby(st).mean()
    n = st.value_counts()
    return {"keys": list(NULL_KEYS), "buckets": list(NULL_LABELS),
            "n": {s: int(n[s]) for s in frac.index},
            "null_frac": {s: {f: round(float(frac.at[s, f]), 6) for f in features}
                          for s in frac.index}}


def train(table: pd.DataFrame, seasons: list[str] = TRAIN_SEASONS,
          log: Callable[[str], None] | None = None) -> Models:
    """Fit every production model on `table` (features.build output with
    outcomes). Features are train.feature_columns(table), stored ORDERED."""
    log = log or (lambda msg: None)
    feats = T.feature_columns(table)
    rows = training_rows(table, list(seasons))
    n_all = len(rows)
    played = rows[rows["played"].eq(True)]  # the gate's filter; astype(bool) would read NaN as played
    bad = [c for c in ["min", *STATS] if played[c].isna().any()]
    if bad:
        raise ValueError(f"played rows with no outcome in {bad}: a played game has a stat line")
    nulls = null_reference(rows, feats)
    clf = availability_model().fit(rows[feats], rows["played"].astype(int))
    del rows  # the full table is ~0.7 GB in memory; keep one filtered copy alive, not two
    log(f"availability: {n_all} rows, {clf.n_iter_} iterations")
    Xp = played[feats]
    mins = regressor("squared_error").fit(Xp, played["min"])
    log(f"minutes: {len(played)} rows, {mins.n_iter_} iterations")
    rates = {}
    for s in STATS:
        y, w = rate_target(played, s)
        rates[s] = regressor("poisson").fit(Xp, y, sample_weight=w)
        log(f"rate_{s}: {rates[s].n_iter_} iterations")
    ests = {"availability": clf, "minutes": mins, **{f"rate_{s}": rates[s] for s in STATS}}
    versions = runtime_versions()
    registry = {
        "model_version": f"{VERSION}-{today_et().isoformat()}",
        "trained_at": now_utc().isoformat(),
        "train_seasons": list(seasons),
        # props.is_in_training_window reads this: a season at or before it is
        # the model's own homework and stays out of the out-of-sample record.
        "trained_through": max(seasons, key=season_end_year),
        "season_type": "regular season (game_id 002)",
        "n_rows": {n: n_all if n == "availability" else len(played) for n in ests},
        "n_iter": {n: int(e.n_iter_) for n, e in ests.items()},
        "features": list(feats),
        "stats": list(STATS),
        "hyperparameters": {**HGB, "max_min": MAX_MIN, "rate_cap": RATE_CAP,
                            "min_exposure": MIN_EXPOSURE},
        "versions": versions,
        # serving_design.md section 2 names the versions at the top level; the
        # same values, and read_registry() refuses a registry where they differ.
        **versions,
        "null_reference": nulls,
        "files": {},
    }
    return Models(clf, mins, rates, registry)


# ── Persistence ──────────────────────────────────────────────────────────────
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(models: Models, dir: Path = MODELS_DIR) -> dict:
    """Write {name}_v1.joblib (compress=3) for every model, then the registry
    LAST, so a crash mid-save leaves a registry whose hashes do not match and
    load() refuses it. Each sha256 is of the exact bytes written. Returns the
    registry (also set on `models`)."""
    dir = Path(dir)
    files = {}
    for name, est in models.estimators().items():
        buf = io.BytesIO()
        joblib.dump(est, buf, compress=3)
        data = buf.getvalue()
        C._atomic_write_bytes(dir / file_name(name), data)
        files[file_name(name)] = _sha256(data)
    models.registry = {**models.registry, "files": files}
    C._atomic_write_bytes(dir / REGISTRY_NAME,
                          (json.dumps(models.registry, indent=2) + "\n").encode())
    return models.registry


_REGISTRY_TYPES = {"model_version": str, "trained_through": str, "train_seasons": list,
                   "features": list, "stats": list, "versions": dict, "null_reference": dict,
                   "files": dict}


def _check_registry(reg: dict) -> None:
    """Raise (any exception; read_registry wraps it) unless `reg` has every
    key a consumer reads, with the right types and consistent copies."""
    for key, typ in _REGISTRY_TYPES.items():
        if not isinstance(reg[key], typ):
            raise TypeError(f"{key} is a {type(reg[key]).__name__}, not a {typ.__name__}")
    feats = reg["features"]
    if not feats or not all(isinstance(f, str) for f in feats) or len(set(feats)) != len(feats):
        raise ValueError("features must be distinct column names")
    if not isinstance(reg["versions"]["sklearn"], str):
        raise TypeError("versions.sklearn is not a version string")
    split = [k for k, v in reg["versions"].items() if reg.get(k) != v]
    if split:
        raise ValueError(f"top-level {split} disagree with versions")
    if reg["trained_through"] != max(reg["train_seasons"], key=season_end_year):
        raise ValueError("trained_through is not the last train season")
    ref = reg["null_reference"]
    if set(ref["n"]) != set(ref["null_frac"]):
        raise ValueError("null_reference strata disagree between n and null_frac")
    off = [s for s, fr in ref["null_frac"].items() if set(fr) != set(feats)]
    if off:
        raise ValueError(f"null_reference strata {off} do not cover exactly the features")


def read_registry(dir: Path = MODELS_DIR) -> dict:
    """The registry, checked. Every way it can be unreadable (truncated JSON,
    a missing or mistyped key, another layout) raises ModelMismatch, the one
    error callers catch to fall back to the committed projections."""
    path = Path(dir) / REGISTRY_NAME
    if not path.exists():
        raise ModelMismatch(f"no model registry at {path}: run scripts/train_production.py")
    try:
        reg = json.loads(path.read_text())
        _check_registry(reg)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ModelMismatch(f"registry unreadable ({path}): {type(e).__name__}: {e}") from e
    return reg


def load(dir: Path = MODELS_DIR) -> Models:
    """The saved models, verified. Raises ModelMismatch when the registry is
    unreadable (read_registry), the runtime scikit-learn differs from the
    registry's (checked BEFORE unpickling), a file is missing, unlisted or not
    a served model, a file's sha256 differs, or a model was fit on other
    columns. The bytes that were hashed are the bytes unpickled, so nothing can
    change between check and load."""
    dir = Path(dir)
    reg = read_registry(dir)
    have, want = sklearn.__version__, reg["versions"]["sklearn"]
    if have != want:
        raise ModelMismatch(f"scikit-learn {have} at runtime but the models were trained with "
                            f"{want}: install the pinned version or retrain")
    names = model_names(reg["stats"])
    expected = [file_name(n) for n in names]
    files = reg.get("files") or {}
    problems = [f"{f}: listed in the registry but not a model this version serves"
                for f in files if f not in expected]
    blobs = {}
    for f in expected:
        path = dir / f
        if f not in files:
            problems.append(f"{f}: not in the registry")
        elif not path.exists():
            problems.append(f"{f}: missing")
        else:
            data = path.read_bytes()
            if _sha256(data) != files[f]:
                problems.append(f"{f}: sha256 differs from the registry")
            blobs[f] = data
    if problems:
        raise ModelMismatch(f"models in {dir} do not match {REGISTRY_NAME}: " + "; ".join(problems))
    est = {n: joblib.load(io.BytesIO(blobs[file_name(n)])) for n in names}
    off = [n for n, e in est.items() if list(getattr(e, "feature_names_in_", [])) != reg["features"]]
    if off:
        raise ModelMismatch(f"models fit on columns other than the registry features: {off}")
    return Models(est["availability"], est["minutes"],
                  {s: est[f"rate_{s}"] for s in reg["stats"]}, reg)


# ── Inference ────────────────────────────────────────────────────────────────
def feature_matrix(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    """rows[registry features], in registry order, as float64. Extra columns
    are ignored; an absent one raises MissingFeatures (never filled); a
    non-numeric one raises in astype rather than being coerced to NaN."""
    feats = models.features
    missing = [f for f in feats if f not in rows.columns]
    if missing:
        raise MissingFeatures(missing)
    dup = sorted(set(rows.columns[rows.columns.duplicated()]) & set(feats))
    if dup:
        raise ValueError(f"feature columns appear more than once in the rows: {dup}")
    return rows[feats].astype("float64")


def null_skew(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    """Every (stratum, feature) cell whose serve null fraction differs from
    the registry's training fraction by more than its tolerance, with
    stratum, feature, n, train, serve, tol. Empty means no skew. Rows with no
    stratum come back as one cell with feature "(stratum keys)". A
    STRUCTURAL_NAN feature is not compared by fraction: one cell (stratum
    "(all: NaN iff <key>)", n = rows checked, serve = the share of rows
    breaking the invariant, train 0) when any row breaks it."""
    cols = ["stratum", "feature", "n", "train", "serve", "tol"]
    X = feature_matrix(models, rows)
    st = null_strata(X)
    cells = []
    if st.isna().any():
        cells.append(("(none)", "(stratum keys)", int(st.isna().sum()), 0.0, 1.0, 0.0))
    for f, key in STRUCTURAL_NAN.items():
        if f not in X.columns or not len(X):
            continue
        flag = X[key] if key in X.columns else rows[key].astype("float64")
        off = X[f].isna() != flag.eq(1.0)
        if off.any():
            cells.append((f"(all: NaN iff {key})", f, len(X), 0.0, float(off.mean()), 0.0))
    ref = models.registry["null_reference"]
    serve = X.isna().groupby(st).mean()
    n_serve = st.value_counts()
    for s in serve.index:
        n, n_ref = int(n_serve[s]), int(ref["n"].get(s, 0))
        if n < NULL_MIN_ROWS or n_ref < NULL_MIN_ROWS:
            continue
        p = pd.Series(ref["null_frac"][s])[models.features]
        q = serve.loc[s, models.features]
        tol = np.maximum(NULL_TOL, NULL_Z * np.sqrt(p * (1 - p) * (1 / n + 1 / n_ref)))
        for f in q.index[(q - p).abs() > tol]:
            if f in STRUCTURAL_NAN:
                continue
            cells.append((s, f, n, float(p[f]), float(q[f]), float(tol[f])))
    return pd.DataFrame(cells, columns=cols)


def check_nulls(models: Models, rows: pd.DataFrame) -> None:
    """Raise NullSkew naming every skewed cell (spec lesson 5). Serving calls
    it on the rows it is about to predict; a raise is a train/serve skew to
    fix, never a column to exempt."""
    bad = null_skew(models, rows)
    if len(bad):
        raise NullSkew(f"{len(bad)} (stratum, feature) null fraction(s) differ from training: "
                       + "; ".join(_skew_text(r) for r in bad.itertuples()))


def _skew_text(r) -> str:
    """One null_skew cell, as check_nulls names it."""
    if r.feature in STRUCTURAL_NAN and str(r.stratum).startswith("(all"):
        return (f"{r.feature} is NaN exactly where {STRUCTURAL_NAN[r.feature]} == 1 in training, "
                f"broken on {round(r.serve * r.n)} of {r.n} serve rows")
    return (f"{r.stratum} {r.feature} serve {r.serve:.2f} vs train {r.train:.2f} "
            f"(n {r.n}, tol {r.tol:.2f})")


def predict(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    """Per row: p_play, mu_min (clipped 0..MAX_MIN), rate_<s>, and the
    E[stat | plays] means under plain stat names (mu_min x rate), plus the
    combos and Fantasy as sums of those means. Indexed like `rows`. It does
    not check null fractions; serving calls check_nulls() on the same rows.

    Every rate is capped at the rates it is a part of (RATE_CEILINGS, in that
    order): fg3a and fgm at fga, fg3m at fg3a and at fgm, ftm at fta. The
    means are built from the capped rates. The rate models are fit
    separately, and their raw rates broke these orders on real rows (v1
    models, 2025-26 at four horizons: ftm above fta on 590 rows of 73
    players, fg3m above fg3a on 86, fg3m above fgm on 14; fg3a above fga on
    21 rows of the training table), where a mean said a player makes more
    than he attempts. A capped row is AT the cap, not a forecast of it: at
    the fg3a cap the FG3A share is exactly 1 and the composed FG2A mean (FGA
    - FG3A) is exactly 0; at the ftm cap the FTM mean equals the FTA mean
    (a perfect line); at the fg3m caps every three goes in, or every make is
    a three (FG2M mean exactly 0). The simulator draws those rows
    degenerately and simprov refuses those draws (simprov._degenerate) and
    the baseline stats a cap pins at zero (simprov._baseline_degenerate).
    The baseline still prices FTM at the ftm cap, as a perfect line: the cap
    removes the impossible, not the rate model's optimism at the top of the
    FT% range (docs/known_issues.md)."""
    X = feature_matrix(models, rows)
    stats = models.stats
    if len(X) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in output_columns(stats)},
                            index=rows.index)
    out = {"p_play": models.availability.predict_proba(X)[:, 1],
           "mu_min": np.clip(models.minutes.predict(X), 0.0, MAX_MIN)}
    for s in stats:
        out[f"rate_{s}"] = models.rates[s].predict(X)
    for s, parts in RATE_CEILINGS:
        out[f"rate_{s}"] = np.minimum.reduce([out[f"rate_{s}"], *(out[f"rate_{p}"] for p in parts)])
    _compose(out, stats)
    return pd.DataFrame(out, index=rows.index)[output_columns(stats)]


def _compose(out, stats: list[str]) -> None:
    """The E[stat | plays] means from mu_min and the (capped) rates, then the
    combos and Fantasy as sums of those means, written into `out` (a dict of
    arrays or a frame). The one composition: predict() builds the means with
    it and cohere() rebuilds them after moving mu_min."""
    for s in stats:
        out[s] = out["mu_min"] * out[f"rate_{s}"]
    for c, parts in T.COMBOS.items():
        out[c] = sum(out[p] for p in parts)
    out["fantasy"] = sum(w * out[s] for s, w in T.FANTASY_W.items())


# ── Team minutes coherence ───────────────────────────────────────────────────
# A team-game's minutes add up: over its roster, the sum of p_play x E[MIN |
# plays] is E[team minutes], 240 plus 25 per overtime period. The models are
# fit row by row and miss that sum with an sd of 12.2 minutes (out of sample,
# 2023-24 and 2024-25), from 148 to 304; Indiana on 2026-11-02, ten players
# Out, summed to 188. cohere() moves every team-game part of the way back
# (docs/model_report.md, "Team minutes coherence").
TEAM_MINUTES = 241.5    # 240 + 25 x E[OT periods]: 241.56 over 2018-19..2022-23
COHERE_S0 = 30.0        # the blend's scale in minutes  } both fitted on 2023-24
COHERE_FULL = 20.0      # full shift up to this mu      } only (validate fold)
COHERE_CAP = 44.0       # no shift at all from here; never above max(this, mu)
COHERE_FLOOR = 1.0      # a shift never lowers E[MIN | plays] below min(this, mu)
COHERE_SPAN = 200.0     # the shift is searched in [-SPAN, SPAN] minutes
COHERE_ITERS = 64       # bisection steps: 400 / 2^64 minutes, exact in practice
COHERE_TAG = "tm1"      # the serving step's version, in Models.version
# Fewer rows than any training team-game had (pipeline.MIN_TEAM_ROWS, the same
# number, which tests pin together): a broken roster pull, whose sum says
# nothing about minutes. Such a team-game is left as predicted.
COHERE_MIN_ROWS = 12


def cohere(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """predict() output with each team-game's minutes moved toward coherence.

    `rows` are the rows predict() scored (same index), and must hold EVERY
    roster row of each team-game (game_id, team) in it: a partial roster has a
    different sum. A team-game with fewer than COHERE_MIN_ROWS rows is a
    broken roster pull (the bot flags it RED and still commits what it wrote)
    and is left exactly as predicted, rather than handing the missing
    players' minutes to the few rows present. Per team-game:
      S       sum of p_play x mu_min, the team minutes the rows imply
      sd      sqrt(sum p (1 - p) mu^2), how much the uncertainty about who
              plays spreads that sum
      lam     COHERE_S0^2 / (COHERE_S0^2 + sd^2)
      target  S + lam x (TEAM_MINUTES - S)
    Each row not listed Out (inj_out) moves by w x shift, with one shift per
    team-game solved so the p-weighted sum hits the target, and
      w = clip((CAP - mu) / (CAP - FULL), 0, 1)
    (the full shift up to FULL minutes, tapering to none at CAP), the result
    clipped to [min(mu, FLOOR), max(mu, CAP)]. A row listed Out (a visitor's
    mark-out included) keeps its mu_min: an absence does not absorb its own
    minutes. The stat means, combos and Fantasy are recomposed from the new
    mu_min and the unchanged rates (_compose); p_play is untouched.

    Why part of the way: a gap is partly availability (players with a low
    p_play who play anyway, or the reverse) and partly E[MIN | plays]. The
    share that is minutes falls as availability gets less certain (2023-24:
    0.92 at sd under 11, 0.38 at sd 27 and over; 2025-26, measured only, 0.96
    and 0.41), and lam follows it; with every player certain in or out, sd is
    0 and the whole gap is minutes. Moving the
    whole gap everywhere in proportion to mu made minutes worse than not
    moving at all (2023-24 RMSE +0.03%). Why equal and tapered: an equal shift
    beat a proportional one, and an untapered one overshot the heaviest
    players (projected over 34: residual +0.37, shift +0.56 on short teams),
    whose minutes give least; the taper took their RMSE from +0.15% to -0.17%.
    """
    if len(pred) == 0:
        return pred
    if not pred.index.equals(rows.index):
        raise ValueError("cohere: pred and rows must share one index (predict's output and its rows)")
    absent = [c for c in ("game_id", "team", "inj_out") if c not in rows.columns]
    if absent:
        raise KeyError(f"cohere needs {absent} on the rows (the team-game key and the Out flag)")
    grp = rows.groupby(["game_id", "team"], sort=False).ngroup().to_numpy()
    n = int(grp.max()) + 1
    p = pred["p_play"].to_numpy(dtype=float)
    mu = pred["mu_min"].to_numpy(dtype=float)
    move = rows["inj_out"].to_numpy(dtype=float) != 1.0

    def by_team(w):
        return np.bincount(grp, weights=w, minlength=n)

    S = by_team(p * mu)
    lam = COHERE_S0 ** 2 / (COHERE_S0 ** 2 + by_team(p * (1 - p) * mu ** 2))
    target = S + lam * (TEAM_MINUTES - S)
    lo, hi = np.minimum(mu, COHERE_FLOOR), np.maximum(mu, COHERE_CAP)
    # w is 0 on a row listed Out, so clip(mu + 0, lo, hi) keeps it exactly.
    w = np.where(move, np.clip((COHERE_CAP - mu) / (COHERE_CAP - COHERE_FULL), 0.0, 1.0), 0.0)
    a, b = np.full(n, -COHERE_SPAN), np.full(n, COHERE_SPAN)
    for _ in range(COHERE_ITERS):
        m = (a + b) / 2
        short = by_team(p * np.clip(mu + w * m[grp], lo, hi)) < target
        a, b = np.where(short, m, a), np.where(short, b, m)
    # Nothing that can move (every row Out or capped, or no weight), nothing
    # to do, or a broken roster pull: the rows stay exactly as predicted, not
    # moved by the bisection's last bit.
    idle = ((by_team(p * w) <= 0) | (np.abs(target - S) < 1e-9)
            | (by_team(np.ones(len(mu))) < COHERE_MIN_ROWS))
    shift = np.where(idle, 0.0, (a + b) / 2)
    out = pred.copy()
    out["mu_min"] = np.clip(mu + w * shift[grp], lo, hi)
    _compose(out, [c[len("rate_"):] for c in pred.columns if c.startswith("rate_")])
    return out
