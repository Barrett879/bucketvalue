"""Fit the simulator's parameters (bucketlib.dist.SimParams) on WALK-FORWARD
predictions, never in-sample ones (lesson: a calibration map fitted on in-sample
predictions learns "no error" and does nothing at serve time).

    python scripts/fit_dist.py [--fit-seasons 2020-21:2022-23]

For each fit season s: train the minutes model and the per-minute rate models
on 2018-19 .. s-1, predict s (played rows only). Pool those out-of-sample rows,
then fit:
  min_resid  actual - projected minutes, per projected-minutes bucket
             (subsampled to <= 4,000 per bucket; the empirical shape carries
             blowouts, foul trouble and overtime)
  beta       per stream, E[Y | M] = lambda * (M / mu) ** beta: a 1-D search
             minimising Poisson deviance with the projected rate as offset
  gamma_k    per stream, method of moments on the residual variance beyond
             Poisson given the realised minutes: Var = lam + lam^2 / k
Validate (2023-24) and test (2024-25) are untouched here; the probabilistic
gate evaluates the fitted simulator on them.

Writes models/sim_params_v1.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import dist as D  # noqa: E402
from bucketlib import train as T  # noqa: E402

TABLE = C.CACHE_DIR / "train_table_full_v1.parquet"
STREAMS = {"fga": "fga", "fta": "fta", "ast": "ast", "oreb": "oreb", "dreb": "dreb",
           "stl": "stl", "blk": "blk", "tov": "tov", "pf": "pf"}
OUT = C.MODELS_DIR / "sim_params_v1.json"


def hgb(loss):
    return HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, random_state=0)


def oos_predictions(df: pd.DataFrame, fit_seasons: list[str]) -> pd.DataFrame:
    feats = T.feature_columns(df)
    rows = []
    for s in fit_seasons:
        prior = T.seasons("2018-19", T.seasons("2018-19", s)[-2])
        tr = df[df["season"].isin(prior) & (df["played"] == True)]  # noqa: E712
        te = df[(df["season"] == s) & (df["played"] == True)].copy()  # noqa: E712
        te["mu_min"] = np.clip(hgb("squared_error").fit(tr[feats], tr["min"]).predict(te[feats]), 0.5, 53)
        w = tr["min"].clip(lower=1 / 60)
        for st in STREAMS.values():
            m = hgb("poisson").fit(tr[feats], (tr[st] / w).clip(upper=5), sample_weight=w)
            te[f"rate_{st}"] = m.predict(te[feats])
        print(f"  {s}: trained on {prior[0]}..{prior[-1]} ({len(tr)} rows), predicted {len(te)}")
        rows.append(te)
    return pd.concat(rows, ignore_index=True)


def fit_beta(y, lam_base, ratio, grid=np.linspace(0.3, 1.7, 29)):
    """lam = lam_base * ratio**beta; pick beta minimising Poisson deviance."""
    best, best_dev = 1.0, np.inf
    lr = np.log(np.clip(ratio, 1e-3, None))
    for b in grid:
        lam = np.clip(lam_base * np.exp(b * lr), 1e-6, None)
        dev = np.mean(lam - y * np.log(lam))
        if dev < best_dev:
            best, best_dev = float(b), dev
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-seasons", default="2020-21:2022-23")
    args = ap.parse_args()
    a, _, b = args.fit_seasons.partition(":")
    df = C.read_parquet(TABLE)
    df = df[df["game_id"].str.startswith("002")]
    oos = oos_predictions(df, T.seasons(a, b))
    params = D.SimParams()
    rng = np.random.default_rng(0)
    oos["bucket"] = oos["mu_min"].map(D.bucket_of)
    for bk, grp in oos.groupby("bucket"):
        r = (grp["min"] - grp["mu_min"]).to_numpy()
        if len(r) > 4000:
            r = rng.choice(r, 4000, replace=False)
        params.min_resid[str(bk)] = [round(float(x), 3) for x in r]
        print(f"bucket {bk} ({D.MIN_BUCKETS[bk]:.0f}-{D.MIN_BUCKETS[bk + 1]:.0f} min): n={len(grp)} "
              f"resid mean {np.mean(grp['min'] - grp['mu_min']):+.2f} sd {np.std(grp['min'] - grp['mu_min']):.2f}")
    ratio = (oos["min"] / oos["mu_min"]).to_numpy()
    for name, st in STREAMS.items():
        y = oos[st].to_numpy(dtype=float)
        if name == "fta":
            y = y / 2.0          # the simulator draws TRIPS
        base = oos[f"rate_{st}"].to_numpy() * oos["mu_min"].to_numpy() / (2.0 if name == "fta" else 1.0)
        beta = fit_beta(y, base, ratio)
        lam = base * ratio ** beta
        extra = ((y - lam) ** 2 - lam).sum()
        k = float((lam ** 2).sum() / extra) if extra > 0 else 1e6
        params.beta[name] = round(beta, 3)
        params.gamma_k[name] = round(min(k, 1e6), 2)
        print(f"{name:5s} beta {beta:.2f}  gamma_k {k:9.2f}")
    params.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
