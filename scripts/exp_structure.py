"""Which model STRUCTURE wins? (spec 3.3, the gate decides)

    python scripts/exp_structure.py [--targets pts,reb,...]

Candidates, all on the same walk-forward folds and paired rows as
validate_models.py (train 2015-16..2022-23, validate 2023-24, test 2024-25):
  direct      Poisson HistGB per stat on all features (validate_models' model)
  mxr_base    model minutes x baseline rate (EWMA20 -> season -> last season)
  mxr_model   model minutes x rate model (Poisson HistGB on stat/min with
              sample_weight = minutes: an exposure model, cameo rows kept)
Reported as gain over baseline 2 (EWMA minutes x EWMA rate), which the direct
models mostly failed to beat on rate-dominated stats (v0, 2026-10-08).
The rate model and the minutes model never see the target game's minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import train as T  # noqa: E402
import validate_models as V  # noqa: E402

TABLE = C.CACHE_DIR / "train_table_logs_v0.parquet"
COMPONENTS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta", "fgm", "fga",
              "fg3a", "oreb", "dreb", "pf"]


def hgb(loss):
    return HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, random_state=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="pts,reb,ast,fg3m,stl,blk,tov,ftm,pra,fantasy,stocks")
    args = ap.parse_args()
    if TABLE.exists():
        df = pd.read_parquet(TABLE)
    else:
        df = V.build_table("logs_v0")
        C.write_parquet(df, TABLE)
    rs = df[df["season_type"] == "RS"]
    train = rs[rs["season"].isin(T.seasons(V.TRAIN_FIRST, V.TRAIN_LAST))]
    folds = {"validate": rs[rs["season"] == V.VALIDATE], "test": rs[rs["season"] == V.TEST]}
    feats = T.feature_columns(df, exclude=V.LOGS_ONLY_EXCLUDE)

    t0 = time.time()
    mm = hgb("squared_error").fit(train[feats], train["min"])
    pmin = {k: np.clip(mm.predict(v[feats]), 0, 48) for k, v in folds.items()}
    print(f"minutes model {time.time() - t0:.0f}s")

    # Rate models: exposure-weighted Poisson on stat per minute.
    rate_pred = {}
    w = train["min"].clip(lower=1 / 60)
    for s in COMPONENTS:
        t0 = time.time()
        m = hgb("poisson").fit(train[feats], (train[s] / w).clip(upper=5), sample_weight=w)
        rate_pred[s] = {k: m.predict(v[feats]) for k, v in folds.items()}
        print(f"rate model {s} {time.time() - t0:.0f}s")

    def compose(target, fold, rates):
        if target in T.COMBOS:
            return sum(pmin[fold] * rates(c, fold) for c in T.COMBOS[target])
        if target == "fantasy":
            return sum(wt * pmin[fold] * rates(c, fold) for c, wt in T.FANTASY_W.items())
        return pmin[fold] * rates(target, fold)

    def base_rate(s, fold):
        d = folds[fold]
        return d[f"r_{s}_ewm20"].fillna(d[f"r_{s}_std"]).fillna(d[f"p1_rate_{s}"]).fillna(
            train[s].sum() / train["min"].sum()).to_numpy()

    def model_rate(s, fold):
        return rate_pred[s][fold]

    rows = []
    for target in args.targets.split(","):
        direct = V.fit_predict(train, folds, target, feats)[0]
        league = train[target].mean()
        for fold, d in folds.items():
            y = d[target].to_numpy(dtype=float)
            b1 = T.baseline_std(d, target).fillna(league).to_numpy(dtype=float)
            b2 = T.baseline_rate(d, target).fillna(pd.Series(b1, index=d.index)).to_numpy(dtype=float)
            cands = {"direct": direct[fold], "mxr_base": compose(target, fold, base_rate),
                     "mxr_model": compose(target, fold, model_rate)}
            sq = target in V.SQUARED
            eb = (y - b2) ** 2 if sq else np.abs(y - b2)
            rec = {"target": target, "fold": fold}
            for name, p in cands.items():
                em = (y - p) ** 2 if sq else np.abs(y - p)
                lo, hi = V.cluster_ci(em, eb, d["person_id"].to_numpy())
                rec[name] = 1 - em.sum() / eb.sum()
                rec[f"{name}_lo"] = lo
            rows.append(rec)
        print("  ", target, "  ".join(f"{r['fold']}: " + " ".join(f"{k} {r[k]:+.2%}" for k in ("direct", "mxr_base", "mxr_model")) for r in rows[-2:]))
    rep = pd.DataFrame(rows)
    rep.to_csv(C.REPO_ROOT / "docs" / "exp_structure_logs_v0.csv", index=False)
    print(rep.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
