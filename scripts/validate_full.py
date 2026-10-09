"""Walk-forward validation on the FULL training table (participation + injury
horizon + lines): availability, minutes | plays, and per-minute rates.

    python scripts/validate_full.py [--train-first 2018-19]

Folds: train train_first..2022-23, VALIDATE 2023-24, TEST 2024-25; 2025-26 is
the holdout and is not touched.

Reported PER HORIZON (d1_1730 / d0_1100 / t_120 / t_15), because the product
serves hours before tip; the spec gates at the morning horizon and reports the
tip horizon separately.

  availability  P(plays at least one second) over every rostered row.
                Log loss and Brier vs (a) a status-lookup baseline (training
                frequency of playing by status/state) and (b) the player's
                recent played rate, shrunk toward the league rate.
  minutes       E[MIN | plays], RMSE vs b2 (EWMA5 -> season -> last season).
  stats         E[stat | plays] = model minutes x per-minute rate model
                (Poisson with exposure), and the direct model, vs a SHRUNK b2
                (EWMA rate pulled toward the league rate by exposure), on MAE
                and Poisson deviance. Shrinking matters: an unshrunk baseline
                can be exactly 0 and deviance punishes that without bound,
                which inflated v0's deviance margins.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import livectx as L  # noqa: E402
from bucketlib import train as T  # noqa: E402
import validate_models as V  # noqa: E402

TABLE = C.CACHE_DIR / "train_table_full_v1.parquet"
VALIDATE, TEST = "2023-24", "2024-25"
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta", "fgm", "fga", "fg3a",
         "oreb", "dreb", "pf"]
COMPOSED = ["pra", "pr", "pa", "ra", "stocks", "fantasy"]
SHRINK_MIN = 150.0   # minutes of league-average production in the shrunk baseline


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def hgb_reg(loss):
    return HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, random_state=0)


def shrunk_rate(df, s, league_rate):
    """EWMA20 rate pulled toward the league rate with SHRINK_MIN minutes of
    ballast, weighted by the minutes behind the EWMA (season-to-date minutes as
    the exposure proxy, last season when none)."""
    r = df[f"r_{s}_ewm20"].fillna(df[f"r_{s}_std"]).fillna(df[f"p1_rate_{s}"])
    expo = (df["min_std_s"] * df["gp_s"]).fillna(0) + (df["p1_mpg"] * df["p1_gp"]).fillna(0) * 0.5
    return ((r.fillna(league_rate) * expo + league_rate * SHRINK_MIN) / (expo + SHRINK_MIN))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-first", default="2018-19")
    args = ap.parse_args()
    df = C.read_parquet(TABLE)
    rs = df[df["game_id"].str.startswith("002")]
    train_seasons = T.seasons(args.train_first, "2022-23")
    train = rs[rs["season"].isin(train_seasons)]
    folds = {"validate": rs[rs["season"] == VALIDATE], "test": rs[rs["season"] == TEST]}
    feats = T.feature_columns(df) + [c for c in L.CTX_COLS if c in df.columns]
    print(f"{len(feats)} features; train rows {len(train)}")
    rows = []

    # ── Availability ─────────────────────────────────────────────────────────
    t0 = time.time()
    y_tr = train["played"].astype(int)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(train[feats], y_tr)
    status_cols = [c for c in L.CTX_COLS if c.startswith("inj_") and not c.startswith("inj_reason")]
    key = train[status_cols].idxmax(axis=1).where(train[status_cols].max(axis=1) > 0, "none")
    lookup = y_tr.groupby(key).mean()
    league_play = y_tr.mean()
    print(f"availability model {time.time() - t0:.0f}s")
    for fold, d in folds.items():
        y = d["played"].astype(int).to_numpy()
        p_m = clf.predict_proba(d[feats])[:, 1]
        k = d[status_cols].idxmax(axis=1).where(d[status_cols].max(axis=1) > 0, "none")
        p_look = k.map(lookup).fillna(league_play).to_numpy()
        n10 = d["events_s"].clip(upper=10).fillna(0)
        p_rate = ((d["played_rate_l10"].fillna(league_play) * n10 + league_play * 3) / (n10 + 3)).to_numpy()
        for h, idx in d.groupby("horizon").indices.items():
            rows.append({"model": "availability", "fold": fold, "horizon": h, "n": len(idx),
                         "logloss_model": logloss(y[idx], p_m[idx]),
                         "logloss_status_lookup": logloss(y[idx], p_look[idx]),
                         "logloss_recent_rate": logloss(y[idx], p_rate[idx]),
                         "brier_model": brier(y[idx], p_m[idx]),
                         "brier_status_lookup": brier(y[idx], p_look[idx]),
                         "sum_p_vs_played": float(p_m[idx].sum() / max(y[idx].sum(), 1))})

    # ── Minutes | plays and rates ────────────────────────────────────────────
    tp = train[train["played"] == True]  # noqa: E712
    fp = {k: v[v["played"] == True] for k, v in folds.items()}  # noqa: E712
    t0 = time.time()
    mm = hgb_reg("squared_error").fit(tp[feats], tp["min"])
    pmin = {k: np.clip(mm.predict(v[feats]), 0, 53) for k, v in fp.items()}
    print(f"minutes model {time.time() - t0:.0f}s")
    w = tp["min"].clip(lower=1 / 60)
    rate = {}
    for s in STATS:
        m = hgb_reg("poisson").fit(tp[feats], (tp[s] / w).clip(upper=5), sample_weight=w)
        rate[s] = {k: m.predict(v[feats]) for k, v in fp.items()}
    league = {s: tp[s].sum() / tp["min"].sum() for s in STATS}
    print(f"rate models {time.time() - t0:.0f}s")

    def compose(target, fold):
        if target in T.COMBOS:
            return sum(pmin[fold] * rate[c][fold] for c in T.COMBOS[target])
        if target == "fantasy":
            return sum(wt * pmin[fold] * rate[c][fold] for c, wt in T.FANTASY_W.items())
        return pmin[fold] * rate[target][fold]

    def base(target, d):
        mins = d["min_ewm5_s"].fillna(d["min_std_s"]).fillna(d["p1_mpg"]).fillna(tp["min"].mean())
        if target in T.COMBOS:
            return sum(mins * shrunk_rate(d, c, league[c]) for c in T.COMBOS[target])
        if target == "fantasy":
            return sum(wt * mins * shrunk_rate(d, c, league[c]) for c, wt in T.FANTASY_W.items())
        return mins * shrunk_rate(d, target, league[target])

    for fold, d in fp.items():
        y_min = d["min"].to_numpy()
        b_min = d["min_ewm5_s"].fillna(d["min_std_s"]).fillna(d["p1_mpg"]).fillna(tp["min"].mean()).to_numpy()
        for h, idx in d.groupby("horizon").indices.items():
            rows.append({"model": "min", "fold": fold, "horizon": h, "n": len(idx),
                         "rmse_model": float(np.sqrt(np.mean((y_min[idx] - pmin[fold][idx]) ** 2))),
                         "rmse_b2": float(np.sqrt(np.mean((y_min[idx] - b_min[idx]) ** 2)))})
        for target in STATS + COMPOSED:
            y = d[target].to_numpy(dtype=float)
            p = compose(target, fold)
            b = base(target, d).to_numpy(dtype=float)
            for h, idx in d.groupby("horizon").indices.items():
                rec = {"model": target, "fold": fold, "horizon": h, "n": len(idx),
                       "mae_model": float(np.mean(np.abs(y[idx] - p[idx]))),
                       "mae_b2s": float(np.mean(np.abs(y[idx] - b[idx])))}
                if target != "fantasy":
                    rec["dev_model"] = float(np.mean(V.poisson_dev(y[idx], p[idx])))
                    rec["dev_b2s"] = float(np.mean(V.poisson_dev(y[idx], b[idx])))
                else:
                    rec["rmse_model"] = float(np.sqrt(np.mean((y[idx] - p[idx]) ** 2)))
                    rec["rmse_b2s"] = float(np.sqrt(np.mean((y[idx] - b[idx]) ** 2)))
                rows.append(rec)
    rep = pd.DataFrame(rows)
    out = C.REPO_ROOT / "docs" / "model_report_full_v1.csv"
    rep.to_csv(out, index=False)
    print(f"wrote {out}")
    a = rep[rep.model == "availability"]
    print(a[["fold", "horizon", "n", "logloss_model", "logloss_status_lookup", "logloss_recent_rate",
             "brier_model", "brier_status_lookup", "sum_p_vs_played"]].round(4).to_string(index=False))
    m = rep[rep.model == "min"].assign(gain=lambda x: 1 - x.rmse_model / x.rmse_b2)
    print(m[["fold", "horizon", "n", "rmse_model", "rmse_b2", "gain"]].round(4).to_string(index=False))
    s = rep[~rep.model.isin(["availability", "min"])].copy()
    s["dev_gain"] = 1 - s["dev_model"] / s["dev_b2s"]
    s["mae_gain"] = 1 - s["mae_model"] / s["mae_b2s"]
    print(s.pivot_table(index="model", columns=["fold"], values=["dev_gain", "mae_gain"], aggfunc="mean").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
