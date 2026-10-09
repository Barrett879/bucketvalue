"""Train once on 2018-19..2022-23 and save OUT-OF-SAMPLE predictions for the
validate (2023-24) and test (2024-25) folds, so the gate, the simulator fit and
the probabilistic check all read the same numbers (2025-26 holdout untouched).

    python scripts/train_eval.py

Writes cache/preds_eval_v1.parquet: every roster row of the two folds with
  p_play                 availability model
  mu_min                 minutes | plays model
  rate_<stat>            per-minute rate models (exposure Poisson)
  pred_<stat>            mu_min x rate (E[stat | plays]); combos and Fantasy composed
  b2_<stat> / b2_min     shrunk baseline (same definition as validate_full.py)
plus the outcomes and the identity / strata columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import train as T  # noqa: E402
import validate_full as VF  # noqa: E402

OUT = C.CACHE_DIR / "preds_eval_v1.parquet"
KEEP = ["person_id", "game_id", "season", "game_date", "team", "opp", "horizon", "played",
        "min", "gp_s", "start_share_s", "n_prior_seasons", "tm_out_min", "games_since_return",
        "last_absence_len", "inj_out", "inj_questionable", "inj_unknown", "min_ewm5_s"]


def main() -> int:
    df = C.read_parquet(VF.TABLE)
    rs = df[df["game_id"].str.startswith("002")]
    train = rs[rs["season"].isin(T.seasons("2018-19", "2022-23"))]
    ev = rs[rs["season"].isin([VF.VALIDATE, VF.TEST])].copy()
    feats = T.feature_columns(df)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(train[feats], train["played"].astype(int))
    out = ev[KEEP + VF.STATS + VF.COMPOSED].copy()
    out["p_play"] = clf.predict_proba(ev[feats])[:, 1]
    tp = train[train["played"] == True]  # noqa: E712
    out["mu_min"] = np.clip(VF.hgb_reg("squared_error").fit(tp[feats], tp["min"]).predict(ev[feats]), 0, 53)
    w = tp["min"].clip(lower=1 / 60)
    league = {s: tp[s].sum() / tp["min"].sum() for s in VF.STATS}
    for s in VF.STATS:
        m = VF.hgb_reg("poisson").fit(tp[feats], (tp[s] / w).clip(upper=5), sample_weight=w)
        out[f"rate_{s}"] = m.predict(ev[feats])
        out[f"pred_{s}"] = out["mu_min"] * out[f"rate_{s}"]
        print("rate", s, flush=True)
    for c, parts in T.COMBOS.items():
        out[f"pred_{c}"] = sum(out[f"pred_{p}"] for p in parts)
    out["pred_fantasy"] = sum(wt * out[f"pred_{s}"] for s, wt in T.FANTASY_W.items())
    mins = ev["min_ewm5_s"].fillna(ev["min_std_s"]).fillna(ev["p1_mpg"]).fillna(tp["min"].mean())
    out["b2_min"] = mins
    for s in VF.STATS:
        out[f"b2_{s}"] = mins * VF.shrunk_rate(ev, s, league[s])
    for c, parts in T.COMBOS.items():
        out[f"b2_{c}"] = sum(out[f"b2_{p}"] for p in parts)
    out["b2_fantasy"] = sum(wt * out[f"b2_{s}"] for s, wt in T.FANTASY_W.items())
    C.write_parquet(out, OUT)
    print(f"wrote {OUT.name} {out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
