"""The ship gate (spec 3.4) on saved out-of-sample predictions.

    python scripts/gate.py

For each target and fold: relative improvement over the shrunk baseline with a
95% PLAYER-CLUSTER bootstrap interval, pooled and per stratum. Primary metric:
RMSE for minutes and Fantasy Score; Poisson deviance for counts (MAE, reported
too, is minimised by the median and penalises an estimate of the MEAN on low
counts: the v0 lesson). PASS = pooled interval above zero on BOTH folds AND no
stratum whose interval lies entirely below zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
import validate_full as VF  # noqa: E402
import validate_models as V  # noqa: E402

PREDS = C.CACHE_DIR / "preds_eval_v1.parquet"
SQUARED = {"min", "fantasy"}


def strata(d: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(d), bool),
        "starters": (d["start_share_s"] >= 0.5).to_numpy(),
        "bench": (d["start_share_s"] < 0.5).to_numpy(),
        "cold<10g": (d["gp_s"] < 10).to_numpy(),
        "rookie": (d["n_prior_seasons"].fillna(0) == 0).to_numpy(),
        "teammate_out": (d["tm_out_min"] > 0).to_numpy(),
        "return<=2g": ((d["last_absence_len"] >= 3) & (d["games_since_return"] <= 2)).to_numpy(),
        "status_unknown": (d["inj_unknown"] == 1).to_numpy(),
    }


def errors(target, y, p):
    if target in SQUARED:
        return (y - p) ** 2
    return V.poisson_dev(y, np.clip(p, 1e-6, None))


def main() -> int:
    df = C.read_parquet(PREDS)
    pl = df[df["played"] == True]  # noqa: E712
    rows = []
    targets = ["min"] + VF.STATS + VF.COMPOSED
    for t in targets:
        for fold in (VF.VALIDATE, VF.TEST):
            d = pl[pl["season"] == fold]
            y = d["min" if t == "min" else t].to_numpy(dtype=float)
            p = d["mu_min" if t == "min" else f"pred_{t}"].to_numpy(dtype=float)
            b = d[f"b2_{t}"].to_numpy(dtype=float)
            em, eb = errors(t, y, p), errors(t, y, b)
            mae_gain = 1 - np.abs(y - p).sum() / np.abs(y - b).sum()
            for sname, m in strata(d).items():
                if m.sum() < 300:
                    continue
                lo, hi = V.cluster_ci(em[m], eb[m], d["person_id"].to_numpy()[m], reps=300)
                rows.append({"target": t, "fold": fold, "stratum": sname, "n": int(m.sum()),
                             "gain": 1 - em[m].sum() / eb[m].sum(), "lo": lo, "hi": hi,
                             "mae_gain": mae_gain if sname == "all" else np.nan})
    rep = pd.DataFrame(rows)
    rep.to_csv(C.REPO_ROOT / "docs" / "gate_v1.csv", index=False)
    allr = rep[rep.stratum == "all"]
    pooled = allr.groupby("target")["lo"].apply(lambda s: bool((s > 0).all()))
    worst = rep[rep.stratum != "all"].groupby("target")["hi"].min()
    verdict = pd.DataFrame({"pooled_ci_above_0_both_folds": pooled,
                            "worst_stratum_hi": worst,
                            "no_stratum_significantly_worse": worst > 0})
    verdict["PASS"] = verdict["pooled_ci_above_0_both_folds"] & verdict["no_stratum_significantly_worse"]
    g = allr.pivot_table(index="target", columns="fold", values=["gain", "lo", "mae_gain"]).round(4)
    print(g.to_string())
    print()
    print(verdict.round(4).to_string())
    losers = rep[(rep.stratum != "all") & (rep.hi < 0)]
    if len(losers):
        print("\nstrata significantly WORSE than the baseline:")
        print(losers.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
