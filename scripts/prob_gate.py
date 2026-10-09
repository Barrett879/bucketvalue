"""The probabilistic gate (spec 3.4c): are the over/under probabilities honest?

    python scripts/prob_gate.py [--n 6000]

On out-of-sample predictions (cache/preds_eval_v1.parquet, folds 2023-24 and
2024-25, played rows), for a deterministic sample of player-games:
  - simulator pmfs (bucketlib.dist with models/sim_params_v1.json), re-centred
    on the model's mean;
  - a NEGATIVE BINOMIAL with the same mean and a per-stat dispersion (the
    simpler candidate the simulator must beat).
Proxy lines are built from the BASELINE projection (never from the model): a
half-point line floor(b)+0.5 and a whole line round(b), because PrizePicks posts
both and ties are their own bucket.
Scores per stat and family:
  ECE   calibration of P(over | not a tie) in 10 bins (decided rows only)
  hit   side-picking hit rate (argmax of P(over), P(under)) vs base = the more
        common side's share (a model that always takes it scores this for free)
  RPS   ranked probability score of the whole pmf (lower is better)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import dist as D  # noqa: E402
from bucketlib.pmf import Pmf  # noqa: E402
import validate_full as VF  # noqa: E402

PREDS = C.CACHE_DIR / "preds_eval_v1.parquet"
BOARD = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra", "pr", "pa", "ra", "stocks", "fantasy"]
NB_DISP = {"pts": 2.2, "reb": 1.1, "ast": 1.1, "fg3m": 1.1, "stl": 1.05, "blk": 1.1, "tov": 1.0,
           "pra": 1.9, "pr": 2.0, "pa": 2.0, "ra": 1.1, "stocks": 1.1}


def player_input(r) -> D.PlayerInput:
    rate = {s: max(float(r[f"rate_{s}"]), 1e-6) for s in ("fga", "fta", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf")}
    fga, fg3a, fgm, fg3m = (max(float(r[f"rate_{s}"]), 1e-9) for s in ("fga", "fg3a", "fgm", "fg3m"))
    fta, ftm = max(float(r["rate_fta"]), 1e-9), max(float(r["rate_ftm"]), 0.0)
    share3 = min(fg3a / fga, 1.0)
    p3 = min(fg3m / fg3a, 1.0)
    p2 = min(max(fgm - fg3m, 0.0) / max(fga - fg3a, 1e-9), 1.0)
    shipped = {s: float(r[f"pred_{s}"]) for s in BOARD}
    return D.PlayerInput(mu_min=float(r["mu_min"]), rate=rate, fg3a_share=share3, p3=p3, p2=p2,
                         pft=min(ftm / fta, 1.0), shipped_mean=shipped)


RPS_GRID = {s: np.arange(0, 121, 1.0) for s in BOARD}
RPS_GRID["fantasy"] = np.arange(-10, 151, 1.0)


def rps(pmf: Pmf, y: float, stat: str) -> float:
    """Ranked probability score on a FIXED grid per stat, so two distributions
    with different supports are scored on the same points. The outcome's CDF
    is the step 1{grid >= y} (an earlier version had it backwards)."""
    g = RPS_GRID[stat]
    cdf = np.array([pmf.cdf(x) for x in g])
    return float(np.sum((cdf - (g >= y)) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="player-games per fold")
    args = ap.parse_args()
    params = D.SimParams.load(C.MODELS_DIR / "sim_params_v1.json")
    df = C.read_parquet(PREDS)
    df = df[(df["played"] == True) & (df["mu_min"] > 3)]  # noqa: E712
    recs = []
    for fold in (VF.VALIDATE, VF.TEST):
        d = df[df["season"] == fold]
        d = d.iloc[np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)]
        for i, (_, r) in enumerate(d.iterrows()):
            sim = D.pmfs(player_input(r), params, seed=i, stats=BOARD, n=2000)
            for s in BOARD:
                y = float(r[s])
                b = float(r[f"b2_{s}"])
                if not np.isfinite(b):
                    continue
                if s == "fantasy":
                    nb = None
                else:
                    nb = Pmf.from_negbin(max(float(r[f"pred_{s}"]), 1e-3), NB_DISP[s])
                for kind, line in (("half", np.floor(b) + 0.5), ("whole", float(np.round(b)))):
                    if s == "fantasy" and kind == "whole":
                        continue
                    for name, p in (("sim", sim[s]), ("nb", nb)):
                        if p is None:
                            continue
                        lt, eq, gt = p.split(line)
                        q_over = gt / (gt + lt) if gt + lt > 0 else 0.5
                        recs.append({"fold": fold, "stat": s, "kind": kind, "dist": name,
                                     "q_over": q_over, "p_eq": eq, "y": y, "line": line,
                                     "rps": rps(p, y, s) if kind == "half" else np.nan})
            if i % 1000 == 0:
                print(f"  {fold}: {i}/{len(d)}", flush=True)
    R = pd.DataFrame(recs)
    R["over"] = (R["y"] > R["line"]).astype(float)
    R["tie"] = R["y"] == R["line"]
    R.to_parquet(C.CACHE_DIR / "prob_gate_rows_v1.parquet", index=False)
    dec = R[~R["tie"]]
    out = []
    for (fold, stat, kind, name), g in dec.groupby(["fold", "stat", "kind", "dist"]):
        bins = np.clip((g["q_over"] * 10).astype(int), 0, 9)
        cal = g.groupby(bins).agg(p=("q_over", "mean"), o=("over", "mean"), n=("over", "size"))
        ece = float((cal["n"] * (cal["p"] - cal["o"]).abs()).sum() / cal["n"].sum())
        lean_over = g["q_over"] > 0.5
        hit = float(np.where(lean_over, g["over"], 1 - g["over"]).mean())
        base = float(max(g["over"].mean(), 1 - g["over"].mean()))
        out.append({"fold": fold, "stat": stat, "kind": kind, "dist": name, "n": len(g),
                    "ece": ece, "hit": hit, "base": base, "edge_pts": (hit - base) * 100,
                    "rps": float(R[(R.fold == fold) & (R.stat == stat) & (R.kind == "half") & (R.dist == name)]["rps"].mean())})
    T = pd.DataFrame(out)
    T.to_csv(C.REPO_ROOT / "docs" / "prob_gate_v1.csv", index=False)
    show = T.pivot_table(index=["stat", "kind"], columns=["dist", "fold"], values=["ece", "edge_pts"]).round(3)
    print(show.to_string())
    print("\nRPS (half lines, lower is better):")
    print(T[T.kind == "half"].pivot_table(index="stat", columns=["dist", "fold"], values="rps").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
