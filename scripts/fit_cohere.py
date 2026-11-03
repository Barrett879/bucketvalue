"""Fit and validate models.cohere, the team-minutes step (docs/model_report.md,
"Team minutes coherence"), on the saved out-of-sample predictions:

  cache/preds_eval_v1.parquet     2023-24 (validate) and 2024-25 (test), from
                                  models fit on 2018-19..2022-23 (train_eval.py)
  cache/preds_holdout_v1.parquet  2025-26, the spent holdout: MEASURED ONLY

    python scripts/fit_cohere.py [--grid]

--grid also prints a grid of the two scales (COHERE_S0, COHERE_FULL) on the
validate fold alone, by minutes RMSE on played rows. It changes nothing: the
shipped constants live in bucketlib/models.py, and they are what every fold is
scored with and what is written. Each score runs the PRODUCTION step
(models.cohere; a grid point swaps the two module constants for that call
only). The summary goes to docs/cohere_tm1.csv, and a cohered copy of each
preds file (mu_min and pred_<s> moved; p_play, rates and outcomes untouched)
to cache/preds_{eval,holdout}_tm1.parquet, which scripts/prob_gate.py takes as
--preds.

REFIT AFTER A RETRAIN. The preds files are scripts/train_eval.py's OWN
walk-forward predictions (models fit on 2018-19..2022-23 for these folds), not
the production models', so this script cannot see a retrain. Regenerate them
first (train_eval.py with the new --train-last and --eval), run --grid, then
edit COHERE_S0 / COHERE_FULL in bucketlib/models.py by hand, update the values
tests/test_models.py pins (test_the_shipped_scales_are_the_validated_ones and
the real-game numbers), rerun this script without --grid, and write the new
numbers into docs/model_report.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bucketlib import cache as C  # noqa: E402
from bucketlib import models as M  # noqa: E402

FOLDS = {"validate": ("preds_eval_v1.parquet", "2023-24"), "test": ("preds_eval_v1.parquet", "2024-25"),
         "holdout": ("preds_holdout_v1.parquet", "2025-26")}
SCORED = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra", "pr", "pa", "ra", "stocks", "fantasy"]
OUT = Path(__file__).resolve().parent.parent / "docs" / "cohere_tm1.csv"


def cohered(df: pd.DataFrame, s0: float | None = None, full: float | None = None) -> pd.DataFrame:
    """models.cohere on an eval frame: its means rebuilt as predict() returns
    them (they must equal the saved pred_<s>), cohered, mapped back. s0 and
    full default to the shipped constants, read when called."""
    pv = pd.DataFrame({"p_play": df["p_play"], "mu_min": df["mu_min"],
                       **{f"rate_{s}": df[f"rate_{s}"] for s in M.STATS}}, index=df.index)
    M._compose(pv, M.STATS)
    drift = (pv[SCORED] - df[[f"pred_{s}" for s in SCORED]].to_numpy()).abs().max().max()
    if drift > 1e-6:
        raise ValueError(f"the saved pred_ columns are not mu_min x rate (off by {drift}): stale preds file")
    keep = (M.COHERE_S0, M.COHERE_FULL)
    M.COHERE_S0 = keep[0] if s0 is None else s0
    M.COHERE_FULL = keep[1] if full is None else full
    try:
        co = M.cohere(pv, df[["game_id", "team", "inj_out"]])
    finally:
        M.COHERE_S0, M.COHERE_FULL = keep
    out = df.copy()
    out["mu_min"] = co["mu_min"]
    for s in [*M.STATS, *M.COMPOSED]:
        if f"pred_{s}" in out:
            out[f"pred_{s}"] = co[s]
    return out


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def score(df: pd.DataFrame, co: pd.DataFrame) -> dict:
    pl = df["played"].eq(True).to_numpy()
    y = df["min"].to_numpy()[pl]
    m0, m1 = df["mu_min"].to_numpy()[pl], co["mu_min"].to_numpy()[pl]
    grp = df.groupby(["game_id", "team"], sort=False).ngroup()
    team = lambda f: (f["p_play"] * f["mu_min"]).groupby(grp).sum() - M.TEAM_MINUTES  # noqa: E731
    r = {"played_rows": int(pl.sum()), "team_games": int(grp.nunique()),
         "team_gap_sd_before": float(team(df).std()), "team_gap_sd_after": float(team(co).std()),
         "min_rmse_before": rmse(y - m0), "min_rmse_after": rmse(y - m1)}
    r["min_rmse_change_pct"] = (r["min_rmse_after"] / r["min_rmse_before"] - 1) * 100
    heavy = m0 > 34
    r["min_rmse_change_pct_over34"] = (rmse((y - m1)[heavy]) / rmse((y - m0)[heavy]) - 1) * 100
    for s in SCORED:
        a = df[s].to_numpy()[pl]
        r[f"{s}_rmse_change_pct"] = (rmse(a - co[f"pred_{s}"].to_numpy()[pl])
                                     / rmse(a - df[f"pred_{s}"].to_numpy()[pl]) - 1) * 100
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="refit COHERE_S0 and COHERE_FULL on 2023-24")
    args = ap.parse_args()
    frames = {f: C.read_parquet(C.CACHE_DIR / f) for f in {v[0] for v in FOLDS.values()}}
    if args.grid:
        val = frames[FOLDS["validate"][0]]
        val = val[val["season"] == FOLDS["validate"][1]].reset_index(drop=True)
        grid = []
        for s0 in (20.0, 24.0, 30.0, 40.0, 60.0):
            for full in (0.0, 12.0, 16.0, 20.0, 24.0, 28.0, 34.0, 44.0 - 1e-9):
                grid.append({"s0": s0, "full": round(full, 1),
                             **score(val, cohered(val, s0, full))})
        g = pd.DataFrame(grid).sort_values("min_rmse_change_pct")
        print("validate 2023-24, best first (full 44 = no taper):")
        print(g[["s0", "full", "min_rmse_change_pct", "min_rmse_change_pct_over34", "pra_rmse_change_pct",
                 "fantasy_rmse_change_pct"]].head(12).round(3).to_string(index=False))
    rows = []
    for fold, (src, season) in FOLDS.items():
        df = frames[src]
        df = df[df["season"] == season].reset_index(drop=True)
        co = cohered(df)
        rows.append({"fold": fold, "season": season, "s0": M.COHERE_S0, "full": M.COHERE_FULL,
                     "measured_only": fold == "holdout", **score(df, co)})
    for src, df in frames.items():
        C.write_parquet(cohered(df), C.CACHE_DIR / src.replace("_v1", "_tm1"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False, float_format="%.4f")
    show = ["fold", "team_gap_sd_before", "team_gap_sd_after", "min_rmse_change_pct", "min_rmse_change_pct_over34",
            "pts_rmse_change_pct", "pra_rmse_change_pct", "fantasy_rmse_change_pct"]
    print(res[show].round(3).to_string(index=False))
    print(f"wrote {OUT} and cache/preds_{{eval,holdout}}_tm1.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
