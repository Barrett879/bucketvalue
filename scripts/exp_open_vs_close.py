"""How much does training/scoring on CLOSING lines flatter the models?

Closing lines already price late scratches; at a 5:30pm-the-day-before horizon
the site sees an earlier line. Same trained models, test folds scored twice:
with the closing line (as trained) and with the OPENING line swapped into every
line feature (open exists for 2023-24+ = both folds). The gap is the optimism
attributable to line timing. Also reports a no-lines model for reference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bucketlib import cache as C  # noqa: E402
from bucketlib import train as T  # noqa: E402

df = C.read_parquet(C.CACHE_DIR / "train_table_full_v1.parquet")
df = df[df.game_id.str.startswith("002") & (df.played == True)]  # noqa: E712
lines = T.load_table("game_lines", T.seasons("2023-24", "2024-25"))[["game_id", "open_home_line", "open_total"]]
feats = T.feature_columns(df)
LINE = ["home_line", "total", "team_line", "implied_total", "abs_line"]
tr = df[df.season.isin(T.seasons("2018-19", "2022-23"))]


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=0)


m_close = hgb().fit(tr[feats], tr["min"])
nol = [f for f in feats if f not in LINE]
m_nol = hgb().fit(tr[nol], tr["min"])
for fold in ("2023-24", "2024-25"):
    te = df[df.season == fold].merge(lines, on="game_id", how="left")
    op = te.copy()
    op["home_line"] = op["open_home_line"].fillna(op["home_line"])
    op["total"] = op["open_total"].fillna(op["total"])
    op["team_line"] = np.where(op["home"].astype(bool), op["home_line"], -op["home_line"])
    op["implied_total"] = op["total"] / 2 - op["team_line"] / 2
    op["abs_line"] = op["team_line"].abs()
    y = te["min"].to_numpy()
    for h in ["d1_1730", "d0_1100", "t_120", "t_15"]:
        k = (te.horizon == h).to_numpy()
        r = lambda p: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))
        rc, ro, rn = r(m_close.predict(te[feats])), r(m_close.predict(op[feats])), r(m_nol.predict(te[nol]))
        print(f"{fold} {h}: minutes RMSE close {rc:.3f}  open {ro:.3f}  (+{(ro / rc - 1) * 100:.2f}%)  no-lines model {rn:.3f}")
    print(f"  open line present for {te.open_home_line.notna().mean():.1%} of rows")
