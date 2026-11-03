"""Spec 3.4(b), two gates the point gate cannot see: TEAM MINUTES COHERENCE and
"MARK OUT CONSERVES TEAM MINUTES". Neither was measured before 2026-10-09.

For each fold (validate 2023-24, test 2024-25, holdout 2025-26) the models are
trained walk-forward on every season before it (models.train: the production
structure and hyperparameters), or with --production the shipped models/ (then
every fold is in sample, and the output says so). On N_GAMES random regular
season games per fold (seeded), the context is rebuilt at the gate horizon
(d0_1100) from the archived injury horizons through livectx.build_context, the
training context path, and models.predict scores it, then models.cohere (the
pipeline's team-minutes step; --raw leaves it out, which is how the v1
measurements of 2026-10-09 in docs/gate_markout_v1.csv were taken).

  coherence   per team-game: full roster sum p_play x mu_min, and played-only
              sum mu_min (who played is postgame information: a coherence
              check, never a forecast), against the actual team minutes and
              240 + 25 x OT periods.
  mark-out    per team-game, the player with the largest p_play x mu_min who is
              not listed Out is marked out (livectx marked_out: status Out at
              age 0, exactly what a visitor's mark-out does on the site), one
              game per call so no other game sees the mark. Reported: the team
              sum before and after, the absorbed share (teammates' gain over
              the star's loss), and the opponent's change (must be 0).

    python scripts/gate_markout.py [--production] [--raw] [--games 300] [--out docs/gate_markout_tm1.csv]

Writes one row per (fold, measure) to the csv, with the step it scored.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bucketlib import cache as C  # noqa: E402
from bucketlib import livectx as L  # noqa: E402
from bucketlib import models as M  # noqa: E402
from bucketlib import state as S  # noqa: E402
from bucketlib import train as T  # noqa: E402

FOLDS = {"validate": "2023-24", "test": "2024-25", "holdout": "2025-26"}
HORIZON = "d0_1100"
TABLE = C.CACHE_DIR / "train_table_full_v1.parquet"
OUT = Path(__file__).resolve().parent.parent / "docs" / "gate_markout_tm1.csv"
OT_MINUTES = 25.0       # five players x five minutes per overtime period


def fold_inputs(season: str, gids) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(history-feature rows without context, resolved injury rows, team
    rows) at HORIZON for the sampled games of `season`."""
    t = pq.read_table(TABLE, filters=[("season", "==", season)]).to_pandas()
    t = t[t["game_id"].isin(set(gids))]
    rows = t.drop(columns=[c for c in [*L.CTX_COLS, "horizon"] if c in t.columns]).reset_index(drop=True)
    ip = C.read_parquet(C.CACHE_DIR / f"injury_horizons_{season}_v1.parquet")
    it = C.read_parquet(C.CACHE_DIR / f"injury_horizon_teams_{season}_v1.parquet")
    ip = ip[(ip["horizon"] == HORIZON) & ip["game_id"].isin(set(gids))]
    it = it[(it["horizon"] == HORIZON) & it["game_id"].isin(set(gids))]
    part = C.read_parquet(C.CACHE_DIR / f"participation_{season}_v1.parquet")
    roster = part.loc[part["game_id"].isin(set(gids)), ["game_id", "team", "person_id", "player_name"]]
    names = C.read_parquet(S.SEED_DIR / S.SEED_FILES["names"])
    return rows, L.resolve_names(ip, roster, names), it


def expected(models: M.Models, ctx: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    """Per row: p_play, mu_min and p_play x mu_min as served (models.cohere
    after predict, the pipeline's path) or, with raw, as predict alone."""
    p = M.predict(models, ctx)
    if not raw:
        p = M.cohere(p, ctx)
    return ctx[["game_id", "team", "person_id", "played", "min", "inj_out"]].assign(
        p_play=p["p_play"].to_numpy(), mu_min=p["mu_min"].to_numpy(),
        em=(p["p_play"] * p["mu_min"]).to_numpy())


def coherence(e: pd.DataFrame, ot: pd.DataFrame) -> dict:
    key = ["game_id", "team"]
    played = e[e["played"].eq(True)]
    g = pd.DataFrame({"full": e.groupby(key)["em"].sum(),
                      "played_only": played.groupby(key)["mu_min"].sum(),
                      "actual": played.groupby(key)["min"].sum()}).join(ot.set_index(key)["ot_periods"])
    g["regulation"] = 240.0 + OT_MINUTES * g["ot_periods"]
    return {"team_games": len(g), "full_roster_mean": g["full"].mean(),
            "played_only_mean": g["played_only"].mean(), "actual_mean": g["actual"].mean(),
            "240_plus_25_EOT": 240.0 + OT_MINUTES * g["ot_periods"].mean(),
            "full_minus_actual_mean": (g["full"] - g["actual"]).mean(),
            "played_only_minus_actual_mean": (g["played_only"] - g["actual"]).mean(),
            "played_only_minus_actual_rmse": float(np.sqrt(((g["played_only"] - g["actual"]) ** 2).mean()))}


def markout(models: M.Models, rows, ip, it, e0: pd.DataFrame, raw: bool = False) -> dict:
    cand = e0[e0["inj_out"].eq(0)]
    stars = cand.loc[cand.groupby(["game_id", "team"])["em"].idxmax()]
    res = []
    for s in stars.itertuples():
        g = s.game_id
        c1 = L.build_context(rows[rows["game_id"] == g], ip[ip["game_id"] == g], it[it["game_id"] == g],
                             marked_out={int(s.person_id)})
        e1 = expected(models, c1, raw)
        b0, b1 = e0[e0["game_id"] == g], e1
        mine0, mine1 = b0[b0["team"] == s.team], b1[b1["team"] == s.team]
        star1 = float(mine1.loc[mine1["person_id"] == s.person_id, "em"].iloc[0])
        mates0 = mine0.loc[mine0["person_id"] != s.person_id, "em"].sum()
        mates1 = mine1.loc[mine1["person_id"] != s.person_id, "em"].sum()
        res.append({"star_em": s.em, "star_p_play_after": float(
                        mine1.loc[mine1["person_id"] == s.person_id, "p_play"].iloc[0]),
                    "team_before": mine0["em"].sum(), "team_after": mine1["em"].sum(),
                    "opp_change": b1.loc[b1["team"] != s.team, "em"].sum() - b0.loc[b0["team"] != s.team, "em"].sum(),
                    "absorbed": (mates1 - mates0) / (s.em - star1)})
    x = pd.DataFrame(res)
    lost = x["team_before"] - x["team_after"]
    return {"team_games": len(x), "star_expected_min_mean": x["star_em"].mean(),
            "star_p_play_after_mean": x["star_p_play_after"].mean(),
            "team_sum_before_mean": x["team_before"].mean(), "team_sum_after_mean": x["team_after"].mean(),
            "team_minutes_lost_mean": lost.mean(), "team_minutes_lost_p90": lost.quantile(0.9),
            "absorbed_share_mean": x["absorbed"].mean(), "absorbed_share_p10": x["absorbed"].quantile(0.1),
            "absorbed_share_median": x["absorbed"].median(),
            "opponent_change_max_abs": x["opp_change"].abs().max()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="store_true",
                    help="score with the shipped models/ (in sample for every fold)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--raw", action="store_true",
                    help="score predict() alone, without models.cohere (the v1 measurements)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    full = None if args.production else C.read_parquet(TABLE)
    out = []
    for fold, season in FOLDS.items():
        if args.production:
            models, trained = M.load(), "production (in sample)"
        else:
            prior = [s for s in T.seasons("2018-19", season) if s != season]
            models, trained = M.train(full, seasons=prior), f"walk-forward {prior[0]}..{prior[-1]}"
        meta = C.read_parquet(C.CACHE_DIR / f"participation_games_{season}_v1.parquet")
        all_ids = sorted(g for g in meta["game_id"].unique() if str(g).startswith("002"))
        gids = np.random.default_rng(0).choice(all_ids, min(args.games, len(all_ids)), replace=False)
        rows, ip, it = fold_inputs(season, gids)
        ctx = L.build_context(rows, ip, it)
        e0 = expected(models, ctx, args.raw)
        ot = T.load_table("team_games", [season])
        ot = ot.loc[ot["game_id"].isin(set(gids)), ["game_id", "team", "ot_periods"]]
        for measure, vals in (("coherence", coherence(e0, ot)),
                              ("markout", markout(models, rows, ip, it, e0, args.raw))):
            row = {"fold": fold, "season": season, "models": trained, "horizon": HORIZON, "measure": measure,
                   "step": "predict" if args.raw else f"predict+cohere ({M.COHERE_TAG})", **vals}
            out.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
