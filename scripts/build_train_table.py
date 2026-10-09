"""Build the full training table: history + context + lines + injury horizon.

    python scripts/build_train_table.py [--first 2018-19] [--last 2025-26]

Rows: every ROSTERED player-game (played, DNP, inactive) in state games of the
participation seasons, so the availability model has its denominator and the
minutes/rate models filter to played rows. Seasons before --first contribute
LOG-ONLY warm-up events (played rows) so long-memory rate features are exact;
their season-scoped availability features reset at the season boundary, so the
warm-up cannot leak into a training row's availability features.

Injury horizon: ONE horizon per GAME, chosen by a stable hash of the game id
(every horizon is represented ~equally; teammates in a game share one snapshot,
which the teammate features require). Rows before 2018-12-17 get the pre-archive
state. Lines are ESPN closing lines (the validation compares close vs open).

Output: cache/train_table_full_v1.parquet (local only; never committed).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import features as F  # noqa: E402
from bucketlib import horizons as HZ  # noqa: E402
from bucketlib import livectx as L  # noqa: E402
from bucketlib import state as S  # noqa: E402
from bucketlib import train as T  # noqa: E402

OUT = C.CACHE_DIR / "train_table_full_v1.parquet"
HORIZON_LIST = list(HZ.HORIZONS)


def horizon_of(game_id: str) -> str:
    h = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
    return HORIZON_LIST[h % len(HORIZON_LIST)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="2018-19")
    ap.add_argument("--last", default="2025-26")
    ap.add_argument("--warmup-from", default="2014-15")
    args = ap.parse_args()
    t0 = time.time()
    part_seasons = T.seasons(args.first, args.last)

    # The history, assembled ONCE for training and the committed serving seed
    # (state.seed_inputs): log-only warm-up events, then participation events
    # through tables.roster_games; RS + PI team games; priors over every log
    # season since 2011-12; schedule dates (schedule loaded through 2026-27);
    # every name spelling. The serving seed is this same history.
    hist = S.seed_inputs(args.last, part_first=args.first, warmup_from=args.warmup_from,
                         schedule_through="2026-27")
    rg, ev = hist["roster_games"], hist["events"]
    warm_seasons = sorted(set(ev["season"]) - set(part_seasons))
    print(rg.groupby("season").size().rename("roster rows").to_string())

    lines = T.load_table("game_lines", T.seasons(args.warmup_from, args.last))
    df = F.build(ev, None, hist["team_games"], hist["schedule_dates"], hist["priors"],
                 lines=lines[["game_id", "home_line", "total"]])
    df = df[df["season"].isin(part_seasons)].copy()
    print(f"features {df.shape} ({time.time() - t0:.0f}s)")

    # ── Injury context at one horizon per game ──────────────────────────────
    df["horizon"] = df["game_id"].map(horizon_of)
    roster = rg[["game_id", "team", "person_id", "player_name"]]
    # Every spelling each person id has carried, from both sources and all seasons.
    known = hist["names"]
    ctx_parts = []
    for s in part_seasons:
        ip = C.read_parquet(C.CACHE_DIR / f"injury_horizons_{s}_v1.parquet")
        it = C.read_parquet(C.CACHE_DIR / f"injury_horizon_teams_{s}_v1.parquet")
        if ip is None or it is None:
            raise FileNotFoundError(f"injury horizons for {s} missing: run build_injury_horizons.py")
        rows = df[df["season"] == s]
        # Keep only each game's assigned horizon.
        it = it.merge(rows[["game_id", "horizon"]].drop_duplicates(), on=["game_id", "horizon"])
        ip = ip.merge(rows[["game_id", "horizon"]].drop_duplicates(), on=["game_id", "horizon"])
        ip = L.resolve_names(ip, roster[roster["game_id"].isin(set(rows["game_id"]))], known)
        print(f"{s}: injury rows {len(ip)}, unresolved names {ip.attrs['unresolved']}")
        # Games before the archive have no horizon rows: give them the
        # pre-archive state explicitly (never "no report").
        have = set(it["game_id"])
        pre = rows.loc[~rows["game_id"].isin(have), ["game_id", "team", "tip_utc"]].drop_duplicates(["game_id", "team"])
        if len(pre):
            add = pre.assign(state="pre_archive", report_ts_et=pd.NaT, cutoff_utc=pd.NaT)
            it = pd.concat([it, add[["game_id", "team", "state", "report_ts_et", "cutoff_utc"]]],
                           ignore_index=True)
        ctx_parts.append(L.build_context(rows, ip, it))
    out = pd.concat(ctx_parts, ignore_index=True)
    out = T.add_derived_targets(out)
    C.write_parquet(out, OUT, meta={"seasons": part_seasons, "warmup": warm_seasons})
    print(f"wrote {OUT.name}: {out.shape} in {time.time() - t0:.0f}s")
    print(out.groupby(["season", "horizon"]).size().unstack().to_string())
    print("status share by horizon (played rows):")
    pl = out[out["played"] == True]  # noqa: E712
    print(pl.groupby("horizon")[["inj_out", "inj_questionable", "inj_unknown", "inj_not_listed",
                                 "inj_pre_archive"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
