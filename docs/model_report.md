# Model report (living document)

## v0, logs-only (2026-10-08)

Events from the box logs alone (no absence rows, no injury reports, no lines).
Train 2015-16..2022-23 regular season, validate 2023-24, test 2024-25; 2025-26
untouched. 159 features. Leakage: the future-mutation check (scramble every
outcome on/after D, D's features unchanged; later dates do change) passed on three
dates before any number below was trusted. Full table: `docs/model_report_logs_v0.csv`.

| | vs b1 season-to-date | vs b2 EWMA min x EWMA rate |
|---|---|---|
| Minutes (RMSE) | +21% / +21% | +7.7% / +10.9% |
| Fantasy (RMSE) | +9.8% / +9.6% | +3.5% / +4.1% |
| PTS, REB, combos (MAE) | +3 to +4% | +0.2 to +1.6% |
| AST, STL, BLK, 3PM, FTM (MAE) | +0 to +1% | -0.5 to -1.5% |
| every count (Poisson deviance) | +18 to +32% | +7 to +10% |

**Reading it.**
- Minutes are where a model earns its keep (the research predicted it: knowing
  minutes cuts PRA error 45% vs ~6% for STL/BLK/TOV).
- The MAE losses on low counts are a METRIC ARTIFACT, the same one GridironValue's
  TD verdict hit: these models estimate the MEAN (Poisson loss), and MAE is
  minimised by the MEDIAN, which for a stat averaging ~1 sits at or below the mean.
  On deviance the same predictions beat both baselines on every count.
- The deviance margins are INFLATED: b1/b2 can be exactly 0 (a player with no
  blocks yet), and deviance punishes a zero prediction for a nonzero outcome
  hard. A fair point comparison needs a shrunk baseline (rate pulled toward the
  league rate with ballast). The deciding gate is the probabilistic one (spec 3.4):
  RPS of the discrete pmf and graded side-picking against proxy lines on the
  PrizePicks grid, because that is what the product shows.

## Structure (exp_structure.py, 2026-10-08)

Gain over b2 on MAE (validate / test), same folds and rows:

| | direct | model min x baseline rate | model min x rate model |
|---|---|---|---|
| PTS | +0.64 / +1.06 | -0.24 / +0.42 | +0.16 / +0.85 |
| PRA | +0.21 / +1.01 | -0.01 / +0.93 | +0.30 / +1.24 |
| Fantasy (RMSE) | +3.54 / +4.08 | +2.88 / +4.16 | **+4.20 / +5.57** |
| AST | -1.37 / -1.25 | -0.46 / -0.10 | -1.23 / -1.05 |

No structure dominates. Minutes x rate-model wins the composite (Fantasy, PRA) and
is close on the rest; direct wins PTS/REB marginally. Decision deferred to the
probabilistic gate with availability and injury features in (they change minutes
most, which favours the decomposition). Full table: `docs/exp_structure_logs_v0.csv`.

## v1, full features (2026-10-09): THE GATE

Training table: 325,371 rostered player-games 2018-19..2025-26 (participation +
history from 2014-15 + ESPN closing lines + the injury report at one of four serve
horizons per game). Train 2018-19..2022-23, validate 2023-24, test 2024-25;
2025-26 untouched. Structure: availability classifier; minutes | plays regressor;
per-minute exposure-Poisson rate models; E[stat | plays] = minutes x rate.

**Availability** (log loss; status lookup / recent played rate in brackets):
5:30pm day before 0.35-0.36 (0.62 / 0.45); 11am 0.29 (0.45-0.48 / 0.44-0.47);
2h before 0.24-0.25; 15 min before 0.20-0.21. Sum of P(play) per team-game /
appearances 0.98-1.02.

**Ship gate** (`scripts/gate.py`, `docs/gate_v1.csv`): improvement over the SHRUNK
baseline with 95% player-cluster bootstrap intervals; RMSE-based for minutes and
Fantasy, Poisson deviance for counts. **All 22 targets pass** on both folds, and no
stratum (starters, bench, cold start, rookies, teammate out, returning from
absence, status unknown) is significantly worse. Minutes 15-17% (squared error),
PRA 9-10%, Fantasy 7-8%, PTS 6-7.5%, REB 6.6%, AST 4-5%, STL/BLK 2-2.6%.

**Closing-line optimism** (`scripts/exp_open_vs_close.py`): scoring the same models
with OPENING lines raises minutes RMSE 0.4-0.9% at 5:30pm the day before and
~0.05% at 15 minutes before tip. Lines help ~1% at early horizons, nothing at tip.

**Probabilistic gate** (`scripts/prob_gate.py`, `docs/prob_gate_v1.csv`): the
simulator (fitted walk-forward on 2020-21..2022-23, `models/sim_params_v1.json`)
vs a negative binomial with the same mean. Simulator calibration error on half
lines 1-2.6 points (NB 2.2-4.9); RPS equal or better on every stat. Side-picking
edges are measured against PROXY lines from the baseline projection, NOT
PrizePicks lines, and become `data/stat_reliability.csv` tiers (worse fold):
Fantasy/PRA/P+R/P+A/PTS/R+A strong, STL/3PM/TOV/REB/Blks+Stls/AST moderate, BLK
weak. These are model-quality tiers, never market edges (DiamondValue: the posted
line is a near-sufficient statistic for its projection).

**Decision: every board stat ships as a model.** Nothing falls back to a baseline.

## v1 holdout, 2025-26 (2026-10-09, run ONCE)

Same code, trained on 2018-19..2024-25, scored on 2025-26 (43,265 rostered
player-games), never looked at before this run. `scripts/train_eval.py
--train-last 2024-25 --eval 2025-26`, then `gate.py` and `prob_gate.py` with
`--tag holdout_2025-26` (`docs/gate_holdout_2025-26.csv`,
`docs/prob_gate_holdout_2025-26.csv`).

**Ship gate: all 22 targets pass**, no stratum significantly worse. Gains match
or beat the earlier folds: minutes 17.5%, PRA 11.1%, Fantasy 8.9%, PTS 7.8%,
REB 7.4%, AST 5.4%, STL 2.6%, BLK 2.1%. On MAE (reported, not the gate metric)
BLK and STL are -0.3% and -0.1%: the mean-vs-median artifact from v0.

**Probabilistic:** simulator calibration error 0.5-2.9 points (NB 1.1-4.7), RPS
equal or better than NB on every stat. One reversal: TOV calibration, simulator
1.8 vs NB 1.1 points (RPS tied).

**What moved:** 3PM and Blks+Stls side-picking edges roughly halved (3.5 to 1.9,
3.2 to 2.0 points). Now that the holdout is spent, the tiers in
`data/stat_reliability.csv` take the WORST of all three seasons, which moves
both from moderate to weak. That is a downgrade-only change; nothing was promoted
on holdout evidence.

## Audit measurements (2026-10-09)

**Opening vs closing lines, part 2** (`scripts/exp_open_vs_close.py --part 2`,
`docs/open_vs_close_v1.csv`). Training uses CLOSING lines at every horizon; the
site has opening or current lines. Scoring the same models with opening lines
costs P(play) log loss +0.55% / +0.87% at 5:30pm the day before, +0.69% / +1.14%
at 11am, +0.55% / +0.82% two hours out, about 0 at 15 minutes (2023-24 / 2024-25).
Where the spread moved 3+ points the early-horizon cost is +0.85% to +4.3%. PTS
and Fantasy move 0 to 0.3% overall. Decision: no retrain on opening lines.

**Mark-out conservation** (spec 3.4(b), `scripts/gate_markout.py`,
`docs/gate_markout_v1.csv`, walk-forward models, 600 team-games per fold, 11am
horizon). Marking out a team's top player (about 33 expected minutes) loses 1.7 to
2.6 team minutes; teammates absorb 92-95% on average (68-70% at the 10th
percentile); the opponent does not move. Coherence: the p_play-weighted full-
roster minutes sum misses actual team minutes by +2.3 (2023-24), -6.0 (2024-25)
and +2.5 (2025-26, measured only); the played-only sum by -0.8 to +2.1. No
rescaling shipped; the 2024-25 shortfall is open (docs/known_issues.md).

**Caveat on the availability numbers above** (section "v1, full features"): they
come from `scripts/validate_full.py`, which fed the context columns twice and
scored with closing lines. The ship gate (`scripts/train_eval.py`, `gate.py`)
and production used the correct 193 columns. Treat those log losses as
unconfirmed until `validate_full.py` is rerun.
