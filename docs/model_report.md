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
(Superseded 2026-11-03: the served rows are cohered, section "Team minutes
coherence" below. The numbers in this paragraph are the raw models'.)

**Caveat on the availability numbers above** (section "v1, full features"): they
come from `scripts/validate_full.py`, which fed the context columns twice and
scored with closing lines. The ship gate (`scripts/train_eval.py`, `gate.py`)
and production used the correct 193 columns. Treat those log losses as
unconfirmed until `validate_full.py` is rerun.

## Team minutes coherence, models.cohere (2026-11-03)

**The problem.** A team-game's minutes add up: over its roster, the sum of
p_play x E[MIN | plays] is E[team minutes], 240 plus 25 per overtime period
(241.56 a team-game over 2018-19..2022-23, so TEAM_MINUTES = 241.5). The
models are fit row by row, and out of sample that sum missed with an sd of
12.2 minutes (2023-24 and 2024-25, `cache/preds_eval_v1.parquet`), from 148 to
304. On 2026-11-02 Indiana listed ten players Out at 6 PM ET; its seven others
summed to 188 projected team minutes. It played six of them, 34.5 to 44.7
minutes each (the raw projections said 21 to 30).

**What ships** (`bucketlib/models.py`, `cohere`, called by pipeline.project
right after predict, so the bot, the app and visitor mark-outs all get it):
per team-game, S = sum p mu, sd = sqrt(sum p (1 - p) mu^2), lam = 30^2 /
(30^2 + sd^2), target = S + lam (241.5 - S). Each row not listed Out moves by
w x shift, w = clip((44 - mu) / (44 - 20), 0, 1), one shift per team-game
solved to hit the target, clipped to [min(mu, 1), max(mu, 44)]. Rows listed Out
keep their minutes; p_play and the per-minute rates are untouched; every stat
mean is recomposed as mu_min x rate, so the board moves with the minutes. A
side with fewer than 12 rows (fewer than any training team-game: a broken
roster pull, which the bot flags RED and still commits) is left as predicted,
so the missing players' minutes are not handed to the few present. The served
model_version carries "+tm1".

**Why that shape** (`scripts/fit_cohere.py`; the two scales fitted on 2023-24
only, then confirmed on 2024-25 without refitting):
- Moving the whole gap, in proportion to mu, made minutes WORSE (2023-24 RMSE
  +0.03%). A big gap is partly availability: players with a low p_play who
  play anyway, or the reverse. Regressing the realised conditional gap (actual
  team minutes minus the sum of mu over who played) on the forecast gap, the
  share that is minutes falls as availability gets less certain: on 2023-24
  alone 0.92 (availability sd under 11 minutes) to 0.38 (sd 27 and over); on
  2025-26, measured only, 0.96 to 0.41 (2024-25: 0.91 to 0.63). lam follows
  it, and is 1 when every player is certain in or out.
- An equal shift beat a proportional one (2023-24 at lam 0.5: -0.70% against
  -0.51%; each at its best scale: -0.78% against -0.52%). Untapered, it
  overshot the heaviest players
  (projected over 34 minutes on short teams: residual +0.37, shift +0.56;
  their RMSE +0.15%); the taper took them to -0.17% and the whole fold from
  -0.77% to -0.87%. S0 = 30 and the taper start 20 are interior optima of
  `scripts/fit_cohere.py --grid` (2023-24 minutes RMSE -0.874%; S0 24 -0.841%,
  40 -0.866%; start 16 -0.873%, 24 -0.858%; no taper is not in the top 12).

| fold | team-sum sd | minutes RMSE | 95% CI | PTS | PRA | Fantasy | over 34 min |
|---|---|---|---|---|---|---|---|
| validate 2023-24 (fitted) | 12.2 to 4.7 | -0.87% | -1.07, -0.68 | -0.29% | -0.43% | -0.41% | -0.17% |
| test 2024-25 | 12.3 to 4.7 | -0.87% | -1.05, -0.67 | -0.30% | -0.42% | -0.40% | -0.00% |
| holdout 2025-26 (measured only) | 11.4 to 3.9 | -0.87% | -1.05, -0.68 | -0.17% | -0.28% | -0.27% | +0.13% |

RMSE change on played rows; the CI is a player-cluster bootstrap (1,000 draws)
of the minutes change. By the size of the team gap, the biggest gain most
(team over by 20+: -4.0%, -5.5%, -3.2%; short by 20+: -2.0%, -2.3%, -4.0%);
within 4 minutes of adding up, -0.03% to -0.10%. Every horizon gains (-0.3%
to -1.3%).

**Probabilistic, same rows before and after** (the simulator unchanged,
`models/sim_params_v1.json`; 6,000 player-games per fold, the gate's proxy
lines, half and whole): mean over stats and line kinds, ECE 0.0197 to 0.0188
(2023-24) and 0.0142 to 0.0139 (2024-25); log loss of P(over) 0.6209 to
0.6195 and 0.6241 to 0.6222; RPS 1.871 to 1.865 and 1.894 to 1.886;
side-picking edge +6.80 to +6.89 and +7.86 to +8.10 points. Log loss and RPS
improve on nearly every stat; no refit of the simulator. On a second sample,
the rows `scripts/prob_gate.py --preds cache/preds_eval_tm1.parquet --tag tm1`
draws (`docs/prob_gate_tm1.csv`), the same direction: ECE 0.0228 to 0.0208 and
0.0181 to 0.0167, log loss 0.6264 to 0.6248 and 0.6245 to 0.6231, RPS 1.870 to
1.862 and 1.903 to 1.897, edge +6.10 to +6.07 and +6.88 to +7.05. Do NOT
compare `docs/prob_gate_tm1.csv` with `docs/prob_gate_v1.csv` cell for cell:
the gate samples played rows with mu_min over 3 by its OWN means, so a step
that moves mu_min changes which player-games are scored (the 2023-24 edge reads
6.80 in one file and 6.07 in the other; on the same rows it is 6.10 and 6.07).

**Mark-outs** (`scripts/gate_markout.py --production`, with and without
`--raw`, the shipped models so the step is the only difference: in sample, 600
team-games per fold at d0_1100, `docs/gate_markout_tm1.csv`). Marking out a
team's top player lost 2.7 to 3.3 team minutes with predict alone, teammates
absorbing 91-92% on average (68-70% at the 10th percentile); with the step it
loses 0.6 to 0.8, teammates absorb 98% (90% at the 10th percentile), and the
opponent still does not move (0.0). The walk-forward rerun that
`docs/gate_markout_v1.csv` records for the raw models has not been repeated.

**What it cannot fix.** The availability side of a gap: Kelly Oubre Jr.
(p_play 0.99 at 6 PM) did not play on 2026-11-02, so Indiana's six played 40
minutes on average against about 34 cohered (the raw models said 27). Late
scratches are invisible before tip. The two scales were fitted on
predictions of models trained through 2022-23; after a retrain, regenerate
those predictions with `scripts/train_eval.py` before `scripts/fit_cohere.py
--grid` means anything (its docstring has the steps), since the script reads
train_eval's own walk-forward predictions, not the production models'.

## Fantasy Score on the shown numbers (2026-11-05)

**Barrett's call.** Fantasy Score is PrizePicks' formula (PTS + 1.2 REB + 1.5
AST + 3 STL + 3 BLK - TOV) on the medians the page shows for its six parts,
each rounded to the tenth it prints (`props.fantasy_shown`). It is the Game
table's FS column and the number a Fantasy Score line is judged against: the
simulated fantasy distribution is slid onto it with its shape kept
(`props.fantasy_on_shown`). Before, FS was the mid-p median of the simulated
fantasy total, which a visitor could not reproduce from the row (Maxey on
2026-11-04: his row gives 39.3, FS showed 40.9). A median of a sum is not the
sum of the medians, so the shown formula runs about 1-2 points under the
simulated median for a starter.

**Measured cost** (`scripts/prob_gate.py`, which now prices fantasy the board's
way, on `cache/preds_eval_tm1.parquet`: the same 6,000 rows per fold and the
same seeds as `docs/prob_gate_tm1.csv`, so the comparison is paired; every
other stat is identical):

| fold | calibration error | side-picking edge | RPS |
|---|---|---|---|
| 2023-24 | 0.025 to 0.052 | +4.28 to +3.83 | 5.214 to 5.243 |
| 2024-25 | 0.019 to 0.043 | +5.72 to +5.49 | 5.335 to 5.380 |
| 2025-26 (measured only) | 0.053 | +6.58 | 5.301 |

The fantasy row of `data/stat_reliability.csv` drops from strong (+6.3, the
simulated median on the v1 predictions) to moderate (+3.83, the worst fold
under the new rule, downgrade-only). Its displayed tier was already weak,
inherited from Blocks. The rule trades calibration for a number that adds up
from the row; switching the board back to the simulated median would restore
the calibration and leave the table's FS as it is.

