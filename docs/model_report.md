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
