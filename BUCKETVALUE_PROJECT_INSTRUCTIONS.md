# BucketValue: build spec

NBA per-game player projections, the basketball sibling of DiamondValue (MLB,
`Desktop/Claude/MLB Predict`) and GridironValue (NFL, `Desktop/Claude/NFL Predict`).
Pick a date, see every game, open a game, see a projected stat line for every
rostered player, compare against PrizePicks lines the user pastes in.

Written 2026-10-08 from a live-verified research pass (7 domains, each re-probed by
an adversarial verifier). Source-level facts and traps live in `docs/data_notes.md`;
this file holds the decisions. Where the two disagree, the data notes are newer.

## 0. Decisions already made

| Decision | Value | By |
|---|---|---|
| Name | **BucketValue**, one constant (`theme.SITE_NAME`) | Barrett, 2026-10-08 |
| League | NBA only (WNBA season is over; SwishValue covers WNBA contracts) | default |
| Shape | Same as the siblings: Streamlit, HistGB models, walk-forward validation, scheduled bot refresh, public Streamlit Cloud deploy | "the same as gridiron and diamond value" |
| PrizePicks | INLINE on the board and game pages, never a tab. PASTE ONLY: no fetcher, ever (403 to servers and ToS 16(l) bars automation) | sibling precedent |
| Repo | `Desktop/Claude/NBA Predict`, package `bucketlib/` | default |
| Accuracy page public or local-only | OPEN, ask Barrett before deploy (DiamondValue removed its page from the public site; GridironValue kept one) | Barrett |
| GitHub repo creation, Actions probe | needs Barrett's go-ahead (outward-facing) | Barrett |

## 1. Lessons that are rules here (each was a shipped bug on a sibling)

1. **Two estimands, never mixed.** Game logs hold only players who appeared, so
   every stat model estimates E[Y | plays]. That is what a prop row shows (a DNP voids
   the pick, so the posted line is itself conditional). `p_play` lives in its own
   column. `{stat}_expected = p_play x conditional` is for team totals and coherence
   checks only.
2. **Point-in-time or nothing.** Every rolling feature is built from games strictly
   before the target game (`shift(1)` inside the production builder). GridironValue's
   `_prior_roll_mean` lacked it and 30-39% of its "accuracy" was the leak.
3. **Never group or rank within the current game's rows.** In an appeared-only table,
   "who is here" is postgame information. Compare against norms from prior games.
4. **A leak replicates in every fold.** Two-stage gating catches mirages, not leaks.
   An implausibly large gain is the alarm. Leakage tests CALL PRODUCTION CODE; a test
   that reimplements the filter it checks tests the copy.
5. **Train/serve parity raises.** Inference RAISES on a missing trained-on column and
   on a trained-on column that is all-NaN at serve time. (GridironValue had four silent
   skews; DiamondValue's `reindex` still fails open.) A gate-passing feature is not
   shipped until it is retrained AND served.
6. **Training-only sources are never features.** A source that does not exist at
   serve time (post-game participation, closing lines treated as known in the
   morning, post-game referee assignment) is simulated as absent in a parity test.
7. **Freshness never reads file mtime.** Committed files get a fresh mtime on every
   clone. Every cached payload embeds `fetched_at` (`bucketlib.cache`).
8. **ET dates only.** `util.today_et()`. A test bans `date.today()` and friends.
9. **Stale beats empty, but a cold failure raises.** An empty or failed fetch never
   overwrites good data. A `Blocked` is never read as "no games".
10. **Paired comparisons only.** Model error and baseline error are compared on the
    same rows (DiamondValue published +4.60% that was really -0.07%). Per-target
    percentages, then averaged; never pooled MAE across targets.
11. **Zero-fill of a missing sub-document is a phantom-actual generator.** A DNP or
    inactive player has NO stat line, not a line of zeros.
12. **Calibration maps are fitted on walk-forward predictions**, never in-sample.
13. **Bump the artifact filename version when SEMANTICS change**, not just schema.
14. **The record is frozen at first sight** and stamped with the model version.
15. **Rank by confidence, never by raw gap.** The gap is an argmax over the model's
    own error and is not comparable across stats.
16. **The board is at its ceiling.** DiamondValue measured that the posted line is a
    near-sufficient statistic for its projection (incremental R^2 0.0007). Present
    projections honestly; never sell the props comparison as an edge. Always report a
    hit rate against the base rate of taking the more common side, and against the
    54.7-58.5% per-leg break-even.
17. **Every query param reaching `unsafe_allow_html` goes through `util.esc`.**
    `teams.canonical` returns None for unknown strings, never a pass-through.
18. **Only link pages that exist in git.** A nav link to a missing page renders the
    shell and then a client-side "not found".
19. **Clone and run before calling a deploy ready.** Derived artifacts must be
    un-ignored one by one; a fresh clone of GridironValue silently lacked two.
20. Receipts for every change set in `Desktop/Claude/receipts/`; no em dashes in UI
    text (the sentinel for a missing value is the one exception).

## 2. Data (details and traps: `docs/data_notes.md`)

| Need | Source | Era | Notes |
|---|---|---|---|
| Player box logs (training + daily) | stats.nba.com PlayerGameLogs Base+Advanced, LeagueGameLog P | 2014-15+ | whole season per call; curl_cffi Chrome + nba_api headers or it hangs |
| Team logs, pace, OT | LeagueGameLog T, TeamGameLogs Advanced | 2014-15+ | |
| Schedule, tip times, Cup final, neutral site | stats `scheduleleaguev2` | current + history if available | current season is incomplete until the Cup group stage ends |
| Participation (DNP, inactive, starters, reasons) | cdn.nba.com liveData box score | 2019-20+ | Referer + Chrome fingerprint; 1 call/game |
| | stats BoxScoreTraditionalV3 + SummaryV3 | 2014-15..2018-19 | starters = first 5 listed; V2 endpoints return silent empties |
| Pre-game injury status | official NBA injury-report PDFs | 2018-12-17+ | 3 layouts, 3 filename eras; title time, not filename time |
| Closing spread/total (training) | ESPN core odds API, per event | 2014-15+ | cross-book median with an allow-list; Kaggle CC0 file only as a LOCAL cross-check |
| Live lines | ESPN scoreboard (DraftKings) + core odds | serve | odds vanish at tip: snapshot before, recover from core after |
| Live injuries | NBA injury PDF (primary) + ESPN injuries endpoint | serve | ESPN historical injuries are TODAY's list: never a training source |
| Rosters, two-way flag | stats `playerindex` (SUPPLEMENTAL_STATUS==1) | daily | camp rosters (18-21) until ~Oct 19 cuts |
| Bio (birth date, height, draft) | CommonTeamRoster per team-season, playerindex | 2014-15+ | |
| Headshots/logos | cdn.nba.com/headshots, /logos | serve | silhouette 200 for unknown ids, never 404 |

Raw payloads are cached gzipped under `cache/raw/` and never committed. Historical
lines are never committed either (provenance); only the current season's own
pre-tip snapshots are, because next season's training needs them and the bot's
runner is ephemeral.

Training window: 2014-15 .. 2025-26 regular season (+ play-in, flagged). Playoffs
are stored but excluded from v1 training (rotations shorten; different estimand).
2019-20 bubble games and the 72-game 2020-21 season are kept and flagged.

## 3. Model

### 3.1 Targets
Per player-game, conditional on playing: MIN, PTS, REB, AST, FG3M, STL, BLK, TOV,
FGM, FGA, FG3A, FTM, FTA, OREB, DREB, PF. Derived from the same draws: PRA, P+R,
P+A, R+A, Blks+Stls, two-pointers made/attempted, and Fantasy Score
(PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV, which equals stats.nba.com
FANTASY_PTS exactly). Overtime counts in full-game props.

### 3.2 Structure (to be confirmed by the ship gate, not assumed)
1. **Availability** P(plays at least one second) per rostered player-game: rows are
   every player on the participation table (played + DNP + inactive). Features:
   injury-report status and reason category as of the last report before tip
   (2018-19+), games missed streak, days since last appearance, recent minutes,
   back-to-back, age/workload, two-way and G League flags.
2. **Minutes | plays**: short-memory minutes EWMA (half-life ~2 games beat
   season-to-date, 6.44 vs 7.08 RMSE), starts share, games-back-from-absence by
   absence length, minutes vacated by teammates ruled out pre-game, rotation depth,
   |spread| and total, rest/B2B, prior-season MPG for cold start (prior worth only
   ~1-4 games for minutes), age.
3. **Per-minute rates** per stat, longer memory (half-life 5-20 games, Marcel-style
   prior: 6/3/1 seasons with league-average ballast), usage vacated by absent
   teammates, opponent allowed rates (shrunk), expected pace (team x opp / league),
   implied team total, home (small).
4. **E[stat | plays] = E[min | plays] x rate**, compared at the gate against a direct
   per-stat model that takes predicted minutes as a feature (out-of-fold).

"Ruled out pre-game" in training = on the inactive list or listed Out on the last
injury report before tip. DNP-coach's-decision is an IN-GAME decision and never
counts as "out" for a teammate feature (that would leak blowouts).

### 3.3 Validation
Walk-forward by season: train <= 2022-23, validate 2023-24, test 2024-25, holdout
2025-26 touched ONCE at the end of the programme. Baselines: (1) season-to-date mean
(this game excluded), (2) shrunk EWMA/Marcel rate x minutes. Gate: beat BOTH baselines
on BOTH validate and test, stratified (starters / bench / cold start <10 games /
rookies) because pooled MAE hides damage. Counts are also scored on Poisson deviance
(MAE is minimised by the median; GridironValue's TD verdict was a metric artifact).
A hypothesis read off a fold is not evidence from that fold (the selection trap).
Targets that fail ship a baseline, labelled as such in the UI.

### 3.4 Distributions and props
Count stats have point mass, and PrizePicks posts whole-number lines (ties are their
own bucket, neither win nor loss). For every line compute P(>), P(=), P(<) from a
discrete distribution; lean = argmax(P(>), P(<)) with an explicit Even branch.
Candidate distribution families (chosen out of sample on the validation season by
calibration error, confirmed on test): per-stat negative binomial with level-dependent
dispersion; empirical conditional pmf by projection bin; simulation from a minutes
distribution x per-stat counts with combos and Fantasy from the SAME draws (keeps
correlations). FTM/FTA and PTS are overdispersed even given minutes; REB/AST/3PM/STL/
BLK/TOV are near-Poisson given minutes; PF is underdispersed.
PrizePicks rules (first-party, 2026): DNP = zero seconds played, pick removed. NBA
Reboot: a MORE pick on a full-game projection is voided when the player leaves in the
1st half and does not return, unless he had already cleared the line. So P(More wins)
excludes P(1H-only exit AND not yet over); Less includes it. Period props (1H, 1Q,
First 5 Minutes), Dunks, and combo-player squares are counted as unsupported.
Display the median (consistent with the lean by construction) and rank by
|P - 0.5|, tier first. Demon/Goblin direction is read PER ROW from the pasted board
(Less/More buttons) or the feed's allowed_wager_types; never assumed (PrizePicks'
own pages contradict each other about NBA as of 2026-10-08).

## 4. Serving

The bot precomputes; the app recomputes the volatile part.

- **Bot** (GitHub Actions, if it can reach the sources; probe first): after each
  night, append box logs and participation for the current season, rebuild the
  player-state table, score yesterday into the accuracy history, snapshot rosters,
  schedule, injury report and ESPN odds, and precompute today's and tomorrow's
  projections. Commit. Actions cron starts were measured at a MEDIAN of 3.9 hours late
  on DiamondValue (range 2.25-7.4h), so: many cron entries, every run idempotent, the
  ET date derived inside the script every time, and nothing time-critical depends on
  one run landing.
- **App** reads the committed state and, for today's slate, fetches the latest
  injury report and ESPN odds live (embedded-stamp cache, ~10 min TTL) and re-runs
  inference (state + context -> rows -> models; milliseconds). If the live fetch fails
  it serves the bot's committed projections with a visible "as of" stamp. A per-game
  "mark out" control re-runs the teammate redistribution in-session (late scratches
  land 30 minutes before tip; no cron can follow them).
- One feature builder serves training and inference: the player-state table is the
  post-game rolling state; training features are that state shifted one game;
  inference features are the latest state. A test asserts the two paths agree on the
  last historical game.

## 5. Site

Pages: `/` (date bar in ET, game cards with team-colour split stripe, tip time,
spread/total, injury count, PrizePicks line-count pill top-left, slate-wide model vs
board), `/Game` (team-colour header, matchup context, two team tables: status badge,
P(play), MIN, PTS, REB, AST, 3PM, STL, BLK, TOV, PRA, Fantasy; expandable rows with
the player's pasted lines and, after the final, actuals), `/About`. Same look family
as DiamondValue/GridironValue (themed HTML tables, light/dark, Space Grotesk headers),
headshots from the NBA CDN. Shareable links minted explicitly with readable params.
Paste via an "Update lines" popover; pastes live in session_state on the shared
deployment; only local pastes reach the frozen record.

## 6. Phase plan (STOP = check in with Barrett)

1. Foundation + source modules + full backfill (running).
2. Tables: player-game, team-game, participation, lines, injury status, bio; joined
   training table with a column contract (`columns.classify` raises on unknowns).
3. Features + baselines + walk-forward validation report. **STOP**: show the gate
   results and what ships as a model vs a baseline.
4. Distributions, calibration, props layer (port GridironValue's hardened props.py
   with the whole-line, combo and Reboot fixes).
5. Inference + app pages + bot script; local end-to-end on a 2025-26 date and on the
   2026-27 preseason.
6. **STOP**: GitHub repo + Actions reachability probe + Streamlit Cloud deploy
   (Barrett creates/approves), accuracy page visibility decision.
7. Full audit workflow before opening night (2026-10-20).
