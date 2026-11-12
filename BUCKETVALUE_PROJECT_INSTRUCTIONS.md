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
| Player lines | INLINE on the board and game pages, never a tab. LICENSED FEED (Barrett, 2026-11-09): lines from DraftKings, FanDuel, BetMGM, PrizePicks and Underdog come from The Odds API under Barrett's key, fetched only by the app's live layer (never the bot, never Actions, never a visitor's browser), for today's ET date, display only. The board shows all five books at once, one column per book, and an up or down arrow on a line marks the model's lean at that line. The paste box stays as the backup (outages, typed picks, Demons and Goblins). PrizePicks itself is never fetched by anything, ever (403 to servers and ToS 16(l) bars automation) | Barrett, 2026-11-09 (was PASTE ONLY, sibling precedent) |
| Repo | `Desktop/Claude/NBA Predict`, package `bucketlib/` | default |
| Accuracy page | **LOCAL ONLY**: `pages/Accuracy.py` and its summary outputs are gitignored, never in the nav of the deployed site (DiamondValue precedent) | Barrett, 2026-10-08 |
| GitHub repo | `Barrett879/bucketvalue`, created PRIVATE 2026-10-08 (token has `workflow` scope, verified) (approved by Barrett; public creation needs his explicit visibility call, and Streamlit only shares apps from public repos, so flip it at deploy) | Barrett, 2026-10-08 |

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
5. **Train/serve parity raises.** Inference RAISES on a missing trained-on column.
   Null fractions are checked STRATIFIED: serve vs training null rates for rows in the
   same games-into-season bucket (0 / 1-2 / 3-9 / 10+) and team-change flag, because on
   opening night every current-season column is legitimately all-NaN and a blanket
   "all-NaN raises" would either stop the site or breed an exemption list. Parity is
   proven by a DATE-REPLAY test (inference rows built from data truncated before D equal
   the training rows bit for bit, on: a season opener, first game after a trade, first
   game back from a 10+ game absence, a B2B second leg, the day after the All-Star
   break, the game after the Cup final, a postponed-game makeup) and a FUTURE-MUTATION
   test (randomize every outcome dated on or after D; all features for date-D rows are
   unchanged). (GridironValue had four silent skews; DiamondValue's `reindex` still
   fails open.) A gate-passing feature is not shipped until it is retrained AND served.
6. **Training-only sources are never features.** A source that does not exist at
   serve time (post-game participation, closing lines treated as known in the
   morning, post-game referee assignment) is simulated as absent in a parity test.
   Player lines from the odds feed are DISPLAY ONLY (Barrett, 2026-11-09): no model reads
   them. Making one a feature needs serve-time parity and a training history, and the
   vendor sells history only on paid plans at 10 times the price.
7. **Freshness never reads file mtime.** Committed files get a fresh mtime on every
   clone. Every cached payload embeds `fetched_at` (`bucketlib.cache`).
8. **ET dates only.** `util.today_et()`. A test bans `date.today()` and friends.
9. **Stale beats empty, but a cold failure raises.** An empty or failed fetch never
   overwrites good data. A `Blocked` is never read as "no games".
   For the odds feed this holds per game and per book: a failed refresh, or one with no
   bookmakers, keeps the last good lines with their own fetched_at; a book that answers
   with no readable line, or with under half the lines it had, keeps its last lines; only
   a line its book took down, while the rest of that book is present, is dropped, and it
   is counted.
10. **Paired comparisons only.** Model error and baseline error are compared on the
    same rows (DiamondValue published +4.60% that was really -0.07%). Per-target
    percentages, then averaged; never pooled MAE across targets.
11. **Zero-fill of a missing sub-document is a phantom-actual generator.** A DNP or
    inactive player has NO stat line, not a line of zeros.
12. **Calibration maps are fitted on walk-forward predictions**, never in-sample.
13. **Bump the artifact filename version when SEMANTICS change**, not just schema.
14. **The record is frozen at first sight** and stamped with the model version.
15. **Rank by confidence, never by raw gap.** The gap is an argmax over the model's
    own error and is not comparable across stats. One display is exempt, by Barrett's
    request (2026-11-05): "Biggest discrepancies" under the board, DiamondValue's strip.
    It sorts the board's shown STANDARD lines by the raw gap (with the odds feed on, every
    book's: the books posting the same line for a player and stat with the same side word
    share one row that names each of them, so a player and stat appear once per line and
    side word, and no book's line is preferred to another's), offers a stat-family filter,
    and never changes the board's own order (props_ui.discrepancies and
    props_ui.merge_books).
16. **The board is at its ceiling.** DiamondValue measured that the posted line is a
    near-sufficient statistic for its projection (incremental R^2 0.0007). Present
    projections honestly; never sell the props comparison as an edge. Always report a
    hit rate against the base rate of taking the more common side, and against the
    break-even of the line's own source: 54.7-58.5% per leg for PrizePicks Power Play
    payouts; for a sportsbook side, the break-even its American price implies (about
    52.4% at -110); none for Underdog until its payouts are verified. DFS prices are
    indicative only and get no price math.
17. **Every query param reaching `unsafe_allow_html` goes through `util.esc`.**
    `teams.canonical` returns None for unknown strings, never a pass-through.
    Odds-feed text (player names) goes through `util.esc` too; book names render from the
    code's own labels, never from the vendor's title strings.
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
| Player lines (display only) | The Odds API event odds, `bookmakers=draftkings,fanduel,betmgm,prizepicks,underdog` (1 region), markets from settings | serve, today ET | app live layer only (bucketlib.live: a background refresher and page views share one walk); never the bot; never PrizePicks itself |
| Live injuries | NBA injury PDF (primary) + ESPN injuries endpoint | serve | ESPN historical injuries are TODAY's list: never a training source |
| Rosters, two-way flag | stats `playerindex` (SUPPLEMENTAL_STATUS==1) | daily | camp rosters (18-21) until ~Oct 19 cuts |
| Bio (birth date, height, draft) | CommonTeamRoster per team-season, playerindex | 2014-15+ | |
| Headshots/logos | cdn.nba.com/headshots, /logos | serve | silhouette 200 for unknown ids, never 404 |

Raw payloads are cached gzipped under `cache/raw/` and never committed. Historical
lines are never committed (provenance). The bot commits only the compact parsed
context each published projection used (injury statuses, line snapshot, state
stamp), never raw JSON or PDFs: ESPN core keeps pregame closes and ak-static keeps
old PDFs, so next season can rebuild from source. Odds-feed payloads and parsed feed
lines are never committed, never written to disk, never used as a test fixture (vendor
fixtures are hand-built), and never put in a visitor's token board or browser copy.

**GAME_UNIVERSE** (one whitelist shared by backfill, bot and app): id prefix `002`
regular season (training + state), `005` play-in (state; NOT projected in v1, the models trained on regular-season rows only; decided 2026-10-09), `004` playoffs
(stored, excluded from v1 training), `001` preseason and `003` All-Star never enter
state, `006` Cup final used ONLY for rest/schedule features (it is in no season log).
Training window 2014-15..2025-26; LeagueGameLog 2011-12..2013-14 is pulled for
priors only. 2019-20 bubble games and the 72-game 2020-21 season are kept and
flagged. Serving features are computed from BOX COUNTS only (possessions, pace and
usage derived from counts in both paths), because the cdn liveData box score the
bot can reach has no pace/usage fields.

**Times.** One tz-aware `tip_et` built from `gameDateTimeUTC`. Trap: the schedule's
`gameDateTimeEst` is ET WALL CLOCK with a "Z" suffix, so parsing it as UTC is off by
4-5 hours (921 of 1,207 2026-27 games have a UTC date different from the ET date).
"Report time < tip" is a production assertion on every injury row. Historical team
membership comes from each game's participation row, never from playerindex (which
returns END-of-season state; 7.8% of 2025-26 rows had a different end-of-season
team). Two-way status as of a date comes from the movement log or the report's
"G League - Two-Way" reason.

## 3. Model

### 3.1 Targets
Per player-game, conditional on playing (row presence or seconds > 0, never MIN > 0;
the played threshold is one constant, to be confirmed from a real paste): MIN, PTS,
REB, AST, FG3M, STL, BLK, TOV, FGM, FGA, FG3A, FTM, FTA, OREB, DREB, PF. Composed from
the same simulated draws: PRA, P+R, P+A, R+A, Blks+Stls, two-pointers made/attempted,
Fantasy Score (PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV == stats.nba.com
FANTASY_PTS exactly). Overtime counts.

### 3.2 The information horizon (design review, adopted)
The product serves and freezes HOURS before tip (bot runs land a median 2-4h late;
the record freezes at first paste, often the night before). So every injury and
absence feature is defined AS OF A HORIZON h, never "at tip":
- own status = the latest injury report with title time <= h; `report_age_min`;
  a three-way encoding (listed with a status / team NOT YET SUBMITTED or no report
  yet / no-report era before 2018-12) and a test that serving never emits the era
  value;
- teammate features = minutes of teammates listed Out, Doubtful, Questionable at h,
  split by status, and NEWLY vacated minutes (out at h but played in the player's
  last game) with the absentee's absence length, because ~70% of vacated rows
  describe absences the player's own EWMA has already absorbed;
- the inactive list and DNP outcomes are TARGETS only, never features;
- rotation depth = rostered players not listed Out at h with an EWMA of 10+ minutes
  (the post-game count of who appeared is a 6.8% leak that looks plausible);
- training rows are built at sampled horizons from the PDF archive: 5:30pm ET the
  day before, game-day late morning, and the last report before tip. The gate is
  scored at the morning horizon; the tip horizon is reported separately.
- lines: training on closes, so validation runs twice (close vs open, where open
  exists: 2014-17 and per book 2023-24+) and reports the gap. A missing line at
  serve never reaches a model as NaN: a no-line fallback model (or a rating-derived
  line with a source flag, trained with random masking).
"Mark out" = status Out at age 0, which the trained definition already covers.

### 3.3 Structure (candidates; the gate decides)
1. **Availability** P(plays) on every participation row (played + DNP + inactive).
   Trained 2018-19+ (or with an era intercept: 28+ MPG absence rose from ~0.10 to
   0.177 by 2025-26; B2B second legs 0.13 -> 0.24). Gated on log loss/Brier against a
   status-lookup baseline and a recent-absence-rate baseline, reliability by status
   and lead time; sum of P(play) per team-game checked against appearances.
2. **Minutes | plays**: EWMA candidates, starts share, return-from-absence by
   absence length, horizon injury features (Questionable players who play may be on
   restrictions), newly vacated minutes, rotation depth, |spread|, total, rest/B2B
   (including the Cup final), prior-season MPG excluding post-elimination games,
   games this season, team-changed flag, age, league-new from DRAFT_YEAR/FROM_YEAR,
   `n_prior_seasons`. SHORT-MEMORY STATE RESETS EACH SEASON (an EWMA carried across
   the summer has opener RMSE 8.98 vs 7.64 for prior-season MPG).
3. **Rates**: Poisson with exposure (target stat/min, sample_weight = precise
   minutes; cameo rows kept), half-life candidates 10/20/40/season per stat with a
   Marcel prior (6/3/1, league ballast as of date), usage vacated, opponent allowed
   rates (shrunk, as-of norms), expected pace from counts, implied team total.
   PF and TOV modelled directly (minutes x rate over-predicts PF 3-8%).
4. **Direct alternative**: Poisson HistGB per board stat with out-of-fold predicted
   minutes and the minutes x rate product as features. Compared within
   predicted-minutes-change strata, not pooled.
League norms (pace, ballast, opponent shrink targets) are AS-OF-DATE expanding means
blended with the prior season, never `groupby(season).mean()` (that includes future
games; first-month pace runs 1.2-2.8% above the season mean).

### 3.4 Validation
Walk-forward: train <= 2022-23, validate 2023-24, test 2024-25, holdout 2025-26 used
ONCE. Calibration fit seasons are produced walk-forward for 2020-21..2022-23.
Baselines use strictly prior games: season-to-date, falling back to prior-season
per-game, then a league role mean (game 1 is 2.1% of rows and every opener row).
Gate: paired, player-cluster bootstrap CIs (pooled must win with the CI above 0; no
stratum may lose beyond its CI); RMSE for MIN and Fantasy, MAE + Poisson deviance for
counts. Strata: starters/bench, cold start (<10 games), rookies/league-new, opener
(first 3 team games), predicted-minutes change, teammate-out, return-from-absence.
Additional gates the point gate cannot see: (a) availability (above); (b) team
minutes coherence, sum p_play x E[MIN|plays] vs 240 + 25 x E[OT], full roster and
played-only, and "mark out" conserves team minutes; (c) probabilistic: RPS/log score
of the discrete pmf and graded side-picking against proxy lines rounded to the
PrizePicks grid (whole numbers included), lines spanning the 10th-90th predictive
percentiles. Targets that fail ship a labelled baseline.

### 3.5 Distributions and props
- Simulation per game (memory-capped: 2-4k draws, float32, per game, cache pmfs not
  draws): minutes from an empirical role-binned residual with an explicit OT component
  and a spread-dependent low tail; counts with a minutes ELASTICITY per stat
  (log E[Y|M] = log(r E[M]) + beta_s log(M/E[M]); PTS/3PM/FTA ~1.2, PF/TOV ~0.6-0.7)
  and Gamma rate uncertainty; shots COMPOSITIONALLY (FGA -> FG3A -> FG3M, FG2M;
  FTA -> FTM; PTS and REB composed) so PTS/3PM/FTM/2PM are coherent and points
  parity falls out. Acceptance: the scaling-by-minutes-surprise table is flat on
  simulated draws; randomized PIT is flat within minutes-surprise bins.
- Every distribution is RE-CENTRED on the shipped mean (test: E_sim[Y] equals the
  shipped E[Y|plays]); baseline-served components draw from the same minutes draw.
- Per line: P(>), P(=), P(<). PrizePicks grading: More wins on Y > L; a 1st-half exit
  with no return before clearing VOIDS More (Reboot); ties are their own bucket. Per
  side q = P_win / (P_win + P_loss), excluding ties and voids; lean = argmax q with an
  Even band; rank tier first, then q. Until an exit model exists (needs play-by-play,
  v1.1), q ignores the Reboot and More leans show a role-based "void risk" note.
- Display the MID-P median m = (k - 0.5) + (0.5 - F(k-1)) / p(k) (agrees with the lean
  on 100% of whole-line and half-line cases measured); P(=) shown on whole lines;
  combos are their own medians (footnoted). Test: sign(m - line) == lean.
- Tiers come from an NBA `validate_lean_accuracy` on validate/test seasons, written to
  `data/stat_reliability.csv`; combos inherit their weakest component; the About trust
  table is generated from it. Displayed confidence is capped at the validated maximum.

## 4. Serving

- **Ingestion** runs on GitHub Actions using ONLY sources the 2026-10-09 probe proved
  reachable from Azure: cdn.nba.com (schedule, liveData box scores, playerIndex.json
  for rosters and two-way flags), ak-static injury PDFs, ESPN. stats.nba.com HANGS from
  Actions and is local-only (backfill, bio). Daily box scores come
  from the cdn liveData box score (one call per game: every counting stat, status,
  starter, seconds, Final flag); stats.nba.com is used for the backfill and slow data
  (playerindex, bio). Fallbacks in order: cdn from Actions; Barrett's Mac as the single
  writer (docs/runbook.md section 5, `scripts/commit_state.sh --local`);
  a visible "data through" stamp. (ESPN summary box scores were dropped from this list
  on 2026-10-09: no parser exists, and the Mac fallback covers the same outage.)
- **State**: player state keyed by PLAYER (season and team boundary rules written once,
  shared); one context function f(state, target game, team schedule, horizon inputs)
  called by training AND serving. A committed end-of-2025-26 state seed plus a
  latest-state snapshot (~600 rows), never the full per-game state history.
- **Completeness**: each team's game sequence comes from the INGESTED participation
  table (never the schedule: an un-ingested game would read as an absence). Before
  serving date D, every Final game of both teams before D must be ingested, else a
  stale-state badge and no freeze. Tomorrow's projections are provisional and never
  freezable.
- **No post-tip leak**: inference for date D uses state < D and context as of
  min(now, tip). Projections FREEZE PER GAME AT TIP: the bot never writes a tipped
  game; the app stops recomputing at tip and shows the frozen pre-tip version. The
  record = first published + last pre-tip snapshot, each stamped; scored per game
  when Final. In-session recomputes (mark-outs) are never scored.
- **App live layer**: render committed projections first, then upgrade with the
  latest injury PDF (floor(now ET, 15 min), step back <= 8 slots) and ESPN odds;
  budget ~5s, 0 retries, failures cached with their own TTL (circuit breaker); the odds
  feed's walk runs on a background pool, started by a refresher thread or a page view,
  and a page waits at most 3 s for it; what is not done by then lands in memory for the
  next run. The app never calls stats.nba.com. The app writes nothing to disk
  (docs/serving_design.md section 0); there is no LIVE_DIR.
- **Bot hygiene**: dense crons at off-minutes (e.g. :17/:43), cadence logic in Python;
  `if_changed=True` on every committed write (byte-stable when unchanged); append-only
  per-date partitions immutable once Final; an allow-list of bot-owned paths, one
  writer per path; pytest + an AppTest smoke render before committing; push with
  `pull --rebase` and retries; the run goes RED on any Blocked source, on logs trailing
  the last Final game by more than a game-day, or on a date with Finals yielding 0 rows;
  a dead-man's switch for runs that never start; an offseason keepalive (scheduled
  workflows in a public repo disable after 60 days idle). The bot never calls the odds
  vendor, and no workflow holds a vendor secret. Opening-night runbook:
  manual runs Oct 19 evening and Oct 20 morning (roster cut-down). BOS@DET tips at
  3pm ET, so the first freeze point is 2:55pm ET Oct 20 (tip minus
  `pipeline.FREEZE_LEAD`). The Oct 20 run must finish well before then: a game past
  its freeze point is never written for the first time, and the Oct 19 run's rows for
  Oct 20 are provisional, which never freeze, so a late run leaves BOS@DET with no
  frozen projection.
- **Memory**: models are 6-13 MB; simulation capped per game.

## 5. Site

- **Pastes and mark-outs survive navigation**: every internal link is a full page load
  (a new Streamlit session), so a visitor's board lives SERVER-SIDE in an in-process
  LRU store keyed by an unguessable `?board=` token carried by every minted URL (nav,
  cards, back links, share); mark-outs live in the URL as `?out=pid,pid`. Never on
  the server's disk, never frozen. Test: paste on `/`, follow a card, the /Game line
  count equals the card pill.
- **A visitor's board survives a reload, a bare link and a restart** (Barrett,
  2026-11-05: "I have to reenter them every time I refresh"): after THEIR paste, the
  page keeps a copy in THEIR browser (localStorage, props_ui.board_keeper) and, when
  a page opens with no live board, brings it back under a fresh token
  (props.restore_saved: the copy is a second paste box, re-validated and re-matched;
  lines for games already played are dropped). Never on the server's disk; a board
  opened from someone else's link is never copied; Clear all removes the copy.
- **Feed lines need no token** (Barrett, 2026-11-09): the odds feed's lines are one set
  per date for every visitor, held in process memory, and the board built from them needs
  no `?board=`. A visitor's token board and browser copy hold only what that visitor
  pasted; feed rows are never copied into either. Where both have a PrizePicks line for
  the same player and stat, the newer line is shown on that visitor's view (a typed pick,
  a Demon or a Goblin from the paste always is), and the board says which and how many.
- **A typed list pastes too** (Barrett, 2026-11-06: "can you incorporate these into
  the script"): one pick per line, as typed or copied from a picks post ("Take the
  OVER on Miles Bridges for a combined 6.5 Rebounds and Assists", "Oso Ighodaro o7.5
  P+R", "7+ R+A (No)", DiamondValue's "Name, Stat, Line"). props.parse_typed_lines
  reads the whole paste when the board parser finds no square naming a stat; in a
  board paste it adds only the typed lines that name a side, and counts the ones it
  could not read. The Over or Under written is the PICK, shown beside the lean
  ("Pick: More") and never used to price or rank; the sides offered stay unknown. A
  stat is priced as written or not at all: a note after it ("LOCK") is read only on a
  line with a side, only when it names no stat ("points, rebounds and assists
  combined" is PRA) and when no qualifier or note word opens the note, filler words
  aside ("rebounds offensive", "rebounds, only offensive", "points in the paint" are
  refused, "coming off a big game" is a note), and a word that changes the stat
  among a note's first four ("rebounds, but only offensive", "points, all from the
  paint", "threes, as attempts", "rebound chances", "all from three") refuses it
  too, but the matchup does not ("elite offensive rating"); a stat run across a comma with a
  note after it ("rebounds, assists are live too") is refused, two stats with only
  a comma between them are a combo only with "combined" after them, and a letter is
  a stat only where "+", "&" or "/" joins it ("R+A", "Pts + R + A", "Reb + A") or
  among capitals joined by "and" ("R and A"): "points - a lock" is Points and
  "points + a lock" is refused; a slash, pipe or bullet between stats joins them
  too ("Pts / R / A", "Pts | R | A"); a
  parenthesis joins the stat when every word in it changes the stat ("(offensive)",
  "(3PA)", "(OREBs)", "(offensive only)", "(+ assists)", "(w/ assists)", "(inc.
  assists)", "(along with assists)", "(assists included)"), and one that changes
  it among other words ("(offensive glass)", "(O boards)",
  "(off glass)") or adds another stat loosely ("(assists too)") refuses the line
  unless it speaks of the matchup ("(weak defensive team)"). A result word before
  the line or first in the note ("o25.5 pts - cashed", "- that one lost"),
  "cashed" anywhere after the stat, or a result alone after it or in a parenthesis
  ("- hit", "(W)", "(cashed it)") makes a recap, counted (typed_recap) and never
  filed; a recap naming another player ends the name line above (one naming the
  name line's own player keeps it); later in a note "won" or "lost" is context,
  and "it cashes tonight" is a pick, but without a word of hope "- this one
  cashes" is a recap; "- missed shootaround" and "- push it" are notes. A Yes
  never turns the pick, so the reader weighs a No (also "Nope", "Nah", "No way", "Not a chance", "(answer: No)"): it is
  the answer right after the pick's own "?" ("26+ 3-PT Made? No", "26+ points on
  Saturday? No way"), first after the line or last on it with closers only ("7+
  R+A No - fade", "26+ points No lol"), before the number of a "7+" line, or in a
  parenthesis that is only the answer ("(No 2u)"); one that starts a note ("no
  doubt", "no Towns", "NO TOWNS TONIGHT") or answers the note's own question ("is
  Towns playing? No") is the note's; anywhere else, above all beside a written Over
  ("No, I like Jalen Brunson over 25.5 points", "over 25.5 points, no for real",
  "(no, not a typo)"), the line is counted (typed_unclear) and never guessed.
  After the pick's own "?" other answers are read too ("Never", "Fat chance",
  "Yeah, no" are a No, "I like it" a Yes) and any other words there leave the side
  unclear; so do "no it is a lock", a correction ("No, wait, yes") and a No idiom
  beside a written side ("over 25.5 points - no way"), while "(no B2B)" is a note. A
  handle or hashtag that may stand for the name ("#KAT", "@Money23Green",
  "#Pickett") names nobody and never borrows the name above; a tag is decoration
  only when all of it is ("#NBA", "#PrizePicks", "@hoopspicks:") or it is a team's
  own hashtag ("#DubNation"), and "@Jalen Brunson" is his name; a tag that splits
  into a tout's words is decoration only with a bet's word in it ("#LockOfTheDay",
  "#LetsCash"), so "@King23" and "#Buckets" are nicknames. A pick outranks the board's line for the same
  player and stat, and a board line it sets aside at another number is reported,
  never counted as loaded; a pick and a square at the same number merge. Picks are
  never held or withheld as period lines, but a quarter square under a typed pick is
  held (and the message names the pick, since a Demon typed above the full-game line
  looks the same), and a pick lends its player (not its line) to the period-board
  check; Goblin and Demon rungs lend players too, and set the ratio (at half the
  bar) only when fewer than two standard players do. A period marker anywhere on a
  line ("(1H)", "Q1", "1st-half", "first qtr"), in a section label above it ("1Q
  props:", "1Q Locks", "## 1Q props"; in a board paste only a label ending with a
  colon or naming the picks), on a name line above it ("Jalen Brunson (1Q)", "Main
  Slate 1H props:", which label every pick under them, named or not, until a label
  naming none), or on its game heading refuses it, and the report says which picks
  a line above made period props (typed_period_label). Any spelling of a period
  counts ("1st 1/2", "Q-1", "q 1", "First Q", "1st period", "1qprops"); a name
  line's "full game" holds for that player and ends only a name line's period, and a
  team label under a period heading keeps the period unless it says "full game".
  Refusing is the safe side,
  so a line naming a period is a label unless the period is cancelled ("No 1Q props
  today", "Full game picks, 1H later:"; a negation reaches the period across no
  mark and must deny the period's picks, so "No more waiting, 1Q props:" and "No 1Q
  parlays:" are labels) or the line reads as a remark
  (a result: "What a 4th quarter!", "1H recap: 3-1"; or a remark's verb with a
  subject other than the period: "Q4 Brunson is different", while "1Q is the move"
  is a label; or "2H plays in the discord"); a heading's own period ("Magic @ Suns
  (1H)") holds over the game's own team headings, a loose pair of them and a
  sub-label naming one ("Magic players:"), and ends at any other. ESPN's "NO" beside a name or after a side word is New Orleans for a
  Pelican and a No for anyone else (alt_pick); after "7+", after a question, or on a
  line naming nobody it is the market's No; "vs NO" is the opponent. A team heading
  ("Suns:") only groups: the roster's team wins. A game is named by a game heading
  ("Magic @ Suns 7:00 PM MST", "ORL vs PHX", "Magic @ Suns on ESPN": a broadcast,
  a call sign or "| late game" is set aside), "vs Lakers"/"@ATL", or team headings
  whose latest game (tipped today or within 6 hours, or last night and before noon
  ET unless both play again today) or next game is against each other; two teams
  joined on one line by "&", "/", a dash or "x" ("ORL-PHX") are a game heading when
  their latest game (unless both play again today) or next game is against each
  other, routed as "@" is, and team headings otherwise. A named game that has
  tipped claims the line, so a list sent after its game is reported as late, never
  filed under each team's next game. The typed reader stops at 5,000 lines or
  40,000 words per paste (MAX_TYPED_LINES, MAX_TYPED_WORDS; a board paste is about
  2 words a line): 5,000 lines of 40 junk words cost seconds, and the daily bot runs
  the whole suite, timing tests included, before every refresh.
- `/`: nav (wordmark, Home, About; right cluster Update lines, Share, theme toggle);
  masthead (NBA PROJECTIONS kicker; tiles: games, first tip ET, injury reports filed
  X/30 with as-of time); date bar (chevrons, date, Yesterday/Today/Tomorrow; the
  default view date is the ET date of now minus 4h so late West Coast games stay on
  screen); player search (`?player=`, sync-on-URL-change); cards grouped by tip time
  (team-colour split stripe, tip time, DraftKings spread/total or the sentinel, B2B
  chips, "Out: X, Y +2" for 15+ MPG players excluding G League/two-way, Upcoming /
  Live / Final, line-count pill top-left counting the board's SHOWN lines only (one
  per player and stat, and with the odds feed on one per player, stat and book));
  "Model vs the board": green confirmation, "N of M shown (hidden / thin / void)", top
  25 one row per player with "+N more", stat-family pills and a game filter inside an
  @st.fragment, "Show all", footnotes (Demon/Goblin and Reboot on PrizePicks rows
  only, break-even per source, board-at-its-ceiling); while the odds feed holds lines
  for the date, the board is a grid: one row per player and stat with a shown line
  (Player with the matchup under it, Stat, Model) and one column per book (DK, FD,
  MGM, PP, UD), each cell that book's line with the model's lean at it (a green up
  arrow for More or Over, a red down arrow for Less or Under, the number alone where
  the model has no lean or the line does not offer its side, the sentinel where the
  book has no line); rows rank by their strongest shown line (tier, then confidence,
  never the gap); on a phone the grid scrolls sideways inside its box with the Player
  column pinned; one legend line under it; and a note names the games a book has not
  posted yet (two at most, then "and N more").
- `/Game`: team-colour header (spread, total, implied team totals, rest/B2B, injury
  report as-of or "report not filed"); 5-chip strip; two team tables with
  team-tinted headers: status badge, player (headshot, projected-starter dot), P(play)
  only when listed or below 90%, **MIN** as the bold lead column, PTS REB AST 3PM STL
  BLK TOV PRA FS as mid-p medians under "if he plays"; starters by minutes then bench;
  deep bench collapsed; Out rows greyed at the bottom with the reason and no numbers;
  expandable rows with each player's lines (with the odds feed on, the same grid per
  player: one row per stat with a shown line, one column per book; sides offered,
  badges, void risk on PrizePicks lines) and, after tip, actuals from the cdn box
  score (120s cache; "Did not play, pick voided" on a PrizePicks line, "Did not play"
  on an Underdog or sportsbook line; a row holding a PrizePicks line and other books'
  lines names the books, "PP: Did not play, pick voided. DK, UD: Did not play"; an exact
  landing reads "landed on the line" (with the odds feed on, followed by the codes of
  the books whose line it was), never "push", and on a sportsbook line the
  chance-of-landing tooltip says the book's own rule decides it);
  per-team mark-out / mark-in (stable options) writing `?out=` with a "your
  adjustments, not the published projection" banner.
- Status badges: Out, Doubtful, Questionable, Probable, G League, Two-way, Not with
  team, Suspended; team-level "Report not filed" and "No report yet". Unknown is
  never rendered as healthy.
- `board_rows` returns (shown, hidden, thin, void); void = Out / inactive / marked
  out. Pills, ledger and rows all read it. Ledger balances per book and game: posted =
  matched + unmatched + ambiguous + refused + unprojected (props.ledger); the board's
  ledger (every book's lines for every game, plus the paste with each overlap counted
  once) balances too, with unmatched names listed.
- Ported from GridironValue props_ui (acceptance items): Demon/Goblin badge and
  footnote, hidden-side counts, "N of M lines" header and refusal reasons, GAP column
  not a percentage, green banner + toast after a paste, each line routed to its own
  game's ET date with a "lines saved for <date>, step the date" pointer, visitor
  Clear all (the link and this browser's copy), VPN warning, FEED_URL as a link only. GridironValue's
  FREEZE_KEYS comment is false (says it includes `line`); do not copy it.
- Preseason dates: "Preseason: no projections". Empty dates: "Next slate <date>".
  Cup knockout placeholders render as TBD.
- `/About` with the generated trust table, the break-even framing and the
  board-at-its-ceiling note. Accuracy is local only (Barrett).
- House rules: no em dashes, no emojis in UI text. Process: a re-sent screenshot
  means the last fix missed the point; the options offered are not the whole space.

## 6. Phase plan (STOP = check in with Barrett)

1. Foundation + source modules + full backfill (running) + GitHub Actions host probe
   (repo created private 2026-10-08).
2. **Slate viewer early** (product review): `/` and `/Game` from the schedule, ESPN
   odds, playerindex, headshots and live injury badges, with BASELINE numbers labelled
   as such. Barrett looks at the preseason and Oct 20 slates. Ask him for one real NBA
   PrizePicks paste (feed JSON + board text) as soon as one posts; turn it into
   fixtures.
3. Tables + horizon injury features + player state + context function; replay and
   future-mutation tests.
4. Models + gates (point, availability, coherence, probabilistic). **STOP**: gate
   results, what ships as a model vs a baseline.
5. Distributions, calibration, tiers, props layer (props.py + props_ui.py ports).
6. Bot (ingestion from the source the probe proved), per-game freeze at tip, record.
   **STOP**: deploy (flip repo public for Streamlit Cloud sharing), clone-and-run.
7. Full audit workflow before opening night (2026-10-20). If the gate is not done,
   open on labelled baselines only if Barrett agrees.
