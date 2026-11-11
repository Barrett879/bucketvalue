# Data notes (live-verified 2026-10-08)

Every fact here was observed live from this machine and re-probed by an
independent verifier. "Reputational" marks the few things that could not be
observed. Egress at the time was a VPN exit (AS62240 Clouvider, NYC), NOT a
residential IP and NOT the GitHub Actions / Streamlit Cloud ranges, so
reachability from those hosts is still UNVERIFIED and must be probed.

## stats.nba.com

- **Client gate.** python-requests and curl (with full browser headers) HANG to a
  read timeout: TLS completes, no byte arrives, no status. curl_cffi
  `impersonate="chrome"` AND nba_api's header set: 200 in 0.1-1.2s. Either half alone
  hangs. `bucketlib.http` does both. A hang is `Blocked`, never "no data".
- curl_cffi sends a None param as the string "None" -> HTTP 400 ->
  nba_api `KeyError('resultSet')`. Hits PlayerGameLogs/TeamGameLogs
  (OpponentTeamID, MeasureType). `http.stats_params` converts None to '' and
  refuses an implicit MeasureType.
- Responses are edge-cached (cache-control max-age 89-1800s; CommonTeamRoster can be
  served STALE for up to a day via Varnish grace).
- **Bulk logs, one call per season:**
  - LeagueGameLog P: 32 cols, MIN int (rounded; 99% = round(sec/60)), no starter, no
    DNPs. Keeps 0-minute cameo rows (77-118/season) that PlayerGameLogs sometimes
    drops (e.g. Schroder 0022200140). 2014-15..2025-26: 1230 games/season except
    2019-20 1059 and 2020-21 1080.
  - PlayerGameLogs Base (72 cols, precise float MIN, MIN_SEC 'MM:SS', PFD, BLKA,
    DD2, TD3), Advanced (USG_PCT, E_USG_PCT, PACE, E_PACE, POSS, PIE, TS_PCT,
    ratings; MIN rounded to 0.1), Usage/Misc/Scoring. 5-9s each. Advanced rates
    explode at tiny minutes (PACE up to 14400, USG 1.0): minutes-weight or clip.
  - AVAILABLE_FLAG equals VIDEO_AVAILABLE. It is NOT player availability.
  - LeagueGameLog T: MIN = team player-minutes, 240 + 25 per OT period.
    TeamGameLogs Advanced: MIN = game minutes (48-58), PACE/POSS per team-game.
- FANTASY_PTS (LGL) == PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV exactly
  (0 mismatches in 21 season files) == PrizePicks Fantasy Score.
- Date formats: LGL 'YYYY-MM-DD', PGL 'YYYY-MM-DDT00:00:00'; season keys LGL
  SEASON_ID '22024' (1 pre, 2 RS, 4 PO, 5 PI) vs PGL '2024-25'. Season types
  'Regular Season', 'PlayIn' (not 'Play In'), 'Playoffs', 'Pre Season'.
- 2026-27 Regular Season currently returns 200 + headers + 0 rows. That is the
  legitimate preseason state; assert status and headers separately from row count.
- **Box scores.** BoxScoreTraditionalV3 lists every DRESSED player incl. DNPs
  ('comment' "DNP - Coach's Decision", 'DND - Injury/Illness', 'NWT - ...';
  minutes '' for a DNP). Starters = first 5 per team in list order; `position` is
  filled for bench players before 2018-19. Inactives appear only in
  BoxScoreSummaryV3 `homeTeam.inactives`, with no reason. **V2 box/summary return
  silent empties for 2025-26 games.**
- LeagueDashPlayerStats(starter_bench='Starters', DateFrom=DateTo=d) returns that
  day's starters, but labels TEAM_ID with the player's LATER team. Join on
  PLAYER_ID + date only. LeagueGameFinder SILENTLY IGNORES the starter filter.
- ScheduleLeagueV2('2026-27'): opener 2026-10-20 (BOS@DET 3pm ET, PHI@NYK 7pm ET,
  OKC@SAS 9:30pm ET). Corrected 2026-10-09: this line said BOS@DET 7pm, but the
  stats and cdn schedules (gameDateTimeUTC 2026-10-20T19:00:00Z, '3:00 pm ET') and
  ESPN 401909088 all have it at 3pm; PHI@NYK is the 7pm game. Tips come from
  gameDateTimeUTC: gameDateTimeEst is the ET wall clock with a false 'Z'. The
  season's first freeze point follows from this tip (opening-night runbook, spec
  section 4).
  1206 of 1230 RS games listed (post-Cup games added later), 7 Cup knockout
  placeholders with team id 0, Cup final 0062600001 on 2026-12-11. gameDate is
  'MM/DD/YYYY 00:00:00': parse it (string min() returns January). Preseason includes
  the non-NBA 'LON'.
- The Cup final (prefix 006) is in neither Regular Season nor Playoffs logs, but
  both teams played it: rest/back-to-back features must take it from the schedule.
- playerindex (one call): every rostered player for a season with PERSON_ID, TEAM_ID,
  POSITION (null for 10), HEIGHT, DRAFT_*, SUPPLEMENTAL_STATUS (1 = two-way, at most 3
  per team, cross-checked against the movement log). Preseason = camp rosters
  (18-21/team). Past seasons return end-of-season state with off-roster rows
  (ROSTER_STATUS NaN): filter.
- CommonTeamRoster: BIRTH_DATE, AGE, EXP, HOW_ACQUIRED. HOW_ACQUIRED is the ORIGINAL
  acquisition, not the current contract: never a transaction date.
- nba_api's bundled static player table lacks 125 of 615 current players (every 2026
  rookie) and returns a retired Brandon Williams first. Take ids from playerindex.

## Reachability from the hosts that will run the site (MEASURED)

GitHub Actions probe, 2026-10-09 00:24 UTC, egress AS8075 Microsoft (San Jose),
`scripts/probe_hosts.py` via `.github/workflows/probe.yml`, results committed under
`docs/probes/`:

| Source | From Actions |
|---|---|
| stats.nba.com (LeagueGameLog, playerindex, CommonTeamRoster) | **BLOCKED**: hangs to timeout, even with curl_cffi Chrome + nba_api headers |
| cdn.nba.com schedule, todaysScoreboard, liveData box score, headshots | OK, 0.07-0.42s |
| cdn.nba.com `staticData/playerIndex.json` | OK, 0.4s, 135 KB (probed 2026-10-11 00:05 UTC, `docs/probes/probe_github-actions_20261011T000505+0000.json`; the first probe had not fetched it). IDENTICAL to stats `playerindex` for 2026-27 (615 players, same teams, same 83 two-way flags, checked from this Mac 2026-10-08) |
| ak-static injury PDFs | OK |
| ESPN scoreboard, core odds, injuries | OK |

So the bot runs on cdn + ak-static + ESPN only. stats.nba.com is a LOCAL-ONLY
source (this Mac) for the historical backfill and bio. Streamlit Cloud has not been
probed yet (do it at deploy; the app never calls stats.nba.com anyway).
A liveData box score for a game that has not tipped is an S3 403 (Absent): rosters
before a team's first game come from playerIndex.json.

## cdn.nba.com

- `/static/json/*` needs a Chrome fingerprint AND `Referer: https://www.nba.com/`.
  Without the Referer: Akamai HTML 403 (431-445 bytes, errors.edgesuite.net) that
  hides whether the path exists. A missing object returns an S3 XML 403
  (`<Code>AccessDenied</Code>`). Classify by body.
- liveData `boxscore_{gid}.json` exists from **2019-20** (older ids: S3 403). Per
  player: status ACTIVE/INACTIVE, starter '1'/'0', played '1'/'0', notPlayingReason
  (INACTIVE_INJURY, DNP_INJURY, DND_INJURY, INACTIVE_HEALTH_AND_SAFETY_PROTOCOLS,
  ...), notPlayingDescription, minutes 'PT25M01.00S'. One call per game.
- **From 2026-27 the liveData box lists ACTIVE players only** (found by the live
  rehearsal of 2026-11-02, never by a test: every fixture and the parity replay are
  2025-26 payloads). Through 2025-26 a box listed every rostered player of both
  teams, 15-22 a team (median 18), the ones who did not dress as INACTIVE with a
  notPlayingReason (11,106 inactive rows in the 1,236 regular-season and play-in
  games of 2025-26). The 92 boxes of 2026-10-20..11-01 list only the players who
  dressed: 11-16 a team (median 13), statuses played 2,079 / dnp 414 / inactive
  0, no INACTIVE row and no inactive field anywhere; game, team and player keys
  are unchanged. Kept as `tests/fixtures/box/cdn_0022600005.json` (GSW at LAL,
  2026-10-21): 13 a side, against 18 per team in the playerIndex. The bot rebuilds
  the inactive list from its roster snapshots (docs/serving_design.md section 9).
- **How to tell the formats apart**: no INACTIVE player on EITHER side. Every
  2025-26 regular-season and play-in box lists one on at least one side; three
  team-games listed none (BOS twice in the regular season and CHA in the
  play-in, 15 dressed each) while the opponent listed its own; the 14 games with
  none on either side were all playoffs (no two-way players).
- **Two-way players after the regular season** (ineligible for the play-in and
  the playoffs; 2019-20 and 2020-21 let them in, 16-17 listed a team): the
  listing mostly leaves them out, not always. 2021-22..2025-26: 38 of 60
  play-in team-games and 824 of 844 playoff team-games listed 15 or fewer; the
  other play-in boxes listed their two-way players inactive
  (INACTIVE_GLEAGUE_TWOWAY, INACTIVE_INELIGIBLE_TO_PLAY, INACTIVE_COACH, no
  reason) or even active (ORL's two on 2026-04-15). In 2025-26's play-in four
  games listed 15 a side, without the three (ORL two) two-way players each
  team's 2026-04-12 box had listed; 0052500111 listed MIA's and CHA's five as
  inactive. The rebuild leaves them out (box.TWO_WAY_INELIGIBLE).
- **The report and the inactive list** (2023-24..2025-26, t_15 report, 33,018
  inactive rows): 89.5% listed Out, 1.0% listed with another status, 9.5% not
  listed (two-way and assignment players, healthy inactives). So the report alone
  cannot rebuild the list; the roster minus the box can.
- **The old listing was close to the roster, not equal to it.** A few rostered
  players were missing from 2025-26 boxes altogether: HOU's Finney-Smith and Tate
  on opening night (both listed Out by HOU's report), ATL's Djurisic in about 40
  games. 86 of 13,993 t_15 report rows resolve to nobody in their game's listing,
  and 112 team-player-games fall inside a stint (listed before and after, not in
  that game). Training has no row for those; a rebuild from a true roster does.
- scheduleLeagueV2(_1).json, todaysScoreboard_00.json, odds_todaysGames.json all 200
  with the Referer.
- Headshots `/headshots/nba/latest/{260x190|1040x760}/{pid}.png` and logos
  `/logos/nba/{team_id}/global/L/logo.svg` work without a Referer. Unknown headshot
  ids return 200 with a silhouette (4937 bytes at 260x190), never a 404.

## Official injury reports (ak-static.cms.nba.com/referee/injury/)

- Downloadable from datacenter IPs (S3 behind an Akamai 7-day edge cache).
- A missing file is an S3 XML 403 for: not published yet, no games, wrong filename
  format, future time. `curl -o` saves the XML as .pdf. Check `%PDF`.
- Archive starts **2018-12-17_05PM**. Filename eras (ET wall clock, 12-hour, 12AM/12PM):
  - to ~2021-10-14: three a day, 01PM/05PM/08PM; 2020 bubble restart: 11AM/02PM/05PM
  - ~2021-10-15 to 2025-12-22 12AM: hourly `_HHAM|PM`
  - from 2025-12-22 (by 10AM): every 15 minutes `_HH_MMAM|PM`
  - DST: fall-back keeps one slot per name; spring-forward slots do not exist.
- **The filename time is not the report time.** In the hourly and 3/day eras the
  file named HH holds the HH:30 (sometimes HH:45) report. Pick "last report before
  tip" by the PDF TITLE time; picking by filename leaks post-tip information.
- Layouts: 2018-19 adds Category and Previous Status columns; 2019-20..2023-04
  (iTextSharp) one row per line; GemBox (by 2023-10, and already in use by
  2025-12-21 under the hourly filename) needs positional parsing (pypdf
  visitor_text), with wrapped reasons centred on the player line, carry-over across
  pages, team forward-fill, hyphen-wrap artefacts ('Two- Way').
- Reasons separate injury from G League two-way / on assignment (about 30% of Out
  rows in-season), Not With Team, Personal, Suspension, Coach's Decision,
  protocols, Trade Pending, Reconditioning. Statuses: Out, Doubtful, Questionable,
  Probable, Available.
- The 5 PM report includes the NEXT day's games: key by Game Date + matchup.
- 'NOT YET SUBMITTED' means the team has not filed: a missing player row then means
  "unknown", not "healthy".
- **QA/test PDFs are published to production** (2026-10-01_12_45PM,
  2026-10-07_09_45AM/10_45AM list games that do not exist, with reasons like
  'Rest - asdfe') and Summer League reports share the bucket. Validate every row
  against the real schedule.
- Preseason coverage is near zero; regular-season coverage starts the day before
  opening night. "No report" means unknown, never "everyone available".
- Deadlines (official.nba.com): status by 5pm local the day before; 2nd game of a
  back-to-back by 1pm local on game day; game-day update 11am-1pm local; lineups due
  30 minutes before tip.

## ESPN

- Scoreboard `?dates=YYYYMMDD` (US-local date param; event dates are UTC). Month form
  `YYYYMM` silently caps at 100 without `limit=`; range form `A-B` is HTTP 400.
- Upcoming-game odds: DraftKings (provider 100), home-side spread (negative = home
  favoured), open and "close" (before tip "close" = current). Lines post gradually
  (28 of 88 October games priced on 10-08; some national games 7 weeks out). Odds
  DISAPPEAR from the scoreboard at tip.
- **Core odds** `sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{id}/competitions/{id}/odds`
  has pregame lines for every FINAL game sampled 2014-15..2025-26 (111/111 on full
  slates) and keeps the pregame close after the final. Eras: 2014-17 offshore books +
  'Opening' (23) + initialSpread; 2017-18..2022-23 current only (no open); 2022-23
  ESPN BET (58) `close{}` holds PRICES in the line slot; 2023-24 open/close/current
  per book; 2024-25 ESPN BET only; DraftKings (id 40 earlier, **100 from 2025-11-25**,
  DK-only from 2025-12-01). Live providers (44, 46, 59, 200 'DraftKings - Live Odds')
  sit beside pregame ones: use an allow-list. Garbage exists (Caesars 31 reporting
  half totals ~58, sign flips near pick'em, stale books): bounds + cross-book median.
- Summary `pickcenter` is EMPTY 2023-24 .. 2025-11-25 although core has the lines.
- Summary `injuries` for a HISTORICAL game is TODAY's injury list. Never a training
  source. ESPN has no injury history.
- Summary box scores: whole-minute strings, 10 starters, `didNotPlay` + `reason`.
  12 FINAL games 2015-16..2017-18 (all CHI or NO) are zeroed (every stat 0, '--').
  `active` means on the floor at the buzzer, not dressed.
- Injuries endpoint: group by the ENCLOSING team (athlete.team is stale after
  trades); athlete id only in `links` href; status lags text (Day-To-Day players
  "ruled out" in comments; use fantasyStatus + returnDate too).
- No ESPN<->NBA crosswalk exists: games by (ET date, home, away); players by
  normalized name + date of birth + team. ESPN abbreviations GS NO NY SA UTAH WSH.
- All-Star games are tagged regular-season with fake team ids; Cup final has comp
  type 'CC'; postponed games keep their ids (STATUS_POSTPONED) and may carry odds.

## Historical lines: provenance decision

Kaggle `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` (CC0 label,
nba_2008-2026.csv, 15,479 games 2014-15..2025-26, sha256 729eb2ea...a8c4) is built
from SportsbookReviewsOnline (whose terms forbid reuse) to 2023-01-16, ESPN BET close
to 2025-06, then an SBR median. It matches SBRO exactly where they overlap. Its
2025-26 rows disagree with DraftKings on several games (one favourite flip). Only the
current version downloads (v3 404s). **Decision: training lines are rebuilt from
ESPN's core API (the same family as the serve-time source); the Kaggle file is used
only as a local cross-check under `cache/local_only/`. No historical line table is
committed.**

## PrizePicks (first-party pages, read 2026-10-08; never call their API)

- Fantasy Score = PTS 1, REB 1.2, AST 1.5, BLK 3, STL 3, TOV -1; no 3PT bonus.
- Overtime counts in full-game, 2H and 4Q projections.
- DNP = did not participate for a single second; pick removed, payout tier drops.
- NBA Reboot: MORE on a full-game projection is voided if the player leaves in the 1H
  and does not return, unless he already cleared the line. Period boards excluded.
- Ties ("down to the decimal") are neither win nor loss; whole-number NBA lines exist
  (e.g. 'Siakam 7 Rebounds', 2026-01-12).
- Settlement uses partner feeds (SportRadar, Genius, Stats Perform, Grid) "as
  originally posted"; stats.nba.com is a proxy for the settled number.
- Demons/Goblins Less picks: announced for MLB/WNBA (2026-08-21) then NFL (2026-09-04);
  no first-party page names the NBA; the NBA scoring article still says More-only.
  Resolve direction per row from the paste.
- Stat labels confirmed: Points, Rebounds, Assists, 3-PT Made, Blocked Shots, Steals,
  Turnovers, Pts+Rebs, Pts+Asts, Rebs+Asts, Blks+Stls, Pts+Rebs+Asts, Fantasy Score;
  announced 2025-04: Free Throws Made, FT Attempted, Two Pointers Made/Attempted,
  Personal Fouls, '... in First 5 Minutes'. Others unverified until a real paste.

## The Odds API (read 2026-11-09, probed 2026-11-11; display-only player lines)

Two kinds of fact here. "Verified by the probe on 2026-11-11" means
scripts/probe_oddsapi.py saw it with Barrett's key from this Mac, at about 10:15
ET on a 10-game day (10 PASS, 0 FAIL, 14 credits spent; its output holds no
key); the bracketed numbers are the probe's own fact numbers. Every other fact
is from the vendor's own pages, read 2026-11-09, and is UNVERIFIED where marked.

- **The calls.** Sport key basketball_nba. Props come only from
  /v4/sports/basketball_nba/events/{id}/odds, one game per call, with the
  books named in `bookmakers`. The key travels only as the apiKey query
  parameter; nothing documents header auth. /events is free (x-requests-last
  0), lists games in commence_time order and lists other dates' games too
  (14 events, 10 of them that day's): verified by the probe on 2026-11-11
  [12a]. The site keeps only today's ET games. That /events also lists
  games in progress is the vendor's docs, UNVERIFIED. /participants costs 1
  credit (verified by the probe on 2026-11-11).
- **Cost.** Cost = unique markets returned x regions; up to 10 books named in
  `bookmakers` count as 1 region, and `bookmakers` wins over `regions` (the
  vendor's docs). Verified by the probe on 2026-11-11: points for the five
  books, named with no `regions`, cost 1 credit per game call in all 7 games
  read [3a], and adding regions=us did not change the bill [3b]. The schema
  marks `regions` required, but a bookmakers-only call is accepted (verified),
  so `oddsapi.SEND_REGIONS` stays False. An empty answer is free (the vendor's
  docs; UNVERIFIED: the probe's free PrizePicks-only check was skipped, since
  PrizePicks had posted every game it read).
- **The free plan** (500 credits a month) serves all five books, PrizePicks
  and Underdog included, with the books named and no regions: verified by the
  probe on 2026-11-11 [2].
- **Headers and errors.** A 200, a 422 and a 404 carry x-requests-remaining,
  x-requests-used and x-requests-last, and a 401 INVALID_KEY carries none:
  verified by the probe on 2026-11-11 [5a, 5b, 5c, 5d]. A 422 INVALID_MARKET
  (an unknown market) and a 404 EVENT_NOT_FOUND (an unknown event) cost 0
  (verified); the 422's body held the fields details_url, error_code and
  message [5b]. A made-up key gets 401 INVALID_KEY (verified [5d]). 401 is a
  bad key OR an exhausted account, and the body's code says which
  (OUT_OF_USAGE_CREDITS); an exhausted account's 401, its headers, and
  whether /events still answers it are UNVERIFIED. An unknown book key is
  dropped without a word: no 422, and the known book still answers, for 1
  credit (verified [4]). 429 limit: 30 calls a second on paid plans (the
  vendor's docs); the free plan's is not stated, and a 429's headers are
  UNVERIFIED.
- **Credits reset** on the 1st of every month (the FAQ, for every plan; the
  hour and time zone are not stated, UNVERIFIED); paid plans are BILLED on the
  sign-up day, which is not a reset.
- **What the books post.** By 10:15 ET on 2026-11-11 all five books had points
  in all 7 games read, at 11 to 15 lines a game on DraftKings and FanDuel, 9 to
  13 on BetMGM, 11 to 17 on PrizePicks and 5 to 10 on Underdog, and every line
  had both sides (verified by the probe [1, 2, 10a]). What the books have
  posted by the 08:50 ET first pull is UNVERIFIED: the probe ran at 10:15.
  Caesars and Fanatics are paid-plan books and not among the five.
- **Timestamps.** Market last_update is when the vendor last READ the market,
  not when the line moved: verified by the probe on 2026-11-11, when within 2
  minutes DraftKings, FanDuel, PrizePicks and Underdog re-stamped every
  market's last_update with no line change, and BetMGM re-stamped none [8]. No
  book carried a bookmaker-level last_update (verified [13]; the vendor calls
  that stamp deprecated). So the site dates each book's lines by its own fetch
  time (fetched_at), never by last_update. That last_update stops advancing on
  suspension and the market drops about 15 minutes later is the vendor's docs,
  UNVERIFIED.
- **DFS prices** (prizepicks, underdog) are indicative only (the vendor's
  docs). They were -137 on every standard line the probe read on 2026-11-11
  (verified [6a]); the parser blanks them anyway, and no price is shown for any
  book. PrizePicks Demons and Goblins sit only in _alternate markets (Goblins
  at the default price, Demons at +100, the vendor's docs); the feed does not
  buy them.
- **Teams and players.** Team names are full names ("Detroit Pistons"), and
  all 30 map through teams.canonical (verified by the probe on 2026-11-11
  [9a]; the participants list also carried two names outside the 30 teams).
  An event naming a team teams.canonical does not know is skipped and logged.
  All 10 of the day's events matched the committed schedule, with no home and
  away swaps (verified). Neutral-site games may list home and away the other
  way round (UNVERIFIED: no neutral-site game that day), so events match on the
  date and the unordered pair. Players are in outcome.description, and every
  player name of every book matched in its own game in all 7 games (verified
  [9b]).
- **Parameters.** commenceTimeFrom is ignored on event odds: lines came back,
  for 1 credit (verified by the probe on 2026-11-11 [14]).
- **Still UNVERIFIED:** what the books post by 08:50 ET; the reset hour
  (`--balance` around a month's end, and on a paid plan around its sign-up
  day); the free plan's rate limit; a 401 OUT_OF_USAGE_CREDITS and a 429;
  whether an empty answer is free; whether /events lists games in progress, and
  whether its `status=pre` filter exists (the probe's `--evening`, while a game
  is on); how large one game's answer gets with 12 markets.

## Measured modelling facts (2023-24..2025-26 unless noted)

- Knowing actual minutes cuts squared error vs a season mean by PTS 31.6%, REB
  28.4%, AST 17.3%, 3PM 13.9%, PRA 45.4%, Fantasy 45.3%, but STL/BLK/TOV only ~6%.
- Minutes memory is short (EWMA half-life 2 best: RMSE 6.44 vs 7.08 season-to-date);
  scoring memory longer (half-life 5-10).
- Back-to-back works through AVAILABILITY (+6pp absence for 28+ MPG players); stars
  who play lose ~0.2 min. Pooled across all players B2B minutes go UP (bench covers):
  a role interaction or the model learns the wrong sign for starters.
- Pre-game expected margin moves the minutes MEAN ~1 min (12+ point spots) and mostly
  fattens the low tail; the FINAL margin moves it ~6 min (a post-game leak trap).
- Expected pace corr 0.54 with actual possessions; Fantasy elasticity ~0.6.
- Top shot-taker out: each of the next four teammates +~2 FGA, +~2.5 PTS (confounded
  with rest days and shutdowns).
- Return from a 10-19 game absence: about -6 / -3 / -2 minutes over games 1/2/3 back.
- Rookies: first-10-game MPG predicts the rest with RMSE 6.8 (veterans ~3.8).
- OT in 4.4-6.5% of games; +6-8 min for 32+ MPG players, +1.5-4 for <28 MPG.
- Dispersion given minutes (var/mean): PTS ~2.2, FTM ~1.8, FTA ~2.0, PRA ~1.9,
  Fantasy ~2.4; REB/AST/3PM/STL/BLK/TOV ~1.0; PF ~0.8.
- 1st-half-only exits (Reboot mass): 3.8% of played games overall, 0.8% for 28+ MPG,
  ~7% for <20 MPG, 20% for games under 10 minutes.
