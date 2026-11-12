# BucketValue runbook: launch and operations

The operator's runbook for the bot (`.github/workflows/daily.yml` ->
`scripts/daily_update.py` -> `scripts/commit_state.sh`) and the Streamlit app
(`app.py`, `pages/`). Rewritten 2026-11-03 for a mid-season launch, with every
claim re-checked against the code at that date. The opening-night version is in
git under its old name: `git show dad9bb9:docs/runbook_opening_night.md`, first
committed in fb9bd55. Spec section 4. Section 8, the licensed odds feed, was
added on 2026-11-11 and checked against the code of that date.

## 0. Where things stand (2026-11-03)

Done:
- **The Actions probes.** The probe of 2026-10-11 reached every source the bot
  read then, the playerIndex included (`docs/probes/`, `docs/known_issues.md`).
  ESPN's team-roster endpoint is read since 2026-11-02 by the in-season roster
  check. It sits on the probed host (site.api.espn.com) but was in no probe;
  a failure there is a WARN only.
- **The bot commits from Actions.** It ran on a private-repo cron of four runs
  a day from 2026-11-02 (bot commits 8ece28a, ebc144b, 8f8a126, 0693e13,
  9a1f01a), and on the full every-30-minutes cron since 2026-11-04. Each ET day
  with at least one run that pulls a good roster and commits saves that day's
  roster snapshot, which a 2026-27 box needs because it lists active players
  only (`docs/known_issues.md`).
- **The repo is public** since 2026-11-04 (Barrett's call). Its Actions runs
  can be read without a token: `https://api.github.com/repos/Barrett879/bucketvalue/actions/runs`.
- **`tests.yml` is green on GitHub** since dad9bb9, through 055d752 and
  6d32676 (Actions runs 6 to 8).
- **The team-minutes step** (models.cohere, a model_version ending in "+tm1")
  is in every projection computed since 055d752.
- **The app is deployed** on Streamlit Community Cloud at
  bucketvalue.streamlit.app (after this list's date; section 3, step 6). The
  licensed odds feed (section 8, added 2026-11-11) runs only once its key is in
  the app's secrets.

Not done:
- **The healthcheck** (`BV_HEALTHCHECK_URL`) is not set.

GitHub starts cron runs late and sometimes not at all. On the private cron,
2026-11-02 to 11-04, it started 5 runs from 10 slots, all successful, each
created 42 minutes to 2 hours 50 minutes after the nearest earlier slot. On 2026-11-03 no bot commit
landed between 04:01 and 21:04 UTC (Barrett's 055d752 at 09:18 UTC was the
only commit), so the bot's first commit of that ET day came at 4:04 PM. `scripts/daily_update.py` cites 2 to 7
hours of lag measured on DiamondValue. The first line of the Bot step's log
(`BucketValue daily update: ET date D, run at ...`) gives each run's real start.

## 1. The dead-man's switch

A missed slot alerts nobody until this exists, and a day without a good run
loses that day's roster snapshot for good:

1. On healthchecks.io (the free tier is enough), create a check with schedule
   type **Cron**, expression **`17,47 13-23,0-6 * * *`**, time zone **UTC**, and
   a grace time of **3 hours**. That is the full cron in `daily.yml`, on since
   2026-11-04. If the repo goes private again, it is `17 2,15,20,23 * * *` with
   an 8-hour grace instead (section 6): those runs are 3 to 13 hours apart.
2. Turn on alerts by email, and phone or push if wanted.
3. Copy the check's ping URL into the repo secret **`BV_HEALTHCHECK_URL`**
   (Settings, Secrets and variables, Actions).

The workflow's last step (`if: always()`) pings it on every run that starts:
success, or `/fail` for a RED or failed run. A run that never starts sends
nothing, and the missing ping is the alert. Until the secret exists, each run
prints a warning instead.

## 2. Before the flip: what becomes public

Making the repo public publishes its whole history. That is Barrett's call,
and these are the things to decide on first:

- **Every commit Barrett makes carries his personal address, as author and as
  committer.** That is 38 of the 43 commits before this file
  (`git log --all --format='%ae %ce' | sort | uniq -c`); the bot's carry
  bucketvalue-bot@users.noreply.github.com.
- **Real betting lines.** The spec says historical lines are never committed,
  but two committed things hold real lines:
  - The ESPN odds test fixture `tests/fixtures/espn/odds_events.json.gz`
    (bb70d2c) holds real spreads and totals of 21 games, 2015-16..2025-26.
  - The bot's 2026-27 captures, `state/season/2026-27/odds_*.parquet`, which
    the app reads.

  Decide on the fixture before the flip. The large line archives
  (`cache/game_lines_*`, `cache/local_only/`) were never committed.
- **The spec, every doc and the other test fixtures:**
  - injury-report PDFs;
  - cdn and ESPN payloads;
  - constructed PrizePicks fixtures, which are not real boards
    (`tests/fixtures/props/PROVENANCE.txt`);
  - constructed odds-feed fixtures with made-up lines, never a captured vendor
    answer (`tests/fixtures/oddsapi/PROVENANCE.txt`).
- **Actions run logs, private-era runs included.** They carry the RED lists and
  the real start times. GitHub's dialog for changing visibility warns that
  Actions history becomes visible; that is platform behaviour, not checkable
  here. Delete old runs first if that matters.

What stays private:
- **cache/** (gitignored, and nothing under it except `cache/.gitkeep` was ever
  committed);
- **the accuracy record**: `pages/Accuracy.py`, `cache/accuracy_summary_*` and
  `cache/props_record_*` are gitignored and never in the nav.

## 3. Launch, in this order

The order is forced. The full cron spends a private repo's free minutes within
about a week. The healthcheck must match whichever cron is on. And the public
site should not show games frozen hours before tip.

1. **Pull** the development tree (`git pull --ff-only origin main`): the bot
   pushes state/ several times a day.
2. **Clone-and-run** (spec lesson 19), outside the iCloud Desktop:
   ```
   git clone https://github.com/Barrett879/bucketvalue ~/bucketvalue-check
   cd ~/bucketvalue-check
   /opt/anaconda3/bin/python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/python -m pytest -q
   .venv/bin/streamlit run app.py
   ```
   - **The suite:** the same command the workflow runs. It must show 0 failed;
     the local-cache tests skip in a clone.
   - **The app:** check `/`, one `/Game?date=...&away=...&home=...` page of the
     newest committed date, and `/About`, against the list in step 7.
   - **What the render tests cover:** `tests/test_pages_smoke.py` renders the
     newest committed slates and every game page from a checkout; the workflow
     reruns it before each commit.
3. **Flip the repo public** (Barrett). Done 2026-11-04.
4. **Swap the cron** (done 2026-11-04). In `daily.yml`, replace the line
   `- cron: "17 2,15,20,23 * * *"` with the two commented full-schedule lines
   (`17,47 13-23 * * *` and `17,47 0-6 * * *`), push, then **dispatch one run by
   hand** (Actions, daily-update, Run workflow, date input EMPTY).
   - **Why dispatch:** a push never triggers daily.yml (it has only `schedule`
     and `workflow_dispatch`), and the new crons take effect on their next slot.
   - **The date input:** never type a date to "run tomorrow early". A future
     date is refused (exit 2), and a plain run already projects tomorrow as
     provisional.
5. **Switch the healthcheck in the same sitting**, if it was set on the private
   cron: expression `17,47 13-23,0-6 * * *`, grace **3 hours**. The full
   expression on the private cron alarms within hours, and an 8-hour grace on a
   30-minute cron hides an outage.
6. **Deploy on Streamlit Community Cloud** (Barrett's account):
   - **Settings:** repository `Barrett879/bucketvalue`, branch `main`, main file
     `app.py` (About and Game come from `pages/`).
   - **Python:** choose **3.12** under Advanced settings, the version the bot
     runs (`runtime.txt`, which the workflow reads). The registry pins
     scikit-learn, and `models.load()` refuses a different one rather than serve
     wrong numbers; `requirements.txt` pins it. No `packages.txt` is needed.
     Keep `.streamlit/config.toml`: its `enableStaticServing` serves the two
     webfonts.
   - **Secrets:** add the root-level secret **`BUCKETVALUE_READONLY = "1"`**. The
     app reads no `st.secrets`. It reads these environment variables, which
     Streamlit fills from the root-level secrets:
     - `BUCKETVALUE_READONLY`, which declares the host shared. The backfill
       injury store then stays in memory, and the props record refuses to
       write.
     - `BUCKETVALUE_LOG`, optional, default WARNING. INFO adds the odds feed's
       spend and restart lines (section 8). It is read when the process starts
       and must be exactly a level name (`DEBUG`, `INFO`, `WARNING`, `ERROR` or
       `CRITICAL`, in any case, no spaces): it has no fallback, and any other
       value makes the first page after a start fail with a `ValueError`
       (section 8, Settings). After setting it, reboot and check that the site
       loads.
     - `BUCKETVALUE_ODDS_API_KEY`, optional: the key for The Odds API (licensed
       player lines). Unset, the site is paste only. Added only by Barrett, in
       the app's Settings, Secrets; never in git, never in Actions, never sent to
       anyone, never in ~/.streamlit/secrets.toml. The other `BUCKETVALUE_ODDS_*`
       settings are in section 8. Every value must be a quoted string at root
       level: Streamlit copies only strings and numbers into the environment.
       After any secrets edit, press Reboot app (a reboot is a restart; section 8
       says when one costs credits).

   **Deploy no earlier than step 4.** On the private cron a game freezes on the
   last run before its tip minus 5 minutes. That is up to about 3 hours before
   an evening tip and up to about 5 before an afternoon one: a tip between
   about 10:30 AM and 3:30 PM EST freezes on the 10:17 AM run, such as the
   1:00 PM games of 11-15 and 11-21 and the 3:30 PM games of 11-08 and 11-22,
   unless the 3:17 PM run reaches its projections by 3:25. A late start freezes
   them earlier still, and the public site would show those numbers.
7. **Check the live app.** A healthy site shows these:
   - **`/` opens on the ET date of now minus 4 hours.** Its tiles read Games,
     First tip, Lines posted n/N and Injury reports filed n/N. The deck reads
     `Injury report: <time> (<date> report).`, and a note reads
     `Box scores through <Month> <day>.`, matching the bot's latest ingested
     finals.
   - **`Provisional.`** appears only on tomorrow's slate or on a game with a
     named gap. A game without both lines says
     `Projections post when the line does`.
   - **A game page** shows both teams' tables; after its tip, `Frozen at tip`
     and the box score.
   - **`/About`** shows the trust table.
   - **Untipped numbers are recomputed live,** so they can differ slightly from
     the committed file.

   **No notices**, and a notice that persists is a host problem to report, not
   an outage: every probe ran from GitHub's runners, never from Streamlit's
   hosts.
   - `Showing a saved schedule from N minutes ago.` is cdn. The page uses the
     newer of the process's last copy and the committed one, and projections
     and mark-outs still run on it.
   - `Game lines are unavailable right now.` is ESPN. The cards have no
     fallback, so untipped cards show no spread or total, and the next notice
     comes with it.
   - With the odds feed on (section 8), three notices are The Odds API's:
     - `Lines as of <time>. The odds feed did not answer just now, so the board
       shows the last lines it had.` A call failed; the board keeps the lines it
       had.
     - `Lines are unavailable right now. The odds feed did not answer.` The same,
       with no lines in memory.
     - `Lines as of <time>. The board is not refreshing these lines for now.`, or
       with no lines `No lines from the odds feed today.` Paid calls are stopped:
       the key was refused, the credits ran out, the reserve, the morning-pull
       floor or the day's spending ceiling held a pull, or three billed answers
       in a row could not be used (the day breaker). The log says which
       (section 8).

     Two are normal, not faults: the note `Lines as of <time>.` and, before a
     book posts a game, the note `No <Book> lines yet for <game>.`
   - `Showing the published projections. The live update could not run (<why>)`
     means the live recompute failed: the committed rows are served, without
     mark-outs. `<why>` names the cause (`bucketlib/store.py`, `_STALE_WHY`):
     - `the lines feed did not answer` is ESPN;
     - `the injury report could not be fetched` or `no injury report could be
       read` means ak-static failed AND no committed report was usable;
     - `the schedule could not be read` is the schedule or the committed state;
     - `this date's history features are not published` means the bot has not
       committed that date's features yet;
     - `the models could not be loaded` is a deploy fault, a wrong
       scikit-learn on the host;
     - `the live inputs look unlike anything the models were trained on` is a
       null skew;
     - `the published features lack a column the models need` is a
       train/serve mismatch;
     - `the live update hit an error` is a bug.
   - An ak-static outage alone shows no notice while the bot's committed report
     is in reach.
   - The game page adds `Showing the box score from N minutes ago.` or
     `The box score is unavailable right now.` (cdn liveData, no committed
     copy).

   Not visible on the site: the `model_version` column of
   `state/serve/<date>/projections_v1.parquet` says which models made a slate
   (2026-11-04: `v1-2026-10-09+tm1`; 2026-11-02: `v1-2026-10-09`).

## 4. Daily operations: reading a run

The Bot step's log ends with `RUN IS RED (n):` and the list. A RED run still
commits what it wrote (stale beats empty), then fails the job.

**Normal lines, not faults**
- **`finals: ... lists no inactive player; N rebuilt from the D roster ...`** on
  every 2026-27 box. These are the inactive rows rebuilt from roster
  snapshots.
- **The ESPN roster cross-check line** for a team over 18 rows (LAC lists 19
  since 2026-11-02). It is an info line when ESPN agrees, and a WARN naming
  who only one source lists when it does not. It is never RED, and nothing is
  dropped.
- **`WARN projections D: <game> froze on rows computed with ..., but the <T> ET
  report was committed before its freeze point`.** The frozen rows missed a
  report the bot already held. On the private cron this is expected (section
  3, step 6); on the full cron it means a projections step failed or did not
  run in time.
- **`WARN finals: N state game(s) before today not ingested yet ...`** after
  late tips. Their teams' next games stay provisional until ingested.
- **`WARN finals: ... (the schedule says Final; retried next run)`.** A box that
  is not final yet, under 6 hours after tip. After that it becomes the
  `still not final ... after tip` RED.
- **`WARN schedule: unrecognised postponedStatus ...`.** A postponedStatus value
  cdn does not know (only "Y", or the status text PPD or Postponed, counts as
  postponed). Check the feed: if those games were postponed, add the code to
  the positive check in `cdn._postponed` (`docs/known_issues.md`); if not, add
  it to `cdn.NOT_POSTPONED`. Never add it as a postponement unchecked.

**RED messages (Bot step)**

| Message | Meaning | Action |
|---|---|---|
| `schedule blocked: ...`, `roster blocked: ...` | A hang, an Akamai 403/503, or repeated 429/5xx on that pull | The committed copy stands in. If it repeats for hours, section 5 |
| `schedule unusable (...)`, `roster unusable (...)`, `schedule parsed empty` | A bad payload or a parser refusal | The committed copy stands in. Look at the payload if it repeats |
| `roster has only N players (under 400)` | A truncated playerIndex | The committed roster stands in, and no snapshot is written for that pull |
| `box score blocked at X` | The box host refused | That run's box fetches stop: the games wait for a later run, and their teams' next games stay provisional |
| `box score absent for final game X`, `box score fetch failed for X`, `parse failed for X` | That game only | Retried every run. If it persists, see "A stuck game" below |
| `box score of final game X is still not final ... after tip` | A cdn box frozen mid-game | See "A stuck game" below |
| `box score of X lists no inactive player and there is no roster snapshot ...` | The season has no snapshot at all on or after its first game day: impossible while `roster_2026-11-02.parquet` is committed | Not ingested; retried next run. A missed day never causes it: the game is rebuilt from the latest snapshot before it, else the earliest after (`state.roster_for_game`) |
| `inactive rows of X could not be rebuilt (...)` | The rebuild failed for that game | Not ingested; retried next run. Read the exception |
| `N state game(s) before today are not ingested 6.0h after tip` | A stuck feed or box | The games named keep their teams' next games provisional (never frozen) until fixed |
| `injury reports blocked: ...`, `the newest injury report could not be fetched` or `does not parse` | The ak-static host, or a new PDF layout | The committed report stands in while in reach (30 hours). A parse failure needs a parser fix |
| `injuries: ..., and the committed report is unusable` | No usable report at all | Projections for the date are refused (no report is a slate training never saw), and the committed rows stand |
| `odds D: <Error>` | The ESPN odds pull failed for D | The committed odds snapshots of D still feed the projections. A game with no spread and total in them is withheld, and one projected earlier keeps its committed rows. The other date is still tried |
| `projections D: N untipped game(s) without a whole roster (under 12 rows a side)` | A broken roster pull for those teams | The game is still projected and committed with the short side. models.cohere leaves a side under 12 rows as predicted, so no minutes move onto the rows present. Fix the roster pull |
| `projections D: nothing new written: ...` (NoReport) or `projections D: nothing new written (a null skew): ...` | No report in reach, or serve rows unlike training's | The committed projections are served |
| `projections D: <Error>: ...` | An unexpected exception projecting D | The other date still ran. A bug: read the traceback |
| `projections: no roster (live or committed); nothing projected` | Neither roster exists | Nothing projected |
| `<step> step raised <Error>: ...` | An unexpected exception in a step | The other steps still ran. A bug: read the traceback |

The short-roster and nothing-new-written rows are RED for today's slate only.
For tomorrow's slate the same text prints as `WARN projections <tomorrow>: ...`,
and the run stays green: every later run retries it. The roster rule
`projections D: the roster pull lists more than 18 players` fired only through
the first game day (camp rosters); in season the ESPN cross-check replaces it.

**Workflow-step failures.** Each fails the job and pings `/fail` once the
secret is set.
- **A red Tests step:** the Bot never runs.
- **A red App smoke step:** the run's writes are discarded (the Commit step has
  no `if: always()`).
- **`the bot changed files outside state/`:** the allow-list in
  `scripts/commit_state.sh`. A bug, and nothing is committed.
- **`state conflicts with a newer push`:** this run computed on a stale base.
  Nothing to do: the next run redoes it.
- **`could not push state after 3 attempts`:** in Actions the run's state commit
  is lost with the runner, and the next run recomputes it. On the Mac
  (`--local`) it stays a local commit: push it, or `git reset --hard
  origin/main`, before the next run.
- **A job over 40 minutes** is cancelled.

**A lost roster day.** If no run of an ET day both pulls a good roster and
commits, that day's snapshot is gone:
- **A RED run that commits keeps it.**
- **These lose it if they hit every run of the day:** a red Tests or App smoke
  step, a conflict, a failed push, or a roster RED (`roster blocked`, `roster
  unusable`, `under 400`).

Fix forward, and dispatch so the roster step runs before midnight ET.

**A stuck game (2026-27).** Do not follow the old override route (backfill X,
then `scripts/build_serving_seed.py --override X`) for this season:
- the auto backfill refuses a live-season box that is not final;
- `state.override_from_backfill` copies the backfill's participation without
  rebuilding inactive rows, so the game would be committed for good without
  its injured players, which the bot itself refuses to do.

The candidate route runs on the Mac (stats.nba.com) and is unverified: the V3
inactives are unchecked (`docs/known_issues.md`).
1. `.venv/bin/python scripts/build_participation.py --games X --source v3`
2. Confirm the V3 rows list the inactive players. BoxScoreSummaryV3 inactives
   carry no reason.
3. `.venv/bin/python scripts/build_logs.py --seasons 2026-27 --types RS`. The
   override also needs player_games and team_games.
4. Only then, `.venv/bin/python scripts/build_serving_seed.py --override X`.
   Pull, then commit and push `state/season/2026-27/override/`.

Waiting has a cost:
- **Every later run is RED** (`N state game(s) before today are not ingested
  6.0h after tip`). So every run pings `/fail`, and the healthcheck stays down,
  which hides any new failure.
- **Every remaining game of both teams this season stays provisional,** never
  frozen.

Decide within days.

## 5. Fallback: the Mac as the single writer

Spec 4's fallback order: cdn from Actions; this Mac as the single writer; the
visible "data through" stamp. Use the Mac when Actions cannot reach cdn for
hours, or Actions is down.

1. **One writer.** Disable the workflow first: Actions, daily-update, the "..."
   menu, Disable workflow. `gh` is not installed on this Mac; installed, it
   would be `gh workflow disable daily-update -R Barrett879/bucketvalue`.
2. **A dedicated clone,** never the development tree:
   `git clone https://github.com/Barrett879/bucketvalue ~/bucketvalue-writer`,
   then a `.venv` from `requirements.txt`.
   - This Mac has no Python 3.12, which is what the bot runs. Either install
     CPython 3.12.8, or knowingly use `/opt/anaconda3/bin/python3 -m venv .venv`
     (3.13.9), as the development tree does.
3. **Each run,** from that clone:
   ```
   git pull --rebase origin main
   .venv/bin/python scripts/daily_update.py
   .venv/bin/python -m pytest -q tests/test_pages_smoke.py && bash scripts/commit_state.sh --local main
   ```
   - **The smoke test gates the commit,** as it does in the workflow. Without
     it, a broken render could be committed.
   - **`commit_state.sh --local`** commits `state/` only and pushes with the
     workflow's pull-rebase rules.
   - **A conflicting rebase** is aborted, but the local state commit stays.
     Reset with `git reset --hard origin/main` before the next run.
4. **Nothing pings the healthcheck from the Mac.** Pause the check while the
   workflow is disabled, or ping it with `curl -fsS <ping URL>` after each run.
5. **To repeat it every 30 minutes**, a launchd agent
   (`~/Library/LaunchAgents/com.bucketvalue.writer.plist`) can run
   `/bin/bash -lc 'cd ~/bucketvalue-writer && git pull -q --rebase origin main && { .venv/bin/python scripts/daily_update.py; .venv/bin/python -m pytest -q tests/test_pages_smoke.py && bash scripts/commit_state.sh --local main; }'`
   with `StartInterval` 1800.
   - Nothing commits after a failed `cd` or pull, or a failed smoke test.
   - Give it `StandardOutPath` and `StandardErrorPath`, or its RED lists are
     never seen.
   - It does not run while the Mac sleeps.
   - Installing it is Barrett's call.
6. **To go back:** unload the agent, re-enable the workflow (the same menu,
   Enable workflow), and resume the healthcheck. Never both writers at once.

## 6. Rollback: back to private

- **Swap the cron back first,** in the same change: the private line in
  `daily.yml`. The full schedule spends a private repo's 2,000 free minutes in
  about a week, and runs then stop starting: one lost roster snapshot a day.
- **Put the healthcheck back** on `17 2,15,20,23 * * *` with the 8-hour grace.
- **Take the app down, or make it private, before the repo.** Per the spec,
  Community Cloud shares apps from public repos only; that is platform
  behaviour, not checkable from here.
- **Taking only the site down** needs no change of visibility.

## 7. Once public, and the offseason

- **Free minutes.** Public runs cost no Actions minutes, which is what makes the
  36-run schedule affordable.
- **The 60-day rule.** GitHub disables a public repo's scheduled workflows after
  60 days without activity. `scripts/commit_state.sh` commits a dated
  `state/keepalive.txt` whenever the newest commit is 45 days old, but only
  inside a run that reaches the Commit step. A suite that stays red stops the
  keepalive too, and a disabled schedule needs a manual re-enable (Actions,
  daily-update, Enable workflow).
- **The seed.** It must be rebuilt for 2027-28 between the Finals and the next
  opener (`docs/known_issues.md`, "The seed serves ONE season").

## 8. The odds feed (The Odds API)

**What it is.** The app's live layer (`bucketlib/live.py`) buys today's player
lines for DraftKings, FanDuel, BetMGM, PrizePicks and Underdog from The Odds
API, one call per game, for today's ET date only and only for games the bot has
published. The board shows every book at once: one row per player and stat,
one column per book (DK, FD, MGM, PP, UD), each line with an up arrow where the
model leans More (Over) at it, a down arrow where it leans Less (Under), the
number alone where it has no lean or the line does not offer the side it
leans, and the dash where that book has no line. A book that has not posted a
game shows the dash on that game's rows, and a note names the game. No price
is shown for any book. The paste box stays as the backup (outages, typed
picks, Demons and Goblins). The bot and Actions never call the vendor, and
nothing ever fetches PrizePicks itself.

- **When it buys.** After the first view of today's slate or of a game today,
  a background thread (the refresher, unless `BUCKETVALUE_ODDS_REFRESHER` is
  off) checks every 60 seconds, with or without visitors, for as long as the
  process runs. It buys each game's first pull from 08:50 ET and its
  refreshes on the plan's schedule, the last planned one 30 minutes before
  tip, and never within 5 minutes of tip, but only for the games of the work
  order that the latest view of today left: that view's day and the next day,
  and none once the order is more than 36 hours old. So after a full day with
  no view of today, as after a restart, nothing is bought until a view of today
  (Streamlit Cloud may stop an app nobody visits; how soon is UNVERIFIED). A
  view of today starts the same walk and waits at most 3 seconds for it; what
  is not done by then fills in by itself.
- **Where the lines live.** In process memory only: a code push and a Clear
  cache keep them, a restart loses them. They never reach disk, git, a
  visitor's token board or a browser copy.
- **What it costs.** On the free plan (500 credits a month) points for the five
  books cost 1 credit per game call (the probe, 2026-11-11). Each game gets its
  08:50 pull and usually one more, 30 minutes before tip. An answer with nothing
  posted is free (the vendor's docs).
- **As shipped:** the free plan and points only; no responsible-gambling line;
  the free PrizePicks-only checks on; the newer of a pasted PrizePicks line and
  the feed's is shown (a typed pick, a Demon or a Goblin always is); today
  only; the default reserve.

### Settings

On Cloud each is a root-level Streamlit secret, written as a quoted string;
locally they are plain environment variables. The key and the feed's settings
are read on every call, `BUCKETVALUE_LOG` only when the process starts. Press
Reboot app after any secrets edit anyway: whether Cloud's secrets editor
reloads them by itself is UNVERIFIED.

| Setting | Default | Accepted | What it does |
|---|---|---|---|
| `BUCKETVALUE_ODDS_API_KEY` | unset | the key | Unset: no vendor call at all, and the site is paste only. Set on Cloud; local runs use the key file below. |
| `BUCKETVALUE_ODDS_ENABLED` | `"1"` | on: `"1"`, `"true"`, `"yes"`, `"on"`; off: `"0"`, `"false"`, `"no"`, `"off"` | The kill switch. Off stops every vendor call, the free one too, and the refresher. A value it cannot read also means off. |
| `BUCKETVALUE_ODDS_MARKETS` | `"player_points"` | market keys, comma separated | What a pull asks for when the budget allows (a re-buy asks the core markets). Alternate, quarter or half, yes/no, first-basket and unknown keys are dropped; an empty list falls back to the default. |
| `BUCKETVALUE_ODDS_CORE_MARKETS` | `"player_points"` | market keys from the list above | What the budget keeps when the full list does not fit, and what every re-buy asks. Keys outside the market list are ignored; with none left, its first market is used. |
| `BUCKETVALUE_ODDS_BOOKS` | `"draftkings,fanduel,betmgm,prizepicks,underdog"` | some of those five | May only narrow the five; any other name is dropped, and an empty list means all five. The board keeps all five columns: a book left out shows the dash on every row (PrizePicks' column still shows a visitor's pasted lines), and while it has no line for a game another book has, the note reads `<Book> lines are not loaded on this site.` |
| `BUCKETVALUE_ODDS_RESERVE` | unset: the larger of 25 and 5% of the plan's credits | a whole number, 0 to 100000000 | Credits the site never plans and never spends: room for the probe, local runs and surprises. |
| `BUCKETVALUE_ODDS_FIRST_PULL_ET` | `"08:50"` | `"HH:MM"`, ET | The earliest first pull of a game, on its own ET date. The restart brake starts 60 minutes later (09:50 by default). |
| `BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN` | `"30"` | 0 to 720 | The last planned refresh, this many minutes before tip. |
| `BUCKETVALUE_ODDS_MIN_REFRESH_MIN` | `"15"` | 1 to 720 | A planned refresh comes no sooner than this after the game's last pull. |
| `BUCKETVALUE_ODDS_TIP_GUARD_MIN` | `"5"` | 0 to 720 | No paid call this close to tip, or later. |
| `BUCKETVALUE_ODDS_MAX_CALLS_10MIN` | `"40"` | 1 to 1000 | The most paid calls in any 10 minutes, in one process. |
| `BUCKETVALUE_ODDS_PP_TOPUP` | `"1"` | the on and off words | Off: a game whose pull came back without PrizePicks waits for its next refresh instead of the free PrizePicks-only checks every 10 minutes. |
| `BUCKETVALUE_ODDS_REFRESHER` | `"1"` | the on and off words | Off: no background refresher; only page views buy. |
| `BUCKETVALUE_ODDS_API_KEY_FILE` | unset | a file path (`~` allowed) | Local runs and the probe only: the private key file. Never on Cloud. |
| `BUCKETVALUE_LOG` | `WARNING` | exactly a level name, in any case: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` | INFO adds the walk's spend line and the restart line. It has no fallback (below). |

- A whole number may be written `"30"`, `"+30"` or `"30.0"`. A value out of
  range, or one that cannot be read, falls back to the default and logs
  `oddsapi: setting <NAME>: ...` (the name, never the value). The kill switch is
  one exception: a value it cannot read turns the feed off, and the log then
  shows only the INFO line `oddsapi: switched off`.
- `BUCKETVALUE_LOG` is the other: it has no fallback. Anything but a level
  name, such as `"INFO "` (a trailing space), `"INF0"`, `"verbose"` or `""`,
  raises `ValueError: Unknown level` when the site's code first loads in a new
  process. The first page after the start then shows a `ValueError` instead of
  the site (its message redacted, as every page error is on the shared site),
  and the log has `ValueError: Unknown level: ...`. The pages after it load,
  with the log stuck at WARNING. After setting it, press Reboot app and open
  the site once; if it shows a `ValueError`, fix the value and reboot again.
- The market keys: player_points, player_rebounds, player_assists,
  player_threes, player_blocks, player_steals, player_blocks_steals,
  player_turnovers, player_points_rebounds_assists, player_points_rebounds,
  player_points_assists, player_rebounds_assists, player_frees_made,
  player_frees_attempts, and player_fantasy_points (kept for PrizePicks only).
  Each market a pull asks for can cost one more credit.
- There is no reset-day setting: credits reset on the 1st for every plan (the
  vendor's FAQ), and the budget reads the balance to see it.
- TOML traps: write `BUCKETVALUE_ODDS_ENABLED = "0"`, never `false`, and
  `BUCKETVALUE_ODDS_MARKETS = "player_points,player_rebounds"`, never a TOML
  list; a bool or a list never reaches the environment. Root level only: a
  value under a table such as `[odds]` is not copied.

### Going live

Only Barrett does this; the key never goes in a chat, in git, in Actions or
in ~/.streamlit/secrets.toml.

1. On share.streamlit.io: the app, then Settings, then Secrets. Add at root
   level, beside `BUCKETVALUE_READONLY = "1"`:
   ```
   BUCKETVALUE_ODDS_API_KEY = "<the key>"
   ```
   and save.
2. Press Reboot app.
3. Open today's slate once: that view starts the walk and the refresher.
4. Note the balance: `scripts/probe_oddsapi.py --balance` on the Mac (below).

Any time of day works. From 08:50 ET the first view after the reboot normally
buys today's games that have not tipped (the restart rules below say how).

### Turning it off fast

1. In Secrets set `BUCKETVALUE_ODDS_ENABLED = "0"` (or delete the key), save,
   and press Reboot app. The site is then paste only, exactly as before the
   feed: no vendor call, no refresher and no feed text on any page.
2. If a leak is suspected, first regenerate the key in the vendor's account
   dashboard. That stops the old key everywhere at once (it is then refused
   with a 401). Then put the new key in Secrets and reboot, or leave the feed
   off; anything else that used the old key needs the new one too.
3. No code change is needed: with no key, or the switch off, no feed code runs.

### The first morning: what to watch

- **Before 08:50 ET**, set `BUCKETVALUE_LOG = "INFO"` for the day and press
  Reboot app (a restart before 08:50 costs no credits), then open today's slate
  once (Today in the date bar) so the refresher is running at 08:50. The slate
  must load: a `ValueError` there (the log says `Unknown level`) means the
  value is not exactly a level name (Settings, above), so fix it and reboot
  again.
- **08:50 to 08:52 ET, the log** (Manage app, Logs). Expect `oddsapi: walk
  pulled <n> game(s), cost <c>, remaining <r>` from the refresher with nobody
  visiting, and no `oddsapi` WARNING.
- **9:00 to 9:15 AM ET, the slate.**
  - "Model vs the board" shows without a paste, with the note `Lines as of
    8:50 AM ET.`
  - The grid has five book columns, DK, FD, MGM, PP and UD, each cell a line,
    with an up or down arrow where the model leans at it, and one legend line
    under it: `Up arrow: model leans More (Over) at that line. Down arrow: Less
    (Under). No arrow: no lean there, or the line does not offer that side. A
    dash: no line from that book.`
  - A game PrizePicks has not posted shows the dash in the PP column, and the
    note names it: `No PrizePicks lines yet for <game>.` (books missing the
    same games share one sentence: `No PrizePicks or Underdog lines yet for
    <game>.`). The free PrizePicks checks fill it in once PrizePicks posts.
  - No `did not answer` or `not refreshing` notice. Each card's count equals
    its Game page's `Lines on this game` count.
  - On a phone, measure, do not eyeball: at 375 px, on the slate and on a Game
    page with one expander open, in light and with the Dark toggle, run this
    check in the browser console:
    ```js
    (() => {
      const m = document.querySelector('[data-testid="stMain"]');
      const all = [...document.querySelectorAll('.bv-grid-wrap')];
      const wraps = all.filter((w) => w.getClientRects().length).map((w) => {
        let ok = true;
        for (let a = w.parentElement; a && a !== m; a = a.parentElement)
          if (a.scrollWidth > a.clientWidth + 1) ok = false;
        return {ancestorsFit: ok, scrolls: w.scrollWidth > w.clientWidth,
                focusable: w.tabIndex === 0, role: w.getAttribute('role')};
      });
      const guard = document.querySelectorAll(
        '[data-testid="stVerticalBlock"].st-key-bv_board_grid [data-testid="stElementContainer"]:has(.bv-grid-wrap),' +
        '[data-testid="stVerticalBlock"].st-key-bv_game_lines [data-testid="stElementContainer"]:has(.bv-grid-wrap)').length;
      return {mainFits: m.scrollWidth <= m.clientWidth, wraps, grids: all.length, guard};
    })()
    ```
    The page's scroller is `section[data-testid="stMain"]`, not the document
    (Streamlit 1.59 never scrolls the document), so it must have scrollWidth <=
    clientWidth; every ancestor of each grid up to stMain must have scrollWidth
    <= clientWidth + 1 while the grid's own box has scrollWidth > clientWidth
    (it is the one that scrolls); the grid's box is focusable with role region;
    and the CSS guard matches exactly one element per grid. So it passes when
    `mainFits` is true, every entry of `wraps` has `ancestorsFit`, `scrolls`
    and `focusable` true and `role` "region", and `guard` equals `grids` (0
    means the guard no longer matches, more means it matches too much). Then
    scroll a grid sideways: the Player column (on a Game page, the Stat column)
    stays put.
  - Note how long the first view after the 08:50 pull took: it compares every
    book's lines once (the cost is UNVERIFIED). If it is often over a couple of
    seconds, say so.
- **The loading line.** A view that shows `Lines for <n> games are still
  loading; this board fills in by itself.` should rerun once by itself within
  a few seconds. Check that a pasted board survives that rerun (UNVERIFIED next
  to the board keeper).
- **Right after the walk**, `--balance`: the drop should equal the games
  pulled (1 credit a game on points), plus 1 for each PrizePicks check that
  found lines.
- **The board's notes.** Names under `Not matched to a current roster:` go into
  `data/name_aliases.csv`.
- **30 minutes before the first evening tip**, the free plan's second pull: the
  time on the lines moves, and the balance drops again by about the evening's
  untipped games, fewer when PrizePicks checks took those credits.
- **After the first tip**, a tipped game's page shows `Pre-tip lines, as of
  <time>.` and its lines settle against the box score. On a tipped game's page,
  a book with no lines for that game reads `The odds feed did not load <Book>
  lines for this game before tip.` A tipped game with no feed lines at all
  reads `No lines were posted for this game before tip.` only when its books
  posted none, else `The odds feed did not load lines for this game before
  tip.`
- **The next morning**: the balance against the plan, and the number of
  `oddsapi: feed memory created (process start)` lines (each is a restart).

### Reading the log

Manage app, then Logs. The site's lines start `[bucketvalue]` and their level;
the feed's then start `oddsapi`. Only WARNING lines show at the default level;
`BUCKETVALUE_LOG = "INFO"` (and a reboot) adds the INFO ones. Below, `<...>`
stands for a number, a code or a name. No line ever holds the key or a URL; the
only vendor text in any line is an unknown team's name, cleaned and cut to 40
characters.

Normal running (INFO, unless marked):

| Line | Meaning |
|---|---|
| `oddsapi: feed memory created (process start)` | A new process: a reboot, a crash or Cloud restarting the app. One per restart. |
| `oddsapi: feed memory created (replaced: version change)` | Pushed code changed the shape of the feed's memory, and the new memory buys today's games again (the push rule below). |
| `oddsapi: refresher started` | The refresher is running. |
| `oddsapi: refresher replaced (version change)` | Pushed code replaced the refresher's loop. |
| `oddsapi: refresher restarted (the last loop had stopped)` | WARNING. The refresher had died and a view restarted it. Tell the developer if it repeats. |
| `oddsapi: budget full (remaining <r>, reserve <s>, <g> games to the reset)` | A process's first plan: everything fits. |
| `oddsapi: walk pulled <n> game(s), cost <c>, remaining <r>` | One walk's calls that may have been billed: n of them, c credits counted, r credits left. |
| `oddsapi: no key; feed off` / `oddsapi: switched off` | Once per process: the feed is off (no key, or the kill switch). |
| `oddsapi: event matched with home and away swapped: <game> (<away> at <home>)` | A neutral-site game; normal. |
| `oddsapi event odds: <code>; the game's pulls end for today` | A 404: the vendor has dropped the game; no more pulls of it today. |
| `oddsapi: the events call answered again; paid calls may resume` | Recovery after a refused key or no credits. |
| `oddsapi: the credits came back; paid calls resume` | The counters show a reset or new credits. |

The budget (WARNING):

| Line | Meaning and what to do |
|---|---|
| `oddsapi: budget <mode> (remaining <r>, reserve <s>, <g> games to the reset)` | The plan changed mode. `core markets only`: the full market list no longer fits, so the core markets until the reset. `short, morning pulls only`: one pull a game fits. `paused at the reserve`: nothing is left above the reserve until the reset. `paused, out of credits`, `paused, the key was refused`, `paused by the day breaker`: see their own lines. `unknown, no balance yet`: no answer has carried the balance yet. `full`: back to normal. |
| `oddsapi: budget held a pull today (<code>)` | Once a day per code. `floor`: a refresh, a PrizePicks check or a re-buy would have spent a credit a later game needs for its morning pull. `reserve`: it would have crossed the reserve. `ceiling`: the day's spending ceiling. While a game is held the day reads paused: `not refreshing`, or with no lines `No lines from the odds feed today.` |
| `oddsapi: spending ceiling reached for today` | Once a day: the month's spending is half again above its pace so far. It eases tomorrow. |
| `oddsapi: paid-call rate cap reached; the rest wait their turn` | `BUCKETVALUE_ODDS_MAX_CALLS_10MIN` paid calls were made in the last 10 minutes; the rest go later. |
| `oddsapi: memory-loss brake: <n> game(s) of today counted as pulled at the ledger's birth` | A restart after 09:50 ET in a month the account was already spending (or a version change): those games are re-bought once, at once, while each has two or more pulls planned and is before its last refresh time (restarts, below). |
| `oddsapi: credits did not reset by midday on the 1st; check the vendor dashboard` | Check the dashboard, and tell the developer. |
| `oddsapi: credits reset outside the 1st; the period rule needs a change` | Tell the developer. |

The key (WARNING):

| Line | Meaning and what to do |
|---|---|
| `oddsapi: 401 <code>; all calls stopped for 30 min` | The key was refused (`<code>` is INVALID_KEY, DEACTIVATED_KEY, MISSING_KEY or HTTP_401). Fix the secret and reboot; until then one free check every 30 minutes. |
| `oddsapi: 401 OUT_OF_USAGE_CREDITS; paid calls stopped until the credits reset` | Out of credits. Wait for the 1st or upgrade the plan; the free events call keeps watching. |
| `oddsapi: key file unreadable (<why>)` | Local runs only: the file `BUCKETVALUE_ODDS_API_KEY_FILE` names cannot be read (`<why>` is an error type, or `not a regular file`). |

Failures (WARNING):

| Line | Meaning and what to do |
|---|---|
| `oddsapi event odds failed: <kind> <code>` | One game's paid call failed (logged at most once per kind and code in 10 minutes); the game backs off, 90 seconds doubling to 60 minutes. Kinds: `timeout`; `server connect` or `server transport` (the network); `oversize` (an answer over 4 MB); `bad_payload` (an answer the site cannot use), `bad_payload no_lines_kept` (books but no readable line) or `bad_payload unknown_team`; `bad_request <CODE>` (a 422, such as INVALID_MARKET for a market the vendor does not know; `bad_request MISSING_REGION` would mean the vendor now requires `regions`, which the probe found it did not on 2026-11-11: tell the developer); `rate_limited <CODE>` (a 429); `server HTTP_<status>` (a 5xx, 403 or 3xx). |
| `oddsapi events failed: <kind> <code>` | The free events call failed: the last list of games is kept, and it retries after 90 seconds doubling to 60 minutes. With no list yet, nothing can be bought. |
| `oddsapi: <n> billed failures in a row; paid calls stopped until tomorrow (parser counters: <counters>)` | The day breaker: three answers in a row (n counts them), any games, came back billed and unusable. Paid calls resume the next ET day. Timeouts never trip it. Tell the developer. |
| `oddsapi: <book> answered 0 of <n> lines for a game; kept the previous lines (<counters>)` | That book's answer had nothing readable; its earlier lines stay. |
| `oddsapi: <book> answered <n> of <m> lines for a game; kept the previous lines` | That book came back with under half its lines; its earlier lines stay. |
| `oddsapi: unknown team name(s): <names>` | The vendor named a team the site does not know; that game is skipped. Tell the developer. |
| `oddsapi: setting <NAME>: <what>` | A setting was dropped or set back to its default: fix the secret. `<what>` is one of `unreadable value; default used`, `out of range; default used`, `empty list; default used`, `unknown key dropped`, `alternate market dropped`, `period market dropped`, `yes/no market dropped`, `first-basket method market dropped`, `book outside the five dropped`, `more than 10 books; the rest dropped`, `not in the market list; the first market used`. |
| `oddsapi: today's published file is unreadable (<why>); the refresher waits for a view` | Once a day: the 08:50 buy waits for the first view. |
| `oddsapi: <n> game(s) of today have no teams in the work order's horizon; the refresher waits for a view` | Once a day: the same. |
| `oddsapi: the date's published file is unreadable (<why>)` | Nothing is bought for that date. |
| `oddsapi: a malformed game entry counted as absent` | A bug: tell the developer. |
| `oddsapi: kick failed (<Type>)`, `oddsapi: walk failed (<Type>)`, `oddsapi: worker failed (<Type>)`, `oddsapi: feed view failed (<Type>)`, `oddsapi: feed check failed (<Type>)`, `oddsapi: feed start failed (<Type>)`, `oddsapi: feed result failed (<Type>)`, `oddsapi: feed tables failed (<Type>)`, `oddsapi: loading watch failed (<Type>)`, `oddsapi: refresher start failed (<Type>)`, `oddsapi: refresher tick failed (<Type>)` | An error inside the feed code, logged by its type only. The page falls back to a notice or the paste board. Tell the developer. |
| `prepare_feed failed (<Type>)`, `compare_books: <book> <game> failed (<Type>)`, `date_board: the feed board failed (<Type>)`, `date_board: no simulator parameters` | The board could not be built from the feed's lines; the page says so. Tell the developer. |

### Restarts, reboots and code pushes

| Event | The feed's memory | What it costs |
|---|---|---|
| A code push | Kept: freshen purges only the bucketlib package, and the memory sits outside it | Nothing. Check the live page after a library push: Cloud can run new pages on old modules until freshen acts. |
| Clear cache (Barrett's, or any visitor's) | Kept; the compared lines are rebuilt on the next view | Nothing (CPU only). |
| A push that changes the memory's shape | Replaced: `oddsapi: feed memory created (replaced: version change)` | From 08:50 ET, today's untipped games are bought again (the push rule below). |
| A restart: Reboot app, a crash, Cloud restarting the app | Lost: `oddsapi: feed memory created (process start)` | Before 08:50 ET, nothing. From 08:50 to 09:50, today's games are bought again: in a month the account was already spending each is a re-buy, bought only while every later game's morning pull stays covered and the day's ceiling holds, and for the core markets only; otherwise (a period's first day, go-live day) they are first pulls, which only the reserve and the ceiling bound. After 09:50, in a month that was already spending, the brake counts today's untipped games as pulled at the restart, and each is re-bought once, at once, while it has two or more pulls planned and before its last refresh time; in a month that was not spending yet, today's games are bought as first pulls. |

- **Tipped games** lose their pre-tip lines in a restart. Their Game pages say
  so: `The odds feed's pre-tip lines for this game were not kept. The site
  restarted after its last update. A pasted board still shows here.` Pasted
  boards come back from each visitor's browser.
- **On the free plan**, a restart after 09:50 in a month that was already
  spending costs more than credits: the re-buy is the game's last pull of the
  day, so there is no evening refresh, and a book that posts after the restart,
  PrizePicks included, is not fetched that day. After a second restart the same
  day, the next process can plan one pull a game, and then the games it counts
  as pulled get neither a re-buy nor an evening refresh. Neither touches the
  9 AM view.
- **What restarts cost**, in the January replays (12 games on Jan 7, free plan,
  points; docs/serving_design.md section 10 has the details): no restart, 24
  credits; one at 15:00, 24; three in the afternoon, 27; five between 09:00 and
  09:40, 72, all in the morning, while the floor and the ceiling held and every
  later game kept its morning pull; one every 10 minutes from 10:00 to 18:00, 27.
- **A crash loop.** If `oddsapi: feed memory created (process start)` shows up
  more than twice in an hour, turn the feed off until the cause is fixed. That
  line is INFO (set `BUCKETVALUE_LOG = "INFO"`); at the default level a restart
  after 09:50 in a spending month shows as the brake's WARNING instead. Each
  restart from 08:50 can buy today's games again. In a month that was already
  spending those are re-buys, which the floor (every later game's morning pull)
  and the day's ceiling bound; on a period's first day, or go-live day, they
  are first pulls, which only the reserve and the ceiling bound. Either way
  each costs credits.
- **The push rule.** A code push keeps the memory, but a change to its shape
  (`FEED_MEMORY_VERSION` in bucketlib/live.py, or any change to the fields
  `new_feed_memory` makes) replaces it. Never push such a change between 08:50
  and 09:50 ET: every one of today's games would be re-bought. After 09:50 the
  same push re-buys them through the brake's one re-buy each, which on the free
  plan is each game's last pull. So push it before 08:50 ET. Any push is best
  outside game hours.

### The balance, the key and local runs

- **The balance.** `scripts/probe_oddsapi.py --balance` on the Mac (one free
  call: remaining, used and allowance), or the vendor's account dashboard.
  Visitors never see a balance, a credit or an error code, and no notice names
  the vendor.
- **One key, used by this site alone.** Sharing the key with another program
  costs this site lines, never more credits than its own rules allow: the
  other program's spending comes off the same balance and counts against this
  site's daily spending ceiling, so the site plans fewer pulls and stops at its
  reserve, while each day's first pulls still come first, in tip order (they
  skip the morning-pull floor, and a true first pull skips the ceiling too),
  so the games that go without lines are the month's last.
- **Local runs share the site's balance.** Set `BUCKETVALUE_ODDS_API_KEY_FILE`
  only when a local run should buy lines: a local run keeps its own memory, so
  it buys today's games once more, and its spending comes off the shared
  balance and counts toward the site's ceiling. The ceiling does not bound the
  local run's own first pulls, though, if it was already running at a game's
  08:50: those are true first pulls (B23), which only the reserve stops. So
  start a local run that should buy lines after 08:50, and stop it (or unset
  `BUCKETVALUE_ODDS_API_KEY_FILE`) when done: one left open overnight can buy
  the next morning's games again. The tests delete every vendor variable,
  switch the refresher off and block the vendor client, so the suite never
  spends. Keep the key in `~/.config/bucketvalue/odds_api_key` (folder mode
  0700, file mode 0600), outside every repo and every iCloud folder, and never
  in ~/.streamlit/secrets.toml, which every local Streamlit app reads. To
  write a new key there, with it on the clipboard:
  ```
  mkdir -p -m 700 ~/.config/bucketvalue
  ( umask 077; pbpaste > ~/.config/bucketvalue/odds_api_key )
  pbcopy < /dev/null
  ```
  `pbpaste` keeps the key out of the shell history, and `pbcopy < /dev/null`
  empties the clipboard after.
- **Support mail to the vendor:** send the path and the parameters with apiKey
  removed, never a copied URL (their pages ask for the full URL, which holds
  the key).

### The probe

`scripts/probe_oddsapi.py` is local only: Barrett runs it by hand from the
development tree, it refuses to run in GitHub Actions, and the bot never runs
it.
```
BUCKETVALUE_ODDS_API_KEY_FILE=~/.config/bucketvalue/odds_api_key .venv/bin/python scripts/probe_oddsapi.py --balance
```
- `--balance`: one free events call; it prints remaining, used and allowance.
- `--evening`: free events calls with and without `status=pre`, while games
  are on.
- With neither, the full run: 13 steps, at most `--max-credits` credits
  (default 16), each paid call's real cost read off its answer. Run it on a
  game day once the books have posted (the first run was at 10:15 ET).
- `--key-file PATH` names the key file instead of the variable.
- It refuses to run (exit 2) without a key file, for a key file under
  ~/Desktop, ~/Documents or ~/Library/Mobile Documents or inside a git
  checkout, for one that group or others can read, is not a plain file or
  holds no key, for a negative `--max-credits`, and in GitHub Actions. Exit 0
  when nothing failed, 1 on a FAIL.
- Its lines hold no key, URL or error text, so its output can be shared as it
  is. It writes no file.
- The first full run, 2026-11-11 at about 10:15 ET: 10 PASS, 0 FAIL, 14 credits
  spent, 486 left. What it settled is in docs/data_notes.md ("The Odds API").
  Still to watch: what the books have posted by 08:50 (the first morning shows
  it) and the reset hour (`--balance` around a month's end: about 7 AM, 9 AM
  and 7 PM ET on the last day, and the next morning).

### The budget in plain words

- **The plan.** Every game of the credit period gets the same number of pulls,
  from the balance, the games still to come and the reserve, worked out again
  on every walk. On the free plan with points that is usually two: 08:50 and 30
  minutes before tip.
- **What counts as a pull.** Every call that may have been billed, whatever
  came of it. An answer with nothing posted is no pull; the game is asked again
  10 minutes later.
- **PrizePicks checks.** A game whose pull came back without PrizePicks is
  asked for PrizePicks alone every 10 minutes, free while PrizePicks has
  nothing (the vendor's docs). The first answer with lines costs a credit and counts as a pull,
  which on the free plan takes the place of the evening refresh.
- **The floor.** No refresh, PrizePicks check or re-buy spends a credit that a
  later game needs for its morning pull.
- **The ceiling.** The month's spending may run at most half again above its
  pace so far, read off the vendor's own counter. It bounds every refresh,
  PrizePicks check and re-buy, and every first pull of a process born after
  the game's 08:50, so restarts and crash loops cannot push past it. A second
  process already running at 08:50 (a local run with the key file started
  before 08:50, or left open overnight) makes true first pulls, which skip it
  (B23, below), and buys that morning's games again, bounded only by the
  reserve.
- **First pulls come first (B1).** A first pull never waits for another game's
  first pull. When credits run short, today's games are bought first, earliest
  tip first, and the month's last games go without lines until the reset. A
  first pull asks for extra markets only when every waiting game's morning pull
  stays covered; otherwise it asks the core markets.
- **A restart in the morning (B11).** After a restart between 08:50 and 09:50
  ET, in a month the account was already spending, the new process cannot tell
  whether the old one bought today's games, so each first pull is treated as a
  re-buy: bought only if the floor and the ceiling hold, and for the core
  markets only.
- **A failed morning pull (B12).** A game whose pulls so far all failed (a
  timeout at 08:50, say) is tried again as soon as its wait allows (90 seconds,
  then doubling), not at the evening refresh. On the free plan (two pulls a
  game) that retry uses the game's second pull, so the game keeps its morning
  lines but gets no pre-tip refresh.
- **A restart after 09:50 (B21).** In a month the account was already
  spending, the new process counts each of today's untipped games as pulled at
  the restart (the brake), so their later pulls keep the normal spacing. A game
  it holds no lines for (after a restart, every one) is then re-bought once, at
  once, while the game has two or more pulls planned and before its last
  refresh time (30 minutes before tip). On the free plan that re-buy is the
  game's last pull of the day.
- **What trips the day breaker (B22).** Only an answer that came back, billed
  and unusable (a 200 the site cannot read, an answer over the size cap, an
  error that was billed) counts toward three in a row. A timeout, a failed
  connection or a download the network cut off still costs its worst case,
  counts as one of the game's pulls when it may have been billed, and makes
  the game wait, but never trips the breaker.
- **A true first pull skips the ceiling (B23).** A first pull by a process
  that was already running at that game's 08:50 can never re-buy what an
  earlier process bought, so it skips the spending ceiling as every first pull
  skips the floor; only the reserve stops it. So a key drained from outside
  early in the month never holds that morning's lines while credits stay
  above the reserve. It cannot see what another process running at the same
  time bought: two processes running at 08:50 (the site and a local run) each
  buy that morning's games, past the ceiling if need be.
- **A failed free call.** A timeout of a call expected to be free (a PrizePicks
  check, or a pull after an answer with nothing posted) still takes its worst
  case off the balance but is no pull, once: until the vendor answers for that
  game again (with nothing posted, or with lines), its next failure counts as a
  pull and its first pull is checked against the ceiling.

### Known limits (kept as built)

- **More than one market.** The plan counts the pulls already made at the
  plan's cost per pull. That matters only with more than one market
  configured: a day's last first pull can then see the plan as full and ask
  every market, spending credits the core plan kept for that evening's
  refreshes. Fix it before adding markets.
- **A vendor that alternates.** A vendor that answered "nothing posted" and
  then billed a timeout, over and over, would re-arm the free-call rule at each
  empty answer. The waits bound it to about 13 credits per game per day, each
  taken off the balance as its worst case, so the plan tightens. Accepted: a
  billed timeout means the game had lines, so an empty answer right after one
  is unlikely.
- **Restarts past 09:50 on the free plan.** The re-buy is the game's last pull,
  so a book that posts after the restart, PrizePicks included, is not fetched
  that day; after a second restart the same day, the next process can plan one
  pull a game, and its games then get neither a re-buy nor an evening refresh
  (in a three-restart replay, 9 of 12 games had no lines from 14:00). Both need
  a restart after 09:50 and leave the 9 AM view untouched. Kept for now
  (Barrett, 2026-11-11); revisit after the test week or a move to a paid plan.
- **The push window.** Never push a change to the feed memory's shape between
  08:50 and 09:50 ET (the push rule above).
