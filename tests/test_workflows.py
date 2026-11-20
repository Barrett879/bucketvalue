"""The daily workflow's operational guards, pinned on the file itself
(.github/workflows/daily.yml). The behaviour behind the commit step is tested
for real in tests/test_commit_state.py; these pin that the workflow keeps
calling it, and the settings no Python test can see.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DAILY = (REPO / ".github" / "workflows" / "daily.yml").read_text()


def _step(name: str) -> str:
    """The text of one step, from its `- name:` (or `- uses:`) line to the next step."""
    m = re.search(rf"^      - (?:name|uses): {re.escape(name)}\n(.*?)(?=^      - |\Z)", DAILY,
                  re.M | re.S)
    assert m, f"no step {name!r} in daily.yml"
    return m.group(0)


def test_the_checkout_is_the_branch_tip_not_the_runs_creation_commit():
    """A run that queued behind another one (concurrency group) must compute
    on the tip that run pushed, or its state files conflict on the rebase."""
    assert re.search(r"ref: \$\{\{ github\.ref_name \}\}", _step("actions/checkout@v4"))


def test_the_commit_goes_through_the_one_commit_script():
    step = _step("Commit state/ only")
    assert "scripts/commit_state.sh" in step
    assert "git push" not in step and "git pull" not in step


def test_a_dead_mans_switch_pings_after_every_run():
    """Runs that never start send no failure email: the last step pings the
    monitor on every run (success, or /fail), reading the URL from a secret."""
    step = _step("Dead-man's switch")
    assert DAILY.rstrip().endswith(step.rstrip()), "the ping must be the last step"
    assert "if: always()" in step and "secrets.BV_HEALTHCHECK_URL" in step
    assert '"$HC_URL/fail"' in step and "job.status" in step


def test_the_date_input_is_for_past_replays_only():
    m = re.search(r"date:\n\s+description: \"([^\"]+)\"", DAILY)
    assert m and "PAST" in m.group(1) and "EMPTY" in m.group(1)


def _cron_slots(expr: str) -> set:
    """(UTC hour, minute) of every slot of a 5-field cron whose minute and hour
    fields are lists of numbers and ranges (all daily.yml uses), every day."""
    minute, hour = expr.split()[:2]

    def expand(field: str) -> list:
        out = []
        for part in field.split(","):
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
        return out
    return {(h, m) for h in expand(hour) for m in expand(minute)}


def test_the_bot_has_slots_before_8_am_et_all_year():
    """Lines by 8 AM (Barrett, 2026-11-19): the schedule offers SLOTS at 7:17
    and 7:47 AM ET, for last night's late games in the box scores by then.
    That is 12:17 and 12:47 UTC in winter (EST) and 11:17 and 11:47 UTC in
    summer (EDT). GitHub starts only some slots, so this pins the offer, not a
    run. Every slot of the schedule before it is kept."""
    crons = re.findall(r'^\s+- cron: "([^"]+)"', DAILY, re.M)
    assert crons, "no live cron in daily.yml"
    slots = set().union(*(_cron_slots(c) for c in crons))
    for utc_minus_et in (5, 4):                              # EST, EDT
        for minute in (17, 47):
            assert ((7 + utc_minus_et) % 24, minute) in slots, (utc_minus_et, minute)
    before = _cron_slots("17,47 13-23 * * *") | _cron_slots("17,47 0-6 * * *")
    assert before <= slots, sorted(before - slots)
    assert len(slots) == 40


def test_the_runbooks_healthcheck_cron_is_the_bots_schedule():
    """docs/runbook.md gives the dead-man's switch the bot's full cron as ONE
    expression (sections 1 and 3). Each must cover exactly daily.yml's slots,
    so a cron change cannot leave the alert's schedule stale (review F7a,
    2026-11-20)."""
    crons = re.findall(r'^\s+- cron: "([^"]+)"', DAILY, re.M)
    bot = set().union(*(_cron_slots(c) for c in crons))
    runbook = (REPO / "docs" / "runbook.md").read_text()
    exprs = re.findall(r"`(17,47 [0-9,-]+,0-6 \* \* \*)`", runbook)
    assert len(exprs) >= 2, exprs
    for expr in exprs:
        assert _cron_slots(expr) == bot, expr
