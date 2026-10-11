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
