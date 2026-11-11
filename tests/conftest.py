"""Make the package importable when pytest runs from the repo root, and keep
every test away from the licensed odds feed (The Odds API).

The vendor block is SESSION scoped, on its own pytest.MonkeyPatch(): a test
that calls `monkeypatch.undo()` on the function-scoped fixture (as
tests/test_pages_smoke.py does) cannot bring back a deleted key or the real
vendor entry point mid-suite. No test ever has a key, a vendor call or the
feed's background refresher thread.

A call that reaches the block cannot pass unseen (review I1, F4):
  - the block raises VendorNetworkReached, a BaseException, so code under
    test that catches Exception (the probe's ask(), the feed's worker and
    walk, design 2.4.4) cannot swallow it;
  - every hit is counted, and a test that leaves a hit behind fails at its
    teardown, which also catches one swallowed in a thread or a Future.
A test that trips the block on purpose asks for the `vendor_block` fixture,
which gives the exception type and the test's own hits and takes them away.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VENDOR_ENV = ("BUCKETVALUE_ODDS_API_KEY", "BUCKETVALUE_ODDS_API_KEY_FILE", "THE_ODDS_API_KEY")
VENDOR_BLOCK_MESSAGE = "a test reached the vendor network: stub http.vendor_get"
VENDOR_HITS: list = []      # one entry per call that reached the block; never its arguments


class VendorNetworkReached(BaseException):
    """What the block raises: not an Exception, so `except Exception` lets it through."""


def _blocked(*a, **k):
    VENDOR_HITS.append("hit")                      # nothing of the call: its params hold the key
    raise VendorNetworkReached(VENDOR_BLOCK_MESSAGE)


@pytest.fixture(scope="session", autouse=True)
def _no_vendor_key_no_vendor_network_no_refresher():
    mp = pytest.MonkeyPatch()
    for k in VENDOR_ENV:
        mp.delenv(k, raising=False)
    mp.setenv("BUCKETVALUE_ODDS_REFRESHER", "0")      # no background thread in any test
    from bucketlib import http

    mp.setattr(http, "vendor_get", _blocked)
    mp.setattr(http, "_vendor_session", _blocked)   # the real vendor_get cannot connect either
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _vendor_env_clean_at_start():
    assert not any(os.environ.get(k) for k in VENDOR_ENV), "a vendor key is in the environment"


@pytest.fixture(autouse=True)
def _no_vendor_hit_left_behind():
    """Fails the test, at its teardown, when a call reached the block and the
    code under test swallowed what the block raised."""
    start = len(VENDOR_HITS)
    yield
    hits = len(VENDOR_HITS) - start
    del VENDOR_HITS[start:]
    if hits:
        pytest.fail(f"{hits} call(s) reached the vendor network block and the code under test "
                    "swallowed it (an except clause, a thread or a Future): stub http.vendor_get",
                    pytrace=False)


class _VendorBlock:
    """The block, for a test that trips it on purpose."""
    error = VendorNetworkReached
    message = VENDOR_BLOCK_MESSAGE

    def __init__(self):
        self.start = len(VENDOR_HITS)

    @property
    def hits(self) -> int:
        return len(VENDOR_HITS) - self.start


@pytest.fixture
def vendor_block():
    """The block's exception type and this test's hits so far. The hits are
    the test's own, made on purpose: they are taken away at its teardown,
    before the left-behind check runs."""
    block = _VendorBlock()
    yield block
    del VENDOR_HITS[block.start:]
