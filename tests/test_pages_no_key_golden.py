"""With no key configured, every page is what it was before the licensed odds
feed (rule 13 of the build; design 1.3's first row; build 4's review, L3-12).

tests/test_pages_feed.py's m9 test compares the no-key pages with the same
pages with the feed stubbed off. Both sides run today's page code, so a
no-key change that both paths share (a sentence of the Game page's notes, a
line of the slate) would pass it. This test pins the no-key pages to a
golden capture instead: tests/fixtures/pages/no_key_golden.json, written by
logs/I4_capture_pages_golden.py from this module's capture() BEFORE build 4's
fix pass touched any page, and checked against stage S9's capture of the
tree before the pages were wired to the feed (the two agree element for
element; see tests/fixtures/pages/PROVENANCE.txt).

Each view runs the REAL pages through AppTest on the serve fixture's world
(serve_harness.install: 2026-03-12, every clock pinned), with no key, so the
real store.feed_start is "off" before any feed code runs: the slate, both
Game pages and About, with and without a pasted board (its paste time pinned
through props._wall), in the morning and in the evening. A capture is every
element of the page in render order, as its proto (what a browser is sent),
with each <style> block replaced by its sha256 (the style sheets are pinned
exactly, without their bulk), the board token and the keeper's per-session
request id masked.

Streamlit's element ids are masked too, on BOTH sides of the comparison
(build 6, R-L3-12): Streamlit 1.59 hashes the page script's absolute path
into every id ("$$ID-<32 hex>-<key>", the hash of the active script among
its inputs), so an unmasked golden passes only in the folder it was
captured in, and fails in Barrett's repo, in GitHub Actions and in any copy.
Only the hash goes: the element's type, its key (the id's tail) and every
other field of its proto are compared exactly as before. The golden file
itself is unchanged (it holds the capture folder's hashes, which are masked
when it is read). Nothing else in a capture depends on the checkout's path:
a copy of the tree at another path passes this test.

NEVER regenerate the golden from the current code to make this test pass: a
change to a no-key page is a change to the live site and needs its own
decision.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from bucketlib import props, props_ui
from bucketlib.util import UTC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_pages_smoke as PS  # noqa: E402  (the page runner)

GOLDEN = HERE / "fixtures" / "pages" / "no_key_golden.json"
STYLE = re.compile(r"<style>.*?</style>", re.S)
CLOCKS = {"morning": H.MORNING, "evening": H.EVENING}
# An element id's path-dependent hash (streamlit.elements.lib.utils._compute_element_id:
# "$$ID-" + the hasher's 32 hex digits + "-" + the user key).
ELEMENT_ID = re.compile(r"\$\$ID-[0-9a-f]{32}-")
ID_MASK = "$$ID-<id>-"


def mask_ids(lines: list[str]) -> list[str]:
    """Every element id's hash masked, the key after it kept: what the
    comparison reads of a capture and of the golden alike."""
    return [ELEMENT_ID.sub(ID_MASK, line) for line in lines]

# (name, clock, page, query, with a pasted board). The views of stage S9's
# capture (logs/S9_capture_pages.py), one for one.
D, CADE = H.DATE.isoformat(), "1630595"
PHI, DEN = {"away": "PHI", "home": "DET"}, {"away": "DEN", "home": "SAS"}
VIEWS = (
    ("slate", "morning", "app.py", {"date": D}, False),
    ("slate_paste", "morning", "app.py", {"date": D}, True),
    ("slate_paste_out", "morning", "app.py", {"date": D, "out": CADE}, True),
    ("slate_paste_book", "morning", "app.py", {"date": D, "book": "draftkings"}, True),
    ("slate_evening_paste", "evening", "app.py", {"date": D}, True),
    ("game_phi", "morning", "pages/Game.py", {"date": D, **PHI}, False),
    ("game_phi_paste", "morning", "pages/Game.py", {"date": D, **PHI}, True),
    ("game_den", "morning", "pages/Game.py", {"date": D, **DEN}, False),
    ("game_den_paste", "morning", "pages/Game.py", {"date": D, **DEN}, True),
    ("game_den_paste_book", "morning", "pages/Game.py", {"date": D, **DEN, "book": "fanduel"}, True),
    ("game_phi_evening_paste", "evening", "pages/Game.py", {"date": D, **PHI}, True),
    ("game_phi_evening", "evening", "pages/Game.py", {"date": D, **PHI}, False),
    ("about", "morning", "pages/About.py", {}, False),
    ("about_paste", "morning", "pages/About.py", {"date": D, "out": CADE, "book": "betmgm"}, True),
)


def _style_digest(m: re.Match) -> str:
    return "<style sha256=" + hashlib.sha256(m.group(0).encode()).hexdigest() + ">"


def capture(at, token: str | None = None) -> list[str]:
    """Every element of a page in render order, as its proto, one line per
    element: the style blocks as their sha256, the token and the nonce
    masked."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(f"{type(node).__name__}: {proto}")
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            for k in sorted(kids):
                walk(kids[k])
    walk(at._tree)
    page = STYLE.sub(_style_digest, "\n".join(out))
    if token:
        page = page.replace(token, "<board>")
    try:
        nonce = at.session_state[props_ui.KEEP_NONCE]
    except (KeyError, AttributeError):
        nonce = None
    if isinstance(nonce, str) and nonce:
        page = page.replace(nonce, "<nonce>")
    return page.split("\n")


def capture_all(monkeypatch, tmp_root: Path) -> dict[str, list[str]]:
    """Every view of VIEWS with no key, each in a fresh serve world."""
    got = {}
    for i, (name, clock, path, query, with_paste) in enumerate(VIEWS):
        at_clock = CLOCKS[clock]
        w = H.install(monkeypatch, tmp_root / f"v{i}", at_clock)
        token = None
        if with_paste:
            monkeypatch.setattr(props, "_wall", lambda: H.MORNING.astimezone(UTC))
            token = H.paste(w, H.MORNING)
        q = {**query, **({"board": token} if token else {})}
        got[name] = capture(PS.run(path, **q), token)
    return got


def test_with_no_key_every_page_is_the_golden_capture_l3_12(monkeypatch, tmp_path):
    """Rule 13: the no-key slate, Game pages and About, element for element,
    are the golden capture, and nothing reached the vendor (conftest's block
    raises on any call). Element ids are compared with their path-dependent
    hash masked on both sides (R-L3-12); everything else exactly."""
    golden = {name: mask_ids(lines) for name, lines in json.loads(GOLDEN.read_text()).items()}
    got = {name: mask_ids(lines) for name, lines in capture_all(monkeypatch, tmp_path).items()}
    assert set(got) == set(golden)
    for name in golden:
        if got[name] != golden[name]:
            changed = [i for i, (a, b) in enumerate(zip(got[name], golden[name])) if a != b]
            first = changed[0] if changed else min(len(got[name]), len(golden[name]))
            raise AssertionError(f"{name}: element {first} changed: "
                                 f"{(golden[name] + [''])[first][:300]!r} -> "
                                 f"{(got[name] + [''])[first][:300]!r}")


def test_the_mask_takes_only_the_ids_path_hash_r_l3_12():
    """R-L3-12's mask removes exactly the hash of every element id in the
    golden (158 of them), never the key after it, and nothing else: a line
    without an id is untouched, and a hash of any other length is left (so
    a change in Streamlit's id format fails the comparison loudly instead of
    being masked away)."""
    golden = json.loads(GOLDEN.read_text())
    raw = [line for lines in golden.values() for line in lines]
    masked = mask_ids(raw)
    ids = [m.group(0) for line in raw for m in ELEMENT_ID.finditer(line)]
    assert len(ids) == 158
    assert sum(line.count(ID_MASK) for line in masked) == 158
    assert not any(ELEMENT_ID.search(line) for line in masked)
    assert [a == b for a, b in zip(raw, masked)].count(False) == 158        # one id per line
    keys = sorted({line.split(ID_MASK, 1)[1].split('"', 1)[0] for line in masked if ID_MASK in line})
    assert "bv_board_keeper" in keys and "bv_out_DEN" in keys and "view_date" in keys
    assert mask_ids(['Button: id: "$$ID-' + "a" * 31 + '-x"']) == ['Button: id: "$$ID-' + "a" * 31 + '-x"']
    line = 'Block: id: "$$ID-' + "0123456789abcdef" * 2 + '-bv_datebar"'
    assert mask_ids([line]) == ['Block: id: "$$ID-<id>-bv_datebar"']


def test_the_golden_is_ascii_and_holds_every_view():
    raw = GOLDEN.read_bytes()
    assert all(b < 128 for b in raw)
    golden = json.loads(raw)
    assert sorted(golden) == sorted(name for name, *_ in VIEWS)
    for name, lines in golden.items():
        assert lines and all(isinstance(x, str) for x in lines), name
        assert "K" * 40 not in "".join(lines)
