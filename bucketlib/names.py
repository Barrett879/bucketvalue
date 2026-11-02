"""Name -> player resolution, used ONLY at boundaries that carry no NBA id
(PrizePicks pastes, injury-report PDFs, ESPN feeds). Every modelling table is
keyed on the stats.nba.com PERSON_ID; names never join two tables internally.

Copied (not imported) from HoopsValue utils.NameIndex: importing that module
runs its disk-cache seeding and pulls in streamlit/bs4 at import. Extended with
the gaps the 2026-10-08 identity research measured against the 615-player
2026-27 index:

  - hyphen vs space: "Karl Anthony Towns", "Shai Gilgeous Alexander"
  - initials: "P.J." == "PJ" (the sibling sites' normalize_name turned "P.J."
    into "p j" and missed PJ/CJ/RJ/TJ, common in the NBA)
  - nicknames that are not prefixes (Moe Wagner, Bones Hyland, Bub Carrington,
    KJ Martin, Lu Dort, Mo Bamba): data/name_aliases.csv, consulted first
  - family-name-first players: the injury report prints "Yang, Hansen" while
    stats.nba.com has Yang Hansen, so `resolve_last_first` tries both orders
  - a suffix between two commas: the report prints "Udeh, Jr., Ernest" (the
    playerIndex last name is "Udeh, Jr."; found live 2026-11-02), which
    `last_first_orders` reads as "Udeh Jr., Ernest"
  - team scoping: when the source names a team, candidates are filtered to it
    BEFORE the prefix fallback, which keeps "Jay Williams" off Jaylin Williams
    unless he is the only Williams on that team.

Build the index from the CURRENT roster only. Suffix stripping merges fathers
and sons (Tim Hardaway -> Tim Hardaway Jr.) when retired players share it.
Every lookup reports HOW it matched so a UI can list non-exact matches instead
of dropping or silently guessing.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")
_SUF = re.compile(rf"\s+({'|'.join(_SUFFIXES)})$")
_PUNC = re.compile(r"[^a-z0-9 ]")

ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "name_aliases.csv"


def normalize(name: str) -> str:
    """Lowercase, accents stripped, apostrophes and periods removed, hyphens
    turned into spaces, whitespace collapsed. 'P.J.' -> 'pj'."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()
    s = s.replace("’", "'")
    s = re.sub(r"[.']", "", s)          # P.J. -> pj, De'Aaron -> deaaron
    s = s.replace("-", " ").replace(",", " ")
    return " ".join(s.split())


def name_keys(n: str) -> list[str]:
    """Exact-match key variants for an already-normalized name."""
    out: list[str] = []
    for base in (n, _SUF.sub("", n)):
        flat = " ".join(_PUNC.sub("", base).split())
        for k in (base, flat, _SUF.sub("", flat), flat.replace(" ", "")):
            if k and k not in out:
                out.append(k)
    return out


def name_split(n: str) -> tuple[str | None, str | None]:
    parts = " ".join(_PUNC.sub("", _SUF.sub("", n)).split()).split()
    return (parts[0], " ".join(parts[1:])) if len(parts) >= 2 else (None, None)


def last_first_orders(last_first: str) -> tuple[str, str]:
    """The two names an injury-report 'Last, First' (it holds a comma) can be,
    in the order to try them: 'First Last', then the printed order for
    family-name-first players ('Yang, Hansen' is Yang Hansen).

    The suffix belongs to the last name. The report usually prints it there
    ('Lively II, Dereck'), but for a player whose playerIndex last name holds
    a comma it prints it between two ('Udeh, Jr., Ernest'). That suffix goes
    back on the last name, so both printings try the same two names: split at
    the first comma alone they were 'Jr., Ernest Udeh' and 'Udeh Jr., Ernest',
    which match nobody."""
    last, first = (p.strip() for p in last_first.split(",", 1))
    suf, sep, rest = first.partition(",")
    if sep and normalize(suf) in _SUFFIXES:
        last, first = f"{last} {suf.strip()}", rest.strip()
    return f"{first} {last}", f"{last} {first}"


@dataclass(frozen=True)
class Match:
    value: object | None
    how: str  # "alias" | "exact" | "prefix" | "none" | "ambiguous"


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """{normalized alternate spelling: canonical NBA display name}."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(r for r in f if not r.lstrip().startswith("#")):
            alt, canon = (row.get("alias") or "").strip(), (row.get("nba_name") or "").strip()
            if alt and canon:
                out[normalize(alt)] = canon
    return out


class NameIndex:
    """Lookup keyed by player name, tolerant of the spellings above.

    add(name, value, team=None). get(name, team=None) -> Match.
    An exact key shared by two different players is AMBIGUOUS and resolves to
    nothing unless a team narrows it to one.
    """

    def __init__(self, aliases: dict[str, str] | None = None):
        self._k: dict[str, list[tuple[object, str | None]]] = {}
        self._s: dict[str, list[tuple[str, object, str | None]]] = {}
        self._aliases = load_aliases() if aliases is None else aliases

    def add(self, name, value, team: str | None = None) -> None:
        n = normalize(name)
        for k in name_keys(n):
            bucket = self._k.setdefault(k, [])
            if not any(v == value for v, _ in bucket):
                bucket.append((value, team))
        given, sur = name_split(n)
        if sur:
            lst = self._s.setdefault(sur, [])
            if not any(v == value for _, v, _ in lst):
                lst.append((given, value, team))

    def _exact(self, n: str, team: str | None) -> Match | None:
        for k in name_keys(n):
            hits = self._k.get(k)
            if not hits:
                continue
            if team is not None:
                scoped = [v for v, t in hits if t == team]
                if len(scoped) == 1:
                    return Match(scoped[0], "exact")
            uniq = {id(v): v for v, _ in hits}
            if len(uniq) == 1:
                return Match(hits[0][0], "exact")
            return Match(None, "ambiguous")
        return None

    def get(self, name, team: str | None = None) -> Match:
        n = normalize(name)
        if not n:
            return Match(None, "none")
        canon = self._aliases.get(n)
        if canon is not None:
            m = self._exact(normalize(canon), team)
            if m is not None and m.value is not None:
                return Match(m.value, "alias")
        m = self._exact(n, team)
        if m is not None:
            return m
        given, sur = name_split(n)
        if given:
            cands = self._s.get(sur, ())
            if team is not None:
                cands = [c for c in cands if c[2] == team] or []
            hit = [v for g, v, _ in cands
                   if g and min(len(given), len(g)) >= 3
                   and (given.startswith(g) or g.startswith(given))]
            if len(hit) == 1:
                return Match(hit[0], "prefix")
            if len(hit) > 1:
                return Match(None, "ambiguous")
        return Match(None, "none")

    def resolve_last_first(self, last_first: str, team: str | None = None) -> Match:
        """Injury-report 'Last, First' (suffix on the last name, 'Lively II,
        Dereck', or between two commas, 'Udeh, Jr., Ernest'). Tries
        'First Last' and, for family-name-first players, the printed order
        (last_first_orders)."""
        if "," in last_first:
            for cand in last_first_orders(last_first):
                m = self.get(cand, team=team)
                if m.value is not None:
                    return m
            return Match(None, "none")
        return self.get(last_first, team=team)
