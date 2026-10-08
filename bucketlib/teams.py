"""The 30 NBA teams: one canonical code (the NBA tricode), every alias the
feeds use, ids for each source, and display colours.

`canonical()` returns None for anything it does not recognise. It never passes
an unknown string through: GridironValue's teams.canonical did, and a crafted
?away= value reached an unsafe_allow_html block as live markup.

Alias sources, all verified 2026-10-08:
  ESPN        GS NO NY SA UTAH WSH (ids 1..30, its own numbering)
  Kaggle/SBR  lowercase ESPN-style (gs, no, ny, sa, utah, wsh)
  BBRef       BRK CHO PHO
  Injury PDFs full names, which pypdf layout mode prints without spaces
              ("LAClippers", "PortlandTrailBlazers"), so names are also
              indexed with spaces removed.
Colours come from HoopsValue utils.TEAM_HEX / TEAM_HEX_2 (accent = brightest
recognisable brand colour, legible as a dot on both themes; never used for
text or a full background).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    code: str        # NBA tricode, the canonical key everywhere
    team_id: int     # stats.nba.com TEAM_ID
    espn_id: int
    espn_abbr: str
    full_name: str
    conference: str
    division: str
    hex1: str
    hex2: str


_ROWS = [
    # code, team_id, espn_id, espn_abbr, full_name, conf, division, hex1, hex2
    ("ATL", 1610612737, 1, "ATL", "Atlanta Hawks", "East", "Southeast", "#E03A3E", "#C1D32F"),
    ("BOS", 1610612738, 2, "BOS", "Boston Celtics", "East", "Atlantic", "#007A33", "#BA9653"),
    ("BKN", 1610612751, 17, "BKN", "Brooklyn Nets", "East", "Atlantic", "#8A8D8F", "#3A3A3A"),
    ("CHA", 1610612766, 30, "CHA", "Charlotte Hornets", "East", "Southeast", "#00788C", "#1D1160"),
    ("CHI", 1610612741, 4, "CHI", "Chicago Bulls", "East", "Central", "#CE1141", "#2E2E2E"),
    ("CLE", 1610612739, 5, "CLE", "Cleveland Cavaliers", "East", "Central", "#A31D3C", "#FDBB30"),
    ("DAL", 1610612742, 6, "DAL", "Dallas Mavericks", "West", "Southwest", "#0064B1", "#B8C4CA"),
    ("DEN", 1610612743, 7, "DEN", "Denver Nuggets", "West", "Northwest", "#FEC524", "#0E2240"),
    ("DET", 1610612765, 8, "DET", "Detroit Pistons", "East", "Central", "#C8102E", "#1D42BA"),
    ("GSW", 1610612744, 9, "GS", "Golden State Warriors", "West", "Pacific", "#1D428A", "#FFC72C"),
    ("HOU", 1610612745, 10, "HOU", "Houston Rockets", "West", "Southwest", "#CE1141", "#2E2E2E"),
    ("IND", 1610612754, 11, "IND", "Indiana Pacers", "East", "Central", "#FDBB30", "#002D62"),
    ("LAC", 1610612746, 12, "LAC", "LA Clippers", "West", "Pacific", "#C8102E", "#1D428A"),
    ("LAL", 1610612747, 13, "LAL", "Los Angeles Lakers", "West", "Pacific", "#552583", "#FDB927"),
    ("MEM", 1610612763, 29, "MEM", "Memphis Grizzlies", "West", "Southwest", "#5D76A9", "#12173F"),
    ("MIA", 1610612748, 14, "MIA", "Miami Heat", "East", "Southeast", "#B62435", "#F9A01B"),
    ("MIL", 1610612749, 15, "MIL", "Milwaukee Bucks", "East", "Central", "#1E7A44", "#C9B072"),
    ("MIN", 1610612750, 16, "MIN", "Minnesota Timberwolves", "West", "Northwest", "#236192", "#78BE20"),
    ("NOP", 1610612740, 3, "NO", "New Orleans Pelicans", "West", "Southwest", "#B4975A", "#0C2340"),
    ("NYK", 1610612752, 18, "NY", "New York Knicks", "East", "Atlantic", "#F58426", "#006BB6"),
    ("OKC", 1610612760, 25, "OKC", "Oklahoma City Thunder", "West", "Northwest", "#007AC1", "#EF3B24"),
    ("ORL", 1610612753, 19, "ORL", "Orlando Magic", "East", "Southeast", "#0077C0", "#C4CED4"),
    ("PHI", 1610612755, 20, "PHI", "Philadelphia 76ers", "East", "Atlantic", "#006BB6", "#ED174C"),
    ("PHX", 1610612756, 21, "PHX", "Phoenix Suns", "West", "Pacific", "#E56020", "#1D1160"),
    ("POR", 1610612757, 22, "POR", "Portland Trail Blazers", "West", "Northwest", "#E03A3E", "#2E2E2E"),
    ("SAC", 1610612758, 23, "SAC", "Sacramento Kings", "West", "Pacific", "#5A2D81", "#63727A"),
    ("SAS", 1610612759, 24, "SA", "San Antonio Spurs", "West", "Southwest", "#8A8D8F", "#2E2E2E"),
    ("TOR", 1610612761, 28, "TOR", "Toronto Raptors", "East", "Atlantic", "#CE1141", "#2E2E2E"),
    ("UTA", 1610612762, 26, "UTAH", "Utah Jazz", "West", "Northwest", "#F9A01B", "#002B5C"),
    ("WAS", 1610612764, 27, "WSH", "Washington Wizards", "East", "Southeast", "#E31837", "#002B5C"),
]

TEAMS: dict[str, Team] = {r[0]: Team(*r) for r in _ROWS}
CODES: tuple[str, ...] = tuple(sorted(TEAMS))

# Extra spellings seen in feeds. Keys are compared upper-cased and with
# spaces/periods removed, so "L.A. Clippers" and "LAClippers" both resolve.
_EXTRA_ALIASES = {
    "BRK": "BKN", "CHO": "CHA", "PHO": "PHX",            # Basketball-Reference
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",  # ESPN / Kaggle
    "UTAH": "UTA", "WSH": "WAS", "NOR": "NOP", "UTH": "UTA",
    "LACLIPPERS": "LAC", "LOSANGELESCLIPPERS": "LAC",
    "SEA": "OKC", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "CHH": "CHA",  # relocations
}


def _key(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


_ALIAS: dict[str, str] = {}
for _t in TEAMS.values():
    _ALIAS[_key(_t.code)] = _t.code
    _ALIAS[_key(_t.espn_abbr)] = _t.code
    _ALIAS[_key(_t.full_name)] = _t.code
for _a, _c in _EXTRA_ALIASES.items():
    _ALIAS[_key(_a)] = _c

_BY_TEAM_ID = {t.team_id: t.code for t in TEAMS.values()}
_BY_ESPN_ID = {t.espn_id: t.code for t in TEAMS.values()}


def canonical(value) -> str | None:
    """NBA tricode for any known spelling; None for anything else."""
    if value is None:
        return None
    return _ALIAS.get(_key(value))


def from_team_id(team_id) -> str | None:
    try:
        return _BY_TEAM_ID.get(int(team_id))
    except (TypeError, ValueError):
        return None


def from_espn_id(espn_id) -> str | None:
    try:
        return _BY_ESPN_ID.get(int(espn_id))
    except (TypeError, ValueError):
        return None


def get(code: str) -> Team:
    c = canonical(code)
    if c is None:
        raise KeyError(f"unknown team {code!r}")
    return TEAMS[c]


def logo_url(code: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{get(code).team_id}/global/L/logo.svg"


def headshot_url(person_id, size: str = "260x190") -> str:
    """NBA headshot. The CDN returns 200 with a silhouette for an unknown id
    (never a 404), so an onerror handler can never detect a missing photo."""
    return f"https://cdn.nba.com/headshots/nba/latest/{size}/{int(person_id)}.png"
