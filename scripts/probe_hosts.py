"""Can THIS machine reach every source BucketValue needs? Prints and saves a table.

Run on GitHub Actions (.github/workflows/probe.yml) and, once deployed, from the
Streamlit Cloud host. Every live-verified fact in docs/data_notes.md came from a
VPN exit in New York; reputationally stats.nba.com hangs and cdn.nba.com 403s for
cloud IP ranges, and the daily refresh design depends on which is true from the
hosts that will actually run it. So it is measured, not assumed.

Each target goes through bucketlib.http, the same client the bot will use, with
retries off and a short timeout, so a hang shows up as Blocked in ~20s rather
than hanging the job. A plain python-requests call to stats.nba.com is included
as a control (it hangs from every host tested so far).

    python scripts/probe_hosts.py --out docs/probes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import http as H  # noqa: E402
from bucketlib.sources import cdn  # noqa: E402
from bucketlib.util import now_utc  # noqa: E402

TARGETS = [
    # name, url, params, expect
    ("stats_leaguegamelog_T_2025-26", H.STATS_BASE + "leaguegamelog",
     {"Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": "T",
      "Season": "2025-26", "SeasonType": "Regular Season", "Sorter": "DATE",
      "DateFrom": "", "DateTo": ""}, "json"),
    ("stats_playerindex_2026-27", H.STATS_BASE + "playerindex",
     {"Historical": 0, "LeagueID": "00", "Season": "2026-27", "TeamID": 0}, "json"),
    ("stats_commonteamroster_DEN_2026-27", H.STATS_BASE + "commonteamroster",
     {"LeagueID": "00", "Season": "2026-27", "TeamID": 1610612743}, "json"),
    ("cdn_schedule", cdn.SCHEDULE_URL, None, "json"),
    # The bot's roster source (step_roster). Not in the 2026-10-09 probe: the
    # "identical to stats playerindex" check was made from the Mac.
    ("cdn_player_index", cdn.PLAYER_INDEX_URL, None, "json"),
    ("cdn_todays_scoreboard", "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json", None, "json"),
    ("cdn_boxscore_0022500600", cdn.BOX_URL.format(gid="0022500600"), None, "json"),
    ("cdn_headshot", "https://cdn.nba.com/headshots/nba/latest/260x190/203999.png", None, "bytes"),
    ("injury_pdf_2026-01-15", "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-01-15_05_00PM.pdf", None, "pdf"),
    ("espn_scoreboard_20261020", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
     {"dates": "20261020"}, "json"),
    ("espn_core_odds_final", "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401810831/competitions/401810831/odds",
     None, "json"),
    ("espn_injuries", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries", None, "json"),
]


def _size(payload) -> int:
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    return len(json.dumps(payload))


def probe_one(name, url, params, expect) -> dict:
    t0 = time.monotonic()
    try:
        payload = H.fetch(url, params=params, timeout=20, retries=0, expect=expect)
        return {"name": name, "ok": True, "class": "ok", "seconds": round(time.monotonic() - t0, 2),
                "bytes": _size(payload), "detail": ""}
    except H.FetchError as e:
        return {"name": name, "ok": False, "class": type(e).__name__,
                "seconds": round(time.monotonic() - t0, 2), "bytes": 0, "detail": e.detail[:200]}


def requests_control() -> dict:
    """python-requests against stats.nba.com: the client that hangs."""
    import requests
    t0 = time.monotonic()
    try:
        r = requests.get(H.STATS_BASE + "leaguegamelog", params=TARGETS[0][2],
                         headers=H.STATS_HEADERS, timeout=15)
        return {"name": "control_python_requests_stats", "ok": r.status_code == 200,
                "class": f"HTTP {r.status_code}", "seconds": round(time.monotonic() - t0, 2),
                "bytes": len(r.content), "detail": ""}
    except Exception as e:  # noqa: BLE001
        return {"name": "control_python_requests_stats", "ok": False, "class": type(e).__name__,
                "seconds": round(time.monotonic() - t0, 2), "bytes": 0, "detail": str(e)[:200]}


def egress() -> dict:
    try:
        j = H.fetch("https://ipinfo.io/json", timeout=10, retries=0)
        return {k: j.get(k) for k in ("org", "city", "region", "country")}
    except H.FetchError as e:
        return {"error": e.detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="local")
    args = ap.parse_args()

    rows = [probe_one(*t) for t in TARGETS] + [requests_control()]
    result = {"label": args.label, "at_utc": now_utc().isoformat(timespec="seconds"),
              "egress": egress(), "results": rows}
    for r in rows:
        print(f"{'OK ' if r['ok'] else 'NO '} {r['name']:<38} {r['class']:<14} {r['seconds']:>6}s "
              f"{r['bytes']:>9}B {r['detail']}")
    print("egress:", result["egress"])
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = result["at_utc"].replace(":", "").replace("-", "")
        (out / f"probe_{args.label}_{stamp}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
