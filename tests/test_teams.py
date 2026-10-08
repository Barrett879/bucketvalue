from bucketlib import teams


def test_thirty_teams_unique_ids():
    assert len(teams.TEAMS) == 30
    assert len({t.team_id for t in teams.TEAMS.values()}) == 30
    assert sorted(t.espn_id for t in teams.TEAMS.values()) == list(range(1, 31))


def test_aliases():
    for alias, code in [("GS", "GSW"), ("utah", "UTA"), ("WSH", "WAS"), ("NO", "NOP"),
                        ("ny", "NYK"), ("SA", "SAS"), ("BRK", "BKN"), ("PHO", "PHX"),
                        ("LAClippers", "LAC"), ("Los Angeles Clippers", "LAC"),
                        ("PortlandTrailBlazers", "POR"), ("Memphis Grizzlies", "MEM")]:
        assert teams.canonical(alias) == code, alias


def test_unknown_is_none_never_passthrough():
    for junk in ("<script>", "XYZ", "", None, "LON", "London Lions"):
        assert teams.canonical(junk) is None


def test_id_maps():
    assert teams.from_team_id(1610612747) == "LAL"
    assert teams.from_espn_id(26) == "UTA"
    assert teams.from_team_id("nope") is None
