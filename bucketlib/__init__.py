"""BucketValue: NBA per-game player projections.

Sibling of DiamondValue (mlblib) and GridironValue (gridlib). Modules:
  util      ET dates, season labels, safe display
  cache     atomic writes, embedded fetch stamps, READ_ONLY
  http      one client for every source, with failure classes
  teams     30 teams, every alias, ids, colours
  names     name -> PERSON_ID at the id-less boundaries
  sources/  one module per external source

CODE_QUICK and CODE_DIGEST record the source this package was imported from,
so the app's freshen.ensure() can tell a stale package from the code on disk
after a deploy (see freshen.py). Without the app root on sys.path there is no
app to refresh, and both are None.
"""
try:
    from freshen import code_digest as _code_digest
    from freshen import code_quick as _code_quick
    CODE_QUICK, CODE_DIGEST = _code_quick(), _code_digest()
except (ImportError, OSError):
    CODE_QUICK = CODE_DIGEST = None
