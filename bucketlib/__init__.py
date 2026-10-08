"""BucketValue: NBA per-game player projections.

Sibling of DiamondValue (mlblib) and GridironValue (gridlib). Modules:
  util      ET dates, season labels, safe display
  cache     atomic writes, embedded fetch stamps, READ_ONLY
  http      one client for every source, with failure classes
  teams     30 teams, every alias, ids, colours
  names     name -> PERSON_ID at the id-less boundaries
  sources/  one module per external source
"""
