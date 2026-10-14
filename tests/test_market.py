"""Sportsbook abbreviations must map to the right team, the same way on every computer.

Plain-language version: betting files name teams with short codes ('ARMY', 'LAF'). If a
rare code is matched to the wrong team, its line gets the wrong sign, and the median of
'Army -33.5' and a mis-signed '+33.5' becomes a fake 0-point line. These tests lock in
the fix: ties go to the team no other code already owns, and row order never matters.
"""

from __future__ import annotations

import pandas as pd

from cfb.data.clean import _abbr_to_team

ARMY, LAFAYETTE, NAVY, AIR_FORCE = 1, 2, 3, 4


def _lines() -> pd.DataFrame:
    rows = []
    # 'ARMY' and 'NAVY' appear in many games, so they are easy to match
    for gid, opp in enumerate([10, 11, 12, 13, 14], start=100):
        rows.append({"game_id": gid, "abbr": "ARMY", "home_team_id": ARMY, "away_team_id": opp})
        rows.append({"game_id": gid + 50, "abbr": "NAVY", "home_team_id": NAVY, "away_team_id": opp + 10})
    # 'LAF' appears once, in Army vs Lafayette: a tie between Army and Lafayette
    rows.append({"game_id": 200, "abbr": "ARMY", "home_team_id": ARMY, "away_team_id": LAFAYETTE})
    rows.append({"game_id": 200, "abbr": "LAF", "home_team_id": ARMY, "away_team_id": LAFAYETTE})
    # 'XYZ' appears once, in Army vs Navy: both teams are already owned -> must be skipped
    rows.append({"game_id": 300, "abbr": "XYZ", "home_team_id": ARMY, "away_team_id": NAVY})
    # 'AF' appears once, vs Navy: Navy is owned, so 'AF' is Air Force
    rows.append({"game_id": 400, "abbr": "AF", "home_team_id": AIR_FORCE, "away_team_id": NAVY})
    return pd.DataFrame(rows)


def test_ties_go_to_the_team_nobody_owns():
    m = _abbr_to_team(_lines())
    assert m["ARMY"] == ARMY and m["NAVY"] == NAVY
    assert m["LAF"] == LAFAYETTE
    assert m["AF"] == AIR_FORCE
    assert "XYZ" not in m                       # unresolvable tie: skipped, never guessed


def test_row_order_never_changes_the_answer():
    base = _abbr_to_team(_lines())
    for seed in range(5):
        shuffled = _lines().sample(frac=1.0, random_state=seed).reset_index(drop=True)
        assert _abbr_to_team(shuffled) == base
