"""The simulator's hypothetical-matchup builder must reproduce the real feature rows exactly.

Plain-language version: if we rebuild a REAL game from the two teams' saved scouting
reports, we must get the very same numbers the feature builder produced. If not, the
playoff simulator would be asking the model about a slightly different game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.config import PROCESSED
from cfb.features.build import MODEL_FEATURES
from cfb.sim.matchup import load_states, matchup_rows, slot_coefs


def test_matchup_rows_match_real_games():
    table = pd.read_parquet(PROCESSED / "features.parquet")
    states, coefs, elo_hfa = load_states()
    real = table[table["fbs_both"] & table["completed"] & table["season"].between(2015, 2025)
                 & (table["season_type"] == "postseason")].head(150)
    checked = 0
    for _, r in real.iterrows():
        st = states[states["game_id"] == r["game_id"]].set_index("side")
        A, B = st.loc[["home"]], st.loc[["away"]]
        ctx = {k: r[k] for k in ("conference_game", "postseason", "playoff", "week_order")}
        rows = matchup_rows(A, B, np.array([r["hf"]]), ctx,
                            slot_coefs(coefs, r["season"], r["week_order"]), elo_hfa)
        got = rows[MODEL_FEATURES].iloc[0].to_numpy(dtype=float)
        want = r[MODEL_FEATURES].to_numpy(dtype=float)
        assert np.allclose(got, want, equal_nan=True, atol=1e-8), r["game_id"]
        checked += 1
    assert checked >= 100
