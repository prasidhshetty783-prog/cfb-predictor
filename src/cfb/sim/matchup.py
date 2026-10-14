"""Build model-ready rows for matchups that may never have happened.

Plain-language version: to ask "who wins Georgia vs. Ohio State in the title game?"
we take each team's latest scouting report (ratings, talent, form) and combine them
exactly the way the feature builder does for real games: differences "A minus B",
plus the game situation (neutral field, playoff, week).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from cfb.config import PROCESSED
from cfb.features.build import MODEL_FEATURES, TEAM_FEATURES, _haversine_km

RATING_TARGETS = ("pts", "sr", "ypp", "pw")


def load_states() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    states = pd.read_parquet(PROCESSED / "team_states.parquet")
    coefs = pd.read_parquet(PROCESSED / "rating_coefs.parquet")
    meta = json.loads((PROCESSED / "features_meta.json").read_text())
    return states, coefs, float(meta["elo_params"]["hfa"])


def latest_state(states: pd.DataFrame, season: int, week_order: int) -> pd.DataFrame:
    """Each team's scouting report from its game in (or most recent before) ``week_order``."""
    s = states[(states["season"] == season) & (states["week_order"] <= week_order)]
    s = s.sort_values(["week_order", "start_date"]).drop_duplicates("team_id", keep="last")
    return s.set_index("team_id")


def slot_coefs(coefs: pd.DataFrame, season: int, week_order: int) -> dict[str, float]:
    c = coefs[(coefs["season"] == season) & (coefs["week_order"] <= week_order)]
    c = c[c["week_order"] == c["week_order"].max()]
    return dict(zip(c["target"], c["hfa"]))


def matchup_rows(A: pd.DataFrame, B: pd.DataFrame, hf: np.ndarray, ctx: dict,
                 hfa: dict[str, float], elo_hfa: float) -> pd.DataFrame:
    """Rows of model features for team A (home slot) vs team B (away slot).

    ``A``/``B`` are state rows aligned one-to-one; ``hf`` is 1 if A plays at home, 0 if neutral.
    ``ctx`` holds conference_game, postseason, playoff, week_order.
    """
    A = A.reset_index(drop=True)
    B = B.reset_index(drop=True)
    hf = np.asarray(hf, dtype=float)
    out = pd.DataFrame(index=A.index)
    for f in TEAM_FEATURES:
        out[f"d_{f}"] = A[f].to_numpy() - B[f].to_numpy()
    out["hf"] = hf
    for k in ("conference_game", "postseason", "playoff", "week_order"):
        out[k] = ctx[k]
    out["travel_km"] = np.where(hf == 1, _haversine_km(A["lat"], A["lon"], B["lat"], B["lon"]),
                                np.nan)
    for t in RATING_TARGETS:
        out[f"pred_{t}"] = (A[f"r_{t}"].to_numpy() - B[f"r_{t}"].to_numpy()) + hfa.get(t, 0.0) * hf
    out["elo_logit"] = (A["elo"].to_numpy() + elo_hfa * hf - B["elo"].to_numpy()) * math.log(10) / 400
    out["home_talent"] = A["talent"].to_numpy()
    out["away_talent"] = B["talent"].to_numpy()
    out["home_is_power"] = A["is_power"].to_numpy()
    out["away_is_power"] = B["is_power"].to_numpy()
    out["elo_prob"] = 1 / (1 + np.exp(-out["elo_logit"]))
    return out[MODEL_FEATURES + ["elo_prob"]]
