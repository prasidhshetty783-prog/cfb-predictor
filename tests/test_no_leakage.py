"""Time-travel test: features for week W must not change when results from week W on are hidden.

Plain-language version: we pretend we are living on the morning of a game week. We erase
every score from that week onward and rebuild the scouting reports. If any report for that
week changes, the model was secretly peeking at the future. This test must always pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb.config import LEAKY_COLUMNS
from cfb.features.build import MODEL_FEATURES, _form, _game_frame, _team_long
from cfb.features.elo import EloParams, run_elo
from cfb.features.extra import EXTRA_FEATURES, extra_form
from cfb.features.ratings import RidgeParams, sequential_ratings

CUT_SEASON, CUT_WEEK = 2013, 8


@pytest.fixture(scope="module")
def frames():
    g, st = _game_frame()
    g = g[g["season"].between(2011, 2013)].reset_index(drop=True)
    cut = (g["season"] > CUT_SEASON) | ((g["season"] == CUT_SEASON) & (g["week_order"] >= CUT_WEEK))
    hidden = g.copy()
    hidden.loc[cut, ["completed"]] = False
    hidden.loc[cut, ["margin", "home_win", "pts_target", "sr_target", "ypp_target",
                     "pw_target"]] = np.nan
    st_hidden = st[~st["game_id"].isin(g.loc[cut, "game_id"])]
    slot = (g["season"] == CUT_SEASON) & (g["week_order"] == CUT_WEEK)
    return g, hidden, st, st_hidden, slot


def test_no_outcome_columns_are_features():
    assert not (set(MODEL_FEATURES) & LEAKY_COLUMNS)


def test_elo_ignores_future(frames):
    g, hidden, _, _, slot = frames
    a = run_elo(g, EloParams())[slot.values]
    b = run_elo(hidden, EloParams())[slot.values]
    assert np.allclose(a[["elo_home", "elo_away", "elo_prob"]], b[["elo_home", "elo_away", "elo_prob"]])


def test_ratings_ignore_future(frames):
    g, hidden, _, _, slot = frames
    params = {"pts_target": RidgeParams(lam=1.0), "sr_target": RidgeParams(lam=1.0, cap=None)}
    a = sequential_ratings(g, params)[slot.values].drop(columns="game_id")
    b = sequential_ratings(hidden, params)[slot.values].drop(columns="game_id")
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_form_ignores_future(frames):
    g, hidden, st, st_hidden, slot = frames
    ids = set(g.loc[slot, "game_id"])
    a = _form(_team_long(g, st))
    b = _form(_team_long(hidden, st_hidden))
    a = a[a["game_id"].isin(ids)].sort_values(["game_id", "team_id"])
    b = b[b["game_id"].isin(ids)].sort_values(["game_id", "team_id"])
    num = a.columns.drop(["game_id", "team_id"])
    assert np.allclose(a[num].to_numpy(), b[num].to_numpy(), equal_nan=True)


def test_luck_clues_ignore_future(frames):
    """EXP-005 luck clues (season luck, close-game record) use earlier games only."""
    g, hidden, st, st_hidden, slot = frames
    ids = set(g.loc[slot, "game_id"])
    a = extra_form(_team_long(g, st))
    b = extra_form(_team_long(hidden, st_hidden))
    a = a[a["game_id"].isin(ids)].sort_values(["game_id", "team_id"])
    b = b[b["game_id"].isin(ids)].sort_values(["game_id", "team_id"])
    assert np.allclose(a[["s_luck", "s_close"]].to_numpy(), b[["s_luck", "s_close"]].to_numpy())
    assert not (set(EXTRA_FEATURES) & LEAKY_COLUMNS)


def test_test_is_sensitive(frames):
    """Sanity check of the test itself: hiding results DOES change the NEXT week's features."""
    g, hidden, _, _, _ = frames
    nxt = ((g["season"] == CUT_SEASON) & (g["week_order"] == CUT_WEEK + 1)).values
    a = run_elo(g, EloParams())[nxt]
    b = run_elo(hidden, EloParams())[nxt]
    assert not np.allclose(a["elo_home"], b["elo_home"])


def test_poll_is_published_before_games():
    """Teams that lose in week W drop between poll W and poll W+1, not between W-1 and W,
    so the poll labeled W is published BEFORE week W games (safe to use for week W)."""
    from cfb.config import INTERIM
    g = pd.read_parquet(INTERIM / "games.parquet")
    p = pd.read_parquet(INTERIM / "polls.parquet")
    ap = p[(p["poll"] == "ap") & (p["season_type"] == 2)][["season", "week", "team_id", "rank"]]
    reg = g[(g["season_type"] == "regular") & g["completed"] & (g["season"] >= 2010)]
    lost = pd.concat([
        reg.loc[reg["margin"] < 0, ["season", "week", "home_id"]].rename(columns={"home_id": "team_id"}),
        reg.loc[reg["margin"] > 0, ["season", "week", "away_id"]].rename(columns={"away_id": "team_id"}),
    ])

    def rank(shift: int) -> pd.Series:
        q = ap.assign(week=ap["week"] - shift)
        return lost.merge(q, on=["season", "week", "team_id"], how="left")["rank"].to_numpy()

    d = pd.DataFrame({"prev": rank(-1), "same": rank(0), "next": rank(1)}).dropna()
    assert len(d) > 300
    assert (d["next"] - d["same"]).mean() > 3.0          # the loss shows up in the NEXT poll
    assert abs((d["same"] - d["prev"]).mean()) < 1.5     # ...not in the same-week poll
