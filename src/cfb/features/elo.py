"""Elo ratings for every college football team, updated one game at a time.

Plain-language version
----------------------
Every team starts with a score (1500 for big-school FBS teams, 1200 for smaller
schools). After each game the winner takes points from the loser. Beating a
strong team, or winning by a lot, earns more points. Before each season every
team drifts part of the way back toward average, because players graduate.

Before a game we read both scores and turn the gap into a win chance:
    P(home wins) = 1 / (1 + 10 ** (-(home_elo + home_edge - away_elo) / 400))

Nothing about a game is used until AFTER we record the pregame ratings for it,
so the ratings can never "peek" at the result.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FBS_MEAN, OTHER_MEAN = 1500.0, 1200.0


@dataclass(frozen=True)
class EloParams:
    k: float = 25.0            # how fast ratings move after one game
    hfa: float = 55.0          # home-field edge, in Elo points
    revert: float = 0.33       # share pulled back to average each offseason
    mov: bool = True           # bigger wins move ratings more (with diminishing returns)


def _expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def run_elo(games: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """Walk through games in time order. Returns pregame ratings + win prob per game.

    ``games`` must be sorted by time and contain: game_id, season, home_id, away_id,
    home_division, away_division, neutral_site, completed, margin.
    Games that are not completed get pregame ratings but cause no update.
    """
    rating: dict[int, float] = {}
    is_fbs: dict[int, bool] = {}
    season_seen: dict[int, int] = {}
    out_h = np.empty(len(games))
    out_a = np.empty(len(games))
    out_p = np.empty(len(games))

    cols = ["season", "home_id", "away_id", "home_division", "away_division",
            "neutral_site", "completed", "margin"]
    for i, (season, h, a, hdiv, adiv, neutral, done, margin) in enumerate(
            games[cols].itertuples(index=False, name=None)):
        for team, div in ((h, hdiv), (a, adiv)):
            fbs_now = div == "fbs"
            if team not in rating:
                rating[team] = FBS_MEAN if fbs_now else OTHER_MEAN
                season_seen[team] = season
            elif season_seen[team] != season:               # new season: regress to mean
                mean = FBS_MEAN if fbs_now else OTHER_MEAN
                gap = season - season_seen[team]
                keep = (1.0 - p.revert) ** gap
                rating[team] = mean + keep * (rating[team] - mean)
                season_seen[team] = season
            is_fbs[team] = fbs_now
        edge = 0.0 if neutral else p.hfa
        rh, ra = rating[h], rating[a]
        prob = _expected(rh + edge - ra)
        out_h[i], out_a[i], out_p[i] = rh, ra, prob
        if not done or margin != margin:                     # skip unplayed / NaN
            continue
        result = 1.0 if margin > 0 else 0.0
        mult = 1.0
        if p.mov:
            winner_diff = (rh + edge - ra) if margin > 0 else (ra - rh - edge)
            mult = math.log(abs(margin) + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))
        delta = p.k * mult * (result - prob)
        rating[h] = rh + delta
        rating[a] = ra - delta

    return pd.DataFrame({"game_id": games["game_id"].values, "elo_home": out_h,
                         "elo_away": out_a, "elo_prob": out_p})


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_elo(games: pd.DataFrame, seasons: list[int]) -> tuple[EloParams, pd.DataFrame]:
    """Grid-search Elo settings using ONLY the given (early) seasons for scoring."""
    grid = itertools.product([25, 30, 35, 40, 50, 60], [45, 55, 65, 75], [0.1, 0.15, 0.2, 0.3])
    mask = games["season"].isin(seasons) & games["completed"] & games["fbs_involved"]
    y = games.loc[mask, "home_win"].to_numpy()
    rows = []
    for k, hfa, rev in grid:
        p = EloParams(k=k, hfa=hfa, revert=rev)
        res = run_elo(games, p)
        rows.append({"k": k, "hfa": hfa, "revert": rev,
                     "log_loss": log_loss(y, res.loc[mask.values, "elo_prob"].to_numpy())})
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0]
    return EloParams(k=float(best.k), hfa=float(best.hfa), revert=float(best.revert)), table
