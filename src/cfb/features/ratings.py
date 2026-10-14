"""Opponent-adjusted power ratings, re-fit every week using only past games.

Plain-language version
----------------------
Beating a weak team by 30 is not the same as beating a strong team by 30.
These ratings fix that. Every week we solve a puzzle: "what strength number
for every team best explains all the scores so far, after accounting for who
played whom and where?" A rating of +10 means about 10 points better than an
average FBS team on a neutral field.

We solve the puzzle for several signals, not just points:
* ``pts``  - final score margin (capped, so running up the score doesn't count much)
* ``sr``   - success-rate margin (how often each play "stayed on schedule")
* ``ypp``  - yards-per-play margin
* ``pw``   - postgame win expectancy (who *should* have won, given the stats)

Early in a season there are few games, so last season's games are included
with a smaller weight. A "ridge" penalty pulls teams with little data toward
average, so one fluky game can't create a crazy rating.

Every rating for week W is fit on games strictly BEFORE week W.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeParams:
    lam: float = 5.0           # pull toward average (bigger = more caution)
    prev_weight: float = 0.35  # weight of last season's games
    cap: float | None = 28.0   # cap on the target's absolute value (None = no cap)


def _fit_slot(train: pd.DataFrame, target: str, p: RidgeParams) -> tuple[dict[int, float], float, float]:
    """Weighted ridge: y = r_home - r_away + hfa*home_field + fcs*(away_non_fbs - home_non_fbs)."""
    d = train.dropna(subset=[target])
    if len(d) < 30:
        return {}, 0.0, 0.0
    teams = pd.unique(np.concatenate([d["home_id"].to_numpy(), d["away_id"].to_numpy()]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    h = d["home_id"].map(idx).to_numpy()
    a = d["away_id"].map(idx).to_numpy()
    c = d["home_field"].to_numpy(dtype=float)                       # home-field column
    f = (d["away_non_fbs"].to_numpy() - d["home_non_fbs"].to_numpy()).astype(float)
    y = d[target].to_numpy(dtype=float)
    if p.cap is not None:
        y = np.clip(y, -p.cap, p.cap)
    w = d["w"].to_numpy(dtype=float)
    # Normal equations (X'WX + ridge) beta = X'Wy, built from the 4 non-zeros per row.
    H, F = n_t, n_t + 1
    A = np.zeros((n_t + 2, n_t + 2))
    np.add.at(A, (h, h), w)
    np.add.at(A, (a, a), w)
    np.add.at(A, (h, a), -w)
    np.add.at(A, (a, h), -w)
    for col, v in ((H, c), (F, f)):
        np.add.at(A, (h, col), w * v)
        np.add.at(A, (a, col), -w * v)
    A[H:, :n_t] = A[:n_t, H:].T
    A[H, H] = np.sum(w * c * c)
    A[F, F] = np.sum(w * f * f)
    A[H, F] = A[F, H] = np.sum(w * c * f)
    A[np.arange(n_t), np.arange(n_t)] += p.lam                      # ridge on team ratings only
    A[F, F] += 1e-6                                                 # keeps the solve stable
    b = np.zeros(n_t + 2)
    np.add.at(b, h, w * y)
    np.add.at(b, a, -w * y)
    b[H] = np.sum(w * c * y)
    b[F] = np.sum(w * f * y)
    beta = np.linalg.solve(A, b)
    return {t: beta[i] for t, i in idx.items()}, beta[H], beta[F]


def sequential_ratings(games: pd.DataFrame, targets: dict[str, RidgeParams],
                       return_coefs: bool = False):
    """Pregame ratings for every game (completed or not) for each target.

    ``games`` needs: game_id, season, week_order, home_id, away_id, home_field (1/0),
    home_non_fbs, away_non_fbs, completed, and one column per target (home perspective).
    Returns one row per game with  r_<t>_home, r_<t>_away, pred_<t>  for each target.
    """
    g = games.reset_index(drop=True)
    out = {"game_id": g["game_id"].to_numpy()}
    for t in targets:
        out[f"r_{t}_home"] = np.full(len(g), np.nan)
        out[f"r_{t}_away"] = np.full(len(g), np.nan)
        out[f"pred_{t}"] = np.full(len(g), np.nan)
    done = g[g["completed"]]
    coefs = []
    for (season, wk), slot in g.groupby(["season", "week_order"], sort=True):
        cur = done[(done["season"] == season) & (done["week_order"] < wk)]
        prev = done[done["season"] == season - 1]
        train = pd.concat([cur.assign(w=1.0), prev.assign(w=np.nan)], ignore_index=True)
        for t, p in targets.items():
            train["w"] = np.where(train["season"] == season, 1.0, p.prev_weight)
            r, hfa, fcs = _fit_slot(train, t, p)
            coefs.append({"season": season, "week_order": wk, "target": t, "hfa": hfa, "fcs": fcs})
            rh = slot["home_id"].map(r).fillna(0.0).to_numpy()
            ra = slot["away_id"].map(r).fillna(0.0).to_numpy()
            pred = (rh - ra + hfa * slot["home_field"].to_numpy()
                    + fcs * (slot["away_non_fbs"].to_numpy() - slot["home_non_fbs"].to_numpy()))
            ix = slot.index.to_numpy()
            out[f"r_{t}_home"][ix] = rh
            out[f"r_{t}_away"][ix] = ra
            out[f"pred_{t}"][ix] = pred
    if return_coefs:
        return pd.DataFrame(out), pd.DataFrame(coefs)
    return pd.DataFrame(out)
