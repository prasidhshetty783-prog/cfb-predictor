"""EXP-007: should picks for games several weeks away be less sure?

Plain-language version
----------------------
Today's pick for a game 5 weeks away uses today's ratings. But 5 more weeks of results
will move both teams' ratings, so that pick is shakier than a pick for this Saturday.
The fix: add the expected rating drift to the uncertainty.

    win chance = bell curve( predicted margin / sqrt(sigma^2 + weeks_ahead * d^2) )

``sigma`` is the usual game-to-game miss (~16 points). ``d`` is how much the matchup can
drift per week. d = 0 means "no softening" (what we do now).

How we test it (same rules as always): for every season, freeze the teams' scouting reports
at week W, predict the games 0-5 weeks later, and compare with what happened.
d is chosen on 2010-2013 only; the exam is 2014-2025.

Run:  python scripts/exp007_soften.py   ->  reports/experiments/exp007/
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb.config import FEATURE_START, RANDOM_SEED, REPORTS, TEST_SEASONS, TUNE_SEASONS
from cfb.eval.analyze import _cluster_boot_idx
from cfb.eval.backtest import load_table
from cfb.models.predictor import Predictor
from cfb.sim.matchup import latest_state, load_states, matchup_rows, slot_coefs

OUT = REPORTS / "experiments" / "exp007"
CUTOFFS = range(2, 12)          # freeze the scouting reports before week W ...
AHEAD = range(0, 6)             # ... and predict games 0-5 weeks later
D_GRID = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def margins_for_season(season: int, table: pd.DataFrame, states, coefs, elo_hfa) -> pd.DataFrame:
    model = Predictor("margin_ridge", {"alpha": 1000.0}).fit(
        table[table["season"].between(FEATURE_START, season - 1)])
    games = table[(table["season"] == season) & (table["season_type"] == "regular")]
    out = []
    for w in CUTOFFS:
        st = latest_state(states, season, w)
        hfa = slot_coefs(coefs, season, w)
        g = games[games["week_order"].between(w, w + max(AHEAD))]
        g = g[g["home_id"].isin(st.index) & g["away_id"].isin(st.index)]
        if g.empty:
            continue
        ctx = {"conference_game": g["conference_game"].to_numpy(), "postseason": 0, "playoff": 0,
               "week_order": g["week_order"].to_numpy()}
        rows = matchup_rows(st.loc[g["home_id"]], st.loc[g["away_id"]], g["hf"].to_numpy(), ctx, hfa, elo_hfa)
        out.append(pd.DataFrame({"season": season, "cutoff": w, "week_order": g["week_order"].to_numpy(),
                                 "ahead": g["week_order"].to_numpy() - w, "game_id": g["game_id"].to_numpy(),
                                 "margin_pred": model.predict_margin(rows), "sigma": model.sigma,
                                 "home_win": g["home_win"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def prob(df: pd.DataFrame, d: float) -> np.ndarray:
    return norm.cdf(df["margin_pred"] / np.sqrt(df["sigma"] ** 2 + df["ahead"] * d ** 2))


def logloss(y, p) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_table()
    states, coefs, elo_hfa = load_states()
    df = pd.concat([margins_for_season(s, table, states, coefs, elo_hfa)
                    for s in TUNE_SEASONS + TEST_SEASONS], ignore_index=True)
    df.to_parquet(OUT / "margins.parquet", index=False)

    tune = df[df["season"].isin(TUNE_SEASONS) & (df["ahead"] >= 1)]
    grid = pd.DataFrame([{"d": d, "tune_log_loss": float(logloss(tune["home_win"], prob(tune, d)).mean())}
                         for d in D_GRID])
    grid.to_csv(OUT / "tuning.csv", index=False)
    d_best = float(grid.loc[grid["tune_log_loss"].idxmin(), "d"])
    print(grid.round(5).to_string(index=False))
    print("chosen d:", d_best)

    test = df[df["season"].isin(TEST_SEASONS)].copy()
    test["p_now"], test["p_soft"] = prob(test, 0.0), prob(test, d_best)
    rows = []
    for k, g in test.groupby("ahead"):
        y = g["home_win"].to_numpy()
        for name in ("p_now", "p_soft"):
            p = g[name].to_numpy()
            conf = np.maximum(p, 1 - p)
            rows.append({"weeks_ahead": int(k), "version": "current" if name == "p_now" else "softened",
                         "games": len(g), "log_loss": float(logloss(y, p).mean()),
                         "accuracy": float(np.mean((p >= .5) == y)), "avg_said": float(conf.mean()),
                         "actually_right": float(np.mean(np.where(p >= .5, y, 1 - y)))})
    by_k = pd.DataFrame(rows)
    by_k.to_csv(OUT / "by_weeks_ahead.csv", index=False)
    print(by_k.round(4).to_string(index=False))

    # paired bootstrap over game-weeks (cutoff x season clusters), games 1-5 weeks ahead
    t = test[test["ahead"] >= 1]
    diff_games = logloss(t["home_win"].to_numpy(), t["p_soft"].to_numpy()) - \
        logloss(t["home_win"].to_numpy(), t["p_now"].to_numpy())
    clusters = (t["season"] * 10000 + t["cutoff"] * 100 + t["week_order"]).to_numpy()
    boot = np.array([diff_games[i].mean() for i in _cluster_boot_idx(clusters, np.random.default_rng(RANDOM_SEED))])
    p_one = float(np.mean(boot >= 0))
    summary = {"d": d_best, "log_loss_change_1to5_weeks_ahead": float(diff_games.mean()),
               "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
               "p_value": min(1.0, 2 * min(p_one, 1 - p_one)), "games": int(len(t))}
    if d_best == 0.0:                       # nothing to compare: the tuning picked "no softening"
        summary.update({"p_value": None, "ci_low": None, "ci_high": None})
    summary["significantly_better"] = bool(summary["log_loss_change_1to5_weeks_ahead"] < 0 and summary["p_value"] < 0.05)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
