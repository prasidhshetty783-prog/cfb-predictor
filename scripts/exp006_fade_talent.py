"""EXP-006: position talent that counts early in the season and fades as games are played.

Plain-language version: EXP-005 showed quarterback / offensive-line talent helps in the
first weeks (when we know little about a team) but hurts later (when the ratings know more).
So each position-talent clue is multiplied by a fade factor:

    fade = 1 / (1 + games_so_far / h)       games_so_far ~ week of the season minus 1

A small h fades fast; a big h fades slowly. h is chosen using 2010-2013 ONLY (pre-registered
before this test), then the exam is the same as always: 2014-2025, one season at a time,
paired bootstrap by game-week against the current model.

Run:  python scripts/exp006_fade_talent.py   ->  reports/experiments/exp006/
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from cfb.config import RANDOM_SEED, REPORTS
from cfb.eval.analyze import _cluster_boot_idx
from cfb.eval.backtest import load_table, tune, walk_forward
from cfb.eval.metrics import score
from cfb.features.build import MODEL_FEATURES
from cfb.features.extra import TALENT_FEATURES
from cfb.models.zoo import BY_NAME

OUT = REPORTS / "experiments" / "exp006"
H_GRID = [1.0, 2.0, 4.0, 8.0]
FADED = [f + "_fade" for f in TALENT_FEATURES]


def add_fade(t: pd.DataFrame, h: float) -> pd.DataFrame:
    t = t.copy()
    games = (t["week_order"].clip(lower=1, upper=14) - 1).astype(float)
    fade = 1.0 / (1.0 + games / h)
    for f in TALENT_FEATURES:
        t[f + "_fade"] = t[f] * fade
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_table = load_table()
    base = BY_NAME["margin_ridge"]
    new = replace(base, name="+ faded position talent", features=MODEL_FEATURES + FADED)

    # 1) choose h on 2010-2013 only
    tune_rows = []
    for h in H_GRID:
        params, rows = tune(new, add_fade(base_table, h))
        best = min(r["tune_log_loss"] for r in rows)
        tune_rows.append({"h": h, "alpha": params["alpha"], "tune_log_loss": best})
        print(f"h={h:<4} alpha={params['alpha']:<7} tune log loss={best:.5f}", flush=True)
    tr = pd.DataFrame(tune_rows).sort_values("tune_log_loss")
    tr.to_csv(OUT / "tuning.csv", index=False)
    h, alpha = float(tr.iloc[0]["h"]), float(tr.iloc[0]["alpha"])
    cur_params, _ = tune(base, base_table)

    # 2) the exam
    table = add_fade(base_table, h)
    p = pd.concat([walk_forward(replace(base, name="current (40 clues)"), cur_params, table),
                   walk_forward(new, {"alpha": alpha}, table)], ignore_index=True)
    df = p.merge(table[["game_id", "season", "week_order", "home_win", "home_spread"]], on=["game_id", "season"])
    df.to_parquet(OUT / "predictions.parquet", index=False)

    rows = []
    subsets = {"all games": lambda d: d["week_order"] > -1, "weeks 1-4": lambda d: d["week_order"] <= 4,
               "week 5 on": lambda d: d["week_order"] > 4}
    for sub, cond in subsets.items():
        for m in df["model"].unique():
            d = df[(df["model"] == m)]
            d = d[cond(d)]
            rows.append({"subset": sub, "variant": m, **score(d["home_win"].to_numpy(), d["p_home"].to_numpy(),
                                                              d["home_spread"].to_numpy())})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    sig = []
    for sub, cond in subsets.items():
        w = df[cond(df)].pivot_table(index=["game_id", "season", "week_order", "home_win"], columns="model",
                                     values="p_home")
        y = w.index.get_level_values("home_win").to_numpy(float)
        P = np.clip(w.to_numpy(), 1e-6, 1 - 1e-6)
        L = -(y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P))
        n = list(w.columns)
        cl = (w.index.get_level_values("season").astype(int) * 100
              + w.index.get_level_values("week_order").astype(int)).to_numpy()
        B = np.stack([L[i].mean(0) for i in _cluster_boot_idx(cl, np.random.default_rng(RANDOM_SEED))])
        b, j = n.index("current (40 clues)"), n.index("+ faded position talent")
        d = B[:, j] - B[:, b]
        p_one = float(np.mean(d >= 0))
        sig.append({"subset": sub, "log_loss_change": float(L[:, j].mean() - L[:, b].mean()),
                    "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                    "p_value": min(1.0, 2 * min(p_one, 1 - p_one)), "games": int(len(y))})
    sig = pd.DataFrame(sig)
    main_row = sig[sig["subset"] == "all games"].iloc[0]
    verdict = bool(main_row["log_loss_change"] < 0 and main_row["p_value"] < 0.05)
    sig.to_csv(OUT / "significance.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"h": h, "alpha": alpha, "significantly_better": verdict}, indent=1))

    print(metrics[["subset", "variant", "log_loss", "accuracy", "recall_home", "recall_away", "ece",
                   "acc_when_80pct_sure"]].round(4).to_string(index=False))
    print(sig.round(5).to_string(index=False))
    print("chosen h:", h, "| significantly better overall:", verdict)


if __name__ == "__main__":
    main()
