"""EXP-005: do luck clues (#1) and position-talent clues (#2) make the winning model better?

Plain-language version: we take the current winner (margin ridge), give it the new clues, and
re-run the exact same exam (tune on 2010-2013, predict 2014-2025 one season at a time). Then
we compare it with the current model on the SAME games, game-week by game-week, 2,000 times
(bootstrap). A version is only "better" if the improvement is bigger than luck could explain.

Run:  python scripts/exp005_luck_talent.py   ->  reports/experiments/exp005/
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
from cfb.features.extra import LUCK_FEATURES, TALENT_FEATURES
from cfb.models.zoo import BY_NAME

OUT = REPORTS / "experiments" / "exp005"
BASE = BY_NAME["margin_ridge"]
VARIANTS = {
    "current (40 clues)": MODEL_FEATURES,
    "+ luck (#1)": MODEL_FEATURES + LUCK_FEATURES,
    "+ position talent (#2)": MODEL_FEATURES + TALENT_FEATURES,
    "+ both": MODEL_FEATURES + LUCK_FEATURES + TALENT_FEATURES,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_table()
    preds, params = [], {}
    for name, feats in VARIANTS.items():
        spec = replace(BASE, name=name, features=list(feats))
        best, _ = tune(spec, table)
        params[name] = best
        preds.append(walk_forward(spec, best, table))
        print(f"{name:26s} alpha={best}", flush=True)
    p = pd.concat(preds, ignore_index=True)
    meta = table[["game_id", "season", "week_order", "home_win", "home_spread"]]
    df = p.merge(meta, on=["game_id", "season"])
    df.to_parquet(OUT / "predictions.parquet", index=False)

    rows = []
    for subset, cond in {"all games": lambda d: d.index == d.index,
                         "weeks 1-4": lambda d: d["week_order"] <= 4,
                         "week 5 on": lambda d: d["week_order"] > 4}.items():
        for name in VARIANTS:
            d = df[df["model"] == name]
            d = d[cond(d)]
            r = score(d["home_win"].to_numpy(), d["p_home"].to_numpy(), d["home_spread"].to_numpy())
            rows.append({"subset": subset, "variant": name, **r})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    # paired bootstrap vs the current model, clustered by game-week, Holm-corrected
    wide = df.pivot_table(index=["game_id", "season", "week_order", "home_win"], columns="model",
                          values="p_home")
    y = wide.index.get_level_values("home_win").to_numpy(float)
    P = np.clip(wide.to_numpy(), 1e-6, 1 - 1e-6)
    L = -(y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P))
    names = list(wide.columns)
    clusters = (wide.index.get_level_values("season").astype(int) * 100
                + wide.index.get_level_values("week_order").astype(int)).to_numpy()
    boot = np.stack([L[i].mean(axis=0) for i in _cluster_boot_idx(clusters, np.random.default_rng(RANDOM_SEED))])
    b = names.index("current (40 clues)")
    sig = []
    for j, n in enumerate(names):
        if j == b:
            continue
        diff = boot[:, j] - boot[:, b]                      # < 0 means the new version is better
        p_one = float(np.mean(diff >= 0))
        sig.append({"variant": n, "log_loss_change": float(L[:, j].mean() - L[:, b].mean()),
                    "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5)),
                    "p_value": min(1.0, 2 * min(p_one, 1 - p_one))})
    sig = pd.DataFrame(sig).sort_values("p_value").reset_index(drop=True)
    k, running = len(sig), 0.0
    holm = []
    for i, pv in enumerate(sig["p_value"]):
        running = max(running, min(1.0, (k - i) * pv))
        holm.append(running)
    sig["p_holm"] = holm
    sig["significantly_better"] = (sig["log_loss_change"] < 0) & (sig["p_holm"] < 0.05)
    sig.to_csv(OUT / "significance.csv", index=False)
    (OUT / "params.json").write_text(json.dumps(params, indent=1))

    show = metrics[metrics["subset"] == "all games"][["variant", "log_loss", "accuracy", "recall_home",
                                                       "recall_away", "ece", "acc_when_80pct_sure",
                                                       "upset_recall"]]
    print(show.round(4).to_string(index=False))
    print(metrics[metrics["subset"] == "weeks 1-4"][["variant", "log_loss", "accuracy"]].round(4).to_string(index=False))
    print(sig.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
