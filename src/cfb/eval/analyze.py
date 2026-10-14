"""Turn backtest predictions into a scorecard, confidence intervals and significance tests.

Plain-language version
----------------------
* **Scorecard**: every metric for every model on the 2014-2025 test games.
* **Ensemble**: the 3 best learners *from the tuning seasons* averaged together
  (chosen without looking at the test seasons).
* **Confidence intervals**: we re-run the season thousands of times by resampling
  whole game-weeks (bootstrap). If a model's score barely moves, we trust it.
* **Head-to-head tests**: is the leader really better than each rival on the *same*
  games, or could the gap be luck? Holm's correction keeps us honest when we run
  many comparisons at once.

Run:  python -m cfb.eval.analyze
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb.eval.backtest import OUT, load_table
from cfb.eval.metrics import ece, score

LEARNER_FAMILIES = {"linear", "tree", "neural", "kernel", "margin", "instance", "probabilistic"}
N_BOOT = 2000
SEED = 7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def add_ensemble(preds: pd.DataFrame, tuning: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, list[str]]:
    from cfb.models.zoo import BY_NAME

    best = tuning.sort_values("tune_log_loss").drop_duplicates("model")
    best = best[best["model"].map(lambda m: BY_NAME[m].family in LEARNER_FAMILIES
                                  if m in BY_NAME else False)]
    members = best.head(k)["model"].tolist()
    wide = preds[preds["model"].isin(members)].pivot_table(
        index=["game_id", "season"], columns="model", values="p_home").dropna()
    z = _logit(wide[members].to_numpy()).mean(axis=1)
    ens = pd.DataFrame({"game_id": wide.index.get_level_values(0),
                        "season": wide.index.get_level_values(1), "model": "ensemble_top3",
                        "p_home": 1 / (1 + np.exp(-z)), "pred_margin": np.nan})
    return pd.concat([preds, ens], ignore_index=True), members


def _cluster_boot_idx(clusters: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    uniq, inv = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def run() -> dict:
    table = load_table()
    meta = table[["game_id", "season", "week_order", "fbs_both", "season_type", "cfp_round",
                  "home_win", "home_spread"]]
    preds = pd.read_parquet(OUT / "predictions.parquet")
    tuning = pd.read_csv(OUT / "tuning.csv")
    preds, members = add_ensemble(preds, tuning)
    preds.to_parquet(OUT / "predictions_with_ensemble.parquet", index=False)
    df = preds.merge(meta, on=["game_id", "season"], how="left")
    models = sorted(df["model"].unique())

    # ---------------- scorecards on several game subsets
    subsets = {
        "all_fbs_games": lambda d: d,
        "fbs_vs_fbs": lambda d: d[d["fbs_both"]],
        "games_with_vegas_line": lambda d: d[d["home_spread"].notna()],
        "bowls_and_playoff": lambda d: d[d["season_type"] == "postseason"],
        "early_season_wk1_4": lambda d: d[d["week_order"] <= 4],
    }
    rows = []
    common_line = set(df.loc[(df["model"] == "vegas_spread") & df["p_home"].notna(), "game_id"])
    for name, f in subsets.items():
        for m in models:
            d = f(df[df["model"] == m]).dropna(subset=["p_home", "home_win"])
            if name == "games_with_vegas_line":
                d = d[d["game_id"].isin(common_line)]
            if len(d) < 50:
                continue
            r = score(d["home_win"].to_numpy(), d["p_home"].to_numpy(), d["home_spread"].to_numpy())
            rows.append({"subset": name, "model": m, **r})
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "metrics_overall.csv", index=False)

    # ---------------- per season (all games)
    per = []
    for (m, s), d in df.dropna(subset=["p_home"]).groupby(["model", "season"]):
        r = score(d["home_win"].to_numpy(), d["p_home"].to_numpy(), d["home_spread"].to_numpy())
        per.append({"model": m, "season": s, **r})
    per = pd.DataFrame(per)
    per.to_csv(OUT / "metrics_by_season.csv", index=False)

    # ---------------- reliability tables
    cal = []
    for m in models:
        d = df[df["model"] == m].dropna(subset=["p_home"])
        _, tab = ece(d["home_win"].to_numpy(), d["p_home"].to_numpy())
        cal += [{"model": m, **t} for t in tab]
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)

    # ---------------- bootstrap CIs + paired tests vs leader (all FBS games, common set)
    # Every model that predicts EVERY game (benchmarks with partial coverage, like CFBD Elo
    # and Vegas, are compared on their own subsets instead, in vs_vegas.csv).
    full = df.groupby("model")["p_home"].apply(lambda s: s.notna().mean())
    learners_and_bench = [m for m in models if full.get(m, 0) > 0.999]
    wide = df[df["model"].isin(learners_and_bench)].pivot_table(
        index=["game_id", "season", "week_order", "home_win"], columns="model",
        values="p_home").dropna()
    y = wide.index.get_level_values("home_win").to_numpy(dtype=float)
    clusters = (wide.index.get_level_values("season").astype(int) * 100
                + wide.index.get_level_values("week_order").astype(int)).to_numpy()
    P = np.clip(wide.to_numpy(), 1e-6, 1 - 1e-6)
    L = -(y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P))   # per-game log loss
    names = list(wide.columns)
    mean_ll = L.mean(axis=0)
    leader = names[int(np.argmin(mean_ll))]
    rng = np.random.default_rng(SEED)
    idx = _cluster_boot_idx(clusters, rng)
    boot = np.stack([L[i].mean(axis=0) for i in idx])                    # N_BOOT x models
    li = names.index(leader)
    sig = []
    for j, m in enumerate(names):
        diff = boot[:, j] - boot[:, li]                                    # >0: worse than leader
        p_one = float(np.mean(diff <= 0)) if m != leader else 1.0
        sig.append({"model": m, "log_loss": float(mean_ll[j]),
                    "ci_low": float(np.percentile(boot[:, j], 2.5)),
                    "ci_high": float(np.percentile(boot[:, j], 97.5)),
                    "diff_vs_leader": float(mean_ll[j] - mean_ll[li]),
                    "diff_ci_low": float(np.percentile(diff, 2.5)),
                    "diff_ci_high": float(np.percentile(diff, 97.5)),
                    "p_value": min(1.0, 2 * min(p_one, 1 - p_one)) if m != leader else np.nan})
    sig = pd.DataFrame(sig).sort_values("log_loss").reset_index(drop=True)
    # Holm correction
    rival = sig["p_value"].notna()
    order = sig.loc[rival, "p_value"].sort_values().index
    k = len(order)
    adj, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * sig.loc[i, "p_value"]))
        adj[i] = running
    sig["p_holm"] = pd.Series(adj)
    sig["significantly_worse"] = sig["p_holm"] < 0.05
    sig.to_csv(OUT / "significance.csv", index=False)

    # ---------------- versus Vegas on games that have a line
    vs_models = learners_and_bench + ["vegas_spread"]
    vs = df[df["game_id"].isin(common_line) & df["model"].isin(vs_models)].pivot_table(
        index=["game_id", "season", "week_order", "home_win"], columns="model",
        values="p_home").dropna()
    yv = vs.index.get_level_values("home_win").to_numpy(dtype=float)
    cv = (vs.index.get_level_values("season").astype(int) * 100
          + vs.index.get_level_values("week_order").astype(int)).to_numpy()
    Pv = np.clip(vs.to_numpy(), 1e-6, 1 - 1e-6)
    Lv = -(yv[:, None] * np.log(Pv) + (1 - yv[:, None]) * np.log(1 - Pv))
    vnames = list(vs.columns)
    vi = vnames.index("vegas_spread")
    idx_v = _cluster_boot_idx(cv, np.random.default_rng(SEED + 1))
    bv = np.stack([Lv[i].mean(axis=0) for i in idx_v])
    vegas_rows = []
    for j, m in enumerate(vnames):
        d = bv[:, j] - bv[:, vi]
        vegas_rows.append({"model": m, "log_loss": float(Lv[:, j].mean()),
                           "gap_to_vegas": float(Lv[:, j].mean() - Lv[:, vi].mean()),
                           "gap_ci_low": float(np.percentile(d, 2.5)),
                           "gap_ci_high": float(np.percentile(d, 97.5)),
                           "accuracy": float(np.mean((Pv[:, j] >= .5) == yv))})
    pd.DataFrame(vegas_rows).sort_values("log_loss").to_csv(OUT / "vs_vegas.csv", index=False)

    summary = {"leader": leader, "ensemble_members": members, "n_test_games": int(len(y)),
               "n_bootstrap": N_BOOT, "bootstrap_unit": "game-week (season x week)"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    s = run()
    print(json.dumps(s, indent=1))
    sig = pd.read_csv(OUT / "significance.csv")
    print(sig[["model", "log_loss", "ci_low", "ci_high", "diff_vs_leader", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
