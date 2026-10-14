"""Pick the winner by the locked rule, train it on all history, explain it, and save it.

Plain-language version
----------------------
1. **Pick**: the rule from the Evaluation Protocol. If the top scorer isn't *significantly*
   better than a simpler model, the simpler one wins (easier to trust and explain).
2. **Train** the winner on every completed game from 2005 to 2025.
3. **Explain**: which clues push the prediction most (standardized coefficients).
4. **Ablation**: re-test the winner with groups of clues removed, to see which data matters.
5. **Save** the model file and a "model card" (its ID badge: what it is, how good it is,
   where it's weak).

Run:  python -m cfb.models.finalize
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from cfb.config import FEATURE_START, MODELS_DIR, TEST_SEASONS
from cfb.eval.backtest import OUT, fit_predict, load_table
from cfb.eval.metrics import score
from cfb.models.predictor import Predictor
from cfb.models.zoo import BY_NAME

SIMPLE_FAMILIES = {"linear", "margin"}
SIMPLE_MODELS = {"logistic_regression", "logistic_lasso", "margin_ridge"}

ABLATION_GROUPS = {
    "1 Elo only": ["elo_logit", "hf"],
    "2 + power ratings": ["elo_logit", "hf", "d_elo", "d_r_pts", "d_r_sr", "d_r_ypp", "d_r_pw",
                          "pred_pts", "pred_sr", "pred_ypp", "pred_pw"],
    "3 + roster (talent, returning, portal)": None,   # filled below
    "4 + season form & AP poll": None,
    "5 all clues (+ context)": None,
}


def choose_winner() -> tuple[str, dict]:
    sig = pd.read_csv(OUT / "significance.csv")
    leader = sig.iloc[0]
    tied = sig[(sig["model"] == leader["model"]) | (sig["p_holm"] >= 0.05)]
    simple = tied[tied["model"].isin(SIMPLE_MODELS)]
    winner = simple.iloc[0]["model"] if len(simple) else leader["model"]
    why = {
        "leader": leader["model"],
        "leader_log_loss": float(leader["log_loss"]),
        "statistically_tied_with_leader": tied["model"].tolist(),
        "rule": "Leader not significantly better than a simpler model -> choose the simplest "
                "tied model with the lowest log loss" if winner != leader["model"]
                else "Leader is significantly better than every simpler model",
        "winner": winner,
    }
    return winner, why


def ablation(winner: str, params: dict, table: pd.DataFrame) -> pd.DataFrame:
    spec = BY_NAME[winner]
    g2 = ABLATION_GROUPS["2 + power ratings"]
    roster = ["d_talent", "d_blue_chip", "d_ret_off", "d_ret_def", "d_portal_net",
              "d_prev_adj_net", "d_is_power", "d_is_fbs", "home_talent", "away_talent",
              "home_is_power", "away_is_power"]
    form = ["d_gp", "d_win_pct", "d_s_margin", "d_s_sr", "d_s_expl", "d_s_havoc", "d_s_to",
            "d_s_ypp", "d_s_pw", "d_last3_margin", "d_rest", "d_ap_rank"]
    groups = dict(ABLATION_GROUPS)
    groups["3 + roster (talent, returning, portal)"] = g2 + roster
    groups["4 + season form & AP poll"] = g2 + roster + form
    groups["5 all clues (+ context)"] = list(spec.features)
    rows = []
    for name, feats in groups.items():
        s = replace(spec, features=feats)
        ys, ps = [], []
        for season in TEST_SEASONS:
            train = table[table["season"].between(FEATURE_START, season - 1)]
            test = table[table["season"] == season]
            p, _, _ = fit_predict(s, params, train, test)
            ys.append(test["home_win"].to_numpy())
            ps.append(p)
        r = score(np.concatenate(ys), np.concatenate(ps))
        rows.append({"feature_set": name, "n_features": len(feats), **r})
        print(f"{name:42s} log_loss={r['log_loss']:.4f} acc={r['accuracy']:.4f}", flush=True)
    return pd.DataFrame(rows)


def coefficients(pred: Predictor) -> pd.DataFrame:
    """Standardized weights: how many points of margin one 'typical' change in a clue is worth."""
    est = pred.est
    final = est[-1]
    coef = getattr(final, "coef_", None)
    if coef is None:
        return pd.DataFrame()
    coef = np.ravel(coef)
    return (pd.DataFrame({"feature": pred.spec.features, "weight": coef})
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False).drop(columns="abs_weight"))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    winner, why = choose_winner()
    chosen = json.loads((OUT / "chosen_params.json").read_text())
    params = chosen.get(winner, {})
    print("winner:", winner, params)

    train = table[table["season"].between(FEATURE_START, max(TEST_SEASONS))]
    pred = Predictor(winner, params).fit(train)
    joblib.dump(pred, MODELS_DIR / "final_model.joblib")

    coefs = coefficients(pred)
    coefs.to_csv(OUT / "final_model_coefficients.csv", index=False)

    abl = ablation(winner, params, table)
    abl.to_csv(OUT / "ablation.csv", index=False)

    m = pd.read_csv(OUT / "metrics_overall.csv")
    mine = m[(m["model"] == winner)].set_index("subset")
    vegas = pd.read_csv(OUT / "vs_vegas.csv").set_index("model")
    card = {
        "name": f"CFB Predictor v0.1 ({winner})",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": winner,
        "plain_description": BY_NAME[winner].plain,
        "params": params,
        "sigma_points": pred.sigma,
        "trained_on_seasons": [FEATURE_START, max(TEST_SEASONS)],
        "n_training_games": int(len(train)),
        "features": BY_NAME[winner].features,
        "selection": why,
        "test_results_2014_2025": mine.loc["all_fbs_games"].drop("model").to_dict(),
        "by_subset_log_loss": mine["log_loss"].to_dict(),
        "gap_to_vegas_log_loss": float(vegas.loc[winner, "gap_to_vegas"]),
        "known_weaknesses": [
            "Bowls/playoff games are close to coin flips for every model (opt-outs, motivation).",
            "No injury, quarterback, weather or betting information is used.",
            "Still about 0.01 log loss behind the Vegas closing line.",
        ],
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    print(json.dumps({k: card[k] for k in ("name", "selection", "gap_to_vegas_log_loss")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
