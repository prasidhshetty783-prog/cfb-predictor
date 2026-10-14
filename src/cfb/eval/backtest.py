"""Walk-forward backtest: tune on 2010-2013, then train-on-the-past / predict-the-next-season.

Plain-language version
----------------------
For each algorithm:
1. **Tuning**: try a few settings. Each setting predicts 2010, 2011, 2012 and 2013,
   always trained only on earlier seasons. Keep the setting with the best score.
2. **Testing**: with that setting locked, predict 2014 using 2005-2013, then 2015 using
   2005-2014, and so on to 2025. These ~10,000 predictions are the exam.

Every game is predicted from both sides (A vs B, B vs A) and averaged, so the two
teams' chances always add to 100%.

Run:  python -m cfb.eval.backtest                 (all models)
      python -m cfb.eval.backtest --only lightgbm logistic_regression
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb.config import FEATURE_START, PROCESSED, REPORTS, TEST_SEASONS, TUNE_SEASONS
from cfb.features.build import flip
from cfb.models.zoo import BY_NAME, SPECS, Spec

OUT = REPORTS / "backtest"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_table() -> pd.DataFrame:
    t = pd.read_parquet(PROCESSED / "features.parquet")
    return t[t["completed"] & (t["season"] >= FEATURE_START)].reset_index(drop=True)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_predict(spec: Spec, params: dict, train: pd.DataFrame, test: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray | None, object]:
    """Train on ``train``, return (P(home wins), predicted margin or None, fitted model)."""
    if spec.kind == "fixed":
        return test[spec.features[0]].to_numpy(dtype=float), None, None
    if spec.needs_column:
        train = train[train[spec.needs_column].notna()]
    aug = pd.concat([train, flip(train)], ignore_index=True)       # both points of view
    X = aug[spec.features]
    est = spec.make(params)
    test_ok = test[spec.needs_column].notna().to_numpy() if spec.needs_column else \
        np.ones(len(test), bool)
    p = np.full(len(test), np.nan)
    margin = None
    tX, fX = test.loc[test_ok, spec.features], flip(test.loc[test_ok])[spec.features]
    if spec.kind == "clf":
        est.fit(X, aug["home_win"].astype(int))
        p1 = est.predict_proba(tX)[:, 1]
        p2 = est.predict_proba(fX)[:, 1]
        p[test_ok] = (p1 + (1.0 - p2)) / 2.0
    else:  # margin model -> probability through a bell curve
        est.fit(X, aug["margin"])
        sigma = float(np.std(aug["margin"] - est.predict(X)))
        m = (est.predict(tX) - est.predict(fX)) / 2.0
        margin = np.full(len(test), np.nan)
        margin[test_ok] = m
        p[test_ok] = norm.cdf(m / sigma)
    return p, margin, est


def tune(spec: Spec, table: pd.DataFrame) -> tuple[dict, list[dict]]:
    rows = []
    for params in spec.grid:
        ys, ps = [], []
        for v in TUNE_SEASONS:
            train = table[table["season"].between(FEATURE_START, v - 1)]
            test = table[table["season"] == v]
            p, _, _ = fit_predict(spec, params, train, test)
            ok = ~np.isnan(p)
            ys.append(test["home_win"].to_numpy()[ok])
            ps.append(p[ok])
        y, p = np.concatenate(ys), np.concatenate(ps)
        rows.append({"model": spec.name, "params": json.dumps(params, default=str),
                     "tune_log_loss": _log_loss(y, p), "tune_accuracy": float(np.mean((p >= .5) == y)),
                     "n": int(len(y))})
    best = min(rows, key=lambda r: r["tune_log_loss"])
    return json.loads(best["params"]), rows


def walk_forward(spec: Spec, params: dict, table: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for s in TEST_SEASONS:
        train = table[table["season"].between(FEATURE_START, s - 1)]
        test = table[table["season"] == s]
        p, m, _ = fit_predict(spec, params, train, test)
        parts.append(pd.DataFrame({"game_id": test["game_id"].to_numpy(), "season": s,
                                   "model": spec.name, "p_home": p,
                                   "pred_margin": m if m is not None else np.nan}))
    return pd.concat(parts, ignore_index=True)


def run(names: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_table()
    specs = [BY_NAME[n] for n in names] if names else SPECS
    tune_path, pred_path = OUT / "tuning.csv", OUT / "predictions.parquet"
    old_tune = pd.read_csv(tune_path) if tune_path.exists() else pd.DataFrame()
    old_pred = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    chosen_path = OUT / "chosen_params.json"
    chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
    for spec in specs:
        t0 = time.time()
        params, rows = tune(spec, table)
        preds = walk_forward(spec, params, table)
        chosen[spec.name] = params
        if len(old_tune):
            old_tune = old_tune[old_tune["model"] != spec.name]
        if len(old_pred):
            old_pred = old_pred[old_pred["model"] != spec.name]
        old_tune = pd.concat([old_tune, pd.DataFrame(rows)], ignore_index=True)
        old_pred = pd.concat([old_pred, preds], ignore_index=True)
        old_tune.to_csv(tune_path, index=False)
        old_pred.to_parquet(pred_path, index=False)
        chosen_path.write_text(json.dumps(chosen, indent=1, default=str))
        best = min(r["tune_log_loss"] for r in rows)
        print(f"{spec.name:24s} tune_ll={best:.4f} params={params} ({time.time() - t0:.0f}s)",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    run(ap.parse_args().only)


if __name__ == "__main__":
    main()
