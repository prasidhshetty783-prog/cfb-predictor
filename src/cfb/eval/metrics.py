"""Scoring functions for win-probability predictions.

Plain-language version
----------------------
* accuracy: how often the team we favored won
* recall_home / recall_away: of the games the home (away) team really won, the share we called
* balanced_accuracy: the average of those two recalls
* log_loss / brier: penalties for wrong probabilities (lower is better)
* ece: calibration error, the average gap between "we said 70%" and "it happened 70%"
* upset_recall: of the games where the betting underdog won, the share we picked correctly
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width bins, plus the reliability table."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, freq, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        total += n * abs(conf - freq)
        table.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "mean_pred": conf,
                      "observed": freq, "n": n})
    return total / len(y), table


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit y ~ a + b*logit(p). Perfect calibration: a = 0, b = 1.
    b < 1 means too extreme (overconfident); b > 1 means too timid."""
    from sklearn.linear_model import LogisticRegression

    z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
    lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def score(y: np.ndarray, p: np.ndarray, market_spread: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pick = (p >= 0.5).astype(float)
    tp = np.sum((pick == 1) & (y == 1))
    tn = np.sum((pick == 0) & (y == 0))
    fp = np.sum((pick == 1) & (y == 0))
    fn = np.sum((pick == 0) & (y == 1))
    rec_h = tp / max(tp + fn, 1)
    rec_a = tn / max(tn + fp, 1)
    prec_h = tp / max(tp + fp, 1)
    prec_a = tn / max(tn + fn, 1)
    f1_h = 2 * prec_h * rec_h / max(prec_h + rec_h, EPS)
    f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, EPS)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    e, _ = ece(y, p)
    a, b = calibration_slope(y, p)
    conf = np.maximum(p, 1 - p)
    hi = conf >= 0.8
    out = {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": brier,
        "brier_skill": 1 - brier / (base * (1 - base)),
        "accuracy": float(np.mean(pick == y)),
        "balanced_accuracy": float((rec_h + rec_a) / 2),
        "recall_home": float(rec_h),
        "recall_away": float(rec_a),
        "precision_home": float(prec_h),
        "f1_macro": float((f1_h + f1_a) / 2),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "ece": float(e),
        "calib_intercept": a,
        "calib_slope": b,
        "acc_when_80pct_sure": float(np.mean(pick[hi] == y[hi])) if hi.any() else float("nan"),
        "share_80pct_sure": float(np.mean(hi)),
    }
    if market_spread is not None:
        s = np.asarray(market_spread, dtype=float)
        ok = ~np.isnan(s) & (s != 0)
        dog_home = s > 0                                # home team is the betting underdog
        upset = ok & (((s > 0) & (y == 1)) | ((s < 0) & (y == 0)))
        picked_dog = ok & (((pick == 1) & dog_home) | ((pick == 0) & ~dog_home))
        out["upsets"] = int(upset.sum())
        out["upset_recall"] = float(np.mean(picked_dog[upset])) if upset.any() else float("nan")
        out["upset_precision"] = (float(np.mean(upset[picked_dog])) if picked_dog.any()
                                  else float("nan"))
    return out
