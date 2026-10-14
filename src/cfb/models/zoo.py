"""Every algorithm in the bake-off, with a plain-language description and a small tuning grid.

Plain-language tour
-------------------
* Baselines / benchmarks: "home team wins", our Elo, CFBD's Elo, the Vegas spread.
* Linear: logistic regression. Adds up weighted clues, then squashes the sum into a %.
* Probabilistic: Naive Bayes. Assumes every clue is independent (they aren't, so it is a weak baseline).
* Instance-based: k-nearest neighbors. "Find the 150 most similar past games; what happened?"
* Trees: random forest, gradient boosting (HistGB, XGBoost, LightGBM). Many small
  yes/no flowcharts that vote (forest) or correct each other's mistakes (boosting).
* Neural network: a small multi-layer perceptron.
* Kernel: an RBF support-vector machine (fast approximate version).
* Margin models: predict the point margin, then turn it into a % with a bell curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from cfb.config import RANDOM_SEED
from cfb.features.build import MODEL_FEATURES
from cfb.features.extra import FADED_FEATURES

SEED = RANDOM_SEED


@dataclass
class Spec:
    name: str
    family: str
    plain: str                                   # one-line explanation for a 14-year-old
    make: Callable[[dict], object] | None        # builds an unfitted estimator from params
    grid: list[dict] = field(default_factory=lambda: [{}])
    features: list[str] = field(default_factory=lambda: list(MODEL_FEATURES))
    kind: str = "clf"                            # clf | margin | fixed
    needs_column: str | None = None              # rows without this column are skipped


def _scaled(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def _imputed(est):
    return make_pipeline(SimpleImputer(strategy="median"), est)


def _lgbm_clf(p):
    return lgb.LGBMClassifier(n_estimators=p.get("n", 500), learning_rate=p["lr"],
                              num_leaves=p["leaves"], min_child_samples=p.get("mcs", 60),
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=p.get("l2", 5.0), random_state=SEED, n_jobs=2,
                              verbose=-1)


def _lgbm_reg(p):
    return lgb.LGBMRegressor(n_estimators=p.get("n", 500), learning_rate=p["lr"],
                             num_leaves=p["leaves"], min_child_samples=p.get("mcs", 60),
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             reg_lambda=p.get("l2", 5.0), random_state=SEED, n_jobs=2,
                             verbose=-1)


SPECS: list[Spec] = [
    # ---------------------------------------------------------------- baselines & benchmarks
    Spec("home_field", "baseline",
         "Always lean toward the home team (as much as history says home teams win).",
         lambda p: LogisticRegression(C=1e4), features=["hf"]),
    Spec("elo", "baseline",
         "Our classic Elo rating: winners take points from losers, then compare scores.",
         None, kind="fixed", features=["elo_prob"]),
    Spec("cfbd_elo", "benchmark",
         "CollegeFootballData's public Elo rating, turned into a % (outside benchmark).",
         lambda p: LogisticRegression(C=1e4), features=["cfbd_elo_diff", "hf"],
         needs_column="cfbd_elo_diff"),
    Spec("vegas_spread", "benchmark",
         "The betting market's closing point spread, turned into a %. The toughest opponent.",
         lambda p: LogisticRegression(C=1e4), features=["home_spread"],
         needs_column="home_spread"),
    # ---------------------------------------------------------------- learners
    Spec("logistic_regression", "linear",
         "Adds up weighted clues (ratings, talent, home field) and squashes the total into a %.",
         lambda p: _scaled(LogisticRegression(C=p["C"], max_iter=2000)),
         grid=[{"C": 0.0005}, {"C": 0.001}, {"C": 0.003}, {"C": 0.01}]),
    Spec("logistic_lasso", "linear",
         "Logistic regression that is forced to ignore weak clues (keeps only the useful ones).",
         lambda p: _scaled(LogisticRegression(C=p["C"], penalty="l1", solver="liblinear",
                                              max_iter=2000)),
         grid=[{"C": 0.002}, {"C": 0.005}, {"C": 0.02}, {"C": 0.05}]),
    Spec("naive_bayes", "probabilistic",
         "Treats every clue as independent evidence. Simple, fast, usually overconfident.",
         lambda p: _scaled(CalibratedClassifierCV(GaussianNB(var_smoothing=p["vs"]),
                                                  method="sigmoid", cv=3)),
         grid=[{"vs": 1e-9}, {"vs": 1e-3}]),
    Spec("knn", "instance",
         "Finds the most similar past matchups and counts who won those.",
         lambda p: _scaled(KNeighborsClassifier(n_neighbors=p["k"], weights="uniform")),
         grid=[{"k": 75}, {"k": 200}, {"k": 500}]),
    Spec("random_forest", "tree",
         "Hundreds of yes/no flowcharts, each built on a random slice of history, then they vote.",
         lambda p: _imputed(RandomForestClassifier(
             n_estimators=300, max_depth=p["depth"], min_samples_leaf=p["leaf"],
             max_features="sqrt", n_jobs=2, random_state=SEED)),
         grid=[{"depth": 8, "leaf": 25}, {"depth": 12, "leaf": 25},
               {"depth": 8, "leaf": 75}, {"depth": None, "leaf": 50}]),
    Spec("hist_gradient_boosting", "tree",
         "Flowcharts built one after another, each fixing the last one's mistakes (scikit-learn).",
         lambda p: HistGradientBoostingClassifier(
             learning_rate=p["lr"], max_depth=p["depth"], max_iter=p.get("n", 300),
             min_samples_leaf=60, l2_regularization=1.0, random_state=SEED),
         grid=[{"lr": 0.03, "depth": 3}, {"lr": 0.03, "depth": 5},
               {"lr": 0.08, "depth": 3}, {"lr": 0.08, "depth": None, "n": 150}]),
    Spec("xgboost", "tree",
         "The famous competition-winning gradient boosting library.",
         lambda p: XGBClassifier(n_estimators=p.get("n", 500), learning_rate=p["lr"],
                                 max_depth=p["depth"], subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=10, reg_lambda=5.0, random_state=SEED,
                                 n_jobs=2, eval_metric="logloss", tree_method="hist"),
         grid=[{"lr": 0.02, "depth": 3}, {"lr": 0.02, "depth": 5},
               {"lr": 0.05, "depth": 3}, {"lr": 0.05, "depth": 5, "n": 300}]),
    Spec("lightgbm", "tree",
         "Microsoft's fast gradient boosting: grows the most useful branches first.",
         _lgbm_clf,
         grid=[{"lr": 0.02, "leaves": 7}, {"lr": 0.02, "leaves": 15},
               {"lr": 0.05, "leaves": 7}, {"lr": 0.05, "leaves": 31, "n": 300}]),
    Spec("neural_net_mlp", "neural",
         "A small neural network: layers of simple math units that learn combinations of clues.",
         lambda p: _scaled(MLPClassifier(hidden_layer_sizes=p["layers"], alpha=p["alpha"],
                                         early_stopping=True, validation_fraction=0.15,
                                         max_iter=400, learning_rate_init=1e-3,
                                         random_state=SEED)),
         grid=[{"layers": (32,), "alpha": 0.3}, {"layers": (32,), "alpha": 1.0},
               {"layers": (32,), "alpha": 3.0}, {"layers": (32,), "alpha": 10.0},
               {"layers": (64, 32), "alpha": 1.0}, {"layers": (64, 32), "alpha": 3.0}]),
    Spec("svm_rbf", "kernel",
         "Draws the best curved boundary between wins and losses (kernel support-vector machine).",
         lambda p: _scaled(CalibratedClassifierCV(
             make_pipeline(Nystroem(gamma=p["gamma"], n_components=400, random_state=SEED),
                           LinearSVC(C=p["C"], max_iter=5000)),
             method="sigmoid", cv=3)),
         grid=[{"gamma": 0.005, "C": 0.1}, {"gamma": 0.005, "C": 1.0},
               {"gamma": 0.02, "C": 0.1}, {"gamma": 0.02, "C": 1.0}]),
    Spec("margin_ridge", "margin",
         "Predicts the final point margin with a straight-line formula, then turns it into a %.",
         lambda p: _scaled(Ridge(alpha=p["alpha"])), kind="margin",
         grid=[{"alpha": 30.0}, {"alpha": 300.0}, {"alpha": 1000.0}, {"alpha": 3000.0}]),
    Spec("margin_lightgbm", "margin",
         "Predicts the point margin with boosted flowcharts, then turns it into a %.",
         _lgbm_reg, kind="margin",
         grid=[{"lr": 0.02, "leaves": 7}, {"lr": 0.02, "leaves": 15}]),
]

BY_NAME = {s.name: s for s in SPECS}

# Challenger tracked live in 2026 (EXP-006). NOT part of the bake-off list (SPECS).
CHALLENGER = "margin_ridge_faded"
BY_NAME[CHALLENGER] = Spec(
    CHALLENGER, "margin",
    "The winning formula plus quarterback / line talent that counts early and fades as games are played.",
    lambda p: _scaled(Ridge(alpha=p["alpha"])), kind="margin",
    features=list(MODEL_FEATURES) + FADED_FEATURES, grid=[{"alpha": 1000.0}])
