"""A trained model you can ask about ANY matchup (real or hypothetical).

Plain-language version: this wraps one algorithm from the zoo. ``fit`` studies past
games (from both teams' points of view). ``predict`` returns P(first team wins),
averaging the answer from both sides so the two chances always add to 100%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb.features.build import flip
from cfb.models.zoo import BY_NAME


class Predictor:
    def __init__(self, model: str, params: dict | None = None):
        self.model = model
        self.spec = BY_NAME[model]
        self.params = params or {}
        self.est = None
        self.sigma = None

    # The spec holds small "recipe" functions (lambdas) that pickle can't save.
    # Save the model NAME instead and look the recipe up again when loading.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("spec", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.spec = BY_NAME[self.model]

    def fit(self, train: pd.DataFrame) -> "Predictor":
        spec = self.spec
        if spec.kind == "fixed":
            return self
        if spec.needs_column:
            train = train[train[spec.needs_column].notna()]
        aug = pd.concat([train, flip(train)], ignore_index=True)
        X = aug[spec.features]
        self.est = spec.make(self.params)
        if spec.kind == "clf":
            self.est.fit(X, aug["home_win"].astype(int))
        else:
            self.est.fit(X, aug["margin"])
            self.sigma = float(np.std(aug["margin"] - self.est.predict(X)))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        spec = self.spec
        if spec.kind == "fixed":
            return rows[spec.features[0]].to_numpy(dtype=float)
        a, b = rows[spec.features], flip(rows)[spec.features]
        if spec.kind == "clf":
            return (self.est.predict_proba(a)[:, 1] + 1.0 - self.est.predict_proba(b)[:, 1]) / 2.0
        m = (self.est.predict(a) - self.est.predict(b)) / 2.0
        return norm.cdf(m / self.sigma)

    def predict_margin(self, rows: pd.DataFrame) -> np.ndarray | None:
        if self.spec.kind != "margin":
            return None
        a, b = rows[self.spec.features], flip(rows)[self.spec.features]
        return (self.est.predict(a) - self.est.predict(b)) / 2.0
