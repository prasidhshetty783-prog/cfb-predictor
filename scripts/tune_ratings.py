"""Tune the opponent-adjusted rating settings on 2005-2013 ONLY (test seasons untouched).

Plain-language version: we try a few "caution" levels (lam), how much last season
counts (prev_weight) and how much a blowout can count (cap), and keep the combo whose
pregame predictions line up best with the real final margins.

Run:  python scripts/tune_ratings.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfb.config import REPORTS  # noqa: E402
from cfb.features.build import _game_frame  # noqa: E402
from cfb.features.ratings import RidgeParams, sequential_ratings  # noqa: E402

TUNE = range(2005, 2014)


def main() -> None:
    g, _ = _game_frame()
    g = g[g["season"].between(2004, 2013)].reset_index(drop=True)
    ev = g["season"].isin(TUNE) & g["completed"]
    grids = {
        "pts_target": [21.0, 28.0, 35.0],
        "sr_target": [None],
        "ypp_target": [3.0, 5.0],
        "pw_target": [None],
    }
    rows = []
    for target, caps in grids.items():
        for lam, prev, cap in itertools.product([2.0, 5.0, 10.0, 20.0], [0.2, 0.35, 0.5, 0.7], caps):
            r = sequential_ratings(g, {target: RidgeParams(lam=lam, prev_weight=prev, cap=cap)})
            pred = r[f"pred_{target}"].to_numpy()[ev.to_numpy()]
            actual = g.loc[ev, "margin"].to_numpy()
            ok = ~np.isnan(pred)
            rows.append({"target": target, "lam": lam, "prev_weight": prev, "cap": cap,
                         "corr_with_margin": float(np.corrcoef(pred[ok], actual[ok])[0, 1]),
                         "n": int(ok.sum())})
            print(rows[-1])
    res = pd.DataFrame(rows)
    out = REPORTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ratings_tuning.csv", index=False)
    best = (res.sort_values("corr_with_margin", ascending=False)
            .drop_duplicates("target").set_index("target"))
    best_params = {t: {"lam": float(b.lam), "prev_weight": float(b.prev_weight),
                       "cap": None if pd.isna(b.cap) else float(b.cap),
                       "corr_with_margin": round(float(b.corr_with_margin), 4)}
                   for t, b in best.iterrows()}
    (out / "ratings_best.json").write_text(json.dumps(best_params, indent=1))
    print(json.dumps(best_params, indent=1))


if __name__ == "__main__":
    main()
