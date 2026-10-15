"""Live 2026 scorecard: log every pick BEFORE kickoff, score it AFTER the game.

Plain-language version
----------------------
Each weekly run, two versions predict every remaining game:
* **current**: the official model (Margin ridge, 40 clues). Its picks are the ones you see.
* **challenger**: the same formula plus quarterback / line talent that fades as games are
  played (EXP-006). It predicts quietly in the background.

Picks are saved in ``reports/tracking/picks_<season>.csv``. A pick can be refreshed until
its game kicks off; after that it is frozen, so nothing can be "fixed" after the fact.
Once a game is played, both picks are scored, and the scorecard is written to
``reports/tracking/scorecard_<season>.csv`` and the vault note ``06 Predictions/Live Scorecard.md``.

Run:  python -m cfb.track          (the weekly pipeline runs it for you)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from cfb.config import CURRENT_SEASON, MODELS_DIR, PROCESSED, REPORTS, VAULT
from cfb.models.zoo import CHALLENGER

OUT = REPORTS / "tracking"
PICKS = OUT / f"picks_{CURRENT_SEASON}.csv"
SCORE = OUT / f"scorecard_{CURRENT_SEASON}.csv"
HISTORY = OUT / f"pick_history_{CURRENT_SEASON}.csv"
CH_PATH = MODELS_DIR / "challenger_model.joblib"
LABEL = {"current": "Current model (official)", "challenger": "Challenger (faded position talent)"}


def _challenger():
    """Load the challenger; (re)train it in seconds if missing or saved by another library version."""
    from cfb.eval.backtest import load_table
    from cfb.models.predictor import Predictor

    if CH_PATH.exists():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                m = joblib.load(CH_PATH)
                if not any("unpickle" in str(w.message).lower() for w in caught):
                    return m
            except Exception:
                pass
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    first, last = card["trained_on_seasons"]
    hist = load_table()
    m = Predictor(CHALLENGER, {"alpha": 1000.0}).fit(hist[hist["season"].between(first, last)])
    joblib.dump(m, CH_PATH)
    print(f"trained challenger model on {first}-{last}")
    return m


def log_picks(table: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    future = table[(table["season"] == CURRENT_SEASON) & ~table["completed"]].copy()
    future = future[pd.to_datetime(future["start_date"], utc=True) > now]
    models = {"current": joblib.load(MODELS_DIR / "final_model.joblib"), "challenger": _challenger()}
    rows = []
    for name, m in models.items():
        p = m.predict(future)
        rows.append(pd.DataFrame({
            "game_id": future["game_id"].to_numpy(), "season": CURRENT_SEASON,
            "week": future["week"].to_numpy(), "start_date": future["start_date"].astype(str).to_numpy(),
            "away_team": future["away_team"].to_numpy(), "home_team": future["home_team"].to_numpy(),
            "model": name, "home_win_prob": p, "home_margin": m.predict_margin(future),
            "logged_at": now.isoformat(timespec="seconds")}))
    fresh = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hist = fresh[["game_id", "model", "home_win_prob", "logged_at"]]          # every update, for "pick moved"
    hist.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)
    if PICKS.exists():
        old = pd.read_csv(PICKS)
        frozen = old[pd.to_datetime(old["start_date"], utc=True) <= now]      # kicked off: never change
        frozen = frozen[~frozen.set_index(["game_id", "model"]).index.isin(
            fresh.set_index(["game_id", "model"]).index)]
        fresh = pd.concat([frozen, fresh], ignore_index=True)
    fresh = fresh.sort_values(["start_date", "game_id", "model"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(PICKS, index=False)
    return fresh


def _score(d: pd.DataFrame) -> dict:
    y, p = d["home_win"].to_numpy(float), np.clip(d["home_win_prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    pick = (p >= 0.5).astype(float)
    conf = np.maximum(p, 1 - p)
    sure = conf >= 0.8
    return {"games": int(len(d)), "picks_right": int((pick == y).sum()),
            "accuracy": float(np.mean(pick == y)) if len(d) else np.nan,
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(d) else np.nan,
            "avg_confidence": float(conf.mean()) if len(d) else np.nan,
            "sure_games": int(sure.sum()),
            "sure_right": float(np.mean(pick[sure] == y[sure])) if sure.any() else np.nan}


def scorecard(picks: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    done = table[(table["season"] == CURRENT_SEASON) & table["completed"]][["game_id", "home_win", "margin"]]
    g = picks.merge(done, on="game_id", how="inner")
    rows = []
    for name in ("current", "challenger"):
        d = g[g["model"] == name]
        rows.append({"week": "season", "model": name, **_score(d)})
        for wk, dw in d.groupby("week"):
            rows.append({"week": int(wk), "model": name, **_score(dw)})
    sc = pd.DataFrame(rows)
    sc.to_csv(SCORE, index=False)
    return sc


def calibration(picks: pd.DataFrame, table: pd.DataFrame, model: str = "current") -> pd.DataFrame:
    done = table[(table["season"] == CURRENT_SEASON) & table["completed"]][["game_id", "home_win"]]
    d = picks[picks["model"] == model].merge(done, on="game_id")
    if d.empty:
        return d
    conf = np.maximum(d["home_win_prob"], 1 - d["home_win_prob"])
    right = np.where(d["home_win_prob"] >= 0.5, d["home_win"], 1 - d["home_win"])
    bins = pd.cut(conf, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], right=False,
                  labels=["50-60%", "60-70%", "70-80%", "80-90%", "90%+"])
    return (pd.DataFrame({"bin": bins, "said": conf, "right": right})
            .groupby("bin", observed=True).agg(games=("right", "size"), said=("said", "mean"), happened=("right", "mean"))
            .reset_index())


def write_note(sc: pd.DataFrame, cal: pd.DataFrame, picks: pd.DataFrame, now: pd.Timestamp) -> None:
    def pct(x):
        return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"

    season = sc[sc["week"] == "season"].set_index("model")
    lines = ["---", "tags: [predictions, auto, tracking]", f"generated: {now:%Y-%m-%d %H:%M} UTC", "---",
             f"# Live Scorecard ({CURRENT_SEASON})", "",
             "> Auto-generated by `python -m cfb.track` (runs in the weekly pipeline). Every pick is saved "
             "before kickoff and frozen once the game starts, then scored here.", ""]
    n = int(season["games"].max()) if len(season) else 0
    if n == 0:
        lines += ["No tracked games have been played yet. Picks start counting from the first games after "
                  "tracking began (2026-10-14). Check back after the next weekly run.", ""]
    else:
        c, ch = season.loc["current"], season.loc["challenger"]
        lead = ("the challenger is ahead" if ch["log_loss"] < c["log_loss"] else "the current model is ahead")
        lines += [f"**{n} tracked games played.** On log loss (lower is better) {lead} so far. "
                  "Backtest reference: 75.7% right, log loss 0.482.", "",
                  "| | Picks right | Accuracy | Log loss | Right when 80%+ sure |", "|---|---|---|---|---|"]
        for k in ("current", "challenger"):
            r = season.loc[k]
            lines.append(f"| {LABEL[k]} | {int(r['picks_right'])} of {int(r['games'])} | {pct(r['accuracy'])} | "
                         f"{r['log_loss']:.3f} | {pct(r['sure_right'])} ({int(r['sure_games'])} games) |")
        wk = sc[sc["week"] != "season"].pivot_table(index="week", columns="model",
                                                     values=["picks_right", "games", "log_loss"])
        lines += ["", "## Week by week", "| Week | Games | Current right | Challenger right | Current log loss | Challenger log loss |",
                  "|---|---|---|---|---|---|"]
        for w in wk.index:
            lines.append(f"| {w} | {int(wk.loc[w, ('games', 'current')])} | {int(wk.loc[w, ('picks_right', 'current')])} | "
                         f"{int(wk.loc[w, ('picks_right', 'challenger')])} | {wk.loc[w, ('log_loss', 'current')]:.3f} | "
                         f"{wk.loc[w, ('log_loss', 'challenger')]:.3f} |")
        if len(cal):
            lines += ["", "## Are the percentages honest? (current model)",
                      "| When it said | Games | Average said | Actually won |", "|---|---|---|---|"]
            for r in cal.itertuples():
                lines.append(f"| {r.bin} | {r.games} | {pct(r.said)} | {pct(r.happened)} |")
    pending = picks[pd.to_datetime(picks["start_date"], utc=True) > now]["game_id"].nunique()
    lines += ["", f"Games picked and waiting to be played: {pending}. All picks: `reports/tracking/picks_{CURRENT_SEASON}.csv`.",
              "Decision rule: after the bowls, re-test the challenger with 2026 added ([[EXP-006 Fading Position Talent]])."]
    path = VAULT / "06 Predictions" / "Live Scorecard.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    now = pd.Timestamp(datetime.now(timezone.utc))
    table = pd.read_parquet(PROCESSED / "features.parquet")
    picks = log_picks(table, now)
    sc = scorecard(picks, table)
    write_note(sc, calibration(picks, table), picks, now)
    season = sc[sc["week"] == "season"]
    print(f"logged {picks['game_id'].nunique()} games x 2 models -> {PICKS.relative_to(REPORTS.parent)}")
    print(season[["model", "games", "picks_right", "accuracy", "log_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
