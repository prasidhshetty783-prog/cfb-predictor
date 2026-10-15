"""Prepare small, ready-to-show files for the public dashboard (app/streamlit_app.py).

Plain-language version: the dashboard never runs the model. Each weekly run writes these
files, the dashboard just reads them. That keeps it fast, free to host, and honest:
past picks come from the frozen pre-kickoff log, so they can't be changed later.

Writes reports/app/:
    games.csv       every 2026 game: teams, kickoff, pick, win %, margin, reasons, result
    teams.csv       team names, colors, logos
    record.json     season record (live-logged picks only, plus re-created early weeks)
    calibration.csv "when it said X%, it won Y%" for finished games
    title_odds.csv  copy of the latest title odds (+ meta.json)
    by_weeks_ahead.csv  accuracy of picks made 0-5 weeks early (EXP-007)

Run:  python -m cfb.report.app_data
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from cfb.config import CURRENT_SEASON, INTERIM, MODELS_DIR, PROCESSED, RAW, REPORTS
from cfb.features.build import flip

OUT = REPORTS / "app"
TRACK = REPORTS / "tracking"

# Readable groups for the "why" line (feature -> plain group name)
GROUPS = {
    "Team ratings": ["d_r_pts", "d_r_sr", "d_r_ypp", "d_r_pw", "pred_pts", "pred_sr", "pred_ypp", "pred_pw"],
    "Elo rating": ["d_elo", "elo_logit"],
    "Recruiting talent": ["d_talent", "home_talent", "away_talent", "d_blue_chip"],
    "Returning players": ["d_ret_off", "d_ret_def"],
    "Transfers": ["d_portal_net"],
    "Last season": ["d_prev_adj_net"],
    "Conference strength": ["d_is_power", "home_is_power", "away_is_power"],
    "Big school vs small school": ["d_is_fbs"],
    "This season's form": ["d_gp", "d_win_pct", "d_s_margin", "d_s_sr", "d_s_expl", "d_s_havoc", "d_s_to",
                           "d_s_ypp", "d_s_pw", "d_last3_margin"],
    "Rest days": ["d_rest"],
    "AP ranking": ["d_ap_rank"],
    "Home field": ["hf"],
}


def contributions(model, rows: pd.DataFrame) -> pd.DataFrame:
    """Points each clue group adds to the HOME team's predicted margin (straight-line model)."""
    pipe = model.est
    imp, scl, ridge = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[-1][1]
    feats = model.spec.features

    def parts(x: pd.DataFrame) -> np.ndarray:
        z = scl.transform(imp.transform(x[feats]))
        return z * ridge.coef_

    c = (parts(rows) - parts(flip(rows))) / 2.0
    c = pd.DataFrame(c, columns=feats, index=rows.index)
    return pd.DataFrame({g: c[[f for f in fs if f in c]].sum(axis=1) for g, fs in GROUPS.items()})


def reasons_text(contrib: pd.Series, home_fav: bool) -> str:
    sign = 1 if home_fav else -1
    push = (contrib * sign).sort_values(ascending=False)
    top = push[push > 0.3].head(3)
    return "; ".join(f"{g} +{v:.1f}" for g, v in top.items())


def teams() -> pd.DataFrame:
    t = pd.read_csv(RAW / "teams_logos" / "teams_colors_logos.csv", encoding="latin-1")
    info = pd.read_parquet(INTERIM / "team_info.parquet")
    info = info[info["season"] == CURRENT_SEASON][["team_id", "conference", "classification"]]
    t = t[["team_id", "school", "mascot", "abbreviation", "color", "alt_color", "logo"]].merge(info, on="team_id", how="left")
    t["logo"] = t["logo"].str.replace("http://", "https://", regex=False)
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    table = pd.read_parquet(PROCESSED / "features.parquet")
    season = table[table["season"] == CURRENT_SEASON].copy()
    model = joblib.load(MODELS_DIR / "final_model.joblib")

    # model's view of every game, from pre-game information only
    season["model_prob"] = model.predict(season)
    season["model_margin"] = model.predict_margin(season)
    contrib = contributions(model, season)

    # official picks: the frozen pre-kickoff log wins whenever it exists
    picks = pd.read_csv(TRACK / f"picks_{CURRENT_SEASON}.csv") if (TRACK / f"picks_{CURRENT_SEASON}.csv").exists() \
        else pd.DataFrame(columns=["game_id", "model", "home_win_prob", "home_margin", "logged_at"])
    cur = picks[picks["model"] == "current"][["game_id", "home_win_prob", "home_margin", "logged_at"]]
    g = season.merge(cur, on="game_id", how="left")
    g["source"] = np.where(g["home_win_prob"].notna(), "logged before kickoff", "re-created from pre-game data")
    g["home_win_prob"] = g["home_win_prob"].fillna(g["model_prob"])
    g["home_margin"] = g["home_margin"].fillna(g["model_margin"])

    # previous logged value, to show "pick moved"
    hist_path = TRACK / f"pick_history_{CURRENT_SEASON}.csv"
    g["prev_prob"] = np.nan
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        h = h[h["model"] == "current"].sort_values("logged_at")
        runs = sorted(h["logged_at"].unique())
        if len(runs) >= 2:
            prev = h[h["logged_at"] == runs[-2]].set_index("game_id")["home_win_prob"]
            g["prev_prob"] = g["game_id"].map(prev)

    games = pd.read_parquet(INTERIM / "games.parquet")[["game_id", "home_points", "away_points"]]
    g = g.merge(games, on="game_id", how="left")
    home_fav = g["home_win_prob"] >= 0.5
    g["pick"] = np.where(home_fav, g["home_team"], g["away_team"])
    g["pick_id"] = np.where(home_fav, g["home_id"], g["away_id"])
    g["pick_prob"] = np.where(home_fav, g["home_win_prob"], 1 - g["home_win_prob"])
    g["pick_by"] = g["home_margin"].abs()
    g["prev_pick_prob"] = np.where(home_fav, g["prev_prob"], 1 - g["prev_prob"])
    g["reasons"] = [reasons_text(contrib.iloc[i], bool(home_fav.iloc[i])) for i in range(len(g))]
    g["status"] = np.where(g["completed"], "final", "upcoming")
    g["pick_right"] = np.where(g["completed"], (g["home_win"] == home_fav.astype(float)), np.nan)
    start = pd.to_datetime(g["start_date"], utc=True)
    g["kickoff_utc"] = start.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    upcoming_weeks = g.loc[~g["completed"], "week"]
    next_week = int(upcoming_weeks.min()) if len(upcoming_weeks) else int(g["week"].max())
    g["weeks_ahead"] = np.where(g["completed"], np.nan, g["week"] - next_week)

    cols = ["game_id", "week", "kickoff_utc", "season_type", "neutral_site", "away_team", "away_id",
            "home_team", "home_id", "home_conference", "away_conference", "away_points", "home_points",
            "status", "pick", "pick_id", "pick_prob", "pick_by", "prev_pick_prob", "reasons", "pick_right",
            "source", "weeks_ahead"]
    g[cols].sort_values(["week", "kickoff_utc", "game_id"]).to_csv(OUT / "games.csv", index=False)
    teams().to_csv(OUT / "teams.csv", index=False)

    done = g[g["completed"] & g["fbs_involved"]]
    def rec(d: pd.DataFrame) -> dict:
        sure = d[d["pick_prob"] >= 0.8]
        return {"games": int(len(d)), "right": int(d["pick_right"].sum()),
                "sure_games": int(len(sure)), "sure_right": int(sure["pick_right"].sum())}
    record = {"season": CURRENT_SEASON, "all": rec(done),
              "logged": rec(done[done["source"] == "logged before kickoff"]),
              "recreated": rec(done[done["source"] != "logged before kickoff"]),
              "next_week": next_week}
    (OUT / "record.json").write_text(json.dumps(record, indent=1))

    conf = done["pick_prob"]
    bins = pd.cut(conf, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], right=False,
                  labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    (pd.DataFrame({"bin": bins, "said": conf, "right": done["pick_right"].astype(float)})
     .groupby("bin", observed=True).agg(games=("right", "size"), said=("said", "mean"), won=("right", "mean"))
     .reset_index().to_csv(OUT / "calibration.csv", index=False))

    ch = REPORTS / "championship"
    shutil.copy2(ch / f"title_odds_{CURRENT_SEASON}.csv", OUT / "title_odds.csv")
    meta_odds = json.loads((ch / f"title_odds_{CURRENT_SEASON}_meta.json").read_text())
    exp7 = REPORTS / "experiments" / "exp007" / "by_weeks_ahead.csv"
    if exp7.exists():
        e = pd.read_csv(exp7)
        e[e["version"] == "current"][["weeks_ahead", "games", "accuracy"]].to_csv(OUT / "by_weeks_ahead.csv", index=False)
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    t = card["test_results_2014_2025"]
    (OUT / "meta.json").write_text(json.dumps({
        "updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "data_through_utc": meta_odds["data_through"],
        "season": CURRENT_SEASON, "next_week": next_week, "model": card["algorithm"],
        "backtest": {"accuracy": t["accuracy"], "log_loss": t["log_loss"], "sure_right": t["acc_when_80pct_sure"],
                     "games": t["n"]}}, indent=1))
    print(f"app data: {len(g)} games, next week {next_week}, record {record['all']}")


if __name__ == "__main__":
    main()
