"""Predict upcoming games with the saved final model.

Plain-language version: loads the trained model and the latest scouting reports,
then writes each remaining game's win chance and predicted point margin.

Run:  python -m cfb.predict                 (current season, all unplayed games)
      python -m cfb.predict --team Michigan (only games involving a team)
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from cfb.config import CURRENT_SEASON, MODELS_DIR, PROCESSED, REPORTS


def upcoming(season: int = CURRENT_SEASON, team: str | None = None) -> pd.DataFrame:
    table = pd.read_parquet(PROCESSED / "features.parquet")
    g = table[(table["season"] == season) & ~table["completed"]].copy()
    if team:
        g = g[(g["home_team"] == team) | (g["away_team"] == team)]
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    g["home_win_prob"] = model.predict(g)
    g["home_margin"] = model.predict_margin(g)
    g["favorite"] = g["home_team"].where(g["home_win_prob"] >= 0.5, g["away_team"])
    g["favorite_win_prob"] = g["home_win_prob"].where(g["home_win_prob"] >= 0.5,
                                                      1 - g["home_win_prob"])
    g["favorite_by_points"] = g["home_margin"].abs()
    cols = ["season", "week", "start_date", "home_team", "away_team", "neutral_site",
            "home_win_prob", "home_margin", "favorite", "favorite_win_prob", "favorite_by_points",
            "home_spread"]
    return g[cols].sort_values(["week", "start_date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--team")
    a = ap.parse_args()
    df = upcoming(a.season, a.team)
    out = REPORTS / f"predictions_{a.season}_upcoming.csv"
    if not a.team:
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} games -> {out}")
    # Print like scoreboards do: "Away @ Home" (or "vs" on a neutral field), Eastern time.
    show = df if a.team else df[df["week"] == df["week"].min()]
    show = show.copy()
    when = pd.to_datetime(show["start_date"], utc=True).dt.tz_convert("America/New_York")
    show["when (ET)"] = [t.strftime("%a %b %d ") + ("TBD" if (t.hour, t.minute) == (0, 0)
                                                    else t.strftime("%I:%M %p")) for t in when]
    show["game (away @ home)"] = [f"{aw} {'vs' if n else '@'} {h}" for aw, h, n in
                                  zip(show["away_team"], show["home_team"],
                                      show["neutral_site"].fillna(False), strict=True)]
    show["pick"] = (show["favorite"] + " " + (show["favorite_win_prob"] * 100).round(0).astype(int)
                    .astype(str) + "% (by " + show["favorite_by_points"].round(1).astype(str) + ")")
    print(show[["week", "when (ET)", "game (away @ home)", "pick"]].to_string(index=False))

if __name__ == "__main__":
    main()
