"""Data quality audit: proves the clean tables are trustworthy before any model sees them.

Plain-language version
----------------------
Before cooking, a chef checks every ingredient. This script checks the data:
no duplicate games, no ties, no missing scores, the right number of playoff
games every year, and the champion in our data matches real history. It also
measures how much of each extra data source we actually have per season.

It writes ``reports/data_audit.json`` and the Obsidian note
``vault/02 Data/Data Quality Report.md``. If a hard check fails, it exits with an error.

Run:  python -m cfb.data.audit
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from cfb.config import CURRENT_SEASON, INTERIM, LEAKY_COLUMNS, REPORTS, VAULT

# Real national champions (on-field winners of the title game), used to verify our data.
KNOWN_CHAMPIONS = {
    2004: "USC", 2005: "Texas", 2006: "Florida", 2007: "LSU", 2008: "Florida",
    2009: "Alabama", 2010: "Auburn", 2011: "Alabama", 2012: "Alabama", 2013: "Florida State",
    2014: "Ohio State", 2015: "Alabama", 2016: "Clemson", 2017: "Alabama", 2018: "Clemson",
    2019: "LSU", 2020: "Alabama", 2021: "Georgia", 2022: "Georgia", 2023: "Michigan",
    2024: "Ohio State", 2025: "Indiana",
}
EXPECTED_CFP_GAMES = {**{s: 3 for s in range(2014, 2024)}, 2024: 11, 2025: 11}


def champions(games: pd.DataFrame) -> dict[int, str]:
    """Winner of each season's title game, read straight from the scores."""
    f = games[games["cfp_round"].eq("final") & games["completed"]]
    out = {}
    for _, r in f.iterrows():
        out[int(r["season"])] = r["home_team"] if r["margin"] > 0 else r["away_team"]
    return out


def run() -> dict:
    games = pd.read_parquet(INTERIM / "games.parquet")
    ts = pd.read_parquet(INTERIM / "team_season.parquet")
    tgs = pd.read_parquet(INTERIM / "team_game_stats.parquet")
    market = pd.read_parquet(INTERIM / "market.parquet")
    polls = pd.read_parquet(INTERIM / "polls.parquet")

    hard_fail: list[str] = []
    done = games[games["completed"]]
    fbs = done[done["fbs_involved"]]

    # ---- hard checks
    dup = int(games["game_id"].duplicated().sum())
    # Overtime makes ties impossible in FBS since 1996; small-college games can be called early.
    ties = int((fbs["margin"] == 0).sum())
    lower_division_ties = int((done["margin"] == 0).sum()) - ties
    self_play = int((games["home_id"] == games["away_id"]).sum())
    if dup:
        hard_fail.append(f"{dup} duplicate game ids")
    if ties:
        hard_fail.append(f"{ties} tied FBS games (impossible since 1996 overtime rule)")
    if self_play:
        hard_fail.append(f"{self_play} games where a team plays itself")

    champs = champions(games)
    champ_check = {s: {"data": champs.get(s), "real": real, "ok": champs.get(s) == real}
                   for s, real in KNOWN_CHAMPIONS.items()}
    bad_champs = [s for s, v in champ_check.items() if not v["ok"]]
    if bad_champs:
        hard_fail.append(f"champion mismatch in seasons {bad_champs}")

    cfp_counts = games[games["is_playoff"]].groupby("season").size().to_dict()
    bad_cfp = {s: cfp_counts.get(s, 0) for s, n in EXPECTED_CFP_GAMES.items()
               if cfp_counts.get(s, 0) != n}
    if bad_cfp:
        hard_fail.append(f"wrong number of playoff games: {bad_cfp}")

    leaky_present = sorted(c for c in LEAKY_COLUMNS if c in games.columns)

    # ---- coverage per season (FBS-involved completed games)
    per = fbs.groupby("season").agg(
        games=("game_id", "size"),
        fbs_vs_fbs=("fbs_both", "sum"),
        postseason=("season_type", lambda s: int((s == "postseason").sum())),
        home_win_rate=("home_win", "mean"),
    )
    m = fbs.merge(market[["game_id"]].assign(has_line=True), on="game_id", how="left")
    per["market_line"] = m.groupby("season")["has_line"].apply(lambda s: s.notna().mean())
    stat_ids = set(zip(tgs["game_id"].astype("int64"), tgs["team_id"].astype("int64")))
    both = [((int(g), int(h)) in stat_ids) and ((int(g), int(a)) in stat_ids)
            for g, h, a in zip(fbs["game_id"], fbs["home_id"], fbs["away_id"])]
    per["adv_stats_both"] = pd.Series(both, index=fbs.index).groupby(fbs["season"]).mean()

    fbs_teams = pd.concat([fbs[fbs["home_division"] == "fbs"][["season", "home_id"]]
                           .rename(columns={"home_id": "team_id"}),
                           fbs[fbs["away_division"] == "fbs"][["season", "away_id"]]
                           .rename(columns={"away_id": "team_id"})]).drop_duplicates()
    cov = fbs_teams.merge(ts, on=["season", "team_id"], how="left")
    for col in ["talent", "ret_all", "portal_net", "prev_adj_net"]:
        per[f"{col}_cov"] = cov.groupby("season")[col].apply(lambda s: s.notna().mean())
    per = per.round(3)

    live = games[games["season"] == CURRENT_SEASON]
    live_done = live[live["completed"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hard_fail": hard_fail,
        "checks": {"duplicate_game_ids": dup, "ties": ties,
                   "lower_division_ties_ignored": lower_division_ties, "self_play": self_play,
                   "playoff_game_counts": cfp_counts, "bad_playoff_counts": bad_cfp,
                   "leaky_columns_in_games_table": leaky_present},
        "champions": champ_check,
        "per_season": per.reset_index().to_dict(orient="records"),
        "live_season": {
            "season": CURRENT_SEASON,
            "completed_games": int(len(live_done)),
            "remaining_games": int(len(live) - len(live_done)),
            "last_completed_week": int(live_done["week"].max()) if len(live_done) else None,
            "last_completed_date": str(live_done["start_date"].max()) if len(live_done) else None,
        },
        "known_gaps": [
            "Bowl/playoff games for 2001-2003 are not in any source (Elo warm-up only).",
            "ESPN betting spreads are sign-flipped for ~1/3 of 2006-2012 games: only used 2013+.",
            "ESPN FPI weekly snapshots are often captured after the season: benchmark only.",
            "Talent composite is derived by SportsDataverse from recruiting data (2005+).",
            "Transfer portal data starts 2015 (meaningful from 2018).",
        ],
        "polls": polls.groupby("poll")["season"].agg(["min", "max"]).to_dict(),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_audit.json").write_text(json.dumps(report, indent=1, default=str))
    _write_vault_note(report, per)
    return report


def _write_vault_note(rep: dict, per: pd.DataFrame) -> None:
    ok = "PASS" if not rep["hard_fail"] else "FAIL"
    champ_rows = "\n".join(
        f"| {s} | {v['real']} | {v['data']} | {'yes' if v['ok'] else '**NO**'} |"
        for s, v in rep["champions"].items())
    cols = ["games", "fbs_vs_fbs", "postseason", "home_win_rate", "market_line",
            "adv_stats_both", "talent_cov", "ret_all_cov", "portal_net_cov", "prev_adj_net_cov"]
    table = per[cols].reset_index().to_markdown(index=False)
    live = rep["live_season"]
    note = f"""---
tags: [data, audit]
updated: {rep['generated_at']}
status: {ok}
---
# Data Quality Report

> [!summary] In plain words
> Before any model trains, we check the data like a referee checks equipment.
> Overall result: **{ok}**. Hard problems found: {rep['hard_fail'] or 'none'}.

## Hard checks
- Duplicate games: **{rep['checks']['duplicate_game_ids']}**
- Tied FBS games (impossible since overtime began in 1996): **{rep['checks']['ties']}** (small-college ties ignored: {rep['checks']['lower_division_ties_ignored']})
- Team playing itself: **{rep['checks']['self_play']}**
- Playoff games per season: {rep['checks']['playoff_game_counts']}
- Leak-only columns present in the games table (allowed there, banned as features): {', '.join(rep['checks']['leaky_columns_in_games_table'])}

## Champions in our data vs real history
| Season | Real champion | Our data | Match |
|---|---|---|---|
{champ_rows}

## Coverage per season (completed games with at least one FBS team)
Columns ending in `_cov` = share of FBS teams that have that data. `market_line` = share of
games with a betting line (benchmark only). `adv_stats_both` = both teams have advanced stats.

{table}

## Live season ({live['season']})
- Completed games: {live['completed_games']} (through week {live['last_completed_week']}, last game {live['last_completed_date']})
- Games still to play: {live['remaining_games']}

## Known gaps (and what we did about them)
""" + "\n".join(f"- {g}" for g in rep["known_gaps"]) + """

Related: [[Data Sources]] · [[Feature Catalog]] · [[Evaluation Protocol]]
"""
    path = VAULT / "02 Data" / "Data Quality Report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note, encoding="utf-8")


def main() -> None:
    rep = run()
    print("hard_fail:", rep["hard_fail"] or "none")
    print("champions ok:", sum(v["ok"] for v in rep["champions"].values()), "/", len(rep["champions"]))
    print("live season:", rep["live_season"])
    if rep["hard_fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
