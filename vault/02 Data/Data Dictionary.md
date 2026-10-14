---
tags: [data, dictionary]
updated: 2026-10-14
---
# Data Dictionary

Data flows in three layers, like ingredients, prep and the finished dish:
`data/raw/` (downloaded, never edited) → `data/interim/` (cleaned tables) → `data/processed/` (model-ready).

## Ids and naming
- `game_id`: unique game number (CFBD id). `team_id`: unique team number. Names can change; ids don't.
- `season`: the year the season started (the 2026 season ends in January 2027).
- `week_order`: a single counter that keeps regular season and postseason in time order.
- `home_*` / `away_*`: the listed home and away team. On a neutral field, "home" is just the first-listed team.
- `d_*`: home minus away (for example `d_talent` > 0 means the home team has more talent).

## data/interim (built by `python -m cfb.data.clean`)
| Table | Rows | One row per | Key columns |
|---|---|---|---|
| `games` | 38,496 | game (all divisions) | `start_date`, `season_type`, `neutral_site`, `conference_game`, `home_points`, `away_points`, `margin` (home minus away), `home_win`, `completed`, `fbs_involved`, `fbs_both`, `is_bowl`, `cfp_round`, `is_playoff`, `cfbd_home_elo`, `cfbd_away_elo`, `source`, `notes` |
| `team_game_stats` | 38,144 | team per game | `yards`, `plays`, `turnovers`, `ypp` (yards per play), advanced margins: `success_margin`, `explosive_margin`, `havoc_margin`, `turnovers_margin`, `paper_share` (postgame win expectancy) |
| `team_season` | 6,148 | team per season | `talent`, `blue_chip`, `ret_off`, `ret_def`, `ret_all`, `portal_in`, `portal_out`, `portal_net`, `prev_adj_net` (last season's rating) |
| `team_info` | 12,149 | team per season | `school`, `conference`, `classification` (fbs, fcs, ...), `lat`, `lon` |
| `polls` | 20,012 | ranked team per poll week | `poll` (AP, Coaches, CFP), `rank`, `points`, `week_order` |
| `market` | 15,816 | game with a betting line | `home_spread` (negative = home favored), `n_books`, `spread_sd`, `market_source`, `espn_spread`. Benchmark only |

## data/processed (built by `python -m cfb.features.build`)
| Table | Rows | What it is |
|---|---|---|
| `features.parquet` | 19,212 | One row per game with an FBS team, 2005-2026. The 40 model inputs ([[Feature Catalog]]), the answers (`home_win`, `margin`) and the benchmarks (`home_spread`, `cfbd_elo_diff`, `elo_prob`) in separate columns |
| `team_states.parquet` | 42,766 | Each team's "scouting report" right before each game (ratings, roster, form). The simulator uses the latest one |
| `rating_coefs.parquet` | 1,660 | Weekly fitted home-field edge (`hfa`) and FBS-vs-small-school gap (`fcs`) for each rating |
| `features_meta.json` | 1 | Tuned Elo and rating settings |

## Answer columns (never model inputs)
`home_points`, `away_points`, `margin`, `home_win`, plus any after-the-game number
(postgame Elo, postgame win probability, excitement index). The list lives in
`LEAKY_COLUMNS` in `src/cfb/config.py`; `tests/test_no_leakage.py` and the code-review hook enforce it.

## Outputs (reports/)
| File | What it is |
|---|---|
| `backtest/predictions.parquet` | Every model's prediction for every test game, 2014-2025 |
| `backtest/metrics_overall.csv`, `metrics_by_season.csv`, `significance.csv`, `vs_vegas.csv`, `calibration.csv` | The scoreboard ([[Algorithm Comparison]]) |
| `backtest/ablation.csv` | Feature-group test ([[EXP-002 Feature Ablation]]) |
| `championship/bracket_backtest.csv` | Champion odds for 2014-2025 ([[EXP-003 Champion Backtest]]) |
| `championship/committee_validation.csv` | Committee emulator test ([[EXP-004 Committee Emulator]]) |
| `championship/title_odds_2026.csv` | Current title odds ([[2026 Title Odds]]) |
| `predictions_2026_upcoming.csv` | Every remaining 2026 game ([[Upcoming Games]]) |
| `model_report.html` | The full visual report |
| `data_audit.json` | Data checks ([[Data Quality Report]]) |
