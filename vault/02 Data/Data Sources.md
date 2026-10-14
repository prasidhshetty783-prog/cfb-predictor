---
tags: [data, sources]
updated: 2026-10-14
---
# Data Sources

All data is free and public, from the **SportsDataverse** project on GitHub, which republishes
data from CollegeFootballData.com (CFBD), ESPN and 247Sports. No API key needed.
- `cfbfastR-data` (raw files): https://github.com/sportsdataverse/cfbfastR-data
- `sportsdataverse-data` (release files): https://github.com/sportsdataverse/sportsdataverse-data/releases

Every file is saved untouched in `data/raw/<dataset>/`. The receipt `data/raw/_manifest.json`
records each file's web address, season, size, download time and sha256 fingerprint (a unique
code for the exact file contents, so we can spot if a source changes). Total: 362 files, about
115 MB. Download code: `src/cfb/data/download.py`.

| Dataset | Seasons | What it is | Used for |
|---|---|---|---|
| `games` | 2001-2026 | Every game: teams, score, date, venue, neutral site, conference, pregame Elo | The backbone. Results, schedule, CFBD Elo benchmark |
| `espn_schedules` | 2004-2022 | ESPN schedules incl. bowls and playoff games | Fills the missing postseason games in `games` before 2023 |
| `team_info` | 2001-2026 | Name, conference, division, stadium location per season | Conference membership, travel distance |
| `talent` | 2005-2026 | 247Sports Team Talent Composite (how talented a roster is) | Roster strength |
| `recruits` | 2002-2026 | Every high-school recruit: stars, rating, school | Blue-chip share (4- and 5-star players) |
| `returning_production` | 2004-2026 | Share of last year's production that returns | Experience |
| `transfer_portal` | 2015-2026 | Transfer gains and losses per team | Roster change |
| `ratings_final` | 2004-2026 | End-of-season ratings | Only as LAST season's rating (a prior) |
| `game_stats_advanced` | 2004-2026 | Per-game success rate, explosiveness, havoc, turnovers, postgame win expectancy | Power ratings and season form |
| `team_box` | 2004-2026 | ESPN team box scores (yards, plays, turnovers) | Yards per play |
| `polls` / `poll_ranks` | 2004-2026 | Weekly AP, Coaches and CFP rankings | AP rank clue; real CFP rankings to train the committee emulator |
| `betting_lines` | all | Sportsbook spreads, totals, moneylines (CFBD) | **Benchmark only**, never a model input |
| `espn_betting` | 2004-2026 | ESPN betting lines | Second benchmark source (2013+ only, after checks) |
| `fpi_weekly`, `ratings_weekly`, `team_summaries_weekly` | 2004-2026 | ESPN FPI and weekly computer ratings | Reference only. **Not** used as inputs: these files were rebuilt after each season, so they "know the future" |
| `teams_logos` | current | Team colors and logo links | For the future app |
| `coach_careers` | current | Head coach career records | Future "new coach" clue (not used yet) |

## Known gaps and fixes (details in [[Data Quality Report]] and [[Lessons Learned]])
- `games` has no postseason games for 2001-2022, so ESPN schedules fill them (2004+).
- ESPN spreads: 8,309 games marked "available" were blank, and about a third of 2006-2012 spreads
  had the sign flipped. Fix: a consensus line across several sportsbooks, ESPN only as a 2013+
  fallback. After the fix, favorites win about 77% and the average miss is 12.2 points, which is normal.
- Sportsbook team codes: a rare code (for example 'LAF' in Army vs Lafayette) can tie between the two
  teams in its only game. A wrong match flips the line's sign, and the median of 'Army -33.5' and a
  flipped '+33.5' became a fake 0-point line in 10 games. Fixed 2026-10-14: ties go to the team no
  other code owns, else the code is skipped; the result no longer depends on the computer
  (`tests/test_market.py`).
- 0-0 "ties" (33 games) were cancellations and are dropped.
- 2004 and 2005 title games are found with an AP #1 vs #2 rule, because in the data they look like ordinary bowls.

## Refreshing
- Live season only: `python -m cfb.pipeline --weekly` (re-downloads 2026 files, then rebuilds everything that depends on them).
- Everything: `python -m cfb.data.download` (skips files already on disk; add `--force` to replace them).
- The mirrors usually update daily during the season.
