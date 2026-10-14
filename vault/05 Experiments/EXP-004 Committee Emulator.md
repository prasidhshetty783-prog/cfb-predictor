---
tags: [experiment, championship]
date: 2026-10-14
status: done
code: src/cfb/sim/committee.py
outputs: reports/championship/committee_validation.csv, committee_summary.json
---
# EXP-004 Committee Emulator

## Question
The playoff field is picked by a 13-person committee, not a formula. Can we learn how they
rank teams, so the 2026 simulation picks a realistic field?

## Setup
- Resume per team at the end of the regular season: wins, losses, rating (our opponent-adjusted
  points rating, on one scale for big and small schools), strength of schedule, conference
  champion, power conference.
- Method: pairwise logistic "learning to rank" ("the committee put A above B"; learn weights
  that explain those choices). Like an Elo, but for resumes.
- Honest test: leave one season out. Train on the other 11 seasons, then rank the held-out season.

## Result (each season ranked by a model that never saw it)
| Season | Real top 4 we matched | Real top 12 we matched | Rank agreement, top 25 |
|---|---|---|---|
| 2014 | 4 of 4 | 10 of 12 | 0.93 |
| 2015 | 4 | 11 | 0.92 |
| 2016 | 4 | 10 | 0.92 |
| 2017 | 3 | 12 | 0.94 |
| 2018 | 3 | 11 | 0.95 |
| 2019 | 4 | 11 | 0.94 |
| 2020 | 3 | 11 | 0.92 |
| 2021 | 4 | 11 | 0.94 |
| 2022 | 4 | 11 | 0.92 |
| 2023 | 3 | 12 | 0.91 |
| 2024 | 4 | 11 | 0.95 |
| 2025 | 3 | 12 | 0.96 |
| **Average** | **3.6 of 4** | **11.1 of 12** | **0.93** |

"Rank agreement" is the Spearman correlation (1.0 = identical order).

What the committee rewards (weights on standardized inputs, all seasons):
wins +3.61, losses -3.16, rating +2.07, schedule strength +2.02, power conference +1.19,
conference title about 0 (once wins and strength are counted, the title adds nothing extra).

## So what
- The emulator agrees closely with the real committee, so simulated playoff fields are realistic.
- Remaining risk: the committee sometimes makes judgment calls (for example 2023 Florida State,
  left out while unbeaten). No formula catches those.
