---
tags: [experiment, features]
date: 2026-10-14
status: done
code: src/cfb/models/finalize.py (ablation)
outputs: reports/backtest/ablation.csv
---
# EXP-002 Feature Ablation

## Question
Which groups of clues actually help? (An ablation adds or removes groups one at a time to see
what each one is worth, like benching players to see who matters.)

## Setup
Winning model (Margin ridge, same settings), same walk-forward exam (2014-2025, 10,371 games).
Groups are added one after another ([[Feature Catalog]]).

## Result
| Clues used | Count | Log loss | Accuracy | Recall: away wins |
|---|---|---|---|---|
| Elo only (plus home field) | 2 | 0.5121 | 73.9% | 57.0% |
| + power ratings (points, success rate, yards per play, postgame win expectancy) | 11 | 0.4958 | 74.7% | 61.6% |
| + roster (talent, blue chips, returning production, portal, last season) | 23 | 0.4844 | 75.5% | 63.2% |
| + season form and AP poll | 35 | 0.4822 | 75.7% | 63.3% |
| + context (conference game, bowl, playoff, week, travel) | 40 | 0.4822 | 75.7% | 63.3% |

## So what
- Power ratings are the biggest step (-0.016 log loss), then roster clues (-0.011).
- Form adds a little (-0.002).
- Not measured yet: whether roster clues matter more early in the season, before ratings
  have many games. Worth checking by week.
- Context adds **nothing** in this model. That's not because context doesn't matter: these
  clues are the same from both teams' side, and a straight-line margin formula can't use a
  two-sided clue. Making them one-sided is a cheap upgrade to test ([[Roadmap]]).
- Away-win recall climbs from 57% to 63% as clues are added, so the extra clues mostly help
  spot road teams that are better than the home crowd thinks.
