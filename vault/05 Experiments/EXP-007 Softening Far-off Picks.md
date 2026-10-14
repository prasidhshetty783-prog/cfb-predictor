---
tags: [experiment, calibration]
date: 2026-10-14
status: done (not needed)
code: scripts/exp007_soften.py
outputs: reports/experiments/exp007/
---
# EXP-007 Softening Far-off Picks

## Question
A pick for a game 5 weeks away uses today's ratings, which will change. Should those picks be
less sure (closer to 50%)?

## Setup
For every season, freeze the teams' scouting reports at week W (2-11) and predict the games
0-5 weeks later with that season's model (trained only on earlier seasons).
Softening: win % = bell curve(margin / sqrt(sigma^2 + weeks ahead x d^2)). d tuned on 2010-2013 (grid 0-5).

## Result (2014-2025, about 37,000 forecasts)
| Weeks ahead | Accuracy | Average confidence | Actually right |
|---|---|---|---|
| 0 (this week) | 75.0% | 73.7% | 75.0% |
| 1 | 73.0% | 72.6% | 73.0% |
| 2 | 72.2% | 72.2% | 72.2% |
| 3 | 71.7% | 71.8% | 71.7% |
| 4 | 71.3% | 71.6% | 71.3% |
| 5 | 70.9% | 71.5% | 70.9% |

The tuning picked **d = 0 (no softening)**: any softening made 2010-2013 forecasts worse.

## Why
Far-off picks already get less accurate (75% this week, 71% five weeks out), but the
model's percentages stay honest at every distance (within 0.6 points). The model is slightly
cautious by design (calibration slope 1.07), and that caution already covers the extra
uncertainty of future weeks.

## Decision
No softening ([[Decisions Log]] D-012). In the dashboard, label later weeks "Early look — will
change" and show the measured accuracy by distance (75% this week down to 71% five weeks out).
