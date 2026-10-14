---
tags: [experiment, features]
date: 2026-10-14
status: done (not adopted, promising)
code: scripts/exp006_fade_talent.py
outputs: reports/experiments/exp006/
---
# EXP-006 Fading Position Talent

## Question
Pre-registered after [[EXP-005 Luck and Position Talent]]: if position talent (QB room, offensive
line, transfer talent, roster turnover) counts fully in week 1 and fades as games are played,
does the model get better?

## Setup
Fade = 1 / (1 + games so far / h). h chosen on 2010-2013 only (grid 1, 2, 4, 8; picked h = 4, so
talent counts half as much after 4 games). Exam: 2014-2025 walk-forward, paired bootstrap by game-week.

## Result
| Games | Current log loss | With faded talent | Change | 95% range | p |
|---|---|---|---|---|---|
| All (10,371) | 0.4822 | 0.4814 | -0.0007 | -0.0021 to +0.0005 | 0.24 |
| Weeks 1-4 (3,356) | 0.3872 | 0.3853 | -0.0019 | -0.0059 to +0.0012 | 0.23 |
| Week 5 on (7,015) | 0.5276 | 0.5274 | -0.0002 | -0.0012 to +0.0008 | 0.73 |

Accuracy 75.7% to 75.8%; right when 80%+ sure 91.7% to 92.0%; calibration about the same.

## Decision
Not adopted: better in every slice, but the gain is still within luck (p = 0.24 > 0.05).
Unlike EXP-005 it no longer hurts after week 4. Best next evidence: the 2026 live season.
Track both versions each week; if the gap holds, re-test after the season with 13 test seasons.
See [[Decisions Log]] D-011.
