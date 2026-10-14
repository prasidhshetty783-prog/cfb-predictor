---
tags: [project]
created: 2026-10-14
owner: Prasidh Shetty
---
# Project Brief

## The goal
Build an AI model that predicts:
1. **Single games**: for any matchup (for example Michigan vs Michigan State), who wins, by how
   many points, and how sure the model is.
2. **The national title**: each team's chance to win the College Football Playoff. First
   target: the **2026 season** (title game in January 2027).

Later, a product (an app or site) will show these predictions. The model comes first: we
only build the product once the model is proven on real past seasons.

## What "good" means (measured, not guessed)
| Need | How we measure it | Result (see [[Model Card]]) |
|---|---|---|
| Picks winners | Accuracy on 2014-2025 games it never saw | about 76% |
| Catches wins on both sides | Recall for home wins and for away wins | 83% / 63% |
| Honest confidence | Calibration: when it says 70%, teams win about 70% | error about 1 point |
| Confident picks are right | Accuracy when it is 80%+ sure | about 92% |
| Beats simple methods | Log loss vs "home team always" and Elo | yes, by a wide margin |
| Close to the experts | Log loss gap to the Vegas closing line | about 0.01 behind |
| Title odds work | Real champions' predicted chance, 2014-2025 | 34% on average vs 22% by chance |

"Recall" here is the share of real wins the model called. "Confidence score" is the win
probability, and calibration is how we check it is honest. See [[Glossary]].

## Scope
- In: every game with at least one FBS team, 2001 to today; title odds for the 12-team playoff.
- Out (for now): player-level news (injuries, quarterback changes), weather, betting lines as
  inputs, live in-game predictions.

## Ground rules
- Test like real life: a model may only learn from games played before the ones it predicts
  ([[Evaluation Protocol]]).
- Decide the winner with rules written before seeing test results.
- Vegas is a yardstick, never an input.
- Every mistake becomes a written rule ([[Lessons Learned]]).

## Where things live
- Code: `src/cfb/` · Data: `data/` · Results: `reports/` · Saved model: `models/`
- Full results report: `reports/model_report.html`
- One command to rerun everything: `python -m cfb.pipeline` (see `README.md`)
