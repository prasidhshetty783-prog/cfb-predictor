---
tags: [experiment, models]
date: 2026-10-14
status: done
code: src/cfb/eval/backtest.py, src/cfb/eval/analyze.py, src/cfb/models/finalize.py
outputs: reports/backtest/
---
# EXP-001 Algorithm Bake-off

## Question
Which algorithm predicts college football games best, with honest confidence?

## Setup (locked in advance: [[Evaluation Protocol]])
- 16 approaches: 12 machine-learning algorithms, an ensemble of the top 3, and 3 simple baselines
  (home team always, our Elo, CFBD Elo), plus the Vegas line as a yardstick ([[Algorithms Explained]]).
  The 12 algorithms and the ensemble all get the same 40 clues ([[Feature Catalog]]).
- Settings tuned only on 2010-2013, each tuning season predicted from earlier seasons.
- Exam: 2014-2025, walk-forward. For season S, train on 2005 to S-1, then predict all of S. 10,371 games.
- Main score: log loss. Also: Brier, accuracy, recall for home and away wins, upset recall, AUC, calibration.
- Significance: paired bootstrap by game-week (2,000 resamples), Holm correction.

## Result
| | Log loss | Accuracy | Recall home / away | When 80%+ sure |
|---|---|---|---|---|
| Leader: Margin LightGBM | 0.4816 | 75.9% | 82.8% / 64.6% | right 91.0% |
| **Winner: Margin ridge** | 0.4822 | 75.7% | 83.2% / 63.3% | right 91.7% |
| Our Elo | 0.5126 | 73.8% | 81.8% / 60.5% | right 89.4% |
| Home team always | 0.6614 | 62.3% | 100% / 0% | n/a |
| Vegas closing line (benchmark) | 0.4690 | 76.4% | | right 91.6% |

- **Six models tied** for the top spot (the gaps were smaller than the noise). Tree ensembles
  (XGBoost, LightGBM, hist. gradient boosting, random forest) were significantly worse, as were
  k-nearest neighbors, SVM, Naive Bayes and Elo.
- The rule picks the simplest tied model: **Margin ridge** ([[Decisions Log]] D-006).
- Calibration is excellent: calibration error 0.013 (when it says 70%, teams win about 70%).
- Gap to Vegas: +0.0117 log loss (95% CI +0.008 to +0.015). Real, but small.
  (Updated 2026-10-14 after fixing 10 fake 0-point Vegas lines; our model's own scores did not change.)
- Hardest games: bowls and playoff games (log loss 0.653, close to a coin flip at 0.693) for every model.

## So what
The model is ready for the product phase: about 3 in 4 winners right, and its percentages
can be trusted as stated. The remaining gap to Vegas is most likely news the model can't see
(injuries, quarterback changes, opt-outs). That's the top upgrade on the [[Roadmap]].

Full scoreboard (auto-updated): [[Algorithm Comparison]]. Chosen model: [[Model Card]].
