---
tags: [models, evaluation, pre-registered]
created: 2026-10-14
status: locked (written BEFORE any test-season result was looked at)
---
# Evaluation Protocol (the rules of the bake-off)

> [!summary] In plain words
> We test every algorithm the way it would really be used: it only gets to study the
> past, then it predicts a season it has never seen. We do that 12 times, for every
> season from 2014 to 2025. The rules for picking the winner were written down
> **before** looking at any test result, so we can't fool ourselves.

## 1. Data split: a "time machine", never a random shuffle
| Purpose | Seasons | Why |
|---|---|---|
| Warm-up for ratings | 2001-2004 | Elo and power ratings need history to settle |
| Tuning (choosing knobs) | 2010-2013, walk-forward | Each tuning season is predicted by a model trained only on earlier seasons |
| **Test (the real exam)** | **2014-2025** | The whole College Football Playoff era. Never used for tuning |

For each test season *S*: train on every completed game from 2005 to *S-1*, then predict all of *S*.
A random train/test split would let the model learn from games played *after* the ones it
predicts. That is cheating, and it is banned (the code-review hook flags `train_test_split`
without `shuffle=False`).

## 2. What counts as one game
- Every completed game with at least one FBS team (about 900 per season).
- Every prediction is made from **both sides** (A vs B and B vs A) and averaged, so
  P(A wins) + P(B wins) = 100% always holds.

## 3. Scorecard (all computed on test seasons only)
| Metric | Plain meaning | Good direction |
|---|---|---|
| **Log loss** (primary) | Penalty for confident wrong answers. Rewards honest probabilities | lower |
| Brier score | Average squared gap between the % and what happened | lower |
| Accuracy | % of games where the favorite we picked won | higher |
| Recall (home / away) | Of the games the home (or away) team really won, how many did we call? | higher |
| Balanced accuracy | Average of the two recalls: fair even if home teams win more often | higher |
| Upset recall | Of the games where the betting underdog won, how many did we see coming? | higher |
| ECE (calibration error) | When we say 70%, does it happen ~70% of the time? Gap in % points | lower |
| AUC | How well we rank likely winners above likely losers (0.5 = coin flip) | higher |

## 4. Benchmarks every model must face
1. **Home team always wins**: the "no brain" baseline.
2. **Our Elo**: the classic rating system, tuned on 2005-2013.
3. **CFBD Elo**: a public Elo rating from CollegeFootballData.
4. **Vegas closing spread**: the toughest benchmark. Thousands of bettors with real money.
   Used ONLY to compare, never as a model input (the product must predict games
   weeks or a season ahead, before betting lines exist).

## 5. Statistical honesty
- 95% confidence intervals from a **cluster bootstrap** (resampling whole game-weeks, 2,000 times).
- Every model is compared head-to-head with the leader on the **same games** (paired test).
- **Holm correction** because we compare many models at once (more comparisons = more lucky flukes).

## 6. How the winner is chosen (locked rule)
1. The leader is the model with the lowest test log loss on all FBS games, 2014-2025.
2. Its calibration error (ECE) must be under 3 percentage points.
3. If the leader is **not** significantly better than a simpler model (Holm-adjusted p > 0.05),
   pick the simpler, easier-to-explain one (Occam's razor).
4. Report its results honestly next to Vegas, including where it loses.

Related: [[Algorithm Comparison]] · [[Feature Catalog]] · [[Glossary]] · [[Decisions Log]]
