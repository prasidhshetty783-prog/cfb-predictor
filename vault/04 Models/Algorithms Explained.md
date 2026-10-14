---
tags: [models, explainer]
updated: 2026-10-14
---
# Algorithms Explained (in plain words)

Every machine-learning algorithm got the same 40 clues ([[Feature Catalog]]), the same
tuning seasons (2010-2013) and the same exam (2014-2025). The simple baselines show how
much the clues add. Scores: [[Algorithm Comparison]].
Code: `src/cfb/models/zoo.py`.

Two kinds of model were tested:
- **Win-chance models** answer "what's the chance the home team wins?" directly.
- **Margin models** answer "by how many points?" and then turn the margin into a win chance
  using how big the usual miss is (about 16 points). A 7-point favorite wins about 67% of the time.

| Algorithm | How it thinks (sports version) | Tuned setting | Result |
|---|---|---|---|
| **Margin ridge** (winner) | Adds up the clues with a weight on each, like a scout's checklist, to predict the point margin. "Ridge" keeps any single weight from getting too big, so it doesn't overreact | alpha = 1000 (strong smoothing) | Tied for best. Easiest to explain |
| Margin LightGBM | Hundreds of small decision trees, each fixing the last one's mistakes, predicting the margin | learning rate 0.02, 15 leaves | Best raw score, but not significantly better than ridge |
| Logistic regression | A weighted checklist that outputs a win chance directly | C = 0.003 | Tied with the leader |
| Logistic lasso | Same, but it can switch useless clues off completely | C = 0.02 | Tied with the leader |
| Neural net (MLP) | Layers of tiny calculators that learn their own combinations of clues | 1 layer of 32, alpha = 1.0 | Tied with the leader |
| Ensemble (top 3) | Averages the three best tuning-season models | n/a | Tied with the leader |
| XGBoost / LightGBM / Hist. gradient boosting | Boosted decision trees ("if Elo gap > 100 and away team travels far, then...") | small trees, slow learning | Slightly but significantly worse |
| Random forest | Hundreds of independent decision trees that vote | depth 12, min 25 games per leaf | Worse |
| k-nearest neighbors | Finds the 200 most similar past games and copies what happened | k = 200 | Worse |
| SVM (RBF kernel) | Draws a curved boundary between "home wins" and "away wins" | gamma 0.005, C 0.1 | Clearly worse; least reliable percentages of the serious models (calibration error 0.032) |
| Naive Bayes | Treats every clue as unrelated to the others (they aren't) | default | Poor: it counts linked clues as separate evidence, and its percentages are off (calibration error 0.052) |
| Our Elo | Just the tuned Elo rating | K 35, home edge 55 | Good baseline, far behind the full models |
| CFBD Elo | A public Elo rating from CollegeFootballData | none | Covers only 8,760 of the test games, so it is scored separately (log loss 0.539) |
| Home team always | Always picks the home team | none | Floor: 62% accuracy |
| Vegas closing line | What the betting market expects at kickoff | none | **Benchmark only**: the best, about 0.01 ahead |

## Why the simple model won
College football has a strong, mostly straight-line signal: better ratings and more talent
mean bigger wins. Fancy models mostly re-learn the same thing and then chase noise. Six
models finished in a statistical tie, so the locked rule ([[Evaluation Protocol]]) picked the
simplest one. A simple formula is also easier to trust and explain in the product: each pick
can show exactly which clues pushed it.

## Why the margin approach helps
Margins carry more information than wins. A 35-point win and a 1-point win are both "wins",
but they say very different things about the teams. Predicting the margin first lets the
model learn from that difference.
