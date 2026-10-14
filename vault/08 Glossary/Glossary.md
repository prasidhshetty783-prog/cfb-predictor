---
tags: [glossary]
updated: 2026-10-14
---
# Glossary

Every technical word used in this project, in plain language.

**Ablation**: testing what a group of clues is worth by adding or removing it, like benching a player to see how much the team misses them. See [[EXP-002 Feature Ablation]].

**Accuracy**: share of games where the team we favored won.

**AP poll**: the weekly media top-25 ranking. We use the poll published *before* each game.

**AUC**: the chance the model gives a higher win % to a random real winner than to a random real loser. 0.5 is a coin flip, 1.0 is perfect. Ours: 0.83.

**Backtest / walk-forward test**: replaying past seasons as if they were live. The model only knows what had happened before each game. See [[Evaluation Protocol]].

**Benchmark**: something to compare against (home team always, Elo, Vegas). Not an input.

**Bootstrap**: re-scoring the model on thousands of reshuffled samples of the test games to see how much a score could move by luck. We resample whole game-weeks, because games in the same week share conditions.

**Brier score**: the average squared gap between the predicted % and what happened (1 or 0). Lower is better.

**Calibration**: whether the percentages are honest. If we say 70% for 100 games, about 70 should be won.

**Calibration error (ECE)**: the average gap between "what we said" and "what happened", across confidence levels. 0.013 means about 1.3 percentage points off.

**Calibration slope**: 1.0 is perfect. Above 1 means the model is a bit too timid (it could be bolder); below 1 means too bold.

**CFP**: College Football Playoff. 12 teams since 2024. See [[Decisions Log]] D-007 for the 2026 rules.

**Committee emulator**: our stand-in for the 13-person selection committee. See [[EXP-004 Committee Emulator]].

**Confidence score**: the model's win probability for its pick (for example 78%). Calibration checks whether it's honest.

**Data leakage**: when a model sees information it wouldn't have had at prediction time (like the final score). It makes test scores look great and real predictions bad. Our tests and hooks block it.

**Elo**: a chess-style rating. Beat a strong team and you gain a lot of points; lose to a weak one and you drop a lot.

**Feature (clue)**: one input to the model, like "talent difference". See [[Feature Catalog]].

**FBS / FCS**: the top two levels of college football. FBS (Football Bowl Subdivision) is the big-school level that plays for the CFP; FCS is the next level down.

**Holm correction**: a fix for testing many models at once. The more comparisons you make, the more likely one looks better by luck, so the bar is raised.

**Hyperparameter (tuned setting)**: a knob set before training, like how much smoothing to use. Tuned only on 2010-2013.

**Log loss**: the main score. It punishes confident wrong answers hard and rewards honest probabilities. Lower is better. A coin flip scores 0.693; ours is 0.482.

**Margin**: home points minus away points.

**Monte Carlo simulation**: playing the rest of the season thousands of times with weighted coin flips, then counting how often each team wins it all.

**Opponent-adjusted rating**: how good a team is after accounting for who it played. Beating a top-10 team by 7 counts more than beating a weak team by 30.

**Overfitting**: memorizing the past instead of learning patterns that repeat. Walk-forward testing exposes it.

**p-value**: roughly, how likely a difference this big would appear by luck alone. Small (under 0.05) means probably real.

**Power conference**: SEC, Big Ten, Big 12, ACC (plus Notre Dame). The Pac-12 counted through 2023.

**Precision**: of the games we picked a team to win, the share it really won.

**Recall**: of the games a team really won, the share we called. We track it for home wins, away wins and upsets.

**Ridge regression**: a straight-line formula with a brake on big weights, so it doesn't overreact to noise. Our winning model uses it to predict the margin.

**Spread (point spread)**: the betting market's expected margin. Negative home spread = home team favored.

**Upset**: the betting underdog wins.

**Walk-forward**: see Backtest.
