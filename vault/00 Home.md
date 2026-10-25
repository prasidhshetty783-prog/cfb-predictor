---
tags: [home]
---
# CFB Predictor: Home

**Goal:** predict every college football game (who wins, by how much, how sure) and each
team's chance to win the national title. First target: the 2026 season, whose title game is
played in January 2027. Full brief: [[Project Brief]].

## Status now
- Updated 2026-10-25 02:05 UTC.
- Model: CFB Predictor v0.1 (margin_ridge). Picks 75.7% of 2014-2025 games right; log loss 0.482; +0.012 behind Vegas. [[Model Card]]
- 2026 title odds (games through Sat Oct 24, 2026): Ohio State 24.7%, Georgia 19.1%, Miami 14.5%. [[2026 Title Odds]]
- Spotlight: Michigan title 0.02%; Michigan State title 0%. [[Upcoming Games]]
- Next steps: [[Roadmap]].

## Next up
1. Every week of the season: `python -m cfb.pipeline --weekly` (new results, new odds, new notes).
2. Product phase: design the app that shows these predictions. See [[Roadmap]].
3. Model upgrades worth testing next: quarterback and injury news, coaching changes, a
   talent-shrinkage fix for overachieving teams. See [[Roadmap]].

## Map of the vault
| Folder | What's inside | Start with |
|---|---|---|
| 01 Project | Why, scope, plan, every big decision | [[Project Brief]], [[Roadmap]], [[Decisions Log]] |
| 02 Data | Where the data comes from, what each table means, data checks | [[Data Sources]], [[Data Dictionary]], [[Data Quality Report]] |
| 03 Features | The 40 pregame clues the model uses | [[Feature Catalog]] |
| 04 Models | Test rules, scoreboard, the chosen model, algorithms in plain words | [[Evaluation Protocol]], [[Algorithm Comparison]], [[Model Card]], [[Algorithms Explained]] |
| 05 Experiments | One note per experiment, with the result | [[EXP-001 Algorithm Bake-off]], [[EXP-002 Feature Ablation]], [[EXP-003 Champion Backtest]], [[EXP-004 Committee Emulator]] |
| 06 Predictions | Latest title odds, game picks, live scorecard | [[2026 Title Odds]], [[Upcoming Games]], [[Live Scorecard]] |
| 07 Lessons | Rules learned from mistakes, raw error log, how the hooks work | [[Lessons Learned]], [[Mistake Log]], [[How the Hooks Work]] |
| 08 Glossary | Every technical word in plain language | [[Glossary]] |
| Templates | Blank notes for new experiments and decisions | `Templates/` |

Notes tagged `auto` are rewritten after every run. Don't edit them by hand.

## For Claude: how to get context fast
- New session: this page plus [[Lessons Learned]] are loaded by the SessionStart hook. Read more only when the task needs it.
- Changing data code: [[Data Sources]], [[Data Dictionary]], and the `data` rules in [[Lessons Learned]].
- Changing features or models: [[Feature Catalog]], [[Evaluation Protocol]] (locked rules), [[Model Card]].
- Answering "who will win": [[Upcoming Games]] and [[2026 Title Odds]]. Rerun the pipeline first if the notes are more than a week old.
- After any experiment: add a note from `Templates/Experiment.md` and a line in [[Decisions Log]] if something changed.
