---
tags: [project, plan]
updated: 2026-10-14
---
# Roadmap

## Phase 1: Prove the model (done, 2026-10-14)
- [x] Download 22+ seasons of public data with receipts (362 files, sha256 fingerprints)
- [x] Clean and audit it (all 22 national champions 2004-2025 match real history)
- [x] Build 40 leak-proof pregame clues; leakage tests pass
- [x] Bake-off: 16 approaches (12 machine-learning algorithms, an ensemble, 3 baselines) plus Vegas as a yardstick, walk-forward over 2014-2025 ([[EXP-001 Algorithm Bake-off]])
- [x] Pick the winner with the locked rule: Margin ridge ([[Model Card]])
- [x] Champion backtest ([[EXP-003 Champion Backtest]]) and committee emulator ([[EXP-004 Committee Emulator]])
- [x] 2026 title odds and game picks ([[2026 Title Odds]], [[Upcoming Games]])
- [x] Results report with charts, Obsidian vault, Claude Code hooks

## Phase 2: Run it every week (in season, Oct 2026 to Jan 2027)
- [ ] Each Sunday or Monday: `python -m cfb.pipeline --weekly`
- [ ] Log each week's picks before games start, then score them after (a true live test)
- [ ] After bowls: compare the January 2027 champion with the odds

Success test: live 2026 log loss within 0.01 of the 2014-2025 backtest (about 0.48).

## Phase 3: Model upgrades (test each one with the same locked protocol)
Ranked by expected value:
1. **Quarterback and injury flags.** The biggest known gap to Vegas. Needs a source for depth charts.
2. **Talent shrinkage for overachievers.** The model leans on recruiting talent, so teams that
   win above their recruiting (Indiana-style) get underrated. Try a smaller talent weight once
   a team's in-season ratings are strong.
3. **One-sided context clues.** Travel distance and bowl/playoff flags get zero weight in the
   straight-line model because they don't point toward either team ([[Feature Catalog]]).
   Try "away team travel" and "bowl game: shrink the margin" versions. Cheap to test.
4. **Coaching change flags** (`coach_careers` is already downloaded).
5. **Better bowl predictions.** Bowls are close to coin flips for every model (opt-outs).
   Ideas: opt-out counts, motivation proxies.
6. **Calibrate title odds** after a full season of live results.

Tested: luck clues and position talent ([[EXP-005 Luck and Position Talent]]): no overall gain;
position talent helps in weeks 1-4 only. Next: EXP-006, fade talent out as games are played.
Tested: [[EXP-006 Fading Position Talent]]: small gain everywhere (-0.0007 log loss), not yet
significant (p = 0.24). Re-test after the 2026 season.

Rule: keep an upgrade only if it beats the current model on 2014-2025 log loss with a
significant paired bootstrap test ([[Evaluation Protocol]]).

## Phase 4: The product (design phase)
- User research: who wants this (fans, fantasy players, bettors, journalists) and what they need to trust a prediction.
- Explainable picks: show the top three reasons for each pick (the model is a straight-line
  formula, so each clue's push is easy to show).
- Show uncertainty honestly: win %, predicted margin and a range, plus "how often our 70% picks win".
- Measure trust and usability (SUS, a trust-in-automation scale), and track whether users'
  confidence matches the model's real accuracy (calibrated trust, not blind trust).
- Data refresh: a scheduled weekly job that runs the pipeline and publishes the results.
