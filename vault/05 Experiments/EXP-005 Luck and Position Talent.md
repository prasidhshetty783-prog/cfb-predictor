---
tags: [experiment, features]
date: 2026-10-14
status: done (not adopted)
code: src/cfb/features/extra.py, scripts/exp005_luck_talent.py
outputs: reports/experiments/exp005/
---
# EXP-005 Luck and Position Talent

## Question
Do free clues we already had (luck, quarterback and offensive-line talent, transfer detail)
make the winning model (Margin ridge) better?

## New clues (all known before kickoff; see `src/cfb/features/extra.py`)
- **Luck (#1):** last season's luck (real win share minus postgame win expectancy), this season's luck so far, one-score-game record so far.
- **Position talent (#2):** top-2 quarterback recruit grades and top-5 offensive line grades over the last 4 classes, transfer talent gained, share of roster from transfers.

## Setup
Same locked protocol as [[EXP-001 Algorithm Bake-off]]: tune on 2010-2013, walk-forward 2014-2025
(10,371 games), paired bootstrap by game-week (2,000), Holm correction.

## Result (all games, 2014-2025)
| Version | Log loss | Accuracy | Recall home / away | Calibration error | Right when 80%+ sure |
|---|---|---|---|---|---|
| Current (40 clues) | 0.4822 | 75.7% | 83.2% / 63.3% | 0.013 | 91.7% |
| + luck | 0.4821 | 75.7% | 83.2% / 63.4% | 0.013 | 91.7% |
| + position talent | 0.4823 | 75.7% | 83.2% / 63.4% | 0.014 | 92.2% |
| + both | 0.4822 | 75.7% | 83.2% / 63.4% | 0.012 | 92.1% |

No version is significantly better (Holm p = 1.0 for all three; changes of 0.0001 or less).

**Where it did help:** weeks 1-4. Position talent cut early-season log loss from 0.3872 to
0.3850 (95% range -0.0048 to -0.00001), but hurt slightly from week 5 on (+0.0012), so the net
was zero. This subset was looked at AFTER the main test, so it is a lead, not proof.

## Why it didn't help overall
- Luck: the model already has win % and postgame win expectancy side by side, so a straight-line
  formula can already compute "luck" itself.
- Position talent: overall recruiting talent and in-season ratings already carry most of it.
  Once a few games are played, the ratings know more than recruiting does.

## Decision
Not adopted (locked rule: keep only significant improvements). The clues stay in the feature
table (unused) so the follow-up is cheap. See [[Decisions Log]] D-010.

## Next (pre-registered before any new test)
EXP-006: let position talent count early and fade it out as games are played
(talent x 1/(1 + games played)), with the fade speed tuned on 2010-2013 only.
