---
tags: [experiment, championship]
date: 2026-10-14
status: done
code: src/cfb/sim/bracket.py
outputs: reports/championship/bracket_backtest.csv, bracket_backtest_summary.json
---
# EXP-003 Champion Backtest

## Question
Once the playoff field is known, does the model give the real champion a good chance?

## Setup
For each playoff season 2014-2025: rebuild the real bracket (4 teams through 2023, 12 teams in
2024-2025), use only the model that season would have had (trained on earlier seasons), and
play the bracket 20,000 times. Compare with two simple guesses: our Elo, and "every team
equally likely".

## Result
| Season | Field | Champion | Model's chance for the champion | Champion's rank in our odds | Model's favorite |
|---|---|---|---|---|---|
| 2014 | 4 | Ohio State | 19.1% | 3 | Alabama 36.3% |
| 2015 | 4 | Alabama | 42.6% | **1** | Alabama |
| 2016 | 4 | Clemson | 20.0% | 2 | Alabama 50.0% |
| 2017 | 4 | Alabama | 42.0% | **1** | Alabama |
| 2018 | 4 | Clemson | 35.1% | 2 | Alabama 53.6% |
| 2019 | 4 | LSU | 40.2% | **1** | LSU |
| 2020 | 4 | Alabama | 55.3% | **1** | Alabama |
| 2021 | 4 | Georgia | 47.7% | **1** | Georgia |
| 2022 | 4 | Georgia | 39.5% | **1** | Georgia |
| 2023 | 4 | Michigan | 38.5% | **1** | Michigan |
| 2024 | 12 | Ohio State | 13.0% | 5 | Oregon 21.1% |
| 2025 | 12 | Indiana | 14.5% | 2 | Ohio State 29.3% |

| Summary (12 seasons) | Model | Elo | Equal chances |
|---|---|---|---|
| Average chance given to the real champion | **34.0%** | 27.3% | 22.2% |
| Log score (lower is better) | **1.18** | 1.42 | 1.57 |

- The model's favorite won 7 of 12 titles. The champion was in the model's top 2 in 10 of 12.

## So what
- The model clearly beats both simple guesses. It knows who the strong teams are.
- But even a great favorite rarely tops 50%: a playoff is a short run of near-coin-flips
  against other elite teams. With 12 teams, the favorite's chance drops further (about 20-30%).
  That's the sport, not a flaw. Forecasts should be read as "chances", never as "will win".
- Small sample: 12 seasons. Treat the averages as rough.

Related: [[2026 Title Odds]], [[EXP-004 Committee Emulator]].
