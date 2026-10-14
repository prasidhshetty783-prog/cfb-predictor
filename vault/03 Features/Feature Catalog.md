---
tags: [features]
updated: 2026-10-14
---
# Feature Catalog (the 40 pregame clues)

A **feature** is one clue the model gets about a game, like a scout's note. Every clue must be
known **before kickoff**. Code: `src/cfb/features/` (`elo.py`, `ratings.py`, `build.py`).
Most clues are written as **home minus away** (`d_` prefix), because the matchup is what counts.

Every game is also shown to the model from the other side (away team as "home", every `d_`
clue and the margin flipped), and the two answers are averaged. That makes the model fair to
both teams: P(A wins) + P(B wins) is always exactly 100%.

## 1. Ratings: how good is each team right now? (10 clues)
| Clue | Meaning |
|---|---|
| `d_elo` | Elo rating gap. Elo is a chess-style rating: beat a good team and you gain a lot. Tuned on 2005-2013: K = 35, home edge 55 Elo points, 15% pulled back toward average each offseason, bigger wins count more |
| `elo_logit` | Elo's own win chance for the home team, home field included (on the log-odds scale) |
| `d_r_pts` | Opponent-adjusted **points** rating gap: how many points better than average, after accounting for who you played |
| `d_r_sr` | Same idea for **success rate** (share of plays that gain "enough" yards) |
| `d_r_ypp` | Same idea for **yards per play** |
| `d_r_pw` | Same idea for **postgame win expectancy** (how often you'd win playing like that; filters out lucky bounces) |
| `pred_pts`, `pred_sr`, `pred_ypp`, `pred_pw` | Each rating's predicted margin for this exact matchup, home edge included |

How the four ratings are built: a ridge regression (a line fit that avoids overreacting) is
refit **every week** using only games already played. It starts from last season's rating
(weight 0.35) and caps blowouts (35 points; 5 yards per play) so beating a tiny school by 60
doesn't inflate a team.

## 2. Roster: what does the team have to work with? (12 clues)
| Clue | Meaning |
|---|---|
| `d_talent`, `home_talent`, `away_talent` | 247Sports Team Talent Composite (recruiting stars on the roster) |
| `d_blue_chip` | Share of 4- and 5-star recruits |
| `d_ret_off`, `d_ret_def` | Share of last year's offensive / defensive production that returns |
| `d_portal_net` | Transfer portal gains minus losses |
| `d_prev_adj_net` | Last season's final opponent-adjusted rating |
| `d_is_power`, `home_is_power`, `away_is_power` | Power-conference team (SEC, Big Ten, Big 12, ACC, plus Notre Dame; Pac-12 through 2023; Big East through 2012) |
| `d_is_fbs` | Big school vs small school (FBS vs FCS or lower) |

Small schools with no data get a low, season-specific fill value (unknown returning
production gets the typical value).

## 3. Season form: how is the season going? (12 clues)
| Clue | Meaning |
|---|---|
| `d_gp` | Games played so far |
| `d_win_pct` | Win share so far, pulled toward .500 by 2 make-believe games (so 1-0 isn't "perfect") |
| `d_s_margin`, `d_s_sr`, `d_s_expl`, `d_s_havoc`, `d_s_to`, `d_s_ypp`, `d_s_pw` | Season averages of scoring margin, success rate, explosive plays, havoc (tackles for loss, forced fumbles, interceptions, pass breakups), turnovers, yards per play and postgame win expectancy. Each is pulled toward 0 by 2 make-believe games |
| `d_last3_margin` | Average margin over the last 3 games |
| `d_rest` | Days since the last game (capped at 21; small schools get a neutral 7) |
| `d_ap_rank` | AP poll rank from the poll released before the game (unranked = 30) |

## 4. Context: the setting of the game (6 clues)
| Clue | Meaning |
|---|---|
| `hf` | 1 = real home game, 0 = neutral field |
| `conference_game`, `postseason`, `playoff` | Type of game |
| `week_order` | How far into the season |
| `travel_km` | Distance between the two campuses (blank on neutral fields) |

**Important finding:** in the winning straight-line model, `conference_game`, `postseason`,
`playoff`, `week_order` and `travel_km` get **exactly zero weight**. They don't point toward
either team, and because every game is also shown flipped, a straight-line margin formula
can't use a clue that is the same from both sides. Tree models can use them through
combinations (for example "bowl games are closer"). Turning them into one-sided clues (like
"away team travels 2,000 km") is on the [[Roadmap]].

## What each group adds (ablation, winning model, 2014-2025)
| Clues used | Count | Log loss | Accuracy |
|---|---|---|---|
| Elo only (plus home field) | 2 | 0.5121 | 73.9% |
| + power ratings | 11 | 0.4958 | 74.7% |
| + roster | 23 | 0.4844 | 75.5% |
| + season form and AP poll | 35 | 0.4822 | 75.7% |
| + context | 40 | 0.4822 | 75.7% |

Ratings and roster do most of the work. Details: [[EXP-002 Feature Ablation]].
Biggest single weights: [[Model Card]].

## Leak safety (how we know no clue peeks at the answer)
- Ratings and form use only games that finished before this game's kickoff.
- The AP rank comes from the poll published before the game (checked against real poll dates).
- Roster clues are preseason numbers; last season's rating is from the season before.
- ESPN FPI and "weekly" computer ratings are **not** used: those files were rebuilt after the season.
- `tests/test_no_leakage.py` time-travel tests: hiding every result from week W on must not change
  any week-W clue (Elo, ratings, form), and the AP poll must be published before the games it is used for.
  A sanity test proves the check would catch a leak (hiding results DOES change the next week's clues).

## Not used (and why)
| Not used | Why |
|---|---|
| Final score and stats of the game being predicted (points, margin, postgame win probability, postgame Elo, excitement) | The answer key. Using it is data leakage |
| Vegas lines (spread, total, moneyline) | Yardstick only: title odds need games weeks ahead, before lines exist, and using them would copy Vegas instead of building our own model |
| ESPN FPI and "weekly" computer ratings | Files were rebuilt after each season, so early weeks secretly know later results |
| CFBD Elo | Another company's model, missing for about 15% of games. Benchmark only |
| Injuries, quarterback changes, suspensions, bowl opt-outs | No reliable free history. Main reason we trail Vegas; top upgrade on the [[Roadmap]] |
| Weather | Not in our data source; affects total points more than who wins |
| Coaching changes | `coach_careers` is downloaded but not tested yet |
| Player-by-player stats | Team-level ratings capture most of it with far less noise |
| Coaches Poll, CFP rankings | Overlap with the AP poll; CFP rankings only start late in the season. CFP rankings train the committee emulator instead |
| Extra advanced stats (field position, points per scoring chance, explosive EPA) | Available, not tested yet; likely overlap with success rate and margin |
| Crowd size, rivalry, "trap game" motivation | No reliable way to measure them |

## Why pregame only
- A clue is only useful if it exists when the prediction is made, which is before kickoff.
- Postgame stats from **earlier** games are used: they become next week's ratings and form clues.
- In-game data (score, clock, down and distance) answers a different question ("who wins from
  here?"). That would be a separate live win-probability model built on play-by-play data. The
  best design starts from this model's pregame win % and updates every play. A possible product
  feature, not a replacement.
