---
tags: [lessons, hooks]
updated: 2026-10-14
---
# Lessons Learned

Every rule here came from a real mistake on this project. The SessionStart hook loads the
**Rules** section into Claude's context at the start of every session, so these mistakes don't
happen twice. Raw errors live in [[Mistake Log]]; how the hooks work is in [[How the Hooks Work]].

**How to add a rule:** one line, at the bottom of the Rules list, in this format:
`- YYYY-MM-DD | area | RULE: what to always/never do | WHY: what went wrong`
Areas: `bridge`, `data`, `features`, `eval`, `code`, `report`, `hooks`.
Keep it short. If a new mistake matches an old rule, sharpen the old rule instead of adding a copy.

## Rules
- 2026-10-14 | bridge | RULE: Run long jobs on the user's computer in foreground chunks under 170 s with `python -u`; never start background jobs. | WHY: the remote shell kills background processes when the call ends, so the first download died halfway.
- 2026-10-14 | bridge | RULE: Stage every sync into a brand-new folder (outputs/sync_HHMMSS). | WHY: re-sending a file overwritten in the same folder delivered the old copy.
- 2026-10-14 | bridge | RULE: Move files in small batches, retry failures, and check each file's sha256 against data/raw/_manifest.json. | WHY: staging failed at random ("upload failed").
- 2026-10-14 | data | RULE: Count games per season and season_type before trusting a source. | WHY: the cfbfastR schedules had no 2001-2022 postseason games; ESPN data filled the gap.
- 2026-10-14 | data | RULE: Check betting data against results (favorites should win about 75%). | WHY: about a third of 2006-2012 ESPN spreads had the sign flipped (favorites "won" only 27%).
- 2026-10-14 | data | RULE: Count real non-null values; never trust an "available" flag. | WHY: ESPN marked 8,309 games as having spreads that were blank.
- 2026-10-14 | data | RULE: Dedupe games across sources by team pair within +/-3 days. | WHY: the 2013 conference title games appeared twice.
- 2026-10-14 | data | RULE: A 0-0 final is a cancelled game, not a tie; drop it from completed games. | WHY: 33 "ties" were cancellations.
- 2026-10-14 | data | RULE: Tag a game as CFP only when its notes say CFP and both teams are FBS. | WHY: FCS playoff rounds were labeled as CFP games.
- 2026-10-14 | data | RULE: Check the computed champion against the real list for every season (data audit). | WHY: 2004 and 2005 were wrong at first because the title game looked like an ordinary bowl.
- 2026-10-14 | data | RULE: Find conference title games by their label, in the regular season only, and exclude Army-Navy. | WHY: Army-Navy counted as a title game and the all-SEC national final counted as an SEC title.
- 2026-10-14 | data | RULE: Read CSV files with a UTF-8 then latin-1 fallback. | WHY: games_info.csv had non-UTF-8 bytes and crashed the loader.
- 2026-10-14 | features | RULE: Prove a poll's timing before using it (the week-W AP poll is published before week-W games). | WHY: a poll taken after a game would leak its result.
- 2026-10-14 | features | RULE: Never use FPI or rating "weekly" snapshots that were captured after the season. | WHY: they were rebuilt at season's end, so they already know the future.
- 2026-10-14 | features | RULE: Keep postgame columns (points, margin, post-win prob, postgame Elo, excitement) out of features; tests/test_no_leakage.py guards this. | WHY: they are the answer key.
- 2026-10-14 | features | RULE: Give lower-division opponents neutral rest (7 days) and games played = week - 1. | WHY: their missing schedules created fake rest-day gaps.
- 2026-10-14 | features | RULE: Put FCS opponents on the FBS rating scale before computing strength of schedule. | WHY: strength of schedule came out mis-scaled.
- 2026-10-14 | code | RULE: Make column names unique before a merge (e.g. rename `source` to `market_source`). | WHY: a name collision caused a KeyError.
- 2026-10-14 | code | RULE: Dedupe column lists with dict.fromkeys before selecting columns. | WHY: `week_order` was selected twice.
- 2026-10-14 | code | RULE: Don't store lambdas on objects that get pickled; add __getstate__/__setstate__. | WHY: saving the Predictor crashed.
- 2026-10-14 | code | RULE: Use boolean masks instead of df.query("@var"). | WHY: ruff can't see the variable being used and reports a false F841.
- 2026-10-14 | code | RULE: Filter rows with explicit column comparisons, not row-wise string joins. | WHY: missing values are floats and broke str.join.
- 2026-10-14 | code | RULE: Never pipe a Python step into `head` (or chain steps through any pipe); write output to a log file, check each exit code, then read the log. | WHY: BrokenPipe killed finalize before it wrote the model card, and `&&` kept going because the pipe hid the exit code (happened twice).
- 2026-10-14 | code | RULE: Use absolute paths in shell commands. | WHY: the working folder changed between calls and relative paths pointed at the wrong place.
- 2026-10-14 | eval | RULE: If the best hyperparameter sits at the edge of the grid, widen the grid and rerun. | WHY: Elo, logistic C, MLP alpha and ridge alpha all started at an edge.
- 2026-10-14 | eval | RULE: Compare models only on games every model covers; report partial-coverage models separately. | WHY: CFBD Elo covered fewer games and quietly shrank the shared test set.
- 2026-10-14 | report | RULE: Match each SVG viewBox to its container width and check the page at phone width. | WHY: screenshots showed blown-up text, clipped labels and overlapping ticks.
- 2026-10-14 | hooks | RULE: Write each lesson as one dated bullet under "## Rules". | WHY: an undated bullet was invisible to the first version of the loader (the loader now reads any bullet in this section).
- 2026-10-14 | code | RULE: Before shipping, run the weekly pipeline and tests under Python 3.10 with the libraries pip picks for it (uv venv -p 3.10). | WHY: the saved model came from scikit-learn 1.9.1 and pandas 2.x warned on a fillna downcast; the pipeline now retrains the model when versions differ.
- 2026-10-14 | data | RULE: Check rule facts (like playoff format) against the newest official source; when sources disagree, the newer and more explicit one wins and gets cited. | WHY: an older article said the Group of Six bid goes to a champion; NCAA.com and AP say the highest-ranked team, champion or not.
- 2026-10-14 | data | RULE: When matching names or codes by counting, break ties on purpose (prefer the unclaimed option, else skip) and sort with a stable, fully specified key. | WHY: tied sportsbook codes were matched differently on Windows and in the cloud, creating fake 0-point lines in 10 games.
