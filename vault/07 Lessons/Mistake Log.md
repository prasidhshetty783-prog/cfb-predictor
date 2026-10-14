# Mistake Log

Raw errors, newest at the bottom. The hooks add a row automatically every time code or a
command fails (see [[How the Hooks Work]]). Each row must end up as a rule in
[[Lessons Learned]]; the Stop hook won't let Claude finish until that happens.
The first rows were logged by hand while the project was built (before the hooks existed).
Keep the table as the last thing in this file, because the hooks append rows to the end.

| When | Kind | Where | What happened |
|---|---|---|---|
| 2026-10-14 | bridge | data download | Background download died when the remote shell call ended. |
| 2026-10-14 | bridge | device sync | Re-sent file was the old version (same staging folder reused). |
| 2026-10-14 | bridge | device staging | Intermittent "upload failed" while staging raw files. |
| 2026-10-14 | data | cfb.data.clean | cfbfastR schedules had no postseason games for 2001-2022. |
| 2026-10-14 | data | cfb.data.clean | ESPN spreads blank for 8,309 games despite the "available" flag; about 1/3 of 2006-2012 spreads sign-flipped. |
| 2026-10-14 | code | cfb.data.clean | KeyError after merge: two tables both had a `source` column. |
| 2026-10-14 | data | cfb.data.clean | 33 games scored 0-0 were cancellations, not ties. |
| 2026-10-14 | data | cfb.data.clean | FCS playoff rounds tagged as CFP games. |
| 2026-10-14 | data | cfb.data.clean | 2013 conference title games duplicated across sources. |
| 2026-10-14 | data | cfb.data.audit | 2004 and 2005 champions wrong (title game looked like a normal bowl). |
| 2026-10-14 | code | cfb.features.build | Duplicate `week_order` column in the selection list. |
| 2026-10-14 | code | shell | BrokenPipe from piping Python output into `head`. |
| 2026-10-14 | code | ruff | False F841 on a variable used only inside df.query("@ids"). |
| 2026-10-14 | features | cfb.features.build | Fake rest-day gaps for lower-division opponents. |
| 2026-10-14 | eval | cfb.eval.backtest | Best hyperparameters at the edge of the grid (Elo, logistic C, MLP alpha, ridge alpha). |
| 2026-10-14 | eval | cfb.eval.analyze | Shared test set shrank because CFBD Elo covers fewer games. |
| 2026-10-14 | code | cfb.models.finalize | Pickling the Predictor failed (it held a lambda). |
| 2026-10-14 | data | cfb.sim.committee | Army-Navy counted as a title game; all-SEC national final counted as SEC title; 2025 title games flagged non-conference. |
| 2026-10-14 | features | cfb.sim.committee | Strength of schedule mis-scaled for FCS opponents. |
| 2026-10-14 | report | model_report.html | Blown-up SVG text, clipped y-label, overlapping labels, table overflow at phone width. |
| 2026-10-14 | data | cfb.data.clean | games_info.csv was not UTF-8 (UnicodeDecodeError). |
| 2026-10-14 | hooks | session_context.py | Lesson written without a date was not loaded by the first loader. |
| 2026-10-14 | code | shell | Relative paths failed after the working folder changed. |
| 2026-10-14 | code | ad-hoc check | TypeError: str.join hit a missing value (float) in a row-wise filter. |
| 2026-10-14 | code | compatibility test (Python 3.10) | Model saved with scikit-learn 1.9.1 loads with a version warning under 1.7.2; pandas 2.3 FutureWarning in clean._to_bool. |
| 2026-10-14 | data | vault Decisions Log | First draft said the Group of Six bid goes to a conference champion; the 2026 rule is the highest-ranked team. |
| 2026-10-14 | data | cfb.data.clean (_abbr_to_team) | Tied sportsbook codes matched differently per computer: fake 0-point lines in 10 games; Vegas score differed between Windows and cloud runs. |
| 2026-10-14 | code | shell (repeat) | Piped finalize into head: BrokenPipe stopped it before the model card was written; the chain continued because the pipe hid the exit code. |
