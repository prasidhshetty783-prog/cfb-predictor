# CFB Predictor

Predicts every college football game (winner, margin, win %) and each team's national-title
chance. Current target: the 2026 season (title game January 2027).

## Start here
- Project notes live in the Obsidian vault: `vault/00 Home.md` (map, status, next steps).
- Rules learned from past mistakes: `vault/07 Lessons/Lessons Learned.md`. The SessionStart hook
  loads them automatically. Never repeat a mistake listed there.
- Read only the notes a task needs. Auto notes (tagged `auto`) are regenerated; don't hand-edit them.

## How to explain things
Think and check like a data scientist with 30 years of experience. Explain like you are talking
to a smart 14-year-old: answer first, short sentences, everyday words, define each technical
term once, a sports example when it helps, exact numbers and what they mean.

## Hard rules
- No data leakage: a model may only use information known before kickoff. Never shuffle games
  across time; split by season. Answer columns are listed in `LEAKY_COLUMNS` (`src/cfb/config.py`).
- The evaluation protocol is locked (`vault/04 Models/Evaluation Protocol.md`): tune on
  2010-2013, test walk-forward on 2014-2025, log loss first. Keep a change only if it is
  significantly better on the test seasons.
- Vegas lines are a benchmark only, never a model input.
- After any error: find the root cause, fix it, then add one dated rule under `## Rules` in
  `vault/07 Lessons/Lessons Learned.md`. The hooks enforce this (`vault/07 Lessons/How the Hooks Work.md`).
- Run `python -m pytest -q` after changing anything in `src/cfb/features/` or `src/cfb/models/`.

## Commands (run from this folder; Python 3.10+)
```
pip install -r requirements.txt
pip install -e .                         # makes `import cfb` work everywhere
python -m cfb.pipeline --weekly          # in season: new data -> features -> forecast -> report -> vault
python -m cfb.pipeline                   # rebuild and re-test everything (slow: runs the full bake-off)
python -m cfb.pipeline --list            # show the steps
python -m cfb.predict --team Michigan    # one team's remaining games
python -m pytest -q                      # leakage and simulator tests
```

## Folder map
```
src/cfb/data       download (with sha256 receipts), clean, audit
src/cfb/features   Elo, opponent-adjusted ratings, the 40 pregame clues
src/cfb/eval       walk-forward backtest, metrics, significance tests
src/cfb/models     algorithm zoo, Predictor wrapper, final model + model card
src/cfb/sim        matchup builder, playoff bracket, committee emulator, 2026 season simulator
src/cfb/report     HTML report, vault notes
data/raw -> data/interim -> data/processed     (raw files are never edited)
models/            final_model.joblib, model_card.json
reports/           backtest/, championship/, model_report.html, predictions_2026_upcoming.csv
vault/             Obsidian vault (project memory)
.claude/           hooks + settings
```
