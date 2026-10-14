---
tags: [hooks, how-to]
updated: 2026-10-14
---
# How the Hooks Work

**What a hook is:** a small script that Claude Code runs on its own at set moments, like a
referee who checks every play without being asked. Ours live in `.claude/hooks/` and are
switched on by `.claude/settings.json`. They run only in **Claude Code** (terminal, desktop
app or IDE) opened in this project folder. They don't run in a claude.ai chat.

## Hook 1: check code and output, learn from mistakes

| Script | When it runs | What it does |
|---|---|---|
| `review_code.py` | After Claude writes or edits a file | Python files: checks syntax, undefined names and broken statements (ruff `E9,F63,F7,F82`), plus two data-leak traps: a shuffled `train_test_split`, or after-the-game columns used as model inputs. JSON files: checks they parse. **Real errors block** (Claude must fix them). Style issues (ruff `E,W,F,B`) are passed along as a note and don't block. |
| `review_output.py` | After every shell command | Scans the output for crashes (tracebacks), exceptions, failed tests, models that didn't converge and pandas' silent-copy warning. Any of these **blocks** until fixed. Accuracy above 85% triggers a "possible data leak" warning (no block). Read-only commands like `cat` or `grep` are ignored. |
| `lesson_gate.py` | When Claude is about to finish | If any error was logged after [[Lessons Learned]] was last changed, Claude is sent back **once** to write the rule. It can't loop forever. |
| `session_context.py` | When a session starts, resumes, or is compacted | Loads every rule in [[Lessons Learned]] (the `## Rules` section), the explanation style, and the current status from [[00 Home]]. |

Every blocked error is written to [[Mistake Log]] and to `.claude/state/pending_errors.jsonl`.
That is how "learn from the mistake" works: error, then fix, then rule, then the rule is loaded
next session, so the mistake doesn't come back.

## Hook 2: explain simply

| Script | When it runs | What it does |
|---|---|---|
| `session_context.py` | Session start | Sets the style: expert-level checking, explained for a smart 14-year-old. |
| `explain_simple.py` | Every message you send | Adds a short style reminder next to your prompt. If the last answer was too hard to read, the reminder says so and gives the score. |
| `readability.py` | When Claude finishes | Scores the answer with the Flesch-Kincaid grade level (a formula based on sentence length and syllables). Grade 9 is about age 14, which is the target. Code, tables and links are ignored. Never blocks. |

## Setup on Windows

1. Python must be on your PATH. Test it: open a terminal and type `python --version`.
   If that fails but `py --version` works, open `.claude/settings.json` and change every
   `"command": "python"` to `"command": "py"`.
2. Install ruff for the full code check: `pip install ruff` (it's in `requirements.txt`).
   Without ruff, only the syntax and leak checks run.
3. Open the folder in Claude Code. Type `/hooks` to see the hooks listed.

## Tested

All five scripts were tested on 2026-10-14 in a throwaway copy of the project, and once end
to end with a real headless Claude Code session (v2.1.299). In that run, the code check caught
an undefined name, Claude fixed the file, and the lesson gate made it write the rule.

## Change the rules

- Make the code check stricter or looser: edit `ERROR_RULES` / `STYLE_RULES` in `review_code.py`.
- Add a leak column: add it to `LEAKY` in `review_code.py` and to `LEAKY_COLUMNS` in `src/cfb/config.py`.
- Change the reading target: `TARGET_GRADE` in `explain_simple.py`.
- Turn a hook off: delete its entry in `.claude/settings.json`.
