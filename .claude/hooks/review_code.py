"""HOOK 1a (PostToolUse on Write | Edit | MultiEdit): review code right after Claude writes it.

Plain-language version: every time Claude saves a Python file, this script checks it
like a strict teammate:
* REAL ERRORS (block until fixed): syntax errors, undefined names, broken imports, and
  data-leak patterns (a random train/test split on time-ordered games, or using
  after-the-game columns as features).
* STYLE NOTES (no block): unused imports/variables and similar, passed along as a note.
Every real error is written to the Mistake Log so it becomes a lesson.
JSON files are checked too (a broken settings.json would silently disable hooks).
"""

from __future__ import annotations

import json
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import LESSON_FORMAT, PROJECT, ascii_safe, emit, log_mistake, read_payload  # noqa: E402

LEAKY = ["home_post_win_prob", "away_post_win_prob", "home_postgame_elo", "away_postgame_elo",
         "excitement_index"]
ERROR_RULES = "E9,F63,F7,F82"            # syntax, invalid comparisons, misplaced statements, undefined names
STYLE_RULES = "E,W,F,B"


def ruff(path: Path, select: str) -> list[str]:
    exe = shutil.which("ruff")
    cmd = [exe] if exe else [sys.executable, "-m", "ruff"]
    try:
        r = subprocess.run(cmd + ["check", "--no-cache", "--select", select, "--output-format", "concise",
                                  "--quiet", str(path)], capture_output=True, text=True, timeout=40,
                           cwd=str(PROJECT))
    except Exception:
        return []
    if r.returncode not in (0, 1):                      # ruff missing or crashed: skip quietly
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and ":" in ln]


def leakage_checks(path: Path, text: str) -> list[str]:
    rel = path.as_posix()
    problems = []
    if "tests/" not in rel and ".claude/" not in rel:
        for m in re.finditer(r"train_test_split\((?P<args>[^)]*)\)", text):
            if "shuffle=False" not in m.group("args").replace(" ", ""):
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:{line}: LEAK train_test_split without shuffle=False "
                                "mixes future games into training. Split by season instead.")
    if "/features/" in rel or "/models/" in rel:
        for col in LEAKY:
            for m in re.finditer(rf"[\"']{col}[\"']", text):
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:{line}: LEAK '{col}' is only known after the game; "
                                "it can't be a model input.")
    return problems


def main() -> int:
    data = read_payload()
    ti = data.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("path") or ""
    if not fp:
        return 0
    path = Path(fp)
    if not path.is_absolute():
        path = Path(data.get("cwd") or PROJECT) / path
    if not path.exists():
        return 0

    errors: list[str] = []
    notes: list[str] = []
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{path.name}: invalid JSON ({e})")
    elif path.suffix == ".py":
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{path.name}: syntax error: {e.msg.strip().splitlines()[-1]}")
        if not errors:
            errors += ruff(path, ERROR_RULES)
            errors += leakage_checks(path, path.read_text(encoding="utf-8", errors="replace"))
            notes += [n for n in ruff(path, STYLE_RULES) if n not in errors][:12]
    else:
        return 0

    rel = str(path.relative_to(PROJECT)) if str(path).startswith(str(PROJECT)) else path.name
    if errors:
        for e in errors[:5]:
            log_mistake("code", rel, e)
        msg = ("Code review hook: real errors found in " + rel + ". Fix them before moving on, then add "
               "a one-line rule so it never happens again: " + LESSON_FORMAT + "\n- "
               + "\n- ".join(errors[:10]))
        sys.stderr.write(ascii_safe(msg))
        return 2                                        # exit 2: Claude sees this and must react
    if notes:
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ascii_safe(
            "Style notes from the code review hook (not blocking, tidy up when convenient) for "
            + rel + ":\n- " + "\n- ".join(notes))}})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)                                      # never break the session because of the hook
