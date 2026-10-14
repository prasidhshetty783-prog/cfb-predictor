"""HOOK 1b (PostToolUse on Bash): read every command's output for signs of trouble.

Plain-language version: after Claude runs a command, this script scans what came back.
* A crash (Python traceback), a failed test, or a model warning that means the result
  can't be trusted -> Claude is told to stop, find the root cause, fix it, and write the
  lesson down. The error goes into the Mistake Log.
* A suspiciously high accuracy (above 85% on college football games) -> a warning that
  the model may be peeking at the answers (data leakage).
Commands that only READ text (grep, cat, head, tail...) are ignored, because reading a log
that mentions an error is not the same as causing one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import LESSON_FORMAT, ascii_safe, emit, log_mistake, read_payload  # noqa: E402

READ_ONLY = re.compile(r"^\s*(grep|rg|cat|head|tail|less|more|ls|find|echo|type|Get-Content|git\s+(log|diff|show|status))\b")
PROBLEMS = [
    (re.compile(r"Traceback \(most recent call last\)"), "Python crashed (traceback)"),
    (re.compile(r"^\s*(\w+Error|\w+Exception): .+", re.M), "an exception was raised"),
    (re.compile(r"^(FAILED|ERROR) [\w/\\.\-]+::", re.M), "a test failed"),
    (re.compile(r"=+ .*\b\d+ (failed|errors?)\b", re.M), "tests failed"),
    (re.compile(r"ConvergenceWarning"), "a model did not converge (results unreliable)"),
    (re.compile(r"SettingWithCopyWarning"), "pandas may have silently not saved a change"),
]
LEAK_SMELL = re.compile(r"(accuracy|acc)[\"']?\s*[:=]\s*(0?\.(8[6-9]|9\d)\d*|(8[6-9]|9\d)(\.\d+)?\s*%)", re.I)


def response_text(resp) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        parts = [str(resp.get(k, "")) for k in ("stdout", "stderr", "output", "error")]
        return "\n".join(p for p in parts if p)
    return str(resp or "")


def main() -> int:
    data = read_payload()
    cmd = str((data.get("tool_input") or {}).get("command", ""))
    if not cmd or READ_ONLY.match(cmd):
        return 0
    text = response_text(data.get("tool_response"))
    if not text:
        return 0
    found = []
    for rx, label in PROBLEMS:
        m = rx.search(text)
        if m:
            line = m.group(0).strip().splitlines()[0][:200]
            found.append(f"{label}: {line}")
    if found:
        for f in found[:3]:
            log_mistake("output", cmd[:80], f)
        sys.stderr.write(ascii_safe(
            "Output review hook: the command output shows a problem. Before moving on: read the full "
            "error, find the ROOT cause (not just the symptom), fix it, re-run, and add a one-line rule "
            "in this format: " + LESSON_FORMAT + "\n- " + "\n- ".join(found[:5])))
        return 2
    if LEAK_SMELL.search(text):
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": (
            "Output review hook: an accuracy above 85% showed up. For college football game picks that "
            "usually means data leakage (the model saw after-the-game information). Check features and "
            "run: python -m pytest tests/test_no_leakage.py")}})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
