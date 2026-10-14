"""HOOK 1c (Stop): don't finish until every mistake from this session became a lesson.

Plain-language version: when Claude is about to say "done", this script checks the
Mistake Log. If errors happened after the last update to Lessons Learned.md, Claude is
sent back once to write the lessons (what went wrong, why, and the rule that prevents it).
It only asks once per stop, so it can never trap Claude in a loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import LESSON_FORMAT, ascii_safe, emit, pending_since_last_lesson, read_payload  # noqa: E402


def main() -> int:
    data = read_payload()
    if data.get("stop_hook_active"):
        return 0                                     # already sent back once: let Claude stop
    pending = pending_since_last_lesson()
    if not pending:
        return 0
    lines = [f"{p.get('when')} [{p.get('kind')}] {p.get('where')}: {p.get('detail')}" for p in pending[-8:]]
    emit({"decision": "block", "reason": ascii_safe(
        f"Lesson gate hook: {len(pending)} error(s) happened since Lessons Learned.md was last updated. "
        "Before finishing, add one short rule per root cause, in this format: " + LESSON_FORMAT + ". "
        "Merge duplicates into one rule. Errors:\n- " + "\n- ".join(lines))})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
