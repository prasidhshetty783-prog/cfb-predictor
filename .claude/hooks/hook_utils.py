"""Shared helpers for the CFB Predictor Claude Code hooks.

Every hook reads one JSON object from stdin (sent by Claude Code), does a quick
check, and exits. Rules every hook follows:
* never crash Claude Code: unexpected problems end with exit code 0 (fail open);
* never print non-ASCII to the console (Windows consoles may not be UTF-8);
* write files as UTF-8.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
STATE = PROJECT / ".claude" / "state"
LESSONS = PROJECT / "vault" / "07 Lessons" / "Lessons Learned.md"
MISTAKES = PROJECT / "vault" / "07 Lessons" / "Mistake Log.md"
PENDING = STATE / "pending_errors.jsonl"

STYLE_RULE = (
    "Explanation style for this project: think and check like a data scientist with 30 years of "
    "experience in AI, LLMs, agentic AI and model building, but explain like you are talking to a "
    "smart 14-year-old. Lead with the answer and why it matters. Use short sentences and everyday "
    "words. Define any technical term the first time in plain words (one line). Use a sports "
    "example when it helps. Give exact numbers, then say what they mean."
)


LESSON_FORMAT = ("'- YYYY-MM-DD | area | RULE: what to always/never do | WHY: what went wrong', "
                 "added under '## Rules' in vault/07 Lessons/Lessons Learned.md")


def read_payload() -> dict:
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def ascii_safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def log_mistake(kind: str, where: str, detail: str) -> None:
    """Append a raw entry to the Mistake Log and mark it 'pending' until a lesson is written."""
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        detail = " ".join(detail.split())[:300]
        entry = {"time": datetime.now().timestamp(), "when": now(), "kind": kind,
                 "where": where, "detail": detail}
        with PENDING.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        MISTAKES.parent.mkdir(parents=True, exist_ok=True)
        if not MISTAKES.exists():
            MISTAKES.write_text("# Mistake Log\n\nRaw errors caught by the hooks. Each one should end up as a"
                                " rule in [[Lessons Learned]].\n\n| When | Kind | Where | What happened |\n"
                                "|---|---|---|---|\n", encoding="utf-8")
        safe = detail.replace("|", "/")
        with MISTAKES.open("a", encoding="utf-8") as f:
            f.write(f"| {entry['when']} | {kind} | {where.replace('|', '/')} | {safe} |\n")
    except Exception:
        pass


def pending_since_last_lesson() -> list[dict]:
    """Errors logged after the last time Lessons Learned.md was updated."""
    if not PENDING.exists():
        return []
    try:
        cutoff = LESSONS.stat().st_mtime if LESSONS.exists() else 0.0
        out = []
        for line in PENDING.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("time", 0) > cutoff:
                out.append(e)
        return out
    except Exception:
        return []


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()
