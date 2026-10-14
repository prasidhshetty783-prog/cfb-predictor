"""HOOK 1d + 2a (SessionStart): load the lessons and the explanation style at the start of
every session (new, resumed, cleared, or compacted).

Plain-language version: like a coach's pre-game talk. Before Claude does anything, it is
reminded of (1) every rule learned from past mistakes, so it doesn't repeat them, (2) how
to explain things, and (3) where the project's notes live.
Plain text printed here is added to Claude's context.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import LESSONS, PROJECT, STYLE_RULE, ascii_safe, read_payload  # noqa: E402

MAX_RULES = 40


def rules() -> list[str]:
    if not LESSONS.exists():
        return []
    text = LESSONS.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^##\s*Rules\s*$(.*?)(?=^##\s|\Z)", text, re.S | re.M)   # only the "## Rules" section
    block = m.group(1) if m else text
    out = [ln.strip() for ln in block.splitlines() if re.match(r"^\s*[-*]\s+\S", ln)]  # any bullet, dated or not
    return out[-MAX_RULES:]


def status() -> str:
    home = PROJECT / "vault" / "00 Home.md"
    if not home.exists():
        return ""
    m = re.search(r"## Status now\s*\n(.*?)(\n## |\Z)", home.read_text(encoding="utf-8", errors="replace"), re.S)
    return m.group(1).strip()[:900] if m else ""


def main() -> int:
    read_payload()
    parts = ["CFB Predictor project context (from the SessionStart hook).", "", "STYLE: " + STYLE_RULE]
    r = rules()
    if r:
        parts += ["", f"LESSONS LEARNED ({len(r)} rule(s) from past mistakes; follow them, never repeat them):"] + r
    s = status()
    if s:
        parts += ["", "STATUS (vault/00 Home.md):", s]
    parts += ["", "Notes live in vault/ (start at vault/00 Home.md). Load only the notes a task needs."]
    sys.stdout.write(ascii_safe("\n".join(parts)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
