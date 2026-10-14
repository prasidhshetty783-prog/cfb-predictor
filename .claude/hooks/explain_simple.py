"""HOOK 2b (UserPromptSubmit): remind Claude of the explanation style on every prompt.

Plain-language version: every time you send a message, Claude gets a short sticky note:
"expert rigor, 14-year-old language". If the previous answer was too hard to read
(scored by readability.py), the note says so, with the score, so the next answer is simpler.
Plain text printed here is added to Claude's context next to your prompt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import STATE, STYLE_RULE, ascii_safe, read_payload  # noqa: E402

TARGET_GRADE = 9.0


def main() -> int:
    read_payload()
    msg = "Reminder (explain-simply hook): " + STYLE_RULE
    f = STATE / "readability.json"
    if f.exists():
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            g = float(r.get("grade", 0))
            if g > TARGET_GRADE:
                msg += (f" Your last answer scored reading grade {g:.1f} (target: {TARGET_GRADE:.0f} or lower, "
                        "about age 14). Use shorter sentences and simpler words this time.")
        except Exception:
            pass
    sys.stdout.write(ascii_safe(msg))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
