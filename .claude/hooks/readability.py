"""HOOK 2c (Stop): score how easy Claude's last answer was to read.

Plain-language version: uses the Flesch-Kincaid grade level, a classic formula based on
sentence length and syllables per word. Grade 9 is about a 14-year-old's reading level.
Code blocks, tables and links are ignored (they aren't prose). The score is saved to
.claude/state/readability.json and the next prompt's reminder uses it. This hook never
blocks: it measures, it doesn't nag.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook_utils import STATE, read_payload  # noqa: E402


def syllables(word: str) -> int:
    w = word.lower().strip(".,;:!?\"'()[]")
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks
    text = re.sub(r"`[^`]*`", " ", text)                        # inline code
    text = re.sub(r"https?://\S+", " ", text)                   # links
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("|")]   # tables
    text = "\n".join(lines)
    text = re.sub(r"[#>*_]+", " ", text)
    return text


def grade(text: str) -> tuple[float, int]:
    t = prose(text)
    sentences = [s for s in re.split(r"[.!?]+(?:\s|$)|\n\s*[-\d]+[.)]?\s", t) if re.search(r"[A-Za-z]", s)]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", t)
    if len(words) < 40 or not sentences:
        return 0.0, len(words)
    syl = sum(syllables(w) for w in words)
    g = 0.39 * (len(words) / len(sentences)) + 11.8 * (syl / len(words)) - 15.59
    return round(g, 1), len(words)


def last_message(data: dict) -> str:
    msg = data.get("last_assistant_message")
    if isinstance(msg, str) and msg.strip():
        return msg
    if isinstance(msg, dict):
        content = msg.get("content") or msg.get("text") or ""
        if isinstance(content, list):
            return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        return str(content)
    tp = data.get("transcript_path")                             # fallback: read the transcript
    if tp and Path(tp).exists():
        for line in reversed(Path(tp).read_text(encoding="utf-8", errors="replace").splitlines()[-200:]):
            try:
                e = json.loads(line)
            except Exception:
                continue
            m = e.get("message") or {}
            if e.get("type") == "assistant" or m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, list):
                    txt = "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                    if txt.strip():
                        return txt
    return ""


def main() -> int:
    data = read_payload()
    text = last_message(data)
    g, n = grade(text)
    if n >= 40:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "readability.json").write_text(json.dumps({
            "grade": g, "words": n, "scored_at": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
