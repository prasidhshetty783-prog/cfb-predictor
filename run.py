"""Shortcut so the pipeline runs without installing anything first.

    python run.py --weekly      (same as: python -m cfb.pipeline --weekly)
    python run.py --list
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_module("cfb.pipeline", run_name="__main__")
