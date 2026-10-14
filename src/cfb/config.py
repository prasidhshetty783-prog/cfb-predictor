"""One place for every path and fixed setting in the project.

Plain-language version: this file is the project's "settings page". If a number
(like which seasons we test on) appears in more than one script, it lives here,
so changing it once changes it everywhere.
"""

from __future__ import annotations

from pathlib import Path

# Project folder = two levels above this file (src/cfb/config.py -> project root)
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
VAULT = ROOT / "vault"

# Seasons ---------------------------------------------------------------
FIRST_SEASON = 2001          # earliest season we download
CURRENT_SEASON = 2026        # the live season (title game is in Jan 2027)
FEATURE_START = 2005         # first season with every pregame feature source
TUNE_SEASONS = list(range(2010, 2014))   # used ONLY to pick hyperparameters
TEST_SEASONS = list(range(2014, 2026))   # walk-forward test: the whole CFP era
RANDOM_SEED = 42

# Columns that are only known AFTER a game ends. Using any of them as a model
# input would be cheating ("data leakage"). The feature builder refuses them.
LEAKY_COLUMNS = {
    "home_points", "away_points", "home_post_win_prob", "away_post_win_prob",
    "home_postgame_elo", "away_postgame_elo", "excitement_index", "highlights",
    "margin", "home_win", "total_points",
}


def ensure_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for p in (RAW, INTERIM, PROCESSED, MODELS_DIR, REPORTS, FIGURES, VAULT):
        p.mkdir(parents=True, exist_ok=True)
