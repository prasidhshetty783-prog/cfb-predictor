"""Download every dataset the CFB Predictor uses (standard library only).

Plain-language version
----------------------
Think of this as a shopping trip with a receipt. The script visits two free,
public GitHub data collections run by the SportsDataverse project:

* ``cfbfastR-data``         - game results, betting lines, team info (updated weekly)
* ``sportsdataverse-data``  - talent, recruiting, returning players, polls, FPI, box scores

Every file is saved exactly as downloaded into ``data/raw/<dataset>/``. Nothing is
changed, so we can always go back to the original. ``data/raw/_manifest.json`` is
the receipt: where each file came from, when, its size, and a fingerprint (sha256)
so we can tell if the source ever changes.

Usage
-----
    python -m cfb.data.download                     # everything
    python -m cfb.data.download --only games        # one dataset
    python -m cfb.data.download --seasons 2026 --force   # refresh the live season

Re-running is safe: files already on disk are skipped unless ``--force`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # works both as "python -m cfb.data.download" and as a plain script
    from cfb.config import CURRENT_SEASON, RAW
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cfb.config import CURRENT_SEASON, RAW

RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
REL_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"


@dataclass(frozen=True)
class Source:
    """One dataset: where it lives and which seasons exist."""

    key: str            # folder name under data/raw
    url: str            # URL template; {y} is replaced by the season
    first: int | None   # first season (None = a single file, no seasons)
    note: str           # what it is, in plain words
    last: int | None = None  # last season published (None = current season)


SOURCES: list[Source] = [
    Source("games", RAW_BASE + "/schedules/csv/cfb_schedules_{y}.csv", 2001,
           "Every game: teams, scores, date, venue, neutral site, conference, pregame Elo"),
    Source("betting_lines", RAW_BASE + "/betting/csv/cfb_line_odds.csv.gz", None,
           "Sportsbook point spreads, totals, moneylines (CFBD). Used ONLY as a benchmark"),
    Source("team_info", RAW_BASE + "/team_info/parquet/cfb_team_info_{y}.parquet", 2001,
           "Team name, conference, division, stadium location per season"),
    Source("teams_logos", RAW_BASE + "/teams/teams_colors_logos.csv", None,
           "Team colors and logo links (for the future app)"),
    Source("talent", REL_BASE + "/cfb_team_talent/cfb_team_talent_{y}.parquet", 2005,
           "247Sports Team Talent Composite: how talented each roster is"),
    Source("recruits", REL_BASE + "/cfb_recruits/cfb_recruits_{y}.parquet", 2002,
           "Every high-school recruit: stars, rating, school signed with"),
    Source("returning_production", REL_BASE
           + "/cfb_returning_production/cfb_returning_production_{y}.parquet", 2004,
           "Share of last year's production (yards, snaps) that returns"),
    Source("transfer_portal", REL_BASE + "/cfb_team_portal/cfb_team_portal_{y}.parquet", 2015,
           "Transfer portal gains and losses per team"),
    Source("espn_schedules", RAW_BASE + "/cfb/schedules/parquet/cfb_schedule_{y}.parquet", 2004,
           "ESPN schedules incl. BOWLS and PLAYOFF games (the main file lacks them before 2023)",
           last=2022),
    Source("poll_ranks", REL_BASE + "/cfb_poll_analytics/cfb_poll_analytics_{y}.parquet", 2004,
           "Weekly poll rankings for every ranked team (AP, Coaches, CFP committee)"),
    Source("polls", REL_BASE + "/cfb_poll_week_summary/cfb_poll_week_summary_{y}.parquet", 2004,
           "Weekly poll summary (volatility, chaos). Reference only, not a model input"),
    Source("game_stats_advanced", REL_BASE
           + "/cfb_paper_index_games/cfb_paper_index_games_{y}.parquet", 2004,
           "Per-game advanced margins: success rate, explosiveness, havoc, turnovers"),
    Source("fpi_weekly", REL_BASE + "/cfb_fpi_weekly/cfb_fpi_weekly_{y}.parquet", 2005,
           "ESPN Football Power Index by week (a professional benchmark)"),
    Source("ratings_weekly", REL_BASE + "/cfb_ratings_weekly/cfb_ratings_weekly_{y}.parquet", 2004,
           "Weekly computer ratings (benchmark / reference)"),
    Source("ratings_final", REL_BASE + "/cfb_ratings/cfb_ratings_{y}.parquet", 2004,
           "End-of-season ratings (only used as LAST season's rating)"),
    Source("team_summaries_weekly", REL_BASE
           + "/cfb_team_summaries_weekly/cfb_team_summaries_weekly_{y}.parquet", 2004,
           "Weekly team efficiency summaries (EPA, success rate, etc.)"),
    Source("team_box", REL_BASE + "/espn_cfb_team_box/team_box_{y}.parquet", 2004,
           "ESPN team box scores per game (yards, turnovers, ...)"),
    Source("espn_betting", REL_BASE + "/espn_cfb_betting/betting_{y}.parquet", 2004,
           "ESPN betting lines (second benchmark source)"),
    Source("coach_careers", REL_BASE + "/espn_cfb_coach_careers/coach_careers.parquet", None,
           "Head coach career records (for 'new coach' flags)"),
]


def _fetch(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> tuple[int, int]:
    """Download one URL to ``dest``. Returns (http_status, bytes). 404 is not an error."""
    req = urllib.request.Request(url, headers={"User-Agent": "cfb-predictor/0.1"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return 200, len(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, 0
            if attempt == retries:
                return e.code, 0
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                return -1, 0
        time.sleep(2 * attempt)
    return -1, 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(only: list[str] | None = None, seasons: list[int] | None = None,
             force: bool = False) -> dict:
    """Download the requested datasets and update the manifest. Returns the manifest."""
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for src in SOURCES:
        if only and src.key not in only:
            continue
        folder = RAW / src.key
        folder.mkdir(exist_ok=True)
        last = src.last or CURRENT_SEASON
        years = [None] if src.first is None else list(range(src.first, last + 1))
        if seasons and src.first is not None:
            years = [y for y in years if y in seasons]
        entry = manifest.get(src.key, {"note": src.note, "files": {}})
        entry["note"] = src.note
        got = missing = failed = 0
        for y in years:
            url = src.url if y is None else src.url.format(y=y)
            dest = folder / url.rsplit("/", 1)[-1]
            if dest.exists() and dest.stat().st_size > 0 and not force:
                got += 1
                if dest.name not in entry["files"]:  # on disk but no receipt yet
                    mtime = datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc)
                    entry["files"][dest.name] = _receipt(url, y, dest, mtime)
                    _save(manifest_path, manifest, src.key, entry)
                continue
            status, _ = _fetch(url, dest)
            if status == 200:
                got += 1
                entry["files"][dest.name] = _receipt(url, y, dest, datetime.now(timezone.utc))
                _save(manifest_path, manifest, src.key, entry)  # save after every file
            elif status == 404:
                missing += 1
            else:
                failed += 1
                print(f"  ! {src.key}: {url} -> status {status}")
        _save(manifest_path, manifest, src.key, entry)
        print(f"{src.key:24s} ok={got:3d} not_published={missing:2d} failed={failed}")
    return manifest


def _receipt(url: str, season: int | None, dest: Path, when: datetime) -> dict:
    """One line of the receipt: where a file came from and its fingerprint."""
    return {
        "url": url,
        "season": season,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "downloaded_at": when.isoformat(timespec="seconds"),
    }


def _save(manifest_path: Path, manifest: dict, key: str, entry: dict) -> None:
    manifest[key] = entry
    manifest_path.write_text(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="dataset keys to download")
    ap.add_argument("--seasons", nargs="*", type=int, help="limit to these seasons")
    ap.add_argument("--force", action="store_true", help="re-download files already on disk")
    args = ap.parse_args()
    download(args.only, args.seasons, args.force)


if __name__ == "__main__":
    main()
