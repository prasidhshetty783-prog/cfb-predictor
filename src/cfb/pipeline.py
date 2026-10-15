"""Run the whole CFB Predictor with one command.

Plain-language version: a checklist. It runs each step in order, shows how long each one
took, and stops at the first step that fails, so a broken step never feeds bad data to the
next one. Every run is logged in reports/pipeline_runs.log.

    python -m cfb.pipeline --weekly         in season, once a week (a few minutes)
    python -m cfb.pipeline                  rebuild + re-test everything from the saved raw data
                                            (slow: the full algorithm bake-off runs again)
    python -m cfb.pipeline --download       same as above, but download all raw data first
    python -m cfb.pipeline --list           show the steps without running them
    python -m cfb.pipeline --weekly --from season      resume at a step
    python -m cfb.pipeline --only report vault         run just these steps

Weekly mode also saves a dated copy of the picks and title odds in reports/archive/, so each
week's predictions can be scored later against what really happened (a true live test).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from cfb.config import CURRENT_SEASON, RAW, REPORTS, ROOT

FULL = {"full", "download"}
ALL = {"full", "download", "weekly"}


@dataclass
class Step:
    name: str
    what: str
    modes: set[str]
    module: list[str] = field(default_factory=list)     # run as: python -m <module...>
    func: Callable[[], None] | None = None              # or run this function in-process


def archive_picks() -> None:
    """Save this week's picks and title odds under reports/archive/<date>/."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest = REPORTS / "archive" / day
    dest.mkdir(parents=True, exist_ok=True)
    files = [REPORTS / f"predictions_{CURRENT_SEASON}_upcoming.csv",
             REPORTS / "championship" / f"title_odds_{CURRENT_SEASON}.csv",
             REPORTS / "championship" / f"title_odds_{CURRENT_SEASON}_meta.json"]
    for f in files:
        if f.exists():
            shutil.copy2(f, dest / f.name)
    print(f"archived {sum(f.exists() for f in files)} files -> {dest.relative_to(ROOT)}")


def check_live_data() -> None:
    """Warn loudly if the live season's game file was not refreshed by this run."""
    manifest = RAW / "_manifest.json"
    name = f"cfb_schedules_{CURRENT_SEASON}.csv"
    try:
        entry = json.loads(manifest.read_text())["games"]["files"][name]
        when = datetime.fromisoformat(entry["downloaded_at"])
    except Exception:
        print(f"WARNING: no download receipt for {name}; predictions may use old results.")
        return
    age_h = (datetime.now(timezone.utc) - when).total_seconds() / 3600
    if age_h > 2:
        print(f"WARNING: {name} was last downloaded {age_h:.0f} hours ago (the download may have "
              "failed). Predictions will use the results already on disk.")
    else:
        print(f"live data fresh: {name} downloaded {when:%Y-%m-%d %H:%M} UTC")


def check_model() -> None:
    """Make sure the saved model works with THIS computer's library versions.

    A model saved by one scikit-learn version may load with a warning (or not at all) in
    another. If that happens, retrain it here: same algorithm, same settings, same seasons.
    For the ridge model that takes seconds, and the result is the same formula.
    """
    import warnings

    import joblib
    import sklearn

    from cfb.config import MODELS_DIR
    from cfb.eval.backtest import load_table
    from cfb.models.predictor import Predictor

    path = MODELS_DIR / "final_model.joblib"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            joblib.load(path)
            stale = any("unpickle" in str(w.message).lower() for w in caught)
            why = "it was saved by a different scikit-learn version"
        except Exception as e:                       # missing file or incompatible pickle
            stale, why = True, f"it can't be loaded here ({type(e).__name__})"
    if not stale:
        print(f"saved model OK with scikit-learn {sklearn.__version__}")
        return
    card_path = MODELS_DIR / "model_card.json"
    card = json.loads(card_path.read_text())
    first, last = card["trained_on_seasons"]
    table = load_table()
    model = Predictor(card["algorithm"], card["params"]).fit(table[table["season"].between(first, last)])
    joblib.dump(model, path)
    card["retrained_with_sklearn"] = sklearn.__version__
    card_path.write_text(json.dumps(card, indent=1, default=str))
    print(f"retrained the final model because {why}: {card['algorithm']} {card['params']}, "
          f"seasons {first}-{last}, scikit-learn {sklearn.__version__}")


STEPS = [
    Step("download_all", "Download every season's raw data (files already saved are kept)", {"download"},
         ["cfb.data.download"]),
    Step("download_live", f"Re-download the {CURRENT_SEASON} season files (new results, polls, ratings)",
         {"weekly"}, ["cfb.data.download", "--seasons", str(CURRENT_SEASON), "--force"]),
    Step("check_live", "Check the live data really was refreshed", {"weekly"}, func=check_live_data),
    Step("clean", "Clean and join the raw files into tidy tables", ALL, ["cfb.data.clean"]),
    Step("audit", "Data checks (stops the run on a hard failure)", ALL, ["cfb.data.audit"]),
    Step("features", "Build the 40 pregame clues for every game", ALL, ["cfb.features.build"]),
    Step("backtest", "Bake-off: tune and test every algorithm on 2014-2025 (slowest step)", FULL,
         ["cfb.eval.backtest"]),
    Step("analyze", "Score the bake-off: metrics, significance tests, Vegas gap", FULL, ["cfb.eval.analyze"]),
    Step("finalize", "Pick the winner with the locked rule, train the final model, write the model card",
         FULL, ["cfb.models.finalize"]),
    Step("bracket", "Champion backtest: replay the 2014-2025 playoffs", FULL, ["cfb.sim.bracket"]),
    Step("committee", "Train and test the committee emulator", FULL, ["cfb.sim.committee"]),
    Step("model_check", "Make sure the saved model works with this computer's library versions", {"weekly"},
         func=check_model),
    Step("season", f"Simulate the rest of {CURRENT_SEASON} 10,000 times -> title odds", ALL, ["cfb.sim.season"]),
    Step("predict", "Pick every remaining game", ALL, ["cfb.predict"]),
    Step("track", "Log both versions' picks before kickoff; score finished games (live scorecard)", ALL,
         ["cfb.track"]),
    Step("app_data", "Write the small files the public dashboard reads (reports/app/)", ALL, ["cfb.report.app_data"]),
    Step("archive", "Save a dated copy of this week's picks and odds", {"weekly"}, func=archive_picks),
    Step("report", "Rebuild reports/model_report.html", ALL, ["cfb.report.build_report"]),
    Step("vault", "Refresh the auto notes in the Obsidian vault", ALL, ["cfb.report.vault_notes"]),
    Step("tests", "Leakage and simulator tests", ALL, ["pytest", "-q", "-p", "no:cacheprovider"]),
]
BY_NAME = {s.name: s for s in STEPS}


def plan(mode: str, start: str | None, only: list[str] | None) -> list[Step]:
    if only:
        unknown = [n for n in only if n not in BY_NAME]
        if unknown:
            sys.exit(f"Unknown step(s): {', '.join(unknown)}. Use --list to see them.")
        return [s for s in STEPS if s.name in only]
    steps = [s for s in STEPS if mode in s.modes]
    if start:
        names = [s.name for s in steps]
        if start not in names:
            sys.exit(f"Step '{start}' is not part of the {mode} run. Steps: {', '.join(names)}")
        steps = steps[names.index(start):]
    return steps


def run_step(step: Step) -> int:
    if step.func is not None:
        try:
            step.func()
            return 0
        except Exception as e:                      # report, then stop the pipeline
            print(f"{type(e).__name__}: {e}")
            return 1
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")     # Windows consoles: avoid UnicodeEncodeError
    return subprocess.run([sys.executable, "-u", "-m", *step.module], cwd=ROOT, env=env).returncode


def log_run(mode: str, results: list[tuple[str, float, int]], ok: bool) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    line = {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
            "ok": ok, "steps": [{"step": n, "seconds": round(t, 1), "exit": c} for n, t, c in results]}
    with (REPORTS / "pipeline_runs.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CFB Predictor pipeline.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--weekly", action="store_true", help="in-season refresh (new data -> odds -> report)")
    g.add_argument("--download", action="store_true", help="download all raw data, then the full rebuild")
    ap.add_argument("--from", dest="start", metavar="STEP", help="start at this step")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--list", action="store_true", help="show the steps and exit")
    a = ap.parse_args()
    mode = "weekly" if a.weekly else "download" if a.download else "full"

    if a.list:
        for m in ("weekly", "full", "download"):
            print(f"\n{m} run:")
            for i, s in enumerate((s for s in STEPS if m in s.modes), 1):
                print(f"  {i:2d}. {s.name:13s} {s.what}")
        return

    steps = plan(mode, a.start, a.only)
    print(f"CFB Predictor pipeline: {mode} run, {len(steps)} steps: {', '.join(s.name for s in steps)}")
    results: list[tuple[str, float, int]] = []
    t_all = time.time()
    for i, s in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {s.name}: {s.what}", flush=True)
        t0 = time.time()
        code = run_step(s)
        results.append((s.name, time.time() - t0, code))
        if code != 0:
            log_run(mode, results, ok=False)
            flag = "--weekly " if mode == "weekly" else "--download " if mode == "download" else ""
            print(f"\nSTOPPED: step '{s.name}' failed (exit code {code}). Fix the error above, then resume with:\n"
                  f"  python -m cfb.pipeline {flag}--from {s.name}")
            sys.exit(code)
        print(f"--- {s.name} done in {time.time() - t0:.0f}s", flush=True)
    log_run(mode, results, ok=True)
    print(f"\nAll {len(steps)} steps finished in {(time.time() - t_all) / 60:.1f} min. "
          "Open reports/model_report.html or vault/00 Home.md.")


if __name__ == "__main__":
    main()
