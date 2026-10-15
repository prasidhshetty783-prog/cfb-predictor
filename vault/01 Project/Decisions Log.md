---
tags: [project, decisions]
---
# Decisions Log

Newest at the bottom. Each entry: what we decided, why, and what we gave up.
Use `Templates/Decision.md` for new entries.

### D-001 · 2026-10-14 · Title target is the 2026 season (January 2027 game)
- **Why:** the season was already under way, so the forecast can be checked within months.
- **Trade-off:** mid-season data only; preseason-only forecasts are not tested yet.

### D-002 · 2026-10-14 · Data from SportsDataverse public mirrors, not live APIs
- **Why:** free, versioned, covers 2001-2026, no API key needed. Every file gets a receipt
  with a sha256 fingerprint (`data/raw/_manifest.json`).
- **Trade-off:** updates come when the mirror updates (usually daily in season).
  See [[Data Sources]].

### D-003 · 2026-10-14 · Walk-forward testing on 2014-2025; tuning on 2010-2013 only
- **Why:** it copies real use (only the past is known). Random splits would leak future games.
- **Trade-off:** fewer training games for early test seasons. See [[Evaluation Protocol]].

### D-004 · 2026-10-14 · Log loss is the main score
- **Why:** it rewards honest probabilities, which is what "confidence score" needs. Accuracy
  alone can't tell a lucky 51% guess from a solid 90% one.
- **Trade-off:** harder to explain than accuracy, so the report shows both.

### D-005 · 2026-10-14 · Vegas lines are a benchmark only, never a model input
- **Why:** the product must predict games weeks ahead and whole seasons, before lines exist.
- **Trade-off:** the model stays about 0.01 log loss behind Vegas.

### D-006 · 2026-10-14 · Winner: Margin ridge (simplest model tied with the leader)
- **Why:** Margin LightGBM had the lowest log loss (0.4816), but Margin ridge (0.4822) is
  not significantly worse (Holm p = 0.74). The locked rule picks the simpler, explainable model.
- **Trade-off:** gives up 0.0006 log loss (not measurable) for a formula anyone can read.
  See [[Algorithm Comparison]].

### D-007 · 2026-10-14 · 2026 playoff rules: 12 teams, straight seeding
- **Rules used:** champions of the ACC, Big Ten, Big 12 and SEC get in. So does the
  highest-ranked team from the six other conferences (American, CUSA, MAC, Mountain West,
  Pac-12, Sun Belt), champion or not. Notre Dame gets in if ranked in the top 12. The rest are
  at-large picks. Seeds follow the committee ranking, so only the top four teams get byes.
  First round at the higher seed; no reseeding.
- **Why:** these are the 2026 rules (checked 2026-10-14 against NCAA.com, Feb 3 2026, and an AP
  preview, Aug 2026; one older article said "highest-ranked Group of Six champion", which the
  newer sources correct). A committee emulator stands in for the real committee.

### D-008 · 2026-10-14 · Hooks block real errors and only warn on style
- **Why:** errors (crashes, undefined names, data leaks) must never slip through, but style
  nags would slow work down. See [[How the Hooks Work]].

### D-009 · 2026-10-14 · Explanations target a reading grade of 9 (about age 14)
- **Why:** the user asked for expert work explained simply. A readability hook scores every answer.

### D-010 · 2026-10-14 · Luck and position-talent clues: not adopted
- **Why:** EXP-005 found no significant gain overall (log loss change of 0.0001 or less, Holm p = 1.0).
  Position talent helped in weeks 1-4 but hurt slightly later.
- **Trade-off:** none; the model is unchanged. Follow-up EXP-006 (fade talent as games are played) is pre-registered.

### D-011 · 2026-10-14 · Faded position talent: not adopted yet
- **Why:** EXP-006 improved every slice (all games -0.0007 log loss) but p = 0.24, so it may be luck.
- **Next:** track it alongside the live 2026 picks; re-test after the season.

### D-012 · 2026-10-14 · No softening for far-off picks
- **Why:** EXP-007: the tuning chose "no softening"; far-off percentages are already honest (within 0.6 points).
- **Instead:** the dashboard labels later weeks as early looks and shows accuracy by weeks ahead.

### D-013 · 2026-10-14 · Public dashboard reads files; it never runs the model
- **What:** Streamlit dashboard (`app/streamlit_app.py`) on Streamlit Community Cloud. It reads
  `reports/app/`, written by the `app_data` pipeline step. A GitHub Actions job runs the weekly
  pipeline every Monday and commits the new files.
- **Why:** past picks stay frozen (trust), the page is fast, and hosting is free. No "predict" button,
  because re-running the model on demand could quietly change old picks.
- **Trade-off:** updates once a week (or by hand from the Actions tab), not live during games.
