"""Turn the raw downloads into clean, joined tables in ``data/interim/``.

Plain-language version
----------------------
Raw data is like groceries straight from the store: some boxes are labeled
differently, some items are missing, some are duplicated. This script unpacks
everything and puts it on labeled shelves:

* ``games.parquet``          one row per game (2001-2026), bowls and playoffs included
* ``team_season.parquet``    one row per team per season: talent, returning players,
                             transfer portal, and LAST season's final rating
* ``team_game_stats.parquet``one row per team per game: yards per play, success rate,
                             explosiveness, havoc, turnovers (used only for FUTURE games)
* ``polls.parquet``          weekly AP / Coaches / CFP rankings
* ``market.parquet``         the Vegas point spread for each game (benchmark only)

Run:  python -m cfb.data.clean
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from cfb.config import INTERIM, RAW


# --------------------------------------------------------------------------- helpers
def _read_all(key: str, pattern: str = "*") -> pd.DataFrame:
    """Read and stack every file of one raw dataset."""
    files = sorted(glob.glob(str(RAW / key / pattern)))
    frames = []
    for f in files:
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(f))
        elif f.endswith((".csv", ".csv.gz")):
            try:
                frames.append(pd.read_csv(f, low_memory=False))
            except UnicodeDecodeError:  # some sources are not UTF-8
                frames.append(pd.read_csv(f, low_memory=False, encoding="latin-1"))
    if not frames:
        raise FileNotFoundError(f"No raw files for '{key}'. Run: python -m cfb.data.download")
    return pd.concat(frames, ignore_index=True)


def _to_int_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _to_bool(s: pd.Series) -> pd.Series:
    """Turn True/False/'TRUE'/1/0 into real booleans (missing -> False)."""
    if s.dtype == bool:
        return s
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    return s.astype(str).str.strip().str.lower().map(m).eq(True)    # unknown/missing -> False


# --------------------------------------------------------------------------- teams
def build_team_info() -> pd.DataFrame:
    """Team identity per season: name, conference, division (fbs/fcs/ii/iii), stadium location."""
    ti = _read_all("team_info", "cfb_team_info_*.parquet")
    if "season" not in ti.columns:  # older files have no season column: recover it from the filename
        parts = []
        for f in sorted(glob.glob(str(RAW / "team_info" / "cfb_team_info_*.parquet"))):
            d = pd.read_parquet(f)
            d["season"] = int(f.rsplit("_", 1)[-1].split(".")[0])
            parts.append(d)
        ti = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({
        "season": ti["season"].astype(int),
        "team_id": _to_int_id(ti["team_id"]),
        "school": ti["school"],
        "conference": ti["conference"],
        "classification": ti["classification"],
        "lat": pd.to_numeric(ti.get("latitude"), errors="coerce"),
        "lon": pd.to_numeric(ti.get("longitude"), errors="coerce"),
    })
    return out.dropna(subset=["team_id"]).drop_duplicates(["season", "team_id"])


# --------------------------------------------------------------------------- games
def _cfbd_games() -> pd.DataFrame:
    g = _read_all("games", "cfb_schedules_*.csv")
    out = pd.DataFrame({
        "game_id": _to_int_id(g["game_id"]),
        "season": g["season"].astype(int),
        "week": pd.to_numeric(g["week"], errors="coerce").astype("Int64"),
        "season_type": g["season_type"].astype(str),
        "start_date": pd.to_datetime(g["start_date"], utc=True, errors="coerce"),
        "neutral_site": _to_bool(g["neutral_site"]),
        "conference_game": _to_bool(g["conference_game"]),
        "venue_id": _to_int_id(g["venue_id"]),
        "home_id": _to_int_id(g["home_id"]),
        "home_team": g["home_team"],
        "home_conference": g["home_conference"],
        "home_division": g["home_division"],
        "home_points": pd.to_numeric(g["home_points"], errors="coerce"),
        "away_id": _to_int_id(g["away_id"]),
        "away_team": g["away_team"],
        "away_conference": g["away_conference"],
        "away_division": g["away_division"],
        "away_points": pd.to_numeric(g["away_points"], errors="coerce"),
        "completed": _to_bool(g["completed"]),
        "notes": g["notes"],
        # CFBD's own pregame Elo: kept ONLY as a benchmark to beat
        "cfbd_home_elo": pd.to_numeric(g["home_pregame_elo"], errors="coerce"),
        "cfbd_away_elo": pd.to_numeric(g["away_pregame_elo"], errors="coerce"),
        "source": "cfbd",
    })
    return out


def _espn_postseason(team_info: pd.DataFrame) -> pd.DataFrame:
    """Bowls + playoff games 2004-2022 (missing from the main schedule files)."""
    e = _read_all("espn_schedules", "*.parquet")
    e = e[pd.to_numeric(e["season_type"], errors="coerce") == 3].copy()
    out = pd.DataFrame({
        "game_id": _to_int_id(e["game_id"]),
        "season": pd.to_numeric(e["season"]).astype(int),
        "week": pd.Series(1, index=e.index, dtype="Int64"),
        "season_type": "postseason",
        "start_date": pd.to_datetime(e["start_date"], utc=True, errors="coerce"),
        "neutral_site": _to_bool(e["neutral_site"]),
        "conference_game": _to_bool(e["conference_competition"]),
        "venue_id": _to_int_id(e["venue_id"]),
        "home_id": _to_int_id(e["home_id"]),
        "home_points": pd.to_numeric(e["home_score"], errors="coerce"),
        "away_id": _to_int_id(e["away_id"]),
        "away_points": pd.to_numeric(e["away_score"], errors="coerce"),
        "completed": _to_bool(e["status_type_completed"]),
        "notes": e["notes_headline"],
        "cfbd_home_elo": np.nan,
        "cfbd_away_elo": np.nan,
        "source": "espn_post",
        "_home_loc": e["home_location"],
        "_away_loc": e["away_location"],
    })
    ti = team_info[["season", "team_id", "school", "conference", "classification"]]
    for side in ("home", "away"):
        m = out.merge(ti, left_on=["season", f"{side}_id"], right_on=["season", "team_id"],
                      how="left")
        out[f"{side}_team"] = m["school"].fillna(m[f"_{side}_loc"]).values
        out[f"{side}_conference"] = m["conference"].values
        out[f"{side}_division"] = m["classification"].values
    return out.drop(columns=["_home_loc", "_away_loc"])


def _tag_playoff(games: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
    """Mark bowl games and College Football Playoff rounds."""
    g = games
    notes = g["notes"].fillna("").str.lower()
    # Only top-division (FBS vs FBS) postseason games can be title games. This keeps the
    # FCS / Division II / Division III playoffs ("FCS Championship - First Round") out.
    post = g["season_type"].eq("postseason") & g["fbs_both"]
    cfp_text = notes.str.contains("college football playoff|cfp")
    g["is_bowl"] = g["season_type"].eq("postseason")
    g["cfp_round"] = pd.Series(pd.NA, index=g.index, dtype="string")
    g.loc[post & cfp_text & notes.str.contains("first round"), "cfp_round"] = "first_round"
    g.loc[post & cfp_text & notes.str.contains("quarterfinal"), "cfp_round"] = "quarterfinal"
    g.loc[post & cfp_text & notes.str.contains("semifinal"), "cfp_round"] = "semifinal"
    g.loc[post & notes.str.contains("national championship")
          & (cfp_text | notes.str.contains("bcs")), "cfp_round"] = "final"
    # 4-team era (2014-2023): a semifinal is a bowl between two of the final CFP top 4.
    # Conference title games (early December) are excluded by the date rule.
    bowl_time = (g["start_date"].dt.month == 1) | (
        (g["start_date"].dt.month == 12) & (g["start_date"].dt.day >= 15))
    cfp = polls[polls["poll"].eq("cfp")]
    for season in range(2014, 2024):
        c = cfp[cfp["season"] == season]
        if c.empty:
            continue
        last = c[c["week_order"] == c["week_order"].max()]
        top4 = set(last.loc[last["rank"] <= 4, "team_id"].astype(int))
        mask = ((g["season"] == season) & post & bowl_time
                & g["home_id"].isin(top4) & g["away_id"].isin(top4) & g["cfp_round"].isna())
        g.loc[mask, "cfp_round"] = "semifinal"
    # 2004-2005: the BCS title game was a regular bowl (Orange / Rose) between AP #1 and #2
    ap = polls[polls["poll"].eq("ap") & polls["season_type"].eq(2)]
    for season in (2004, 2005):
        a = ap[ap["season"] == season]
        if a.empty:
            continue
        last = a[a["week_order"] == a["week_order"].max()]
        top2 = set(last.loc[last["rank"] <= 2, "team_id"].astype(int))
        mask = ((g["season"] == season) & post & g["home_id"].isin(top2)
                & g["away_id"].isin(top2))
        g.loc[mask, "cfp_round"] = "final"
    g["is_playoff"] = g["cfp_round"].notna() & (g["season"] >= 2014)
    return g


def _drop_cross_source_duplicates(cfbd: pd.DataFrame, espn: pd.DataFrame) -> pd.DataFrame:
    """ESPN sometimes files a game (e.g. 2013 conference title games) as 'postseason'
    under a different id than the main source. Same two teams within 3 days = same game.
    (A real rematch, like the 2021 SEC title game and national final, is weeks apart.)"""
    def pair(d: pd.DataFrame) -> pd.Series:
        lo = d[["home_id", "away_id"]].min(axis=1).astype("int64").astype(str)
        hi = d[["home_id", "away_id"]].max(axis=1).astype("int64").astype(str)
        return lo + "-" + hi
    c = cfbd.assign(_pair=pair(cfbd))[["_pair", "start_date"]]
    e = espn.assign(_pair=pair(espn), _row=np.arange(len(espn)))
    m = e[["_pair", "start_date", "_row"]].merge(c, on="_pair", suffixes=("", "_c"))
    close = (m["start_date"] - m["start_date_c"]).abs() <= pd.Timedelta(days=3)
    dup_rows = set(m.loc[close, "_row"])
    return espn[~e["_row"].isin(dup_rows).values]


def build_games(team_info: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
    cfbd = _cfbd_games()
    espn = _espn_postseason(team_info)
    espn = espn[~espn["game_id"].isin(cfbd["game_id"])]          # never double count
    espn = espn.dropna(subset=["home_id", "away_id", "start_date"])
    espn = _drop_cross_source_duplicates(cfbd.dropna(subset=["home_id", "away_id"]), espn)
    g = pd.concat([cfbd, espn], ignore_index=True)
    g = g.dropna(subset=["game_id", "home_id", "away_id"])
    g = g.drop_duplicates("game_id", keep="first")
    g["fbs_involved"] = g["home_division"].eq("fbs") | g["away_division"].eq("fbs")
    g["fbs_both"] = g["home_division"].eq("fbs") & g["away_division"].eq("fbs")
    g["has_score"] = g["home_points"].notna() & g["away_points"].notna()
    # A 0-0 "final" is a cancelled or forfeited game, not a real result.
    zero_zero = g["home_points"].eq(0) & g["away_points"].eq(0)
    g["completed"] = g["completed"] & g["has_score"] & ~zero_zero
    g["margin"] = g["home_points"] - g["away_points"]             # outcome (never a feature)
    g["home_win"] = np.where(g["completed"], (g["margin"] > 0).astype(float), np.nan)
    # Order inside a season: regular weeks 1..16, then all postseason games as week 20
    g["week_order"] = np.where(g["season_type"].eq("postseason"), 20,
                               g["week"].fillna(0).astype(int))
    g = _tag_playoff(g, polls)
    g = g.sort_values(["season", "week_order", "start_date", "game_id"]).reset_index(drop=True)
    return g


# --------------------------------------------------------------------------- polls
def build_polls() -> pd.DataFrame:
    p = _read_all("poll_ranks", "*.parquet")
    out = pd.DataFrame({
        "season": p["season"].astype(int),
        "season_type": pd.to_numeric(p["season_type"], errors="coerce").astype("Int64"),
        "week": pd.to_numeric(p["week"], errors="coerce").astype("Int64"),
        "poll": p["poll"].astype(str).str.lower(),
        "team_id": _to_int_id(p["team_id"]),
        "rank": pd.to_numeric(p["rank"], errors="coerce"),
        "points": pd.to_numeric(p.get("points"), errors="coerce"),
    })
    # postseason polls (season_type 3) come after every regular week
    out["week_order"] = np.where(out["season_type"].eq(3), 21, out["week"].fillna(0).astype(int))
    return out.dropna(subset=["team_id", "rank"])


# --------------------------------------------------------------------------- team-season priors
def build_team_season() -> pd.DataFrame:
    """Things known BEFORE a season starts (except last season's rating, which is known too)."""
    tal = _read_all("talent")
    tal = pd.DataFrame({"season": tal["season"].astype(int), "team_id": _to_int_id(tal["team_id"]),
                        "talent": pd.to_numeric(tal["talent_composite"], errors="coerce"),
                        "blue_chip": pd.to_numeric(tal["blue_chip_ratio"], errors="coerce")})
    rp = _read_all("returning_production")
    rp = pd.DataFrame({"season": rp["season"].astype(int), "team_id": _to_int_id(rp["team_id"]),
                       "ret_off": pd.to_numeric(rp["off_returning"], errors="coerce"),
                       "ret_def": pd.to_numeric(rp["def_returning"], errors="coerce"),
                       "ret_all": pd.to_numeric(rp["overall_returning"], errors="coerce")})
    tp = _read_all("transfer_portal")
    tp = pd.DataFrame({"season": tp["season"].astype(int), "team_id": _to_int_id(tp["team_id"]),
                       "portal_net": pd.to_numeric(tp["net_transfer_talent"], errors="coerce"),
                       "portal_in": pd.to_numeric(tp["transfers_in_n"], errors="coerce"),
                       "portal_out": pd.to_numeric(tp["transfers_out_n"], errors="coerce")})
    rf = _read_all("ratings_final")
    rf = pd.DataFrame({"season": rf["season"].astype(int) + 1,     # LAST season -> this season
                       "team_id": _to_int_id(rf["team_id"]),
                       "prev_adj_net": pd.to_numeric(rf["adj_net"], errors="coerce"),
                       "prev_adj_off": pd.to_numeric(rf["adj_off_epa"], errors="coerce"),
                       "prev_adj_def": pd.to_numeric(rf["adj_def_epa"], errors="coerce"),
                       "prev_fei_net": pd.to_numeric(rf["fei_net"], errors="coerce")})
    ts = tal.merge(rp, on=["season", "team_id"], how="outer")
    ts = ts.merge(tp, on=["season", "team_id"], how="outer")
    ts = ts.merge(rf, on=["season", "team_id"], how="outer")
    ts = ts.dropna(subset=["team_id"]).drop_duplicates(["season", "team_id"])
    return ts.sort_values(["season", "team_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- per-game team stats
def _split_pair(s: pd.Series, sep: str) -> tuple[pd.Series, pd.Series]:
    parts = s.astype(str).str.split(sep, n=1, expand=True)
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    return a, b


def build_team_game_stats() -> pd.DataFrame:
    """Per team per game box-score and advanced stats. Used ONLY to describe PAST games."""
    tb = _read_all("team_box")
    comp, att = _split_pair(tb["completionAttempts"], "/")
    rush_att = pd.to_numeric(tb["rushingAttempts"], errors="coerce")
    box = pd.DataFrame({
        "game_id": _to_int_id(tb["game_id"]),
        "team_id": _to_int_id(tb["team_id"]),
        "yards": pd.to_numeric(tb["totalYards"], errors="coerce"),
        "plays": att + rush_att,
        "turnovers": pd.to_numeric(tb["turnovers"], errors="coerce"),
    })
    box["ypp"] = box["yards"] / box["plays"].where(box["plays"] > 0)
    box = box.dropna(subset=["game_id", "team_id"]).drop_duplicates(["game_id", "team_id"])
    # opponent's numbers -> defense
    opp = box.rename(columns={"team_id": "opp_id", "yards": "opp_yards", "plays": "opp_plays",
                              "turnovers": "opp_turnovers", "ypp": "opp_ypp"})
    box = box.merge(opp, on="game_id")
    box = box[box["team_id"] != box["opp_id"]]

    adv = _read_all("game_stats_advanced")
    keep = ["success_margin", "explosive_margin", "explosive_epa_margin", "havoc_margin",
            "turnovers_margin", "field_position_margin", "pts_per_opp_margin",
            "opp_conversion_margin", "paper_share"]
    adv = pd.concat([pd.DataFrame({"game_id": _to_int_id(adv["game_id"]),
                                   "team_id": _to_int_id(adv["team_id"])}),
                     adv[keep].apply(pd.to_numeric, errors="coerce")], axis=1)
    adv = adv.dropna(subset=["game_id", "team_id"]).drop_duplicates(["game_id", "team_id"])
    out = box.merge(adv, on=["game_id", "team_id"], how="outer")
    return out


# --------------------------------------------------------------------------- market (benchmark)
def _abbr_to_team(lines: pd.DataFrame) -> dict[str, int]:
    """Learn which sportsbook abbreviation means which team.

    Trick: an abbreviation like 'UTH' shows up in every Utah game. Utah's team id is
    the only id present in ALL those games (the opponents keep changing), so the
    most frequent id among its games is the answer.

    Ties: a small school's abbreviation may appear in just one game (say 'LAF' in
    Army vs Lafayette). Then Army and Lafayette are tied. We give it to the team that no
    clearer abbreviation already owns ('ARMY' owns Army after 100+ games, so 'LAF' is
    Lafayette). If that still doesn't decide it, the abbreviation is skipped: no line is
    better than a line with the wrong sign. Sorting is fully deterministic, so every
    computer and every pandas version gets the same answer.
    """
    rows = lines.dropna(subset=["abbr", "home_team_id", "away_team_id"])
    pairs = rows.drop_duplicates(["game_id", "abbr"])
    long = pd.concat([pairs[["abbr", "home_team_id"]].rename(columns={"home_team_id": "tid"}),
                      pairs[["abbr", "away_team_id"]].rename(columns={"away_team_id": "tid"})])
    long["tid"] = long["tid"].astype("int64")
    counts = long.groupby(["abbr", "tid"]).size().rename("n").reset_index()
    counts = counts.join(pairs.groupby("abbr").size().rename("games"), on="abbr")
    counts = counts[counts["n"] >= 0.8 * counts["games"]]          # must be in >=80% of its games
    top = counts[counts["n"] == counts.groupby("abbr")["n"].transform("max")]
    top = top.sort_values(["abbr", "tid"], kind="mergesort")
    size = top.groupby("abbr")["tid"].transform("size")
    out = {str(a): int(t) for a, t in zip(top.loc[size == 1, "abbr"], top.loc[size == 1, "tid"], strict=True)}
    owned = set(out.values())
    for abbr, grp in top[size > 1].groupby("abbr", sort=True):
        free = [int(t) for t in grp["tid"] if int(t) not in owned]
        if len(free) == 1:
            out[str(abbr)] = free[0]
    return out


def build_market() -> pd.DataFrame:
    """Consensus closing point spread per game (median across sportsbooks).

    ``home_spread`` < 0 means the home team is favored. Used ONLY as a benchmark.
    Main source: multi-book lines (2006-2025). Backup: ESPN lines (2012+).
    """
    bl = pd.read_csv(RAW / "betting_lines" / "cfb_line_odds.csv.gz", low_memory=False)
    sp = bl[bl["market_type"].eq("spread")].dropna(subset=["game_id", "lines"]).copy()
    amap = _abbr_to_team(sp)
    sp["tid"] = sp["abbr"].map(amap)
    sp["home_line"] = np.where(sp["tid"].eq(sp["home_team_id"]), sp["lines"],
                               np.where(sp["tid"].eq(sp["away_team_id"]), -sp["lines"], np.nan))
    sp = sp.dropna(subset=["home_line"])
    per_book = sp.groupby(["game_id", "book"])["home_line"].median()
    cons = per_book.groupby("game_id").agg(home_spread="median", n_books="size",
                                           spread_sd="std").reset_index()
    cons["game_id"] = _to_int_id(cons["game_id"])
    cons["market_source"] = "books"

    # ESPN backup. Audit finding: ESPN's home/away sign is flipped for about a third
    # of 2006-2012 games (its "favorites" won only 27% of those). From 2013 on it agrees
    # with the multi-book consensus about 98% of the time, so only 2013+ is trusted.
    eb = _read_all("espn_betting")
    eb = eb[pd.to_numeric(eb["season"], errors="coerce") >= 2013]
    espn = pd.DataFrame({
        "game_id": _to_int_id(eb["game_id"]),
        "espn_spread": pd.to_numeric(eb["home_team_spread"], errors="coerce"),
    }).dropna().drop_duplicates("game_id")
    out = cons.merge(espn, on="game_id", how="outer")
    use_espn = out["home_spread"].isna() & out["espn_spread"].notna()
    out.loc[use_espn, "home_spread"] = out.loc[use_espn, "espn_spread"]
    out.loc[use_espn, "market_source"] = "espn"
    out["n_books"] = out["n_books"].fillna(1)
    return out.dropna(subset=["game_id", "home_spread"]).drop_duplicates("game_id")


# --------------------------------------------------------------------------- main
def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    team_info = build_team_info()
    polls = build_polls()
    games = build_games(team_info, polls)
    tables = {
        "team_info": team_info,
        "polls": polls,
        "games": games,
        "team_season": build_team_season(),
        "team_game_stats": build_team_game_stats(),
        "market": build_market(),
    }
    for name, df in tables.items():
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)
        print(f"{name:16s} rows={len(df):7d} cols={df.shape[1]:3d}")


if __name__ == "__main__":
    main()
