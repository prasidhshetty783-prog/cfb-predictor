"""Candidate clues for EXP-005: luck correction (#1) and position talent (#2).

Plain-language version
----------------------
* **Luck.** A team can win more games than it "deserved" (close wins, lucky bounces). We
  measure that as: real win share minus postgame win expectancy (how often a team that
  played like that would usually win). Luck tends to even out, so a lucky team is often
  weaker than its record says.
  - ``prev_luck``: last season's luck (known before this season starts).
  - ``s_luck``: this season's luck so far, from earlier games only.
  - ``s_close``: record in one-score games (8 points or less) so far, a classic luck sign.
* **Position talent.** Not all recruits matter equally. The quarterback room matters most,
  then the offensive line.
  - ``qb_talent``: average 247Sports grade of the team's top 2 quarterback recruits from the
    last 4 recruiting classes (this year's class included: they sign before the season).
  - ``ol_talent``: same for the top 5 offensive linemen.
  - ``portal_in_talent``: talent gained from transfers (we already used the net number).
  - ``portal_share``: share of the roster that is new through the portal (turnover).

Every number uses only information available before the game. These clues are NOT used by
the current model; the experiment decides whether they earn a place.
"""

from __future__ import annotations

import pandas as pd

from cfb.config import INTERIM, RAW

EXTRA_TEAM_FEATURES = ["prev_luck", "s_luck", "s_close", "qb_talent", "ol_talent",
                       "portal_in_talent", "portal_share"]
EXTRA_FEATURES = [f"d_{f}" for f in EXTRA_TEAM_FEATURES]
LUCK_FEATURES = ["d_prev_luck", "d_s_luck", "d_s_close"]
TALENT_FEATURES = ["d_qb_talent", "d_ol_talent", "d_portal_in_talent", "d_portal_share"]

# EXP-006 "challenger": position talent that fades as games are played (h chosen on 2010-2013)
FADE_H = 4.0
FADED_FEATURES = [f + "_fade" for f in TALENT_FEATURES]

QB_POS = {"QB", "PRO", "DUAL"}
OL_POS = {"OT", "OG", "IOL", "OC"}
CLOSE = 8          # one-score game


def _read(key: str) -> pd.DataFrame:
    files = sorted((RAW / key).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def position_talent() -> pd.DataFrame:
    rec = _read("recruits")
    rec = rec[["season", "team_id", "position", "grade"]].dropna()
    rec["season"] = rec["season"].astype(int)
    rec["team_id"] = rec["team_id"].astype("int64")
    rec = rec[rec["grade"] > 30]                                  # a few broken grades (e.g. 7)
    out = []
    for season in range(int(rec["season"].min()) + 3, int(rec["season"].max()) + 1):
        win = rec[rec["season"].between(season - 3, season)]
        for name, pos, top in (("qb_talent", QB_POS, 2), ("ol_talent", OL_POS, 5)):
            g = (win[win["position"].isin(pos)].sort_values("grade", ascending=False)
                 .groupby("team_id")["grade"].apply(lambda s, t=top: s.head(t).mean()))
            out.append(pd.DataFrame({"season": season, "team_id": g.index, "name": name, "v": g.values}))
    long = pd.concat(out, ignore_index=True)
    return long.pivot_table(index=["season", "team_id"], columns="name", values="v").reset_index()


def portal_extra() -> pd.DataFrame:
    tp = _read("transfer_portal")
    return pd.DataFrame({"season": tp["season"].astype(int), "team_id": tp["team_id"].astype("int64"),
                         "portal_in_talent": pd.to_numeric(tp["transfer_talent_in"], errors="coerce"),
                         "portal_share": pd.to_numeric(tp["portal_share"], errors="coerce")}
                        ).drop_duplicates(["season", "team_id"])


def prev_luck() -> pd.DataFrame:
    """Last season's (real win share - average postgame win expectancy), moved to this season."""
    g = pd.read_parquet(INTERIM / "games.parquet")
    g = g[g["completed"] & g["fbs_involved"] & g["margin"].notna()]
    st = pd.read_parquet(INTERIM / "team_game_stats.parquet")[["game_id", "team_id", "paper_share"]]
    rows = pd.concat([
        pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "team_id": g["home_id"],
                      "win": (g["margin"] > 0).astype(float)}),
        pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "team_id": g["away_id"],
                      "win": (g["margin"] < 0).astype(float)})])
    rows = rows.merge(st, on=["game_id", "team_id"], how="inner").dropna(subset=["paper_share"])
    agg = rows.groupby(["season", "team_id"]).agg(w=("win", "mean"), x=("paper_share", "mean"),
                                                  n=("win", "size")).reset_index()
    agg = agg[agg["n"] >= 4]
    return pd.DataFrame({"season": agg["season"].astype(int) + 1, "team_id": agg["team_id"].astype("int64"),
                         "prev_luck": agg["w"] - agg["x"]})


def priors() -> pd.DataFrame:
    p = position_talent().merge(portal_extra(), on=["season", "team_id"], how="outer")
    return p.merge(prev_luck(), on=["season", "team_id"], how="outer")


def extra_form(long: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """In-season luck from EARLIER games only (shifted cumulative sums), shrunk toward 0."""
    long = long.copy()
    done = long["completed"].fillna(False).astype(bool)
    luck = (long["win"] - long["paper_share"]).where(done & long["paper_share"].notna())
    close = done & (long["margin"].abs() <= CLOSE)
    keys = [long["team_id"], long["season"]]

    def prev_sum(x: pd.Series) -> pd.Series:
        return x.groupby(keys).transform(lambda s: s.cumsum().shift(1).fillna(0.0))

    out = long[["game_id", "team_id"]].copy()
    out["s_luck"] = prev_sum(luck.fillna(0.0)) / (prev_sum(luck.notna().astype(float)) + k)
    out["s_close"] = (prev_sum(((long["win"] - 0.5) * 2).where(close, 0.0).fillna(0.0))
                      / (prev_sum(close.astype(float)) + k))
    return out


def fill_missing(g: pd.DataFrame) -> pd.DataFrame:
    """Teams with no data: low talent for small schools, neutral (0) for luck and portal."""
    for side in ("home", "away"):
        for c in ("qb_talent", "ol_talent"):
            col = f"{side}_{c}"
            g[col] = g[col].fillna(g.groupby("season")[col].transform(lambda s: s.quantile(0.05)))
        for c in ("prev_luck", "s_luck", "s_close", "portal_in_talent", "portal_share"):
            g[f"{side}_{c}"] = g[f"{side}_{c}"].fillna(0.0)
    return g


def add_fade(t: pd.DataFrame, h: float = FADE_H) -> pd.DataFrame:
    """Position-talent clues x 1 / (1 + games so far / h); games so far ~ week of season - 1."""
    games = (t["week_order"].clip(lower=1, upper=14) - 1).astype(float)
    fade = 1.0 / (1.0 + games / h)
    for f in TALENT_FEATURES:
        t[f + "_fade"] = t[f] * fade
    return t
