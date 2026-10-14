"""Build the model table: one row per game, only information known BEFORE kickoff.

Plain-language version
----------------------
For every game we write down a "scouting report" about both teams as it looked
the morning of the game:

* **Ratings**: Elo, plus opponent-adjusted power ratings for points, success
  rate, yards per play and postgame win expectancy (see ``ratings.py``).
* **Roster**: recruiting talent, share of production returning, transfer
  portal gains, last season's final rating, power-conference membership.
* **Form**: this season's results so far (average margin, success rate,
  explosiveness, havoc, turnovers), rest days, and the AP poll rank.
* **Context**: home field, neutral site, conference game, bowl/playoff, week,
  travel distance.

Most team numbers are given as a difference, "home minus away" (names start
with ``d_``), because the matchup is what matters.

The table also keeps the ANSWERS (``home_win``, ``margin``) and the BENCHMARKS
(Vegas spread, CFBD Elo) in clearly separate columns that are never fed to a model.

Run:  python -m cfb.features.build
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb.config import FEATURE_START, INTERIM, LEAKY_COLUMNS, PROCESSED
from cfb.features import extra
from cfb.features.elo import EloParams, run_elo, tune_elo
from cfb.features.ratings import RidgeParams, sequential_ratings

POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}

# Team-level features (each exists as home_<x>, away_<x>, and d_<x> = home - away)
TEAM_FEATURES = [
    "elo", "r_pts", "r_sr", "r_ypp", "r_pw",
    "talent", "blue_chip", "ret_off", "ret_def", "portal_net", "prev_adj_net",
    "is_power", "is_fbs",
    "gp", "win_pct", "s_margin", "s_sr", "s_expl", "s_havoc", "s_to", "s_ypp", "s_pw",
    "last3_margin", "rest", "ap_rank",
]
CONTEXT_FEATURES = ["hf", "conference_game", "postseason", "playoff", "week_order",
                    "travel_km", "pred_pts", "pred_sr", "pred_ypp", "pred_pw", "elo_logit"]
MODEL_FEATURES = [f"d_{f}" for f in TEAM_FEATURES] + CONTEXT_FEATURES + [
    "home_talent", "away_talent", "home_is_power", "away_is_power"]
TARGETS = ["home_win", "margin"]
BENCHMARKS = ["home_spread", "cfbd_elo_diff", "elo_prob"]


# --------------------------------------------------------------------------- inputs
def _game_frame() -> pd.DataFrame:
    g = pd.read_parquet(INTERIM / "games.parquet")
    g = g[g["fbs_involved"]].copy()
    g["home_field"] = (~g["neutral_site"]).astype(int)
    g["home_non_fbs"] = (g["home_division"] != "fbs").astype(int)
    g["away_non_fbs"] = (g["away_division"] != "fbs").astype(int)
    tgs = pd.read_parquet(INTERIM / "team_game_stats.parquet")
    st = tgs[["game_id", "team_id", "success_margin", "ypp", "opp_ypp", "paper_share",
              "explosive_margin", "havoc_margin", "turnovers_margin"]].copy()
    st["ypp_margin"] = st["ypp"] - st["opp_ypp"]
    st["pw_margin"] = 2.0 * st["paper_share"] - 1.0
    home = st.rename(columns={"team_id": "home_id"})[
        ["game_id", "home_id", "success_margin", "ypp_margin", "pw_margin"]]
    g = g.merge(home, on=["game_id", "home_id"], how="left")
    g = g.rename(columns={"success_margin": "sr_target", "ypp_margin": "ypp_target",
                          "pw_margin": "pw_target"})
    g["pts_target"] = g["margin"]
    # a stat target only counts if the game actually finished
    for t in ("pts_target", "sr_target", "ypp_target", "pw_target"):
        g.loc[~g["completed"], t] = np.nan
    return g.sort_values(["season", "week_order", "start_date", "game_id"]).reset_index(drop=True), st


# --------------------------------------------------------------------------- season-to-date form
def _team_long(g: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team), team's point of view."""
    base = ["game_id", "season", "week_order", "start_date", "completed"]
    h = g[base + ["home_id", "away_id", "margin"]].rename(
        columns={"home_id": "team_id", "away_id": "opp_id"})
    a = g[base + ["away_id", "home_id", "margin"]].rename(
        columns={"away_id": "team_id", "home_id": "opp_id"})
    a["margin"] = -a["margin"]
    long = pd.concat([h, a], ignore_index=True)
    long = long.merge(st, on=["game_id", "team_id"], how="left")
    long["win"] = np.where(long["completed"], (long["margin"] > 0).astype(float), np.nan)
    return long.sort_values(["team_id", "start_date", "game_id"]).reset_index(drop=True)


def _form(long: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Season-to-date averages from PREVIOUS games only, shrunk toward 0 by k pseudo-games."""
    stats = {"margin": "s_margin", "success_margin": "s_sr", "explosive_margin": "s_expl",
             "havoc_margin": "s_havoc", "turnovers_margin": "s_to", "ypp_margin": "s_ypp",
             "pw_margin": "s_pw"}
    long = long.copy()
    long["ypp_margin"] = long["ypp"] - long["opp_ypp"]
    long["pw_margin"] = 2.0 * long["paper_share"] - 1.0
    played = long["completed"].astype(float)
    grp = long.groupby(["team_id", "season"], sort=False)
    out = long[["game_id", "team_id"]].copy()
    prev_n = grp["completed"].transform(lambda s: s.astype(float).cumsum().shift(1).fillna(0.0))
    out["gp"] = prev_n
    wins = long["win"].fillna(0.0) * played
    prev_w = wins.groupby([long["team_id"], long["season"]]).transform(
        lambda s: s.cumsum().shift(1).fillna(0.0))
    out["win_pct"] = (prev_w + 0.5 * k) / (prev_n + k)              # shrunk toward .500
    for src, name in stats.items():
        v = long[src].where(long["completed"]).fillna(0.0)
        cnt = long[src].where(long["completed"]).notna().astype(float)
        s_sum = v.groupby([long["team_id"], long["season"]]).transform(
            lambda s: s.cumsum().shift(1).fillna(0.0))
        s_cnt = cnt.groupby([long["team_id"], long["season"]]).transform(
            lambda s: s.cumsum().shift(1).fillna(0.0))
        out[name] = s_sum / (s_cnt + k)
    m = long["margin"].where(long["completed"])
    out["last3_margin"] = m.groupby([long["team_id"], long["season"]]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    last_date = long.groupby(["team_id", "season"])["start_date"].shift(1)
    rest = (long["start_date"] - last_date).dt.total_seconds() / 86400.0
    out["rest"] = rest.clip(upper=21).fillna(21.0)                  # first game: fully rested
    return out


# --------------------------------------------------------------------------- polls
def _ap_rank(g: pd.DataFrame) -> pd.DataFrame:
    """AP rank from the latest poll released BEFORE the game (unranked = 30).

    Verified in tests/test_poll_timing.py: the poll labeled week W is published before
    week W games. The post-bowl final poll (week_order 21) is never used.
    """
    p = pd.read_parquet(INTERIM / "polls.parquet")
    ap = p[p["poll"].eq("ap")][["season", "week_order", "team_id", "rank"]]
    rows = []
    for side in ("home", "away"):
        q = g[["game_id", "season", "week_order", f"{side}_id"]].rename(
            columns={f"{side}_id": "team_id"})
        m = q.merge(ap, on=["season", "team_id"], how="left", suffixes=("", "_poll"))
        m = m[m["week_order_poll"] <= m["week_order"]]
        m = m.sort_values("week_order_poll").drop_duplicates(["game_id", "team_id"], keep="last")
        q = q.merge(m[["game_id", "team_id", "rank"]], on=["game_id", "team_id"], how="left")
        rows.append(q[["game_id", "team_id", "rank"]])
    r = pd.concat(rows).drop_duplicates(["game_id", "team_id"])
    r["ap_rank"] = r["rank"].fillna(30.0)
    return r[["game_id", "team_id", "ap_rank"]]


# --------------------------------------------------------------------------- priors & travel
def _priors(g: pd.DataFrame) -> pd.DataFrame:
    ts = pd.read_parquet(INTERIM / "team_season.parquet")
    ti = pd.read_parquet(INTERIM / "team_info.parquet")
    base = ti[["season", "team_id", "conference", "classification", "school", "lat", "lon"]]
    pri = base.merge(ts, on=["season", "team_id"], how="left")
    pri["is_power"] = (pri["conference"].isin(POWER)
                       & ~((pri["conference"] == "Pac-12") & (pri["season"] >= 2024))
                       | (pri["school"] == "Notre Dame")).astype(int)
    pri.loc[(pri["conference"] == "Big East") & (pri["season"] <= 2012), "is_power"] = 1
    pri["is_fbs"] = (pri["classification"] == "fbs").astype(int)
    return pri


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- main builder
def build(elo_params: EloParams | None = None,
          ridge: dict[str, RidgeParams] | None = None) -> tuple[pd.DataFrame, dict]:
    g, st = _game_frame()
    meta: dict = {}

    # 1) Elo (tuned on 2005-2013 only, unless params are given)
    if elo_params is None:
        elo_params, table = tune_elo(g, list(range(2005, 2014)))
        meta["elo_tuning_top5"] = table.head(5).to_dict(orient="records")
    meta["elo_params"] = elo_params.__dict__
    g = g.merge(run_elo(g, elo_params), on="game_id", how="left")

    # 2) Opponent-adjusted ratings
    # Settings chosen by scripts/tune_ratings.py on 2005-2013 only (see reports/tuning/)
    ridge = ridge or {"pts_target": RidgeParams(lam=1.0, prev_weight=0.35, cap=35.0),
                      "sr_target": RidgeParams(lam=1.0, prev_weight=0.35, cap=None),
                      "ypp_target": RidgeParams(lam=1.0, prev_weight=0.35, cap=5.0),
                      "pw_target": RidgeParams(lam=1.0, prev_weight=0.35, cap=None)}
    meta["ridge_params"] = {k: v.__dict__ for k, v in ridge.items()}
    rat, coefs = sequential_ratings(g, ridge, return_coefs=True)
    rat = rat.rename(columns=lambda c: c.replace("_target", ""))
    coefs["target"] = coefs["target"].str.replace("_target", "", regex=False)
    meta["_coefs"] = coefs
    g = g.merge(rat, on="game_id", how="left")

    # 3) Form, polls, priors
    long = _team_long(g, st)
    form = _form(long)
    ap = _ap_rank(g)
    pri = _priors(g)
    xform = extra.extra_form(long)
    xpri = extra.priors()
    team_cols = ["talent", "blue_chip", "ret_off", "ret_def", "portal_net", "prev_adj_net",
                 "is_power", "is_fbs", "lat", "lon"]
    for side in ("home", "away"):
        tid = f"{side}_id"
        f = form.rename(columns={"team_id": tid})
        g = g.merge(f.add_prefix(f"{side}_").rename(
            columns={f"{side}_game_id": "game_id", f"{side}_{tid}": tid}),
            on=["game_id", tid], how="left")
        a = ap.rename(columns={"team_id": tid, "ap_rank": f"{side}_ap_rank"})
        g = g.merge(a, on=["game_id", tid], how="left")
        p = pri[["season", "team_id"] + team_cols].rename(
            columns={"team_id": tid, **{c: f"{side}_{c}" for c in team_cols}})
        g = g.merge(p, on=["season", tid], how="left")
        xf = xform.rename(columns={"team_id": tid, **{c: f"{side}_{c}" for c in ("s_luck", "s_close")}})
        g = g.merge(xf, on=["game_id", tid], how="left")
        xp = xpri.rename(columns={"team_id": tid, **{c: f"{side}_{c}" for c in xpri.columns
                                                    if c not in ("season", "team_id")}})
        g = g.merge(xp, on=["season", tid], how="left")
        g[f"{side}_elo"] = g[f"elo_{side}"]
        for t in ("pts", "sr", "ypp", "pw"):
            g[f"{side}_r_{t}"] = g[f"r_{t}_{side}"]

    # Fill priors for teams with no data (small schools): a low, season-specific value
    for c in ["talent", "blue_chip", "ret_off", "ret_def", "portal_net", "prev_adj_net"]:
        for side in ("home", "away"):
            col = f"{side}_{c}"
            low = g.groupby("season")[col].transform(lambda s: s.quantile(0.05))
            if c in ("ret_off", "ret_def"):                         # unknown -> typical
                low = g.groupby("season")[col].transform("median")
            if c == "portal_net":
                low = pd.Series(0.0, index=g.index)
            g[col] = g[col].fillna(low)
    g["home_is_fbs"] = g["home_is_fbs"].fillna(1 - g["home_non_fbs"])
    g["away_is_fbs"] = g["away_is_fbs"].fillna(1 - g["away_non_fbs"])
    # Small schools' games against each other are missing before 2022, so their "days of
    # rest" and "games played" are unreliable. Use neutral values for non-FBS teams.
    for side in ("home", "away"):
        small = g[f"{side}_non_fbs"] == 1
        g.loc[small, f"{side}_rest"] = 7.0
        g.loc[small, f"{side}_gp"] = g.loc[small, "week_order"].clip(upper=13) - 1
    g = extra.fill_missing(g)
    g["home_is_power"] = g["home_is_power"].fillna(0)
    g["away_is_power"] = g["away_is_power"].fillna(0)

    g = g.copy()                                   # de-fragment after many merges
    # 4) Differences and context
    for f in TEAM_FEATURES + extra.EXTRA_TEAM_FEATURES:
        g[f"d_{f}"] = g[f"home_{f}"] - g[f"away_{f}"]
    g["hf"] = g["home_field"]
    g["postseason"] = g["season_type"].eq("postseason").astype(int)
    g["playoff"] = g["cfp_round"].notna().astype(int)
    g["conference_game"] = g["conference_game"].astype(int)
    g["travel_km"] = np.where(g["home_field"] == 1,
                              _haversine_km(g["home_lat"], g["home_lon"],
                                            g["away_lat"], g["away_lon"]), np.nan)
    g["elo_logit"] = np.log(g["elo_prob"] / (1 - g["elo_prob"]))
    g = extra.add_fade(g)
    g["cfbd_elo_diff"] = g["cfbd_home_elo"] - g["cfbd_away_elo"]
    market = pd.read_parquet(INTERIM / "market.parquet")[["game_id", "home_spread"]]
    g = g.merge(market, on="game_id", how="left")

    # 5) Keep only what a model may see + answers + benchmarks + ids
    ids = ["game_id", "season", "week", "week_order", "start_date", "season_type", "cfp_round",
           "home_id", "away_id", "home_team", "away_team", "home_conference", "away_conference",
           "neutral_site", "fbs_both", "completed", "notes", "home_division", "away_division",
           "fbs_involved"]
    # Team "state" (level values, not differences) for building hypothetical matchups later
    state_cols = TEAM_FEATURES + extra.EXTRA_TEAM_FEATURES + ["lat", "lon"]
    states = []
    for side in ("home", "away"):
        st_ = g[["game_id", "season", "week_order", "start_date", f"{side}_id"]
                + [f"{side}_{c}" for c in state_cols]].copy()
        st_.columns = ["game_id", "season", "week_order", "start_date", "team_id"] + state_cols
        st_["side"] = side
        states.append(st_)
    meta["_states"] = pd.concat(states, ignore_index=True)
    meta["_elo_hfa"] = elo_params.hfa

    cols = list(dict.fromkeys(ids + MODEL_FEATURES + extra.EXTRA_FEATURES + extra.FADED_FEATURES + TARGETS + BENCHMARKS))  # no duplicates
    table = g[cols].copy()
    table = table[table["season"] >= FEATURE_START - 1].reset_index(drop=True)
    leaked = set(MODEL_FEATURES) & LEAKY_COLUMNS
    if leaked:
        raise ValueError(f"Leakage! These after-the-game columns are model inputs: {leaked}")
    meta["n_rows"] = int(len(table))
    meta["features"] = MODEL_FEATURES
    return table, meta


def flip(df: pd.DataFrame) -> pd.DataFrame:
    """Same games seen from the other side (home <-> away). Used to double the
    training data and make P(A beats B) = 1 - P(B beats A) hold exactly."""
    f = df.copy()
    for c in df.columns:
        if c.startswith("d_") or c in ("hf", "pred_pts", "pred_sr", "pred_ypp", "pred_pw",
                                       "elo_logit", "margin", "home_spread", "cfbd_elo_diff"):
            f[c] = -df[c]
    for a, b in (("home_talent", "away_talent"), ("home_is_power", "away_is_power")):
        f[a], f[b] = df[b], df[a]
    if "home_win" in df:
        f["home_win"] = 1.0 - df["home_win"]
    if "elo_prob" in df:
        f["elo_prob"] = 1.0 - df["elo_prob"]
    # travel distance belongs to the away team: unknown from the flipped side
    f["travel_km"] = df["travel_km"]
    return f


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table, meta = build()
    table.to_parquet(PROCESSED / "features.parquet", index=False)
    meta.pop("_states").to_parquet(PROCESSED / "team_states.parquet", index=False)
    meta.pop("_coefs").to_parquet(PROCESSED / "rating_coefs.parquet", index=False)
    (PROCESSED / "features_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    done = table[table["completed"]]
    print(f"rows={len(table)} completed={len(done)} features={len(MODEL_FEATURES)}")
    print("elo params:", meta["elo_params"])
    print(done[MODEL_FEATURES].isna().mean().sort_values(ascending=False).head(6).round(3).to_dict())


if __name__ == "__main__":
    main()
