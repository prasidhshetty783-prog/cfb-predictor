"""Committee emulator: learn how the CFP selection committee ranks teams (2014-2025).

Plain-language version
----------------------
The playoff field is picked by 13 people in a room, not by a formula. To simulate
2026 we need a stand-in for them. We study their final rankings from 2014 to 2025
and learn what they reward: wins, losses, how strong a team really is (our rating),
schedule strength, winning the conference, and being in a power conference.

The method compares teams in pairs ("committee put A above B") and learns weights,
like an Elo for resumes. We test it honestly: for each season we train on the OTHER
seasons, then check how many of that season's real top-4 / top-12 it picks.

Run:  python -m cfb.sim.committee
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfb.config import INTERIM, PROCESSED, REPORTS
from cfb.features.build import POWER

RESUME_FEATURES = ["wins", "losses", "rating", "sos", "conf_champ", "is_power"]

# Seasons where the data can't reveal the title game (2020 COVID scheduling put several
# Pac-12 games on championship weekend). Verified against the real result.
CHAMPION_OVERRIDES = {(2020, "Pac-12"): "Oregon"}


def conference_champions(games: pd.DataFrame, season: int) -> dict[str, list[int]]:
    """Champion(s) of each conference: title-game winner if there was one, else best record.

    Title games are found by their label ("SEC Championship") when the data has one,
    otherwise by the calendar: a conference's single, final game about a week after
    all its other conference games. Army-Navy never counts in conference standings.
    """
    s = games[(games["season"] == season) & games["completed"] & games["fbs_both"]
              & (games["home_conference"] == games["away_conference"])]
    army_navy = s["home_team"].isin(["Army", "Navy"]) & s["away_team"].isin(["Army", "Navy"])
    notes = s["notes"].fillna("").str.lower() if "notes" in s else pd.Series("", index=s.index)
    labeled = (notes.str.contains("championship")
               & ~notes.str.contains("national|playoff|bowl|fcs"))
    regular = s["season_type"].eq("regular")       # bowls/playoff never decide a conference
    g = s[((s["conference_game"].astype(bool) & regular) | (labeled & regular)) & ~army_navy]
    champs: dict[str, list[int]] = {}
    for conf, c in g.groupby("home_conference"):
        c = c.sort_values("start_date")
        lab = c[labeled.reindex(c.index, fill_value=False)]
        if len(lab) == 1:
            r = lab.iloc[0]
            champs[conf] = [int(r["home_id"] if r["margin"] > 0 else r["away_id"])]
            continue
        last_day = c["start_date"].max()
        final = c[(last_day - c["start_date"]) <= pd.Timedelta(days=1)]
        before = c[(last_day - c["start_date"]) > pd.Timedelta(days=1)]
        is_ccg = (len(final) == 1 and len(before) > 0
                  and (last_day - before["start_date"].max()) >= pd.Timedelta(days=5))
        if is_ccg:
            r = final.iloc[0]
            champs[conf] = [int(r["home_id"] if r["margin"] > 0 else r["away_id"])]
            continue
        c = c[~labeled.reindex(c.index, fill_value=False)]
        rec = pd.concat([
            pd.DataFrame({"t": c["home_id"], "w": (c["margin"] > 0).astype(int)}),
            pd.DataFrame({"t": c["away_id"], "w": (c["margin"] < 0).astype(int)})])
        pct = rec.groupby("t")["w"].mean()
        champs[conf] = [int(t) for t in pct[pct == pct.max()].index]
    for (yr, conf), team in CHAMPION_OVERRIDES.items():
        if yr == season:
            ids = s.loc[s["home_team"] == team, "home_id"]
            ids = ids if len(ids) else s.loc[s["away_team"] == team, "away_id"]
            if len(ids):
                champs[conf] = [int(ids.iloc[0])]
    return champs


def absolute_rating(state_rows: pd.DataFrame, season: int) -> pd.Series:
    """Power rating on one scale for everyone. Small schools' ratings are measured against
    other small schools, so we subtract the fitted FBS-vs-small-school gap (about 20+ points)."""
    coefs = pd.read_parquet(PROCESSED / "rating_coefs.parquet")
    c = coefs[(coefs["season"] == season) & (coefs["target"] == "pts")]
    gap = float(c.loc[c["week_order"].idxmax(), "fcs"]) if len(c) else 0.0
    r = state_rows.set_index("team_id")
    return r["r_pts"] - gap * (1 - r["is_fbs"].fillna(0))


def season_resumes(season: int, games: pd.DataFrame, states: pd.DataFrame,
                   through_week: int = 19) -> pd.DataFrame:
    """One row per FBS team: record and strength before the bowls."""
    g = games[(games["season"] == season) & games["completed"] & games["fbs_involved"]
              & (games["week_order"] <= through_week)]
    st = states[(states["season"] == season) & (states["week_order"] <= 20)]
    st = st.sort_values(["week_order", "start_date"]).drop_duplicates("team_id", keep="last")
    rating = absolute_rating(st, season)
    long = pd.concat([
        pd.DataFrame({"team_id": g["home_id"], "opp": g["away_id"], "win": g["margin"] > 0,
                      "conf": g["home_conference"], "div": g["home_division"]}),
        pd.DataFrame({"team_id": g["away_id"], "opp": g["home_id"], "win": g["margin"] < 0,
                      "conf": g["away_conference"], "div": g["away_division"]})])
    long = long[long["div"] == "fbs"]
    long["opp_rating"] = long["opp"].map(rating).fillna(rating.min() if len(rating) else 0)
    res = long.groupby("team_id").agg(wins=("win", "sum"), games=("win", "size"),
                                      sos=("opp_rating", "mean"), conference=("conf", "last"))
    res["losses"] = res["games"] - res["wins"]
    res["rating"] = rating.reindex(res.index).fillna(rating.min() if len(rating) else 0)
    champs = conference_champions(games, season)
    champ_ids = {t for ts in champs.values() for t in ts}
    res["conf_champ"] = res.index.isin(champ_ids).astype(int)
    res["is_power"] = res["conference"].isin(POWER).astype(int)
    res.loc[(res["conference"] == "Pac-12") & (season >= 2024), "is_power"] = 0
    res["season"] = season
    return res.reset_index()


def final_cfp_ranking(season: int) -> pd.Series:
    p = pd.read_parquet(INTERIM / "polls.parquet")
    c = p[(p["poll"] == "cfp") & (p["season"] == season) & (p["season_type"] == 2)]
    c = c[c["week_order"] == c["week_order"].max()]
    return c.set_index("team_id")["rank"]


def _pairs(res: pd.DataFrame, ranks: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Training pairs: (features of A - features of B, 1 if committee ranked A above B)."""
    r = res.set_index("team_id")
    r["rank"] = ranks.reindex(r.index).fillna(26.0)      # unranked = 26
    pool = r[(r["rank"] <= 25) | (r["wins"] >= r["games"] - 3)]
    X, y = [], []
    idx = pool.index.to_numpy()
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = pool.loc[idx[i]], pool.loc[idx[j]]
            if a["rank"] == b["rank"]:
                continue
            d = (a[RESUME_FEATURES] - b[RESUME_FEATURES]).to_numpy(dtype=float)
            X += [d, -d]
            better = float(a["rank"] < b["rank"])
            y += [better, 1 - better]
    return np.array(X), np.array(y)


class Committee:
    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
        self.mu = None
        self.sd = None

    def fit(self, resumes: list[pd.DataFrame], rankings: list[pd.Series]) -> "Committee":
        allx = pd.concat(resumes)[RESUME_FEATURES]
        self.mu, self.sd = allx.mean(), allx.std().replace(0, 1)
        X, y = [], []
        for res, rk in zip(resumes, rankings):
            z = res.copy()
            z[RESUME_FEATURES] = (z[RESUME_FEATURES] - self.mu) / self.sd
            a, b = _pairs(z, rk)
            X.append(a)
            y.append(b)
        self.lr.fit(np.vstack(X), np.concatenate(y))
        return self

    def score(self, res: pd.DataFrame) -> np.ndarray:
        z = (res[RESUME_FEATURES] - self.mu) / self.sd
        return z.to_numpy(dtype=float) @ self.lr.coef_.ravel()

    def weights(self) -> dict[str, float]:
        return dict(zip(RESUME_FEATURES, self.lr.coef_.ravel().round(3)))


def validate(seasons: list[int]) -> tuple[pd.DataFrame, Committee]:
    games = pd.read_parquet(PROCESSED / "features.parquet")
    states = pd.read_parquet(PROCESSED / "team_states.parquet")
    res = {s: season_resumes(s, games, states) for s in seasons}
    rk = {s: final_cfp_ranking(s) for s in seasons}
    rows = []
    for s in seasons:
        others = [o for o in seasons if o != s]
        com = Committee().fit([res[o] for o in others], [rk[o] for o in others])
        r = res[s].assign(score=com.score(res[s])).sort_values("score", ascending=False)
        actual = rk[s].sort_values()
        pred4, act4 = set(r["team_id"].head(4)), set(actual.index[:4])
        pred12, act12 = set(r["team_id"].head(12)), set(actual.index[:12])
        top25 = actual.index[:25]
        pr = r.reset_index(drop=True).reset_index().set_index("team_id")["index"] + 1
        rho = pd.Series(pr.reindex(top25).to_numpy()).corr(
            pd.Series(actual.reindex(top25).to_numpy()), method="spearman")
        rows.append({"season": s, "top4_overlap": len(pred4 & act4),
                     "top12_overlap": len(pred12 & act12), "spearman_top25": rho})
    full = Committee().fit([res[s] for s in seasons], [rk[s] for s in seasons])
    return pd.DataFrame(rows), full


def main() -> None:
    seasons = list(range(2014, 2026))
    table, com = validate(seasons)
    out = REPORTS / "championship"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "committee_validation.csv", index=False)
    summary = {"weights_standardized": com.weights(),
               "avg_top4_overlap": float(table["top4_overlap"].mean()),
               "avg_top12_overlap": float(table["top12_overlap"].mean()),
               "avg_spearman_top25": float(table["spearman_top25"].mean())}
    (out / "committee_summary.json").write_text(json.dumps(summary, indent=1))
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
