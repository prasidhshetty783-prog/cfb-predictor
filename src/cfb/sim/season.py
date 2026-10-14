"""Simulate the rest of the 2026 season 10,000 times: conference titles, playoff field, champion.

Plain-language version
----------------------
1. **Rest of the regular season**: every remaining game is played in the computer
   using the model's predicted point margin plus random game-day luck.
2. **"Are they really that good?"** In each simulated season every team also gets a
   secret boost or penalty (about 3 points, the typical rating change from week 7 to the
   end of a season). So a team that is better than we think tends to win MANY of its games.
3. **Conference title games**: the top two in each conference's standings meet
   (Sun Belt: East winner vs West winner).
4. **Selection Day**: our committee emulator (trained on 2014-2025 CFP rankings) ranks
   teams. 2026 rules: the ACC, Big Ten, Big 12 and SEC champions get in, plus the
   highest-ranked team from the other six conferences, plus Notre Dame if ranked in the
   top 12. The rest of the 12 spots go to the highest-ranked teams. Seeds follow the ranking.
5. **The playoff**: first round at the higher seed's stadium, then neutral sites.

Title odds = share of the 10,000 seasons each team won the national title.

Run:  python -m cfb.sim.season
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from cfb.config import CURRENT_SEASON, INTERIM, MODELS_DIR, PROCESSED, REPORTS
from cfb.sim.committee import RESUME_FEATURES, Committee, absolute_rating, final_cfp_ranking, \
    season_resumes
from cfb.sim.matchup import load_states, latest_state, matchup_rows, slot_coefs

N_SIMS = 10000
SEED = 2026
P4 = ["ACC", "Big Ten", "Big 12", "SEC"]
G6 = ["American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt"]
CCG_NEUTRAL = {"ACC", "Big Ten", "Big 12", "SEC", "Mid-American"}
TRANSITION = {"North Dakota State", "Sacramento State"}     # first FBS year: no title/CFP
PAC12_FLEX = [("Utah State", "Boise State"), ("Colorado State", "Texas State"),
              ("Fresno State", "San Diego State"), ("Washington State", "Oregon State")]
POWER_2026 = {"ACC", "Big Ten", "Big 12", "SEC"}


def _rating_uncertainty(states: pd.DataFrame) -> float:
    """How much a team's rating typically moves between week 7 and the end of a season."""
    a = (states[states["week_order"] == 7].drop_duplicates(["season", "team_id"])
         .set_index(["season", "team_id"])["r_pts"])
    b = (states[states["week_order"] <= 20].sort_values("week_order")
         .drop_duplicates(["season", "team_id"], keep="last").set_index(["season", "team_id"])["r_pts"])
    d = (b - a).dropna()
    d = d[d.index.get_level_values(0).isin(range(2014, CURRENT_SEASON))]
    return float(d.std())


def run(n_sims: int = N_SIMS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    table = pd.read_parquet(PROCESSED / "features.parquet")
    s = table[table["season"] == CURRENT_SEASON].copy()
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    states, coefs, elo_hfa = load_states()
    tau = _rating_uncertainty(states)
    sigma_game = float(np.sqrt(max(model.sigma ** 2 - 2 * tau ** 2, 1.0)))

    ti = pd.read_parquet(INTERIM / "team_info.parquet")
    ti = ti[ti["season"] == CURRENT_SEASON]
    raw = pd.read_parquet(sorted((PROCESSED.parent / "raw" / "team_info").glob(
        f"*_{CURRENT_SEASON}.parquet"))[0])
    division = dict(zip(raw["team_id"].astype(int), raw["division"]))
    fbs = ti[ti["classification"] == "fbs"].set_index("team_id")
    name = dict(zip(ti["team_id"], ti["school"]))
    conf = fbs["conference"].to_dict()
    by_name = {v: k for k, v in name.items()}

    # ---------------- team index (FBS + every small-school opponent on the schedule)
    ids = sorted(set(s["home_id"]) | set(s["away_id"]) | set(fbs.index))
    ix = {t: i for i, t in enumerate(ids)}
    T = len(ids)
    fbs_mask = np.array([t in fbs.index for t in ids])
    eligible = np.array([t in fbs.index and name.get(t) not in TRANSITION for t in ids])

    now = latest_state(states, CURRENT_SEASON, 20)
    cur_rating = absolute_rating(now.reset_index(), CURRENT_SEASON).reindex(ids).fillna(-25.0).to_numpy()

    # ---------------- schedule: completed results and remaining games
    done = s[s["completed"]]
    rem = s[~s["completed"]].copy()
    rem["m"] = model.predict_margin(rem)
    hfa = slot_coefs(coefs, CURRENT_SEASON, 13)
    flex = []
    for home, away in PAC12_FLEX:                       # projected Pac-12 flex games (week 13)
        h, a = by_name.get(home), by_name.get(away)
        if h in now.index and a in now.index:
            row = matchup_rows(now.loc[[h]], now.loc[[a]], np.array([1.0]),
                               {"conference_game": 0, "postseason": 0, "playoff": 0,
                                "week_order": 13}, hfa, elo_hfa)
            flex.append({"home_id": h, "away_id": a, "m": float(model.predict_margin(row)[0]),
                         "counts_in_conference": False})
    rem["counts_in_conference"] = True
    games_rem = pd.concat([rem[["home_id", "away_id", "m", "home_conference", "away_conference",
                                "conference_game", "home_team", "away_team"]].assign(
        counts_in_conference=True), pd.DataFrame(flex)], ignore_index=True)

    def conf_game(df: pd.DataFrame) -> np.ndarray:
        same = df["home_id"].map(conf).eq(df["away_id"].map(conf)) & df["home_id"].map(conf).notna()
        army_navy = df["home_id"].map(name).isin(["Army", "Navy"]) & df["away_id"].map(name).isin(["Army", "Navy"])
        flag = df.get("counts_in_conference", pd.Series(True, index=df.index)).fillna(True).astype(bool)
        return (same & ~army_navy & flag).to_numpy()

    W = np.zeros((n_sims, T))
    L = np.zeros((n_sims, T))
    CW = np.zeros((n_sims, T))
    CL = np.zeros((n_sims, T))
    hi = done["home_id"].map(ix).to_numpy()
    ai = done["away_id"].map(ix).to_numpy()
    hw = (done["margin"] > 0).to_numpy()
    cg = conf_game(done)
    for h, a, w, c in zip(hi, ai, hw, cg):
        W[:, h if w else a] += 1
        L[:, a if w else h] += 1
        if c:
            CW[:, h if w else a] += 1
            CL[:, a if w else h] += 1

    delta = rng.normal(0.0, tau, (n_sims, T))           # "are they really that good?"
    strength = cur_rating[None, :] + delta
    rh = games_rem["home_id"].map(ix).to_numpy()
    ra = games_rem["away_id"].map(ix).to_numpy()
    rc = conf_game(games_rem)
    for k in range(len(games_rem)):
        h, a = rh[k], ra[k]
        margin = games_rem["m"].iat[k] + delta[:, h] - delta[:, a] + rng.normal(0, sigma_game, n_sims)
        home_won = margin > 0
        W[:, h] += home_won
        L[:, h] += ~home_won
        W[:, a] += ~home_won
        L[:, a] += home_won
        if rc[k]:
            CW[:, h] += home_won
            CL[:, h] += ~home_won
            CW[:, a] += ~home_won
            CL[:, a] += home_won

    # ---------------- pairwise margins for title games and playoff games (FBS teams)
    fb = [t for t in ids if t in fbs.index and t in now.index]
    def pair_matrix(ctx: dict, hf: float) -> np.ndarray:
        A = [a for a in fb for b in fb if a != b]
        B = [b for a in fb for b in fb if a != b]
        rows = matchup_rows(now.loc[A], now.loc[B], np.full(len(A), hf), ctx,
                            slot_coefs(coefs, CURRENT_SEASON, 20), elo_hfa)
        m = model.predict_margin(rows)
        M = np.zeros((T, T))
        M[[ix[a] for a in A], [ix[b] for b in B]] = m
        return M
    M_ccg_n = pair_matrix({"conference_game": 1, "postseason": 0, "playoff": 0, "week_order": 15}, 0.0)
    M_ccg_h = pair_matrix({"conference_game": 1, "postseason": 0, "playoff": 0, "week_order": 15}, 1.0)
    M_cfp_n = pair_matrix({"conference_game": 0, "postseason": 1, "playoff": 1, "week_order": 20}, 0.0)
    M_cfp_h = pair_matrix({"conference_game": 0, "postseason": 1, "playoff": 1, "week_order": 20}, 1.0)

    def play(M: np.ndarray, a: np.ndarray, b: np.ndarray, sims: np.ndarray) -> np.ndarray:
        """Vectorized game: a vs b in each listed simulation; returns the winners."""
        m = M[a, b] + delta[sims, a] - delta[sims, b] + rng.normal(0, sigma_game, len(sims))
        return np.where(m > 0, a, b)

    # ---------------- conference championship games
    sims = np.arange(n_sims)
    champ = {}                                          # conference -> winners per sim
    for c in P4 + G6:
        members = [ix[t] for t in fbs.index if conf.get(t) == c and t in ix and eligible[ix[t]]]
        if len(members) < 2:
            continue
        mem = np.array(members)
        cpct = CW[:, mem] / np.maximum(CW[:, mem] + CL[:, mem], 1)
        opct = W[:, mem] / np.maximum(W[:, mem] + L[:, mem], 1)
        key = cpct * 1e6 + opct * 1e3 + strength[:, mem] * 1e-3
        if c == "Sun Belt":
            east = np.array([division.get(ids[m]) == "East" for m in mem])
            k_e = np.where(east[None, :], key, -np.inf)
            k_w = np.where(~east[None, :], key, -np.inf)
            first, second = mem[k_e.argmax(1)], mem[k_w.argmax(1)]
            swap = key[sims, k_w.argmax(1)] > key[sims, k_e.argmax(1)]
            first, second = np.where(swap, second, first), np.where(swap, first, second)
        else:
            order = np.argsort(-key, axis=1)
            first, second = mem[order[:, 0]], mem[order[:, 1]]
        M = M_ccg_n if c in CCG_NEUTRAL else M_ccg_h     # otherwise the #1 seed hosts
        win = play(M, first, second, sims)
        lose = np.where(win == first, second, first)
        W[sims, win] += 1
        L[sims, lose] += 1
        champ[c] = win

    # ---------------- committee emulator (trained on every CFP season 2014-2025)
    games_hist = pd.read_parquet(PROCESSED / "features.parquet")
    hist = list(range(2014, CURRENT_SEASON))
    com = Committee().fit([season_resumes(y, games_hist, states) for y in hist],
                          [final_cfp_ranking(y) for y in hist])
    # strength of schedule = average (absolute) strength of FBS-season opponents
    opp = np.zeros((T, T))
    for h, a in zip(np.concatenate([hi, rh]), np.concatenate([ai, ra])):
        opp[h, a] += 1
        opp[a, h] += 1
    n_opp = opp.sum(1)
    sos = (strength @ opp.T) / np.maximum(n_opp, 1)
    is_power = np.array([conf.get(t) in POWER_2026 for t in ids], dtype=float)
    conf_champ = np.zeros((n_sims, T))
    for c, w in champ.items():
        conf_champ[sims, w] = 1
    feats = {"wins": W, "losses": L, "rating": strength, "sos": sos,
             "conf_champ": conf_champ, "is_power": np.broadcast_to(is_power, (n_sims, T))}
    wts = com.lr.coef_.ravel()
    score = sum(wts[i] * (feats[f] - com.mu[f]) / com.sd[f] for i, f in enumerate(RESUME_FEATURES))
    score = np.where(eligible[None, :], score, -np.inf)

    # ---------------- selection + 12-team bracket
    nd = ix.get(by_name.get("Notre Dame"))
    g6_mask = np.array([conf.get(t) in G6 for t in ids])
    made = np.zeros((n_sims, T))
    bye = np.zeros((n_sims, T))
    semis = np.zeros((n_sims, T))
    final = np.zeros((n_sims, T))
    title = np.zeros((n_sims, T))
    fields = []
    for i in range(n_sims):
        rank = np.argsort(-score[i])
        pos = {t: r for r, t in enumerate(rank)}
        auto = [champ[c][i] for c in P4 if c in champ]
        g6_best = rank[np.flatnonzero(g6_mask[rank])[0]]
        auto.append(g6_best)
        if nd is not None and pos[nd] < 12:
            auto.append(nd)
        auto = list(dict.fromkeys(auto))
        field = list(auto)
        for t in rank:
            if len(field) >= 12:
                break
            if t not in field:
                field.append(t)
        inside = sorted([t for t in field if pos[t] < 12 or t not in auto], key=pos.get)
        outside = sorted([t for t in field if t not in inside], key=pos.get)
        seeds = (inside + outside)[:12]                     # auto bids outside the top 12 go last
        if len(seeds) < 12:
            continue
        sid = np.full(1, i)
        made[i, seeds] = 1
        bye[i, seeds[:4]] = 1
        r1 = [play(M_cfp_h, np.array([seeds[a]]), np.array([seeds[b]]), sid)[0]
              for a, b in ((4, 11), (5, 10), (6, 9), (7, 8))]          # 5v12, 6v11, 7v10, 8v9
        qf_pairs = [(seeds[0], r1[3]), (seeds[1], r1[2]), (seeds[2], r1[1]), (seeds[3], r1[0])]
        qf = [play(M_cfp_n, np.array([a]), np.array([b]), sid)[0] for a, b in qf_pairs]
        semis[i, qf] = 1
        sf = [play(M_cfp_n, np.array([qf[0]]), np.array([qf[3]]), sid)[0],
              play(M_cfp_n, np.array([qf[1]]), np.array([qf[2]]), sid)[0]]
        final[i, sf] = 1
        champ_i = play(M_cfp_n, np.array([sf[0]]), np.array([sf[1]]), sid)[0]
        title[i, champ_i] = 1
        if i < 2000:
            fields.append(tuple(sorted(seeds)))

    # ---------------- summary table
    rec_w = W.mean(0)
    cur_w = np.zeros(T)
    cur_l = np.zeros(T)
    for h, a, w in zip(hi, ai, hw):
        cur_w[h if w else a] += 1
        cur_l[a if w else h] += 1
    conf_title = np.zeros(T)
    for c, w in champ.items():
        conf_title += np.bincount(w, minlength=T) / n_sims
    out = pd.DataFrame({
        "team": [name.get(t, str(t)) for t in ids], "team_id": ids,
        "conference": [conf.get(t, "") for t in ids],
        "record_now": [f"{int(a)}-{int(b)}" for a, b in zip(cur_w, cur_l)],
        "power_rating": cur_rating, "exp_wins": rec_w,
        "p_conf_title": conf_title, "p_playoff": made.mean(0), "p_top4_bye": bye.mean(0),
        "p_semifinal": semis.mean(0), "p_final": final.mean(0), "p_title": title.mean(0),
    })[fbs_mask].sort_values("p_title", ascending=False).reset_index(drop=True)
    common = pd.Series(fields).value_counts().head(1)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT_SEASON, "n_sims": n_sims, "seed": seed,
        "games_completed": int(len(done)), "games_simulated": int(len(games_rem)),
        "data_through": str(done["start_date"].max()),
        "rating_uncertainty_tau_points": tau, "game_noise_sigma_points": sigma_game,
        "committee_weights": com.weights(),
        "assumptions": [
            "2026 CFP: 12 teams; ACC/Big Ten/Big 12/SEC champions + highest-ranked team from the "
            "other six conferences + Notre Dame if top 12; seeds follow the ranking.",
            "Conference title game = top two by conference record (ties: overall record, then "
            "strength). Sun Belt = East winner vs West winner.",
            "Pac-12 flex games use the league's projected matchups and don't count in standings.",
            "North Dakota State and Sacramento State (first FBS season) are ineligible for "
            "conference titles and the CFP.",
            "Team states are frozen at today's data; uncertainty enters through tau and game noise.",
        ],
        "most_common_field_frequency": float(common.iloc[0] / max(len(fields), 1)) if len(common) else 0,
        "most_common_field": [name.get(ids[t]) for t in common.index[0]] if len(common) else [],
    }
    return {"table": out, "meta": meta}


def main() -> None:
    res = run()
    out = REPORTS / "championship"
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / f"title_odds_{CURRENT_SEASON}.csv", index=False)
    (out / f"title_odds_{CURRENT_SEASON}_meta.json").write_text(json.dumps(res["meta"], indent=1))
    t = res["table"].head(20)
    print(t[["team", "conference", "record_now", "power_rating", "exp_wins", "p_conf_title",
             "p_playoff", "p_top4_bye", "p_final", "p_title"]].round(3).to_string(index=False))
    print(json.dumps({k: res["meta"][k] for k in ("games_completed", "games_simulated", "data_through",
                                                  "rating_uncertainty_tau_points",
                                                  "game_noise_sigma_points")}, indent=1))


if __name__ == "__main__":
    main()
