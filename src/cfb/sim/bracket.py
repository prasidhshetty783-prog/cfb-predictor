"""College Football Playoff bracket simulator + historical champion backtest.

Plain-language version
----------------------
Picture the playoff bracket as a tree. We play the whole tournament 20,000 times
in the computer. In each game, a weighted coin decides the winner, weighted by the
model's win chance for that exact matchup. A team's title odds = the share of the
20,000 tournaments it won.

Backtest: for every playoff season 2014-2025 we rebuild the real bracket and use the
model that season would have had (trained only on EARLIER seasons). Then we check
how much probability it gave to the team that actually won.

Run:  python -m cfb.sim.bracket
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.config import FEATURE_START, PROCESSED, REPORTS, TEST_SEASONS
from cfb.eval.backtest import load_table
from cfb.models.predictor import Predictor
from cfb.sim.matchup import load_states, matchup_rows, slot_coefs

N_SIMS = 20000
ROUND_ORDER = {"first_round": 1, "quarterfinal": 2, "semifinal": 3, "final": 4}


@dataclass
class Node:
    round: str
    slots: list[tuple[str, int]]        # ("team", team_id) or ("winner", node_index)
    home_id: int | None                 # team with home field, None if neutral
    actual_winner: int | None


def build_bracket(games: pd.DataFrame, season: int) -> list[Node]:
    """Rebuild the bracket tree from the real playoff games of a season."""
    cfp = games[(games["season"] == season) & games["cfp_round"].notna()].copy()
    cfp["r"] = cfp["cfp_round"].map(ROUND_ORDER)
    cfp = cfp.sort_values(["r", "start_date"])
    nodes: list[Node] = []
    won_at: dict[int, int] = {}
    for _, g in cfp.iterrows():
        slots = []
        for t in (int(g["home_id"]), int(g["away_id"])):
            slots.append(("winner", won_at.pop(t)) if t in won_at else ("team", t))
        winner = None
        if g["completed"] and not math.isnan(g["margin"]):
            winner = int(g["home_id"]) if g["margin"] > 0 else int(g["away_id"])
            won_at[winner] = len(nodes)
        home = None if bool(g["neutral_site"]) else int(g["home_id"])
        nodes.append(Node(g["cfp_round"], slots, home, winner))
    return nodes


def entrants(nodes: list[Node]) -> list[int]:
    return [v for n in nodes for k, v in n.slots if k == "team"]


def simulate(nodes: list[Node], prob: dict[tuple[int, int], float], n: int = N_SIMS,
             seed: int = 0) -> dict[int, float]:
    """prob[(a, b)] = P(a beats b) when a is listed first (a has home field if any)."""
    rng = np.random.default_rng(seed)
    teams = entrants(nodes)
    ix = {t: i for i, t in enumerate(teams)}
    P = np.full((len(teams), len(teams)), 0.5)
    for (a, b), p in prob.items():
        P[ix[a], ix[b]] = p
    winners = []
    for node in nodes:
        sides = []
        for kind, v in node.slots:
            sides.append(np.full(n, ix[v]) if kind == "team" else winners[v])
        a, b = sides
        # the team with home field (first-round campus games) is always the "a" side
        if node.home_id is not None and node.slots[1] == ("team", node.home_id):
            a, b = b, a
        win_a = rng.random(n) < P[a, b]
        winners.append(np.where(win_a, a, b))
    champ = winners[-1]
    counts = np.bincount(champ, minlength=len(teams)) / n
    return {t: float(counts[ix[t]]) for t in teams}


def pairwise_probs(nodes: list[Node], predictor: Predictor, season: int, states, coefs,
                   elo_hfa: float, games: pd.DataFrame) -> tuple[dict, dict]:
    """Model and Elo P(a beats b) for every pair of teams in the bracket (pre-playoff state)."""
    teams = entrants(nodes)
    st = states[(states["season"] == season) & (states["week_order"] == 20)]
    st = st[st["team_id"].isin(teams)].drop_duplicates("team_id").set_index("team_id")
    hfa = slot_coefs(coefs, season, 20)
    home_of = {}
    for n in nodes:
        if n.home_id is not None:
            other = [v for k, v in n.slots if k == "team" and v != n.home_id]
            if other:
                home_of[(n.home_id, other[0])] = True
    pairs = [(a, b) for a in teams for b in teams if a != b]
    hf = np.array([1.0 if home_of.get((a, b)) else 0.0 for a, b in pairs])
    ctx = {"conference_game": 0, "postseason": 1, "playoff": 1, "week_order": 20}
    rows = matchup_rows(st.loc[[a for a, _ in pairs]], st.loc[[b for _, b in pairs]], hf, ctx,
                        hfa, elo_hfa)
    p_model = predictor.predict(rows)
    p_elo = rows["elo_prob"].to_numpy()
    return dict(zip(pairs, p_model)), dict(zip(pairs, p_elo))


def backtest(model: str, params: dict) -> pd.DataFrame:
    table = load_table()
    games = pd.read_parquet(PROCESSED / "features.parquet")
    states, coefs, elo_hfa = load_states()
    rows = []
    for season in TEST_SEASONS:
        nodes = build_bracket(games, season)
        if not nodes:
            continue
        pred = Predictor(model, params).fit(table[table["season"].between(FEATURE_START, season - 1)])
        pm, pe = pairwise_probs(nodes, pred, season, states, coefs, elo_hfa, games)
        model_odds = simulate(nodes, pm, seed=season)
        elo_odds = simulate(nodes, pe, seed=season)
        champ = nodes[-1].actual_winner
        teams = entrants(nodes)
        names = dict(zip(games["home_id"], games["home_team"])) | dict(zip(games["away_id"], games["away_team"]))
        ranked = sorted(model_odds, key=model_odds.get, reverse=True)
        rows.append({
            "season": season, "field_size": len(teams), "champion": names.get(champ),
            "p_champion_model": model_odds[champ], "p_champion_elo": elo_odds[champ],
            "p_champion_uniform": 1 / len(teams),
            "champion_rank_model": ranked.index(champ) + 1,
            "model_favorite": names.get(ranked[0]), "p_favorite": model_odds[ranked[0]],
            "odds": json.dumps({names.get(t): round(model_odds[t], 4) for t in ranked}),
        })
        print(f"{season}: champion {names.get(champ):15s} model {model_odds[champ]:.1%} "
              f"(rank {ranked.index(champ) + 1}/{len(teams)}), elo {elo_odds[champ]:.1%}, "
              f"favorite {names.get(ranked[0])} {model_odds[ranked[0]]:.1%}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    card = json.loads((PROCESSED.parent.parent / "models" / "model_card.json").read_text())
    res = backtest(card["algorithm"], card["params"])
    out = REPORTS / "championship"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "bracket_backtest.csv", index=False)
    summary = {
        "seasons": int(len(res)),
        "avg_p_champion_model": float(res["p_champion_model"].mean()),
        "avg_p_champion_elo": float(res["p_champion_elo"].mean()),
        "avg_p_champion_uniform": float(res["p_champion_uniform"].mean()),
        "log_score_model": float(-np.log(res["p_champion_model"]).mean()),
        "log_score_elo": float(-np.log(res["p_champion_elo"]).mean()),
        "log_score_uniform": float(-np.log(res["p_champion_uniform"]).mean()),
        "favorite_won": int((res["model_favorite"] == res["champion"]).sum()),
        "champion_in_model_top2": int((res["champion_rank_model"] <= 2).sum()),
    }
    (out / "bracket_backtest_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
