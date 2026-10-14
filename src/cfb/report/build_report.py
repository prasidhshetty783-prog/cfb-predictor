"""Build the results report: one self-contained HTML file (opens offline, no internet needed
except optional web fonts). Charts are drawn here as inline SVG / HTML with hover tooltips,
light + dark themes, and a table view under every chart.

Run:  python -m cfb.report.build_report      ->  reports/model_report.html
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cfb.config import CURRENT_SEASON, MODELS_DIR, REPORTS
from cfb.eval.backtest import OUT as BT

CH = REPORTS / "championship"
NICE = {
    "margin_ridge": "Margin ridge (winner)", "margin_lightgbm": "Margin LightGBM",
    "ensemble_top3": "Ensemble (top 3)", "logistic_lasso": "Logistic lasso",
    "logistic_regression": "Logistic regression", "neural_net_mlp": "Neural net (MLP)",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "hist_gradient_boosting": "Hist. gradient boosting",
    "random_forest": "Random forest", "knn": "k-nearest neighbors", "svm_rbf": "SVM (RBF kernel)",
    "elo": "Our Elo", "cfbd_elo": "CFBD Elo", "naive_bayes": "Naive Bayes",
    "home_field": "Home team always", "vegas_spread": "Vegas closing line",
}
WINNER = "margin_ridge"
FEATURE_LABELS = {
    "pred_pts": "Point-margin power rating gap", "pred_sr": "Success-rate rating gap",
    "pred_ypp": "Yards-per-play rating gap", "pred_pw": "Postgame win-expectancy gap",
    "elo_logit": "Elo win chance (with home field)", "d_elo": "Elo rating difference",
    "d_is_fbs": "Big school vs small school", "d_r_pts": "Point rating difference",
    "d_ret_off": "Returning offense difference", "d_is_power": "Power-conference edge",
    "d_s_margin": "Season scoring margin so far", "hf": "Home field",
    "d_talent": "Recruiting talent difference", "d_ret_def": "Returning defense difference",
    "d_prev_adj_net": "Last season's rating difference", "d_last3_margin": "Last-3-games margin",
}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ HTML "row" charts
def dot_whisker_rows(df: pd.DataFrame, lo: float, hi: float, winner: str) -> str:
    """Rows of: name | track with CI whisker + dot | value. Zero line = Vegas."""
    def x(v):
        return 100 * (v - lo) / (hi - lo)
    rows = []
    for _, r in df.iterrows():
        cls = "is-win" if r["model"] == winner else ""
        tip = (f"{NICE.get(r['model'], r['model'])}\nBehind Vegas: +{r['gap_to_vegas']:.4f} log loss"
               f"\n95% CI: +{r['gap_ci_low']:.4f} to +{r['gap_ci_high']:.4f}"
               f"\nAccuracy on these games: {pct(r['accuracy'])}")
        rows.append(f"""
      <div class="dw-row {cls}" tabindex="0" data-tip="{esc(tip)}">
        <span class="dw-name">{esc(NICE.get(r['model'], r['model']))}</span>
        <span class="dw-track"><span class="dw-ci" style="left:{x(r['gap_ci_low']):.2f}%;width:{x(r['gap_ci_high']) - x(r['gap_ci_low']):.2f}%"></span><span class="dw-dot" style="left:{x(r['gap_to_vegas']):.2f}%"></span></span>
        <span class="dw-val num">+{r['gap_to_vegas']:.3f}</span>
      </div>""")
    ticks = "".join(
        f'<span class="dw-tick{"" if (round(t * 100) % 4 == 0) else " minor"}" style="left:{x(t):.2f}%">'
        f'{"Vegas" if t == 0 else f"+{t:.2f}"}</span>'
        for t in np.arange(0, hi + 1e-9, 0.01) if t >= lo)
    return f"""<div class="dw" role="img" aria-label="Distance behind the Vegas line, per algorithm">
      <div class="dw-axis"><span></span><span class="dw-ticks">{ticks}</span><span></span></div>
      {''.join(rows)}
    </div>"""


def bar_rows(items: list[tuple[str, float, str, str]], vmax: float, highlight: set[str] | None = None) -> str:
    """Horizontal bars from a zero baseline. items = (label, value, value_text, tooltip)."""
    out = []
    for label, v, vt, tip in items:
        cls = "is-win" if highlight and label in highlight else ""
        w = max(0.0, 100 * v / vmax)
        out.append(f"""<div class="bar-row {cls}" tabindex="0" data-tip="{esc(tip)}">
        <span class="bar-name">{esc(label)}</span>
        <span class="bar-track"><span class="bar" style="width:{w:.2f}%"></span><span class="bar-val num" style="left:{w:.2f}%">{esc(vt)}</span></span>
      </div>""")
    return f'<div class="bars">{"".join(out)}</div>'


# ------------------------------------------------------------------ SVG charts
def _axis_y(ticks, y, x0, x1, fmt):
    g = []
    for t in ticks:
        yy = y(t)
        g.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                 f'<text class="tick" x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end">{esc(fmt(t))}</text>')
    return "".join(g)


def svg_lines(series, xs, y_ticks, y_fmt, title, x_fmt=str, height=320, width=960,
              y_label="", legend=True) -> str:
    """Multi-series line chart with a crosshair column hover (one tooltip, every series)."""
    m = {"l": 58, "r": 120, "t": 18, "b": 34}
    x0, x1, y0, y1 = m["l"], width - m["r"], m["t"], height - m["b"]
    lo, hi = min(y_ticks), max(y_ticks)

    def X(i):
        return x0 + (x1 - x0) * (i / (len(xs) - 1))

    def Y(v):
        return y1 - (y1 - y0) * (v - lo) / (hi - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(title)}">',
             _axis_y(y_ticks, Y, x0, x1, y_fmt),
             f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>']
    for i, xv in enumerate(xs):
        parts.append(f'<text class="tick" x="{X(i):.1f}" y="{y1 + 20}" text-anchor="middle">{esc(x_fmt(xv))}</text>')
    if y_label:
        parts.append(f'<text class="tick" x="{x0 - 8}" y="{y0 - 6}" text-anchor="end">{esc(y_label)}</text>')
    ends = []
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"]) if v is not None and not np.isnan(v)]
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        parts.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for a, b in pts:
            parts.append(f'<circle class="dot {s["cls"]}" cx="{a:.1f}" cy="{b:.1f}" r="4"/>')
        ends.append((pts[-1][1], s))
    # direct end labels with simple collision avoidance (leader lines when moved)
    ends.sort(key=lambda e: e[0])
    placed = []
    for yy, s in ends:
        ty = yy
        for p in placed:
            if abs(ty - p) < 16:
                ty = p + 16
        placed.append(ty)
        if abs(ty - yy) > 2:
            parts.append(f'<line class="leader" x1="{x1 + 4}" y1="{yy:.1f}" x2="{x1 + 10}" y2="{ty - 4:.1f}"/>')
        parts.append(f'<text class="end-label" x="{x1 + 12}" y="{ty:.1f}">{esc(s["name"])}</text>')
    # hover columns
    step = (x1 - x0) / (len(xs) - 1)
    for i, xv in enumerate(xs):
        lines = [f"{x_fmt(xv)}"] + [f"{s['name']}: {y_fmt(s['values'][i])}" for s in series
                                    if s['values'][i] is not None and not np.isnan(s['values'][i])]
        parts.append(f'<rect class="hit" x="{X(i) - step / 2:.1f}" y="{y0}" width="{step:.1f}" '
                     f'height="{y1 - y0}" tabindex="0" data-tip="{esc(chr(10).join(lines))}" data-x="{X(i):.1f}"/>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{y0}" y2="{y1}" visibility="hidden"/>')
    parts.append("</svg>")
    leg = ""
    if legend and len(series) > 1:
        leg = '<div class="legend">' + "".join(
            f'<span class="key"><span class="swatch-line {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series) + "</div>"
    return leg + '<div class="svg-wrap">' + "".join(parts) + "</div>"


def svg_reliability(series, title, size=420) -> str:
    m = 52
    x0, x1, y0, y1 = m, size - 18, 14, size - 40

    def X(v):
        return x0 + (x1 - x0) * v

    def Y(v):
        return y1 - (y1 - y0) * v
    p = [f'<svg viewBox="0 0 {size} {size}" class="chart chart-sq" role="img" aria-label="{esc(title)}">']
    for t in np.linspace(0, 1, 6):
        p.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{x0 - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>'
                 f'<text class="tick" x="{X(t):.1f}" y="{y1 + 18}" text-anchor="middle">{int(t * 100)}%</text>')
    p.append(f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>')
    p.append(f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{size - 6}" text-anchor="middle">What the model said (home team win chance)</text>')
    for s in series:
        pts = [(X(a), Y(b), a, b, n) for a, b, n in s["points"]]
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b, *_ in pts)
        p.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for a, b, ra, rb, n in pts:
            tip = f"{s['name']}\nSaid: {ra * 100:.0f}%\nHappened: {rb * 100:.0f}%\nGames: {n:,}"
            p.append(f'<g class="pt" tabindex="0" data-tip="{esc(tip)}"><circle class="hit-c" cx="{a:.1f}" cy="{b:.1f}" r="12"/>'
                     f'<circle class="dot {s["cls"]}" cx="{a:.1f}" cy="{b:.1f}" r="4"/></g>')
    p.append("</svg>")
    leg = '<div class="legend">' + "".join(
        f'<span class="key"><span class="swatch-line {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    leg += '<span class="key"><span class="swatch-line diag-key"></span>Perfect calibration</span></div>'
    return leg + '<div class="svg-wrap sq">' + "".join(p) + "</div>"


def svg_champions(rows: pd.DataFrame, title: str, height=330, width=960) -> str:
    m = {"l": 52, "r": 16, "t": 16, "b": 52}
    x0, x1, y0, y1 = m["l"], width - m["r"], m["t"], height - m["b"]
    n = len(rows)
    hi = 0.6

    def X(i):
        return x0 + (x1 - x0) * (i + 0.5) / n

    def Y(v):
        return y1 - (y1 - y0) * v / hi
    p = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(_axis_y([0, .1, .2, .3, .4, .5, .6], Y, x0, x1, lambda t: f"{int(round(t * 100))}%"))
    p.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    for i, r in enumerate(rows.itertuples()):
        cx = X(i)
        p.append(f'<text class="tick" x="{cx:.1f}" y="{y1 + 17}" text-anchor="middle">{r.season}</text>'
                 f'<text class="tick small" x="{cx:.1f}" y="{y1 + 32}" text-anchor="middle">{esc(r.champion)}</text>')
        p.append(f'<line class="uni" x1="{cx - 9:.1f}" x2="{cx + 9:.1f}" y1="{Y(r.p_champion_uniform):.1f}" y2="{Y(r.p_champion_uniform):.1f}"/>')
        p.append(f'<line class="stem" x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(r.p_champion_elo):.1f}" y2="{Y(r.p_champion_model):.1f}"/>')
        p.append(f'<circle class="dot s2" cx="{cx:.1f}" cy="{Y(r.p_champion_elo):.1f}" r="4.5"/>')
        p.append(f'<circle class="dot s1" cx="{cx:.1f}" cy="{Y(r.p_champion_model):.1f}" r="5"/>')
        tip = (f"{r.season}: {r.champion} won\nOur model gave them {pct(r.p_champion_model)} "
               f"(rank {r.champion_rank_model} of {r.field_size})\nElo gave them {pct(r.p_champion_elo)}"
               f"\nRandom guess: {pct(r.p_champion_uniform)}\nModel favorite: {r.model_favorite} ({pct(r.p_favorite)})")
        p.append(f'<rect class="hit" x="{cx - (x1 - x0) / n / 2:.1f}" y="{y0}" width="{(x1 - x0) / n:.1f}" height="{y1 - y0}" tabindex="0" data-tip="{esc(tip)}" data-x="{cx:.1f}"/>')
    p.append(f'<line class="crosshair" x1="0" x2="0" y1="{y0}" y2="{y1}" visibility="hidden"/>')
    p.append("</svg>")
    leg = ('<div class="legend"><span class="key"><span class="swatch-dot s1"></span>Our model</span>'
           '<span class="key"><span class="swatch-dot s2"></span>Elo only</span>'
           '<span class="key"><span class="swatch-tick"></span>Random guess (1 in 4, or 1 in 12)</span></div>')
    return leg + '<div class="svg-wrap">' + "".join(p) + "</div>"


HEADERS = {
    "algorithm": "Algorithm", "family": "Type", "n": "Games", "log_loss": "Log loss", "brier": "Brier",
    "accuracy": "Accuracy", "balanced_accuracy": "Bal. acc.", "recall_home": "Recall home",
    "recall_away": "Recall road", "upset_recall": "Upsets caught", "auc": "AUC", "ece": "Calib. error",
    "holm_p_vs_leader": "p vs leader", "gap_to_vegas": "Gap to Vegas", "gap_ci_low": "95% low",
    "gap_ci_high": "95% high", "model": "Model", "lo": "Bin from", "hi": "Bin to", "mean_pred": "Said",
    "observed": "Happened", "season": "Season", "our_model": "Our model", "elo": "Elo", "vegas": "Vegas",
    "feature_set": "Clues used", "n_features": "Clues", "champion": "Champion", "field_size": "Field",
    "p_champion_model": "Model gave champ", "champion_rank_model": "Champ rank", "p_champion_elo": "Elo gave champ",
    "model_favorite": "Model favorite", "p_favorite": "Favorite odds", "team": "Team", "conference": "Conference",
    "record_now": "Record", "exp_wins": "Exp. wins", "p_conf_title": "Conf. title", "p_playoff": "Playoff",
    "p_top4_bye": "Bye (top 4)", "p_semifinal": "Semifinal", "p_final": "Final", "p_title": "Title",
}


def table(df: pd.DataFrame, fmt: dict, caption: str, num_cols: set[str] | None = None,
          open_: bool = False) -> str:
    num_cols = num_cols or set(fmt)
    head = "".join(f'<th scope="col" class="{"num" if c in num_cols else ""}">{esc(HEADERS.get(c, c))}</th>'
                   for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            s = fmt[c](v) if c in fmt else v
            tds.append(f'<td class="{"num" if c in num_cols else ""}">{esc(s)}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    if open_:
        return (f'<div class="tbl-wrap"><table><caption class="sr">{esc(caption)}</caption>'
                f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return (f'<details class="tv"><summary>Table view: {esc(caption)}</summary><div class="tbl-wrap">'
            f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')


# ------------------------------------------------------------------ page
def build() -> str:
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    m = pd.read_csv(BT / "metrics_overall.csv")
    allg = m[m["subset"] == "all_fbs_games"].set_index("model")
    vs = pd.read_csv(BT / "vs_vegas.csv")
    sig = pd.read_csv(BT / "significance.csv").set_index("model")
    per = pd.read_csv(BT / "metrics_by_season.csv")
    cal = pd.read_csv(BT / "calibration.csv")
    abl = pd.read_csv(BT / "ablation.csv")
    coef = pd.read_csv(BT / "final_model_coefficients.csv")
    br = pd.read_csv(CH / "bracket_backtest.csv")
    brs = json.loads((CH / "bracket_backtest_summary.json").read_text())
    com = json.loads((CH / "committee_summary.json").read_text())
    odds = pd.read_csv(CH / f"title_odds_{CURRENT_SEASON}.csv")
    ometa = json.loads((CH / f"title_odds_{CURRENT_SEASON}_meta.json").read_text())
    up = pd.read_csv(REPORTS / f"predictions_{CURRENT_SEASON}_upcoming.csv")
    audit = json.loads((REPORTS / "data_audit.json").read_text())
    w = allg.loc[WINNER]
    vgap = float(vs.set_index("model").loc[WINNER, "gap_to_vegas"])
    n_test = int(w["n"])
    data_through = pd.Timestamp(ometa["data_through"]).tz_convert("America/New_York")

    # ---- chart 1: distance behind Vegas
    vv = vs[vs["model"] != "vegas_spread"].sort_values("gap_to_vegas")
    vv = vv[vv["model"] != "home_field"]
    chart_gap = dot_whisker_rows(vv, 0.0, float(np.ceil(vv["gap_ci_high"].max() * 100) / 100), WINNER)
    gap_tbl = vs.sort_values("log_loss").assign(
        algorithm=lambda d: d["model"].map(lambda x: NICE.get(x, x)))[[
            "algorithm", "log_loss", "gap_to_vegas", "gap_ci_low", "gap_ci_high", "accuracy"]]
    gap_table = table(gap_tbl, {"log_loss": lambda v: f"{v:.4f}", "gap_to_vegas": lambda v: f"{v:+.4f}",
                                "gap_ci_low": lambda v: f"{v:+.4f}", "gap_ci_high": lambda v: f"{v:+.4f}",
                                "accuracy": pct}, f"all {len(gap_tbl)} entries on the same {int(allg.loc['vegas_spread', 'n']):,} games")

    # ---- full scorecard table
    sc = allg.drop(index=["vegas_spread"], errors="ignore").copy()
    leader = sig["log_loss"].idxmin()
    sc["holm_p_vs_leader"] = [("leader" if mdl == leader else
                               ("n/a" if pd.isna(sig["p_holm"].get(mdl, np.nan)) else f"{sig['p_holm'][mdl]:.3f}"))
                              for mdl in sc.index]
    sc = sc.sort_values("log_loss").reset_index()
    sc["algorithm"] = sc["model"].map(lambda x: NICE.get(x, x))
    sc.loc[sc["n"] < n_test, "algorithm"] = sc["algorithm"] + " (" + sc["n"].map(lambda v: f"{int(v):,}") + " games)"
    score_tbl = table(sc[["algorithm", "log_loss", "brier", "accuracy", "balanced_accuracy",
                          "recall_home", "recall_away", "upset_recall", "auc", "ece", "holm_p_vs_leader"]],
                      {"log_loss": lambda v: f"{v:.4f}", "brier": lambda v: f"{v:.4f}",
                       "accuracy": pct, "balanced_accuracy": pct, "recall_home": pct, "recall_away": pct,
                       "upset_recall": pct, "auc": lambda v: f"{v:.3f}", "ece": lambda v: f"{100 * v:.1f} pts",
                       "holm_p_vs_leader": lambda v: v},
                      "every algorithm, all test games 2014-2025", open_=True)

    # ---- chart 2: reliability
    def rel(model, name, cls):
        c = cal[cal["model"] == model]
        c = c[c["n"] >= 40]
        return {"name": name, "cls": cls,
                "points": list(zip(c["mean_pred"], c["observed"], c["n"].astype(int)))}
    rel_series = [rel(WINNER, "Our model", "s1"), rel("elo", "Elo only", "s2"),
                  rel("vegas_spread", "Vegas", "s3")]
    chart_rel = svg_reliability(rel_series, "Calibration: predicted chance vs how often it happened")
    rel_tbl = cal[cal["model"].isin([WINNER, "elo", "vegas_spread"])].assign(
        model=lambda d: d["model"].map(lambda x: NICE.get(x, x)))[["model", "lo", "hi", "mean_pred", "observed", "n"]]
    rel_table = table(rel_tbl, {"lo": pct, "hi": pct, "mean_pred": pct, "observed": pct, "n": lambda v: f"{int(v):,}"},
                      "calibration bins", {"lo", "hi", "mean_pred", "observed", "n"})

    # ---- chart 3: per season
    seasons = sorted(per["season"].unique())
    pv = per.pivot_table(index="season", columns="model", values="log_loss")
    chart_season = svg_lines(
        [{"name": "Our model", "cls": "s1", "values": pv[WINNER].tolist()},
         {"name": "Elo only", "cls": "s2", "values": pv["elo"].tolist()},
         {"name": "Vegas", "cls": "s3", "values": pv["vegas_spread"].tolist()}],
        seasons, [0.42, 0.45, 0.48, 0.51, 0.54, 0.57], lambda v: f"{v:.2f}",
        "Log loss by season")
    per_tbl = pv[[WINNER, "elo", "vegas_spread"]].reset_index().rename(
        columns={WINNER: "our_model", "vegas_spread": "vegas"})
    season_table = table(per_tbl, {"our_model": lambda v: f"{v:.4f}", "elo": lambda v: f"{v:.4f}",
                                   "vegas": lambda v: f"{v:.4f}"}, "log loss per season")
    beat_elo = int((pv[WINNER] < pv["elo"]).sum())

    # ---- chart 4: ablation (log loss saved vs Elo only)
    base = float(abl.iloc[0]["log_loss"])
    abl_items = []
    for _, r in abl.iterrows():
        saved = base - r["log_loss"]
        label = r["feature_set"].split(" ", 1)[1].replace("+ ", "+ ")
        tip = f"{label}\nLog loss {r['log_loss']:.4f}\nAccuracy {pct(r['accuracy'])}\nFeatures: {int(r['n_features'])}"
        abl_items.append((label, saved, f"{saved:.3f} saved" if saved > 0 else "starting point", tip))
    chart_abl = bar_rows(abl_items, max(i[1] for i in abl_items) * 1.18)
    abl_table = table(abl[["feature_set", "n_features", "log_loss", "accuracy"]],
                      {"n_features": lambda v: f"{int(v)}", "log_loss": lambda v: f"{v:.4f}", "accuracy": pct},
                      "feature-group ablation", {"n_features", "log_loss", "accuracy"})

    # ---- chart 5: champions
    chart_ch = svg_champions(br, "Probability given to the eventual champion, right before each playoff")
    ch_table = table(br[["season", "champion", "field_size", "p_champion_model", "champion_rank_model",
                         "p_champion_elo", "model_favorite", "p_favorite"]],
                     {"p_champion_model": pct, "p_champion_elo": pct, "p_favorite": pct,
                      "field_size": lambda v: f"{int(v)}", "champion_rank_model": lambda v: f"{int(v)}"},
                     "champion backtest", {"field_size", "p_champion_model", "champion_rank_model",
                                           "p_champion_elo", "p_favorite"})

    # ---- chart 6: 2026 title odds
    top = odds.head(12)
    t_items = [(r.team, r.p_title, pct(r.p_title),
                f"{r.team} ({r.conference}, {r.record_now})\nTitle: {pct(r.p_title)}\nMake playoff: {pct(r.p_playoff)}"
                f"\nWin conference: {pct(r.p_conf_title)}\nTop-4 seed (bye): {pct(r.p_top4_bye)}"
                f"\nReach final: {pct(r.p_final)}\nExpected wins: {r.exp_wins:.1f}")
               for r in top.itertuples()]
    chart_odds = bar_rows(t_items, float(top["p_title"].max()) * 1.2, highlight={top.iloc[0]["team"]})
    o_tbl = odds.head(25)[["team", "conference", "record_now", "exp_wins", "p_conf_title", "p_playoff",
                           "p_top4_bye", "p_semifinal", "p_final", "p_title"]]
    odds_table = table(o_tbl, {"exp_wins": lambda v: f"{v:.1f}", "p_conf_title": pct, "p_playoff": pct,
                               "p_top4_bye": pct, "p_semifinal": pct, "p_final": pct, "p_title": pct},
                       "top 25 title odds", {"exp_wins", "p_conf_title", "p_playoff", "p_top4_bye",
                                              "p_semifinal", "p_final", "p_title"})
    mich = odds[odds["team"] == "Michigan"].iloc[0]
    mich_games = up[(up["home_team"] == "Michigan") | (up["away_team"] == "Michigan")]
    mg_rows = []
    for r in mich_games.itertuples():
        opp = r.away_team if r.home_team == "Michigan" else r.home_team
        where = "vs" if r.home_team == "Michigan" else "at"
        p_m = r.home_win_prob if r.home_team == "Michigan" else 1 - r.home_win_prob
        spread = -r.home_margin if r.home_team == "Michigan" else r.home_margin
        mg_rows.append(f'<tr><td class="num">{int(r.week)}</td><td class="wrap">{where} {esc(opp)}</td>'
                       f'<td class="num">MICH {spread:+.1f}</td><td class="num">{pct(p_m, 0)}</td>'
                       f'<td class="hide-sm"><span class="pill {"pill-w" if p_m >= .5 else "pill-l"}">{"Michigan" if p_m >= .5 else esc(opp)}</span></td></tr>')
    msu = mich_games[(mich_games["home_team"] == "Michigan State") | (mich_games["away_team"] == "Michigan State")].iloc[0]

    weights_note = ", ".join(f"{k.replace('_', ' ')} {v:+.1f}" for k, v in com["weights_standardized"].items())
    top_coef = coef.head(8)
    coef_list = "".join(
        f"<li><span>{esc(FEATURE_LABELS.get(r.feature, r.feature))}</span> "
        f"<span class='num coef'>{r.weight:+.2f}</span> <code class='muted'>{esc(r.feature)}</code></li>"
        for r in top_coef.itertuples())
    g = datetime.now(timezone.utc).strftime("%b %d, %Y")
    champs_ok = sum(1 for v in audit["champions"].values() if v["ok"])

    return f"""<title>CFB Predictor Bake-off</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column like a game-day program: scoreboard header, verdict, film-room charts, title odds, method. */
:root {{
  --bg: #f5f6f2; --surface: #fcfcfb; --ink: #0e1210; --ink-2: #4b524e; --muted: #7d837d;
  --grid: #e2e4dd; --axis: #c3c6bc; --accent: #1d6b40; --accent-wash: #e3efe6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --deemph: #b8bcb3; --win: #006300; --loss: #b4322f;
  --ring: rgba(14,18,16,.10);
  --font-display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --font-body: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0d0f0e; --surface: #151816; --ink: #eef0ec; --ink-2: #b8beb8; --muted: #8b918b;
  --grid: #272b28; --axis: #3a3f3b; --accent: #5fc08a; --accent-wash: #16261c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --deemph: #4b504c; --win: #0ca30c; --loss: #e66767;
  --ring: rgba(255,255,255,.10); color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0d0f0e; --surface: #151816; --ink: #eef0ec; --ink-2: #b8beb8; --muted: #8b918b;
  --grid: #272b28; --axis: #3a3f3b; --accent: #5fc08a; --accent-wash: #16261c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --deemph: #4b504c; --win: #0ca30c; --loss: #e66767;
  --ring: rgba(255,255,255,.10); color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 16px; line-height: 1.55; }}
.page {{ max-width: 1040px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
h1, h2, h3 {{ font-family: var(--font-display); text-wrap: balance; letter-spacing: .01em; margin: 0; }}
h1 {{ font-size: clamp(2.2rem, 6vw, 3.6rem); line-height: 1.02; font-weight: 700; }}
h2 {{ font-size: clamp(1.55rem, 3.6vw, 2.1rem); line-height: 1.1; font-weight: 600; }}
h3 {{ font-size: 1.25rem; font-weight: 600; }}
p {{ margin: 0; max-width: 68ch; }}
.eyebrow {{ font-family: var(--font-data); font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}
.num {{ font-variant-numeric: tabular-nums; }}
code {{ font-family: var(--font-data); font-size: .85em; }}
a {{ color: var(--accent); }}
a:focus-visible, [tabindex]:focus-visible, summary:focus-visible {{ outline: 2px solid var(--s1); outline-offset: 2px; border-radius: 4px; }}
header.top {{ display: grid; gap: 14px; padding-bottom: 22px; border-bottom: 1px solid var(--grid); }}
.lede {{ font-size: 1.12rem; color: var(--ink-2); }}
.meta {{ font-family: var(--font-data); font-size: .78rem; color: var(--muted); }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 8px; }}
.kpi {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; display: grid; gap: 2px; }}
.kpi .label {{ font-size: .86rem; color: var(--ink-2); }}
.kpi .value {{ font-size: 2rem; font-weight: 700; line-height: 1.1; }}
.kpi .sub {{ font-size: .8rem; color: var(--muted); }}
section {{ display: grid; gap: 14px; padding-block: 34px 6px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 18px; display: grid; gap: 12px; min-width: 0; }}
.two {{ display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr); gap: 16px; align-items: start; }}
@media (max-width: 820px) {{ .two {{ grid-template-columns: minmax(0, 1fr); }} }}
.verdict {{ background: var(--accent-wash); border-radius: 12px; padding: 18px 20px; display: grid; gap: 8px; }}
.verdict strong {{ font-weight: 700; }}
.note {{ font-size: .9rem; color: var(--ink-2); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .86rem; color: var(--ink-2); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch-line {{ width: 16px; height: 2px; border-radius: 2px; background: currentColor; }}
.swatch-dot {{ width: 9px; height: 9px; border-radius: 50%; }}
.swatch-tick {{ width: 14px; height: 2px; background: var(--muted); }}
.swatch-line.s1, .swatch-dot.s1 {{ background: var(--s1); }} .swatch-line.s2, .swatch-dot.s2 {{ background: var(--s2); }}
.swatch-line.s3, .swatch-dot.s3 {{ background: var(--s3); }}
.svg-wrap {{ overflow-x: auto; }} .svg-wrap svg {{ min-width: 540px; }} .svg-wrap.sq svg {{ min-width: 300px; max-width: 460px; }}
svg.chart {{ width: 100%; height: auto; display: block; font-family: var(--font-body); }}
svg .grid {{ stroke: var(--grid); stroke-width: 1; }} svg .axis {{ stroke: var(--axis); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font-size: 12px; }} svg .tick.small {{ font-size: 10.5px; }}
svg .end-label {{ fill: var(--ink-2); font-size: 12.5px; dominant-baseline: middle; }}
svg .leader {{ stroke: var(--axis); stroke-width: 1; }}
svg .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
svg .line.s1 {{ stroke: var(--s1); }} svg .line.s2 {{ stroke: var(--s2); }} svg .line.s3 {{ stroke: var(--s3); }}
svg .dot {{ stroke: var(--surface); stroke-width: 2; }}
svg .dot.s1 {{ fill: var(--s1); }} svg .dot.s2 {{ fill: var(--s2); }} svg .dot.s3 {{ fill: var(--s3); }}
svg .diag {{ stroke: var(--axis); stroke-width: 1.5; }} svg .uni {{ stroke: var(--muted); stroke-width: 2; }}
svg .stem {{ stroke: var(--axis); stroke-width: 1.5; }}
svg .hit {{ fill: transparent; cursor: crosshair; }} svg .hit-c {{ fill: transparent; }}
svg .crosshair {{ stroke: var(--ink-2); stroke-width: 1; pointer-events: none; }}
svg .pt:hover .dot, svg .pt:focus .dot {{ r: 6; }}
.dw {{ display: grid; gap: 4px; font-size: .9rem; }}
.dw-row, .dw-axis {{ display: grid; grid-template-columns: minmax(118px, 190px) minmax(0, 1fr) 58px; gap: 10px; align-items: center; }}
.dw-row {{ padding: 3px 4px; border-radius: 6px; }} .dw-row:hover, .dw-row:focus {{ background: var(--accent-wash); }}
.dw-name {{ color: var(--ink-2); line-height: 1.2; }} .dw-row.is-win .dw-name {{ color: var(--ink); font-weight: 700; }}
.dw-track, .dw-ticks {{ position: relative; height: 20px; }}
.dw-track::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; border-left: 1.5px solid var(--axis); }}
.dw-ci {{ position: absolute; top: 9px; height: 2px; background: var(--deemph); border-radius: 2px; }}
.dw-dot {{ position: absolute; top: 5px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; background: var(--deemph); box-shadow: 0 0 0 2px var(--surface); }}
.dw-row.is-win .dw-dot {{ background: var(--s1); }} .dw-row.is-win .dw-ci {{ background: var(--s1); opacity: .55; }}
.dw-val {{ text-align: right; color: var(--ink-2); }}
.dw-tick {{ position: absolute; top: 0; transform: translateX(-50%); font-size: .72rem; color: var(--muted); font-family: var(--font-data); white-space: nowrap; }}
.bars {{ display: grid; gap: 6px; font-size: .92rem; }}
.bar-row {{ display: grid; grid-template-columns: minmax(110px, 230px) minmax(0, 1fr); gap: 10px; align-items: center; padding: 2px 4px; border-radius: 6px; }}
.bar-row:hover, .bar-row:focus {{ background: var(--accent-wash); }}
.bar-name {{ color: var(--ink-2); line-height: 1.2; }} .bar-row.is-win .bar-name {{ color: var(--ink); font-weight: 700; }}
.bar-track {{ position: relative; height: 22px; margin-right: 64px; }}
.bar {{ position: absolute; left: 0; top: 3px; height: 16px; background: var(--s1); border-radius: 0 4px 4px 0; }}
.bar-val {{ position: absolute; top: 1px; padding-left: 8px; font-size: .85rem; color: var(--ink); white-space: nowrap; }}
details.tv {{ font-size: .88rem; }} details.tv summary {{ cursor: pointer; color: var(--ink-2); width: fit-content; }}
.tbl-wrap {{ overflow-x: auto; margin-top: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
td.wrap {{ white-space: normal; }}
.dw-tick:first-child {{ transform: none; }}
th {{ color: var(--ink-2); font-weight: 600; }} td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.stats4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
@media (max-width: 760px) {{ .stats4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.sr {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }}
.swatch-line.diag-key {{ background: var(--axis); }}
ul.clean li .coef {{ font-weight: 700; margin-left: 4px; }} code.muted {{ color: var(--muted); font-size: .76rem; margin-left: 4px; }}
@media (max-width: 560px) {{
  .dw-row {{ grid-template-columns: minmax(0, 1fr) 52px; grid-template-areas: "name val" "track track"; row-gap: 0; }}
  .dw-row .dw-name {{ grid-area: name; }} .dw-row .dw-val {{ grid-area: val; }} .dw-row .dw-track {{ grid-area: track; }}
  .dw-axis {{ grid-template-columns: minmax(0, 1fr); }} .dw-axis > span:first-child, .dw-axis > span:last-child {{ display: none; }}
  .dw-tick.minor {{ display: none; }}
  .bar-row {{ grid-template-columns: minmax(80px, 110px) minmax(0, 1fr); }} .bar-track {{ margin-right: 92px; }}
  .hide-sm {{ display: none; }}
  th, td {{ padding: 6px 5px; }}
}}
.stat {{ border-left: 3px solid var(--s1); padding: 2px 0 2px 12px; display: grid; }}
.stat .v {{ font-size: 1.6rem; font-weight: 700; }} .stat .l {{ font-size: .86rem; color: var(--ink-2); }}
.pill {{ font-size: .78rem; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--ring); }}
.pill-w {{ color: var(--win); }} .pill-l {{ color: var(--loss); }}
ul.clean {{ margin: 0; padding-left: 18px; display: grid; gap: 4px; }}
dl.gloss {{ display: grid; grid-template-columns: minmax(120px, 190px) minmax(0, 1fr); gap: 8px 16px; margin: 0; }}
dl.gloss dt {{ font-weight: 700; }} dl.gloss dd {{ margin: 0; color: var(--ink-2); }}
@media (max-width: 520px) {{ dl.gloss {{ grid-template-columns: minmax(0, 1fr); }} dl.gloss dd {{ margin-bottom: 6px; }} }}
#tip {{ position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  box-shadow: 0 6px 20px rgba(0,0,0,.16); border-radius: 8px; padding: 8px 10px; font-size: .82rem; max-width: 280px; white-space: pre-line; }}
#tip .lead {{ font-weight: 700; }}
footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--grid); font-size: .82rem; color: var(--muted); display: grid; gap: 6px; }}
@media (prefers-reduced-motion: no-preference) {{ .bar {{ transition: width .4s ease; }} }}
</style>

<div class="page">
<header class="top">
  <span class="eyebrow">CFB Predictor · Model bake-off · Test seasons 2014–2025</span>
  <h1>16 algorithms took the same exam. A simple one passed best.</h1>
  <p class="lede">Every algorithm studied only the past, then predicted a season it had never seen, twelve times in a row.
  The winner predicts the final point margin, then turns it into a win chance.</p>
  <span class="meta">Built {esc(g)} · Data through {esc(data_through.strftime('%b %d, %Y'))} (2026 week 6) · {n_test:,} test games</span>
  <div class="kpis">
    <div class="kpi"><span class="label">Picked the winner</span><span class="value">{pct(w['accuracy'])}</span><span class="sub">of {n_test:,} games, 2014–2025</span></div>
    <div class="kpi"><span class="label">Right when 80%+ sure</span><span class="value">{pct(w['acc_when_80pct_sure'])}</span><span class="sub">{pct(w['share_80pct_sure'], 0)} of games</span></div>
    <div class="kpi"><span class="label">Calibration error</span><span class="value">{100 * w['ece']:.1f} pts</span><span class="sub">"70%" really means ≈70%</span></div>
    <div class="kpi"><span class="label">Behind Vegas</span><span class="value">+{vgap:.3f}</span><span class="sub">log loss; Vegas still leads</span></div>
  </div>
</header>

<section aria-labelledby="verdict">
  <div class="verdict">
    <span class="eyebrow">The verdict</span>
    <h2 id="verdict">Winner: margin ridge, a straight-line point-spread model</h2>
    <p>Six algorithms finished in a statistical tie at the top. The highest scorer, margin LightGBM, was not significantly better
    than the simpler linear models (Holm-adjusted p = {sig.loc['margin_ridge', 'p_holm']:.2f}). Our rule, written down
    <em>before</em> the test, says to pick the simplest model in a tie. So the winner is <strong>margin ridge</strong>: it predicts
    "Team A by 6.5" from about 40 pregame clues, then a bell curve turns that into a win chance. It is easy to explain,
    fast, and gives both a point spread and a percentage.</p>
    <p class="note">It beats our own Elo rating in {beat_elo} of 12 seasons and trails the Vegas closing line by about {vgap:.3f} log loss
    (Vegas knows about injuries, quarterbacks and weather; we don't use them). Its average miss on the final margin is
    12.9 points, against 12.4 for Vegas.</p>
  </div>
</section>

<section aria-labelledby="h-gap">
  <span class="eyebrow">Leaderboard</span>
  <h2 id="h-gap">How far behind Vegas is each algorithm?</h2>
  <p>Same {int(allg.loc['vegas_spread', 'n']):,} games for everyone (every test game with a betting line). The dot is the gap in log loss;
  the line through it is the 95% range from 2,000 bootstrap re-runs. Further left is better; 0 would mean tied with Vegas.</p>
  <div class="card">{chart_gap}{gap_table}</div>
</section>

<section aria-labelledby="h-recall">
  <span class="eyebrow">Recall</span>
  <h2 id="h-recall">Which winners did it catch?</h2>
  <div class="card">
    <div class="stats4">
      <div class="stat"><span class="v">{pct(w['recall_home'])}</span><span class="l">of home-team wins caught</span></div>
      <div class="stat"><span class="v">{pct(w['recall_away'])}</span><span class="l">of road-team wins caught</span></div>
      <div class="stat"><span class="v">{pct(w['balanced_accuracy'])}</span><span class="l">balanced accuracy (average of both)</span></div>
      <div class="stat"><span class="v">{pct(w['upset_recall'])}</span><span class="l">of Vegas upsets called ({int(w['upsets']):,} upsets)</span></div>
    </div>
    <p class="note">Recall means: of all the games a side really won, how many did we see coming? Road teams are usually the
    underdogs, so they are harder to call. Upsets are rare by definition: a well-calibrated model should <em>not</em> call many of them,
    or it would be wrong much more often. Chasing upset recall makes the overall picks worse.</p>
  </div>
  <div class="card">
    <h3>Every algorithm, every metric</h3>
    <p class="note">All {n_test:,} test games, sorted by log loss (the primary score). The last column is the Holm-adjusted p-value
    against the leader: above 0.05 means "statistically tied with the leader". CFBD Elo covers fewer games, so it is
    compared with Vegas on its own games instead (n/a here).</p>
    {score_tbl}
  </div>
</section>

<section aria-labelledby="h-cal">
  <span class="eyebrow">Confidence</span>
  <h2 id="h-cal">When it says 70%, does it happen 70% of the time?</h2>
  <div class="two">
    <div class="card">{chart_rel}{rel_table}</div>
    <div class="card">
      <p>Each dot groups games by what the model said before kickoff, then checks how often that team really won.
      Dots on the diagonal mean the confidence is honest. Our model's average gap is <strong>{100 * w['ece']:.1f} percentage points</strong>
      (Elo: {100 * allg.loc['elo', 'ece']:.1f}, Vegas: {100 * allg.loc['vegas_spread', 'ece']:.1f}).</p>
      <p class="note">Calibration slope {w['calib_slope']:.2f} (1.00 is perfect; above 1 means slightly too cautious, below 1 means overconfident).
      When the model is at least 80% sure, which happens in {pct(w['share_80pct_sure'], 0)} of games, it is right {pct(w['acc_when_80pct_sure'])} of the time.</p>
    </div>
  </div>
</section>

<section aria-labelledby="h-season">
  <span class="eyebrow">Stability</span>
  <h2 id="h-season">Season by season, 2014–2025</h2>
  <p>Log loss each test season (lower is better). Some seasons are harder for everyone: 2020 (COVID schedules) and 2021–2022 (transfer portal churn) stand out.</p>
  <div class="card">{chart_season}{season_table}</div>
</section>

<section aria-labelledby="h-abl">
  <span class="eyebrow">What matters</span>
  <h2 id="h-abl">Which data earns its place?</h2>
  <p>We re-ran the winner five times, adding one group of clues each time. The bar is how much log loss it saved compared with Elo alone.
  Roster data (recruiting talent, returning players, transfer portal) is the biggest single gain after the opponent-adjusted ratings.</p>
  <div class="two">
    <div class="card">{chart_abl}{abl_table}</div>
    <div class="card">
      <h3>Strongest clues in the final model</h3>
      <p class="note">Weights on a common scale (points of margin per typical change). Clues overlap, so read them as a group, not one by one.</p>
      <ul class="clean">{coef_list}</ul>
    </div>
  </div>
</section>

<section aria-labelledby="h-champ">
  <span class="eyebrow">Champion backtest</span>
  <h2 id="h-champ">Could it have called the champion?</h2>
  <p>For every playoff season we rebuilt the real bracket, used the model that season would have had (trained only on earlier years),
  and played the playoff 20,000 times. The model's pre-playoff favorite won <strong>{brs['favorite_won']} of 12</strong> titles, and the champion was
  in its top two <strong>{brs['champion_in_model_top2']} of 12</strong> times. On average it gave the real champion {pct(brs['avg_p_champion_model'])},
  against {pct(brs['avg_p_champion_elo'])} for Elo and {pct(brs['avg_p_champion_uniform'])} for a random guess.</p>
  <div class="card">{chart_ch}{ch_table}</div>
</section>

<section aria-labelledby="h-2026">
  <span class="eyebrow">Live forecast · {CURRENT_SEASON} season · title game January 2027</span>
  <h2 id="h-2026">Who wins the national title in January 2027?</h2>
  <p>10,000 simulated seasons from today's data: the remaining {ometa['games_simulated']} games, conference title games, a committee
  emulator (it picks {com['avg_top4_overlap']:.1f} of the real top 4 and {com['avg_top12_overlap']:.1f} of the real top 12 in seasons it never saw),
  then the 12-team playoff under the 2026 rules.</p>
  <div class="two">
    <div class="card"><h3>Title odds, top 12</h3>{chart_odds}{odds_table}</div>
    <div class="card">
      <h3>Michigan, as of today</h3>
      <p>{esc(mich['record_now'])} so far. Expected regular-season finish: {mich['exp_wins']:.1f} wins.
      Playoff chance {pct(mich['p_playoff'])}; title chance {pct(mich['p_title'], 2)}.</p>
      <div class="tbl-wrap"><table><thead><tr><th class="num">Wk</th><th>Game</th><th class="num">Line</th><th class="num">Win %</th><th class="hide-sm">Pick</th></tr></thead>
      <tbody>{''.join(mg_rows)}</tbody></table></div>
      <p class="note">Michigan vs Michigan State (week {int(msu['week'])}): Michigan by {abs(msu['home_margin']):.1f},
      {pct(msu['home_win_prob'] if msu['home_team'] == 'Michigan' else 1 - msu['home_win_prob'], 0)} to win.</p>
    </div>
  </div>
  <p class="note">Watch-out: the model weighs recruiting talent heavily, so it rates high-talent teams (Georgia, Ohio State) above
  teams whose results run ahead of their recruiting (Indiana has the best results-based rating but is a 6-point underdog to Georgia on a neutral field).
  2025 Indiana won the title as exactly that kind of team. Simulation noise is about ±0.5 percentage points.</p>
</section>

<section aria-labelledby="h-method">
  <span class="eyebrow">How we tested</span>
  <h2 id="h-method">The method, and its limits</h2>
  <div class="two">
    <div class="card">
      <ul class="clean">
        <li><strong>Time machine, never a shuffle.</strong> Tune on 2010–2013 only, then train on all earlier seasons and predict the next one, 2014 through 2025.</li>
        <li><strong>No peeking.</strong> Every clue is computed from games before kickoff. An automated "time-travel" test erases future scores and checks that nothing changes.</li>
        <li><strong>Both sides.</strong> Each game is predicted as A-vs-B and B-vs-A and averaged, so the two chances always add to 100%.</li>
        <li><strong>Fair fights.</strong> Every algorithm sees the same games. Gaps get 95% ranges from a game-week bootstrap and Holm-corrected head-to-head tests.</li>
        <li><strong>Clean data.</strong> {audit['checks']['duplicate_game_ids']} duplicates, {audit['checks']['ties']} impossible ties, and {champs_ok} of 22 champions match real history.</li>
      </ul>
    </div>
    <div class="card">
      <h3>Known limits</h3>
      <ul class="clean">
        <li>No injuries, quarterback changes, weather or betting data go into the model.</li>
        <li>Bowls and playoff games are close to coin flips for every model (opt-outs and motivation).</li>
        <li>The committee is people, not a formula; the emulator copies their past habits.</li>
        <li>The 2026 forecast freezes today's ratings; uncertainty comes from simulated luck and a ±{ometa['rating_uncertainty_tau_points']:.1f}-point "true strength" wobble per team.</li>
      </ul>
      <p class="note">Committee emulator weights (standardized): {esc(weights_note)}.</p>
    </div>
  </div>
  <div class="card">
    <h3>Glossary</h3>
    <dl class="gloss">
      <dt>Log loss</dt><dd>A penalty score for predictions. Being confident and wrong costs a lot. Lower is better; a coin flip scores 0.693.</dd>
      <dt>Brier score</dt><dd>The average squared gap between the chance we gave and what happened (1 or 0). Lower is better.</dd>
      <dt>Recall</dt><dd>Of the games a side really won, the share we predicted correctly.</dd>
      <dt>Calibration</dt><dd>Whether our percentages mean what they say: 70% should come true about 70% of the time.</dd>
      <dt>Walk-forward test</dt><dd>Train on the past, predict the next season, step forward a year, repeat.</dd>
      <dt>Ridge regression</dt><dd>A straight-line formula with a "seat belt" that stops it from trusting any one clue too much.</dd>
    </dl>
  </div>
</section>

<footer>
  <span>Data: SportsDataverse open data (cfbfastR-data and sportsdataverse-data on GitHub), originally from CollegeFootballData.com and ESPN.
  Betting lines are a benchmark only. Model: {esc(card['name'])}, trained on {card['n_training_games']:,} games ({card['trained_on_seasons'][0]}–{card['trained_on_seasons'][1]}).</span>
  <span>Re-create everything from the project folder: <code>python -m cfb.pipeline</code>.</span>
</footer>
</div>
<div id="tip" hidden></div>
<script>
(() => {{
  const tip = document.getElementById("tip");
  const show = (el, x, y) => {{
    const lines = (el.getAttribute("data-tip") || "").split("\\n");
    tip.replaceChildren();
    lines.forEach((t, i) => {{ const d = document.createElement("div"); d.textContent = t; if (i === 0) d.className = "lead"; tip.appendChild(d); }});
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
    const svg = el.ownerSVGElement; const xh = svg && svg.querySelector(".crosshair");
    if (xh && el.dataset.x) {{ xh.setAttribute("x1", el.dataset.x); xh.setAttribute("x2", el.dataset.x); xh.setAttribute("visibility", "visible"); }}
  }};
  const hide = (el) => {{ tip.hidden = true; const svg = el.ownerSVGElement; const xh = svg && svg.querySelector(".crosshair"); if (xh) xh.setAttribute("visibility", "hidden"); }};
  document.querySelectorAll("[data-tip]").forEach(el => {{
    el.addEventListener("pointermove", e => show(el, e.clientX, e.clientY));
    el.addEventListener("pointerleave", () => hide(el));
    el.addEventListener("focus", () => {{ const r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); }});
    el.addEventListener("blur", () => hide(el));
  }});
}})();
</script>
"""


SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
{content}
</body>
</html>
"""


def main(artifact_copy: str | None = None) -> None:
    """Write the local report (full HTML document). Optionally also the bare page body
    used for publishing as an artifact (the publisher adds the document wrapper itself)."""
    page = build()
    head, _, rest = page.partition("<div class=\"page\">")
    local = SKELETON.format(content=head + "</head>\n<body>\n<div class=\"page\">" + rest)
    out = REPORTS / "model_report.html"
    out.write_text(local, encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")
    if artifact_copy:
        from pathlib import Path
        Path(artifact_copy).write_text(page, encoding="utf-8")
        print("wrote", artifact_copy)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
