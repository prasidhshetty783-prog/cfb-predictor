"""CFB Predictor: public dashboard.

The dashboard only READS files made by the weekly pipeline (reports/app/). It never runs
the model, so it is fast, free to host, and past picks can't change after kickoff.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "reports" / "app"
ET = ZoneInfo("America/New_York")
REPO = "https://github.com/prasidhshetty783-prog/cfb-predictor"
POWER = {"SEC", "Big Ten", "Big 12", "ACC"}

TURF, INK, MUTED, LINE, PYLON, MISS = "#1F7A47", "#10241A", "#5B6B62", "#DCE3DC", "#C2541A", "#B23A2B"

st.set_page_config(page_title="CFB Predictor", page_icon="🏈", layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input {{ font-family: 'Public Sans', system-ui, sans-serif; }}
h1, h2, h3, .num {{ font-family: 'Saira Condensed', 'Arial Narrow', sans-serif !important; letter-spacing: .01em; }}
.block-container {{ padding-top: 1.2rem; max-width: 1180px; }}
a, a:visited {{ color: {TURF} !important; }}
:focus-visible {{ outline: 2px solid {TURF}; outline-offset: 2px; }}

/* scoreboard header: the one loud element */
.board {{ background: {INK}; color: #F3F7F2; border-radius: 14px; padding: 22px 26px 18px;
  background-image: repeating-linear-gradient(90deg, rgba(255,255,255,.07) 0 2px, transparent 2px 64px);
  display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: end; }}
.board h1 {{ font-size: 2.6rem; line-height: 1; margin: 0; color: #fff; font-weight: 700; }}
.board .sub {{ color: #B9CBBE; font-size: .95rem; margin-top: 6px; }}
.board .stats {{ display: flex; gap: 28px; }}
.board .stat .v {{ font-family: 'Saira Condensed', sans-serif; font-size: 2.1rem; font-weight: 700; line-height: 1; color: #fff; }}
.board .stat .l {{ font-size: .8rem; color: #B9CBBE; max-width: 150px; }}
@media (max-width: 760px) {{ .board {{ grid-template-columns: 1fr; align-items: start; padding: 18px; }} .board h1 {{ font-size: 2.1rem; }}
  .board .stats {{ gap: 20px; flex-wrap: wrap; }} .board .stat .v {{ font-size: 1.7rem; }} }}
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ height: 0; background: transparent; }}

.weeknote {{ color: {MUTED}; font-size: .95rem; margin: 2px 0 10px; }}
.tag {{ display: inline-block; font-size: .74rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; border: 1px solid; }}
.tag.early {{ color: {PYLON}; border-color: {PYLON}; }}
.tag.ok {{ color: {TURF}; border-color: {TURF}; }}
.tag.miss {{ color: {MISS}; border-color: {MISS}; }}
.tag.soft {{ color: {MUTED}; border-color: {LINE}; }}

.game {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px 12px; margin-bottom: 12px; }}
.game .top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; color: {MUTED}; font-size: .82rem; }}
.team {{ display: grid; grid-template-columns: 30px 1fr auto; align-items: center; gap: 10px; margin-top: 8px; }}
.logo {{ width: 28px; height: 28px; background: center / contain no-repeat; display: inline-block; }}
.team .nm {{ font-weight: 600; font-size: 1rem; color: {INK}; }}
.team .nm.pick::after {{ content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {TURF}; margin-left: 7px; vertical-align: middle; }}
.team .sc {{ font-family: 'Saira Condensed', sans-serif; font-size: 1.35rem; font-weight: 700; color: {INK}; }}
.team .sc.lose {{ color: {MUTED}; font-weight: 500; }}
.bar {{ height: 8px; background: {LINE}; border-radius: 4px; margin: 12px 0 6px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; border-radius: 4px; }}
.pickline {{ display: flex; justify-content: space-between; font-size: .92rem; color: {INK}; }}
.pickline b {{ font-family: 'Saira Condensed', sans-serif; font-size: 1.1rem; }}
.why {{ color: {MUTED}; font-size: .8rem; margin-top: 4px; }}
.moved {{ color: {PYLON}; font-size: .8rem; }}

.odds-row {{ display: grid; grid-template-columns: 28px 30px minmax(110px, 190px) 1fr 64px 70px; gap: 10px; align-items: center;
  padding: 7px 4px; border-bottom: 1px solid {LINE}; }}

.odds-row .rk {{ color: {MUTED}; font-family: 'Saira Condensed', sans-serif; font-size: 1.05rem; text-align: right; }}
.odds-row .tm {{ font-weight: 600; }} .odds-row .cf {{ color: {MUTED}; font-size: .78rem; display: block; font-weight: 400; }}
.odds-row .track {{ height: 10px; background: {LINE}; border-radius: 4px; overflow: hidden; }}
.odds-row .track span {{ display: block; height: 100%; background: {TURF}; border-radius: 0 4px 4px 0; }}
.odds-row .v {{ font-family: 'Saira Condensed', sans-serif; font-weight: 700; font-size: 1.1rem; text-align: right; }}
.odds-row .p {{ color: {MUTED}; font-size: .85rem; text-align: right; }}
.odds-head {{ color: {MUTED}; font-size: .78rem; }} .odds-head .p {{ font-size: .78rem; }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 6px 0 8px; }}
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
.kpi .l {{ color: {MUTED}; font-size: .85rem; }} .kpi .s {{ color: {MUTED}; font-size: .8rem; }}
.kpi .v {{ font-family: 'Saira Condensed', sans-serif; font-size: 2.2rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
@media (max-width: 640px) {{ .kpis {{ grid-template-columns: 1fr; }} }}
@media (max-width: 640px) {{ .odds-row {{ grid-template-columns: 22px 26px 1fr 54px; }} .odds-row .track, .odds-row .p {{ display: none; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(ttl=600)
def load():
    games = pd.read_csv(DATA / "games.csv")
    teams = pd.read_csv(DATA / "teams.csv").drop_duplicates("team_id").set_index("team_id")
    meta = json.loads((DATA / "meta.json").read_text())
    record = json.loads((DATA / "record.json").read_text())
    cal = pd.read_csv(DATA / "calibration.csv")
    odds = pd.read_csv(DATA / "title_odds.csv")
    ahead = pd.read_csv(DATA / "by_weeks_ahead.csv") if (DATA / "by_weeks_ahead.csv").exists() else None
    games["kickoff"] = pd.to_datetime(games["kickoff_utc"], utc=True).dt.tz_convert(ET)
    return games, teams, meta, record, cal, odds, ahead


games, teams, meta, record, cal, odds, ahead = load()


def esc(x) -> str:
    return html.escape(str(x))


def logo(tid) -> str:
    try:
        url = teams.loc[int(tid), "logo"]
        return url if isinstance(url, str) else ""
    except (KeyError, ValueError):
        return ""


def logo_html(tid) -> str:
    url = logo(tid)
    style = f' style="background-image:url(\'{esc(url)}\')"' if url else ""
    return f'<span class="logo" aria-hidden="true"{style}></span>'


def team_row(name, tid, pts, is_pick: bool, won: bool, is_final: bool) -> str:
    sc = f'<span class="sc{"" if won else " lose"}">{int(pts)}</span>' if is_final else ""
    return (f'<div class="team">{logo_html(tid)}<span class="nm{" pick" if is_pick else ""}">'
            f'{esc(name)}</span>{sc}</div>')


def team_color(tid) -> str:
    """Team color for the pick bar; fall back to turf if missing or too light to see."""
    try:
        c = str(teams.loc[int(tid), "color"]).strip()
    except (KeyError, ValueError):
        return TURF
    if not c.startswith("#") or len(c) != 7:
        return TURF
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return TURF if (0.299 * r + 0.587 * g + 0.114 * b) > 205 else c


def conf(tid) -> str:
    try:
        v = teams.loc[int(tid), "conference"]
        return v if isinstance(v, str) else ""
    except (KeyError, ValueError):
        return ""


def pct(x) -> str:
    if x >= 0.995:
        return ">99%"
    return f"{100 * x:.0f}%"


def kickoff_text(t: pd.Timestamp) -> str:
    if t.hour == 0 and t.minute == 0:
        return t.strftime("%a %b %-d") + " · time TBA"
    return t.strftime("%a %b %-d · %-I:%M %p ET")


# ------------------------------------------------------------------ header (scoreboard)
r_all = record["all"]
updated = datetime.fromisoformat(meta["updated_utc"].replace("Z", "+00:00")).astimezone(ET)
acc = r_all["right"] / r_all["games"] if r_all["games"] else 0
sure = r_all["sure_right"] / r_all["sure_games"] if r_all["sure_games"] else 0
st.markdown(f"""
<div class="board" role="banner">
  <div>
    <h1>CFB Predictor</h1>
    <div class="sub">{meta['season']} college football picks from a model tested on 10,371 past games.
    Updated {updated.strftime('%a %b %-d, %-I:%M %p ET')}.</div>
  </div>
  <div class="stats">
    <div class="stat"><div class="v">{r_all['right']}–{r_all['games'] - r_all['right']}</div><div class="l">right–wrong in {meta['season']} ({pct(acc)})</div></div>
    <div class="stat"><div class="v">{pct(sure)}</div><div class="l">right when 80%+ sure</div></div>
    <div class="stat"><div class="v">Wk {meta['next_week']}</div><div class="l">up next</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_games, tab_record, tab_odds, tab_how = st.tabs(["Games", "Season record", "Title odds", "How it works"])

# ------------------------------------------------------------------ games
with tab_games:
    weeks = sorted(int(w) for w in games["week"].unique())
    week = st.pills("Week", options=weeks, default=meta["next_week"], key="week") or meta["next_week"]
    c2, c3 = st.columns([2, 1], vertical_alignment="bottom")
    with c2:
        query = st.text_input("Find a team", placeholder="e.g. Michigan")
    with c3:
        power_only = st.toggle("Power 4 games", value=True, help="SEC, Big Ten, Big 12, ACC and Notre Dame")

    q = query.strip().lower()
    if q:   # a team search covers the whole season, not just the chosen week
        wk = games[games["home_team"].str.lower().str.contains(q, regex=False)
                   | games["away_team"].str.lower().str.contains(q, regex=False)].sort_values("kickoff")
    else:
        wk = games[games["week"] == week].sort_values("kickoff")
    if not q and power_only:
        keep = wk.apply(lambda r: conf(r["home_id"]) in POWER or conf(r["away_id"]) in POWER
                        or "Notre Dame" in (r["home_team"], r["away_team"]), axis=1)
        wk = wk[keep] if len(wk) else wk

    done = wk[wk["status"] == "final"]
    if q:
        note = (f"Every {meta['season']} game for teams matching “{query.strip()}”. "
                f"Finished games: {int(done['pick_right'].sum())} of {len(done)} picked right.")
    elif week < meta["next_week"]:
        right = int(done["pick_right"].sum())
        note = f"Final. The model picked {right} of {len(done)} of these games right."
        if (done["source"] != "logged before kickoff").any():
            note += " Picks for weeks before tracking began were re-created afterward from pre-game data only."
    elif week == meta["next_week"]:
        note = "This week's picks. They can still update until each game kicks off, then they lock."
    else:
        k = week - meta["next_week"]
        hist = ""
        if ahead is not None and k in set(ahead["weeks_ahead"]):
            hist = f" Picks made {k} week{'s' if k > 1 else ''} early were right {pct(float(ahead.set_index('weeks_ahead').loc[k, 'accuracy']))} of the time in past seasons."
        note = f"Early look. These picks will change as results come in.{hist}"
    st.markdown(f'<p class="weeknote">{esc(note)}</p>', unsafe_allow_html=True)

    if wk.empty:
        st.info("No games match. Clear the search or switch off “Power 4 games” to see every game.")

    cards = []
    for g in wk.itertuples():
        is_final = g.status == "final"
        away_win = is_final and g.away_points > g.home_points
        pick_home = g.pick == g.home_team
        tags = []
        if is_final:
            tags.append('<span class="tag ok">✓ Correct</span>' if g.pick_right == 1 else '<span class="tag miss">✗ Missed</span>')
        elif g.weeks_ahead and g.weeks_ahead > 0:
            tags.append('<span class="tag early">Early look</span>')
        if g.neutral_site:
            tags.append('<span class="tag soft">Neutral site</span>')

        moved = ""
        if not is_final and pd.notna(g.prev_pick_prob) and abs(g.pick_prob - g.prev_pick_prob) >= 0.03:
            moved = f'<div class="moved">Was {pct(g.prev_pick_prob)} last update</div>'
        why = f'<div class="why">Why: {esc(g.reasons)} points</div>' if isinstance(g.reasons, str) and g.reasons else ""
        cards.append(f"""
<div class="game" aria-label="{esc(g.away_team)} at {esc(g.home_team)}">
  <div class="top"><span>{f"Week {g.week} · " if q else ""}{esc(kickoff_text(g.kickoff))}</span><span>{' '.join(tags)}</span></div>
  {team_row(g.away_team, g.away_id, g.away_points, not pick_home, away_win, is_final)}
  {team_row(g.home_team, g.home_id, g.home_points, pick_home, is_final and not away_win, is_final)}
  <div class="bar" role="img" aria-label="{esc(g.pick)} {pct(g.pick_prob)} to win"><span style="width:{100 * g.pick_prob:.1f}%; background:{team_color(g.pick_id)}"></span></div>
  <div class="pickline"><span>Pick: <b>{esc(g.pick)}</b></span><span><b>{pct(g.pick_prob)}</b> · by {'<1' if g.pick_by < 0.5 else f'{g.pick_by:.0f}'}</span></div>
  {moved}{why}
</div>""")
    for i in range(0, len(cards), 2):        # row by row, so phones keep kickoff order
        for col, c in zip(st.columns(2)[:len(cards[i:i + 2])], cards[i:i + 2], strict=True):
            col.markdown(c, unsafe_allow_html=True)

# ------------------------------------------------------------------ season record
with tab_record:
    lg, rc = record["logged"], record["recreated"]
    bt = meta["backtest"]
    tiles = [("Picks right this season", pct(acc), f"{r_all['right']} of {r_all['games']} games"),
             ("Right when 80%+ sure", pct(sure), f"{r_all['sure_right']} of {r_all['sure_games']} games"),
             ("Past-season test, 2014–2025", pct(bt["accuracy"]), f"{bt['games']:,} games")]
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{esc(a)}</div><div class="v">{b}</div><div class="s">{esc(c)}</div></div>'
        for a, b, c in tiles) + "</div>", unsafe_allow_html=True)
    logged_txt = (f"Picks saved before kickoff: {lg['right']} of {lg['games']} right. " if lg["games"]
                  else "Live tracking starts with week 7, so every finished game so far is a re-created pick. ")
    st.caption(logged_txt + f"Re-created picks use only information known before each game: {rc['right']} of {rc['games']} right. "
               "Early-season weeks include many easy games against small schools, so expect this to drift down "
               "toward the 76% test result.")

    fin = games[(games["status"] == "final")]
    by_week = fin.groupby("week").agg(games=("pick_right", "size"), right=("pick_right", "sum")).reset_index()
    by_week["accuracy"] = by_week["right"] / by_week["games"]
    st.subheader("Picks right, week by week")
    chart = (alt.Chart(by_week).mark_bar(color=TURF, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=26)
             .encode(x=alt.X("week:O", title="Week", axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("accuracy:Q", title="Share of picks right", scale=alt.Scale(domain=[0, 1]),
                             axis=alt.Axis(format="%", grid=True, gridColor=LINE)),
                     tooltip=[alt.Tooltip("week:O", title="Week"), alt.Tooltip("right:Q", title="Right"),
                              alt.Tooltip("games:Q", title="Games"), alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%")])
             .properties(height=260))
    st.altair_chart(chart.configure_view(strokeWidth=0).configure_axis(labelColor=MUTED, titleColor=MUTED,
                                                                         domainColor=LINE, tickColor=LINE),
                    use_container_width=True)

    st.subheader("Are the percentages honest?")
    st.caption("Each dot groups this season's finished games by how sure the model was. "
               "On the line means honest: when it says 70%, it wins about 70%.")
    line = alt.Chart(pd.DataFrame({"x": [0.5, 1], "y": [0.5, 1]})).mark_line(color="#AEBBB2", strokeWidth=2, strokeDash=[5, 4]).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.5, 1], zero=False)), y=alt.Y("y:Q", scale=alt.Scale(domain=[0.5, 1], zero=False)))
    dots = (alt.Chart(cal).mark_circle(color=TURF, opacity=1, stroke="#fff", strokeWidth=2)
            .encode(x=alt.X("said:Q", title="How sure the model was", scale=alt.Scale(domain=[0.5, 1], zero=False), axis=alt.Axis(format="%", gridColor=LINE, tickCount=6)),
                    y=alt.Y("won:Q", title="How often the pick won", scale=alt.Scale(domain=[0.5, 1], zero=False), axis=alt.Axis(format="%", gridColor=LINE, tickCount=6)),
                    size=alt.Size("games:Q", legend=None, scale=alt.Scale(range=[80, 500])),
                    tooltip=[alt.Tooltip("bin:N", title="Confidence"), alt.Tooltip("games:Q", title="Games"),
                             alt.Tooltip("said:Q", title="Said", format=".0%"), alt.Tooltip("won:Q", title="Won", format=".0%")]))
    st.altair_chart((line + dots).properties(height=320).configure_view(strokeWidth=0)
                    .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor=LINE, tickColor=LINE),
                    use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(cal.rename(columns={"bin": "Confidence", "games": "Games", "said": "Average said", "won": "Actually won"})
                     .style.format({"Average said": "{:.0%}", "Actually won": "{:.0%}"}), hide_index=True)

# ------------------------------------------------------------------ title odds
with tab_odds:
    data_through = datetime.fromisoformat(meta["data_through_utc"].replace("Z", "+00:00")).astimezone(ET)
    st.markdown(f'<p class="weeknote">Chance to win the national title in January {meta["season"] + 1}, from 10,000 simulated '
                f'seasons with the 12-team playoff rules. Games through {data_through.strftime("%b %-d")}.</p>',
                unsafe_allow_html=True)
    top = odds.head(25)
    scale = max(float(top["p_title"].max()), 0.01)
    rows = ['<div class="odds-row odds-head"><span></span><span></span><span>Team</span><span class="track" style="background:none"></span>'
            '<span style="text-align:right">Title</span><span class="p">Playoff</span></div>']
    for i, r in enumerate(top.itertuples(), 1):
        title_txt = "<0.1%" if 0 < r.p_title < 0.001 else f"{100 * r.p_title:.1f}%"
        rows.append(f'<div class="odds-row" title="{esc(r.team)}: title {title_txt}, playoff {pct(r.p_playoff)}">'
                    f'<span class="rk">{i}</span>'
                    f'{logo_html(r.team_id)}'
                    f'<span class="tm">{esc(r.team)}<span class="cf">{esc(r.conference)} · {esc(r.record_now)}</span></span>'
                    f'<span class="track"><span style="width:{100 * r.p_title / scale:.1f}%"></span></span>'
                    f'<span class="v">{title_txt}</span><span class="p">{pct(r.p_playoff)}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("No team is above 50%: in a 12-team playoff even the best team usually has to win three or four games. "
               "Read these as chances, not promises.")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown(f"""
### How the picks are made
1. **Clues.** For every game the model compares the two teams on 40 things known before kickoff:
   power ratings adjusted for opponents, Elo, recruiting talent, returning players, transfers,
   this season's form, AP rank and home field.
2. **Margin.** A straight-line formula (ridge regression) weighs those clues to predict the point margin.
3. **Win chance.** Picks usually miss by about 16 points, so a bell curve turns the margin into a chance.
   A 7-point favorite wins about 67% of the time.

### How it was tested
16 methods (including simple baselines) took the same exam: predict each season from 2014 to 2025 using only earlier seasons
({bt['games']:,} games). This one picked **{pct(bt['accuracy'])}** of winners and was right
**{pct(bt['sure_right'])}** of the time when at least 80% sure. The Vegas line is a little better
(0.012 lower log loss); it can react to injury and lineup news, which this model doesn't see.

### Why past picks never change
Every pick is saved before kickoff and locked when the game starts. Each Monday an automatic job
downloads the weekend's results, updates every team's ratings, and re-picks only the games still
to be played. The saved picks live in the public repository, so anyone can check them.

Full method, code and results: [GitHub repository]({REPO}). Not betting advice.
""")
