import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data
import ui

s = data.summary()
p = data.panel()
model, service = data.plan_choice()
with st.spinner("Simulating the plan…"):
    res = data.plan(model, service)
bench = data.table("benchmark_weekly.csv").set_index("week")
brow = data.benchmark_rows()
heroes = data.heroes()

ui.md(f"""
<div class="oc-hero oc-anim">
  <div class="oc-eyebrow">Stock planner · VN2 Inventory Planning Challenge</div>
  <h1>Predict demand.<br/>Order exactly what the shelf needs.</h1>
  <p>A retail chain with <b>{s['stores']} stores</b> sells <b>{s['rows']} store-products</b> (one product in one
  store). Every Monday it must order stock that arrives two weeks later. OptiChain learns from three years of real
  weekly sales, predicts the coming weeks, and orders just enough: not too little (empty shelves), not too much
  (stock gathering dust).</p>
</div>
""")
ui.stepper(0)

# ------------------------------------------------------------------ money headline
saved = s["benchmark_score"] - res["score"]
per_year = saved * 52 / 6
left_p, left_b = res["weekly"].loc[3:8, "end"].mean(), bench.loc[3:8, "end"].mean()
lost_p, lost_b = res["weekly"].loc[3:8, "lost"].sum(), bench.loc[3:8, "lost"].sum()
ui.kpis([
    {"label": "Saved vs the official method", "value": ui.eur(saved), "accent": True,
     "sub": f"in 6 weeks · {saved / s['benchmark_score']:.0%} cheaper", "tone": "good" if saved > 0 else "bad"},
    {"label": "Over a year (estimate)", "value": f"≈ {ui.eur(round(per_year, -2))}",
     "sub": "if the 6 test weeks are typical"},
    {"label": "Stock sitting on shelves", "value": f"−{1 - left_p / left_b:.0%}",
     "sub": f"{left_p:,.0f} vs {left_b:,.0f} units left each week", "tone": "good" if left_p < left_b else ""},
    {"label": "To order this week", "value": f"{res['next_order'].sum():,.0f}", "unit": "units",
     "sub": f"for {(res['next_order'] > 0).sum()} of {s['rows']} store-products"},
])
ui.note(f"<b>The trade-off, honestly:</b> the plan keeps far less stock on the shelves, and accepts some more missed "
        f"sales ({lost_p:,.0f} vs {lost_b:,.0f} units over 6 weeks). That is the right deal under the rules: a box sitting a week "
        f"costs €0.20, a missed sale €1, and the plan's total is still {ui.eur(saved)} lower.")

# ------------------------------------------------------------------ the five heroes
ui.section("Meet the products", "Five products, five different stories",
           "Stores and products in this data are anonymous codes. These nicknames describe how each one behaves; "
           "the real code is underneath. Pick one to follow it from sales history to the shelf.")
cur = st.session_state.get("hero_row", heroes[0]["row"])
cols = st.columns(len(heroes))
for col, h in zip(cols, heroes):
    sel = h["row"] == cur
    recent = p.sales.iloc[h["row"], p.t0 - 51:p.t0 + 1].rolling(4, min_periods=1).mean().to_numpy()
    with col:
        with st.container(key=f"cardhover_{'sel_' if sel else ''}hero{h['row']}"):
            ui.md(f"""
            <div style="font-size:1.6rem;line-height:1">{h['icon']}</div>
            <div style="font-weight:750;color:{ui.TEXT};font-size:1rem;margin-top:8px;line-height:1.25">{h['name']}</div>
            <div style="color:{ui.MUTED};font-size:.74rem;margin-top:2px">{h['label']}</div>
            {ui.sparkline(recent)}
            <div style="font-size:.82rem;color:{ui.TEXT_2};line-height:1.5;min-height:3.8em">{h['why']}</div>
            """)
            if st.button("Following ✓" if sel else "Follow →", key=f"hero_{h['row']}", width="stretch",
                         type="primary" if sel else "secondary"):
                st.session_state["hero_row"] = h["row"]
                st.session_state["row"] = h["row"]
                st.rerun()

# ------------------------------------------------------------------ the story of one product
h = data.hero_of(cur) or heroes[0]
r = h["row"]
t0 = p.t0
ui.section("The story", f"{h['icon']} {h['name']}, from sales history to the shelf",
           f"{h['label']} · read the four steps from left to right, top to bottom.")


def step_card(key: str, n: int, title: str, fig: go.Figure, text: str, height: int = 250):
    with st.container(key=f"card_{key}"):
        ui.md(f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:2px">'
              f'<span style="width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:.8rem;'
              f'font-weight:800;color:white;background:linear-gradient(135deg,{ui.BLUE},{ui.BLUE_INK})">{n}</span>'
              f'<span style="font-weight:750;color:{ui.TEXT};font-size:1rem">{title}</span></div>')
        ui.show(fig, height, key=f"fig_{key}")
        ui.md(f'<div style="color:{ui.TEXT_2};font-size:.88rem;line-height:1.5;margin-top:-4px">{text}</div>')


wk_dates = p.dates[t0 + 1:t0 + 9]
demand = p.sales.iloc[r, t0 + 1:t0 + 9].to_numpy()
c1, c2 = st.columns(2, gap="medium")
with c1:   # 1 · history
    hx = p.dates[t0 - 51:t0 + 1]
    hs = p.sales.iloc[r, t0 - 51:t0 + 1]
    empty = ~p.in_stock.iloc[r, t0 - 51:t0 + 1].to_numpy()
    fig = go.Figure()
    fig.add_bar(x=hx, y=np.where(empty, np.nan, hs), name="Units sold", marker=dict(color=ui.BLUE, cornerradius=2),
                hovertemplate="%{x|%d %b %Y}: %{y:.0f} sold<extra></extra>")
    if empty.any():
        top = max(float(hs.max()), 1.0)
        fig.add_bar(x=hx[empty], y=np.full(empty.sum(), top), name="Shelf empty", marker=dict(color="rgba(235,104,52,.25)"),
                    hovertemplate="%{x|%d %b %Y}: shelf empty<extra></extra>")
    fig.update_layout(barmode="overlay", bargap=.15, yaxis=dict(title="Units"), hovermode="closest", margin=dict(t=30),
                      legend=dict(y=1.02, yanchor="bottom"))
    step_card("s1", 1, "It sold like this (last 52 weeks)", fig,
              f"About <b>{np.nanmean(np.where(empty, np.nan, hs)):.0f} units a week</b>"
              + (f"; the shelf was empty in <b>{empty.sum()} of 52 weeks</b>." if empty.any() else ", and it never ran out."))
with c2:   # 2 · forecast vs actual
    rounds = data.table("rounds.parquet")
    f1 = rounds[(rounds.model == model) & (rounds.row == r) & (rounds.h == 1) & (rounds.week <= 8)].sort_values("week")
    fig = go.Figure()
    fig.add_scatter(x=wk_dates, y=demand, name="What really sold", mode="lines+markers",
                    line=dict(color=ui.TEXT, width=2, dash="dot"), marker=dict(size=8))
    fig.add_scatter(x=wk_dates, y=f1.forecast, name="Our prediction (made the week before)", mode="lines+markers",
                    line=dict(color=ui.BLUE, width=2.5), marker=dict(size=8))
    fig.update_layout(yaxis=dict(title="Units"), hovermode="x unified", margin=dict(t=30), legend=dict(y=1.02, yanchor="bottom"))
    err = np.abs(f1.forecast.to_numpy() - demand)
    step_card("s2", 2, "We predicted this", fig,
              f"Over the 8 test weeks the prediction was off by <b>{err.mean():.0f} units a week</b> on average "
              f"({err.sum() / max(demand.sum(), 1):.0%} of what sold).")
c3, c4 = st.columns(2, gap="medium")
with c3:   # 3 · orders
    po, bo = res["orders"][r, :6], brow["orders"][r]
    labels = [f"Order {i}<br>(arrives wk {i + 2})" for i in range(1, 7)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=bo, name="Official method", marker=dict(color=ui.CONTEXT, cornerradius=3))
    fig.add_bar(x=labels, y=po, name="Our plan", marker=dict(color=ui.BLUE, cornerradius=3))
    fig.update_layout(barmode="group", yaxis=dict(title="Units ordered"), hovermode="x unified", margin=dict(t=30),
                      legend=dict(y=1.02, yanchor="bottom"))
    step_card("s3", 3, "So we ordered this", fig,
              f"Six weekly orders, <b>{po.sum():,.0f} units</b> in total, against {bo.sum():,.0f} for the official method. "
              "Each one lands on the shelf two weeks after it is placed.")
with c4:   # 4 · shelf result
    pr = res["per_row"]
    start, lost, endv = pr["start"][r], pr["lost"][r], pr["end"][r]
    wk = np.arange(1, 9)
    fig = go.Figure()
    fig.add_bar(x=wk, y=start, name="Stock on the shelf", marker=dict(color="#b7d3f6", cornerradius=3))
    fig.add_scatter(x=wk, y=demand, name="Customers wanted", mode="lines+markers", line=dict(color=ui.TEXT, width=2), marker=dict(size=7))
    miss = lost > 0
    fig.add_scatter(x=wk[miss], y=demand[miss], name="Ran out", mode="markers", marker=dict(size=13, color=ui.BAD, symbol="x"))
    fig.add_vrect(x0=.5, x1=2.5, fillcolor=ui.CONTEXT_FILL, line_width=0)
    fig.update_layout(xaxis=dict(title="Week", dtick=1), yaxis=dict(title="Units"), hovermode="x unified", margin=dict(t=30),
                      legend=dict(y=1.02, yanchor="bottom"))
    sc = slice(2, 8)
    hc, sc_ = data.costs().holding, data.costs().shortage
    cost_p = float((endv[sc] * hc + lost[sc] * sc_).sum())
    cost_b = float((brow["end"][r][sc] * hc + brow["lost"][r][sc] * sc_).sum())
    out_p, out_b = int((lost[sc] > 0).sum()), int((brow["lost"][r][sc] > 0).sum())
    verdict = ("<b>never ran out</b>" if out_p == 0 else f"ran out in <b>{out_p} of 6 weeks</b>")
    step_card("s4", 4, "And the shelf…", fig,
              f"In the 6 weeks our orders controlled (3–8), the shelf {verdict} (official method: {out_b}). "
              f"Cost for this product: <b>{ui.eur(cost_p, 2)}</b> vs {ui.eur(cost_b, 2)}. Weeks 1–2 (grey) were already decided."
              + ("" if cost_p <= cost_b else
                 f" <i>Here the official method did slightly better: no plan wins on every product. Added up over all "
                 f"{s['rows']}, ours saves {ui.eur(saved)}.</i>"))
st.page_link("views/demand.py", label=f"See {h['name']}'s full history  →")

# ------------------------------------------------------------------ the result for the whole chain
ui.section("The result", "All 599 store-products, eight real weeks, the official rules",
           "Each unit left on the shelf costs €0.20 per week; each sale missed because the shelf was empty costs €1. "
           "Weeks 1–2 were fixed before the first order (same for everyone).")
c = st.columns([3, 2], gap="medium")
with c[0]:
    with st.container(key="card_race"):
        wk = np.arange(3, 9)
        cum_p = res["weekly"].loc[3:8, "cost"].cumsum().to_numpy()
        cum_b = bench.loc[3:8, "cost"].cumsum().to_numpy()
        fig = go.Figure()
        fig.add_scatter(x=wk, y=cum_b, name="Official method", mode="lines+markers", line=dict(color=ui.ORANGE, width=2),
                        marker=dict(size=8), hovertemplate="Week %{x}<br>Official method: €%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=wk, y=cum_p, name="Our plan", mode="lines+markers", line=dict(color=ui.BLUE, width=2.5),
                        marker=dict(size=8), fill="tonexty", fillcolor="rgba(42,120,214,.07)",
                        hovertemplate="Week %{x}<br>Our plan: €%{y:,.0f}<extra></extra>")
        for y, name in ((cum_b[-1], f"Official {ui.eur(cum_b[-1])}"), (cum_p[-1], f"Our plan {ui.eur(cum_p[-1])}")):
            fig.add_annotation(x=8, y=y, text=name, showarrow=False, xanchor="left", xshift=10, font=dict(color=ui.TEXT, size=12))
        fig.update_layout(title=dict(text="Money spent so far (lower is better)"),
                          xaxis=dict(title="Competition week", tickvals=list(range(3, 9)), range=[2.7, 9.6]), yaxis=dict(tickprefix="€"),
                          hovermode="closest", showlegend=False, margin=dict(r=20))
        ui.show(fig, 340, key="race")
with c[1]:
    with st.container(key="card_rank"):
        rank = [(n, v) for n, v in s["ranking"]] + [("Our plan", res["score"])]
        rank.sort(key=lambda x: x[1])
        names = [n.replace("VN2 benchmark", "Official method") for n, _ in rank]
        vals = [v for _, v in rank]
        colors = [ui.BLUE if n == "Our plan" else (ui.ORANGE if n == "Official method" else ui.CONTEXT) for n in names]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color=colors, cornerradius=4),
                               text=[ui.eur(v) for v in vals], textposition="outside", cliponaxis=False,
                               textfont=dict(color=ui.TEXT_2), hovertemplate="%{y}: €%{x:,.1f}<extra></extra>"))
        fig.update_layout(title=dict(text="Against the real competition's top 5"),
                          xaxis=dict(range=[3000, max(vals) * 1.1], showgrid=True, tickprefix="€"),
                          yaxis=dict(autorange="reversed", showgrid=False), hovermode="closest", margin=dict(r=40))
        ui.show(fig, 340, key="rank")
ui.note("<b>Read with care:</b> the plan only used information available each week, and its settings were tuned on "
        "past weeks, never on these eight. But it was built after the competition, knowing which methods did well, so "
        "the comparison with the leaderboard carries some hindsight. 180 teams took part; 25 beat the official method.")

# ------------------------------------------------------------------ how it works
ui.section("How it works", "From sales history to an order, in four steps")
ui.insights([
    ("🧹", "1 · Clean the history", "Weeks that ended with an **empty shelf** say nothing about demand, so they are set "
     "aside. So are the weeks before a product was launched."),
    ("🔮", "2 · Predict the next weeks", "Ten methods, from *last week's sales* to **ARIMA** and **machine learning**, are "
     "tested on 30 past weeks they never saw."),
    ("🎲", "3 · Simulate what could happen", "1,000 possible futures per store-product show how much stock will be left when "
     "the next order arrives."),
    ("📦", "4 · Order the right amount", "Each order covers its arrival week with the **service level** that cost least "
     "when replayed on the past."),
])

ui.section("Explore", "Choose where to start")
cols = st.columns(3)
cards = [
    ("Demand", "views/demand.py", "See how each store-product sells, when shelves ran empty, and the yearly rhythm."),
    ("Stock planner", "views/planner.py", "Change the forecast method and service level, and replay the eight weeks live."),
    ("Order outlook", "views/outlook.py", "This week's recommended orders and the store-products at risk of running out."),
]
for col, (title, path, text) in zip(cols, cards):
    with col:
        with st.container(key=f"cardhover_{title.replace(' ', '_')}"):
            ui.md(f'<div style="font-weight:750;color:{ui.TEXT};font-size:1.05rem">{title}</div>'
                  f'<div style="color:{ui.TEXT_2};font-size:.9rem;line-height:1.55;margin:6px 0 12px">{text}</div>')
            st.page_link(path, label=f"Open {title}  →")
ui.note("Data: VN2 Inventory Planning Challenge (Vandeput, 2025, SupChains), real weekly sales of an anonymous retail "
        "chain, April 2021 – June 2024, CC BY 4.0. Costs and lead time are the competition's official rules.")
