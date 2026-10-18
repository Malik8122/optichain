import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui

ui.page_header(4, "Stock planner",
               "Choose how to forecast and how safe to play, then replay the eight VN2 weeks live with the official rules.")
s = data.summary()
p = data.panel()
bench = data.table("benchmark_weekly.csv").set_index("week")
tuning = data.table("tuning.csv")
models = list(ui.MODEL_NAMES)

# Widget state is dropped when the page is left, so the choice also lives in plain keys read by other pages
cur_model, cur_service = data.plan_choice()
st.session_state.setdefault("w_plan_model", cur_model)
st.session_state.setdefault("w_plan_service", cur_service)


def reset():
    st.session_state["w_plan_model"] = st.session_state["plan_model"] = s["plan_model"]
    st.session_state["w_plan_service"] = st.session_state["plan_service"] = s["plan_service"]


# ------------------------------------------------------------------ controls
with st.container(key="card_controls"):
    ui.md(f'<div style="font-weight:750;color:{ui.TEXT};margin-bottom:2px">Your plan</div>'
          f'<div style="color:{ui.MUTED};font-size:.86rem;margin-bottom:8px">The recommended settings had the lowest cost '
          'when replayed on 18 past weeks. Change them and the eight weeks are replayed instantly.</div>')
    k = st.columns([1.4, 1.6, 0.7], vertical_alignment="bottom")
    model = k[0].selectbox("Forecast method", models, key="w_plan_model",
                           format_func=lambda m: ui.MODEL_NAMES[m] + ("  ★ recommended" if m == s["plan_model"] else ""),
                           help="The prediction each order is based on.")
    service = k[1].slider("Service level: chance an order covers its week", 0.50, 0.97, step=0.01, key="w_plan_service",
                          help="Higher = more safety stock: fewer empty shelves but more stock left over. With €1 per lost sale "
                               "and €0.20 per unit left, the textbook value is 0.83; the best value on past weeks was "
                               f"{s['best_service'].get(model, 0.8):.2f} for this method.")
    k[2].button("Reset", on_click=reset, width="stretch", help="Back to the recommended plan")
st.session_state["plan_model"], st.session_state["plan_service"] = model, float(service)

with st.spinner("Replaying eight weeks with 1,000 possible futures per store-product…"):
    res = data.plan(model, service)
w = res["weekly"]
sc = w.loc[3:8]
delta = res["score"] - s["benchmark_score"]
fill = sc["sales"].sum() / (sc["sales"].sum() + sc["lost"].sum())
bfill = bench.loc[3:8, "sales"].sum() / (bench.loc[3:8, "sales"].sum() + bench.loc[3:8, "lost"].sum())
ui.kpis([
    {"label": "VN2 score (cost of weeks 3–8)", "value": ui.eur(res["score"]), "accent": True,
     "sub": f"{'▼' if delta < 0 else '▲'} {ui.eur(abs(delta))} vs benchmark", "tone": "good" if delta < 0 else "bad"},
    {"label": "Cost of leftover stock", "value": ui.eur(sc["holding"].sum()),
     "sub": f"benchmark {ui.eur(bench.loc[3:8, 'holding'].sum())}"},
    {"label": "Cost of empty shelves", "value": ui.eur(sc["shortage"].sum()),
     "sub": f"benchmark {ui.eur(bench.loc[3:8, 'shortage'].sum())}"},
    {"label": "Demand served", "value": f"{fill:.1%}", "sub": f"benchmark {bfill:.1%}"},
])

# ------------------------------------------------------------------ weekly costs
c = st.columns([3, 2], gap="medium")
with c[0]:
    with st.container(key="card_weekly"):
        wk = np.arange(1, 9)
        fig = go.Figure()
        for name, col, grp, y, base in (
            ("Our plan · leftover stock", "#86b6ef", "plan", w["holding"], None),
            ("Our plan · empty shelves", ui.BLUE, "plan", w["shortage"], w["holding"]),
            ("Benchmark · leftover stock", "#f5b393", "bench", bench["holding"], None),
            ("Benchmark · empty shelves", ui.ORANGE, "bench", bench["shortage"], bench["holding"]),
        ):
            fig.add_bar(x=wk, y=y, base=base, name=name, offsetgroup=grp, marker=dict(color=col, cornerradius=3,
                        line=dict(color="white", width=1)), hovertemplate=f"Week %{{x}}<br>{name}: €%{{y:,.0f}}<extra></extra>")
        fig.add_vrect(x0=0.5, x1=2.5, fillcolor=ui.CONTEXT_FILL, line_width=0, annotation_text="fixed before the first order",
                      annotation_position="top left", annotation_font=dict(color=ui.TEXT_2, size=11))
        fig.update_layout(title=dict(text="Cost per week: our plan (left) vs benchmark (right)", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(title="Competition week", dtick=1), yaxis=dict(tickprefix="€"),
                          barmode="group", bargroupgap=0.08, hovermode="closest", legend=dict(font=dict(size=11)))
        ui.show(fig, 390, key="weekly")
with c[1]:
    with st.container(key="card_curve"):
        comp = tuning[(tuning.policy == "competition") & (tuning.model == model)].sort_values("param")
        past = tuning[(tuning.policy == "newsvendor") & (tuning.model == model)].sort_values("param")
        fig = go.Figure()
        fig.add_scatter(x=past.param, y=past.cost / 15, name="18 past weeks (used to tune)", mode="lines+markers",
                        line=dict(color=ui.CONTEXT, width=2), marker=dict(size=6),
                        hovertemplate="service %{x:.2f}<br>past weeks: €%{y:,.0f} per week<extra></extra>")
        fig.add_scatter(x=comp.param, y=comp.cost / 6, name="The 8 VN2 weeks", mode="lines+markers",
                        line=dict(color=ui.BLUE, width=2.5), marker=dict(size=7),
                        hovertemplate="service %{x:.2f}<br>VN2 weeks: €%{y:,.0f} per week<extra></extra>")
        fig.add_scatter(x=[service], y=[res["score"] / 6], name="Your choice", mode="markers",
                        marker=dict(size=16, color=ui.BLUE, symbol="diamond", line=dict(color="white", width=2)),
                        hovertemplate="your choice %{x:.2f}<br>€%{y:,.0f} per week<extra></extra>")
        fig.update_layout(title=dict(text="The safety trade-off: cost per scored week", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(title="Service level"), yaxis=dict(title="€ per week", tickprefix="€", rangemode="normal"),
                          hovermode="closest", legend=dict(font=dict(size=11)))
        ui.show(fig, 390, key="curve")
ui.note("Too little safety and shelves run empty (€1 per missed sale); too much and stock piles up (€0.20 per unit per "
        "week). The curve's low point is the sweet spot. The grey curve is what the plan was tuned on; the blue curve shows "
        "the same choice held up on the eight unseen weeks.")

# ------------------------------------------------------------------ drill-down
ui.section("Drill down", "Follow one store-product week by week")
c = st.columns([2, 3])
with c[0]:
    row = data.row_picker("plan")
info = data.rows().iloc[row]
pr = res["per_row"]
d = pd.DataFrame({
    "week": np.arange(1, 9), "date": p.dates[p.t0 + 1:p.t0 + 9],
    "stock at start": pr["start"][row], "demand": p.sales.iloc[row, p.t0 + 1:p.t0 + 9].to_numpy(),
    "sold": pr["sales"][row], "missed": pr["lost"][row], "left at end": pr["end"][row],
    "order placed": [res["orders"][row, i] if i < 6 else 0 for i in range(8)],
})
d["arrives"] = [f"week {i + 3}" if v > 0 else "" for i, v in enumerate(d["order placed"])]
with st.container(key="card_drill"):
    fig = go.Figure()
    fig.add_bar(x=d.week, y=d["stock at start"], name="Stock on the shelf at the start", marker=dict(color="#b7d3f6", cornerradius=3),
                hovertemplate="Week %{x}<br>%{y:.0f} on the shelf<extra></extra>")
    fig.add_scatter(x=d.week, y=d.demand, name="Demand", mode="lines+markers", line=dict(color=ui.TEXT, width=2), marker=dict(size=8),
                    hovertemplate="Week %{x}<br>demand %{y:.0f}<extra></extra>")
    miss = d[d.missed > 0]
    fig.add_scatter(x=miss.week, y=miss.demand, name="Missed sales", mode="markers", marker=dict(size=14, color=ui.BAD, symbol="x"),
                    customdata=miss.missed, hovertemplate="Week %{x}<br>%{customdata:.0f} sales missed<extra></extra>")
    fig.update_layout(title=dict(text=f"{info['label']}: stock vs demand", font=dict(size=14, color=ui.TEXT)),
                      xaxis=dict(title="Competition week", dtick=1), yaxis=dict(title="Units"), hovermode="closest")
    ui.show(fig, 330, key="drill")
st.dataframe(d.drop(columns=["date"]), hide_index=True, width="stretch",
             column_config={c_: st.column_config.NumberColumn(format="%d") for c_ in
                            ["stock at start", "demand", "sold", "missed", "left at end", "order placed"]})

orders = pd.DataFrame(res["orders"][:, :6], columns=[f"Order {i} ({(p.week0 + pd.Timedelta(weeks=i)):%d %b})" for i in range(1, 7)])
orders.insert(0, "Product", p.index.get_level_values("Product"))
orders.insert(0, "Store", p.index.get_level_values("Store"))
st.download_button("Download the six orders (CSV)", orders.to_csv(index=False), file_name=f"orders_{model}_{service:.2f}.csv",
                   mime="text/csv", icon=":material/download:")
ui.next_step(4)
