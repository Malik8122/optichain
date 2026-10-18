import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from optichain.data_pipeline import d_to_date

cfg = data.cfg()
sa = cfg.simulation_assumptions
RULE, PLAN, SMART, IDEAL = "P0 baseline", "P1 forecast", "P2 empirical", "Perfect foresight"

ui.page_header(4, "Plan stock", "Turn the forecast into reorder decisions, then replay Jan–Apr 2016 to see how it would have done.")
c = st.columns([2, 3])
with c[0]:
    item = data.product_picker("stock")


def delta_card(label, plan, rule, fmt, unit="", better="lower", accent=False):
    diff = plan - rule
    good = diff < 0 if better == "lower" else diff > 0
    same = abs(diff) < 1e-9
    arrow = "▲" if diff > 0 else "▼"
    sub = "Same as the simple rule" if same else f"{arrow} {fmt(abs(diff))} vs simple rule ({fmt(rule)})"
    return {"label": label, "value": fmt(plan), "unit": unit, "sub": sub,
            "tone": "" if same else ("good" if good else "bad"), "accent": accent}


@st.fragment
def planner():
    limit = data.lookahead_limit()
    with st.container(key="card_controls"):
        ui.md(f'<div style="font-weight:750;color:{ui.TEXT};margin-bottom:4px">Your scenario</div>'
              f'<div style="color:{ui.MUTED};font-size:.85rem;margin-bottom:6px">These are assumptions, not Walmart data. '
              'Move them to see how the plan responds.</div>')
        k = st.columns([1, 1, 1, 0.8])
        L = k[0].slider("Supplier delivery time (days)", 1, limit - 1, sa.lead_time_days, key="sp_L",
                        help="Days between placing an order and the stock arriving.")
        cover = k[1].slider("Extra days of stock per order", 1, limit - L, min(sa.cover_days, limit - L), key="sp_cover",
                            help="Bigger orders mean fewer deliveries but more stock sitting on the shelf.")
        csl = k[2].select_slider("Aim to avoid running out on", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
                                 value=sa.target_csl, format_func=lambda v: f"{v:.0%} of days", key="sp_csl")
        with k[3].popover("Cost assumptions", width="stretch"):
            hold = st.slider("Storage cost (% of price per year)", 5, 60, int(sa.holding_rate_annual * 100), key="sp_h") / 100
            short = st.slider("Cost of a missed sale (% of price)", 0, 200, int(sa.shortage_penalty_rate * 100), key="sp_p") / 100
            kcost = st.number_input("Cost per delivery ($)", 0.0, 100.0, float(sa.order_cost), 0.5, key="sp_k")

    res = data.simulate(item, L, cover, csl, hold, short, kcost)
    pm = res["metrics"].set_index("policy")
    tr = res["trace"]
    rule, plan = pm.loc[RULE], pm.loc[PLAN]

    # ------------------------------------------------------------ recommendation
    last = tr[tr["policy"] == PLAN].iloc[-1]
    today = last["date"]
    fc_next = data.tables()["fc"]
    fc_next = fc_next[(fc_next["item_id"] == item) & (fc_next["origin_d"] == max(cfg.forecast.origins))]
    model_now = res["model_choice"].iloc[-1]["model"]
    next_week = fc_next[(fc_next["model"] == model_now) & (fc_next["d"] > last["d"]) & (fc_next["d"] <= last["d"] + 7)]["yhat"].sum()
    if last["order_qty"] > 0:
        action = f'<span class="oc-badge">Order today</span>&nbsp; Place an order for <b>{last["order_qty"]:.0f} units</b>.'
    else:
        action = (f'<span class="oc-badge good">No order needed today</span>&nbsp; '
                  f'{last["on_hand"]:.0f} units on hand, {last["on_order"]:.0f} already on the way.')
    ui.md(f"""
    <div class="oc-reco oc-anim" style="margin-top:16px">
      <div class="oc-eyebrow">Recommendation · as of {ui.fmt_date(today)}</div>
      <div class="big" style="margin-top:8px">Reorder when stock falls to <b>{np.ceil(last['rop']):.0f} units</b>,
      then order enough to reach <b>{np.ceil(last['order_up_to']):.0f} units</b>.</div>
      <div class="meta" style="margin-top:12px">{action}
      &nbsp;Expected demand over the next 7 days: <b>{next_week:.0f} units</b>
      (forecast method: {ui.MODEL_NAMES.get(model_now, model_now)}).</div>
    </div>""")

    # ------------------------------------------------------------ results
    first, last_d = d_to_date(min(cfg.forecast.origins) + 1 + cfg.simulation.warmup_days), today
    ui.section("Replay", f"What would have happened, {first:%b %d} – {last_d:%b %d, %Y}",
               "Both approaches start with the same stock, face the same real sales and the same assumptions. "
               "The only difference is how they decide when and how much to order.")
    pct = lambda v: f"{v:.1%}"
    num = lambda v: f"{v:,.0f}"
    usd = lambda v: f"${v:,.2f}"
    ui.kpis([
        delta_card("Sales served from stock", plan["fill_rate"], rule["fill_rate"], pct, better="higher", accent=True),
        delta_card("Sales missed (units)", plan["lost_units"], rule["lost_units"], num),
        delta_card("Average stock on shelf", plan["avg_on_hand"], rule["avg_on_hand"], lambda v: f"{v:,.1f}", "units"),
        delta_card("Total cost", plan["total_cost"], rule["total_cost"], usd),
    ])

    # ------------------------------------------------------------ animated replay
    names = {RULE: ui.POLICY_NAMES[RULE], PLAN: ui.POLICY_NAMES[PLAN]}
    a = tr[tr["policy"] == RULE].reset_index(drop=True)
    b = tr[tr["policy"] == PLAN].reset_index(drop=True)
    x = a["date"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a["demand"], name="Daily sales", marker=dict(color=ui.DEMAND_BAR),
                         hovertemplate="%{y:.0f} sold<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=a["on_hand"], name=names[RULE], mode="lines",
                             line=dict(color=ui.ORANGE, width=2.4, shape="hv"), hovertemplate="%{y:.0f} on shelf"))
    fig.add_trace(go.Scatter(x=x, y=b["on_hand"], name=names[PLAN], mode="lines",
                             line=dict(color=ui.BLUE, width=2.8, shape="hv"), hovertemplate="%{y:.0f} on shelf"))
    for df_, col, nm in [(a, ui.ORANGE, RULE), (b, ui.BLUE, PLAN)]:
        so = df_[df_["lost"] > 0]
        fig.add_trace(go.Scatter(x=so["date"], y=np.zeros(len(so)), mode="markers", name=f"Ran out · {names[nm]}",
                                 marker=dict(color=col, size=11, symbol="x-thin", line=dict(width=3, color=col)),
                                 hovertemplate="%{customdata:.0f} sales missed<extra></extra>", customdata=so["lost"]))
    step = 2
    frames = [go.Frame(data=[go.Scatter(x=x[:k], y=a["on_hand"][:k]), go.Scatter(x=x[:k], y=b["on_hand"][:k])],
                       traces=[1, 2], name=str(k)) for k in range(2, len(x) + step, step)]
    fig.frames = frames
    ymax = max(a["on_hand"].max(), b["on_hand"].max(), a["demand"].max()) * 1.12
    fig.update_layout(
        yaxis=dict(title="Units", range=[-ymax * 0.04, ymax]),
        xaxis=dict(range=[x.iloc[0] - pd.Timedelta(days=1), x.iloc[-1] + pd.Timedelta(days=1)]),
        updatemenus=[dict(type="buttons", showactive=False, x=1, xanchor="right", y=1.16, yanchor="top",
                          bgcolor=ui.CARD, bordercolor=ui.AXIS, font=dict(color=ui.BLUE_INK),
                          buttons=[dict(label="▶  Replay", method="animate",
                                        args=[None, dict(frame=dict(duration=35, redraw=False),
                                                         transition=dict(duration=0), fromcurrent=False, mode="immediate")])])],
    )
    for o in cfg.forecast.origins[1:]:
        fig.add_vline(x=d_to_date(o), line=dict(color=ui.AXIS, dash="dot", width=1))
    ui.section("Stock on the shelf", "Day by day",
               "Press Replay to watch both approaches play out. ✕ marks days when the shelf ran empty. "
               "Dotted lines show when a fresh forecast arrived.")
    ui.show(fig, 440, key=f"replay_{item}_{L}_{cover}_{csl}")

    # ------------------------------------------------------------ costs + what-if
    c1, c2 = st.columns(2)
    with c1:
        ui.section("Where the money goes", "Cost breakdown", "Storage, deliveries and the penalty for missed sales.")
        pols = [p for p in (IDEAL, SMART, PLAN, RULE) if p in pm.index]
        parts = [("holding_cost", "Storage"), ("ordering_cost", "Deliveries"), ("shortage_cost", "Missed sales")]
        fc_ = go.Figure()
        for (col, lab), color in zip(parts, ui.COST_STEPS):
            fc_.add_trace(go.Bar(y=[ui.POLICY_NAMES[p] for p in pols], x=[pm.loc[p, col] for p in pols], name=lab,
                                 orientation="h", marker=dict(color=color, line=dict(color=ui.CARD, width=2)),
                                 hovertemplate=f"{lab}: $%{{x:.2f}}<extra></extra>"))
        for p in pols:
            fc_.add_annotation(x=pm.loc[p, "total_cost"], y=ui.POLICY_NAMES[p], text=f"  ${pm.loc[p, 'total_cost']:.0f}",
                               showarrow=False, xanchor="left", font=dict(color=ui.TEXT, size=13))
        fc_.update_layout(barmode="stack", hovermode="closest", legend=dict(traceorder="normal"), xaxis=dict(title="Cost over the period ($)", showgrid=True,
                          range=[0, pm.loc[pols, "total_cost"].max() * 1.18]), yaxis=dict(showgrid=False))
        ui.show(fc_, 330, key=f"cost_{item}_{L}_{cover}_{csl}_{hold}_{short}_{kcost}")
        st.caption("Perfect hindsight uses the real future sales as its forecast. It is the best fill rate "
                   "possible, but not always the cheapest under this simple reorder rule.")
    with c2:
        sens = data.tables()["sensitivity"]
        sens = sens[sens["item_id"] == item]
        p0 = sens[sens["policy"] == RULE].set_index(["lead_time", "target_csl"])["total_cost"]
        p1 = sens[sens["policy"] == PLAN].set_index(["lead_time", "target_csl"])["total_cost"]
        sav = ((p0 - p1) / p0).unstack("target_csl")
        lim = max(abs(np.nanmin(sav.values)), abs(np.nanmax(sav.values)), 0.05)
        ui.section("What if?", "Savings across scenarios",
                   "How much cheaper the forecast-driven plan is than the simple rule, using the default cost assumptions.")
        hm = go.Figure(go.Heatmap(
            z=sav.values, x=[f"{v:.0%}" for v in sav.columns], y=[f"{int(v)} days" for v in sav.index],
            zmin=-lim, zmax=lim, colorscale=[[0, ui.DIVERGING[0]], [0.5, ui.DIVERGING[1]], [1, ui.DIVERGING[2]]],
            text=[[f"{v:+.0%}" for v in row] for row in sav.values], texttemplate="%{text}",
            textfont=dict(color=ui.TEXT, size=14), xgap=4, ygap=4, showscale=False,
            hovertemplate="Delivery %{y}, aim %{x}: %{text} cost vs simple rule<extra></extra>"))
        hm.update_layout(hovermode="closest",
                         xaxis=dict(title="Aim to avoid running out on…", type="category", showgrid=False),
                         yaxis=dict(title="Delivery time", type="category", autorange="reversed", showgrid=False))
        ui.show(hm, 300, key=f"sens_{item}")
        st.caption("Blue = forecast plan cheaper · red = simple rule cheaper.")

    with st.expander("For analysts: every policy, and how the rules work"):
        show_cols = ["fill_rate", "cycle_service", "stockout_days", "lost_units", "avg_on_hand", "peak_on_hand",
                     "n_orders", "avg_order_qty", "holding_cost", "ordering_cost", "shortage_cost", "total_cost"]
        nice = pm[show_cols].rename(index=lambda p: ui.POLICY_NAMES.get(p, p.replace("P1 [", "Plan using ").rstrip("]")))
        st.dataframe(nice.style.format("{:.3f}"), width="stretch")
        ch = res["model_choice"].assign(
            forecast_from=lambda d: d["origin_d"].map(lambda o: ui.fmt_date(d_to_date(o + 1))),
            method=lambda d: d["model"].map(ui.MODEL_NAMES))[["forecast_from", "method", "basis"]]
        st.dataframe(ch, hide_index=True, width="stretch")
        st.markdown(
            f"- **Simple rule:** reorder point = average daily sales × delivery time + safety buffer. Average and "
            f"spread come from the last {cfg.simulation.baseline_window} days and are refreshed monthly.\n"
            "- **Forecast-driven plan:** reorder point = forecast sales over the delivery time + safety buffer. "
            "The buffer is sized from the method's own past errors.\n"
            "- Both approaches review stock daily, lose sales when the shelf is empty, and receive orders after the "
            "delivery time. The first week is a warm-up and is not scored.")


planner()
ui.note("All stock levels, delivery times and costs are hypothetical. The M5 dataset records sales, not inventory, "
        "and recorded sales may already be capped by real stockouts.")
ui.next_step(4)
