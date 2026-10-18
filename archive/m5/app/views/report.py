import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from optichain.config import PROJECT_ROOT

cfg = data.cfg()
t = data.tables()
sa = cfg.simulation_assumptions
RULE, PLAN = "P0 baseline", "P1 forecast"

ui.page_header(5, "Report", "The results across all products, the assumptions behind them, and the data to download.")

pol = t["policy"].set_index(["item_id", "policy"])
summ = t["summary"].set_index(["item_id", "model"])
rows = []
for item in cfg.series.items:
    s = summ.loc[item]
    best = s["rmse"].idxmin()  # same winners as RMSSE, in units/day
    r, p = pol.loc[(item, RULE)], pol.loc[(item, PLAN)]
    rows.append({
        "Product": ui.item_short(item), "Item code": item, "Sales pattern": ui.CLASS_NAMES[data.demand_class(item)],
        "Best method": ui.MODEL_NAMES[best], "Typical error (units/day)": s.loc[best, "rmse"],
        "Served · simple rule": r["fill_rate"], "Served · forecast plan": p["fill_rate"],
        "Missed sales · rule": r["lost_units"], "Missed sales · plan": p["lost_units"],
        "Cost · rule ($)": r["total_cost"], "Cost · plan ($)": p["total_cost"],
        "Cost change": p["total_cost"] / r["total_cost"] - 1,
    })
df = pd.DataFrame(rows)

cheaper = int((df["Cost change"] < -0.005).sum())
cost_r, cost_p = df["Cost · rule ($)"].sum(), df["Cost · plan ($)"].sum()
lost_r, lost_p = df["Missed sales · rule"].sum(), df["Missed sales · plan"].sum()
ui.kpis([
    {"label": "Products where the plan was cheaper", "value": f"{cheaper} of {len(df)}", "accent": True,
     "sub": "Under the default assumptions"},
    {"label": "Total cost, all products", "value": f"${cost_p:,.0f}", "sub": f"vs ${cost_r:,.0f} with the simple rule "
     f"({cost_p / cost_r - 1:+.0%})", "tone": "good" if cost_p < cost_r else "bad"},
    {"label": "Sales missed, all products", "value": f"{lost_p:,.0f}", "unit": "units",
     "sub": f"vs {lost_r:,.0f} with the simple rule", "tone": "good" if lost_p < lost_r else "bad"},
    {"label": "Scenario", "value": f"{sa.lead_time_days}-day", "unit": "delivery",
     "sub": f"Aim: no stockout on {sa.target_csl:.0%} of days"},
])

ui.section("By product", "Simple rule vs forecast-driven plan", "Total cost over the replay period, per product.")
order = df.sort_values("Cost · rule ($)")
fig = go.Figure()
fig.add_trace(go.Bar(y=order["Product"], x=order["Cost · rule ($)"], name=ui.POLICY_NAMES[RULE], orientation="h",
                     marker=dict(color=ui.ORANGE, cornerradius=4), hovertemplate="$%{x:.2f}<extra>Simple rule</extra>"))
fig.add_trace(go.Bar(y=order["Product"], x=order["Cost · plan ($)"], name=ui.POLICY_NAMES[PLAN], orientation="h",
                     marker=dict(color=ui.BLUE, cornerradius=4), hovertemplate="$%{x:.2f}<extra>Forecast plan</extra>"))
fig.update_layout(barmode="group", bargroupgap=0.12, hovermode="closest",
                  xaxis=dict(title="Cost ($, hypothetical)", showgrid=True), yaxis=dict(showgrid=False))
ui.show(fig, 470, key="report_costs")

st.dataframe(
    df.drop(columns=["Item code", "Missed sales · rule", "Missed sales · plan"]), hide_index=True, width="stretch",
    column_config={
        "Typical error (units/day)": st.column_config.NumberColumn(format="±%.2f"),
        "Served · simple rule": st.column_config.ProgressColumn(format="percent", min_value=0.5, max_value=1),
        "Served · forecast plan": st.column_config.ProgressColumn(format="percent", min_value=0.5, max_value=1),
        "Cost · rule ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Cost · plan ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Cost change": st.column_config.NumberColumn(format="percent"),
    })

ui.section("Robustness", "Does the result hold under other assumptions?",
           "Total cost of the forecast-driven plan vs the simple rule, all 8 products, for other delivery times and "
           "service targets. Blue = forecast plan cheaper.")
sens = t["sensitivity"]
agg = sens.groupby(["lead_time", "target_csl", "policy"])["total_cost"].sum().unstack("policy")
chg = (agg[PLAN] / agg[RULE] - 1).unstack("target_csl")
lim = max(0.05, float(np.nanmax(np.abs(chg.values))))
hm = go.Figure(go.Heatmap(
    z=chg.values, x=[f"{c:.0%}" for c in chg.columns], y=[f"{int(i)} days" for i in chg.index],
    zmin=-lim, zmax=lim, colorscale=[[0, ui.DIVERGING[2]], [0.5, ui.DIVERGING[1]], [1, ui.DIVERGING[0]]],
    text=[[f"{v:+.0%}" for v in row] for row in chg.values], texttemplate="%{text}", textfont=dict(size=15, color=ui.TEXT),
    xgap=4, ygap=4, showscale=False, hovertemplate="Delivery %{y}, target %{x}: %{text} cost<extra></extra>"))
hm.update_layout(hovermode="closest", xaxis=dict(title="Aim to avoid running out on…", type="category"),
                 yaxis=dict(title="Delivery time", type="category", autorange="reversed"))
c1, c2 = st.columns([3, 2])
with c1:
    ui.show(hm, 300, key="report_robust")
with c2:
    n_better = int((chg.values < 0).sum())
    ui.insights([("🧪", f"Cheaper in {n_better} of {chg.size} scenarios",
                  f"Savings range from {-chg.values.min():.0%} down to {-chg.values.max():+.0%}. The default scenario "
                  "(4-day delivery, 95%) is among the most favourable, so quote the range, not just the headline.")])

final_path = PROJECT_ROOT / "reports" / "final_test" / "metrics.csv"
if final_path.exists():
    fm = pd.read_csv(final_path)
    sel = fm[fm["selected"]]
    finv = pd.read_csv(PROJECT_ROOT / "reports" / "final_test" / "inventory.csv").groupby("policy")["total_cost"].sum()
    ui.section("Final check", "Does it hold on a month nobody had seen?",
               "Before loading 25 Apr – 22 May 2016, the method for each product was written down in advance. Then it was tested once.")
    ui.kpis([
        {"label": "Forecast accuracy on the unseen month", "value": f"{sel['rmsse'].mean():.3f}", "unit": "RMSSE",
         "sub": f"vs {sel['backtest_rmsse'].mean():.3f} in testing: no sign of overfitting", "tone": "good", "accent": True},
        {"label": "Best methods on the unseen month", "value": "SARIMA & ETS", "sub": "Same leaders as before"},
        {"label": "Forecast plan cost (4 weeks)", "value": f"${finv['P1 forecast']:.0f}",
         "sub": f"vs ${finv['P0 baseline']:.0f} simple rule, mostly from one product", "tone": "good"},
    ])

c1, c2 = st.columns(2)
with c1:
    ui.section("Assumptions", "What the replay assumes")
    ui.md(f"""<div class="oc-card oc-anim"><ul style="margin:0;padding-left:18px;line-height:1.9">
      <li>Supplier delivery time: <b>{sa.lead_time_days} days</b></li>
      <li>Each order adds <b>{sa.cover_days} extra days</b> of stock</li>
      <li>Aim to avoid running out on <b>{sa.target_csl:.0%}</b> of days</li>
      <li>Storage cost: <b>{sa.holding_rate_annual:.0%}</b> of price per year</li>
      <li>Missed sale penalty: <b>{sa.shortage_penalty_rate:.0%}</b> of price · delivery cost <b>${sa.order_cost:.2f}</b></li>
      <li>Stock reviewed daily; missed sales are lost, not delayed</li>
    </ul></div>""")
with c2:
    ui.section("Limitations", "Read results with care")
    ui.md("""<div class="oc-card oc-anim"><ul style="margin:0;padding-left:18px;line-height:1.9">
      <li>Inventory, delivery times and costs are <b>hypothetical</b>. The data contains sales only.</li>
      <li>Recorded sales may already be reduced by real stockouts.</li>
      <li>Eight products from one US store: results are illustrative.</li>
      <li>Price and event links are associations, not proven causes.</li>
      <li>Safety stock is sized from in-sample errors; real errors were 18–26% larger for four products.</li>
      <li>Forecasts are refreshed monthly; daily re-forecasting would likely help.</li>
    </ul></div>""")

ui.section("Downloads", "Take the results with you")
item = data.current_item()
sim = data.simulate(item, sa.lead_time_days, sa.cover_days, sa.target_csl, sa.holding_rate_annual,
                    sa.shortage_penalty_rate, sa.order_cost)
d = st.columns(4)
d[0].download_button("Summary table (CSV)", df.to_csv(index=False), "optichain_summary.csv", "text/csv", width="stretch")
d[1].download_button("All forecasts (CSV)", t["fc"].to_csv(index=False), "optichain_forecasts.csv", "text/csv",
                     width="stretch")
d[2].download_button("Accuracy by method (CSV)", t["summary"].to_csv(index=False), "optichain_accuracy.csv",
                     "text/csv", width="stretch")
d[3].download_button(f"Daily stock plan · {ui.item_short(item)}", sim["trace"].to_csv(index=False),
                     f"optichain_stock_plan_{item}.csv", "text/csv", width="stretch")
