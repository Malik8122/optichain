import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui

ui.page_header(5, "Order outlook",
               "The data ends on 3 June 2024. If the plan kept running, this is what to order now and where shelves may run empty.")
p = data.panel()
r = data.rows()
model, service = data.plan_choice()
with st.spinner("Simulating the plan…"):
    res = data.plan(model, service)
out = res["outlook"]
end = res["end"]
mu = res["mu9"]
wk_dates = [p.dates[-1] + pd.Timedelta(weeks=i) for i in (1, 2, 3)]

ui.md(f"""
<div class="oc-reco oc-anim" style="margin:6px 0 18px">
  <div class="oc-eyebrow">This week's order · placed before {wk_dates[0]:%d %b %Y}</div>
  <div class="big" style="margin-top:8px">Order <b>{res['next_order'].sum():,.0f} units</b> for
  <b>{(res['next_order'] > 0).sum()}</b> of {len(r)} store-products. They arrive in week 11 ({wk_dates[2]:%d %b}).</div>
  <div class="meta" style="margin-top:12px">Plan: {ui.MODEL_NAMES[model]} at service level {service:.2f}
  &nbsp;·&nbsp; change it in the Stock planner.</div>
</div>""")

HIGH, MID = 0.5, 0.2
ui.kpis([
    {"label": "On the shelf now", "value": f"{end.end.sum():,.0f}", "unit": "units",
     "sub": f"+ {end.t1.sum() + end.t2.sum():,.0f} already on the way"},
    {"label": "Expected demand, next 3 weeks", "value": f"{mu[:, :3].sum():,.0f}", "unit": "units"},
    {"label": "High risk next week", "value": str(int((out['p_stockout'][:, 0] > HIGH).sum())),
     "sub": "store-products, already decided", "tone": "bad"},
    {"label": "High risk in week 11", "value": str(int((out['p_stockout'][:, 2] > HIGH).sum())),
     "sub": "after this order arrives"},
])

# ------------------------------------------------------------------ risk bands
c = st.columns([2, 3], gap="medium")
with c[0]:
    with st.container(key="card_bands"):
        bands = []
        for i in range(3):
            pr = out["p_stockout"][:, i]
            bands.append([(pr <= MID).sum(), ((pr > MID) & (pr <= HIGH)).sum(), (pr > HIGH).sum()])
        bands = np.array(bands)
        labels = [f"Week {9 + i}<br>{d:%d %b}" for i, d in enumerate(wk_dates)]
        fig = go.Figure()
        for j, (name, col) in enumerate((("✓ Low (≤20%)", "#9ec5f4"), ("! Medium (20–50%)", "#eda100"), ("✕ High (>50%)", ui.BAD))):
            fig.add_bar(x=labels, y=bands[:, j], name=name, marker=dict(color=col, line=dict(color="white", width=2)),
                        text=bands[:, j], textposition="inside", textfont=dict(color="white" if j else ui.TEXT),
                        hovertemplate=f"%{{x}}<br>{name}: %{{y}} store-products<extra></extra>")
        fig.update_layout(title=dict(text="Chance of an empty shelf, by week", font=dict(size=14, color=ui.TEXT)),
                          barmode="stack", yaxis=dict(title="Store-products"), hovermode="closest", legend=dict(font=dict(size=11)))
        ui.show(fig, 360, key="bands")
with c[1]:
    with st.container(key="card_stores"):
        df = pd.DataFrame({"Store": r.Store, "order": res["next_order"], "risk": out["p_stockout"][:, 0]})
        g = df.groupby("Store").agg(order=("order", "sum"), risky=("risk", lambda x: (x > HIGH).sum())).sort_values("order", ascending=False).head(15)
        fig = go.Figure(go.Bar(x=[f"Store {i}" for i in g.index], y=g.order, marker=dict(color=ui.BLUE, cornerradius=4),
                               customdata=g.risky, text=g.order.astype(int), textposition="outside", cliponaxis=False,
                               textfont=dict(color=ui.TEXT_2),
                               hovertemplate="%{x}<br>%{y:,.0f} units to order<br>%{customdata} products at high risk next week<extra></extra>"))
        fig.update_layout(title=dict(text="The 15 stores ordering the most this week", font=dict(size=14, color=ui.TEXT)),
                          yaxis=dict(title="Units to order"), xaxis=dict(tickangle=-40), hovermode="closest")
        ui.show(fig, 360, key="stores")
ui.note("Weeks 9 and 10 are already decided: only the stock on the shelf and the orders on their way can serve them. "
        "This week's order protects week 11, so its risk sits near 1 − service level for most products.")

# ------------------------------------------------------------------ action list
ui.section("Action list", "Every store-product, riskiest first")
f = st.columns([1.2, 1.2, 1.6])
stores = f[0].multiselect("Stores", sorted(r.Store.unique()), placeholder="All stores", key="ol_stores")
pats = f[1].multiselect("Sales pattern", list(ui.PATTERN_NAMES), format_func=ui.PATTERN_NAMES.get, placeholder="All patterns",
                        key="ol_pats")
min_risk = f[2].slider("Minimum risk next week", 0, 100, 0, step=5, format="%d%%", key="ol_risk")
tab = pd.DataFrame({
    "Store": r.Store, "Product": r.Product, "Pattern": r.pattern.map(ui.PATTERN_NAMES),
    "On shelf": end.end, "Arriving wk 9": end.t1, "Arriving wk 10": end.t2, "Expected demand (3 wks)": mu[:, :3].sum(axis=1),
    "Order now": res["next_order"], "Risk wk 9": out["p_stockout"][:, 0], "Risk wk 10": out["p_stockout"][:, 1],
    "Risk wk 11": out["p_stockout"][:, 2],
})
mask = np.ones(len(tab), bool)
if stores:
    mask &= tab.Store.isin(stores).to_numpy()
if pats:
    mask &= r.pattern.isin(pats).to_numpy()
mask &= tab["Risk wk 9"].to_numpy() * 100 >= min_risk
view = tab[mask].sort_values(["Risk wk 9", "Risk wk 10"], ascending=False)
st.caption(f"{len(view)} of {len(tab)} store-products")
risk_col = lambda t: st.column_config.ProgressColumn(t, min_value=0, max_value=1, format="percent")
st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
    "On shelf": st.column_config.NumberColumn(format="%d"), "Arriving wk 9": st.column_config.NumberColumn(format="%d"),
    "Arriving wk 10": st.column_config.NumberColumn(format="%d"),
    "Expected demand (3 wks)": st.column_config.NumberColumn(format="%.1f"), "Order now": st.column_config.NumberColumn(format="%d"),
    "Risk wk 9": risk_col("Risk wk 9"), "Risk wk 10": risk_col("Risk wk 10"), "Risk wk 11": risk_col("Risk wk 11")})
st.download_button("Download the action list (CSV)", view.to_csv(index=False), file_name="order_outlook.csv", mime="text/csv",
                   icon=":material/download:")
ui.next_step(5)
