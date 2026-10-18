import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from optichain.planning import k_vector, nb_quantile

ui.page_header(2, "Predict the next weeks",
               "Nine forecasting methods, tested on 30 past weeks they never saw. Lower error means fewer empty shelves "
               "and less excess stock.")
p = data.panel()
s = data.summary()
plan_model, _ = data.plan_choice()
lb = data.table("leaderboard.csv")

FEATURE_NAMES = {
    "mean_4w": "Average, last 4 weeks", "mean_8w": "Average, last 8 weeks", "mean_26w": "Average, last 26 weeks",
    "mean_52w": "Average, last year", "std_13w": "Volatility, last 13 weeks", "zero_share_13w": "Share of zero weeks",
    "instock_share_13w": "Share of weeks in stock", "weeks_since_launch": "Weeks since launch", "log_level": "Sales level",
    "same_week_ly": "Same week last year", "same_week_2y": "Same week 2 years ago", "season_ratio": "Season: target vs now",
    "season_target": "Season of the target week", "week_of_year": "Week of the year", "horizon": "Weeks ahead",
    "ProductGroup": "Product group", "DepartmentGroup": "Department group", "StoreFormat": "Store format",
    **{f"demand_lag{i}": f"Demand {i} week{'s' if i > 1 else ''} ago" for i in range(1, 9)},
    **{f"sales_lag{i}": f"Sales {i} week{'s' if i > 1 else ''} ago" for i in range(1, 5)},
    **{f"instock_lag{i}": f"In stock {i} week{'s' if i > 1 else ''} ago" for i in range(1, 5)},
}
METRICS = {"Score %": "Forecast error (WAPE + |bias|)", "WAPE %": "Typical error (WAPE)", "MAE": "Average miss (units)",
           "Bias %": "Bias (+ = too high)"}

# ------------------------------------------------------------------ leaderboard
ui.section("Leaderboard", "Which method predicts best?",
           "Each method forecast the next 3 weeks at 30 past dates, using only what was known then. Errors are measured on "
           "weeks where the shelf did not run empty.")
metric = st.segmented_control("Measure", list(METRICS), default="Score %", format_func=lambda m: METRICS[m],
                              key="fc_metric") or "Score %"
c = st.columns([3, 2], gap="medium")
with c[0]:
    with st.container(key="card_lb"):
        d = (lb.reindex(lb[metric].abs().sort_values(ascending=False).index) if metric == "Bias %"
             else lb.sort_values(metric, ascending=False))
        colors = [ui.BLUE if m == plan_model else (ui.ORANGE if m == "seasonal_ma" else ui.CONTEXT) for m in d.model]
        fmt = (lambda v: f"{v:.2f}") if metric == "MAE" else (lambda v: f"{v:+.1f}%" if metric == "Bias %" else f"{v:.1f}%")
        fig = go.Figure(go.Bar(
            x=d[metric], y=d.model.map(ui.MODEL_NAMES), orientation="h", marker=dict(color=colors, cornerradius=4),
            text=[fmt(v) for v in d[metric]], textposition="outside", cliponaxis=False, textfont=dict(color=ui.TEXT_2),
            customdata=np.c_[d.model.map(ui.MODEL_FAMILY), d.model.map(ui.MODEL_BLURBS)],
            hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        lo = min(0, d[metric].min() * 1.25)
        fig.update_layout(title=dict(text=f"{METRICS[metric]} by method, lower is better" if metric != "Bias %"
                                     else "Bias by method, closer to 0 is better", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(range=[lo, d[metric].max() * 1.18], showgrid=True), yaxis=dict(showgrid=False),
                          hovermode="closest", margin=dict(r=40))
        ui.show(fig, 400, key="lb")
with c[1]:
    best = lb.iloc[0]
    bench = lb.set_index("model").loc["seasonal_ma"]
    ui.kpis([
        {"label": "Most accurate", "value": ui.MODEL_NAMES[best.model], "sub": f"error {best['Score %']:.1f}%", "accent": True},
        {"label": "VN2 benchmark forecast", "value": f"{bench['Score %']:.1f}", "unit": "%",
         "sub": f"{bench['Score %'] - best['Score %']:.1f} points worse"},
    ])
    ui.md(f"""<div class="oc-card oc-anim"><div class="oc-insight"><div class="ic">💡</div><div>
    <div class="t">Why one model for all stores wins</div>
    <div class="d">Most store-products sell only a few units a week, which is too little to learn from alone.
    LightGBM learns one set of rules from all 599 at once, so a slow seller borrows patterns from similar products
    and stores.</div></div></div></div>""")

# ------------------------------------------------------------------ series explorer
ui.section("Try it", "See the forecasts for one store-product",
           "Pick a date: the methods only see the weeks before it. The shaded band is the range demand should fall in "
           "8 times out of 10.")
bt = data.table("backtest.parquet")
rounds = data.table("rounds.parquet")
disp = data.table("dispersion.parquet")
origin_dates = sorted(bt.origin_date.unique())
round_dates = [p.week0 + pd.Timedelta(weeks=r - 1) for r in range(1, 10)]
all_dates = [pd.Timestamp(x) for x in origin_dates] + round_dates
c = st.columns([2, 2, 2])
with c[0]:
    row = data.row_picker("fc")
with c[1]:
    when = st.select_slider("Forecast made after the week of", options=all_dates, value=round_dates[0],
                            format_func=lambda d: f"{d:%d %b %Y}", key="fc_when")
with c[2]:
    pick = st.multiselect("Methods", list(ui.MODEL_NAMES), default=[plan_model, "seasonal_ma"],
                          format_func=lambda m: ui.MODEL_NAMES[m], key="fc_models", max_selections=3)
info = data.rows().iloc[row]
o = int(p.dates.get_loc(when))
if when >= p.week0:
    rr = o - p.t0 + 1
    f = rounds[(rounds["round"] == rr) & (rounds.row == row)]
    fc = {m: g.sort_values("h").forecast.to_numpy() for m, g in f.groupby("model")}
else:
    f = bt[(bt.origin == o) & (bt.row == row)]
    fc = {m: g.sort_values("h").forecast.to_numpy() for m, g in f.groupby("model")}

with st.container(key="card_try"):
    lo = max(0, o - 26)
    hx = p.dates[lo:o + 1]
    fig = go.Figure()
    ins = p.in_stock.iloc[row, lo:o + 1].to_numpy()
    fig.add_bar(x=hx, y=p.sales.iloc[row, lo:o + 1], name="Sales (known)",
                marker=dict(color=ui.CONTEXT, cornerradius=3,
                            pattern=dict(shape=["" if v else "/" for v in ins], fgcolor="white", size=5)),
                hovertemplate="%{x|%d %b}<br>%{y:.0f} sold<extra>known</extra>")
    if pick:
        H = len(fc[pick[0]])
        fx = p.dates[o + 1:o + 1 + H] if o + 1 + H <= len(p.dates) else pd.date_range(p.dates[o] + pd.Timedelta(weeks=1), periods=H, freq="7D")
        known = min(H, len(p.dates) - o - 1)
        if known > 0:
            fig.add_bar(x=fx[:known], y=p.sales.iloc[row, o + 1:o + 1 + known], name="What actually sold",
                        marker=dict(color="rgba(15,23,42,.0)", line=dict(color=ui.TEXT, width=1.5), cornerradius=3),
                        hovertemplate="%{x|%d %b}<br>%{y:.0f} actually sold<extra></extra>")
        m0 = pick[0]
        k = k_vector(disp, m0)[row]
        lo80 = nb_quantile(fc[m0], np.full(H, k), 0.1)
        hi80 = nb_quantile(fc[m0], np.full(H, k), 0.9)
        fig.add_scatter(x=np.r_[fx, fx[::-1]], y=np.r_[hi80, lo80[::-1]], fill="toself", fillcolor=ui.BLUE_BAND,
                        mode="lines", line=dict(width=0), name=f"80% range ({ui.MODEL_NAMES[m0]})", hoverinfo="skip")
        palette = [ui.BLUE, ui.ORANGE, ui.AQUA]
        for i, m in enumerate(pick):
            fig.add_scatter(x=np.r_[[p.dates[o]], fx], y=np.r_[[np.nan], fc[m]], name=ui.MODEL_NAMES[m], mode="lines+markers",
                            line=dict(color=palette[i], width=2.5), marker=dict(size=9),
                            hovertemplate=f"%{{x|%d %b}}<br>{ui.MODEL_NAMES[m]}: %{{y:.1f}}<extra></extra>")
    fig.add_vline(x=p.dates[o] + pd.Timedelta(days=3.5), line=dict(color=ui.TEXT_2, dash="dot"))
    fig.update_layout(title=dict(text=f"{info['label']}: forecast made after {when:%d %b %Y}", font=dict(size=14, color=ui.TEXT)),
                      yaxis=dict(title="Units"), barmode="overlay", hovermode="closest")
    ui.show(fig, 380, key="try")

# ------------------------------------------------------------------ breakdowns
ui.section("Details", "Where forecasting is hard")
c = st.columns(2, gap="medium")
with c[0]:
    with st.container(key="card_pat"):
        lp = data.table("leaderboard_pattern.csv")
        piv = lp.pivot(index="model", columns="pattern", values="Score %")[["smooth", "erratic", "intermittent", "lumpy"]]
        piv = piv.loc[lb.model]
        fig = go.Figure(go.Heatmap(
            z=piv.to_numpy(), x=[ui.PATTERN_NAMES[c] for c in piv.columns], y=piv.index.map(ui.MODEL_NAMES),
            colorscale=[[0, "#f4f8fe"], [0.5, "#6da7ec"], [1, "#184f95"]], xgap=2, ygap=2,
            text=np.round(piv.to_numpy()).astype(int), texttemplate="%{text}%", showscale=False,
            hovertemplate="%{y} · %{x}<br>error %{z:.1f}%<extra></extra>"))
        fig.update_layout(title=dict(text="Error by sales pattern (%)", font=dict(size=14, color=ui.TEXT)),
                          yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False, side="top", tickangle=0),
                          hovermode="closest", margin=dict(t=70))
        ui.show(fig, 400, key="pat")
with c[1]:
    with st.container(key="card_imp"):
        imp = data.table("importance.csv").head(12)[::-1]
        nice = imp.feature.map(lambda f: FEATURE_NAMES.get(f, f.replace("_", " ")))
        fig = go.Figure(go.Bar(x=100 * imp.gain, y=nice, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.1f}% of the model's gain<extra></extra>"))
        fig.update_layout(title=dict(text="What LightGBM relies on most", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(title="% of total gain", showgrid=True), yaxis=dict(showgrid=False), hovermode="closest")
        ui.show(fig, 400, key="imp")

with st.expander("Accuracy on the eight competition weeks (out-of-sample check)"):
    lc = data.table("leaderboard_competition.csv")
    st.dataframe(lc[["label", "MAE", "Bias %", "WAPE %", "Score %"]].rename(columns={"label": "Method"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ["MAE", "Bias %", "WAPE %", "Score %"]})
    st.caption("Same measures, on forecasts made each competition round (1 to 3 weeks ahead). Over only eight weeks the "
               "ranking reshuffles: the simple seasonal average does best here, yet the ensemble's orders still cost least.")

ui.next_step(2)
