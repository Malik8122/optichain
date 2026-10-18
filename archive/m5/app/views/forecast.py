import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from optichain.data_pipeline import d_to_date

cfg = data.cfg()
t = data.tables()

ui.page_header(2, "Predict the next weeks",
               "Each method made its forecast without seeing what came next. Compare it with what actually sold.")
c = st.columns([2, 3])
with c[0]:
    item = data.product_picker("forecast")

fc_item = t["fc"][t["fc"]["item_id"] == item]
summ = t["summary"][t["summary"]["item_id"] == item].set_index("model")
models = [m for m in ui.MODEL_NAMES if m in set(fc_item["model"])]
# Rank by RMSE: same winners as RMSSE (used for model selection), but in units/day.
best = summ["rmse"].idxmin()

ctl = st.columns([3, 2, 3])
origin = ctl[0].segmented_control(
    "Forecast made on", cfg.forecast.origins, default=cfg.forecast.origins[-1],
    format_func=lambda o: f"{d_to_date(o):%b %d}", key="fc_origin") or cfg.forecast.origins[-1]
model = st.pills("Method", models, default=best, format_func=lambda m: ui.MODEL_NAMES[m], key="fc_model") or best
weeks = ctl[1].segmented_control("Look ahead", [1, 2, 4], default=4, format_func=lambda w: f"{w} wk",
                                 key="fc_weeks") or 4
H = weeks * 7

f = fc_item[(fc_item["model"] == model) & (fc_item["origin_d"] == origin) & (fc_item["h"] <= H)]
s = data.series(item)
actual = s.reindex(pd.to_datetime(f["date"])).to_numpy()
err = f["yhat"].to_numpy() - actual
mae = np.mean(np.abs(err))
tot_f, tot_a = f["yhat"].sum(), np.nansum(actual)
inside = np.mean((actual >= f["lo80"].to_numpy()) & (actual <= f["hi80"].to_numpy()))
start = d_to_date(origin + 1)

ui.kpis([
    {"label": "Typical daily error", "value": f"±{mae:.1f}", "unit": "units",
     "sub": f"{mae / max(np.mean(actual), 1e-9):.0%} of an average day's sales" if np.mean(actual) > 0 else "",
     "accent": True},
    {"label": f"Total over {weeks} week{'s' * (weeks > 1)}", "value": f"{tot_f:,.0f}", "unit": "forecast",
     "sub": f"vs {tot_a:,.0f} actually sold ({tot_f / tot_a - 1:+.0%})" if tot_a else "",
     "tone": "good" if tot_a and abs(tot_f / tot_a - 1) < 0.1 else ""},
    {"label": "Days inside the likely range", "value": f"{inside:.0%}", "sub": "The shaded band aims for 80%"},
    {"label": "Most accurate method here", "value": ui.MODEL_NAMES[best], "sub": "Averaged over all 4 test months",
     "tone": "good" if model == best else ""},
])

# ---------------------------------------------------------------- main chart
hist = s.loc[d_to_date(origin - 55):d_to_date(origin)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=f["date"], y=f["hi80"], line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=f["date"], y=f["lo80"], line=dict(width=0), fill="tonexty",
                         fillcolor=ui.BLUE_BAND, name="Likely range", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Before the forecast", mode="lines",
                         line=dict(color=ui.CONTEXT, width=1.6), hovertemplate="%{y:.0f} units"))
fig.add_trace(go.Scatter(x=f["date"], y=actual, name="What actually sold", mode="lines+markers",
                         line=dict(color=ui.TEXT_2, width=1.6), marker=dict(size=5), hovertemplate="%{y:.0f} sold"))
fig.add_trace(go.Scatter(x=f["date"], y=f["yhat"], name=f"Forecast · {ui.MODEL_NAMES[model]}", mode="lines",
                         line=dict(color=ui.BLUE, width=3), hovertemplate="%{y:.1f} forecast"))
fig.add_vline(x=d_to_date(origin), line=dict(color=ui.MUTED, dash="dot", width=1))
fig.add_annotation(x=d_to_date(origin), y=1, yref="paper", text="forecast made", showarrow=False,
                   xanchor="right", xshift=-6, font=dict(color=ui.MUTED, size=11))
fig.update_layout(yaxis_title="Units per day")
ui.show(fig, 430, key=f"fc_{item}_{model}_{origin}_{H}")
st.caption(f"{ui.MODEL_NAMES[model]}: {ui.MODEL_BLURBS.get(model, '')} Forecast starts {ui.fmt_date(start)}.")

# ---------------------------------------------------------------- leaderboard
ui.section("Method comparison", "Which method predicts this product best?",
           "Typical error size across all four test months, with big misses counting extra (RMSE). Shorter bars are better.")
lb = summ.loc[models, "rmse"].sort_values(ascending=False)
f2 = go.Figure(go.Bar(
    y=[ui.MODEL_NAMES[m] for m in lb.index], x=lb.values, orientation="h",
    marker=dict(color=[ui.BLUE if m == best else ui.CONTEXT for m in lb.index], cornerradius=6),
    text=[f"±{v:.2f} units/day" for v in lb.values], textposition="outside", textfont=dict(color=ui.TEXT_2),
    hovertemplate="%{y}: ±%{x:.2f} units/day<extra></extra>"))
f2.update_layout(hovermode="closest", showlegend=False, xaxis=dict(showgrid=True, gridcolor=ui.GRID,
                 range=[0, lb.max() * 1.35], title="Typical error size, units per day (RMSE)"), yaxis=dict(showgrid=False))
c1, c2 = st.columns([3, 2])
with c1:
    ui.show(f2, 60 + 56 * len(lb), key=f"lb_{item}")
with c2:
    worst = lb.idxmax()
    gain = 1 - lb[best] / lb[worst]
    ui.insights([("🏆", f"{ui.MODEL_NAMES[best]} wins",
                  f"It is {gain:.0%} more accurate than {ui.MODEL_NAMES[worst]} for this product. "
                  "The stock plan uses whichever method has done best so far.")])

with st.expander("For analysts: full accuracy table"):
    cols = ["mae", "rmse", "rmsse", "bias", "smape", "cov80", "cov95", "skill_vs_snaive"]
    tbl = summ[[c for c in cols if c in summ.columns]].rename(index=ui.MODEL_NAMES).sort_values("rmsse")
    st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    st.caption("Mean over 4 rolling 28-day windows. RMSSE is scaled by in-sample naive error (M5 metric); "
               "bias > 0 means over-forecasting; cov80/cov95 are interval coverage rates.")
ui.next_step(2)
