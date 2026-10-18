import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
import ui
from optichain import diagnostics as D
from optichain.data_pipeline import d_to_date

cfg = data.cfg()
t = data.tables()
LAST = max(cfg.forecast.origins)

ui.page_header(3, "Model check", "Can we trust the forecast? Checks on the sales pattern and on what the model got wrong.")
c = st.columns([2, 3])
with c[0]:
    item = data.product_picker("check")

idr = t["ident"]
rec = idr[(idr["item_id"] == item) & (idr["origin_d"] == LAST)].iloc[0]
rd = t["resdiag"]
sel = rd[(rd["item_id"] == item) & rd["selected"] & (rd["origin_d"] == LAST)].iloc[0]
fam = sel["model"]
g = t["garch"]
gd = g[(g["item_id"] == item) & (g["model"] == fam)].iloc[0]

# ---------------------------------------------------------------- verdict cards
bias_ok = sel["mean_p"] >= 0.05
acf_small = sel["max_abs_acf"] < 0.1
ui.kpis([
    {"label": "Model checked", "value": ui.MODEL_NAMES.get(fam, fam), "sub": f"{sel['spec']} · fitted up to {ui.fmt_date(d_to_date(LAST))}",
     "accent": True},
    {"label": "Systematic over/under-forecasting", "value": "None" if bias_ok else "Detected",
     "sub": f"Average error {sel['resid_mean']:+.2f} units/day", "tone": "good" if bias_ok else "bad"},
    {"label": "Pattern left in the errors", "value": "Negligible" if acf_small else "Some",
     "sub": f"Largest correlation {sel['max_abs_acf']:.2f} (0 = none, 1 = total)", "tone": "good" if acf_small else "bad"},
    {"label": "Volatility model (GARCH)", "value": "Not needed" if not gd["recommend_garch"] else "Used",
     "sub": ("No volatility clustering found" if "no ARCH" in str(gd["reason"])
             else "Too many zero-sales days for GARCH" if "zero-heavy" in str(gd["reason"])
             else "Tested: gain on held-out months too small"), "tone": "good" if not gd["recommend_garch"] else ""},
])

# ---------------------------------------------------------------- 1. pattern
verdict_text = {
    "stationary": "Sales move around a **stable level**, so no differencing is needed.",
    "unit_root": "The sales level **drifts** over time, so the model works with day-to-day *changes* instead (d = 1).",
    "conflict": "The level **shifts now and then**. The tests disagreed, so both options were tried and the more accurate one kept.",
    "inconclusive": "The tests could not tell whether the level is stable, so both options were tried.",
}
y = data.series_model(item).loc[:d_to_date(LAST)].iloc[-cfg.forecast.fit_window_days:]
res = D.stl_decompose(D.apply_transform(y, rec["transform"]))
fs, ft = rec["seasonal_strength"], rec["trend_strength"]
ui.section("Step 1", "How does this product's demand behave?",
           "Sales split into a slow trend, a repeating weekly pattern and day-to-day noise (STL decomposition, last 26 weeks).")
last = 182
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("Trend (slow movement)", "Weekly pattern", "Noise left over"))
fig.add_trace(go.Scatter(x=res.trend.index[-last:], y=res.trend.values[-last:], line=dict(color=ui.BLUE, width=2.4),
                         name="Trend", hovertemplate="%{y:.2f}"), 1, 1)
fig.add_trace(go.Scatter(x=res.seasonal.index[-last:], y=res.seasonal.values[-last:], line=dict(color=ui.BLUE, width=1.6),
                         name="Weekly", hovertemplate="%{y:+.2f}"), 2, 1)
fig.add_trace(go.Bar(x=res.resid.index[-last:], y=res.resid.values[-last:], marker_color=ui.CONTEXT, name="Noise",
                     hovertemplate="%{y:+.2f}"), 3, 1)
fig.update_layout(showlegend=False, hovermode="x unified")
for a in fig.layout.annotations:
    a.update(x=0, xanchor="left", font=dict(size=12, color=ui.TEXT_2))
ui.show(fig, 470, key=f"stl_{item}")
scale = " (on a log scale, because variability grows with the level)" if rec["transform"] == "log1p" else ""
ui.insights([
    ("📐", "Is the level stable?", verdict_text.get(rec["pre_d_verdict"], "") + scale),
    ("🗓️", f"Weekly pattern strength: {fs:.0%}",
     "Strong enough that the model uses weekly terms." if fs >= 0.2 else "Weak: the weekly rhythm explains little of the variation."),
    ("🎲", "Is there anything to predict?",
     "Sales are **essentially random** around their average, so no model can do much better than a good average."
     if rec["white_noise"] else "Yes: today's sales are related to recent days, which models can learn from."),
])

# ---------------------------------------------------------------- 2. ACF / PACF
ui.section("Step 2", "How strongly is today linked to previous days?",
           "Bars outside the shaded band are real links. Spikes at 7, 14, 21… days are the weekly rhythm.")
ident_rec = {"transform": rec["transform"], "d": int(rec["d"]), "D": int(rec["D"])}
ap = D.acf_pacf(D.modelling_series(y, ident_rec), 35)
lags = np.arange(1, 36)
fig2 = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Autocorrelation (ACF)", "Partial autocorrelation (PACF)"))
for col, key in [(1, "acf"), (2, "pacf")]:
    v = ap[key][1:36]
    fig2.add_trace(go.Bar(x=lags, y=v, marker_color=[ui.BLUE if abs(x) > ap["bound"] else ui.CONTEXT for x in v],
                          width=0.5, hovertemplate="lag %{x}: %{y:.2f}<extra></extra>"), 1, col)
    fig2.add_hrect(y0=-ap["bound"], y1=ap["bound"], fillcolor=ui.BLUE_BAND, line_width=0, row=1, col=col)
fig2.update_layout(showlegend=False, hovermode="closest")
fig2.update_xaxes(title_text="Days back", dtick=7)
for a in fig2.layout.annotations:
    a.update(font=dict(size=12, color=ui.TEXT_2))
ui.show(fig2, 300, key=f"acf_{item}")
ops = [f"d = {int(rec['d'])}", f"D = {int(rec['D'])}"] + (["log1p"] if rec["transform"] == "log1p" else [])
st.caption("Computed on the series as the model sees it (" + ", ".join(ops) + ").")

# ---------------------------------------------------------------- 3. residuals
ui.section("Step 3", "What did the model get wrong?",
           "Its one-day-ahead errors over the training period. Good errors look like random noise centred on zero.")
r = t["resid"]
r = r[(r["item_id"] == item) & (r["model"] == fam) & (r["origin_d"] == LAST)].set_index("date")["resid"]
fig3 = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25], horizontal_spacing=0.07,
                     subplot_titles=("Errors over time", "Pattern left in errors", "How errors are spread"))
fig3.add_trace(go.Bar(x=r.index, y=r.values, marker_color=ui.CONTEXT, hovertemplate="%{y:+.1f}<extra></extra>"), 1, 1)
fig3.add_trace(go.Scatter(x=r.index, y=r.rolling(28).mean(), line=dict(color=ui.BLUE, width=2),
                          hovertemplate="28-day avg %{y:+.2f}<extra></extra>"), 1, 1)
ra = D.acf_pacf(r, 21)
fig3.add_trace(go.Bar(x=np.arange(1, 22), y=ra["acf"][1:22], width=0.5,
                      marker_color=[ui.BLUE if abs(x) > ra["bound"] else ui.CONTEXT for x in ra["acf"][1:22]],
                      hovertemplate="lag %{x}: %{y:.2f}<extra></extra>"), 1, 2)
fig3.add_hrect(y0=-ra["bound"], y1=ra["bound"], fillcolor=ui.BLUE_BAND, line_width=0, row=1, col=2)
fig3.add_trace(go.Histogram(x=r.values, nbinsx=40, marker_color=ui.BLUE, hovertemplate="%{x}: %{y} days<extra></extra>"), 1, 3)
fig3.update_layout(showlegend=False, hovermode="closest", bargap=0.05)
fig3.update_yaxes(rangemode="normal", row=1, col=1)
fig3.update_yaxes(rangemode="normal", range=[-0.25, 0.25], row=1, col=2)
for a in fig3.layout.annotations:
    a.update(font=dict(size=12, color=ui.TEXT_2))
ui.show(fig3, 300, key=f"res_{item}")
kurt = sel["excess_kurtosis"]
ui.insights([
    ("🎯", "No systematic bias" if bias_ok else "Systematic bias",
     f"On average the model is off by {sel['resid_mean']:+.2f} units/day, statistically indistinguishable from zero."
     if bias_ok else f"On average the model is off by {sel['resid_mean']:+.2f} units/day."),
    ("🔁", "Leftover weekly pattern" if not sel["no_autocorrelation"] else "No leftover pattern",
     f"A formal test flags a little structure (p = {sel['lb_p_7']:.3f}), but the largest correlation is only "
     f"{sel['max_abs_acf']:.2f}, too small to matter in practice." if not sel["no_autocorrelation"]
     else f"The errors behave like random noise (largest correlation {sel['max_abs_acf']:.2f})."),
    ("📊", "Occasional big surprises" if kurt > 3 else "Errors roughly bell-shaped",
     f"Large spikes happen more often than a bell curve predicts (excess kurtosis {kurt:.1f}). "
     "Safety stock based on a bell curve may be too small." if kurt > 3
     else "Most errors are small and big misses are rare, close to a normal bell curve."),
])

# ---------------------------------------------------------------- 4. volatility
ui.section("Step 4", "Does uncertainty come in waves?",
           "If calm and turbulent periods alternated, a volatility (GARCH) model could widen or narrow the forecast range over time.")
roll = r.rolling(28).std()
fig4 = go.Figure(go.Scatter(x=roll.index, y=roll.values, line=dict(color=ui.ORANGE, width=2), fill="tozeroy",
                            fillcolor="rgba(235,104,52,.08)", hovertemplate="%{y:.2f}<extra>28-day error spread</extra>"))
fig4.update_layout(yaxis_title="Error spread (units)", hovermode="x unified")
c1, c2 = st.columns([3, 2])
with c1:
    ui.show(fig4, 260, key=f"vol_{item}")
with c2:
    if gd["recommend_garch"]:
        msg = "A volatility model improves the forecast ranges for this product."
    elif "zero-heavy" in str(gd["reason"]):
        msg = "Some clustering was detected, but this product has too many zero-sales days for a volatility model to be appropriate."
    elif sel["arch_effects"]:
        gain = gd.get("score_gain_pct", np.nan)
        msg = (f"Mild clustering is statistically detectable (ARCH-LM p = {sel['arch_lm_p']:.3f}), but a GARCH model "
               f"changed the forecast ranges by only {abs(gain):.1f}% on held-out months, so it is **not used**.")
    else:
        msg = f"No: big and small errors are not grouped in time (ARCH-LM p = {sel['arch_lm_p']:.2f}). A volatility model would add nothing."
    ui.insights([("🌊", "Verdict", msg)])

with st.expander("For analysts: identification and residual test details"):
    cols = ["origin_d", "transform", "level_adf_p", "level_kpss_p", "pre_d_verdict", "seasonal_strength", "trend_strength",
            "d", "D", "lb_p_7", "white_noise", "p_hint", "q_hint", "P_hint", "Q_hint"]
    st.markdown("**Identification (M4), per forecast origin**")
    st.dataframe(idr[idr["item_id"] == item][cols].set_index("origin_d").style.format(precision=3), width="stretch")
    st.markdown("**Residual diagnostics (M6), ETS and SARIMA at every origin**")
    cols = ["model", "origin_d", "spec", "resid_mean", "mean_p", "lb_p_7", "lb_p_14", "lb_p_21", "max_abs_acf",
            "excess_kurtosis", "jb_p", "arch_lm_p", "selected"]
    st.dataframe(rd[rd["item_id"] == item][cols].style.format(precision=3), hide_index=True, width="stretch")
    st.caption("ADF H₀: unit root · KPSS H₀: stationary (p reported within [0.01, 0.10]) · Ljung–Box H₀: no autocorrelation · "
               "ARCH-LM H₀: no ARCH effects. Full methodology in notebooks 02 and 04.")
ui.next_step(3)
