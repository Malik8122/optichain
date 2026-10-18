import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import data
import ui
from optichain import timeseries as TS

ui.page_header(3, "Model check",
               "The classical time-series checks: is the series stable, seasonal, which model fits, and is what the "
               "model leaves behind just random noise?")
p = data.panel()
net = json.loads(data.art("ts_network.json").read_text())


def tick(ok: bool) -> str:
    return "✓" if ok else "✕"


# ------------------------------------------------------------------ data for one view
@st.cache_data(show_spinner=False)
def network_view() -> dict:
    tot = p.sales.iloc[:, :p.t0 + 1].sum(axis=0)
    stat = pd.DataFrame(net["stationarity"]).set_index("series")
    for c in ("ADF statistic", "ADF p-value", "KPSS p-value"):
        stat[c] = stat[c].astype(float)
    dec = data.table("ts_decomposition.parquet").set_index("date")
    res = data.table("ts_residuals.parquet")
    g = net["garch_tests"]
    keys = list(g)
    return {
        "name": "Whole network (all 599 store-products added up)", "y": tot.to_numpy(), "dates": tot.index,
        "stationarity": stat, "d": net["d"], "decomposition": dec, "strength": net["strength"],
        "acf_pacf": TS.acf_pacf(tot.to_numpy(), 56), "identify": net["identify"],
        "aic_grid": pd.DataFrame(net["aic"]), "best": net["best"], "best_bic": net["best_bic"],
        "resid": res["resid"].to_numpy(), "resid_dates": pd.to_datetime(res["date"]),
        "resid_tests": g[keys[0]], "resid_acf": TS.acf_pacf(res["resid"].to_numpy(), 30),
        "garch": {"params": net["garch_params"], "std_tests": g[keys[1]], "sq_tests": g[keys[2]],
                  "volatility": res["volatility"].to_numpy()},
        "holdout": data.table("ts_holdout.parquet"),
    }


@st.cache_data(show_spinner=False)
def row_view(row: int) -> dict:
    y = data.panel().sales.iloc[row, :p.t0 + 1].where(p.in_stock.iloc[row, :p.t0 + 1])
    y = y.where(p.sales.iloc[row, :p.t0 + 1].cumsum() > 0).to_numpy()
    a = TS.analyze(y, p.dates[:p.t0 + 1], period=52, seasonal=False)
    g = a["garch"]
    return {
        "name": data.rows().iloc[row]["label"], "y": TS._prep(y)[0], "dates": p.dates[:p.t0 + 1][-a["n"]:],
        "stationarity": a["stationarity"], "d": a["d"], "decomposition": a.get("decomposition"),
        "strength": a.get("strength"), "acf_pacf": a["acf_pacf"], "identify": a["identify"],
        "aic_grid": a["aic_grid"][["model", "family", "AIC", "BIC", "ΔAIC", "parameters"]], "best": a["best"]["model"],
        "best_bic": a["aic_grid"].sort_values("BIC").iloc[0]["model"], "resid": a["resid"], "resid_dates": a["resid_dates"],
        "resid_tests": a["resid_tests"], "resid_acf": a["resid_acf"],
        "garch": {k: g[k] for k in ("std_tests", "sq_tests", "volatility")} | {"params": dict(g["params"])} if "params" in g else None,
        "missing": a["missing"],
    }


# ------------------------------------------------------------------ choose the series
mode = st.segmented_control("Analyse", ["Whole network", "One store-product"], default="Whole network", key="mc_mode") \
    or "Whole network"
if mode == "One store-product":
    c = st.columns([2, 3])
    with c[0]:
        row = data.row_picker("mc")
    with st.spinner("Running the tests and fitting the models for this store-product…"):
        try:
            v = row_view(row)
        except Exception as e:  # noqa: BLE001
            st.warning(f"This store-product has too little clean history for the full analysis ({e}). Pick another one.")
            st.stop()
    arima_rows = data.table("arima_rows.parquet")
    used = arima_rows.iloc[row]
    ui.note(f"In the forecasting race this store-product uses <b>{used['family']}</b> {used['order'] if used['order'] != 'None' else ''} "
            f"on its season-adjusted demand (selected by AIC). {v['missing']} out-of-stock weeks are treated as missing.")
else:
    v = network_view()

stat = v["stationarity"]
orig = stat.iloc[0]
rt = v["resid_tests"]
gt = v["garch"]["std_tests"] if v["garch"] else None
clean = bool(rt["white noise"]) or (gt is not None and bool(gt["white noise"]))
ui.kpis([
    {"label": "Stable over time? (ADF)", "value": f"{tick(orig['ADF p-value'] < .05)} {'Yes' if orig['ADF p-value'] < .05 else 'No'}",
     "sub": f"ADF p = {orig['ADF p-value']:.3f} · differencing d = {v['d']}"},
    {"label": "Yearly season", "value": f"{v['strength']['seasonal strength']:.2f}" if v["strength"] else "n/a",
     "sub": "strength, 0 = none, 1 = dominant" if v["strength"] else "needs 2 full years of history"},
    {"label": "Best model (AIC)", "value": v["best"], "sub": f"BIC prefers {v['best_bic']}", "accent": True},
    {"label": "Leftover is random noise?", "value": f"{tick(clean)} {'Yes' if clean else 'No'}",
     "sub": "Ljung-Box" + (" after GARCH" if (not rt["white noise"] and clean) else ""), "tone": "good" if clean else "bad"},
])

tabs = st.tabs(["1 · Stationarity", "2 · Season & trend", "3 · ACF / PACF", "4 · Model choice", "5 · Residuals & GARCH"]
               + (["6 · Forecast check"] if "holdout" in v else []))

# ------------------------------------------------------------------ 1 stationarity
with tabs[0]:
    y = pd.Series(v["y"], index=v["dates"])
    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y, name="Weekly units", line=dict(color=ui.BLUE, width=1.6))
    fig.add_scatter(x=y.index, y=y.rolling(13, min_periods=4).mean(), name="13-week rolling mean", line=dict(color=ui.TEXT, width=2))
    fig.add_scatter(x=y.index, y=y.rolling(13, min_periods=4).std(), name="13-week rolling std", line=dict(color=ui.ORANGE, width=2, dash="dot"))
    fig.update_layout(title=dict(text=f"{v['name']}: does the level or the spread drift?"), yaxis=dict(title="Units"))
    ui.show(fig, 340, key="mc_stat")
    tab = stat[["ADF p-value", "ADF says", "KPSS p-value", "KPSS says", "verdict"]].copy()
    tab.index.name = "transformation"
    st.dataframe(tab.reset_index(), hide_index=True, width="stretch", column_config={
        "ADF p-value": st.column_config.NumberColumn(format="%.4f"), "KPSS p-value": st.column_config.NumberColumn(format="%.3f")})
    ui.note("<b>ADF</b> asks “does the series wander like a random walk?” A p-value below 0.05 says no: it is "
            "<b>stationary</b> (stable mean and spread), which ARMA models need. <b>KPSS</b> asks the opposite question as "
            "a cross-check. If the original fails, a log or a difference (week minus the week before) usually fixes it.")

# ------------------------------------------------------------------ 2 decomposition
with tabs[1]:
    dec = v["decomposition"]
    if dec is None:
        st.info("A 52-week decomposition needs at least two full years of clean history; this store-product has less.")
    else:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("Observed", "Trend (slow movement)", "Yearly season", "Remainder (noise)"))
        for i, (k, col) in enumerate((("observed", ui.TEXT), ("trend", ui.BLUE), ("seasonal", ui.ORANGE), ("remainder", ui.CONTEXT)), 1):
            fig.add_scatter(x=dec.index, y=dec[k], row=i, col=1, line=dict(color=col, width=1.6), showlegend=False,
                            hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra></extra>")
        fig.update_annotations(font=dict(size=12, color=ui.TEXT_2), x=0, xanchor="left")
        fig.update_layout(hovermode="closest", margin=dict(t=30))
        ui.show(fig, 560, key="mc_dec")
        s = v["strength"]
        ui.note(f"STL splits the series into a slow <b>trend</b>, a repeating <b>yearly season</b> and the <b>remainder</b>. "
                f"Seasonal strength {s['seasonal strength']:.2f}, trend strength {s['trend strength']:.2f} (1 = that part explains "
                "almost all the movement).")

# ------------------------------------------------------------------ 3 ACF / PACF
with tabs[2]:
    ap = v["acf_pacf"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("ACF: correlation with k weeks ago",
                                                                           "PACF: direct effect of k weeks ago"))
    for j, col in enumerate(("acf", "pacf"), 1):
        sig = ap[col].abs() > ap.band
        fig.add_bar(x=ap.lag, y=ap[col], row=1, col=j, marker=dict(color=np.where(sig, ui.BLUE, ui.CONTEXT)), showlegend=False,
                    hovertemplate="lag %{x}: %{y:.2f}<extra></extra>")
        fig.add_scatter(x=np.r_[ap.lag, ap.lag[::-1]], y=np.r_[ap.band, -ap.band[::-1]], fill="toself", mode="lines",
                        fillcolor="rgba(235,104,52,.10)", line=dict(width=0), row=1, col=j, showlegend=False, hoverinfo="skip")
    fig.update_annotations(font=dict(size=12, color=ui.TEXT_2))
    fig.update_xaxes(title_text="Lag (weeks)")
    fig.update_layout(hovermode="closest", margin=dict(t=40))
    ui.show(fig, 330, key="mc_acf")
    idn = v["identify"]
    ui.kpis([
        {"label": "Pattern read from the plots", "value": str(idn["family"]),
         "sub": f"suggested p = {idn['suggested p']}, d = {idn['suggested d']}, q = {idn['suggested q']}", "accent": True},
        {"label": "Significant ACF lags", "value": ", ".join(map(str, list(idn["significant ACF lags"])[:8])) or "none",
         "sub": "weeks back with a real correlation"},
        {"label": "Significant PACF lags", "value": ", ".join(map(str, list(idn["significant PACF lags"])[:8])) or "none",
         "sub": "weeks back with a direct effect"},
    ])
    ui.note("Blue bars cross the orange band: they are significant. <b>PACF cuts off, ACF fades → AR</b>; "
            "<b>ACF cuts off, PACF fades → MA</b>; both fade → <b>ARMA</b>; needed differencing → <b>ARIMA</b>; "
            "a spike at lag 52 → a seasonal part (<b>SARIMA</b>).")

# ------------------------------------------------------------------ 4 AIC / BIC
with tabs[3]:
    g = v["aic_grid"].dropna(subset=["AIC"]).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=g["ΔAIC"], y=g["model"], orientation="h",
                           marker=dict(color=[ui.BLUE if m == v["best"] else ui.CONTEXT for m in g["model"]], cornerradius=4),
                           text=[f"AIC {a:,.1f} · BIC {b:,.1f}" for a, b in zip(g["AIC"], g["BIC"])], textposition="outside",
                           cliponaxis=False, textfont=dict(color=ui.TEXT_2), hovertemplate="%{y}<br>ΔAIC %{x:.1f}<extra></extra>"))
    fig.update_layout(title=dict(text="The 10 best candidates: AIC distance from the best (0 = best)"),
                      xaxis=dict(title="ΔAIC (lower is better)", range=[0, max(g["ΔAIC"].max(), 1) * 1.6], showgrid=True),
                      yaxis=dict(showgrid=False), hovermode="closest", margin=dict(r=40))
    ui.show(fig, 380, key="mc_aic")
    ui.note("Every candidate is fitted and scored. <b>AIC</b> and <b>BIC</b> reward a good fit and penalise extra "
            "parameters (BIC more strictly). A ΔAIC under 2 means two models are practically as good.")

# ------------------------------------------------------------------ 5 residuals + GARCH
with tabs[4]:
    r = np.asarray(v["resid"], float)
    c = st.columns(2, gap="medium")
    with c[0]:
        fig = go.Figure()
        fig.add_scatter(x=v["resid_dates"], y=r, name="Residual", line=dict(color=ui.BLUE, width=1.3))
        if v["garch"]:
            vol = v["garch"]["volatility"]
            fig.add_scatter(x=v["resid_dates"][-len(vol):], y=vol, name="GARCH volatility (±)", line=dict(color=ui.ORANGE, width=2))
            fig.add_scatter(x=v["resid_dates"][-len(vol):], y=-vol, showlegend=False, line=dict(color=ui.ORANGE, width=2), hoverinfo="skip")
        fig.update_layout(title=dict(text="What the model could not explain"), yaxis=dict(title="Units", rangemode="normal"))
        ui.show(fig, 320, key="mc_res")
    with c[1]:
        (osm, osr), (sl, ic, _) = stats.probplot(r, dist="norm")
        fig = go.Figure()
        fig.add_scatter(x=osm, y=osr, mode="markers", name="Residuals", marker=dict(color=ui.BLUE, size=6))
        fig.add_scatter(x=osm, y=sl * osm + ic, name="Normal line", line=dict(color=ui.ORANGE, width=2))
        fig.update_layout(title=dict(text="QQ plot: are the residuals normal?"), xaxis=dict(title="Normal quantiles"),
                          yaxis=dict(title="Residual quantiles", rangemode="normal"), hovermode="closest")
        ui.show(fig, 320, key="mc_qq")

    def show_tests(t):
        return {k: (("yes" if t[k] else "no") if isinstance(t[k], bool) else f"{t[k]:.3f}") for k in t}
    cols = {"Model residuals": show_tests(rt)}
    if v["garch"]:
        cols["After GARCH(1,1)"] = show_tests(v["garch"]["std_tests"])
        cols["After GARCH, squared"] = show_tests(v["garch"]["sq_tests"])
    st.dataframe(pd.DataFrame(cols).rename_axis("test").reset_index(), hide_index=True, width="stretch")
    if v["garch"]:
        gp = v["garch"]["params"]
        ui.note(f"<b>Ljung-Box</b>: p above 0.05 = no pattern left (white noise). <b>ARCH-LM</b>: p below 0.05 = calm and "
                f"turbulent periods alternate. <b>GARCH(1,1)</b> models that changing spread: ω = {gp.get('omega', 0):.2f}, "
                f"α = {gp.get('alpha[1]', 0):.2f} (reaction to the last shock), β = {gp.get('beta[1]', 0):.2f} (memory). "
                "After GARCH, the residuals are tested again.")

# ------------------------------------------------------------------ 6 forecast check (network)
if "holdout" in v:
    with tabs[5]:
        h = v["holdout"]
        hist_t = pd.Series(v["y"], index=v["dates"]).iloc[-30:]
        fig = go.Figure()
        fig.add_scatter(x=hist_t.index, y=hist_t, name="History", line=dict(color=ui.TEXT, width=1.6))
        fig.add_scatter(x=np.r_[h.date, h.date[::-1]], y=np.r_[h.hi95, h.lo95[::-1]], fill="toself", mode="lines",
                        fillcolor="rgba(42,120,214,.10)", line=dict(width=0), name="95% interval", hoverinfo="skip")
        fig.add_scatter(x=np.r_[h.date, h.date[::-1]], y=np.r_[h.hi80, h.lo80[::-1]], fill="toself", mode="lines",
                        fillcolor="rgba(42,120,214,.20)", line=dict(width=0), name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=h.date, y=h.forecast, name=f"{v['best']} forecast", mode="lines+markers",
                        line=dict(color=ui.BLUE, width=2.5), marker=dict(size=8))
        fig.add_scatter(x=h.date, y=h.actual, name="What actually sold", mode="lines+markers",
                        line=dict(color=ui.TEXT, width=1.6, dash="dot"), marker=dict(size=8))
        fig.update_layout(title=dict(text="Forecast of the 8 competition weeks, made before they happened"), yaxis=dict(title="Units"))
        ui.show(fig, 360, key="mc_hold")
        inside = ((h.actual >= h.lo80) & (h.actual <= h.hi80)).mean()
        ui.kpis([
            {"label": "Forecast error (MAPE)", "value": f"{net['holdout_mape']:.1f}%", "accent": True,
             "sub": f"same week last year: {net['holdout_mape_snaive']:.1f}%"},
            {"label": "Weeks inside the 80% range", "value": f"{inside:.0%}", "sub": "should be about 80% or more"},
        ])

# ------------------------------------------------------------------ all store-products
ui.section("All 599", "The same checks on every store-product",
           "Each store-product gets its own ARIMA (difference order by ADF, p and q by AIC) on season-adjusted demand.")
ar = data.table("arima_rows.parquet")
wh = data.table("error_whiteness.csv").rename(columns={"Unnamed: 0": "model"})
c = st.columns([2, 3], gap="medium")
with c[0]:
    fam = ar["family"].value_counts()
    fig = go.Figure(go.Bar(x=fam.values, y=fam.index, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                           text=fam.values, textposition="outside", cliponaxis=False, textfont=dict(color=ui.TEXT_2),
                           hovertemplate="%{y}: %{x} store-products<extra></extra>"))
    fig.update_layout(title=dict(text="Model family chosen by AIC"), yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(range=[0, fam.max() * 1.2]), hovermode="closest")
    ui.show(fig, 300, key="mc_fam")
with c[1]:
    ui.kpis([
        {"label": "Stationary (ADF)", "value": f"{net['adf_rows_stationary']:.0%}", "sub": "of store-products"},
        {"label": "Residuals = white noise", "value": f"{net['rows_white_noise']:.0%}", "sub": "Ljung-Box, lags 4, 8, 13",
         "accent": True},
        {"label": "Volatility clustering", "value": f"{net['rows_arch']:.0%}",
         "sub": f"ARCH-LM; GARCH fixes {net['garch_rows_fixed']:.0%} of them" if net.get("garch_rows_fixed") is not None else "ARCH-LM"},
    ])
    ui.note("With a 5% test, about 5% of perfectly good models fail by chance alone, so 90%+ passing is what a "
            "well-specified model looks like.")

with st.container(key="card_wh"):
    w = wh.sort_values("white_noise_share")
    fig = go.Figure(go.Bar(x=100 * w["white_noise_share"], y=w["model"].map(ui.MODEL_NAMES), orientation="h",
                           marker=dict(color=[ui.BLUE if m == data.plan_choice()[0] else ui.CONTEXT for m in w["model"]], cornerradius=4),
                           text=[f"{x:.0%}" for x in w["white_noise_share"]], textposition="outside", cliponaxis=False,
                           textfont=dict(color=ui.TEXT_2), hovertemplate="%{y}: %{x:.0f}% of store-products<extra></extra>"))
    fig.update_layout(title=dict(text="Out-of-sample check: share of store-products whose forecast errors are white noise"),
                      xaxis=dict(range=[0, 110], ticksuffix="%", showgrid=True), yaxis=dict(showgrid=False), hovermode="closest")
    ui.show(fig, 380, key="mc_wh")
ui.note("For each method and store-product, the 1-week-ahead errors over 30 past weeks were tested with Ljung-Box. "
        "Passing means the errors have no pattern left to learn. It does not prove a forecast is accurate on its own "
        "(see the Forecast page for accuracy); the two together are the evidence.")
ui.next_step(3)
