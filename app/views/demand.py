import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui
from optichain.data import ADI_CUT, CV2_CUT

ui.page_header(1, "Understand demand", "How much does each store-product sell, when, and how often did the shelf run empty?")
p = data.panel()
r = data.rows()

c = st.columns([2, 3])
with c[0]:
    row = data.row_picker("dem")
info = r.iloc[row]
sales = p.sales.iloc[row]
instock = p.in_stock.iloc[row]
pattern = info["pattern"]

ui.kpis([
    {"label": "Sales pattern", "value": ui.PATTERN_NAMES[pattern], "sub": ui.PATTERN_BLURBS[pattern], "accent": True},
    {"label": "Average week", "value": f"{info['mean_week']:.1f}", "unit": "units", "sub": f"last 13 weeks: {info['last13']:.1f}"},
    {"label": "Weeks with no sale", "value": f"{info['zero_share']:.0%}", "sub": "share of weeks since launch"},
    {"label": "Weeks ending out of stock", "value": f"{info['oos_share']:.0%}",
     "sub": "these weeks under-count demand", "tone": "bad" if info["oos_share"] > .15 else ""},
])

# ------------------------------------------------------------------ history
with st.container(key="card_hist"):
    weeks = p.dates
    clean = sales.where(instock & (sales.cumsum() > 0))
    avg = clean.rolling(13, min_periods=4).mean()
    fig = go.Figure()
    fig.add_bar(x=weeks[instock.values], y=sales[instock.values], name="Normal week", marker=dict(color=ui.BLUE, cornerradius=2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} units sold<extra>Normal week</extra>")
    launched = (sales.cumsum() > 0).to_numpy()
    empty = ~instock.to_numpy() & launched
    top = max(float(sales.max()), 1.0) * 1.05
    # out-of-stock weeks sell 0, so they are shown as full-height bands behind the bars
    fig.add_bar(x=weeks[empty], y=np.full(empty.sum(), top), name="Shelf ran empty (set aside)",
                marker=dict(color="rgba(235,104,52,.22)", line=dict(width=0)), width=6.5 * 86400000,
                hovertemplate="%{x|%d %b %Y}<br>shelf empty: no sales possible<extra></extra>")
    if (~launched).any():
        fig.add_vrect(x0=weeks[0] - pd.Timedelta(days=3.5), x1=weeks[int(launched.argmax())] - pd.Timedelta(days=3.5),
                      fillcolor=ui.CONTEXT_FILL, line_width=0, annotation_text="not launched yet",
                      annotation_position="top left", annotation_font=dict(color=ui.TEXT_2, size=11))
    fig.add_scatter(x=weeks, y=avg, name="13-week average of normal weeks", line=dict(color=ui.TEXT, width=2),
                    hovertemplate="%{x|%d %b %Y}<br>average %{y:.1f}<extra></extra>")
    fig.add_vrect(x0=p.week0 + pd.Timedelta(days=3.5), x1=weeks[-1] + pd.Timedelta(days=3.5), fillcolor=ui.CONTEXT_FILL,
                  line_width=0, annotation_text="competition weeks", annotation_position="top left",
                  annotation_font=dict(color=ui.TEXT_2, size=11))
    fig.update_layout(title=dict(text=f"{info['label']}: units sold per week", font=dict(size=14, color=ui.TEXT)),
                      yaxis=dict(title="Units", range=[0, top]), barmode="overlay", bargap=0.15, hovermode="closest")
    ui.show(fig, 380, key="hist")
ui.note("Orange bands are weeks flagged <b>out of stock</b>: the shelf was empty, so nothing could be sold (they record "
        "zero sales). That zero is about supply, not demand, so the models learn only from the blue weeks.")

# ------------------------------------------------------------------ seasonality + pattern map
ui.section("Patterns", "The yearly rhythm and how this product compares")
c = st.columns(2, gap="medium")
with c[0]:
    with st.container(key="card_season"):
        season = data.table("season.csv")
        cur_week = int(p.week0.isocalendar().week)
        colors = [ui.BLUE if v >= 1 else ui.CONTEXT for v in season.factor]
        fig = go.Figure(go.Bar(x=season.iso_week, y=season.factor, marker=dict(color=colors, cornerradius=3),
                               hovertemplate="Week %{x}: %{y:.2f}× an average week<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=ui.TEXT_2, width=1))
        fig.add_vrect(x0=15.5, x1=23.5, fillcolor=ui.CONTEXT_FILL, line_width=0, annotation_text="competition",
                      annotation_position="top left", annotation_font=dict(color=ui.TEXT_2, size=11))
        fig.update_layout(title=dict(text="Seasonality across all stores", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(title="Week of the year", dtick=4), yaxis=dict(title="× average week", rangemode="normal",
                                                                                   range=[season.factor.min() * .9, season.factor.max() * 1.06]),
                          hovermode="closest", showlegend=False)
        ui.show(fig, 330, key="season")
with c[1]:
    with st.container(key="card_pattern"):
        adi = r["adi"].clip(upper=30)
        cv2 = r["cv2"].clip(upper=4)
        fig = go.Figure()
        fig.add_scatter(x=adi, y=cv2, mode="markers", marker=dict(size=7, color=ui.CONTEXT, line=dict(color="white", width=1)),
                        text=r["label"], customdata=r["pattern"].map(ui.PATTERN_NAMES),
                        hovertemplate="%{text}<br>%{customdata}<br>sells every %{x:.1f} weeks · size variation %{y:.2f}<extra></extra>",
                        name="All store-products")
        fig.add_scatter(x=[adi.iloc[row]], y=[cv2.iloc[row]], mode="markers", name=info["label"],
                        marker=dict(size=15, color=ui.BLUE, line=dict(color="white", width=2)),
                        hovertemplate=f"{info['label']}<extra></extra>")
        fig.add_vline(x=ADI_CUT, line=dict(color=ui.AXIS, dash="dash"))
        fig.add_hline(y=CV2_CUT, line=dict(color=ui.AXIS, dash="dash"))
        for x, y, t_ in ((1.0, 3.85, "Spiky"), (1.5, 3.85, "Unpredictable"), (1.0, -0.02, "Steady"), (1.5, -0.02, "Occasional")):
            fig.add_annotation(x=np.log10(x), y=y, text=t_, showarrow=False, xanchor="left", font=dict(color=ui.MUTED, size=12))
        fig.update_layout(title=dict(text="Sales patterns of all 599 store-products", font=dict(size=14, color=ui.TEXT)),
                          xaxis=dict(title="Weeks between sales (log scale)", type="log", showgrid=True, tickvals=[1, 1.32, 2, 5, 10, 30],
                                     ticktext=["1", "1.32", "2", "5", "10", "30+"], range=[np.log10(0.95), np.log10(33)]),
                          yaxis=dict(title="Variation in sale size", range=[-0.1, 4.1]), hovermode="closest", showlegend=False)
        ui.show(fig, 330, key="pattern")

# ------------------------------------------------------------------ network availability
ui.section("Availability", "Where and when shelves ran empty",
           "Share of each store's launched products whose shelf was empty at the end of the week, month by month. "
           "Darker means more empty shelves.")
with st.container(key="card_heat"):
    oos = (~p.in_stock).astype(float).where(p.sales.cumsum(axis=1) > 0)   # launched weeks only
    oos.columns = oos.columns.to_period("M")
    by_store = oos.T.groupby(level=0).mean().T.groupby(level="Store").mean()
    order = by_store.mean(axis=1).sort_values(ascending=False).index
    by_store = by_store.loc[order]
    fig = go.Figure(go.Heatmap(
        z=100 * by_store.to_numpy(), x=[c.to_timestamp() for c in by_store.columns], y=[f"Store {s}" for s in by_store.index],
        colorscale=[[0, "#f4f8fe"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1, "#0d366b"]],
        zmin=0, zmax=50, xgap=1, ygap=1, colorbar=dict(title="% out of stock", ticksuffix="%", thickness=12),
        hovertemplate="%{y} · %{x|%b %Y}<br>%{z:.0f}% of weeks ended out of stock<extra></extra>"))
    fig.update_layout(yaxis=dict(showticklabels=False, title="67 stores (worst at the top)", showgrid=False, autorange="reversed"),
                      xaxis=dict(showgrid=False, dtick="M6", tickformat="%b %Y"), hovermode="closest")
    ui.show(fig, 430, key="heat")

ui.next_step(1)
