import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import ui

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ui.page_header(1, "Understand demand", "How much does this product sell, and when?")
c = st.columns([2, 3])
with c[0]:
    item = data.product_picker("demand")

s = data.series(item)
cls = data.demand_class(item)
recent = s.iloc[-730:]
mean = recent.mean()
wd = recent.groupby(recent.index.day_name()).mean().reindex(WEEKDAYS)
wd_idx = wd / mean
busiest, quietest = wd_idx.idxmax(), wd_idx.idxmin()
zero_share = (recent == 0).mean()
price = data.price(item).dropna()

ui.kpis([
    {"label": "Average sold per day", "value": f"{mean:.1f}", "unit": "units", "sub": "Last two years", "accent": True},
    {"label": "Busiest day", "value": busiest, "sub": f"{wd_idx[busiest] - 1:+.0%} vs an average day"},
    {"label": "Days with no sales", "value": f"{zero_share:.0%}",
     "sub": ui.CLASS_BLURBS[cls]},
    {"label": "Current price", "value": f"${price.iloc[-1]:.2f}", "sub": f"Ranged ${price.min():.2f}–${price.max():.2f} over 5 years"},
])

# ---------------------------------------------------------------- history
ui.section("Sales history", "Daily sales and the 4-week trend",
           "Use the buttons to zoom. The blue line smooths out day-to-day noise.")
trend = s.rolling(28, min_periods=7).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Daily sales", mode="lines",
                         line=dict(color=ui.CONTEXT, width=1), fill="tozeroy", fillcolor=ui.CONTEXT_FILL,
                         hovertemplate="%{y:.0f} units"))
fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name="4-week trend", mode="lines",
                         line=dict(color=ui.BLUE, width=2.6), hovertemplate="%{y:.1f} / day"))
fig.update_xaxes(range=[s.index[-365], s.index[-1]], rangeselector=dict(
    buttons=[dict(count=3, label="3M", step="month", stepmode="backward"),
             dict(count=6, label="6M", step="month", stepmode="backward"),
             dict(count=1, label="1Y", step="year", stepmode="backward"),
             dict(step="all", label="All")],
    bgcolor=ui.CARD_2, activecolor="#cfe2fb", font=dict(color=ui.TEXT), x=1, xanchor="right", y=1.08))
fig.update_layout(yaxis_title="Units sold", legend=dict(x=0, y=1.08))
ui.show(fig, 400, key="hist")

# ---------------------------------------------------------------- rhythm
c1, c2 = st.columns(2)
with c1:
    ui.section("Weekly rhythm", "Which days are busiest?", "Average sales on each weekday, last two years.")
    colors = [ui.BLUE if d == busiest else ui.CONTEXT for d in WEEKDAYS]
    f2 = go.Figure(go.Bar(x=[d[:3] for d in WEEKDAYS], y=wd.values, marker=dict(color=colors, cornerradius=6),
                          text=[f"{v - 1:+.0%}" for v in wd_idx], textposition="outside",
                          textfont=dict(color=ui.TEXT_2, size=12),
                          hovertemplate="%{x}: %{y:.1f} units/day<extra></extra>"))
    f2.add_hline(y=mean, line=dict(color=ui.MUTED, dash="dot", width=1),
                 annotation_text="average day", annotation_font_color=ui.MUTED, annotation_position="right")
    f2.update_layout(yaxis_title="Units / day", hovermode="closest", showlegend=False, margin=dict(r=95),
                     yaxis_range=[0, wd.max() * 1.22])
    ui.show(f2, 320, key="wd")
with c2:
    full_years = s[(s.index.year > s.index[0].year) & (s.index.year < s.index[-1].year)]
    by_month = (full_years if len(full_years) > 365 else s).groupby(lambda x: x.month).mean().reindex(range(1, 13))
    peak, low = int(by_month.idxmax()), int(by_month.idxmin())
    ui.section("Seasons", "Which months are strongest?", "Average daily sales by calendar month, complete years only.")
    f3 = go.Figure(go.Scatter(x=MONTHS, y=by_month.values, mode="lines+markers", line=dict(color=ui.BLUE, width=2.6, shape="spline"),
                              marker=dict(size=8, color=[ui.BLUE if m == peak else ui.CARD for m in range(1, 13)],
                                          line=dict(color=ui.BLUE, width=2)),
                              fill="tozeroy", fillcolor=ui.BLUE_FILL,
                              hovertemplate="%{x}: %{y:.1f} units/day<extra></extra>"))
    f3.update_layout(yaxis_title="Units / day", hovermode="closest", showlegend=False)
    ui.show(f3, 320, key="month")

# ---------------------------------------------------------------- insights
yoy_now, yoy_prev = s.iloc[-90:].mean(), s.iloc[-455:-365].mean()
yoy = yoy_now / yoy_prev - 1 if yoy_prev > 0 else np.nan
season_swing = by_month[peak] / by_month[low] - 1
ui.section("What this means", "Key takeaways for planning")
cards = [
    ("📅", f"{busiest}s are busiest",
     f"{busiest} sells {wd_idx[busiest] - 1:+.0%} vs an average day, while {quietest} sells {wd_idx[quietest] - 1:+.0%}. "
     "Stock needs to be in place before the peak, not on the day."),
    ("🌤️", f"{MONTHS[peak - 1]} is the strongest month",
     f"Sales in {MONTHS[peak - 1]} run {season_swing:.0%} above {MONTHS[low - 1]}, the quietest month."
     if season_swing > 0.1 else "Sales are fairly even across the year, with no strong season."),
]
if not np.isnan(yoy):
    cards.append(("📊", "Compared with a year ago",
                  f"The last 90 days averaged {yoy_now:.1f} units a day, {yoy:+.0%} vs the same period last year."))
cards.append(("🧩", ui.CLASS_NAMES[cls],
              f"{ui.CLASS_BLURBS[cls]} {zero_share:.0%} of days had no sales. "
              + ("Simple averages often forecast this kind of product as well as clever models."
                 if cls in ("intermittent", "lumpy") else "Clear patterns like this are what forecasting models pick up.")))
ui.insights(cards)
ui.next_step(1)
