"""Design system for the dashboard: colours, CSS, Plotly template and HTML components.

Light, modern theme. Series colours are the validated light-mode categorical slots
(blue = forecast / plan, orange = simple rule) on a white card surface; text never wears
a series colour.
"""
from __future__ import annotations

import html
import re

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- tokens
BG = "#f6f7fb"
CARD = "#ffffff"
CARD_2 = "#f1f4f9"
BORDER = "#e7eaf1"
TEXT = "#0f172a"
TEXT_2 = "#475569"
MUTED = "#8a94a6"
GRID = "#eef1f6"
AXIS = "#d9dee8"
BLUE = "#2a78d6"       # forecast / forecast-driven plan
BLUE_INK = "#1c5cab"   # blue for emphasised text on light surfaces
ORANGE = "#eb6834"     # simple rule
AQUA = "#1baf7a"       # decorative accent only (below 3:1 on white, never a data series)
CONTEXT = "#c3cad6"    # de-emphasised data
CONTEXT_FILL = "rgba(148,163,184,.16)"
BLUE_FILL = "rgba(42,120,214,.08)"
BLUE_BAND = "rgba(42,120,214,.14)"
DEMAND_BAR = "rgba(148,163,184,.32)"
GOOD = "#006300"       # success text on light
BAD = "#d03b3b"
DIVERGING = ["#ec8b8a", "#f0efec", "#6da7ec"]  # worse / same / better
COST_STEPS = ["#b7d3f6", "#5598e7", "#1c5cab"]  # storage, deliveries, missed sales (one ordered hue)

MODEL_NAMES = {
    "naive": "Repeat yesterday",
    "snaive": "Same day last week",
    "ma28": "4-week average",
    "ets": "Exponential smoothing",
    "arima": "ARIMA",
    "sarima": "Seasonal ARIMA",
    "croston": "Croston (slow movers)",
}
MODEL_BLURBS = {
    "naive": "Tomorrow will look like today.",
    "snaive": "Each day repeats the same weekday from last week.",
    "ma28": "Every day gets the average of the last four weeks.",
    "ets": "A weighted average that favours recent days, with an optional trend and weekly rhythm.",
    "arima": "A statistical model that learns how today's sales relate to the last few days.",
    "sarima": "Like ARIMA, but also learns the weekly rhythm (same weekday last week, two weeks ago…).",
    "croston": "Built for slow movers: predicts how big a sale is and how often one happens, separately.",
}
POLICY_NAMES = {
    "P0 baseline": "Simple rule",
    "P1 forecast": "Forecast-driven plan",
    "P2 empirical": "Forecast plan + smart buffer",
    "Perfect foresight": "Perfect hindsight",
}
CLASS_NAMES = {
    "smooth": "Steady seller",
    "erratic": "Spiky seller",
    "intermittent": "Occasional seller",
    "lumpy": "Unpredictable seller",
}
CLASS_BLURBS = {
    "smooth": "Sells almost every day in similar amounts.",
    "erratic": "Sells most days, but quantities swing a lot.",
    "intermittent": "Many days with no sales; small amounts when it sells.",
    "lumpy": "Irregular days and irregular amounts: the hardest to plan.",
}
CATEGORY = {"FOODS": "Food", "HOUSEHOLD": "Household", "HOBBIES": "Hobbies"}

FLOW = [
    ("Overview", "views/overview.py"),
    ("Understand demand", "views/demand.py"),
    ("Predict", "views/forecast.py"),
    ("Model check", "views/model_check.py"),
    ("Plan stock", "views/stock_plan.py"),
    ("Report", "views/report.py"),
]


def item_category(item: str) -> str:
    return CATEGORY.get(item.split("_")[0], item.split("_")[0].title())


def item_short(item: str) -> str:
    return f"{item_category(item)} #{'-'.join(item.split('_')[1:])}"


def fmt_date(ts) -> str:
    return f"{ts:%b} {ts.day}, {ts:%Y}"


# ---------------------------------------------------------------- CSS
CSS = f"""
<style>
.stApp {{
  background:
    radial-gradient(900px 480px at -5% -10%, #e3eefc 0%, transparent 60%),
    radial-gradient(760px 420px at 105% -5%, #e5f6ef 0%, transparent 60%),
    radial-gradient(700px 500px at 50% 120%, #efeafd 0%, transparent 65%),
    {BG};
}}
[data-testid="stHeader"] {{ background: rgba(255,255,255,.72); backdrop-filter: saturate(180%) blur(14px);
  border-bottom: 1px solid {BORDER}; }}
.block-container {{ padding-top: 4.8rem; padding-bottom: 4rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: -.025em; color: {TEXT}; }}
p, li {{ color: {TEXT_2}; }}

/* motion ----------------------------------------------------------------- */
@keyframes ocFadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes ocDraw {{ from {{ stroke-dashoffset: 1; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes ocPulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(42,120,214,0), 0 6px 18px rgba(42,120,214,.14); }}
                      50% {{ box-shadow: 0 0 0 5px rgba(42,120,214,.10), 0 6px 18px rgba(42,120,214,.14); }} }}
@keyframes ocShine {{ from {{ background-position: 0% 50%; }} to {{ background-position: 100% 50%; }} }}
.oc-anim {{ animation: ocFadeUp .65s cubic-bezier(.2,.7,.2,1) both; }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ animation: ocFadeUp .7s cubic-bezier(.2,.7,.2,1) both; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}

/* hero ------------------------------------------------------------------- */
.oc-hero {{ padding: 10px 0 6px; }}
.oc-eyebrow {{ display: inline-block; font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: {BLUE_INK}; background: #e8f1fd; border: 1px solid #d3e4fa; padding: 5px 12px; border-radius: 999px; }}
.oc-hero h1 {{ font-size: clamp(2.2rem, 4.4vw, 3.5rem); font-weight: 800; line-height: 1.05; margin: .9rem 0 .7rem;
  background: linear-gradient(100deg, {TEXT} 0%, #1e3a8a 40%, {BLUE} 70%, #0e8a5f 100%); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: ocShine 6s ease-in-out infinite alternate; }}
.oc-hero p {{ font-size: 1.08rem; max-width: 760px; line-height: 1.65; }}
.oc-page-title {{ font-size: 2rem; font-weight: 800; color: {TEXT}; margin: .2rem 0 .1rem; letter-spacing: -.03em; }}
.oc-page-sub {{ color: {TEXT_2}; font-size: 1.02rem; margin-bottom: .5rem; }}

/* flow stepper ----------------------------------------------------------- */
.oc-steps {{ display: flex; gap: 8px; margin: 6px 0 24px; flex-wrap: wrap; padding: 6px; border-radius: 16px;
  background: rgba(255,255,255,.6); border: 1px solid {BORDER}; backdrop-filter: blur(8px); }}
.oc-step {{ flex: 1 1 0; min-width: 120px; display: flex; align-items: center; gap: 8px; padding: 9px 10px; white-space: nowrap;
  border-radius: 11px; color: {MUTED}; font-size: .84rem; font-weight: 600; transition: all .3s ease; }}
.oc-step .n {{ width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: .74rem;
  background: {CARD_2}; color: {MUTED}; flex: none; font-weight: 700; }}
.oc-step.done {{ color: {TEXT_2}; }}
.oc-step.done .n {{ background: #e3eefc; color: {BLUE_INK}; }}
.oc-step.active {{ color: {TEXT}; background: {CARD}; animation: ocPulse 2.8s ease-in-out infinite; }}
.oc-step.active .n {{ background: linear-gradient(135deg, {BLUE}, {BLUE_INK}); color: white; }}

/* cards ------------------------------------------------------------------ */
.oc-card, [class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 20px rgba(15,23,42,.05); }}
.oc-card {{ padding: 18px 20px; }}
[class*="st-key-card"] {{ padding: 18px 20px 14px; animation: ocFadeUp .65s cubic-bezier(.2,.7,.2,1) both;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }}
[class*="st-key-cardhover"]:hover {{ transform: translateY(-4px); border-color: #cfe0f8;
  box-shadow: 0 2px 4px rgba(15,23,42,.04), 0 18px 40px rgba(42,120,214,.12); }}
[class*="st-key-cardhover_sel"] {{ border-color: #b9d4f6; background: linear-gradient(165deg, #eef5fe, {CARD} 55%);
  box-shadow: 0 0 0 3px rgba(42,120,214,.10), 0 10px 28px rgba(42,120,214,.12); }}
.oc-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin: 6px 0 18px; }}
.oc-kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 20px rgba(15,23,42,.05); transition: transform .25s ease, box-shadow .25s ease; }}
.oc-kpi:hover {{ transform: translateY(-3px); box-shadow: 0 2px 4px rgba(15,23,42,.04), 0 16px 34px rgba(15,23,42,.08); }}
.oc-kpi .lbl {{ color: {MUTED}; font-size: .8rem; font-weight: 600; }}
.oc-kpi .val {{ color: {TEXT}; font-size: 1.9rem; font-weight: 800; letter-spacing: -.03em; margin-top: 4px; line-height: 1.15; }}
.oc-kpi .val .u {{ font-size: .9rem; color: {TEXT_2}; font-weight: 600; margin-left: 4px; letter-spacing: 0; }}
.oc-kpi.small .val {{ font-size: 1.3rem; padding-top: 6px; }}
.oc-kpi .sub {{ color: {TEXT_2}; font-size: .82rem; margin-top: 6px; }}
.oc-kpi .sub.good {{ color: {GOOD}; font-weight: 600; }} .oc-kpi .sub.bad {{ color: {BAD}; font-weight: 600; }}
.oc-kpi.accent {{ border-color: #cfe0f8; background: linear-gradient(150deg, #eaf3fe 0%, {CARD} 70%); }}

.oc-insight {{ display: flex; gap: 14px; align-items: flex-start; }}
.oc-insight .ic {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; flex: none;
  background: linear-gradient(135deg, #e8f1fd, #e7f7f0); font-size: 1.15rem; }}
.oc-insight .t {{ color: {TEXT}; font-weight: 700; font-size: .98rem; }}
.oc-insight .d {{ color: {TEXT_2}; font-size: .9rem; line-height: 1.55; margin-top: 2px; }}

.oc-badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700;
  background: #eaf3fe; color: {BLUE_INK}; border: 1px solid #d3e4fa; }}
.oc-badge.neutral {{ background: {CARD_2}; color: {TEXT_2}; border-color: {BORDER}; }}
.oc-badge.good {{ background: #e8f6e8; color: {GOOD}; border-color: #c7e8c7; }}

.oc-section {{ margin: 34px 0 10px; }}
.oc-section .k {{ font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {BLUE_INK}; }}
.oc-section h3 {{ margin: 4px 0 2px; font-size: 1.35rem; color: {TEXT}; font-weight: 750; }}
.oc-section p {{ margin: 0; font-size: .94rem; }}

.oc-reco {{ border-radius: 20px; padding: 24px 26px; color: white; position: relative; overflow: hidden;
  background: radial-gradient(500px 220px at 100% 0%, rgba(27,175,122,.45), transparent 70%),
              linear-gradient(130deg, #1e3a8a 0%, {BLUE} 60%, #3d8ae0 100%);
  box-shadow: 0 18px 40px rgba(30,58,138,.25); }}
.oc-reco .oc-eyebrow {{ background: rgba(255,255,255,.14); border-color: rgba(255,255,255,.28); color: white; }}
.oc-reco .big {{ font-size: 1.6rem; font-weight: 800; color: white; line-height: 1.35; letter-spacing: -.02em; }}
.oc-reco .big b {{ color: white; background: rgba(255,255,255,.16); padding: 0 8px; border-radius: 8px; }}
.oc-reco .meta, .oc-reco .meta b {{ color: rgba(255,255,255,.92) !important; }}
.oc-reco .oc-badge {{ background: rgba(255,255,255,.95); color: {BLUE_INK}; border: none; }}
.oc-reco .oc-badge.good {{ color: {GOOD}; }}
.oc-note {{ border: 1px solid {BORDER}; padding: 10px 14px; color: {TEXT_2}; font-size: .85rem; background: rgba(255,255,255,.7);
  border-radius: 12px; }}

.oc-spark {{ width: 100%; height: 46px; display: block; margin: 8px 0 2px; }}
.oc-spark path.line {{ fill: none; stroke: {BLUE}; stroke-width: 2; stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: ocDraw 1.6s cubic-bezier(.4,0,.2,1) .2s forwards; }}
.oc-spark path.area {{ fill: url(#ocg); opacity: 0; animation: ocFadeUp 1s ease .9s forwards; }}

/* widgets ---------------------------------------------------------------- */
[data-testid="stPageLink"] a {{ background: linear-gradient(135deg, {BLUE}, {BLUE_INK}); border-radius: 12px; padding: 10px 18px;
  box-shadow: 0 8px 20px rgba(42,120,214,.25); transition: transform .2s ease, box-shadow .2s ease; justify-content: center; }}
[data-testid="stPageLink"] a:hover {{ transform: translateY(-2px); box-shadow: 0 14px 28px rgba(42,120,214,.32); }}
[data-testid="stPageLink"] a p {{ color: white !important; font-weight: 700; }}
.stButton > button {{ border-radius: 11px; font-weight: 600; transition: all .2s ease; background: {CARD}; border-color: {BORDER}; }}
.stButton > button[kind="primary"] {{ background: linear-gradient(135deg, {BLUE}, {BLUE_INK}); border: none;
  box-shadow: 0 6px 16px rgba(42,120,214,.25); }}
.stButton > button[kind="primary"] p {{ color: #ffffff; font-weight: 700; }}
.stButton > button:hover {{ transform: translateY(-1px); border-color: {BLUE}; color: {BLUE_INK}; }}
[data-testid="stExpander"] details {{ border-radius: 16px; border-color: {BORDER}; background: {CARD}; }}
[data-testid="stSlider"] label p, [data-testid="stSelectbox"] label p {{ color: {TEXT_2}; font-weight: 600; }}
[data-baseweb="select"] > div {{ background: {CARD}; border-color: {BORDER}; border-radius: 12px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)
    _register_template()


# ---------------------------------------------------------------- plotly
def _register_template() -> None:
    if "optichain" in pio.templates:
        return
    axis = dict(automargin=True, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=12),
                title=dict(font=dict(color=TEXT_2, size=12)))
    pio.templates["optichain"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT_2, size=13),
        colorway=[BLUE, ORANGE, AQUA],
        xaxis=axis | dict(showgrid=False), yaxis=axis | dict(showgrid=True, rangemode="tozero"),
        hoverlabel=dict(bgcolor=CARD, bordercolor=AXIS, font=dict(color=TEXT, family="Inter, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2)),
        margin=dict(l=8, r=12, t=40, b=8), hovermode="x unified", bargap=0.25,
    ))


def show(fig: go.Figure, height: int = 380, key: str | None = None) -> None:
    # Streamlit installs its own Plotly default template on import, so apply ours explicitly.
    fig.update_layout(template=pio.templates["optichain"], height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displayModeBar": False, "scrollZoom": False})


# ---------------------------------------------------------------- components
def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def stepper(active: int) -> None:
    parts = []
    for i, (label, _) in enumerate(FLOW):
        cls = "active" if i == active else ("done" if i < active else "")
        n = "✓" if i < active else str(i + 1)
        parts.append(f'<div class="oc-step {cls}"><span class="n">{n}</span>{label}</div>')
    md(f'<div class="oc-steps oc-anim">{"".join(parts)}</div>')


def page_header(step: int, title: str, subtitle: str) -> None:
    stepper(step)
    md(f'<div class="oc-anim"><div class="oc-page-title">{title}</div><div class="oc-page-sub">{subtitle}</div></div>')


def section(kicker: str, title: str, text: str = "") -> None:
    md(f'<div class="oc-section oc-anim"><div class="k">{kicker}</div><h3>{title}</h3>'
       f'{f"<p>{text}</p>" if text else ""}</div>')


def kpis(cards: list[dict]) -> None:
    """cards: label, value, unit?, sub?, tone? ('good'|'bad'), accent? (bool)."""
    out = []
    for i, c in enumerate(cards):
        sub = f'<div class="sub {c.get("tone", "")}">{c["sub"]}</div>' if c.get("sub") else ""
        unit = f'<span class="u">{c["unit"]}</span>' if c.get("unit") else ""
        acc = (" accent" if c.get("accent") else "") + (" small" if len(str(c["value"])) > 12 else "")
        out.append(f'<div class="oc-kpi oc-anim{acc}" style="animation-delay:{i * 0.07:.2f}s">'
                   f'<div class="lbl">{c["label"]}</div><div class="val">{c["value"]}{unit}</div>{sub}</div>')
    md(f'<div class="oc-grid">{"".join(out)}</div>')


def insights(items: list[tuple[str, str, str]]) -> None:
    """items: (icon, title, text)."""
    out = []
    for i, (ic, t, d) in enumerate(items):
        d = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", d)  # **bold**
        d = re.sub(r"(?<![\w*])\*(?!\s)(.+?)\*(?!\w)", r"<i>\1</i>", d)  # *italic*
        out.append(f'<div class="oc-card oc-anim" style="animation-delay:{0.1 + i * 0.08:.2f}s">'
                   f'<div class="oc-insight"><div class="ic">{ic}</div><div><div class="t">{t}</div>'
                   f'<div class="d">{d}</div></div></div></div>')
    md(f'<div class="oc-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">{"".join(out)}</div>')


def note(text: str) -> None:
    md(f'<div class="oc-note oc-anim">{text}</div>')


def sparkline(values: np.ndarray, width: int = 240, height: int = 46) -> str:
    v = np.asarray(values, float)
    if len(v) < 2 or np.nanmax(v) == np.nanmin(v):
        v = np.r_[v, v + 1e-9]
    x = np.linspace(0, width, len(v))
    y = height - 4 - (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) * (height - 8)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    line = f"M{pts.replace(' ', ' L')}"
    area = f"{line} L{width},{height} L0,{height} Z"
    return (f'<svg class="oc-spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
            f'<defs><linearGradient id="ocg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{BLUE}" stop-opacity=".35"/>'
            f'<stop offset="100%" stop-color="{BLUE}" stop-opacity="0"/></linearGradient></defs>'
            f'<path class="area" d="{area}"/><path class="line" pathLength="1" d="{line}"/></svg>')


def next_step(step: int) -> None:
    if step + 1 < len(FLOW):
        label, path = FLOW[step + 1]
        st.write("")
        c = st.columns([3, 2])
        c[1].page_link(path, label=f"Next: {label}  →")


def esc(s: str) -> str:
    return html.escape(str(s))
