"""Build docs/OptiChain_Project_Guide.pdf: the A-to-Z explanation of the project, with charts and dashboard screenshots.

    python scripts/build_guide.py [--shots FOLDER]

Numbers are read from artifacts/ (run the notebook first). --shots points to a folder of dashboard screenshots
named pw_<page>.png (overview, demand, forecast, check, planner, outlook, method); pages without one are skipped.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "OptiChain_Project_Guide.pdf"

# ------------------------------------------------------------------ fonts & styles
FONT_DIR = Path("C:/Windows/Fonts")
try:  # Arial has €, →, ≈, ², × which the built-in Helvetica lacks
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "ariali.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    F, FB = "Body", "Body-Bold"
except Exception:  # noqa: BLE001
    F, FB = "Helvetica", "Helvetica-Bold"

INK = colors.HexColor("#0f172a")
INK2 = colors.HexColor("#475569")
MUTED = colors.HexColor("#8a94a6")
BLUE = colors.HexColor("#2a78d6")
BLUE_INK = colors.HexColor("#1c5cab")
BLUE_BG = colors.HexColor("#eef5fe")
ORANGE = colors.HexColor("#eb6834")
LINE = colors.HexColor("#e2e7ef")
ZEBRA = colors.HexColor("#f6f8fb")
AMBER_BG = colors.HexColor("#fff7ed")

S = {
    "title": ParagraphStyle("title", fontName=FB, fontSize=30, leading=35, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName=F, fontSize=13, leading=19, textColor=INK2),
    "kicker": ParagraphStyle("kicker", fontName=FB, fontSize=8.5, leading=11, textColor=BLUE_INK, spaceBefore=4, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName=FB, fontSize=19, leading=24, textColor=INK, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName=FB, fontSize=13.5, leading=18, textColor=INK, spaceBefore=12, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName=FB, fontSize=11, leading=15, textColor=BLUE_INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=F, fontSize=10, leading=15, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName=F, fontSize=10, leading=14.5, textColor=INK, leftIndent=14, bulletIndent=3,
                             spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName=F, fontSize=8.8, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=FB, fontSize=8.8, leading=12, textColor=INK),
    "head": ParagraphStyle("head", fontName=FB, fontSize=8.8, leading=12, textColor=colors.white),
    "callout": ParagraphStyle("callout", fontName=F, fontSize=9.6, leading=14, textColor=INK),
    "caption": ParagraphStyle("caption", fontName=F, fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=3, spaceAfter=10),
    "toc": ParagraphStyle("toc", fontName=F, fontSize=10.5, leading=17, textColor=INK),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=11.5, textColor=INK, backColor=ZEBRA,
                           borderPadding=6, spaceAfter=8),
}
W = A4[0] - 4.2 * cm  # text width


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, bold_first=False):
    data = []
    for i, r in enumerate(rows):
        st_ = "head" if (header and i == 0) else "cell"
        data.append([Paragraph(str(c), S["cellb"] if (bold_first and j == 0 and st_ == "cell") else S[st_])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE_INK)]
    for i in range(1 if header else 0, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def callout(text, kind="info"):
    bg, bar = (BLUE_BG, BLUE) if kind == "info" else (AMBER_BG, ORANGE)
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 8)])


def kpi_row(items):
    lbl_s = ParagraphStyle("kl", fontName=F, fontSize=8, leading=10, textColor=MUTED)
    val_s = ParagraphStyle("kv", fontName=FB, fontSize=17, leading=22, textColor=INK, spaceBefore=2)
    sub_s = ParagraphStyle("ks", fontName=F, fontSize=8, leading=10.5, textColor=INK2)
    cells = [[[Paragraph(lbl, lbl_s), Paragraph(val, val_s), Paragraph(sub, sub_s)] for lbl, val, sub in items]]
    t = Table(cells, colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                           ("BACKGROUND", (0, 0), (0, 0), BLUE_BG), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return KeepTogether([t, Spacer(1, 10)])


def image(path, width=W, max_h=None):
    im = Image.open(path)
    w, h = im.size
    height = width * h / w
    if max_h and height > max_h:
        width, height = width * max_h / height, max_h
    return RLImage(str(path), width=width, height=height)


def section(kicker, title):
    return [P(kicker.upper(), "kicker"), P(title, "h1")]


# ------------------------------------------------------------------ charts
def make_charts(tmp: Path, summ, plan_weekly, bench_weekly, lb, comp) -> dict:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#64748b", "xtick.color": "#64748b",
                         "ytick.color": "#64748b", "axes.grid": True, "grid.color": "#eef1f6", "axes.axisbelow": True})
    out = {}
    # 1 cost race
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    wk = np.arange(3, 9)
    cb, cp = bench_weekly.loc[3:8, "cost"].cumsum(), plan_weekly.loc[3:8, "cost"].cumsum()
    ax.plot(wk, cb, color="#eb6834", lw=2, marker="o", ms=4)
    ax.plot(wk, cp, color="#2a78d6", lw=2.4, marker="o", ms=4)
    ax.fill_between(wk, cp, cb, color="#2a78d6", alpha=.08)
    ax.text(8.12, cb.iloc[-1], f"Official method €{cb.iloc[-1]:,.0f}", va="center", fontsize=8.5, color="#0f172a")
    ax.text(8.12, cp.iloc[-1], f"Our plan €{cp.iloc[-1]:,.0f}", va="center", fontsize=8.5, color="#0f172a")
    ax.set_xlim(2.8, 9.9); ax.set_xticks(wk); ax.set_xlabel("Competition week")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_title("Money spent so far, weeks 3-8 (lower is better)", loc="left", fontsize=10, fontweight="bold", color="#0f172a")
    fig.tight_layout(); out["race"] = tmp / "race.png"; fig.savefig(out["race"]); plt.close(fig)
    # 2 leaderboard
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=200)
    rank = [(n.replace("VN2 benchmark", "Official method"), v) for n, v in summ["ranking"]] + [("Our plan", summ["plan_score"])]
    rank.sort(key=lambda x: x[1], reverse=True)
    names, vals = [n for n, _ in rank], [v for _, v in rank]
    cols = ["#2a78d6" if n == "Our plan" else ("#eb6834" if n == "Official method" else "#c3cad6") for n in names]
    ax.barh(names, vals, color=cols, height=.65)
    for i, v in enumerate(vals):
        ax.text(v + 15, i, f"€{v:,.0f}", va="center", fontsize=8.5)
    ax.set_xlim(3000, max(vals) * 1.08); ax.grid(axis="y", visible=False)
    ax.set_title("Against the real competition's top 5 (cost, lower is better)", loc="left", fontsize=10, fontweight="bold",
                 color="#0f172a")
    fig.tight_layout(); out["rank"] = tmp / "rank.png"; fig.savefig(out["rank"]); plt.close(fig)
    # 3 forecast accuracy
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    d = lb.sort_values("Score %", ascending=False)
    cols = ["#2a78d6" if m == lb.iloc[0]["model"] else ("#eb6834" if m == "seasonal_ma" else "#c3cad6") for m in d["model"]]
    ax.barh(d["label"], d["Score %"], color=cols, height=.65)
    for i, v in enumerate(d["Score %"]):
        ax.text(v + .6, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, d["Score %"].max() * 1.12); ax.grid(axis="y", visible=False)
    ax.set_title("Forecast error on 30 past weeks (Score % = WAPE + |bias|, lower is better)", loc="left", fontsize=10,
                 fontweight="bold", color="#0f172a")
    fig.tight_layout(); out["acc"] = tmp / "acc.png"; fig.savefig(out["acc"]); plt.close(fig)
    # 4 VN2 score per method
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    d = comp.sort_values("VN2 score €", ascending=False)
    cols = ["#2a78d6" if m == summ["plan_model"] else "#c3cad6" for m in d["model"]]
    ax.barh(d["label"], d["VN2 score €"], color=cols, height=.65)
    ax.axvline(summ["benchmark_score"], color="#eb6834", lw=1.4, ls="--")
    ax.text(summ["benchmark_score"], len(d) - .45, " official method €4,334", color="#eb6834", fontsize=8)
    for i, v in enumerate(d["VN2 score €"]):
        ax.text(v + 20, i, f"€{v:,.0f}", va="center", fontsize=8.5)
    ax.set_xlim(3000, d["VN2 score €"].max() * 1.09); ax.grid(axis="y", visible=False)
    ax.set_title("Cost of the orders each forecast leads to, weeks 3-8", loc="left", fontsize=10, fontweight="bold",
                 color="#0f172a")
    fig.tight_layout(); out["comp"] = tmp / "comp.png"; fig.savefig(out["comp"]); plt.close(fig)
    return out


# ------------------------------------------------------------------ page decoration
def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE); canvas.setLineWidth(.5)
        canvas.line(2.1 * cm, A4[1] - 1.5 * cm, A4[0] - 2.1 * cm, A4[1] - 1.5 * cm)
        canvas.setFont(FB, 8); canvas.setFillColor(BLUE_INK)
        canvas.drawString(2.1 * cm, A4[1] - 1.3 * cm, "OptiChain")
        canvas.setFont(F, 8); canvas.setFillColor(MUTED)
        canvas.drawString(3.6 * cm, A4[1] - 1.3 * cm, "Project guide, A to Z")
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
        canvas.drawString(2.1 * cm, 1.2 * cm, "Data: VN2 Inventory Planning Challenge (Vandeput, 2025), CC BY 4.0")
    else:
        canvas.setFillColor(BLUE_INK); canvas.rect(0, A4[1] - 0.9 * cm, A4[0], 0.9 * cm, stroke=0, fill=1)
        canvas.setFillColor(BLUE); canvas.rect(0, A4[1] - 1.05 * cm, A4[0], 0.15 * cm, stroke=0, fill=1)
    canvas.restoreState()


# ------------------------------------------------------------------ content
def build(shots: Path | None):
    summ = json.loads((ART / "summary.json").read_text())
    ts = json.loads((ART / "ts_network.json").read_text())
    lb = pd.read_csv(ART / "leaderboard.csv")
    comp = pd.read_csv(ART / "competition_scores.csv")
    bench_weekly = pd.read_csv(ART / "benchmark_weekly.csv").set_index("week")
    wh = pd.read_csv(ART / "error_whiteness.csv")
    ar = pd.read_parquet(ART / "arima_rows.parquet")

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from optichain import data as D, inventory as I, planning as P_
    from optichain.config import load_config
    cfg = load_config()
    panel = D.load_panel(cfg.paths.raw_dir)
    rounds = pd.read_parquet(ART / "rounds.parquet")
    k = P_.k_vector(pd.read_parquet(ART / "dispersion.parquet"), summ["plan_model"])
    plan = P_.run_plan(I.State.from_initial(panel.initial), panel.sales.iloc[:, panel.t0 + 1:panel.t0 + 9].to_numpy(), rounds,
                       k, summ["plan_model"], summ["plan_service"], I.Costs(), cfg.policy.samples, cfg.seed)
    pw = plan.weekly
    saved = summ["benchmark_score"] - plan.score
    left_p, left_b = pw.loc[3:8, "end"].mean(), bench_weekly.loc[3:8, "end"].mean()
    lost_p, lost_b = pw.loc[3:8, "lost"].sum(), bench_weekly.loc[3:8, "lost"].sum()
    labels = {"naive": "Last week", "ma13": "13-week average", "snaive": "Same week last year",
              "seasonal_ma": "Official benchmark", "ses": "Exponential smoothing", "arima": "ARIMA (AIC-selected)",
              "croston_sba": "Croston SBA", "tsb": "TSB", "lightgbm": "LightGBM", "ensemble": "Ensemble"}
    sc = lambda m: lb.set_index("model").loc[m, "Score %"]
    cs = comp.set_index("model")["VN2 score €"]
    fam = ar["family"].value_counts(normalize=True)
    whs = wh.set_index(wh.columns[0])["white_noise_share"]

    tmp = Path(tempfile.mkdtemp())
    ch = make_charts(tmp, {**summ, "plan_score": plan.score}, pw, bench_weekly, lb.assign(label=lb["model"].map(labels)),
                     comp.assign(label=comp["model"].map(labels)))
    st = []

    # ---------------------------------------------------------------- cover
    st += [Spacer(1, 3.2 * cm), P("PROJECT GUIDE", "kicker"),
           P("OptiChain", "title"),
           P("Demand forecasting and stock planning for a real retail chain, explained from A to Z: what was done, "
             "what every number means, and why each choice was made.", "subtitle"), Spacer(1, 1.0 * cm),
           kpi_row([("Saved vs the official method", f"€{saved:,.0f}", f"in 6 weeks · {saved / summ['benchmark_score']:.0%} cheaper"),
                    ("Over a year (estimate)", f"≈ €{round(saved * 52 / 6, -2):,.0f}", "if the test weeks are typical"),
                    ("Stock on shelves", f"−{1 - left_p / left_b:.0%}", f"{left_p:,.0f} vs {left_b:,.0f} units a week"),
                    ("Store-products planned", f"{summ['rows']}", f"{summ['stores']} stores · {summ['products']} products")]),
           Spacer(1, .6 * cm),
           P("<b>Contents</b>", "h3"),
           Table([[P(f"<b>{a}</b> · {b}", "toc"), P(f"<b>{c}</b> · {d}", "toc")] for (a, b), (c, d) in zip(
               [("A", "The problem"), ("B", "The data"), ("C", "The pipeline"), ("D", "The 20 steps"), ("E", "The dashboard")],
               [("F", "The code"), ("G", "Results"), ("H", "Limitations"), ("I", "Questions and answers"), ("J", "How to run")])],
                 colWidths=[W / 2, W / 2], style=[("LEFTPADDING", (0, 0), (-1, -1), 0)]),
           Spacer(1, 2.5 * cm),
           P("Data: Vandeput, N. (2025). <i>VN2 Inventory Planning Challenge</i>. Organised by Nicolas Vandeput "
             "(SupChains); sponsored by Flieber, Daybreak and SupChains. https://supchains.com/vn2-inventory-competition/ "
             "· Licence CC BY 4.0.", "caption"),
           PageBreak()]

    # ---------------------------------------------------------------- A problem
    st += section("A", "The problem we are solving")
    st += [P("A shop chain must decide <b>every Monday how much stock to order</b> for each product in each shop. The "
             "order takes <b>two weeks</b> to arrive. Two mistakes cost money:"),
           table([["Mistake", "What happens", "Cost (official rule)"],
                  ["Order too little", "The shelf goes empty and the customer leaves", "<b>€1 per missed sale</b>"],
                  ["Order too much", "Boxes sit unsold on the shelf", "<b>€0.20 per unit per week</b>"]], [.24, .46, .30]),
           Spacer(1, 8),
           P("<b>Goal:</b> predict demand, then order the amount that makes the total cost as small as possible."),
           callout("<b>Why this matters.</b> Retailers lose money both ways: empty shelves lose customers, extra stock ties "
                   "up money and space. Balancing the two is the core problem of supply-chain inventory planning.")]

    # ---------------------------------------------------------------- B data
    st += section("B", "The data: the VN2 Inventory Planning Challenge")
    st += [P("<b>What it is.</b> A real competition held in 2025 by Nicolas Vandeput (SupChains), with about 180 teams. "
             "The data is <b>real weekly sales</b> of an anonymous retail chain, published on GitHub (SupChains/VN2)."),
           P("<b>Why we chose it over Walmart M5</b> (the project's first version, kept in <i>archive/m5</i>):"),
           *bullets(["M5 is older (2011-2016) and has <b>no stock data</b>, so stock levels and costs had to be invented.",
                     "VN2 is recent (2021-2024) and comes with <b>real stock, real costs, real rules</b> and an official "
                     "score to compare against.",
                     "It is free to use (CC BY 4.0) as long as the competition is cited."]),
           P("Size of the data", "h3"),
           table([["Item", "Value", "Meaning"],
                  ["Stores", str(summ["stores"]), "shops in the chain"],
                  ["Products", str(summ["products"]), "different items sold"],
                  ["Store-products", str(summ["rows"]), "one product in one store; not every store sells every product. "
                                                        "<b>Each is planned separately.</b>"],
                  ["Weeks", f"{summ['weeks']}", f"{summ['first_week']} to {summ['last_week']}"],
                  ["History", f"{summ['history_weeks']} weeks", "up to week 0 (8 April 2024), used for learning"],
                  ["Competition", "8 weeks", "weeks 1-8, used <b>only</b> for the final test"]], [.2, .17, .63]),
           P("The files", "h3"),
           table([["File", "What is inside"],
                  ["Sales (Week 0 ... Week 8)", "units sold per store-product per week; each file adds one week"],
                  ["In Stock", "True/False per week: was stock left at the end of the week?"],
                  ["Master", "groupings: product group, department, store format"],
                  ["Initial State", "the real stock on the shelf and orders on their way at the start"],
                  ["Benchmark orders", "the official simple method's six orders, used to verify our simulator"]], [.32, .68]),
           callout("<b>Why stores and products are numbers.</b> The company hid its identity. The dashboard therefore gives "
                   "five example products descriptive nicknames (\"The Christmas Rocket\") based on how they behave, while "
                   "always showing the real code next to them. Nothing is invented.")]

    # ---------------------------------------------------------------- C pipeline
    st += [PageBreak()] + section("C", "The whole pipeline in one picture")
    flow = [["1", "Load the data and check its quality"], ["2", "Explore it with charts and statistics"],
            ["3", "Find and handle hidden missing weeks"],
            ["4-11", "Classical time-series analysis: stationarity, seasonality, ACF/PACF, ARIMA, AIC/BIC, residual tests, GARCH"],
            ["12-13", "Ten forecasting methods, tested fairly on 30 past weeks"],
            ["14", "Measure how uncertain each forecast is"], ["15", "Simulate the shop with the official rules"],
            ["16", "Decide how much to order (Monte Carlo newsvendor)"], ["17", "Final test on the 8 real competition weeks"],
            ["18", "Recommend next week's order"], ["19-20", "Save results for the dashboard and conclude"]]
    st += [table([["Step", "What happens"]] + flow, [.12, .88], bold_first=True), Spacer(1, 8),
           P("Everything lives in <b>one notebook</b>, <i>notebooks/OptiChain_VN2_Pipeline.ipynb</i>, in 20 explained "
             "steps. The dashboard only displays its results, except the stock simulation, which it re-runs live when you "
             "change a setting.")]

    # ---------------------------------------------------------------- D steps
    st += [PageBreak()] + section("D", "The 20 steps, one by one")
    st += [P("Step 1 · Data collection and audit", "h2"),
           *bullets(["<b>What:</b> download the files if missing, then check: no empty cells, no negative sales, only whole "
                     "numbers, and all nine sales files agree with each other.",
                     "<b>Why:</b> if the data is broken, every later result is wrong.",
                     "<b>Found:</b> 43% of weeks have zero sales (most products sell slowly); about 1 week in 10 is flagged "
                     "out of stock."]),
           P("Step 2 · Exploratory data analysis (EDA)", "h2"),
           P("Looking at the data before modelling. Each chart answers one question:"),
           table([["Chart", "Question", "What we found"],
                  ["Line graph of total sales", "Is there a trend or season?", "Stable level with a strong Christmas peak each year"],
                  ["Seasonal profile", "Which weeks of the year sell more?", "Weeks 48-52 sell up to 2.4× an average week"],
                  ["Histograms", "How much does a typical product sell?", "Median about 1 unit a week; very skewed"],
                  ["Pareto curve", "Where is the volume?", "A few products make most of the sales"],
                  ["Scatter plots", "Volume vs variability; this week vs last week", "Slow sellers are most erratic; this week follows last week"],
                  ["QQ plots", "Are sales normally distributed?", "No: heavy tails, especially single products"],
                  ["Descriptive statistics", "Mean, spread, skewness, kurtosis", "Strong right skew, many zeros; Jarque-Bera rejects normality"],
                  ["Stockout charts", "Which stores run out?", "From 0% to 59% of weeks, depending on the store"]], [.25, .33, .42]),
           callout("<b>Why it matters:</b> slow, zero-heavy demand needs special methods (Croston, TSB), and non-normal data "
                   "needs a count distribution rather than a bell curve. The EDA decided both."),
           P("Step 3 · Missing values", "h2"),
           P("The file has no empty cells, but two kinds of weeks are <b>hidden missing values</b>:"),
           *bullets(["<b>Before launch.</b> A product shows 0 before it was ever sold in that store. Those zeros are not "
                     "\"nobody wanted it\": the product did not exist yet. Every week before the first sale is treated as missing.",
                     "<b>Out-of-stock weeks.</b> <b>100% of out-of-stock weeks recorded zero sales</b>: the shelf was empty, "
                     "so nothing could sell. That zero measures supply, not demand (statisticians call it <i>censored "
                     "demand</i>), so it is treated as missing."]),
           P("Why not fill the gaps with zero or a guess? A zero would teach the models that demand is lower than it is, so "
             "they would under-order; a guess would look like real information. Each method handles the gap honestly:"),
           table([["Method", "Treatment of a missing week"],
                  ["Averages, smoothing, Croston, TSB", "skip it"],
                  ["ARIMA / SARIMA", "kept missing: the Kalman filter skips it exactly"],
                  ["ADF, KPSS, STL, ACF (need an unbroken series)", "straight-line interpolation between the neighbouring weeks"],
                  ["LightGBM", "never learned as a target"]], [.48, .52]),
           P("<b>Demand patterns.</b> Each store-product is also labelled by two numbers: <b>ADI</b> (average weeks between "
             "sales) and <b>CV²</b> (how much sale sizes vary). Together they give four types: <b>steady, spiky, occasional, "
             "unpredictable</b> (Syntetos-Boylan classification)."),
           ]

    st += [P("Step 4 · Stationarity (ADF test)", "h2"),
           *bullets(["<b>Stationary</b> means the average and the spread do not drift over time. ARIMA-type models need it.",
                     "<b>ADF</b> asks \"does the series wander like a random walk?\" A p-value below 0.05 means stationary. "
                     "<b>KPSS</b> asks the opposite question, as a second opinion.",
                     f"<b>Result:</b> the network total is stationary by ADF; <b>{ts['adf_rows_stationary']:.0%} of the 599 "
                     "store-products</b> are stationary."]),
           P("Step 5 · Transformations", "h2"),
           P("If a series is not stationary, transform it: a <b>log</b> (calms a growing spread), a <b>difference</b> (this week "
             f"minus last week, removes drift) or a <b>seasonal difference</b> (minus the same week last year). All were tested; "
             f"<b>no differencing was needed (d = {ts['d']})</b>, so the original units were kept, which is also what ordering needs."),
           P("Step 6 · Seasonality (STL decomposition)", "h2"),
           P("STL splits the series into <b>trend</b> (slow movement), <b>season</b> (the yearly repeat) and <b>remainder</b> "
             f"(noise). Seasonal strength <b>{ts['strength']['seasonal strength']:.2f}</b> on a 0-1 scale: very seasonal (Christmas). "
             "Most single products are too noisy and too short to estimate their own season, so the models take the season "
             "from the whole network and apply it to each product."),
           P("Step 7 · ACF and PACF", "h2"),
           *bullets(["<b>ACF</b>: how strongly this week relates to k weeks ago.",
                     "<b>PACF</b>: the same, keeping only the direct link (the weeks in between removed).",
                     "Found: strong links at lags 1-5, and around lag 52 (one year back)."]),
           P("Step 8 · Identify the model", "h2"),
           table([["What the plots show", "Model"],
                  ["PACF cuts off, ACF fades", "<b>AR</b>: depends on past values"],
                  ["ACF cuts off, PACF fades", "<b>MA</b>: depends on past errors"],
                  ["Both fade", "<b>ARMA</b>"], ["Differencing was needed", "<b>ARIMA</b>"],
                  ["Spike at the seasonal lag (52)", "<b>SARIMA</b>"]], [.5, .5]),
           P("For the network total: an AR pattern plus a yearly spike, so SARIMA."),
           P("Step 9 · Lags", "h2"),
           P("The useful lags are <b>1-5</b> (recent weeks) and <b>52</b> (the same week last year). This also chose the inputs "
             "of the machine-learning model in step 12."),
           P("Step 10 · AIC and BIC", "h2"),
           *bullets(["<b>AIC</b> = fit quality minus a penalty for complexity; <b>BIC</b> = the same with a stricter penalty. "
                     "Lower is better.",
                     f"<b>Network total:</b> 27 candidates compared; the winner is <b>{ts['best']}</b> (BIC prefers {ts['best_bic']}). "
                     "This week depends on the last two weeks and on the same week last year.",
                     f"<b>Every store-product</b> gets its own ARIMA: ADF picks d, then AIC picks p and q. Result: "
                     f"{fam.get('ARMA', 0):.0%} ARMA, {fam.get('AR', 0):.0%} AR, {fam.get('MA', 0):.0%} MA, "
                     f"{fam.get('ARIMA', 0):.0%} ARIMA, and {fam.get('mean only', 0):.0%} \"mean only\" (slow sellers with "
                     "so little signal that a plain average wins)."]),
           ]

    rt, gt = ts["resid_tests"], ts["garch_tests"]
    gkeys = list(gt)
    st += [P("Step 11 · Residual diagnostics, ARCH and GARCH", "h2"),
           P("<b>Residuals</b> are what the model could not explain. A good model leaves only random noise (<b>white noise</b>)."),
           table([["Test", "Question", "Rule"],
                  ["Ljung-Box", "Is any pattern left?", "p above 0.05: white noise"],
                  ["Jarque-Bera", "Are the residuals normal?", "p above 0.05: normal"],
                  ["ARCH-LM", "Do calm and turbulent periods alternate?", "p below 0.05: yes, GARCH is needed"]], [.2, .45, .35]),
           Spacer(1, 6),
           table([["Network total", "SARIMA residuals", "After GARCH(1,1)"],
                  ["Ljung-Box p (lag 8)", f"{rt['Ljung-Box p (lag 8)']:.3f}", f"{gt[gkeys[1]]['Ljung-Box p (lag 8)']:.3f}"],
                  ["Ljung-Box p (lag 13)", f"{rt['Ljung-Box p (lag 13)']:.3f}", f"{gt[gkeys[1]]['Ljung-Box p (lag 13)']:.3f}"],
                  ["ARCH-LM p", f"{rt['ARCH-LM p']:.3f}", f"{gt[gkeys[1]]['ARCH-LM p']:.3f}"],
                  ["White noise?", "no" if not rt["white noise"] else "yes", "yes" if gt[gkeys[1]]["white noise"] else "no"],
                  ["ARCH effect?", "yes" if rt["ARCH effect"] else "no", "yes" if gt[gkeys[1]]["ARCH effect"] else "no"]],
                 [.4, .3, .3], bold_first=True),
           Spacer(1, 6),
           *bullets(["The SARIMA residuals showed an <b>ARCH effect</b>: the noise is bigger in some periods (around Christmas).",
                     "<b>GARCH(1,1)</b> models that changing noise level. After it, Ljung-Box passes and the ARCH effect is gone.",
                     "Normality is still rejected because of a few extreme weeks. That does not bias the forecast; it only "
                     "affects how wide the intervals should be.",
                     f"<b>Test on unseen weeks:</b> the SARIMA forecast the 8 competition weeks with <b>{ts['holdout_mape']:.0f}% error</b>, "
                     f"against {ts['holdout_mape_snaive']:.0f}% for \"same week last year\", and every real week fell inside its 80% range.",
                     f"<b>All 599 store-products:</b> {ts['rows_white_noise']:.0%} have white-noise residuals; {ts['rows_arch']:.0%} showed "
                     f"an ARCH effect, and GARCH removed it in {ts['garch_rows_fixed']:.0%} of those."]),
           ]

    st += [P("Step 12 · Ten forecasting methods", "h2"),
           P("Ten methods are compared so that the choice is proven, not assumed."),
           table([["Method", "Idea", "Why include it"],
                  ["Last week", "Next week = last week", "Simplest baseline"],
                  ["13-week average", "Average of the last three months", "Simple and stable"],
                  ["Same week last year", "Copy last year", "Captures the season simply"],
                  ["<b>Official benchmark</b>", "13-week average × seasonal factor", "The competition's own method: the target to beat"],
                  ["Exponential smoothing", "Weighted average that trusts recent weeks more", "Classic statistical method"],
                  ["<b>ARIMA</b>", "The Box-Jenkins model of steps 4-11, one per store-product", "The classical time-series approach"],
                  ["Croston SBA", "Predicts sale size and gap between sales separately", "Built for slow sellers"],
                  ["TSB", "Like Croston, but notices products that stop selling", "Built for slow sellers"],
                  ["<b>LightGBM</b>", "One machine-learning model trained on all 599 at once", "Learns shared patterns; the VN2 winners used this type"],
                  ["<b>Ensemble</b>", "Average of LightGBM, smoothing and the benchmark", "Combining usually reduces error"]],
                 [.24, .44, .32]),
           Spacer(1, 6),
           P("<b>LightGBM inputs:</b> the last 8 weeks of demand, rolling averages, the same week last year, the season factor, "
             "in-stock flags, weeks since launch, and the store and product groups. Its loss function is <b>Tweedie</b>, "
             "designed for data with many zeros; it beat the alternatives in a test on past weeks."),
           P("Step 13 · Backtest: testing on the past, fairly", "h2"),
           *bullets(["At <b>30 past weeks</b>, each method sees <b>only the data before that week</b> and predicts the next "
                     "three weeks; the prediction is then compared with what really sold.",
                     "<b>Why three weeks?</b> An order placed now arrives in two weeks, so the planner needs this week, next week "
                     "and the arrival week.",
                     "<b>Why only past data?</b> Using future data would be cheating (<i>leakage</i>) and make results look "
                     "better than reality."]),
           table([["Metric", "Meaning"],
                  ["MAE", "average miss, in units"], ["WAPE", "total error ÷ total sales"],
                  ["Bias", "do we guess too high (+) or too low (−) overall?"],
                  ["Score = WAPE + |Bias|", "the official metric of the VN1 forecasting competition; lower is better"]], [.3, .7]),
           Spacer(1, 8), image(ch["acc"]),
           P("Why do errors look large (about 60%)? Most products sell 1-2 units a week: guessing 2 when 1 sells is already "
             "\"100% wrong\". Errors for slow sellers always look big in percentage terms.", "caption"),
           P(f"<b>Ljung-Box on the real forecast errors:</b> for most methods, {whs.drop('naive', errors='ignore').min():.0%}-"
             f"{whs.max():.0%} of store-products leave no pattern in their mistakes; \"Last week\" only {whs.get('naive', 0):.0%}. "
             "A pass shows that no pattern is left; it does not by itself prove accuracy, which the leaderboard measures."),
           ]

    st += [P("Step 14 · Uncertainty: how wrong can a forecast be?", "h2"),
           *bullets(["<b>Why:</b> ordering exactly the average runs out about half the time. The spread matters.",
                     "<b>What:</b> demand is modelled as a <b>negative binomial distribution</b>, which suits counts (whole "
                     "numbers, many zeros, occasional spikes). Its mean is the forecast; its spread <b>k</b> is learned per product "
                     "from past errors.",
                     "<b>Calibration check:</b> when the model says \"80% chance demand stays below X\", it does about 85% of the "
                     "time: slightly cautious, which step 16 corrects."]),
           P("Step 15 · The inventory simulator", "h2"),
           P("Our code plays the exact official rules every week:"),
           P("stock at start = last week's leftover + the arriving order<br/>sales = smaller of (stock, demand)<br/>"
             "missed sales = demand − sales (lost, no back-orders)<br/>cost = €0.20 × leftover + €1 × missed sales", "code"),
           callout(f"<b>Verified.</b> With the organiser's own benchmark orders, our simulator gives exactly "
                   f"<b>€{summ['benchmark_score']:,.1f}</b>, the official number, and our rebuild of the benchmark places the same "
                   "orders. Every score in this project is therefore directly comparable with the real leaderboard."),
           P("Step 16 · The ordering policy: how much to order", "h2"),
           P("<b>Monte Carlo newsvendor</b>, for each store-product, every week:"),
           *bullets(["Simulate <b>1,000 possible futures</b> of demand for the next three weeks.",
                     "In each future, play the two weeks before the order arrives, to see how much stock will be left.",
                     "Need = demand in the arrival week − stock left.",
                     "Order enough to cover that need in a chosen share of futures: the <b>service level</b>."]),
           P(f"The textbook service level is 1 ÷ (1 + 0.2) = <b>0.833</b>. Instead, the level was <b>tuned by replaying 18 past "
             f"weeks</b>: <b>{summ['plan_service']}</b> gave the lowest cost. It sits below the textbook value because the spread "
             "estimate is slightly cautious and leftover stock is not wasted (it sells next week). Simulation captures both, "
             "which is what the competition's top teams did."),
           ]

    st += [P("Step 17 · Final test: the eight real competition weeks", "h2"),
           P("Each week the models were retrained on the data known up to that week; the service level was tuned on the past, "
             "<b>never on these weeks</b>."),
           image(ch["race"]), Spacer(1, 4), image(ch["rank"]), Spacer(1, 4),
           table([["Plan", "Cost, weeks 3-8"],
                  [f"<b>Our plan ({labels[summ['plan_model']]} @ {summ['plan_service']})</b>", f"<b>€{plan.score:,.0f}</b>"],
                  ["LightGBM plan", f"€{cs['lightgbm']:,.0f}"], ["ARIMA plan", f"€{cs['arima']:,.0f}"],
                  ["Real 1st place", f"€{summ['winner_score']:,.0f}"], ["<b>Official benchmark</b>", f"<b>€{summ['benchmark_score']:,.0f}</b>"]],
                 [.6, .4]),
           callout(f"<b>Two honest points.</b> (1) Hindsight: this was built after the competition, knowing which methods worked, "
                   "so \"we beat the official method by 16%\" is fair, \"we beat 1st place\" is not fully fair. (2) On these 8 weeks "
                   "the simple benchmark <i>forecast</i> was slightly more accurate, yet our <i>orders</i> still cost less: short "
                   "test windows shuffle rankings.", "warn"),
           P(f"<b>The trade-off:</b> the plan keeps <b>{1 - left_p / left_b:.0%} less stock</b> on the shelves "
             f"({left_p:,.0f} vs {left_b:,.0f} units a week) and accepts more missed sales ({lost_p:,.0f} vs {lost_b:,.0f} units). "
             "Under the €1 vs €0.20 rules, that is the cheaper deal overall."),
           image(ch["comp"]),
           P("Step 18 · Stock outlook: the next order", "h2"),
           P(f"The data ends on 3 June 2024, so the plan treats <b>10 June 2024 as \"today\"</b>: it orders "
             f"<b>{plan.next_order.sum():,.0f} units for {(plan.next_order > 0).sum()} store-products</b>, arriving on 24 June "
             "(two weeks later), and flags which products might run out. Today's real date cannot be used: there is no data "
             "after June 2024, so a plan for today would be invented."),
           P("Steps 19-20 · Save and conclude", "h2"),
           P("All results are saved to <i>artifacts/</i>, which the dashboard reads; the notebook ends with a summary and its limitations."),
           ]

    # ---------------------------------------------------------------- E dashboard
    st += [PageBreak()] + section("E", "The dashboard: seven pages")
    pages = [("overview", "Overview", "Did it work?",
              "Money headlines (saved, per year, stock on shelves, units to order); five hero products; a four-step story "
              "per product (sold, predicted, ordered, shelf result); the result against the competition."),
             ("demand", "Demand", "How does a product sell?",
              "Sales history (orange bands = empty shelf), yearly seasonality, a map of sales patterns, stockouts by store and month."),
             ("forecast", "Forecast", "Which method predicts best?",
              "Ranking of the 10 methods, a \"try it\" forecast for any product and date with an 80% range, error by "
              "pattern, LightGBM's most important inputs."),
             ("check", "Model check", "Are the statistics sound?",
              "Tabs for ADF stationarity, STL season and trend, ACF/PACF, AIC/BIC model choice, residuals and GARCH, "
              "forecast check; for the whole network or live for any product."),
             ("planner", "Stock planner", "What if I change the settings?",
              "Choose a method and a service level; the eight weeks replay live; a cost curve shows the sweet spot; "
              "a week-by-week drill-down; download the orders."),
             ("outlook", "Order outlook", "What do I order this week?",
              "The order total, a risk chart, every store-product with the riskiest first, CSV download."),
             ("method", "Method", "How was it built?",
              "The rules, a glossary, calibration, limitations and the data citation.")]
    st += [table([["Page", "Question it answers", "What is on it"]] + [[f"<b>{t}</b>", q, d] for _, t, q, d in pages],
                 [.17, .25, .58]),
           Spacer(1, 6),
           P("<b>Design choices:</b> blue is always our plan, orange the official method, grey is context; every chart has a "
             "plain-language note; the palette is colour-blind safe.")]
    if shots:
        for key, title, q, _ in pages:
            f = shots / f"pw_{key}.png"
            if f.exists():
                st += [PageBreak(), P(f"Dashboard · {title}", "h2"), P(q, "caption"), image(f, max_h=21.5 * cm)]

    # ---------------------------------------------------------------- F code
    st += [PageBreak()] + section("F", "The code: where everything lives")
    st += [table([["File", "Job"],
                  ["config.yaml", "all settings in one place: costs, models, test windows"],
                  ["src/optichain/data.py", "load files, find launches and stockouts, classify demand patterns"],
                  ["src/optichain/timeseries.py", "ADF, KPSS, STL, ACF/PACF, AIC/BIC, Ljung-Box, ARCH, GARCH, per-product ARIMA"],
                  ["src/optichain/models.py", "the forecasting methods, including LightGBM"],
                  ["src/optichain/evaluation.py", "backtest and accuracy metrics"],
                  ["src/optichain/inventory.py", "official simulator, demand sampling, newsvendor orders"],
                  ["src/optichain/planning.py", "weekly rounds and the live plan"],
                  ["scripts/build_notebook.py", "writes the notebook (edit this, not the .ipynb)"],
                  ["scripts/build_guide.py", "writes this PDF"],
                  ["app/", "the dashboard: Home.py, one file per page, ui.py design system, data.py loaders"],
                  ["tests/", "automatic tests, including \"the simulator gives exactly €4,334\""],
                  ["archive/m5/", "the earlier Walmart version, kept, not deleted"]], [.36, .64])]

    # ---------------------------------------------------------------- G results
    st += section("G", "Results at a glance")
    st += [table([["Question", "Answer"],
                  ["Best forecaster (30 past weeks)", f"LightGBM, error {sc('lightgbm'):.1f}% (official benchmark {sc('seasonal_ma'):.1f}%)"],
                  ["Best plan (8 real weeks)", f"{labels[summ['plan_model']]} at service level {summ['plan_service']}: €{plan.score:,.0f}"],
                  ["Saving vs the official method", f"€{saved:,.0f} ({saved / summ['benchmark_score']:.0%}) in 6 weeks; ≈ €{round(saved * 52 / 6, -2):,.0f} a year"],
                  ["Network time-series model", f"{ts['best']}; residuals white noise after GARCH(1,1)"],
                  ["Per-product ARIMA residuals", f"{ts['rows_white_noise']:.0%} white noise (Ljung-Box)"],
                  ["Simulator check", f"reproduces the official €{summ['benchmark_score']:,.1f} exactly"]], [.38, .62], bold_first=True)]

    # ---------------------------------------------------------------- H limitations
    st += section("H", "Limitations")
    st += bullets(["<b>Hindsight:</b> the design was informed by the published competition results.",
                   "<b>Censored demand:</b> the simulation uses recorded sales as demand, so weeks that ran out in reality "
                   "understate true demand (the competition used the same convention).",
                   "<b>Short test:</b> eight weeks is short; a few big products can move the score.",
                   "<b>Anonymous data:</b> what the products really are is unknown.",
                   "<b>Simple uncertainty model:</b> quantile models could calibrate the risk ranges better."])

    # ---------------------------------------------------------------- I Q&A
    st += [PageBreak()] + section("I", "Questions you might be asked")
    qa = [("Why VN2 and not Walmart?", "It is recent, real, and has real stock levels, real costs and an official score to compare against."),
          ("Why remove out-of-stock weeks?", "All of them show zero sales: those zeros measure an empty shelf, not customer demand."),
          ("What does \"store-product\" mean?", "One product in one store. 67 stores × 297 products, but only 599 pairs exist; each is planned separately."),
          ("Why is ARIMA not the final model?", "It was tested fairly: LightGBM and the ensemble were more accurate and cheaper. ARIMA still beat the official method."),
          ("What does Ljung-Box prove?", "That no pattern is left in the errors. Accuracy is proven separately, on unseen weeks."),
          ("Why GARCH?", "The noise was bigger in some periods (Christmas). GARCH models that, and the tests then pass."),
          ("Why service level 0.75 and not 0.833?", "It was tuned by simulating real costs on past weeks, which beats trusting the formula."),
          ("Why does the plan miss more sales than the official method?", "Because a missed sale costs €1 but a leftover costs €0.20 a week; keeping 45% less stock is worth some extra misses, and the total is lower."),
          ("Why does the dashboard say 24 June 2024?", "That is the first planning week after the data ends; an order placed on 10 June arrives two weeks later."),
          ("Is the comparison with 1st place fair?", "Not fully: the project was built after the competition. The fair claim is the 16% saving over the official method.")]
    st += [table([["Question", "Short answer"]] + [[f"<b>{q}</b>", a] for q, a in qa], [.38, .62])]

    # ---------------------------------------------------------------- J run
    st += section("J", "How to run it")
    st += [P("python -m venv .venv<br/>.venv/Scripts/python -m pip install -r requirements.txt<br/>"
             ".venv/Scripts/python -m pip install -e .<br/><br/>"
             "python scripts/build_notebook.py --run &nbsp;&nbsp;# the whole pipeline, about 20 minutes<br/>"
             "streamlit run app/Home.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# the dashboard, http://localhost:8501<br/>"
             "python -m pytest &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# the tests<br/>"
             "python scripts/build_guide.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# this PDF", "code")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.0 * cm,
                            bottomMargin=1.8 * cm, title="OptiChain: project guide, A to Z", author="OptiChain",
                            subject="Demand forecasting and stock planning on the VN2 Inventory Planning Challenge")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, default=None, help="folder with pw_<page>.png dashboard screenshots")
    build(ap.parse_args().shots)
