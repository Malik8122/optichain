"""Build docs/OptiChain_Complete_Walkthrough.pdf: the complete, in-depth walkthrough of the project in one PDF.

Every idea in plain words, then its formula, its real result and its graph: the problem, the data, cleaning, all EDA
graphs, the full time-series (Box-Jenkins) analysis, all ten forecasting models with their equations, the backtest,
uncertainty, the simulator, the ordering policy, the final test, the outlook, every dashboard page, the code, the
settings, the tests, limitations, a glossary and Q&A. The cell-by-cell notebook guide is appended at the end.

    python scripts/build_notebook_guide.py      # the appendix (run first)
    python scripts/build_complete_report.py

Numbers and graphs are read from artifacts/ and from the executed notebook, so they always match the last run.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import build_guide as G  # noqa: E402  (shared fonts, styles, tables and charts)

ART = ROOT / "artifacts"
NB = ROOT / "notebooks" / "OptiChain_VN2_Pipeline.ipynb"
SHOTS = ROOT / "docs" / "assets" / "dashboard"
APPENDIX = ROOT / "docs" / "OptiChain_Notebook_Cell_by_Cell.pdf"
OUT = ROOT / "docs" / "OptiChain_Complete_Walkthrough.pdf"

S, W, F, FB = G.S, G.W, G.F, G.FB
P, bullets, table, callout, kpi_row, image = G.P, G.bullets, G.table, G.callout, G.kpi_row, G.image
S["eq"] = ParagraphStyle("eq", fontName=F, fontSize=10.5, leading=16, textColor=G.INK, alignment=1,
                         backColor=colors.HexColor("#f6f8fb"), borderPadding=7, spaceBefore=4, spaceAfter=10)
S["part"] = ParagraphStyle("part", fontName=FB, fontSize=26, leading=31, textColor=G.INK, spaceAfter=10)
S["fig"] = ParagraphStyle("fig", fontName=F, fontSize=8.8, leading=12.5, textColor=G.INK2, spaceBefore=3, spaceAfter=12)
S["h2"].keepWithNext = 1
S["h3"].keepWithNext = 1
S["h1"].keepWithNext = 1

LABELS = {"naive": "Last week", "ma13": "13-week average", "snaive": "Same week last year",
          "seasonal_ma": "Official benchmark (seasonal average)", "ses": "Exponential smoothing",
          "arima": "ARIMA (AIC-selected)", "croston_sba": "Croston SBA", "tsb": "TSB", "lightgbm": "LightGBM",
          "ensemble": "Ensemble"}


def EQ(text):
    return Paragraph(text, S["eq"])


def H(n, title):
    return [PageBreak(), P(f"PART {n}", "kicker"), Paragraph(title, S["part"])]


# ---------------------------------------------------------------- graphs from the executed notebook
_nb = nbf.read(NB, as_version=4)
_fig_no = [0]


def nb_figure(chart: str, caption: str, which: int = 0, max_h: float = 10.5 * cm):
    """A graph from the executed notebook, found by the chart function its cell calls (for example "pareto")."""
    cell = next(c for c in _nb.cells if c.cell_type == "code" and f"plots.{chart}(" in c.source)
    imgs = [o["data"]["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})]
    raw = base64.b64decode(imgs[which])
    from PIL import Image
    w, h = Image.open(io.BytesIO(raw)).size
    width = W
    height = width * h / w
    if height > max_h:
        width, height = width * max_h / height, max_h
    _fig_no[0] += 1
    return KeepTogether([RLImage(io.BytesIO(raw), width=width, height=height),
                         Paragraph(f"<b>Figure {_fig_no[0]}.</b> {caption}", S["fig"])])


def png_figure(path, caption, max_h=10.5 * cm):
    _fig_no[0] += 1
    return KeepTogether([image(path, max_h=max_h), Paragraph(f"<b>Figure {_fig_no[0]}.</b> {caption}", S["fig"])])


def df_table(df: pd.DataFrame, widths=None, fmt=None, first_bold=True):
    fmt = fmt or {}
    rows = [list(df.columns)]
    for _, r in df.iterrows():
        rows.append([fmt.get(c, lambda v: v)(r[c]) if c in fmt else r[c] for c in df.columns])
    return table(rows, widths or [1 / len(df.columns)] * len(df.columns), bold_first=first_bold)


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(G.LINE); canvas.setLineWidth(.5)
        canvas.line(2.1 * cm, A4[1] - 1.5 * cm, A4[0] - 2.1 * cm, A4[1] - 1.5 * cm)
        canvas.setFont(FB, 8); canvas.setFillColor(G.BLUE_INK)
        canvas.drawString(2.1 * cm, A4[1] - 1.3 * cm, "OptiChain")
        canvas.setFont(F, 8); canvas.setFillColor(G.MUTED)
        canvas.drawString(3.6 * cm, A4[1] - 1.3 * cm, "Complete walkthrough")
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
        canvas.drawString(2.1 * cm, 1.2 * cm, "Data: VN2 Inventory Planning Challenge (Vandeput, 2025), CC BY 4.0")
    else:
        canvas.setFillColor(G.BLUE_INK); canvas.rect(0, A4[1] - 0.9 * cm, A4[0], 0.9 * cm, stroke=0, fill=1)
        canvas.setFillColor(G.BLUE); canvas.rect(0, A4[1] - 1.05 * cm, A4[0], 0.15 * cm, stroke=0, fill=1)
    canvas.restoreState()


# ---------------------------------------------------------------- build
def build():
    from optichain import data as D, inventory as I, planning as PL
    from optichain.config import load_config

    cfg = load_config()
    summ = json.loads((ART / "summary.json").read_text())
    ts = json.loads((ART / "ts_network.json").read_text())
    lb = pd.read_csv(ART / "leaderboard.csv")
    lbh = pd.read_csv(ART / "leaderboard_h.csv")
    lbp = pd.read_csv(ART / "leaderboard_pattern.csv")
    lbc = pd.read_csv(ART / "leaderboard_competition.csv")
    comp = pd.read_csv(ART / "competition_scores.csv")
    tuning = pd.read_csv(ART / "tuning.csv")
    calib = pd.read_csv(ART / "calibration.csv")
    wh = pd.read_csv(ART / "error_whiteness.csv")
    imp = pd.read_csv(ART / "importance.csv")
    audit = pd.read_csv(ART / "audit.csv")
    classes = pd.read_parquet(ART / "classes.parquet")
    ar = pd.read_parquet(ART / "arima_rows.parquet")
    bench_w = pd.read_csv(ART / "benchmark_weekly.csv").set_index("week")
    outlook = pd.read_csv(ART / "outlook.csv")
    panel = D.load_panel(cfg.paths.raw_dir)
    rounds = pd.read_parquet(ART / "rounds.parquet")
    k = PL.k_vector(pd.read_parquet(ART / "dispersion.parquet"), summ["plan_model"])
    plan = PL.run_plan(I.State.from_initial(panel.initial), panel.sales.iloc[:, panel.t0 + 1:panel.t0 + 9].to_numpy(),
                       rounds, k, summ["plan_model"], summ["plan_service"], I.Costs(), cfg.policy.samples, cfg.seed)
    pw = plan.weekly
    saved = summ["benchmark_score"] - plan.score
    left_p, left_b = pw.loc[3:8, "end"].mean(), bench_w.loc[3:8, "end"].mean()
    lost_p, lost_b = pw.loc[3:8, "lost"].sum(), bench_w.loc[3:8, "lost"].sum()
    sc = lb.set_index("model")["Score %"]
    cs = comp.set_index("model")
    fam = ar["family"].value_counts()
    pat = classes["pattern"].value_counts()
    whs = wh.set_index(wh.columns[0])["white_noise_share"]
    a = dict(zip(audit["fact"], audit["value"]))

    tmp = Path(tempfile.mkdtemp())
    ch = G.make_charts(tmp, {**summ, "plan_score": plan.score}, pw, bench_w, lb.assign(label=lb["model"].map(LABELS)),
                       comp.assign(label=comp["model"].map(LABELS)))

    st = []
    # ================================================================ cover
    st += [Spacer(1, 2.6 * cm), P("COMPLETE WALKTHROUGH", "kicker"), P("OptiChain", "title"),
           P("Demand forecasting and stock planning for a real retail chain. Every step, every graph, every model and "
             "every number of the project, explained in depth: what it is, how it works, and why it was chosen.", "subtitle"),
           Spacer(1, .8 * cm),
           kpi_row([("Saved vs the official method", f"€{saved:,.0f}", f"in 6 weeks · {saved / summ['benchmark_score']:.0%} cheaper"),
                    ("Over a year (estimate)", f"≈ €{round(saved * 52 / 6, -2):,.0f}", "if the test weeks are typical"),
                    ("Stock on shelves", f"−{1 - left_p / left_b:.0%}", f"{left_p:,.0f} vs {left_b:,.0f} units a week"),
                    ("Forecast error", f"{sc['lightgbm']:.1f}%", f"LightGBM · benchmark {sc['seasonal_ma']:.1f}%")]),
           Spacer(1, .4 * cm), P("<b>Contents</b>", "h3")]
    toc = ["1 · The problem", "2 · The data", "3 · Data audit and cleaning", "4 · Exploratory data analysis",
           "5 · Time-series analysis (Box-Jenkins)", "6 · The ten forecasting models", "7 · Testing the forecasts (backtest)",
           "8 · Uncertainty", "9 · The inventory simulator", "10 · The ordering policy", "11 · The final test",
           "12 · The stock outlook", "13 · The dashboard, page by page", "14 · The engineering",
           "15 · Limitations and next steps", "Appendix A · Glossary", "Appendix B · Questions and answers",
           "Appendix C · The notebook, cell by cell"]
    half = (len(toc) + 1) // 2
    tt = Table([[P(toc[i], "toc"), P(toc[i + half], "toc") if i + half < len(toc) else ""] for i in range(half)],
               colWidths=[W / 2, W / 2])
    tt.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    st += [tt, Spacer(1, .8 * cm),
           P("Data: Vandeput, N. (2025). <i>VN2 Inventory Planning Challenge</i>. Organised by Nicolas Vandeput (SupChains); "
             "sponsored by Flieber, Daybreak and SupChains. https://supchains.com/vn2-inventory-competition/ · CC BY 4.0.",
             "caption")]

    # ================================================================ 1 problem
    st += H(1, "The problem")
    st += [P("1.1 · The business situation", "h2"),
           P("A retail chain has 67 stores. Each store sells some of 297 products. Every Monday, before the stores open, the "
             "planner orders stock for every product in every store from the central warehouse. The order is delivered "
             "<b>two weeks later</b>. Customers then buy what is on the shelf. If the shelf is empty, the customer leaves and "
             "the sale is <b>lost</b> (there is no waiting list, no back-order)."),
           P("1.2 · The two costs", "h2"),
           table([["Event", "Cost", "Real meaning"],
                  ["A unit is still on the shelf at the end of a week", "€0.20 per unit per week",
                   "money tied up in stock, shelf space, risk of damage or expiry (the <b>holding cost</b>)"],
                  ["A customer wants a unit but the shelf is empty", "€1.00 per unit",
                   "lost margin and an unhappy customer (the <b>shortage cost</b>)"]], [.36, .2, .44]),
           Spacer(1, 6),
           P("A missed sale is <b>five times</b> more expensive than keeping a unit for a week. That ratio drives every "
             "decision later: it is better to keep a little extra stock than to run out, but not too much."),
           P("1.3 · Why it is hard", "h2"),
           *bullets(["<b>The future is unknown.</b> Orders are placed two weeks ahead, so the planner must predict demand.",
                     "<b>Demand is small and irregular.</b> The typical store-product sells about one unit a week, often zero.",
                     "<b>The data hides demand.</b> When the shelf was empty, the sales record shows zero even if customers came.",
                     "<b>Stock already on its way matters.</b> Each order must account for stock on the shelf and orders not yet "
                     "delivered, which are themselves eaten by the next two weeks of demand.",
                     "<b>Many items.</b> 599 store-products, each needing its own decision every week."]),
           P("1.4 · The goal of the project", "h2"),
           P("Build a system that <b>(1)</b> learns from three years of weekly sales, <b>(2)</b> predicts the coming weeks for "
             "every store-product, <b>(3)</b> turns predictions into orders that minimise the total of the two costs, "
             "<b>(4)</b> proves it on real weeks it never saw, scored with the official competition rules, and <b>(5)</b> "
             "presents everything in an explanatory notebook and an interactive dashboard."),
           callout("<b>In one sentence:</b> predict how much each product will sell, then order just enough, not too little "
                   "(empty shelves cost €1 a unit) and not too much (leftovers cost €0.20 a unit per week).")]

    # ================================================================ 2 data
    st += H(2, "The data")
    st += [P("2.1 · Source and choice", "h2"),
           P("The data is the <b>VN2 Inventory Planning Challenge</b>, a real competition organised in 2025 by Nicolas "
             "Vandeput (SupChains), sponsored by Flieber and Daybreak and hosted by DataSource.ai. About 180 teams took part. "
             "The sales are real weekly sales of an anonymous retail chain. The organiser published the data, the official "
             "simulator and the official benchmark on GitHub (SupChains/VN2) under the CC BY 4.0 licence."),
           P("The project first used the Walmart M5 dataset (2011-2016). It was replaced because M5 has no stock levels, no "
             "costs and no ordering rules, so those had to be invented. VN2 brings <b>real stock, real costs, real lead time "
             "and an official score</b> to compare with 180 real teams. The M5 version is kept in <i>archive/m5</i>."),
           P("2.2 · Size and time span", "h2"),
           table([["Item", "Value"],
                  ["Stores / products / store-products", f"{a['Stores']} / {a['Products']} / {a['Store-product rows']}"],
                  ["Weeks in total", f"{a['Weeks in total']} (Mondays from {a['First week']} to {a['Last week']})"],
                  ["History (used for learning)", f"{a['History weeks (to week 0)']} weeks, up to week 0 = 8 April 2024"],
                  ["Competition (final test only)", "8 weeks: week 1 = 15 April 2024 ... week 8 = 3 June 2024"],
                  ["Units sold in total", f"{int(a['Units sold in total']):,}"]], [.42, .58], bold_first=True),
           P("2.3 · The files", "h2"),
           table([["File", "Content", "How it is used"],
                  ["Week 0 - 2024-04-08 - Sales.csv", "157 weeks of sales per store-product", "what planners knew before round 1"],
                  ["Week 1 ... Week 8 - Sales.csv", "the same, one more week each round", "Week 8 holds every week: demand of the simulation"],
                  ["Week 0 - In Stock.csv", "True/False per week: was stock left at the end of the week?", "finds the censored weeks"],
                  ["Week 0 - Master.csv", "ProductGroup, Division, Department, DepartmentGroup, StoreFormat, Format",
                   "inputs of the machine-learning model; charts by group"],
                  ["Week 0 - Initial State.csv", "stock on the shelf and orders in transit at the end of week 0",
                   "the real starting point of the simulation"],
                  ["Benchmark order files (6)", "the official benchmark's orders", "verify our simulator to the cent"]],
                 [.3, .38, .32]),
           P("2.4 · The shape of the data", "h2"),
           P("Each row is one <b>store-product</b> (one product in one store), identified by two anonymised integer codes, "
             "<i>Store</i> and <i>Product</i>. Each column is one week, labelled with its Monday date. A cell holds the number "
             "of units sold that week. The in-stock file has exactly the same shape with True/False values."),
           table([["Store", "Product", "2021-04-12", "2021-04-19", "...", "2024-06-03"],
                  ["0", "126", "0", "1", "...", "1"], ["0", "182", "0", "0", "...", "0"], ["...", "...", "...", "...", "...", "..."]],
                 [.12, .12, .19, .19, .19, .19], bold_first=False),
           P("2.5 · Why the stores and products are numbers", "h2"),
           P("The company is anonymous. The data gives codes, not names. The project never invents names: on the dashboard, "
             "five example products get descriptive nicknames based on how they behave (for example \"The Christmas Rocket\", "
             "which sells 2.8 times more in December), always shown next to their real code."),
           P("2.6 · The competition game", "h2"),
           table([["Round", "Planner knows sales up to", "Order placed before", "Order arrives"],
                  ["1", "week 0 (8 Apr)", "week 1 (15 Apr)", "week 3 (29 Apr)"], ["2", "week 1", "week 2", "week 4"],
                  ["3", "week 2", "week 3", "week 5"], ["4", "week 3", "week 4", "week 6"], ["5", "week 4", "week 5", "week 7"],
                  ["6", "week 5", "week 6", "week 8"]], [.12, .32, .28, .28]),
           Spacer(1, 6),
           P("Weeks 1 and 2 are served only by the stock and orders that already exist, so their cost (€913.80) is the same "
             "for every player. The <b>score is the total cost of weeks 3 to 8</b>; lower is better.")]

    # ================================================================ 3 audit & cleaning
    st += H(3, "Data audit and cleaning")
    st += [P("3.1 · Data-quality audit", "h2"),
           table([["Fact", "Value"]] + [[f, v] for f, v in zip(audit["fact"], audit["value"])], [.55, .45], bold_first=True),
           Spacer(1, 6),
           P("The data is technically perfect: no missing cells, no negative or fractional sales, and the nine weekly files "
             "agree with each other (each repeats the earlier weeks unchanged and adds one week). But two facts stand out: "
             f"<b>{a['Weeks with zero sales']}</b> of weeks sell nothing, and <b>{a['Weeks ending out of stock']}</b> of weeks are "
             f"flagged out of stock ({a['Rows ever out of stock']} of store-products ran out at least once)."),
           P("3.2 · Hidden missing value 1: weeks before launch", "h2"),
           P(f"{a['Weeks before launch']} of all cells are zeros before a product's first ever sale in that store. The product "
             "was not on the shelf yet, so those zeros say nothing about demand. 38% of store-products sell from the first "
             "week; 371 were launched later (median 4 weeks in). <b>Rule:</b> every week before the first sale is set to "
             "missing (NaN)."),
           nb_figure("launch_histogram", "When products were launched. Each bar counts store-products whose first sale fell in that week."),
           P("3.3 · Hidden missing value 2: out-of-stock weeks (censored demand)", "h2"),
           P("When a week ends with an empty shelf, the recorded sales are a <b>lower bound</b> of demand: some customers found "
             "nothing. Statisticians call this <b>censored demand</b>. In this data the effect is extreme: <b>100% of "
             "out-of-stock weeks recorded zero sales</b>; the shelf was empty the whole week. In the history, 4.3% of weeks "
             "after launch are censored. <b>Rule:</b> these weeks are set to missing (NaN). The result is called "
             "<b>clean demand</b>."),
           nb_figure("stockout_example", "One store-product with frequent stockouts. Orange bands mark weeks that ended out of stock: they fall "
                         "exactly on the zero-sales weeks, which is why those zeros are not treated as demand."),
           callout("<b>Why not fill the gaps with zero?</b> A zero would teach every model that demand is lower than it really "
                   "is, so the plan would under-order and run out even more often. <b>Why not fill them with a guess?</b> An "
                   "invented number would look like real information. The honest choice is to leave them out and let each "
                   "method skip them."),
           table([["Method", "Treatment of a missing week"],
                  ["Moving averages, exponential smoothing, Croston, TSB", "skipped: averages use clean weeks only"],
                  ["ARIMA / SARIMA", "kept missing: the Kalman filter of the state-space model skips it exactly"],
                  ["ADF, KPSS, STL, ACF/PACF (need an unbroken series)", "linear interpolation between neighbouring clean weeks"],
                  ["LightGBM", "never used as a training target; missing lag features are handled by the trees"]], [.5, .5]),
           P("3.4 · Demand patterns (Syntetos-Boylan classification)", "h2"),
           P("Each store-product is described by two numbers computed on its clean history:"),
           EQ("ADI = number of clean weeks ÷ number of weeks with a sale &nbsp;&nbsp;&nbsp; "
              "CV<super>2</super> = (standard deviation ÷ mean of the non-zero sales)<super>2</super>"),
           P("ADI is the average gap between sales (1 = sells every week). CV² measures how much the size of a sale varies. "
             "With the standard cut-offs (ADI 1.32 and CV² 0.49) each store-product falls into one of four patterns:"),
           table([["Pattern", "Rule", "Plain meaning", "Store-products"],
                  ["Smooth (steady seller)", "ADI < 1.32, CV² < 0.49", "sells almost every week, similar amounts", str(pat.get("smooth", 0))],
                  ["Erratic (spiky seller)", "ADI < 1.32, CV² ≥ 0.49", "sells most weeks, amounts swing", str(pat.get("erratic", 0))],
                  ["Intermittent (occasional)", "ADI ≥ 1.32, CV² < 0.49", "many zero weeks, steady amounts", str(pat.get("intermittent", 0))],
                  ["Lumpy (unpredictable)", "ADI ≥ 1.32, CV² ≥ 0.49", "irregular weeks and amounts", str(pat.get("lumpy", 0))]],
                 [.25, .25, .34, .16]),
           nb_figure("pattern_map", "Demand patterns, one dot per store-product. Most are intermittent or lumpy: hard to forecast."),
           P("3.5 · The time split: no peeking at the future", "h2"),
           P("Everything is evaluated as if it were happening live. The weeks are used in strictly separate roles:"),
           table([["Window", "Weeks", "Used for"],
                  ["Training history", "up to each forecast date", "fitting models (never later weeks)"],
                  ["Backtest dates 1-12", "33 to 22 weeks before week 0", "measuring the spread of forecast errors"],
                  ["Backtest dates 13-30", "21 to 3 weeks before week 0", "accuracy and tuning the ordering policy"],
                  ["Competition weeks 1-8", "15 April to 3 June 2024", "<b>final test only</b>: nothing is tuned on them"]],
                 [.28, .32, .4]),
           nb_figure("time_split", "How the 165 weeks are used: history, the error-spread window, the accuracy and tuning window, and the final test.",
                     max_h=4 * cm)]

    # ================================================================ 4 EDA
    st += H(4, "Exploratory data analysis")
    st += [P("Exploratory data analysis (EDA) means looking at the data from many angles before modelling it, to understand "
             "its behaviour and choose suitable methods. Each graph below answers one question."),
           P("4.1 · The network over time (line graph)", "h2"),
           nb_figure("network_over_time", "Top: total units sold per week by all 599 store-products (dashed line = week 0). Bottom: share of "
                         "store-products with stock left at the end of each week."),
           P("The total sells about 1,761 units a week with a stable level and a sharp peak every December. Availability "
             "is not stable: in some periods 15-20% of store-products ended the week with an empty shelf."),
           P("4.2 · The seasonal profile", "h2"),
           P("The seasonal factor of a calendar week is the average sales of that ISO week number divided by the average week "
             "(the official benchmark's own definition):"),
           EQ("s<sub>w</sub> = mean sales in ISO week w ÷ mean of the 52 weekly means &nbsp;&nbsp;(1.0 = an average week)"),
           nb_figure("seasonal_factors", "Seasonal factor per ISO week. Blue bars are above average. The shaded area marks the competition weeks (16-23)."),
           P("Weeks 48-52 sell up to 2.4 times an average week (the Christmas peak). The competition weeks sit in a calm, "
             "slightly below-average part of the year."),
           P("4.3 · How much does a store-product sell? (histograms)", "h2"),
           nb_figure("sales_histograms", "Left: average units per week per store-product (log scale). Right: share of zero-sales weeks per store-product."),
           P("The median store-product sells about 1.1 units a week, and 75% sell under 2 a week. Demand is extremely "
             "skewed: a few sell hundreds, most sell a handful. This is why error percentages look large later and why "
             "special slow-seller methods are needed."),
           P("4.4 · Where the volume is (Pareto curve)", "h2"),
           nb_figure("pareto", "Cumulative share of units sold, store-products sorted from biggest to smallest seller."),
           P("A small group of store-products makes most of the units. Mistakes on those few items dominate the total cost."),
           P("4.5 · Scatter plots", "h2"),
           nb_figure("scatter_plots", "Left: volume against variability (coefficient of variation), one dot per store-product. Right: lag "
                         "plot of the network total, this week against last week."),
           P("Slow sellers are the most variable relative to their size, typical of count data. The lag plot is tight around "
             "the diagonal: this week's total is strongly related to last week's, the property autoregressive (AR) models use."),
           P("4.6 · QQ plots (normality)", "h2"),
           P("A QQ plot compares the quantiles of the data with those of a normal (bell-curve) distribution; points on the "
             "line mean normal."),
           nb_figure("qq_plots", "QQ plots of the network total, its logarithm, and the biggest single store-product."),
           P("The total is close to normal in the middle with a heavy upper tail (the Christmas weeks); the log helps. A "
             "single store-product is far from normal: small whole numbers with many zeros. Consequence: uncertainty is later "
             "modelled with a <b>count distribution</b> (negative binomial), not a bell curve."),
           P("4.7 · Descriptive statistics", "h2"),
           P("For the network total, the biggest seller and the median store-product, and across all 599: mean, standard "
             "deviation, quartiles, coefficient of variation, skewness, excess kurtosis, share of zero weeks and the "
             "Jarque-Bera normality test. Results: strong right skew (skewness above 1), heavy tails (positive excess "
             "kurtosis), many zero weeks, and normality rejected almost everywhere."),
           P("4.8 · Stockouts and product groups", "h2"),
           nb_figure("stockouts_and_groups", "Left: out-of-stock rate per store, worst first. Right: units sold per department group."),
           P("Out-of-stock rates range from 0% to 59% of weeks across stores: empty shelves are a real, uneven problem."),
           callout("<b>What the EDA decided.</b> (1) Slow, intermittent demand: include Croston and TSB and use a count "
                   "distribution. (2) A strong yearly season shared across the network: take the season from the network. "
                   "(3) Frequent censoring: learn only from clean weeks. (4) Store-products share structure: a single global "
                   "machine-learning model can borrow strength across them.")]

    # ================================================================ 5 time series
    st += H(5, "Time-series analysis (Box-Jenkins)")
    st += [P("The Box-Jenkins method is the classical way to build a forecasting model for one series: check stationarity, "
             "transform if needed, read the autocorrelations to identify a model, choose its order with information criteria, "
             "then check that the residuals are white noise. It is applied to two series and then automatically to all 599:"),
           *bullets(["<b>The network total:</b> all 599 store-products added up, a smooth series of 157 weeks.",
                     "<b>A representative store-product:</b> Store 61 · Product 124, a high-volume item sold from week 1, never "
                     "out of stock (about 99 units a week)."]),
           P("5.1 · Stationarity: ADF and KPSS", "h2"),
           P("A series is <b>stationary</b> when its mean, variance and correlations do not change over time. AR, MA and ARMA "
             "models assume it. The <b>Augmented Dickey-Fuller (ADF)</b> test fits:"),
           EQ("Δy<sub>t</sub> = α + γ·y<sub>t−1</sub> + δ<sub>1</sub>Δy<sub>t−1</sub> + ... + δ<sub>p</sub>Δy<sub>t−p</sub> + ε<sub>t</sub>"),
           P("H0: γ = 0 (a unit root, the series wanders like a random walk, not stationary). A p-value below 0.05 rejects "
             "H0: the series is stationary. The <b>KPSS</b> test reverses the question (H0: stationary), so the two together "
             "guard against a wrong conclusion. Results:"),
           table([["Series", "ADF p-value", "ADF says", "KPSS says"],
                  ["Network total", "0.006", "stationary", "not stationary (borderline)"],
                  ["Store 61 · Product 124", "0.000", "stationary", "stationary"],
                  ["All 599 store-products", f"{ts['adf_rows_stationary']:.0%} below 0.05", f"{ts['adf_rows_stationary']:.0%} stationary", "–"]],
                 [.34, .2, .2, .26]),
           Spacer(1, 6),
           P("KPSS hesitates on the network total because of the strong yearly peaks; this is resolved below."),
           P("5.2 · Transformations", "h2"),
           P("When a series is not stationary, standard fixes are: the <b>log</b> log(1 + y), which stabilises a variance that "
             "grows with the level; the <b>first difference</b> y<sub>t</sub> − y<sub>t−1</sub>, which removes a drifting level; "
             "and the <b>seasonal difference</b> y<sub>t</sub> − y<sub>t−52</sub>, which removes a repeating yearly pattern."),
           table([["Transformation"] + ["ADF p", "KPSS p", "verdict"]] +
                 [[r["series"], f"{float(r['ADF p-value']):.4f}", f"{float(r['KPSS p-value']):.3f}", r["verdict"]] for r in ts["stationarity"]],
                 [.34, .16, .16, .34]),
           nb_figure("transformations", "The network total: original, log and first difference."),
           P(f"The rule \"difference only while ADF cannot reject a unit root\" gives <b>d = {ts['d']}</b>: no differencing is "
             "needed, so the original units are kept (orders need forecasts in units). The AIC comparison in 5.6 also tests "
             "differenced (d = 1) models as a double check."),
           P("5.3 · Seasonality: STL decomposition", "h2"),
           P("STL (Seasonal-Trend decomposition using LOESS) splits the series into three additive parts with a 52-week season:"),
           EQ("y<sub>t</sub> = T<sub>t</sub> (trend) + S<sub>t</sub> (season) + R<sub>t</sub> (remainder)"),
           EQ("trend strength = max(0, 1 − Var(R) ÷ Var(T + R)) &nbsp;&nbsp;&nbsp; seasonal strength = max(0, 1 − Var(R) ÷ Var(S + R))"),
           nb_figure("decomposition", "STL decomposition of the network total: observed, trend, yearly season and remainder."),
           P(f"Seasonal strength <b>{ts['strength']['seasonal strength']:.2f}</b> and trend strength {ts['strength']['trend strength']:.2f}: "
             "the series is dominated by the yearly season. Single small store-products are too noisy and too short to "
             "estimate their own season, which is why the forecasting models take the season from the whole network."),
           P("5.4 · ACF and PACF", "h2"),
           EQ("ACF: ρ<sub>k</sub> = Σ(y<sub>t</sub> − ȳ)(y<sub>t−k</sub> − ȳ) ÷ Σ(y<sub>t</sub> − ȳ)<super>2</super> &nbsp;&nbsp;&nbsp; "
              "significance band = ±1.96 ÷ √n"),
           P("The <b>ACF</b> is the correlation of the series with itself k weeks earlier. The <b>PACF</b> is the same "
             "correlation after removing the effect of the weeks in between (the direct link only). Bars outside the band are "
             "significant at 5%."),
           nb_figure("acf_pacf", "ACF and PACF of the network total (56 lags). Blue bars are significant.", 0),
           nb_figure("acf_pacf", "ACF and PACF of Store 61 · Product 124 (30 lags).", 1),
           P("5.5 · Identifying the model and the lags", "h2"),
           table([["What the plots show", "Model", "Equation"],
                  ["PACF cuts off after p, ACF decays", "AR(p)", "y<sub>t</sub> = c + φ<sub>1</sub>y<sub>t−1</sub> + ... + φ<sub>p</sub>y<sub>t−p</sub> + ε<sub>t</sub>"],
                  ["ACF cuts off after q, PACF decays", "MA(q)", "y<sub>t</sub> = c + ε<sub>t</sub> + θ<sub>1</sub>ε<sub>t−1</sub> + ... + θ<sub>q</sub>ε<sub>t−q</sub>"],
                  ["both decay", "ARMA(p, q)", "AR and MA terms together"],
                  ["differencing needed", "ARIMA(p, d, q)", "ARMA on the d-times differenced series"],
                  ["spike at the seasonal lag", "SARIMA(p,d,q)(P,D,Q)[52]", "adds terms at lags 52, 104, ..."],
                  ["nothing significant", "white noise", "no model can predict it"]], [.32, .22, .46]),
           Spacer(1, 6),
           P(f"Network total: significant ACF lags {ts['identify']['significant ACF lags']}, PACF lags "
             f"{ts['identify']['significant PACF lags']}. The ACF decays and the PACF has one dominant spike at lag 1: an "
             "<b>AR</b> signature, plus the yearly lags around 52, so <b>SARIMA</b>. The representative store-product shows lags "
             "1 and 5 only: a low-order ARMA. <b>Lags</b> 1-5 (recent weeks) and 52 (same week last year) carry the information; "
             "these also became the inputs of the machine-learning model."),
           P("5.6 · Choosing the order: AIC and BIC", "h2"),
           EQ("AIC = 2k − 2·ln L &nbsp;&nbsp;&nbsp;&nbsp; BIC = k·ln n − 2·ln L &nbsp;&nbsp;&nbsp; (k = parameters, n = weeks, L = likelihood)"),
           P("Both reward a good fit (high likelihood) and penalise complexity (more parameters); BIC penalises harder. Lower is "
             "better; a difference below 2 means two models are practically equal. 27 candidates were fitted to the network "
             "total: ARMA(p, q) and SARIMA(p, 0, q)(1, 0, 0)[52] for p, q in 0-2, and ARIMA(p, 1, q)."),
           table([["Model", "Family", "AIC", "BIC", "ΔAIC"]] +
                 [[r["model"], r["family"], f"{r['AIC']:.1f}", f"{r['BIC']:.1f}", f"{r['ΔAIC']:.1f}"] for r in ts["aic"][:8]],
                 [.4, .14, .16, .16, .14]),
           Spacer(1, 6),
           P(f"Best by AIC: <b>{ts['best']}</b>; best by BIC: {ts['best_bic']} (one parameter fewer, ΔAIC 0.5). In words: "
             "this week's total depends on the last two weeks and on the same week one year earlier:"),
           EQ("(1 − φ<sub>1</sub>B − φ<sub>2</sub>B<super>2</super>)(1 − Φ<sub>1</sub>B<super>52</super>)(y<sub>t</sub> − μ) = ε<sub>t</sub>"
              " &nbsp;&nbsp; φ<sub>1</sub> = 0.32, φ<sub>2</sub> = 0.15, Φ<sub>1</sub> = 0.70 (all significant)"),
           P("(B is the backshift operator: B·y<sub>t</sub> = y<sub>t−1</sub>.)"),
           P("5.7 · ARIMA for every store-product", "h2"),
           P("The same recipe runs automatically on all 599 store-products: remove the yearly season with the network factors, "
             "choose d with the ADF rule, then choose (p, q) among 9 candidates by AIC. Store-products with fewer than 26 clean "
             "weeks get a mean-only model. Missing weeks stay missing for the Kalman filter."),
           table([["Family chosen by AIC", "Store-products", "Share"]] +
                 [[f, str(n), f"{n / fam.sum():.0%}"] for f, n in fam.items()], [.5, .25, .25]),
           P("Most store-products need no differencing and follow low-order AR, MA or ARMA models; 28% are slow sellers with so "
             "little signal that a constant mean wins."),
           P("5.8 · Residual diagnostics", "h2"),
           P("The <b>residuals</b> are what the model could not explain. A good model leaves <b>white noise</b>: no pattern, "
             "constant variance."),
           EQ("Ljung-Box: Q = n(n + 2) · Σ<sub>k=1..h</sub> ρ̂<sub>k</sub><super>2</super> ÷ (n − k) &nbsp;&nbsp; (H0: no autocorrelation; p > 0.05 = white noise)"),
           EQ("Jarque-Bera: JB = n/6 · (S<super>2</super> + (K − 3)<super>2</super>/4) &nbsp;&nbsp; (H0: normal residuals; S = skewness, K = kurtosis)"),
           EQ("ARCH-LM: regress ε<sub>t</sub><super>2</super> on ε<sub>t−1</sub><super>2</super>...ε<sub>t−q</sub><super>2</super>; "
              "LM = n·R<super>2</super> &nbsp;&nbsp; (H0: constant variance; p < 0.05 = volatility clustering)"),
           nb_figure("residual_diagnostics", "Residuals of the SARIMA on the network total: over time, their ACF, and their QQ plot."),
           P("5.9 · ARCH and GARCH", "h2"),
           P("ARCH-LM found <b>volatility clustering</b> (p = 0.027): calm periods and turbulent periods (around Christmas) "
             "alternate, so the size of the error is itself predictable. A <b>GARCH(1, 1)</b> model describes this changing "
             "variance:"),
           EQ("ε<sub>t</sub> = σ<sub>t</sub>·z<sub>t</sub>, &nbsp; σ<sub>t</sub><super>2</super> = ω + α·ε<sub>t−1</sub><super>2</super> + β·σ<sub>t−1</sub><super>2</super>"
              f" &nbsp;&nbsp; ω = {ts['garch_params']['omega']:.2f}, α = {ts['garch_params']['alpha[1]']:.2f}, β = {ts['garch_params']['beta[1]']:.2f}"),
           P("α measures how strongly variance reacts to the last shock, β how long it remembers; α + β is the persistence. "
             "After GARCH, the <b>standardised residuals</b> z<sub>t</sub> = ε<sub>t</sub> ÷ σ<sub>t</sub> are tested again:"),
           ]
    gk = list(ts["garch_tests"])
    rows = []
    for test in ["Ljung-Box p (lag 4)", "Ljung-Box p (lag 8)", "Ljung-Box p (lag 13)", "Ljung-Box p (lag 26)", "Jarque-Bera p",
                 "ARCH-LM p", "white noise", "ARCH effect"]:
        rows.append([test] + [("yes" if ts["garch_tests"][g][test] else "no") if isinstance(ts["garch_tests"][g][test], bool)
                              else f"{ts['garch_tests'][g][test]:.3f}" for g in gk])
    st += [table([["Test", "SARIMA residuals", "After GARCH(1,1)", "Squared, after GARCH"]] + rows, [.34, .22, .22, .22],
                 bold_first=True),
           nb_figure("garch_volatility", "GARCH(1,1) volatility (orange) against the size of the residuals (grey)."),
           P("After GARCH: no autocorrelation and no volatility clustering are left. Normality is still rejected because of "
             "a few extreme weeks, which does not bias the forecast but means normal-based intervals would be too narrow in "
             "those weeks."),
           P("5.10 · Does the model forecast unseen weeks?", "h2"),
           nb_figure("holdout_forecast", "SARIMA forecast of the 8 competition weeks (made with history only), with 80% and 95% ranges, against what actually sold."),
           P(f"Mean absolute percentage error <b>{ts['holdout_mape']:.1f}%</b>, against {ts['holdout_mape_snaive']:.1f}% for "
             "\"same week last year\"; <b>all 8 real weeks fall inside the 80% range</b>."),
           P("5.11 · The same tests on all 599 store-products", "h2"),
           nb_figure("pvalue_histograms", "Ljung-Box (lag 8) and ARCH-LM p-values of the 599 per-product ARIMA models."),
           P(f"<b>{ts['rows_white_noise']:.0%}</b> of store-products leave white-noise residuals (Ljung-Box at lags 4, 8 and 13). "
             f"{ts['rows_arch']:.0%} show an ARCH effect; GARCH(1,1) removes it in {ts['garch_rows_fixed']:.0%} of them. With a "
             "5% test level about 5% of perfectly good models fail by chance, so these numbers describe well-specified models."),
           callout("<b>Conclusion of the time-series analysis.</b> The series are mostly stationary, strongly seasonal, and well "
                   "described by low-order (S)ARIMA models whose residuals are white noise once GARCH absorbs the volatility "
                   "clustering. The per-product ARIMA becomes one of the ten forecasting models.")]

    # ================================================================ 6 models
    st += H(6, "The ten forecasting models")
    st += [P("Each model sees the clean demand up to the forecast date and predicts the next weeks (3 for the backtest, 4 "
             "for the competition rounds). <i>y</i> is clean demand, <i>ŷ</i> the forecast, <i>t</i> the last known week and "
             "<i>h</i> the number of weeks ahead. Ten models are compared so that the final choice is proven, not assumed."),
           P("6.1 · Last week (naive)", "h2"),
           EQ("ŷ<sub>t+h</sub> = last clean value of y"),
           P("The simplest possible forecast. It reacts instantly but copies every random spike. It is the baseline any useful "
             "method must beat."),
           P("6.2 · 13-week average", "h2"),
           EQ("ŷ<sub>t+h</sub> = mean of the clean weeks among the last 13"),
           P("A quarter of history smooths out noise but ignores the season and reacts slowly to change."),
           P("6.3 · Same week last year (seasonal naive)", "h2"),
           EQ("ŷ<sub>t+h</sub> = y<sub>t+h−52</sub> (falls back to the 13-week average if that week is missing)"),
           P("Captures the season automatically, but repeats last year's noise as well."),
           P("6.4 · Official benchmark: seasonal moving average", "h2"),
           EQ("base = mean over the last 13 weeks of (y<sub>t</sub> ÷ s<sub>week(t)</sub>) &nbsp;&nbsp; ŷ<sub>t+h</sub> = base × s<sub>week(t+h)</sub>"),
           P("The competition's own method (vn2_benchmark.py), reproduced with the same arithmetic: out-of-stock weeks are set "
             "aside, sales are deseasonalised with the network factors s, averaged over 13 weeks, and reseasonalised for each "
             "future week. It is the target to beat."),
           P("6.5 · Exponential smoothing (SES on deseasonalised demand)", "h2"),
           EQ("ℓ<sub>t</sub> = ℓ<sub>t−1</sub> + α·(y<sub>t</sub>/s<sub>t</sub> − ℓ<sub>t−1</sub>) &nbsp;&nbsp; ŷ<sub>t+h</sub> = ℓ<sub>t</sub> × s<sub>t+h</sub>"),
           P("A weighted average whose weights fade geometrically into the past (weight α on the newest week, α(1−α) on the "
             "one before, ...). α is chosen per store-product among 0.05, 0.1, 0.2, 0.3 and 0.5 by the smallest one-step "
             "in-sample error. Missing weeks are skipped."),
           P("6.6 · ARIMA (AIC-selected, per store-product)", "h2"),
           EQ("φ(B)(1 − B)<super>d</super>(y<sub>t</sub>/s<sub>t</sub>) = c + θ(B)ε<sub>t</sub> &nbsp;&nbsp; ŷ = forecast × s<sub>t+h</sub>"),
           P("The Box-Jenkins model of Part 5: d by ADF, (p, q) by AIC, on season-adjusted clean demand. Orders and parameters "
             "are re-selected every 8 backtest dates (once for the competition) and run on newer data with the Kalman filter."),
           P("6.7 · Croston SBA (for intermittent demand)", "h2"),
           EQ("when a sale occurs: z ← z + α(y − z), &nbsp; x ← x + α(gap − x) &nbsp;&nbsp; ŷ = (1 − α/2)·z ÷ x, &nbsp; α = 0.1"),
           P("Croston's idea: for slow sellers, predict separately <b>how big a sale is</b> (z) and <b>how many weeks pass "
             "between sales</b> (x); demand per week is z ÷ x. The factor (1 − α/2) is the Syntetos-Boylan correction of "
             "Croston's known upward bias."),
           P("6.8 · TSB (Teunter-Syntetos-Babai)", "h2"),
           EQ("every week: p ← p + β(1<sub>sale</sub> − p); &nbsp; on a sale: z ← z + α(y − z) &nbsp;&nbsp; ŷ = p·z, &nbsp; α = β = 0.1"),
           P("Like Croston but smoothing the <b>probability of a sale</b> p every week, so the forecast decays when a product "
             "stops selling (Croston would keep forecasting the old rate)."),
           P("6.9 · LightGBM (global machine learning)", "h2"),
           P("<b>One</b> gradient-boosted decision-tree model is trained on all 599 store-products at once. Gradient boosting "
             "adds trees one after another, each correcting the errors of the previous ones:"),
           EQ("F(x) = f<sub>1</sub>(x) + f<sub>2</sub>(x) + ... + f<sub>300</sub>(x), &nbsp; each f a small decision tree"),
           P("<b>Target:</b> clean demand h weeks ahead divided by the store-product's recent level (13-week mean + 0.1), so "
             "slow and fast sellers share one model; the prediction is multiplied back by the level. <b>Horizon</b> is an input, "
             "so one model serves 1, 2, 3 and 4 weeks ahead. Out-of-stock weeks are never used as targets."),
           table([["Feature group", "Features (all known at the forecast date)"],
                  ["Recent demand", "clean demand 1-8 weeks ago; raw sales 1-4 weeks ago; in-stock flags 1-4 weeks ago"],
                  ["Rolling statistics", "means over 4, 8, 26 and 52 weeks; 13-week standard deviation; share of zero weeks; in-stock share"],
                  ["Season and calendar", "same week last year and two years ago; seasonal factor of the target week; target/now season ratio; ISO week; horizon"],
                  ["Product life", "weeks since launch; log of the level"],
                  ["Hierarchy (categorical)", "Store, Product, ProductGroup, Division, Department, DepartmentGroup, StoreFormat, Format"]],
                 [.28, .72]),
           Spacer(1, 6),
           P("<b>Loss function: Tweedie</b> (power 1.5), designed for non-negative data with many exact zeros. In a test on past "
             "weeks it beat squared error, absolute error and Huber loss. <b>Settings:</b> 300 trees, learning rate 0.05, 31 "
             "leaves, at least 40 rows per leaf, 80% row and column sampling, L2 penalty 1, rows weighted by the level. Refitted "
             "every 4 backtest dates and every competition round."),
           nb_figure("importance", "LightGBM: the 15 most important features (share of total gain)."),
           P("The target week's seasonal factor and the product's identity lead, then the yearly average and the store: the "
             "model learns both the calendar and each item's typical level."),
           P("6.10 · Ensemble", "h2"),
           EQ("ŷ = (ŷ<sub>LightGBM</sub> + ŷ<sub>exponential smoothing</sub> + ŷ<sub>official benchmark</sub>) ÷ 3"),
           P("Averaging different kinds of models cancels part of their individual errors; it is one of the most reliable "
             "findings in forecasting research."),
           nb_figure("model_forecasts", "All models' forecasts at week 0 for one store-product (three highlighted, seven in grey), with the history and what then sold.")]

    # ================================================================ 7 backtest
    st += H(7, "Testing the forecasts: the backtest")
    st += [P("7.1 · Design", "h2"),
           P("A <b>rolling-origin backtest</b>: at each of 30 past weeks (the origins), every model is given only the data up "
             "to that week and forecasts the next 3 weeks; the forecasts are then compared with what really sold. Using any "
             "later data would be <i>leakage</i> and make results look better than reality. Three weeks because an order "
             "placed now arrives in two weeks: the planner needs this week, next week and the arrival week. Errors are scored "
             "only on clean weeks."),
           P("7.2 · Error measures", "h2"),
           EQ("e = ŷ − y &nbsp;&nbsp; MAE = mean|e| &nbsp;&nbsp; WAPE = Σ|e| ÷ Σy &nbsp;&nbsp; Bias = Σe ÷ Σy &nbsp;&nbsp; Score = WAPE + |Bias|"),
           P("MAE is the average miss in units. WAPE scales the total error by total demand. Bias shows whether forecasts are "
             "too high (+) or too low (−) overall. <b>Score</b>, the official metric of the VN1 forecasting competition, "
             "punishes both noise and systematic bias."),
           P("7.3 · Results", "h2"),
           df_table(lb.assign(Model=lb["model"].map(LABELS))[["Model", "MAE", "RMSE", "Bias %", "WAPE %", "Score %"]],
                    [.34, .12, .12, .14, .14, .14],
                    {c: (lambda v: f"{v:.2f}") for c in ["MAE", "RMSE"]} | {c: (lambda v: f"{v:.1f}") for c in ["Bias %", "WAPE %", "Score %"]}),
           Spacer(1, 6), nb_figure("leaderboard", "Score % by model (lower is better). LightGBM in blue, the official benchmark in orange."),
           P("Why are errors around 60%? Most store-products sell one or two units: forecasting 2 when 1 sells is a 100% error. "
             "Percentages for slow sellers always look large; what matters is the comparison between models and, later, the cost."),
           P("By horizon (Score %)", "h3"),
           df_table(lbh.pivot(index="model", columns="h", values="Score %").reindex(lb["model"]).reset_index()
                    .assign(model=lambda d: d["model"].map(LABELS)).rename(columns={"model": "Model", 1: "1 week ahead", 2: "2 weeks", 3: "3 weeks"}),
                    [.4, .2, .2, .2], {c: (lambda v: f"{v:.1f}") for c in ["1 week ahead", "2 weeks", "3 weeks"]}),
           P("By demand pattern (Score %)", "h3"),
           df_table(lbp.pivot(index="model", columns="pattern", values="Score %").reindex(lb["model"])[["smooth", "erratic", "intermittent", "lumpy"]]
                    .reset_index().assign(model=lambda d: d["model"].map(LABELS)).rename(columns={"model": "Model"}),
                    [.36, .16, .16, .16, .16], {c: (lambda v: f"{v:.1f}") for c in ["smooth", "erratic", "intermittent", "lumpy"]}),
           Spacer(1, 6),
           P("Forecasting gets slightly harder further ahead and much harder for intermittent and lumpy store-products, where "
             "errors above 100% are common. LightGBM leads in every pattern except erratic."),
           P("7.4 · Are the forecast errors white noise?", "h2"),
           P("For each model and store-product, the sequence of 1-week-ahead errors over the 30 dates is tested with "
             "Ljung-Box (lag 4). Passing means the model leaves no pattern in its mistakes."),
           df_table(wh.rename(columns={wh.columns[0]: "model"}).assign(Model=lambda d: d["model"].map(LABELS))
                    [["Model", "rows_tested", "white_noise_share", "median_p"]]
                    .rename(columns={"rows_tested": "Store-products tested", "white_noise_share": "White-noise share", "median_p": "Median p"}),
                    [.4, .2, .2, .2], {"White-noise share": lambda v: f"{v:.0%}", "Median p": lambda v: f"{v:.2f}"}),
           callout("A Ljung-Box pass shows that no exploitable pattern is left; it does not by itself prove accuracy (a flat "
                   "average can pass). Accuracy (7.3) and white-noise errors (7.4) together are the evidence.")]

    # ================================================================ 8 uncertainty
    st += H(8, "Uncertainty: how wrong can a forecast be?")
    st += [P("Ordering exactly the forecast would run out about half the time. The policy needs the whole <b>distribution</b> of "
             "demand around each forecast. Weekly unit counts are modelled with a <b>negative binomial</b> distribution, "
             "built for counts with extra spread:"),
           EQ("mean = μ (the forecast), &nbsp; variance = k·μ &nbsp;&nbsp; (k = 1: Poisson noise; k > 1: spikier)"),
           EQ("Demand = Poisson(λ) with λ ~ Gamma(shape r, mean μ), &nbsp; r = μ ÷ (k − 1)"),
           P("<b>Estimating k.</b> For each model and store-product, from the backtest errors on clean weeks:"),
           EQ("raw k = Σ(ŷ − y)<super>2</super> ÷ Σŷ &nbsp;&nbsp; k = w·raw + (1 − w)·pooled k, &nbsp; w = n ÷ (n + 6), &nbsp; kept in [1, 30]"),
           P("Each store-product has few clean weeks, so its own estimate is shrunk towards the model's pooled value. To keep "
             "the test honest, k for the policy tuning comes only from backtest dates 1-12; dates 13-30 check calibration: "
             "demand should stay below the 50%, 80% and 95% quantiles about 50%, 80% and 95% of the time."),
           df_table(calib.rename(columns={"model": "Model"}), [.34, .14, .17, .17, .18],
                    {"pooled k": lambda v: f"{v:.1f}"} | {c: (lambda v: f"{v:.0%}") for c in calib.columns if c.startswith("demand")}),
           Spacer(1, 6),
           P("The quantiles are close but slightly generous (about 85% instead of 80%): the spread is estimated mostly from "
             "large, volatile items. This is corrected in Part 10 by tuning the service level with real simulated costs.")]

    # ================================================================ 9 simulator
    st += H(9, "The inventory simulator")
    st += [P("The simulator plays the official rules (the organiser's vn2_simulate.py) for every store-product, every week:"),
           EQ("start = end<sub>last week</sub> + arriving order &nbsp;&nbsp; sales = min(start, demand) &nbsp;&nbsp; lost = demand − sales"),
           EQ("end = start − sales &nbsp;&nbsp; cost = 0.2 × end + 1.0 × lost &nbsp;&nbsp; in transit: W+1 ← W+2, W+2 ← new order"),
           P("Demand in the simulation is the recorded sales, as in the competition. The state of each store-product is three "
             "numbers: stock on the shelf, the order arriving next week, and the order arriving in two weeks."),
           P("9.1 · Verification against the official result", "h2"),
           P("The organiser published the six benchmark order files and their score. Playing them through our simulator, and "
             "re-implementing the benchmark policy (seasonal average, order up to 4 weeks of forecast), must give the same "
             "number:"),
           table([["Check", "Result"],
                  ["Setup cost, weeks 1-2", "€913.80 (official: €913.80)"],
                  ["Score of the published benchmark orders, weeks 3-8", "€4,334.00 (official: €4,334.00)"],
                  ["Our re-implementation of the benchmark", "€4,334.00, identical orders for all 599 × 6 decisions"]], [.6, .4]),
           Spacer(1, 6),
           df_table(bench_w.reset_index()[["week", "start", "sales", "lost", "end", "holding", "shortage", "cost"]]
                    .rename(columns=str.capitalize), [1 / 8] * 8,
                    {c: (lambda v: f"{v:,.1f}") for c in ["Start", "Sales", "Lost", "End", "Holding", "Shortage", "Cost"]}),
           P("The official benchmark, week by week (all 599 store-products).", "caption"),
           callout("Because the simulator reproduces the official result exactly, every score in this project is directly "
                   "comparable with the real competition leaderboard.")]

    # ================================================================ 10 policy
    st += H(10, "The ordering policy")
    st += [P("10.1 · The idea: a newsvendor problem", "h2"),
           P("Before week k the planner orders Q units that land at the start of week k + 2. When they land, the shelf holds "
             "X, whatever is left after weeks k and k + 1 (with lost sales). The order must cover Y = demand in week k + 2 − X. "
             "With cost h per unit left and p per unit short, the expected cost of the arrival week is:"),
           EQ("E[cost] = h·E[(Q − Y)<super>+</super>] + p·E[(Y − Q)<super>+</super>]"),
           P("The classic newsvendor result: the best Q is the <b>critical-ratio quantile</b> of Y:"),
           EQ("Q* = F<sub>Y</sub><super>−1</super>( p ÷ (p + h) ) = F<sub>Y</sub><super>−1</super>(1 ÷ 1.2) = F<sub>Y</sub><super>−1</super>(0.833)"),
           P("The probability used is called the <b>service level</b>: the chance that the order covers its week."),
           P("10.2 · Monte Carlo: computing the distribution of Y", "h2"),
           *bullets(["Draw 1,000 demand paths for weeks k, k+1 and k+2 from the negative binomial (forecast μ, spread k).",
                     "On each path, play weeks k and k+1 with the stock on the shelf and the two orders on their way, with lost "
                     "sales: X = max(0, max(0, shelf + arriving<sub>k</sub> − D<sub>k</sub>) + arriving<sub>k+1</sub> − D<sub>k+1</sub>).",
                     "Y = D<sub>k+2</sub> − X on each path; Q = the service-level quantile of the 1,000 values of Y, at least 0, a whole number.",
                     "Common random numbers: all service levels share the same paths, so differences come from the policy alone."]),
           P("10.3 · Tuning the service level by simulation", "h2"),
           P("The textbook 0.833 assumes a perfect demand distribution. Instead, every model is replayed over 18 past weeks "
             "(backtest dates 13-30) at 12 service levels; the first 3 weeks are warm-up and not scored. The benchmark's "
             "simpler order-up-to-N-weeks rule is replayed too."),
           ]
    nv = tuning[tuning.policy == "newsvendor"].pivot(index="param", columns="model", values="cost")
    show = ["ensemble", "lightgbm", "seasonal_ma", "arima", "ses", "tsb"]
    st += [df_table(nv[show].reset_index().rename(columns={"param": "Service level"} | {m: LABELS[m].split(" (")[0] for m in show}),
                    [.16] + [.14] * 6, {"Service level": lambda v: f"{v:.3f}"} | {LABELS[m].split(" (")[0]: (lambda v: f"{v:,.0f}") for m in show}),
           P("Historical replay: total cost (€) over 15 scored weeks by service level, for six models.", "caption"),
           nb_figure("tuning_curves", "Replay cost against service level (grey: other models). The dashed line is the textbook 0.833."),
           P(f"Every curve is U-shaped: too little safety means empty shelves, too much means leftovers. The lowest cost is at "
             f"<b>{summ['plan_service']}</b> for the best models, below the textbook 0.833, for two reasons: the estimated "
             "spread is slightly generous (Part 8), and leftover stock is not wasted (it sells the following week). The chosen "
             f"plan is the cheapest in the replay: <b>{LABELS[summ['plan_model']]} at {summ['plan_service']}</b>. Simulation-based "
             "tuning is exactly what the competition's top teams reported doing.")]

    # ================================================================ 11 final test
    st += H(11, "The final test: the eight competition weeks")
    st += [P("The competition is replayed exactly: before each round the models are refitted on the data available then "
             "(sales and in-stock flags up to the previous week) and forecast 4 weeks ahead; each model orders with the service "
             "level chosen on the past. <b>Nothing is tuned on these weeks.</b>"),
           P("11.1 · Forecast accuracy on the competition weeks", "h2"),
           df_table(lbc.assign(Model=lbc["model"].map(LABELS))[["Model", "MAE", "Bias %", "WAPE %", "Score %"]],
                    [.4, .15, .15, .15, .15], {c: (lambda v: f"{v:.2f}") for c in ["MAE"]} | {c: (lambda v: f"{v:.1f}") for c in ["Bias %", "WAPE %", "Score %"]}),
           Spacer(1, 6),
           P("An honest surprise: over these eight weeks the simple official benchmark forecast is the most accurate, and "
             "LightGBM (best over 30 backtest weeks) slips. Eight weeks is a short window and rankings move. What matters is "
             "the cost of the orders each forecast leads to:"),
           P("11.2 · The score", "h2"),
           df_table(comp.assign(Model=comp["model"].map(LABELS))[["Model", "service (tuned on past)", "VN2 score €", "order-up-to score €",
                                                                    "hindsight best €"]]
                    .rename(columns={"service (tuned on past)": "Service level", "VN2 score €": "Newsvendor score",
                                     "order-up-to score €": "Order-up-to score", "hindsight best €": "Hindsight best"}),
                    [.34, .14, .17, .18, .17], {"Service level": lambda v: f"{v:.2f}"} |
                    {c: (lambda v: f"€{v:,.1f}") for c in ["Newsvendor score", "Order-up-to score", "Hindsight best"]}),
           Spacer(1, 6), png_figure(ch["comp"], "Cost of weeks 3-8 for each model's orders; the dashed line is the official method."),
           png_figure(ch["race"], "Money spent so far, weeks 3-8: our plan (blue) against the official method (orange)."),
           png_figure(ch["rank"], "Our plan against the real competition's top five and the official method."),
           table([["Plan", "Cost, weeks 3-8"],
                  [f"<b>Our plan: {LABELS[summ['plan_model']]} @ {summ['plan_service']}</b>", f"<b>€{plan.score:,.1f}</b>"],
                  ["Real 1st place", f"€{summ['winner_score']:,.1f}"],
                  ["<b>Official benchmark</b>", f"<b>€{summ['benchmark_score']:,.1f}</b>"],
                  ["Saving vs the official benchmark", f"€{saved:,.1f} ({saved / summ['benchmark_score']:.1%})"]], [.62, .38]),
           Spacer(1, 6),
           P("11.3 · Where the saving comes from", "h2"),
           nb_figure("weekly_costs", "Weekly cost: our plan (left bars: light = leftover stock, dark = missed sales) against the benchmark (right bars)."),
           P(f"Our plan keeps <b>{1 - left_p / left_b:.0%} less stock</b> on the shelves ({left_p:,.0f} vs {left_b:,.0f} units at the "
             f"end of an average week) and accepts more missed sales ({lost_p:,.0f} vs {lost_b:,.0f} units over weeks 3-8). Under "
             f"the official costs that is the cheaper deal: {left_b - left_p:,.0f} fewer units held each week save about "
             f"€{(left_b - left_p) * 0.2 * 6:,.0f} over the six weeks, more than the €{lost_p - lost_b:,.0f} of extra missed sales."),
           callout("<b>How to read this honestly.</b> Every decision used only information available at the time, and the service "
                   "level was chosen on past weeks. But the project was built after the competition, knowing which kinds of "
                   "methods did well (global gradient boosting, simulation-tuned policies). The fair claim is: the pipeline "
                   f"clearly beats the official method ({saved / summ['benchmark_score']:.0%} cheaper) and performs in the range "
                   "of the top entries. \"We beat 1st place\" is not a fair claim.", "warn")]

    # ================================================================ 12 outlook
    st += H(12, "The stock outlook: the next order")
    st += [P("The data ends on 3 June 2024 (week 8). The plan treats <b>10 June 2024 as \"today\"</b>: it continues the plan "
             "for rounds 7 and 8 (which the competition did not play), then forecasts weeks 9-12 from all data and computes the "
             "next order, which arrives on 24 June (week 11). Today's real date cannot be used: there is no data after June "
             "2024, so a plan for today would be invented."),
           table([["Outlook figure", "Value"],
                  ["Units to order now", f"{int(outlook['order now'].sum()):,}"],
                  ["Store-products receiving an order", f"{int((outlook['order now'] > 0).sum())} of {len(outlook)}"],
                  ["High risk (> 50%) of an empty shelf in week 9", str(int((outlook['risk wk 9'] > .5).sum()))],
                  ["High risk in week 10", str(int((outlook['risk wk 10'] > .5).sum()))],
                  ["High risk in week 11 (after this order arrives)", str(int((outlook['risk wk 11'] > .5).sum()))]], [.62, .38], bold_first=True),
           Spacer(1, 6),
           P("Weeks 9 and 10 are already decided: only the stock on the shelf and the orders on their way can serve them. The "
             "new order protects week 11, which is why its risk is close to 1 − service level for most store-products. The risk "
             "of a week is the share of the 1,000 simulated demand paths in which the shelf runs empty.")]

    # ================================================================ 13 dashboard
    st += H(13, "The dashboard, page by page")
    st += [P("The dashboard (Streamlit, <i>streamlit run app/Home.py</i>, http://localhost:8501) has seven pages. Heavy results "
             "come from the notebook's artefacts; the stock plan is re-simulated live when a setting changes. Colours are "
             "consistent everywhere: <b>blue = our plan</b>, <b>orange = the official method</b>, grey = context, and the "
             "palette is colour-blind safe. Every chart has a plain-language note.")]
    pages = [
        ("overview", "Overview: did it work?",
         ["Hero header: the business problem in two lines.",
          "Money headlines: saved vs the official method, the yearly estimate, stock on shelves, units to order this week, "
          "and an honest note on the trade-off (more missed sales, much less stock).",
          "Meet the products: five hero store-products chosen by fixed rules (The Steady Seller, The Christmas Rocket, The "
          "Often-Empty Shelf, The Slow Mover, The Newcomer), each with its real code, a sparkline and a Follow button.",
          "The story: four charts for the followed product (it sold like this, we predicted this, so we ordered this, and the "
          "shelf ...), with one plain sentence each, including when the official method did better on that product.",
          "The result: money spent so far, our plan vs the official method; the real competition's top five.",
          "How it works (four steps) and links to the other pages."]),
        ("demand", "Demand: how does a product sell?",
         ["Store and Product selectors (shared by all pages; hero products show their nickname).",
          "Cards: sales pattern, average week, share of weeks with no sale, share of weeks out of stock.",
          "Weekly sales: blue bars, orange bands on out-of-stock weeks, the 13-week average, grey band before launch.",
          "Seasonality across all stores; the pattern map (weeks between sales vs size variation) with the product highlighted.",
          "Availability heatmap: share of each store's launched products out of stock, month by month."]),
        ("forecast", "Forecast: which method predicts best?",
         ["Leaderboard of the 10 methods with a choice of measure (Score, WAPE, MAE, Bias); hover for each method's idea.",
          "Try it: pick a product, a forecast date (30 backtest dates and 9 competition rounds) and up to three methods; see "
          "the forecasts, the 80% range and what actually sold.",
          "Error by sales pattern (heatmap) and LightGBM's most important inputs.",
          "Accuracy on the eight competition weeks."]),
        ("check", "Model check: are the statistics sound?",
         ["Whole network or one store-product (computed live).",
          "Verdict cards: stationary (ADF), seasonal strength, best model (AIC, with the BIC choice), residuals white noise.",
          "Tabs: stationarity (rolling mean and spread, ADF/KPSS table), season and trend (STL), ACF/PACF with the "
          "identified model and lags, model choice (ΔAIC bars), residuals and GARCH (tests before and after), forecast check.",
          "All 599: model families, share stationary, share white noise, ARCH share, and out-of-sample error whiteness per method."]),
        ("planner", "Stock planner: what if I change the settings?",
         ["Choose the forecast method and the service level (slider); Reset returns to the recommended plan.",
          "Cards: VN2 score vs the benchmark, cost of leftover stock, cost of empty shelves, share of demand served.",
          "Cost per week (plan vs benchmark, stacked leftover/missed) and the safety trade-off curve (past replay and the "
          "8 VN2 weeks, with your choice marked).",
          "Drill-down for one product: stock at the start of each week, demand, missed sales, the orders and when they arrive; "
          "download the six orders as CSV."]),
        ("outlook", "Order outlook: what do I order this week?",
         ["Headline: units to order now, for how many store-products, arrival date.",
          "Cards: stock on the shelf and on its way, expected demand, high-risk counts.",
          "Chance of an empty shelf by week (low, medium, high) and the 15 stores ordering most.",
          "Action list of all 599 with filters (stores, pattern, minimum risk), progress bars for risk, CSV download."]),
        ("method", "Method: how was it built?",
         ["The six stages of the pipeline, the official rules, a glossary, the calibration table, limitations, the data "
          "citation and downloads of the main results."]),
    ]
    for key, title, items in pages:
        st += [P(title, "h2"), *bullets(items)]
        f = SHOTS / f"pw_{key}.png"
        if f.exists():
            st.append(png_figure(f, f"Dashboard: {title.split(':')[0]} page.", max_h=19 * cm))

    # ================================================================ 14 engineering
    st += H(14, "The engineering")
    st += [P("14.1 · Project structure", "h2"),
           table([["Path", "Role"],
                  ["config.yaml", "every setting in one place"],
                  ["src/optichain/data.py", "load VN2, launch and censoring masks, Syntetos-Boylan classes, audit"],
                  ["src/optichain/timeseries.py", "ADF/KPSS, transformations, STL, ACF/PACF, identification, AIC/BIC, Ljung-Box, "
                                                 "Jarque-Bera, ARCH-LM, GARCH, per-product ARIMA (RowARIMA), error whiteness"],
                  ["src/optichain/models.py", "the forecasting models, seasonal factors, LightGBM features and training"],
                  ["src/optichain/evaluation.py", "rolling-origin backtest, metrics, spread (k) estimation"],
                  ["src/optichain/inventory.py", "official simulator, negative-binomial sampling, newsvendor orders, replays, outlook"],
                  ["src/optichain/planning.py", "round-by-round forecasts, policy tuning, competition scoring, the live plan"],
                  ["src/optichain/plots.py", "the notebook's charts, one function per chart, so the notebook shows only the analysis"],
                  ["scripts/build_notebook.py", "writes and runs the pipeline notebook"],
                  ["scripts/build_*.py", "build the PDF documents (guide, cell-by-cell, this walkthrough)"],
                  ["app/", "the Streamlit dashboard: Home.py, views/ (one file per page), ui.py (design system), data.py (cached loaders)"],
                  ["artifacts/", "results written by the notebook and read by the dashboard"],
                  ["tests/", "23 automatic tests"], ["archive/m5/", "the earlier Walmart M5 version"]], [.32, .68]),
           P("14.2 · Settings (config.yaml)", "h2"),
           table([["Setting", "Value", "Meaning"],
                  ["holding_cost / shortage_cost", "0.2 / 1.0 €", "official costs"],
                  ["lead_time_weeks", "2", "order k arrives at the start of week k + 2"],
                  ["forecast.horizon", "3", "weeks forecast in the backtest"],
                  ["forecast.backtest_origins", "30", "rolling forecast dates"],
                  ["forecast.dispersion_origins", "12", "first dates, used only for the spread k"],
                  ["forecast.lgbm_retrain_every / arima_refit_every", "4 / 8", "weeks between refits in the backtest"],
                  ["forecast.ensemble_of", "LightGBM, smoothing, benchmark", "ensemble members"],
                  ["policy.samples", "1,000", "Monte Carlo demand paths per store-product"],
                  ["policy.service_grid", "0.50 ... 0.96 (12 values)", "service levels tried in the tuning"],
                  ["policy.replay_warmup", "3", "replay weeks not scored"], ["seed", "42", "same numbers on every run"]],
                 [.38, .27, .35]),
           P("14.3 · Tests", "h2"),
           P("23 automatic tests (python -m pytest) guard the logic: censoring masks and pattern classes; every model returns "
             "finite non-negative forecasts and ignores censored weeks; Croston on a known intermittent series; the official "
             "weekly arithmetic; <b>the simulator reproduces €913.8 and €4,334.0</b>; the benchmark re-implementation places the "
             "official first order; negative-binomial sampling has the right mean and variance; newsvendor orders grow with the "
             "service level; ADF separates noise from a random walk; AIC recovers a known AR(1) with white-noise residuals; "
             "GARCH removes a simulated ARCH effect; every dashboard page renders; the planner reacts and keeps the choice; the "
             "Model check works for one product; following a hero product updates the story."),
           P("14.4 · Reproducibility and run time", "h2"),
           P("The whole pipeline runs from one command (python scripts/build_notebook.py --run) in about 20-25 minutes on a "
             "12-core laptop: the backtest about 6 minutes, the competition rounds about 6, the policy tuning about 3, the "
             "599 ARIMA fits about 1 (in parallel). A fixed seed makes every run identical. The VN2 files ship with the "
             "project and are downloaded automatically if missing.")]

    # ================================================================ 15 limitations
    st += H(15, "Limitations and next steps")
    st += [P("15.1 · Limitations", "h2"),
           *bullets(["<b>Hindsight.</b> The design was informed by the published competition results.",
                     "<b>Censored demand in the simulation.</b> Demand is the recorded sales (as in the competition), so weeks "
                     "that ran out in reality understate true demand.",
                     "<b>Short test.</b> Eight weeks is a short window; a few large store-products can move the score.",
                     "<b>Anonymous data.</b> What the products are, and their prices and margins, is unknown; costs are uniform.",
                     "<b>Simple spread model.</b> Variance = k × mean is simple; the risk ranges are slightly generous.",
                     "<b>Myopic policy.</b> Each order looks only at its arrival week, not at the weeks after it."]),
           P("15.2 · Next steps", "h2"),
           *bullets(["Quantile or distributional LightGBM, or conformal intervals, for better-calibrated risk.",
                     "Service levels per demand pattern or per product group.",
                     "A multi-week policy (dynamic programming or reinforcement learning, as the 2nd place did).",
                     "Estimating true demand in censored weeks instead of only skipping them.",
                     "A permanent deployment (Streamlit Community Cloud) with weekly data updates."])]

    # ================================================================ appendices
    st += H("A", "Glossary")
    gl = [("ACF / PACF", "correlation of a series with itself k weeks earlier; PACF keeps only the direct link"),
          ("ADF test", "Augmented Dickey-Fuller; H0 = unit root (not stationary)"),
          ("AIC / BIC", "information criteria: fit minus a complexity penalty; lower is better"),
          ("ARCH effect", "volatility clustering: the size of errors changes over time in runs"),
          ("AR, MA, ARMA, ARIMA, SARIMA", "autoregressive, moving-average, both, with differencing, with a seasonal part"),
          ("Backtest", "forecasting past weeks with only the data known at the time"),
          ("Bias", "total over-forecast ÷ total demand"),
          ("Censored demand", "sales lower than demand because the shelf ran empty"),
          ("Clean demand", "sales with out-of-stock and pre-launch weeks removed"),
          ("Critical ratio", "p ÷ (p + h) = 0.833: the textbook service level"),
          ("Croston SBA / TSB", "forecasting methods for intermittent demand"),
          ("Ensemble", "the average of several models' forecasts"),
          ("GARCH(1,1)", "a model of changing variance: σ² = ω + α·ε² + β·σ²"),
          ("Holding cost", "€0.20 per unit left on the shelf at the end of a week"),
          ("Jarque-Bera", "test of normality from skewness and kurtosis"),
          ("Kalman filter", "the algorithm that estimates state-space (ARIMA) models and skips missing weeks"),
          ("KPSS test", "H0 = stationary (the mirror of ADF)"),
          ("Lead time", "time between ordering and delivery: 2 weeks"),
          ("LightGBM", "a fast gradient-boosted decision-tree library"),
          ("Ljung-Box", "test for leftover autocorrelation (white noise if p > 0.05)"),
          ("Lost sale / shortage cost", "demand the empty shelf could not serve; €1 per unit"),
          ("MAE / WAPE / Score", "error measures; Score = WAPE + |Bias|"),
          ("Monte Carlo", "estimating by simulating many random scenarios"),
          ("Negative binomial", "a count distribution with variance above the mean"),
          ("Newsvendor", "the classic single-period ordering problem with an optimal quantile"),
          ("NaN", "a missing value"),
          ("Order-up-to", "order enough to bring stock + in transit up to N weeks of forecast"),
          ("p-value", "chance of the observed result if H0 were true; below 0.05 rejects H0"),
          ("Service level", "chance that an order covers demand in its arrival week"),
          ("Stationary", "mean and variance do not drift over time"),
          ("STL", "Seasonal-Trend decomposition using LOESS"),
          ("Store-product", "one product in one store; 599 in total"),
          ("Tweedie loss", "a loss function for non-negative data with many zeros"),
          ("Week 0", "8 April 2024, the last history week before the competition"),
          ("White noise", "a series with no pattern left: uncorrelated, constant variance")]
    st += [table([["Term", "Meaning"]] + [[f"<b>{t}</b>", d] for t, d in gl], [.3, .7])]

    st += H("B", "Questions and answers")
    qa = [("Why VN2 and not Walmart?", "It is recent and real, with real stock levels, costs, rules and an official score."),
          ("What is a store-product?", "One product in one store. 67 stores and 297 products give 599 existing pairs; each is planned separately."),
          ("Why set out-of-stock weeks aside?", "All of them recorded zero sales: those zeros measure an empty shelf, not customer demand."),
          ("Why both ADF and KPSS?", "They test opposite hypotheses; agreement makes the conclusion safer."),
          ("Why no differencing?", "ADF already rejects a unit root on the levels (d = 0), and AIC confirmed it."),
          ("Why SARIMA for the total?", "AR signature in ACF/PACF, strong yearly season (strength 0.93), and the lowest AIC."),
          ("Why GARCH?", "ARCH-LM found volatility clustering; after GARCH the residuals are white noise."),
          ("What does Ljung-Box prove?", "That no pattern is left in the residuals or errors. Accuracy is proven separately on unseen weeks."),
          ("Why is ARIMA not the final model?", "Tested fairly, LightGBM and the ensemble were more accurate and cheaper; ARIMA still beat the official method."),
          ("Why LightGBM?", "One model learns from all 599 store-products at once, so slow sellers borrow patterns from similar ones."),
          ("Why a negative binomial?", "Weekly sales are small whole numbers with many zeros and extra spread; a bell curve does not fit."),
          ("Why 0.75 and not 0.833?", "Tuned by simulating real costs on past weeks; the spread is slightly generous and leftovers sell later."),
          ("Why does the plan miss more sales than the benchmark?", "Holding 45% less stock saves more than the extra missed sales cost under the €1 vs €0.20 rules."),
          ("Is beating 1st place a fair claim?", "No: the project was built after the competition. The fair claim is 16% cheaper than the official method."),
          ("Why does the outlook say 24 June 2024?", "The data ends on 3 June; the order placed on 10 June arrives two weeks later."),
          ("Is the simulator trustworthy?", "It reproduces the official €4,334.0 to the cent with identical orders.")]
    st += [table([["Question", "Answer"]] + [[f"<b>{q}</b>", ans] for q, ans in qa], [.38, .62])]

    st += H("C", "The notebook, cell by cell")
    st += [P("The following pages explain every one of the 108 cells of the pipeline notebook in order: what each cell does, "
             "what it shows and why it matters, with the start of its code and its real output.")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    main_pdf = tmp / "main.pdf"
    doc = SimpleDocTemplate(str(main_pdf), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.0 * cm,
                            bottomMargin=1.8 * cm, title="OptiChain: complete walkthrough", author="OptiChain",
                            subject="Demand forecasting and stock planning on the VN2 Inventory Planning Challenge")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)

    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for src in [main_pdf] + ([APPENDIX] if APPENDIX.exists() else []):
        for page in PdfReader(str(src)).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "OptiChain: complete walkthrough", "/Author": "OptiChain"})
    with open(OUT, "wb") as f:
        w.write(f)
    print(f"Wrote {OUT} ({len(w.pages)} pages; appendix {'included' if APPENDIX.exists() else 'missing'})")


if __name__ == "__main__":
    build()
