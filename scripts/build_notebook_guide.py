"""Build docs/OptiChain_Notebook_Cell_by_Cell.pdf: every cell of the pipeline notebook explained in simple words,
with a short look at its code and its real output (text and charts) taken from the executed notebook.

    python scripts/build_notebook_guide.py

The explanations below are keyed by cell number. If the notebook's cells change (scripts/build_notebook.py), check
that the first words in CHECK still match, or the script stops and names the cell that moved.
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import nbformat as nbf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "OptiChain_VN2_Pipeline.ipynb"
OUT = ROOT / "docs" / "OptiChain_Notebook_Cell_by_Cell.pdf"

# ---------------------------------------------------------------- explanations, one per cell
# Text (markdown) cells: one line. Code cells: (what it does, what you see, why it matters).
EXPL: dict[int, str | tuple[str, str, str]] = {
    0: "The title page of the notebook: the business problem, the data, and a table of all 20 steps.",
    1: "Heading: step 0, setup.",
    2: ("Loads the tools (pandas for tables, numpy for maths) and the project's own modules, each with a short "
        "comment: D = data, E = evaluation, I = inventory, M = models, P = planning, plots = charts, TS = time series. "
        "It reads the settings from config.yaml, creates the artifacts folder, fixes the random seed, hides "
        "statsmodels' messages and sets one chart style.",
        "Nothing is printed.",
        "Fixing the seed (42) means every run gives exactly the same numbers. Hiding the warnings keeps the output readable."),
    3: "Heading: step 1, data collection. A table lists the competition files and what each one contains.",
    4: ("Checks that all VN2 files are in data/raw/vn2. Any missing file is downloaded from the organiser's GitHub "
        "page, and reports what it downloaded.",
        "\"Downloaded: nothing, every file is already here\".",
        "The notebook works on any computer: if the data is missing, it fetches it itself."),
    5: ("Reads all the files into one object called panel: sales, in-stock flags, product groups and the starting "
        "stock. It also makes hist, the same data as fast number arrays for the models, and t0, the position of "
        "week 0 (8 April 2024).",
        "599 store-products × 165 weeks, and the first 5 rows × 8 weeks of sales.",
        "Each row is one product in one store; each column is one week. Everything later works on this table."),
    6: "Text: explains the shape of the data (rows are store-products, columns are weeks).",
    7: ("Shows the first rows of three more tables: the in-stock flags (True/False), the product and store groups, "
        "and the stock on the shelf and on its way at the start.",
        "Three small tables.",
        "To see what each file looks like before using it."),
    8: "Heading: step 1.2, the data-quality audit.",
    9: ("Runs a quality check and puts the results in one table: number of stores, products and weeks, missing "
        "values, negative values, decimals, share of zero weeks, share of out-of-stock weeks.",
        "A table of facts: 0 missing, 0 negative, 0 decimals, 43% zero weeks, about 10% out-of-stock weeks.",
        "If the data were broken, every later result would be wrong. Checking first is basic good practice."),
    10: ("Opens all 9 weekly sales files one by one and checks that each repeats the earlier weeks unchanged, keeps "
         "the same 599 rows in the same order, and adds exactly one new week.",
         "\"All 9 sales files are consistent\".",
         "Proves the files agree with each other, so using the last file (which holds every week) is safe."),
    11: "Text: reads the audit. Data is complete; 43% of weeks sell nothing; about 1 week in 10 is out of stock.",
    12: "Heading: step 2, exploratory data analysis (EDA).",
    13: ("Turns the wide table (one column per week) into a long table (one row per store-product-week) with the "
         "product groups attached. Also defines hist_cols, which marks the 157 history weeks.",
         "The first rows of the long table (98,835 rows in total).",
         "A long table is easier for charts and group totals. hist_cols makes sure the 8 test weeks stay hidden."),
    14: "Heading: 2.1, line graph of the whole network over time.",
    15: ("Adds up the sales of all 599 store-products for each week, and the share of store-products that had stock "
         "left. Draws both as line graphs.",
         "Top: total units per week with yearly peaks. Bottom: the in-stock share, which dips in some periods. "
         "Mean about 1,761 units a week.",
         "Shows the big picture: a stable level, a strong Christmas peak, and periods with many empty shelves."),
    16: "Text: sales are stable with a yearly rhythm; availability is not, so the in-stock flag matters.",
    17: ("Computes the seasonal factor of each week of the year: the average sales in that week ÷ the average week. "
         "This is exactly how the official benchmark measures seasonality.",
         "A bar chart with 52 bars; weeks above 1 sell more than average; the test weeks (16-23) are shaded.",
         "Every seasonal model later uses these factors."),
    18: "Heading: 2.3, histograms (how much does one store-product sell?).",
    19: ("Computes, for each store-product, its average weekly sales and its share of zero weeks, and draws two "
         "histograms.",
         "Most store-products sell about 1 unit a week; 75% sell under 2 a week.",
         "Shows that demand is small and slow, so the models must handle many zeros."),
    20: "Text: demand is very skewed; errors measured in units are dominated by the big sellers. Heading 2.4: Pareto.",
    21: ("Sorts store-products from biggest to smallest seller and draws the running total of their share of sales "
         "(a Pareto curve).",
         "A curve with two marked points, for example \"top 20% of rows = X% of units\".",
         "Shows where the volume is: a small group of products makes most of the sales."),
    22: "Heading: 2.5, scatter plots, and what they show.",
    23: ("Left: each store-product as a dot, average sales against variability (coefficient of variation). Right: "
         "the network total, this week against last week (a lag plot).",
         "Slow sellers are the most variable. The lag-plot dots lie near a diagonal line (high correlation).",
         "The lag plot shows that this week depends on last week, which is what AR (autoregressive) models use."),
    24: "Text: reads the scatter plots. Heading 2.6: QQ plots, and what a QQ plot is.",
    25: ("Draws QQ plots for the network total, its log, and the biggest single store-product. Each plot compares the "
         "data with a perfect normal (bell-curve) distribution.",
         "Points follow the line in the middle but bend away at the top (heavy tail).",
         "Sales are not normally distributed. That decides later choices, such as using a count distribution for risk."),
    26: "Text: the total is close to normal in the middle with a heavy upper tail; single products are far from normal. Heading 2.7.",
    27: ("Computes descriptive statistics (mean, standard deviation, quartiles, CV, skewness, kurtosis, share of zero "
         "weeks, Jarque-Bera p-value) for the total, the biggest seller and the median product, then summarises the "
         "same statistics across all 599.",
         "Two tables of numbers.",
         "Puts numbers on what the charts showed: right-skewed, heavy-tailed, many zeros, not normal."),
    28: "Text: reads the statistics. Heading 2.8: stockouts and product groups.",
    29: ("Computes the out-of-stock rate of each of the 67 stores and total sales per department group, and charts both.",
         "Stores range from 0% to 59% of weeks out of stock; one chart of units per department group.",
         "Shows that empty shelves are a real and uneven problem across stores."),
    30: "Text: what the EDA teaches us and how it shapes the models (slow demand, shared season, censoring, shared patterns).",
    31: "Heading: step 3, missing values. There are no empty cells, but two kinds of weeks are hidden missing values.",
    32: "Heading 3.1 and text: zeros before a product's first sale mean it was not on sale yet, not that nobody wanted it.",
    33: ("Finds each store-product's first week with a sale (its launch) and marks every earlier week. Draws a "
         "histogram of launch weeks.",
         "38% sell from the first week; 371 store-products were launched later.",
         "Those early zeros would wrongly drag forecasts down, so they are treated as missing."),
    34: "Heading 3.2 and text: out-of-stock weeks are censored (sales understate demand), so they are set aside.",
    35: ("Builds demand: the sales table with out-of-stock weeks and pre-launch weeks replaced by NaN (empty). Then "
         "measures how much is set aside and what out-of-stock weeks look like.",
         "4.3% of weeks set aside; 100% of out-of-stock weeks have zero sales.",
         "This is the key cleaning step. An empty shelf means nothing could sell, so that zero says nothing about demand."),
    36: ("Picks a busy store-product (more than 3 units a week) that often ran out (over 20% of weeks), and charts "
         "its sales with orange bands on out-of-stock weeks.",
         "A bar chart with orange bands.",
         "Makes the problem visible: the bands fall exactly on the weeks with zero sales."),
    37: ("Text: every flagged week had zero sales. Then 3.3, a table of how each method treats a missing week, and "
         "3.4, demand patterns (ADI and CV²)."),
    38: ("Classifies every store-product with two numbers: ADI (average weeks between sales) and CV² (how much sale "
         "sizes vary). The standard cut-offs give 4 patterns: smooth, erratic, intermittent, lumpy.",
         "A count table and a scatter plot with four coloured groups.",
         "Each pattern suits different methods; Croston and TSB were designed for the intermittent and lumpy ones."),
    39: "Heading 3.5 and a table: how the 165 weeks are split (training, backtest, tuning, final test). No peeking at the future.",
    40: ("Defines the 30 backtest dates (origins): the first 12 for measuring forecast spread, the last 18 for "
         "accuracy and tuning. Draws a timeline of how the weeks are used.",
         "A coloured bar timeline.",
         "Keeps the test honest: the 8 competition weeks are never used to tune anything."),
    41: ("Heading: step 4, stationarity. Explains the ADF test (H0: not stationary, p < 0.05 means stationary) and "
         "KPSS (the opposite), and the two series studied."),
    42: ("Picks the representative store-product (high volume, on sale from week 1, rarely out of stock). Builds "
         "y_net (the network total) and y_row (that product's clean demand, with gaps filled for the tests). Runs "
         "ADF and KPSS on both.",
         "Store 61 · Product 124 (about 99 a week). A table: ADF says stationary for both.",
         "ARIMA-type models need a stationary series, so this is checked first."),
    43: ("Runs the ADF test on every one of the 599 store-products (clean weeks, gaps filled).",
         "94% stationary, 6% not.",
         "Shows the result holds for almost every product, not just the two examples."),
    44: "Text: ADF and KPSS readings. Heading: step 5, transformations (log, first difference, seasonal difference).",
    45: ("Applies each transformation to the network total, re-tests ADF and KPSS, and chooses d (how many times to "
         "difference) with the ADF rule. Plots three versions.",
         "A results table and \"d = 0\" for both series (no differencing needed).",
         "Proves the original series can be used as it is, so forecasts stay in real units."),
    46: "Text: why d = 0 is kept. Heading: step 6, STL decomposition and what trend and season strength mean.",
    47: ("Splits the network total into trend, seasonal and remainder with STL (52-week season), plots the four parts, "
         "and computes trend and seasonal strength for both series.",
         "A four-panel chart; seasonal strength 0.93 for the total.",
         "The data is very seasonal (Christmas), so a seasonal part is needed in the models."),
    48: "Text: reads the decomposition. Heading: step 7, what ACF and PACF are.",
    49: ("Computes the ACF and PACF of the network total (56 lags) and of the representative product (30 lags), "
         "and draws them with the 95% band. No differencing is applied because d = 0.",
         "Two pairs of bar charts; blue bars cross the band (significant), grey bars do not.",
         "These plots show which past weeks matter, which guides the choice of model."),
    50: "Heading: step 8, model identification, with the textbook rules table (AR, MA, ARMA, ARIMA, SARIMA).",
    51: ("Reads the ACF/PACF automatically with those rules and suggests a model family and orders (p, d, q).",
         "Network total: AR with a seasonal spike, so SARIMA. The product: ARMA.",
         "Turns the plots into a starting model, which step 10 then checks with AIC/BIC."),
    52: "Text: why the total looks like a seasonal AR process. Heading: step 9, lags.",
    53: ("Lists the significant lags of the ACF and PACF for both series.",
         "Network total: lags 1-5 and around 52. The product: lags 1 and 5.",
         "Recent weeks and the same week last year carry the information; this also chose LightGBM's inputs."),
    54: "Text: what the lags mean. Heading: step 10, AIC and BIC formulas and the grid to compare.",
    55: ("Fits 27 candidate models to the network total (ARMA, ARIMA with d = 1, and SARIMA with a 52-week seasonal "
         "term) and ranks them by AIC; also shows the BIC winner.",
         "Best by AIC: SARIMA(2,0,0)(1,0,0)[52]; best by BIC: SARIMA(1,0,0)(1,0,0)[52]. A table of the top 10.",
         "Picks the best model objectively, balancing fit against complexity."),
    56: ("Does the same AIC/BIC comparison for the representative product. Missing weeks stay missing; the Kalman "
         "filter skips them.",
         "Best: MA(1) by both AIC and BIC.",
         "Shows the method works for a single product too."),
    57: "Text: the same recipe will now run automatically on all 599 store-products (season removed first).",
    58: ("Fits an ARIMA to every store-product in parallel: ADF chooses d, AIC chooses p and q, and products with "
         "fewer than 26 clean weeks get a simple mean.",
         "Counts per family (ARMA 229, mean only 166, AR 119, MA 53, ARIMA 32) and the most common orders.",
         "This per-product ARIMA becomes one of the forecasting methods in step 12."),
    59: "Text: reads the result. Heading: step 11, residual tests (Ljung-Box, Jarque-Bera, ARCH-LM) and GARCH.",
    60: ("Fits the winning SARIMA to the network total, prints its coefficients, and runs the three tests on its "
         "residuals (what it could not explain).",
         "A coefficient table (all significant) and test results: Ljung-Box borderline at lag 8, ARCH effect found (p = 0.027).",
         "Checks whether the model left any pattern behind."),
    61: ("Draws three diagnostic charts of the residuals: over time, their ACF, and a QQ plot.",
         "Residuals hover around 0; almost all ACF bars are inside the band; the QQ plot has heavy tails.",
         "A visual check that matches the tests."),
    62: "Text: reads the charts; the ARCH effect means calm and turbulent periods alternate, so GARCH is next.",
    63: ("Fits GARCH(1,1) to the residuals, plots the changing volatility, and repeats the tests on the standardised "
         "residuals and their squares.",
         "ω = 0.35, α = 0.18, β = 0.45. After GARCH, Ljung-Box passes and the ARCH effect is gone.",
         "Confirms the leftover is white noise once the changing noise level is modelled."),
    64: "Text: what the GARCH result means (normality still fails because of a few extreme weeks). Heading 11.2.",
    65: ("Uses the SARIMA fitted only on history to forecast the 8 competition weeks, with 80% and 95% ranges, and "
         "compares with what really sold and with \"same week last year\".",
         "Error 14.2% vs 20.0%; all 8 real weeks fall inside the 80% range.",
         "A real out-of-sample test: the model predicts weeks it never saw."),
    66: "Heading 11.3: the same tests on every store-product.",
    67: ("Summarises the residual tests of the 599 ARIMA models, draws p-value histograms, then fits GARCH(1,1) to "
         "every product with an ARCH effect and re-tests.",
         "91% white noise; 10% ARCH effect; GARCH fixes 85% of those.",
         "Shows the classical models are well specified across the whole chain."),
    68: "Text: summary of steps 4 to 11.",
    69: ("Heading: step 12, forecasting methods. Why three weeks ahead, and a table of all 10 methods with their idea."),
    70: "Text: LightGBM's input features and its Tweedie loss, and why.",
    71: ("Trains LightGBM once on all history, makes every method forecast the next 3 weeks for one example product, "
         "and plots them against the history and what really sold.",
         "A chart with three highlighted methods and the others in grey.",
         "Lets you see the methods side by side before measuring them properly."),
    72: "Heading: step 13, the backtest and the four error measures (MAE, Bias, WAPE, Score).",
    73: ("The slowest cell (about 6 minutes). At each of 30 past dates, every method forecasts 3 weeks using only "
         "earlier data (LightGBM refitted every 4 dates, ARIMA every 8). All forecasts are compared with what sold.",
         "The leaderboard table: MAE, RMSE, Bias, WAPE and Score for every method.",
         "A fair test: no method ever sees the weeks it predicts."),
    74: ("Draws the leaderboard as a bar chart.",
         "LightGBM best (57.1%); the official benchmark forecast 63.7%; \"last week\" worst.",
         "Shows clearly which forecast is most accurate."),
    75: "Text: next, accuracy by forecast horizon and by demand pattern.",
    76: ("Two tables: error 1, 2 and 3 weeks ahead, and error per demand pattern.",
         "Further ahead is a little harder; intermittent and lumpy products are much harder.",
         "Shows where forecasting struggles."),
    77: "Text: reads the tables. Then: what drives LightGBM?",
    78: ("Charts LightGBM's 15 most important inputs (share of total gain).",
         "Recent averages and lags lead; season and product groups follow.",
         "Explains the machine-learning model in plain terms."),
    79: "Text and heading 13.1: Ljung-Box on the real forecast errors of the backtest.",
    80: ("For every method and store-product, tests whether the 1-week-ahead errors over 30 weeks are white noise "
         "(Ljung-Box, lag 4).",
         "Most methods: 89-93% of products pass; \"last week\": 52%.",
         "A good forecast leaves no pattern in its mistakes. A pass does not prove accuracy; the leaderboard does that."),
    81: "Text: how to read the result, and why accuracy and white noise together are the evidence.",
    82: "Heading: step 14, uncertainty. Demand is modelled as a negative binomial with mean = forecast and variance k × mean.",
    83: ("Estimates the spread k for every product and method from backtest errors, then checks calibration: does "
         "demand stay below the 50%, 80% and 95% marks about 50%, 80% and 95% of the time?",
         "A table: for example LightGBM 51%, 85%, 98%.",
         "Without the spread, orders would cover only the average and run out about half the time."),
    84: "Text: the ranges are slightly generous, and step 16 corrects for that. Heading: step 15, the simulator, with "
        "the weekly rules written as five lines.",
    85: "Text: before trusting the simulator, reproduce the official benchmark score of €4,334.0.",
    86: ("Plays the organiser's published benchmark orders through our simulator, then rebuilds the benchmark method "
         "and compares its orders. Stops with an error if anything differs.",
         "Setup €913.8 · score €4,334.0 · identical orders: True ✓ simulator verified, and the benchmark week by week.",
         "Proves our scores are directly comparable with the real competition leaderboard."),
    87: "Text: the check passed. Heading: step 16, the ordering policy in 3 steps; its code is newsvendor_orders in "
        "src/optichain/inventory.py.",
    88: "Text: tuning by simulation, replaying 18 past weeks for every method and service level.",
    89: ("Replays the 18 past weeks for every method at 12 service levels, and the benchmark's order-up-to rule "
         "(function tune_service_levels), then keeps the cheapest setting per method (best_settings).",
         "A table of the best service level and its replay cost per method (about 2.5 minutes).",
         "Chooses the safety level from real simulated costs, not from a formula."),
    90: ("Draws cost against service level for every method and picks the overall winner.",
         "U-shaped curves; the chosen plan is Ensemble at service level 0.75.",
         "The bottom of the U is the best balance between empty shelves and leftover stock."),
    91: "Text: better forecasts mean lower cost, and why 0.75 is below the textbook 0.833. Heading: step 17, the final test.",
    92: ("Makes forecasts for each of the 9 rounds (weekly, retraining LightGBM on the data known then) and measures "
         "accuracy on the competition weeks.",
         "An accuracy table for the 8 test weeks (about 6 minutes to compute the forecasts).",
         "Simulates exactly how the competition worked, week by week."),
    93: "Text: an honest surprise. On these 8 weeks the simple benchmark forecast was the most accurate.",
    94: ("Plays the 8 competition weeks for every method with its service level tuned on the past, and also the "
         "order-up-to rule; records the score and the \"hindsight best\".",
         "A table: Ensemble €3,626 is lowest; LightGBM €3,680; ARIMA €3,757.",
         "The real test: what each forecast would have cost in the actual competition."),
    95: ("Charts every method's score against the official benchmark and the real winner.",
         "Plan: Ensemble at 0.75, €3,626.4, 16.3% below the benchmark.",
         "The headline result of the project."),
    96: "Text: how to read the result honestly (the hindsight caveat), and the weekly cost comparison.",
    97: ("Re-runs the chosen plan week by week (P.run_plan) and charts the weekly costs (leftover and missed "
          "sales) against the benchmark.",
          "Stacked bars per week, and a table.",
          "Shows where the saving comes from: much less leftover stock."),
    98: "Heading: step 18, stock outlook. What to order after the data ends.",
    99: ("Builds a table for every store-product: stock on the shelf, stock on its way, forecast, order now, and "
          "the chance of running out in weeks 9, 10 and 11.",
          "Order 1,162 units for 271 products; 19 products have a high risk next week.",
          "Turns the analysis into a concrete action list for the planner."),
    100: "Text: weeks 9 and 10 are already decided; the new order protects week 11. Heading: step 19, save.",
    101: ("Saves every result (tables, forecasts, scores, time-series results) to the artifacts folder and lists the files.",
          "\"Saved N files\" to the artifacts folder.",
          "The dashboard reads these files, so the dashboard and the notebook always agree."),
    102: "Heading: step 20, conclusions.",
    103: ("Prints the conclusions with the real numbers filled in automatically.",
          "Four lines: time-series result, forecasting, stock policy, final result.",
          "Numbers come from the run itself, so the text can never be out of date."),
    104: "Text: limitations and next steps, and the data citation.",
}
CHECK = {2: "import json", 55: "# 27 candidates", 73: "# At each of the 30", 86: "costs = I.Costs", 101: "# Tables"}

# ---------------------------------------------------------------- style
FD = Path("C:/Windows/Fonts")
try:
    pdfmetrics.registerFont(TTFont("Body", str(FD / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FD / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(FD / "ariali.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FD / "consola.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    F, FB, FM = "Body", "Body-Bold", "Mono"
except Exception:  # noqa: BLE001
    F, FB, FM = "Helvetica", "Helvetica-Bold", "Courier"

INK, INK2, MUTED = colors.HexColor("#0f172a"), colors.HexColor("#475569"), colors.HexColor("#8a94a6")
BLUE, BLUE_INK, LINE = colors.HexColor("#2a78d6"), colors.HexColor("#1c5cab"), colors.HexColor("#e2e7ef")
CODE_BG, OUT_BG, MD_BG = colors.HexColor("#f6f8fb"), colors.HexColor("#fbfcfe"), colors.HexColor("#f4f6f9")
GREEN = colors.HexColor("#1b7a4b")

S = {
    "title": ParagraphStyle("t", fontName=FB, fontSize=26, leading=31, textColor=INK, spaceAfter=8),
    "sub": ParagraphStyle("s", fontName=F, fontSize=12, leading=18, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName=FB, fontSize=16, leading=21, textColor=INK, spaceBefore=10, spaceAfter=6,
                         keepWithNext=1),
    "body": ParagraphStyle("b", fontName=F, fontSize=10, leading=14.5, textColor=INK, spaceAfter=5),
    "label": ParagraphStyle("l", fontName=FB, fontSize=8.5, leading=11, textColor=BLUE_INK),
    "explain": ParagraphStyle("e", fontName=F, fontSize=9.6, leading=13.6, textColor=INK),
    "md": ParagraphStyle("m", fontName=F, fontSize=9.2, leading=13, textColor=INK2),
    "code": ParagraphStyle("c", fontName=FM, fontSize=7.6, leading=9.8, textColor=INK),
    "out": ParagraphStyle("o", fontName=FM, fontSize=7.4, leading=9.4, textColor=GREEN),
    "cell": ParagraphStyle("ce", fontName=F, fontSize=8.8, leading=12, textColor=INK),
    "head": ParagraphStyle("he", fontName=FB, fontSize=8.8, leading=12, textColor=colors.white),
}
W = A4[0] - 4 * cm


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono(text: str, style: str, max_lines: int) -> Paragraph:
    lines = text.rstrip().splitlines()
    more = len(lines) - max_lines
    lines = [ln[:118] + ("…" if len(ln) > 118 else "") for ln in lines[:max_lines]]
    if more > 0:
        lines.append(f"… ({more} more lines)")
    body = "<br/>".join(esc(ln).replace(" ", "&nbsp;") for ln in lines)
    return Paragraph(body, S[style])


def boxed(flow, bg, bar=None):
    t = Table([[flow]], colWidths=[W])
    st = [("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
          ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if bar:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 2.5, bar))
    t.setStyle(TableStyle(st))
    return t


def cell_header(i: int, kind: str):
    tag = "CODE CELL" if kind == "code" else "TEXT CELL"
    col = "#1c5cab" if kind == "code" else "#8a94a6"
    return Paragraph(f'<font color="{col}"><b>Cell {i}</b> &nbsp;·&nbsp; {tag}</font>', S["label"])


def outputs_of(cell):
    texts, images, tables = [], [], 0
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            texts.append(re.sub(r"origin \d+/\d+|round \d+/\d+", "", o.text).strip())
        elif o.output_type in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                images.append(d["image/png"])
            elif "text/html" in d or "text/plain" in d:
                tables += 1
    return [t for t in texts if t], images, tables


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont(FB, 8); canvas.setFillColor(BLUE_INK)
        canvas.drawString(2 * cm, A4[1] - 1.25 * cm, "OptiChain")
        canvas.setFont(F, 8); canvas.setFillColor(MUTED)
        canvas.drawString(3.45 * cm, A4[1] - 1.25 * cm, "The notebook, cell by cell")
        canvas.setStrokeColor(LINE); canvas.line(2 * cm, A4[1] - 1.42 * cm, A4[0] - 2 * cm, A4[1] - 1.42 * cm)
        canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, str(doc.page))
    else:
        canvas.setFillColor(BLUE_INK); canvas.rect(0, A4[1] - .9 * cm, A4[0], .9 * cm, stroke=0, fill=1)
    canvas.restoreState()


def build():
    nb = nbf.read(NB, as_version=4)
    for i, start in CHECK.items():
        if not nb.cells[i].source.lstrip().startswith(start):
            raise SystemExit(f"Cell {i} no longer starts with {start!r}: update EXPL in this script.")
    if len(nb.cells) != len(EXPL):
        raise SystemExit(f"The notebook has {len(nb.cells)} cells but {len(EXPL)} explanations: update EXPL.")
    n_code = sum(c.cell_type == "code" for c in nb.cells)

    st = [Spacer(1, 3 * cm), Paragraph("NOTEBOOK GUIDE", S["label"]), Spacer(1, 4),
          Paragraph("The OptiChain notebook, cell by cell", S["title"]),
          Paragraph(f"Every one of the {len(nb.cells)} cells of <i>notebooks/OptiChain_VN2_Pipeline.ipynb</i> explained in "
                    f"simple words: {n_code} code cells and {len(nb.cells) - n_code} text cells, with a short look at each "
                    "cell's code and its real output from the last run.", S["sub"]),
          Spacer(1, 1 * cm), Paragraph("How to read this guide", S["h1"])]
    legend = [["Box", "What it shows"],
              ["What it does", "the cell's job, in plain words"],
              ["What you see", "the result it prints or draws, as in the last run"],
              ["Why it matters", "why this step is in the project"],
              ["Code (start)", "the first lines of the cell's code, for reference"],
              ["Output", "the real printed output; charts are shown as small pictures"],
              ["Text cell", "a short summary of the explanation written in the notebook"]]
    lt = Table([[Paragraph(c, S["head"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(legend)],
               colWidths=[.28 * W, .72 * W])
    lt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE_INK), ("LINEBELOW", (0, 0), (-1, -1), .4, LINE),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    st += [lt, Spacer(1, .8 * cm),
           Paragraph("<b>Short vocabulary.</b> <i>Store-product</i> = one product in one store (599 in total). "
                     "<i>Week 0</i> = 8 April 2024, the last week of history. <i>Clean demand</i> = sales with out-of-stock "
                     "and pre-launch weeks removed. <i>NaN</i> = an empty (missing) value. <i>p-value</i> = the chance of "
                     "seeing the result if the test's starting assumption were true; below 0.05 means \"reject it\".",
                     S["body"]),
           PageBreak()]

    section = None
    for i, cell in enumerate(nb.cells):
        exp = EXPL[i]
        if cell.cell_type == "markdown":
            m = re.search(r"^## (.+)$", cell.source, flags=re.M)
            if m and m.group(1) != section:
                section = m.group(1)
                st.append(Paragraph(esc(section.replace("·", "·")), S["h1"]))
            st.append(KeepTogether([cell_header(i, "md"), Spacer(1, 2), boxed(Paragraph(esc(exp), S["md"]), MD_BG),
                                    Spacer(1, 9)]))
            continue
        what, see, why = exp
        texts, images, n_tables = outputs_of(cell)
        block = [cell_header(i, "code"), Spacer(1, 3),
                 boxed(Paragraph(f"<b>What it does.</b> {esc(what)}<br/><b>What you see.</b> {esc(see)}<br/>"
                                 f"<b>Why it matters.</b> {esc(why)}", S["explain"]), colors.HexColor("#eef5fe"), BLUE),
                 Spacer(1, 4), Paragraph("Code (start)", S["label"]), Spacer(1, 1),
                 boxed(mono(cell.source, "code", 7), CODE_BG)]
        if texts:
            block += [Spacer(1, 4), Paragraph("Output", S["label"]), Spacer(1, 1),
                      boxed(mono("\n".join(texts), "out", 9), OUT_BG)]
        elif n_tables and not images:
            block += [Spacer(1, 3), Paragraph(f'<font color="#8a94a6">Output: {n_tables} table(s), shown in the notebook.</font>',
                                              S["md"])]
        st.append(KeepTogether(block))
        for b64 in images[:2]:
            im = Image.open(io.BytesIO(base64.b64decode(b64)))
            w, h = im.size
            width = min(W * .82, 15 * cm)
            height = width * h / w
            if height > 11 * cm:
                width, height = width * 11 * cm / height, 11 * cm
            st += [Spacer(1, 4), RLImage(io.BytesIO(base64.b64decode(b64)), width=width, height=height)]
        if len(images) > 2:
            st.append(Paragraph(f'<font color="#8a94a6">+ {len(images) - 2} more chart(s) in the notebook.</font>', S["md"]))
        st.append(Spacer(1, 12))

    st += [PageBreak(), Paragraph("The whole notebook in one table", S["h1"]),
           Paragraph("Each step, the cells that do it, and its key result.", S["body"])]
    starts = [(i, int(re.match(r"## (\d+)", c.source).group(1))) for i, c in enumerate(nb.cells)
              if c.cell_type == "markdown" and re.match(r"## \d+", c.source)]
    span = {n: f"{i}-{(starts[k + 1][0] - 1) if k + 1 < len(starts) else len(nb.cells) - 1}" for k, (i, n) in enumerate(starts)}
    result = {0: "tools loaded, seed fixed",
              1: "599 store-products × 165 weeks; files consistent; 43% zero weeks",
              2: "strong Christmas peak; about 1 unit a week typical; not normal; stores 0-59% out of stock",
              3: "pre-launch and out-of-stock weeks set aside (100% of out-of-stock weeks = 0 sales)",
              4: "ADF: total stationary; 94% of products stationary", 5: "no differencing needed (d = 0)",
              6: "seasonal strength 0.93", 7: "lags 1-5 and about 52 matter", 8: "the total looks like SARIMA",
              9: "recent weeks + the same week last year", 10: "SARIMA(2,0,0)(1,0,0)[52]; ARIMA fitted for all 599",
              11: "ARCH found → GARCH(1,1) → white noise; 91% of products white noise", 12: "10 methods",
              13: "LightGBM most accurate (57.1% vs 63.7% benchmark)", 14: "negative binomial spread; ranges slightly generous",
              15: "reproduces the official €4,334.0 exactly", 16: "service level 0.75, chosen by replaying 18 past weeks",
              17: "Ensemble plan €3,626 vs benchmark €4,334 (−16%)", 18: "order 1,162 units for 271 products",
              19: "results saved for the dashboard", 20: "conclusions"}
    titles = {i: re.match(r"## (.+)", nb.cells[i].source).group(1) for i, _ in starts}
    summary = [["Step", "Cells", "Key result"]] + [[titles[i], span[n], result.get(n, "")] for i, n in starts]
    t = Table([[Paragraph(c, S["head"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(summary)],
              colWidths=[.3 * W, .12 * W, .58 * W], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE_INK), ("LINEBELOW", (0, 0), (-1, -1), .4, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           *[("BACKGROUND", (0, r), (-1, r), CODE_BG) for r in range(2, len(summary), 2)]]))
    st.append(t)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm,
                            bottomMargin=1.7 * cm, title="OptiChain notebook, cell by cell", author="OptiChain")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
