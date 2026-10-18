"""Build docs/OptiChain_Explained_Simply.pdf: the whole project in simple language, in one PDF.

Part 1 the full story · Part 2 every term explained (what it is, an example, why we used it) · Part 3 every model
explained (how it works, strengths, weaknesses, why it is in the comparison, how it scored) · Part 4 why we compare
models · Part 5 the short recap.

    python scripts/build_simple_guide.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import build_guide as G  # noqa: E402  (shared fonts, styles and charts)

ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "OptiChain_Explained_Simply.pdf"
S, W, F, FB = G.S, G.W, G.F, G.FB
P, bullets, table, callout, kpi_row, image = G.P, G.bullets, G.table, G.callout, G.kpi_row, G.image
S["part"] = ParagraphStyle("part", fontName=FB, fontSize=26, leading=31, textColor=G.INK, spaceAfter=10)
S["term"] = ParagraphStyle("term", fontName=FB, fontSize=12, leading=16, textColor=G.BLUE_INK, spaceBefore=10, spaceAfter=3,
                           keepWithNext=1)
S["lab"] = ParagraphStyle("lab", fontName=FB, fontSize=9.6, leading=14, textColor=G.INK)
for k in ("h1", "h2", "h3"):
    S[k].keepWithNext = 1


def H(n, title):
    return [PageBreak(), P(f"PART {n}", "kicker"), Paragraph(title, S["part"])]


def term_card(name, what, example, why):
    rows = [[Paragraph("<b>What it is</b>", S["lab"]), Paragraph(what, S["body"])],
            [Paragraph("<b>Example</b>", S["lab"]), Paragraph(example, S["body"])],
            [Paragraph("<b>Why we used it</b>", S["lab"]), Paragraph(why, S["body"])]]
    t = Table(rows, colWidths=[.2 * W, .8 * W])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), .4, G.LINE),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f6f8fb")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("BOX", (0, 0), (-1, -1), .5, G.LINE)]))
    return KeepTogether([Paragraph(name, S["term"]), t, Spacer(1, 4)])


def model_card(name, how, example, strong, weak, why, result):
    rows = [["How it works", how], ["Small example", example], ["Strengths", strong], ["Weaknesses", weak],
            ["Why it is in the comparison", why], ["How it did", result]]
    t = Table([[Paragraph(f"<b>{a}</b>", S["lab"]), Paragraph(b, S["body"])] for a, b in rows], colWidths=[.24 * W, .76 * W])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), .4, G.LINE),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f6f8fb")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("BOX", (0, 0), (-1, -1), .5, G.LINE)]))
    return KeepTogether([Paragraph(name, S["h2"]), t, Spacer(1, 6)])


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(G.LINE); canvas.setLineWidth(.5)
        canvas.line(2.1 * cm, A4[1] - 1.5 * cm, A4[0] - 2.1 * cm, A4[1] - 1.5 * cm)
        canvas.setFont(FB, 8); canvas.setFillColor(G.BLUE_INK)
        canvas.drawString(2.1 * cm, A4[1] - 1.3 * cm, "OptiChain")
        canvas.setFont(F, 8); canvas.setFillColor(G.MUTED)
        canvas.drawString(3.6 * cm, A4[1] - 1.3 * cm, "Explained simply")
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
    else:
        canvas.setFillColor(G.BLUE_INK); canvas.rect(0, A4[1] - 0.9 * cm, A4[0], 0.9 * cm, stroke=0, fill=1)
    canvas.restoreState()


def build():
    summ = json.loads((ART / "summary.json").read_text())
    ts = json.loads((ART / "ts_network.json").read_text())
    lb = pd.read_csv(ART / "leaderboard.csv").set_index("model")["Score %"]
    comp = pd.read_csv(ART / "competition_scores.csv").set_index("model")["VN2 score €"]
    sc = lambda m: f"{lb[m]:.1f}%"
    eur = lambda m: f"€{comp[m]:,.0f}"
    saved = summ["benchmark_score"] - summ["plan_score"]

    st = []
    # ---------------------------------------------------------------- cover
    st += [Spacer(1, 3 * cm), P("EXPLAINED SIMPLY", "kicker"), P("OptiChain", "title"),
           P("The whole project in plain language: the full story, every term (from stationarity and the ADF test to "
             "ARCH and GARCH), every model, and the reason behind every choice.", "subtitle"), Spacer(1, 1 * cm),
           kpi_row([("Saved vs the official method", f"€{saved:,.0f}", "in 6 weeks · 16% cheaper"),
                    ("Over a year (estimate)", f"≈ €{round(saved * 52 / 6, -2):,.0f}", "if the test weeks are typical"),
                    ("Stock on shelves", "−45%", "much less stock gathering dust"),
                    ("Best forecaster", "LightGBM", f"error {sc('lightgbm')} vs {sc('seasonal_ma')}")]),
           Spacer(1, .6 * cm), P("<b>Contents</b>", "h3"),
           P("<b>Part 1</b> · The full story, start to finish<br/><b>Part 2</b> · Every term explained<br/>"
             "<b>Part 3</b> · Every model explained<br/><b>Part 4</b> · Why we compare ten models<br/>"
             "<b>Part 5</b> · The short recap", "toc")]

    # ================================================================ PART 1
    st += H(1, "The full story, start to finish")
    story = [
        ("1 · The problem we are trying to solve", [
            "Imagine you run a chain of 67 shops. Every shop sells some products, and every Monday morning you place an "
            "order with a central warehouse to refill the shelves. The catch is that the delivery takes two weeks: what "
            "you order today only reaches the shelf two Mondays from now. So every order is a bet on the future. You must "
            "decide today how much customers will want in two weeks' time.",
            "Two things can go wrong. If you order too little, the shelf goes empty, a customer comes in, finds nothing "
            "and leaves. You lose that sale for good, because the customer does not wait. The competition puts a price "
            "of <b>€1 on every missed sale</b>. If you order too much, the extra items sit on the shelf. Holding stock "
            "costs money (space, capital, risk of damage), and the competition charges <b>€0.20 for every item still on "
            "the shelf at the end of each week</b>.",
            "A missed sale is five times more expensive than keeping one item for a week. That one fact shapes the whole "
            "project: it is better to keep a little extra stock than to run out, but extra stock is not free either. The "
            "goal is to find the balance, ordering the amount that keeps the <b>total cost</b> (missed sales plus leftover "
            "stock) as low as possible, for every product in every shop, every week."]),
        ("2 · The data and why we chose it", [
            "The data comes from a real competition called the <b>VN2 Inventory Planning Challenge</b>, run in 2025 by a "
            "supply-chain expert, Nicolas Vandeput. About 180 teams entered. The sales numbers are real weekly sales from "
            "an anonymous retail chain, covering April 2021 to June 2024: 165 weeks in total.",
            "The chain has 67 shops and 297 products, but not every shop sells every product. When you count the "
            "combinations that actually exist, there are <b>599 store-products</b>. A store-product is simply one product "
            "in one shop. \"Product 124 in Shop 61\" is one store-product, and \"Product 124 in Shop 63\" is a different one. "
            "They are treated separately because the same product can sell very differently in two places.",
            "Shops and products appear only as numbers because the company chose to stay anonymous. We never invented real "
            "names. On the dashboard five example products get descriptive nicknames such as \"The Christmas Rocket\" "
            "(it sells almost three times more in December), but the nickname only describes how the product behaves, "
            "and its real code is always shown next to it.",
            "The project first used Walmart sales data (the M5 dataset). We switched because that data is older "
            "(2011-2016) and has no information about stock, costs or delivery times, so those had to be made up. VN2 "
            "gives us the <b>real stock levels, the real costs, the real two-week delivery rule and an official score</b>, "
            "so we can measure our work against 180 real teams.",
            "We split the 165 weeks into two parts. The first <b>157 weeks</b> are history, used for learning. The last "
            "<b>8 weeks</b> are kept hidden and used only once, at the very end, as the final exam. If we tuned anything "
            "on those 8 weeks, the final score would be dishonestly good."]),
        ("3 · Checking the data before trusting it", [
            "Before building anything, we checked that the data was healthy: no empty cells, no negative sales, no "
            "fractions of an item, and all nine weekly files agreed with each other. Everything passed. This matters "
            "because any problem here would quietly corrupt every result later.",
            "Two facts stood out. First, <b>43% of all weeks have zero sales</b>: most products are slow sellers, about one "
            "unit a week, and many weeks they sell nothing at all. That makes forecasting hard, because there is very "
            "little signal in such small numbers. Second, <b>about one week in ten was flagged as out of stock</b>, meaning "
            "the shelf was empty at the end of that week. That turns out to be crucial."]),
        ("4 · Looking at the data with charts (EDA)", [
            "EDA (exploratory data analysis) means studying the data with charts and simple statistics before choosing "
            "any method. Each chart answers one question.",
            "<b>The total sales line graph</b> adds up all 599 store-products week by week. It shows a steady level of "
            "about 1,760 units a week with a large spike every December: a strong <b>Christmas season</b>. <b>The seasonal "
            "profile</b> puts numbers on it: Christmas weeks sell about <b>2.4 times</b> a normal week.",
            "<b>The histograms</b> show that the middle product sells about 1.1 units a week and three quarters sell fewer "
            "than 2. <b>The Pareto curve</b> shows that a small group of products makes most of the sales, so mistakes on "
            "big sellers matter most. <b>The scatter plots</b> show that small sellers are the most irregular, and that this "
            "week's total is strongly linked to last week's. <b>The QQ plots</b> show that sales do not follow a bell curve. "
            "<b>The stockout charts</b> show that empty-shelf weeks range from 0% in the best store to 59% in the worst.",
            "From this we learned four things that shaped every later choice: demand is slow and often zero; there is a "
            "strong yearly season shared by all shops; empty shelves are common and distort the data; and products share "
            "patterns, so learning from all of them together should help."]),
        ("5 · Hidden missing values: the most important cleaning step", [
            "The audit said there are no missing values: every cell has a number. But some of those numbers do not mean "
            "what they seem to mean.",
            "<b>The first problem is products before their launch.</b> Many products show zeros in their first weeks, not "
            "because nobody wanted them but because they were not on sale in that shop yet. Every week before a product's "
            "first sale is treated as missing.",
            "<b>The second, bigger problem is empty-shelf weeks.</b> When the shelf was empty, sales were zero, but customers "
            "may well have wanted the product; they just could not buy it. This is called <b>censored demand</b>. We found "
            "that <b>100% of out-of-stock weeks recorded exactly zero sales</b>: those zeros measure supply, not demand.",
            "We set these weeks aside, and call what remains <b>clean demand</b>. Filling them with zero would teach the "
            "models that people want less than they do, so they would order too little and cause even more empty shelves. "
            "Filling them with a guess would be invented information. So each method simply skips these weeks.",
            "We also sorted every product into one of four <b>demand patterns</b>: steady (sells almost every week, similar "
            "amounts), spiky (sells most weeks, amounts jump), occasional (many weeks with no sales) and unpredictable "
            "(irregular weeks and amounts). Most of the 599 are occasional or unpredictable."]),
        ("6 · The classical time-series analysis", [
            "A time series is a list of numbers in time order, like weekly sales. Statisticians have a classic recipe "
            "for building a forecasting model from one, the <b>Box-Jenkins method</b>, and we followed it step by step on "
            "the network total, on one representative product (Store 61 · Product 124), and automatically on all 599.",
            "<b>Is it stable?</b> The ADF test said yes for the total and for 94% of products. <b>Does it need transforming?</b> "
            "No, it was already stable. <b>Trend, season and noise:</b> STL found a seasonal strength of 0.93 out of 1, so "
            "the yearly pattern dominates. <b>Which past weeks matter?</b> The ACF and PACF charts showed the last one to five "
            "weeks and the same week last year. <b>Which model type?</b> The pattern pointed to SARIMA. <b>Which exact "
            f"version?</b> Out of 27 candidates, AIC chose <b>{ts['best']}</b>: this week depends on the last two weeks and "
            "on the same week last year.",
            "<b>What does the model leave behind?</b> The ARCH test found calm and wild periods (Christmas is wild), so we "
            "added <b>GARCH</b>, after which no pattern remained: the leftovers are pure random noise. <b>Does it predict "
            f"unseen weeks?</b> It forecast the 8 hidden weeks with {ts['holdout_mape']:.0f}% error (against 20% for copying "
            "last year), and all 8 real weeks fell inside its predicted range. Across all 599 products, "
            f"{ts['rows_white_noise']:.0%} of the individual ARIMA models left pure noise behind.",
            "Part 2 explains every one of these terms in detail."]),
        ("7 · The ten forecasting models", [
            "We did not pick one method and hope for the best. We built ten, from the simplest guess (\"next week = last "
            "week\") to machine learning, and compared them fairly so that the final choice is proven rather than assumed. "
            "Part 3 explains each one; Part 4 explains why comparing them matters."]),
        ("8 · Testing the models fairly (the backtest)", [
            "We went back in time to <b>30 different past weeks</b>. At each one, every model could see only the data up to "
            "that week and had to forecast the next three weeks; then we compared the forecasts with what actually sold. "
            "Letting a model see the future would be cheating (leakage). Three weeks because an order placed now arrives "
            "in two weeks, so the planner needs this week, next week and the arrival week.",
            f"<b>LightGBM won clearly, with an error of {sc('lightgbm')}.</b> The ensemble came second ({sc('ensemble')}), "
            f"the official benchmark scored {sc('seasonal_ma')}, ARIMA {sc('arima')}, and \"last week\" was worst "
            f"({sc('naive')}). The errors look large because most products sell one or two units a week: guessing 2 when 1 "
            "sells is already a 100% error. What matters is the comparison, and later the cost."]),
        ("9 · Knowing how uncertain a forecast is", [
            "A forecast is just the most likely value. Ordering exactly that would run out about half the time. So we "
            "modelled how spread out demand is around each forecast, with a <b>negative binomial distribution</b> (a shape "
            "for counts with many zeros). When it said \"80% chance demand stays below X\", demand stayed below X about 85% "
            "of the time: slightly cautious, and corrected in the next steps."]),
        ("10 · The shop simulator", [
            "We wrote a simulator that plays the shops week by week with the exact competition rules. Fed the organiser's "
            "own benchmark orders, it produced exactly <b>€4,334.0</b>, the official number, to the cent. So every score in "
            "the project is directly comparable with the real competition."]),
        ("11 · Deciding how much to order", [
            "We used the <b>Monte Carlo newsvendor</b>: imagine 1,000 possible futures, see how much stock will be left when "
            "the new order arrives, and order enough to cover demand in a chosen share of those futures (the <b>service "
            "level</b>). The textbook share is 83%, but replaying 18 past weeks showed <b>75%</b> was cheaper, because our "
            "uncertainty estimate was slightly cautious and leftover stock is not wasted (it sells the next week)."]),
        ("12 · The final exam: the eight hidden weeks", [
            f"Replaying the competition exactly, with nothing tuned on these weeks, our plan (the ensemble at 75%) cost "
            f"<b>€{summ['plan_score']:,.0f}</b>, against <b>€{summ['benchmark_score']:,.0f}</b> for the official method: "
            f"<b>€{saved:,.0f} (16%) cheaper</b>, roughly €6,100 a year. The real winner scored €3,763.",
            "The saving comes from keeping about <b>45% less stock</b> on the shelves, in exchange for somewhat more missed "
            "sales. Because a leftover costs €0.20 every week while a missed sale costs €1 once, holding much less stock "
            "saves more than the extra missed sales cost.",
            "Two honest caveats: we built this after the competition, knowing which methods worked, so \"we beat the "
            "official method\" is fair but \"we beat the winner\" is not fully fair; and on these 8 weeks the simple "
            "benchmark made slightly more accurate forecasts, yet our orders still cost less."]),
        ("13 · What to order next", [
            "The data ends on 3 June 2024, so the system treats 10 June 2024 as \"today\" and recommends ordering "
            "<b>1,162 units across 271 store-products</b>, arriving on 24 June. It flags 19 products likely to run out next "
            "week, which is already too late to fix because any order takes two weeks. It cannot plan for today's real "
            "date: there is no data after June 2024."]),
        ("14 · The lessons", [
            "Cleaning the data mattered as much as the models. Seasonality is strong and best learned across all shops. "
            "The classical statistics check out. Machine learning predicts best because it learns from all products "
            "together. A good forecast is not enough: the ordering rule must handle uncertainty and real costs. And "
            "testing decisions with simulated costs beats trusting a formula."]),
    ]
    for title, paras in story:
        st.append(P(title, "h2"))
        st += [P(p) for p in paras]

    # ================================================================ PART 2 terms
    st += H(2, "Every term explained")
    st += [P("Each term has three parts: <b>what it is</b> in plain words, a small <b>example</b>, and <b>why we used "
             "it</b> in this project. The terms are grouped by topic, in the order you meet them.")]
    groups = [
        ("The business terms", [
            ("Store-product", "One product in one shop. It is the unit of every decision in this project.",
             "Product 124 in Shop 61 and Product 124 in Shop 63 are two different store-products.",
             "Each one sells differently, so each gets its own forecast and its own order. There are 599."),
            ("Lead time", "The time between placing an order and the stock arriving on the shelf.",
             "Order on Monday 15 April, the stock arrives on Monday 29 April: a lead time of two weeks.",
             "Because of it, every order must be based on a forecast of the future, not on what is on the shelf today."),
            ("Holding cost", "The cost of keeping one unsold item on the shelf for one week. In VN2 it is €0.20.",
             "10 items left at the end of a week cost €2.", "It is half of the total cost we try to minimise."),
            ("Shortage cost (lost sale)", "The cost of a customer wanting an item when the shelf is empty. In VN2 it is "
             "€1 per unit, and the sale is lost (the customer does not wait).", "5 customers find an empty shelf: €5.",
             "It is the other half of the total cost, and five times more expensive than holding."),
            ("Week 0", "8 April 2024, the last week of history before the competition weeks.",
             "Week 1 is 15 April 2024, week 8 is 3 June 2024.", "Everything up to week 0 is for learning; weeks 1-8 are the final test."),
        ]),
        ("The data terms", [
            ("Censored demand", "Sales that are lower than real demand because the shelf ran out. The real demand is "
             "hidden, or \"censored\", behind the recorded number.", "10 customers wanted an item, but only 6 were on the "
             "shelf: the record says 6, the real demand was 10. In our data the shelf was empty all week, so the record says 0.",
             "We found that every out-of-stock week shows zero sales, so we set those weeks aside instead of treating them as demand."),
            ("Clean demand", "Sales with the untrustworthy weeks removed: weeks before launch and out-of-stock weeks.",
             "A product launched in week 5 and out of stock in week 20: weeks 1-4 and 20 are removed.",
             "All models learn only from clean demand, so they are not dragged down by false zeros."),
            ("NaN (missing value)", "\"Not a number\": an empty value used to mark a week we do not trust.",
             "An out-of-stock week becomes NaN instead of 0.", "Models skip NaN weeks instead of treating them as real zeros."),
            ("Demand pattern (ADI and CV²)", "A label for how a product sells. ADI is the average number of weeks between "
             "sales; CV² measures how much the size of a sale varies. Together they give four patterns: steady, spiky, "
             "occasional, unpredictable.", "A product that sells 2 units every week is steady; one that sells 0, 0, 7, 0, 1 is unpredictable.",
             "It tells us which products are hard to forecast and which special methods (Croston, TSB) to include."),
            ("EDA (exploratory data analysis)", "Studying the data with charts and simple statistics before building models.",
             "Drawing total sales over time and spotting the Christmas peak.", "It decided many later choices: the season, slow demand, censoring."),
            ("Seasonality and seasonal factor", "A pattern that repeats at the same time every year. The seasonal factor of a "
             "week is its average sales divided by an average week.", "Christmas weeks have a factor of about 2.4: they sell 2.4 times a normal week.",
             "Forecasts must know the time of year, so most models use these factors."),
        ]),
        ("The statistics terms: distributions and tests", [
            ("Hypothesis test and p-value", "A test starts from an assumption (the \"null hypothesis\", H0) and asks how "
             "likely the data would be if it were true. The p-value is that probability. If it is below 0.05 (5%), we reject "
             "the assumption.", "Assumption: \"the coin is fair\". Ten heads in a row has a p-value of about 0.001, so we "
             "reject it: the coin is probably not fair.", "Every statistical test in the project (ADF, KPSS, Ljung-Box, "
             "Jarque-Bera, ARCH-LM) gives a p-value that we read this way."),
            ("Normal distribution (bell curve)", "The familiar symmetric bell shape, where most values are near the average "
             "and extreme values are rare.", "People's heights roughly follow a bell curve.",
             "Many classical methods assume it. We checked whether our data follows it (it does not), which changed how we model uncertainty."),
            ("QQ plot", "A chart that compares the data with a bell curve. If the points lie on a straight line, the data is "
             "normal; if they bend away, it is not.", "Our sales bend sharply upwards at the end: a few very big weeks.",
             "It showed that sales are not normal, so we model uncertainty with a count distribution instead."),
            ("Skewness and kurtosis", "Skewness measures how lopsided the data is; kurtosis measures how heavy the extremes "
             "are.", "Most weeks sell 0-2 units but a few sell 30: strongly skewed, heavy extremes.",
             "Both confirmed that sales are far from a bell curve."),
            ("Jarque-Bera test", "A test of normality based on skewness and kurtosis. H0: the data is normal. p below 0.05 means not normal.",
             "Our residuals gave p = 0.000: not normal.", "It checks whether normal-based prediction ranges would be trustworthy."),
            ("Poisson and negative binomial distributions", "Shapes for counts (0, 1, 2, ...). Poisson is for pure random "
             "counts; the negative binomial allows more spread, for counts that are sometimes zero and sometimes spike.",
             "A product averaging 2 a week might sell 0, 1, 2, 5 or 9: the negative binomial describes that spread.",
             "We use it to describe the uncertainty around each forecast, because weekly sales are small counts with many zeros."),
        ]),
        ("The statistics terms: stability and seasons", [
            ("Stationarity (a stationary series)", "A series is stationary when its average level and its spread stay the "
             "same over time. It moves around, but around a stable centre, and does not drift away.", "Daily temperature "
             "differences from the yearly average are roughly stationary; a company's growing revenue is not (it drifts upward).",
             "AR, MA, ARMA and ARIMA models only work properly on stationary data, so stationarity is the first thing to check."),
            ("Unit root and random walk", "A random walk is a series where each value is the previous value plus a random "
             "step, so it can wander anywhere and never returns to a centre. Having a \"unit root\" means behaving like this.",
             "A drunk person's walk: each step starts from wherever the last one ended.",
             "A unit root means the series is not stationary, and the ADF test is designed to detect one."),
            ("ADF test (Augmented Dickey-Fuller)", "The standard test for stationarity. Its assumption (H0) is that the "
             "series has a unit root, i.e. it is NOT stationary. A p-value below 0.05 rejects that, so the series is stationary. "
             "\"Augmented\" means it also accounts for the effect of recent weeks.",
             "Network total: ADF p-value 0.006, below 0.05, so the series is stationary.",
             "To check that the classical models can be used, and to decide whether differencing is needed. 94% of our products passed."),
            ("KPSS test", "A second stationarity test that asks the opposite question: its H0 is that the series IS stationary. "
             "p below 0.05 means not stationary.", "If ADF says stationary and KPSS agrees, we are confident; if they disagree, "
             "we look closer.", "A cross-check on ADF, to avoid a wrong conclusion from one test alone."),
            ("Transformation: log and differencing", "Ways to make a series stationary. The log shrinks big values (calming a "
             "spread that grows with the level). Differencing replaces each value by its change from the previous week. A "
             "seasonal difference subtracts the same week last year.", "Sales of 100, 110, 125 become differences of 10 and 15.",
             "We tested them all; none was needed because the data was already stationary (d = 0)."),
            ("d (order of differencing)", "How many times the series is differenced before modelling. d = 0 means use it as it is.",
             "d = 1 means model the weekly changes instead of the levels.", "The ADF test chose d = 0 for our series."),
            ("Trend", "The slow, long-term movement of a series, up or down.", "Sales gently rising over three years.",
             "One of the three parts STL separates."),
            ("STL decomposition", "Seasonal-Trend decomposition using LOESS: it splits a series into trend + season + "
             "remainder (noise), and gives each part a strength from 0 to 1.", "Our total: seasonal strength 0.93, the yearly "
             "pattern dominates.", "To measure how seasonal the data is, which led to taking the season from the whole network."),
        ]),
        ("The statistics terms: models and choosing them", [
            ("Lag", "A past value of the series. Lag 1 is last week, lag 52 is the same week last year.",
             "To predict this week, we might look at lag 1 (last week) and lag 52 (a year ago).",
             "We found lags 1-5 and 52 matter, and gave them to the machine-learning model as inputs."),
            ("ACF (autocorrelation function)", "A chart of how strongly the series is linked to itself k weeks earlier, for "
             "k = 1, 2, 3, ... Bars outside the shaded band are significant.", "A tall bar at lag 1 means this week is a lot like last week.",
             "It shows which past weeks matter and helps identify the model type."),
            ("PACF (partial autocorrelation)", "Like the ACF, but it keeps only the direct link with k weeks ago, removing "
             "the influence of the weeks in between.", "If this week links to two weeks ago only because both link to last "
             "week, the PACF at lag 2 is small.", "Together with the ACF, it tells AR and MA patterns apart."),
            ("AR model (autoregressive)", "This week is predicted from a weighted sum of the last few weeks' values. AR(p) uses p past weeks.",
             "AR(2): this week = 0.32 × last week + 0.15 × two weeks ago + a constant.", "Our total shows an AR pattern."),
            ("MA model (moving average)", "This week is predicted from the last few surprises (forecast errors). MA(q) uses q past surprises.",
             "MA(1): if last week was 10 units higher than expected, part of that surprise carries into this week.",
             "Our representative product was best described by MA(1)."),
            ("ARMA, ARIMA, SARIMA", "ARMA combines AR and MA. ARIMA is ARMA applied after differencing (the I is for "
             "\"integrated\"). SARIMA adds a seasonal part that links each week to the same week in earlier years. They are "
             "written (p, d, q) for the regular part and (P, D, Q)[52] for the seasonal part.",
             f"{ts['best']}: two past weeks (p = 2), no differencing, and one seasonal AR term at lag 52.",
             "They are the classical forecasting models. SARIMA describes our network total; ARIMA is one of our ten forecasting models."),
            ("Likelihood", "How probable the observed data is under a given model. A better-fitting model has a higher likelihood.",
             "A model that predicts the past weeks closely has a high likelihood.", "AIC and BIC are built from it."),
            ("AIC and BIC", "Scores for choosing between models. Both reward a good fit (likelihood) and punish complexity "
             "(number of parameters); BIC punishes more strictly. Lower is better; a difference below 2 means the models are about equal.",
             "Of 27 candidates, AIC chose SARIMA(2,0,0)(1,0,0)[52]; BIC chose a version with one parameter fewer.",
             "To pick the best model objectively, without over-complicating it (overfitting)."),
            ("Overfitting", "When a model learns the noise of the past instead of the real pattern, so it looks great on old "
             "data but predicts the future badly.", "Memorising last year's exact random spikes.",
             "AIC/BIC penalties and the fair backtest protect against it."),
            ("Kalman filter", "The algorithm that fits ARIMA-type models step by step through time. It can simply skip missing weeks.",
             "An out-of-stock week is skipped instead of being filled in.", "It lets ARIMA learn from clean demand with gaps."),
        ]),
        ("The statistics terms: checking the leftovers", [
            ("Residuals", "What a model could not explain: actual value minus the model's prediction.",
             "The model predicted 1,800, and 1,900 sold: the residual is +100.", "If residuals still show a pattern, the model missed something."),
            ("White noise", "A series with no pattern at all: no links between weeks and a constant spread. Pure randomness.",
             "The results of rolling a die every week.", "A good model leaves white-noise residuals: everything predictable has been captured."),
            ("Ljung-Box test", "Tests whether residuals are white noise by checking many lags at once. H0: no autocorrelation. "
             "p above 0.05 means white noise (good).", "After GARCH, our residuals gave p = 0.25 at lag 13: white noise.",
             "To confirm the models leave no pattern behind; also used on real forecast errors."),
            ("Volatility and volatility clustering", "Volatility is the size of the ups and downs. Clustering means calm periods "
             "and wild periods come in runs: after a wild week, more wild weeks tend to follow.", "Around Christmas, forecast "
             "errors are much bigger for several weeks, then calm down.", "Our residuals showed clustering, which led to ARCH and GARCH."),
            ("ARCH effect and the ARCH-LM test", "ARCH (Autoregressive Conditional Heteroskedasticity) means the size of the "
             "errors depends on the size of recent errors. The ARCH-LM test checks for it. H0: constant spread. p below 0.05 "
             "means an ARCH effect is present.", "Our SARIMA residuals gave p = 0.027: there is an ARCH effect.",
             "To detect volatility clustering, so we know whether a GARCH model is needed."),
            ("GARCH(1,1)", "Generalised ARCH: a model of how volatility changes over time. Today's volatility = a base level "
             "(ω) + a reaction to yesterday's shock (α) + a memory of yesterday's volatility (β). α + β is the persistence: how "
             "long wild periods last.", f"Our total: ω = {ts['garch_params']['omega']:.2f}, α = {ts['garch_params']['alpha[1]']:.2f}, "
             f"β = {ts['garch_params']['beta[1]']:.2f}.", "After GARCH, the residuals pass Ljung-Box and ARCH-LM: no pattern and "
             "no clustering left. It shows the leftover is pure noise once changing volatility is accounted for."),
            ("Standardised residuals", "Residuals divided by their GARCH volatility, so wild and calm periods are put on the same scale.",
             "A +200 error in a wild week and a +50 error in a calm week might both become +1.", "These are what we re-test after GARCH."),
        ]),
        ("The forecasting terms", [
            ("Forecast and horizon", "A forecast is a prediction of future demand. The horizon is how many weeks ahead.",
             "A 3-week horizon predicts this week, next week and the week after.", "Orders need 3 weeks: two to wait, one to cover."),
            ("Backtest (rolling origin)", "Testing a model by pretending to be in the past: at each of many past dates, it "
             "may only use data up to that date, and its forecasts are compared with what really happened.",
             "Standing on 1 October 2023, forecast October; then move to 8 October and repeat.", "The fair way to measure accuracy (30 dates)."),
            ("Leakage", "Accidentally letting a model see information from the future. It makes results look better than reality.",
             "Training on next month's sales and then \"predicting\" next month.", "The whole design avoids it: models only ever see the past."),
            ("MAE, WAPE, bias and Score", "MAE: the average miss in units. WAPE: the total miss divided by total sales. "
             "Bias: whether forecasts are too high (+) or too low (−) overall. Score = WAPE + |bias|, the official metric of "
             "the VN1 forecasting competition.", "Forecasts 3, 2, 1 against sales 2, 2, 2: MAE 0.67, WAPE 33%, bias 0%.",
             "To compare models on accuracy and on systematic over- or under-forecasting."),
            ("Feature", "An input given to a machine-learning model.", "Last week's sales, the time of year, the store.",
             "LightGBM uses about 30 features."),
            ("Decision tree and gradient boosting", "A decision tree makes a prediction through a series of yes/no questions. "
             "Gradient boosting builds many small trees one after another, each correcting the errors of the previous ones.",
             "Tree 1: \"Is it December?\". Tree 2 fixes what tree 1 got wrong for slow sellers.", "This is how LightGBM works."),
            ("Tweedie loss", "The measure of error that LightGBM tries to reduce while learning, designed for data that is "
             "never negative and has many exact zeros.", "Weekly sales of 0, 0, 1, 0, 3.", "It beat the usual alternatives on our data."),
            ("Ensemble", "Combining several models, here by averaging their forecasts.", "The average of three forecasts of 4, 6 and 5 is 5.",
             "Different models make different mistakes; averaging cancels some of them."),
            ("Calibration", "Whether stated probabilities come true as often as claimed.", "If \"80% sure\" is right 80% of the time, the model is calibrated.",
             "We checked it for the uncertainty ranges (they were right about 85% of the time)."),
        ]),
        ("The inventory terms", [
            ("Simulator", "Code that plays the shops week by week with the official rules and adds up the costs.",
             "Start 10 units, demand 12: sell 10, lose 2 sales (€2), end with 0.", "To score any ordering plan exactly like the competition did (it reproduced €4,334.0)."),
            ("Newsvendor problem and critical ratio", "The classic problem of choosing an order size before knowing demand, "
             "with a cost for too much and a cost for too little. The best order covers demand with a probability equal to the "
             "critical ratio = shortage cost ÷ (shortage cost + holding cost).", "With €1 and €0.20: 1 ÷ 1.2 = 0.833, so cover demand 83% of the time.",
             "It is the theory behind our ordering rule."),
            ("Service level", "The chance that an order covers the demand of the week it arrives.", "75% means covering demand in 3 out of 4 imagined futures.",
             "The main setting of the ordering rule; we tuned it to 75%."),
            ("Monte Carlo simulation", "Estimating something by simulating many random scenarios and looking at the results.",
             "Imagine 1,000 possible next weeks of demand and see how often the shelf runs out.", "It lets us account for the stock "
             "already on its way and the uncertainty of the next two weeks."),
            ("Order-up-to rule", "Order enough to bring stock on hand plus stock on its way up to a target, such as N weeks of forecast.",
             "Target 4 weeks of forecast = 40 units; 25 on hand and on the way, so order 15.", "It is the official benchmark's rule, kept for comparison."),
            ("Safety stock", "Extra stock kept above the expected demand to protect against surprises.", "Expecting 10, keeping 13.",
             "The service level decides how much safety stock the plan holds."),
            ("Tuning by replay", "Choosing a setting by replaying past weeks with each candidate setting and keeping the cheapest.",
             "Replaying 18 past weeks at service levels from 50% to 96%.", "It found 75% was cheaper than the textbook 83%."),
        ]),
    ]
    for gname, terms in groups:
        cards = [term_card(*t) for t in terms]
        st += [KeepTogether([P(gname, "h1"), cards[0]])] + cards[1:]   # never leave a heading alone at a page end

    # ================================================================ PART 3 models
    st += H(3, "Every model explained")
    st += [P("Ten models were built and compared on the same 30 past weeks (accuracy: error %, lower is better) and on the "
             "8 competition weeks (cost of the orders each model leads to, lower is better). The official benchmark is "
             f"{sc('seasonal_ma')} error and {eur('seasonal_ma')} on our replay of the 8 weeks.")]
    models = [
        ("1 · Last week (naive)",
         "Next week will be exactly like the last normal week.", "Sold 5 last week, so predict 5 for each coming week.",
         "Instant reaction to change; trivial to understand.", "Copies every random spike; ignores the season.",
         "It is the simplest possible guess: any serious method must beat it, or it is not worth its complexity.",
         f"Worst: error {sc('naive')}, orders cost {eur('naive')}."),
        ("2 · 13-week average",
         "Average of the last 13 normal weeks (about three months).", "Sales 4, 6, 5, ... over 13 weeks average 5, so predict 5.",
         "Smooths out random ups and downs; stable.", "Ignores the season; slow to react when demand really changes.",
         "A classic simple baseline that shows how much smoothing alone helps.", f"Error {sc('ma13')}, orders {eur('ma13')}."),
        ("3 · Same week last year (seasonal naive)",
         "Copy what sold in the same week one year earlier.", "Sold 20 in Christmas week last year, so predict 20 this Christmas.",
         "Captures the season automatically.", "Repeats last year's random noise; useless for new products.",
         "Tests whether the season alone is enough to forecast.", f"Error {sc('snaive')}, orders {eur('snaive')}: the season alone is not enough."),
        ("4 · Official benchmark (seasonal moving average)",
         "Removes the season from recent sales, averages the last 13 weeks, then puts the season back for each future week.",
         "Average 5 per normal week; a Christmas week has factor 2.4, so predict 12.",
         "Sensible, simple and seasonal; the competition's own method.", "Same weight for all 13 weeks; one season for all products.",
         "It is the official yardstick of the competition: beating it is the goal.", f"Error {sc('seasonal_ma')}, orders {eur('seasonal_ma')}."),
        ("5 · Exponential smoothing",
         "A weighted average that trusts recent weeks more: weights fade gradually into the past. The speed of fading is "
         "chosen per product. The season is removed first and added back afterwards.",
         "Weights 30% on last week, 21% on the week before, 15% on the one before, and so on.",
         "Reacts faster than a plain average while still smoothing noise.", "Cannot model trends or links to specific past weeks.",
         "A classic, reliable statistical method that is hard to beat in practice.", f"Error {sc('ses')}, orders {eur('ses')}."),
        ("6 · ARIMA (AIC-selected)",
         "The Box-Jenkins model of Part 2, one per product: ADF picks the differencing, AIC picks how many past weeks and "
         "past surprises to use. The season is removed first.",
         "AR(1): this week = 0.4 × last week + a constant, then the season is added back.",
         "Statistically principled; captures how weeks depend on each other; residuals checked.",
         "Each product learns alone from little data; struggles with many zeros.",
         "It is the classical time-series approach, and the subject of the whole Box-Jenkins analysis.",
         f"Error {sc('arima')}, orders {eur('arima')}: it beats the official method on cost."),
        ("7 · Croston (SBA)",
         "Built for slow sellers. It separately tracks how big a sale usually is and how many weeks usually pass between "
         "sales; the forecast is size ÷ gap, with a small correction (SBA) for a known upward bias.",
         "A sale of 4 roughly every 4 weeks: about 1 per week.", "Designed for intermittent demand with many zeros.",
         "Keeps forecasting the old rate even if the product stops selling.",
         "Most of our products are occasional or unpredictable sellers, the case Croston was invented for.",
         f"Error {sc('croston_sba')}, orders {eur('croston_sba')}."),
        ("8 · TSB",
         "Like Croston, but it tracks the probability of a sale each week (updated every week, sale or not) and the usual "
         "size of a sale; forecast = probability × size.", "Sells in 1 week out of 4, 4 units each time: 0.25 × 4 = 1 per week.",
         "Notices when a product dies: the probability fades towards zero.", "Still treats each product alone.",
         "A modern improvement on Croston for products that may stop selling.", f"Error {sc('tsb')}, orders {eur('tsb')}."),
        ("9 · LightGBM (machine learning)",
         "One gradient-boosted decision-tree model trained on all 599 products at once. Hundreds of small trees ask questions "
         "like \"Is it December? Were sales high last month? Was the shelf empty recently?\", each tree correcting the previous ones.",
         "It learns that products like this one, in December, after a strong month, usually sell about 30% more.",
         "Learns from all products together, so slow sellers borrow patterns from similar ones; uses many inputs.",
         "Harder to explain; needs careful testing to avoid overfitting.",
         "Global machine-learning models like this won the VN2 competition; we need to know if it beats the classics here.",
         f"<b>Best forecaster</b>: error {sc('lightgbm')}, orders {eur('lightgbm')}."),
        ("10 · Ensemble",
         "The average of three different models: LightGBM, exponential smoothing and the official benchmark.",
         "Forecasts of 6, 4 and 5 average to 5.", "Different models make different mistakes, and averaging cancels some of them.",
         "Not always the most accurate single forecast.", "Combining forecasts is one of the most reliable findings in forecasting research.",
         f"<b>Cheapest orders</b>: error {sc('ensemble')}, orders {eur('ensemble')}: our chosen plan."),
    ]
    st += [model_card(*m) for m in models]

    # ================================================================ PART 4 why compare
    st += H(4, "Why we compare ten models")
    st += [P("Choosing a single model and trusting it would be guesswork. Comparing ten models on exactly the same data, the "
             "same past weeks and the same rules turns the choice into evidence. Each model is in the comparison for a reason:"),
           table([["Group", "Models", "What the comparison tells us"],
                  ["Simple baselines", "Last week, 13-week average, same week last year",
                   "How much a method gains over the simplest possible guesses. A complex model that cannot beat them is useless."],
                  ["The official yardstick", "Official benchmark", "Whether we beat the competition's own method: the real target."],
                  ["Classical statistics", "Exponential smoothing, ARIMA", "Whether the textbook approaches (including the full "
                   "Box-Jenkins analysis) are good enough on their own."],
                  ["Slow-seller specialists", "Croston, TSB", "Whether methods built for intermittent demand help, since most products are slow sellers."],
                  ["Machine learning", "LightGBM", "Whether learning across all products together beats per-product methods."],
                  ["Combination", "Ensemble", "Whether averaging different models gives safer forecasts and cheaper orders."]],
                 [.22, .3, .48]),
           Spacer(1, 8),
           P("<b>What the comparison showed.</b> LightGBM made the most accurate forecasts over 30 past weeks. Every "
             "advanced method beat the simple baselines. ARIMA, exponential smoothing and the benchmark were close to each "
             "other. The ensemble led to the cheapest orders. Accuracy and cost are related but not identical: what finally "
             "matters to the shop is the cost of the orders, so the plan was chosen on cost, tested on past weeks."),
           P("<b>Why test on cost and not only on accuracy?</b> The ordering rule turns forecasts into decisions, and small "
             "differences in how a model over- or under-predicts can change the cost more than its average accuracy suggests. "
             "That is why every model was also played through the official simulator."),
           image(G.make_charts(Path(tempfile.mkdtemp()), summ, pd.read_csv(ART / "benchmark_weekly.csv").set_index("week"),
                               pd.read_csv(ART / "benchmark_weekly.csv").set_index("week"),
                               pd.read_csv(ART / "leaderboard.csv").assign(label=lambda d: d["model"].map(
                                   {"naive": "Last week", "ma13": "13-week average", "snaive": "Same week last year",
                                    "seasonal_ma": "Official benchmark", "ses": "Exponential smoothing", "arima": "ARIMA",
                                    "croston_sba": "Croston SBA", "tsb": "TSB", "lightgbm": "LightGBM", "ensemble": "Ensemble"})),
                               pd.read_csv(ART / "competition_scores.csv").assign(label=lambda d: d["model"].map(
                                   {"naive": "Last week", "ma13": "13-week average", "snaive": "Same week last year",
                                    "seasonal_ma": "Official benchmark", "ses": "Exponential smoothing", "arima": "ARIMA",
                                    "croston_sba": "Croston SBA", "tsb": "TSB", "lightgbm": "LightGBM", "ensemble": "Ensemble"})))["acc"]),
           P("Forecast error of the ten models on 30 past weeks (lower is better).", "caption")]

    # ================================================================ PART 5 recap
    st += H(5, "The short recap")
    recap = [
        ("The problem", "Order stock every Monday that arrives two weeks later. Empty shelf: €1 per lost sale. Leftover: €0.20 per item per week. Find the cheapest balance."),
        ("The data", "Real VN2 competition data: 67 shops, 297 products, 599 store-products, 165 weeks. The last 8 weeks are the final test."),
        ("The cleaning", "Zeros before a launch and in empty-shelf weeks are not real demand (100% of empty-shelf weeks show zero), so they are set aside."),
        ("The charts", "A strong Christmas season (2.4× a normal week), slow sellers (about 1 a week), not bell-shaped, many empty shelves."),
        ("The statistics", f"Stationary (ADF), very seasonal (0.93), best classical model {ts['best']}, leftovers are pure noise after GARCH."),
        ("The models", "Ten models, from \"last week\" to machine learning, compared fairly on 30 past weeks."),
        ("The best forecaster", f"LightGBM, error {sc('lightgbm')} against {sc('seasonal_ma')} for the official benchmark."),
        ("The ordering rule", "Imagine 1,000 futures and cover demand in 75% of them (tested on past weeks; the textbook says 83%)."),
        ("The result", f"€{summ['plan_score']:,.0f} against €{summ['benchmark_score']:,.0f}: 16% cheaper, with 45% less stock on the shelves."),
        ("The honest limits", "Built with hindsight; only 8 test weeks; empty-shelf weeks still understate demand; products are anonymous."),
    ]
    st += [table([["Topic", "In one or two sentences"]] + [[f"<b>{a}</b>", b] for a, b in recap], [.24, .76])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.0 * cm,
                            bottomMargin=1.8 * cm, title="OptiChain explained simply", author="OptiChain")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
