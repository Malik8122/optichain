"""Build the final project report (PDF) from saved results.

Tables are generated from the result files, so numbers always match the analysis.
Usage: python scripts/build_report.py [--author "Name"] [--course "Course"]
Output: reports/OptiChain_Final_Report.pdf
"""
import argparse
import json
import os
from datetime import date

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from optichain.config import PROJECT_ROOT, load_config

R = PROJECT_ROOT / "reports"
FIG = R / "figures"
NAVY = colors.HexColor("#1e3a8a")
BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0f172a")
INK2 = colors.HexColor("#475569")
RULE = colors.HexColor("#d9dee8")
ZEBRA = colors.HexColor("#f4f6fa")
HEAD = colors.HexColor("#e8f0fb")

# ---------------------------------------------------------------- fonts & styles
ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
                ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"), ("Sans-Oblique", "DejaVuSans-Oblique.ttf"),
                ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(ttf, f)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique", boldItalic="Sans-Bold")

S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.8, leading=14.2, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=9.8, leading=13.6, leftIndent=14, bulletIndent=4, textColor=INK, spaceAfter=2.5),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10, leading=13, textColor=INK2, spaceBefore=6, spaceAfter=3),
    "eq": ParagraphStyle("eq", fontName="Serif-Italic", fontSize=10, leading=15, alignment=TA_CENTER, textColor=INK, spaceBefore=3, spaceAfter=7),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.2, leading=11, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.6, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=7.6, leading=9.6, textColor=INK),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=8.8, leading=12, leftIndent=14, firstLineIndent=-14, textColor=INK, spaceAfter=4),
    "note": ParagraphStyle("note", fontName="Sans", fontSize=8.6, leading=12, textColor=INK2, borderPadding=7,
                           backColor=colors.HexColor("#f4f7fc"), borderColor=RULE, borderWidth=0.6, spaceBefore=4, spaceAfter=10),
    "abs": ParagraphStyle("abs", fontName="Serif", fontSize=9.6, leading=14, alignment=TA_JUSTIFY, textColor=INK, leftIndent=18, rightIndent=18),
    "toc1": ParagraphStyle("toc1", fontName="Sans-Bold", fontSize=10, leading=17, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Sans", fontSize=9, leading=13.5, leftIndent=14, textColor=INK2),
}


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.1 * cm,
                         bottomMargin=2 * cm, title="Demand Forecasting for Supply Chain Inventory Optimization",
                         author=kw.pop("author_meta", ""), subject="Walmart M5 · OptiChain final report", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("main", [frame], onPage=self._decorate)])

    def _decorate(self, c, d):
        c.saveState()
        c.setFont("Sans", 7.5)
        c.setFillColor(INK2)
        c.drawString(self.leftMargin, A4[1] - 1.3 * cm, "Demand Forecasting for Supply Chain Inventory Optimization · Walmart M5")
        c.drawRightString(A4[0] - self.rightMargin, 1.2 * cm, f"{d.page}")
        c.setStrokeColor(RULE)
        c.line(self.leftMargin, A4[1] - 1.45 * cm, A4[0] - self.rightMargin, A4[1] - 1.45 * cm)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            lvl = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = f"h{lvl}-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=lvl, closed=lvl > 0)
            self.notify("TOCEntry", (lvl, text, self.page, key))


# ---------------------------------------------------------------- helpers
story = []
FIGN = [0]
TABN = [0]


def p(text, style="body"):
    story.append(Paragraph(text, S[style]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))
    story.append(Spacer(1, 4))


def h1(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def h3(t):
    story.append(Paragraph(t, S["h3"]))


def eq(t):
    # reportlab collapses runs of spaces; keep visible gaps between equation parts
    story.append(Paragraph(t.replace("   ", "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;").replace("  ", "&nbsp;&nbsp;&nbsp;"), S["eq"]))


def note(t):
    story.append(Paragraph(t, S["note"]))


def fig(path, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = width_cm * cm
    FIGN[0] += 1
    story.append(KeepTogether([Image(str(path), width=width, height=width * h / w),
                               Paragraph(f"<b>Figure {FIGN[0]}.</b> {caption}", S["cap"])]))


def table(header, rows, widths, caption, bold_cells=None, align_right_from=1):
    TABN[0] += 1
    bold_cells = bold_cells or set()
    data = [[Paragraph(f"<b>{h}</b>", S["cellb"]) for h in header]]
    for i, r in enumerate(rows):
        data.append([Paragraph(str(v), S["cellb"] if (i, j) in bold_cells else S["cell"]) for j, v in enumerate(r)])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
             ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    story.append(KeepTogether([Paragraph(f"<b>Table {TABN[0]}.</b> {caption}", S["cap"]), t, Spacer(1, 10)]))


def f3(v):
    return f"{v:.3f}"


# ---------------------------------------------------------------- content
def build(author: str, course: str):
    cfg = load_config()
    items = cfg.series.items
    M = R / "metrics"
    summ = pd.read_csv(M / "forecast_summary.csv")
    ident = pd.read_csv(M / "identification.csv")
    sel_items = pd.read_csv(M / "selected_items.csv")
    rd = pd.read_csv(M / "residual_diagnostics.csv")
    gdec = pd.read_csv(M / "garch_decision.csv")
    pm = pd.read_csv(R / "inventory" / "policy_metrics.csv")
    sens = pd.read_csv(R / "inventory" / "sensitivity.csv")
    fmet = pd.read_csv(R / "final_test" / "metrics.csv")
    finv = pd.read_csv(R / "final_test" / "inventory.csv")
    prereg = json.loads((R / "final_test" / "preregistration.json").read_text())

    # ------------------------------------------------ cover
    story.append(Spacer(1, 3.2 * cm))
    story.append(Paragraph("FINAL PROJECT REPORT", ParagraphStyle("k", fontName="Sans-Bold", fontSize=9.5, textColor=BLUE, alignment=TA_CENTER, spaceAfter=14)))
    story.append(Paragraph("Demand Forecasting for<br/>Supply Chain Inventory Optimization", ParagraphStyle(
        "t", fontName="Sans-Bold", fontSize=20, leading=27, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12)))
    story.append(Paragraph("Using Time-Series Analysis and the Walmart M5 Dataset", ParagraphStyle(
        "st", fontName="Sans", fontSize=13, leading=17, textColor=INK2, alignment=TA_CENTER, spaceAfter=40)))
    story.append(Paragraph(
        "Statistical forecasting (ETS, ARIMA/SARIMA, Croston) with stationarity and residual diagnostics, "
        "a gated ARCH/GARCH evaluation, an inventory-replenishment simulation, and an interactive dashboard "
        "(OptiChain), with a pre-registered final test on unseen data.",
        ParagraphStyle("d", fontName="Serif-Italic", fontSize=10.5, leading=15, textColor=INK2, alignment=TA_CENTER, leftIndent=40, rightIndent=40, spaceAfter=70)))
    meta = [("Author", author or "&nbsp;"), ("Course", course or "&nbsp;"), ("Date", date.today().strftime("%B %Y")),
            ("Data", "Walmart M5 Forecasting – Accuracy (Kaggle), store CA_1"),
            ("Code", "OptiChain repository (Python 3.13, statsmodels, Streamlit)")]
    t = Table([[Paragraph(f"<b>{k}</b>", S["cellb"]), Paragraph(v, S["cell"])] for k, v in meta],
              colWidths=[3 * cm, 9 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(t)
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # ------------------------------------------------ abstract + TOC
    story.append(Paragraph("Abstract", ParagraphStyle("abs_h", parent=S["h2"], alignment=TA_CENTER)))
    bt_sel = fmet[fmet.selected]
    p(f"""Retailers must decide how much stock to order before demand is known. This project builds and evaluates an
    end-to-end pipeline from raw Walmart M5 sales to replenishment decisions, for eight products in store CA_1 chosen
    to cover the four Syntetos–Boylan demand classes. Each series is identified with ADF/KPSS tests, STL decomposition
    and ACF/PACF analysis, separately at each forecast origin. Seven methods (naive, seasonal naive, 28-day moving
    average, ETS, ARIMA, SARIMA and Croston-SBA) are compared by rolling-origin evaluation over four 28-day windows.
    ETS (mean RMSSE 0.773) and SARIMA (0.779) are jointly best and about 21% more accurate than seasonal naive, with
    well-calibrated prediction intervals. Residuals are unbiased with only small leftover autocorrelation. ARCH effects
    appear in three series, but GARCH(1,1) does not improve held-out interval scores materially and is not adopted.
    In a hypothetical inventory simulation, a forecast-driven reorder policy is cheaper than a simple trailing-average
    rule in 8 of 9 lead-time/service scenarios (3–11%). The main limitation identified is that in-sample error estimates
    misjudge out-of-sample uncertainty when safety stock is sized. A pre-registered test on four unseen weeks confirms
    the forecasting results: mean RMSSE {bt_sel['rmsse'].mean():.3f}, against {bt_sel['backtest_rmsse'].mean():.3f}
    in the backtest. All results are delivered through a six-step Streamlit dashboard.""", "abs")
    story.append(Spacer(1, 14))
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story.append(Paragraph("Contents", S["h3"]))
    story.append(toc)
    story.append(PageBreak())

    # ------------------------------------------------ 1 introduction
    h1("1. Introduction")
    p("""Replenishment decisions are made before customer demand is observed. Under-forecasting leads to stockouts
    and lost sales, while over-forecasting ties up capital and shelf space. Daily retail demand varies by weekday,
    season, event and price, and many products sell only intermittently. This project asks whether time-series
    forecasts improve demand estimates and, more importantly, whether feeding those forecasts into a transparent
    reorder policy improves simulated inventory outcomes compared with a simple rule of thumb.""")
    p("""The work follows the project blueprint: a reproducible data pipeline, exploratory analysis, stationarity and
    seasonality diagnostics, statistical model development and comparison, residual diagnostics with an optional
    ARCH/GARCH extension, an inventory simulation under explicit hypothetical assumptions, and a deployable
    dashboard. Methodological choices that the blueprint left open are recorded as decisions in the repository's
    <font face="Mono">DESIGN.md</font> and summarised in this report.""")
    h2("1.1 Research questions")
    bullets([
        "<b>RQ1.</b> Which forecasting method performs best on held-out dates for the selected product-store series?",
        "<b>RQ2.</b> How do weekday effects, events and price changes relate to demand in the chosen examples?",
        "<b>RQ3.</b> Do residual diagnostics support the adequacy of the selected mean models?",
        "<b>RQ4.</b> Do any series show conditional heteroscedasticity that warrants ARCH/GARCH?",
        "<b>RQ5.</b> Does a forecast-driven replenishment policy improve simulated service level or cost relative to a simple baseline under the same assumptions?",
        "<b>RQ6.</b> How sensitive are inventory outcomes to lead time, safety-stock targets and forecast errors?",
    ])
    note("<b>Important boundary.</b> M5 provides daily unit sales, a calendar and weekly prices. It does not contain "
         "on-hand inventory, lead times, purchase orders or observed stockouts. All inventory results are therefore a "
         "controlled simulation under clearly labelled hypothetical assumptions, not a reconstruction of Walmart's "
         "inventory system or an estimate of its savings.")

    # ------------------------------------------------ 2 data
    h1("2. Data and preparation")
    h2("2.1 Dataset")
    p("""The Walmart M5 Forecasting – Accuracy dataset (Makridakis et al., 2022a, 2022b) contains daily unit sales for
    3,049 products in each of 10 stores from 29 January 2011 (d_1) to 22 May 2016 (d_1941), together with a
    calendar (events and SNAP days) and weekly selling prices. The project uses store CA_1. Sales are read up to
    d_1913 (24 April 2016) for all development work. The final four weeks (d_1914–d_1941) were kept physically
    excluded by a <font face="Mono">max_day</font> cutoff until the single final test (Section 9).""")
    table(["File", "Content", "Use in this project"], [
        ["sales_train_evaluation.csv", "30,490 series × daily sales d_1–d_1941", "Single sales source; read only up to d_1913 until the final test"],
        ["calendar.csv", "Day ID → date, weekday, events, SNAP flags", "Date mapping; event/SNAP analysis; store-closure detection"],
        ["sell_prices.csv", "Weekly price per item-store", "Launch detection, price analysis, price-linked costs"],
    ], [4.3, 5.4, 6.9], "Source files and their role.")
    h2("2.2 Pipeline and data quality")
    bullets([
        "The wide sales table is filtered to CA_1 <i>before</i> reshaping to one row per item-day, giving 3,049 items × 1,913 days = 5,832,737 rows. It is then joined to the calendar and to prices, with row counts checked after each join.",
        "<b>Launch trimming.</b> 1,845 of 3,049 items were introduced after d_1, and 19.4% of all rows precede an item's first priced week. Those zeros mean \"not yet sold\", not \"no demand\", so each series starts at its first priced week. No sales occur before launch, which confirms the rule.",
        "Prices are forward-filled within an item only, never back-filled, to avoid using future prices. In practice no post-launch gaps occurred.",
        "<b>Store closures.</b> Every M5 store sells zero units on 25 December. For modelling and simulation, these days are replaced by the same weekday one week earlier (past data only). The raw series is kept for description.",
        "Validation checks (unique keys, non-negative sales, contiguous dates, d_1 = 2011-01-29) run inside the pipeline and as automated tests.",
    ])
    h2("2.3 Product selection")
    p("""Products were chosen by demand pattern rather than by hand. On the 730 days before the first forecast origin
    (d_1800), each item with at least two years of history was classified by its average inter-demand interval (ADI)
    and the squared coefficient of variation of non-zero sales (CV²), following Syntetos, Boylan and Croston (2005):""")
    eq("ADI = n / n<sub>+</sub>,    CV² = (s<sub>+</sub> / m<sub>+</sub>)²")
    eq("smooth: ADI &lt; 1.32, CV² &lt; 0.49  ·  erratic: ADI &lt; 1.32, CV² ≥ 0.49  ·  intermittent: ADI ≥ 1.32, CV² &lt; 0.49  ·  lumpy: ADI ≥ 1.32, CV² ≥ 0.49")
    p("Here n is the number of days, n<sub>+</sub> the days with a sale, and s<sub>+</sub>, m<sub>+</sub> the standard deviation and mean of the non-zero sales.")
    p("""Of 2,495 eligible items, 72% are intermittent and only 10% smooth, so zero-heavy demand is the norm in this
    store. Items with a run of more than 28 consecutive zero-sales days were excluded. At their sales rates such runs
    almost certainly reflect stockouts or delistings (censored demand) rather than true zero demand. This affected 13%
    of smooth items and about two-thirds of intermittent and lumpy items. Within each class, the item closest to a
    target volume quantile was chosen deterministically, preferring categories not yet represented.""")
    rows = [[r.item_id, r.cat_id.title(), r.demand_class, f"{r['mean']:.2f}", f"{r.zero_share:.1%}", f"{r.adi:.2f}", f"{r.cv2:.2f}"]
            for _, r in sel_items.iterrows()]
    table(["Item", "Category", "Class", "Mean/day", "Zero days", "ADI", "CV²"], rows, [3.3, 2.3, 2.4, 1.9, 1.9, 1.6, 1.6],
          "The eight study products (classification window d_1071–d_1800).")

    # ------------------------------------------------ 3 EDA
    h1("3. Exploratory analysis")
    fig(FIG / "eda" / "01_overview_small_multiples.png", 15.5, "Daily sales (grey) and 28-day rolling mean (blue), last two years before d_1800.")
    p("""The high-volume FOODS_3_586 (about 45 units/day) has a stable level with summer peaks. HOUSEHOLD_1_447
    trends upward. The HOBBIES items show large isolated spikes (up to ten times their mean), and the intermittent and
    lumpy items mostly sell 0–2 units a day. The coefficient of variation rises from 0.35 for FOODS_3_586 to 1.0–1.25
    for the spiky and intermittent items, and zero-sales shares range from 0.3% to 51%. MAPE is therefore unusable as
    a ranking metric.""")
    fig(FIG / "eda" / "02_weekday_index_heatmap.png", 13, "Weekday index (weekday mean ÷ overall mean). Seven of eight products peak at the weekend.")
    p("""<b>RQ2: calendar and price effects.</b> Weekend demand is 1.1–1.6 times an average day for seven of eight
    products. HOBBIES_1_189 has an almost flat weekly profile. Calendar events have small and inconsistent effects at
    item level (event-day to normal-day ratio 0.82–1.09). FOODS_3_586 sells about 13% more on Cultural events and 9%
    more on National holidays. SNAP days lift the FOODS items by 3–14% and have no effect on non-food items, as
    expected. Prices change rarely and in one direction over time, so price is confounded with time trends and no
    elasticity conclusion is drawn. These are associations, not causal effects. The practical consequences were a
    weekly seasonal period (s = 7) in all models, and univariate models without event regressors.""")

    # ------------------------------------------------ 4 identification
    h1("4. Stationarity, seasonality and identification")
    h2("4.1 Procedure")
    p("""Identification was run separately at each of the four forecast origins, using only the 730 days before that
    origin, so every modelling decision could have been made at that time. The Augmented Dickey–Fuller test
    (H<sub>0</sub>: unit root; Dickey and Fuller, 1979) and the KPSS test (H<sub>0</sub>: level-stationary;
    Kwiatkowski et al., 1992) were combined because their null hypotheses are opposite:""")
    table(["ADF", "KPSS", "Verdict", "Differencing"], [
        ["rejects", "does not reject", "stationary", "d = 0"],
        ["does not reject", "rejects", "unit root", "d = 1"],
        ["rejects", "rejects", "conflict (level shifts / long memory)", "d ∈ {0, 1}, settled out-of-sample"],
        ["does not reject", "does not reject", "inconclusive (low power)", "d ∈ {0, 1}, settled out-of-sample"],
    ], [3, 3, 5.5, 5.1], "Decision table combining ADF and KPSS (α = 0.05).")
    p("""A log1p transform was applied only where the 28-day rolling standard deviation rose with the rolling mean
    (correlation ≥ 0.6) and fewer than 10% of days were zero. Seasonal differencing was set by the STL seasonal
    strength F<sub>s</sub> (Cleveland et al., 1990; Hyndman and Athanasopoulos, 2021), with D = 1 if
    F<sub>s</sub> ≥ 0.64. Where the seasonal ACF decayed slowly (lags 7–28 all above 0.2), D = 1 was also kept as a
    candidate. AIC cannot compare models fitted to differently differenced data, so ambiguous d and D were resolved
    later by out-of-sample accuracy.""")
    h2("4.2 Results")
    i0 = ident[ident.origin_d == 1800].set_index("item_id").loc[items]
    rows = [[i, r["transform"], r["pre_d_verdict"], f"{r.seasonal_strength:.2f}", int(r.d), int(r.D), f"{r.lb_p_7:.3f}", "yes" if r.white_noise else "no"]
            for i, r in i0.iterrows()]
    table(["Item", "Transform", "ADF/KPSS verdict", "F<sub>s</sub>", "d", "D", "Ljung–Box p (7)", "White noise"], rows,
          [3.3, 2.0, 3.0, 1.3, 0.9, 0.9, 2.6, 2.0], "Identification at origin d_1800.")
    fig(FIG / "diagnostics" / "FOODS_3_586" / "acf_pacf.png", 15.5,
        "FOODS_3_586 after log1p and first differencing: ACF spikes at every multiple of 7 point to seasonal AR/MA terms.")
    bullets([
        "No product reaches the seasonal-strength threshold (F<sub>s</sub> ≤ 0.59) despite clear weekend peaks, because daily noise is large relative to the weekly pattern. The weekly structure is therefore modelled with seasonal AR/MA terms rather than seasonal differencing.",
        "FOODS_3_586 is the only clear unit-root case (log1p, d = 1). Its slowly decaying seasonal ACF also keeps D = 1 as a candidate.",
        "Four products show ADF/KPSS conflicts, consistent with level shifts. Two of them flip their d decision at one origin each, which justifies carrying both options forward.",
        "FOODS_3_242 and HOUSEHOLD_1_046 are statistically white noise (Ljung–Box p ≈ 0.3), so little can be gained beyond a well-estimated mean.",
    ])

    # ------------------------------------------------ 5 models
    h1("5. Forecasting models and evaluation design")
    h2("5.1 Models")
    p("<b>Benchmarks.</b> Naive (last value), seasonal naive (same weekday last week) and a 28-day moving average.")
    p("""<b>Exponential smoothing (ETS)</b> with additive errors (Hyndman et al., 2002). The form is chosen by AICc
    from {no trend, damped trend} × {no season, weekly season}. The most frequently selected form, ETS(A,N,A), is:""")
    eq("<font face='Sans'>ℓ</font><sub>t</sub> = α(y<sub>t</sub> − s<sub>t−7</sub>) + (1 − α)<font face='Sans'>ℓ</font><sub>t−1</sub>,   s<sub>t</sub> = γ(y<sub>t</sub> − <font face='Sans'>ℓ</font><sub>t−1</sub>) + (1 − γ)s<sub>t−7</sub>,   ŷ<sub>t+h|t</sub> = <font face='Sans'>ℓ</font><sub>t</sub> + s<sub>t+h−7k</sub>")
    p("<b>ARIMA and SARIMA</b>, fitted on the transformed scale chosen in Section 4:")
    eq("φ(B) Φ(B<super>7</super>) (1 − B)<super>d</super> (1 − B<super>7</super>)<super>D</super> y<sub>t</sub> = c + θ(B) Θ(B<super>7</super>) ε<sub>t</sub>")
    p("""Orders p, q ∈ {0, 1, 2} and P, Q ∈ {0, 1} are searched by AIC within each allowed (d, D), and non-converged
    fits are discarded. When several (d, D) combinations are allowed, the AIC-best order of each is scored on the last
    28 days of the training window and the winner is refit on the full window. The real test window is never used for
    any choice. Log-scale forecasts are back-transformed with expm1, which gives approximately the median.""")
    p("<b>Croston with the Syntetos–Boylan correction (SBA)</b> for intermittent demand (Croston, 1972; Syntetos and Boylan, 2005):")
    eq("ŷ = (1 − α/2) · ẑ<sub>t</sub> / p̂<sub>t</sub>")
    p("Here ẑ and p̂ are smoothed non-zero demand sizes and inter-demand intervals, and α ∈ {0.05, 0.1, 0.2, 0.3} is chosen by in-sample MSE.")
    h2("5.2 Evaluation design")
    p("""Rolling-origin evaluation uses origins d_1800, d_1828, d_1856 and d_1885, each followed by a 28-day test
    window (3 January – 24 April 2016). Models are fitted on the 730 days before each origin. The main metric is the
    M5 root mean squared scaled error (Hyndman and Koehler, 2006):""")
    eq("RMSSE = √[ (1/h) Σ<sub>t</sub> (y<sub>t</sub> − ŷ<sub>t</sub>)² / ( (1/(n−1)) Σ<sub>t=2..n</sub> (y<sub>t</sub> − y<sub>t−1</sub>)² ) ]")
    p("""The denominator is computed on the training series from its first non-zero sale. MAE, RMSE, bias, sMAPE (with
    0/0 = 0) and 80%/95% interval coverage are also reported. A parallel, cached backtest ran 2,342 candidate fits;
    each cache key hashes only training data up to the origin, so data after the origin cannot affect a cached fit.""")

    # ------------------------------------------------ 6 results forecasting
    h1("6. Forecasting results")
    order = ["naive", "snaive", "ma28", "ets", "arima", "sarima", "croston"]
    piv = summ.pivot(index="item_id", columns="model", values="rmsse").loc[items, order]
    rows, bold = [], set()
    for i, (it, r) in enumerate(piv.iterrows()):
        rows.append([it] + [f3(v) for v in r.values])
        bold.add((i, 1 + list(r.values).index(r.min())))
    mean = piv.mean()
    rows.append(["<b>Mean</b>"] + [f"<b>{v:.3f}</b>" for v in mean.values])
    table(["Item", "Naive", "S-naive", "MA28", "ETS", "ARIMA", "SARIMA", "Croston"], rows,
          [3.4, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], "Mean RMSSE over the four backtest folds (best per item in bold).", bold_cells=bold)
    fig(FIG / "final" / "model_comparison.png", 14.5, "Mean RMSSE by method in the backtest and in the pre-registered final test.")
    cal = summ.groupby("model")[["bias", "cov80", "cov95"]].mean().loc[order]
    p(f"""<b>RQ1.</b> ETS is the most accurate method overall (mean RMSSE {mean['ets']:.3f}, best on 6 of 8 products),
    with SARIMA a close second ({mean['sarima']:.3f}, best on HOUSEHOLD_1_469 and FOODS_3_242). Both are about 21%
    more accurate than seasonal naive ({mean['snaive']:.3f}), and every fitted model beats both naive benchmarks on
    average. The two leaders differ by under 1%, which is within fold-to-fold variation over four windows, so they
    are treated as jointly best. Gains are largest for the steady, strongly weekly FOODS_3_586 (0.664 → 0.580) and
    smallest for the white-noise items, as identification predicted. AIC was used only to choose orders within a
    family: across families (different scales and likelihoods) AIC values are not comparable, so families were
    compared on held-out data only.""")
    p(f"""Prediction intervals are well calibrated for the fitted models. ETS 80% intervals contain the actual value
    {cal.loc['ets','cov80']:.0%} of the time and 95% intervals {cal.loc['ets','cov95']:.0%}; for SARIMA the figures
    are {cal.loc['sarima','cov80']:.0%} and {cal.loc['sarima','cov95']:.0%}. The naive intervals over-cover (99%).
    Mean bias is at most ±0.25 units/day for every fitted model.""")
    fig(FIG / "final" / "forecast_example.png", 15, "FOODS_3_586: ETS and seasonal-naive forecasts from d_1885 against held-out actuals.")

    # ------------------------------------------------ 7 residuals + GARCH
    h1("7. Residual diagnostics and the ARCH/GARCH decision")
    h2("7.1 Residual adequacy")
    p("""For each product, the ETS and SARIMA models chosen by the backtest were refit at every origin (64 fits).
    Their one-step residuals were tested for bias (t-test), autocorrelation (Ljung–Box at lags 7, 14 and 21, with
    degrees of freedom reduced by the number of fitted parameters; Ljung and Box, 1978), normality (Jarque–Bera) and
    ARCH effects (Engle's ARCH-LM test at lag 7; Engle, 1982).""")
    s85 = rd[rd.selected & (rd.origin_d == 1885)].set_index("item_id").loc[items]
    rows = [[i, r.spec, f"{r.resid_mean:+.3f}", f"{r.lb_p_7:.3f}", f"{r.max_abs_acf:.3f}", f"{r.excess_kurtosis:.1f}", f"{r.arch_lm_p:.3f}"]
            for i, r in s85.iterrows()]
    table(["Item", "Selected model", "Mean resid.", "LB p (7)", "max |ACF|", "Excess kurt.", "ARCH-LM p"], rows,
          [3.2, 4.1, 1.9, 1.7, 1.8, 1.9, 1.9], "Residual diagnostics of the selected models fitted at d_1885.")
    sel = rd[rd.selected]
    p(f"""<b>RQ3.</b> Residuals are unbiased in {int(sel.unbiased.sum())} of {len(sel)} selected fits. Ljung–Box rejects
    at lag 7 in about half of them, but the largest residual autocorrelation in any fit is {sel.max_abs_acf.max():.3f},
    against a 95% band of ±0.074 at n ≈ 700. With this many observations even very small correlations are
    significant, and a correlation of 0.1 explains about 1% of residual variance. The leftover structure is recorded
    as a quantified limitation rather than removed by adding parameters, since extra SARIMA flexibility did not beat
    ETS out-of-sample. FOODS_3_586 has near-normal residuals. The spiky and lumpy products have heavy right tails
    (excess kurtosis 4–9), which matters for normal-theory safety stock.""")
    fig(FIG / "residuals" / "FOODS_3_586_ets.png", 15.5, "Residual diagnostics for FOODS_3_586 · ETS(A,N,A) fitted at d_1885.")
    h2("7.2 ARCH/GARCH")
    p("""<b>RQ4.</b> A GARCH(1,1) model (Bollerslev, 1986) was fitted only where a gate fixed in advance passed:
    ARCH-LM p &lt; 0.05 and at most 30% zero days. GARCH never alters the point forecast. It rescales the mean model's
    interval half-widths by the predicted variance relative to its long-run level, and was kept only if it improved
    the held-out interval score (Gneiting and Raftery, 2007):""")
    eq("σ<super>2</super><sub>t</sub> = ω + α ε<super>2</super><sub>t−1</sub> + β σ<super>2</super><sub>t−1</sub>;   IS<sub>a</sub> = (u − l) + (2/a)(l − y)·1{y &lt; l} + (2/a)(y − u)·1{y &gt; u}")
    g = gdec[gdec.origins_garch_eligible > 0]
    rows = [[r.item_id, r.model, int(r.origins_garch_eligible), f"{r.score_gain_pct:+.1f}%" if pd.notna(r.score_gain_pct) else "–",
             f"{r.alpha_significant_share:.0%}" if pd.notna(r.alpha_significant_share) else "–",
             "yes" if r.use_garch else "no", "yes" if r.recommend_garch else "no"] for _, r in g.iterrows()]
    table(["Item", "Model", "Eligible origins", "Interval-score gain", "α significant", "Pre-set rule", "Strict rule"], rows,
          [3.2, 1.6, 2.2, 2.6, 2.1, 2.4, 2.4], "GARCH(1,1) where the gate passed.")
    p("""ARCH effects appear mainly in the three highest-volume, near-continuous series, and not in the spiky or
    intermittent ones. Where GARCH was fitted, α was small (0.01–0.05) and mostly insignificant, and persistence was
    0.89–1.00. Held-out interval-score gains were at most 2.4% for ETS. Every SARIMA variant got worse, by up to 79%.
    The pre-specified rule technically favours GARCH for two ETS series. A stricter criterion, added after inspecting
    the fits and reported alongside (α significant at most origins and a gain of at least 5%), selects none.
    <b>GARCH is therefore not adopted</b>, a negative result the blueprint explicitly treats as valid. Simulated GARCH
    data is recovered correctly by the same code (an automated test), so the negative result reflects the sales data,
    not the tooling.""")

    # ------------------------------------------------ 8 inventory
    h1("8. Inventory simulation")
    h2("8.1 Design")
    sa = cfg.simulation_assumptions
    p(f"""Each product is simulated over d_1801–d_1913 with daily review, lost sales (no backorders) and a deterministic
    lead time L. An order placed at the end of day t arrives at the start of day t + L. Every policy shares the same
    opening stock, the same real sales and the same hypothetical costs: holding {sa.holding_rate_annual:.0%} of price per
    year, missed-sale penalty {sa.shortage_penalty_rate:.0%} of price, ${sa.order_cost:.0f} per order. The first
    {cfg.simulation.warmup_days} days are excluded from the metrics. With inventory position IP = on-hand + on-order,
    an order up to S<sub>t</sub> is placed whenever IP ≤ ROP<sub>t</sub> (Silver et al., 2017):""")
    eq("P0:  ROP = μ·L + z·σ<sub>d</sub>·√L,   S = μ·(L + c) + SS   (μ, σ<sub>d</sub>: trailing 28 days)")
    eq("P1:  ROP<sub>t</sub> = Σ<sub>k=1..L</sub> ŷ<sub>t+k</sub> + z·σ<sub>e</sub>·√L,   S<sub>t</sub> = Σ<sub>k=1..L+c</sub> ŷ<sub>t+k</sub> + SS")
    eq("P2:  SS = Q<sub>CSL</sub>( Σ<sub>k=1..L</sub> e<sub>t−k</sub> )   (empirical quantile of L-day summed one-step errors)")
    p(f"""Defaults: L = {sa.lead_time_days} days, c = {sa.cover_days} days of cover, target cycle service level
    {sa.target_csl:.0%}. P1 and P2 use the model chosen at each origin from folds already completed by then: seasonal
    naive at the first origin, then mostly ETS or SARIMA. A perfect-hindsight policy, which uses actual future demand
    and no safety stock, bounds what forecasting could achieve in service and stock. It is not a cost optimum, because
    the (s, S) rule is not cost-optimised.""")
    h2("8.2 Results")
    main = ["P0 baseline", "P1 forecast", "P2 empirical", "Perfect foresight"]
    tot = pm[pm.policy.isin(main)].groupby("policy").agg(c=("total_cost", "sum"), l=("lost_units", "sum"), s=("avg_on_hand", "sum"),
                                                        sold=("sold_units", "sum"), d=("demand_units", "sum")).loc[main]
    names = {"P0 baseline": "P0 simple rule", "P1 forecast": "P1 forecast-driven", "P2 empirical": "P2 empirical SS", "Perfect foresight": "Perfect hindsight"}
    rows = [[names[k], f"${r.c:,.0f}", f"{r.c / tot.loc['P0 baseline', 'c'] - 1:+.0%}", f"{r.sold / r.d:.2%}", f"{r.l:.0f}", f"{r.s:.0f}"] for k, r in tot.iterrows()]
    table(["Policy", "Total cost", "vs P0", "Fill rate", "Lost units", "Σ avg stock"], rows, [4.2, 2.3, 1.8, 2.2, 2.2, 2.6],
          f"Totals over 8 products, default scenario (L = {sa.lead_time_days}, CSL = {sa.target_csl:.0%}); costs hypothetical.")
    p("""In the default scenario P1 costs 11% less than the simple rule, holds 4% less stock and misses slightly fewer
    sales. It is cheaper on 7 of 8 products. P2 is essentially tied with P1 overall. Perfect hindsight shows that 29%
    less stock with zero lost sales would be possible with exact forecasts.""")
    fig(FIG / "final" / "frontier.png", 16, "Service vs stock trade-off: each point is one target service level (80–99%). Up and to the left is better.")
    p("""<b>RQ5.</b> The trade-off curves show where forecasting helps. The forecast policies are more stock-efficient
    for FOODS_3_586, HOUSEHOLD_1_447 and FOODS_3_242, mostly so for HOUSEHOLD_1_046, and reach far higher fill rates
    for HOUSEHOLD_1_469. The simple rule is more efficient for FOODS_3_203 and the spiky HOBBIES_1_189. Fill rate is
    not always monotonic in the target, a sign of simulation noise over a short window with few stockout events.""")
    fig(FIG / "final" / "sensitivity.png", 14.5, "Total-cost change relative to P0 across lead times and service targets (blue = cheaper).")
    agg = sens.groupby(["lead_time", "target_csl", "policy"])["total_cost"].sum().unstack("policy")
    ch = agg["P1 forecast"] / agg["P0 baseline"] - 1
    p(f"""<b>RQ6.</b> P1 is cheaper than P0 in {int((ch < 0).sum())} of {len(ch)} scenarios, by {-ch.max():.0%} to
    {-ch.min():.0%}. It is more expensive only at L = 2 with a 98% target. The default scenario is the most favourable,
    so the result should be quoted as a range, not as the headline figure. P2's effect is less consistent (−12% to
    +3.5%). Re-running P1 with each model fixed shows that forecast quality matters but maps only loosely to cost.
    Naive forecasts cost $880 (worse than P0), ETS gives the lowest cost ($593), and SARIMA costs $644 despite
    near-best accuracy. The adaptive P1 ($600) lands within 1% of the best fixed model.""")
    p("""<b>Main inventory finding.</b> Buffer accuracy is limited by the <i>error estimate</i>, not by its
    distributional form. The empirical 4-day buffers of P2 are only 0–15% larger than normal-theory ones. In contrast,
    out-of-sample forecast errors are 18–26% larger than the in-sample one-step σ for the HOBBIES products, FOODS_3_242
    and HOUSEHOLD_1_447, so their buffers come out too small. They are smaller for FOODS_3_586 (0.87×) and
    HOUSEHOLD_1_469 (0.59×), so those items are over-protected. Each decision relies on a forecast issued up to four
    weeks earlier, so buffers should be sized from h-step out-of-sample errors, or forecasts should be updated daily.""")

    # ------------------------------------------------ 9 final test
    h1("9. Pre-registered final test")
    p(f"""Before any sales after d_1913 were loaded, the model for each product (lowest backtest RMSSE) and the
    inventory settings were written to a pre-registration file ({prereg['created_utc']} UTC). The final window
    d_1914–d_1941 ({prereg['test_window']['dates'][0]} to {prereg['test_window']['dates'][1]}) was then loaded once.
    A check confirmed that history up to d_1913 is identical to the frozen dataset, and all models were fitted at
    d_1913 and scored.""")
    fs = fmet[fmet.selected].set_index("item_id").loc[items]
    rows = [[i, r.model.upper(), f3(r.backtest_rmsse), f3(r.rmsse), f"{int(r.rank_in_test) if r.rank_in_test == int(r.rank_in_test) else r.rank_in_test} / 7",
             f"{r.cov80:.0%}", f"{r.cov95:.0%}"] for i, r in fs.iterrows()]
    rows.append(["<b>Mean</b>", "", f"<b>{fs.backtest_rmsse.mean():.3f}</b>", f"<b>{fs.rmsse.mean():.3f}</b>", "", f"{fs.cov80.mean():.0%}", f"{fs.cov95.mean():.0%}"])
    table(["Item", "Model", "Backtest RMSSE", "Test RMSSE", "Rank in test", "Cov. 80%", "Cov. 95%"], rows,
          [3.3, 1.8, 2.5, 2.2, 2.2, 1.8, 1.8], "Pre-registered models on the unseen four weeks.")
    fig(FIG / "final" / "final_test.png", 15, "Backtest vs final-test RMSSE for each pre-registered model.")
    ft = fmet.groupby("model")["rmsse"].mean()
    p(f"""<b>Forecasting generalises.</b> The pre-registered models score a mean RMSSE of {fs.rmsse.mean():.3f}
    against {fs.backtest_rmsse.mean():.3f} in the backtest. SARIMA ({ft['sarima']:.3f}) and ETS ({ft['ets']:.3f})
    remain jointly best, and every fitted model again beats seasonal naive ({ft['snaive']:.3f}). Individual products
    move in both directions. HOUSEHOLD_1_469 deteriorates (0.41 → 0.97) while HOBBIES_1_030 and FOODS_3_242 improve
    markedly, a reminder that four folds give noisy per-product estimates. The 80% intervals remain close to nominal on
    average.""")
    fi = finv.groupby("policy").agg(c=("total_cost", "sum"), l=("lost_units", "sum"), s=("avg_on_hand", "sum"))
    fx = finv[finv.item_id != "HOUSEHOLD_1_469"].groupby("policy").agg(c=("total_cost", "sum"), l=("lost_units", "sum"), s=("avg_on_hand", "sum"))
    rows = [[names[k], f"${fi.loc[k, 'c']:.0f}", f"{fi.loc[k, 'l']:.0f}", f"{fi.loc[k, 's']:.0f}", f"${fx.loc[k, 'c']:.0f}", f"{fx.loc[k, 's']:.0f}"] for k in main]
    table(["Policy", "Cost (all 8)", "Lost units", "Σ avg stock", "Cost excl. 469", "Σ stock excl. 469"], rows,
          [4.2, 2.3, 2.1, 2.3, 2.7, 2.9], "Inventory over the final four weeks (L = 4, CSL = 95%, no warm-up; hypothetical costs).")
    p(f"""Over the final four weeks P1 costs ${fi.loc['P1 forecast', 'c']:.0f} against ${fi.loc['P0 baseline', 'c']:.0f}
    for the simple rule, with {fi.loc['P1 forecast', 'l']:.0f} lost units against {fi.loc['P0 baseline', 'l']:.0f}.
    The difference comes almost entirely from HOUSEHOLD_1_469, an $18 item where the simple rule lost six units.
    Excluding it, costs are equal (${fx.loc['P1 forecast', 'c']:.0f} vs ${fx.loc['P0 baseline', 'c']:.0f}), but P1
    holds about 10% less stock with no more lost sales. Over 28 days ordering fees dominate total cost, so one extra
    order (as for FOODS_3_586) can outweigh holding savings. The window is too short to measure service differences
    precisely.""")

    # ------------------------------------------------ 10 dashboard
    h1("10. Application: the OptiChain dashboard")
    p("""Results are delivered through a Streamlit dashboard designed for non-technical users. It follows a six-step
    flow: Overview → Understand demand → Predict → Model check → Plan stock → Report. Methods and policies carry plain
    names ("Same day last week", "Simple rule", "Forecast-driven plan"), and technical detail sits in analyst
    sections. The app reads precomputed artefacts (about 12 MB) and runs the inventory simulator live, so scenario
    sliders respond instantly. Every page has an automated rendering test, and the app is prepared for Streamlit
    Community Cloud.""")
    fig(FIG / "final" / "dashboard_plan.png", 15.5, "Plan stock page: scenario inputs (hypothetical) and the reorder recommendation.")
    fig(FIG / "final" / "dashboard_check.png", 15.5, "Model check page: plain-language verdicts on bias, leftover pattern and volatility.")

    # ------------------------------------------------ 11 discussion
    h1("11. Discussion and limitations")
    bullets([
        "<b>Hypothetical inventory.</b> Lead times, costs and opening stock are assumptions, and outcomes depend on them (Section 8). Results are a controlled experiment, not Walmart savings.",
        "<b>Censored demand.</b> Recorded sales are capped by real stockouts. Long zero runs were screened out at selection, but shorter censoring remains and observed sales are treated as demand.",
        "<b>Small sample.</b> Eight products from one US store, four backtest folds and a 106-day replay. Per-product rankings and inventory differences of a few percent are within noise.",
        "<b>Leftover autocorrelation.</b> Statistically detectable in about half the residual sets, though practically small (|ACF| ≤ 0.1).",
        "<b>Error estimates for safety stock.</b> Buffers use in-sample one-step errors, while decisions use multi-step forecasts. This was the main cause of under- and over-protection.",
        "<b>Monthly forecast refresh.</b> Forecasts are issued once per fold. Daily state updating would shorten effective horizons.",
        "<b>Back-transformation.</b> expm1 of a log-scale mean gives roughly the median, which runs slightly low for right-skewed demand (affects the two log1p series).",
        "<b>Associations only.</b> Event, SNAP and price relationships are descriptive and confounded with time.",
    ])
    h1("12. Conclusions and future work")
    p("""A carefully identified statistical pipeline produces accurate, well-calibrated daily forecasts for
    heterogeneous retail demand. ETS and SARIMA are jointly best and roughly 21% better than seasonal naive, and the
    result held on a pre-registered unseen month. Diagnostics confirm adequate mean models, and the evidence does not
    support a volatility model. Turning forecasts into reorder decisions lowers simulated cost in most scenarios. The
    benefit depends on the product and the assumptions, and it is limited less by forecast accuracy than by how
    forecast uncertainty is converted into safety stock.""")
    bullets([
        "Size safety stock from h-step out-of-sample errors collected on completed folds, and re-forecast daily from updated model state.",
        "Set the service target from the shortage-to-holding cost ratio (newsvendor) rather than fixing it, and optimise (s, S) for cost.",
        "Add a global feature-based model (e.g. LightGBM with lag, calendar and price features) as a benchmark across many more series.",
        "Model censored demand explicitly and scale to a department or the full store.",
    ])

    # ------------------------------------------------ references
    h1("References")
    for ref in [
        "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. <i>Journal of Econometrics</i>, 31(3), 307–327.",
        "Cleveland, R. B., Cleveland, W. S., McRae, J. E., &amp; Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. <i>Journal of Official Statistics</i>, 6(1), 3–73.",
        "Croston, J. D. (1972). Forecasting and stock control for intermittent demands. <i>Operational Research Quarterly</i>, 23(3), 289–303.",
        "Dickey, D. A., &amp; Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. <i>Journal of the American Statistical Association</i>, 74(366), 427–431.",
        "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. <i>Econometrica</i>, 50(4), 987–1007.",
        "Gneiting, T., &amp; Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. <i>Journal of the American Statistical Association</i>, 102(477), 359–378.",
        "Hyndman, R. J., &amp; Athanasopoulos, G. (2021). <i>Forecasting: Principles and Practice</i> (3rd ed.). OTexts.",
        "Hyndman, R. J., &amp; Koehler, A. B. (2006). Another look at measures of forecast accuracy. <i>International Journal of Forecasting</i>, 22(4), 679–688.",
        "Hyndman, R. J., Koehler, A. B., Snyder, R. D., &amp; Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. <i>International Journal of Forecasting</i>, 18(3), 439–454.",
        "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., &amp; Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. <i>Journal of Econometrics</i>, 54(1–3), 159–178.",
        "Ljung, G. M., &amp; Box, G. E. P. (1978). On a measure of lack of fit in time series models. <i>Biometrika</i>, 65(2), 297–303.",
        "Makridakis, S., Spiliotis, E., &amp; Assimakopoulos, V. (2022a). M5 accuracy competition: Results, findings, and conclusions. <i>International Journal of Forecasting</i>, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013",
        "Makridakis, S., Spiliotis, E., &amp; Assimakopoulos, V. (2022b). The M5 competition: Background, organization, and implementation. <i>International Journal of Forecasting</i>, 38(4), 1325–1336. https://doi.org/10.1016/j.ijforecast.2021.07.007",
        "Seabold, S., &amp; Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. <i>Proceedings of the 9th Python in Science Conference</i>, 92–96.",
        "Silver, E. A., Pyke, D. F., &amp; Thomas, D. J. (2017). <i>Inventory and Production Management in Supply Chains</i> (4th ed.). CRC Press.",
        "Syntetos, A. A., &amp; Boylan, J. E. (2005). The accuracy of intermittent demand estimates. <i>International Journal of Forecasting</i>, 21(2), 303–314.",
        "Syntetos, A. A., Boylan, J. E., &amp; Croston, J. D. (2005). On the categorization of demand patterns. <i>Journal of the Operational Research Society</i>, 56(5), 495–503.",
        "Walmart / University of Nicosia (2020). M5 Forecasting – Accuracy [dataset]. Kaggle. https://www.kaggle.com/competitions/m5-forecasting-accuracy",
    ]:
        story.append(Paragraph(ref, S["ref"]))

    h1("Appendix A. Reproducibility")
    p("""All results can be regenerated from the raw M5 files with the scripts below, run in order from the project
    root (Python 3.13; pinned versions in <font face="Mono">requirements.txt</font>). The test suite (97 tests) runs
    on synthetic data and does not need the Kaggle files.""")
    table(["Step", "Command", "Produces"], [
        ["1", "python scripts/build_dataset.py", "CA_1 long table, data audit"],
        ["2", "python scripts/select_products.py", "8 products frozen in config.yaml"],
        ["3", "python scripts/run_eda.py", "EDA figures and tables"],
        ["4", "python scripts/run_diagnostics.py", "models/identification.json (per origin)"],
        ["5", "python scripts/run_backtest.py", "Forecasts, metrics, AIC tables (parallel, cached)"],
        ["6", "python scripts/run_residuals.py", "Residual diagnostics, GARCH evaluation"],
        ["7", "python scripts/run_simulation.py", "Inventory policies, sensitivity, trade-off curves"],
        ["8", "python scripts/run_final_test.py", "Pre-registered final test (run once)"],
        ["9", "python scripts/make_report_figures.py; python scripts/build_report.py", "This report"],
        ["10", "streamlit run app/Home.py", "Dashboard"],
    ], [1.2, 8.6, 6.8], "Pipeline commands.")
    p("""Analysis notebooks 01–05 document each stage with interpretation. <font face="Mono">DESIGN.md</font> records
    every methodological decision, including those added during the project and the reason for each.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="")
    ap.add_argument("--course", default="")
    args = ap.parse_args()
    build(args.author, args.course)
    out = R / "OptiChain_Final_Report.pdf"
    doc = Doc(out, author_meta=args.author)
    doc.multiBuild(story)
    print("wrote", out)


if __name__ == "__main__":
    main()
