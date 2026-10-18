// Build the presentation deck (PowerPoint) from saved results and figures.
// Requires: npm install pptxgenjs   (run: node scripts/build_deck.js  from the project root)
// Output: reports/OptiChain_Presentation.pptx
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const R = (...p) => path.join(ROOT, "reports", ...p);
const OUT = R("OptiChain_Presentation.pptx");
const DATA = JSON.parse(fs.readFileSync(R("figures", "final", "deck_data.json"), "utf8"));
const AUTHOR = process.argv[2] || "";

const THEME = {
  name: "OptiChain",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "1E2761", lt2: "EEF2F8",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "CADCFC",
    accent5: "64748B", accent6: "F2B134", hlink: "2A78D6", folHlink: "6D28D9",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Demand Forecasting for Supply Chain Inventory Optimization";
pres.subject = "Walmart M5 · OptiChain";
pres.author = AUTHOR || "OptiChain";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: W - 2 * M, h: 1.6, fontSize: 40, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.95, w: W - 2 * M, h: 3.0, fontSize: 18,
      color: C.accent4, fontFace: THEME.bodyFontFace, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.38, w: W - 2 * M, h: 0.85, fontSize: 32, bold: true,
      color: C.text2, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "OptiChain · Walmart M5 demand forecasting & inventory", options: { x: M, y: H - 0.42, w: 7, h: 0.3,
      fontSize: 10, color: C.accent5, fontFace: THEME.bodyFontFace, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
});

// ---------------------------------------------------------------- helpers
const shadow = () => ({ type: "outer", color: "0F172A", opacity: 0.12, blur: 8, offset: 2, angle: 90 });
let slideNo = 0;
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  slideNo += 1;
  return s;
}
function card(s, x, y, w, h, name, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill },
    shadow: shadow(), objectName: name });
}
function stat(s, x, y, w, value, label, color = C.accent1, size = 40) {
  s.addText(value, { x, y, w, h: 0.8, fontSize: size, bold: true, color, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText(label, { x, y: y + 0.85, w, h: 0.75, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}
function badge(s, x, y, n, color = C.accent1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color }, line: { color }, objectName: `badge ${n}` });
  s.addText(String(n), { x, y, w: 0.46, h: 0.46, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function bullets(s, items, x, y, w, h, size = 16) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { x, y, w, h, fontSize: size, color: C.text1, valign: "top", margin: 0, isTextBox: true });
}
function img(s, file, x, y, w, h, name) {
  s.addImage({ path: file, x, y, w, h, objectName: name, sizing: { type: "contain", w, h } });
}
function caption(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.35, fontSize: 11, italic: true, color: C.accent5, margin: 0, isTextBox: true });
}

// ================================================================ slides
// 1 title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addText("Demand Forecasting for Supply Chain Inventory Optimization", { placeholder: "title" });
  s.addText("Time-series analysis on the Walmart M5 dataset, from daily sales to reorder decisions" + (AUTHOR ? `\n${AUTHOR}` : ""),
    { placeholder: "body" });
  s.addText("OptiChain", { x: M, y: 0.6, w: 4, h: 0.5, fontSize: 18, bold: true, color: C.accent4, margin: 0, isTextBox: true });
  s.addNotes("Opening (30s). The project asks two questions: can statistical time-series models forecast daily retail demand well, and does using those forecasts to set reorder points actually improve inventory outcomes compared with a simple rule of thumb? Everything is built on the public Walmart M5 dataset and delivered as an interactive dashboard called OptiChain.");
}

// 2 problem
{
  const s = content("Stock has to be ordered before demand is known", "Introduction");
  bullets(s, [
    "Under-forecast → empty shelves and lost sales",
    "Over-forecast → cash tied up in excess stock",
    "Daily demand shifts with weekday, season, events and price, and many products sell only now and then",
    "Question: do better forecasts lead to better reorder decisions?",
  ], M, 1.6, 6.3, 3.8);
  const cards = [["3,049", "products in Walmart store CA_1"], ["5.2 yrs", "of daily sales history (2011–2016)"], ["58%", "of days with zero sales for the typical product"]];
  cards.forEach(([v, l], i) => { const y = 1.55 + i * 1.75; card(s, 7.55, y, 5.2, 1.55, `stat card ${i + 1}`); stat(s, 7.85, y + 0.05, 4.7, v, l, i === 2 ? C.accent2 : C.accent1, 34); });
  s.addText("M5 contains sales only: inventory, lead times and costs in this project are hypothetical", { x: M, y: 5.75, w: 6.3, h: 0.7, fontSize: 13, italic: true, color: C.accent5, margin: 0, isTextBox: true });
  s.addNotes("Frame the problem (1 min). Ordering happens before demand is known, so forecast errors become either stockouts or excess stock. Retail demand is hard: weekly rhythm, seasons, events, and in this store the typical product has zero sales on 58% of days. Important caveat to state early: M5 has sales but no inventory data, so all inventory results are a simulation under stated assumptions.");
}

// 3 approach
{
  const s = content("An end-to-end pipeline with no look-ahead", "Introduction");
  const steps = [["Data", "Reshape and join 5.8M rows; launch and closure fixes"], ["Understand", "Weekly rhythm, seasons, events, price"],
    ["Identify", "ADF + KPSS, STL, ACF/PACF per origin"], ["Forecast", "7 methods, 4 rolling test months"],
    ["Check", "Residuals, ARCH-LM, gated GARCH"], ["Plan stock", "3 reorder policies, simulated"], ["Final test", "Pre-registered, unseen month"]];
  const bw = (W - 2 * M - 6 * 0.18) / 7;
  steps.forEach(([t, d], i) => {
    const x = M + i * (bw + 0.18);
    card(s, x, 1.75, bw, 3.1, `step ${i + 1}`, i === 6 ? C.accent4 : C.background2);
    badge(s, x + 0.22, 1.98, i + 1, i === 6 ? C.text2 : C.accent1);
    s.addText(t, { x: x + 0.22, y: 2.6, w: bw - 0.4, h: 0.5, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.22, y: 3.1, w: bw - 0.4, h: 1.6, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  });
  bullets(s, ["Every choice uses only data available at its forecast date: identification, model orders, safety stock and policy model selection",
    "The final four weeks (25 Apr – 22 May 2016) stayed locked until a single pre-registered test"], M, 5.2, W - 2 * M, 1.4, 15);
  s.addNotes("Method overview (1 min). Seven stages. The key discipline is no look-ahead: every decision is made using only data before the forecast date, and the last four weeks of data were locked away until one final test whose model choices were written down in advance.");
}

// 4 products
pres.addSection({ title: "Data & demand" });
{
  const s = content("Eight products chosen to cover every demand pattern", "Data & demand");
  img(s, R("figures", "eda", "00_adi_cv2_classification.png"), M, 1.4, 7.2, 4.8, "ADI CV2 chart");
  const rows = [["Steady", "3", "sells daily, similar amounts"], ["Spiky", "2", "sells most days, big swings"], ["Occasional", "2", "many zero days, small sales"], ["Unpredictable", "1", "irregular timing and size"]];
  rows.forEach(([k, n, d], i) => {
    const y = 1.5 + i * 1.12; card(s, 8.2, y, 4.55, 0.95, `class ${k}`);
    s.addText(n, { x: 8.4, y, w: 0.6, h: 0.95, fontSize: 30, bold: true, color: C.accent1, valign: "middle", margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    s.addText([{ text: `${k}\n`, options: { bold: true, color: C.text2 } }, { text: d, options: { color: C.text1 } }],
      { x: 9.1, y, w: 3.5, h: 0.95, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  });
  caption(s, "Syntetos–Boylan classification of 2,495 items; items with >28-day zero runs (likely stockouts) excluded", M, 6.3, 7.2);
  s.addNotes("Product selection (1 min). Rather than hand-pick, products were classified by how often they sell (ADI) and how much quantities vary (CV squared). 72% of eligible items are intermittent. One decision worth mentioning: items with month-long zero runs were excluded, because those are almost certainly stockouts rather than zero demand.");
}

// 5 demand
{
  const s = content("Weekends drive demand; events matter far less", "Data & demand");
  img(s, R("figures", "eda", "02_weekday_index_heatmap.png"), M, 1.4, 7.4, 3.6, "weekday heatmap");
  card(s, 8.4, 1.45, 4.35, 4.6, "insights card");
  bullets(s, ["Weekend sales are 1.1–1.6× an average day for 7 of 8 products",
    "Events change sales by only −18% to +9%",
    "SNAP days lift food items by 3–14%",
    "Christmas: the store is closed, so those zeros are filled for modelling",
    "Price is confounded with time, so no causal claims"], 8.65, 1.7, 3.9, 4.2, 15);
  caption(s, "Weekday index = weekday mean ÷ overall mean (2014–2015)", M, 5.15, 7.4);
  s.addNotes("EDA (1 min). The dominant pattern is the weekend peak, which is why every model uses a 7-day season. Events and SNAP have small effects. A practical detail: Christmas Day is a store closure with zero sales; it badly distorted the statistical tests until it was filled with the previous week's value.");
}

// 6 identification
pres.addSection({ title: "Forecasting" });
{
  const s = content("Identification: what each series needs before modelling", "Forecasting");
  img(s, R("figures", "diagnostics", "FOODS_3_586", "acf_pacf.png"), M, 1.45, 7.6, 2.5, "ACF PACF");
  caption(s, "FOODS_3_586 after log + differencing: spikes at 7, 14, 21… = weekly structure", M, 4.0, 7.6);
  const pts = [["No seasonal differencing", "Daily noise is larger than the weekly pattern (Fs ≤ 0.59), so seasonal AR/MA terms are used instead"],
    ["One clear unit root", "FOODS_3_586: log transform and first difference"],
    ["Ambiguous cases tested on data", "ADF/KPSS conflicts → both options tried on held-back days"],
    ["Two series are white noise", "Little to gain beyond a good average"]];
  pts.forEach(([t, d], i) => { const y = 1.45 + i * 1.2; badge(s, 8.5, y + 0.05, i + 1); s.addText([{ text: t + "\n", options: { bold: true, color: C.text2 } }, { text: d, options: { color: C.text1 } }], { x: 9.1, y, w: 3.65, h: 1.1, fontSize: 13.5, valign: "top", margin: 0, isTextBox: true }); });
  s.addNotes("Identification (1.5 min). ADF and KPSS were combined because they have opposite null hypotheses. The surprise: despite strong weekend peaks, no product passed the seasonal-strength threshold, so weekly structure goes into seasonal AR/MA terms. Where the tests disagreed, I didn't guess: both options were carried forward and decided by out-of-sample accuracy, because AIC can't compare different amounts of differencing.");
}

// 7 accuracy (native chart)
{
  const s = content("ETS and SARIMA forecast about 21% better", "Forecasting");
  const order = DATA.accuracy.order;
  s.addChart(pres.charts.BAR, [
    { name: "Backtest (4 folds)", labels: order.map((m) => DATA.accuracy.names[m]), values: order.map((m) => DATA.accuracy.backtest[m]) },
    { name: "Final test (unseen)", labels: order.map((m) => DATA.accuracy.names[m]), values: order.map((m) => DATA.accuracy.final[m]) },
  ], {
    x: M, y: 1.4, w: 8.0, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.accent1, HEX.accent4], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
    valAxisLabelColor: HEX.accent5, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 1.5,
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
    showTitle: true, title: "Mean RMSSE across 8 products (lower is better)", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
    objectName: "accuracy chart",
  });
  card(s, 8.95, 1.45, 3.8, 4.95, "accuracy callouts");
  stat(s, 9.2, 1.55, 3.4, "0.773", "ETS mean RMSSE in the backtest (best on 6 of 8)", C.accent1, 36);
  stat(s, 9.2, 3.15, 3.4, "82–84%", "of actuals fall inside the 80% prediction interval", C.accent3, 36);
  s.addText("Every fitted model beats both naive benchmarks; the ETS–SARIMA gap (<1%) is within noise", { x: 9.2, y: 4.8, w: 3.4, h: 1.4, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("Forecast accuracy (1.5 min). Seven methods, four rolling test months. ETS and SARIMA lead and are about 21% more accurate than repeating last week. Their difference is under 1%, so I treat them as jointly best. The light bars are the final unseen test, which tells the same story. Prediction intervals are well calibrated, which matters because safety stock is sized from them.");
}

// 8 forecast example
{
  const s = content("What a forecast looks like", "Forecasting");
  img(s, R("figures", "final", "forecast_example.png"), M, 1.4, 8.4, 3.2, "forecast example");
  card(s, 9.3, 1.45, 3.45, 4.6, "example notes");
  bullets(s, ["ETS follows the weekend peaks at a smoothed height", "Seasonal naive copies one noisy week", "Actual sales mostly stay inside the shaded 80% band", "Model orders and settings were chosen without seeing this month"], 9.55, 1.7, 3.0, 4.2, 14);
  caption(s, "FOODS_3_586, 28-day forecast made on 27 March 2016 vs held-out actuals", M, 4.7, 8.4);
  s.addNotes("Example (45s). This is the high-volume food item. The model captures the weekly rhythm but averages it over many weeks instead of copying last week's noise; that's where the accuracy gain comes from.");
}

// 9 model check
{
  const s = content("Model checks pass; volatility modelling is not needed", "Forecasting");
  const st = [["32 / 32", "fits with no systematic bias", C.accent3], ["0.096", "largest leftover correlation in errors (tiny)", C.accent1], ["≤ 2.4%", "interval gain from GARCH, and worse for SARIMA", C.accent2]];
  st.forEach(([v, l, c], i) => { const x = M + i * 4.1; card(s, x, 1.5, 3.85, 2.0, `check ${i + 1}`); stat(s, x + 0.3, 1.6, 3.3, v, l, c, 38); });
  bullets(s, ["Ljung–Box flags small leftover weekly structure in about half the fits; with ~700 points, tiny effects become 'significant'",
    "ARCH-LM finds volatility clustering in only 3 high-volume series",
    "GARCH(1,1) α is small and mostly insignificant, so it is not adopted (a valid negative result)",
    "What matters for stock is heavy tails (big spikes), not volatility waves"], M, 3.85, W - 2 * M, 2.6, 15);
  s.addNotes("Residual diagnostics (1.5 min). Errors are unbiased in all 32 fits. There is a little leftover weekly pattern, statistically detectable, practically tiny. ARCH-LM found volatility clustering in three series, so GARCH was fitted there, but it improved interval scores by at most 2.4% and made SARIMA intervals worse. So GARCH is not used. The plan explicitly allows that negative result. I'm transparent that a stricter acceptance check was added after seeing the fits; both are reported.");
}

// 10 policies
pres.addSection({ title: "Inventory" });
{
  const s = content("Three ways to decide when and how much to reorder", "Inventory");
  const pol = [["Simple rule (P0)", "Last 28 days' average × lead time + safety buffer", C.accent2, "$674"],
    ["Forecast-driven (P1)", "Sum of forecasts over the lead time + buffer from model errors", C.accent1, "$600"],
    ["Empirical buffer (P2)", "P1 with the buffer taken from the real spread of past errors", C.accent3, "$595"],
    ["Perfect hindsight", "Uses actual future sales (upper bound on what forecasting can deliver)", C.accent5, "$595"]];
  pol.forEach(([t, d, c, v], i) => {
    const x = M + i * 3.07; card(s, x, 1.5, 2.87, 3.55, `policy ${i + 1}`);
    s.addText(t, { x: x + 0.25, y: 1.7, w: 2.4, h: 0.5, fontSize: 16, bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: 2.25, w: 2.4, h: 1.5, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    s.addText(v, { x: x + 0.25, y: 3.85, w: 2.4, h: 0.7, fontSize: 32, bold: true, color: c, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    s.addText("total cost, 8 products", { x: x + 0.25, y: 4.55, w: 2.4, h: 0.35, fontSize: 11, color: C.accent5, margin: 0, isTextBox: true });
  });
  bullets(s, ["Replay of 3 Jan – 24 Apr 2016: same real sales and opening stock for every policy, daily review, lost sales",
    "Hypothetical: 4-day lead time, 95% service target, holding 25%/yr, missed-sale penalty 50% of price, $5 per order",
    "Forecast-driven plan: 11% cheaper, 4% less stock, fewer missed sales (44 vs 49 units)"], M, 5.3, W - 2 * M, 1.6, 14);
  s.addNotes("Inventory simulation (1.5 min). Every policy faces the same real sales and starting stock; only the reorder rule differs. The forecast-driven plan sums the forecast over the delivery window. Policy 2 replaces the bell-curve buffer with the real spread of past errors. In the default scenario the forecast plan is 11% cheaper. Perfect hindsight isn't the cheapest because the rule isn't cost-optimised; it bounds service and stock, not cost.");
}

// 11 frontier
{
  const s = content("Where forecasting helps, and where it doesn't", "Inventory");
  img(s, R("figures", "final", "frontier.png"), M, 1.35, 8.9, 4.15, "frontier chart");
  card(s, 9.75, 1.4, 3.0, 4.6, "frontier notes");
  bullets(s, ["Up and left = more service from less stock", "Forecasting wins for steady and some slow movers", "The simple rule is more efficient for FOODS_3_203 and spiky HOBBIES_1_189", "Short window: few stockouts, so small gaps are noise"], 9.95, 1.6, 2.65, 4.3, 13);
  caption(s, "Each point = one service target (80–99%); lead time 4 days", M, 5.55, 8.9);
  s.addNotes("Trade-off curves (1 min). Sweeping the service target shows the full trade-off. For steady products the forecast curves sit up and to the left: more service from less stock. But for one calm food item and one spiky hobby item the simple rule is more efficient, so forecasting is not a universal win.");
}

// 12 robustness + key finding
{
  const s = content("Savings hold in 8 of 9 scenarios, but they vary", "Inventory");
  img(s, R("figures", "final", "sensitivity.png"), M, 1.4, 7.6, 2.55, "sensitivity heatmap");
  caption(s, "Total-cost change vs simple rule across lead time × service target (blue = cheaper)", M, 4.0, 7.6);
  stat(s, M, 4.5, 3.4, "3–11%", "cheaper than the simple rule; the default scenario is the best case", C.accent1, 36);
  stat(s, 4.3, 4.5, 3.6, "+18–26%", "how much real errors exceeded the in-sample estimate for 4 products", C.accent2, 36);
  card(s, 8.6, 1.45, 4.15, 4.95, "key finding");
  s.addText([{ text: "Key finding\n", options: { bold: true, color: C.text2, fontSize: 17 } },
    { text: "The limit is not the forecast's shape but how its uncertainty is estimated. Buffers use in-sample one-step errors, while decisions use forecasts made weeks earlier.\n\nFix: size buffers from out-of-sample errors and re-forecast daily.", options: { color: C.text1, fontSize: 14 } }],
    { x: 8.85, y: 1.7, w: 3.7, h: 4.5, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Robustness and key finding (1.5 min). Across nine lead-time and service scenarios, the forecast plan is cheaper in eight, by 3 to 11%. I quote the range because the default scenario happens to be the most favourable. The most useful finding: the bell-curve versus empirical buffer choice barely matters (0–15%). What matters is that in-sample errors misjudge real errors by up to 26%, because decisions rely on forecasts made weeks earlier. That points directly at the next improvement.");
}

// 13 final test
pres.addSection({ title: "Validation" });
{
  const s = content("Pre-registered final test: the results hold on unseen data", "Validation");
  img(s, R("figures", "final", "final_test.png"), M, 1.4, 8.2, 3.0, "final test chart");
  caption(s, `Model per product written down before the month (25 Apr – 22 May 2016) was loaded`, M, 4.45, 8.2);
  stat(s, 9.2, 1.45, 3.5, `${DATA.final.rmsse_test.toFixed(3)}`, `mean RMSSE on the unseen month (backtest: ${DATA.final.rmsse_backtest.toFixed(3)})`, C.accent3, 40);
  stat(s, 9.2, 3.2, 3.5, `$${DATA.final.cost_p1} vs $${DATA.final.cost_p0}`, "forecast plan vs simple rule, mostly from one product", C.accent1, 30);
  bullets(s, ["No sign of overfitting; SARIMA and ETS stay jointly best",
    "Excluding HOUSEHOLD_1_469, costs are equal but the forecast plan holds about 10% less stock",
    "Per-product scores move both ways, a reminder that 4 folds are a small sample"], M, 4.95, W - 2 * M, 1.6, 14);
  s.addNotes("Final test (1 min). Before loading the last four weeks I wrote down which model each product would use. On that unseen month the forecasts were just as accurate as in the backtest, slightly better in fact. Inventory: the forecast plan is cheaper overall, but honestly most of that comes from one product; without it, costs are equal while the forecast plan holds about 10% less stock.");
}

// 14 dashboard
{
  const s = content("OptiChain: the results as a six-step dashboard", "Validation");
  img(s, R("figures", "final", "dashboard_plan.png"), M, 1.4, 7.5, 4.69, "dashboard screenshot");
  card(s, 8.45, 1.45, 4.3, 4.6, "dashboard steps");
  const steps = ["Overview", "Understand demand", "Predict", "Model check", "Plan stock", "Report"];
  steps.forEach((t, i) => { const y = 1.65 + i * 0.7; badge(s, 8.7, y, i + 1, i === 4 ? C.accent2 : C.accent1); s.addText(t, { x: 9.3, y, w: 3.2, h: 0.46, fontSize: 15, color: C.text2, bold: i === 4, valign: "middle", margin: 0, isTextBox: true }); });
  caption(s, "Plain-language labels, live what-if sliders, analyst detail on demand; ready for Streamlit Cloud", M, 6.15, 12);
  s.addNotes("LIVE DEMO (3 min): (1) Overview: pick Food #3-586. (2) Understand demand: Sunday is busiest, +37%. (3) Predict: switch methods and forecast dates; the method bars. (4) Model check: verdict cards. (5) Plan stock: read the recommendation, move delivery time to 7 days, press Replay to watch stock levels. (6) Report: the 8-of-9 robustness heatmap and downloads. Backup if the demo fails: this screenshot and the report figures.");
}

// 15 conclusions
pres.addSection({ title: "Conclusion" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  s.addText("Better forecasts help, and better uncertainty estimates would help more", { placeholder: "title", fontSize: 32 });
  s.addText([
    { text: "ETS and SARIMA forecast about 21% better than seasonal naive, confirmed on an unseen month", options: { bullet: true, breakLine: true } },
    { text: "Forecast-driven reordering is cheaper in 8 of 9 scenarios (3–11%), but not for every product", options: { bullet: true, breakLine: true } },
    { text: "GARCH not needed; heavy-tailed errors and error estimation matter more", options: { bullet: true, breakLine: true } },
    { text: "Next: out-of-sample error buffers, daily re-forecasting, cost-optimal service targets, more products", options: { bullet: true } },
  ], { placeholder: "body", fontSize: 17, paraSpaceAfter: 6 });
  s.addNotes("Close (1 min). Three takeaways and next steps. Inventory figures are simulated under stated assumptions, and all code, notebooks, the report and the dashboard are reproducible from the repository. Invite questions.");
}

pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, "slides:", slideNo + 2);
});
