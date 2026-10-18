# OptiChain — demonstration script

**Format:** 15-minute talk using `reports/OptiChain_Presentation.pptx` (speaker notes on every slide), with a
3-minute live dashboard demo at slide 14. The flow follows the project plan's recommended demonstration story:
one product from raw data through forecast to inventory action.

**Before you start**
- From the project root, run `streamlit run app/Home.py` and open `http://localhost:8501` once to warm it up. Leave it on the Overview page.
- Keep the PDF report open as a backup for every figure.
- Product for the demo: **Food #3-586 (FOODS_3_586)**, the steady, high-volume item.

## Talk (slides 1–13, about 11 minutes)

| Slide | Time | Key line |
|---|---|---|
| 1 Title | 0:30 | Can statistical forecasts improve reorder decisions, not just predictions? |
| 2 Problem | 1:00 | Orders happen before demand is known; the typical product sells nothing on 58% of days. Inventory figures are simulated. |
| 3 Approach | 1:00 | Seven stages, with no look-ahead anywhere; the last four weeks stayed locked. |
| 4 Products | 1:00 | Eight products chosen by demand pattern (ADI/CV²); month-long zero runs excluded as likely stockouts. |
| 5 Demand | 1:00 | Weekends drive demand (1.1–1.6×); events matter little; Christmas is a closure. |
| 6 Identification | 1:30 | ADF + KPSS together; no seasonal differencing; ambiguous cases decided on held-back data. |
| 7 Accuracy | 1:30 | ETS and SARIMA jointly best, ~21% better than seasonal naive; calibrated intervals. |
| 8 Example | 0:45 | The model averages the weekly rhythm instead of copying last week's noise. |
| 9 Model checks | 1:30 | Unbiased residuals; tiny leftover pattern; GARCH tested and rejected (state the post-hoc stricter check). |
| 10 Policies | 1:30 | Same sales, same stock, only the rule differs; forecast plan 11% cheaper in the default scenario. |
| 11 Trade-off | 1:00 | Forecasting wins for steady items, not for every product. |
| 12 Robustness | 1:30 | Cheaper in 8 of 9 scenarios (3–11%); the real limit is the error estimate, off by up to 26%. |
| 13 Final test | 1:00 | Pre-registered month: RMSSE 0.752 vs 0.769, so no overfitting; inventory gain mostly from one product. |

## Live demo (slide 14, about 3 minutes)

1. **Overview** (20 s): "Each card is one product; the sparkline draws its recent sales." Click **Explore** on *Food #3-586*.
2. **Understand demand** (30 s): point at *Busiest day: Sunday, +37%* and the weekday bars. Press **1Y** on the history chart.
3. **Predict** (40 s): forecast date **Mar 27**, method **Exponential smoothing**. "The shaded band is the likely range, and actual sales stay inside it." Switch to **Same day last week** to show the noisier forecast. Point at the method-comparison bars.
4. **Model check** (30 s): read the four verdict cards: no bias, negligible pattern, GARCH not needed.
5. **Plan stock** (50 s): read the recommendation card. Move **Supplier delivery time** to 7 days and watch the plan update. Press **▶ Replay** on the stock chart. Point at the cost breakdown and the what-if heatmap.
6. **Report** (20 s): the 8-of-9 robustness heatmap and the downloads.

**If the demo fails:** stay on slide 14 (screenshot) and use report figures 9–12.

## Close (slide 15, about 1 minute)

Three takeaways, next steps, and the reminder that inventory results are simulated under stated assumptions.

## Likely questions

- **Why not machine learning (LightGBM)?** The plan prioritised a fully diagnosed statistical pipeline. A global ML model is listed as future work.
- **Why is Perfect hindsight not the cheapest?** The (s, S) rule is not cost-optimised. Exact forecasts with no buffer can trigger more orders. It bounds service and stock, not cost.
- **Is the GARCH rejection post hoc?** The pre-specified rule flags two ETS series with gains of 1–2.4%. A stricter check, added after inspection, flags none. Both are reported.
- **Can results transfer to Walmart?** No claim is made: inventory, lead times and costs are hypothetical, and sales are censored by real stockouts.
