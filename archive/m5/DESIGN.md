# OptiChain — Design Document

**Demand Forecasting for Supply Chain Inventory Optimization Using Time-Series Analysis (Walmart M5)**

This document turns the project plan (PDF blueprint) into concrete engineering decisions: what gets built, in what order, with what interfaces, and which open choices in the plan have been settled (and why). Anything marked **[Decision]** is a choice the plan left open.

---

## 0. Guiding principles

1. **Vertical slice first.** One store, one product, naive + seasonal-naive, Policy 0 vs Policy 1, one dashboard page — working end to end before anything is widened.
2. **No leakage, ever.** Every number used at forecast origin *T* is computed from data with date ≤ *T*. Product selection, transformations, model orders, safety-stock sigmas — all of it.
3. **Precompute heavy, serve light.** Notebooks/scripts fit models and write artefacts; the Streamlit app only reads artefacts and runs the (cheap) inventory simulator live.
4. **Inventory is a labelled simulation.** All lead times, costs and starting stock live in `config.yaml` under a `simulation_assumptions` key and are shown as "hypothetical" in every output.
5. **Negative results are results.** If SARIMA loses to seasonal-naive or ARCH-LM is not significant, that goes in the report as-is.

---

## 1. Data design

### 1.1 Files and time windows

| Day ID | Date | Role |
|---|---|---|
| d_1 | 2011-01-29 (Sat) | Series start |
| d_1801–d_1828 | 2016-01-03 → 2016-01-30 | Rolling fold 1 test |
| d_1829–d_1856 | 2016-01-31 → 2016-02-27 | Rolling fold 2 test |
| d_1857–d_1884 | 2016-02-28 → 2016-03-26 | Rolling fold 3 test |
| d_1886–d_1913 | 2016-03-28 → 2016-04-24 | **Primary holdout** (plan's recommended split; train ≤ d_1885) |
| d_1914–d_1941 | 2016-04-25 → 2016-05-22 | **Final untouched test** (from `sales_train_evaluation.csv`) |

**[Decision]** Use `sales_train_evaluation.csv` as the single sales source (it is a superset of the validation file, d_1–d_1941), but the pipeline enforces a `max_day` cutoff so d_1914–d_1941 are physically excluded from every step until the one final evaluation run. This gives a genuinely unseen test set without a second ingestion path.

> Note: fold 3 ends at d_1884 and the primary holdout starts at d_1886 because the plan fixes train-through-d_1885. Folds are defined by origins `[1800, 1828, 1856, 1885]`, each followed by a 28-day test window.

### 1.2 Ingestion strategy (memory)

The full sales table is ~30k rows × 1,947 columns; full melt is ~59M rows. We never melt everything.

1. Read sales CSV with `dtype` for id columns = `category`, `d_*` = `int16`.
2. **Filter to `store_id == CA_1` before melting** (3,049 series × 1,941 days ≈ 5.9M rows — fine in memory).
3. Read `sell_prices.csv` filtered to the same store (`float32` price).
4. Melt → join `calendar.csv` on `d` → join prices on `(store_id, item_id, wm_yr_wk)`.
5. Validate after each join (see §1.4), write `data/processed/ca1_long.parquet`.
6. Write a second, tiny file `data/processed/selected_series.parquet` (selected items only) — this is what the app ships with.

`data/raw/` is read-only (set file attribute read-only after download; the pipeline only ever opens it for reading).

### 1.3 Long-form schema

| Column | Type | Notes |
|---|---|---|
| `date` | datetime64 | from calendar |
| `d` | int16 | numeric day index (strip `d_`) — makes cutoffs trivial |
| `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | category | |
| `sales` | int16 | 0 is a real observation |
| `wm_yr_wk` | int32 | |
| `weekday`, `wday`, `month`, `year` | category/int8 | |
| `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` | category | NaN = no event |
| `snap_CA` | int8 | only the CA flag is relevant for a CA store |
| `sell_price` | float32 | NaN = no price row (see §1.5) |
| `on_sale` | bool | `sell_price.notna()` — derived |

### 1.4 Validation checks (run in pipeline, also as pytest)

- Row count after melt = n_series × n_days; unchanged after both left joins.
- `(item_id, store_id, date)` unique.
- `sales >= 0`, dates contiguous daily, `d ↔ date` mapping monotonic.
- Calendar join: zero NaN dates. Price join: report % NaN per item (expected for pre-launch periods).
- `max(d) <= max_day` cutoff assertion.

### 1.5 Missing-value policy

| Situation | Handling |
|---|---|
| `sales == 0` | Kept. Never treated as missing. |
| Leading zeros **before product launch** (no price row yet) | **[Decision]** Trim each series to start at the first date with a `sell_price`. Zeros before launch are not demand — they are "product didn't exist". Trimming rule uses only the price table, so no leakage. |
| `sell_price` NaN after launch | Forward-fill within item (price persists until changed); flag count in data-quality report. Never back-fill (leaks future price). |
| Event columns NaN | Means "no event"; encoded as `"none"`. |
| NaNs from differencing | Dropped inside the model-fitting function; effective n is logged. |

---

## 2. Product selection

**[Decision]** Select products with a data-driven demand classification instead of hand-picking, so the "why does model performance vary" story is built in.

Compute on **training data only (d ≤ 1800, the earliest fold origin)** for each CA_1 item with ≥ 2 years of post-launch history:

- **ADI** (average inter-demand interval) = n_days / n_nonzero_days
- **CV²** of non-zero demand sizes

Syntetos–Boylan quadrants (cut-offs ADI = 1.32, CV² = 0.49):

| Class | Rule | Pick | Why |
|---|---|---|---|
| Smooth | ADI < 1.32, CV² < 0.49 | 3 (incl. 1 very high-volume FOODS_3 item) | ARIMA/SARIMA home turf; weekly seasonality visible |
| Erratic | ADI < 1.32, CV² ≥ 0.49 | 2 | Frequent sales, spiky sizes — tests variance handling, ARCH candidate |
| Intermittent | ADI ≥ 1.32, CV² < 0.49 | 2 | Many zeros — where ARIMA struggles, Croston is relevant |
| Lumpy | ADI ≥ 1.32, CV² ≥ 0.49 | 1 | Hardest case; honest negative example |

Total **8 products**, spanning FOODS / HOUSEHOLD / HOBBIES where possible. Within each quadrant, pick deterministically (e.g. the item closest to the quadrant's median volume, seeded tie-break) so selection is reproducible. The output is written into `config.yaml` → `series.items` and frozen; the selection notebook is re-runnable but not re-run silently.

**[Decision] Zero-run exclusion (added in M2).** Items with any run of more than 28 consecutive zero-sales days in the classification window are excluded. At these sales rates such runs are almost certainly stockouts or delistings (censored demand), not true zero demand. 13% of smooth and about 66% of intermittent and lumpy items have such runs, so censoring is widespread in M5 and is listed as a limitation.

**Frozen selection** (`scripts/select_products.py`, window d_1071–d_1800, 2,495 eligible items):

| Item | Class | Mean/day | Zero share |
|---|---|---|---|
| FOODS_3_586 | smooth (top volume) | 44.9 | 0.3% |
| HOUSEHOLD_1_447 | smooth | 6.2 | 5.5% |
| FOODS_3_203 | smooth | 3.5 | 8.2% |
| HOBBIES_1_189 | erratic | 7.3 | 16.4% |
| HOBBIES_1_030 | erratic | 4.8 | 20.4% |
| HOUSEHOLD_1_469 | intermittent | 1.1 | 37.4% |
| FOODS_3_242 | intermittent | 0.7 | 51.2% |
| HOUSEHOLD_1_046 | lumpy | 1.3 | 36.0% |

**EDA cutoff.** All exploratory analysis uses d ≤ 1800 (the earliest forecast origin), so no test window influences any modelling choice.

---

## 3. Analysis pipeline (EDA → stationarity → identification)

### 3.1 EDA outputs (per selected series, saved to `reports/figures/eda/`)

- Daily line plot + 28-day rolling mean; weekly-aggregated plot.
- Histogram + box plot of daily sales.
- Weekday bar chart (mean ± CI) and month summary; calendar heatmap for the highest-volume item.
- Sales vs price scatter (coloured by year to avoid confounding trend with price).
- Event vs non-event mean demand table; SNAP vs non-SNAP for FOODS items.
- **Variability table** (one row per item): mean, median, std, CV, zero-share %, ADI, CV², class, mean price.

QQ plots are produced only for *model residuals* (§5), not for raw sales.

### 3.2 Stationarity procedure (scripted, per series, on training window)

1. Plot series; ADF (`regression="c"`, lag selection AIC) **and** KPSS (`regression="c"`). Using both avoids the classic ADF over-interpretation: ADF null = unit root, KPSS null = stationary.
2. If the variance–level relationship is visible (rolling std vs rolling mean correlation > threshold, smooth/erratic only) → test `log1p`. Back-transform: `expm1`, documented as returning approximately the median, not the mean.
3. If ADF fails to reject / KPSS rejects → first difference, re-test.
4. Seasonal unit root: check with OCSB / seasonal strength from STL; seasonal differencing (D=1, s=7) only if supported.
5. Record per-series decision `{transform, d, D}` in `models/identification.json` — consumed by the modelling step, so the decision is traceable.

### 3.3 Decomposition

STL with period 7 (robust=True) on the last 2 years of training data. Additive by default; multiplicative (i.e. STL on log1p) only for series flagged in 3.2. Annual pattern: shown descriptively on weekly aggregates only, not modelled (only ~5 annual cycles; not enough to justify s=365 SARIMA, which is also computationally impractical).

### 3.4 ACF/PACF

Plotted to 35 lags (5 weeks) on the transformed/differenced series. Used to set a **small candidate grid**, not to read off a single order.

### 3.5 Decisions added in M4 (implemented in `diagnostics.py`, notebook 02)

- **[Decision] Store-closure adjustment.** Christmas (25 Dec) is a store closure with zero sales. After log1p it became a large outlier that distorted the tests and the STL remainder. All modelling and simulation now use `get_series(fill_closures=True)`, which replaces closure days with the same weekday one week earlier (past data only). EDA still shows the raw series. The backtest and simulation were re-run with this adjustment.
- **[Decision] Identification per origin.** Identification is run separately at each of the 4 origins, on the 730 days before that origin, and stored in `models/identification.json`. M5 reads the record for the matching origin.
- **[Decision] Ambiguous cases become candidates, not guesses.** ADF/KPSS "conflict" or "inconclusive" → `d_candidates = [0, 1]`. A slowly decaying seasonal ACF (lags 7–28 all > 0.2) → `D_candidates` includes 1. AIC cannot compare different d or D, so M5 settles these by out-of-sample RMSSE.
- **[Decision] log1p only for series with ≤ 10% zero days** and a variance–level correlation ≥ 0.6.
- **Results at d_1800:**
  - No product reaches the seasonal-strength threshold (F_s ≤ 0.59), so D = 0 by default. The weekly pattern goes into seasonal AR/MA terms.
  - FOODS_3_586 is the only clear unit root (log1p, d = 1, with D = 1 as a candidate).
  - 4 products are conflict cases.
  - FOODS_3_242 and HOUSEHOLD_1_046 are white noise (Ljung–Box p ≈ 0.3), so ARIMA is expected to add little for them.

---

## 4. Forecasting models

### 4.1 Model set

| Key | Model | Implementation | Notes |
|---|---|---|---|
| `naive` | Last value | own code | Benchmark |
| `snaive` | Same weekday last week | own code | **The bar to beat** for daily retail |
| `ma28` | 28-day moving average | own code | Flat forecast; what the baseline inventory policy uses |
| `ets` | Holt–Winters / ETS, additive error, damped trend optional, seasonal period 7 | `statsmodels.tsa.holtwinters.ExponentialSmoothing` | Trend/no-trend chosen by AICc |
| `arima` | ARIMA(p,d,q) | `statsmodels SARIMAX` | grid p,q ∈ {0,1,2}, d from §3.2 |
| `sarima` | SARIMA(p,d,q)(P,D,Q)₇ | `statsmodels SARIMAX` | grid p,q ∈ {0,1,2}, P,Q ∈ {0,1}, D from §3.2 (≤ 36 fits/series) |
| `croston` | Croston / SBA | own code (~30 lines) | **[Decision] added**: the plan stresses intermittent demand but lists no intermittent model; SBA is the standard benchmark there |
| `lgbm` *(optional, phase 2)* | LightGBM on lags/calendar/price | `lightgbm` | Only after slice is complete |

**[Decision]** Do not use `pmdarima` — fragile with recent NumPy/Python. A ~40-line grid search over `SARIMAX` is transparent and reportable (the AIC table itself is a deliverable).

**[Decision]** Fit window = **last 730 days before origin** for ARIMA/SARIMA/ETS (configurable). Reasons: SARIMAX on 1,900 daily points × 36 candidates × 8 series × 4 folds is slow; older data reflects different assortment/pricing; all candidates compared by AIC/BIC use the same window, as the plan requires.

### 4.2 Selection vs evaluation (kept separate)

- **Within a model family**: order chosen by AIC (BIC reported alongside) on the training window. Non-converged fits are discarded and logged.
- **Across families**: compared only by out-of-sample metrics on the rolling folds. The final "selected model" per series = lowest mean RMSSE across the 4 folds (origins 1800, 1828, 1856, 1885).
- **Final test** (d_1914–1941): run once, with the selected model per series refit on data through d_1913. Reported separately.

### 4.2b Decisions and results added in M5 (implemented in `models_stat.py`, `backtest.py`, notebook 03)

- **[Decision] (d, D) resolved by an inner holdout.** When identification allows several (d, D) combinations, the AIC-best order for each is scored on the last 28 days of the training window, and the winner is refit on the full window. The real test window is never used for any choice.
- **[Decision] ETS restricted to additive error.** The form is chosen by AICc over {none, damped trend} × {none, weekly season}. Multiplicative forms need strictly positive data.
- **[Decision] Croston uses the SBA bias correction**, with α ∈ {0.05, 0.1, 0.2, 0.3} chosen by in-sample one-step MSE.
- **[Decision] Parallel, cached backtest.** One task = (item, model, origin). The cache key hashes only training data up to the origin, plus settings and identification, so later data cannot affect a cached fit. 2,342 candidate fits run in about 4 minutes on 11 workers.
- **Results (mean RMSSE, 4 folds):**
  - ETS 0.773 (best on 6/8), SARIMA 0.779 (best on 2/8), ARIMA 0.812, Croston 0.812, MA28 0.821, seasonal naive 1.012, naive 1.062.
  - Prediction intervals are calibrated: 80% → 82–84% coverage, 95% → 92–94%.
  - P1 (forecast-driven) is now cheaper than P0 on 7/8 products ($600 vs $674 in total, −11%), with fewer lost units (44 vs 49).

### 4.3 Post-processing

- Inverse-transform (`expm1`) if applicable.
- Clip at 0. Keep unrounded values for metrics; round only in the UI/order quantities.
- Intervals (80% / 95%) from model where available (`get_forecast().conf_int()`), stored separately from point forecasts.
- Assertion: every forecast date > origin date; horizon length == requested.

### 4.4 Forecast artefact schema — `reports/forecasts/forecasts.parquet`

`item_id, model, origin_d, date, h, yhat, lo80, hi80, lo95, hi95, y_true` (y_true joined after the fact for evaluation; never present at fit time).

---

## 5. Evaluation & diagnostics

### 5.1 Metrics (`src/evaluation.py`)

| Metric | Definition / zero handling |
|---|---|
| MAE | mean \|y − ŷ\| |
| RMSE | √mean (y − ŷ)² |
| Bias (ME) | mean (ŷ − y) — positive = over-forecast; directly relevant to inventory |
| sMAPE | mean 2\|y−ŷ\|/(\|y\|+\|ŷ\|), terms where both are 0 count as 0 error |
| RMSSE | RMSE / √(mean of squared one-step naive differences on the *training* series, from first non-zero sale) — M5 definition. If denominator = 0 → return NaN and log, never inf |
| MASE | MAE-scaled equivalent, same denominator logic |
| PI coverage | share of y within 80%/95% intervals |

MAPE is computed only for completeness on smooth series and never used for ranking. WRMSSE: out of scope (we have 8 series, not the 42,840-series hierarchy).

Outputs: `reports/metrics/forecast_metrics.csv` (item × model × fold), plus a pivot (item × model, mean RMSSE) and a "skill vs snaive" column (1 − RMSSE_model / RMSSE_snaive).

### 5.2 Residual diagnostics (selected model per series, `src/diagnostics.py`)

- Residuals over time, histogram, QQ plot, ACF of residuals.
- **Ljung–Box** at lags 7, 14, 21 (weekly-multiples; `model_df` = number of ARMA params).
- **ARCH-LM** (`het_arch`) at lag 7 on residuals; ACF of squared residuals.
- Output `reports/metrics/diagnostics.csv`: item, model, LB p-values, ARCH-LM stat/p, residual mean, adequacy flag.

### 5.3 ARCH/GARCH gate

Proceed to GARCH(1,1) (`arch` package, on mean-model residuals) **only** for series with ARCH-LM p < 0.05 **and** ≤ 30% zero days. Evaluate by interval coverage on the folds vs constant-variance intervals. Expected outcome: GARCH justified for at most the smooth/erratic high-volume items; intermittent ones documented as "not applicable". If used, its conditional σ feeds Policy 2 (§6.4).


**Outcome (M6, `residual_checks.py`, notebook 04):**
- **Residuals:** unbiased in 32/32 selected fits. Ljung–Box rejects at lag 7 in about half, but the largest |ACF| is 0.096, so the leftover autocorrelation is practically small; documented as a limitation. Heavy tails (excess kurtosis 4–9) for the spiky and lumpy products.
- **ARCH effects:** only FOODS_3_586, FOODS_3_203 and HOUSEHOLD_1_447 (plus HOBBIES_1_189 under SARIMA).
- **GARCH(1,1)** where the gate passed: α 0.01–0.05, mostly insignificant; held-out interval-score gains ≤ 2.4% for ETS, and worse for every SARIMA variant (up to −79%). The pre-specified rule flags two ETS series. A stricter recommendation (α significant at most origins and ≥ 5% gain), added after inspection and reported alongside, flags none.
- **[Decision] GARCH is not adopted.** Heavy tails, not volatility clustering, are the relevant uncertainty issue, so M7's Policy 2 uses empirical error quantiles.
---

## 6. Inventory simulation

### 6.1 Model of the world (all hypothetical)

- **Single item, single store, daily periodic review**, simulated independently per item.
- **Lost sales** (not backorders) — closer to retail shelves. Fill rate = units sold / units demanded.
- **Lead time L** deterministic, configurable (default 4; sensitivity {2, 4, 7}).
- Orders arrive at the **start** of day t + L.

**Daily event order** (fixed and tested):
1. Receive orders due today → on-hand += qty
2. Demand realised (actual `sales`): sold = min(on_hand, demand); lost = demand − sold
3. Holding cost on end-of-day on-hand
4. Review: inventory position IP = on_hand + on_order. If IP ≤ ROPₜ → order Qₜ = ceil_to_pack(Sₜ − IP)
5. Ordering cost if an order is placed

> Caveat in the report: observed M5 sales are *censored* demand (Walmart's own stockouts already cut sales). We treat observed sales as demand — stated as a limitation.

### 6.2 Policies (identical L, costs, service target, start stock)

Let R = review interval (1 day), z = Φ⁻¹(target CSL), μ/σ from training only.

| Policy | ROPₜ | Order-up-to Sₜ | Safety stock |
|---|---|---|---|
| **P0 Baseline** (static) | μ₂₈·L + SS₀ | ROP + μ₂₈·D_cover | SS₀ = z·σ_d·√L, σ_d = std of daily demand over last 28 training days. Recomputed only at each fold origin. |
| **P1 Forecast-driven** | Σ ŷ(t+1 … t+L) + SS₁ | Σ ŷ(t+1 … t+L+D_cover) + SS₁ | SS₁ = z·σ_e·√L, σ_e = std of one-step in-sample forecast errors of the selected model |
| **P2 Uncertainty-aware** *(optional)* | Σ ŷ(t+1 … t+L) + SS₂ | same as P1 with SS₂ | SS₂ = empirical z-quantile of **L-day cumulative** forecast errors from earlier folds (or GARCH σ if §5.3 passed) |

`D_cover` (days of cover above ROP, default 7) controls order size / frequency; same for all policies. The plan's "sum of forecasts over the lead-time window" rule is used for P1/P2.

**[Decision] Forecast availability during simulation:** forecasts are issued at fold origins with horizon **28 + L_max + D_cover** so the lead-time window never runs off the end of the forecast. v1 uses these static forecasts; v2 (stretch) updates the SARIMA state daily with `results.append(new_obs, refit=False)` — no refit, no leakage, more realistic.

**[Decision] Simulation horizon** = the 4 folds back-to-back (d_1801–d_1913, 113 days) — 28 days alone is too short for stable inventory metrics. Starting on-hand at d_1801 = μ·(L + D_cover) for **both** policies; first 7 days reported but optionally excluded as warm-up.

**Decisions added in M3 (implemented in `inventory.py` / `simulation.py`):**
- **P1 model choice is rolling and leak-free.** At each origin, P1 uses the model with the best mean RMSSE over folds whose test window has *already ended* by that origin. At d_1800 no fold is complete, so it uses `snaive`. Choosing the model on the same windows that are simulated would quietly favour P1.
- **Opening stock is rounded up to whole units** (ceil of μ·(L+cover)), so lost sales are always whole units.
- **Lookahead limit:** forecasts run 42 days (28 + max L + cover). Origins are up to 29 days apart, so **L + cover ≤ 14**, and the dashboard sliders enforce this.
- **Perfect foresight is a bound on forecast accuracy, not on cost.** Under this non-optimised (s,S) rule, exact forecasts with no safety stock can order more often and cost more than P1. It shows the best achievable fill rate, not the minimum cost.
- P1 is also run with each model fixed (`P1 [naive]`, `P1 [snaive]`, …) to show how sensitive inventory outcomes are to forecast quality.

**Decisions and results added in M7 (`inventory.py`, `simulation.py`, notebook 05):**
- **[Decision] Policy 2 = empirical safety stock.** SS is the CSL-quantile of L-day rolling sums of the chosen model's in-sample one-step errors at that origin. The quantiles are computed in the backtest (`residual_quantiles.parquet`, L = 1–14). They keep heavy tails and autocorrelation without assuming normality, and use no future data. GARCH was rejected in M6.
- **[Decision] Service-vs-stock frontier.** The service target is swept from 80% to 99% for every policy, giving (average stock, fill rate) curves.
- **Results:**
  - P1 is cheaper than P0 in 8/9 lead-time × service scenarios (−3% to −11%; +2.8% at L = 2, 98%). The default (L = 4, 95%) is the most favourable at −11%.
  - P2 ≈ P1 overall: empirical buffers are only 0–15% larger than normal ones, and P2's savings are less consistent (−12% to +3.5%).
  - The frontiers favour forecasting for 4–5 products, while the simple rule is more stock-efficient for FOODS_3_203 and HOBBIES_1_189.
- **Key finding:** buffer sizing is limited by in-sample error estimates. Out-of-sample errors are 18–26% larger for the HOBBIES products, FOODS_3_242 and HOUSEHOLD_1_447, and smaller for FOODS_3_586 and HOUSEHOLD_1_469. The normal-vs-empirical choice matters less.
  - Recommended next steps: out-of-sample h-step error quantiles, and daily state-updating re-forecasts.
- P1 with a fixed model: naive is worst ($880, worse than P0); ETS is best ($593); SARIMA costs $644 despite near-best RMSSE. Accuracy and inventory cost are only loosely linked.

### 6.3 Costs (hypothetical, price-linked so they scale per item)

| Parameter | Default | Formula |
|---|---|---|
| Holding | 25% annual carrying rate | h = 0.25 × sell_price / 365 per unit-day |
| Shortage penalty | 50% of price | p = 0.5 × sell_price per lost unit (labelled *penalty*, not "lost profit") |
| Ordering | $5 | K per order |
| Case pack | 1 | configurable rounding |

### 6.4 Outputs & metrics

Per item × policy × scenario: fill rate, cycle service level (share of days with no lost sales), stockout days, lost units, avg & peak on-hand, # orders, avg order qty, holding/ordering/shortage/total cost. Daily trace saved for plotting (`reports/inventory/trace.parquet`).

### 6.5 Sensitivity grid

L ∈ {2, 4, 7} × target CSL ∈ {0.90, 0.95, 0.98} × forecast source ∈ {selected model, snaive, "perfect foresight" (actuals — upper bound)}. The perfect-foresight row is the key context: it shows how much of the gap between P0 and the ideal the forecast actually closes.

---

## 7. Code architecture

```
OptiChain/
├── data/
│   ├── raw/                     # M5 CSVs (git-ignored, read-only)
│   └── processed/               # ca1_long.parquet (ignored), selected_series.parquet (committed, small)
├── notebooks/                   # narrative + figures only; logic lives in src/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_stationarity_seasonality.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_diagnostics_evaluation.ipynb
│   └── 05_inventory_simulation.ipynb
├── src/optichain/
│   ├── config.py                # load + validate config.yaml (pydantic)
│   ├── data_pipeline.py
│   ├── selection.py             # ADI/CV² classification
│   ├── eda.py
│   ├── diagnostics.py           # ADF/KPSS, STL, ACF, Ljung-Box, ARCH-LM
│   ├── forecasting.py           # model registry + fit/forecast
│   ├── evaluation.py            # metrics, rolling-origin backtest
│   ├── inventory.py             # policies + simulator
│   └── plots.py
├── scripts/                     # CLI entry points → reproducible full run
│   ├── build_dataset.py
│   ├── run_backtest.py
│   └── run_simulation.py
├── app/
│   ├── Home.py                  # Streamlit entry (Overview)
│   └── pages/ 2_Forecasting.py … 7_Export.py
├── models/                      # identification.json, per-series fitted params (json) + pickles
├── reports/ figures/ metrics/ forecasts/ inventory/
├── tests/                       # pytest, synthetic fixtures — no Kaggle data needed
├── config.yaml
├── pyproject.toml / requirements.txt
└── README.md
```

### 7.1 Key interfaces

```python
# data_pipeline.py
def build_long_table(raw_dir: Path, store_id: str, max_day: int) -> pd.DataFrame
def get_series(df, item_id, store_id, end_d: int) -> pd.Series   # date-indexed, launch-trimmed, ≤ end_d

# forecasting.py — every model implements the same protocol
class Forecaster(Protocol):
    name: str
    def fit(self, y: pd.Series) -> "Forecaster": ...
    def predict(self, h: int, alpha: list[float] = (0.2, 0.05)) -> pd.DataFrame  # yhat, lo/hi
    def residuals(self) -> pd.Series: ...
    def params(self) -> dict: ...                                                 # for json persistence
MODEL_REGISTRY: dict[str, Callable[..., Forecaster]]

def forecast(item_id: str, store_id: str, origin_d: int, model: str, h: int) -> pd.DataFrame
    # the plan's "deterministic inference function"

# evaluation.py
def rolling_origin(y_full, model_name, origins: list[int], h: int) -> pd.DataFrame  # long forecasts
def score(forecasts, y_train_by_origin) -> pd.DataFrame

# inventory.py
@dataclass(frozen=True)
class SimConfig: lead_time: int; review: int; cover_days: int; csl: float;
                 h_rate: float; p_rate: float; order_cost: float; pack: int; start_stock: float
class Policy(Protocol):
    def targets(self, t: int) -> tuple[float, float]  # (ROP_t, S_t), uses only info ≤ t
def simulate(demand: np.ndarray, policy: Policy, cfg: SimConfig) -> SimResult  # trace + metrics
```

The rolling-origin backtest is the only place that slices train/test — models never see an index past `origin_d`, which is what the leakage tests target.

### 7.2 `config.yaml` (draft)

```yaml
data:
  raw_dir: data/raw
  store_id: CA_1
  max_day: 1913            # raise to 1941 only for the final test run
series:
  items: []                # filled by selection notebook, then frozen
forecast:
  horizon: 28
  fit_window_days: 730
  origins: [1800, 1828, 1856, 1885]
  models: [naive, snaive, ma28, ets, arima, sarima, croston]
  sarima_grid: {p: [0,1,2], q: [0,1,2], P: [0,1], Q: [0,1], s: 7}
  interval_levels: [0.80, 0.95]
diagnostics:
  ljung_box_lags: [7, 14, 21]
  arch_lm_lag: 7
  alpha: 0.05
simulation_assumptions:     # HYPOTHETICAL — not Walmart data
  lead_time_days: 4
  review_days: 1
  cover_days: 7
  target_csl: 0.95
  holding_rate_annual: 0.25
  shortage_penalty_rate: 0.5
  order_cost: 5.0
  case_pack: 1
  sensitivity: {lead_time_days: [2,4,7], target_csl: [0.90,0.95,0.98]}
seed: 42
```

---

## 8. Dashboard design (Streamlit)

> **Implemented (before M4):** a 5-step story flow for non-technical readers: **Overview → Understand demand → Predict → Plan stock → Report**. It uses top navigation, a progress stepper and a "Next" button on every page, and the chosen product carries across pages.
>
> - **Plain language:** methods and policies use friendly names ("Same day last week", "Simple rule" vs "Forecast-driven plan"). Technical metrics sit in "For analysts" expanders.
> - **Look:** dark navy theme (`.streamlit/config.toml`, Inter font). The design system lives in `app/ui.py`: validated dark palette (blue = forecast/plan, orange = simple rule), CSS fade-up entrances, self-drawing sparklines and an animated Plotly stock replay. Motion is turned off under `prefers-reduced-motion`.
> - **Files:** pages are in `app/views/` and cached data access is in `app/data.py`.
>
> **M8 additions:**
> - A sixth step, **Model check** (`views/model_check.py`), gives a plain-language view of the M4/M6 diagnostics: STL breakdown, ACF/PACF, residual bias/pattern/spread, and the volatility verdict, with analyst tables in an expander.
> - The Report page adds an all-scenario robustness heatmap and updated limitations.
> - **[Decision]** The dashboard ranks methods by **RMSE**, which picks the same winners as RMSSE on all 8 products and is shown in units/day. It no longer uses MAE, which favoured "repeat yesterday" for slow movers.
> - Deployment: `src/` path fallback in `Home.py`, a README guide for Streamlit Community Cloud, and `tests/test_app.py` page smoke tests. The page table below is the original plan; the implemented pages cover the same content, regrouped into this flow.

**Serving model:** app loads `selected_series.parquet`, `forecasts.parquet`, `forecast_metrics.csv`, `diagnostics.csv`, and pre-rendered diagnostic data via `@st.cache_data`. No model fitting in the app (keeps Streamlit Cloud within memory/time limits). The **inventory page runs `simulate()` live** because it's milliseconds and lets users move sliders.

| Page | Contents | Data source |
|---|---|---|
| 1 Overview | store/item selector, history plot, variability table, demand class badge, data-quality summary | selected_series, variability table |
| 2 Forecasting | model + origin + horizon (7/14/28) selector, actual vs forecast with 80/95% bands | forecasts.parquet |
| 3 Model comparison | metric table, RMSSE heatmap item × model, skill-vs-snaive bars | forecast_metrics.csv |
| 4 Diagnostics | ADF/KPSS table, STL plot, ACF/PACF, residual panel, Ljung-Box, ARCH-LM (+GARCH if any) | diagnostics.csv, residual arrays |
| 5 Inventory control | sliders: L, CSL, cover days, costs, start stock → today's ROP, suggested order qty, low-stock alert; **"Hypothetical assumptions" banner** | live simulate() |
| 6 Policy comparison | P0 vs P1 (vs P2) metric cards, on-hand trace plot, cost breakdown, sensitivity heatmap | live + precomputed sensitivity |
| 7 Export | CSV download of forecasts, recommendations, metrics | in-memory |

Input validation: unknown item, missing artefact files → `st.error` with the script name that produces the file.

---

## 9. Testing (pytest, synthetic data)

`tests/conftest.py` builds a tiny fake M5 (3 items, 120 days, known prices/events) so CI never needs Kaggle data.

| Area | Test |
|---|---|
| Data | melt row count; unique keys; no negative sales; `max_day` cutoff respected; launch trimming |
| Leakage | `get_series(..., end_d=T).index.max() == date(T)`; backtest asserts model fit input ends at origin; price ffill never back-fills |
| Split | all train dates < all test dates per fold |
| Forecast | output length == h, finite, index starts at origin+1, non-negative after clipping |
| Metrics | hand-computed MAE/RMSE/RMSSE on toy arrays; zero-denominator RMSSE → NaN; sMAPE 0/0 → 0 |
| Inventory | on-hand never < 0 (lost-sales model); order placed at t arrives exactly at t+L; conservation: start + received − sold = end; with perfect foresight + z=0 and L=1, lost ≈ 0 |
| Reproducibility | two runs with same config → identical metric CSV |

---

## 10. Build order (milestones)

| # | Milestone | Done when | Plan WP |
|---|---|---|---|
| M0 | Repo setup | venv, requirements, `git init` in OptiChain, M5 downloaded to `data/raw` | WP1 |
| M1 ✅ | Data pipeline | `build_dataset.py` produces `ca1_long.parquet`; data tests pass; audit notebook | WP1–2 |
| M2 ✅ | Selection + EDA | 8 items frozen in config; EDA figures + variability table | WP2 |
| **M3 ✅** | **Vertical slice** | 1 item: naive/snaive → backtest → metrics → P0 vs P1 sim → 1 Streamlit page | — |
| M4 ✅ | Diagnostics | stationarity decisions json, STL, ACF/PACF for all 8 | WP3 |
| M5 ✅ | Full models | ETS, ARIMA, SARIMA, Croston across 8 items × 4 folds; comparison table | WP4 |
| M6 ✅ | Residuals + ARCH gate | diagnostics.csv; GARCH only where gate passes | WP5 |
| M7 ✅ | Inventory full | P0/P1/(P2) all items; sensitivity grid; perfect-foresight bound | WP6 |
| M8 ✅ | Dashboard | all 7 pages; deploy to Streamlit Community Cloud | WP7 |
| M9 ✅ | Final test + report | single run on d_1914–1941; report, slides, demo script | WP8 |
| (stretch) | LightGBM global model; daily state-updating forecasts in sim; scale to all CA_1 FOODS_3 | | |

---

## 10b. Final test outcome (M9)

- **[Decision] Pre-registration.** `reports/final_test/preregistration.json` (model per item = lowest backtest RMSSE, plus the inventory settings) is written before any data after d_1913 is loaded. The script never overwrites it. An integrity check confirms that history up to d_1913 equals the frozen dataset.
- **Forecasting:** mean RMSSE of the pre-registered models is 0.752 on 25 Apr – 22 May 2016, vs 0.769 in the backtest. SARIMA (0.756) and ETS (0.760) remain jointly best. Seasonal naive scores 1.077 and naive 1.362. 80% intervals cover 80–84%.
- **Inventory (28 days, hypothetical):** P1 costs $185 vs P0 $219 (6 vs 11 lost units), almost entirely from HOUSEHOLD_1_469. Excluding it, costs are equal ($125 vs $124) and P1 holds about 10% less stock. Ordering fees dominate over 28 days.
- **Deliverables:** `reports/OptiChain_Final_Report.pdf` (`scripts/build_report.py`, with tables generated from the result files), `reports/OptiChain_Presentation.pptx` (`scripts/build_deck.js`, 15 slides with speaker notes), and `docs/DEMO_SCRIPT.md`.

## 11. Tech stack

Python 3.12 (Streamlit Cloud–friendly; 3.13 also works), pandas, numpy, pyarrow, statsmodels, scipy, arch, matplotlib/plotly, streamlit, pydantic, pyyaml, pytest. Optional: lightgbm. Pin exact versions in `requirements.txt`.

---

## 12. Limitations to state up front (report + dashboard footer)

- Observed sales are censored demand; true demand during Walmart's own stockouts is unknown.
- Inventory, lead times and costs are hypothetical; results are a controlled experiment, not Walmart savings.
- US data; any Indian-retail framing is a separate hypothetical scenario.
- Price/event associations are not causal.
- 8 series from one store — findings are illustrative, not general.
