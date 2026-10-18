# OptiChain

Demand forecasting and stock planning on the **VN2 Inventory Planning Challenge** (2025): real weekly sales of
599 store-products in 67 stores of a retail chain, April 2021 – June 2024. OptiChain forecasts demand, turns the
forecast into weekly orders with a simulation-tuned newsvendor policy, and scores them with the competition's
official rules (lead time 2 weeks, €0.20 per unit left per week, €1 per lost sale).

| | VN2 score (weeks 3–8) |
|---|---|
| Official benchmark | €4,334.0 |
| VN2 winner | €3,763.2 |
| **OptiChain plan** | see `artifacts/summary.json` after running the notebook |

The plan's settings are tuned on past weeks only. It was built after the competition, so a leaderboard comparison
carries some hindsight (see the notebook, Section 10).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pip install -e .
```

The VN2 files ship in `data/raw/vn2/` (CC BY 4.0). If they are missing, the notebook downloads them from
<https://github.com/SupChains/VN2>.

## Run

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/build_notebook.py --run` (or run all cells of `notebooks/OptiChain_VN2_Pipeline.ipynb`) | every result, in `artifacts/` (~20 min) |
| 2 | `streamlit run app/Home.py` | the dashboard at http://localhost:8501 |
| – | `python -m pytest` | tests, including the simulator reproducing the official benchmark score |

### The notebook

`notebooks/OptiChain_VN2_Pipeline.ipynb` runs the whole pipeline with explanations, in 20 steps:

1. data collection and audit · 2. EDA (line graphs, scatter plots, histograms, QQ plots, descriptive statistics) ·
3. missing values (out-of-stock and pre-launch weeks) · 4. stationarity (ADF, KPSS) · 5. transformations (log,
differencing) · 6. seasonality (STL decomposition) · 7. ACF/PACF · 8. AR/MA/ARMA/ARIMA/SARIMA identification ·
9. lags · 10. AIC/BIC model selection (network total and every store-product) · 11. residual diagnostics
(Ljung-Box, Jarque-Bera, ARCH-LM) and GARCH(1,1) · 12. ten forecasting models (baselines, seasonal benchmark,
exponential smoothing, ARIMA, Croston SBA, TSB, global LightGBM, ensemble) · 13. 30-origin backtest and Ljung-Box on
the forecast errors · 14. uncertainty and calibration · 15. official simulator (verified) · 16. Monte Carlo
newsvendor tuned by historical replay · 17. final test on the 8 VN2 weeks · 18. next-order outlook · 19-20. save and
conclude.

`scripts/build_notebook.py` generates the notebook, so edit that file rather than the `.ipynb`.

### The dashboard

Seven pages: **Overview → Demand → Forecast → Model check → Stock planner → Order outlook → Method**. Model check shows the time-series diagnostics (ADF, STL, ACF/PACF, AIC/BIC, Ljung-Box, ARCH/GARCH) for the network or, live, for any store-product. Everything heavy is
precomputed by the notebook. The Stock planner replays the eight VN2 weeks live for any forecast method and
service level, and the Order outlook shows the next order and the stockout risk per store-product.

## Layout

```
src/optichain/   data.py (load, clean, classify) · models.py (forecasters) · evaluation.py (backtest, metrics)
                 timeseries.py (ADF/KPSS, STL, ACF/PACF, AIC/BIC, Ljung-Box, ARCH/GARCH, per-row ARIMA)
                 inventory.py (official simulator, policies) · planning.py (rounds, live plan) · config.py
app/             Streamlit dashboard (Home.py, views/, ui.py design system, data.py cached loaders)
notebooks/       the pipeline notebook          scripts/   notebook builder
data/raw/vn2/    official VN2 files             artifacts/ results read by the dashboard
archive/m5/      the earlier Walmart M5 version of the project
```

## Data

Vandeput, N. (2025). *VN2 Inventory Planning Challenge.* Organised by Nicolas Vandeput (SupChains). Sponsored by
Flieber, Daybreak and SupChains. <https://supchains.com/vn2-inventory-competition/> · Data licence CC BY 4.0.
