# OptiChain

Demand forecasting for supply-chain inventory optimisation on the Walmart M5 dataset (store CA_1).
Statistical forecasts feed a **hypothetical** inventory simulation that compares a baseline reorder rule
with a forecast-driven one. See [DESIGN.md](DESIGN.md) for all decisions and milestones.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pip install -e .
```

Download the M5 data from <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data> (accept the competition rules first).
Put `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` in `data/raw/`.

## Run order

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/build_dataset.py` | `data/processed/ca_1_long.parquet`, data audit |
| 2 | `python scripts/select_products.py` | 8 items frozen in `config.yaml`, `selected_series.parquet` |
| 3 | `python scripts/run_eda.py` | `reports/figures/eda/`, EDA tables |
| 4 | `python scripts/run_diagnostics.py` | `models/identification.json`, stationarity / STL / ACF figures |
| 5 | `python scripts/run_backtest.py` | forecasts for 7 models, accuracy metrics, AIC tables (parallel; fits cached in `models/cache/`, ~4 min first run) |
| 6 | `python scripts/run_residuals.py` | residual diagnostics, ARCH-LM, gated GARCH evaluation |
| 7 | `python scripts/run_simulation.py` | `reports/inventory/` policy metrics, traces, sensitivity grid |
| 8 | `streamlit run app/Home.py` | dashboard (see below) |
| 9 | `python scripts/run_final_test.py` | **one-time** pre-registered test on d_1914–d_1941 (`reports/final_test/`) |
| 10 | `python scripts/make_report_figures.py` then `python scripts/build_report.py --author "Name" --course "Course"` | `reports/OptiChain_Final_Report.pdf` |
| 11 | `npm install pptxgenjs` then `node scripts/build_deck.js "Name"` | `reports/OptiChain_Presentation.pptx` (speaker notes = talk track) |

Deliverables: the final report (PDF), the slides (PPTX), and the demo script in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

Tests use synthetic data and do not need the Kaggle files: `python -m pytest`.

## Dashboard

`streamlit run app/Home.py` (run from the project root so `.streamlit/config.toml` is picked up).
It is a six-step flow: **Overview → Understand demand → Predict → Model check → Plan stock → Report**.
The app reads only precomputed artefacts and runs the inventory simulator live.

## Deploying to Streamlit Community Cloud

The app is ready to deploy. Nothing is fitted at runtime, and the artefacts total about 12 MB.

1. Run the full pipeline (steps 1–7 above), so these files exist. They are **not** git-ignored and must be committed:
   - `data/processed/selected_series.parquet`
   - `reports/forecasts/`, `reports/metrics/`, `reports/inventory/`
   - `models/identification.json`
   - `.streamlit/config.toml`

   `data/raw/` (Kaggle data), `ca_1_long.parquet` and `models/cache/` stay out of git.
2. Push the repository to GitHub.
3. At <https://share.streamlit.io>, choose **Create app** and pick the repository and branch. Set the **main file path** to `app/Home.py`, and select Python 3.12 or 3.13 under *Advanced settings*.
4. Cloud installs `requirements.txt`. The entry point adds `src/` to the import path, so the `optichain` package does not need to be pip-installed there.
5. Check the [Kaggle M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules) before making the app public. It ships a small derived extract (8 products), not the raw files.

Known quirk: opening a deep link (for example `/report`) as the very first request after the server starts can show a brief
"Page not found" notice before the correct page loads. This is Streamlit registering its pages on first run, and it is harmless.

Dashboard smoke tests: `python -m pytest tests/test_app.py` (renders every page; skipped if artefacts are missing).

## Important limitations

- Inventory levels, lead times and costs are **hypothetical assumptions** (`config.yaml → simulation_assumptions`). M5 has no inventory data.
- Observed sales are used as demand, but they are censored by real stockouts.
- Results come from 8 series in one US store and are illustrative.
