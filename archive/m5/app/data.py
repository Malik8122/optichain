"""Cached data access shared by all dashboard pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import get_series
from optichain.inventory import SimConfig
from optichain.simulation import run_item

REQUIRED = {
    "data/processed/selected_series.parquet": "scripts/select_products.py",
    "reports/metrics/ca_1_demand_classes.csv": "scripts/select_products.py",
    "reports/forecasts/forecasts.parquet": "scripts/run_backtest.py",
    "reports/forecasts/residual_quantiles.parquet": "scripts/run_backtest.py",
    "reports/metrics/forecast_metrics.csv": "scripts/run_backtest.py",
    "reports/metrics/forecast_summary.csv": "scripts/run_backtest.py",
    "reports/inventory/sensitivity.csv": "scripts/run_simulation.py",
    "reports/inventory/policy_metrics.csv": "scripts/run_simulation.py",
    "reports/inventory/frontier.csv": "scripts/run_simulation.py",
    "models/identification.json": "scripts/run_diagnostics.py",
    "reports/metrics/residual_diagnostics.csv": "scripts/run_residuals.py",
    "reports/metrics/garch_decision.csv": "scripts/run_residuals.py",
    "reports/metrics/residuals.parquet": "scripts/run_residuals.py",
}


@st.cache_resource
def cfg():
    return load_config()


def check_artifacts() -> None:
    missing = [(p, s) for p, s in REQUIRED.items() if not (PROJECT_ROOT / p).exists()]
    if missing:
        st.error("Some results haven't been generated yet. Run these commands, then refresh:\n\n"
                 + "\n".join(f"- `python {s}`  →  {p}" for p, s in missing))
        st.stop()


@st.cache_data
def tables() -> dict[str, pd.DataFrame]:
    r = lambda p: PROJECT_ROOT / p
    return {
        "long": pd.read_parquet(r("data/processed/selected_series.parquet")),
        "classes": pd.read_csv(r("reports/metrics/ca_1_demand_classes.csv")).set_index("item_id"),
        "fc": pd.read_parquet(r("reports/forecasts/forecasts.parquet")),
        "rq": pd.read_parquet(r("reports/forecasts/residual_quantiles.parquet")),
        "metrics": pd.read_csv(r("reports/metrics/forecast_metrics.csv")),
        "summary": pd.read_csv(r("reports/metrics/forecast_summary.csv")),
        "sensitivity": pd.read_csv(r("reports/inventory/sensitivity.csv")),
        "policy": pd.read_csv(r("reports/inventory/policy_metrics.csv")),
        "frontier": pd.read_csv(r("reports/inventory/frontier.csv")),
        "ident": pd.read_csv(r("reports/metrics/identification.csv")),
        "resdiag": pd.read_csv(r("reports/metrics/residual_diagnostics.csv")),
        "garch": pd.read_csv(r("reports/metrics/garch_decision.csv")),
        "resid": pd.read_parquet(r("reports/metrics/residuals.parquet")),
        "specs": pd.read_csv(r("reports/metrics/model_specs.csv")),
    }


@st.cache_data
def series(item: str) -> pd.Series:
    return get_series(tables()["long"], item, cfg().data.max_day)


@st.cache_data
def series_model(item: str) -> pd.Series:
    """Series used for modelling/simulation: store-closure days filled (see get_series)."""
    return get_series(tables()["long"], item, cfg().data.max_day, fill_closures=True)


@st.cache_data
def price(item: str) -> pd.Series:
    long = tables()["long"]
    return long[long["item_id"] == item].set_index("d")["sell_price"]


def demand_class(item: str) -> str:
    return tables()["classes"].loc[item, "demand_class"]


def item_label(item: str) -> str:
    return f"{ui.item_short(item)}  ·  {ui.CLASS_NAMES[demand_class(item)]}"


def current_item() -> str:
    items = cfg().series.items
    if st.session_state.get("item") not in items:
        st.session_state["item"] = items[0]
    return st.session_state["item"]


def product_picker(page_key: str) -> str:
    """Product selector whose choice follows the user across pages."""
    items = cfg().series.items
    cur = current_item()
    choice = st.selectbox("Product", items, index=items.index(cur), key=f"pick_{page_key}",
                          format_func=item_label, label_visibility="collapsed")
    st.session_state["item"] = choice
    return choice


@st.cache_data(show_spinner="Simulating the stock plan…")
def simulate(item: str, lead_time: int, cover_days: int, csl: float, holding: float, shortage: float,
             order_cost: float) -> dict:
    c = cfg()
    t = tables()
    sim_cfg = SimConfig(lead_time=lead_time, cover_days=cover_days, csl=csl, holding_rate_annual=holding,
                        shortage_penalty_rate=shortage, order_cost=order_cost,
                        case_pack=c.simulation_assumptions.case_pack)
    fc = t["fc"][t["fc"]["item_id"] == item]
    met = t["metrics"][t["metrics"]["item_id"] == item]
    rq = t["rq"][t["rq"]["item_id"] == item]
    return run_item(item, series_model(item), price(item), fc, met, c.forecast.origins, c.forecast.horizon,
                    c.data.max_day, sim_cfg, c.simulation, rq)


def lookahead_limit() -> int:
    """Max lead time + cover days the stored forecasts can support from any day."""
    o = cfg().forecast.origins
    return int(tables()["fc"]["h"].max()) - max(b - a for a, b in zip(o, o[1:]))
