"""Inventory simulation: baseline vs forecast-driven policies over all fold windows.

Policies: P0 baseline (trailing mean/std), P1 forecast (normal safety stock), P2 empirical
(safety stock from empirical quantiles of summed forecast errors), Perfect foresight, and P1
with each model fixed. Simulated days: d_(first origin + 1) .. d_max_day. All costs, lead
times and stock levels are HYPOTHETICAL assumptions from config.yaml.

Outputs (reports/inventory/): policy_metrics.csv, trace.parquet, sensitivity.csv, frontier.csv,
                              model_choice.csv, sim_config.json
Usage:   python scripts/run_simulation.py   (run scripts/run_backtest.py first)
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import get_series
from optichain.inventory import SimConfig, config_dict
from optichain.simulation import run_item, sensitivity_grid, service_frontier

FRONTIER_CSLS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]


def _item_job(args):
    item, y, price, fc, met, q, cfg_d, sim, origins, horizon, max_day, grid = args
    base = SimConfig(**cfg_d)
    res = run_item(item, y, price, fc, met, origins, horizon, max_day, base, sim, q)
    sens = sensitivity_grid(item, y, price, fc, met, origins, horizon, max_day, base, sim, grid, q)
    front = service_frontier(item, y, price, fc, met, origins, horizon, max_day, base, sim, FRONTIER_CSLS, q)
    return res, sens, front


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    base_cfg = SimConfig.from_assumptions(cfg.simulation_assumptions)
    sim = cfg.simulation
    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    fc = pd.read_parquet(PROJECT_ROOT / "reports" / "forecasts" / "forecasts.parquet")
    rq = pd.read_parquet(PROJECT_ROOT / "reports" / "forecasts" / "residual_quantiles.parquet")
    metrics = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "forecast_metrics.csv")
    cfg_d = {k: v for k, v in config_dict(base_cfg).items() if k != "z"}

    jobs = []
    for item in cfg.series.items:
        y = get_series(long, item, cfg.data.max_day, fill_closures=True)
        price = long[long["item_id"] == item].set_index("d")["sell_price"]
        jobs.append((item, y, price, fc[fc["item_id"] == item], metrics[metrics["item_id"] == item],
                     rq[rq["item_id"] == item], cfg_d, sim, cfg.forecast.origins, cfg.forecast.horizon,
                     cfg.data.max_day, cfg.simulation_assumptions.sensitivity))
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_item_job, jobs))

    pm = pd.concat([r["metrics"] for r, _, _ in out], ignore_index=True)
    trace = pd.concat([r["trace"] for r, _, _ in out], ignore_index=True)
    mc = pd.concat([r["model_choice"].assign(item_id=j[0]) for (r, _, _), j in zip(out, jobs)], ignore_index=True)
    sens = pd.concat([s for _, s, _ in out], ignore_index=True)
    front = pd.concat([f for _, _, f in out], ignore_index=True)

    o = PROJECT_ROOT / "reports" / "inventory"
    o.mkdir(parents=True, exist_ok=True)
    pm.to_csv(o / "policy_metrics.csv", index=False)
    trace.to_parquet(o / "trace.parquet", index=False)
    sens.to_csv(o / "sensitivity.csv", index=False)
    front.to_csv(o / "frontier.csv", index=False)
    mc.to_csv(o / "model_choice.csv", index=False)
    (o / "sim_config.json").write_text(json.dumps({"HYPOTHETICAL": True, **config_dict(base_cfg), **sim.model_dump()},
                                                  indent=1))

    pd.set_option("display.width", 220)
    main_p = ["P0 baseline", "P1 forecast", "P2 empirical", "Perfect foresight"]
    print(f"Done in {time.time() - t0:.0f}s. L={base_cfg.lead_time}, CSL={base_cfg.csl} (HYPOTHETICAL costs)\n")
    for col in ["total_cost", "fill_rate", "avg_on_hand", "lost_units"]:
        print(col)
        print(pm.pivot(index="item_id", columns="policy", values=col).loc[cfg.series.items, main_p].round(3))
    tot = pm[pm["policy"].isin(main_p)].groupby("policy")[["total_cost", "lost_units", "avg_on_hand"]].sum()
    print("\nTotals:\n", tot.loc[main_p].round(1))


if __name__ == "__main__":
    main()
