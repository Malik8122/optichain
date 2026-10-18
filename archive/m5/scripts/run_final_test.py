"""ONE-TIME final test on the untouched window d_1914-d_1941 (2016-04-25 .. 2016-05-22).

Protocol:
1. Pre-register the model chosen for each item (best mean RMSSE in the 4-fold backtest) and the
   inventory settings in reports/final_test/preregistration.json BEFORE any test data is read.
   An existing pre-registration is never overwritten (use --force only to reproduce deliberately).
2. Load sales up to d_1941 and check that d_1..d_1913 equal the frozen dataset exactly.
3. Identify at origin d_1913, fit every model on the 730 days up to d_1913, and forecast 28 days.
4. Score the forecasts and replay the inventory policies on the same 28 days.

Outputs: reports/final_test/{preregistration.json, forecasts.parquet, metrics.csv, inventory.csv, trace.parquet}
Usage:   python scripts/run_final_test.py
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from optichain import diagnostics as D
from optichain.backtest import Task, run_tasks
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import build_long_table, d_to_date, get_series
from optichain.evaluation import score
from optichain.forecasting import MODEL_REGISTRY
from optichain.inventory import (BaselinePolicy, EmpiricalForecastPolicy, ForecastPolicy, SimConfig,
                                 perfect_foresight_forecasts, simulate)
from optichain.simulation import start_on_hand

FINAL_END = 1941


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run even if a pre-registration exists")
    args = ap.parse_args()

    cfg = load_config()
    origin, H = cfg.data.max_day, cfg.forecast.horizon
    assert origin + H == FINAL_END, "final test expects origin d_1913 and a 28-day horizon"
    out = PROJECT_ROOT / "reports" / "final_test"
    out.mkdir(parents=True, exist_ok=True)
    sa = cfg.simulation_assumptions
    sim_cfg = SimConfig.from_assumptions(sa)

    # ------------------------------------------------------------ 1. pre-registration
    prereg_path = out / "preregistration.json"
    if prereg_path.exists() and not args.force:
        prereg = json.loads(prereg_path.read_text())
        print(f"Using existing pre-registration from {prereg['created_utc']}")
    else:
        summary = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "forecast_summary.csv")
        best = summary.loc[summary.groupby("item_id")["rmsse"].idxmin()].set_index("item_id")
        prereg = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_window": {"first_d": origin + 1, "last_d": FINAL_END,
                            "dates": [str(d_to_date(origin + 1).date()), str(d_to_date(FINAL_END).date())]},
            "fit_origin_d": origin,
            "selected_model": best["model"].to_dict(),
            "backtest_rmsse_of_selected": best["rmsse"].round(4).to_dict(),
            "inventory": {"lead_time": sim_cfg.lead_time, "cover_days": sim_cfg.cover_days, "csl": sim_cfg.csl,
                          "warmup_days": 0, "policies": ["P0 baseline", "P1 forecast", "P2 empirical", "Perfect foresight"]},
            "note": "Written before any sales after d_1913 were loaded. Selection = lowest mean RMSSE over the 4 backtest folds.",
        }
        prereg_path.write_text(json.dumps(prereg, indent=1))
        print("Pre-registered:", prereg["selected_model"])

    # ------------------------------------------------------------ 2. data + integrity check
    long = build_long_table(cfg.data.raw_dir, cfg.data.store_id, FINAL_END, cfg.data.sales_file)
    long = long[long["item_id"].isin(cfg.series.items)].copy()
    long["item_id"] = long["item_id"].cat.remove_unused_categories()
    frozen = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    a = long[long["d"] <= origin].sort_values(["item_id", "d"])["sales"].to_numpy()
    b = frozen.sort_values(["item_id", "d"])["sales"].to_numpy()
    if not np.array_equal(a, b):
        raise SystemExit("Integrity check failed: d_1..d_1913 differ from the frozen dataset")
    long.to_parquet(cfg.data.processed_dir / "final_test_series.parquet", index=False)
    print("Integrity check passed: history up to d_1913 is identical to the frozen dataset")

    # ------------------------------------------------------------ 3. identify + fit all models at d_1913
    win = cfg.forecast.fit_window_days
    h_ext = H + sim_cfg.lead_time + sim_cfg.cover_days
    levels = tuple(cfg.forecast.interval_levels)
    grid = dict(cfg.forecast.sarima_grid)
    tasks, ys = [], {}
    for item in cfg.series.items:
        y = get_series(long, item, FINAL_END, fill_closures=True)
        ys[item] = y
        rec = D.identify(y.loc[:d_to_date(origin)].iloc[-win:], alpha=cfg.diagnostics.alpha,
                         lb_lags=tuple(cfg.diagnostics.ljung_box_lags))
        for m in cfg.forecast.models:
            if m not in MODEL_REGISTRY:
                continue
            ctx = ({"ident": {k: rec[k] for k in ("transform", "d_candidates", "D_candidates")}, "grid": grid}
                   if m in ("arima", "sarima") else {})
            tasks.append(Task(item, m, origin, y, h_ext, win, levels, ctx))
    fc, params, _ = run_tasks(tasks, PROJECT_ROOT / "models" / "cache_final")
    fc.to_parquet(out / "forecasts.parquet", index=False)

    # ------------------------------------------------------------ 4a. forecast accuracy
    met = score(fc, h_eval=H, levels=levels)
    bt = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "forecast_summary.csv").set_index(["item_id", "model"])
    met["backtest_rmsse"] = [bt.loc[(i, m), "rmsse"] for i, m in zip(met["item_id"], met["model"])]
    met["selected"] = [prereg["selected_model"][i] == m for i, m in zip(met["item_id"], met["model"])]
    met["rank_in_test"] = met.groupby("item_id")["rmsse"].rank()
    met.to_csv(out / "metrics.csv", index=False)

    # ------------------------------------------------------------ 4b. inventory on the same 28 days
    rq = pd.DataFrame([{"item_id": k.split("/")[0], "model": k.split("/")[1], "origin_d": p["origin_d"], **row}
                       for k, plist in params.items() for p in plist for row in p.get("resid_sum_q", [])])
    inv, traces = [], []
    for item in cfg.series.items:
        y = ys[item]
        m = prereg["selected_model"][item]
        f = fc[(fc["item_id"] == item) & (fc["model"] == m)]
        q = rq[(rq["item_id"] == item) & (rq["model"] == m)]
        price = long[long["item_id"] == item].set_index("d")["sell_price"]
        days = list(range(origin + 1, FINAL_END + 1))
        demand = pd.Series([float(y.loc[d_to_date(d)]) for d in days], index=days)
        s0 = start_on_hand(y.loc[:d_to_date(origin)], origin + 1, sim_cfg, cfg.simulation.baseline_window)
        policies = [BaselinePolicy(y.loc[:d_to_date(origin)], [origin], sim_cfg, cfg.simulation.baseline_window),
                    ForecastPolicy(f, sim_cfg, "P1 forecast"),
                    EmpiricalForecastPolicy(f, q, sim_cfg, "P2 empirical"),
                    ForecastPolicy(perfect_foresight_forecasts(y, [origin], h_ext), sim_cfg, "Perfect foresight")]
        for pol in policies:
            r = simulate(demand, price, pol, sim_cfg, s0, warmup_days=0)
            inv.append({"item_id": item, "policy": pol.name, "model": m, **r.metrics})
            traces.append(r.trace.assign(item_id=item))
    inv = pd.DataFrame(inv)
    inv.to_csv(out / "inventory.csv", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(out / "trace.parquet", index=False)

    # ------------------------------------------------------------ report
    pd.set_option("display.width", 220)
    sel = met[met["selected"]].set_index("item_id").loc[cfg.series.items]
    print(f"\nFINAL TEST d_{origin + 1}-d_{FINAL_END} ({prereg['test_window']['dates'][0]} .. {prereg['test_window']['dates'][1]})")
    print("\nPre-registered model per item:")
    print(sel[["model", "rmsse", "backtest_rmsse", "mae", "bias", "cov80", "cov95", "rank_in_test"]].round(3))
    piv = met.pivot(index="item_id", columns="model", values="rmsse").loc[cfg.series.items]
    print("\nTest RMSSE, every model:")
    print(piv[[m for m in cfg.forecast.models if m in piv.columns]].round(3))
    print("\nMean test RMSSE by model:", piv.mean().sort_values().round(3).to_dict())
    print("Mean test RMSSE of pre-registered models:", round(sel["rmsse"].mean(), 3),
          "| backtest:", round(sel["backtest_rmsse"].mean(), 3))
    tot = inv.groupby("policy")[["total_cost", "lost_units", "avg_on_hand", "sold_units", "demand_units"]].sum()
    tot["fill_rate"] = tot["sold_units"] / tot["demand_units"]
    print("\nInventory over the test window (HYPOTHETICAL costs):")
    print(tot.loc[prereg["inventory"]["policies"]].round(3))
    print(inv.pivot(index="item_id", columns="policy", values="total_cost").loc[cfg.series.items].round(2))


if __name__ == "__main__":
    main()
