"""Rolling-origin backtest for every configured item x model x origin (parallel, cached).

Forecasts are produced for horizon + max lead time + cover days so the inventory simulator
can look ahead from any day in a fold; accuracy metrics use only the first `horizon` steps.
ARIMA/SARIMA read models/identification.json (run scripts/run_diagnostics.py first).

Outputs: reports/forecasts/forecasts.parquet, reports/metrics/forecast_metrics.csv,
         reports/metrics/forecast_summary.csv, reports/metrics/model_specs.csv,
         reports/metrics/aic_candidates.csv, models/backtest_params.json
Usage:   python scripts/run_backtest.py [--workers N] [--models ets,sarima]
"""
import argparse
import json
import time

import pandas as pd

from optichain.backtest import Task, run_tasks
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import get_series
from optichain.evaluation import score, summarise
from optichain.forecasting import MODEL_REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--models", default=None, help="comma-separated subset (default: all configured)")
    args = ap.parse_args()

    cfg = load_config()
    fc_cfg, sa = cfg.forecast, cfg.simulation_assumptions
    wanted = args.models.split(",") if args.models else fc_cfg.models
    models = [m for m in wanted if m in MODEL_REGISTRY]
    h_ext = fc_cfg.horizon + int(max(sa.sensitivity["lead_time_days"] + [sa.lead_time_days])) + sa.cover_days
    levels = tuple(fc_cfg.interval_levels)
    grid = {k: v for k, v in fc_cfg.sarima_grid.items()}

    ident_path = PROJECT_ROOT / "models" / "identification.json"
    ident = json.loads(ident_path.read_text()) if ident_path.exists() else {}
    if {"arima", "sarima"} & set(models) and not ident:
        raise SystemExit("models/identification.json missing: run scripts/run_diagnostics.py first")

    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    tasks = []
    for item in cfg.series.items:
        y = get_series(long, item, cfg.data.max_day, fill_closures=True)
        for m in models:
            for o in fc_cfg.origins:
                ctx = {}
                if m in ("arima", "sarima"):
                    rec = ident[item][str(o)]
                    ctx = {"ident": {k: rec[k] for k in ("transform", "d_candidates", "D_candidates")}, "grid": grid}
                tasks.append(Task(item, m, o, y, h_ext, fc_cfg.fit_window_days, levels, ctx))

    t0 = time.time()
    fc, params, timing = run_tasks(tasks, PROJECT_ROOT / "models" / "cache", workers=args.workers)
    metrics = score(fc, h_eval=fc_cfg.horizon, levels=levels)
    summary = summarise(metrics)

    out_f, out_m = PROJECT_ROOT / "reports" / "forecasts", PROJECT_ROOT / "reports" / "metrics"
    out_f.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_f / "forecasts.parquet", index=False)
    metrics.to_csv(out_m / "forecast_metrics.csv", index=False)
    summary.to_csv(out_m / "forecast_summary.csv", index=False)
    timing.to_csv(out_m / "fit_timing.csv", index=False)

    specs, cands = [], []
    for key, plist in params.items():
        item, model = key.split("/")
        for p in plist:
            spec = p.get("spec") or p.get("form") or (f"alpha={p['alpha']}" if "alpha" in p else model)
            specs.append({"item_id": item, "model": model, "origin_d": p["origin_d"], "spec": spec,
                          "transform": p.get("transform", ""), "aic": p.get("aic", p.get("aicc")),
                          "sigma_resid": p["sigma"], "combo_scores": json.dumps(p.get("combo_scores", []))})
            for c in p.get("candidates", []):
                cands.append({"item_id": item, "model": model, "origin_d": p["origin_d"], **c})
    pd.DataFrame(specs).to_csv(out_m / "model_specs.csv", index=False)
    rq = [{"item_id": k.split("/")[0], "model": k.split("/")[1], "origin_d": p["origin_d"], **row}
          for k, plist in params.items() for p in plist for row in p.get("resid_sum_q", [])]
    pd.DataFrame(rq).to_parquet(out_f / "residual_quantiles.parquet", index=False)
    pd.DataFrame(cands).to_csv(out_m / "aic_candidates.csv", index=False)
    slim = {k: [{kk: vv for kk, vv in p.items() if kk not in ("candidates", "resid_sum_q")} for p in v]
            for k, v in params.items()}
    (PROJECT_ROOT / "models" / "backtest_params.json").write_text(json.dumps(slim, indent=1, default=float))

    print(f"\nDone in {time.time() - t0:.0f}s. {len(cfg.series.items)} items x {len(models)} models x "
          f"{len(fc_cfg.origins)} origins (forecast length {h_ext}, scored on {fc_cfg.horizon})")
    pd.set_option("display.width", 220)
    piv = summary.pivot(index="item_id", columns="model", values="rmsse").loc[cfg.series.items]
    print("\nMean RMSSE over origins (lower is better):")
    print(piv[[m for m in fc_cfg.models if m in piv.columns]].round(3))
    print("\nBest model per item:", piv.idxmin(axis=1).to_dict())


if __name__ == "__main__":
    main()
