"""Residual diagnostics (ETS and SARIMA, every item and origin) and the gated GARCH evaluation.

Outputs: reports/metrics/residual_diagnostics.csv, reports/metrics/garch_evaluation.csv,
         reports/metrics/garch_decision.csv, reports/figures/residuals/<item>_<model>.png
Usage:   python scripts/run_residuals.py   (after run_backtest.py)
"""
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optichain import plots as P
from optichain import residual_checks as R
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import d_to_date, get_series

FAMILIES = ("ets", "sarima")
LEVELS = (80, 95)


def main() -> None:
    cfg = load_config()
    P.apply_style()
    dg, win, H = cfg.diagnostics, cfg.forecast.fit_window_days, cfg.forecast.horizon
    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    params = json.loads((PROJECT_ROOT / "models" / "backtest_params.json").read_text())
    fc = pd.read_parquet(PROJECT_ROOT / "reports" / "forecasts" / "forecasts.parquet")
    summary = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "forecast_summary.csv")
    best = (summary[summary["model"].isin(FAMILIES)].sort_values("rmsse").groupby("item_id").first()["model"])

    jobs = []
    for item in cfg.series.items:
        y = get_series(long, item, cfg.data.max_day, fill_closures=True)
        for fam in FAMILIES:
            for p in params[f"{item}/{fam}"]:
                o = p["origin_d"]
                jobs.append({"item": item, "family": fam, "origin": o, "params": p,
                             "y_train": y.loc[:d_to_date(o)].iloc[-win:], "lags": tuple(dg.ljung_box_lags),
                             "arch_lag": dg.arch_lm_lag, "alpha": dg.alpha})
    print(f"Refitting {len(jobs)} models for residual diagnostics…")
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(R.refit_and_test, jobs))

    diag = pd.DataFrame([o["rec"] for o in out])
    diag["selected"] = diag.apply(lambda r: best[r["item_id"]] == r["model"], axis=1)
    diag["garch_eligible"] = diag["arch_effects"] & (diag["zero_share"] <= R.GARCH_MAX_ZERO_SHARE)
    resid = {(o["rec"]["item_id"], o["rec"]["model"], o["rec"]["origin_d"]): o["resid"] for o in out}

    # ---------------------------------------------------------------- GARCH (gated)
    rows = []
    for _, r in diag[diag["garch_eligible"]].iterrows():
        key = (r["item_id"], r["model"], r["origin_d"])
        res, info = R.fit_garch(resid[key])
        f = fc[(fc["item_id"] == r["item_id"]) & (fc["model"] == r["model"]) & (fc["origin_d"] == r["origin_d"])
               & (fc["h"] <= H)].sort_values("h")
        ratio = R.garch_variance_ratio(res, info, H, resid[key])
        g = R.scale_intervals(f, ratio, LEVELS)
        row = {"item_id": r["item_id"], "model": r["model"], "origin_d": r["origin_d"], **info,
               "var_ratio_h1": float(ratio[0]), "var_ratio_h28": float(ratio[-1])}
        for lv in LEVELS:
            y_true = f["y_true"].to_numpy()
            for tag, frame in (("const", f), ("garch", g)):
                lo, hi = frame[f"lo{lv}"].to_numpy(), frame[f"hi{lv}"].to_numpy()
                row[f"cov{lv}_{tag}"] = float(np.mean((y_true >= lo) & (y_true <= hi)))
                row[f"score{lv}_{tag}"] = R.interval_score(y_true, lo, hi, lv / 100)
        rows.append(row)
    garch = pd.DataFrame(rows)

    decision = []
    for (item, model), g in diag.groupby(["item_id", "model"]):
        e = garch[(garch["item_id"] == item) & (garch["model"] == model)] if len(garch) else pd.DataFrame()
        d = {"item_id": item, "model": model, "selected": bool(g["selected"].iloc[0]),
             "origins_with_arch_effects": int(g["arch_effects"].sum()), "origins_garch_eligible": int(g["garch_eligible"].sum())}
        if len(e):
            better = sum(e[f"score{lv}_garch"].mean() < e[f"score{lv}_const"].mean() for lv in LEVELS)
            d |= {"garch_valid_fits": int(e["valid"].sum()),
                  "score80_const": e["score80_const"].mean(), "score80_garch": e["score80_garch"].mean(),
                  "score95_const": e["score95_const"].mean(), "score95_garch": e["score95_garch"].mean(),
                  "use_garch": bool(better == len(LEVELS) and e["valid"].all())}
            d["reason"] = "GARCH improves interval scores" if d["use_garch"] else "GARCH does not improve held-out interval scores"
            # Stricter recommendation (added after inspecting the fits; reported alongside the
            # pre-specified rule): ARCH coefficient significant at most origins AND >= 5% gain.
            gain = np.mean([1 - d[f"score{lv}_garch"] / d[f"score{lv}_const"] for lv in LEVELS])
            sig = float((e["alpha_p"] < 0.05).mean())
            d |= {"score_gain_pct": 100 * gain, "alpha_significant_share": sig,
                  "recommend_garch": bool(d["use_garch"] and sig >= 0.5 and gain >= 0.05)}
        else:
            d |= {"use_garch": False, "recommend_garch": False,
                  "reason": "no ARCH effects" if d["origins_with_arch_effects"] == 0 else "ARCH effects but zero-heavy (gate)"}
        decision.append(d)
    decision = pd.DataFrame(decision)

    m = PROJECT_ROOT / "reports" / "metrics"
    diag.to_csv(m / "residual_diagnostics.csv", index=False)
    garch.to_csv(m / "garch_evaluation.csv", index=False)
    decision.to_csv(m / "garch_decision.csv", index=False)
    pd.concat([s.dropna().rename("resid").rename_axis("date").reset_index().assign(item_id=k[0], model=k[1], origin_d=k[2])
               for k, s in resid.items()], ignore_index=True).to_parquet(m / "residuals.parquet", index=False)

    figs = PROJECT_ROOT / "reports" / "figures" / "residuals"
    last = max(cfg.forecast.origins)
    for item in cfg.series.items:
        for fam in FAMILIES:
            r = diag[(diag["item_id"] == item) & (diag["model"] == fam) & (diag["origin_d"] == last)].iloc[0]
            title = (f"{item} · {r['spec']} residuals (fit at d_{last}) · Ljung-Box p(7)={r['lb_p_7']:.3f}, "
                     f"ARCH-LM p={r['arch_lm_p']:.3f}")
            P.save(R.fig_residuals(resid[(item, fam, last)], title), figs / f"{item}_{fam}.png")
            plt.close("all")

    pd.set_option("display.width", 250)
    sel = diag[diag["selected"]]
    cols = ["model", "spec", "resid_mean", "mean_p", "lb_p_7", "lb_p_14", "lb_p_21", "max_abs_acf", "jb_p", "excess_kurtosis",
            "arch_lm_p", "zero_share", "no_autocorrelation", "arch_effects", "garch_eligible"]
    print(f"\nSelected model per item, fit at d_{last}:")
    print(sel[sel["origin_d"] == last].set_index("item_id")[cols].round(3).to_string())
    print("\nAdequacy across all origins (selected model): no autocorrelation in",
          int(sel["no_autocorrelation"].sum()), "of", len(sel), "fits; ARCH effects in", int(sel["arch_effects"].sum()))
    print("\nGARCH decision:")
    print(decision.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
