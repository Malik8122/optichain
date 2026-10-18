"""Identification for every item at every forecast origin (training data only).

Outputs: models/identification.json (item -> origin -> record, consumed by M5),
         reports/metrics/identification.csv, reports/figures/diagnostics/<item>/*.png (first origin)
Usage:   python scripts/run_diagnostics.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optichain import diagnostics as D
from optichain import plots as P
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import get_series


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main() -> None:
    cfg = load_config()
    P.apply_style()
    dg = cfg.diagnostics
    window = cfg.forecast.fit_window_days
    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    classes = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "ca_1_demand_classes.csv").set_index("item_id")
    figs = PROJECT_ROOT / "reports" / "figures" / "diagnostics"

    ident, rows = {}, []
    first = min(cfg.forecast.origins)
    for item in cfg.series.items:
        ident[item] = {}
        for o in cfg.forecast.origins:
            y = get_series(long, item, o, fill_closures=True).iloc[-window:]
            rec = D.identify(y, alpha=dg.alpha, lb_lags=tuple(dg.ljung_box_lags))
            ident[item][str(o)] = {k: _clean(v) for k, v in rec.items()}
            rows.append({"item_id": item, "origin_d": o, "demand_class": classes.loc[item, "demand_class"], **rec})
            if o == first:
                label = f"{item} ({classes.loc[item, 'demand_class']})"
                P.save(D.fig_stationarity(y, rec, f"{label} — level and spread"), figs / item / "stationarity.png")
                P.save(D.fig_stl(y, rec, label), figs / item / "stl.png")
                P.save(D.fig_acf_pacf(D.modelling_series(y, rec), rec, label), figs / item / "acf_pacf.png")
                plt.close("all")

    (PROJECT_ROOT / "models" / "identification.json").write_text(json.dumps(ident, indent=1))
    df = pd.DataFrame(rows)
    df.drop(columns=["acf_sig_lags", "pacf_sig_lags", "seasonal_acf"]).to_csv(PROJECT_ROOT / "reports" / "metrics" / "identification.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["demand_class", "zero_share", "var_level_corr", "transform", "level_adf_p", "level_kpss_p", "level_verdict",
            "seasonal_strength", "trend_strength", "D", "pre_d_verdict", "d", "flag", "p_hint", "q_hint", "P_hint",
            "Q_hint", "lb_p_7", "white_noise", "d_candidates", "D_candidates"]
    print(f"Identification at d_{first} (window {window} days):")
    print(df[df["origin_d"] == first].set_index("item_id")[cols].round(3).to_string())
    stab = df.groupby("item_id")[["transform", "d", "D"]].nunique().max(axis=1)
    print("\nDecisions identical across all origins:", (stab == 1).sum(), "of", len(stab), "items")
    if (stab > 1).any():
        print(df[df["item_id"].isin(stab[stab > 1].index)][["item_id", "origin_d", "transform", "d", "D", "seasonal_strength"]].to_string(index=False))


if __name__ == "__main__":
    main()
