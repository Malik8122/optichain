"""Figures for the final report and slides (reads saved results only).

Outputs: reports/figures/final/*.png
Usage:   python scripts/make_report_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from optichain import plots as P
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import d_to_date, get_series

R = PROJECT_ROOT / "reports"
OUT = R / "figures" / "final"
NAMES = {"naive": "Naive", "snaive": "Seasonal naive", "ma28": "28-day average", "ets": "ETS", "arima": "ARIMA",
         "sarima": "SARIMA", "croston": "Croston-SBA"}
POL = {"P0 baseline": ("P0 simple rule", P.SERIES[1], "s"), "P1 forecast": ("P1 forecast-driven", P.SERIES[0], "o"),
       "P2 empirical": ("P2 forecast + empirical SS", P.SERIES[2], "^")}


def model_comparison(summary, final):
    bt = summary.groupby("model")["rmsse"].mean()
    ft = final.groupby("model")["rmsse"].mean()
    order = bt.sort_values(ascending=False).index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(y + 0.2, bt[order], height=0.38, color=P.SERIES[0], label="Backtest (4 folds, Jan–Apr 2016)")
    ax.barh(y - 0.2, ft[order], height=0.38, color=P.SEQ_BLUE[2], label="Final test (25 Apr – 22 May 2016)")
    for i, m in enumerate(order):
        ax.text(bt[m] + 0.01, i + 0.2, f"{bt[m]:.3f}", va="center", fontsize=8, color=P.INK_2)
        ax.text(ft[m] + 0.01, i - 0.2, f"{ft[m]:.3f}", va="center", fontsize=8, color=P.INK_2)
    ax.set_yticks(y, [NAMES[m] for m in order])
    ax.axvline(1, color=P.MUTED, ls="--", lw=1)
    ax.set_xlabel("Mean RMSSE across 8 products (lower is better)")
    ax.set_title("Forecast accuracy by method")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.5)
    ax.legend(loc="upper right", fontsize=8)
    P.save(fig, OUT / "model_comparison.png")


def forecast_example(fc, long):
    item, origin = "FOODS_3_586", 1885
    y = get_series(long, item, 1913)
    hist = y.loc[d_to_date(origin - 34):d_to_date(origin)]
    f = fc[(fc.item_id == item) & (fc.origin_d == origin) & (fc.h <= 28)]
    ets = f[f.model == "ets"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(hist.index, hist.values, color=P.CONTEXT, lw=1.3, label="History")
    d = pd.to_datetime(ets["date"])
    ax.fill_between(d, ets["lo80"], ets["hi80"], color=P.SEQ_BLUE[0], lw=0, label="ETS 80% interval")
    ax.plot(d, ets["y_true"], color=P.INK, lw=1.3, marker="o", ms=3, label="Actual (held out)")
    ax.plot(d, ets["yhat"], color=P.SERIES[0], lw=2.2, label="ETS")
    s = f[f.model == "snaive"]
    ax.plot(pd.to_datetime(s["date"]), s["yhat"], color=P.MUTED, lw=1.3, ls=":", label="Seasonal naive")
    ax.axvline(d_to_date(origin), color=P.MUTED, ls="--", lw=1)
    ax.set_ylabel("Units per day"); ax.set_ylim(bottom=0)
    ax.set_title(f"{item}: 28-day forecast from {d_to_date(origin):%d %b %Y}")
    ax.legend(ncols=3, fontsize=8, loc="upper left")
    P.save(fig, OUT / "forecast_example.png")


def frontier(front, items):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for ax, item in zip(axes.flat, items):
        f = front[front.item_id == item]
        for pol, (lab, c, mk) in POL.items():
            g = f[f.policy == pol].sort_values("target_csl")
            ax.plot(g["avg_on_hand"], g["fill_rate"] * 100, color=c, marker=mk, ms=4, lw=1.5, label=lab)
        pf = f[f.policy == "Perfect foresight"].iloc[0]
        ax.scatter(pf["avg_on_hand"], 100, marker="*", s=90, color=P.MUTED, zorder=5, label="Perfect hindsight")
        ax.set_title(item, fontsize=9); ax.grid(axis="x")
    for ax in axes[1]:
        ax.set_xlabel("Average stock (units)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fill rate (%)")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Service vs stock as the target service level moves from 80% to 99% (L = 4 days)", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold")
    P.save(fig, OUT / "frontier.png")


def sensitivity(sens):
    agg = sens.groupby(["lead_time", "target_csl", "policy"])["total_cost"].sum().unstack("policy")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    cmap = LinearSegmentedColormap.from_list("div", [P.SERIES[0], "#f0efec", P.SERIES[7]])
    for ax, pol, title in [(axes[0], "P1 forecast", "P1 forecast-driven vs P0"), (axes[1], "P2 empirical", "P2 empirical SS vs P0")]:
        m = (agg[pol] / agg["P0 baseline"] - 1).unstack("target_csl")
        ax.imshow(m.values, cmap=cmap, norm=TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15), aspect="auto")
        ax.set_xticks(range(m.shape[1]), [f"{c:.0%}" for c in m.columns])
        ax.set_yticks(range(m.shape[0]), [f"{int(l)} days" for l in m.index])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:+.1%}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Total cost change, {title}", fontsize=9.5)
        ax.set_xlabel("Target service level"); ax.grid(False)
    axes[0].set_ylabel("Lead time")
    P.save(fig, OUT / "sensitivity.png")


def final_test(final, prereg_items):
    sel = final[final.selected].set_index("item_id").loc[prereg_items]
    x = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.bar(x - 0.2, sel["backtest_rmsse"], width=0.38, color=P.SERIES[0], label="Backtest (mean of 4 folds)")
    ax.bar(x + 0.2, sel["rmsse"], width=0.38, color=P.SEQ_BLUE[2], label="Final test")
    ax.set_xticks(x, list(sel.index), rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMSSE")
    ax.set_title(f"Pre-registered model per product: backtest vs final test "
                 f"(means {sel['backtest_rmsse'].mean():.3f} vs {sel['rmsse'].mean():.3f})")
    ax.legend(fontsize=8)
    P.save(fig, OUT / "final_test.png")


def main():
    P.apply_style()
    cfg = load_config()
    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    summary = pd.read_csv(R / "metrics" / "forecast_summary.csv")
    final = pd.read_csv(R / "final_test" / "metrics.csv")
    fc = pd.read_parquet(R / "forecasts" / "forecasts.parquet")
    model_comparison(summary, final)
    forecast_example(fc, long)
    frontier(pd.read_csv(R / "inventory" / "frontier.csv"), cfg.series.items)
    sensitivity(pd.read_csv(R / "inventory" / "sensitivity.csv"))
    final_test(final, cfg.series.items)
    plt.close("all")
    deck_data(summary, final)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")), "+ deck_data.json")


def deck_data(summary, final):
    """Numbers used by scripts/build_deck.js, so the slides always match the results."""
    import json
    bt = summary.groupby("model")["rmsse"].mean()
    ft = final.groupby("model")["rmsse"].mean()
    sel = final[final.selected]
    inv = pd.read_csv(R / "final_test" / "inventory.csv").groupby("policy")["total_cost"].sum()
    data = {"accuracy": {"order": list(bt.sort_values().index), "names": NAMES,
                         "backtest": bt.round(3).to_dict(), "final": ft.round(3).to_dict()},
            "final": {"rmsse_test": round(float(sel["rmsse"].mean()), 3), "rmsse_backtest": round(float(sel["backtest_rmsse"].mean()), 3),
                      "cost_p1": int(round(inv["P1 forecast"])), "cost_p0": int(round(inv["P0 baseline"]))}}
    (OUT / "deck_data.json").write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
