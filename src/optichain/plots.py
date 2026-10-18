"""The notebook's charts, kept out of the notebook so its cells show only the analysis.

One function per chart. Colours: blue = our model, orange = the comparison (benchmark), grey = context.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b8c0cc", "#0f172a", "#64748b"


def style() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 4), "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#eef1f6", "axes.axisbelow": True, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False,
    })
    pd.set_option("display.max_columns", 30, "display.width", 160, "display.float_format", "{:,.2f}".format)


def _done():
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------- 2 · EDA
def network_over_time(total: pd.Series, instock_share: pd.Series, week0) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(total.index, total.values, color=BLUE, lw=2)
    a1.fill_between(total.index, total.values, color=BLUE, alpha=.08)
    a1.axvline(week0, color=INK, lw=1, ls="--")
    a1.text(week0, total.max(), "  week 0", fontsize=9)
    a1.set(title="Total units sold per week, all 599 store-products", ylabel="units")
    a2.plot(instock_share.index, 100 * instock_share.values, color=ORANGE, lw=1.8)
    a2.set(title="Share of store-products with stock left at the end of the week", ylabel="%", ylim=(70, 100))
    _done()


def seasonal_factors(season: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(season.index, season.values, color=[BLUE if v >= 1 else GREY for v in season.values], width=.75)
    ax.axhline(1, color=INK, lw=.8)
    ax.axvspan(15.5, 23.5, color=ORANGE, alpha=.1, lw=0)  # competition weeks = ISO weeks 16-23
    ax.text(16, season.max() * .98, "competition weeks", color=ORANGE, fontsize=9)
    ax.set(title="Seasonal factor per ISO week (1 = average week)", xlabel="ISO week", ylabel="factor",
           ylim=(season.min() * .9, season.max() * 1.05))
    _done()


def sales_histograms(mean_week: pd.Series, zero_share: pd.Series) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.hist(np.log10(mean_week.clip(lower=.05)), bins=40, color=BLUE, edgecolor="white")
    a1.set(title="Mean units per week (log scale)", xlabel="units per week", ylabel="store-products")
    a1.set_xticks([-1, 0, 1, 2], ["0.1", "1", "10", "100"])
    a2.hist(100 * zero_share, bins=30, color=BLUE, edgecolor="white")
    a2.set(title="Share of weeks with zero sales", xlabel="%", ylabel="store-products")
    _done()


def pareto(cum_share: pd.Series) -> None:
    x = np.arange(1, len(cum_share) + 1) / len(cum_share) * 100
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, 100 * cum_share.values, color=BLUE, lw=2)
    for share in (0.2, 0.5):
        y = 100 * cum_share.iloc[int(share * len(cum_share)) - 1]
        ax.scatter(share * 100, y, color=BLUE, zorder=3)
        ax.annotate(f"top {share:.0%} of store-products = {y:.0f}% of units", (share * 100, y), xytext=(8, -14),
                    textcoords="offset points")
    ax.set(title="Cumulative share of units sold", xlabel="% of store-products (biggest first)", ylabel="% of units")
    _done()


def scatter_plots(mean_w: pd.Series, cv_w: pd.Series, total: np.ndarray) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.scatter(mean_w, cv_w, s=12, color=BLUE, alpha=.6)
    a1.set_xscale("log")
    a1.set(title="Volume vs variability, one dot per store-product", xlabel="mean units per week (log)",
           ylabel="coefficient of variation")
    a2.scatter(total[:-1], total[1:], s=14, color=BLUE, alpha=.7)
    lim = [total.min() * .95, total.max() * 1.02]
    a2.plot(lim, lim, color=GREY, lw=1, ls="--")
    r = np.corrcoef(total[:-1], total[1:])[0, 1]
    a2.set(title=f"Lag plot of the network total (correlation {r:.2f})", xlabel="units, week t-1", ylabel="units, week t")
    _done()


def qq_plots(series: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(series), figsize=(11, 3.8))
    for ax, (name, v) in zip(np.atleast_1d(axes), series.items()):
        (osm, osr), (slope, inter, r) = stats.probplot(v, dist="norm")
        ax.scatter(osm, osr, s=10, color=BLUE)
        ax.plot(osm, slope * osm + inter, color=ORANGE, lw=1.5)
        ax.set(title=f"{name}  (r = {r:.3f})", xlabel="normal quantiles", ylabel="data quantiles")
    _done()


def stockouts_and_groups(store_oos: pd.Series, dept_sales: pd.Series) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.bar(range(len(store_oos)), 100 * store_oos.values, color=BLUE, width=.8)
    a1.set(title="Out-of-stock rate per store (67 stores)", xlabel="stores, worst first", ylabel="% of weeks")
    a1.set_xticks([])
    a2.barh(dept_sales.index.astype(str), dept_sales.values, color=BLUE)
    a2.set(title="Units sold by department group", xlabel="units (history)")
    _done()


# ----------------------------------------------------------------- 3 · missing values
def launch_histogram(first_sale: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.hist(first_sale[first_sale > 0], bins=40, color=BLUE, edgecolor="white")
    ax.set(title="When products were launched (store-products not selling from week 1)",
           xlabel="weeks after 12 April 2021", ylabel="store-products")
    _done()


def stockout_example(sales: pd.Series, in_stock: pd.Series, week0, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(sales.index, sales.values, width=5, color=BLUE, label="weekly sales")
    for day in sales.index[~in_stock.to_numpy()]:
        ax.axvspan(day - pd.Timedelta(days=3.5), day + pd.Timedelta(days=3.5), color=ORANGE, alpha=.18, lw=0)
    ax.axvline(week0, color=INK, lw=1, ls="--")
    ax.set(title=f"{title}: orange bands = week ended out of stock (set aside)", ylabel="units")
    ax.legend(loc="upper left")
    _done()


def pattern_map(classes: pd.DataFrame, adi_cut: float, cv2_cut: float) -> None:
    colors = {"smooth": BLUE, "erratic": ORANGE, "intermittent": AQUA, "lumpy": "#6250d6"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pat, g in classes.groupby("pattern"):
        ax.scatter(g["adi"].clip(upper=30), g["cv2"].clip(upper=4), s=14, alpha=.7, color=colors[pat], label=pat)
    ax.axvline(adi_cut, color=INK, lw=.8, ls="--")
    ax.axhline(cv2_cut, color=INK, lw=.8, ls="--")
    ax.set_xscale("log")
    ax.set(xlabel="ADI: weeks between sales (log)", ylabel="CV²: variability of sale size",
           title="Demand patterns, one dot per store-product")
    ax.legend(title="pattern")
    _done()


def time_split(n_weeks: int, t0: int, spread: list[int], tune: list[int], dates) -> None:
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.barh(0, t0 + 1, color="#e2e8f0", height=.5)
    ax.barh(0, len(spread), left=spread[0], color=GREY, height=.5)
    ax.barh(0, len(tune) + 3, left=tune[0], color=BLUE, height=.5)
    ax.barh(0, 8, left=t0 + 1, color=ORANGE, height=.5)
    ax.text(4, 0, "history (training)", va="center", color=INK)
    ax.text(spread[0] + 1, .42, "spread", color=MUTED, fontsize=8)
    ax.text(tune[0] + 1, .42, "accuracy + policy tuning", color=BLUE, fontsize=8)
    ax.text(t0 + 1, .42, "final test", color=ORANGE, fontsize=8)
    ticks = [0, 52, 104, t0, n_weeks - 1]
    ax.set_xticks(ticks, [f"{dates[i]:%b %Y}" for i in ticks])
    ax.set(yticks=[], xlim=(0, n_weeks), title="How the weeks are used")
    ax.grid(False)
    _done()


# ----------------------------------------------------------------- 4-11 · time series
def transformations(tr: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, k in zip(axes, ["original", "log(1+y)", "1st difference"]):
        ax.plot(tr[k], color=BLUE, lw=1.4)
        ax.set_title(k)
    _done()


def decomposition(dec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, (k, c) in zip(axes, [("observed", INK), ("trend", BLUE), ("seasonal", ORANGE), ("remainder", GREY)]):
        ax.plot(dec.index, dec[k], color=c, lw=1.5)
        ax.set_ylabel(k)
    axes[0].set_title("STL decomposition of the network total (period 52 weeks)")
    _done()


def acf_pacf(ap: pd.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, col in zip(axes, ["acf", "pacf"]):
        ax.bar(ap.lag, ap[col], color=[BLUE if abs(x) > b else GREY for x, b in zip(ap[col], ap.band)], width=.6)
        ax.fill_between(ap.lag, -ap.band, ap.band, color=ORANGE, alpha=.12, lw=0)
        ax.axhline(0, color=INK, lw=.6)
        ax.set(title=f"{col.upper()}: {title}", xlabel="lag (weeks)")
    _done()


def residual_diagnostics(dates, resid: np.ndarray, resid_acf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].plot(dates, resid, color=BLUE, lw=1.2)
    axes[0].axhline(0, color=INK, lw=.6)
    axes[0].set_title("Residuals over time")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[1].bar(resid_acf.lag, resid_acf.acf, width=.6,
                color=[BLUE if abs(x) > b else GREY for x, b in zip(resid_acf.acf, resid_acf.band)])
    axes[1].fill_between(resid_acf.lag, -resid_acf.band, resid_acf.band, color=ORANGE, alpha=.12, lw=0)
    axes[1].set_title("ACF of the residuals")
    (osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
    axes[2].scatter(osm, osr, s=10, color=BLUE)
    axes[2].plot(osm, sl * osm + ic, color=ORANGE)
    axes[2].set_title("QQ plot of the residuals")
    _done()


def garch_volatility(dates, resid: np.ndarray, volatility: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(dates, np.abs(resid), color=GREY, lw=1, label="|residual|")
    ax.plot(dates, volatility, color=ORANGE, lw=2, label="GARCH volatility")
    ax.set(title="GARCH(1,1): the variance of the network total changes over time", ylabel="units")
    ax.legend()
    _done()


def holdout_forecast(history: pd.Series, fc_dates, mean, ci80, ci95, actual, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(history.index, history.values, color=INK, lw=1.5, label="history")
    ax.fill_between(fc_dates, ci95[:, 0], ci95[:, 1], color=BLUE, alpha=.1, lw=0, label="95% interval")
    ax.fill_between(fc_dates, ci80[:, 0], ci80[:, 1], color=BLUE, alpha=.18, lw=0, label="80% interval")
    ax.plot(fc_dates, mean, color=BLUE, lw=2.2, marker="o", ms=4, label=label)
    ax.plot(fc_dates, actual, color=INK, lw=1.5, ls=":", marker="o", ms=4, label="actual (unseen)")
    ax.set(title="Network total: SARIMA forecast of the 8 competition weeks", ylabel="units")
    ax.legend(ncol=3)
    _done()


def pvalue_histograms(lb_p: pd.Series, arch_p: pd.Series) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, v, title, xl in ((a1, lb_p, "Ljung-Box p-values (lag 8), per store-product", "p-value (right of the line: white noise)"),
                             (a2, arch_p, "ARCH-LM p-values, per store-product", "p-value (left of the line: ARCH effect)")):
        ax.hist(v.dropna(), bins=20, color=BLUE, edgecolor="white")
        ax.axvline(.05, color=ORANGE, lw=1.5)
        ax.set(title=title, xlabel=xl)
    _done()


# ----------------------------------------------------------------- 12-17 · forecasting and stock
def model_forecasts(history: pd.Series, actual: pd.Series, forecasts: dict[str, np.ndarray], labels: dict, title: str) -> None:
    highlight = {"lightgbm": BLUE, "seasonal_ma": ORANGE, "ensemble": AQUA}
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history.index, history.values, color=INK, lw=1.6, marker="o", ms=3, label="sales (history)")
    ax.plot(actual.index, actual.values, color=INK, lw=1.6, ls=":", marker="o", ms=4, label="actual (unseen)")
    for m, f in forecasts.items():
        hi = m in highlight
        ax.plot(actual.index, f, lw=2.4 if hi else 1, color=highlight.get(m, GREY), alpha=1 if hi else .8,
                label=labels[m] if hi else None)
    ax.set(title=f"{title} (grey: the other models)", ylabel="units")
    ax.legend(ncol=2)
    _done()


def leaderboard(lb: pd.DataFrame) -> None:
    d = lb.sort_values("Score %", ascending=False)
    best = lb.sort_values("Score %").iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(d["label"], d["Score %"], color=[BLUE if m == best else (ORANGE if m == "seasonal_ma" else GREY) for m in d["model"]])
    for y, v in enumerate(d["Score %"]):
        ax.text(v + .4, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set(title="Forecast error by model (Score % = WAPE + |bias|, lower is better)", xlabel="Score %",
           xlim=(0, d["Score %"].max() * 1.12))
    _done()


def importance(imp: pd.Series, top: int = 15) -> None:
    d = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.barh(d.index, 100 * d.values, color=BLUE)
    ax.set(title=f"LightGBM: top {top} features (share of total gain)", xlabel="%")
    _done()


def tuning_curves(nv: pd.DataFrame, labels: dict, critical_ratio: float) -> None:
    highlight = {"lightgbm": BLUE, "seasonal_ma": ORANGE, "ensemble": AQUA}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for m, g in nv.groupby("model"):
        hi = highlight.get(m)
        ax.plot(g["param"], g["cost"] / 1000, color=hi or GREY, lw=2.4 if hi else 1, marker="o" if hi else None, ms=4,
                label=labels[m] if hi else None, alpha=1 if hi else .7)
    ax.axvline(critical_ratio, color=INK, lw=.8, ls="--")
    ax.set_ylim(top=nv[nv.model.isin(list(highlight))]["cost"].max() / 1000 * 1.05)
    ax.text(critical_ratio, ax.get_ylim()[1] * .97, " textbook 0.833", fontsize=9)
    ax.set(title="Historical replay: total cost vs service level (grey: other models)", xlabel="service level",
           ylabel="cost over 15 scored weeks (k€)")
    ax.legend()
    _done()


def competition_scores(scores: pd.DataFrame, plan_model: str, benchmark: float, winner: float) -> None:
    d = scores.sort_values("VN2 score €", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.barh(d["label"], d["VN2 score €"], color=[BLUE if m == plan_model else GREY for m in d["model"]])
    for y, v in enumerate(d["VN2 score €"]):
        ax.text(v + 20, y, f"{v:,.0f} €", va="center", fontsize=9, color=INK)
    ax.axvline(benchmark, color=ORANGE, lw=1.5, ls="--")
    ax.text(benchmark, len(d) - .4, f" benchmark {benchmark:,.0f} €", color=ORANGE, fontsize=9)
    ax.axvline(winner, color=INK, lw=1, ls=":")
    ax.text(winner, -.9, f" winner {winner:,.0f} €", color=INK, fontsize=9)
    ax.set(title="VN2 score (cost of weeks 3-8, lower is better), service level tuned on the past", xlabel="€",
           xlim=(3000, d["VN2 score €"].max() * 1.08))
    _done()


def weekly_costs(plan_weekly: pd.DataFrame, bench_weekly: pd.DataFrame) -> None:
    wk = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(wk - .2, plan_weekly["holding"], .38, color="#86b6ef", label="plan: leftover stock")
    ax.bar(wk - .2, plan_weekly["shortage"], .38, bottom=plan_weekly["holding"], color=BLUE, label="plan: missed sales")
    ax.bar(wk + .2, bench_weekly["holding"], .38, color="#f5b393", label="benchmark: leftover stock")
    ax.bar(wk + .2, bench_weekly["shortage"], .38, bottom=bench_weekly["holding"], color=ORANGE, label="benchmark: missed sales")
    ax.axvspan(.5, 2.5, color="#e2e8f0", alpha=.6, lw=0)
    ax.text(.6, ax.get_ylim()[1] * .92, "setup (same for all)", fontsize=9, color=MUTED)
    ax.set(title="Weekly cost: chosen plan (left bars) vs benchmark (right bars)", xlabel="competition week", ylabel="€")
    ax.set_xticks(wk)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    _done()
