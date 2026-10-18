"""Exploratory analysis tables and figures for the selected series.

All functions take an explicit ``end_d`` so EDA never looks past the chosen cutoff.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import plots as P
from .data_pipeline import STORE_CLOSED_EVENTS, get_series
from .selection import ADI_CUT, CV2_CUT, demand_stats

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WD_SHORT = [w[:3] for w in WEEKDAYS]


# ---------------------------------------------------------------- tables

def variability_table(long: pd.DataFrame, items: list[str], end_d: int, window_days: int = 730) -> pd.DataFrame:
    rows = []
    for item in items:
        s = get_series(long, item, end_d).iloc[-window_days:]
        price = long.loc[(long["item_id"] == item) & long["d"].between(end_d - window_days + 1, end_d), "sell_price"]
        rows.append({"item_id": item, **demand_stats(s.to_numpy()), "mean_price": float(price.mean())})
    return pd.DataFrame(rows).set_index("item_id")


def weekday_profile(s: pd.Series) -> pd.DataFrame:
    """Mean daily sales by weekday with a normal-approx 95% CI."""
    g = s.groupby(s.index.day_name())
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()}).reindex(WEEKDAYS)
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["n"])
    out["index"] = out["mean"] / s.mean()
    return out


def weekday_index_table(long: pd.DataFrame, items: list[str], end_d: int, window_days: int = 730) -> pd.DataFrame:
    """Item x weekday: weekday mean / overall mean (1.0 = average day)."""
    return pd.DataFrame({
        item: weekday_profile(get_series(long, item, end_d).iloc[-window_days:])["index"] for item in items
    }).T[WEEKDAYS].rename_axis(columns=None)


def monthly_profile(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.month).mean().rename("mean_sales")


def calendar_effects(long: pd.DataFrame, items: list[str], end_d: int, window_days: int = 730) -> pd.DataFrame:
    """Mean sales on event vs non-event days, and SNAP vs non-SNAP days (post-launch window).

    Store-closure days (Christmas) are excluded: they are zero by construction, not an event effect.
    """
    rows = []
    for item in items:
        w = long[(long["item_id"] == item) & long["launched"] & long["d"].between(end_d - window_days + 1, end_d)
                 & ~long["event_name_1"].isin(STORE_CLOSED_EVENTS)]
        ev = w["event_name_1"] != "none"
        snap = w["snap"] == 1
        rows.append({
            "item_id": item,
            "cat_id": w["cat_id"].iloc[0],
            "event_days": int(ev.sum()),
            "mean_event": w.loc[ev, "sales"].mean(),
            "mean_no_event": w.loc[~ev, "sales"].mean(),
            "event_ratio": w.loc[ev, "sales"].mean() / w.loc[~ev, "sales"].mean(),
            "snap_ratio": w.loc[snap, "sales"].mean() / w.loc[~snap, "sales"].mean(),
        })
    return pd.DataFrame(rows).set_index("item_id")


def event_type_effects(long: pd.DataFrame, item: str, end_d: int) -> pd.DataFrame:
    """Mean sales by event type relative to non-event days, over the full post-launch history."""
    w = long[(long["item_id"] == item) & long["launched"] & (long["d"] <= end_d)
             & ~long["event_name_1"].isin(STORE_CLOSED_EVENTS)]
    base = w.loc[w["event_name_1"] == "none", "sales"].mean()
    out = w.groupby("event_type_1", observed=True)["sales"].agg(["mean", "size"])
    out["ratio_vs_no_event"] = out["mean"] / base
    return out.sort_values("ratio_vs_no_event")


def weekly_price_sales(long: pd.DataFrame, item: str, end_d: int) -> pd.DataFrame:
    w = long[(long["item_id"] == item) & long["launched"] & (long["d"] <= end_d)]
    return (w.groupby("wm_yr_wk")
             .agg(date=("date", "min"), sales=("sales", "sum"), days=("sales", "size"),
                  price=("sell_price", "mean"))
             .query("days == 7")
             .assign(year=lambda d: d["date"].dt.year))


# ---------------------------------------------------------------- figures

def _date_axis(ax, years: bool = True) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator() if years else mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if years else "%b %Y"))


def fig_daily(s: pd.Series, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s.index, s.values, color=P.CONTEXT, lw=0.6, label="Daily units")
    ax.plot(s.index, s.rolling(28).mean(), color=P.BLUE, lw=2, label="28-day rolling mean")
    ax.set_title(title)
    ax.set_ylabel("Units sold")
    ax.set_ylim(bottom=0)
    _date_axis(ax)
    ax.legend(loc="upper left", ncols=2)
    return fig


def fig_weekly(s: pd.Series, title: str) -> plt.Figure:
    w = s.resample("W-SUN").sum().iloc[1:-1]  # drop partial first/last weeks
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(w.index, w.values, color=P.BLUE, lw=1.6)
    ax.set_title(title)
    ax.set_ylabel("Units per week")
    ax.set_ylim(bottom=0)
    _date_axis(ax)
    return fig


def fig_distribution(s: pd.Series, title: str) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.8), width_ratios=[3, 1])
    vmax = int(s.max())
    bins = np.arange(-0.5, vmax + 1.5, 1) if vmax <= 40 else 40
    a1.hist(s.values, bins=bins, color=P.BLUE, edgecolor=P.SURFACE, linewidth=1)
    a1.set_title(title)
    a1.set_xlabel("Units sold per day")
    a1.set_ylabel("Number of days")
    a1.axvline(s.mean(), color=P.INK, lw=1, ls="--")
    a1.annotate(f"mean {s.mean():.1f}", (s.mean(), a1.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", color=P.INK_2, fontsize=8.5)
    a2.boxplot(s.values, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=P.SEQ_BLUE[0], edgecolor=P.BLUE),
               medianprops=dict(color=P.INK), whiskerprops=dict(color=P.MUTED),
               capprops=dict(color=P.MUTED), flierprops=dict(marker="o", ms=3, mfc=P.CONTEXT, mec="none"))
    a2.set_xticks([])
    a2.set_title("Box plot")
    return fig


def fig_weekday(s: pd.Series, title: str) -> plt.Figure:
    prof = weekday_profile(s)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(7)
    ax.bar(x, prof["mean"], color=P.BLUE, width=0.6)
    ax.errorbar(x, prof["mean"], yerr=prof["ci95"], fmt="none", ecolor=P.INK_2, elinewidth=1, capsize=3)
    ax.axhline(s.mean(), color=P.MUTED, lw=1, ls="--")
    ax.set_xticks(x, WD_SHORT)
    ax.set_ylabel("Mean units / day")
    ax.set_title(title)
    return fig


def fig_price_scatter(wk: pd.DataFrame, title: str) -> plt.Figure:
    years = sorted(wk["year"].unique())
    ramp = P.SEQ_BLUE[-len(years):] if len(years) <= len(P.SEQ_BLUE) else P.SEQ_BLUE
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for y, c in zip(years, ramp):
        d = wk[wk["year"] == y]
        ax.scatter(d["price"], d["sales"], s=22, color=c, edgecolor=P.SURFACE, linewidth=0.6, label=str(y))
    ax.set_xlabel("Weekly selling price ($)")
    ax.set_ylabel("Units per week")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.legend(title="Year", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title_fontsize=8)
    return fig


def fig_small_multiples(long: pd.DataFrame, items: list[str], classes: pd.Series, end_d: int,
                        window_days: int = 730) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    for ax, item in zip(axes.flat, items):
        s = get_series(long, item, end_d).iloc[-window_days:]
        ax.plot(s.index, s.values, color=P.CONTEXT, lw=0.5)
        ax.plot(s.index, s.rolling(28).mean(), color=P.BLUE, lw=1.6)
        ax.set_title(f"{item}  ·  {classes[item]}", fontsize=9.5)
        ax.set_ylim(bottom=0)
        _date_axis(ax, years=False)
    fig.suptitle("Daily sales, last 2 years before d_1800 (grey: daily, blue: 28-day mean)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def _place_labels(ax, xs, ys, labels) -> None:
    """Greedy direct labels: try four offsets per point, keep the first that overlaps nothing."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(6, 3, "left"), (6, -11, "left"), (-6, 3, "right"), (-6, -11, "right")]
    for x, y, lab in sorted(zip(xs, ys, labels)):
        for dx, dy, ha in offsets:
            a = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, fontsize=8, color=P.INK)
            box = a.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            a.remove()
        else:
            ax.annotate(lab, (x, y), xytext=(6, 3), textcoords="offset points", fontsize=8, color=P.INK)


def fig_adi_cv2(classes: pd.DataFrame, selected: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    other = classes[~classes["item_id"].isin(selected)]
    sel = classes[classes["item_id"].isin(selected)]
    ax.scatter(other["adi"], other["cv2"], s=8, color=P.CONTEXT, alpha=0.6, linewidth=0, label="Other CA_1 items")
    ax.scatter(sel["adi"], sel["cv2"], s=48, color=P.BLUE, edgecolor=P.SURFACE, linewidth=1.5, label="Selected", zorder=3)
    _place_labels(ax, sel["adi"], sel["cv2"], sel["item_id"])
    ax.axvline(ADI_CUT, color=P.INK_2, lw=1, ls="--")
    ax.axhline(CV2_CUT, color=P.INK_2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.95, 12)
    ticks = [1, ADI_CUT, 2, 3, 5, 10]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 2.0)
    ax.grid(axis="x")
    for x, y, lab in [(1.0, 0.03, "Smooth"), (1.0, 1.9, "Erratic"), (11.5, 0.03, "Intermittent"), (11.5, 1.9, "Lumpy")]:
        ax.text(x, y, lab, color=P.INK_2, fontsize=9, fontweight="bold",
                ha="left" if x < 2 else "right", va="bottom" if y < 1 else "top")
    ax.set_xlabel("ADI — average days between sales (log scale)")
    ax.set_ylabel("CV² of non-zero sales")
    ax.set_title("Demand classification of CA_1 items (d_1071–d_1800)")
    ax.legend(loc="upper center")
    return fig


def fig_weekday_heatmap(idx: pd.DataFrame) -> plt.Figure:
    cmap = LinearSegmentedColormap.from_list("div", P.DIVERGING)
    lim = max(abs(idx.values.min() - 1), abs(idx.values.max() - 1))
    norm = TwoSlopeNorm(vmin=1 - lim, vcenter=1.0, vmax=1 + lim)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(idx.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(7), WD_SHORT)
    ax.set_yticks(range(len(idx)), idx.index)
    ax.grid(False)
    for i in range(idx.shape[0]):
        for j in range(7):
            v = idx.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v - 1) > lim * 0.6 else P.INK)
    ax.set_title("Weekday index (weekday mean ÷ overall mean; 1.00 = average day)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def fig_calendar_heatmap(s: pd.Series, title: str, years: list[int]) -> plt.Figure:
    cmap = LinearSegmentedColormap.from_list("seq", P.SEQ_BLUE)
    fig, axes = plt.subplots(len(years), 1, figsize=(11, 1.9 * len(years)))
    axes = np.atleast_1d(axes)
    vmax = s.quantile(0.99)
    for ax, y in zip(axes, years):
        d = s[s.index.year == y]
        grid = np.full((7, 54), np.nan)
        for ts, v in d.items():
            week = int(ts.strftime("%W"))
            grid[ts.dayofweek, week] = v
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_yticks(range(7), WD_SHORT, fontsize=7.5)
        ax.set_xticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(str(y), rotation=0, ha="right", va="center", fontweight="bold", color=P.INK)
    axes[0].set_title(title)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Units")
    return fig
