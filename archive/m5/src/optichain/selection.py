"""Demand classification (Syntetos-Boylan ADI/CV²) and deterministic product selection.

Everything here uses only data with d <= ``end_d`` (default: earliest forecast origin),
so the choice of products cannot be influenced by any test window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49
CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]


def classify(adi: float, cv2: float) -> str:
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def max_zero_run(sales: np.ndarray) -> int:
    """Longest run of consecutive zero-sales days (long runs in busy items hint at stockouts)."""
    best = run = 0
    for v in sales:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    return best


def demand_stats(sales: np.ndarray) -> dict:
    """ADI, CV² of non-zero sizes, and basic variability stats for one window."""
    sales = np.asarray(sales, dtype=float)
    nz = sales[sales > 0]
    n_nz = len(nz)
    adi = len(sales) / n_nz if n_nz else np.inf
    cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2) if n_nz > 1 else np.nan
    mean = sales.mean()
    return {
        "max_zero_run": max_zero_run(sales),
        "mean": mean,
        "median": float(np.median(sales)),
        "std": sales.std(ddof=1),
        "cv": sales.std(ddof=1) / mean if mean > 0 else np.nan,
        "zero_share": 1 - n_nz / len(sales),
        "adi": adi,
        "cv2": cv2,
        "demand_class": classify(adi, cv2) if n_nz > 1 else "no_demand",
    }


def classify_items(
    long: pd.DataFrame, end_d: int, window_days: int = 730, recent_days: int = 90
) -> pd.DataFrame:
    """Classify every item with a full ``window_days`` of post-launch history before ``end_d``.

    Eligibility: launched on or before the window start, and at least one sale in the
    last ``recent_days`` (screens out items that look discontinued).
    """
    start_d = end_d - window_days + 1
    w = long[(long["d"] >= start_d) & (long["d"] <= end_d)]
    launch = long[long["launched"]].groupby("item_id", observed=True)["d"].min()
    eligible = launch.index[launch <= start_d]
    w = w[w["item_id"].isin(eligible)]

    rows = []
    for item, g in w.groupby("item_id", observed=True):
        s = g.sort_values("d")
        if s["sales"].iloc[-recent_days:].sum() == 0:
            continue
        rows.append({
            "item_id": item,
            "dept_id": s["dept_id"].iloc[0],
            "cat_id": s["cat_id"].iloc[0],
            **demand_stats(s["sales"].to_numpy()),
            "mean_price": float(s["sell_price"].mean()),
        })
    out = pd.DataFrame(rows)
    out["window"] = f"d_{start_d}-d_{end_d}"
    return out.sort_values("item_id", ignore_index=True)


# Per class: which point of the class's volume distribution to sample.
# 1.0 = the highest-volume item in the class (the "very high-volume" smooth series).
DEFAULT_QUOTAS = {
    "smooth": [1.0, 0.75, 0.5],
    "erratic": [0.75, 0.5],
    "intermittent": [0.75, 0.5],
    "lumpy": [0.5],
}


def select_items(
    classes: pd.DataFrame,
    quotas: dict[str, list[float]] = DEFAULT_QUOTAS,
    n_near: int = 15,
    max_zero_run_days: int = 28,
) -> pd.DataFrame:
    """Deterministically pick items per class at target volume quantiles.

    Items with a zero-sales run longer than ``max_zero_run_days`` are excluded: such runs
    are far more likely supply/assortment gaps (censored demand) than true zero demand,
    and would confound the model comparison.

    For each target, consider the ``n_near`` items closest in log-volume and take the one
    from the least-represented category so far (ties: closest volume, then item_id).
    """
    classes = classes[classes["max_zero_run"] <= max_zero_run_days]
    picks: list[dict] = []
    used: set[str] = set()
    cat_count: dict[str, int] = {}
    for cls, targets in quotas.items():
        pool = classes[classes["demand_class"] == cls]
        if len(pool) < len(targets):
            raise ValueError(f"Only {len(pool)} {cls} items for {len(targets)} picks")
        log_vol = np.log(pool["mean"])
        for q in targets:
            target = log_vol.quantile(q)
            cand = pool[~pool["item_id"].isin(used)].assign(dist=(log_vol - target).abs())
            cand = cand.nsmallest(n_near, "dist")
            if q == 1.0:  # the top item itself, not a category-balanced neighbour
                cand = cand.nsmallest(1, "dist")
            cand = cand.assign(cat_n=cand["cat_id"].astype(str).map(cat_count).fillna(0))
            best = cand.sort_values(["cat_n", "dist", "item_id"]).iloc[0]
            used.add(best["item_id"])
            cat_count[str(best["cat_id"])] = cat_count.get(str(best["cat_id"]), 0) + 1
            picks.append({**best.drop(["dist", "cat_n"]).to_dict(), "volume_quantile": q})
    return pd.DataFrame(picks)
