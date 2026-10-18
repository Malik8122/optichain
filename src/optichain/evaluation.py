"""Rolling-origin backtest and forecast-accuracy metrics.

At each origin we pretend it is that week: models see the history up to the origin only, and forecast
the next ``horizon`` weeks. Errors are measured only on weeks whose sales are a trustworthy measure of
demand (the shelf did not run empty and the product was launched).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .models import GlobalLGBM, History, forecast_all
from .timeseries import RowARIMA


def backtest(hist_full: History, origins: list[int], models: list[str], horizon: int,
             ensemble_of: list[str], lgbm_retrain_every: int = 4, seed: int = 42,
             progress=None, arima_refit_every: int = 8) -> tuple[pd.DataFrame, dict]:
    """Forecast at every origin. Returns a long table (model, origin, h, row, forecast, actual, clean) and
    info (fit times, last LightGBM model)."""
    demand = hist_full.demand
    rows, lgbm, arima, info = [], None, None, {"lgbm_fit_seconds": [], "arima_fit_seconds": [], "model_seconds": {}}
    for k, o in enumerate(origins):
        hist = hist_full.upto(o)
        if "lightgbm" in models and (lgbm is None or k % lgbm_retrain_every == 0):
            t0 = time.perf_counter()
            lgbm = GlobalLGBM(horizon, seed=seed).fit(hist)
            info["lgbm_fit_seconds"].append(time.perf_counter() - t0)
        if "arima" in models and (arima is None or k % arima_refit_every == 0):
            t0 = time.perf_counter()
            arima = RowARIMA().fit(hist)
            info["arima_fit_seconds"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fc = forecast_all(hist, horizon, models, lgbm, ensemble_of, arima)
        info["model_seconds"][o] = time.perf_counter() - t0
        for m, f in fc.items():
            for h in range(1, horizon + 1):
                j = o + h
                rows.append(pd.DataFrame({
                    "model": m, "origin": o, "origin_date": hist_full.dates[o], "h": h,
                    "target_date": hist_full.dates[j], "row": np.arange(hist_full.n),
                    "forecast": f[:, h - 1], "sales": hist_full.sales[:, j], "demand": demand[:, j],
                }))
        if progress:
            progress(k + 1, len(origins))
    info["lgbm"] = lgbm
    info["arima"] = arima
    return pd.concat(rows, ignore_index=True), info


def _metrics(g: pd.DataFrame) -> pd.Series:
    g = g.dropna(subset=["demand"])
    e = g["forecast"] - g["demand"]
    total = g["demand"].sum()
    return pd.Series({
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "Bias %": 100 * e.sum() / total,
        "WAPE %": 100 * e.abs().sum() / total,
        "Score %": 100 * (e.abs().sum() + abs(e.sum())) / total,  # VN1 competition metric: WAPE + |bias|
        "Weeks scored": len(g),
    })


def leaderboard(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy per model (and optional extra keys), best first by Score %."""
    keys = ["model"] + (by or [])
    out = bt.groupby(keys, observed=True)[["forecast", "demand"]].apply(_metrics).reset_index()
    return out.sort_values(keys[1:] + ["Score %"]).reset_index(drop=True)


def dispersion(bt: pd.DataFrame, n_rows: int, shrink: float = 6.0, cap: float = 30.0) -> pd.DataFrame:
    """Forecast-error spread per model and row, as a variance-to-mean ratio k:
        Var(demand | forecast) ~= k * forecast
    k = 1 is Poisson noise; k > 1 is more spread. Each row's k is shrunk towards the model's pooled k
    (rows have few clean weeks), and kept in [1, cap]."""
    g = bt.dropna(subset=["demand"]).copy()
    g["e2"] = (g["forecast"] - g["demand"]) ** 2
    g["mu"] = g["forecast"].clip(lower=0.25)
    out = []
    for m, gm in g.groupby("model"):
        pooled = gm["e2"].sum() / gm["mu"].sum()
        per = gm.groupby("row").agg(e2=("e2", "sum"), mu=("mu", "sum"), n=("e2", "size")).reindex(range(n_rows))
        per = per.fillna({"e2": 0.0, "mu": 0.0, "n": 0})
        raw = (per["e2"] / per["mu"].where(per["mu"] > 0)).fillna(pooled)
        w = per["n"] / (per["n"] + shrink)
        k = (w * raw + (1 - w) * pooled).clip(1.0, cap)
        out.append(pd.DataFrame({"model": m, "row": per.index, "k": k.to_numpy(), "pooled_k": pooled}))
    return pd.concat(out, ignore_index=True)


def calibration(bt: pd.DataFrame, disp: pd.DataFrame, labels: dict, quantiles=(0.5, 0.8, 0.95)) -> pd.DataFrame:
    """Share of clean weeks whose demand stayed below each forecast quantile (ideal: 50%, 80%, 95%)."""
    from .planning import nb_quantile
    g = bt.dropna(subset=["demand"]).merge(disp[["model", "row", "k"]], on=["model", "row"])
    rows = []
    for m, gm in g.groupby("model"):
        r = {"model": labels[m], "pooled k": disp.loc[disp.model == m, "pooled_k"].iloc[0]}
        for q in quantiles:
            r[f"demand ≤ q{int(q * 100)}"] = (gm["demand"] <= nb_quantile(gm["forecast"].to_numpy(), gm["k"].to_numpy(), q)).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")
