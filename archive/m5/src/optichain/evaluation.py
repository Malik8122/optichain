"""Forecast accuracy metrics and rolling-origin backtesting.

``rolling_origin`` is the only place that splits train from test: each model sees the
series up to the origin day and nothing after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_pipeline import d_to_date
from .forecasting import DEFAULT_LEVELS, make_model


def scale_denominators(y_train: pd.Series) -> dict[str, float]:
    """M5-style scales: in-sample one-step naive errors from the first non-zero sale."""
    y = y_train.to_numpy(dtype=float)
    nz = np.flatnonzero(y)
    if len(nz) == 0:
        return {"mse_scale": np.nan, "mae_scale": np.nan}
    d = np.diff(y[nz[0]:])
    if len(d) == 0:
        return {"mse_scale": np.nan, "mae_scale": np.nan}
    return {"mse_scale": float(np.mean(d**2)), "mae_scale": float(np.mean(np.abs(d)))}


RESID_SUM_LEADS = tuple(range(1, 15))
RESID_SUM_QUANTILES = (0.5, 0.8, 0.85, 0.9, 0.95, 0.975, 0.98, 0.99)


def residual_sum_quantiles(resid: pd.Series, leads=RESID_SUM_LEADS, qs=RESID_SUM_QUANTILES) -> list[dict]:
    """Empirical quantiles of L-day rolling sums of in-sample one-step residuals (actual - fitted).

    Used by the empirical safety-stock policy: unlike z*sigma*sqrt(L) it keeps heavy tails,
    skew and any residual autocorrelation. Uses only the training window.
    """
    r = resid.dropna()
    rows = []
    for L in leads:
        s = r.rolling(L).sum().dropna().to_numpy()
        if len(s) < 30:
            continue
        for q, v in zip(qs, np.quantile(s, qs)):
            rows.append({"L": int(L), "q": float(q), "value": float(v), "n": int(len(s))})
    return rows


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in [0, 2]; a day where both actual and forecast are 0 counts as 0 error."""
    denom = np.abs(y) + np.abs(yhat)
    terms = np.where(denom == 0, 0.0, 2 * np.abs(y - yhat) / np.where(denom == 0, 1, denom))
    return float(np.mean(terms))


def point_metrics(y: np.ndarray, yhat: np.ndarray, mse_scale: float, mae_scale: float) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(err)),  # > 0 means over-forecasting
        "smape": smape(y, yhat),
        # degenerate (zero / NaN) scale -> NaN, never inf
        "rmsse": float(np.sqrt(np.mean(err**2) / mse_scale)) if mse_scale and mse_scale > 0 else np.nan,
        "mase": float(mae / mae_scale) if mae_scale and mae_scale > 0 else np.nan,
    }


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def rolling_origin(
    y_full: pd.Series,
    model_name: str,
    origins: list[int],
    h: int,
    fit_window_days: int | None = None,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    context_by_origin: dict[int, dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit at each origin on data up to that day and forecast ``h`` days ahead.

    ``context_by_origin[o]`` is passed to the model factory (e.g. the identification record
    computed from data up to o). Returns (forecasts with y_true joined, per-origin params).
    """
    frames, params = [], []
    for o in origins:
        origin_date = d_to_date(o)
        y_train = y_full.loc[:origin_date]
        if y_train.index[-1] != origin_date:
            raise ValueError(f"series has no observation on origin d_{o} ({origin_date.date()})")
        y_fit = y_train.iloc[-fit_window_days:] if fit_window_days else y_train
        model = make_model(model_name, **(context_by_origin or {}).get(o, {})).fit(y_fit)
        fc = model.predict(h, levels)
        assert fc.index.min() > origin_date, "forecast dates must follow the origin"
        fc = fc.assign(model=model_name, origin_d=o, d=o + fc["h"].to_numpy(),
                       y_true=y_full.reindex(fc.index).to_numpy(),
                       sigma_resid=model.sigma_, **scale_denominators(y_train))
        frames.append(fc.reset_index())
        params.append({"origin_d": o, **model.params(), "resid_sum_q": residual_sum_quantiles(model.residuals())})
    return pd.concat(frames, ignore_index=True), params


def score(forecasts: pd.DataFrame, h_eval: int, levels: tuple[float, ...] = DEFAULT_LEVELS) -> pd.DataFrame:
    """Per (item, model, origin) metrics over the first ``h_eval`` steps."""
    f = forecasts[forecasts["h"] <= h_eval]
    keys = [k for k in ("item_id", "model", "origin_d") if k in f.columns]
    rows = []
    for key, g in f.groupby(keys, observed=True):
        g = g.dropna(subset=["y_true"])
        if len(g) < h_eval:
            continue  # incomplete window (e.g. beyond max_day)
        m = point_metrics(g["y_true"], g["yhat"], g["mse_scale"].iloc[0], g["mae_scale"].iloc[0])
        for lv in levels:
            tag = int(round(lv * 100))
            m[f"cov{tag}"] = coverage(g["y_true"], g[f"lo{tag}"], g[f"hi{tag}"])
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,))) | m)
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame, baseline: str = "snaive") -> pd.DataFrame:
    """Mean over origins per item x model, plus skill vs a baseline (1 - RMSSE/RMSSE_baseline)."""
    s = metrics.groupby(["item_id", "model"], observed=True).mean(numeric_only=True).drop(columns="origin_d")
    if baseline in s.index.get_level_values("model"):
        base = s.xs(baseline, level="model")["rmsse"]
        s["skill_vs_" + baseline] = 1 - s["rmsse"] / base.reindex(s.index.get_level_values("item_id")).to_numpy()
    return s.reset_index()
