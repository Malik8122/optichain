"""Forecasting models for weekly demand.

Every model takes a ``History`` (all 599 rows, weeks up to the forecast origin, nothing later)
and returns an array of shape (rows, horizon): the expected demand of the next weeks.

Models, from simplest to most advanced:
    naive         last clean week, repeated
    ma13          mean of the last 13 clean weeks
    snaive        the same week one year earlier
    seasonal_ma   the official VN2 benchmark: 13-week moving average, deseasonalised and reseasonalised
    ses           simple exponential smoothing on deseasonalised demand (smoothing tuned per row)
    arima         ARIMA per row on deseasonalised demand: d by ADF test, p and q by AIC (timeseries.RowARIMA)
    croston_sba   Croston with the Syntetos-Boylan correction, for slow sellers
    tsb           Teunter-Syntetos-Babai, for slow sellers that may fade out
    lightgbm      one gradient-boosted tree model for all rows, on lags, rolling stats and hierarchy
    ensemble      mean of lightgbm, ses and seasonal_ma
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Panel

CAT_COLS = ["Store", "Product", "ProductGroup", "Division", "Department", "DepartmentGroup", "StoreFormat", "Format"]

MODEL_LABELS = {
    "naive": "Last week",
    "ma13": "13-week average",
    "snaive": "Same week last year",
    "seasonal_ma": "Seasonal average (VN2 benchmark)",
    "ses": "Exponential smoothing",
    "arima": "ARIMA (AIC-selected)",
    "croston_sba": "Croston SBA",
    "tsb": "TSB",
    "lightgbm": "LightGBM (global ML)",
    "ensemble": "Ensemble",
}
MODEL_FAMILY = {
    "naive": "Baseline", "ma13": "Baseline", "snaive": "Baseline",
    "seasonal_ma": "Statistical", "ses": "Statistical", "arima": "Statistical",
    "croston_sba": "Intermittent", "tsb": "Intermittent",
    "lightgbm": "Machine learning", "ensemble": "Combination",
}


@dataclass
class History:
    """Weekly data up to (and including) a forecast origin. Arrays are (rows, weeks)."""
    sales: np.ndarray
    in_stock: np.ndarray
    dates: pd.DatetimeIndex
    cats: pd.DataFrame = field(repr=False)

    @classmethod
    def from_panel(cls, panel: Panel) -> "History":
        cats = panel.master.reindex(panel.index).reset_index()[CAT_COLS].astype(int)
        return cls(panel.sales.to_numpy(float), panel.in_stock.to_numpy(bool), panel.dates, cats)

    def upto(self, t: int) -> "History":
        """Only the weeks 0..t: what a planner knew at origin t."""
        return History(self.sales[:, :t + 1], self.in_stock[:, :t + 1], self.dates[:t + 1], self.cats)

    @property
    def n(self) -> int:
        return self.sales.shape[0]

    @property
    def T(self) -> int:
        return self.sales.shape[1]

    @property
    def launched(self) -> np.ndarray:
        return np.cumsum(self.sales, axis=1) > 0

    @property
    def demand(self) -> np.ndarray:
        """Clean demand: NaN when the week ended out of stock or the product was not launched yet."""
        return np.where(self.in_stock & self.launched, self.sales, np.nan)

    def future_dates(self, horizon: int) -> pd.DatetimeIndex:
        return pd.date_range(self.dates[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")


# ----------------------------------------------------------------- helpers
def _nanmean(a: np.ndarray, axis: int = 1) -> np.ndarray:
    cnt = np.sum(~np.isnan(a), axis=axis)
    tot = np.nansum(a, axis=axis)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _fallback(x: np.ndarray, hist: History) -> np.ndarray:
    """Replace NaN forecasts by the long-run clean mean, then by 0."""
    long_run = _nanmean(hist.demand)
    x = np.where(np.isnan(x), long_run[:, None] if x.ndim == 2 else long_run, x)
    return np.nan_to_num(x, nan=0.0)


def season_factors(hist: History) -> pd.Series:
    """Network-wide seasonality, exactly as the VN2 benchmark computes it: mean sales of all rows per week
    (out-of-stock weeks set aside), averaged per ISO week number, normalised to a mean of 1."""
    masked = np.where(hist.in_stock, hist.sales, np.nan)
    weekly = pd.Series(_nanmean(masked, axis=0), index=hist.dates)
    by_week = weekly.groupby(hist.dates.isocalendar().week.to_numpy()).mean()
    return by_week / by_week.mean()


def season_of(factors: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    weeks = dates.isocalendar().week.to_numpy()
    return factors.reindex(weeks).fillna(1.0).to_numpy()


# ----------------------------------------------------------------- baselines
def naive(hist: History, horizon: int) -> np.ndarray:
    d = pd.DataFrame(hist.demand).ffill(axis=1).to_numpy()[:, -1]
    return np.repeat(_fallback(d, hist)[:, None], horizon, axis=1)


def ma13(hist: History, horizon: int, window: int = 13) -> np.ndarray:
    m = _nanmean(hist.demand[:, -window:])
    return np.repeat(_fallback(m, hist)[:, None], horizon, axis=1)


def snaive(hist: History, horizon: int, season: int = 52) -> np.ndarray:
    d = hist.demand
    out = np.full((hist.n, horizon), np.nan)
    for h in range(1, horizon + 1):
        j = hist.T - 1 + h - season
        if 0 <= j < hist.T:
            out[:, h - 1] = d[:, j]
    fill = ma13(hist, horizon)
    return np.where(np.isnan(out), fill, out)


def seasonal_ma(hist: History, horizon: int, window: int = 13) -> np.ndarray:
    """The VN2 benchmark forecast (vn2_benchmark.py), same arithmetic."""
    factors = season_factors(hist)
    masked = np.where(hist.in_stock, hist.sales, np.nan)
    deseason = masked / season_of(factors, hist.dates)[None, :]
    base = _nanmean(deseason[:, -window:])
    base = np.where(np.isnan(base), _nanmean(deseason), base)
    base = np.nan_to_num(base, nan=0.0)
    return base[:, None] * season_of(factors, hist.future_dates(horizon))[None, :]


# ----------------------------------------------------------------- exponential smoothing
def _ses_level(y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Run SES over the columns, skipping NaN weeks. Returns (final level, one-step squared-error sum)."""
    n, T = y.shape
    level = np.full(n, np.nan)
    sse = np.zeros(n)
    for t in range(T):
        v = y[:, t]
        ok = ~np.isnan(v)
        first = ok & np.isnan(level)
        level[first] = v[first]
        upd = ok & ~first
        err = v[upd] - level[upd]
        sse[upd] += err ** 2
        level[upd] += alpha * err
    return level, sse


def ses(hist: History, horizon: int, alphas=(0.05, 0.1, 0.2, 0.3, 0.5)) -> np.ndarray:
    """Simple exponential smoothing on deseasonalised clean demand; alpha picked per row by in-sample error."""
    factors = season_factors(hist)
    y = hist.demand / season_of(factors, hist.dates)[None, :]
    best_lvl, best_sse = None, None
    for a in alphas:
        lvl, sse = _ses_level(y, a)
        if best_lvl is None:
            best_lvl, best_sse = lvl, sse
        else:
            better = sse < best_sse
            best_lvl = np.where(better, lvl, best_lvl)
            best_sse = np.where(better, sse, best_sse)
    lvl = _fallback(best_lvl, hist)
    return np.maximum(lvl, 0)[:, None] * season_of(factors, hist.future_dates(horizon))[None, :]


# ----------------------------------------------------------------- intermittent demand
def croston_sba(hist: History, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Croston's method: smooth the size of sales (z) and the gap between sales (x) separately.
    Forecast = (1 - alpha/2) * z / x  (the Syntetos-Boylan bias correction)."""
    y = hist.demand
    n, T = y.shape
    z = np.full(n, np.nan)
    x = np.full(n, np.nan)
    gap = np.ones(n)
    for t in range(T):
        v = y[:, t]
        ok = ~np.isnan(v)
        sale = ok & (v > 0)
        first = sale & np.isnan(z)          # the first sale is the launch: no gap observed yet
        z[first] = v[first]
        upd = sale & ~first
        z[upd] += alpha * (v[upd] - z[upd])
        x_new = upd & np.isnan(x)           # the first real gap, between two sales
        x[x_new] = gap[x_new]
        x_upd = upd & ~x_new
        x[x_upd] += alpha * (gap[x_upd] - x[x_upd])
        gap = np.where(sale, 1, gap + ok)
    f = (1 - alpha / 2) * z / x
    f = np.where(np.isnan(z) & ~np.all(np.isnan(y), axis=1), 0.0, f)  # never sold: forecast 0; one sale: fallback
    return np.repeat(_fallback(f, hist)[:, None], horizon, axis=1)


def tsb(hist: History, horizon: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """TSB: smooth the size of sales (z, when there is a sale) and the probability of a sale (p, every week).
    Forecast = p * z. Unlike Croston, p decays when the product stops selling."""
    y = hist.demand
    n, T = y.shape
    z = np.full(n, np.nan)
    p = np.full(n, np.nan)
    for t in range(T):
        v = y[:, t]
        ok = ~np.isnan(v)
        sale = ok & (v > 0)
        start = ok & np.isnan(p)
        p[start] = sale[start].astype(float)
        z[start & sale] = v[start & sale]
        upd = ok & ~start
        p[upd] += beta * (sale[upd] - p[upd])
        zu = upd & sale
        z[zu] = np.where(np.isnan(z[zu]), v[zu], z[zu] + alpha * (v[zu] - z[zu]))
    f = np.nan_to_num(p) * np.nan_to_num(z)
    f = np.where(np.all(np.isnan(y), axis=1), np.nan, f)
    return np.repeat(_fallback(f, hist)[:, None], horizon, axis=1)


# ----------------------------------------------------------------- global machine learning
LGBM_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=300,
                   verbose=-1)


def _roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    df = pd.DataFrame(a.T)
    r = df.rolling(w, min_periods=1)
    return getattr(r, fn)().to_numpy().T


def _origin_features(hist: History) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Features known at the end of every week t (columns of the history). Returns (n, T, k), names, scale."""
    d, s, i = hist.demand, hist.sales, hist.in_stock.astype(float)
    scale = _roll(d, 13, "mean")
    longrun = _roll(d, 156, "mean")
    scale = np.where(np.isnan(scale), longrun, scale)
    scale = np.nan_to_num(scale, nan=0.0) + 0.1
    feats, names = [], []

    def add(name, arr):
        names.append(name)
        feats.append(arr)

    def lag(a, k):
        out = np.full_like(a, np.nan, dtype=float)
        out[:, k:] = a[:, :-k] if k else a
        return out

    for k in range(0, 8):
        add(f"demand_lag{k + 1}", lag(d, k) / scale)
    for k in range(0, 4):
        add(f"sales_lag{k + 1}", lag(s, k) / scale)
        add(f"instock_lag{k + 1}", lag(i, k))
    for w in (4, 8, 26, 52):
        add(f"mean_{w}w", _roll(d, w, "mean") / scale)
    add("std_13w", _roll(d, 13, "std") / scale)
    add("zero_share_13w", _roll(np.where(np.isnan(d), np.nan, (d == 0).astype(float)), 13, "mean"))
    add("instock_share_13w", _roll(i, 13, "mean"))
    add("weeks_since_launch", np.cumsum(hist.launched, axis=1).astype(float))
    add("log_level", np.log1p(scale))
    return np.stack(feats, axis=2), names, scale


def _target_features(hist: History, t: np.ndarray, h: int, factors: pd.Series, dates_ext: pd.DatetimeIndex):
    """Features that depend on the target week t+h (known in advance: calendar, last year's sales)."""
    d = hist.demand
    n = hist.n
    tgt = t + h
    out = []
    for back in (52, 104):
        j = tgt - back
        col = np.full((n, len(t)), np.nan)
        ok = (j >= 0) & (j < hist.T)
        col[:, ok] = d[:, j[ok]]
        out.append(col)
    season_t = season_of(factors, dates_ext[tgt])
    season_o = season_of(factors, dates_ext[t])
    week = dates_ext[tgt].isocalendar().week.to_numpy().astype(float)
    return out, np.broadcast_to(season_t / season_o, (n, len(t))), np.broadcast_to(season_t, (n, len(t))), \
        np.broadcast_to(week, (n, len(t)))


class GlobalLGBM:
    """One LightGBM model for all 599 rows and all horizons (the horizon is a feature).

    The target is next weeks' clean demand divided by the row's recent level, so slow and fast sellers
    share one model. Weeks that ended out of stock are never used as targets."""

    def __init__(self, horizon: int, seed: int = 42, params: dict | None = None, min_origin: int = 26):
        self.horizon = horizon
        self.seed = seed
        self.params = {**LGBM_PARAMS, **(params or {}), "random_state": seed}
        self.min_origin = min_origin
        self.model = None
        self.feature_names: list[str] = []

    def _rows(self, hist: History, t: np.ndarray, h: int, factors: pd.Series):
        F, names, scale = _origin_features(hist)
        dates_ext = hist.dates.append(hist.future_dates(self.horizon))
        lags, ratio, season_t, week = _target_features(hist, t, h, factors, dates_ext)
        sc = scale[:, t]
        blocks = [F[:, t, :]] + [x[..., None] for x in (lags[0] / sc, lags[1] / sc, ratio, season_t, week,
                                                         np.full_like(sc, h))]
        X = np.concatenate(blocks, axis=2)
        cats = np.broadcast_to(hist.cats.to_numpy()[:, None, :], (hist.n, len(t), len(CAT_COLS)))
        X = np.concatenate([X, cats], axis=2)
        cols = names + ["same_week_ly", "same_week_2y", "season_ratio", "season_target", "week_of_year", "horizon"] \
            + CAT_COLS
        return X.reshape(-1, X.shape[2]), cols, sc.reshape(-1)

    def fit(self, hist: History) -> "GlobalLGBM":
        import lightgbm as lgb
        factors = season_factors(hist)
        d = hist.demand
        Xs, ys, ws = [], [], []
        for h in range(1, self.horizon + 1):
            t = np.arange(self.min_origin, hist.T - h)
            X, cols, sc = self._rows(hist, t, h, factors)
            y = d[:, t + h].reshape(-1)
            ok = ~np.isnan(y)
            Xs.append(X[ok]); ys.append(y[ok] / sc[ok]); ws.append(sc[ok])
        X = pd.DataFrame(np.vstack(Xs), columns=cols)
        X[CAT_COLS] = X[CAT_COLS].astype(int).astype("category")
        self.categories = {c: X[c].cat.categories for c in CAT_COLS}
        self.feature_names = cols
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, np.concatenate(ys), sample_weight=np.concatenate(ws))
        return self

    def predict(self, hist: History) -> np.ndarray:
        factors = season_factors(hist)
        t = np.array([hist.T - 1])
        out = np.zeros((hist.n, self.horizon))
        for h in range(1, self.horizon + 1):
            X, cols, sc = self._rows(hist, t, h, factors)
            X = pd.DataFrame(X, columns=cols)
            for c in CAT_COLS:
                X[c] = pd.Categorical(X[c].astype(int), categories=self.categories[c])
            out[:, h - 1] = np.maximum(self.model.predict(X), 0) * sc
        return out

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp / imp.sum(), index=self.feature_names).sort_values(ascending=False)


# ----------------------------------------------------------------- registry
SIMPLE_MODELS = {
    "naive": naive, "ma13": ma13, "snaive": snaive, "seasonal_ma": seasonal_ma,
    "ses": ses, "croston_sba": croston_sba, "tsb": tsb,
}


def forecast_all(hist: History, horizon: int, models: list[str], lgbm: GlobalLGBM | None = None,
                 ensemble_of: list[str] | None = None, arima=None) -> dict[str, np.ndarray]:
    """Forecasts of every requested model at the last week of ``hist``."""
    out = {}
    for m in models:
        if m in SIMPLE_MODELS:
            out[m] = SIMPLE_MODELS[m](hist, horizon)
        elif m == "lightgbm":
            if lgbm is None:
                raise ValueError("lightgbm needs a fitted GlobalLGBM")
            out[m] = lgbm.predict(hist)
        elif m == "arima":
            if arima is None:
                raise ValueError("arima needs a fitted timeseries.RowARIMA")
            out[m] = arima.predict(hist, horizon)
    if "ensemble" in models:
        members = ensemble_of or ["lightgbm", "ses", "seasonal_ma"]
        out["ensemble"] = np.mean([out[m] for m in members], axis=0)
    return out
