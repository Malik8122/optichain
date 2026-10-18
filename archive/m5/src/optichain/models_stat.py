"""Statistical forecasters: ETS, ARIMA/SARIMA and Croston-SBA.

All follow the Forecaster interface (fit on a daily series ending at the origin, predict h
days with intervals). ARIMA/SARIMA read the identification record of their origin
(transform, d/D candidates) produced by diagnostics.identify.
"""
from __future__ import annotations

import itertools
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .diagnostics import apply_transform, inverse_transform
from .forecasting import Forecaster


@contextmanager
def _quiet():
    """statsmodels emits many convergence/start-parameter warnings during grid search;
    convergence is checked explicitly instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


# ---------------------------------------------------------------- ETS

class ETS(Forecaster):
    """Additive-error ETS; the form (none / damped trend) x (none / weekly seasonal) is chosen by AICc.

    Multiplicative forms are excluded because they need strictly positive data, which
    zero-heavy series do not have.
    """
    name = "ets"
    FORMS = [
        {"trend": None, "damped_trend": False, "seasonal": None},
        {"trend": "add", "damped_trend": True, "seasonal": None},
        {"trend": None, "damped_trend": False, "seasonal": "add"},
        {"trend": "add", "damped_trend": True, "seasonal": "add"},
    ]

    def __init__(self, s: int = 7, **_):
        self.s = s

    def _fit(self, y):
        self.table_, best = [], None
        with _quiet():
            for form in self.FORMS:
                try:
                    res = ETSModel(y.astype(float), error="add", seasonal_periods=self.s if form["seasonal"] else None,
                                   **form).fit(disp=False, maxiter=2000)
                    ok = bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True
                    row = {**form, "aicc": float(res.aicc), "aic": float(res.aic), "bic": float(res.bic), "converged": ok}
                except Exception as e:  # noqa: BLE001 - record and skip failing forms
                    self.table_.append({**form, "error": str(e)[:120]})
                    continue
                self.table_.append(row)
                if ok and np.isfinite(row["aicc"]) and (best is None or row["aicc"] < best[0]):
                    best = (row["aicc"], res, form)
        if best is None:
            raise RuntimeError("no ETS form converged")
        _, self.res_, self.form_ = best

    def _forecast(self, steps, levels):
        n, h = len(self.y_), len(steps)
        with _quiet():
            pred = self.res_.get_prediction(start=n, end=n + h - 1)
            mean = np.asarray(pred.predicted_mean)
            bands = {}
            for lv in levels:
                sf = pred.summary_frame(alpha=1 - lv)
                bands[lv] = (sf["pi_lower"].to_numpy(), sf["pi_upper"].to_numpy())
        return mean, bands

    def _point(self, steps):  # not used (_forecast overridden)
        raise NotImplementedError

    def residuals(self):
        return pd.Series(self.y_.to_numpy() - np.asarray(self.res_.fittedvalues), index=self.y_.index)

    def params(self):
        f = self.form_
        label = f"ETS(A,{'Ad' if f['trend'] else 'N'},{'A' if f['seasonal'] else 'N'})"
        return super().params() | {"form": label, "aicc": float(self.res_.aicc), "candidates": self.table_}


# ---------------------------------------------------------------- ARIMA / SARIMA

DEFAULT_GRID = {"p": [0, 1, 2], "q": [0, 1, 2], "P": [0, 1], "Q": [0, 1], "s": 7}


class SarimaFamily(Forecaster):
    """ARIMA (seasonal=False) or SARIMA (seasonal=True) with identification-driven preparation.

    1. Transform per the identification record (log1p or none).
    2. For each allowed (d, D) combination, pick (p, q[, P, Q]) by AIC. Only converged fits
       count, and AIC is compared only within one (d, D), i.e. on the same differenced data.
    3. If several (d, D) combinations are allowed, the last ``inner_holdout`` days of the
       training window decide (RMSE on the original scale). The winning order is refit on
       the full window. The real test window is never used.
    """

    def __init__(self, seasonal: bool, ident: dict | None = None, grid: dict | None = None,
                 inner_holdout: int = 28, maxiter: int = 200, fixed: dict | None = None, **_):
        self.seasonal = seasonal
        self.name = "sarima" if seasonal else "arima"
        self.ident = ident or {"transform": "none", "d_candidates": [0], "D_candidates": [0]}
        self.grid = {**DEFAULT_GRID, **(grid or {})}
        self.inner_holdout = inner_holdout
        self.maxiter = maxiter
        self.fixed = fixed  # {"order": [p, d, q], "seasonal_order": [P, D, Q, s] | None}: refit, no search

    # -- helpers
    def _orders(self):
        g = self.grid
        ps = itertools.product(g["p"], g["q"], g["P"] if self.seasonal else [0], g["Q"] if self.seasonal else [0])
        return list(ps)

    def _fit_one(self, yt: np.ndarray, d, D, p, q, P, Q):
        s = self.grid["s"]
        trend = "c" if d + D == 0 else "n"
        with _quiet():
            res = SARIMAX(yt, order=(p, d, q), seasonal_order=(P, D, Q, s if (P or D or Q) else 0),
                          trend=trend).fit(disp=False, maxiter=self.maxiter)
        return res, bool(res.mle_retvals.get("converged", False))

    def _search(self, yt: np.ndarray, d: int, D: int, tag: str):
        best = None
        for p, q, P, Q in self._orders():
            try:
                res, ok = self._fit_one(yt, d, D, p, q, P, Q)
                aic, bic = float(res.aic), float(res.bic)
            except Exception:  # noqa: BLE001 - numerical failure on one order
                res, ok, aic, bic = None, False, np.nan, np.nan
            self.table_.append({"stage": tag, "d": d, "D": D, "p": p, "q": q, "P": P, "Q": Q,
                                "aic": aic, "bic": bic, "converged": ok})
            if ok and np.isfinite(aic) and (best is None or aic < best[0]):
                best = (aic, res, (p, q, P, Q))
        return best

    def _back(self, x):
        return inverse_transform(np.asarray(x, float), self.transform_)

    # -- Forecaster hooks
    def _fit(self, y):
        t0 = time.time()
        self.table_, self.combo_scores_ = [], []
        self.transform_ = self.ident.get("transform", "none")
        yt = apply_transform(y, self.transform_).to_numpy()
        d_c = self.ident.get("d_candidates", [0])
        D_c = self.ident.get("D_candidates", [0]) if self.seasonal else [0]
        combos = [(d, D) for d in d_c for D in D_c]

        if self.fixed:
            p, d, q = self.fixed["order"]
            P, D, Q = (self.fixed.get("seasonal_order") or [0, 0, 0])[:3]
            order = (p, q, P, Q)
            res, ok = self._fit_one(yt, d, D, *order)
            if not ok:
                raise RuntimeError(f"fixed-order refit did not converge: {order}")
        elif len(combos) > 1 and len(yt) > 3 * self.inner_holdout:
            H = self.inner_holdout
            truth = y.to_numpy()[-H:]
            winners = []
            for d, D in combos:
                best = self._search(yt[:-H], d, D, "inner")
                if best is None:
                    continue
                with _quiet():
                    fc = self._back(best[1].forecast(H))
                rmse = float(np.sqrt(np.mean((np.clip(fc, 0, None) - truth) ** 2)))
                self.combo_scores_.append({"d": d, "D": D, "order": best[2], "inner_rmse": rmse})
                winners.append((rmse, d, D, best[2]))
            if not winners:
                raise RuntimeError("no converged ARIMA candidate")
            _, d, D, order = min(winners)
            res, ok = self._fit_one(yt, d, D, *order)
            if not ok:  # fall back to a full search on the chosen (d, D)
                best = self._search(yt, d, D, "final")
                res, order = best[1], best[2]
        else:
            d, D = combos[0]
            best = self._search(yt, d, D, "final")
            if best is None:
                raise RuntimeError("no converged ARIMA candidate")
            res, order = best[1], best[2]

        self.res_, self.d_, self.D_, self.order_ = res, d, D, order
        self.fit_seconds_ = time.time() - t0

    def _forecast(self, steps, levels):
        h = len(steps)
        with _quiet():
            fc = self.res_.get_forecast(h)
            mean = self._back(fc.predicted_mean)
            bands = {}
            for lv in levels:
                ci = np.asarray(fc.conf_int(alpha=1 - lv))
                bands[lv] = (self._back(ci[:, 0]), self._back(ci[:, 1]))  # monotone transform: quantiles map
        return mean, bands

    def _point(self, steps):
        raise NotImplementedError

    def residuals(self):
        fitted = self._back(self.res_.fittedvalues)
        r = pd.Series(self.y_.to_numpy() - fitted, index=self.y_.index)
        p, q, P, Q = self.order_
        burn = self.d_ + self.D_ * self.grid["s"] + max(p, q, (P + Q) * self.grid["s"]) + 7
        r.iloc[:burn] = np.nan  # state-space start-up residuals are not genuine one-step errors
        return r

    def params(self):
        p, q, P, Q = self.order_
        s = self.grid["s"]
        label = f"ARIMA({p},{self.d_},{q})" + (f"({P},{self.D_},{Q})[{s}]" if self.seasonal else "")
        return super().params() | {
            "spec": label, "transform": self.transform_, "order": [p, self.d_, q],
            "seasonal_order": [P, self.D_, Q, s] if self.seasonal else None,
            "aic": float(self.res_.aic), "bic": float(self.res_.bic),
            "combo_scores": self.combo_scores_, "n_fits": len(self.table_), "fit_seconds": self.fit_seconds_,
            "candidates": self.table_,
        }


# ---------------------------------------------------------------- Croston / SBA

class Croston(Forecaster):
    """Croston's method with the Syntetos-Boylan bias correction (SBA).

    Smooths non-zero demand sizes (z) and inter-demand intervals (p) separately; forecast =
    (1 - alpha/2) * z / p, flat over the horizon. alpha is chosen from a small set by
    in-sample one-step MSE.
    """
    name = "croston"
    ALPHAS = (0.05, 0.1, 0.2, 0.3)

    def __init__(self, **_):
        pass

    @staticmethod
    def _run(y: np.ndarray, a: float):
        nz = np.flatnonzero(y > 0)
        f = np.full(len(y), np.nan)
        if len(nz) == 0:
            return f, 0.0, 1.0
        z, p, q = float(y[nz[0]]), float(nz[0] + 1), 1
        for t in range(nz[0] + 1, len(y)):
            f[t] = (1 - a / 2) * z / p
            if y[t] > 0:
                z += a * (y[t] - z)
                p += a * (q - p)
                q = 1
            else:
                q += 1
        return f, z, p

    def _fit(self, y):
        v = y.to_numpy()
        best = None
        for a in self.ALPHAS:
            f, z, p = self._run(v, a)
            mse = float(np.nanmean((v - f) ** 2)) if np.isfinite(f).any() else np.inf
            if best is None or mse < best[0]:
                best = (mse, a, f, z, p)
        _, self.alpha_, self.fitted_, z, p = best
        self.level_ = (1 - self.alpha_ / 2) * z / p

    def _point(self, steps):
        return np.full(len(steps), self.level_)

    def residuals(self):
        return pd.Series(self.y_.to_numpy() - self.fitted_, index=self.y_.index)

    def params(self):
        return super().params() | {"alpha": self.alpha_, "level": self.level_}
