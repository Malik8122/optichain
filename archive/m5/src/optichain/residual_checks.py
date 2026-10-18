"""Residual diagnostics for fitted models and the ARCH/GARCH decision.

Residuals are the in-sample one-step errors on the original scale (Forecaster.residuals),
with state-space start-up values dropped.

GARCH gate (set in DESIGN.md): fit GARCH(1,1) only if ARCH-LM p < alpha AND the series has
<= 30% zero days. GARCH never changes the point forecast. It rescales the mean model's own
prediction intervals by the predicted volatility relative to its long-run level, and is kept
only if that improves interval scores on held-out windows.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

from . import plots as P

GARCH_MAX_ZERO_SHARE = 0.30


def model_df(params: dict) -> int:
    """Number of estimated dynamic parameters, used as Ljung-Box degrees-of-freedom correction."""
    if params.get("order") is not None:  # ARIMA / SARIMA
        p, _, q = params["order"]
        P_, _, Q_, _ = params.get("seasonal_order") or [0, 0, 0, 0]
        return int(p + q + P_ + Q_)
    form = params.get("form", "")
    if form.startswith("ETS"):  # alpha [+ beta + phi] [+ gamma]
        trend, season = form[4:-1].split(",")[1:]
        return 1 + (2 if trend == "Ad" else 0) + (1 if season == "A" else 0)
    return 0


def residual_tests(resid: pd.Series, lags=(7, 14, 21), df: int = 0, arch_lag: int = 7, alpha: float = 0.05) -> dict:
    r = resid.dropna().to_numpy(dtype=float)
    out = {"n_resid": len(r), "resid_mean": float(r.mean()), "resid_sd": float(r.std(ddof=1))}
    out["mean_p"] = float(stats.ttest_1samp(r, 0.0).pvalue)
    lb = acorr_ljungbox(r, lags=list(lags), model_df=df, return_df=True)
    for l in lags:
        out[f"lb_p_{l}"] = float(lb.loc[l, "lb_pvalue"]) if l > df else np.nan
    jb = stats.jarque_bera(r)
    out |= {"skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)), "jb_p": float(jb.pvalue)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels return-type deprecation notice
        lm, lm_p, _, _ = het_arch(r, nlags=arch_lag)
    out |= {"arch_lm_stat": float(lm), "arch_lm_p": float(lm_p)}
    # effect size next to the p-values: with ~700 points Ljung-Box flags very small correlations
    out["max_abs_acf"] = float(np.max(np.abs(acf(r, nlags=max(lags), fft=True)[1:])))
    out["no_autocorrelation"] = bool(all(out[f"lb_p_{l}"] >= alpha for l in lags if l > df))
    out["unbiased"] = bool(out["mean_p"] >= alpha)
    out["arch_effects"] = bool(lm_p < alpha)
    return out


# ---------------------------------------------------------------- GARCH

def fit_garch(resid: pd.Series):
    """Zero-mean GARCH(1,1), normal errors, on residuals (rescaled internally by arch)."""
    from arch import arch_model

    r = resid.dropna().to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=True).fit(disp="off")
    pr = res.params
    a, b = float(pr["alpha[1]"]), float(pr["beta[1]"])
    info = {"omega": float(pr["omega"]), "alpha": a, "beta": b, "persistence": a + b,
            "converged": bool(res.convergence_flag == 0), "scale": float(res.scale),
            "alpha_p": float(res.pvalues["alpha[1]"]), "beta_p": float(res.pvalues["beta[1]"])}
    info["valid"] = info["converged"] and a >= 0 and b >= 0 and a + b < 1
    return res, info


def garch_variance_ratio(res, info: dict, h: int, resid: pd.Series) -> np.ndarray:
    """Predicted variance h steps ahead divided by the long-run variance (same units)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_h = res.forecast(horizon=h, reindex=False).variance.to_numpy()[-1]
    if info["persistence"] < 1:
        long_run = info["omega"] / (1 - info["persistence"])
    else:  # integrated: fall back to the sample variance in rescaled units
        long_run = float(np.var(resid.dropna().to_numpy() * info["scale"]))
    return var_h / long_run


def scale_intervals(fc: pd.DataFrame, ratio: np.ndarray, levels=(80, 95)) -> pd.DataFrame:
    """Rescale each band's half-widths (around yhat, separately below/above) by sqrt(ratio)."""
    out = fc.copy()
    k = np.sqrt(ratio[: len(fc)])
    for lv in levels:
        lo, hi, mid = fc[f"lo{lv}"].to_numpy(), fc[f"hi{lv}"].to_numpy(), fc["yhat"].to_numpy()
        out[f"lo{lv}"] = np.clip(mid - (mid - lo) * k, 0, None)
        out[f"hi{lv}"] = mid + (hi - mid) * k
    return out


def interval_score(y, lo, hi, level: float) -> float:
    """Winkler / interval score (lower is better): width + penalties for misses."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    a = 1 - level
    return float(np.mean((hi - lo) + (2 / a) * np.clip(lo - y, 0, None) + (2 / a) * np.clip(y - hi, 0, None)))


# ---------------------------------------------------------------- figures

def fig_residuals(resid: pd.Series, title: str, nlags: int = 35) -> plt.Figure:
    r = resid.dropna()
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))
    (a_ts, a_acf, a_sq), (a_hist, a_qq, a_roll) = axes

    a_ts.bar(r.index, r.values, color=P.CONTEXT, width=1)
    a_ts.axhline(0, color=P.INK_2, lw=0.8)
    a_ts.set_title("Residuals over time")

    bound = 1.96 / np.sqrt(len(r))
    for ax, series, name in [(a_acf, r, "ACF of residuals"), (a_sq, r**2, "ACF of squared residuals")]:
        v = acf(series.to_numpy(), nlags=nlags, fft=True)[1:]
        lags = np.arange(1, nlags + 1)
        colors = [P.BLUE if abs(x) > bound else P.CONTEXT for x in v]
        ax.vlines(lags, 0, v, colors=colors, lw=2)
        ax.axhspan(-bound, bound, color=P.SEQ_BLUE[0], alpha=0.6, lw=0)
        ax.axhline(0, color=P.AXIS, lw=0.8)
        for s in (7, 14, 21, 28, 35):
            ax.axvline(s, color=P.GRID, lw=1, zorder=0)
        ax.set_title(name)
        ax.set_xlabel("Lag (days)")

    a_hist.hist(r.values, bins=40, color=P.BLUE, edgecolor=P.SURFACE, density=True)
    x = np.linspace(r.min(), r.max(), 200)
    a_hist.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=P.INK, lw=1.2)
    a_hist.set_title("Distribution (line: normal)")

    (osm, osr), (slope, intercept, _) = stats.probplot(r.values, dist="norm")
    a_qq.scatter(osm, osr, s=6, color=P.BLUE, alpha=0.6, linewidth=0)
    a_qq.plot(osm, slope * np.asarray(osm) + intercept, color=P.INK, lw=1.2)
    a_qq.set_title("Normal QQ plot")
    a_qq.set_xlabel("Theoretical quantiles")

    roll = r.rolling(28).std()
    a_roll.plot(roll.index, roll.values, color=P.SERIES[1], lw=1.6)
    a_roll.set_title("28-day rolling std (volatility clustering?)")
    a_roll.set_ylim(bottom=0)
    for ax in (a_ts, a_roll):
        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


# ---------------------------------------------------------------- refit helper (picklable for process pools)

def refit_and_test(job: dict) -> dict:
    """Refit the model the backtest chose at one origin and run residual tests on it.

    job: item, family ('ets' | 'sarima' | 'arima'), origin, y_train (Series), params (backtest params),
         lags, arch_lag, alpha.
    """
    from .forecasting import make_model

    fam, p = job["family"], job["params"]
    if fam in ("sarima", "arima"):
        model = make_model(fam, ident={"transform": p["transform"]},
                           fixed={"order": p["order"], "seasonal_order": p["seasonal_order"]})
    else:
        model = make_model(fam)
    model.fit(job["y_train"])
    resid = model.residuals()
    rec = {"item_id": job["item"], "model": fam, "origin_d": job["origin"], "spec": model.params().get("spec")
           or model.params().get("form"), "model_df": model_df(model.params()),
           "zero_share": float((job["y_train"] == 0).mean())}
    rec |= residual_tests(resid, job["lags"], rec["model_df"], job["arch_lag"], job["alpha"])
    return {"rec": rec, "resid": resid}
