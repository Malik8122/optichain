"""Time-series identification: stationarity, transformation, seasonality, ACF/PACF.

Decision procedure (per series, per forecast origin, training data only):
1. Transform: log1p if variability clearly grows with the level (rolling std vs rolling
   mean correlation >= ``var_corr_threshold``) and the series is not zero-heavy.
2. Seasonal differencing D: STL (period 7) seasonal strength Fs >= 0.64 -> D = 1
   (the threshold used by Hyndman's nsdiffs).
3. Ordinary differencing d, tested on the (seasonally differenced) series with ADF and KPSS:
       ADF rejects, KPSS does not  -> stationary            -> d = 0
       ADF fails,   KPSS rejects   -> unit root             -> d = 1
       both reject                 -> conflict (level shifts / long memory) -> d = 0, flagged
       neither rejects             -> inconclusive (low power)               -> d = 0, flagged
   Flagged cases carry d_candidates = [0, 1]; M5 settles them out-of-sample (AIC cannot
   compare models with different d).
   ADF H0: unit root.  KPSS H0: level-stationary.  Neither "proves" stationarity.
4. ACF/PACF of the final modelling series -> a small candidate order grid (guides, not rules).
5. Ljung-Box on the modelling series: is there autocorrelation for ARIMA to exploit at all?
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import plots as P

SEASONAL_STRENGTH_D = 0.64
SEASONAL_PERSISTENCE = 0.2  # ACF at lags s..4s all above this -> seasonal unit root plausible


# ---------------------------------------------------------------- tests

def adf_test(y: pd.Series) -> dict:
    stat, p, lag, nobs, crit, _ = adfuller(np.asarray(y, float), regression="c", autolag="AIC", result_object=False)
    return {"adf_stat": float(stat), "adf_p": float(p), "adf_lags": int(lag), "adf_nobs": int(nobs),
            "adf_crit5": float(crit["5%"])}


def kpss_test(y: pd.Series) -> dict:
    """KPSS (level). statsmodels truncates p to [0.01, 0.10]; we record which bound was hit."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InterpolationWarning)
        stat, p, lags, _ = kpss(np.asarray(y, float), regression="c", nlags="auto")
    bound = ""
    if any(issubclass(x.category, InterpolationWarning) for x in w):
        bound = ">=" if p >= 0.1 else "<="
    return {"kpss_stat": float(stat), "kpss_p": float(p), "kpss_p_bound": bound, "kpss_lags": int(lags)}


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "stationary"
    if not adf_rej and kpss_rej:
        return "unit_root"
    if adf_rej and kpss_rej:
        return "conflict"
    return "inconclusive"


def ljung_box(y: pd.Series, lags: list[int]) -> dict:
    lb = acorr_ljungbox(np.asarray(y, float), lags=lags, return_df=True)
    return {f"lb_p_{l}": float(lb.loc[l, "lb_pvalue"]) for l in lags}


# ---------------------------------------------------------------- transforms

def variance_level_corr(y: pd.Series, window: int = 28) -> float:
    m = y.rolling(window).mean()
    s = y.rolling(window).std()
    ok = m.notna() & s.notna()
    if ok.sum() < 3 or s[ok].std() == 0 or m[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(m[ok], s[ok])[0, 1])


def choose_transform(y: pd.Series, threshold: float = 0.6, max_zero_share: float = 0.10) -> str:
    corr = variance_level_corr(y)
    if np.isnan(corr) or (y == 0).mean() > max_zero_share:
        return "none"
    return "log1p" if corr >= threshold else "none"


def apply_transform(y: pd.Series, kind: str) -> pd.Series:
    if kind == "none":
        return y.astype(float)
    if kind == "log1p":
        if (y < 0).any():
            raise ValueError("log1p needs non-negative values")
        return np.log1p(y.astype(float))
    raise ValueError(f"unknown transform {kind}")


def inverse_transform(y, kind: str):
    """expm1 returns (approximately) the median on the original scale, not the mean."""
    if kind == "none":
        return y
    if kind == "log1p":
        return np.expm1(y)
    raise ValueError(f"unknown transform {kind}")


def difference(y: pd.Series, d: int, D: int, s: int = 7) -> pd.Series:
    """Seasonal differencing first, then ordinary; leading NaNs are dropped."""
    out = y.copy()
    for _ in range(D):
        out = out.diff(s)
    for _ in range(d):
        out = out.diff()
    return out.dropna()


# ---------------------------------------------------------------- seasonality

def stl_decompose(y: pd.Series, period: int = 7):
    return STL(y.astype(float), period=period, robust=True).fit()


def strengths(res) -> dict:
    """Hyndman & Athanasopoulos (FPP3) trend and seasonal strength in [0, 1]."""
    r, s, t = res.resid, res.seasonal, res.trend
    vr = np.var(r)
    fs = max(0.0, 1 - vr / np.var(s + r)) if np.var(s + r) > 0 else 0.0
    ft = max(0.0, 1 - vr / np.var(t + r)) if np.var(t + r) > 0 else 0.0
    return {"seasonal_strength": float(fs), "trend_strength": float(ft)}


# ---------------------------------------------------------------- ACF / PACF

def acf_pacf(y: pd.Series, nlags: int = 35) -> dict:
    y = np.asarray(y, float)
    a = acf(y, nlags=nlags, fft=True)
    p = pacf(y, nlags=nlags, method="ywm")
    return {"acf": a, "pacf": p, "bound": 1.96 / np.sqrt(len(y)), "n": len(y)}


def _cutoff(values: np.ndarray, bound: float, max_order: int) -> int:
    """Largest lag <= max_order (among 1..max_order) that is significant, contiguous from lag 1."""
    k = 0
    for lag in range(1, max_order + 1):
        if abs(values[lag]) > bound:
            k = lag
        else:
            break
    return k


def seasonal_persistence(ap: dict, s: int = 7, k: int = 4) -> dict:
    """Slowly decaying ACF at seasonal lags (s, 2s, ..., ks) suggests a seasonal unit root."""
    vals = [float(ap["acf"][s * i]) for i in range(1, k + 1) if s * i < len(ap["acf"])]
    persistent = len(vals) == k and min(vals) > SEASONAL_PERSISTENCE
    return {"seasonal_acf": vals, "seasonal_acf_persistent": bool(persistent)}


def suggest_orders(ap: dict, s: int = 7, max_pq: int = 2) -> dict:
    a, p, b = ap["acf"], ap["pacf"], ap["bound"]
    sig_a = [int(l) for l in range(1, len(a)) if abs(a[l]) > b]
    sig_p = [int(l) for l in range(1, len(p)) if abs(p[l]) > b]
    return {
        "p_hint": _cutoff(p, b, max_pq),
        "q_hint": _cutoff(a, b, max_pq),
        "P_hint": int(abs(p[s]) > b) if len(p) > s else 0,
        "Q_hint": int(abs(a[s]) > b) if len(a) > s else 0,
        "acf_sig_lags": sig_a[:12],
        "pacf_sig_lags": sig_p[:12],
    }


# ---------------------------------------------------------------- orchestration

def identify(y: pd.Series, alpha: float = 0.05, lb_lags: tuple[int, ...] = (7, 14, 21), s: int = 7) -> dict:
    """Full identification record for one training series (already windowed)."""
    rec: dict = {"n": len(y), "start": str(y.index[0].date()), "end": str(y.index[-1].date()),
                 "zero_share": float((y == 0).mean()), "var_level_corr": variance_level_corr(y)}
    rec["transform"] = choose_transform(y)
    yt = apply_transform(y, rec["transform"])

    rec |= {f"level_{k}": v for k, v in (adf_test(yt) | kpss_test(yt)).items()}
    rec["level_verdict"] = stationarity_verdict(rec["level_adf_p"], rec["level_kpss_p"], alpha)

    rec |= strengths(stl_decompose(yt, s))
    rec["D"] = int(rec["seasonal_strength"] >= SEASONAL_STRENGTH_D)

    ys = difference(yt, 0, rec["D"], s)
    t0 = adf_test(ys) | kpss_test(ys)
    rec |= {f"pre_d_{k}": v for k, v in t0.items()}
    verdict = stationarity_verdict(t0["adf_p"], t0["kpss_p"], alpha)
    rec["pre_d_verdict"] = verdict
    rec["d"] = 1 if verdict == "unit_root" else 0
    rec["flag"] = verdict if verdict in ("conflict", "inconclusive") else ""
    # AIC cannot compare different d (different data), so ambiguous cases are settled out-of-sample in M5.
    rec["d_candidates"] = [0, 1] if verdict in ("conflict", "inconclusive") else [rec["d"]]

    ym = difference(yt, rec["d"], rec["D"], s)
    if rec["d"]:
        t1 = adf_test(ym) | kpss_test(ym)
        rec |= {f"post_d_{k}": v for k, v in t1.items()}
        rec["post_d_verdict"] = stationarity_verdict(t1["adf_p"], t1["kpss_p"], alpha)
    # Seasonal persistence is checked on the series after ordinary differencing only (D = 0), so it
    # flags weekly structure that a strength threshold alone can miss. Settled out-of-sample in M5.
    sp = seasonal_persistence(acf_pacf(difference(yt, rec["d"], 0, s)), s)
    rec |= sp
    rec["D_candidates"] = sorted({rec["D"], 1}) if sp["seasonal_acf_persistent"] else [rec["D"]]
    rec["n_model"] = len(ym)
    rec |= ljung_box(ym, list(lb_lags))
    rec |= suggest_orders(acf_pacf(ym), s=s)
    rec["white_noise"] = bool(all(rec[f"lb_p_{l}"] >= alpha for l in lb_lags))
    return rec


def modelling_series(y: pd.Series, rec: dict, s: int = 7) -> pd.Series:
    return difference(apply_transform(y, rec["transform"]), rec["d"], rec["D"], s)


# ---------------------------------------------------------------- figures

def fig_stationarity(y: pd.Series, rec: dict, title: str) -> plt.Figure:
    yt = apply_transform(y, rec["transform"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, height_ratios=[2, 1])
    a1.plot(yt.index, yt.values, color=P.CONTEXT, lw=0.6, label="Series" + (" (log1p)" if rec["transform"] == "log1p" else ""))
    a1.plot(yt.index, yt.rolling(28).mean(), color=P.BLUE, lw=2, label="28-day rolling mean")
    a1.set_title(title)
    a1.legend(loc="upper left", ncols=2)
    a2.plot(yt.index, yt.rolling(28).std(), color=P.SERIES[1], lw=1.6, label="28-day rolling std")
    a2.legend(loc="upper left")
    a2.set_ylim(bottom=0)
    txt = (f"ADF p = {rec['level_adf_p']:.3f}   KPSS p {rec['level_kpss_p_bound'] or '='} {rec['level_kpss_p']:.2f}"
           f"   → {rec['level_verdict']}   ·   chosen: transform={rec['transform']}, d={rec['d']}, D={rec['D']}")
    fig.text(0.01, -0.02, txt, color=P.INK_2, fontsize=8.5)
    return fig


def fig_stl(y: pd.Series, rec: dict, title: str, last_days: int = 182) -> plt.Figure:
    res = stl_decompose(apply_transform(y, rec["transform"]))
    parts = [("Observed", res.observed), ("Trend", res.trend), ("Weekly seasonal", res.seasonal), ("Remainder", res.resid)]
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.4), sharex=True)
    for ax, (name, comp) in zip(axes, parts):
        c = comp.iloc[-last_days:]
        if name == "Remainder":
            ax.bar(c.index, c.values, color=P.CONTEXT, width=1)
        else:
            ax.plot(c.index, c.values, color=P.BLUE if name != "Observed" else P.INK_2, lw=1.4)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.5)
    axes[0].set_title(f"{title} · STL (period 7), last {last_days} days · "
                      f"seasonal strength {rec['seasonal_strength']:.2f}, trend strength {rec['trend_strength']:.2f}")
    return fig


def fig_acf_pacf(ym: pd.Series, rec: dict, title: str, nlags: int = 35) -> plt.Figure:
    ap = acf_pacf(ym, nlags)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    lags = np.arange(1, nlags + 1)
    for ax, key, name in [(axes[0], "acf", "ACF"), (axes[1], "pacf", "PACF")]:
        v = ap[key][1:nlags + 1]
        colors = [P.BLUE if abs(x) > ap["bound"] else P.CONTEXT for x in v]
        ax.vlines(lags, 0, v, colors=colors, lw=2)
        ax.scatter(lags, v, s=10, color=colors, zorder=3)
        ax.axhspan(-ap["bound"], ap["bound"], color=P.SEQ_BLUE[0], alpha=0.6, lw=0)
        ax.axhline(0, color=P.AXIS, lw=0.8)
        for s_lag in (7, 14, 21, 28, 35):
            ax.axvline(s_lag, color=P.GRID, lw=1, zorder=0)
        ax.set_title(name)
        ax.set_xlabel("Lag (days)")
    ops = f"d={rec['d']}, D={rec['D']}" + (", log1p" if rec["transform"] == "log1p" else "")
    fig.suptitle(f"{title} · modelling series ({ops})", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    return fig
