"""Classical (Box-Jenkins) time-series toolkit: descriptive statistics, stationarity tests, transformations,
seasonal decomposition, ACF/PACF, ARIMA-family identification and AIC/BIC selection, residual diagnostics
(Ljung-Box, Jarque-Bera, ARCH-LM) and GARCH.

Two uses:
* the step-by-step analysis of one series (the network total, or one store-product) in the notebook and on
  the dashboard's Model check page;
* ``RowARIMA``, an AIC-selected ARIMA fitted to every store-product, which joins the forecasting models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .models import History, season_factors, season_of

# p, d, q candidates. d is decided by the ADF test first; these are then searched by AIC.
PQ_GRID = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]


def _quiet():
    warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- 2 · descriptive statistics
def describe(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """Descriptive statistics per series (NaN ignored)."""
    rows = {}
    for name, y in series.items():
        v = np.asarray(y, float)
        v = v[~np.isnan(v)]
        jb = stats.jarque_bera(v) if len(v) > 7 else None
        rows[name] = {
            "weeks": len(v), "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(), "25%": np.percentile(v, 25),
            "median": np.median(v), "75%": np.percentile(v, 75), "max": v.max(),
            "CV": v.std(ddof=1) / v.mean() if v.mean() else np.nan, "skewness": stats.skew(v),
            "excess kurtosis": stats.kurtosis(v), "zero weeks %": 100 * (v == 0).mean(),
            "Jarque-Bera p": jb.pvalue if jb is not None else np.nan,
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------- 4 · stationarity
def adf_test(y: np.ndarray, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller. H0: the series has a unit root (NOT stationary). p < 0.05 -> stationary."""
    from statsmodels.tsa.stattools import adfuller
    _quiet()
    v = np.asarray(y, float)
    v = v[~np.isnan(v)]
    stat, p, lags, nobs, crit, _ = adfuller(v, regression=regression, autolag="AIC")
    return {"ADF statistic": stat, "ADF p-value": p, "ADF lags used": lags, "ADF 5% critical": crit["5%"],
            "ADF says": "stationary" if p < 0.05 else "not stationary"}


def kpss_test(y: np.ndarray, regression: str = "c") -> dict:
    """KPSS, the mirror test. H0: the series IS stationary. p < 0.05 -> not stationary."""
    from statsmodels.tsa.stattools import kpss
    _quiet()
    v = np.asarray(y, float)
    v = v[~np.isnan(v)]
    stat, p, lags, crit = kpss(v, regression=regression, nlags="auto")
    return {"KPSS statistic": stat, "KPSS p-value": p, "KPSS says": "not stationary" if p < 0.05 else "stationary"}


def stationarity(series: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {name: {**adf_test(y), **kpss_test(y)} for name, y in series.items()}
    out = pd.DataFrame(rows).T
    out["verdict"] = np.where((out["ADF says"] == "stationary") & (out["KPSS says"] == "stationary"), "stationary",
                              np.where((out["ADF says"] != "stationary") & (out["KPSS says"] != "stationary"),
                                       "not stationary", "borderline (tests disagree)"))
    return out


def choose_d(y: np.ndarray, alpha: float = 0.05, max_d: int = 1) -> int:
    """Differencing order: difference while the ADF test cannot reject a unit root."""
    v = np.asarray(y, float)
    v = v[~np.isnan(v)]
    for d in range(max_d + 1):
        if len(v) < 20 or np.nanstd(v) == 0:
            return d
        try:
            if adf_test(v)["ADF p-value"] < alpha:
                return d
        except Exception:
            return d
        v = np.diff(v)
    return max_d


# ----------------------------------------------------------------- 5 · transformations
def transformations(y: np.ndarray, season: int = 52) -> dict[str, np.ndarray]:
    """The usual candidates to make a series stationary."""
    y = np.asarray(y, float)
    log = np.log1p(y)
    return {
        "original": y,
        "log(1+y)": log,
        "1st difference": np.diff(y),
        "log, 1st difference": np.diff(log),
        f"seasonal difference ({season})": y[season:] - y[:-season],
    }


def fill_gaps(y: np.ndarray) -> np.ndarray:
    """Linear interpolation inside the series, for tools that cannot take NaN (STL, ADF on a calendar).
    Leading NaN (before launch) are dropped by the caller."""
    s = pd.Series(np.asarray(y, float))
    return s.interpolate(limit_direction="both").to_numpy()


# ----------------------------------------------------------------- 6 · seasonality
def decompose(y: np.ndarray, dates: pd.DatetimeIndex, period: int = 52) -> pd.DataFrame:
    """STL decomposition (robust LOESS): observed = trend + seasonal + remainder."""
    from statsmodels.tsa.seasonal import STL
    res = STL(pd.Series(fill_gaps(y), index=dates), period=period, robust=True).fit()
    return pd.DataFrame({"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal, "remainder": res.resid})


def strength(dec: pd.DataFrame) -> dict:
    """Hyndman's strengths in [0, 1]: how much of the variation the trend / the season explains."""
    r = dec["remainder"].var()
    return {"trend strength": max(0.0, 1 - r / (dec["trend"] + dec["remainder"]).var()),
            "seasonal strength": max(0.0, 1 - r / (dec["seasonal"] + dec["remainder"]).var())}


# ----------------------------------------------------------------- 7 · ACF / PACF and 9 · lags
def acf_pacf(y: np.ndarray, nlags: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    from statsmodels.tsa.stattools import acf, pacf
    v = np.asarray(y, float)
    v = v[~np.isnan(v)]
    nlags = min(nlags, len(v) // 2 - 1)
    a = acf(v, nlags=nlags, fft=True)
    p = pacf(v, nlags=nlags, method="ywm")
    band = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(v))
    return pd.DataFrame({"lag": np.arange(nlags + 1), "acf": a, "pacf": p, "band": band}).iloc[1:].reset_index(drop=True)


def significant_lags(ap: pd.DataFrame) -> dict:
    return {"acf": ap.loc[ap.acf.abs() > ap.band, "lag"].tolist(), "pacf": ap.loc[ap.pacf.abs() > ap.band, "lag"].tolist()}


def identify(ap: pd.DataFrame, d: int, seasonal_lag: int | None = None) -> dict:
    """Box-Jenkins reading of the ACF/PACF of the (differenced) series.

    * PACF cuts off after p and ACF decays  -> AR(p)
    * ACF cuts off after q and PACF decays  -> MA(q)
    * both decay                             -> ARMA(p, q)
    * no significant lag                     -> white noise
    With d differences the model becomes ARIMA(p, d, q); a spike at the seasonal lag adds a seasonal part (SARIMA)."""
    sig = significant_lags(ap)
    first = lambda lags: [l for l in lags if l <= 6]
    a, pa = first(sig["acf"]), first(sig["pacf"])

    def cutoff(lags):  # the last lag of the leading run of significant lags (1, 2, ... without a gap)
        k = 0
        while k + 1 in lags:
            k += 1
        return k

    p, q = cutoff(pa), cutoff(a)
    if not a and not pa:
        kind, p, q = "white noise", 0, 0
    elif p and (q == 0 or len(a) > len(pa) + 1):
        kind, q = "AR", 0
    elif q and (p == 0 or len(pa) > len(a) + 1):
        kind, p = "MA", 0
    else:
        kind = "ARMA"
    p, q = min(p, 2), min(q, 2)
    seasonal = bool(seasonal_lag) and seasonal_lag in sig["acf"]
    family = kind if d == 0 else ("ARIMA" if kind != "white noise" else "random walk")
    if seasonal:
        family = "SARIMA"
    return {"pattern": kind, "suggested p": p, "suggested d": d, "suggested q": q, "seasonal spike": seasonal,
            "family": family, "significant ACF lags": sig["acf"][:10], "significant PACF lags": sig["pacf"][:10]}


# ----------------------------------------------------------------- 8 · 10 · AIC / BIC
def family_of(order, seasonal=(0, 0, 0, 0)) -> str:
    p, d, q = order
    if any(seasonal[:3]):
        return "SARIMA"
    if d:
        return "ARIMA"
    if p and q:
        return "ARMA"
    if p:
        return "AR"
    if q:
        return "MA"
    return "mean only"


def fit_arima(y: np.ndarray, order, seasonal=(0, 0, 0, 0)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _quiet()
    trend = "c" if order[1] == 0 and seasonal[1] == 0 else "n"
    return SARIMAX(np.asarray(y, float), order=order, seasonal_order=seasonal, trend=trend).fit(disp=False)


def aic_bic_grid(y: np.ndarray, d: int, seasonal_orders=((0, 0, 0, 0),), pq=PQ_GRID) -> pd.DataFrame:
    """Fit every candidate and rank by AIC (BIC shown too: it penalises extra parameters harder)."""
    rows = []
    for so in seasonal_orders:
        for p, q in pq:
            order = (p, d, q)
            try:
                res = fit_arima(y, order, so)
                rows.append({"model": f"{family_of(order, so)}{order}" + (f"x{so[:3]}[{so[3]}]" if any(so[:3]) else ""),
                             "family": family_of(order, so), "order": order, "seasonal": so, "AIC": res.aic, "BIC": res.bic,
                             "log-likelihood": res.llf, "parameters": len(res.params),
                             "converged": bool(res.mle_retvals.get("converged", True))})
            except Exception as e:  # noqa: BLE001  (a candidate that cannot be fitted is simply skipped)
                rows.append({"model": f"{order}{so}", "family": family_of(order, so), "order": order, "seasonal": so,
                             "AIC": np.nan, "BIC": np.nan, "converged": False, "error": str(e)[:60]})
    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out["ΔAIC"] = out["AIC"] - out["AIC"].min()
    return out


# ----------------------------------------------------------------- 11 · residuals, ARCH, GARCH
def residual_tests(resid: np.ndarray, lags=(4, 8, 13, 26), arch_lag: int = 4, n_params: int = 0) -> dict:
    """Is what the model leaves behind white noise?

    Ljung-Box  H0: no autocorrelation up to lag L  (p > 0.05 -> looks like white noise)
    Jarque-Bera H0: residuals are normal
    ARCH-LM    H0: no volatility clustering (p < 0.05 -> variance changes over time: try GARCH)"""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    v = np.asarray(resid, float)
    v = v[~np.isnan(v)]
    lags = [l for l in lags if l < len(v) // 2]
    out = {}
    if lags:
        lb = acorr_ljungbox(v, lags=lags, model_df=min(n_params, min(lags) - 1))
        for l in lags:
            out[f"Ljung-Box p (lag {l})"] = float(lb.loc[l, "lb_pvalue"])
    out["Jarque-Bera p"] = float(stats.jarque_bera(v).pvalue)
    try:
        out["ARCH-LM p"] = float(het_arch(v, nlags=arch_lag)[1])
    except Exception:  # noqa: BLE001
        out["ARCH-LM p"] = np.nan
    lb_ps = [v_ for k, v_ in out.items() if k.startswith("Ljung")]
    out["white noise"] = bool(lb_ps) and min(lb_ps) > 0.05
    out["ARCH effect"] = bool(out["ARCH-LM p"] < 0.05) if np.isfinite(out["ARCH-LM p"]) else False
    return out


def fit_garch(resid: np.ndarray, p: int = 1, q: int = 1):
    """GARCH(p, q) on residuals: variance today = ω + α·shock² yesterday + β·variance yesterday.
    Residuals are rescaled to unit variance for a stable fit; volatility is scaled back."""
    from arch import arch_model
    v = np.asarray(resid, float)
    v = v[~np.isnan(v)]
    scale = v.std()
    res = arch_model(v / scale, mean="Zero", vol="GARCH", p=p, q=q, dist="normal").fit(disp="off")
    return res, scale


# ----------------------------------------------------------------- the whole workflow on one series
def analyze(y: np.ndarray, dates: pd.DatetimeIndex, period: int = 52, seasonal: bool = False, nlags: int = 30) -> dict:
    """Steps 2-11 on one series (NaN = missing week). Used live by the dashboard's Model check page."""
    v, start = _prep(np.asarray(y, float))
    dts = dates[start:start + len(v)]
    filled = fill_gaps(v)
    out = {"n": len(v), "missing": int(np.isnan(v).sum()), "start": dts[0]}
    out["describe"] = describe({"series": v})
    tr = {k: t for k, t in transformations(filled, period).items() if len(t) > 20 and np.std(t) > 0}
    out["stationarity"] = stationarity(tr)
    d = choose_d(filled)
    work = np.diff(filled, n=d) if d else filled
    out["d"] = d
    out["work"] = work
    if len(filled) >= 2 * period + 4:
        dec = decompose(filled, dts, period)
        out["decomposition"], out["strength"] = dec, strength(dec)
    ap = acf_pacf(work, nlags)
    out["acf_pacf"] = ap
    out["identify"] = identify(ap, d, period if len(work) > period + 5 else None)
    so = [(0, 0, 0, 0)] + ([(1, 0, 0, period)] if seasonal and len(v) > 2 * period else [])
    grid = aic_bic_grid(v, d, so)
    out["aic_grid"] = grid
    best = grid.dropna(subset=["AIC"]).iloc[0]
    res = fit_arima(v, best["order"], best["seasonal"])
    burn = max(int(getattr(res, "loglikelihood_burn", 0)), d) + 1
    resid = np.asarray(res.resid)[burn:]
    out["best"], out["fit"], out["resid"], out["resid_dates"] = best, res, resid, dts[burn:]
    out["resid_tests"] = residual_tests(resid, n_params=len(res.params) - 1)
    out["resid_acf"] = acf_pacf(resid, nlags)
    try:
        g, scale = fit_garch(resid)
        std = np.asarray(g.std_resid)
        out["garch"] = {"result": g, "scale": scale, "params": g.params,
                        "std_tests": residual_tests(std), "sq_tests": residual_tests(std ** 2),
                        "volatility": np.asarray(g.conditional_volatility) * scale}
    except Exception as e:  # noqa: BLE001
        out["garch"] = {"error": str(e)}
    return out


# ----------------------------------------------------------------- ARIMA for every store-product
def _prep(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop the weeks before the first clean observation (not launched yet)."""
    ok = np.flatnonzero(~np.isnan(y))
    start = int(ok[0]) if len(ok) else len(y)
    return y[start:], start


def _select_one(y: np.ndarray, min_obs: int = 26) -> dict:
    """AIC selection for one store-product (d from ADF, then p, q by AIC). Returns plain numbers only."""
    _quiet()
    v, _ = _prep(y)
    n_obs = int(np.sum(~np.isnan(v)))
    if n_obs < min_obs or np.nanstd(v) == 0:
        return {"order": None, "params": None, "aic": np.nan, "bic": np.nan, "d": 0, "n_obs": n_obs}
    d = choose_d(v)
    best = None
    for p, q in PQ_GRID:
        try:
            res = fit_arima(v, (p, d, q))
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(res.aic):
            continue
        if best is None or res.aic < best.aic:
            best = res
    if best is None:
        return {"order": None, "params": None, "aic": np.nan, "bic": np.nan, "d": d, "n_obs": n_obs}
    tests = residual_tests(best.resid[d + 1:], lags=(4, 8, 13), n_params=len(best.params) - 1)
    return {"order": tuple(best.model.order), "params": np.asarray(best.params), "aic": best.aic, "bic": best.bic,
            "d": d, "n_obs": n_obs, **tests}


def _forecast_one(y: np.ndarray, order, params, horizon: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _quiet()
    v, _ = _prep(y)
    trend = "c" if order[1] == 0 else "n"
    res = SARIMAX(v, order=order, trend=trend).smooth(params)
    return np.asarray(res.forecast(horizon))


@dataclass
class RowARIMA:
    """An ARIMA per store-product on deseasonalised clean demand.

    The yearly season (52 weeks) is removed with the network seasonal factors first, because a per-row
    SARIMA with s = 52 cannot be estimated from 2-3 years of mostly small counts. The ARIMA then models what
    is left: the level, its drift and the week-to-week dependence. Out-of-stock weeks stay missing: the
    Kalman filter of the state-space ARIMA skips them. ``fit`` selects d (ADF) and p, q (AIC) for every row;
    ``predict`` reuses those parameters on newer data (no refit), so it is cheap to call every week."""
    n_jobs: int = -1
    table: pd.DataFrame = field(default=None, repr=False)
    _fits: list = field(default=None, repr=False)

    @staticmethod
    def _deseason(hist: History) -> np.ndarray:
        return hist.demand / season_of(season_factors(hist), hist.dates)[None, :]

    def fit(self, hist: History) -> "RowARIMA":
        from joblib import Parallel, delayed
        y = self._deseason(hist)
        self._fits = Parallel(n_jobs=self.n_jobs)(delayed(_select_one)(y[i]) for i in range(hist.n))
        self.table = pd.DataFrame([{k: v for k, v in f.items() if k != "params"} for f in self._fits])
        self.table["family"] = [family_of(o) if o else "too short: mean" for o in self.table["order"]]
        return self

    def predict(self, hist: History, horizon: int) -> np.ndarray:
        from joblib import Parallel, delayed
        y = self._deseason(hist)
        cnt = np.sum(~np.isnan(y), axis=1)
        mean = np.where(cnt > 0, np.nansum(y, axis=1) / np.maximum(cnt, 1), 0.0)
        idx = [i for i, f in enumerate(self._fits) if f["order"] is not None]
        chunks = np.array_split(np.array(idx), 12)
        parts = Parallel(n_jobs=self.n_jobs)(delayed(_forecast_chunk)(y[c], [self._fits[i] for i in c], horizon,
                                                                      mean[c]) for c in chunks if len(c))
        out = np.repeat(mean[:, None], horizon, axis=1)
        for c, part in zip([c for c in chunks if len(c)], parts):
            out[c] = part
        season = season_of(season_factors(hist), hist.future_dates(horizon))[None, :]
        return np.clip(out, 0, None) * season


def _forecast_chunk(ys: np.ndarray, fits: list[dict], horizon: int, means: np.ndarray) -> np.ndarray:
    out = np.empty((len(fits), horizon))
    for j, f in enumerate(fits):
        try:
            out[j] = _forecast_one(ys[j], f["order"], f["params"], horizon)
        except Exception:  # noqa: BLE001  (fall back to the row's mean)
            out[j] = means[j]
    return out


def error_whiteness(bt: pd.DataFrame, models: list[str], lag: int = 4) -> pd.DataFrame:
    """Ljung-Box on each store-product's sequence of 1-week-ahead forecast errors across the backtest origins.
    A good forecast leaves errors without pattern: the share of rows with p > 0.05 should be near 95%."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _quiet()
    g = bt[(bt.h == 1) & bt.model.isin(models)].dropna(subset=["demand"]).sort_values("origin")
    rows = []
    for (m, r), e in g.groupby(["model", "row"]):
        err = (e["forecast"] - e["demand"]).to_numpy()
        if len(err) < 2 * lag + 2 or np.std(err) == 0:
            continue
        p = float(acorr_ljungbox(err, lags=[lag])["lb_pvalue"].iloc[0])
        rows.append({"model": m, "row": r, "p": p, "mean error": err.mean()})
    d = pd.DataFrame(rows)
    return d.groupby("model").agg(rows_tested=("p", "size"), white_noise_share=("p", lambda x: (x > 0.05).mean()),
                                  median_p=("p", "median")).sort_values("white_noise_share", ascending=False)


def adf_all_rows(demand: pd.DataFrame, min_weeks: int = 30) -> np.ndarray:
    """ADF p-value of every store-product (clean weeks after launch, gaps interpolated)."""
    pvalues = []
    for row in demand.to_numpy(float):
        v = fill_gaps(_prep(row)[0])
        if len(v) >= min_weeks and np.std(v) > 0:
            pvalues.append(adf_test(v)["ADF p-value"])
    return np.array(pvalues)


def garch_on_arch_rows(arima_table: pd.DataFrame, demand: pd.DataFrame, season: np.ndarray) -> pd.DataFrame:
    """Fit GARCH(1,1) to the ARIMA residuals of every store-product with an ARCH effect, and re-test them."""
    out = []
    flagged = arima_table.dropna(subset=["aic"])
    for i in flagged[flagged["ARCH effect"].astype(bool)].index:
        order = arima_table.loc[i, "order"]
        v, _ = _prep(demand.iloc[i].to_numpy(float) / season)
        resid = np.asarray(fit_arima(v, order).resid)[order[1] + 1:]
        try:
            g, _ = fit_garch(resid)
        except Exception:  # noqa: BLE001  (a series too short or flat for GARCH is skipped)
            continue
        out.append({"row": i, "alpha+beta": g.params["alpha[1]"] + g.params["beta[1]"],
                    "ARCH left after GARCH": residual_tests(np.asarray(g.std_resid))["ARCH effect"]})
    return pd.DataFrame(out)


def json_safe(d: dict) -> dict:
    """Numbers as plain floats (numpy types cannot be written to JSON); booleans and text unchanged."""
    return {k: v if isinstance(v, (bool, str)) else float(v) for k, v in d.items()}
