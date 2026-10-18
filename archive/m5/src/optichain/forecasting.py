"""Forecasting models behind one interface.

Every model is fitted on a date-indexed daily series that ends at the forecast origin and
returns point forecasts plus prediction intervals for the next ``h`` days. Forecasts and
lower bounds are clipped at zero (demand cannot be negative); unrounded values are kept.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_LEVELS = (0.80, 0.95)


class Forecaster(ABC):
    name: str = "base"

    def fit(self, y: pd.Series) -> "Forecaster":
        if not isinstance(y.index, pd.DatetimeIndex) or y.index.freq is None:
            raise ValueError("y must have a daily DatetimeIndex with freq set")
        if y.isna().any():
            raise ValueError("y contains NaN")
        self.y_ = y.astype(float)
        self._fit(self.y_)
        resid = self.residuals().dropna()
        # RMS of one-step residuals (includes any bias), used for interval widths
        self.sigma_ = float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
        return self

    def predict(self, h: int, levels: tuple[float, ...] = DEFAULT_LEVELS) -> pd.DataFrame:
        if h < 1:
            raise ValueError("h must be >= 1")
        steps = np.arange(1, h + 1)
        idx = pd.date_range(self.y_.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
        mean, bands = self._forecast(steps, levels)
        out = pd.DataFrame({"h": steps, "yhat": np.clip(mean, 0, None)}, index=idx)
        for lv, (lo, hi) in bands.items():
            tag = int(round(lv * 100))
            out[f"lo{tag}"] = np.clip(lo, 0, None)
            out[f"hi{tag}"] = np.clip(hi, 0, None)
        out.index.name = "date"
        return out

    def _forecast(self, steps: np.ndarray, levels) -> tuple[np.ndarray, dict]:
        """Point forecast + {level: (lo, hi)}; default = normal bands from _point/_sd."""
        mean, sd = self._point(steps), self._sd(steps)
        return mean, {lv: (mean - norm.ppf(0.5 + lv / 2) * sd, mean + norm.ppf(0.5 + lv / 2) * sd) for lv in levels}

    @abstractmethod
    def _fit(self, y: pd.Series) -> None: ...

    @abstractmethod
    def _point(self, steps: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def residuals(self) -> pd.Series:
        """In-sample one-step-ahead residuals aligned to y (NaN where undefined)."""

    def _sd(self, steps: np.ndarray) -> np.ndarray:
        return np.full(len(steps), self.sigma_)

    def params(self) -> dict:
        return {"model": self.name, "sigma": self.sigma_, "n_train": len(self.y_)}


class Naive(Forecaster):
    """Every future day = last observed value."""
    name = "naive"

    def _fit(self, y):
        self.last_ = float(y.iloc[-1])

    def _point(self, steps):
        return np.full(len(steps), self.last_)

    def residuals(self):
        return self.y_.diff()

    def _sd(self, steps):
        return self.sigma_ * np.sqrt(steps)


class SeasonalNaive(Forecaster):
    """Every future day = the same weekday in the last observed week."""
    name = "snaive"

    def __init__(self, m: int = 7):
        self.m = m

    def _fit(self, y):
        if len(y) < self.m:
            raise ValueError(f"need at least {self.m} observations")
        self.last_season_ = y.iloc[-self.m:].to_numpy()

    def _point(self, steps):
        return self.last_season_[(steps - 1) % self.m]

    def residuals(self):
        return self.y_ - self.y_.shift(self.m)

    def _sd(self, steps):
        return self.sigma_ * np.sqrt((steps - 1) // self.m + 1)

    def params(self):
        return super().params() | {"m": self.m}


class MovingAverage(Forecaster):
    """Every future day = mean of the last k days (flat forecast)."""

    def __init__(self, k: int = 28):
        self.k = k
        self.name = f"ma{k}"

    def _fit(self, y):
        self.level_ = float(y.iloc[-self.k:].mean())

    def _point(self, steps):
        return np.full(len(steps), self.level_)

    def residuals(self):
        return self.y_ - self.y_.rolling(self.k).mean().shift(1)

    def params(self):
        return super().params() | {"k": self.k, "level": self.level_}


from .models_stat import ETS, Croston, SarimaFamily  # noqa: E402  (needs Forecaster defined above)

# Every factory accepts optional context: ident (identification record for the origin), grid.
MODEL_REGISTRY: dict[str, callable] = {
    "naive": lambda **_: Naive(),
    "snaive": lambda **_: SeasonalNaive(),
    "ma28": lambda **_: MovingAverage(28),
    "ets": lambda **kw: ETS(**kw),
    "arima": lambda **kw: SarimaFamily(seasonal=False, **kw),
    "sarima": lambda **kw: SarimaFamily(seasonal=True, **kw),
    "croston": lambda **kw: Croston(**kw),
}


def make_model(name: str, **context) -> Forecaster:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}") from None
    return factory(**context)
