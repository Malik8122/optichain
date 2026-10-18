"""Single-item periodic-review inventory simulation (HYPOTHETICAL assumptions).

Model of the world
------------------
- Daily review; unmet demand is lost (no backorders), so on-hand never goes negative.
- Deterministic lead time L: an order placed at the end of day t arrives at the start of day t+L.
- Observed M5 sales are used as demand. They are censored by Walmart's own stockouts,
  so true demand is unknown: a stated limitation.

Daily event order (fixed, tested)
1. receive orders due today   2. serve demand (excess lost)   3. holding cost on end-of-day stock
4. review: IP = on-hand + on-order; if IP <= ROP_t order up to S_t (rounded up to case pack)
5. ordering cost if an order was placed

All policies share one SimConfig, so comparisons differ only in how ROP_t and S_t are set.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data_pipeline import d_to_date


@dataclass(frozen=True)
class SimConfig:
    lead_time: int = 4
    cover_days: int = 7
    csl: float = 0.95  # target cycle service level -> safety factor z
    holding_rate_annual: float = 0.25
    shortage_penalty_rate: float = 0.5
    order_cost: float = 5.0
    case_pack: int = 1
    review_days: int = 1

    def __post_init__(self):
        if self.lead_time < 1 or self.review_days < 1 or self.case_pack < 1:
            raise ValueError("lead_time, review_days and case_pack must be >= 1")
        if not 0 < self.csl < 1:
            raise ValueError("csl must be in (0, 1)")

    @property
    def z(self) -> float:
        return float(norm.ppf(self.csl))

    @classmethod
    def from_assumptions(cls, sa) -> "SimConfig":
        return cls(lead_time=sa.lead_time_days, cover_days=sa.cover_days, csl=sa.target_csl,
                   holding_rate_annual=sa.holding_rate_annual, shortage_penalty_rate=sa.shortage_penalty_rate,
                   order_cost=sa.order_cost, case_pack=sa.case_pack, review_days=sa.review_days)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)


class Policy(Protocol):
    name: str

    def targets(self, d: int) -> tuple[float, float]:
        """(ROP_d, S_d) using only information available at the end of day d."""


def _latest_origin(origins: np.ndarray, d: int) -> int:
    i = np.searchsorted(origins, d, side="right") - 1
    if i < 0:
        raise ValueError(f"no forecast origin on or before d_{d}")
    return int(origins[i])


class BaselinePolicy:
    """Policy 0: static (s, S) from the trailing-window mean/std at each origin.

    ROP = mu*L + z*sigma_d*sqrt(L);  S = mu*(L + cover) + SS.  mu, sigma_d come from the
    ``window`` days up to the latest origin <= d, so they change only at origins.
    """

    def __init__(self, y: pd.Series, origins: list[int], cfg: SimConfig, window: int = 28, name: str = "P0 baseline"):
        self.cfg, self.name = cfg, name
        self.origins = np.array(sorted(origins))
        self.stats = {}
        for o in self.origins:
            hist = y.loc[:d_to_date(o)].iloc[-window:]
            self.stats[int(o)] = (float(hist.mean()), float(hist.std(ddof=1)))

    def targets(self, d):
        mu, sd = self.stats[_latest_origin(self.origins, d)]
        L, c = self.cfg.lead_time, self.cfg.cover_days
        ss = self.cfg.z * sd * math.sqrt(L)
        return mu * L + ss, mu * (L + c) + ss


class ForecastPolicy:
    """Policy 1: ROP_d = sum of forecasts d+1..d+L + SS;  S_d = sum d+1..d+L+cover + SS.

    Uses the forecast issued at the latest origin <= d.  SS = z * sigma_e * sqrt(L), where
    sigma_e is the RMS of that model's in-sample one-step residuals at the origin.
    """

    def __init__(self, forecasts: pd.DataFrame, cfg: SimConfig, name: str = "P1 forecast"):
        self.cfg, self.name = cfg, name
        self.origins = np.array(sorted(forecasts["origin_d"].unique()))
        self.paths = {int(o): g.set_index("d")["yhat"] for o, g in forecasts.groupby("origin_d")}
        self.sigma = forecasts.groupby("origin_d")["sigma_resid"].first().to_dict()

    def targets(self, d):
        o = _latest_origin(self.origins, d)
        L, c = self.cfg.lead_time, self.cfg.cover_days
        path = self.paths[o]
        need = range(d + 1, d + L + c + 1)
        missing = [x for x in need if x not in path.index]
        if missing:
            raise ValueError(f"forecast from origin d_{o} does not reach d_{missing[-1]}; extend the horizon")
        f = path.loc[list(need)].to_numpy()
        ss = self.safety_stock(o)
        return f[:L].sum() + ss, f.sum() + ss

    def safety_stock(self, o: int) -> float:
        return self.cfg.z * self.sigma[o] * math.sqrt(self.cfg.lead_time)


class EmpiricalForecastPolicy(ForecastPolicy):
    """Policy 2: same targets as P1, but SS = empirical CSL-quantile of the model's L-day summed
    in-sample one-step errors at that origin (no normality or independence assumption).

    ``quantiles``: origin_d, L, q, value (from evaluation.residual_sum_quantiles).
    """

    def __init__(self, forecasts: pd.DataFrame, quantiles: pd.DataFrame, cfg: SimConfig, name: str = "P2 empirical"):
        super().__init__(forecasts, cfg, name)
        q = quantiles[(quantiles["L"] == cfg.lead_time) & np.isclose(quantiles["q"], cfg.csl)]
        self.ss = {}
        for o in self.origins:
            row = q[q["origin_d"] == o]
            if row.empty:
                raise ValueError(f"no residual quantile for L={cfg.lead_time}, CSL={cfg.csl} at origin d_{o}")
            self.ss[int(o)] = max(0.0, float(row["value"].iloc[0]))

    def safety_stock(self, o: int) -> float:
        return self.ss[o]


def perfect_foresight_forecasts(y: pd.Series, origins: list[int], horizon: int) -> pd.DataFrame:
    """Forecast = actual demand (upper bound on forecast value), sigma_e = 0.

    Days beyond the end of the data are filled with the trailing 28-day mean; this only
    affects orders that could not arrive inside the simulated horizon anyway.
    """
    last_d = int((y.index[-1] - d_to_date(1)).days) + 1
    fill = float(y.iloc[-28:].mean())
    rows = []
    for o in origins:
        for h in range(1, horizon + 1):
            d = o + h
            rows.append({"origin_d": o, "d": d, "h": h, "sigma_resid": 0.0,
                         "yhat": float(y.loc[d_to_date(d)]) if d <= last_d else fill})
    return pd.DataFrame(rows)


@dataclass
class SimResult:
    policy: str
    trace: pd.DataFrame
    metrics: dict


def simulate(demand: pd.Series, price: pd.Series, policy: Policy, cfg: SimConfig,
             start_on_hand: float, warmup_days: int = 0) -> SimResult:
    """Run the daily loop over ``demand`` (index = day number d, consecutive)."""
    days = demand.index.to_numpy()
    if not np.array_equal(days, np.arange(days[0], days[0] + len(days))):
        raise ValueError("demand must be indexed by consecutive day numbers")
    h_daily = cfg.holding_rate_annual / 365

    on_hand = float(start_on_hand)
    pipeline: dict[int, float] = {}
    rows = []
    for i, d in enumerate(days):
        d = int(d)
        received = pipeline.pop(d, 0.0)
        on_hand += received
        dem = float(demand.iloc[i])
        sold = min(on_hand, dem)
        lost = dem - sold
        on_hand -= sold
        p = float(price.loc[d])
        holding = on_hand * h_daily * p

        rop, S = policy.targets(d)
        on_order = sum(pipeline.values())
        ip = on_hand + on_order
        q = 0.0
        if (i % cfg.review_days == 0) and ip <= rop and S > ip:
            q = math.ceil((S - ip) / cfg.case_pack) * cfg.case_pack
            pipeline[d + cfg.lead_time] = pipeline.get(d + cfg.lead_time, 0.0) + q
        rows.append({
            "d": d, "date": d_to_date(d), "demand": dem, "received": received, "sold": sold,
            "lost": lost, "on_hand": on_hand, "on_order": on_order + q, "inv_position": ip + q,
            "rop": rop, "order_up_to": S, "order_qty": q, "price": p,
            "holding_cost": holding, "ordering_cost": cfg.order_cost if q > 0 else 0.0,
            "shortage_cost": lost * cfg.shortage_penalty_rate * p,
        })
    trace = pd.DataFrame(rows)
    trace.insert(0, "policy", policy.name)
    return SimResult(policy.name, trace, inventory_metrics(trace.iloc[warmup_days:]))


def inventory_metrics(trace: pd.DataFrame) -> dict:
    demand = trace["demand"].sum()
    orders = trace["order_qty"] > 0
    costs = {k: float(trace[k].sum()) for k in ("holding_cost", "ordering_cost", "shortage_cost")}
    return {
        "days": len(trace),
        "demand_units": float(demand),
        "sold_units": float(trace["sold"].sum()),
        "lost_units": float(trace["lost"].sum()),
        # fill rate: share of demanded units served from stock (NaN if no demand)
        "fill_rate": float(trace["sold"].sum() / demand) if demand > 0 else np.nan,
        # cycle service: share of days with no lost sales
        "cycle_service": float((trace["lost"] == 0).mean()),
        "stockout_days": int((trace["lost"] > 0).sum()),
        "avg_on_hand": float(trace["on_hand"].mean()),
        "peak_on_hand": float(trace["on_hand"].max()),
        "n_orders": int(orders.sum()),
        "avg_order_qty": float(trace.loc[orders, "order_qty"].mean()) if orders.any() else 0.0,
        **costs,
        "total_cost": sum(costs.values()),
    }


def config_dict(cfg: SimConfig) -> dict:
    return asdict(cfg) | {"z": cfg.z}
