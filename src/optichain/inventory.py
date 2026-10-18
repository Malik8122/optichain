"""Inventory simulation with the official VN2 rules, and ordering policies.

The rules (vn2_simulate.py, the organiser's code) each week:
    start  = end inventory last week + the order arriving this week
    sales  = min(start, demand)          lost sales = demand - sales (no backorders)
    end    = start - sales
    cost   = 0.2 EUR x end + 1 EUR x lost sales
An order placed before week k is on the shelf at the start of week k+2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class State:
    """Inventory position of every row at the end of a week."""
    end: np.ndarray      # units on the shelf
    t1: np.ndarray       # arriving at the start of next week
    t2: np.ndarray       # arriving the week after

    @classmethod
    def from_initial(cls, initial: pd.DataFrame) -> "State":
        return cls(initial["End Inventory"].to_numpy(float), initial["In Transit W+1"].to_numpy(float),
                   initial["In Transit W+2"].to_numpy(float))

    def copy(self) -> "State":
        return State(self.end.copy(), self.t1.copy(), self.t2.copy())

    @property
    def position(self) -> np.ndarray:
        return self.end + self.t1 + self.t2


@dataclass
class Costs:
    holding: float = 0.2
    shortage: float = 1.0

    @property
    def critical_ratio(self) -> float:
        """Newsvendor service level: shortage / (shortage + holding) = 0.833 for VN2."""
        return self.shortage / (self.shortage + self.holding)


def play_week(prev: State, demand: np.ndarray, orders: np.ndarray, costs: Costs) -> tuple[State, dict]:
    """One week with the official arithmetic."""
    start = prev.end + prev.t1
    sales = np.minimum(start, demand)
    lost = demand - sales
    end = start - sales
    new = State(end, prev.t2.copy(), np.asarray(orders, float))
    rec = {"start": start, "sales": sales, "lost": lost, "end": end, "orders": np.asarray(orders, float),
           "holding": end * costs.holding, "shortage": lost * costs.shortage}
    return new, rec


def simulate(initial: State, demand: np.ndarray, orders: list[np.ndarray], costs: Costs) -> tuple[pd.DataFrame, list[dict], State]:
    """Play the weeks in ``demand`` (rows x weeks). ``orders[w]`` is placed before week w+1; missing = 0.
    Returns the weekly totals, the per-row records and the final state."""
    state, recs = initial.copy(), []
    n, W = demand.shape
    for w in range(W):
        o = orders[w] if w < len(orders) and orders[w] is not None else np.zeros(n)
        state, rec = play_week(state, demand[:, w], o, costs)
        recs.append(rec)
    weekly = pd.DataFrame([{k: float(np.sum(v)) for k, v in r.items()} for r in recs], index=pd.RangeIndex(1, W + 1, name="week"))
    weekly["cost"] = weekly["holding"] + weekly["shortage"]
    return weekly, recs, state


# ----------------------------------------------------------------- policies
def order_up_to(mu: np.ndarray, state: State, coverage: float) -> np.ndarray:
    """Benchmark policy: bring stock + in transit up to ``coverage`` weeks of forecast (fractions allowed)."""
    full = int(np.floor(coverage))
    if mu.shape[1] < np.ceil(coverage):  # extend a short forecast with its last week
        mu = np.concatenate([mu, np.repeat(mu[:, -1:], int(np.ceil(coverage)) - mu.shape[1], axis=1)], axis=1)
    target = mu[:, :full].sum(axis=1)
    if coverage > full:
        target = target + (coverage - full) * mu[:, full]
    return np.clip(np.round(target - state.position), 0, None).astype(int)


def sample_demand(mu: np.ndarray, k: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Demand paths (rows, samples, weeks): negative binomial with mean mu and variance k * mu
    (Poisson when k is ~1). Captures that low sellers are noisy and that some rows are spikier than others."""
    n, H = mu.shape
    mu = np.maximum(mu, 0)[:, None, :]
    kk = np.broadcast_to(np.asarray(k, float)[:, None, None], (n, 1, H))
    shape = (n, samples, H)
    # Negative binomial = Poisson whose rate is gamma-distributed (shape r, mean mu): Var = mu + mu^2 / r = k * mu
    pois = (kk <= 1.02) | (mu <= 1e-9)
    r = np.where(pois, 1.0, mu / np.maximum(kk - 1, 1e-6))
    rate = rng.gamma(np.broadcast_to(r, shape), np.broadcast_to(np.where(pois, 1.0, mu / r), shape))
    rate = np.where(np.broadcast_to(pois, shape), np.broadcast_to(mu, shape), rate)
    return rng.poisson(rate)


def project(state: State, paths: np.ndarray) -> np.ndarray:
    """Stock on the shelf at the start of the arrival week (before the new order lands), per path:
    play the next two weeks with lost sales, using the stock and orders already on their way."""
    end0 = state.end[:, None]
    after1 = np.maximum(end0 + state.t1[:, None] - paths[:, :, 0], 0)
    after2 = np.maximum(after1 + state.t2[:, None] - paths[:, :, 1], 0)
    return after2


def newsvendor_orders(mu: np.ndarray, k: np.ndarray, state: State, service: float, samples: int = 1000,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Order Q so that, with probability ``service``, the stock in the arrival week covers its demand.

    Y = demand in the arrival week - stock left when the order lands  (simulated, lost sales included)
    Q = service-quantile of Y, at least 0.
    With cost 0.2 per unit left and 1 per unit short, the textbook service level is 1/1.2 = 0.833."""
    rng = rng or np.random.default_rng(0)
    paths = sample_demand(mu[:, :3], k, samples, rng)
    y = paths[:, :, 2] - project(state, paths)
    q = np.quantile(y, service, axis=1, method="inverted_cdf")
    return np.clip(q, 0, None).astype(int)


def stockout_outlook(mu: np.ndarray, k: np.ndarray, state: State, order: np.ndarray, samples: int = 1000,
                     rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Risk view for the next three weeks if ``order`` is placed now: chance of running out each week and
    expected stock left at the end of each week."""
    rng = rng or np.random.default_rng(0)
    paths = sample_demand(mu[:, :3], k, samples, rng)
    arrivals = [state.t1, state.t2, order]
    stock = np.repeat(state.end[:, None], samples, axis=1)
    p_out, end_mean, lost_mean = [], [], []
    for w in range(3):
        start = stock + arrivals[w][:, None]
        lost = np.maximum(paths[:, :, w] - start, 0)
        stock = np.maximum(start - paths[:, :, w], 0)
        p_out.append((lost > 0).mean(axis=1))
        end_mean.append(stock.mean(axis=1))
        lost_mean.append(lost.mean(axis=1))
    return {"p_stockout": np.array(p_out).T, "end_stock": np.array(end_mean).T, "lost": np.array(lost_mean).T}


# ----------------------------------------------------------------- competition replay
def replay(initial: State, demand: np.ndarray, forecasts: list[np.ndarray], k: np.ndarray | None,
           costs: Costs, policy: str = "newsvendor", service: float = 0.833, coverage: float = 4.0,
           n_orders: int | None = None, samples: int = 1000, seed: int = 42):
    """Play len(demand) weeks. ``forecasts[r]`` (rows x >=3) is the forecast made before week r+1, from data
    up to week r. Only the first ``n_orders`` rounds order (the rest order 0, as in VN2 weeks 7-8)."""
    n, W = demand.shape
    n_orders = W if n_orders is None else n_orders
    state, recs, orders = initial.copy(), [], []
    for w in range(W):
        if w < n_orders:
            mu = forecasts[w]
            if policy == "newsvendor":
                o = newsvendor_orders(mu, k, state, service, samples, np.random.default_rng(seed + w))
            else:
                o = order_up_to(mu, state, coverage)
        else:
            o = np.zeros(n, int)
        orders.append(o)
        state, rec = play_week(state, demand[:, w], o, costs)
        recs.append(rec)
    weekly = pd.DataFrame([{k_: float(np.sum(v)) for k_, v in r.items()} for r in recs],
                          index=pd.RangeIndex(1, W + 1, name="week"))
    weekly["cost"] = weekly["holding"] + weekly["shortage"]
    return weekly, recs, orders, state


def replay_grid(initial: State, demand: np.ndarray, forecasts: list[np.ndarray], k: np.ndarray, costs: Costs,
                services: list[float], n_orders: int | None = None, samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """``replay`` for many service levels at once. The demand paths of a round do not depend on the service
    level, so they are drawn once and shared (common random numbers: differences come from the policy only).
    Returns one row per service level and week."""
    n, W = demand.shape
    n_orders = W if n_orders is None else n_orders
    S = len(services)
    end = np.repeat(initial.end[None], S, 0)
    t1 = np.repeat(initial.t1[None], S, 0)
    t2 = np.repeat(initial.t2[None], S, 0)
    rows = []
    for w in range(W):
        if w < n_orders:
            paths = sample_demand(forecasts[w][:, :3], k, samples, np.random.default_rng(seed + w))
            orders = np.zeros((S, n))
            for i, s in enumerate(services):
                x = project(State(end[i], t1[i], t2[i]), paths)
                q = np.quantile(paths[:, :, 2] - x, s, axis=1, method="inverted_cdf")
                orders[i] = np.clip(q, 0, None)
        else:
            orders = np.zeros((S, n))
        start = end + t1
        sales = np.minimum(start, demand[None, :, w])
        lost = demand[None, :, w] - sales
        end, t1, t2 = start - sales, t2, orders
        for i, s in enumerate(services):
            rows.append({"service": s, "week": w + 1, "holding": end[i].sum() * costs.holding,
                         "shortage": lost[i].sum() * costs.shortage, "end": end[i].sum(), "lost": lost[i].sum(),
                         "orders": orders[i].sum()})
    out = pd.DataFrame(rows)
    out["cost"] = out["holding"] + out["shortage"]
    return out
