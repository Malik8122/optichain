"""Glue between forecasts and stock decisions, shared by the notebook and the dashboard.

Round r (r = 1..9) is the decision taken before week r, with data up to week r-1:
rounds 1-6 are the official VN2 orders, rounds 7-8 continue the plan (VN2 itself stops ordering there,
so its score is computed with zero orders), and round 9 is the next order after the data ends.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .inventory import Costs, State, newsvendor_orders, play_week, stockout_outlook
from .models import GlobalLGBM, History, forecast_all
from .timeseries import RowARIMA


def round_forecasts(hist: History, t0: int, models: list[str], ensemble_of: list[str], n_rounds: int = 9,
                    horizon: int = 4, seed: int = 42, progress=None) -> pd.DataFrame:
    """Forecasts of every model for every round. LightGBM is refitted each round on the data known then;
    the ARIMA orders and parameters are selected once, at round 1, and then run on each round's newer data."""
    rows, arima = [], None
    for r in range(1, n_rounds + 1):
        o = t0 + r - 1                      # last week known before round r
        h = hist.upto(o)
        lgbm = GlobalLGBM(horizon, seed=seed).fit(h) if "lightgbm" in models else None
        if "arima" in models and arima is None:
            arima = RowARIMA().fit(h)
        fc = forecast_all(h, horizon, models, lgbm, ensemble_of, arima)
        for m, f in fc.items():
            for k in range(horizon):
                rows.append(pd.DataFrame({"model": m, "round": r, "h": k + 1, "week": r + k, "row": np.arange(hist.n),
                                          "forecast": f[:, k]}))
        if progress:
            progress(r, n_rounds)
    return pd.concat(rows, ignore_index=True)


def forecast_matrix(rounds: pd.DataFrame, model: str, r: int) -> np.ndarray:
    g = rounds[(rounds["model"] == model) & (rounds["round"] == r)]
    return g.pivot(index="row", columns="h", values="forecast").sort_index().to_numpy()


def k_vector(disp: pd.DataFrame, model: str) -> np.ndarray:
    return disp[disp["model"] == model].sort_values("row")["k"].to_numpy()


def nb_quantile(mu: np.ndarray, k: np.ndarray, q: float) -> np.ndarray:
    """Quantile of the demand distribution used by the policy (negative binomial, variance k * mu)."""
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    k = np.broadcast_to(np.asarray(k, float), mu.shape)
    r = mu / np.maximum(k - 1, 1e-6)
    nb = stats.nbinom.ppf(q, r, 1 / np.maximum(k, 1 + 1e-6))
    po = stats.poisson.ppf(q, mu)
    return np.where(k <= 1.02, po, nb)


@dataclass
class PlanResult:
    weekly: pd.DataFrame          # official VN2 rules: orders 1-6, zero orders before weeks 7-8
    records: list[dict]           # per-row records of each official week
    orders: list[np.ndarray]      # orders of rounds 1..8 (7-8 from the continued plan)
    end_state: State              # end of week 8 if the plan keeps ordering (rounds 7-8 placed)
    next_order: np.ndarray        # round 9: the order to place now, arrives in week 11
    outlook: dict                 # stockout risk for weeks 9-11

    @property
    def score(self) -> float:
        return round(float(self.weekly.loc[3:8, "cost"].sum()), 1)

    @property
    def setup(self) -> float:
        return round(float(self.weekly.loc[1:2, "cost"].sum()), 1)


def run_plan(initial: State, demand: np.ndarray, rounds: pd.DataFrame, k: np.ndarray, model: str, service: float,
             costs: Costs, samples: int = 1000, seed: int = 42) -> PlanResult:
    """Play the eight VN2 weeks with the newsvendor policy, then look ahead."""
    n, W = demand.shape
    official = cont = initial.copy()
    recs, orders = [], []
    for w in range(W):
        mu = forecast_matrix(rounds, model, w + 1)
        o = newsvendor_orders(mu, k, cont, service, samples, np.random.default_rng(seed + w))
        orders.append(o)
        if w < 6:   # official rounds: both tracks are the same
            official, rec = play_week(official, demand[:, w], o, costs)
            cont = official
        else:       # VN2 stops ordering; the continued plan keeps going
            official, rec = play_week(official, demand[:, w], np.zeros(n), costs)
            cont, _ = play_week(cont, demand[:, w], o, costs)
        recs.append(rec)
    weekly = pd.DataFrame([{key: float(np.sum(v)) for key, v in r.items()} for r in recs],
                          index=pd.RangeIndex(1, W + 1, name="week"))
    weekly["cost"] = weekly["holding"] + weekly["shortage"]
    mu9 = forecast_matrix(rounds, model, W + 1)
    nxt = newsvendor_orders(mu9, k, cont, service, samples, np.random.default_rng(seed + W))
    outlook = stockout_outlook(mu9, k, cont, nxt, samples, np.random.default_rng(seed + 100))
    return PlanResult(weekly, recs, orders, cont, nxt, outlook)


def backtest_matrices(bt: pd.DataFrame, model: str, origins: list[int]) -> list[np.ndarray]:
    """The backtest forecasts of one model, one (rows x horizon) matrix per origin."""
    sub = bt[(bt["model"] == model) & bt["origin"].isin(origins)]
    return [g.pivot(index="row", columns="h", values="forecast").sort_index().to_numpy() for _, g in sub.groupby("origin")]


def tune_service_levels(bt: pd.DataFrame, origins: list[int], sales: np.ndarray, disp: pd.DataFrame, costs: Costs,
                        models: list[str], grid: list[float], coverages=(2.5, 3.0, 3.5, 4.0), samples: int = 1000,
                        seed: int = 42, warmup: int = 3) -> pd.DataFrame:
    """Replay the past weeks after ``origins`` for every model: the newsvendor policy at each service level and the
    benchmark's order-up-to rule at each coverage. Returns the total cost of the scored weeks for each setting."""
    from .inventory import replay, replay_grid
    demand = sales[:, [o + 1 for o in origins]]
    rows = []
    for m in models:
        fcs = backtest_matrices(bt, m, origins)
        start = State(np.round(fcs[0][:, 0]), np.round(fcs[0][:, 1]), np.round(fcs[0][:, 2]))  # neutral starting stock
        k = k_vector(disp, m)
        g = replay_grid(start, demand, fcs, k, costs, grid, samples=samples, seed=seed)
        for s, c in g[g.week > warmup].groupby("service")["cost"].sum().items():
            rows.append({"model": m, "policy": "newsvendor", "param": s, "cost": c})
        for cov in coverages:
            w, *_ = replay(start, demand, fcs, k, costs, policy="order_up_to", coverage=cov)
            rows.append({"model": m, "policy": "order_up_to", "param": cov, "cost": w.iloc[warmup:]["cost"].sum()})
    return pd.DataFrame(rows)


def best_settings(tuning: pd.DataFrame, policy: str) -> pd.DataFrame:
    """The cheapest setting of each model for one policy."""
    t = tuning[tuning["policy"] == policy]
    return t.loc[t.groupby("model")["cost"].idxmin()].set_index("model")


def score_competition(initial: State, demand: np.ndarray, rounds: pd.DataFrame, disp: pd.DataFrame, costs: Costs,
                      models: list[str], labels: dict, best_service: pd.DataFrame, best_cov: pd.DataFrame,
                      grid: list[float], samples: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Play the eight VN2 weeks for every model with the settings tuned on the past. Also returns the cost at every
    service level (to show what hindsight would have chosen)."""
    from .inventory import replay, replay_grid
    scores, grid_rows = [], []
    for m in models:
        k = k_vector(disp, m)
        fcs = [forecast_matrix(rounds, m, r) for r in range(1, 7)]
        g = replay_grid(initial, demand, fcs, k, costs, grid, n_orders=6, samples=samples, seed=seed)
        by_s = g[g.week >= 3].groupby("service")["cost"].sum().round(1)
        s_star = float(best_service.loc[m, "param"])
        w_cov, *_ = replay(initial, demand, fcs, k, costs, policy="order_up_to", coverage=float(best_cov.loc[m, "param"]),
                           n_orders=6)
        scores.append({"model": m, "label": labels[m], "service (tuned on past)": s_star, "VN2 score €": by_s.loc[s_star],
                       "order-up-to score €": round(w_cov.loc[3:8, "cost"].sum(), 1), "hindsight best €": by_s.min(),
                       "hindsight service": by_s.idxmin()})
        grid_rows += [{"model": m, "policy": "competition", "param": s, "cost": c} for s, c in by_s.items()]
    return pd.DataFrame(scores).sort_values("VN2 score €").reset_index(drop=True), pd.DataFrame(grid_rows)


def outlook_table(index: pd.MultiIndex, patterns: np.ndarray, plan: PlanResult, next_forecast: np.ndarray) -> pd.DataFrame:
    """One line per store-product: stock now, stock on its way, forecast, the order to place now and the risk of an
    empty shelf in each of the next three weeks."""
    return pd.DataFrame({
        "store": index.get_level_values("Store"), "product": index.get_level_values("Product"), "pattern": patterns,
        "on shelf": plan.end_state.end, "arriving wk 9": plan.end_state.t1, "arriving wk 10": plan.end_state.t2,
        "forecast wk 9-11": next_forecast[:, :3].sum(axis=1), "order now": plan.next_order,
        "risk wk 9": plan.outlook["p_stockout"][:, 0], "risk wk 10": plan.outlook["p_stockout"][:, 1],
        "risk wk 11": plan.outlook["p_stockout"][:, 2],
    })
