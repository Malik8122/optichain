"""Orchestrates policy comparisons for one item (shared by scripts and the dashboard)."""
from __future__ import annotations

import math

import pandas as pd

from .data_pipeline import d_to_date
from .inventory import (
    BaselinePolicy,
    EmpiricalForecastPolicy,
    ForecastPolicy,
    SimConfig,
    perfect_foresight_forecasts,
    simulate,
)


def select_model_per_origin(metrics: pd.DataFrame, origins: list[int], horizon: int,
                            default: str = "snaive", criterion: str = "rmsse") -> pd.DataFrame:
    """At each origin, pick the model with the best mean ``criterion`` over folds already
    finished by that origin (test window end <= origin). No finished folds -> ``default``."""
    rows = []
    for o in sorted(origins):
        done = metrics[metrics["origin_d"] + horizon <= o]
        if done.empty:
            rows.append({"origin_d": o, "model": default, "basis": "default (no completed folds)"})
        else:
            means = done.groupby("model")[criterion].mean()
            rows.append({"origin_d": o, "model": means.idxmin(),
                         "basis": f"best mean {criterion} on {done['origin_d'].nunique()} completed fold(s)"})
    return pd.DataFrame(rows)


def stitch_forecasts(fc_item: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    parts = [fc_item[(fc_item["origin_d"] == r.origin_d) & (fc_item["model"] == r.model)] for r in choice.itertuples()]
    return pd.concat(parts, ignore_index=True)


def demand_by_day(y: pd.Series, first_d: int, last_d: int) -> pd.Series:
    days = range(first_d, last_d + 1)
    return pd.Series([float(y.loc[d_to_date(d)]) for d in days], index=list(days), name="demand")


def start_on_hand(y: pd.Series, first_d: int, cfg: SimConfig, window: int = 28) -> float:
    """Same opening stock for every policy: trailing mean demand x (L + cover), whole units."""
    mu = float(y.loc[:d_to_date(first_d - 1)].iloc[-window:].mean())
    return float(math.ceil(mu * (cfg.lead_time + cfg.cover_days)))


def stitch_quantiles(q_item: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    parts = [q_item[(q_item["origin_d"] == r.origin_d) & (q_item["model"] == r.model)] for r in choice.itertuples()]
    return pd.concat(parts, ignore_index=True)


def build_policies(y, fc_item, metrics_item, origins, horizon, cfg: SimConfig, sim,
                   q_item: pd.DataFrame | None = None) -> tuple[list, pd.DataFrame]:
    if sim.forecast_model == "auto":
        choice = select_model_per_origin(metrics_item, origins, horizon, default=sim.default_model)
    else:
        choice = pd.DataFrame({"origin_d": sorted(origins), "model": sim.forecast_model, "basis": "fixed in config"})
    policies = [
        BaselinePolicy(y, origins, cfg, window=sim.baseline_window),
        ForecastPolicy(stitch_forecasts(fc_item, choice), cfg, name="P1 forecast"),
    ]
    if q_item is not None and len(q_item):
        policies.append(EmpiricalForecastPolicy(stitch_forecasts(fc_item, choice), stitch_quantiles(q_item, choice), cfg))
    if sim.include_fixed_models:
        for m in sorted(fc_item["model"].unique()):
            policies.append(ForecastPolicy(fc_item[fc_item["model"] == m], cfg, name=f"P1 [{m}]"))
    h_ext = int(fc_item["h"].max())
    policies.append(ForecastPolicy(perfect_foresight_forecasts(y, origins, h_ext), cfg, name="Perfect foresight"))
    return policies, choice


def run_policies(y, price, fc_item, metrics_item, origins, horizon, max_day, cfg: SimConfig, sim, q_item=None):
    first_d = min(origins) + 1
    demand = demand_by_day(y, first_d, max_day)
    s0 = start_on_hand(y, first_d, cfg, sim.baseline_window)
    policies, choice = build_policies(y, fc_item, metrics_item, origins, horizon, cfg, sim, q_item)
    results = [simulate(demand, price, p, cfg, s0, warmup_days=sim.warmup_days) for p in policies]
    return results, choice


def run_item(item, y, price, fc_item, metrics_item, origins, horizon, max_day, cfg: SimConfig, sim,
             q_item=None) -> dict:
    results, choice = run_policies(y, price, fc_item, metrics_item, origins, horizon, max_day, cfg, sim, q_item)
    metrics = pd.DataFrame([{"item_id": item, "policy": r.policy, **r.metrics} for r in results])
    trace = pd.concat([r.trace.assign(item_id=item) for r in results], ignore_index=True)
    return {"metrics": metrics, "trace": trace, "model_choice": choice}


def sensitivity_grid(item, y, price, fc_item, metrics_item, origins, horizon, max_day, cfg: SimConfig, sim,
                     grid: dict[str, list[float]], q_item=None) -> pd.DataFrame:
    rows = []
    for L in grid["lead_time_days"]:
        for csl in grid["target_csl"]:
            c = cfg.with_(lead_time=int(L), csl=float(csl))
            res, _ = run_policies(y, price, fc_item, metrics_item, origins, horizon, max_day, c, sim, q_item)
            rows += [{"item_id": item, "lead_time": int(L), "target_csl": float(csl), "policy": r.policy, **r.metrics}
                     for r in res if not r.policy.startswith("P1 [")]
    return pd.DataFrame(rows)


MAIN_POLICIES = ("P0 baseline", "P1 forecast", "P2 empirical", "Perfect foresight")


def service_frontier(item, y, price, fc_item, metrics_item, origins, horizon, max_day, cfg: SimConfig, sim,
                     csls: list[float], q_item=None) -> pd.DataFrame:
    """Sweep the service target: each policy traces (average stock held, fill rate, cost).

    A policy is better if it reaches the same fill rate with less stock (its curve lies up and
    to the left).
    """
    rows = []
    for csl in csls:
        res, _ = run_policies(y, price, fc_item, metrics_item, origins, horizon, max_day, cfg.with_(csl=float(csl)),
                              sim, q_item)
        rows += [{"item_id": item, "target_csl": float(csl), "policy": r.policy, **r.metrics}
                 for r in res if r.policy in MAIN_POLICIES]
    return pd.DataFrame(rows)
