import numpy as np
import pandas as pd
import pytest

from optichain.inventory import (
    BaselinePolicy,
    ForecastPolicy,
    SimConfig,
    inventory_metrics,
    perfect_foresight_forecasts,
    simulate,
)
from optichain.data_pipeline import d_to_date


class FixedPolicy:
    def __init__(self, rop, S, name="fixed"):
        self.rop, self.S, self.name = rop, S, name

    def targets(self, d):
        return self.rop, self.S


def run(demand, policy, start=0.0, **cfg_kw):
    days = list(range(101, 101 + len(demand)))
    dem = pd.Series(demand, index=days, dtype=float)
    price = pd.Series(1.0, index=days)
    return simulate(dem, price, policy, SimConfig(**cfg_kw), start_on_hand=start)


def test_order_arrives_exactly_after_lead_time():
    # start with 0 stock, rop 0 / S 10: first review (end of day 101) orders 10
    res = run([0] * 10, FixedPolicy(0, 10), lead_time=3)
    t = res.trace.set_index("d")
    assert t.loc[101, "order_qty"] == 10
    assert t.loc[[102, 103], "received"].sum() == 0
    assert t.loc[104, "received"] == 10


def test_stock_never_negative_and_conserved():
    rng = np.random.default_rng(0)
    demand = rng.poisson(5, 60)
    res = run(demand, FixedPolicy(15, 30), start=20, lead_time=2)
    t = res.trace
    assert (t["on_hand"] >= 0).all()
    assert (t["sold"] + t["lost"] == t["demand"]).all()
    prev = np.concatenate([[20.0], t["on_hand"].to_numpy()[:-1]])
    np.testing.assert_allclose(prev + t["received"] - t["sold"], t["on_hand"])


def test_lost_sales_when_empty():
    res = run([3, 3], FixedPolicy(-1, 0), start=2)  # never orders
    t = res.trace
    assert list(t["sold"]) == [2, 0] and list(t["lost"]) == [1, 3]
    assert res.metrics["fill_rate"] == pytest.approx(2 / 6)
    assert res.metrics["stockout_days"] == 2


def test_case_pack_rounding_and_order_cost():
    res = run([0, 0], FixedPolicy(5, 7), start=0, case_pack=4, order_cost=2.5)
    t = res.trace
    assert t["order_qty"].iloc[0] == 8  # 7 rounded up to a multiple of 4
    assert res.metrics["ordering_cost"] == 2.5


def test_no_order_when_position_covers_reorder_point():
    res = run([1] * 5, FixedPolicy(2, 10), start=100)
    assert res.metrics["n_orders"] == 0


def _series_and_forecasts(demand_value=4.0, n_days=200, origins=(150,), horizon=42):
    y = pd.Series(demand_value, index=pd.date_range(d_to_date(1), periods=n_days, freq="D"))
    return y, perfect_foresight_forecasts(y, list(origins), horizon)


def test_perfect_foresight_constant_demand_has_no_lost_sales():
    y, fc = _series_and_forecasts()
    cfg = SimConfig(lead_time=1, csl=0.5, cover_days=3)
    days = list(range(151, 181))  # last decision needs d+L+cover <= 150+42
    demand = pd.Series(4.0, index=days)
    res = simulate(demand, pd.Series(1.0, index=days), ForecastPolicy(fc, cfg), cfg, start_on_hand=16)
    assert res.metrics["lost_units"] == 0


def test_forecast_policy_targets_sum_over_lead_time():
    y, fc = _series_and_forecasts(demand_value=2.0)
    cfg = SimConfig(lead_time=4, cover_days=7, csl=0.5)  # z = 0 -> no safety stock
    rop, S = ForecastPolicy(fc, cfg).targets(160)
    assert rop == pytest.approx(8.0) and S == pytest.approx(22.0)


def test_forecast_policy_requires_long_enough_horizon():
    y, fc = _series_and_forecasts(horizon=10)
    pol = ForecastPolicy(fc, SimConfig(lead_time=4, cover_days=7))
    with pytest.raises(ValueError, match="extend the horizon"):
        pol.targets(150)


def test_baseline_uses_only_history_up_to_origin():
    vals = np.r_[np.full(150, 2.0), np.full(50, 100.0)]  # big jump after the origin
    y = pd.Series(vals, index=pd.date_range(d_to_date(1), periods=200, freq="D"))
    cfg = SimConfig(lead_time=2, cover_days=3, csl=0.5)
    rop, S = BaselinePolicy(y, [150], cfg).targets(170)
    assert rop == pytest.approx(4.0) and S == pytest.approx(10.0)


def test_policy_without_origin_before_day_fails():
    y, fc = _series_and_forecasts(origins=(150,))
    with pytest.raises(ValueError):
        ForecastPolicy(fc, SimConfig()).targets(149)


def test_config_validation():
    with pytest.raises(ValueError):
        SimConfig(lead_time=0)
    with pytest.raises(ValueError):
        SimConfig(csl=1.0)


def test_metrics_no_demand():
    t = pd.DataFrame({"demand": [0.0], "sold": [0.0], "lost": [0.0], "on_hand": [5.0], "order_qty": [0.0],
                      "holding_cost": [0.1], "ordering_cost": [0.0], "shortage_cost": [0.0]})
    m = inventory_metrics(t)
    assert np.isnan(m["fill_rate"]) and m["cycle_service"] == 1.0


# ---------------------------------------------------------------- Policy 2 (empirical safety stock)

from optichain.evaluation import residual_sum_quantiles  # noqa: E402
from optichain.inventory import EmpiricalForecastPolicy  # noqa: E402


def _quantile_table(origin=150, L=4, csl=0.95, value=7.5):
    return pd.DataFrame([{"origin_d": origin, "L": L, "q": csl, "value": value, "n": 500}])


def test_empirical_policy_uses_quantile_as_safety_stock():
    y, fc = _series_and_forecasts(demand_value=2.0)
    cfg = SimConfig(lead_time=4, cover_days=7, csl=0.95)
    rop, S = EmpiricalForecastPolicy(fc, _quantile_table(value=7.5), cfg).targets(160)
    assert rop == pytest.approx(8.0 + 7.5) and S == pytest.approx(22.0 + 7.5)


def test_empirical_policy_negative_quantile_floors_at_zero():
    y, fc = _series_and_forecasts(demand_value=2.0)
    cfg = SimConfig(lead_time=4, cover_days=7, csl=0.95)
    rop, _ = EmpiricalForecastPolicy(fc, _quantile_table(value=-3.0), cfg).targets(160)
    assert rop == pytest.approx(8.0)


def test_empirical_policy_requires_matching_quantile():
    y, fc = _series_and_forecasts()
    with pytest.raises(ValueError, match="no residual quantile"):
        EmpiricalForecastPolicy(fc, _quantile_table(L=2), SimConfig(lead_time=4, csl=0.95))


def test_residual_sum_quantiles_match_normal_theory_for_iid_errors():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 2.0, 20000), index=pd.date_range("2000-01-01", periods=20000, freq="D"))
    rows = pd.DataFrame(residual_sum_quantiles(r, leads=(1, 4), qs=(0.95,)))
    from scipy.stats import norm
    for L in (1, 4):
        v = rows[(rows.L == L)]["value"].item()
        assert v == pytest.approx(norm.ppf(0.95) * 2.0 * np.sqrt(L), rel=0.05)


def test_residual_sum_quantiles_capture_heavy_right_tail():
    rng = np.random.default_rng(1)
    e = rng.exponential(3.0, 5000) - 3.0  # mean 0, right-skewed
    r = pd.Series(e, index=pd.date_range("2000-01-01", periods=5000, freq="D"))
    q99 = pd.DataFrame(residual_sum_quantiles(r, leads=(1,), qs=(0.99,)))["value"].item()
    assert q99 > 2.326 * e.std() * 1.2  # empirical 99% quantile well above the normal-theory one
