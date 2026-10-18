import numpy as np
import pandas as pd
import pytest

from optichain import diagnostics as D
from optichain.data_pipeline import build_long_table, get_series

from .conftest import N_DAYS


def daily(values, start="2013-01-05"):
    return pd.Series(np.asarray(values, float), index=pd.date_range(start, periods=len(values), freq="D"))


def test_white_noise_is_stationary_and_left_alone():
    rng = np.random.default_rng(0)
    y = daily(10 + rng.normal(size=730))
    rec = D.identify(y)
    assert rec["level_verdict"] == "stationary"
    assert rec["d"] == 0 and rec["D"] == 0
    assert rec["white_noise"]
    assert rec["p_hint"] == 0 and rec["q_hint"] == 0


def test_random_walk_gets_first_difference():
    rng = np.random.default_rng(0)
    y = daily(100 + np.cumsum(rng.normal(size=730)))
    rec = D.identify(y)
    assert rec["level_verdict"] == "unit_root"
    assert rec["d"] == 1
    assert rec["post_d_verdict"] == "stationary"


def test_strong_weekly_pattern_gets_seasonal_difference():
    rng = np.random.default_rng(0)
    pattern = np.tile([1, 1, 1, 2, 4, 9, 8], 105)[:730]
    y = daily(20 + 5 * pattern + 0.5 * rng.normal(size=730))
    rec = D.identify(y)
    assert rec["seasonal_strength"] > D.SEASONAL_STRENGTH_D
    assert rec["D"] == 1


def test_ar1_suggests_ar_term():
    rng = np.random.default_rng(0)
    e = rng.normal(size=1000)
    x = np.zeros(1000)
    for t in range(1, 1000):
        x[t] = 0.7 * x[t - 1] + e[t]
    rec = D.identify(daily(50 + x[270:]))
    assert rec["p_hint"] >= 1
    assert not rec["white_noise"]


@pytest.mark.parametrize("adf_p,kpss_p,expected", [
    (0.01, 0.10, "stationary"), (0.40, 0.01, "unit_root"), (0.01, 0.01, "conflict"), (0.40, 0.10, "inconclusive"),
])
def test_verdict_table(adf_p, kpss_p, expected):
    assert D.stationarity_verdict(adf_p, kpss_p) == expected


def test_conflict_offers_both_d_candidates():
    rng = np.random.default_rng(0)
    # stationary noise with a level shift: ADF rejects a unit root, KPSS rejects level-stationarity
    y = daily(np.r_[np.full(365, 10.0), np.full(365, 16.0)] + rng.normal(size=730))
    rec = D.identify(y)
    if rec["pre_d_verdict"] == "conflict":
        assert rec["d"] == 0 and rec["d_candidates"] == [0, 1]
    elif rec["pre_d_verdict"] == "inconclusive":
        assert rec["d_candidates"] == [0, 1]
    else:
        assert rec["d_candidates"] == [rec["d"]]


def test_transform_roundtrip_and_rules():
    y = daily([0, 1, 5, 20, 100])
    np.testing.assert_allclose(D.inverse_transform(D.apply_transform(y, "log1p"), "log1p"), y)
    assert D.choose_transform(daily(np.r_[np.zeros(200), np.ones(200)])) == "none"  # zero-heavy
    with pytest.raises(ValueError):
        D.apply_transform(daily([-1, 2]), "log1p")


def test_difference_drops_leading_nans():
    y = daily(np.arange(30, dtype=float))
    out = D.difference(y, d=1, D=1, s=7)
    assert len(out) == 30 - 7 - 1 and out.notna().all()


def test_fill_closures_uses_same_weekday_last_week(raw_dir):
    long = build_long_table(raw_dir, "CA_1", max_day=N_DAYS)
    long["event_name_1"] = long["event_name_1"].cat.add_categories(["Christmas"])
    long.loc[long["d"] == 20, "event_name_1"] = "Christmas"
    long.loc[long["d"] == 20, "sales"] = 0
    raw = get_series(long, "ITEM_A", N_DAYS)
    filled = get_series(long, "ITEM_A", N_DAYS, fill_closures=True)
    day = raw.index[19]
    assert raw[day] == 0
    assert filled[day] == raw[day - pd.Timedelta(days=7)]
    assert (filled.drop(day) == raw.drop(day)).all()  # nothing else changes
