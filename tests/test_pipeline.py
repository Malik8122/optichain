"""Core checks: data loading, models, simulator (against the official result) and the policy."""
import numpy as np
import pandas as pd
import pytest

from optichain import data as D
from optichain import inventory as I
from optichain import models as M
from optichain.config import load_config

cfg = load_config()
HAS_DATA = (cfg.paths.raw_dir / D.SALES_FILES[-1]).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="VN2 files not downloaded")


@pytest.fixture(scope="module")
def panel():
    return D.load_panel(cfg.paths.raw_dir)


def tiny_history(sales, in_stock=None):
    sales = np.asarray(sales, float)
    in_stock = np.ones_like(sales, bool) if in_stock is None else np.asarray(in_stock, bool)
    dates = pd.date_range("2023-01-02", periods=sales.shape[1], freq="7D")
    cats = pd.DataFrame(0, index=range(sales.shape[0]), columns=M.CAT_COLS)
    return M.History(sales, in_stock, dates, cats)


# ------------------------------------------------------------------ preprocessing
def test_clean_demand_masks_stockouts_and_prelaunch():
    s = pd.DataFrame([[0, 0, 3, 5, 2]], columns=pd.date_range("2024-01-01", periods=5, freq="7D"))
    i = pd.DataFrame([[True, True, True, False, True]], columns=s.columns)
    d = D.clean_demand(s, i).iloc[0]
    assert d.isna().tolist() == [True, True, False, True, False]


def test_classify_patterns():
    cols = pd.date_range("2024-01-01", periods=8, freq="7D")
    dem = pd.DataFrame([[5, 5, 5, 5, 5, 5, 5, 5], [0, 0, 0, 4, 0, 0, 0, 4]], columns=cols,
                       index=pd.MultiIndex.from_tuples([(0, 1), (0, 2)], names=D.INDEX))
    cls = D.classify_demand(dem)
    assert cls["pattern"].tolist() == ["smooth", "intermittent"]


# ------------------------------------------------------------------ models
def test_simple_models_shapes_and_values():
    h = tiny_history([[2, 2, 2, 2, 2, 2] * 3, [0, 0, 1, 0, 0, 1] * 3])
    assert np.allclose(M.ma13(h, 3)[0], 2)
    assert np.allclose(M.naive(h, 3)[1], 1)
    for name, fn in M.SIMPLE_MODELS.items():
        f = fn(h, 3)
        assert f.shape == (2, 3) and np.isfinite(f).all() and (f >= 0).all(), name


def test_models_ignore_censored_weeks():
    h = tiny_history([[4, 4, 4, 0, 0, 4, 4, 4]], [[1, 1, 1, 0, 0, 1, 1, 1]])
    assert np.allclose(M.ma13(h, 1), 4)


def test_croston_on_intermittent():
    h = tiny_history([[0, 0, 0, 4] * 10])
    f = M.croston_sba(h, 1)[0, 0]
    assert 0.8 < f < 1.1   # one sale of 4 every 4 weeks ~ 1 per week, minus the SBA correction


# ------------------------------------------------------------------ simulator & policy
def test_play_week_official_arithmetic():
    st = I.State(np.array([5.0]), np.array([2.0]), np.array([3.0]))
    new, rec = I.play_week(st, np.array([10.0]), np.array([4.0]), I.Costs())
    assert rec["start"][0] == 7 and rec["sales"][0] == 7 and rec["lost"][0] == 3 and rec["end"][0] == 0
    assert new.t1[0] == 3 and new.t2[0] == 4 and rec["shortage"][0] == 3.0


@needs_data
def test_simulator_reproduces_official_benchmark(panel):
    init = I.State.from_initial(panel.initial)
    dem = panel.sales.iloc[:, panel.t0 + 1:panel.t0 + 9].to_numpy()
    orders = [o.to_numpy() for o in D.load_benchmark_orders(cfg.paths.raw_dir)]
    weekly, _, _ = I.simulate(init, dem, orders, I.Costs())
    assert round(weekly.loc[1:2, "cost"].sum(), 1) == 913.8
    assert round(weekly.loc[3:8, "cost"].sum(), 1) == 4334.0


@needs_data
def test_benchmark_reimplementation_matches_orders(panel):
    h = M.History.from_panel(panel)
    first = D.load_benchmark_orders(cfg.paths.raw_dir)[0].to_numpy()
    o = I.order_up_to(M.seasonal_ma(h.upto(panel.t0), 4), I.State.from_initial(panel.initial), 4)
    assert (o == first).all()   # round 1; the notebook checks all six rounds


def test_sampler_moments():
    rng = np.random.default_rng(1)
    mu = np.array([[4.0, 4.0, 4.0]])
    d = I.sample_demand(mu, np.array([5.0]), 20000, rng)
    assert abs(d.mean() - 4) < 0.1 and abs(d.var() - 20) < 1.5


def test_newsvendor_orders_more_when_safer():
    mu = np.full((3, 3), 5.0)
    st = I.State(np.zeros(3), np.zeros(3), np.zeros(3))
    lo = I.newsvendor_orders(mu, np.full(3, 2.0), st, 0.6, 4000)
    hi = I.newsvendor_orders(mu, np.full(3, 2.0), st, 0.95, 4000)
    assert (hi > lo).all()
    full = I.State(np.full(3, 100.0), np.zeros(3), np.zeros(3))
    assert (I.newsvendor_orders(mu, np.full(3, 2.0), full, 0.95, 4000) == 0).all()


# ------------------------------------------------------------------ time-series toolkit
def test_adf_separates_stationary_from_random_walk():
    from optichain import timeseries as TS
    rng = np.random.default_rng(0)
    noise = rng.normal(size=300)
    assert TS.adf_test(noise)["ADF p-value"] < 0.05
    assert TS.adf_test(np.cumsum(noise))["ADF p-value"] > 0.05
    assert TS.choose_d(np.cumsum(noise)) == 1 and TS.choose_d(noise) == 0


def test_aic_finds_an_ar1_and_residuals_are_white():
    from optichain import timeseries as TS
    rng = np.random.default_rng(1)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.7 * y[t - 1] + rng.normal()
    grid = TS.aic_bic_grid(y, 0)
    assert grid.iloc[0]["order"][0] >= 1                         # an AR term is chosen
    res = TS.fit_arima(y, grid.iloc[0]["order"])
    assert TS.residual_tests(res.resid[1:])["white noise"]
    assert not TS.residual_tests(y)["white noise"]                # the raw AR(1) series is not white noise


def test_arch_detected_and_garch_removes_it():
    from optichain import timeseries as TS
    rng = np.random.default_rng(2)
    e, s2 = np.zeros(1500), np.ones(1500)
    for t in range(1, 1500):
        s2[t] = 0.1 + 0.3 * e[t - 1] ** 2 + 0.6 * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * rng.normal()
    assert TS.residual_tests(e)["ARCH effect"]
    g, _ = TS.fit_garch(e)
    assert not TS.residual_tests(np.asarray(g.std_resid))["ARCH effect"]
