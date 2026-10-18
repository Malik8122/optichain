import numpy as np
import pandas as pd
import pytest

from optichain.backtest import Task
from optichain.forecasting import make_model
from optichain.models_stat import Croston

SMALL = {"p": [0, 1], "q": [0, 1], "P": [0, 1], "Q": [0, 1], "s": 7}


def daily(values, start="2013-01-05"):
    return pd.Series(np.asarray(values, float), index=pd.date_range(start, periods=len(values), freq="D"))


def weekly(n_weeks=40, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.tile([10, 10, 11, 12, 16, 24, 20], n_weeks).astype(float)
    return daily(base + noise * rng.normal(size=len(base)))


def test_ets_picks_seasonal_form_on_weekly_data():
    m = make_model("ets").fit(weekly())
    assert m.params()["form"].endswith(",A)")  # seasonal component selected
    fc = m.predict(14)
    # forecast reproduces the weekly shape: peak day well above trough day
    assert fc["yhat"].max() - fc["yhat"].min() > 8


def test_sarima_captures_weekly_pattern_and_beats_flat():
    y = weekly(noise=1.0)
    train, test = y.iloc[:-14], y.iloc[-14:]
    ident = {"transform": "none", "d_candidates": [0], "D_candidates": [0]}
    m = make_model("sarima", ident=ident, grid=SMALL).fit(train)
    fc = m.predict(14)["yhat"].to_numpy()
    flat = np.full(14, train.iloc[-28:].mean())
    assert np.mean(np.abs(fc - test.to_numpy())) < 0.6 * np.mean(np.abs(flat - test.to_numpy()))
    assert m.params()["seasonal_order"][3] == 7


def test_arima_respects_identification_and_records_candidates():
    rng = np.random.default_rng(1)
    y = daily(50 + np.cumsum(rng.normal(size=300)))
    m = make_model("arima", ident={"transform": "none", "d_candidates": [1], "D_candidates": [0]}, grid=SMALL).fit(y)
    p = m.params()
    assert p["order"][1] == 1 and p["seasonal_order"] is None
    assert p["n_fits"] == 4  # p, q in {0, 1}, seasonal terms ignored for ARIMA
    assert all(c["d"] == 1 for c in p["candidates"])


def test_inner_holdout_chooses_between_d_candidates():
    rng = np.random.default_rng(2)
    y = daily(50 + np.cumsum(rng.normal(size=300)))
    m = make_model("arima", ident={"transform": "none", "d_candidates": [0, 1], "D_candidates": [0]}, grid=SMALL).fit(y)
    scores = m.params()["combo_scores"]
    assert {s["d"] for s in scores} == {0, 1}
    best = min(scores, key=lambda s: s["inner_rmse"])
    assert m.params()["order"][1] == best["d"]
    # inner fits use only the window minus the holdout; the final fit uses the whole window
    assert m.res_.nobs == len(y)


def test_log1p_back_transform_keeps_bands_ordered_and_nonnegative():
    rng = np.random.default_rng(3)
    y = daily(np.exp(2 + 0.3 * rng.normal(size=200)).round())
    m = make_model("arima", ident={"transform": "log1p", "d_candidates": [0], "D_candidates": [0]}, grid=SMALL).fit(y)
    fc = m.predict(10)
    assert (fc["lo95"] >= 0).all()
    assert (fc["lo95"] <= fc["lo80"]).all() and (fc["lo80"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["hi80"]).all() and (fc["hi80"] <= fc["hi95"]).all()


def test_croston_sba_on_known_series():
    # demand of 4 every 4th day -> true mean rate 1/day; SBA = (1 - a/2) * z / p
    y = daily(np.tile([0, 0, 0, 4], 60))
    m = make_model("croston").fit(y)
    expected = (1 - m.alpha_ / 2) * 4 / 4
    assert m.predict(5)["yhat"].iloc[0] == pytest.approx(expected, rel=0.02)


def test_croston_all_zero_series_forecasts_zero():
    m = Croston().fit(daily(np.zeros(50)))
    assert (m.predict(3)["yhat"] == 0).all()


def test_cache_key_changes_with_context_and_data():
    y = weekly()
    y.index = pd.date_range("2011-01-29", periods=len(y), freq="D")  # d_1 = 2011-01-29, so origin 250 is inside
    base = Task("X", "arima", 250, y, 28, 200, (0.8, 0.95), {"ident": {"transform": "none"}})
    same = Task("X", "arima", 250, y, 28, 200, (0.8, 0.95), {"ident": {"transform": "none"}})
    other_ctx = Task("X", "arima", 250, y, 28, 200, (0.8, 0.95), {"ident": {"transform": "log1p"}})
    y2 = y.copy()
    y2.iloc[100] += 1
    other_data = Task("X", "arima", 250, y2, 28, 200, (0.8, 0.95), {"ident": {"transform": "none"}})
    future_change = y.copy()
    future_change.iloc[-1] += 99  # after the origin: must NOT change the key (no leakage into cache)
    after = Task("X", "arima", 250, future_change, 28, 200, (0.8, 0.95), {"ident": {"transform": "none"}})
    assert base.key() == same.key() == after.key()
    assert base.key() != other_ctx.key()
    assert base.key() != other_data.key()
