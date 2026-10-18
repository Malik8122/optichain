import numpy as np
import pandas as pd
import pytest

from optichain.data_pipeline import d_to_date, date_to_d
from optichain.evaluation import point_metrics, rolling_origin, scale_denominators, score, smape
from optichain.forecasting import MODEL_REGISTRY, make_model
from optichain.simulation import select_model_per_origin


def daily(values, start="2011-01-29"):
    return pd.Series(np.asarray(values, float), index=pd.date_range(start, periods=len(values), freq="D"))


def weekly_pattern(n_weeks=20, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.tile([2, 2, 2, 3, 5, 8, 6], n_weeks).astype(float)
    return daily(base + noise * rng.normal(size=len(base)).clip(-1, 1))


def test_d_date_roundtrip():
    assert d_to_date(1) == pd.Timestamp("2011-01-29")
    assert d_to_date(1913) == pd.Timestamp("2016-04-24")
    assert date_to_d(d_to_date(1885)) == 1885


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_forecast_shape_index_and_bounds(name):
    y = weekly_pattern(noise=0.5)
    fc = make_model(name).fit(y).predict(28)
    assert len(fc) == 28 and list(fc["h"]) == list(range(1, 29))
    assert fc.index[0] == y.index[-1] + pd.Timedelta(days=1)
    assert np.isfinite(fc.to_numpy()).all()
    assert (fc["yhat"] >= 0).all() and (fc["lo95"] >= 0).all()
    assert (fc["lo95"] <= fc["lo80"]).all() and (fc["hi80"] <= fc["hi95"]).all()


def test_snaive_repeats_last_week_and_widens_weekly():
    y = weekly_pattern(noise=0.5)
    fc = make_model("snaive").fit(y).predict(14)
    np.testing.assert_allclose(fc["yhat"].to_numpy()[:7], y.iloc[-7:].to_numpy().clip(0))
    width = (fc["hi95"] - fc["lo95"]).to_numpy()
    assert width[7] > width[6]  # second week is less certain
    assert np.allclose(width[:7], width[0]) or (fc["lo95"] == 0).any()


def test_naive_and_ma():
    y = daily([1, 2, 3, 4] * 10)
    assert (make_model("naive").fit(y).predict(3)["yhat"] == 4).all()
    assert make_model("ma28").fit(y).predict(1)["yhat"].iloc[0] == pytest.approx(2.5)


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        make_model("naive").fit(pd.Series([1.0, 2.0]))  # no DatetimeIndex
    with pytest.raises(ValueError):
        make_model("naive").fit(daily([1, np.nan, 3]))
    with pytest.raises(KeyError):
        make_model("nope")


def test_point_metrics_hand_computed():
    m = point_metrics([0, 2, 4], [1, 2, 2], mse_scale=4.0, mae_scale=2.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["bias"] == pytest.approx(-1 / 3)
    assert m["rmsse"] == pytest.approx(np.sqrt((5 / 3) / 4))
    assert m["mase"] == pytest.approx(0.5)


def test_degenerate_scale_gives_nan_not_inf():
    assert np.isnan(point_metrics([1, 1], [1, 2], mse_scale=0.0, mae_scale=0.0)["rmsse"])
    sc = scale_denominators(daily([0, 0, 0]))
    assert np.isnan(sc["mse_scale"])


def test_scale_starts_at_first_nonzero_sale():
    sc = scale_denominators(daily([0, 0, 0, 2, 4, 4]))
    assert sc["mse_scale"] == pytest.approx((4 + 0) / 2)


def test_smape_zero_zero_is_zero():
    assert smape(np.array([0, 0]), np.array([0, 0])) == 0
    assert smape(np.array([0]), np.array([1])) == pytest.approx(2.0)


def test_rolling_origin_has_no_leakage():
    y = weekly_pattern(n_weeks=30, noise=0.5)
    origins = [100, 150]
    fc, _ = rolling_origin(y, "snaive", origins, h=28, fit_window_days=60)
    tampered = y.copy()
    tampered.loc[d_to_date(101):] = 999.0  # change everything after the first origin
    fc2, _ = rolling_origin(tampered, "snaive", [100], h=28, fit_window_days=60)
    a = fc[fc["origin_d"] == 100]["yhat"].to_numpy()
    np.testing.assert_allclose(a, fc2["yhat"].to_numpy())
    for o, g in fc.groupby("origin_d"):
        assert (g["d"] > o).all()
        assert g["date"].min() == d_to_date(o + 1)
    # y_true is the real future value
    row = fc[(fc["origin_d"] == 100) & (fc["h"] == 1)].iloc[0]
    assert row["y_true"] == y.loc[d_to_date(101)]


def test_score_skips_incomplete_windows():
    y = weekly_pattern(n_weeks=20)  # 140 days
    fc, _ = rolling_origin(y, "naive", [100, 130], h=28)
    out = score(fc.assign(item_id="X"), h_eval=28)
    assert list(out["origin_d"]) == [100]  # origin 130 runs past the data


def test_model_selection_uses_only_completed_folds():
    metrics = pd.DataFrame({
        "origin_d": [100, 100, 128, 128],
        "model": ["a", "b", "a", "b"],
        "rmsse": [0.5, 0.9, 2.0, 0.1],
    })
    ch = select_model_per_origin(metrics, [100, 128, 156], horizon=28, default="d").set_index("origin_d")
    assert ch.loc[100, "model"] == "d"  # nothing completed yet
    assert ch.loc[128, "model"] == "a"  # only fold 100 is complete at 128
    assert ch.loc[156, "model"] == "b"  # mean over both folds: a=1.25, b=0.5
