import numpy as np
import pandas as pd
import pytest

from optichain import residual_checks as R


def daily(values):
    return pd.Series(np.asarray(values, float), index=pd.date_range("2014-01-01", periods=len(values), freq="D"))


def simulate_garch(n=1500, omega=0.1, a=0.15, b=0.8, seed=0):
    rng = np.random.default_rng(seed)
    e, s2 = np.zeros(n), np.full(n, omega / (1 - a - b))
    for t in range(1, n):
        s2[t] = omega + a * e[t - 1] ** 2 + b * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * rng.normal()
    return daily(e[200:])


def test_model_df_counts_parameters():
    assert R.model_df({"order": [1, 1, 2], "seasonal_order": [1, 0, 1, 7]}) == 5
    assert R.model_df({"order": [0, 0, 0], "seasonal_order": None}) == 0
    assert R.model_df({"form": "ETS(A,N,A)"}) == 2      # alpha, gamma
    assert R.model_df({"form": "ETS(A,Ad,A)"}) == 4     # alpha, beta, phi, gamma
    assert R.model_df({"form": "ETS(A,N,N)"}) == 1


def test_white_noise_residuals_pass():
    r = daily(np.random.default_rng(1).normal(size=700))
    t = R.residual_tests(r)
    assert t["no_autocorrelation"] and t["unbiased"] and not t["arch_effects"]
    assert t["max_abs_acf"] < 0.12


def test_autocorrelated_and_biased_residuals_fail():
    rng = np.random.default_rng(2)
    x = np.zeros(700)
    for i in range(1, 700):
        x[i] = 0.5 * x[i - 1] + rng.normal()
    t = R.residual_tests(daily(x + 1.0))
    assert not t["no_autocorrelation"]
    assert not t["unbiased"]


def test_arch_effects_detected_and_garch_recovers_parameters():
    r = simulate_garch()
    assert R.residual_tests(r)["arch_effects"]
    _, info = R.fit_garch(r)
    assert info["valid"]
    assert info["alpha"] == pytest.approx(0.15, abs=0.08)
    assert info["persistence"] == pytest.approx(0.95, abs=0.06)


def test_lags_not_above_model_df_are_skipped():
    r = daily(np.random.default_rng(3).normal(size=400))
    t = R.residual_tests(r, lags=(7, 14), df=8)
    assert np.isnan(t["lb_p_7"]) and not np.isnan(t["lb_p_14"])


def test_scale_intervals():
    fc = pd.DataFrame({"yhat": [10.0, 10.0], "lo80": [6.0, 4.0], "hi80": [14.0, 18.0],
                       "lo95": [4.0, 2.0], "hi95": [16.0, 20.0]})
    same = R.scale_intervals(fc, np.array([1.0, 1.0]))
    pd.testing.assert_frame_equal(same, fc)
    wide = R.scale_intervals(fc, np.array([4.0, 0.25]))  # sqrt -> x2 and x0.5
    assert wide.loc[0, "hi80"] == pytest.approx(18.0) and wide.loc[0, "lo80"] == pytest.approx(2.0)
    assert wide.loc[1, "hi80"] == pytest.approx(14.0) and wide.loc[1, "lo80"] == pytest.approx(7.0)
    assert (R.scale_intervals(fc, np.array([100.0, 100.0]))["lo95"] >= 0).all()  # clipped at zero


def test_interval_score():
    # inside: score = width; miss below by 1 at 80%: width + (2 / 0.2) * 1
    assert R.interval_score([5], [4], [6], 0.8) == pytest.approx(2.0)
    assert R.interval_score([3], [4], [6], 0.8) == pytest.approx(2.0 + 10.0)
    assert R.interval_score([7], [4], [6], 0.95) == pytest.approx(2.0 + 40.0)
