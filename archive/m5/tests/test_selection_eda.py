import numpy as np
import pandas as pd
import pytest

from optichain.data_pipeline import build_long_table
from optichain.eda import STORE_CLOSED_EVENTS, calendar_effects, weekday_profile
from optichain.selection import classify, classify_items, demand_stats, max_zero_run, select_items

from .conftest import N_DAYS


@pytest.mark.parametrize("adi,cv2,expected", [
    (1.0, 0.1, "smooth"), (1.2, 0.8, "erratic"), (2.0, 0.2, "intermittent"), (3.0, 1.0, "lumpy"),
    (1.32, 0.1, "intermittent"), (1.0, 0.49, "erratic"),  # cut-offs belong to the upper class
])
def test_classify_quadrants(adi, cv2, expected):
    assert classify(adi, cv2) == expected


def test_demand_stats_known_values():
    st = demand_stats(np.array([0, 2, 0, 2, 0, 0, 4, 0]))
    assert st["adi"] == pytest.approx(8 / 3)
    assert st["zero_share"] == pytest.approx(5 / 8)
    assert st["max_zero_run"] == 2
    sizes = np.array([2, 2, 4])
    assert st["cv2"] == pytest.approx((sizes.std(ddof=1) / sizes.mean()) ** 2)


def test_max_zero_run_edges():
    assert max_zero_run(np.array([1, 2, 3])) == 0
    assert max_zero_run(np.array([0, 0, 0])) == 3
    assert max_zero_run(np.array([0, 1, 0, 0, 1, 0, 0, 0])) == 3


def test_classify_items_requires_full_post_launch_window(raw_dir):
    long = build_long_table(raw_dir, "CA_1", max_day=N_DAYS)
    # ITEM_C launches at d_15: eligible for a 14-day window ending d_42, not a 30-day one
    short = classify_items(long, end_d=N_DAYS, window_days=14, recent_days=7)
    assert "ITEM_C" in set(short["item_id"])
    long_w = classify_items(long, end_d=N_DAYS, window_days=30, recent_days=7)
    assert "ITEM_C" not in set(long_w["item_id"])
    assert {"ITEM_A", "ITEM_B"} <= set(long_w["item_id"])


def test_classify_items_ignores_data_after_end_d(raw_dir):
    long = build_long_table(raw_dir, "CA_1", max_day=N_DAYS)
    a = classify_items(long, end_d=30, window_days=14, recent_days=7)
    tampered = long.copy()
    tampered.loc[tampered["d"] > 30, "sales"] = 999
    b = classify_items(tampered, end_d=30, window_days=14, recent_days=7)
    pd.testing.assert_frame_equal(a, b)


def _fake_classes() -> pd.DataFrame:
    rows = []
    for cls, base in [("smooth", 10), ("erratic", 5), ("intermittent", 1), ("lumpy", 1)]:
        for i in range(6):
            rows.append({"item_id": f"{cls}_{i}", "cat_id": ["FOODS", "HOBBIES", "HOUSEHOLD"][i % 3],
                         "demand_class": cls, "mean": base * (i + 1), "max_zero_run": 3})
    return pd.DataFrame(rows)


def test_select_items_is_deterministic_and_unique():
    c = _fake_classes()
    a, b = select_items(c), select_items(c.sample(frac=1, random_state=1))
    assert list(a["item_id"]) == list(b["item_id"])
    assert a["item_id"].is_unique
    assert len(a) == 8
    assert a.loc[a["volume_quantile"] == 1.0, "item_id"].item() == "smooth_5"  # top-volume item


def test_select_items_excludes_long_zero_runs():
    c = _fake_classes()
    c.loc[c["item_id"] == "smooth_5", "max_zero_run"] = 60
    picked = select_items(c)
    assert "smooth_5" not in set(picked["item_id"])


def test_weekday_profile_index_averages_to_one():
    idx = pd.date_range("2015-01-05", periods=28, freq="D")  # 4 full weeks
    s = pd.Series(np.tile([1, 1, 1, 1, 2, 4, 4], 4), index=idx, dtype=float)
    prof = weekday_profile(s)
    assert prof["index"].mean() == pytest.approx(1.0)
    assert prof.loc["Saturday", "mean"] == 4 and prof.loc["Monday", "mean"] == 1


def test_calendar_effects_excludes_closures(raw_dir):
    long = build_long_table(raw_dir, "CA_1", max_day=N_DAYS)
    long["event_name_1"] = long["event_name_1"].cat.add_categories(STORE_CLOSED_EVENTS)
    long.loc[long["d"] == 10, "event_name_1"] = "Christmas"
    out = calendar_effects(long, ["ITEM_A"], end_d=N_DAYS, window_days=N_DAYS)
    assert out.loc["ITEM_A", "event_days"] == 1  # only the SuperBowl day counts
