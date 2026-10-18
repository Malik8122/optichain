"""Tiny synthetic M5 dataset with known properties.

- 2 stores (CA_1, TX_1), 3 items, 42 days (6 weeks), d_1 = 2011-01-29.
- ITEM_A: priced from week 1, sales = d % 5.
- ITEM_B: priced from week 1 but price row missing in week 3 (tests forward-fill).
- ITEM_C: launches in week 3 (no price rows before); sales 0 before launch.
"""
import numpy as np
import pandas as pd
import pytest

N_DAYS = 42
ITEMS = ["ITEM_A", "ITEM_B", "ITEM_C"]
STORES = ["CA_1", "TX_1"]
WEEK0 = 11101
LAUNCH_C_WEEK = 2  # zero-based -> d_15


def _week(d: int) -> int:
    return WEEK0 + (d - 1) // 7


@pytest.fixture
def raw_dir(tmp_path):
    days = np.arange(1, N_DAYS + 1)
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": [_week(d) for d in days],
        "weekday": dates.day_name(),
        "wday": (dates.dayofweek + 2) % 7 + 1,  # M5: Saturday = 1
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{d}" for d in days],
        "event_name_1": [("SuperBowl" if d == 8 else None) for d in days],
        "event_type_1": [("Sporting" if d == 8 else None) for d in days],
        "event_name_2": None,
        "event_type_2": None,
        "snap_CA": (days % 2).astype(int),
        "snap_TX": 0,
        "snap_WI": 0,
    })
    cal.to_csv(tmp_path / "calendar.csv", index=False)

    rows = []
    for store in STORES:
        for item in ITEMS:
            sales = days % 5
            if item == "ITEM_C":
                sales = np.where(days >= 1 + 7 * LAUNCH_C_WEEK, sales, 0)
            rows.append({
                "id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": "FOODS_1",
                "cat_id": "FOODS", "store_id": store, "state_id": store[:2],
                **{f"d_{d}": int(s) for d, s in zip(days, sales)},
            })
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)

    prices = []
    for store in STORES:
        for item in ITEMS:
            for w in range(N_DAYS // 7):
                if item == "ITEM_B" and w == 2:
                    continue
                if item == "ITEM_C" and w < LAUNCH_C_WEEK:
                    continue
                prices.append({"store_id": store, "item_id": item,
                               "wm_yr_wk": WEEK0 + w, "sell_price": 1.0 + w})
    pd.DataFrame(prices).to_csv(tmp_path / "sell_prices.csv", index=False)
    return tmp_path
