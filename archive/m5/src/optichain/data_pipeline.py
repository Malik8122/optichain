"""Raw M5 CSVs -> validated long-format table for one store.

Leakage rules enforced here:
- sales columns beyond ``max_day`` are never read;
- prices are only forward-filled within an item (never back-filled);
- a series "launches" on its first priced week; earlier zeros are not demand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
KEY = ["item_id", "store_id", "d"]


D1_DATE = pd.Timestamp("2011-01-29")  # date of d_1; build_long_table validates the mapping


class DataValidationError(Exception):
    pass


def d_to_date(d: int) -> pd.Timestamp:
    return D1_DATE + pd.Timedelta(days=int(d) - 1)


def date_to_d(date: pd.Timestamp) -> int:
    return int((pd.Timestamp(date) - D1_DATE).days) + 1


def _day_num(d: pd.Series) -> pd.Series:
    return d.str.removeprefix("d_").astype("int16")


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    cal["d"] = _day_num(cal["d"])
    for c in EVENT_COLS:
        cal[c] = cal[c].fillna("none").astype("category")
    cal["weekday"] = cal["weekday"].astype("category")
    cal[["wday", "month"]] = cal[["wday", "month"]].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int32")
    return cal


def load_sales_wide(raw_dir: Path, sales_file: str, store_id: str, max_day: int) -> pd.DataFrame:
    """Read only id columns and d_1..d_max_day, then keep one store."""
    header = pd.read_csv(raw_dir / sales_file, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_") and int(c[2:]) <= max_day]
    if len(day_cols) != max_day:
        raise DataValidationError(f"{sales_file} has {len(day_cols)} day columns <= d_{max_day}")
    dtypes = {c: "int16" for c in day_cols} | {c: "string" for c in ID_COLS}
    wide = pd.read_csv(raw_dir / sales_file, usecols=ID_COLS + day_cols, dtype=dtypes)
    wide = wide[wide["store_id"] == store_id].reset_index(drop=True)
    if wide.empty:
        raise DataValidationError(f"No series found for store {store_id}")
    return wide


def load_prices(raw_dir: Path, store_id: str) -> pd.DataFrame:
    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    return prices[prices["store_id"] == store_id].reset_index(drop=True)


def melt_sales(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.drop(columns="id").melt(
        id_vars=ID_COLS[1:], var_name="d", value_name="sales"
    )
    long["d"] = _day_num(long["d"])
    return long


def build_long_table(
    raw_dir: Path, store_id: str, max_day: int, sales_file: str = "sales_train_evaluation.csv"
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    wide = load_sales_wide(raw_dir, sales_file, store_id, max_day)
    n_series = len(wide)
    long = melt_sales(wide)
    del wide

    cal = load_calendar(raw_dir)
    state = store_id.split("_")[0]
    cal = cal.rename(columns={f"snap_{state}": "snap"})
    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year", *EVENT_COLS, "snap"]
    n = len(long)
    long = long.merge(cal[cal_cols], on="d", how="left", validate="many_to_one")
    _check(len(long) == n, "calendar join changed row count")
    _check(long["date"].notna().all(), "calendar join left unmapped day ids")
    first = long.loc[long["d"] == 1, "date"].iloc[0]
    _check(first == D1_DATE, f"d_1 maps to {first.date()}, expected {D1_DATE.date()}")
    long["snap"] = long["snap"].astype("int8")

    prices = load_prices(raw_dir, store_id)
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )
    _check(len(long) == n, "price join changed row count")

    long = long.sort_values(["item_id", "d"], ignore_index=True)
    long["on_sale"] = long["sell_price"].notna()
    # A series launches at its first priced week; this uses only the price table.
    long["launched"] = long.groupby("item_id")["on_sale"].cummax()
    long["price_imputed"] = long["launched"] & ~long["on_sale"]
    long["sell_price"] = long.groupby("item_id")["sell_price"].ffill()

    for c in ID_COLS[1:]:
        long[c] = long[c].astype("category")

    validate_long(long, n_series=n_series, max_day=max_day)
    return long


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise DataValidationError(msg)


def validate_long(long: pd.DataFrame, n_series: int, max_day: int) -> None:
    _check(len(long) == n_series * max_day, f"expected {n_series * max_day} rows, got {len(long)}")
    _check(not long.duplicated(KEY).any(), "duplicate item-store-day rows")
    _check(not long.duplicated(["item_id", "store_id", "date"]).any(), "duplicate item-store-date rows")
    _check((long["sales"] >= 0).all(), "negative sales")
    _check(int(long["d"].max()) <= max_day, "rows beyond max_day")
    _check(long["d"].min() == 1, "series do not start at d_1")
    _check(not long.loc[long["launched"], "sell_price"].isna().any(), "missing price after launch")
    _check((long["sell_price"].dropna() > 0).all(), "non-positive price")
    days = long.drop_duplicates("d").sort_values("d")
    _check(
        (days["date"].diff().dropna() == pd.Timedelta(days=1)).all(),
        "day ids do not map to contiguous daily dates",
    )


STORE_CLOSED_EVENTS = ("Christmas",)  # every M5 store sells 0 units on 25 December


def get_series(long: pd.DataFrame, item_id: str, end_d: int, store_id: str | None = None,
               fill_closures: bool = False) -> pd.Series:
    """Date-indexed post-launch daily sales for one item with d <= end_d.

    ``fill_closures=True`` replaces store-closure days (zero by construction, not demand) with
    the same weekday one week earlier. It uses only past values, so it cannot leak. Use it for
    modelling and simulation; keep the raw series for descriptive EDA.
    """
    mask = (long["item_id"] == item_id) & long["launched"] & (long["d"] <= end_d)
    if store_id is not None:
        mask &= long["store_id"] == store_id
    rows = long.loc[mask, ["date", "sales", "event_name_1"]]
    if rows.empty:
        raise KeyError(f"No launched sales for {item_id} up to d_{end_d}")
    rows = rows.set_index("date")
    s = rows["sales"].astype("float64").asfreq("D")
    _check(s.notna().all(), f"{item_id}: gaps in daily index")
    if fill_closures:
        closed = rows["event_name_1"].isin(STORE_CLOSED_EVENTS).reindex(s.index, fill_value=False)
        for day in s.index[closed]:
            prev = day - pd.Timedelta(days=7)
            if prev in s.index:
                s.loc[day] = s.loc[prev]
    s.name = item_id
    return s


def audit_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per item: launch, coverage and zero-share summary (post-launch only)."""
    post = long[long["launched"]]
    g = post.groupby("item_id", observed=True)
    out = pd.DataFrame({
        "dept_id": g["dept_id"].first(),
        "cat_id": g["cat_id"].first(),
        "launch_d": g["d"].min(),
        "launch_date": g["date"].min(),
        "n_days": g.size(),
        "total_sales": g["sales"].sum(),
        "mean_sales": g["sales"].mean(),
        "zero_share": g["sales"].apply(lambda s: float(np.mean(s == 0))),
        "price_imputed_days": g["price_imputed"].sum(),
        "mean_price": g["sell_price"].mean(),
    })
    return out.reset_index()
