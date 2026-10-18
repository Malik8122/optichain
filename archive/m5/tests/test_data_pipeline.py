import pandas as pd
import pytest

from optichain.data_pipeline import (
    DataValidationError,
    audit_table,
    build_long_table,
    get_series,
    validate_long,
)

from .conftest import ITEMS, N_DAYS


@pytest.fixture
def long(raw_dir):
    return build_long_table(raw_dir, "CA_1", max_day=N_DAYS)


def test_shape_and_store_filter(long):
    assert len(long) == len(ITEMS) * N_DAYS
    assert set(long["store_id"].unique()) == {"CA_1"}
    assert not long.duplicated(["item_id", "store_id", "date"]).any()


def test_calendar_mapping(long):
    first = long[long["d"] == 1]["date"].iloc[0]
    assert first == pd.Timestamp("2011-01-29")
    assert long.loc[long["d"] == 8, "event_name_1"].eq("SuperBowl").all()
    assert long.loc[long["d"] == 9, "event_name_1"].eq("none").all()
    # CA store gets the CA snap flag (odd days = 1 in fixture)
    assert long.loc[long["d"] == 3, "snap"].eq(1).all()


def test_max_day_cutoff_never_reads_future(raw_dir):
    long = build_long_table(raw_dir, "CA_1", max_day=20)
    assert long["d"].max() == 20
    assert len(long) == len(ITEMS) * 20


def test_price_forward_filled_not_back_filled(long):
    b = long[long["item_id"] == "ITEM_B"].set_index("d")
    # week 3 (d_15..d_21) has no price row -> carries week-2 price (2.0) forward
    assert b.loc[15:21, "sell_price"].eq(2.0).all()
    assert b.loc[15:21, "price_imputed"].all()
    assert not b.loc[1:14, "price_imputed"].any()

    c = long[long["item_id"] == "ITEM_C"].set_index("d")
    # pre-launch must stay NaN: back-filling would leak the launch price into the past
    assert c.loc[1:14, "sell_price"].isna().all()


def test_launch_trimming(long):
    c = long[long["item_id"] == "ITEM_C"].set_index("d")
    assert not c.loc[1:14, "launched"].any()
    assert c.loc[15:, "launched"].all()

    s = get_series(long, "ITEM_C", end_d=N_DAYS)
    assert s.index[0] == pd.Timestamp("2011-01-29") + pd.Timedelta(days=14)
    assert len(s) == N_DAYS - 14


def test_get_series_respects_end_d(long):
    s = get_series(long, "ITEM_A", end_d=30)
    assert s.index.max() == pd.Timestamp("2011-01-29") + pd.Timedelta(days=29)
    assert s.index.freq == "D"
    assert (s.values == [d % 5 for d in range(1, 31)]).all()


def test_get_series_unknown_item(long):
    with pytest.raises(KeyError):
        get_series(long, "NOPE", end_d=10)


def test_zero_sales_are_kept(long):
    s = get_series(long, "ITEM_A", end_d=N_DAYS)
    assert (s == 0).sum() == N_DAYS // 5  # d % 5 == 0


def test_validation_catches_duplicates_and_negatives(long):
    dup = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_long(dup, n_series=len(ITEMS), max_day=N_DAYS)

    neg = long.copy()
    neg.loc[0, "sales"] = -1
    with pytest.raises(DataValidationError, match="negative"):
        validate_long(neg, n_series=len(ITEMS), max_day=N_DAYS)


def test_unknown_store(raw_dir):
    with pytest.raises(DataValidationError):
        build_long_table(raw_dir, "ZZ_9", max_day=N_DAYS)


def test_audit_table(long):
    audit = audit_table(long).set_index("item_id")
    assert audit.loc["ITEM_C", "launch_d"] == 15
    assert audit.loc["ITEM_A", "n_days"] == N_DAYS
    assert audit.loc["ITEM_B", "price_imputed_days"] == 7
