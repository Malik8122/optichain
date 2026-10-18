"""Load the VN2 files and turn raw weekly sales into a clean demand history.

The VN2 data is wide: one row per store-product (599 rows) and one column per week, labelled
with its Monday date. We keep that layout as a ``Panel`` because every model works on the whole
matrix at once, and offer ``to_long`` for plotting and tables.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX = ["Store", "Product"]
SALES_FILES = [f"Week {w} - {(pd.Timestamp('2024-04-08') + pd.Timedelta(weeks=w)):%Y-%m-%d} - Sales.csv"
               for w in range(0, 9)]
OTHER_FILES = ["Week 0 - In Stock.csv", "Week 0 - Master.csv", "Week 0 - 2024-04-08 - Initial State.csv",
               "Week 0 - Submission Template.csv"]
BENCHMARK_FILES = [f"Week {w} - {(pd.Timestamp('2024-04-08') + pd.Timedelta(weeks=w)):%Y-%m-%d} - Order.csv"
                   for w in range(1, 7)]

# Syntetos-Boylan cut-offs for demand patterns
ADI_CUT = 1.32
CV2_CUT = 0.49


@dataclass
class Panel:
    """All VN2 data, aligned on the same 599 rows."""
    sales: pd.DataFrame      # units sold per week; columns are Monday dates (Timestamp)
    in_stock: pd.DataFrame   # True when stock was left at the end of the week
    master: pd.DataFrame     # product and store hierarchy (anonymised codes)
    initial: pd.DataFrame    # stock and orders in transit at the end of week 0
    week0: pd.Timestamp      # last week of history before round 1

    @property
    def index(self) -> pd.MultiIndex:
        return self.sales.index

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.sales.columns

    @property
    def t0(self) -> int:
        """Column position of week 0."""
        return int(self.dates.get_loc(self.week0))

    def week_index(self, week: int) -> int:
        """Column position of competition week ``week`` (0 = 2024-04-08, 1 = 2024-04-15, ...)."""
        return self.t0 + week


def series_labels(index: pd.MultiIndex) -> list[str]:
    return [f"Store {s} · Product {p}" for s, p in index]


# ----------------------------------------------------------------- loading
def download_vn2(raw_dir: Path, base_url: str) -> list[str]:
    """Fetch any missing VN2 file from the official repository (SupChains/VN2). Returns what was fetched."""
    raw_dir = Path(raw_dir)
    (raw_dir / "Benchmark").mkdir(parents=True, exist_ok=True)
    wanted = [(f"Data/{f}", raw_dir / f) for f in SALES_FILES + OTHER_FILES]
    wanted += [(f"Benchmark/Orders - vn2_benchmark/{f}", raw_dir / "Benchmark" / f) for f in BENCHMARK_FILES]
    fetched = []
    for remote, local in wanted:
        if not local.exists():
            url = f"{base_url}/{urllib.parse.quote(remote)}"
            urllib.request.urlretrieve(url, local)
            fetched.append(local.name)
    return fetched


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index(INDEX)


def _dated(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.to_datetime(df.columns)
    return df


def load_panel(raw_dir: Path, week0: str = "2024-04-08") -> Panel:
    """Read the VN2 files. The Week 8 sales file holds every week (history + the 8 played weeks)."""
    raw_dir = Path(raw_dir)
    sales = _dated(_read(raw_dir / SALES_FILES[-1])).astype(float)
    in_stock = _dated(_read(raw_dir / "Week 0 - In Stock.csv")).astype(bool)
    in_stock = in_stock.reindex(index=sales.index, columns=sales.columns)
    return Panel(sales=sales, in_stock=in_stock, master=_read(raw_dir / "Week 0 - Master.csv"),
                 initial=_read(raw_dir / "Week 0 - 2024-04-08 - Initial State.csv"),
                 week0=pd.Timestamp(week0))


def load_benchmark_orders(raw_dir: Path) -> list[pd.Series]:
    """The six official benchmark order files, as published by the organiser."""
    out = []
    for f in BENCHMARK_FILES:
        df = _read(Path(raw_dir) / "Benchmark" / f)
        out.append(df.iloc[:, 0].astype(int))
    return out


# ----------------------------------------------------------------- preprocessing
def launch_mask(sales: pd.DataFrame) -> pd.DataFrame:
    """True from the first week with a sale onwards. Weeks before it: the product was not on sale yet."""
    return sales.cumsum(axis=1) > 0


def clean_demand(sales: pd.DataFrame, in_stock: pd.DataFrame) -> pd.DataFrame:
    """Sales we can trust as demand: NaN where the shelf ran empty (sales understate demand)
    or before the product's first sale (it was not on the shelf yet)."""
    return sales.where(in_stock & launch_mask(sales))


def to_long(panel: Panel) -> pd.DataFrame:
    """One row per store-product-week, with flags and the clean demand."""
    launched = launch_mask(panel.sales)
    demand = clean_demand(panel.sales, panel.in_stock)
    long = pd.DataFrame({
        "sales": panel.sales.stack(),
        "in_stock": panel.in_stock.stack(),
        "launched": launched.stack(),
        "demand": demand.stack(future_stack=True),
    }).reset_index()
    long = long.rename(columns={"level_2": "week"})
    long["phase"] = np.where(long["week"] <= panel.week0, "history", "competition")
    return long.merge(panel.master.reset_index(), on=INDEX, how="left")


def classify_demand(demand: pd.DataFrame) -> pd.DataFrame:
    """ADI (average weeks between sales) and CV² (variability of sale sizes) per row, on clean weeks.

    smooth: regular and steady · erratic: regular, sizes swing · intermittent: rare, steady sizes ·
    lumpy: rare and swinging. Cut-offs from Syntetos & Boylan (2005).
    """
    rows = []
    for key, row in demand.iterrows():
        v = row.dropna().to_numpy()
        nz = v[v > 0]
        adi = len(v) / len(nz) if len(nz) else np.inf
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        rows.append((*key, len(v), len(nz), v.mean() if len(v) else 0.0, adi, cv2))
    out = pd.DataFrame(rows, columns=INDEX + ["clean_weeks", "selling_weeks", "mean_demand", "adi", "cv2"])
    out["pattern"] = np.select(
        [(out.adi < ADI_CUT) & (out.cv2 < CV2_CUT), (out.adi < ADI_CUT), (out.cv2 < CV2_CUT)],
        ["smooth", "erratic", "intermittent"], default="lumpy")
    return out.set_index(INDEX)


def audit(panel: Panel) -> pd.DataFrame:
    """Headline facts about the raw data, as a two-column table."""
    s, i = panel.sales, panel.in_stock
    hist = s.loc[:, :panel.week0]
    launched = launch_mask(s)
    facts = {
        "Store-product rows": len(s),
        "Stores": s.index.get_level_values("Store").nunique(),
        "Products": s.index.get_level_values("Product").nunique(),
        "Weeks in total": s.shape[1],
        "History weeks (to week 0)": hist.shape[1],
        "Competition weeks (1-8)": s.shape[1] - hist.shape[1],
        "First week": f"{s.columns[0]:%Y-%m-%d}",
        "Last week": f"{s.columns[-1]:%Y-%m-%d}",
        "Missing values": int(s.isna().sum().sum()),
        "Negative sales": int((s < 0).sum().sum()),
        "Non-integer sales": int((s % 1 != 0).sum().sum()),
        "Units sold in total": int(s.to_numpy().sum()),
        "Weeks with zero sales": f"{(s == 0).to_numpy().mean():.1%}",
        "Weeks ending out of stock": f"{(~i).to_numpy().mean():.1%}",
        "Weeks before launch": f"{(~launched).to_numpy().mean():.1%}",
        "Rows ever out of stock": f"{(~i).any(axis=1).mean():.1%}",
    }
    return pd.DataFrame({"value": [str(v) for v in facts.values()]}, index=pd.Index(facts.keys(), name="fact"))


def check_weekly_files(raw_dir: Path, panel: Panel) -> str:
    """Each weekly sales file must keep the same rows, repeat the earlier weeks unchanged and add one week."""
    for w, f in enumerate(SALES_FILES):
        s = _dated(_read(Path(raw_dir) / f))
        assert s.index.equals(panel.sales.index), f
        assert np.array_equal(s.to_numpy(float), panel.sales.loc[:, s.columns].to_numpy()), f
        assert s.columns[-1] == panel.week0 + pd.Timedelta(weeks=w), f
    return f"All {len(SALES_FILES)} sales files are consistent: same rows, same history, one extra week each."
