"""Generate all EDA figures and tables for the selected series (data up to the first origin).

Usage: python scripts/run_eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from optichain import eda
from optichain import plots as P
from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import get_series


def main() -> None:
    cfg = load_config()
    P.apply_style()
    items = cfg.series.items
    end_d = min(cfg.forecast.origins)
    window = cfg.forecast.fit_window_days
    store = cfg.data.store_id.lower()

    long = pd.read_parquet(cfg.data.processed_dir / "selected_series.parquet")
    classes = pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / f"{store}_demand_classes.csv")
    item_class = classes.set_index("item_id")["demand_class"]

    figs = PROJECT_ROOT / "reports" / "figures" / "eda"
    metrics = PROJECT_ROOT / "reports" / "metrics"

    var = eda.variability_table(long, items, end_d, window)
    var.to_csv(metrics / "eda_variability.csv")
    wd = eda.weekday_index_table(long, items, end_d, window)
    wd.to_csv(metrics / "eda_weekday_index.csv")
    cal = eda.calendar_effects(long, items, end_d, window)
    cal.to_csv(metrics / "eda_calendar_effects.csv")

    P.save(eda.fig_adi_cv2(classes, items), figs / "00_adi_cv2_classification.png")
    P.save(eda.fig_small_multiples(long, items, item_class, end_d, window), figs / "01_overview_small_multiples.png")
    P.save(eda.fig_weekday_heatmap(wd), figs / "02_weekday_index_heatmap.png")
    top = items[0]
    s_top = get_series(long, top, end_d)
    years = sorted(s_top.index.year.unique())[-3:-1]  # last two complete years
    P.save(eda.fig_calendar_heatmap(s_top, f"{top} — daily units sold (calendar view)", years),
           figs / f"03_calendar_heatmap_{top}.png")
    plt.close("all")

    for item in items:
        s = get_series(long, item, end_d)
        label = f"{item} ({item_class[item]})"
        P.save(eda.fig_daily(s, f"{label} — daily sales"), figs / item / "daily.png")
        P.save(eda.fig_weekly(s, f"{label} — weekly total"), figs / item / "weekly.png")
        recent = s.iloc[-window:]
        P.save(eda.fig_distribution(recent, f"{label} — distribution of daily sales (last 2 years)"),
               figs / item / "distribution.png")
        P.save(eda.fig_weekday(recent, f"{label} — mean by weekday (±95% CI)"), figs / item / "weekday.png")
        wk = eda.weekly_price_sales(long, item, end_d)
        P.save(eda.fig_price_scatter(wk, f"{label} — weekly sales vs price"), figs / item / "price_scatter.png")
        plt.close("all")

    pd.set_option("display.width", 200)
    print("Variability (last 2 years before d_%d):" % end_d)
    print(var[["mean", "median", "std", "cv", "zero_share", "adi", "cv2", "demand_class", "max_zero_run", "mean_price"]].round(3))
    print("\nWeekday index:\n", wd.round(2))
    print("\nCalendar effects:\n", cal.round(3))
    print(f"\nFigures in {figs.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
