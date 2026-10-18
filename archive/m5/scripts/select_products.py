"""Classify items by demand pattern and freeze the selection into config.yaml.

Uses only data up to the earliest forecast origin. Refuses to overwrite an existing
selection unless --force is given, so the frozen product set never changes silently.

Usage: python scripts/select_products.py [--force]
"""
import argparse
import re

import pandas as pd

from optichain.config import PROJECT_ROOT, load_config
from optichain.selection import classify_items, select_items


def write_items_to_config(items: list[str]) -> None:
    path = PROJECT_ROOT / "config.yaml"
    text = path.read_text()
    block = "  items:\n" + "".join(f"    - {i}\n" for i in items)
    # Replace the existing items value (inline list or block list) inside `series:`.
    new, n = re.subn(r"  items:.*?\n(?:    - .*\n)*", block, text, count=1)
    if n != 1:
        raise RuntimeError("Could not find series.items in config.yaml")
    path.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if cfg.series.items and not args.force:
        raise SystemExit(f"config.yaml already has {len(cfg.series.items)} items; use --force to reselect")

    store = cfg.data.store_id.lower()
    long = pd.read_parquet(cfg.data.processed_dir / f"{store}_long.parquet")
    end_d = min(cfg.forecast.origins)
    classes = classify_items(long, end_d=end_d, window_days=cfg.forecast.fit_window_days)
    picks = select_items(classes)

    metrics = PROJECT_ROOT / "reports" / "metrics"
    classes.to_csv(metrics / f"{store}_demand_classes.csv", index=False)
    picks.to_csv(metrics / "selected_items.csv", index=False)

    items = picks["item_id"].tolist()
    subset = long[long["item_id"].isin(items)].copy()
    subset["item_id"] = subset["item_id"].cat.remove_unused_categories()
    subset.to_parquet(cfg.data.processed_dir / "selected_series.parquet", index=False)
    write_items_to_config(items)

    print(f"{len(classes)} eligible items classified on {classes['window'].iloc[0]}")
    print(classes["demand_class"].value_counts().to_string())
    print("\nSelected:")
    print(picks[["item_id", "demand_class", "mean", "zero_share", "adi", "cv2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
