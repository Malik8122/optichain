"""Build data/processed/<store>_long.parquet and a per-item audit table.

Usage: python scripts/build_dataset.py [--config config.yaml]
"""
import argparse
import time

from optichain.config import PROJECT_ROOT, load_config
from optichain.data_pipeline import audit_table, build_long_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    d = cfg.data

    t0 = time.time()
    long = build_long_table(d.raw_dir, d.store_id, d.max_day, d.sales_file)
    d.processed_dir.mkdir(parents=True, exist_ok=True)
    out = d.processed_dir / f"{d.store_id.lower()}_long.parquet"
    long.to_parquet(out, index=False)

    audit = audit_table(long)
    audit_path = PROJECT_ROOT / "reports" / "metrics" / f"{d.store_id.lower()}_data_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    pre_launch = (~long["launched"]).mean()
    print(f"Store {d.store_id}: {long['item_id'].nunique()} items x {d.max_day} days = {len(long):,} rows")
    print(f"Dates {long['date'].min().date()} -> {long['date'].max().date()}")
    print(f"Pre-launch rows (excluded from modelling): {pre_launch:.1%}")
    print(f"Post-launch days with forward-filled price: {int(long['price_imputed'].sum())}")
    print(f"Items launched after d_1: {(audit['launch_d'] > 1).sum()}")
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} and {audit_path.relative_to(PROJECT_ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
