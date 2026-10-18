"""Cached data access shared by all dashboard pages.

Everything heavy (backtest, model fits) comes precomputed from the notebook in artifacts/.
The stock plan itself is simulated live, so the planner reacts to the model and service level chosen.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from optichain import data as D
from optichain import inventory as I
from optichain import planning as P
from optichain.config import load_config

REQUIRED = ["summary.json", "long.parquet", "classes.parquet", "backtest.parquet", "leaderboard.csv",
            "leaderboard_h.csv", "leaderboard_pattern.csv", "leaderboard_competition.csv", "importance.csv",
            "dispersion.parquet", "tuning.csv", "rounds.parquet", "competition_scores.csv", "benchmark_weekly.csv",
            "season.csv", "audit.csv", "calibration.csv",
            "ts_network.json", "ts_decomposition.parquet", "ts_residuals.parquet", "ts_holdout.parquet",
            "arima_rows.parquet", "error_whiteness.csv"]


@st.cache_resource
def cfg():
    return load_config()


def art(name: str):
    return cfg().paths.artifacts_dir / name


def check_artifacts() -> None:
    missing = [f for f in REQUIRED if not art(f).exists()]
    if missing:
        st.error("The results haven't been generated yet. Run the pipeline notebook once, then refresh:\n\n"
                 "`python scripts/build_notebook.py --run`\n\n(or open `notebooks/OptiChain_VN2_Pipeline.ipynb` and "
                 f"run all cells). Missing: {', '.join(missing)}")
        st.stop()


@st.cache_data
def summary() -> dict:
    return json.loads(art("summary.json").read_text())


@st.cache_data
def table(name: str) -> pd.DataFrame:
    p = art(name)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


@st.cache_resource
def panel() -> D.Panel:
    return D.load_panel(cfg().paths.raw_dir, cfg().competition.week_0)


@st.cache_data
def rows() -> pd.DataFrame:
    """One line per store-product: ids, hierarchy, pattern and headline stats (history only)."""
    p = panel()
    hist = p.sales.loc[:, :p.week0]
    cls = table("classes.parquet").set_index(D.INDEX)
    out = p.master.copy()
    out["pattern"] = cls["pattern"]
    out["adi"], out["cv2"] = cls["adi"], cls["cv2"]
    out["mean_week"] = hist.mean(axis=1)
    out["last13"] = hist.iloc[:, -13:].mean(axis=1)
    launched = D.launch_mask(p.sales).loc[:, :p.week0]
    out["zero_share"] = (hist == 0).where(launched).mean(axis=1)                       # after launch only
    out["oos_share"] = (~p.in_stock.loc[:, :p.week0]).where(launched).mean(axis=1)
    out["total"] = hist.sum(axis=1)
    out = out.reset_index()
    out["row"] = np.arange(len(out))
    out["label"] = [f"Store {s} · Product {q}" for s, q in zip(out.Store, out.Product)]
    return out


@st.cache_data
def heroes() -> list[dict]:
    """Five store-products that each tell a different story, picked by fixed rules from the history.
    Nicknames describe how the product behaves; the real (anonymised) codes stay visible next to them."""
    p = panel()
    hist, ins = p.sales.loc[:, :p.week0], p.in_stock.loc[:, :p.week0]
    launched = D.launch_mask(p.sales).loc[:, :p.week0]
    r = rows().set_index(D.INDEX)
    first = launched.to_numpy().argmax(axis=1)
    mean = hist.where(launched).mean(axis=1).to_numpy()
    wk = hist.columns.isocalendar().week.to_numpy()
    dec = (wk >= 49) & (wk <= 52)
    xmas = (hist.loc[:, dec].where(ins.loc[:, dec]).mean(axis=1) / hist.where(ins & launched).mean(axis=1)).to_numpy()
    oos, pat = r["oos_share"].to_numpy(), r["pattern"].to_numpy()
    rules = [
        ("🎯", "The Steady Seller", "Sells about {mean:.0f} a week, almost never runs out. Easy to plan.",
         (pat == "smooth") & (first == 0) & (oos < .05), mean),
        ("🎄", "The Christmas Rocket", "Sells {xmas:.1f}× more in December than in a normal week.",
         (mean > 5) & (first == 0), np.nan_to_num(xmas)),
        ("🕳️", "The Often-Empty Shelf", "The shelf was empty in {oos:.0%} of weeks: customers left with nothing.",
         mean > 3, oos),
        ("🐢", "The Slow Mover", "Only about {mean:.0f} a week, with many weeks of no sales at all.",
         (mean > .5) & (mean < 2) & (first == 0) & (oos < .1), mean),
        ("🌱", "The Newcomer", "Launched {first} weeks into the data, so it has a short history to learn from.",
         (first > 60) & (mean > 4), mean),
    ]
    out, used = [], set()
    for icon, name, why, ok, score in rules:
        cand = [i for i in np.flatnonzero(ok) if i not in used]
        if not cand:
            continue
        i = int(max(cand, key=lambda j: score[j]))
        used.add(i)
        out.append({"row": i, "icon": icon, "name": name, "label": r.iloc[i]["label"],
                    "why": why.format(mean=mean[i], xmas=xmas[i], oos=oos[i], first=first[i])})
    return out


def hero_of(row: int) -> dict | None:
    return next((h for h in heroes() if h["row"] == row), None)


@st.cache_data
def benchmark_rows() -> dict:
    """The official benchmark's orders, played per store-product (to compare one product's story)."""
    p = panel()
    orders = [o.to_numpy() for o in D.load_benchmark_orders(cfg().paths.raw_dir)]
    _, recs, _ = I.simulate(I.State.from_initial(p.initial), p.sales.iloc[:, p.t0 + 1:p.t0 + 9].to_numpy(), orders, costs())
    out = {key: np.stack([r[key] for r in recs], axis=1) for key in recs[0]}
    out["orders"] = np.stack(orders, axis=1)
    return out


def costs() -> I.Costs:
    c = cfg().competition
    return I.Costs(c.holding_cost, c.shortage_cost)


@st.cache_data(show_spinner=False)
def plan(model: str, service: float) -> dict:
    """Play the VN2 weeks live with the chosen forecast model and service level."""
    p = panel()
    t0 = p.t0
    rounds = table("rounds.parquet")
    k = P.k_vector(table("dispersion.parquet"), model)
    demand = p.sales.iloc[:, t0 + 1:t0 + 9].to_numpy()
    res = P.run_plan(I.State.from_initial(p.initial), demand, rounds, k, model, float(service), costs(),
                     cfg().policy.samples, cfg().seed)
    per_row = {key: np.stack([r[key] for r in res.records], axis=1) for key in res.records[0]}
    return {"weekly": res.weekly, "score": res.score, "setup": res.setup, "orders": np.stack(res.orders, axis=1),
            "per_row": per_row, "end": res.end_state, "next_order": res.next_order, "outlook": res.outlook,
            "k": k, "mu9": P.forecast_matrix(rounds, model, 9)}


def current_row() -> int:
    return int(st.session_state.get("row", default_row()))


@st.cache_data
def default_row() -> int:
    r = rows()
    busy = r[(r.oos_share > 0.08) & (r.pattern == "smooth")].sort_values("total", ascending=False)
    return int(busy.iloc[0]["row"] if len(busy) else r.sort_values("total").iloc[-1]["row"])


def row_picker(key: str) -> int:
    """Store, then product. Shared across pages through st.session_state['row']."""
    r = rows()
    cur = r.iloc[current_row()]
    stores = sorted(r.Store.unique())
    c = st.columns(2)
    store = c[0].selectbox("Store", stores, index=stores.index(cur.Store), key=f"{key}_store")
    prods = sorted(r.loc[r.Store == store, "Product"])
    pi = prods.index(cur.Product) if cur.Store == store and cur.Product in prods else 0
    prod = c[1].selectbox("Product", prods, index=pi, key=f"{key}_prod",
                          help="Only the products sold in this store are listed.")
    row = int(r[(r.Store == store) & (r.Product == prod)].iloc[0]["row"])
    st.session_state["row"] = row
    h = hero_of(row)
    if h:
        st.caption(f"{h['icon']} This is **{h['name']}**, one of the five products followed on the Overview page.")
    return row


def plan_choice() -> tuple[str, float]:
    s = summary()
    return st.session_state.get("plan_model", s["plan_model"]), float(st.session_state.get("plan_service", s["plan_service"]))
