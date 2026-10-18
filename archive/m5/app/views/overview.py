import streamlit as st

import data
import ui

cfg = data.cfg()
t = data.tables()

ui.md("""
<div class="oc-hero oc-anim">
  <div class="oc-eyebrow">Demand &amp; stock planner · Walmart store CA_1</div>
  <h1>Know what will sell.<br/>Stock exactly what's needed.</h1>
  <p>OptiChain studies five years of daily sales, predicts the coming weeks, and turns that prediction
  into a simple reorder plan. It then replays the past to show whether planning with forecasts beats a
  simple rule of thumb.</p>
</div>
""")
ui.stepper(0)

n_days = t["long"]["d"].nunique()
ui.kpis([
    {"label": "Products in the store", "value": "3,049", "sub": "Food, household and hobbies"},
    {"label": "History analysed", "value": f"{n_days / 365.25:.1f}", "unit": "years", "sub": f"{n_days:,} days of daily sales"},
    {"label": "Products followed closely", "value": str(len(cfg.series.items)), "sub": "Chosen to cover every sales pattern"},
    {"label": "Months used to test", "value": "4", "sub": "Jan–Apr 2016, never seen when predicting"},
])

ui.section("Step 1", "Choose a product to follow",
           "Each one represents a different way products sell, from steady staples to unpredictable items.")

current = data.current_item()
cols = st.columns(4)
for i, item in enumerate(cfg.series.items):
    s = data.series(item)
    cls = data.demand_class(item)
    recent = s.iloc[-140:].rolling(7, min_periods=1).mean().to_numpy()
    with cols[i % 4]:
        with st.container(key=f"cardhover_{'sel_' if item == current else ''}{item}"):
            ui.md(f"""
            <span class="oc-badge">{ui.CLASS_NAMES[cls]}</span>
            <div style="font-weight:750;color:{ui.TEXT};font-size:1.08rem;margin-top:10px">{ui.item_short(item)}</div>
            <div style="color:{ui.MUTED};font-size:.78rem">{ui.item_category(item)} · {item}</div>
            {ui.sparkline(recent)}
            <div style="font-size:.84rem;color:{ui.TEXT_2};line-height:1.55">
              <div><b style="color:{ui.TEXT};font-size:1rem">{s.iloc[-365:].mean():.1f}</b> sold per day</div>
              <div>{(s.iloc[-365:] == 0).mean():.0%} of days with no sales</div>
            </div>
            """)
            if st.button("Following  ✓" if item == current else "Explore  →", key=f"go_{item}", width="stretch",
                         type="primary" if item == current else "secondary"):
                st.session_state["item"] = item
                st.switch_page("views/demand.py")

ui.section("How it works", "From sales history to a stock decision in three steps")
ui.insights([
    ("📈", "1 · Understand demand", "See when the product sells: busy weekdays, seasons, and how often it doesn't sell at all."),
    ("🔮", "2 · Predict the next weeks", "Several forecasting methods are tested on months they never saw, and the most accurate one is picked."),
    ("📦", "3 · Plan stock", "The forecast sets when to reorder and how much. Replaying Jan–Apr 2016 compares it with a simple rule."),
])
ui.note("Research prototype built on the public Walmart M5 dataset. Stock levels, delivery times and costs are "
        "<b>hypothetical assumptions</b>, because the dataset contains sales only.")
ui.next_step(0)
