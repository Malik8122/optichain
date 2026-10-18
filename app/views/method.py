import pandas as pd
import streamlit as st

import data
import ui

ui.page_header(6, "Method", "How the numbers are made, the rules of the game, and what to keep in mind.")
s = data.summary()

ui.section("Pipeline", "Six stages, all in one notebook",
           "Every number in this dashboard comes from notebooks/OptiChain_VN2_Pipeline.ipynb, run top to bottom.")
ui.insights([
    ("📥", "1 · Load & audit", f"Official VN2 files: {s['rows']} store-products, {s['weeks']} weeks "
     f"({s['first_week']} to {s['last_week']}). No gaps, no negative values."),
    ("🧹", "2 · Clean", "Weeks ending with an **empty shelf** and weeks **before launch** are set aside, because their "
     "sales under-count demand."),
    ("📐", "3 · Time-series checks", "ADF stationarity, STL season, ACF/PACF, AIC/BIC model choice, Ljung-Box and "
     "ARCH tests, GARCH: the classical Box-Jenkins workflow (see Model check)."),
    ("🔮", "4 · Forecast & backtest", "10 methods, including an AIC-selected ARIMA per store-product, forecast 1–3 weeks "
     "ahead at 30 past dates. Errors are measured on clean weeks only."),
    ("🎲", "5 · Simulate & tune", "A newsvendor policy orders the service-level quantile of what the arrival week needs. "
     "The level is tuned by replaying 18 past weeks."),
    ("🏁", "6 · Final test", "The 8 VN2 weeks are played once with the official rules, with models refitted each round on "
     "data known then."),
])

c = st.columns([2, 3], gap="medium")
with c[0]:
    ui.section("The rules", "VN2, as played by the organiser")
    st.dataframe(pd.DataFrame({
        "Rule": ["Order frequency", "Lead time", "Leftover stock", "Missed sale", "Backorders", "Weeks 1–2", "Score"],
        "Value": ["Weekly, before the week starts", "2 weeks (order k is on the shelf at the start of week k+2)",
                  "€0.20 per unit per week", "€1.00 per unit", "None: unmet demand is lost",
                  f"Fixed before the first order ({ui.eur(s['setup'], 1)} for everyone)", "Total cost of weeks 3–8"],
    }), hide_index=True, width="stretch")
    st.caption(f"The simulator reproduces the official benchmark score exactly ({ui.eur(s['benchmark_score'], 1)}), "
               "with identical orders.")
with c[1]:
    ui.section("Glossary", "Words used in this dashboard")
    st.markdown(r"""
| Term | Meaning |
|---|---|
| Store-product | One product in one store: the unit of every decision (599 in total). |
| Censored week | A week flagged out of stock: the shelf was empty, so its sales (zero) say nothing about demand. |
| Forecast error | WAPE + \|bias\|, the metric of the VN1 forecasting competition. |
| WAPE | Total absolute error ÷ total demand. |
| Bias | Total over-forecast ÷ total demand (+ = forecasts too high). |
| Service level | Chance that an order covers demand in the week it arrives. |
| Newsvendor | Order the demand quantile that balances leftover cost against missed-sale cost. |
| Backtest | Forecasting past weeks with only the data known then, to measure accuracy honestly. |
""")

ui.section("Uncertainty", "Are the risk ranges trustworthy?",
           "Demand around each forecast is modelled as a negative binomial (variance = k × mean, k estimated per store-product). "
           "On later past weeks, the share of weeks where demand stayed below each quantile:")
cal = data.table("calibration.csv").rename(columns={"model": "Method"})
q_cols = [c_ for c_ in cal.columns if c_.startswith("demand")]
cal[q_cols] = 100 * cal[q_cols]
st.dataframe(cal, hide_index=True, width="stretch", column_config={
    "pooled k": st.column_config.NumberColumn("spread k", format="%.1f"),
    **{c_: st.column_config.NumberColumn(c_, format="%.0f%%") for c_ in q_cols}})
st.caption("A perfect model would show 50%, 80% and 95%. The upper ranges are on the generous side, which is why the "
           "service level is tuned by simulating real costs rather than taken from the textbook.")

ui.section("Limitations", "What to keep in mind")
ui.insights([
    ("⚖️", "Hindsight", "The plan was designed after VN2 ended, knowing which methods did well. Its settings were never "
     "tuned on the eight test weeks, but the leaderboard comparison is not a blind result."),
    ("📉", "Censored demand", "The simulation uses recorded sales as demand, as the competition did. Weeks that ran out "
     "in reality under-state true demand."),
    ("📆", "Short test", "Eight weeks is a short window: a few large store-products can move the score by tens of euros."),
])

ui.section("Data", "Source and citation")
ui.note("Vandeput, N. (2025). <b>VN2 Inventory Planning Challenge</b>. Organised by Nicolas Vandeput (SupChains). "
        "Sponsored by Flieber, Daybreak and SupChains. https://supchains.com/vn2-inventory-competition/ · "
        "Data licence: CC BY 4.0 · Repository: github.com/SupChains/VN2")
audit = data.table("audit.csv")
with st.expander("Data audit"):
    st.dataframe(audit.rename(columns={"fact": "Fact", "value": "Value"}), hide_index=True, width="stretch")
with st.expander("Download the results"):
    for f, label in (("leaderboard.csv", "Forecast leaderboard"), ("competition_scores.csv", "VN2 scores by method"),
                     ("tuning.csv", "Service-level tuning grid")):
        st.download_button(label, data.art(f).read_bytes(), file_name=f, mime="text/csv", key=f"dl_{f}")
