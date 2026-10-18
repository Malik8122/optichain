"""OptiChain dashboard entry point.

Run from the project root:  streamlit run app/Home.py
"""
import sys
from pathlib import Path

import streamlit as st

try:  # installed with `pip install -e .` locally; on Streamlit Cloud fall back to the source tree
    import optichain  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

st.set_page_config(page_title="OptiChain · Demand & Stock Planner", page_icon="📦", layout="wide")

import data  # noqa: E402  (after set_page_config)
import ui  # noqa: E402

ui.inject_css()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True, url_path="overview"),
    st.Page("views/demand.py", title="Understand demand", icon=":material/insights:", url_path="demand"),
    st.Page("views/forecast.py", title="Predict", icon=":material/trending_up:", url_path="predict"),
    st.Page("views/model_check.py", title="Model check", icon=":material/verified:", url_path="check"),
    st.Page("views/stock_plan.py", title="Plan stock", icon=":material/inventory_2:", url_path="plan"),
    st.Page("views/report.py", title="Report", icon=":material/summarize:", url_path="report"),
]
nav = st.navigation(pages, position="top")
data.check_artifacts()
nav.run()
