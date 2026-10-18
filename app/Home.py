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

st.set_page_config(page_title="OptiChain · Stock Planner", page_icon="📦", layout="wide")

import data  # noqa: E402  (after set_page_config)
import ui  # noqa: E402

ui.inject_css()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True, url_path="overview"),
    st.Page("views/demand.py", title="Demand", icon=":material/insights:", url_path="demand"),
    st.Page("views/forecast.py", title="Forecast", icon=":material/trending_up:", url_path="forecast"),
    st.Page("views/model_check.py", title="Model check", icon=":material/verified:", url_path="check"),
    st.Page("views/planner.py", title="Stock planner", icon=":material/inventory_2:", url_path="planner"),
    st.Page("views/outlook.py", title="Order outlook", icon=":material/notifications_active:", url_path="outlook"),
    st.Page("views/method.py", title="Method", icon=":material/menu_book:", url_path="method"),
]
nav = st.navigation(pages, position="top")
data.check_artifacts()
nav.run()
