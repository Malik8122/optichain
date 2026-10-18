"""Smoke tests: every dashboard page renders without exceptions (needs generated artefacts)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

needed = [ROOT / "reports/forecasts/forecasts.parquet", ROOT / "reports/inventory/frontier.csv",
          ROOT / "reports/metrics/residuals.parquet", ROOT / "data/processed/selected_series.parquet"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in needed), reason="pipeline artefacts not generated")

PAGES = ["views/overview.py", "views/demand.py", "views/forecast.py", "views/model_check.py",
         "views/stock_plan.py", "views/report.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180).run()
    at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("item", ["FOODS_3_586", "HOBBIES_1_030", "FOODS_3_242"])
def test_model_check_for_different_demand_types(item):
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180).run()
    at.session_state["item"] = item
    at.switch_page("views/model_check.py").run()
    assert not at.exception, [e.message for e in at.exception]


def test_product_choice_carries_across_pages():
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180).run()
    at.button(key="go_HOUSEHOLD_1_046").click().run()
    at.switch_page("views/model_check.py").run()
    assert at.selectbox(key="pick_check").value == "HOUSEHOLD_1_046"
