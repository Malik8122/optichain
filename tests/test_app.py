"""Dashboard smoke test: every page renders without an exception (skipped until the notebook has run)."""
from pathlib import Path

import pytest

from optichain.config import load_config

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (load_config().paths.artifacts_dir / "summary.json").exists(),
                                reason="run the notebook first: python scripts/build_notebook.py --run")
PAGES = ["overview", "demand", "forecast", "model_check", "planner", "outlook", "method"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_planner_reacts_and_choice_carries_over(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    at.switch_page("views/planner.py").run()
    at.slider(key="w_plan_service").set_value(0.9).run()
    assert not at.exception, [e.message for e in at.exception]
    at.switch_page("views/outlook.py").run()
    assert not at.exception and at.session_state["plan_service"] == 0.9


def test_model_check_for_one_store_product(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    at.switch_page("views/model_check.py").run()
    at.segmented_control(key="mc_mode").set_value("One store-product").run()
    assert not at.exception, [e.message for e in at.exception]


def test_following_a_hero_product_updates_the_story(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    hero_buttons = [b for b in at.button if b.key and b.key.startswith("hero_")]
    assert len(hero_buttons) >= 4
    hero_buttons[1].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["hero_row"] == int(hero_buttons[1].key.split("_")[1])
