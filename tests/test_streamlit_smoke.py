"""Smoke tests for all Streamlit pages using Streamlit's official AppTest harness."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_home_page_renders_without_error():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_streamlit_overview_dashboard_renders_without_error():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/pages/1_Overview_Dashboard.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_streamlit_customer_prediction_page_renders_without_error():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/pages/2_Customer_Prediction.py"), default_timeout=20)
    at.run()
    assert not at.exception


def test_streamlit_model_performance_page_renders_without_error():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/pages/3_Model_Performance.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 6


def test_streamlit_ai_assistant_page_renders_without_error():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_AI_Assistant.py"), default_timeout=20)
    at.run()
    assert not at.exception
