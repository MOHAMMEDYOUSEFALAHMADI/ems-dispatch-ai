from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (ROOT / "results" / "simulation_metrics.json").exists(), reason="run experiment first")
def test_dashboard_renders():
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
