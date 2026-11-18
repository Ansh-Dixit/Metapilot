"""UI helpers: user text is escaped before it reaches HTML."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import gradio_demo as ui  # noqa: E402

EVIL = '<img src=x onerror="alert(1)">'


def _csv(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({EVIL: rng.randn(90), "b": rng.randn(90), "label": ["a", "b", "c"] * 30})
    p = tmp_path / "<script>x.csv"
    df.to_csv(p, index=False)
    return p


def test_file_and_column_names_escaped_in_status(tmp_path):
    _, _, msg = ui.on_file(str(_csv(tmp_path)))
    assert "<script>" not in msg and "&lt;script&gt;" in msg


@pytest.mark.skipif(not (Path(ui.ROOT) / "models" / "metapilot_models.joblib").exists(), reason="models not trained")
def test_recommend_html_escapes_target_and_has_no_jargon(tmp_path, monkeypatch):
    monkeypatch.setattr(ui.hist, "HISTORY_PATH", tmp_path / "history.json")  # keep the real history clean
    p = _csv(tmp_path)
    df = pd.read_csv(p).rename(columns={"label": EVIL + "2"})
    df.to_csv(p, index=False)
    out = ui.do_recommend(str(p), EVIL + "2", 3, "Smart ranker (recommended)")
    html_parts = " ".join(str(x) for x in out[1:6])
    assert "<img" not in html_parts and "onerror=&quot;" in out[1]
    assert "TRY THIS FIRST" in out[2]
    low = html_parts.lower()
    assert "lambdarank" not in low and "regret" not in low and "meta-feature" not in low
    saved = ui.hist.load()
    assert len(saved) == 1 and saved[0]["id"] == out[10]
    panel = ui.render_entry(saved[0])
    assert "<img" not in panel and "TRY THIS FIRST" in panel
