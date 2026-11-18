"""History storage: summaries only, cap at 100, robust to missing/corrupt files, CSV export."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import history as hist  # noqa: E402

SUMMARY = {"rows": 150, "columns": 4, "classes": 3, "minority_share_pct": 33.3, "missing_pct": 0.0,
           "numeric_columns": 4, "categorical_columns": 0, "dropped_columns": ["secret_id (looks like an ID)"],
           "warnings": [], "class_names": ["SECRET_CLASS_A", "b", "c"], "feature_names": ["SECRET_COL"],
           "SECRET_VALUE": 123.456}
TOP = [{"rank": 1, "pipeline_id": "rf_100", "family": "Random forest", "description": "Random forest of 100 trees",
        "score": 0.9, "secret": "SECRET_VALUE"},
       {"rank": 2, "pipeline_id": "svm_c1", "family": "Support vector machine", "description": "SVM", "score": 0.5}]
SAFE = [{"rank": 1, "pipeline_id": "rf_100", "family": "Random forest", "description": "RF", "score": None}]


def _entry(i=0):
    return hist.make_entry(f"data{i}.csv", SUMMARY, "species", "Smart ranker (recommended)", TOP, SAFE,
                           ["Your data has many classes — this model handles that well."])


def test_entry_contains_only_whitelisted_summary(tmp_path):
    p = tmp_path / "h.json"
    hist.add(_entry(), p)
    raw = p.read_text()
    for forbidden in ("SECRET_VALUE", "SECRET_COL", "SECRET_CLASS_A", "secret_id", "123.456", "score"):
        assert forbidden not in raw
    e = json.loads(raw)[0]
    assert set(e) == {"id", "timestamp", "dataset", "rows", "columns", "classes", "target", "model",
                      "top_pick", "picks", "safe_list", "tests"}
    assert e["top_pick"] == "rf_100" and e["tests"] is None


def test_cap_at_100_keeps_latest(tmp_path):
    p = tmp_path / "h.json"
    ids = [hist.add(_entry(i), p) for i in range(105)]
    saved = hist.load(p)
    assert len(saved) == 100 and [e["id"] for e in saved] == ids[-100:]


def test_missing_and_corrupt_files_read_as_empty(tmp_path):
    assert hist.load(tmp_path / "nope.json") == []
    p = tmp_path / "h.json"
    p.write_text("{not json")
    assert hist.load(p) == []
    assert list(tmp_path.glob("h.corrupt-*.json"))  # corrupt file kept aside, not lost
    hist.add(_entry(), p)  # and writing works again
    assert len(hist.load(p)) == 1
    p.write_text(json.dumps({"a": 1}))  # valid JSON, wrong shape
    assert hist.load(p) == []


def test_tests_update_same_entry_and_csv_export(tmp_path):
    p = tmp_path / "h.json"
    eid = hist.add(_entry(), p)
    other = hist.add(_entry(1), p)
    assert hist.set_tests(eid, [{"pipeline_id": "rf_100", "accuracy_pct": 95.3, "status": "ok", "rows_used": 150},
                                {"pipeline_id": "svm_c1", "accuracy_pct": float("nan"), "status": "timeout"}], p)
    saved = {e["id"]: e for e in hist.load(p)}
    assert saved[eid]["tests"][0] == {"pipeline_id": "rf_100", "accuracy_pct": 95.3, "status": "ok"}
    assert saved[eid]["tests"][1]["accuracy_pct"] is None and saved[other]["tests"] is None
    assert not hist.set_tests("missing", [], p)

    rows = hist.table_rows(hist.load(p))
    assert {r[5] for r in rows} == {"95.3% (rf_100)", "not tested"}

    out = hist.export_csv(tmp_path / "out.csv", p)
    with open(out) as f:
        recs = list(csv.DictReader(f))
    assert len(recs) == 2 and set(recs[0]) == set(hist.CSV_FIELDS)
    tested = next(r for r in recs if r["id"] == eid)
    assert tested["best_tested_accuracy_pct"] == "95.3" and tested["safe_list"] == "rf_100"


def test_csv_neutralises_formulas_and_clear(tmp_path):
    p = tmp_path / "h.json"
    e = _entry()
    e["dataset"] = "=HYPERLINK(\"x\")"
    hist.add(e, p)
    out = hist.export_csv(tmp_path / "o.csv", p)
    assert "'=HYPERLINK" in out.read_text()
    hist.clear(p)
    assert hist.load(p) == []
