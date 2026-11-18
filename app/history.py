"""Local recommendation history for the MetaPilot demo (history/history.json).

Stores ONLY summaries: never the uploaded data or any column values. Each entry
is built from a whitelist of fields (`make_entry`), written atomically (temp file,
then rename) and capped at the latest MAX_ENTRIES. A missing or corrupt file
reads as an empty history (a corrupt file is kept aside as *.corrupt-<time>.json).
"""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = Path(os.environ.get("METAPILOT_HISTORY", ROOT / "history" / "history.json"))
MAX_ENTRIES = 100
_LOCK = threading.Lock()

CSV_FIELDS = ["id", "timestamp", "dataset", "rows", "columns", "classes", "target", "model",
              "top_pick", "safe_list", "tested", "best_tested_pipeline", "best_tested_accuracy_pct"]


def _clean_pick(r: dict, with_reasons: bool = False) -> dict:
    out = {"rank": int(r["rank"]), "pipeline_id": str(r["pipeline_id"]),
           "family": str(r.get("family", "")), "description": str(r.get("description", ""))}
    if with_reasons:
        out["reasons"] = [str(x) for x in r.get("reasons", [])][:3]
    return out


def make_entry(dataset: str, summary: dict, target: str, model: str, top_k: list[dict],
               safe_list: list[dict], reasons: list[str]) -> dict:
    """Whitelist-only summary of one recommendation (no data values, no column names besides the target)."""
    picks = [_clean_pick(r) for r in top_k]
    if picks:
        picks[0]["reasons"] = [str(x) for x in reasons][:3]  # plain trait phrases, no values
    return {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "rows": int(summary["rows"]),
        "columns": int(summary["columns"]),
        "classes": int(summary["classes"]),
        "target": str(target),
        "model": str(model),
        "top_pick": picks[0]["pipeline_id"] if picks else "",
        "picks": picks,
        "safe_list": [_clean_pick(r) for r in safe_list],
        "tests": None,
    }


def load(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else HISTORY_PATH
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(e, dict) and "id" in e for e in data):
            raise ValueError("unexpected format")
        return data
    except (ValueError, OSError):
        try:
            path.rename(path.with_name(f"{path.stem}.corrupt-{datetime.now():%Y%m%d%H%M%S}.json"))
        except OSError:
            pass
        return []


def save(entries: list[dict], path: Path | None = None) -> None:
    path = Path(path) if path is not None else HISTORY_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = entries[-MAX_ENTRIES:]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    tmp.replace(path)


def add(entry: dict, path: Path | None = None) -> str:
    path = Path(path) if path is not None else HISTORY_PATH
    with _LOCK:
        entries = load(path)
        entries.append(entry)
        save(entries, path)
    return entry["id"]


def set_tests(entry_id: str, results: list[dict], path: Path | None = None) -> bool:
    """Attach 'Test them on my data' results (pipeline id, accuracy, status) to an entry."""
    path = Path(path) if path is not None else HISTORY_PATH
    clean = [{"pipeline_id": str(r["pipeline_id"]),
              "accuracy_pct": None if r.get("accuracy_pct") is None or r["accuracy_pct"] != r["accuracy_pct"]
              else float(r["accuracy_pct"]),
              "status": str(r.get("status", ""))} for r in results]
    with _LOCK:
        entries = load(path)
        for e in entries:
            if e["id"] == entry_id:
                e["tests"] = clean
                save(entries, path)
                return True
    return False


def clear(path: Path | None = None) -> None:
    path = Path(path) if path is not None else HISTORY_PATH
    with _LOCK:
        save([], path)


def best_test(entry: dict):
    ok = [t for t in (entry.get("tests") or []) if t.get("accuracy_pct") is not None]
    return max(ok, key=lambda t: t["accuracy_pct"]) if ok else None


def newest_first(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)


def table_rows(entries: list[dict]) -> list[list]:
    rows = []
    for e in newest_first(entries):
        b = best_test(e)
        rows.append([e["timestamp"].replace("T", " ")[:16], e["dataset"],
                     f"{e['rows']:,} rows × {e['columns']} cols", e["target"], e["top_pick"],
                     f"{b['accuracy_pct']:.1f}% ({b['pipeline_id']})" if b else "not tested"])
    return rows


def _cell(v):
    """Neutralise spreadsheet formulas in user-provided text (=, +, -, @)."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def export_csv(out_path: Path, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else HISTORY_PATH
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for e in newest_first(load(path)):
            b = best_test(e)
            w.writerow({**{k: _cell(e.get(k)) for k in CSV_FIELDS if k in e},
                        "safe_list": " | ".join(p["pipeline_id"] for p in e.get("safe_list", [])),
                        "tested": bool(e.get("tests")),
                        "best_tested_pipeline": b["pipeline_id"] if b else "",
                        "best_tested_accuracy_pct": b["accuracy_pct"] if b else ""})
    return out_path
