"""Regenerate data/raw/load_failures.csv from the current state of data/raw.

Lists only OpenML datasets that were attempted and are NOT available as a
parent under the full-mode limits (configs/full.yaml). Loaded parents (e.g.
40978, 1458) are never listed here; loaded parents that full mode skips for
the feature limit are in data/meta_full/excluded_parents.csv instead.

Names and sizes come from the local OpenML cache, so no network is needed.
"""
import csv
import re
from datetime import datetime
from pathlib import Path

from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config("full")
oml = cfg["data"]["openml"]
raw = ROOT / "data" / "raw"
cache = raw / "openml_cache" / "org" / "openml" / "www" / "datasets"


def _quality(qxml: str, name: str):
    m = re.search(rf"<oml:name>{name}</oml:name>\s*<oml:value>([^<]*)</oml:value>", qxml)
    return int(float(m.group(1))) if m and m.group(1) else None


rows = []
for d in sorted(cache.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
    if not d.name.isdigit() or (raw / f"openml_{d.name}.npz").exists():
        continue
    desc = (d / "description.xml").read_text() if (d / "description.xml").exists() else ""
    qual = (d / "qualities.xml").read_text() if (d / "qualities.xml").exists() else ""
    m = re.search(r"<oml:name>([^<]*)</oml:name>", desc)
    name = m.group(1) if m else f"did_{d.name}"
    n_rows, n_feat = _quality(qual, "NumberOfInstances"), _quality(qual, "NumberOfFeatures")
    if n_rows is not None and n_rows > oml["max_rows"]:
        err = f"Too many rows: {n_rows} > {oml['max_rows']}"
    elif n_feat is not None and n_feat > oml["max_features"]:
        err = f"Too many features: {n_feat} > {oml['max_features']}"
    else:
        err = "Not loaded (unknown cause; re-run src.data_loading --mode full to retry)"
    rows.append([int(d.name), name, err, datetime.utcnow().isoformat()])

out = raw / "load_failures.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["did", "dataset_name", "error", "timestamp"])
    w.writerows(rows)
print(f"Wrote {len(rows)} failures to {out}")
