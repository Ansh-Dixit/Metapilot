"""Build phase2_bundle.zip for Google Drive / Colab.

Contents (paths inside the zip):
    data/meta_full/{performance_matrix,meta_features,text_metadata,task_info}.parquet, folds.json
    data/phase2/...           prompts (A/B x folds 0-2, train+test), sb_order.json, candidates.json, ...
    src/{__init__,llm_infer,llm_finetune,serialize_text}.py   (code the notebook imports)
    configs/full.yaml

Usage:  .venv/bin/python -m scripts.make_bundle   (run src.serialize_text --mode full first)
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "phase2_bundle.zip"

FILES = [
    "data/meta_full/performance_matrix.parquet",
    "data/meta_full/meta_features.parquet",
    "data/meta_full/text_metadata.parquet",
    "data/meta_full/task_info.parquet",
    "data/meta_full/folds.json",
    "src/__init__.py",
    "src/llm_infer.py",
    "src/llm_finetune.py",
    "src/serialize_text.py",
    "configs/full.yaml",
]


def main() -> None:
    phase2 = ROOT / "data" / "phase2"
    if not (phase2 / "sb_order.json").exists():
        raise SystemExit("data/phase2 missing: run `python -m src.serialize_text --mode full` first")
    files = [ROOT / f for f in FILES] + sorted(p for p in phase2.rglob("*") if p.is_file())
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
