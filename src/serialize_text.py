"""Serialize tasks to text prompts and targets for Phase 2 (LLM pipeline ranking).

Variants
--------
A : numeric only. The 14 meta-features (3 decimals), n_rows, n_cols,
    n_classes, minority %, missing % and column dtype counts.
B : A plus up to the first 40 column names and the OpenML dataset description
    (cleaned, truncated to 300 characters; empty for synthetic/sklearn tasks).

Every prompt lists the 30 candidate pipeline_ids and asks for the top 5, best first.
The target is the 5 pipelines with the highest measured accuracy for the task,
comma separated. Ties are broken by the training-fold Single-best order.

Leakage rules
-------------
- `build_prompt` never sees the performance matrix: prompts carry no accuracy or label info.
- For test fold f, the training set contains only tasks whose fold != f.
- Single-best orders (used for tie-breaking and by the parser) come from
  training-fold tasks only.

Output layout (``data/phase2/``)::

    prompts/{A,B}/fold{f}_train.jsonl   {"task_id", "prompt", "completion"}
    prompts/{A,B}/fold{f}_test.jsonl    {"task_id", "prompt"}       (no target)
    sb_order.json                       {"fold{f}": [pipeline_id, ...]}
    candidates.json                     the 30 pipeline_ids in prompt order
    openml_descriptions.csv             parent_id, description (cleaned, full)

Usage::

    python -m src.serialize_text --mode full
"""

from __future__ import annotations

import ast
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_mode_from_args, load_config

TEST_FOLDS = (0, 1, 2)
TOP_K = 5
MAX_COLUMN_NAMES = 40
MAX_COLUMN_NAME_CHARS = 40
MAX_DESCRIPTION_CHARS = 300
VARIANTS = ("A", "B")

SYSTEM_PROMPT = (
    "You are an AutoML assistant. Given a description of a tabular classification "
    "dataset, rank machine-learning pipelines by expected 3-fold CV accuracy."
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OPENML_CACHE = _PROJECT_ROOT / "data" / "raw" / "openml_cache" / "org" / "openml" / "www" / "datasets"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def clean_description(raw: str) -> str:
    """Strip OpenML boilerplate (author/source/cite lines, markdown, URLs) and collapse whitespace."""
    text = html.unescape(raw or "")
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*\**\s*(author|source|please cite|citation|creators?|donor|date)\b", line, re.I):
            continue
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # markdown links -> text
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*#_`>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_openml_descriptions(parent_ids, cache_dir: Path = _OPENML_CACHE) -> dict[str, str]:
    """Read cleaned descriptions from the local OpenML cache (no network)."""
    out = {}
    for pid in parent_ids:
        if not pid.startswith("openml_"):
            continue
        f = cache_dir / pid.split("_", 1)[1] / "description.xml"
        if f.exists():
            m = re.search(r"<oml:description>(.*?)</oml:description>", f.read_text(errors="ignore"), re.S)
            out[pid] = clean_description(m.group(1)) if m else ""
    return out


def load_column_names(tasks_dir: Path, task_ids) -> dict[str, list[str]]:
    names = {}
    for t in task_ids:
        with np.load(tasks_dir / f"{t}.npz", allow_pickle=True) as d:
            names[t] = [str(n) for n in d["feature_names"]]
    return names


def dtype_counts(dtypes_repr: str) -> dict[str, int]:
    """Group the per-column type names recorded in text_metadata into numeric / string / other."""
    try:
        types = ast.literal_eval(dtypes_repr)
    except (ValueError, SyntaxError):
        types = []
    counts = {"numeric": 0, "string": 0, "other": 0}
    for t in types:
        t = str(t)
        if t.startswith(("float", "int", "uint", "bool")):
            counts["numeric"] += 1
        elif t in ("str", "object", "category"):
            counts["string"] += 1
        else:
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# Prompt / target construction
# ---------------------------------------------------------------------------

def build_prompt(
    variant: str,
    meta_row: pd.Series,
    text_row: pd.Series,
    candidates: list[str],
    column_names: list[str] | None = None,
    description: str = "",
) -> str:
    """Build the user prompt. Deliberately takes NO performance information."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    dc = dtype_counts(text_row["dtypes"])
    lines = [
        "Dataset summary:",
        f"- rows: {int(text_row['n_rows'])}, columns: {int(text_row['n_cols'])}, classes: {int(text_row['n_classes'])}",
        f"- minority class: {float(text_row['minority_pct']):.1f}%, missing values: {float(text_row['missing_pct']):.1f}%",
        f"- column types: {dc['numeric']} numeric, {dc['string']} string, {dc['other']} other",
        "Meta-features:",
    ]
    lines += [f"- {k}: {float(v):.3f}" for k, v in meta_row.items()]
    if variant == "B":
        cols = [c[:MAX_COLUMN_NAME_CHARS] for c in (column_names or [])[:MAX_COLUMN_NAMES]]
        more = len(column_names or []) - len(cols)
        lines.append("Column names: " + ", ".join(cols) + (f" (+{more} more)" if more > 0 else ""))
        lines.append("Description: " + (description[:MAX_DESCRIPTION_CHARS] if description else "(none)"))
    lines += [
        "Candidate pipelines: " + ", ".join(candidates),
        f"Return the {TOP_K} pipelines most likely to achieve the highest accuracy, best first, "
        "as a comma-separated list of pipeline ids and nothing else.",
    ]
    return "\n".join(lines)


def single_best_order(perf: pd.DataFrame, train_tasks: list[str]) -> list[str]:
    """Pipelines sorted by mean training-fold accuracy (desc); ties by pipeline_id."""
    means = perf.loc[train_tasks].mean(axis=0)
    return sorted(perf.columns, key=lambda p: (-means[p], p))


def build_target(acc_row: pd.Series, sb_order: list[str], k: int = TOP_K) -> str:
    """Top-k pipelines by measured accuracy, ties broken by Single-best order."""
    rank = {p: i for i, p in enumerate(sb_order)}
    ordered = sorted(acc_row.index, key=lambda p: (-acc_row[p], rank[p]))
    return ", ".join(ordered[:k])


def build_fold_sets(folds: dict[str, int], test_fold: int, task_ids: list[str]):
    """(train_tasks, test_tasks) for a test fold; train = every other fold."""
    train = [t for t in task_ids if folds[t] != test_fold]
    test = [t for t in task_ids if folds[t] == test_fold]
    return train, test


def serialize_all(cfg: dict, out_dir: Path | None = None) -> dict:
    meta_dir = Path(cfg["paths"]["meta"])
    tasks_dir = Path(cfg["paths"]["tasks"])
    out_dir = Path(out_dir) if out_dir else _PROJECT_ROOT / "data" / "phase2"

    perf = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta = pd.read_parquet(meta_dir / "meta_features.parquet")
    text = pd.read_parquet(meta_dir / "text_metadata.parquet").set_index("task_id")
    info = pd.read_parquet(meta_dir / "task_info.parquet").set_index("task_id")
    with open(cfg["splits"]["file"]) as f:
        folds = json.load(f)

    task_ids = list(perf.index)  # only tasks that are in the performance matrix
    candidates = list(perf.columns)
    descriptions = load_openml_descriptions(sorted(info["parent_id"].unique()))
    col_names = load_column_names(tasks_dir, task_ids)

    prompts = {
        v: {t: build_prompt(v, meta.loc[t], text.loc[t], candidates, col_names[t],
                            descriptions.get(info.loc[t, "parent_id"], ""))
            for t in task_ids}
        for v in VARIANTS
    }

    sb_orders, stats = {}, {"n_tasks": len(task_ids), "folds": {}}
    for f in TEST_FOLDS:
        train, test = build_fold_sets(folds, f, task_ids)
        sb = single_best_order(perf, train)
        sb_orders[f"fold{f}"] = sb
        stats["folds"][f"fold{f}"] = {"train": len(train), "test": len(test)}
        for v in VARIANTS:
            d = out_dir / "prompts" / v
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"fold{f}_train.jsonl", "w") as fh:
                for t in train:
                    fh.write(json.dumps({"task_id": t, "prompt": prompts[v][t],
                                         "completion": build_target(perf.loc[t], sb)}) + "\n")
            with open(d / f"fold{f}_test.jsonl", "w") as fh:
                for t in test:
                    fh.write(json.dumps({"task_id": t, "prompt": prompts[v][t]}) + "\n")

    (out_dir / "sb_order.json").write_text(json.dumps(sb_orders, indent=1))
    (out_dir / "candidates.json").write_text(json.dumps(candidates))
    (out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT)
    pd.DataFrame(sorted(descriptions.items()), columns=["parent_id", "description"]).to_csv(
        out_dir / "openml_descriptions.csv", index=False)
    for v in VARIANTS:
        lens = [len(p) for p in prompts[v].values()]
        stats[f"prompt_chars_{v}"] = {"mean": float(np.mean(lens)), "max": int(np.max(lens))}
    (out_dir / "serialize_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    mode = get_mode_from_args()
    print(json.dumps(serialize_all(load_config(mode)), indent=1))
