"""Tests for meta-level data splits and leakage prevention.

Verifies:
(a) No parent_id appears in both train and test of any fold.
(b) GroupKFold correctly partitions tasks by parent_id.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.splits import create_splits, verify_no_parent_leakage


def test_no_parent_leakage_synthetic(tmp_path):
    """Test with synthetic task info that no parent appears in both train and test."""
    n_parents = 20
    n_subsamples = 4
    tasks = []
    parents = []
    for p in range(n_parents):
        pid = f"parent_{p:03d}"
        for s in range(n_subsamples):
            tasks.append(f"{pid}__s{s}")
            parents.append(pid)

    df = pd.DataFrame({
        "task_id": tasks,
        "parent_id": parents,
        "source": ["synthetic"] * len(tasks),
    })

    folds_file = str(tmp_path / "test_folds.json")
    n_folds = 5
    folds = create_splits(df, n_folds=n_folds, seed=42, folds_file=folds_file)

    assert len(folds) == len(tasks)
    assert verify_no_parent_leakage(folds, df, n_folds=n_folds)


def test_folds_file_if_exists():
    """If data/meta/folds.json and task_info.parquet exist, verify no parent leakage."""
    folds_path = Path("data/meta/folds.json")
    task_info_path = Path("data/meta/task_info.parquet")

    if not folds_path.exists() or not task_info_path.exists():
        pytest.skip("data/meta/folds.json or task_info.parquet not yet generated")

    with open(folds_path) as f:
        folds = json.load(f)

    task_info = pd.read_parquet(task_info_path)
    n_folds = len(set(folds.values()))

    assert verify_no_parent_leakage(folds, task_info, n_folds=n_folds)

