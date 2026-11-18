"""Meta-dataset assembly: targets, design matrix, GroupKFold splits."""

import json
import os

import numpy as np
import pandas as pd

from src.config import load_config, get_mode_from_args


def build_targets(perf_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build regression targets: gap = acc - max_acc per task."""
    max_acc = perf_matrix.max(axis=1)
    gap = perf_matrix.subtract(max_acc, axis=0)
    return gap


def build_lambdarank_labels(perf_matrix: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """Per-task min-max rescale of accuracy to integers 0..n_levels."""
    labels = perf_matrix.copy()
    for task_id in labels.index:
        row = labels.loc[task_id]
        rmin, rmax = row.min(), row.max()
        if rmax == rmin:
            labels.loc[task_id] = 0
        else:
            labels.loc[task_id] = np.round((row - rmin) / (rmax - rmin) * n_levels).astype(int)
    return labels


if __name__ == "__main__":
    mode = get_mode_from_args()
    cfg = load_config(mode)
    meta_dir = cfg["paths"]["meta"]

    perf = pd.read_parquet(os.path.join(meta_dir, "performance_matrix.parquet"))
    gap = build_targets(perf)
    gap.to_parquet(os.path.join(meta_dir, "gap_targets.parquet"))
    print(f"Built gap targets: {gap.shape}")

    labels = build_lambdarank_labels(perf, cfg["metamodels"]["relevance_levels"])
    labels.to_parquet(os.path.join(meta_dir, "lambdarank_labels.parquet"))
    print(f"Built LambdaRank labels: {labels.shape}")
