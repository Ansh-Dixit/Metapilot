"""GroupKFold splits by parent_id. Saved once to folds.json."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import load_config, get_mode_from_args


def create_splits(task_info: pd.DataFrame, n_folds: int, seed: int, folds_file: str) -> dict:
    """Create GroupKFold splits by parent_id and save to folds.json.

    Returns dict: {task_id: fold_index}.
    """
    task_ids = task_info["task_id"].values
    parent_ids = task_info["parent_id"].values

    # GroupKFold doesn't use random_state, but we sort to ensure determinism
    # Sort by task_id for reproducibility
    sort_idx = np.argsort(task_ids)
    task_ids = task_ids[sort_idx]
    parent_ids = parent_ids[sort_idx]

    try:  # sklearn >= 1.6: seeded shuffling of groups across folds
        gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:
        gkf = GroupKFold(n_splits=n_folds)
    # GroupKFold splits by groups - we use parent_id as the group
    # We need dummy X and y for the API
    X_dummy = np.zeros((len(task_ids), 1))
    y_dummy = np.zeros(len(task_ids))

    folds = {}
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=parent_ids)):
        for idx in test_idx:
            folds[task_ids[idx]] = fold_idx

    # Save
    os.makedirs(os.path.dirname(folds_file), exist_ok=True)
    with open(folds_file, "w") as f:
        json.dump(folds, f, indent=2)

    print(f"Saved {n_folds}-fold GroupKFold splits to {folds_file}")
    print(f"  Tasks per fold: {[list(folds.values()).count(i) for i in range(n_folds)]}")

    return folds


def load_splits(folds_file: str) -> dict:
    """Load folds.json."""
    with open(folds_file) as f:
        return json.load(f)


def verify_no_parent_leakage(folds: dict, task_info: pd.DataFrame, n_folds: int) -> bool:
    """Verify no parent_id appears in both train and test of any fold."""
    task_to_parent = dict(zip(task_info["task_id"], task_info["parent_id"]))

    for fold_idx in range(n_folds):
        test_tasks = [t for t, f in folds.items() if f == fold_idx]
        train_tasks = [t for t, f in folds.items() if f != fold_idx]

        test_parents = set(task_to_parent[t] for t in test_tasks)
        train_parents = set(task_to_parent[t] for t in train_tasks)

        overlap = test_parents & train_parents
        if overlap:
            print(f"LEAKAGE in fold {fold_idx}: parents {overlap} in both train and test!")
            return False

    return True


if __name__ == "__main__":
    mode = get_mode_from_args()
    cfg = load_config(mode)

    task_info = pd.read_parquet(os.path.join(cfg["paths"]["meta"], "task_info.parquet"))
    folds = create_splits(task_info, cfg["splits"]["n_folds"], cfg["seed"], cfg["splits"]["file"])

    ok = verify_no_parent_leakage(folds, task_info, cfg["splits"]["n_folds"])
    print(f"Parent leakage check: {'PASS' if ok else 'FAIL'}")
