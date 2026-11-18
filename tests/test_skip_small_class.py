"""A task whose row-capped labels leave a class with < n_folds members is skipped, not crashed.

Also covers a singleton class, where the stratified row cap itself is impossible
(the openml_488__s0 shard crash).
"""

import numpy as np

from src.preprocessing import apply_row_cap, encode_labels
from src.run_portfolio import _process_task, task_skip_reason


def _cfg(n_folds=3):
    return {"cv": {"n_folds": n_folds, "fit_timeout_sec": 30}, "preprocessing": {"row_cap": 1000}}


def _save_task(path, y):
    rng = np.random.RandomState(0)
    X = rng.randn(len(y), 4)
    np.savez(path, X=X, y=y, feature_names=np.array([f"f{i}" for i in range(4)]),
             task_id=path.stem, parent_id="parent_x", source="synthetic",
             row_frac=1.0, col_frac=1.0)


def test_class_with_two_members_after_cap_is_skipped(tmp_path):
    # 1500 rows, rare class of 3 -> stratified cap to 1000 keeps 2 of them
    y = np.array([0] * 1497 + [1] * 3)
    _, y_cap = apply_row_cap(np.zeros((len(y), 1)), y, 1000, 42)
    assert np.bincount(y_cap)[1] == 2

    task = tmp_path / "rare__s0.npz"
    _save_task(task, y)
    cache = tmp_path / "cache"
    cache.mkdir()

    info, rows, fails = _process_task(task, ["gnb", "dt_d3"], cache, 1000, 42, _cfg())
    assert "_skipped" in info and "2 member" in info["_skipped"]
    assert rows == [] and fails == 0
    assert list(cache.iterdir()) == []  # no fits written


def test_singleton_class_cap_impossible_is_skipped_without_exception():
    y, _ = encode_labels(np.array([0] * 1160 + [1]))
    reason = task_skip_reason(y, len(y), 1000, 42, 3)
    assert reason is not None and "row cap impossible" in reason


def test_normal_task_not_skipped():
    y = np.array([0] * 700 + [1] * 800)
    assert task_skip_reason(y, len(y), 1000, 42, 3) is None
