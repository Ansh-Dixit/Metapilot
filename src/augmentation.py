"""Create task subsamples from parent datasets.

For each parent .npz in ``paths.raw``, generate ``n_subsamples`` tasks
(random row/column subsamples) and persist them as .npz files in
``paths.tasks``.  Subsample 0 always keeps the full data so the
original parent is represented.  Subsequent subsamples draw
``row_frac`` and ``col_frac`` uniformly from the configured ranges,
using stratified row sampling to preserve class proportions.

Tasks whose smallest class has fewer than ``min_class_count`` samples
are dropped (subsample 0 is never dropped).  Existing task files are
skipped to support resumable runs.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.config import get_mode_from_args, load_config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _stratified_sample_indices(
    y: np.ndarray,
    frac: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return indices for a stratified subsample of *y* at the given fraction."""
    classes, counts = np.unique(y, return_counts=True)
    indices: List[np.ndarray] = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_keep = max(1, int(round(frac * cnt)))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        indices.append(chosen)
    return np.sort(np.concatenate(indices))


def _select_columns(
    n_cols: int,
    col_frac: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return sorted column indices for a random column subsample."""
    n_keep = max(2, math.ceil(col_frac * n_cols))
    n_keep = min(n_keep, n_cols)
    chosen = rng.choice(n_cols, size=n_keep, replace=False)
    return np.sort(chosen)


def _passes_min_class_count(y: np.ndarray, min_count: int) -> bool:
    """Return True if every class has at least *min_count* samples."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) >= min_count


def select_parent_files(cfg: Dict) -> List[Path]:
    """Return the parent .npz files that belong to the active mode.

    data/raw is shared by pilot and full mode, so parents are selected from
    the config rather than by globbing everything:
      - sklearn parents listed in ``data.sklearn_parents``
      - ``synth_cls_000 .. synth_cls_{n-1}`` and the configured shapes
      - OpenML parents only if ``data.openml.enabled``; those with more than
        ``data.openml.max_features`` columns are skipped and written to
        ``<meta>/excluded_parents.csv``.
    """
    raw_dir = Path(cfg["paths"]["raw"])
    data_cfg = cfg["data"]
    wanted: List[str] = [f"sk_{n}" for n in data_cfg.get("sklearn_parents", [])]
    n_syn = data_cfg.get("synthetic_classification", {}).get("n_parents", 0)
    wanted += [f"synth_cls_{i:03d}" for i in range(n_syn)]
    shapes = data_cfg.get("synthetic_shapes", {})
    kinds = shapes.get("kinds", [])
    if kinds:
        per_kind = shapes.get("n_parents", 0) // len(kinds)
        wanted += [f"{k}_{i:03d}" for k in kinds for i in range(per_kind)]

    files = [raw_dir / f"{pid}.npz" for pid in wanted if (raw_dir / f"{pid}.npz").exists()]

    oml = data_cfg.get("openml", {})
    excluded = []
    if oml.get("enabled", False):
        max_feat = int(oml.get("max_features", 10**9))
        for f in sorted(raw_dir.glob("openml_*.npz")):
            with np.load(f, allow_pickle=True) as d:
                n_r, n_c = d["X"].shape
            if n_c > max_feat:
                excluded.append({"parent_id": f.stem, "rows": n_r, "columns": n_c,
                                 "reason": f"columns {n_c} > max_features {max_feat}"})
                continue
            files.append(f)
    meta_dir = Path(cfg["paths"]["meta"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    if oml.get("enabled", False):
        import pandas as pd
        pd.DataFrame(excluded, columns=["parent_id", "rows", "columns", "reason"]).to_csv(
            meta_dir / "excluded_parents.csv", index=False)
    return sorted(files)


# ------------------------------------------------------------------
# core
# ------------------------------------------------------------------

def generate_tasks(cfg: Dict) -> List[str]:
    """Generate all task subsamples and return a list of saved task_ids.

    Parameters
    ----------
    cfg : dict
        Full config dict (as returned by :func:`src.config.load_config`).

    Returns
    -------
    list[str]
        The ``task_id`` values for every task that was saved (including
        those that already existed on disk and were skipped).
    """
    raw_dir = Path(cfg["paths"]["raw"])
    tasks_dir = Path(cfg["paths"]["tasks"])
    tasks_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = cfg["augmentation"]
    n_subsamples: int = aug_cfg["n_subsamples"]
    row_frac_lo, row_frac_hi = aug_cfg["row_frac"]
    col_frac_lo, col_frac_hi = aug_cfg["col_frac"]
    min_class_count: int = aug_cfg["min_class_count"]
    base_seed: int = cfg["seed"]

    parent_files = select_parent_files(cfg)
    if not parent_files:
        logger.warning("No parent .npz files found in %s", raw_dir)
        return []

    saved_task_ids: List[str] = []

    for parent_idx, parent_path in enumerate(tqdm(parent_files, desc="augmentation", unit="parent")):
        parent_id = parent_path.stem
        parent_data = np.load(parent_path, allow_pickle=True)
        X_parent = parent_data["X"]
        y_parent = parent_data["y"]
        feature_names_parent = parent_data["feature_names"]
        source = str(parent_data["source"]) if "source" in parent_data else parent_id
        if source in ("moons", "circles", "make_classification") or parent_id.startswith(("synth_cls_", "moons_", "circles_")):
            source = "synthetic"
        elif parent_id.startswith("sk_"):
            source = "sklearn"
        elif parent_id.startswith("openml_"):
            source = "openml"

        n_rows, n_cols = X_parent.shape

        for k in range(n_subsamples):
            task_id = f"{parent_id}__s{k}"
            task_path = tasks_dir / f"{task_id}.npz"

            # ---- resumable: skip existing ---------------------------
            if task_path.exists():
                logger.debug("Skipping existing task %s", task_id)
                saved_task_ids.append(task_id)
                continue

            rng = np.random.RandomState(base_seed + parent_idx * n_subsamples + k)

            # ---- determine row_frac & col_frac ----------------------
            if k == 0:
                row_frac = 1.0
                col_frac = 1.0
            else:
                row_frac = rng.uniform(row_frac_lo, row_frac_hi)
                col_frac = rng.uniform(col_frac_lo, col_frac_hi)

            # ---- column subsample -----------------------------------
            if col_frac >= 1.0:
                col_idx = np.arange(n_cols)
            else:
                col_idx = _select_columns(n_cols, col_frac, rng)

            X_sub = X_parent[:, col_idx]
            feat_sub = feature_names_parent[col_idx]

            # ---- row subsample (stratified) -------------------------
            if row_frac >= 1.0:
                row_idx = np.arange(n_rows)
            else:
                row_idx = _stratified_sample_indices(y_parent, row_frac, rng)

            X_sub = X_sub[row_idx]
            y_sub = y_parent[row_idx]

            # ---- min class count check ------------------------------
            if k > 0 and not _passes_min_class_count(y_sub, min_class_count):
                logger.warning(
                    "Dropping task %s: a class has fewer than %d samples",
                    task_id,
                    min_class_count,
                )
                continue

            # ---- save -----------------------------------------------
            np.savez(
                task_path,
                X=X_sub,
                y=y_sub,
                feature_names=feat_sub,
                task_id=task_id,
                parent_id=parent_id,
                source=source,
                row_frac=np.float64(row_frac),
                col_frac=np.float64(col_frac),
            )
            saved_task_ids.append(task_id)
            logger.info("Saved task %s  (rows=%d, cols=%d)", task_id, len(y_sub), len(col_idx))

    return saved_task_ids


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    mode = get_mode_from_args()
    cfg = load_config(mode)
    task_ids = generate_tasks(cfg)
    print(f"\n=== Augmentation summary ({mode} mode) ===")
    print(f"  Total tasks saved/found: {len(task_ids)}")
    if task_ids:
        parents = sorted({tid.rsplit('__s', 1)[0] for tid in task_ids})
        print(f"  Unique parents:         {len(parents)}")
        print(f"  Task IDs (first 10):    {task_ids[:10]}")
