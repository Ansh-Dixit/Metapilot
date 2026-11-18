"""Dry run script for MetaPilot full-study scaling extrapolation.

Runs on a representative sample of 8 parents:
- openml_23517: Largest by rows (96,320 rows)
- openml_1458: Widest by columns (10,000 columns, features > rows)
- openml_1049: Mid-size tabular (1,458 rows, 37 cols)
- openml_12: Multiclass benchmark (2,000 rows, 216 cols, 10 classes)
- openml_1562: Microarray / text-body (64 rows, 4,702 cols)
- sk_breast_cancer: Sklearn real benchmark (569 rows, 30 cols)
- synth_cls_000: Synthetic classification (rectangular rule logic)
- moons_000: Synthetic geometric shape (moons)

10 subsamples each = 80 tasks, evaluated across all 42 pipelines with 3-fold CV
(80 tasks x 42 pipelines x 3 folds = 10,080 fits).
Measures seconds per fit by family, records the 5 slowest (parent, pipeline) pairs,
and updates results/run_progress.json every 100 completed fits.
"""

import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import load_config
from src.portfolio import PIPELINE_REGISTRY, get_all_pipeline_ids, get_family, get_pipeline
from src.preprocessing import apply_row_cap, encode_labels


def _stratified_sample_indices(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_keep = max(1, int(round(frac * cnt)))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        indices.append(chosen)
    return np.sort(np.concatenate(indices))


def _select_columns(n_cols: int, col_frac: float, rng: np.random.RandomState) -> np.ndarray:
    n_keep = max(2, math.ceil(col_frac * n_cols))
    n_keep = min(n_keep, n_cols)
    chosen = rng.choice(n_cols, size=n_keep, replace=False)
    return np.sort(chosen)


def generate_dry_run_tasks(parent_ids: list[str], raw_dir: str, dry_tasks_dir: str, seed: int = 42) -> list[str]:
    """Generate 10 subsamples for each parent."""
    os.makedirs(dry_tasks_dir, exist_ok=True)
    saved_task_ids = []

    for p_idx, pid in enumerate(parent_ids):
        p_path = os.path.join(raw_dir, f"{pid}.npz")
        data = np.load(p_path, allow_pickle=True)
        X_parent = data["X"]
        y_parent = data["y"]
        feat_parent = data["feature_names"]
        source = str(data["source"]) if "source" in data else pid

        n_rows, n_cols = X_parent.shape

        for k in range(10):
            task_id = f"{pid}__s{k}"
            task_path = os.path.join(dry_tasks_dir, f"{task_id}.npz")
            rng = np.random.RandomState(seed + p_idx * 10 + k)

            if k == 0:
                row_frac, col_frac = 1.0, 1.0
            else:
                row_frac = float(rng.uniform(0.3, 1.0))
                col_frac = float(rng.uniform(0.5, 1.0))

            if col_frac >= 1.0:
                col_idx = np.arange(n_cols)
            else:
                col_idx = _select_columns(n_cols, col_frac, rng)

            X_sub = X_parent[:, col_idx]
            feat_sub = feat_parent[col_idx]

            if row_frac >= 1.0:
                row_idx = np.arange(n_rows)
            else:
                row_idx = _stratified_sample_indices(y_parent, row_frac, rng)

            X_sub = X_sub[row_idx]
            y_sub = y_parent[row_idx]

            np.savez(
                task_path,
                X=X_sub,
                y=y_sub,
                feature_names=feat_sub,
                task_id=task_id,
                parent_id=pid,
                source=source,
                row_frac=row_frac,
                col_frac=col_frac,
            )
            saved_task_ids.append(task_id)

    return saved_task_ids


def evaluate_task_dry_run(task_path: str, pipeline_ids: list[str], cfg: dict, seed: int = 42) -> list[dict]:
    """Evaluate all 42 pipelines on one dry-run task, recording precise timing per fold."""
    data = np.load(task_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    task_id = str(data["task_id"])
    parent_id = str(data["parent_id"]) if "parent_id" in data else task_id.split("__")[0]

    y_enc, _ = encode_labels(y)
    row_cap = cfg["preprocessing"]["row_cap"]
    X_eval, y_eval = apply_row_cap(X, y_enc, row_cap, seed)

    n_folds = cfg["cv"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []
    for pid in pipeline_ids:
        family = get_family(pid)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_eval, y_eval)):
            X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
            X_te, y_te = X_eval[test_idx], y_eval[test_idx]

            pipe = get_pipeline(pid, seed=seed + fold_i, X_sample=X_tr, cfg=cfg)

            t0 = time.perf_counter()
            try:
                pipe.fit(X_tr, y_tr)
                acc = float(pipe.score(X_te, y_te))
                status = "ok"
            except Exception as e:
                acc = 0.0
                status = f"error: {e}"
            t_fit = time.perf_counter() - t0

            rows.append({
                "parent_id": parent_id,
                "task_id": task_id,
                "pipeline_id": pid,
                "family": family,
                "fold": fold_i,
                "accuracy": acc,
                "fit_seconds": t_fit,
                "status": status,
            })

    return rows


def _update_run_progress(progress_file: Path, completed_fits: int, total_fits: int, start_time: float) -> None:
    """Atomically write run_progress.json every 100 fits."""
    elapsed = time.perf_counter() - start_time
    pct = (completed_fits / total_fits * 100.0) if total_fits > 0 else 0.0
    fits_per_sec = completed_fits / elapsed if elapsed > 0 else 0.0
    rem = max(0, total_fits - completed_fits)
    eta = rem / fits_per_sec if fits_per_sec > 0 else 0.0
    payload = {
        "completed_fits": completed_fits,
        "total_fits": total_fits,
        "pct_complete": round(pct, 2),
        "elapsed_sec": round(elapsed, 2),
        "fits_per_sec": round(fits_per_sec, 2),
        "eta_sec": round(eta, 2),
        "eta_min": round(eta / 60.0, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = progress_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(payload, f, indent=2)
    temp_file.replace(progress_file)


def run_dry_run(cfg: dict, core_count: int = 10) -> dict:
    """Execute dry run on representative 8 parents and calculate runtime extrapolation."""
    raw_dir = cfg["paths"]["raw"]
    dry_tasks_dir = "data/tasks_dryrun"
    # The dry run has its own progress file so it can never be mistaken for the real run.
    results_dir = Path("results_dry_run")
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_file = results_dir / "run_progress.json"

    # 8 representative parents
    selected_parents = [
        "openml_23517",     # Largest by rows (96,320 rows)
        "openml_1458",      # Widest by columns (10,000 columns, features > rows)
        "openml_1049",      # Mid-size tabular (1,458 rows, 37 cols)
        "openml_12",        # Multiclass benchmark (2,000 rows, 216 cols, 10 classes)
        "openml_1562",      # Microarray / text (64 rows, 4,702 cols)
        "sk_breast_cancer", # Sklearn real benchmark (569 rows, 30 cols)
        "synth_cls_000",    # Diversified synthetic classification (rule logic)
        "moons_000",        # Synthetic geometric shape (moons)
    ]

    print(f"[dry_run] Selected 8 representative parents: {selected_parents}")
    task_ids = generate_dry_run_tasks(selected_parents, raw_dir, dry_tasks_dir, seed=cfg["seed"])
    print(f"[dry_run] Generated {len(task_ids)} tasks (10 subsamples per parent)")

    pipeline_ids = get_all_pipeline_ids()
    n_tasks = len(task_ids)
    n_pipes = len(pipeline_ids)
    n_folds = cfg["cv"]["n_folds"]
    total_fits = n_tasks * n_pipes * n_folds

    print(f"[dry_run] Evaluating {n_pipes} pipelines on {n_tasks} tasks with {n_folds}-fold CV...")
    print(f"[dry_run] Total fits: {total_fits} (parallel execution across {core_count} cores)")

    t_start = time.perf_counter()
    task_paths = [os.path.join(dry_tasks_dir, f"{tid}.npz") for tid in task_ids]

    # Run tasks with progress updates
    results_nested = []
    completed_fits = 0

    # Using Parallel with batching for progress monitoring
    batch_size = max(1, core_count)
    for i in range(0, len(task_paths), batch_size):
        batch = task_paths[i:i + batch_size]
        batch_res = Parallel(n_jobs=core_count)(
            delayed(evaluate_task_dry_run)(t_path, pipeline_ids, cfg, seed=cfg["seed"])
            for t_path in batch
        )
        for task_rows in batch_res:
            results_nested.append(task_rows)
            completed_fits += len(task_rows)
            _update_run_progress(progress_file, completed_fits, total_fits, t_start)

    wall_time = time.perf_counter() - t_start
    _update_run_progress(progress_file, total_fits, total_fits, t_start)

    all_fits = [row for task_rows in results_nested for row in task_rows]
    fits_df = pd.DataFrame(all_fits)

    # Clean up temporary dry run tasks directory
    if os.path.exists(dry_tasks_dir):
        shutil.rmtree(dry_tasks_dir)

    # 1. Compute statistics by family
    family_stats = []
    total_cpu_seconds = fits_df["fit_seconds"].sum()

    for fam, group in fits_df.groupby("family"):
        mean_sec = group["fit_seconds"].mean()
        sum_sec = group["fit_seconds"].sum()
        n_fits = len(group)
        share_pct = (sum_sec / total_cpu_seconds) * 100.0 if total_cpu_seconds > 0 else 0.0
        n_p = len(group["pipeline_id"].unique())
        family_stats.append({
            "family": fam,
            "pipelines": n_p,
            "fits": n_fits,
            "mean_sec_per_fit": mean_sec,
            "total_sec": sum_sec,
            "share_pct": share_pct,
        })

    fam_df = pd.DataFrame(family_stats).sort_values("share_pct", ascending=False).reset_index(drop=True)

    # 2. 5 slowest (parent, pipeline) pairs
    slowest_pairs = (
        fits_df.groupby(["parent_id", "pipeline_id", "family"])["fit_seconds"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(5)
    )
    slowest_pairs.columns = ["Parent", "Pipeline", "Family", "Mean Sec/Fit", "Max Sec/Fit", "Fits"]

    # 3. Full study extrapolation
    # Total parents in full study: 195 parents on disk (161 OpenML + 4 sklearn + 30 synthetic)
    # 195 parents x 10 subsamples = 1,950 tasks
    # 1,950 tasks x 42 pipelines x 3 folds = 245,700 fits
    full_n_tasks = 1950
    extrap_rows = []
    total_extrap_sec = 0.0

    for _, r in fam_df.iterrows():
        fam = r["family"]
        n_p = r["pipelines"]
        fam_total_fits = full_n_tasks * n_p * 3
        fam_extrap_sec = fam_total_fits * r["mean_sec_per_fit"]
        total_extrap_sec += fam_extrap_sec
        extrap_rows.append({
            "Family": fam,
            "Configs": n_p,
            "Total Fits": fam_total_fits,
            "Sec/Fit": f"{r['mean_sec_per_fit']:.4f}s",
            "Extrap Total (h)": f"{fam_extrap_sec / 3600.0:.2f}h",
            "Cost Share (%)": f"{r['share_pct']:.1f}%",
        })

    extrap_df = pd.DataFrame(extrap_rows)
    hours_1core = total_extrap_sec / 3600.0
    # Amdahl / parallel efficiency factor ~0.85
    hours_cores = (total_extrap_sec / (core_count * 0.85)) / 3600.0

    return {
        "wall_time": wall_time,
        "total_fits": len(fits_df),
        "core_count": core_count,
        "fam_df": fam_df,
        "slowest_pairs": slowest_pairs,
        "extrap_df": extrap_df,
        "hours_1core": hours_1core,
        "hours_cores": hours_cores,
        "total_extrap_sec": total_extrap_sec,
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print(f"System real CPU count: os.cpu_count() = {os.cpu_count()}")
    print("=" * 60)
    cfg = load_config("full")
    core_count = os.cpu_count() or 10
    res = run_dry_run(cfg, core_count=core_count)

    print("\n" + "=" * 60)
    print(f"REPRESENTATIVE DRY RUN COMPLETE in {res['wall_time']:.1f}s ({res['total_fits']} fits across 8 parents)")
    print(f"Extrapolated Full Study (195 parents × 10 subsamples = 245,700 fits):")
    print(f"  Single Core (n_jobs=1): {res['hours_1core']:.2f} hours ({res['hours_1core']*60:.1f} minutes)")
    print(f"  {res['core_count']} Cores (n_jobs={res['core_count']}):  {res['hours_cores']:.2f} hours ({res['hours_cores']*60:.1f} minutes)")
    print("=" * 60)

    print("\n1. Family Cost Breakdown:")
    df_out = res["extrap_df"]
    header = "| " + " | ".join(df_out.columns) + " |"
    div = "| " + " | ".join(["---"] * len(df_out.columns)) + " |"
    rows = ["| " + " | ".join(str(val) for val in row) + " |" for _, row in df_out.iterrows()]
    print("\n".join([header, div] + rows))

    print("\n2. Top 5 Slowest (Parent, Pipeline) Pairs:")
    df_slow = res["slowest_pairs"].copy()
    df_slow["Mean Sec/Fit"] = df_slow["Mean Sec/Fit"].apply(lambda v: f"{v:.4f}s")
    df_slow["Max Sec/Fit"] = df_slow["Max Sec/Fit"].apply(lambda v: f"{v:.4f}s")
    header_s = "| " + " | ".join(df_slow.columns) + " |"
    div_s = "| " + " | ".join(["---"] * len(df_slow.columns)) + " |"
    rows_s = ["| " + " | ".join(str(val) for val in row) + " |" for _, row in df_slow.iterrows()]
    print("\n".join([header_s, div_s] + rows_s))
