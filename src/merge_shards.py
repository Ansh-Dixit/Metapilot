"""Merge shard outputs and cached results into canonical meta-datasets.

Usage::

    python -m src.merge_shards --mode full
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_mode_from_args, load_config
from src.portfolio import get_all_pipeline_ids
from src.preprocessing import encode_labels
from src.run_portfolio import task_skip_reason


def merge_shards(cfg: dict) -> None:
    """Read all cached (task, pipeline) JSONs and build canonical parquets."""
    meta_dir = Path(cfg["paths"]["meta"])
    cache_dir = meta_dir / "cache"
    tasks_dir = Path(cfg["paths"]["tasks"])

    if not cache_dir.exists():
        print(f"[merge_shards] Cache directory does not exist: {cache_dir}")
        return

    cache_files = sorted(cache_dir.glob("*__*.json"))
    print(f"[merge_shards] Found {len(cache_files)} cached fit files in {cache_dir}")

    fit_rows = []
    task_ids_seen = set()

    for cf in cache_files:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
            tid = data["task_id"]
            pid = data["pipeline_id"]
            status = data.get("status", "ok")
            fit_sec = data.get("fit_seconds", 0.0)
            task_ids_seen.add(tid)

            for fold_i, acc in enumerate(data.get("fold_accuracies", [])):
                fit_rows.append({
                    "task_id": tid,
                    "pipeline_id": pid,
                    "fold": fold_i,
                    "accuracy": acc,
                    "fit_seconds": fit_sec,
                    "status": status,
                })
        except Exception as e:
            print(f"[merge_shards] Error reading {cf}: {e}")

    fit_log_df = pd.DataFrame(fit_rows)

    # ---- Exclusions: skip rule (same function as run_portfolio) + incomplete tasks ----
    pipeline_ids = get_all_pipeline_ids(cfg.get("mode"))
    results_dir = Path(cfg["paths"].get("results", "results"))
    row_cap = cfg["preprocessing"]["row_cap"]
    n_folds = cfg["cv"]["n_folds"]
    skipped_rows = []
    for tf in sorted(tasks_dir.glob("*.npz")):
        with np.load(tf, allow_pickle=True) as d:
            y_enc, _ = encode_labels(d["y"])
            reason = task_skip_reason(y_enc, len(y_enc), row_cap, cfg["seed"], n_folds)
            if reason is not None:
                skipped_rows.append({"task_id": tf.stem, "parent_id": str(d["parent_id"]), "reason": reason})
    skipped_ids = {r["task_id"] for r in skipped_rows}
    pd.DataFrame(skipped_rows, columns=["task_id", "parent_id", "reason"]).to_csv(
        results_dir / "skipped_tasks.csv", index=False)
    print(f"[merge_shards] Skipped tasks (excluded): {len(skipped_ids)}")

    if not fit_log_df.empty:
        fit_log_df = fit_log_df[~fit_log_df["task_id"].isin(skipped_ids)]
        coverage = fit_log_df.groupby("task_id")["pipeline_id"].nunique()
        incomplete = coverage[coverage < len(pipeline_ids)]
        if len(incomplete):
            pd.DataFrame({"task_id": incomplete.index, "n_pipelines_cached": incomplete.values}).to_csv(
                results_dir / "incomplete_tasks.csv", index=False)
            print(f"[merge_shards] WARNING: {len(incomplete)} tasks lack some pipelines; excluded "
                  f"(see incomplete_tasks.csv)")
            fit_log_df = fit_log_df[~fit_log_df["task_id"].isin(set(incomplete.index))]
        task_ids_seen = set(fit_log_df["task_id"])

    fit_log_path = meta_dir / "fit_log.parquet"
    fit_log_df.to_parquet(fit_log_path, index=False)
    print(f"[merge_shards] Saved fit_log.parquet ({len(fit_log_df)} rows)")

    # Build performance matrix
    if not fit_log_df.empty:
        perf_matrix = (
            fit_log_df
            .groupby(["task_id", "pipeline_id"])["accuracy"]
            .mean()
            .unstack(fill_value=np.nan)
        )

        # NaN replacement: per task, replace NaN with that task's lowest observed score
        for task_id in perf_matrix.index:
            row = perf_matrix.loc[task_id]
            min_score = row.min()
            if pd.isna(min_score):
                min_score = 0.0
            perf_matrix.loc[task_id] = row.fillna(min_score)

        present_cols = [p for p in pipeline_ids if p in perf_matrix.columns]
        perf_matrix = perf_matrix.reindex(columns=present_cols)
        perf_matrix_path = meta_dir / "performance_matrix.parquet"
        perf_matrix.to_parquet(perf_matrix_path)
        print(f"[merge_shards] Saved performance_matrix.parquet {perf_matrix.shape}")
    else:
        perf_matrix = pd.DataFrame()

    # Build task_info.parquet
    task_info_rows = []
    task_files = sorted(tasks_dir.glob("*.npz"))
    for tf in task_files:
        tid = tf.stem
        if tid in task_ids_seen:  # task_info only for tasks in the performance matrix
            data = np.load(tf, allow_pickle=True)
            X = data["X"]
            y = data["y"]
            parent_id = str(data["parent_id"])
            source = str(data["source"]) if "source" in data else parent_id
            row_frac = float(data["row_frac"]) if "row_frac" in data else 1.0
            col_frac = float(data["col_frac"]) if "col_frac" in data else 1.0
            y_enc, _ = encode_labels(y)

            task_info_rows.append({
                "task_id": tid,
                "parent_id": parent_id,
                "source": source,
                "n_rows": int(X.shape[0]),
                "n_cols": int(X.shape[1]),
                "n_classes": int(len(np.unique(y_enc))),
                "row_frac": row_frac,
                "col_frac": col_frac,
            })

    if task_info_rows:
        task_info_df = pd.DataFrame(task_info_rows)
        task_info_path = meta_dir / "task_info.parquet"
        task_info_df.to_parquet(task_info_path, index=False)
        print(f"[merge_shards] Saved task_info.parquet ({len(task_info_df)} rows)")

    print("[merge_shards] Merge complete!")


if __name__ == "__main__":
    mode = get_mode_from_args()
    cfg = load_config(mode)
    merge_shards(cfg)

