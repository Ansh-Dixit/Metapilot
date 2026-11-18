"""Run every portfolio pipeline (18 pilot / 30 full) on every task with 3-fold stratified CV.

For each (task, pipeline) pair the module:
1. Checks a JSON cache and skips if already evaluated (resumable).
2. Applies row-cap downsampling when the task exceeds ``row_cap`` rows.
3. Builds the full sklearn Pipeline (preprocessor + estimator) via
   ``portfolio.get_pipeline()``, so imputation / scaling are fitted
   per CV fold only — no leakage.
4. Runs stratified k-fold CV, recording per-fold accuracy and fit time.
5. Handles errors / timeouts gracefully.
6. Writes three summary parquet files to ``data/meta/``:
   - fit_log.parquet
   - performance_matrix.parquet
   - task_info.parquet

Usage::

    python -m src.run_portfolio --mode pilot
"""

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import get_mode_from_args, load_config
from src.portfolio import get_all_pipeline_ids, get_pipeline
from src.preprocessing import apply_row_cap, encode_labels


# ------------------------------------------------------------------
# Timeout helper
# ------------------------------------------------------------------

class _FitTimeout(Exception):
    """Raised when a single fit exceeds the time budget."""


def _timeout_handler(signum, frame):
    raise _FitTimeout("Fit timed out")


_HAS_ALARM = hasattr(signal, "SIGALRM")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _write_aggregate_progress(results_dir: Path, cache_dir: Path, total_fits: int, n_folds: int) -> None:
    """Combine per-shard progress files + cache count into results/run_progress.json.

    completed_fits counts every cached (task, pipeline) pair across all shards,
    so concurrently running shards all report the same global number.
    """
    shard_dir = results_dir / "progress"
    shards = []
    for sf in sorted(shard_dir.glob("shard_*.json")):
        try:
            with open(sf) as f:
                shards.append(json.load(f))
        except Exception:
            pass
    n_cached = sum(1 for _ in os.scandir(cache_dir)) if cache_dir.exists() else 0
    completed = n_cached * n_folds
    elapsed = max([sh.get("elapsed_sec", 0.0) for sh in shards] or [0.0])
    new_fits = sum(sh.get("new_fits", 0) for sh in shards)
    rate = new_fits / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_fits - completed)
    eta = remaining / rate if rate > 0 else 0.0
    _atomic_json(results_dir / "run_progress.json", {
        "completed_fits": completed,
        "total_fits": total_fits,
        "pct_complete": round(100.0 * completed / total_fits, 2) if total_fits else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "fits_per_sec": round(rate, 2),
        "eta_sec": round(eta, 1),
        "eta_min": round(eta / 60.0, 1),
        "failures": sum(sh.get("failures", 0) for sh in shards),
        "timeouts": sum(sh.get("timeouts", 0) for sh in shards),
        "shards_reporting": len(shards),
        "shards_done": sum(1 for sh in shards if sh.get("done")),
        "timestamp": datetime.utcnow().isoformat(),
    })


def _write_progress(
    progress_file: Path,
    completed_fits: int,
    total_fits: int,
    elapsed_sec: float,
    failures: int,
    timeouts: int,
) -> None:
    """Atomically write run progress JSON every 100 completed fits."""
    pct = (completed_fits / total_fits * 100.0) if total_fits > 0 else 0.0
    fits_per_sec = completed_fits / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining_fits = max(0, total_fits - completed_fits)
    eta_sec = remaining_fits / fits_per_sec if fits_per_sec > 0 else 0.0
    payload = {
        "completed_fits": completed_fits,
        "total_fits": total_fits,
        "pct_complete": round(pct, 2),
        "elapsed_sec": round(elapsed_sec, 2),
        "fits_per_sec": round(fits_per_sec, 2),
        "eta_sec": round(eta_sec, 2),
        "eta_min": round(eta_sec / 60.0, 2),
        "failures": failures,
        "timeouts": timeouts,
        "timestamp": datetime.utcnow().isoformat(),
    }
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(progress_file, payload)


def _patch_progress(progress_file: Path, **fields) -> None:
    with open(progress_file) as f:
        payload = json.load(f)
    payload.update(fields)
    _atomic_json(progress_file, payload)


# ------------------------------------------------------------------
# Evaluation of a single (task, pipeline) pair
# ------------------------------------------------------------------

def _evaluate_pair(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_id: str,
    cfg: dict,
    seed: int,
) -> dict:
    """Run stratified CV for one pipeline on one (possibly row-capped) task.

    Returns
    -------
    dict with keys: fold_accuracies, mean_accuracy, fit_seconds, status
    """
    n_folds: int = cfg["cv"]["n_folds"]
    timeout: int = cfg["cv"]["fit_timeout_sec"]
    # Allow longer fit timeout for high-dimensional datasets (> 1000 features)
    if X.shape[1] > 1000:
        timeout = max(timeout, cfg["cv"].get("fit_timeout_wide_sec", 300))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accuracies: list[float] = []
    fold_times: list[float] = []
    status = "ok"

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build a fresh pipeline each fold (unfitted clone)
        pipe = get_pipeline(pipeline_id, seed, X_train, cfg)

        # Special handling for knn_15: cap n_neighbors
        if pipeline_id == "knn_15":
            estimator_step = pipe.named_steps.get("model") or pipe.named_steps.get("clf") or pipe.steps[-1][1]
            max_k = X_train.shape[0] - 1
            if hasattr(estimator_step, "n_neighbors") and estimator_step.n_neighbors > max_k:
                estimator_step.n_neighbors = max(1, max_k)

        can_alarm = _HAS_ALARM and threading.current_thread() is threading.main_thread()

        try:
            if can_alarm:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)

            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            if can_alarm:
                signal.alarm(0)

            acc = float(pipe.score(X_test, y_test))
            fold_accuracies.append(acc)
            fold_times.append(elapsed)

        except _FitTimeout:
            if can_alarm:
                signal.alarm(0)
            return {
                "fold_accuracies": [float("nan")] * n_folds,
                "mean_accuracy": float("nan"),
                "fit_seconds": float("nan"),
                "status": "timeout",
            }
        except Exception as exc:  # noqa: BLE001
            if can_alarm:
                signal.alarm(0)
            return {
                "fold_accuracies": [float("nan")] * n_folds,
                "mean_accuracy": float("nan"),
                "fit_seconds": float("nan"),
                "status": f"failed: {type(exc).__name__}: {exc}",
            }

    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "fit_seconds": float(np.sum(fold_times)),
        "status": status,
    }


# ------------------------------------------------------------------
# Skip rule + per-shard CSV logs
# ------------------------------------------------------------------

def task_skip_reason(y_encoded: np.ndarray, n_rows: int, row_cap: int, seed: int, n_folds: int):
    """Return a skip reason if the task cannot be evaluated with n_folds-fold CV, else None.

    Shared by run_portfolio and merge_shards so the rule is identical for every
    task regardless of which code version produced its cache. It does NOT change
    how tasks are capped: it replays the same apply_row_cap call.
    """
    try:
        _, y_cap = apply_row_cap(np.zeros((n_rows, 1)), y_encoded, row_cap, seed)
    except ValueError as exc:
        return f"stratified row cap impossible: {exc}"
    _, counts = np.unique(y_cap, return_counts=True)
    if counts.min() < n_folds:
        return f"after row cap a class has {int(counts.min())} member(s) < {n_folds} CV folds"
    return None


def _append_csv(path: Path, header: list[str], row: list) -> None:
    """Append one row (single write, O_APPEND) so concurrent shards don't interleave."""
    import csv
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    if not path.exists():
        w.writerow(header)
    w.writerow(row)
    with open(path, "a", newline="") as f:
        f.write(buf.getvalue())


# ------------------------------------------------------------------
# Per-task processing worker (resumable per task+pipeline)
# ------------------------------------------------------------------

def _process_task(
    task_path: Path,
    pipeline_ids: list[str],
    cache_dir: Path,
    row_cap: int,
    seed: int,
    cfg: dict,
) -> tuple[dict, list[dict], int]:
    """Process all pipelines for a single task."""
    data = np.load(task_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    task_id = str(data["task_id"])
    parent_id = str(data["parent_id"])
    source = str(data["source"]) if "source" in data else ""
    row_frac = float(data["row_frac"]) if "row_frac" in data else 1.0
    col_frac = float(data["col_frac"]) if "col_frac" in data else 1.0

    y, _ = encode_labels(y)
    n_rows_orig, n_cols = X.shape
    n_classes = int(len(np.unique(y)))

    task_info = {
        "task_id": task_id,
        "parent_id": parent_id,
        "source": source,
        "n_rows": n_rows_orig,
        "n_cols": n_cols,
        "n_classes": n_classes,
        "row_frac": row_frac,
        "col_frac": col_frac,
    }

    skip = task_skip_reason(y, n_rows_orig, row_cap, seed, cfg["cv"]["n_folds"])
    if skip is not None:
        task_info["_skipped"] = skip
        task_info["_new_pairs"] = 0
        return task_info, [], 0

    X_eval, y_eval = apply_row_cap(X, y, row_cap, seed)

    fit_rows: list[dict] = []
    failures = 0
    new_pairs = 0

    for pid in pipeline_ids:
        cache_file = cache_dir / f"{task_id}__{pid}.json"
        cached = None
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
            except Exception:
                cached = None

        if cached is not None:
            for fold_i, acc in enumerate(cached["fold_accuracies"]):
                fit_rows.append({
                    "task_id": task_id,
                    "pipeline_id": pid,
                    "fold": fold_i,
                    "accuracy": acc,
                    "fit_seconds": cached["fit_seconds"],
                    "status": cached["status"],
                })
            if cached["status"] != "ok":
                failures += 1
            continue

        result = _evaluate_pair(X_eval, y_eval, pid, cfg, seed)
        new_pairs += 1
        cache_payload = {
            "task_id": task_id,
            "pipeline_id": pid,
            **result,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_payload, f, indent=2)

        for fold_i, acc in enumerate(result["fold_accuracies"]):
            fit_rows.append({
                "task_id": task_id,
                "pipeline_id": pid,
                "fold": fold_i,
                "accuracy": acc,
                "fit_seconds": result["fit_seconds"],
                "status": result["status"],
            })
        if result["status"] != "ok":
            failures += 1

    task_info["_new_pairs"] = new_pairs
    return task_info, fit_rows, failures


# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------

def run_portfolio(cfg: dict, shard: str | None = None) -> None:
    """Evaluate every (task, pipeline) pair and write summary artefacts."""
    seed: int = cfg["seed"]
    tasks_dir = Path(cfg["paths"]["tasks"])
    meta_dir = Path(cfg["paths"]["meta"])
    results_dir = Path(cfg.get("paths", {}).get("results", "results"))
    cache_dir = meta_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    row_cap: int = cfg["preprocessing"]["row_cap"]
    n_jobs: int = cfg.get("n_jobs", 1)

    task_files = sorted(tasks_dir.glob("*.npz"))
    if not task_files:
        print("[run_portfolio] No task files found in", tasks_dir)
        return

    pipeline_ids = get_all_pipeline_ids(cfg.get("mode"))
    n_folds = cfg["cv"]["n_folds"]
    global_total_fits = len(task_files) * len(pipeline_ids) * n_folds

    shard_idx = None
    if shard is not None:
        # 0-based: "--shard i/N" runs tasks i, i+N, i+2N, ... for 0 <= i < N
        parts = shard.strip().split("/")
        if len(parts) != 2:
            raise ValueError(f"--shard must be i/N, got {shard!r}")
        shard_idx, s_total = int(parts[0]), int(parts[1])
        if not (0 <= shard_idx < s_total):
            raise ValueError(f"--shard index must satisfy 0 <= i < N, got {shard!r}")
        task_files = task_files[shard_idx::s_total]
        print(f"[run_portfolio] Shard {shard} selected: {len(task_files)} tasks")

    n_tasks = len(task_files)
    n_pipelines = len(pipeline_ids)
    total_pairs = n_tasks * n_pipelines
    total_fits = total_pairs * n_folds

    print(f"[run_portfolio] {n_tasks} tasks × {n_pipelines} pipelines = {total_pairs} pairs ({total_fits} fits, n_jobs={n_jobs})")

    # Progress tracking
    shard_name = f"shard_{shard_idx}" if shard_idx is not None else "shard_all"
    progress_file = results_dir / "progress" / f"{shard_name}.json"
    completed_fits = 0
    new_fits = 0
    timeout_log: list[dict] = []

    wall_start = time.perf_counter()

    # Sequential with progress tracking
    task_info_rows = []
    fit_log_rows = []
    total_failures = 0

    skipped_csv = results_dir / "skipped_tasks.csv"
    errors_csv = results_dir / "task_errors.csv"
    n_skipped = n_errors = 0

    for task_path in tqdm(task_files, desc="portfolio eval", unit="task"):
        try:
            t_info, f_rows, f_fails = _process_task(
                task_path, pipeline_ids, cache_dir, row_cap, seed, cfg,
            )
        except Exception as exc:  # noqa: BLE001 — a task must never kill a shard
            n_errors += 1
            _append_csv(errors_csv, ["task_id", "error"],
                        [task_path.stem, f"{type(exc).__name__}: {exc}"])
            continue
        if "_skipped" in t_info:
            n_skipped += 1
            _append_csv(skipped_csv, ["task_id", "parent_id", "reason"],
                        [t_info["task_id"], t_info["parent_id"], t_info["_skipped"]])
            continue
        new_fits += t_info.pop("_new_pairs", 0) * n_folds
        task_info_rows.append(t_info)
        fit_log_rows.extend(f_rows)
        total_failures += f_fails

        # Track timeouts
        for row in f_rows:
            if row.get("status", "") == "timeout":
                timeout_log.append({
                    "task_id": row["task_id"],
                    "pipeline_id": row["pipeline_id"],
                    "n_cols": t_info.get("n_cols", 0),
                })

        # Update progress every 100 completed fits
        prev_bucket = completed_fits // 100
        completed_fits += len(f_rows)
        curr_bucket = completed_fits // 100
        if curr_bucket > prev_bucket:
            elapsed = time.perf_counter() - wall_start
            _write_progress(progress_file, completed_fits, total_fits,
                            elapsed, total_failures, len(timeout_log))
            _patch_progress(progress_file, new_fits=new_fits, done=False)
            _write_aggregate_progress(results_dir, cache_dir, global_total_fits, n_folds)

    wall_elapsed = time.perf_counter() - wall_start

    # Final progress update
    _write_progress(progress_file, completed_fits, total_fits,
                    wall_elapsed, total_failures, len(timeout_log))
    _patch_progress(progress_file, new_fits=new_fits, done=True)
    _write_aggregate_progress(results_dir, cache_dir, global_total_fits, n_folds)

    # Log timeouts
    if timeout_log:
        timeout_csv = results_dir / f"timeout_log_{shard_name}.csv"
        pd.DataFrame(timeout_log).to_csv(timeout_csv, index=False)
        print(f"[run_portfolio] {len(timeout_log)} timeouts logged to {timeout_csv}")

    if shard_idx is not None:
        # Shards never write partial canonical parquets; run merge_shards afterwards.
        print(f"[run_portfolio] Shard {shard} done in {wall_elapsed:.1f}s "
              f"({n_tasks} tasks, {n_skipped} skipped, {n_errors} errored, "
              f"{total_failures} failed pairs). Run src.merge_shards next.")
        return

    # 1. fit_log.parquet
    fit_log_df = pd.DataFrame(fit_log_rows)
    fit_log_df.to_parquet(meta_dir / "fit_log.parquet", index=False)

    # 2. performance_matrix.parquet
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

        # Reindex columns to canonical pipeline_id order
        perf_matrix = perf_matrix.reindex(columns=pipeline_ids)
        perf_matrix.to_parquet(meta_dir / "performance_matrix.parquet")
    else:
        perf_matrix = pd.DataFrame()

    # 3. task_info.parquet
    task_info_df = pd.DataFrame(task_info_rows)
    task_info_df.to_parquet(meta_dir / "task_info.parquet", index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Portfolio evaluation complete ({wall_elapsed:.1f}s)")
    print(f"  Tasks:      {n_tasks}")
    print(f"  Pipelines:  {n_pipelines}")
    print(f"  Total fits: {total_fits}")
    print(f"  Failures:   {total_failures}")
    print(f"  Artefacts saved to: {meta_dir}")
    print(f"    - fit_log.parquet            ({len(fit_log_df)} rows)")
    print(f"    - performance_matrix.parquet ({perf_matrix.shape})")
    print(f"    - task_info.parquet          ({len(task_info_df)} rows)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, choices=["pilot", "full"])
    parser.add_argument("--shard", default=None, help="Shard in i/N format (e.g. 1/4 or 0/3)")
    args, _ = parser.parse_known_args()

    mode = args.mode or get_mode_from_args()
    cfg = load_config(mode)
    print(f"[run_portfolio] mode={mode} seed={cfg['seed']} shard={args.shard}")
    run_portfolio(cfg, shard=args.shard)
