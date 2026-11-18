"""Load parent datasets for MetaPilot.

Generates sklearn real datasets, synthetic classification parents,
synthetic moons/circles parents, and OpenML benchmark datasets.
Each parent is saved as a .npz file in the raw data directory with keys:
X, y, feature_names, parent_id, source.
"""

import csv
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn.datasets import make_circles, make_classification, make_moons
from tqdm import tqdm

from src.config import get_mode_from_args, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sklearn loaders keyed by config name
# ---------------------------------------------------------------------------
_SKLEARN_LOADERS = {
    "iris": sk_datasets.load_iris,
    "wine": sk_datasets.load_wine,
    "breast_cancer": sk_datasets.load_breast_cancer,
    "digits": sk_datasets.load_digits,
}


def _save_parent(raw_dir: str, parent_id: str, X: np.ndarray,
                 y: np.ndarray, feature_names: list, source: str) -> bool:
    """Save a single parent as .npz. Returns True if saved, False if skipped."""
    out_path = os.path.join(raw_dir, f"{parent_id}.npz")
    if os.path.exists(out_path):
        return False
    np.savez(
        out_path,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=str),
        parent_id=np.array(parent_id),
        source=np.array(source),
    )
    return True


# ---------------------------------------------------------------------------
# 1. Sklearn real datasets
# ---------------------------------------------------------------------------

def load_sklearn_parents(cfg: dict) -> list[str]:
    """Load sklearn parents listed in config and save as .npz.

    Returns list of parent_ids that were processed.
    """
    raw_dir = cfg["paths"]["raw"]
    names = cfg["data"]["sklearn_parents"]
    parent_ids: list[str] = []

    for name in tqdm(names, desc="sklearn parents"):
        parent_id = f"sk_{name}"
        parent_ids.append(parent_id)
        loader = _SKLEARN_LOADERS[name]
        ds = loader()
        X = ds.data.astype(np.float64)
        y = ds.target.astype(np.int64)
        feature_names = list(ds.feature_names)
        _save_parent(raw_dir, parent_id, X, y, feature_names, source="sklearn")

    return parent_ids


# ---------------------------------------------------------------------------
# 2. Diversified synthetic classification parents
# ---------------------------------------------------------------------------

def load_synthetic_classification_parents(cfg: dict) -> list[str]:
    """Generate diversified synthetic classification parents.

    To prevent RBF SVM dominance and ensure all 10 algorithm families are
    competitive, we vary archetypes across parents:
    - axis-aligned rule-based labels (suits decision_tree & random_forest)
    - XOR / checkerboard grid (suits tree ensembles & MLP)
    - conditionally independent Gaussians (Bayes-optimal for naive_bayes)
    - linear boundaries with high-dimensional noise (suits logreg & ridge)
    - Friedman additive non-linear equations (suits grad_boosting)
    - multi-cluster non-hypercube make_classification (suits knn & extra_trees)

    Returns list of parent_ids that were processed.
    """
    seed = cfg["seed"]
    raw_dir = cfg["paths"]["raw"]
    sc = cfg["data"]["synthetic_classification"]
    n_parents = sc["n_parents"]
    parent_ids: list[str] = []

    for i in tqdm(range(n_parents), desc="synthetic classification"):
        parent_id = f"synth_cls_{i:03d}"
        parent_ids.append(parent_id)

        out_path = os.path.join(raw_dir, f"{parent_id}.npz")
        if os.path.exists(out_path):
            continue

        rng = np.random.RandomState(seed + i)
        n_samples = int(rng.randint(sc["n_samples"][0], sc["n_samples"][1] + 1))
        n_features = int(rng.randint(sc["n_features"][0], sc["n_features"][1] + 1))
        info_frac = float(rng.uniform(sc["informative_frac"][0], sc["informative_frac"][1]))
        class_sep = float(rng.uniform(sc["class_sep"][0], sc["class_sep"][1]))
        flip_y = float(rng.uniform(sc["flip_y"][0], sc["flip_y"][1]))
        n_classes = int(rng.randint(sc["n_classes"][0], sc["n_classes"][1] + 1))

        n_clusters_per_class = int(rng.randint(1, 4))
        hypercube = bool(rng.choice([True, False]))

        archetype = i % 6

        if archetype == 0:
            # Axis-aligned box / rule thresholding
            X = rng.randn(n_samples, n_features)
            n_rules = min(4, n_features)
            cond = np.zeros(n_samples, dtype=int)
            for r in range(n_rules):
                thresh = float(np.median(X[:, r]))
                cond = cond * 2 + (X[:, r] > thresh).astype(int)
            y = cond % n_classes
            if flip_y > 0:
                flip_mask = rng.rand(n_samples) < flip_y
                y[flip_mask] = rng.randint(0, n_classes, size=int(flip_mask.sum()))

        elif archetype == 1:
            # 2D/3D XOR checkerboard grid + noise features
            dim = min(3, n_features)
            X_core = rng.uniform(-2, 2, size=(n_samples, dim))
            grid = np.floor(X_core) % 2
            y_core = (grid.sum(axis=1) % n_classes).astype(int)
            if n_features > dim:
                X_noise = rng.randn(n_samples, n_features - dim)
                X = np.hstack([X_core, X_noise])
            else:
                X = X_core
            y = y_core
            if flip_y > 0:
                flip_mask = rng.rand(n_samples) < flip_y
                y[flip_mask] = rng.randint(0, n_classes, size=int(flip_mask.sum()))

        elif archetype == 2:
            # Conditionally independent Gaussians (Naive Bayes optimal)
            samples_per_class = n_samples // n_classes
            X_list, y_list = [], []
            for c in range(n_classes):
                mu = rng.uniform(-1.5, 1.5, size=n_features)
                sig = rng.uniform(0.5, 2.0, size=n_features)
                Xc = rng.normal(loc=mu, scale=sig, size=(samples_per_class, n_features))
                X_list.append(Xc)
                y_list.append(np.full(samples_per_class, c, dtype=int))
            X = np.vstack(X_list)
            y = np.concatenate(y_list)

        elif archetype == 3:
            # Linear hyperplane with high-dimensional uninformative noise
            n_info = max(2, int(round(n_features * 0.15)))
            n_info = min(n_info, n_features)
            X = rng.randn(n_samples, n_features)
            weights = rng.randn(n_info)
            linear_comb = X[:, :n_info] @ weights
            quantiles = np.quantile(linear_comb, np.linspace(0, 1, n_classes + 1)[1:-1])
            y = np.digitize(linear_comb, quantiles)
            if flip_y > 0:
                flip_mask = rng.rand(n_samples) < flip_y
                y[flip_mask] = rng.randint(0, n_classes, size=int(flip_mask.sum()))

        elif archetype == 4:
            # Friedman additive non-linear boundary
            X = rng.uniform(0, 1, size=(n_samples, n_features))
            f_val = (
                10 * np.sin(np.pi * X[:, 0] * X[:, 1])
                + 20 * (X[:, 2] - 0.5) ** 2
                + 10 * X[:, 3]
                + 5 * X[:, 4]
                if n_features >= 5
                else np.sin(X[:, 0]) + (X[:, 1] ** 2)
            )
            quantiles = np.quantile(f_val, np.linspace(0, 1, n_classes + 1)[1:-1])
            y = np.digitize(f_val, quantiles)

        else:
            # Multi-cluster non-hypercube make_classification
            n_informative = max(2, int(round(n_features * info_frac)))
            n_informative = min(n_informative, n_features)
            max_info_for_clusters = n_classes * n_clusters_per_class
            if (2 ** n_informative) < max_info_for_clusters:
                n_informative = int(np.ceil(np.log2(max_info_for_clusters)))
                n_informative = min(n_informative, n_features)

            remaining = n_features - n_informative
            n_redundant = min(remaining, max(0, int(round(remaining * 0.3))))
            if (n_informative + n_redundant) > n_features:
                n_redundant = max(0, n_features - n_informative)

            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_repeated=0,
                n_classes=n_classes,
                n_clusters_per_class=n_clusters_per_class,
                hypercube=hypercube,
                class_sep=class_sep,
                flip_y=flip_y,
                random_state=seed + i,
            )

        feature_names = [f"f{j}" for j in range(X.shape[1])]
        _save_parent(
            raw_dir,
            parent_id,
            X.astype(np.float64),
            y.astype(np.int64),
            feature_names,
            source="synthetic",
        )

    return parent_ids


# ---------------------------------------------------------------------------
# 3. Synthetic shapes parents (moons & circles)
# ---------------------------------------------------------------------------

def load_synthetic_shapes_parents(cfg: dict) -> list[str]:
    """Generate moons and circles parents with source='synthetic'."""
    seed = cfg["seed"]
    raw_dir = cfg["paths"]["raw"]
    ss = cfg["data"]["synthetic_shapes"]

    n_parents = ss["n_parents"]
    kinds = ss["kinds"]
    n_per_kind = n_parents // len(kinds)

    parent_ids: list[str] = []
    global_idx = 0

    for kind in kinds:
        for k in tqdm(range(n_per_kind), desc=f"shapes ({kind})"):
            parent_id = f"{kind}_{k:03d}"
            parent_ids.append(parent_id)

            out_path = os.path.join(raw_dir, f"{parent_id}.npz")
            if os.path.exists(out_path):
                global_idx += 1
                continue

            rng = np.random.RandomState(seed + global_idx)
            n_samples = int(rng.randint(ss["n_samples"][0], ss["n_samples"][1] + 1))
            noise = float(rng.uniform(ss["noise"][0], ss["noise"][1]))

            if kind == "moons":
                X, y = make_moons(
                    n_samples=n_samples,
                    noise=noise,
                    random_state=seed + global_idx,
                )
            else:
                X, y = make_circles(
                    n_samples=n_samples,
                    noise=noise,
                    random_state=seed + global_idx,
                )

            feature_names = ["x0", "x1"]
            _save_parent(
                raw_dir,
                parent_id,
                X.astype(np.float64),
                y.astype(np.int64),
                feature_names,
                source="synthetic",
            )

            global_idx += 1

    return parent_ids


# ---------------------------------------------------------------------------
# 4. OpenML parent datasets (CC18 Suite 99 + small + high-dimensional sets)
# ---------------------------------------------------------------------------

def _log_failure(raw_dir: str, did: int, name: str, error_msg: str) -> None:
    """Log download/loading failure to data/raw/load_failures.csv."""
    failures_file = os.path.join(raw_dir, "load_failures.csv")
    file_exists = os.path.exists(failures_file)
    with open(failures_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["did", "dataset_name", "error", "timestamp"])
        # Keep name and error in separate columns; never let the error text leak into the name.
        safe_name = name if name and name != str(error_msg) else f"did_{did}"
        writer.writerow([did, safe_name, str(error_msg), datetime.utcnow().isoformat()])


def load_openml_parents(cfg: dict) -> list[str]:
    """Download and save OpenML parents (Suite 99 + named small/high-dim sets).

    Retries failed downloads up to 2 times (3 total attempts).
    Logs failures to data/raw/load_failures.csv.
    Caches downloads locally.
    """
    import openml

    raw_dir = cfg["paths"]["raw"]
    openml_cfg = cfg["data"]["openml"]
    suite_id = openml_cfg.get("suite_id", 99)
    target_downloads = openml_cfg.get("target_downloads", 165)
    target_parents = openml_cfg.get("target_parents", 150)
    cache_dir = os.path.abspath(openml_cfg.get("cache_dir", os.path.join(raw_dir, "openml_cache")))

    os.makedirs(cache_dir, exist_ok=True)
    openml.config.set_root_cache_directory(cache_dir)

    print(f"[openml] Configuring local cache at: {cache_dir}")
    print(f"[openml] Target downloads: {target_downloads}, Target successful parents: {target_parents}")

    # 1. Fetch Suite 99 IDs
    try:
        suite = openml.study.get_suite(suite_id)
        suite_dids = [int(did) for did in suite.data]
        print(f"[openml] Found {len(suite_dids)} datasets in Suite {suite_id}")
    except Exception as e:
        logger.warning("Could not fetch suite %s: %s", suite_id, e)
        suite_dids = []

    # 2. Named datasets from SPEC.md Section 6
    named_dids = [59, 40, 53, 55, 43, 1458, 1485, 41158]

    candidate_dids = []
    for did in suite_dids + named_dids:
        if did not in candidate_dids:
            candidate_dids.append(did)

    # 3. Add additional tabular classification datasets to reach target_downloads
    if len(candidate_dids) < target_downloads:
        try:
            extra_df = openml.datasets.list_datasets(
                number_classes="2..10",
                number_instances="100..20000",
                number_features="4..200",
                status="active",
                output_format="dataframe",
            )
            for did in extra_df["did"]:
                did = int(did)
                if did not in candidate_dids:
                    candidate_dids.append(did)
                if len(candidate_dids) >= target_downloads:
                    break
        except Exception as e:
            logger.warning("Could not query additional OpenML datasets: %s", e)

    candidate_dids = candidate_dids[:target_downloads]
    print(f"[openml] Total candidate datasets to attempt: {len(candidate_dids)}")

    # Size limits: skip very large image/text datasets that are unsuitable
    max_rows = openml_cfg.get("max_rows", 50000)
    max_features = openml_cfg.get("max_features", 11000)
    download_timeout = openml_cfg.get("download_timeout_sec", 180)

    successful_parent_ids: list[str] = []
    max_retries = 2

    for did in tqdm(candidate_dids, desc="openml parents"):
        parent_id = f"openml_{did}"
        out_path = os.path.join(raw_dir, f"{parent_id}.npz")

        if os.path.exists(out_path):
            successful_parent_ids.append(parent_id)
            if len(successful_parent_ids) >= target_parents:
                print(f"[openml] Target of {target_parents} parents reached.")
                break
            continue

        dataset_name = f"did_{did}"
        success = False
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Pre-check metadata size to skip image/very-large datasets
                ds = openml.datasets.get_dataset(did, download_data=False)
                dataset_name = ds.name or dataset_name
                nrow = ds.qualities.get("NumberOfInstances", 0) if ds.qualities else 0
                nfeat = ds.qualities.get("NumberOfFeatures", 0) if ds.qualities else 0
                try:
                    nrow = int(float(nrow))
                    nfeat = int(float(nfeat))
                except (ValueError, TypeError):
                    nrow, nfeat = 0, 0

                if nrow > max_rows:
                    raise ValueError(f"Too many rows: {nrow} > {max_rows}")
                if nfeat > max_features:
                    raise ValueError(f"Too many features: {nfeat} > {max_features}")

                # Download with alarm-based timeout (Unix only)
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Download timed out after {download_timeout}s")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(download_timeout)
                try:
                    ds = openml.datasets.get_dataset(did, download_data=True)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                target_col = ds.default_target_attribute
                if not target_col:
                    raise ValueError("No default target attribute specified")

                X, y, _, feat_names = ds.get_data(target=target_col)

                if y is None or len(y) == 0:
                    raise ValueError("Target array is empty")

                # Filter valid target rows
                valid_mask = pd.notna(y)
                if valid_mask.sum() < 30:
                    raise ValueError(f"Too few valid samples: {valid_mask.sum()} < 30")

                y_valid = y[valid_mask]
                unique_classes = np.unique(y_valid)
                if len(unique_classes) < 2:
                    raise ValueError(f"Too few classes: {len(unique_classes)} < 2")

                # Factorize target 0..K-1
                # np.asarray: some OpenML targets come back as a list, which
                # pd.factorize rejects (the old failure for dids 386/392/394/401).
                y_encoded, _ = pd.factorize(np.asarray(y_valid))
                y_arr = y_encoded.astype(np.int64)

                X_valid = X.iloc[valid_mask] if hasattr(X, "iloc") else X[valid_mask]

                # Convert feature names
                if feat_names is None or len(feat_names) != X_valid.shape[1]:
                    if hasattr(X_valid, "columns"):
                        feat_names = list(X_valid.columns)
                    else:
                        feat_names = [f"f{j}" for j in range(X_valid.shape[1])]

                # Convert X array: numeric if possible, object otherwise
                if hasattr(X_valid, "select_dtypes") and len(X_valid.select_dtypes(exclude=["number"]).columns) == 0:
                    X_arr = X_valid.to_numpy(dtype=np.float64)
                elif hasattr(X_valid, "to_numpy"):
                    X_arr = X_valid.to_numpy(dtype=object)
                else:
                    X_arr = np.asarray(X_valid)

                _save_parent(
                    raw_dir,
                    parent_id,
                    X_arr,
                    y_arr,
                    feat_names,
                    source="openml",
                )
                success = True
                successful_parent_ids.append(parent_id)
                break

            except Exception as e:
                last_error = e
                if isinstance(e, (ValueError, TimeoutError)):
                    break  # Don't retry size/timeout failures
                time.sleep(0.5)

        if not success:
            logger.warning("Failed to load OpenML dataset did=%d (%s): %s", did, dataset_name, last_error)
            _log_failure(raw_dir, did, dataset_name, str(last_error))

        if len(successful_parent_ids) >= target_parents:
            print(f"[openml] Target of {target_parents} parents reached.")
            break

    return successful_parent_ids


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------

def load_all_parents(cfg: dict) -> list[str]:
    """Generate / load all parent datasets. Returns list of all parent_ids."""
    sk_ids = load_sklearn_parents(cfg)
    synth_ids = load_synthetic_classification_parents(cfg)
    shape_ids = load_synthetic_shapes_parents(cfg)
    openml_ids = []
    if cfg.get("data", {}).get("openml", {}).get("enabled", False):
        openml_ids = load_openml_parents(cfg)
    return sk_ids + synth_ids + shape_ids + openml_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = get_mode_from_args()
    cfg = load_config(mode)

    print(f"[data_loading] mode={mode} seed={cfg['seed']} raw_dir={cfg['paths']['raw']}")
    all_ids = load_all_parents(cfg)

    print(f"\n{'='*50}")
    print(f"Total parents generated/loaded: {len(all_ids)}")
    print(f"  sklearn:       {sum(1 for p in all_ids if p.startswith('sk_'))}")
    print(f"  synth_cls:     {sum(1 for p in all_ids if p.startswith('synth_cls_'))}")
    print(f"  moons:         {sum(1 for p in all_ids if p.startswith('moons_'))}")
    print(f"  circles:       {sum(1 for p in all_ids if p.startswith('circles_'))}")
    print(f"  openml:        {sum(1 for p in all_ids if p.startswith('openml_'))}")
    print(f"{'='*50}")
