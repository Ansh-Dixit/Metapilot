"""Extract meta-features and text metadata for every task.

Computes 14 numeric meta-features (simple, statistical, info-theoretic,
and landmarker) plus a text-metadata table for downstream metamodels.

Outputs
-------
- data/meta/meta_features.parquet   – 14 numeric columns, task_id as index
- data/meta/text_metadata.parquet   – text/structural metadata per task
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy, kurtosis as sp_kurtosis, skew as sp_skew
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.config import get_mode_from_args, load_config
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Column names for the 14 meta-features
# ------------------------------------------------------------------
META_FEATURE_COLS = [
    # Simple (4)
    "log_n_rows",
    "log_n_cols",
    "cols_to_rows",
    "n_classes",
    # Statistical (4)
    "mean_abs_corr",
    "mean_abs_skew",
    "mean_kurtosis",
    "pca95_frac",
    # Info-theoretic (2)
    "norm_class_entropy",
    "minority_majority_ratio",
    # Landmarkers (4)
    "lm_stump",
    "lm_1nn",
    "lm_gnb",
    "lm_lda",
]


# ------------------------------------------------------------------
# Helpers — subsample for landmarkers
# ------------------------------------------------------------------

def _stratified_subsample(X, y, max_rows, rng):
    """Stratified subsample to at most *max_rows* rows."""
    if X.shape[0] <= max_rows:
        return X, y
    # Stratified selection via index manipulation
    from sklearn.model_selection import train_test_split

    # Stratify only when every class has >= 2 members (OpenML s0 tasks can keep
    # a singleton class; train_test_split refuses to stratify those).
    _, counts = np.unique(y, return_counts=True)
    strat = y if counts.min() >= 2 else None
    try:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=max_rows, random_state=rng, stratify=strat,
        )
    except ValueError:
        # e.g. held-out part smaller than n_classes: fall back to a plain random draw
        idx = np.random.RandomState(rng).choice(X.shape[0], size=max_rows, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    return X_sub, y_sub


# ------------------------------------------------------------------
# Simple meta-features
# ------------------------------------------------------------------

def _simple_features(X, y):
    """Return dict of 4 simple meta-features."""
    n_rows, n_cols = X.shape
    n_classes = len(np.unique(y))
    return {
        "log_n_rows": np.log(n_rows),
        "log_n_cols": np.log(n_cols),
        "cols_to_rows": n_cols / n_rows,
        "n_classes": n_classes,
    }


# ------------------------------------------------------------------
# Statistical meta-features
# ------------------------------------------------------------------

def _statistical_features(X):
    """Return dict of 4 statistical meta-features."""
    n_rows, n_cols = X.shape

    # Identify non-constant columns
    col_stds = np.nanstd(X, axis=0)
    non_const_mask = col_stds > 0
    n_non_const = int(non_const_mask.sum())

    # --- mean_abs_corr ---
    if n_non_const <= 1:
        mean_abs_corr = 0.0
    else:
        X_nc = X[:, non_const_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(X_nc, rowvar=False)
        # Replace NaN with 0 (can happen if std is near-zero)
        corr = np.nan_to_num(corr, nan=0.0)
        # Mean of absolute off-diagonal
        mask_offdiag = ~np.eye(corr.shape[0], dtype=bool)
        mean_abs_corr = float(np.mean(np.abs(corr[mask_offdiag])))

    # --- mean_abs_skew ---
    if n_non_const == 0:
        mean_abs_skew = 0.0
    else:
        X_nc = X[:, non_const_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skews = sp_skew(X_nc, axis=0, nan_policy="omit")
        skews = np.nan_to_num(skews, nan=0.0)
        mean_abs_skew = float(np.mean(np.abs(skews)))

    # --- mean_kurtosis ---
    if n_non_const == 0:
        mean_kurtosis = 0.0
    else:
        X_nc = X[:, non_const_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kurts = sp_kurtosis(X_nc, axis=0, nan_policy="omit")
        kurts = np.nan_to_num(kurts, nan=0.0)
        mean_kurtosis = float(np.mean(kurts))

    # --- pca95_frac ---
    if n_cols <= 1:
        pca95_frac = 1.0
    elif n_non_const == 0:
        pca95_frac = 1.0
    else:
        X_nc = X[:, non_const_mask]
        # Standardise before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_nc)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        n_components = min(X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95 = int(np.searchsorted(cumvar, 0.95) + 1)
        n95 = min(n95, n_components)
        pca95_frac = n95 / n_cols  # normalise by total cols, not just non-const

    return {
        "mean_abs_corr": mean_abs_corr,
        "mean_abs_skew": mean_abs_skew,
        "mean_kurtosis": mean_kurtosis,
        "pca95_frac": pca95_frac,
    }


# ------------------------------------------------------------------
# Info-theoretic meta-features
# ------------------------------------------------------------------

def _info_theoretic_features(y):
    """Return dict of 2 info-theoretic meta-features."""
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # normalised class entropy
    if n_classes <= 1:
        norm_class_entropy = 0.0
    else:
        probs = counts / counts.sum()
        raw_entropy = float(sp_entropy(probs, base=np.e))
        norm_class_entropy = raw_entropy / np.log(n_classes)

    # minority / majority ratio
    minority_majority_ratio = float(counts.min()) / float(counts.max())

    return {
        "norm_class_entropy": norm_class_entropy,
        "minority_majority_ratio": minority_majority_ratio,
    }


# ------------------------------------------------------------------
# Landmarker meta-features
# ------------------------------------------------------------------

def _landmarker_features(X, y, seed, landmarker_rows, landmarker_folds):
    """Return dict of 4 landmarker meta-features."""
    # Stratified subsample
    X_lm, y_lm = _stratified_subsample(X, y, landmarker_rows, rng=seed)

    cv = StratifiedKFold(
        n_splits=landmarker_folds, shuffle=True, random_state=seed,
    )

    results = {}

    # --- Decision stump (no scaling needed) ---
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    scores = cross_val_score(stump, X_lm, y_lm, cv=cv, scoring="accuracy")
    results["lm_stump"] = float(np.mean(scores))

    # --- 1-NN (needs imputer + scaler) ---
    pipe_1nn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=1)),
    ])
    scores = cross_val_score(pipe_1nn, X_lm, y_lm, cv=cv, scoring="accuracy")
    results["lm_1nn"] = float(np.mean(scores))

    # --- Gaussian NB ---
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X_lm, y_lm, cv=cv, scoring="accuracy")
    results["lm_gnb"] = float(np.mean(scores))

    # --- LDA (can fail on singular matrices) ---
    pipe_lda = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])
    try:
        scores = cross_val_score(pipe_lda, X_lm, y_lm, cv=cv, scoring="accuracy")
        results["lm_lda"] = float(np.mean(scores))
    except Exception:
        results["lm_lda"] = 0.5

    return results


# ------------------------------------------------------------------
# Main extraction for a single task
# ------------------------------------------------------------------

def extract_meta_features(X, y, seed, landmarker_rows, landmarker_folds):
    """Extract all 14 meta-features for one task.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already imputed with median).
    y : np.ndarray
        Integer target labels.
    seed : int
        Random seed.
    landmarker_rows : int
        Max rows for landmarker evaluation.
    landmarker_folds : int
        Number of CV folds for landmarkers.

    Returns
    -------
    dict
        Keys matching ``META_FEATURE_COLS``.
    """
    feats = {}
    feats.update(_simple_features(X, y))
    feats.update(_statistical_features(X))
    feats.update(_info_theoretic_features(y))
    feats.update(_landmarker_features(X, y, seed, landmarker_rows, landmarker_folds))
    return feats


def compute_task_meta_features(X_raw, y, seed, landmarker_rows, landmarker_folds):
    """The single code path from a raw task array to the 14 meta-features.

    Used by `extract_all` (training) and by `src.recommend` (demo), so the demo
    computes meta-features exactly as in training: impute + encode with
    `build_preprocessor` (no scaling) on the UNCAPPED task, then
    `extract_meta_features` (landmarkers use their own <= landmarker_rows subsample).
    """
    prep = build_preprocessor(X_raw, needs_scaling=False)
    X = prep.fit_transform(X_raw)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    return extract_meta_features(X, np.asarray(y).astype(np.int64), seed, landmarker_rows, landmarker_folds)


# ------------------------------------------------------------------
# Text metadata for a single task
# ------------------------------------------------------------------

def _build_text_metadata(task_id, X, y, feature_names, parent_id, source):
    """Build a single-row dict for text_metadata."""
    n_rows, n_cols = X.shape
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    minority_pct = float(counts.min()) / float(counts.sum()) * 100.0

    # Compute missing percentage from original (pre-imputation) data
    total_cells = n_rows * n_cols
    if total_cells == 0:
        missing_pct = 0.0
    else:
        try:
            missing_pct = float(pd.isna(X).sum()) / total_cells * 100.0
        except Exception:
            missing_pct = 0.0

    if isinstance(X, np.ndarray) and X.dtype != object:
        dtypes = [str(X.dtype)] * n_cols
    else:
        dtypes = [type(X[0, j]).__name__ if n_rows > 0 else "float64" for j in range(n_cols)]

    return {
        "task_id": task_id,
        "column_names": repr(list(feature_names)),
        "dtypes": repr(dtypes),
        "missing_pct": missing_pct,
        "description": "",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_classes": n_classes,
        "minority_pct": minority_pct,
        "parent_id": str(parent_id),
        "source": str(source),
    }


# ------------------------------------------------------------------
# Batch extraction
# ------------------------------------------------------------------

def extract_all(cfg):
    """Extract meta-features and text metadata for all tasks.

    Parameters
    ----------
    cfg : dict
        Full config dict.

    Returns
    -------
    meta_features_df : pd.DataFrame
        14 numeric meta-features, task_id as index.
    text_metadata_df : pd.DataFrame
        Text/structural metadata per task.
    """
    seed = cfg["seed"]
    tasks_dir = Path(cfg["paths"]["tasks"])
    meta_dir = Path(cfg["paths"]["meta"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    landmarker_rows = cfg["meta"]["landmarker_rows"]
    landmarker_folds = cfg["meta"]["landmarker_folds"]

    task_files = sorted(tasks_dir.glob("*.npz"))
    if not task_files:
        logger.warning("No task files found in %s", tasks_dir)
        return pd.DataFrame(columns=META_FEATURE_COLS), pd.DataFrame()

    meta_rows = []
    text_rows = []

    for task_path in tqdm(task_files, desc="meta-features", unit="task"):
        task_data = np.load(task_path, allow_pickle=True)
        task_id = str(task_data["task_id"])
        X_raw = task_data["X"]
        y = task_data["y"].astype(np.int64)
        feature_names = list(task_data["feature_names"])
        parent_id = str(task_data["parent_id"])
        source = str(task_data["source"]) if "source" in task_data else parent_id

        # Text metadata (use raw X before imputation for missing_pct)
        text_rows.append(_build_text_metadata(
            task_id, X_raw, y, feature_names, parent_id, source,
        ))

        feats = compute_task_meta_features(X_raw, y, seed, landmarker_rows, landmarker_folds)
        feats["task_id"] = task_id
        meta_rows.append(feats)

    # --- Assemble DataFrames ---
    meta_df = pd.DataFrame(meta_rows).set_index("task_id")
    meta_df = meta_df[META_FEATURE_COLS]

    # --- Final NaN / inf check ---
    bad_mask = ~np.isfinite(meta_df.values)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.warning(
            "Replacing %d NaN/inf values in meta_features with 0.0", n_bad,
        )
        meta_df = meta_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    text_df = pd.DataFrame(text_rows)

    # --- Save ---
    mf_path = Path(meta_dir) / "meta_features.parquet"
    meta_df.to_parquet(mf_path, engine="pyarrow")
    logger.info("Saved meta_features → %s  (%d tasks, %d features)",
                mf_path, len(meta_df), len(META_FEATURE_COLS))

    tm_path = Path(meta_dir) / "text_metadata.parquet"
    text_df.to_parquet(tm_path, engine="pyarrow", index=False)
    logger.info("Saved text_metadata → %s  (%d tasks)", tm_path, len(text_df))

    return meta_df, text_df


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

    print(f"[metafeatures] mode={mode}  seed={cfg['seed']}")
    print(f"  tasks_dir = {cfg['paths']['tasks']}")
    print(f"  meta_dir  = {cfg['paths']['meta']}")

    meta_df, text_df = extract_all(cfg)

    print(f"\n{'='*60}")
    print(f"Meta-features extracted for {len(meta_df)} tasks")
    print(f"  Columns: {list(meta_df.columns)}")
    print(f"\nSummary statistics:")
    print(meta_df.describe().round(4).to_string())
    print(f"\nText metadata: {len(text_df)} rows, cols={list(text_df.columns)}")
    print(f"{'='*60}")
