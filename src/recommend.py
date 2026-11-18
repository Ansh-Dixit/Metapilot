"""Reusable recommendation core for the MetaPilot demo.

- train_final(): fit kNN, LightGBM Regressor, LightGBM LambdaRank and the Static
  portfolio on ALL full-mode tasks and save them (joblib) to models/.
- recommend(df, target_col, k, model_name): validate an uploaded table, compute the
  14 meta-features with the training code path (src.metafeatures.compute_task_meta_features),
  rank the 30 pipelines and explain the top pick in plain words.
- evaluate_pipelines(): actually run a few pipelines on the user's data (3-fold CV),
  reusing src.run_portfolio._evaluate_pair in a separate process with a per-fit time limit.

No logic is duplicated: preprocessing, meta-features, pipelines and selectors all come
from the existing modules.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.metafeatures import META_FEATURE_COLS, compute_task_meta_features
from src.metamodels import (
    KNNMetaLearner,
    LGBMLambdaRankSelector,
    LGBMRegressorSelector,
    StaticPortfolioSelector,
    _build_design_matrix,
)
from src.portfolio import get_all_pipeline_ids, get_family
from src.preprocessing import apply_row_cap, encode_labels

ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "models" / "metapilot_models.joblib"
EXAMPLES_DIR = ROOT / "app" / "examples"

MODEL_NAMES = ["LightGBM LambdaRank", "LightGBM Regressor", "kNN", "Static portfolio"]
MAX_FILE_MB = 50
MIN_ROWS = 50
MIN_CLASS_ROWS = 6
CONTINUOUS_UNIQUE = 20


class RecommendError(ValueError):
    """A problem with the user's data, with a message meant for a non-expert."""


# ---------------------------------------------------------------------------
# Plain-language vocabulary
# ---------------------------------------------------------------------------

FAMILY_LABELS = {
    "logreg": "Logistic regression", "ridge": "Ridge classifier", "naive_bayes": "Naive Bayes",
    "knn": "Nearest neighbours", "decision_tree": "Decision tree", "random_forest": "Random forest",
    "extra_trees": "Extra trees", "grad_boosting": "Gradient boosting", "svm": "Support vector machine",
    "mlp": "Neural network",
}

PIPELINE_DESCRIPTIONS = {
    "logreg_c0.1": "Logistic regression, strongly regularised (simple straight-line boundaries)",
    "logreg_c1": "Logistic regression, standard settings",
    "logreg_c10": "Logistic regression, lightly regularised (fits the data more closely)",
    "ridge": "Ridge classifier, standard settings (fast linear model)",
    "ridge_alpha0.1": "Ridge classifier, light regularisation",
    "ridge_alpha10": "Ridge classifier, strong regularisation",
    "gnb": "Naive Bayes (very fast; assumes columns are independent)",
    "gnb_var1e_7": "Naive Bayes with slightly more smoothing",
    "gnb_var1e_5": "Naive Bayes with more smoothing",
    "knn_1": "Nearest neighbour: copies the label of the single most similar row",
    "knn_5": "Nearest neighbours: majority vote of the 5 most similar rows",
    "knn_15": "Nearest neighbours: majority vote of the 15 most similar rows (smoother)",
    "dt_d3": "Decision tree with at most 3 levels (easy to read)",
    "dt_d10": "Decision tree with up to 10 levels",
    "dt_full": "Decision tree grown fully (can overfit)",
    "rf_100": "Random forest of 100 trees",
    "rf_100_d5": "Random forest of 100 shallow trees (depth 5)",
    "rf_100_entropy": "Random forest of 100 trees using an information-gain split rule",
    "et_100": "Extra trees: 100 extra-randomised trees",
    "et_100_d5": "Extra trees: 100 shallow extra-randomised trees (depth 5)",
    "et_100_entropy": "Extra trees with an information-gain split rule",
    "hgb_default": "Gradient boosting (histogram-based), standard settings",
    "hgb_shallow_lr05": "Gradient boosting with shallow trees and slow learning",
    "hgb_l2_1": "Gradient boosting with extra regularisation",
    "svm_c0.1": "Support vector machine (curved boundaries), strongly regularised",
    "svm_c1": "Support vector machine (curved boundaries), standard settings",
    "svm_c10": "Support vector machine (curved boundaries), lightly regularised",
    "mlp_32": "Small neural network (32 hidden units)",
    "mlp_128": "Neural network (128 hidden units)",
    "mlp_alpha01": "Neural network (64 hidden units) with extra regularisation",
}

# (phrase when the value is high, phrase when it is low) relative to the 1,040 benchmark tasks
FEATURE_PHRASES = {
    "log_n_rows": ("a large dataset (many rows)", "a small dataset (few rows)"),
    "log_n_cols": ("many columns", "few columns"),
    "cols_to_rows": ("many columns compared with rows", "many rows per column"),
    "n_classes": ("many classes", "few classes"),
    "mean_abs_corr": ("strongly related (correlated) columns", "mostly unrelated columns"),
    "mean_abs_skew": ("lopsided (skewed) numeric values", "fairly symmetric numeric values"),
    "mean_kurtosis": ("extreme values / outliers", "few extreme values"),
    "pca95_frac": ("information spread over many columns", "a few columns carrying most of the information"),
    "norm_class_entropy": ("evenly balanced classes", "unbalanced classes"),
    "minority_majority_ratio": ("no rare class", "a rare minority class"),
    "lm_stump": ("one column that alone separates the classes well", "no single column that separates the classes well"),
    "lm_1nn": ("similar rows that usually share a class", "similar rows that often have different classes (noisy)"),
    "lm_gnb": ("a structure simple probabilistic models handle well", "a structure simple probabilistic models struggle with"),
    "lm_lda": ("classes that are close to linearly separable", "classes that are not linearly separable"),
}

SCORE_MEANING = {
    "LightGBM LambdaRank": "ranking score (higher = better)",
    "LightGBM Regressor": "predicted accuracy gap to the best pipeline, in % points (closer to 0 = better)",
    "kNN": "average relative score on the 5 most similar benchmark datasets (0-1, higher = better)",
    "Static portfolio": "position in the fixed default order (higher = earlier)",
}


# ---------------------------------------------------------------------------
# Training the final models
# ---------------------------------------------------------------------------

def train_final(mode: str = "full", out_path: Path = MODELS_PATH) -> Path:
    """Fit the selectors on ALL tasks of `mode` and save one joblib bundle."""
    cfg = load_config(mode)
    meta_dir = Path(cfg["paths"]["meta"])
    perf = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta = pd.read_parquet(meta_dir / "meta_features.parquet").loc[perf.index, META_FEATURE_COLS]
    gap = pd.read_parquet(meta_dir / "gap_targets.parquet").loc[perf.index]
    lam = pd.read_parquet(meta_dir / "lambdarank_labels.parquet").loc[perf.index]
    tasks = list(perf.index)
    lg = cfg["metamodels"]["lgbm"]
    models = {
        "kNN": KNNMetaLearner(k=cfg["metamodels"]["knn_k"]),
        "LightGBM Regressor": LGBMRegressorSelector(seed=cfg["seed"], **lg),
        "LightGBM LambdaRank": LGBMLambdaRankSelector(seed=cfg["seed"], **lg),
        "Static portfolio": StaticPortfolioSelector(),
    }
    for m in models.values():
        m.fit(tasks, meta, perf, gap, lam)
    pids = list(perf.columns)
    bundle = {
        "models": models,
        "pipeline_ids": pids,
        "families": {p: get_family(p) for p in pids},
        "meta_cols": META_FEATURE_COLS,
        "feature_train_values": meta.copy(),  # for plain-language percentiles
        "feature_stats": meta.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T,
        "n_train_tasks": len(tasks),
        "meta_cfg": {"seed": cfg["seed"], "landmarker_rows": cfg["meta"]["landmarker_rows"],
                     "landmarker_folds": cfg["meta"]["landmarker_folds"]},
        "mode": mode,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


_BUNDLE_CACHE: dict = {}


def load_bundle(path: Path = MODELS_PATH) -> dict:
    path = Path(path)
    if str(path) not in _BUNDLE_CACHE:
        if not path.exists():
            raise RecommendError("The recommendation models are not trained yet. Run `make demo` "
                                 "(or `python -m src.train_final`) first.")
        _BUNDLE_CACHE[str(path)] = joblib.load(path)
    return _BUNDLE_CACHE[str(path)]


# ---------------------------------------------------------------------------
# Reading and validating user data
# ---------------------------------------------------------------------------

def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_FILE_MB:
        raise RecommendError(f"This file is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB. "
                             "Try a sample of your rows.")
    try:
        df = pd.read_csv(path)  # comma first: sniffing misreads headers with quotes or '='
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=None, engine="python")  # then sniff ; or tab
    except Exception as exc:  # noqa: BLE001
        raise RecommendError(f"Could not read this file as a CSV table ({type(exc).__name__}).") from exc
    if df.shape[1] < 2:
        raise RecommendError("The table needs at least two columns: one or more inputs and the column to predict.")
    return df


_ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|index|key)$|^unnamed:\s*\d+$", re.I)


def _is_id_like(name: str, col: pd.Series) -> bool:
    if _ID_NAME.search(str(name).strip()):
        return True
    s = col.dropna()
    if len(s) < 20 or s.nunique() < len(col):
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return True  # unique text on every row (names, codes)
    vals = s.to_numpy(dtype=float)
    return bool(np.all(np.mod(vals, 1) == 0) and np.all(np.diff(vals) > 0))  # row counter


def prepare_dataset(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate and convert a user table into (X_raw, y, info) in the training representation.

    X_raw follows src.data_loading: float64 if every column is numeric, otherwise an
    object array whose numeric columns hold numbers and other columns hold strings.
    """
    if target_col not in df.columns:
        raise RecommendError(f"Column '{target_col}' is not in the file.")
    warnings: list[str] = []
    n_before = len(df)
    df = df[df[target_col].notna()].copy()
    if len(df) < n_before:
        warnings.append(f"{n_before - len(df)} rows without a value in '{target_col}' were ignored.")

    target = df.pop(target_col)
    tnum = pd.to_numeric(target, errors="coerce")
    if tnum.notna().all() and tnum.nunique() > CONTINUOUS_UNIQUE:
        warnings.append(f"'{target_col}' has {tnum.nunique()} different numbers: it looks like a quantity, "
                        "not categories. MetaPilot recommends classification pipelines; consider "
                        "grouping it into a few categories.")

    dropped = []
    for c in list(df.columns):
        if df[c].isna().all():
            dropped.append(f"{c} (empty)")
            df.pop(c)
        elif _is_id_like(c, df[c]):
            dropped.append(f"{c} (looks like an ID)")
            df.pop(c)
    if df.shape[1] == 0:
        raise RecommendError("No usable input columns are left after removing empty and ID-like columns.")

    y_raw = target.to_numpy()
    classes, counts = np.unique(y_raw.astype(str), return_counts=True)
    if len(classes) < 2:
        raise RecommendError(f"'{target_col}' has only one value ({classes[0]!r}); there is nothing to predict. "
                             "Pick a column with at least two categories.")
    if len(df) < MIN_ROWS:
        raise RecommendError(f"The file has only {len(df)} usable rows; MetaPilot needs at least {MIN_ROWS}.")
    small = [f"{c!r} ({n} rows)" for c, n in zip(classes, counts) if n < MIN_CLASS_ROWS]
    if small:
        raise RecommendError(f"Every category of '{target_col}' needs at least {MIN_CLASS_ROWS} rows. "
                             f"Too small: {', '.join(small[:5])}{' …' if len(small) > 5 else ''}. "
                             "Merge rare categories or pick another column.")

    # numeric-looking text columns become numbers (as OpenML/sklearn data were)
    numeric_cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            numeric_cols.append(c)
    if len(numeric_cols) == df.shape[1]:
        X = df.to_numpy(dtype=np.float64)
    else:
        obj = df.astype(object)
        X = obj.where(df.notna(), np.nan).to_numpy(dtype=object)
        for j, c in enumerate(df.columns):
            if c not in numeric_cols:
                X[:, j] = [v if (isinstance(v, float) and np.isnan(v)) else str(v) for v in X[:, j]]

    y, _ = encode_labels(y_raw.astype(str) if not pd.api.types.is_numeric_dtype(target) else y_raw)
    info = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "classes": int(len(classes)),
        "class_names": [str(c) for c in classes],
        "minority_share_pct": float(100 * counts.min() / counts.sum()),
        "missing_pct": float(100 * df.isna().to_numpy().mean()),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": int(df.shape[1] - len(numeric_cols)),
        "dropped_columns": dropped,
        "warnings": warnings,
        "feature_names": [str(c) for c in df.columns],
    }
    return X, np.asarray(y, dtype=np.int64), info


def meta_features_for(X_raw: np.ndarray, y: np.ndarray, bundle: dict) -> pd.DataFrame:
    """The 14 meta-features via the exact training code path, as a one-row frame."""
    mc = bundle["meta_cfg"]
    feats = compute_task_meta_features(X_raw, y, mc["seed"], mc["landmarker_rows"], mc["landmarker_folds"])
    return pd.DataFrame([feats], index=["your_data"])[bundle["meta_cols"]]


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def _percentile(bundle: dict, feature: str, value: float) -> float:
    train = bundle["feature_train_values"][feature].to_numpy()
    return float((train < value).mean() + 0.5 * (train == value).mean())


def describe_feature(bundle: dict, feature: str, value: float) -> str:
    pct = _percentile(bundle, feature, value)
    hi, lo = FEATURE_PHRASES[feature]
    if pct >= 0.6:
        return hi
    if pct <= 0.4:
        return lo
    return f"an average level of '{lo.split('(')[0].strip()}' vs '{hi.split('(')[0].strip()}'"


def explain_top_pick(bundle: dict, meta_row: pd.DataFrame, top_pid: str, model_name: str) -> dict:
    """Top-3 meta-features by |LightGBM pred_contrib| for the top pick, in plain words."""
    src_name = model_name if model_name in ("LightGBM Regressor", "LightGBM LambdaRank") else "LightGBM LambdaRank"
    sel = bundle["models"][src_name]
    X_design, _, _ = _build_design_matrix(["your_data"], meta_row, sel.pipeline_ids, scaler=sel.scaler_)
    contrib = sel.model_.predict(X_design, pred_contrib=True)[sel.pipeline_ids.index(top_pid)]
    n_meta = len(bundle["meta_cols"])
    meta_contrib = pd.Series(contrib[:n_meta], index=bundle["meta_cols"])
    top = meta_contrib.reindex(meta_contrib.abs().sort_values(ascending=False).index)[:3]
    reasons = []
    for feat, c in top.items():
        phrase = describe_feature(bundle, feat, float(meta_row.iloc[0][feat]))
        direction = "makes it more likely to do well" if c > 0 else "makes it less likely to do well"
        reasons.append({"feature": feat, "contribution": float(c), "plain": phrase,
                        "sentence": f"Your data has {phrase}, which {direction}."})
    note = "" if src_name == model_name else f" (explanation from the {src_name} model)"
    return {"source_model": src_name, "reasons": reasons,
            "headline": f"Why {top_pid} is first{note}:"}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def _rows(bundle: dict, ranking: list[str], scores: dict | None, k: int) -> list[dict]:
    out = []
    for i, pid in enumerate(ranking[:k], start=1):
        fam = bundle["families"][pid]
        out.append({"rank": i, "pipeline_id": pid, "family": FAMILY_LABELS.get(fam, fam),
                    "description": PIPELINE_DESCRIPTIONS.get(pid, pid),
                    "score": None if scores is None else round(float(scores[pid]), 4)})
    return out


def recommend(df: pd.DataFrame, target_col: str, k: int = 3, model_name: str = "LightGBM LambdaRank",
              bundle: dict | None = None) -> dict:
    """Recommend the top-k pipelines for a table. Raises RecommendError with a friendly message."""
    bundle = bundle or load_bundle()
    if model_name not in MODEL_NAMES:
        raise RecommendError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}.")
    k = int(max(1, min(int(k), len(bundle["pipeline_ids"]))))
    X_raw, y, info = prepare_dataset(df, target_col)
    meta_row = meta_features_for(X_raw, y, bundle)

    sel = bundle["models"][model_name]
    rankings, scores = sel.predict(["your_data"], meta_row)
    ranking = rankings["your_data"]
    static = bundle["models"]["Static portfolio"].order_

    return {
        "summary": {k_: v for k_, v in info.items() if k_ != "feature_names"},
        "meta_features": {c: float(meta_row.iloc[0][c]) for c in bundle["meta_cols"]},
        "model": model_name,
        "score_meaning": SCORE_MEANING[model_name],
        "top_k": _rows(bundle, ranking, scores.loc["your_data"].to_dict(), k),
        "full_ranking": ranking,
        "static_top_k": _rows(bundle, static, None, k),
        "explanation": explain_top_pick(bundle, meta_row, ranking[0], model_name),
        "warnings": info["warnings"],
    }


def plain_summary(summary: dict) -> str:
    """One short paragraph describing the dataset for a non-expert."""
    s = summary
    size = "small" if s["rows"] < 1000 else ("medium-sized" if s["rows"] < 20000 else "large")
    bal = ("balanced" if s["minority_share_pct"] >= 100 / s["classes"] * 0.7 else
           f"unbalanced (the rarest category is {s['minority_share_pct']:.1f}% of rows)")
    parts = [f"A {size} table: {s['rows']:,} rows and {s['columns']} input columns "
             f"({s['numeric_columns']} numeric, {s['categorical_columns']} text/category).",
             f"You are predicting one of {s['classes']} categories; the categories are {bal}."]
    parts.append(f"{s['missing_pct']:.1f}% of the values are missing." if s["missing_pct"] > 0
                 else "No values are missing.")
    if s["dropped_columns"]:
        parts.append("Ignored columns: " + ", ".join(s["dropped_columns"]) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Running pipelines on the user's data
# ---------------------------------------------------------------------------

def _eval_one(pid: str, X: np.ndarray, y: np.ndarray, timeout: int, seed: int) -> dict:
    """Runs in a worker process (main thread), so the SIGALRM per-fit timeout works."""
    from src.run_portfolio import _evaluate_pair
    cfg = load_config("full")  # same preprocessing settings as the benchmark
    cfg["cv"] = {**cfg["cv"], "n_folds": 3, "fit_timeout_sec": timeout, "fit_timeout_wide_sec": timeout}
    return _evaluate_pair(X, y, pid, cfg, seed)


def evaluate_pipelines(df: pd.DataFrame, target_col: str, pipeline_ids: list[str], row_cap: int = 2000,
                       timeout_per_fit: int = 60, seed: int = 42, progress=None) -> pd.DataFrame:
    """3-fold stratified CV of the given pipelines on the user's data (row cap, per-fit time limit).

    Each pipeline runs in a separate worker process via src.run_portfolio._evaluate_pair; a
    hard outer limit (3 folds x limit + 30 s) kills a stuck worker.
    """
    X_raw, y, _ = prepare_dataset(df, target_col)
    X, y = apply_row_cap(X_raw, y, row_cap, seed)
    ctx = mp.get_context("spawn")
    rows = []
    pool = ctx.Pool(1)
    try:
        for i, pid in enumerate(pipeline_ids):
            if progress is not None:
                progress(i / len(pipeline_ids), desc=f"Running {pid} ({i + 1}/{len(pipeline_ids)})")
            job = pool.apply_async(_eval_one, (pid, X, y, timeout_per_fit, seed))
            try:
                res = job.get(timeout=3 * timeout_per_fit + 30)
            except mp.TimeoutError:
                pool.terminate()
                pool = ctx.Pool(1)
                res = {"mean_accuracy": float("nan"), "status": "timeout"}
            status = res["status"]
            rows.append({
                "pipeline_id": pid,
                "accuracy_pct": None if status != "ok" else round(100 * res["mean_accuracy"], 2),
                "status": "ok" if status == "ok" else ("too slow (time limit)" if status == "timeout"
                                                       else "failed"),
            })
    finally:
        pool.terminate()
    if progress is not None:
        progress(1.0, desc="Done")
    out = pd.DataFrame(rows)
    out["rows_used"] = len(y)
    return out


# ---------------------------------------------------------------------------
# Example datasets (written once to app/examples/)
# ---------------------------------------------------------------------------

OPENML_EXAMPLES = {"openml_31": "credit-g", "openml_37": "diabetes", "openml_1049": "pc4"}


def ensure_examples(out_dir: Path = EXAMPLES_DIR) -> dict[str, Path]:
    """Write example CSVs (target = last column). Returns {display name: path}."""
    from sklearn import datasets as skd

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, loader in [("iris", skd.load_iris), ("wine", skd.load_wine), ("breast_cancer", skd.load_breast_cancer)]:
        p = out_dir / f"{name}.csv"
        if not p.exists():
            ds = loader()
            df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df["target"] = [str(ds.target_names[t]) for t in ds.target]
            df.to_csv(p, index=False)
        paths[f"{name} (scikit-learn)"] = p
    for pid, name in OPENML_EXAMPLES.items():
        p = out_dir / f"{name}.csv"
        raw = ROOT / "data" / "raw" / f"{pid}.npz"
        if not p.exists() and raw.exists():
            with np.load(raw, allow_pickle=True) as d:
                df = pd.DataFrame(d["X"], columns=[str(c) for c in d["feature_names"]])
                df["class"] = [f"class_{int(v)}" for v in d["y"]]
            df.to_csv(p, index=False)
        if p.exists():
            paths[f"{name} (OpenML)"] = p
    return paths
