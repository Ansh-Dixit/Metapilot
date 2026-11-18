"""Pipeline registry for MetaPilot.

Defines the classification pipelines (18 in pilot mode, 30 in full mode),
each wrapped in a sklearn Pipeline with a preprocessor (ColumnTransformer)
as the first step and the estimator as the second step.
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

# ---------------------------------------------------------------------------
# Pilot pipeline identifiers (must remain unchanged)
# ---------------------------------------------------------------------------

PILOT_PIPELINE_IDS = [
    "logreg_c0.1",
    "logreg_c10",
    "ridge",
    "gnb",
    "knn_1",
    "knn_5",
    "knn_15",
    "dt_d3",
    "dt_full",
    "rf_100",
    "rf_100_d5",
    "et_100",
    "hgb_default",
    "hgb_shallow_lr05",
    "svm_c1",
    "svm_c10",
    "mlp_32",
    "mlp_128",
]

# ---------------------------------------------------------------------------
# Pipeline metadata: pipeline_id -> (family, needs_scaling)
# ---------------------------------------------------------------------------

_PIPELINE_META = {
    # --- Pilot 18 pipelines ---
    "logreg_c0.1":       ("logreg",          True),
    "logreg_c10":        ("logreg",          True),
    "ridge":             ("ridge",           True),
    "gnb":               ("naive_bayes",     False),
    "knn_1":             ("knn",             True),
    "knn_5":             ("knn",             True),
    "knn_15":            ("knn",             True),
    "dt_d3":             ("decision_tree",   False),
    "dt_full":           ("decision_tree",   False),
    "rf_100":            ("random_forest",   False),
    "rf_100_d5":         ("random_forest",   False),
    "et_100":            ("extra_trees",     False),
    "hgb_default":       ("grad_boosting",   False),
    "hgb_shallow_lr05":  ("grad_boosting",   False),
    "svm_c1":            ("svm",             True),
    "svm_c10":           ("svm",             True),
    "mlp_32":            ("mlp",             True),
    "mlp_128":           ("mlp",             True),

    # --- Extended pipelines for full study (3 to 6 per family) ---
    # logreg (total 4)
    "logreg_c1":         ("logreg",          True),
    "logreg_l1_c1":      ("logreg",          True),

    # ridge (total 3)
    "ridge_alpha0.1":    ("ridge",           True),
    "ridge_alpha10":     ("ridge",           True),

    # naive_bayes (total 3)
    "gnb_var1e_7":       ("naive_bayes",     False),
    "gnb_var1e_5":       ("naive_bayes",     False),

    # knn (total 5)
    "knn_3":             ("knn",             True),
    "knn_7_dist":        ("knn",             True),

    # decision_tree (total 4)
    "dt_d5":             ("decision_tree",   False),
    "dt_d10":            ("decision_tree",   False),

    # random_forest (total 5)
    "rf_200":            ("random_forest",   False),
    "rf_100_entropy":    ("random_forest",   False),
    "rf_200_d10":        ("random_forest",   False),

    # extra_trees (total 4)
    "et_200":            ("extra_trees",     False),
    "et_100_d5":         ("extra_trees",     False),
    "et_100_entropy":    ("extra_trees",     False),

    # grad_boosting (total 5)
    "hgb_lr01":          ("grad_boosting",   False),
    "hgb_deep_d6":       ("grad_boosting",   False),
    "hgb_l2_1":          ("grad_boosting",   False),

    # svm (total 5)
    "svm_c0.1":          ("svm",             True),
    "svm_linear_c1":     ("svm",             True),
    "svm_poly_d3":       ("svm",             True),

    # mlp (total 4)
    "mlp_64_32":         ("mlp",             True),
    "mlp_alpha01":       ("mlp",             True),
}


# ---------------------------------------------------------------------------
# Budget override (see docs/PROGRESS.md, "Deliberate deviations"):
# the full study uses 30 pipelines = 18 pilot + 12 extended, with at most
# 3 configs per family. The 12 dropped extended configs (the costliest MLP /
# RF ones first) are removed from the registry so no code path can run them.
# ---------------------------------------------------------------------------

FULL_EXTRA_PIPELINE_IDS = [
    "logreg_c1",
    "ridge_alpha0.1", "ridge_alpha10",
    "gnb_var1e_7", "gnb_var1e_5",
    "dt_d10",
    "rf_100_entropy",
    "et_100_d5", "et_100_entropy",
    "hgb_l2_1",
    "svm_c0.1",
    "mlp_alpha01",
]

_DROPPED_FOR_BUDGET = [
    "logreg_l1_c1", "knn_3", "knn_7_dist", "dt_d5", "rf_200", "rf_200_d10",
    "et_200", "hgb_lr01", "hgb_deep_d6", "svm_linear_c1", "svm_poly_d3", "mlp_64_32",
]

_PIPELINE_META = {
    pid: meta for pid, meta in _PIPELINE_META.items()
    if pid in PILOT_PIPELINE_IDS or pid in FULL_EXTRA_PIPELINE_IDS
}
assert len(_PIPELINE_META) == 30, len(_PIPELINE_META)


# ---------------------------------------------------------------------------
# Model builder functions (seed -> estimator)
# ---------------------------------------------------------------------------

# logreg
def _logreg_c01(seed):
    return LogisticRegression(C=0.1, max_iter=1000, random_state=seed)

def _logreg_c10(seed):
    return LogisticRegression(C=10, max_iter=1000, random_state=seed)

def _logreg_c1(seed):
    return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)

def _logreg_l1_c1(seed):
    return LogisticRegression(C=1.0, penalty="l1", solver="saga", max_iter=1000, random_state=seed)

# ridge
def _ridge(seed):
    return RidgeClassifier(random_state=seed)

def _ridge_alpha01(seed):
    return RidgeClassifier(alpha=0.1, random_state=seed)

def _ridge_alpha10(seed):
    return RidgeClassifier(alpha=10.0, random_state=seed)

# naive_bayes
def _gnb(seed):
    return GaussianNB()

def _gnb_var1e_7(seed):
    return GaussianNB(var_smoothing=1e-7)

def _gnb_var1e_5(seed):
    return GaussianNB(var_smoothing=1e-5)

# knn
def _knn_1(seed):
    return KNeighborsClassifier(n_neighbors=1)

def _knn_3(seed):
    return KNeighborsClassifier(n_neighbors=3)

def _knn_5(seed):
    return KNeighborsClassifier(n_neighbors=5)

def _knn_7_dist(seed):
    return KNeighborsClassifier(n_neighbors=7, weights="distance")

def _knn_15(seed):
    return KNeighborsClassifier(n_neighbors=15)

# decision_tree
def _dt_d3(seed):
    return DecisionTreeClassifier(max_depth=3, random_state=seed)

def _dt_d5(seed):
    return DecisionTreeClassifier(max_depth=5, random_state=seed)

def _dt_d10(seed):
    return DecisionTreeClassifier(max_depth=10, random_state=seed)

def _dt_full(seed):
    return DecisionTreeClassifier(random_state=seed)

# random_forest
def _rf_100(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed)

def _rf_100_d5(seed):
    return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)

def _rf_100_entropy(seed):
    return RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=seed)

def _rf_200(seed):
    return RandomForestClassifier(n_estimators=200, random_state=seed)

def _rf_200_d10(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)

# extra_trees
def _et_100(seed):
    return ExtraTreesClassifier(n_estimators=100, random_state=seed)

def _et_100_d5(seed):
    return ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)

def _et_100_entropy(seed):
    return ExtraTreesClassifier(n_estimators=100, criterion="entropy", random_state=seed)

def _et_200(seed):
    return ExtraTreesClassifier(n_estimators=200, random_state=seed)

# grad_boosting
def _hgb_default(seed):
    return HistGradientBoostingClassifier(random_state=seed)

def _hgb_shallow_lr05(seed):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=seed)

def _hgb_lr01(seed):
    return HistGradientBoostingClassifier(learning_rate=0.01, random_state=seed)

def _hgb_deep_d6(seed):
    return HistGradientBoostingClassifier(max_depth=6, random_state=seed)

def _hgb_l2_1(seed):
    return HistGradientBoostingClassifier(l2_regularization=1.0, random_state=seed)

# svm
def _svm_c01(seed):
    return SVC(C=0.1, kernel="rbf", random_state=seed)

def _svm_c1(seed):
    return SVC(C=1.0, kernel="rbf", random_state=seed)

def _svm_c10(seed):
    return SVC(C=10.0, kernel="rbf", random_state=seed)

def _svm_linear_c1(seed):
    return SVC(C=1.0, kernel="linear", random_state=seed)

def _svm_poly_d3(seed):
    return SVC(C=1.0, kernel="poly", degree=3, random_state=seed)

# mlp
def _mlp_32(seed):
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed)

def _mlp_128(seed):
    return MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)

def _mlp_64_32(seed):
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)

def _mlp_alpha01(seed):
    return MLPClassifier(hidden_layer_sizes=(64,), alpha=0.1, max_iter=300, random_state=seed)

# tabpfn optional
def _tabpfn_builder(seed):
    try:
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu", random_state=seed)
    except Exception as e:
        raise ImportError(f"TabPFN requested but failed to initialize: {e}")


# Map pipeline_id -> model builder function
_MODEL_BUILDERS = {
    # Pilot 18
    "logreg_c0.1":      _logreg_c01,
    "logreg_c10":       _logreg_c10,
    "ridge":            _ridge,
    "gnb":              _gnb,
    "knn_1":            _knn_1,
    "knn_5":            _knn_5,
    "knn_15":           _knn_15,
    "dt_d3":            _dt_d3,
    "dt_full":          _dt_full,
    "rf_100":           _rf_100,
    "rf_100_d5":        _rf_100_d5,
    "et_100":           _et_100,
    "hgb_default":      _hgb_default,
    "hgb_shallow_lr05": _hgb_shallow_lr05,
    "svm_c1":           _svm_c1,
    "svm_c10":          _svm_c10,
    "mlp_32":           _mlp_32,
    "mlp_128":          _mlp_128,

    # Extended
    "logreg_c1":        _logreg_c1,
    "logreg_l1_c1":     _logreg_l1_c1,
    "ridge_alpha0.1":   _ridge_alpha01,
    "ridge_alpha10":    _ridge_alpha10,
    "gnb_var1e_7":      _gnb_var1e_7,
    "gnb_var1e_5":      _gnb_var1e_5,
    "knn_3":            _knn_3,
    "knn_7_dist":       _knn_7_dist,
    "dt_d5":            _dt_d5,
    "dt_d10":           _dt_d10,
    "rf_200":           _rf_200,
    "rf_100_entropy":   _rf_100_entropy,
    "rf_200_d10":       _rf_200_d10,
    "et_200":           _et_200,
    "et_100_d5":        _et_100_d5,
    "et_100_entropy":   _et_100_entropy,
    "hgb_lr01":         _hgb_lr01,
    "hgb_deep_d6":      _hgb_deep_d6,
    "hgb_l2_1":         _hgb_l2_1,
    "svm_c0.1":         _svm_c01,
    "svm_linear_c1":    _svm_linear_c1,
    "svm_poly_d3":      _svm_poly_d3,
    "mlp_64_32":        _mlp_64_32,
    "mlp_alpha01":      _mlp_alpha01,
}


# ---------------------------------------------------------------------------
# Public registry: pipeline_id -> (family, builder_fn)
# ---------------------------------------------------------------------------

def _make_pipeline_builder(pipeline_id):
    """Create a builder_fn(seed, preprocessor) -> Pipeline for a pipeline_id."""
    model_fn = _MODEL_BUILDERS[pipeline_id]

    def builder_fn(seed, preprocessor):
        model = model_fn(seed)
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

    return builder_fn


PIPELINE_REGISTRY = {
    pid: (_PIPELINE_META[pid][0], _make_pipeline_builder(pid))
    for pid in _PIPELINE_META
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pipeline(pipeline_id, seed, X_sample=None, cfg=None):
    """Build the full sklearn Pipeline for a given pipeline_id."""
    if pipeline_id == "tabpfn":
        use_tabpfn = cfg.get("use_tabpfn", False) if cfg else False
        if not use_tabpfn:
            raise KeyError("TabPFN is disabled in config (use_tabpfn: false).")
        import numpy as np
        if X_sample is None:
            X_sample = np.zeros((2, 2))
        preprocessor = build_preprocessor(X_sample, y=None, cfg=cfg, needs_scaling=False)
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", _tabpfn_builder(seed)),
        ])

    if pipeline_id not in PIPELINE_REGISTRY:
        raise KeyError(
            f"Unknown pipeline_id '{pipeline_id}'. "
            f"Valid ids: {get_all_pipeline_ids()}"
        )

    import numpy as np

    scaling = needs_scaling(pipeline_id)

    if X_sample is None:
        X_sample = np.zeros((2, 2))

    preprocessor = build_preprocessor(
        X_sample, y=None, cfg=cfg, needs_scaling=scaling
    )

    _, builder_fn = PIPELINE_REGISTRY[pipeline_id]
    return builder_fn(seed, preprocessor)


def get_all_pipeline_ids(mode: str | None = None) -> list[str]:
    """Return a sorted list of pipeline identifiers.

    If mode == 'pilot', returns the 18 pilot pipeline IDs.
    Otherwise returns all 30 registered pipeline IDs.
    """
    if mode == "pilot":
        return sorted(PILOT_PIPELINE_IDS)
    return sorted(PIPELINE_REGISTRY.keys())


def get_family(pipeline_id: str) -> str:
    """Return the family string for a pipeline_id."""
    if pipeline_id == "tabpfn":
        return "tabpfn"
    if pipeline_id not in PIPELINE_REGISTRY:
        raise KeyError(f"Unknown pipeline_id '{pipeline_id}'.")
    return PIPELINE_REGISTRY[pipeline_id][0]


def needs_scaling(pipeline_id: str) -> bool:
    """Return True if the pipeline requires feature scaling."""
    if pipeline_id == "tabpfn":
        return False
    if pipeline_id not in _PIPELINE_META:
        raise KeyError(f"Unknown pipeline_id '{pipeline_id}'.")
    return _PIPELINE_META[pipeline_id][1]
