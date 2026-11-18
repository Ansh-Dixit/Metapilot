"""Tests for preprocessing leakage prevention.

Verifies:
(a) Preprocessing (imputer and scaler) is embedded inside the sklearn Pipeline.
(b) Imputation and scaling statistics are computed strictly on training folds only.
"""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.portfolio import get_pipeline, get_all_pipeline_ids, needs_scaling


def test_imputation_and_scaling_fitted_on_train_only():
    """Verify imputer and scaler statistics match train fold exactly and exclude test fold."""
    # Create synthetic data where train and test have vastly different values
    rng = np.random.RandomState(42)
    
    # Fold 1: values around 10.0 with some NaNs
    X_train = rng.normal(loc=10.0, scale=1.0, size=(100, 4))
    X_train[0, 0] = np.nan
    X_train[5, 1] = np.nan
    y_train = rng.randint(0, 2, size=100)

    # Fold 2 (test): values around 1000.0 with some NaNs
    X_test = rng.normal(loc=1000.0, scale=1.0, size=(50, 4))
    X_test[1, 0] = np.nan
    y_test = rng.randint(0, 2, size=50)

    # Calculate expected training-only statistics
    expected_medians = np.nanmedian(X_train, axis=0)
    # Scaler mean will be computed after median imputation
    X_train_imputed = X_train.copy()
    for col_idx in range(4):
        mask = np.isnan(X_train_imputed[:, col_idx])
        X_train_imputed[mask, col_idx] = expected_medians[col_idx]
    expected_means = np.mean(X_train_imputed, axis=0)

    # Use a scaled pipeline, e.g. logreg_c10
    pipe = get_pipeline("logreg_c10", seed=42, X_sample=X_train)

    # Fit ONLY on training fold
    pipe.fit(X_train, y_train)

    # Extract transformers from preprocessor
    preprocessor = pipe.named_steps["preprocessor"]
    num_pipeline = preprocessor.named_transformers_["num"]
    imputer = num_pipeline.named_steps["imputer"]
    scaler = num_pipeline.named_steps["scaler"]

    # Verify imputer learned ONLY from train fold
    np.testing.assert_allclose(imputer.statistics_, expected_medians, rtol=1e-5)

    # Verify scaler learned ONLY from train fold (around 10.0, not contaminated by 1000.0 from test)
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)
    assert np.all(scaler.mean_ < 50.0), f"Scaler mean leaked test data: {scaler.mean_}"

    # Prediction on test should succeed without changing train statistics
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)
    np.testing.assert_allclose(imputer.statistics_, expected_medians, rtol=1e-5)


def test_cv_folds_isolate_preprocessing():
    """Verify that during K-fold CV, each fold fits distinct preprocessor statistics."""
    rng = np.random.RandomState(42)
    # Create 3-group dataset where each group has distinct means
    X = np.vstack([
        rng.normal(loc=10.0, scale=1.0, size=(30, 3)),
        rng.normal(loc=50.0, scale=1.0, size=(30, 3)),
        rng.normal(loc=100.0, scale=1.0, size=(30, 3)),
    ])
    y = np.array([0] * 45 + [1] * 45)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_means = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        pipe = get_pipeline("knn_5", seed=42, X_sample=X_train)
        pipe.fit(X_train, y_train)

        preprocessor = pipe.named_steps["preprocessor"]
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        fold_means.append(scaler.mean_.copy())

        # Assert scaler mean equals the training fold mean exactly
        np.testing.assert_allclose(scaler.mean_, np.mean(X_train, axis=0), rtol=1e-5)

    # Different folds must have slightly different training means (proving isolation)
    assert not np.allclose(fold_means[0], fold_means[1])


def test_all_18_pipelines_have_preprocessor_inside():
    """Assert all pipelines (pilot 18 and all extended) include ColumnTransformer as the first step."""
    pilot_pids = get_all_pipeline_ids(mode="pilot")
    assert len(pilot_pids) == 18

    pipeline_ids = get_all_pipeline_ids()
    assert len(pipeline_ids) >= 18

    dummy_X = np.ones((10, 3))
    for pid in pipeline_ids:
        pipe = get_pipeline(pid, seed=42, X_sample=dummy_X)
        step_names = [name for name, _ in pipe.steps]
        assert step_names[0] == "preprocessor", f"{pid} first step is not preprocessor"
        assert len(pipe.steps) == 2, f"{pid} does not have exactly 2 steps (preprocessor, model)"

