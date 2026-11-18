"""Preprocessing utilities for MetaPilot pipelines.

Builds a sklearn ColumnTransformer for numeric and categorical columns,
handles row-cap downsampling, and label encoding.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler


def _cast_to_float64(arr):
    """Safely cast numeric array slice to float64."""
    # Coerce unparseable values to NaN (then imputed). Needed when a sparse
    # categorical column is all-NaN in the training fold, gets typed numeric,
    # and a string such as 'Y' appears at transform time (OpenML anneal, did=2).
    try:
        if isinstance(arr, pd.DataFrame):
            return arr.astype(np.float64)
        return np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError):
        pass
    df = arr if isinstance(arr, pd.DataFrame) else pd.DataFrame(np.asarray(arr, dtype=object))
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Custom frequency encoder for high-cardinality categoricals
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features by their training-set frequency.

    Each category is replaced by its relative frequency in the training
    data.  Unseen categories at transform time get frequency 0.
    """

    def __init__(self):
        self.freq_maps_ = None

    def fit(self, X, y=None):
        """Learn frequency mapping for each column."""
        X = self._to_dataframe(X)
        self.freq_maps_ = {}
        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()
        return self

    def transform(self, X):
        """Replace categories with their learned frequencies."""
        X = self._to_dataframe(X)
        out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float64)
        for j, col in enumerate(X.columns):
            mapping = self.freq_maps_.get(col, {})
            out[:, j] = X[col].map(mapping).fillna(0.0).values
        return out

    @staticmethod
    def _to_dataframe(X):
        """Ensure input is a DataFrame for uniform processing."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


# ---------------------------------------------------------------------------
# Preprocessor builder
# ---------------------------------------------------------------------------

def build_preprocessor(X, y=None, cfg=None, needs_scaling=False):
    """Build a sklearn ColumnTransformer for the given data.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix used to detect column types.
    y : ignored
        Kept for API consistency.
    cfg : dict or None
        Full config dict.  Uses ``cfg["preprocessing"]["high_card_threshold"]``
        to distinguish low- vs high-cardinality categoricals.
        Defaults to threshold=20 if cfg is None.
    needs_scaling : bool
        If True, append a StandardScaler to the numeric pipeline.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    high_card_threshold = 20
    if cfg is not None:
        high_card_threshold = cfg["preprocessing"].get(
            "high_card_threshold", 20
        )

    # ---- detect column types ----
    if isinstance(X, np.ndarray):
        if X.dtype == object:
            numeric_cols = []
            cat_low_cols = []
            cat_high_cols = []
            for j in range(X.shape[1]):
                col = X[:, j]
                non_null = col[pd.notna(col)]
                try:
                    pd.to_numeric(non_null, errors="raise")
                    numeric_cols.append(j)
                except (ValueError, TypeError):
                    n_uniq = len(set(non_null))
                    if n_uniq <= high_card_threshold:
                        cat_low_cols.append(j)
                    else:
                        cat_high_cols.append(j)
        else:
            # ndarray: treat everything as numeric (common for synthetic data)
            numeric_cols = list(range(X.shape[1]))
            cat_low_cols = []
            cat_high_cols = []
    else:
        # DataFrame: split by dtype
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        # Split categoricals by cardinality
        cat_low_cols = [
            c for c in cat_cols if X[c].nunique() <= high_card_threshold
        ]
        cat_high_cols = [
            c for c in cat_cols if X[c].nunique() > high_card_threshold
        ]

    # ---- numeric pipeline ----
    num_steps = [
        ("cast_float", FunctionTransformer(_cast_to_float64)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if needs_scaling:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(num_steps)

    # ---- categorical pipelines ----
    cat_low_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    cat_high_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq_enc", FrequencyEncoder()),
    ])

    # ---- assemble transformers ----
    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if len(cat_low_cols) > 0:
        transformers.append(("cat_low", cat_low_pipeline, cat_low_cols))
    if len(cat_high_cols) > 0:
        transformers.append(("cat_high", cat_high_pipeline, cat_high_cols))

    # If no columns at all (edge case), add a passthrough numeric transformer
    if len(transformers) == 0:
        transformers.append(("num", numeric_pipeline, []))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Row-cap downsampling
# ---------------------------------------------------------------------------

def apply_row_cap(X, y, row_cap, seed):
    """Downsample X and y to at most ``row_cap`` rows.

    Uses stratified sampling to preserve class proportions.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    y : np.ndarray
    row_cap : int
        Maximum number of rows to keep.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_capped, y_capped : same types as inputs
    """
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    if n <= row_cap:
        return X, y

    # Fraction to keep
    keep_frac = row_cap / n

    # Use train_test_split to do stratified downsampling.
    # We keep the "train" portion sized to row_cap.
    X_keep, _, y_keep, _ = train_test_split(
        X, y,
        train_size=row_cap,
        random_state=seed,
        stratify=y,
    )
    return X_keep, y_keep


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(y):
    """Encode target labels to contiguous integers.

    Parameters
    ----------
    y : array-like
        Raw target labels (strings, ints, etc.).

    Returns
    -------
    y_encoded : np.ndarray of int
        Integer-encoded labels.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder for inverse transformation.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder
