"""Categorical columns must reach their encoders intact (never cast to float).

Verifies:
1. `_cast_to_float64` sits only in the numeric transformer, and string columns
   are routed to the one-hot (low-cardinality) or frequency (high-cardinality)
   encoder with their original string values.
2. A sparse categorical column that is all-NaN in the training fold (typed
   numeric) does not crash when a string such as 'Y' appears at transform time
   (the OpenML anneal, did=2, failure).
"""

import numpy as np

from src.preprocessing import build_preprocessor


def _mixed_X(n=60, seed=0):
    rng = np.random.RandomState(seed)
    num = rng.randn(n)
    low = rng.choice(["a", "b", "c"], size=n)
    high = np.array([f"k{i % 25}" for i in range(n)])
    return np.column_stack([num.astype(object), low, high]).astype(object), low, high


def test_cast_only_in_numeric_transformer_and_strings_reach_encoders():
    X, low, high = _mixed_X()
    ct = build_preprocessor(X, needs_scaling=False)

    routing = {name: cols for name, _, cols in ct.transformers}
    assert routing == {"num": [0], "cat_low": [1], "cat_high": [2]}
    for name, pipe, _ in ct.transformers:
        has_cast = "cast_float" in pipe.named_steps
        assert has_cast == (name == "num"), f"cast_float misplaced in {name}"

    out = ct.fit_transform(X)
    assert out.shape == (len(X), 1 + 3 + 1)

    # One-hot encoder saw the original strings, not floats/NaN
    onehot = ct.named_transformers_["cat_low"].named_steps["onehot"]
    assert list(onehot.categories_[0]) == ["a", "b", "c"]
    onehot_block = out[:, 1:4]
    expected = (low[:, None] == np.array(["a", "b", "c"])[None, :]).astype(float)
    np.testing.assert_array_equal(onehot_block, expected)

    # Frequency encoder learned frequencies keyed by the original strings
    freq = ct.named_transformers_["cat_high"].named_steps["freq_enc"]
    fmap = next(iter(freq.freq_maps_.values()))
    assert set(fmap) == set(high)
    expected_freq = np.array([np.mean(high == v) for v in high])
    np.testing.assert_allclose(out[:, 4], expected_freq)


def test_all_nan_train_column_with_string_at_test_time():
    rng = np.random.RandomState(1)
    n = 40
    X_train = np.column_stack([rng.randn(n), np.full(n, np.nan)]).astype(object)
    X_test = np.column_stack([rng.randn(5), np.array(["Y", np.nan, "Y", np.nan, "Y"], dtype=object)])
    X_test = X_test.astype(object)

    ct = build_preprocessor(X_train, needs_scaling=True)
    ct.fit(X_train)
    out = ct.transform(X_test)  # used to raise "could not convert string to float: 'Y'"
    assert np.isfinite(out).all()
