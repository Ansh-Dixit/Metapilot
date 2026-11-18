"""Demo core (src/recommend.py): end-to-end run, training consistency, friendly errors."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metafeatures import META_FEATURE_COLS
from src.recommend import (
    MODEL_NAMES,
    MODELS_PATH,
    RecommendError,
    load_bundle,
    meta_features_for,
    prepare_dataset,
    recommend,
    train_final,
)

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks_full"
META = ROOT / "data" / "meta_full" / "meta_features.parquet"
needs_full = pytest.mark.skipif(not META.exists(), reason="full-mode meta-features not built")

# numeric OpenML, mixed numeric+categorical OpenML (credit-g), categorical with missing values (anneal),
# sklearn, synthetic
CONSISTENCY_TASKS = ["openml_1049__s4", "openml_31__s0", "openml_2__s3", "sk_wine__s2", "synth_cls_005__s1"]


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    if MODELS_PATH.exists():
        return load_bundle(MODELS_PATH)
    if not META.exists():
        pytest.skip("full-mode data not available to train the demo models")
    return load_bundle(train_final("full", tmp_path_factory.mktemp("models") / "m.joblib"))


def _iris_like(n_per_class=30, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c, mu in [("setosa", 0.0), ("versicolor", 2.0), ("virginica", 4.0)]:
        for _ in range(n_per_class):
            rows.append({"customer_id": len(rows), "a": rng.randn() + mu, "b": rng.randn(),
                         "colour": rng.choice(["red", "blue"]), "species": c})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (a) end to end
@needs_full
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_recommend_end_to_end_on_small_csv(tmp_path, bundle, model_name):
    p = tmp_path / "sample.csv"
    _iris_like().to_csv(p, index=False)
    from src.recommend import read_csv
    res = recommend(read_csv(p), "species", 3, model_name, bundle=bundle)
    assert [r["rank"] for r in res["top_k"]] == [1, 2, 3]
    assert len(res["full_ranking"]) == 30 and len(set(res["full_ranking"])) == 30
    assert len(res["static_top_k"]) == 3
    assert set(res["meta_features"]) == set(META_FEATURE_COLS)
    assert res["summary"]["classes"] == 3 and res["summary"]["rows"] == 90
    assert any("customer_id" in d for d in res["summary"]["dropped_columns"])  # ID-like column dropped
    assert len(res["explanation"]["reasons"]) == 3
    assert all(r["sentence"].startswith("Your data has") for r in res["explanation"]["reasons"])


# ---------------------------------------------------------------- (b) consistency with training
@needs_full
@pytest.mark.parametrize("task_id", CONSISTENCY_TASKS)
def test_meta_features_match_training(bundle, task_id):
    """Saved task array -> DataFrame -> recommend()'s prepare + meta-feature path == training parquet."""
    with np.load(TASKS / f"{task_id}.npz", allow_pickle=True) as d:
        X, y = d["X"], d["y"]
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    df["target"] = y
    X_raw, y_enc, info = prepare_dataset(df, "target")
    # all-empty columns may be dropped (training's imputer drops them too); nothing else
    assert all(d.endswith("(empty)") for d in info["dropped_columns"])
    got = meta_features_for(X_raw, y_enc, bundle).iloc[0]
    want = pd.read_parquet(META).loc[task_id, META_FEATURE_COLS]
    np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=0, atol=1e-6)


# ---------------------------------------------------------------- (c) friendly errors
def test_one_class_is_rejected_with_friendly_message():
    df = pd.DataFrame({"x": np.arange(100) % 7, "y": ["yes"] * 100})
    with pytest.raises(RecommendError, match="only one value"):
        prepare_dataset(df, "y")


def test_tiny_file_is_rejected():
    df = pd.DataFrame({"x": np.arange(20) % 5, "y": ["a", "b"] * 10})
    with pytest.raises(RecommendError, match="at least 50"):
        prepare_dataset(df, "y")


def test_rare_class_is_rejected():
    df = pd.DataFrame({"x": np.arange(100) % 9, "y": ["a"] * 60 + ["b"] * 37 + ["c"] * 3})
    with pytest.raises(RecommendError, match="at least 6 rows"):
        prepare_dataset(df, "y")


def test_continuous_target_warns_and_unknown_column_errors():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.randn(300) % 3, "price": np.repeat(np.arange(30), 10).astype(float)})
    _, _, info = prepare_dataset(df, "price")
    assert any("looks like a quantity" in w for w in info["warnings"])
    with pytest.raises(RecommendError, match="not in the file"):
        prepare_dataset(df, "nope")


def test_all_nan_column_dropped_and_file_size_limit(tmp_path, monkeypatch):
    import src.recommend as r
    df = pd.DataFrame({"x": np.arange(60) % 4, "empty": np.nan, "y": ["a", "b", "c"] * 20})
    _, _, info = prepare_dataset(df, "y")
    assert info["dropped_columns"] == ["empty (empty)"]
    p = tmp_path / "f.csv"
    df.to_csv(p, index=False)
    monkeypatch.setattr(r, "MAX_FILE_MB", 0.000001)
    with pytest.raises(RecommendError, match="limit"):
        r.read_csv(p)
