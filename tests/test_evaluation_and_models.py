"""Unit tests for Stage B meta-models, evaluation metrics, and leakage controls.

Verifies:
(a) A perfect selector achieves Regret@1 = 0.0 and Top-1 hit rate = 1.0.
(b) A random selector matches the theoretical expectation of mean regret.
(c) Every selector loads and evaluates on the exact same folds file without leakage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluation import compute_task_regret, evaluate_predictions
from src.metamodels import (
    KNNMetaLearner,
    LGBMLambdaRankSelector,
    LGBMRegressorSelector,
    RandomSelector,
    SingleBestSelector,
)
from src.portfolio import get_all_pipeline_ids
from src.splits import verify_no_parent_leakage


def test_perfect_selector_zero_regret():
    """Assert that a perfect selector has Regret@1 = 0.0 and Top-1 hit rate = 1.0."""
    pids = get_all_pipeline_ids()
    n_tasks = 20

    # Create synthetic performance matrix
    rng = np.random.RandomState(42)
    perf_data = rng.uniform(0.6, 0.95, size=(n_tasks, len(pids)))
    task_ids = [f"task_{i:03d}" for i in range(n_tasks)]
    perf_df = pd.DataFrame(perf_data, index=task_ids, columns=pids)

    # Perfect selector picks the true best pipeline for each task
    perfect_rankings = {}
    perfect_scores_dict = {}
    for tid in task_ids:
        row = perf_df.loc[tid]
        best_pids = row.sort_values(ascending=False).index.tolist()
        perfect_rankings[tid] = best_pids
        perfect_scores_dict[tid] = row.to_dict()

    perfect_scores = pd.DataFrame.from_dict(perfect_scores_dict, orient="index")

    metrics = evaluate_predictions(
        rankings=perfect_rankings,
        scores=perfect_scores,
        perf_matrix=perf_df,
        n_bootstrap=200,
        hit_tolerance_pts=0.5,
    )

    assert metrics["regret_1_mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["regret_3_mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hit_rate"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["regret_vs_k"][0] == pytest.approx(0.0, abs=1e-6)


def test_random_selector_expected_regret():
    """Assert that a random selector converges to the analytical expected regret."""
    pids = get_all_pipeline_ids()
    n_tasks = 30
    M = len(pids)

    rng = np.random.RandomState(42)
    perf_data = rng.uniform(0.5, 0.9, size=(n_tasks, M))
    task_ids = [f"task_{i:03d}" for i in range(n_tasks)]
    perf_df = pd.DataFrame(perf_data, index=task_ids, columns=pids)

    # Theoretical expected regret for uniform random choice of 1 pipeline:
    # E[Regret@1(t)] = 100 * [ max_p a(t, p) - (1/M) * sum_p a(t, p) ]
    theoretical_task_regrets = []
    for tid in task_ids:
        accs = perf_df.loc[tid].values
        r = 100.0 * (np.max(accs) - np.mean(accs))
        theoretical_task_regrets.append(r)
    expected_mean_regret = float(np.mean(theoretical_task_regrets))

    # Monte Carlo over 100 random rankings
    mc_regrets = []
    for sim_seed in range(100):
        selector = RandomSelector(seed=sim_seed)
        rankings, scores = selector.predict(task_ids, meta_df=None)
        res = evaluate_predictions(rankings, scores, perf_df, n_bootstrap=100)
        mc_regrets.append(res["regret_1_mean"])

    simulated_mean_regret = float(np.mean(mc_regrets))

    # Should match analytical expectation within 1.0 percentage point
    assert simulated_mean_regret == pytest.approx(expected_mean_regret, abs=1.0)


def test_every_method_loads_same_folds_file():
    """Confirm data/meta/folds.json exists and every method evaluates on the exact same folds."""
    cfg = load_config("pilot")
    folds_path = Path(cfg["splits"]["file"])
    assert folds_path.exists(), f"Folds file not found at {folds_path}"

    with open(folds_path) as f:
        folds = json.load(f)

    # Assert exactly 5 folds
    unique_folds = set(folds.values())
    assert unique_folds == {0, 1, 2, 3, 4}

    # Verify no parent leakage across any fold
    task_info_path = Path(cfg["paths"]["meta"]) / "task_info.parquet"
    if task_info_path.exists():
        task_info = pd.read_parquet(task_info_path)
        assert verify_no_parent_leakage(folds, task_info, n_folds=5)

    # Verify all 5 selectors can be instantiated and predict on fold 0 test tasks
    meta_path = Path(cfg["paths"]["meta"]) / "meta_features.parquet"
    perf_path = Path(cfg["paths"]["meta"]) / "performance_matrix.parquet"
    gap_path = Path(cfg["paths"]["meta"]) / "gap_targets.parquet"
    rank_path = Path(cfg["paths"]["meta"]) / "lambdarank_labels.parquet"

    if meta_path.exists() and perf_path.exists():
        meta_df = pd.read_parquet(meta_path)
        perf_df = pd.read_parquet(perf_path)
        gap_df = pd.read_parquet(gap_path)
        rank_df = pd.read_parquet(rank_path)

        train_tasks = [t for t, f in folds.items() if f != 0]
        test_tasks = [t for t, f in folds.items() if f == 0]

        selectors = [
            RandomSelector(seed=42),
            SingleBestSelector(),
            KNNMetaLearner(k=5),
            LGBMRegressorSelector(n_estimators=10, seed=42),
            LGBMLambdaRankSelector(n_estimators=10, seed=42),
        ]

        for s in selectors:
            s.fit(train_tasks, meta_df, perf_df, gap_df, rank_df)
            ranks, scores = s.predict(test_tasks, meta_df)
            assert len(ranks) == len(test_tasks)
            assert len(scores) == len(test_tasks)
            for tid in test_tasks:
                assert len(ranks[tid]) == 18

