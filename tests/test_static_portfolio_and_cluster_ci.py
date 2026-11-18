"""Static portfolio greedy order and parent-cluster bootstrap."""

import numpy as np
import pandas as pd

from src.evaluation import cluster_bootstrap_ci
from src.metamodels import StaticPortfolioSelector


def test_static_portfolio_picks_complement_not_second_best():
    # p0 best on average; p1 is a near-copy of p0; p2 is worse on average but wins where p0 loses.
    perf = pd.DataFrame(
        {"p0": [0.90, 0.90, 0.60, 0.60], "p1": [0.89, 0.89, 0.59, 0.59], "p2": [0.50, 0.50, 0.95, 0.95]},
        index=["t0", "t1", "t2", "t3"],
    )
    sel = StaticPortfolioSelector().fit(list(perf.index), None, perf, None, None)
    assert sel.order_[:2] == ["p0", "p2"]
    ranks, scores = sel.predict(["new"], None)
    assert ranks["new"] == sel.order_ and scores.loc["new", "p0"] > scores.loc["new", "p2"]


def test_static_portfolio_uses_only_training_tasks():
    perf = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]}, index=["train", "test"])
    sel = StaticPortfolioSelector().fit(["train"], None, perf, None, None)
    assert sel.order_[0] == "a"


def test_cluster_bootstrap_resamples_parents():
    # Two parents with very different means and 50 identical tasks each:
    # resampling parents gives a much wider CI than the within-parent spread suggests.
    values = np.r_[np.zeros(50), np.ones(50)]
    groups = np.r_[["A"] * 50, ["B"] * 50]
    mean, lo, hi = cluster_bootstrap_ci(values, groups, n_resamples=2000, seed=0)
    assert mean == 0.5 and lo == 0.0 and hi == 1.0
    # One parent -> degenerate CI
    assert cluster_bootstrap_ci(np.ones(5), np.array(["A"] * 5)) == (1.0, 1.0, 1.0)
