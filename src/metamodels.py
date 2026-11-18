"""Meta-models for pipeline selection: kNN, LightGBM Regressor, LightGBM LambdaRank, and baselines."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.portfolio import get_all_pipeline_ids


class BaseSelector:
    """Abstract base class for meta-learning pipeline selectors."""

    def fit(
        self,
        train_task_ids: list[str],
        meta_df: pd.DataFrame,
        perf_matrix: pd.DataFrame,
        gap_df: pd.DataFrame,
        lambdarank_df: pd.DataFrame,
    ) -> "BaseSelector":
        raise NotImplementedError

    def predict(
        self,
        test_task_ids: list[str],
        meta_df: pd.DataFrame,
    ) -> tuple[dict[str, list[str]], pd.DataFrame]:
        """Predict rankings and scores for test tasks.

        Returns
        -------
        rankings : dict[task_id, list[pipeline_id]]
            Pipelines sorted descending by predicted suitability (best to worst).
        scores_df : pd.DataFrame
            DataFrame (test_task_ids x pipeline_ids) of predicted scores/gaps.
        """
        raise NotImplementedError


class RandomSelector(BaseSelector):
    """Random baseline: permutes candidate pipelines uniformly at random."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pipeline_ids = get_all_pipeline_ids()

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        if perf_matrix is not None and not perf_matrix.empty:
            self.pipeline_ids = list(perf_matrix.columns)
        return self

    def predict(self, test_task_ids, meta_df):
        rankings = {}
        scores_dict = {}
        for idx, task_id in enumerate(test_task_ids):
            rng = np.random.RandomState(self.seed + idx)
            perm = list(self.pipeline_ids)
            rng.shuffle(perm)
            rankings[task_id] = perm
            # Scores correspond to reverse rank so rank 1 has highest score
            scores_dict[task_id] = {pid: float(len(perm) - i) for i, pid in enumerate(perm)}

        scores_df = pd.DataFrame.from_dict(scores_dict, orient="index")[self.pipeline_ids]
        return rankings, scores_df


class SingleBestSelector(BaseSelector):
    """Single-best baseline: selects the pipeline with highest mean training accuracy."""

    def __init__(self):
        self.pipeline_ids = get_all_pipeline_ids()
        self.mean_scores_ = None
        self.ranked_pipelines_ = None

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        if perf_matrix is not None and not perf_matrix.empty:
            self.pipeline_ids = list(perf_matrix.columns)
        train_perf = perf_matrix.loc[train_task_ids, self.pipeline_ids]
        self.mean_scores_ = train_perf.mean(axis=0)
        # Sort descending by mean training accuracy
        self.ranked_pipelines_ = self.mean_scores_.sort_values(ascending=False).index.tolist()
        return self

    def predict(self, test_task_ids, meta_df):
        rankings = {t: list(self.ranked_pipelines_) for t in test_task_ids}
        scores_dict = {t: self.mean_scores_.to_dict() for t in test_task_ids}
        scores_df = pd.DataFrame.from_dict(scores_dict, orient="index")[self.pipeline_ids]
        return rankings, scores_df


class KNNMetaLearner(BaseSelector):
    """kNN meta-learner (auto-sklearn style).

    Finds k nearest training tasks in standardised meta-feature space;
    averages their normalised pipeline scores; ranks descending.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.pipeline_ids = get_all_pipeline_ids()
        self.scaler_ = None
        self.nn_ = None
        self.train_task_ids_ = None
        self.norm_scores_ = None

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        if perf_matrix is not None and not perf_matrix.empty:
            self.pipeline_ids = list(perf_matrix.columns)
        self.train_task_ids_ = list(train_task_ids)
        # 1. Standardise meta-features with training-fold statistics only
        self.scaler_ = StandardScaler()
        X_train = self.scaler_.fit_transform(meta_df.loc[train_task_ids].values)

        actual_k = min(self.k, len(train_task_ids))
        self.nn_ = NearestNeighbors(n_neighbors=actual_k, metric="euclidean")
        self.nn_.fit(X_train)

        # 2. Compute normalised pipeline scores per training task in [0, 1]
        train_perf = perf_matrix.loc[train_task_ids]
        norm_scores = pd.DataFrame(index=train_task_ids, columns=self.pipeline_ids, dtype=float)
        for tid in train_task_ids:
            row = train_perf.loc[tid]
            rmin, rmax = row.min(), row.max()
            if rmax > rmin:
                norm_scores.loc[tid] = (row - rmin) / (rmax - rmin)
            else:
                norm_scores.loc[tid] = 0.0
        self.norm_scores_ = norm_scores
        return self

    def predict(self, test_task_ids, meta_df):
        X_test = self.scaler_.transform(meta_df.loc[test_task_ids].values)
        neighbor_indices = self.nn_.kneighbors(X_test, return_distance=False)

        rankings = {}
        scores_dict = {}
        for idx, task_id in enumerate(test_task_ids):
            n_idxs = neighbor_indices[idx]
            neighbor_tasks = [self.train_task_ids_[i] for i in n_idxs]
            # Average normalised scores across the k neighbors
            mean_score = self.norm_scores_.loc[neighbor_tasks].mean(axis=0)
            scores_dict[task_id] = mean_score.to_dict()
            rankings[task_id] = mean_score.sort_values(ascending=False).index.tolist()

        scores_df = pd.DataFrame.from_dict(scores_dict, orient="index")[self.pipeline_ids]
        return rankings, scores_df


def _build_design_matrix(
    task_ids: list[str],
    meta_df: pd.DataFrame,
    pipeline_ids: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Build design matrix: standardised meta-features + one-hot pipeline_id.

    One row per (task, pipeline) pair, ordered by task.
    """
    raw_meta = meta_df.loc[task_ids].values
    if fit_scaler:
        scaler = StandardScaler()
        meta_scaled = scaler.fit_transform(raw_meta)
    else:
        meta_scaled = scaler.transform(raw_meta)

    n_tasks = len(task_ids)
    n_pipes = len(pipeline_ids)
    n_meta = meta_scaled.shape[1]

    # One-hot encoding matrix for pipelines: identity (n_pipes x n_pipes)
    eye_pipes = np.eye(n_pipes)

    # Repeat each task's meta-features n_pipes times
    meta_rep = np.repeat(meta_scaled, n_pipes, axis=0)

    # Tile the pipeline one-hot vectors n_tasks times
    pipes_tile = np.tile(eye_pipes, (n_tasks, 1))

    X_design = np.hstack([meta_rep, pipes_tile])

    meta_feature_names = list(meta_df.columns)
    onehot_feature_names = [f"pipe_{pid}" for pid in pipeline_ids]
    feature_names = meta_feature_names + onehot_feature_names

    return X_design, scaler, feature_names


class LGBMRegressorSelector(BaseSelector):
    """LightGBM regressor predicting gap = acc - max_acc per (task, pipeline)."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.pipeline_ids = get_all_pipeline_ids()
        self.scaler_ = None
        self.model_ = None
        self.feature_names_ = None
        self.feature_importances_ = None

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        if perf_matrix is not None and not perf_matrix.empty:
            self.pipeline_ids = list(perf_matrix.columns)
        X_train, self.scaler_, self.feature_names_ = _build_design_matrix(
            train_task_ids, meta_df, self.pipeline_ids, fit_scaler=True
        )

        # Construct target y_train (gap per task, pipeline)
        y_rows = []
        for tid in train_task_ids:
            for pid in self.pipeline_ids:
                y_rows.append(gap_df.loc[tid, pid])
        y_train = np.array(y_rows, dtype=np.float64)

        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1,
        )
        self.model_.fit(X_train, y_train)
        self.feature_importances_ = self.model_.booster_.feature_importance(importance_type="gain")
        return self

    def predict(self, test_task_ids, meta_df):
        X_test, _, _ = _build_design_matrix(
            test_task_ids, meta_df, self.pipeline_ids, scaler=self.scaler_, fit_scaler=False
        )
        preds = self.model_.predict(X_test)
        n_pipes = len(self.pipeline_ids)

        rankings = {}
        scores_dict = {}
        for idx, task_id in enumerate(test_task_ids):
            task_preds = preds[idx * n_pipes : (idx + 1) * n_pipes]
            task_series = pd.Series(task_preds, index=self.pipeline_ids)
            scores_dict[task_id] = task_series.to_dict()
            rankings[task_id] = task_series.sort_values(ascending=False).index.tolist()

        scores_df = pd.DataFrame.from_dict(scores_dict, orient="index")[self.pipeline_ids]
        return rankings, scores_df


class LGBMLambdaRankSelector(BaseSelector):
    """LightGBM LambdaRank predicting rank scores from per-task relevance labels."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.pipeline_ids = get_all_pipeline_ids()
        self.scaler_ = None
        self.model_ = None
        self.feature_names_ = None
        self.feature_importances_ = None

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        if perf_matrix is not None and not perf_matrix.empty:
            self.pipeline_ids = list(perf_matrix.columns)
        X_train, self.scaler_, self.feature_names_ = _build_design_matrix(
            train_task_ids, meta_df, self.pipeline_ids, fit_scaler=True
        )

        y_rows = []
        for tid in train_task_ids:
            for pid in self.pipeline_ids:
                y_rows.append(int(lambdarank_df.loc[tid, pid]))
        y_train = np.array(y_rows, dtype=np.int32)
        group_train = [len(self.pipeline_ids)] * len(train_task_ids)

        self.model_ = LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1,
        )
        self.model_.fit(X_train, y_train, group=group_train)
        self.feature_importances_ = self.model_.booster_.feature_importance(importance_type="gain")
        return self

    def predict(self, test_task_ids, meta_df):
        X_test, _, _ = _build_design_matrix(
            test_task_ids, meta_df, self.pipeline_ids, scaler=self.scaler_, fit_scaler=False
        )
        preds = self.model_.predict(X_test)
        n_pipes = len(self.pipeline_ids)

        rankings = {}
        scores_dict = {}
        for idx, task_id in enumerate(test_task_ids):
            task_preds = preds[idx * n_pipes : (idx + 1) * n_pipes]
            task_series = pd.Series(task_preds, index=self.pipeline_ids)
            scores_dict[task_id] = task_series.to_dict()
            rankings[task_id] = task_series.sort_values(ascending=False).index.tolist()

        scores_df = pd.DataFrame.from_dict(scores_dict, orient="index")[self.pipeline_ids]
        return rankings, scores_df


class StaticPortfolioSelector(BaseSelector):
    """Greedy static portfolio (train-fold only).

    Step k adds the pipeline that minimises the mean Regret@k over the TRAINING
    tasks of the portfolio built so far, i.e.
        mean_t [ max_p a(t,p) - max_{q in P_k} a(t,q) ].
    The resulting order is used unchanged for every test task. Step 1 picks the
    pipeline with the highest mean training accuracy (same as Single-best);
    later steps pick complements rather than the next-best pipeline on average.
    """

    def __init__(self):
        self.pipeline_ids = get_all_pipeline_ids()
        self.order_ = None

    def fit(self, train_task_ids, meta_df, perf_matrix, gap_df, lambdarank_df):
        self.pipeline_ids = list(perf_matrix.columns)
        A = perf_matrix.loc[train_task_ids, self.pipeline_ids].to_numpy(dtype=float)
        best = A.max(axis=1)
        covered = np.full(A.shape[0], -np.inf)
        remaining = list(range(A.shape[1]))
        order = []
        while remaining:
            # mean regret if candidate j were added: best - max(covered, A[:, j])
            cand = np.maximum(covered[:, None], A[:, remaining])
            mean_regret = (best[:, None] - cand).mean(axis=0)
            j = remaining[int(np.argmin(mean_regret))]  # ties -> earliest column
            order.append(j)
            covered = np.maximum(covered, A[:, j])
            remaining.remove(j)
        self.order_ = [self.pipeline_ids[j] for j in order]
        return self

    def predict(self, test_task_ids, meta_df):
        n = len(self.order_)
        score = {pid: float(n - i) for i, pid in enumerate(self.order_)}
        rankings = {t: list(self.order_) for t in test_task_ids}
        scores_df = pd.DataFrame.from_dict({t: score for t in test_task_ids}, orient="index")[self.pipeline_ids]
        return rankings, scores_df
