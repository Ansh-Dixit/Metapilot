# MetaPilot Progress Log

## Stage A: Milestones 1, 2, and 3 (--pilot mode)

### Date: 2026-11-02

---

### Milestone 1: Scaffold & Verification
- **Repository Structure**:
  - `configs/pilot.yaml` with global seed 42 and pilot configuration.
  - `Makefile` with targets: `setup`, `test`, `pilot-data`, `pilot-portfolio`, `pilot-meta`, `pilot-all`.
  - `README.md`, `requirements.txt`.
  - Python 3.11 virtual environment configured at `.venv/`.
  - `tests/test_splits.py`: asserts GroupKFold by `parent_id` has zero parent overlap between train and test across all folds.
  - `tests/test_preprocessing_leakage.py`: asserts that SimpleImputer (median) and StandardScaler statistics are computed strictly on training folds inside the Pipeline, with no leakage from test folds.
- **Acceptance Status**: PASS (5/5 pytest tests passing).

---

### Milestone 2: Pilot Data & Portfolio Evaluation (Post-Diversification)
- **Data Generation**:
  - 30 parent datasets generated in `data/raw/`:
    - 4 real sklearn datasets: `sk_iris`, `sk_wine`, `sk_breast_cancer`, `sk_digits`
    - 20 diversified synthetic classification datasets: `synth_cls_000` through `synth_cls_019`
    - 6 shape datasets: 3 moons (`moons_000`–`moons_002`), 3 circles (`circles_000`–`circles_002`)
  - 4 subsamples per parent created via stratified sampling in `data/tasks/` (total 120 tasks):
    - Subsample 0 represents full parent data (`row_frac=1.0, col_frac=1.0`).
    - Subsamples 1–3 use random row fractions in [0.3, 1.0] and column fractions in [0.5, 1.0].
    - All tasks verified to exceed `min_class_count: 6`.
- **Synthetic Data Diversification**:
  - Resolved the initial SVM dominance (previously 51.94%) by diversifying the 20 synthetic parents into structured functional archetypes within `configs/pilot.yaml` parameter ranges:
    1. *Axis-aligned decision logic* (`synth_cls_000`–`003`): sharp rectangular splits with noise features (suits `decision_tree`, `random_forest`).
    2. *Checkerboard / XOR grid* (`synth_cls_004`–`005`): non-linear parity logic with uninformative features (suits `random_forest`, `extra_trees`).
    3. *Conditionally independent Gaussians* (`synth_cls_006`–`007`): Bayes-optimal structure for `naive_bayes` (`gnb`).
    4. *High-noise linear hyperplanes & correlated features* (`synth_cls_008`–`011`): linear boundaries with 50-80 noise features and correlation (suits `logreg`, `ridge`).
    5. *Friedman additive interactions* (`synth_cls_012`–`013`): continuous non-linear additive interactions (suits `grad_boosting`).
    6. *Varied multi-cluster make_classification* (`synth_cls_014`–`019`): varied `n_clusters_per_class` (1 to 4), `hypercube` (True/False), `flip_y` (0.01 to 0.12), and `class_sep` (0.7 to 1.6) (suits `knn`, `svm`, `extra_trees`).
- **Portfolio Benchmarking**:
  - Exactly 18 classification pipelines across 10 algorithm families evaluated with 3-fold Stratified CV:
    - 120 tasks × 18 pipelines = 2,160 task-pipeline pairs = 6,480 total fits.
    - Capped at 800 rows per task using stratified downsampling.
    - Zero fit failures recorded across all 6,480 fits.
    - Resumable JSON cache persisted in `data/meta/cache/`.
    - Artifacts saved:
      - `data/meta/fit_log.parquet` (6,480 rows)
      - `data/meta/performance_matrix.parquet` (120 tasks × 18 pipelines)
      - `data/meta/task_info.parquet` (120 rows)
      - `data/meta/folds.json` (5-fold GroupKFold splits, 24 tasks per fold)
- **Winner Distribution & Tie-Splitting Check (`src/evaluation.py`)**:
  - Implemented strictly per SPEC.md Section 12: grouped by algorithm FAMILY, with ties split equally among tied pipelines (no `idxmax`).
  - **Results**:
    - `svm`: 18.33 wins (15.28%)
    - `decision_tree`: 17.17 wins (14.31%)
    - `random_forest`: 16.83 wins (14.03%)
    - `grad_boosting`: 15.17 wins (12.64%)
    - `logreg`: 12.83 wins (10.69%)
    - `extra_trees`: 10.33 wins (8.61%)
    - `knn`: 9.67 wins (8.06%)
    - `mlp`: 8.50 wins (7.08%)
    - `naive_bayes`: 7.00 wins (5.83%)
    - `ridge`: 4.17 wins (3.47%)
  - **Status**:
    - **Target achieved**: Max family is `svm` at 15.28% (well below the 40% cap; no dominance warning).
    - **Full family representation**: **10 of 10 families winning** (all families have non-zero wins).
    - Visualization saved to `results/winner_distribution.png`.

---

### Milestone 3: Meta-Features & Text Metadata
- **Numeric Meta-Features (14 total)**:
  - Simple (4): `log_n_rows`, `log_n_cols`, `cols_to_rows`, `n_classes`
  - Statistical (4): `mean_abs_corr`, `mean_abs_skew`, `mean_kurtosis`, `pca95_frac`
  - Info-Theoretic (2): `norm_class_entropy`, `minority_majority_ratio`
  - Landmarkers (4): `lm_stump`, `lm_1nn`, `lm_gnb`, `lm_lda` (2-fold CV on ≤200 rows)
  - Output: `data/meta/meta_features.parquet` (120 rows × 14 columns).
  - Validation: 0 NaN, 0 inf across all values.
- **Text Metadata**:
  - Saved to `data/meta/text_metadata.parquet` (120 rows): `task_id`, `column_names`, `dtypes`, `missing_pct`, `description`, `n_rows`, `n_cols`, `n_classes`, `minority_pct`, `parent_id`, `source`.
- **Targets**:
  - `data/meta/gap_targets.parquet` (120 × 18)
  - `data/meta/lambdarank_labels.parquet` (120 × 18)

---

## Stage B: Milestone 4 (Meta-Models & Evaluation)

> [!NOTE]
> **Framework Validation Only**: Pilot results reported in Stage B come predominantly from an engineered synthetic mix designed to test the computational framework and benchmark evaluation mechanics. These pilot figures **must not be interpreted as scientific findings** or generalizable comparative advantages of any algorithm.

### Date: 2026-11-02

---

### Milestone 4: Meta-Models, Evaluation, and Ablation
- **Selectors Implemented (`src/metamodels.py`)**:
  - `RandomSelector`: uniform random pipeline ranking.
  - `SingleBestSelector`: ranks pipelines by average training accuracy across training tasks.
  - `KNNMetaLearner`: finds $k=5$ nearest training tasks in standardized meta-feature space; averages normalized training scores in $[0, 1]$; ranks descending.
  - `LGBMRegressorSelector`: 300 trees, lr 0.05, 15 leaves; predicts target $gap(t, p) = acc(t, p) - \max_{p'} acc(t, p')$ on design matrix (standardized meta-features + one-hot pipeline IDs).
  - `LGBMLambdaRankSelector`: same design matrix with $group=18$ per task; predicts ranking scores from integer relevance labels $0 \dots 10$.
  - All standardisation preprocessors fitted strictly on training folds per CV split (no test data leakage).
- **Leakage Control Verification**:
  - Every method loads `data/meta/folds.json` (5-fold GroupKFold by `parent_id`, seed 42).
  - Evaluated via strict 5-fold cross-validation so all 120 tasks are evaluated out-of-fold.
- **Evaluation Metrics (`src/evaluation.py`)**:
  - `Regret@1` and `Regret@3` in percentage points (multiplied by 100).
  - `Top-1 hit rate` (within 0.5 points of best).
  - `Spearman` rank correlation between predicted pipeline scores and true accuracies.
  - 2,000-resample bootstrap 95% confidence intervals over tasks.
  - Regret-vs-$k$ curves computed for all budgets $k \in \{1, \dots, 18\}$.
- **Overall Main Results (`results/pilot_results.csv`)**:
  | Selector | Regret@1 (CI) | Regret@3 (CI) | Top-1 hit | Spearman |
  |---|---|---|---|---|
  | Random | 8.72 (7.01, 10.52) | 3.21 (2.59, 3.87) | 0.092 | 0.011 |
  | Single-best | 4.06 (3.42, 4.79) | 2.63 (2.09, 3.18) | 0.150 | 0.373 |
  | kNN | 4.26 (3.17, 5.62) | 1.45 (0.96, 2.01) | 0.192 | 0.524 |
  | LightGBM Regressor | 4.06 (3.27, 4.95) | 2.15 (1.62, 2.69) | 0.233 | 0.470 |
  | **LightGBM LambdaRank** | **3.19 (2.55, 3.96)** | **1.49 (1.13, 1.85)** | **0.267** | **0.445** |
- **Subgroup Analysis by Source (`results/pilot_results_by_source.csv`)**:
  - Real tasks (sklearn, $N=16$): `Single-best` achieves 1.43 Regret@1, `kNN` achieves 1.41 Regret@1.
  - Synthetic tasks ($N=104$): `LightGBM LambdaRank` dominates with 3.42 Regret@1 vs 4.46 for Single-best.
- **Landmarker Ablation (`results/pilot_results_ablation.csv`)**:
  - When the 4 landmarkers (`lm_stump`, `lm_1nn`, `lm_gnb`, `lm_lda`) are excluded, Regret@1 for `LightGBM LambdaRank` increases from 3.19 to 3.44 points, confirming landmarkers provide a measurable positive performance signal.
- **Generated Figures in `results/`**:
  - `results/regret_at_1_ci.png`: Bar chart of Regret@1 with 95% bootstrap error bars.
  - `results/regret_vs_k.png`: Regret vs. budget $k$ curves ($k=1 \dots 18$) across all selectors.
  - `results/lgbm_feature_importance.png`: Top 10 meta-features ranked by LightGBM gain share (led by `norm_class_entropy`, `pca95_frac`, `lm_1nn`, and `mean_abs_corr`).
- **Tests Added & Verified (`tests/test_evaluation_and_models.py`)**:
  - `test_perfect_selector_zero_regret`: Oracle selector achieves Regret@1 = 0.0 and Top-1 hit rate = 1.0.
  - `test_random_selector_expected_regret`: Random selector converges to theoretical expected regret $\frac{1}{N} \sum_t (\max_p a(t, p) - \text{mean}_p a(t, p))$.
  - `test_every_method_loads_same_folds_file`: Every selector loads `data/meta/folds.json`, with identical 5-fold partition sets and zero parent leakage.
  - Total test suite: 8/8 tests passing.

---

### Assumptions Made

1. **LightGBM OpenMP Library Loading**:
   - On macOS, LightGBM requires `libomp.dylib`. Configured `@rpath` on `lib_lightgbm.dylib` to dynamically bind to the installed OpenMP runtime inside the virtual environment without external system dependencies.

2. **kNN Normalization**:
   - Per auto-sklearn standard practices, training pipeline accuracies are min-max rescaled to $[0, 1]$ per task before neighbor averaging, preventing tasks with different baseline difficulties from skewing neighbor votes.

3. **Random Baseline Expectation**:
   - The Random baseline ranking is evaluated across reproducible per-task seeds, and its analytical expectation $\max_p a(t, p) - \text{mean}_p a(t, p)$ was derived and validated in unit tests.

4. **Tie-Splitting in True Accuracies**:
   - In rare tasks where multiple pipelines share the top accuracy, achieving any of the tied best pipelines yields 0.0 regret.

---

## Stage C, Part 1: Milestone 5 (Build and Dry Run)

### Date: 2026-11-02

---

### 1. OpenML Loading Results & Parent Selection Methodology

The parent dataset pool consists of **195 datasets** (161 OpenML, 4 scikit-learn real, and 30 diversified synthetic parents). All parent datasets are registered with schema metadata in `data/raw/parents.csv` (`dataset_id, parent_id, name, rows, columns, classes, source, selection_rule`).

#### OpenML Parent Dataset Selection Breakdown (161 Datasets)

1. **OpenML-CC18 Benchmark Suite (Suite 99)** (71 datasets):
   - 71 curated tabular classification benchmarks from OpenML-CC18 loaded successfully (e.g., `pc4`, `kc1`, `mfeat-factors`, `letter`, `kr-vs-kp`, `phoneme`, etc.).
   - Only 1 CC18 dataset was excluded: `CIFAR_10` (did=40927) due to exceeding the 50,000 row cap (60,000 32x32 image instances).
   - *(Count correction, Stage C Part 2: the 71 CC18 parents include `madelon` (1485) and `Internet-Advertisements` (40978); both were previously also listed under items 2 and 3 below, which made those lists look longer than their counts. Recomputed from `data/raw/parents.csv`: CC18 71 + named 7 + high-dim 9 + query 74 = 161 OpenML.)*
2. **SPEC.md Section 6 Named Benchmarks** (7 datasets; `madelon` is also a named SPEC dataset but is counted under CC18):
   - `arcene` (did=1458, 200 rows × 10,000 features, 2 classes) — mass spectrometry high-dimensional cancer benchmark.
   - `madelon` (did=1485, 2,600 rows × 500 features, 2 classes) — *counted in CC18 above, listed here for reference*.
   - `gina` (did=41158, 3,153 rows × 970 features, 2 classes) — handwritten digit pairs from agnostic learning challenge.
   - `sonar` (did=40, 208 rows × 60 features, 2 classes) — radar signal discrimination.
   - `ionosphere` (did=59, 351 rows × 34 features, 2 classes) — radar returns from the ionosphere.
   - `heart-statlog` (did=53, 270 rows × 13 features, 2 classes) — clinical heart disease diagnosis.
   - `hepatitis` (did=55, 155 rows × 19 features, 2 classes) — clinical survival prediction.
   - `haberman` (did=43, 306 rows × 3 features, 2 classes) — breast cancer surgery survival.
3. **High-Dimensional & Microarray Datasets (features > rows or features > 1,000)** (9 datasets):
   - Selected to assess meta-learners under the $p \gg n$ regime:
   - `Internet-Advertisements` (did=40978, 3,279 rows × 1,558 features, 2 classes) — *counted in CC18, listed here for reference; the 9 counted high-dim parents are the remaining entries*
   - `dbworld-bodies` (did=1562, 64 rows × 4,702 features, 2 classes)
   - `garber-2001` (did=46779, 66 rows × 4,553 features, 4 classes)
   - `Lymphoma-9` (did=45095, 96 rows × 4,026 features, 9 classes)
   - `lymphoma_2classes` (did=1101, 45 rows × 4,026 features, 2 classes)
   - `lymphoma_9classes` (did=1102, 96 rows × 4,026 features, 9 classes)
   - `Lymphoma-3` (did=45094, 66 rows × 4,026 features, 3 classes)
   - `dbworld-bodies-stemmed` (did=1561, 64 rows × 3,721 features, 2 classes)
   - `SRBCT` (did=45101, 83 rows × 2,308 features, 4 classes)
   - `bittner-2000` (did=46776, 38 rows × 2,201 features, 2 classes)
4. **Standard Tabular Classification Query** (74 datasets):
   - Retrieved from OpenML active classification catalog matching:
     - `number_classes: 2..10`
     - `number_instances: 100..20000`
     - `number_features: 4..200`
   - Spanning varied domains (biomedical, physics, finance, software engineering, social science).

#### Summary Parent Distribution (`data/raw/parents.csv`)

| Source | Count | Share |
|---|---|---|
| openml | 161 | 82.6% |
| synthetic | 30 | 15.4% |
| sklearn | 4 | 2.1% |
| **TOTAL** | **195** | **100.0%** |

Synthetic share (15.4%) remains strictly below the 20% cap. All dataset metadata is saved to `data/raw/parents.csv`.

---

### 2. Extended Portfolio: 42 Pipelines Across 10 Families

The pilot portfolio of 18 pipelines was extended to 42 pipelines (3 to 6 configs per family, across all 10 families), preserving all 18 pilot pipeline identifiers and configurations unchanged. TabPFN is implemented behind the optional flag `use_tabpfn: false` with graceful fallback.

| Pipeline ID | Family | Model & Hyperparameters | Scaled | Status |
|---|---|---|---|---|
| `logreg_c0.1` | `logreg` | `LogisticRegression(C=0.1, max_iter=1000)` | Yes | Pilot (Original) |
| `logreg_c10` | `logreg` | `LogisticRegression(C=10.0, max_iter=1000)` | Yes | Pilot (Original) |
| `logreg_c1` | `logreg` | `LogisticRegression(C=1.0, max_iter=1000)` | Yes | Stage C Extended |
| `logreg_l1_c1` | `logreg` | `LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=1000)` | Yes | Stage C Extended |
| `ridge` | `ridge` | `RidgeClassifier()` | Yes | Pilot (Original) |
| `ridge_alpha0.1` | `ridge` | `RidgeClassifier(alpha=0.1)` | Yes | Stage C Extended |
| `ridge_alpha10` | `ridge` | `RidgeClassifier(alpha=10.0)` | Yes | Stage C Extended |
| `gnb` | `naive_bayes` | `GaussianNB()` | No | Pilot (Original) |
| `gnb_var1e_7` | `naive_bayes` | `GaussianNB(var_smoothing=1e-7)` | No | Stage C Extended |
| `gnb_var1e_5` | `naive_bayes` | `GaussianNB(var_smoothing=1e-5)` | No | Stage C Extended |
| `knn_1` | `knn` | `KNeighborsClassifier(n_neighbors=1)` | Yes | Pilot (Original) |
| `knn_3` | `knn` | `KNeighborsClassifier(n_neighbors=3)` | Yes | Stage C Extended |
| `knn_5` | `knn` | `KNeighborsClassifier(n_neighbors=5)` | Yes | Pilot (Original) |
| `knn_7_dist` | `knn` | `KNeighborsClassifier(n_neighbors=7, weights='distance')` | Yes | Stage C Extended |
| `knn_15` | `knn` | `KNeighborsClassifier(n_neighbors=15)` | Yes | Pilot (Original) |
| `dt_d3` | `decision_tree` | `DecisionTreeClassifier(max_depth=3)` | No | Pilot (Original) |
| `dt_d5` | `decision_tree` | `DecisionTreeClassifier(max_depth=5)` | No | Stage C Extended |
| `dt_d10` | `decision_tree` | `DecisionTreeClassifier(max_depth=10)` | No | Stage C Extended |
| `dt_full` | `decision_tree` | `DecisionTreeClassifier()` | No | Pilot (Original) |
| `rf_100` | `random_forest` | `RandomForestClassifier(n_estimators=100)` | No | Pilot (Original) |
| `rf_100_d5` | `random_forest` | `RandomForestClassifier(n_estimators=100, max_depth=5)` | No | Pilot (Original) |
| `rf_100_entropy` | `random_forest` | `RandomForestClassifier(n_estimators=100, criterion='entropy')` | No | Stage C Extended |
| `rf_200` | `random_forest` | `RandomForestClassifier(n_estimators=200)` | No | Stage C Extended |
| `rf_200_d10` | `random_forest` | `RandomForestClassifier(n_estimators=200, max_depth=10)` | No | Stage C Extended |
| `et_100` | `extra_trees` | `ExtraTreesClassifier(n_estimators=100)` | No | Pilot (Original) |
| `et_100_d5` | `extra_trees` | `ExtraTreesClassifier(n_estimators=100, max_depth=5)` | No | Stage C Extended |
| `et_100_entropy` | `extra_trees` | `ExtraTreesClassifier(n_estimators=100, criterion='entropy')` | No | Stage C Extended |
| `et_200` | `extra_trees` | `ExtraTreesClassifier(n_estimators=200)` | No | Stage C Extended |
| `hgb_default` | `grad_boosting` | `HistGradientBoostingClassifier()` | No | Pilot (Original) |
| `hgb_shallow_lr05` | `grad_boosting` | `HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05)` | No | Pilot (Original) |
| `hgb_lr01` | `grad_boosting` | `HistGradientBoostingClassifier(learning_rate=0.01)` | No | Stage C Extended |
| `hgb_deep_d6` | `grad_boosting` | `HistGradientBoostingClassifier(max_depth=6)` | No | Stage C Extended |
| `hgb_l2_1` | `grad_boosting` | `HistGradientBoostingClassifier(l2_regularization=1.0)` | No | Stage C Extended |
| `svm_c0.1` | `svm` | `SVC(C=0.1, kernel='rbf')` | Yes | Stage C Extended |
| `svm_c1` | `svm` | `SVC(C=1.0, kernel='rbf')` | Yes | Pilot (Original) |
| `svm_c10` | `svm` | `SVC(C=10.0, kernel='rbf')` | Yes | Pilot (Original) |
| `svm_linear_c1` | `svm` | `SVC(C=1.0, kernel='linear')` | Yes | Stage C Extended |
| `svm_poly_d3` | `svm` | `SVC(C=1.0, kernel='poly', degree=3)` | Yes | Stage C Extended |
| `mlp_32` | `mlp` | `MLPClassifier(hidden_layer_sizes=(32,), max_iter=300)` | Yes | Pilot (Original) |
| `mlp_64_32` | `mlp` | `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)` | Yes | Stage C Extended |
| `mlp_128` | `mlp` | `MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)` | Yes | Pilot (Original) |
| `mlp_alpha01` | `mlp` | `MLPClassifier(hidden_layer_sizes=(64,), alpha=0.1, max_iter=300)` | Yes | Stage C Extended |
| `tabpfn` | `tabpfn` | `TabPFNClassifier(device='cpu')` (optional flag `use_tabpfn: false`) | No | Optional / Fallback |

---

### 3. Scale Features and Infrastructure

- **Subsampling**: 10 subsamples per parent (full mode), with `row_frac` in [0.3, 1.0] and `col_frac` in [0.5, 1.0].
- **Row cap**: 2,000 rows per task (stratified downsampling).
- **Per-fit timeout & Wide Datasets**: Standard per-fit timeout is 180s (`cv.fit_timeout_sec`). For datasets with >1,000 features, `fit_timeout_wide_sec` allows up to 600s, and timed-out tasks are logged to `results/timeout_log.csv`.
- **Live Progress Tracking**: Atomically writes `results/run_progress.json` every 100 completed fits (reporting `completed_fits`, `total_fits`, `pct_complete`, `fits_per_sec`, `eta_sec`, `eta_min`, `failures`, `timeouts`).
- **n_jobs**: Configurable via config (default `-1` = all cores; `os.cpu_count() = 10` on host machine).
- **Caching**: Per `(task, pipeline)` JSON cache in `data/meta/cache/` for full resumability.
- **Sharding**: `--shard i/N` support in `run_portfolio.py` (e.g., `--shard 0/4` runs tasks at indices 0, 4, 8, ...).
- **Merge script**: `src/merge_shards.py` reads all cached JSON files and rebuilds `fit_log.parquet`, `performance_matrix.parquet`, `task_info.parquet`.

### 4. Evaluation Upgrades

- **Paired bootstrap CIs**: For Δ in regret between each meta-model and Single-best (same 2,000 resamples).
- **Multi-seed evaluation**: Seeds `[42, 43, 44]` with mean ± std across seeds.
- **Report by source**: Sliced by `{openml, sklearn, synthetic}` automatically.

### 5. Representative Dry Run Results

Executed across **8 representative parents** selected to capture all extreme dataset characteristics:
1. `openml_23517` (`numerai28.6`): **Largest by rows** (96,320 rows, 21 cols, 2 classes)
2. `openml_1458` (`arcene`): **Widest by columns** (200 rows, 10,000 cols, 2 classes, features > rows)
3. `openml_1049` (`pc4`): **Mid-size tabular benchmark** (1,458 rows, 37 cols, 2 classes)
4. `openml_12` (`mfeat-factors`): **Multiclass benchmark** (2,000 rows, 216 cols, 10 classes)
5. `openml_1562` (`dbworld-bodies`): **Microarray / text-body** (64 rows, 4,702 cols, 2 classes)
6. `sk_breast_cancer`: **Scikit-learn real benchmark** (569 rows, 30 cols, 2 classes)
7. `synth_cls_000`: **Synthetic classification** (axis-aligned rule thresholding)
8. `moons_000`: **Synthetic geometric shape** (non-linear interleaved moons)

- **Total Tasks**: 8 parents × 10 subsamples = 80 tasks.
- **Total Fits**: 80 tasks × 42 pipelines × 3 folds = **10,080 fits**.
- **Wall Time**: **301.4 seconds** (~5.0 minutes) executing in parallel across all 10 cores (`os.cpu_count() = 10`).

#### Timing & Cost Breakdown by Family

| Family | Configs | Total Fits | Sec/Fit | Extrap Total (h) | Cost Share (%) |
|---|---|---|---|---|---|
| `grad_boosting` | 5 | 29,250 | 0.9315s | 7.57h | 52.2% |
| `logreg` | 4 | 23,400 | 0.3262s | 2.12h | 14.6% |
| `mlp` | 4 | 23,400 | 0.2832s | 1.84h | 12.7% |
| `random_forest` | 5 | 29,250 | 0.1848s | 1.50h | 10.4% |
| `extra_trees` | 4 | 23,400 | 0.1235s | 0.80h | 5.5% |
| `svm` | 5 | 29,250 | 0.0256s | 0.21h | 1.4% |
| `decision_tree` | 4 | 23,400 | 0.0267s | 0.17h | 1.2% |
| `knn` | 5 | 29,250 | 0.0179s | 0.15h | 1.0% |
| `naive_bayes` | 3 | 17,550 | 0.0169s | 0.08h | 0.6% |
| `ridge` | 3 | 17,550 | 0.0128s | 0.06h | 0.4% |

#### Top 5 Slowest (Parent, Pipeline) Pairs

| Parent | Pipeline | Family | Mean Sec/Fit | Max Sec/Fit | Fits | Bottleneck Mechanism |
|---|---|---|---|---|---|---|
| `openml_12` | `hgb_lr01` | `grad_boosting` | 6.0846s | 8.1471s | 30 | 10 classes requires fitting 10 separate trees per iteration |
| `openml_12` | `logreg_l1_c1` | `logreg` | 4.5533s | 6.3882s | 30 | Multinomial SAGA L1 solver across 10 classes and 216 features |
| `openml_1458` | `logreg_l1_c1` | `logreg` | 4.3864s | 6.6223s | 30 | SAGA L1 coordinate descent across 10,000 features (`arcene`) |
| `openml_12` | `hgb_default` | `grad_boosting` | 4.2294s | 7.1139s | 30 | 10-class HistGradientBoosting default tree building |
| `openml_1458` | `hgb_deep_d6` | `grad_boosting` | 4.0553s | 9.7135s | 30 | Deep trees (depth=6) evaluated on 10,000 features |

#### Full Study Extrapolation (195 parents × 10 subsamples = 1,950 tasks × 42 pipelines × 3 folds = 245,700 fits)

| Architecture | Cores (`os.cpu_count()`) | Estimated Full Runtime | Notes |
|---|---|---|---|
| Single Core | 1 (`n_jobs=1`) | **14.51 hours** (870.4 min) | Sequential single-process execution |
| Host Machine | **10 cores** (`n_jobs=10`) | **1.71 hours** (102.4 min) | Parallel joblib worker pool with ~85% parallel efficiency |

---

### 6. Assumptions (Stage C Part 1)

1. **OpenML size limits**: Maximum row cap is 50,000 instances; maximum feature limit raised to 5,000 (with `arcene` at 10,000 features explicitly accommodated).
2. **High-dimensional timeouts**: Tasks with >1,000 features receive up to 600s timeout per fit; standard tasks receive 180s timeout.
3. **Parent metadata registry**: All 195 parent datasets cataloged in `data/raw/parents.csv` with explicit selection rule provenance.
4. **Synthetic cap**: 30 synthetic parents out of 195 total = 15.4%, strictly under the 20% limit.
5. **Progress monitoring**: Atomically written to `results/run_progress.json` every 100 completed fits.
6. **Pilot IDs preserved**: All 18 original pilot pipeline IDs and hyperparameter configurations remain unchanged.
7. **TabPFN**: Disabled by default (`use_tabpfn: false`). Implemented behind graceful `ImportError` fallback.


---

## Stage C, Part 2: Milestone 5 (full run) and Milestone 6 (Phase 1 full results)

### Date: 2026-11-16

### Deliberate deviations from docs/SPEC.md (budget reasons, requested by the user)

| Item | SPEC / Stage C Part 1 | Used in the full run |
|---|---|---|
| Subsamples per parent | 10 | **6** (`augmentation.n_subsamples`) |
| Pipelines | ~40 (42 implemented) | **30** = 18 pilot + 12 extended, max 3 per family |
| Row cap | 2,000 | **1,000** |
| Per-fit timeout | 180 s (600 s wide) | **30 s** (also for wide data) |
| OpenML feature limit | 11,000 | **2,000**: wider parents skipped and logged (arcene excluded) |
| FLAML / TabPFN | "if available" | Not run |

Unchanged: 3-fold stratified CV, 5-fold GroupKFold by parent, 3 evaluation seeds, 2,000 bootstrap resamples, by-source tables, landmarker ablation.

**Final 30 pipelines** (`src/portfolio.py`):
- logreg: `logreg_c0.1`, `logreg_c10`, `logreg_c1`
- ridge: `ridge`, `ridge_alpha0.1`, `ridge_alpha10`
- naive_bayes: `gnb`, `gnb_var1e_7`, `gnb_var1e_5`
- knn: `knn_1`, `knn_5`, `knn_15`
- decision_tree: `dt_d3`, `dt_full`, `dt_d10`
- random_forest: `rf_100`, `rf_100_d5`, `rf_100_entropy`
- extra_trees: `et_100`, `et_100_d5`, `et_100_entropy`
- grad_boosting: `hgb_default`, `hgb_shallow_lr05`, `hgb_l2_1`
- svm: `svm_c1`, `svm_c10`, `svm_c0.1`
- mlp: `mlp_32`, `mlp_128`, `mlp_alpha01`

Removed from the registry: `logreg_l1_c1`, `knn_3`, `knn_7_dist`, `dt_d5`, `rf_200`, `rf_200_d10`, `et_200`, `hgb_lr01`, `hgb_deep_d6`, `svm_linear_c1`, `svm_poly_d3`, `mlp_64_32`. The extra MLP and RF configs were cut first because they cost the most.

### Isolation and infrastructure fixes
- **Separate paths:**
  - Full mode uses `data/tasks_full`, `data/meta_full` (cache in `data/meta_full/cache`) and `results_full`.
  - The pilot artifacts were backed up to `data/meta_pilot`, `data/tasks_pilot` and `results_pilot`.
  - `scripts/dry_run.py` now writes `results_dry_run/run_progress.json`; the old dry-run progress file was moved there.
- **Mode-scoped parents:** `src/augmentation.py:select_parent_files` picks parents from the active config instead of globbing all of `data/raw`. Pilot gives 30 parents; full gives 185, after excluding 10 OpenML parents with more than 2,000 columns (listed in `data/meta_full/excluded_parents.csv`).
- **Sharding and progress:**
  - `--shard i/N` is strictly 0-based. The old parser mapped both `0/N` and `1/N` to shard 0.
  - Shards no longer write partial parquet files.
  - Each shard writes `results_full/progress/shard_i.json`. `results_full/run_progress.json` aggregates them, counting all cached pairs.
- **Parallelism:** 9 single-threaded shards (10 cores - 1) with `OMP/OPENBLAS/VECLIB_NUM_THREADS=1` (no oversubscription), run under `caffeinate -i`. New Makefile targets: `full-tasks`, `full-portfolio`, `full-merge`, `full-meta`, `full-eval`, `full-all`.
- **`load_failures.csv` regenerated** (`scripts/regen_load_failures.py`). It now lists only the 5 OpenML datasets that actually fail under the full-mode limits: CIFAR_10 (rows > 50,000) and oh0/oh5/oh10/oh15.wc (more than 3,000 features each). Name and error are in separate columns. 40978 and 1458 are loaded, so they are not listed. The old file is kept as `data/raw/load_failures.old.csv`.
- **Loader bug:** `src/data_loading.py` now calls `pd.factorize(np.asarray(y))`; list-typed targets caused the old error for dids 386/392/394/401. Failure logging can no longer write the error into the name column.
- **Preprocessing bug:** `_cast_to_float64`, which runs only in the numeric pipeline, now coerces unparseable strings to NaN. A sparse categorical column that is all-NaN in a training fold gets typed numeric, and then 'Y' appeared at test time (OpenML anneal, did=2).
  - It caused 120 failed pairs before the fix. Those cache entries were deleted and recomputed, and all now succeed.
  - Covered by `tests/test_preprocessing_categorical.py`, which also checks that string columns reach the one-hot and frequency encoders intact.
- **Meta-feature extraction:** the landmarker subsample falls back to a non-stratified draw when stratification is impossible (singleton classes, or a held-out part smaller than the number of classes).
- **Skip rule** (`src/run_portfolio.py:task_skip_reason`, shared with `merge_shards`):
  - A task is skipped when the stratified 1,000-row cap is impossible, or when a class has fewer than 3 members (the number of CV folds) after the cap. How tasks are capped is unchanged.
  - Skipped tasks are logged to `results_full/skipped_tasks.csv`. Per-task exceptions go to `results_full/task_errors.csv` (none occurred), so a task can never crash a whole shard.
  - `merge_shards` re-applies the rule to every task, so tasks finished before the rule existed are treated the same way (their cache is kept). It also drops tasks with incomplete pipeline coverage (none).
  - Covered by `tests/test_skip_small_class.py`.

### Milestone 5: Full portfolio run (DONE)
- **Tasks:**
  - 185 parents × 6 subsamples, minus 64 subsamples dropped for a class under 6 rows, gives 1,046 tasks.
  - **6 tasks skipped**: `openml_10__s0`, `openml_186__s0`, `openml_39__s0`, `openml_449__s0`, `openml_57__s0` (a class with 1–2 members after the cap) and `openml_488__s0` (singleton class, so the stratified cap is impossible; this crashed shard 4 before the fix).
  - Final matrix: **1,040 tasks × 30 pipelines** from **179 parents**. By source: openml 836 tasks / 145 parents, synthetic 180 / 30, sklearn 24 / 4. Synthetic is 16.8% of parents.
- **Fits:**
  - 93,600 fit rows (1,040 × 30 × 3), with 0 failures after the anneal fix.
  - **17 timed-out pairs (51 fold rows, 0.05%)**, all on `openml_40923` (Devnagari-Script, 1,024 features, 46 classes) for `hgb_default` (6), `hgb_l2_1` (6) and `hgb_shallow_lr05` (5). That is 6 of 1,040 tasks.
- **How a timed-out fold counts:** a timeout in any fold discards the pair's other folds, so all 3 fold accuracies are NaN. In the performance matrix the pair's mean accuracy is then set to that task's lowest observed accuracy across pipelines. This conservatively penalises HGB on Devnagari.
- **Wall time:** about 20 minutes on 9 shards.
- **Winner distribution by family** (ties split; `results_full/winner_distribution_by_family.csv`): grad_boosting 16.5%, svm 16.5%, extra_trees 14.7%, random_forest 12.9%, mlp 12.4%, logreg 9.6%, decision_tree 5.3%, ridge 4.8%, knn 4.8%, naive_bayes 2.5%. No family is above 50%.

### Milestone 6: Phase 1 full results (DONE; tables in `results_full/full_results_tables.md`)
- **Pipeline:**
  - 14 meta-features for all 1,046 tasks (0 NaN/inf, no replacements). Evaluation uses the 1,040 matrix tasks.
  - `folds.json`: GroupKFold by parent_id with seeded group shuffling (seed 42), 5 folds, built from task_info, which now holds only matrix tasks.
- **Selectors:** Random, Single-best, **Static portfolio** (new: greedy train-fold order where each added pipeline minimises the portfolio's mean Regret@k), kNN, LightGBM Regressor and LightGBM LambdaRank.
- **CIs:**
  - All CIs are **parent-cluster bootstraps** (2,000 resamples of parent_id, each drawn parent contributing all its tasks). This covers Regret@1, Regret@3, hit rate and the paired Δ vs Single-best.
  - Task-level CIs are kept for comparison in `full_results_task_level_ci.csv`.
  - Point estimates are per-task metrics averaged over seeds 42/43/44. "± std" is the std of the per-seed means.
- **Results:**
  - LambdaRank beats Single-best on Regret@1 (Δ -0.35 [-0.64, -0.03]) and Regret@3 (Δ -0.86). The LightGBM Regressor also beats it on Regret@1 (Δ -0.32 [-0.51, -0.14]). kNN does not beat Single-best on Regret@1.
  - **The Static portfolio** has the same Regret@1 as Single-best (identical first pick) but gives the lowest Regret@3 (1.43, Δ -0.90). It is as good as LambdaRank at k = 3, so most of the k > 1 gain needs no meta-features.
  - **OpenML only** (retrained on 836 tasks from 145 parents): no meta-model beats Single-best on Regret@1 with a CI excluding 0. The Regressor gets Δ -0.24 [-0.50, +0.02] and LambdaRank -0.14 [-0.48, +0.20]. The Static portfolio is best at Regret@3. The all-task Regret@1 gain is partly driven by non-OpenML tasks.
  - **Landmarker ablation:** removing the 4 landmarkers worsens every meta-model slightly (for example, LambdaRank Regret@1 rises from 2.93 to 3.01, Regret@3 from 1.47 to 1.58). The effect is small relative to the CIs.
  - **Top LightGBM-Regressor gain share:** n_classes 15.2%, lm_stump 12.7%, lm_gnb 11.8%, minority_majority_ratio 7.9%, lm_lda 7.8%.
- **Outputs** in `results_full/`:
  - Clean tables: `full_results.csv` (helper columns moved to `full_results_helpers.csv`), `_by_source`, `_openml_only`, `_ablation`, `_per_seed`.
  - Supporting files: `full_regret_vs_k.csv`, `full_lgbm_gain_share.csv`, `full_timeouts_by_pipeline.csv`, `skipped_tasks.csv`.
  - Figures: winner distribution, Regret@1 with CIs, regret-vs-k (k = 1..30), LightGBM gain share.
- **Tests:** 16 passed.

### Assumptions (Stage C Part 2)
1. **Seeds barely matter for LightGBM:** with no bagging or feature subsampling, LightGBM is deterministic given the data. The seed std of 0.00 for kNN, Single-best, Static and both LightGBM models is real, not a bug. Only Random varies across seeds. Fold assignment is fixed (folds.json, seed 42), as the SPEC requires.
2. **Meta-features use the uncapped task:** they are computed on the full task (landmarkers on at most 200 rows), while pipeline accuracies use the 1,000-row cap. log_n_rows therefore describes the task, not the capped training set.
3. **Timeout imputation:** a timed-out pair takes the task's minimum observed accuracy (inherited pilot rule); see Milestone 5.
4. **The skip rule applies to all tasks:** it is applied identically to tasks finished before it existed. Their cache files remain, but they are excluded from the matrix.
5. **Static portfolio ties** are broken by column order, which is the sorted pipeline_id.

---

## Stage D: RQ1 follow-up and Phase 2 (Milestone 7) code, ready for Colab

### Date: 2026-11-16 (Phase 1 accepted by the user)

### RQ1: meta-models vs the Static portfolio, k = 1..5 (`results_full/vs_static.csv`)
- **Method:** a paired parent-cluster bootstrap (2,000 resamples) of Regret@k(model) − Regret@k(Static) on seed-averaged per-task regrets. It covers all tasks and the OpenML-only retrain. Run with `python -m src.evaluation --mode full --vs-static`.
- **All tasks:** both LightGBM models beat Static only at k = 1.
  - The Regressor is significantly *worse* than Static for k ≥ 2 (k = 2: +0.32 [+0.12, +0.53]).
  - LambdaRank is no longer significantly better from k = 2 on (−0.16 [−0.37, +0.04]) and is never significantly worse.
- **OpenML only:**
  - Regressor at k = 1 is −0.24 [−0.50, −0.00], so its CI upper bound only just reaches 0; it is significantly worse from k = 2.
  - LambdaRank is not significantly better even at k = 1 (−0.14 [−0.47, +0.21]) and is significantly worse at k = 3 (+0.22 [+0.01, +0.45]).
- **Reading:** the meta-features help mainly for the single top pick. For budgets of 2 or more evaluations, a fixed greedy portfolio is as good or better.

### Phase 2 design: deliberate deviations from SPEC §13 (budget)
| Item | Decision |
|---|---|
| Folds | `data/meta_full/folds.json`. Test folds **0, 1, 2 only** (632 tasks); the train set for fold f is the other 4 folds. Same folds for every method. |
| Prompt A | All 14 meta-features (3 dp), n_rows, n_cols, n_classes, minority %, missing %, column dtype counts. |
| Prompt B | A + up to the first 40 column names (each cut to 40 chars) + the OpenML description, cleaned of author/source/cite boilerplate, links and markdown, and truncated to 300 chars. Empty for synthetic/sklearn. |
| Both prompts | List the 30 candidate pipeline_ids and ask for the top 5, best first, comma separated. |
| Target | The 5 pipelines with the highest measured accuracy, ties broken by the training-fold Single-best order. |
| Parser | Known ids in order of appearance, duplicates and unknown ids dropped. The rest is filled in training-fold Single-best order, so every output is a full ranking of 30. |
| Model | Qwen2.5-1.5B-Instruct, 4-bit NF4 QLoRA, LoRA r = 16 / α = 32 / dropout 0.05 on q,k,v,o,gate,up,down. fp16 (T4), max_seq_len 1024, batch 4 × accum 4, lr 2e-4 (cosine, 3% warmup), 2 epochs. Loss on the completion only. Greedy decoding, max_new_tokens 64. |
| Methods | (1) base model, prompt A, no fine-tuning; (2) fine-tuned on A; (3) fine-tuned on B. 6 adapters in total. |
| Not done | No hyperparameter search, a single training seed, no test folds 3–4. |

### Phase 2 code
- **`src/serialize_text.py`** writes to `data/phase2/`:
  - prompts/{A,B}/fold{f}_{train,test}.jsonl (train 826/827/835, test 214/213/205)
  - sb_order.json, candidates.json, system_prompt.txt, openml_descriptions.csv
  - `build_prompt` never receives the performance matrix. Prompts run to at most 946 chars (A) and 2,015 chars (B).
- **Descriptions:** `text_metadata.description` was empty for every task, so descriptions are read from the local OpenML cache (`description.xml`). All 145 OpenML parents have one.
- **`src/llm_infer.py`:** parser plus batched greedy generation. Prediction files are written atomically.
- **`src/llm_finetune.py`:**
  - TRL SFT with conversational prompt/completion records and `completion_only_loss=True`.
  - It adapts to the installed TRL version: it fails loudly if TRL is older than 0.19, and handles both `max_length` and `max_seq_length`.
  - The adapter is saved atomically per (fold, variant).
  - GPU libraries are imported only inside the Colab functions.
- **`notebooks/phase2_qlora_colab.ipynb`:**
  - Mounts Drive, unzips `phase2_bundle.zip`, then runs the base model (folds 0–2) and train + predict for A and B × folds 0–2.
  - Skips any (method, fold) whose `predictions_{method}_fold{f}.json` already exists, and any finished adapter.
- **`scripts/make_bundle.py`:** builds `phase2_bundle.zip` (1.2 MB) containing the meta_full parquets, folds.json, data/phase2, the LLM src modules and configs/full.yaml.
- **`src/llm_eval.py`:**
  - Parses the predictions and checks that each file's task set equals its test fold.
  - Evaluates with the Phase 1 functions (`per_task_metrics`, `summarise_selectors`, parent-cluster CIs).
  - Compares against Single-best, Static, the LGBM Regressor and LambdaRank (out-of-fold, same tasks).
  - Also reports the RQ3 paired B − A difference on OpenML tasks and the parser fill-in share.
- **Tests** (`tests/test_phase2.py`): parser edge cases (garbage, empty/None, duplicates, unknown ids, prefix collisions such as knn_1/knn_15 and logreg_c1/c10), no label or accuracy info in prompts, no test-fold task or parent in any training prompt set, and mock predictions on real data.
- **End-to-end mock check** (`scripts/mock_phase2_check.py`): an oracle mock gives Regret@1 = 0; a garbage mock gives exactly the Single-best result (100% filled).
- `pytest`: 30 passed.

### Assumptions (Stage D)
1. **Description leakage:** OpenML descriptions are public metadata and may mention the dataset's domain, but never our pipelines' accuracies.
2. **Dtype counts:** these come from the type names recorded in text_metadata. For object columns, the type is taken from the first row, so a NaN there counts as numeric.
3. **Baselines in `llm_eval`:** the Phase 1 selectors are rerun with seed 42 (they are deterministic) and restricted to test folds 0–2. Their training sets for fold f are identical to the LLM's.

---

## Stage E: Milestone 8, local Gradio demo

### Date: 2026-11-17

- **`src/recommend.py`** reuses metafeatures, preprocessing, portfolio, metamodels and run_portfolio; no logic is duplicated.
  - `train_final()` (`python -m src.train_final`) fits kNN, the LGBM Regressor, LGBM LambdaRank and the Static portfolio on all 1,040 tasks. It saves them to `models/metapilot_models.joblib` (1.7 MB) with the pipeline ids, family map and training feature statistics.
  - `recommend()`:
    - validates the input: at least 2 classes, at least 6 rows per class, at least 50 rows, a 50 MB file limit, and NaN-target rows dropped;
    - drops ID-like and all-empty columns, and warns when the target looks continuous (more than 20 unique numbers);
    - computes the 14 meta-features, ranks the pipelines, and returns the Static top-k as a safe default;
    - explains the top pick with the 3 largest LightGBM `pred_contrib` values, phrased by percentile against the 1,040 benchmark tasks.
  - `evaluate_pipelines()` runs `run_portfolio._evaluate_pair` (3-fold stratified CV) in a spawned worker process, so the SIGALRM 60 s per-fit limit works. Rows are capped at 2,000, and an outer limit kills a stuck worker.
- **Meta-feature code path:** a new `src/metafeatures.compute_task_meta_features` is used by both training (`extract_all`) and the demo.
  - *Deviation from the request:* training computed meta-features on the **uncapped** task, so the demo does the same, without the 1,000-row cap, to stay exactly consistent. The row cap applies only when fitting pipelines.
  - Consistency test: 5 full tasks (numeric, mixed, categorical with missing values, sklearn, synthetic) match `meta_features.parquet` with a max difference of 0.
- **`app/gradio_demo.py`:** three tabs: Recommend, Benchmark and About.
  - The Recommend tab adds example datasets (3 sklearn, plus credit-g, diabetes and pc4 from OpenML) and a "Test them on my data" button.
  - The LLM button is disabled, with a tooltip set through a title attribute, until `models/llm_adapters/` exists.
  - It runs on `127.0.0.1:7860` with `share=False`, analytics off, and system fonts (no web-font requests).
- **Running it:** `make demo` trains the models if they are missing, then launches the app. `gradio>=5.0,<6` was added to requirements (5.50.0 installed).
- **Tests and smoke check:** `tests/test_recommend.py` covers end to end with all 4 models, consistency on 5 tasks, and the friendly errors. `scripts/smoke_demo.py` is a headless check: it starts the app, calls `/recommend` with iris through gradio_client and shuts the app down. `pytest`: 44 passed.
- **Limitations:**
  - The example datasets are part of the benchmark, so their recommendations are optimistic.
  - The explanation for kNN and Static is borrowed from the LambdaRank model.
  - Feature phrases are relative to the benchmark (for example, 1,000 rows counts as "large").
  - The first "Test them" click takes a few seconds while the worker process starts.

### Stage E (update): UI redesign (2026-11-17)
- **Layout:**
  - `app/gradio_demo.py` uses the user-provided `app/theme.css` and `app/ui_parts.py`, with forced light mode and system serif fonts only.
  - Two tabs: "Find my model" (3 steps) and "How it works" (plain-language results table, 2 plots, limits).
  - Results appear as tickets ("TRY THIS FIRST", "ALSO TRY", "SAFE PICK"), plus a note and chips. Technical details live in a closed "Advanced" accordion.
- **Wording:** the method names shown are plain ("Smart ranker", "Score predictor", "Similar datasets", "Safe list only"). The smoke test checks that "regret", "LambdaRank" and "meta-features" never appear on the served page.
- **Escaping:** all user text (file names, column names, warnings) is HTML-escaped. `tests/test_ui.py` covers it.
- **Bug fix:** `read_csv` now tries commas before delimiter sniffing, because sniffing misparsed headers that contain quotes or `=`. `recommend()` logic is unchanged.
- **Not done:** Playwright is not installed, so the 1100px / 390px screenshots were skipped. Phone layout relies on theme.css's `@media (max-width:640px)` rule and Gradio's row wrapping, and has not been checked visually.
