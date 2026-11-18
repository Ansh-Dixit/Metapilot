# MetaPilot: Master Specification

You are a senior ML engineer. Build MetaPilot from this spec in milestones. After each milestone run its acceptance checks and append a note to `docs/PROGRESS.md` (assumptions, results, issues). Stop at the stop-points named in the instruction I give you. Ask me only if something is ambiguous in a way that changes results.

## 1. Goal
Pipeline selection for tabular classification as a learning problem. Run a fixed portfolio of ML pipelines on many classification tasks, describe each task with meta-features, train a meta-model to predict the best pipelines for unseen datasets.
- Phase 1: kNN, LightGBM regressor, LightGBM LambdaRank on 14 numeric meta-features.
- Phase 2: fine-tune Qwen2.5-1.5B-Instruct (4-bit QLoRA) to read a text description of a dataset and output a ranked pipeline list.

Research questions:
1. Does a meta-model beat a strong fixed default (single-best), and at what evaluation budget k does the advantage disappear?
2. Can a fine-tuned LLM match LightGBM given the same numeric information?
3. Does semantic info (column names, description) improve selection over numeric meta-features alone?

## 2. Global conventions
- Python 3.10+. Packages: scikit-learn, lightgbm, pandas, numpy, scipy, pyarrow, joblib, pyyaml, tqdm, matplotlib, pytest, openml (full study), gradio (demo). Phase 2 only: transformers, peft, bitsandbytes, trl, datasets. Optional: pymfe, tabpfn, flaml.
- All paths and parameters come from `configs/*.yaml`. Nothing hard-coded.
- Global seed: 42 (config key `seed`). Use it for every random operation. Pass `random_state` everywhere.
- Every expensive step is parallel (joblib, `n_jobs` from config), resumable (cache per task+pipeline to disk, skip completed work) and logged with tqdm.
- Use `--mode pilot|full` on every CLI entry point. Build and verify `pilot` first.
- snake_case for files, functions, columns. IDs: `parent_id` (string, e.g. `synth_cls_007`, `sk_iris`, `moons_002`), `task_id` (string, `<parent_id>__s<k>`, k=0..n_sub-1), `pipeline_id` (string, see section 5).
- Tables are parquet. Config is YAML. JSON for folds.
- Log to `logs/` and keep stdout concise.

## 3. Repo structure
```
metapilot/
  configs/{pilot.yaml, full.yaml}
  data/{raw, tasks, meta}/        # tasks/: one .npz or parquet per task; meta/: matrices and tables
  results/                        # tables, figures
  logs/
  src/
    __init__.py
    config.py            # load yaml, resolve paths
    data_loading.py      # sklearn built-ins, synthetic; OpenML (full mode)
    augmentation.py      # task subsampling
    preprocessing.py     # shared preprocessing (ColumnTransformer builder)
    portfolio.py         # pipeline registry
    run_portfolio.py     # evaluate every pipeline on every task
    metafeatures.py      # 14 meta-features
    meta_dataset.py      # assemble matrix, targets, text metadata
    splits.py            # GroupKFold by parent, saved once
    metamodels.py        # kNN, LGBM regressor, LGBM LambdaRank, baselines
    evaluation.py        # metrics, bootstrap, regret-vs-k
    serialize_text.py    # Phase 2 prompts
    llm_finetune.py      # QLoRA
    llm_infer.py         # generation + ranking parser
    plots.py
  app/gradio_demo.py
  notebooks/phase2_qlora_colab.ipynb
  tests/
  docs/{SPEC.md, PROGRESS.md}
  README.md  Makefile  requirements.txt
```
Makefile targets: `setup`, `test`, `pilot-data`, `pilot-portfolio`, `pilot-meta`, `pilot-all`.

## 4. Configs
`configs/pilot.yaml` schema (full.yaml uses the same keys with full-study values):
```yaml
mode: pilot
seed: 42
paths: {raw: data/raw, tasks: data/tasks, meta: data/meta, results: results, logs: logs}
n_jobs: 1                    # pilot: 1 core ok; allow -1
data:
  sklearn_parents: [iris, wine, breast_cancer, digits]
  synthetic_classification: {n_parents: 20, n_samples: [60, 800], n_features: [2, 150], informative_frac: [0.2, 0.8], class_sep: [0.5, 2.0], flip_y: [0.0, 0.15], n_classes: [2, 5]}
  synthetic_shapes: {n_parents: 6, kinds: [moons, circles], n_samples: [120, 600], noise: [0.05, 0.4]}
  openml: {enabled: false, suite_id: 99}      # full mode: enabled true
augmentation: {n_subsamples: 4, row_frac: [0.3, 1.0], col_frac: [0.5, 1.0], min_class_count: 6}
preprocessing: {row_cap: 800, high_card_threshold: 20}
cv: {n_folds: 3, scoring: accuracy, fit_timeout_sec: 60}
meta: {landmarker_rows: 200, landmarker_folds: 2}
splits: {n_folds: 5, file: data/meta/folds.json}
metamodels: {knn_k: 5, lgbm: {n_estimators: 300, learning_rate: 0.05, num_leaves: 15}, relevance_levels: 10}
evaluation: {bootstrap_resamples: 2000, hit_tolerance_pts: 0.5, ks: [1, 2, 3, 5, 10]}
```
Task `row_frac`/`col_frac` are sampled uniformly per subsample. Always keep at least 2 columns. Subsample 0 of each parent may use the full data (frac 1.0) so the original is represented; document your choice.

## 5. Pipeline portfolio (pilot: exactly 18, `pipeline_id` and family)
Scaled = `StandardScaler` inside the sklearn Pipeline. Tree models are not scaled.

| pipeline_id | family | model |
|---|---|---|
| logreg_c0.1 | logreg | LogisticRegression(C=0.1, max_iter=1000), scaled |
| logreg_c10 | logreg | LogisticRegression(C=10, max_iter=1000), scaled |
| ridge | ridge | RidgeClassifier(), scaled |
| gnb | naive_bayes | GaussianNB() |
| knn_1 | knn | KNeighborsClassifier(1), scaled |
| knn_5 | knn | KNeighborsClassifier(5), scaled |
| knn_15 | knn | KNeighborsClassifier(15), scaled (cap k at n_train-1) |
| dt_d3 | decision_tree | DecisionTreeClassifier(max_depth=3) |
| dt_full | decision_tree | DecisionTreeClassifier() |
| rf_100 | random_forest | RandomForestClassifier(100) |
| rf_100_d5 | random_forest | RandomForestClassifier(100, max_depth=5) |
| et_100 | extra_trees | ExtraTreesClassifier(100) |
| hgb_default | grad_boosting | HistGradientBoostingClassifier() |
| hgb_shallow_lr05 | grad_boosting | HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05) |
| svm_c1 | svm | SVC(C=1, kernel='rbf'), scaled |
| svm_c10 | svm | SVC(C=10, kernel='rbf'), scaled |
| mlp_32 | mlp | MLPClassifier((32,), max_iter=300), scaled |
| mlp_128 | mlp | MLPClassifier((128,), max_iter=300), scaled |

`portfolio.py` exposes a registry `{pipeline_id: (family, builder_fn)}` so the full study can extend to about 40 pipelines (3 to 6 per family) plus optional TabPFN without changing other code. Set `random_state` on all stochastic models.

## 6. Data, preprocessing, augmentation
Parents: pilot = 4 sklearn (iris, wine, breast_cancer, digits) + 20 `make_classification` + 6 moons/circles = 30. Full = about 150 from OpenML-CC18 (suite 99) + small sets (ionosphere, sonar, heart-statlog, hepatitis, haberman + sklearn ones) + high-dim sets (arcene, madelon, gina, microarray) + synthetic. Wrap each OpenML load in try/except, log failures to `data/raw/load_failures.csv`, cache locally.

Preprocessing (identical for all pipelines, built as a `ColumnTransformer` placed INSIDE each sklearn Pipeline so it is fitted per CV training fold only):
1. Encode labels 0..K-1.
2. One-hot encode categoricals; frequency-encode those with more than `high_card_threshold` categories.
3. Impute inside the pipeline: median (numeric), most-frequent (categorical).
4. StandardScaler only for logreg, ridge, knn, svm, mlp.
5. Stratified row cap (`row_cap`) applied once per task before CV.
6. Per-fit timeout; failed or timed-out fit -> NaN. After all pipelines run on a task, replace NaN with that task's lowest observed score.

Augmentation: per parent make `n_subsamples` tasks by random row/col fractions (config). Drop any subsample where a class has fewer than `min_class_count` samples. Each task is independent: own meta-features, own performance row, same `parent_id`.

## 7. Performance matrix
`data/meta/performance_matrix.parquet`: index `task_id`; columns = `pipeline_id` (one per pipeline); values = mean 3-fold stratified CV accuracy in [0,1] (after NaN replacement). Also save `data/meta/fit_log.parquet` with columns `task_id, pipeline_id, fold, accuracy, fit_seconds, status` (status: ok/failed/timeout) so the NaN replacement is auditable. Save `data/meta/task_info.parquet`: `task_id, parent_id, source, n_rows, n_cols, n_classes, row_frac, col_frac`.

## 8. Meta-features (exactly 14, parquet `data/meta/meta_features.parquet`, index `task_id`)
Column names:
- Simple: `log_n_rows`, `log_n_cols`, `cols_to_rows`, `n_classes`
- Statistical: `mean_abs_corr`, `mean_abs_skew`, `mean_kurtosis`, `pca95_frac`
- Info-theoretic: `norm_class_entropy`, `minority_majority_ratio`
- Landmarkers (2-fold accuracy on at most `landmarker_rows` random rows): `lm_stump`, `lm_1nn`, `lm_gnb`, `lm_lda`

Definitions: `cols_to_rows` = n_cols/n_rows (after preprocessing/one-hot); `mean_abs_corr` = mean absolute off-diagonal correlation of numeric features; `pca95_frac` = components needed for 95% variance divided by n_cols; `norm_class_entropy` = entropy / log(n_classes). Handle constant features, a single column and NaN safely (no NaN/inf in the output). Landmarker features use the same imputation as the pipelines.

Text metadata `data/meta/text_metadata.parquet`: `task_id, column_names (list), dtypes (list), missing_pct, description (empty string if none), n_rows, n_cols, n_classes, minority_pct`. Collected now for Phase 2, not used in Phase 1.

## 9. Targets
- Regression target per (task, pipeline): `gap = acc(t,p) - max_p acc(t,p)` (0 for best, negative otherwise).
- LambdaRank label: per-task min-max rescale of accuracy to integers 0..10 (ties allowed; if all equal, all 0).
- Meta-model design matrix: meta-features concatenated with one-hot `pipeline_id`; one row per (task, pipeline). Group by task for LambdaRank.
- kNN meta-learner: standardise meta-features with training-fold statistics only.

## 10. Splits (leakage control)
5-fold GroupKFold by `parent_id`, seed fixed. Save once to `data/meta/folds.json` as `{task_id: fold_index}`. Every method, including the LLM, must load this file. Tests must assert: (a) no `parent_id` appears in both train and test of any fold; (b) imputer/scaler statistics come from training folds only.

## 11. Phase 1 methods
- kNN meta-learner (auto-sklearn style): 5 nearest training tasks in standardised meta-feature space; average their normalised pipeline scores; rank.
- LightGBM regressor: 300 trees, lr 0.05, 15 leaves; predict gap per (task, pipeline); sort descending.
- LightGBM LambdaRank: same features, `objective=lambdarank`, grouped by task, labels 0..10.
- Baselines: Random; Single-best (best mean accuracy on training tasks, applied to every test task). Full study: TabPFN, FLAML with equal time budget (optional, graceful fallback).

## 12. Evaluation
- Regret(t,k) = max_p a(t,p) - max over top-k recommended of a(t,p). Report Regret@1 and Regret@3 in accuracy percentage points.
- Top-1 hit rate: selected pipeline within 0.5 points of best.
- Spearman between predicted and true scores per task, averaged.
- 95% bootstrap CIs over tasks, 2000 resamples.
- Regret-vs-k curve for all k.
- Sanity check BEFORE modelling: print and plot the distribution of winning algorithm families; warn loudly if one family wins more than about 50% of tasks. Ties: split credit equally among tied pipelines/families.
- Ablation: rerun without landmarkers (NB landmarker may leak into NB pipeline).
- Full study: at least 3 seeds, report mean +/- std.
- Output table format: `Selector | Regret@1 (CI) | Regret@3 (CI) | Top-1 hit | Spearman`.

## 13. Phase 2 (LLM)
- Two prompt versions per task: (A) numeric-only (rows, features, classes, minority %, missing %, dtypes, landmarkers); (B) A + column names + description. Example: `Dataset: 768 rows, 8 features, binary target, minority class 35%. Columns: ... Missing: 0%. All numeric. Landmarkers: stump 0.71, 1-NN 0.68, naive Bayes 0.75, LDA 0.77. Description: ... Rank the candidate pipelines from best to worst.`
- Target: pipeline_ids ordered by measured accuracy.
- Base: Qwen2.5-1.5B-Instruct, 4-bit QLoRA, LoRA rank 16 on attention and MLP layers; must fit a free Colab T4. Train per fold on training-fold prompts only, using folds.json.
- Parse generations against the known pipeline_id list; append missing/unparseable ones at the end in single-best order so every output is a full ranking.
- Compare: base model (no fine-tune), fine-tuned numeric-only, fine-tuned semantic. Same folds and metrics as Phase 1. Report results with and without synthetic tasks (semantic info is weak on synthetic data).

## 14. Milestones
1. Scaffold: repo, configs, Makefile, README, requirements, venv, tests.
2. Pilot data + portfolio: 120 tasks x 18 pipelines matrix (about 6,480 fits, about 5 min on one core). Winner distribution table + plot.
3. Meta-features + text metadata: 14 features per task, no NaN/inf.
4. Phase 1 models + evaluation on pilot, results table and plots (winner distribution, Regret@1 with CIs, regret-vs-k, LightGBM gain share).
5. Scale-up: `--mode full` (about 150 parents, 10 subsamples, about 40 pipelines), resumable, shardable (`--shard i/N`) with a merge script.
6. Phase 1 completion: full results, seeds, landmarker ablation, FLAML/TabPFN if available.
7. Phase 2: serializer, Colab notebook + script, inference + parser, 3-variant evaluation.
8. Gradio demo: upload CSV, choose target column, compute meta-features, show ranked pipelines with a plain-language explanation (and optionally the LLM ranking).
9. Report assets in `results/` and `docs/REPORT_DRAFT.md` answering the 3 RQs honestly.

## 15. Quality rules
- Never tune or select anything on test folds; every fitted object lives inside the training fold.
- Be honest: if meta-models do not beat single-best, say so and show the regret-vs-k trade-off.
- Document every assumption in `docs/PROGRESS.md`.
- Do not start a later milestone until the earlier one's acceptance checks pass and I confirm.
