# MetaPilot Status

**Updated:** 2026-11-16 (after Milestones 5 and 6, plus the Phase 2 code). The earlier read-only audit is superseded; details are in docs/PROGRESS.md, Stage C Part 2.

## Milestones (SPEC §14)

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Scaffold | DONE | Makefile (pilot + `full-*` targets), configs, tests |
| 2 | Pilot data + portfolio | DONE | `data/meta_pilot/performance_matrix.parquet` 120×18 (backup of `data/meta`) |
| 3 | Meta-features + text | DONE | pilot 120×14, full 1,046×14, 0 NaN/inf |
| 4 | Phase 1 on pilot | DONE | `results_pilot/` (backup of `results/`) |
| 5 | Scale-up (full mode) | DONE | `data/meta_full/performance_matrix.parquet` 1,040×30 from 179 parents; 93,600 fit rows |
| 6 | Phase 1 completion | DONE | `results_full/full_results_tables.md`; parent-cluster CIs, 3 seeds, ablation, by source, openml-only |
| 7 | Phase 2 (LLM) | PARTIAL: code done, GPU run pending | serializer, parser, QLoRA trainer, notebook, bundle and `llm_eval` are implemented and tested with a mock model; Colab run not yet done |
| 8 | Gradio demo | DONE | `make demo` → http://127.0.0.1:7860 (local only). Tabs: Find my model, How it works, History. Code: `src/recommend.py`, `app/gradio_demo.py`, `app/history.py`. Headless smoke test passed |
| 9 | Report | NOT STARTED | no `docs/REPORT_DRAFT.md` |

Not run: FLAML and TabPFN (skipped by decision).

## Full run in numbers
- **Parents:** 185 selected (151 OpenML ≤ 2,000 features, 4 sklearn, 30 synthetic). 10 wide OpenML parents were excluded (`data/meta_full/excluded_parents.csv`).
- **Tasks:** 1,046 generated at 6 subsamples each. **6 were skipped** (`results_full/skipped_tasks.csv`): `openml_10__s0`, `openml_186__s0`, `openml_39__s0`, `openml_449__s0`, `openml_57__s0`, `openml_488__s0`. That leaves **1,040 tasks**: openml 836, synthetic 180, sklearn 24.
- **Fits:** 0 failures. 17 timed-out pairs (0.05% of 31,200 pairs) on 6 tasks, all `openml_40923` × HistGradientBoosting (`results_full/full_timeouts_by_pipeline.csv`).
- **Winner families** (ties split): grad_boosting 16.5%, svm 16.5%, extra_trees 14.7%, random_forest 12.9%, mlp 12.4%, others below 10%.

## Deliberate deviations from SPEC (budget)
- 6 subsamples per parent instead of 10.
- 30 pipelines instead of ~40.
- Row cap 1,000 instead of 2,000.
- Fit timeout 30 s.
- OpenML feature limit 2,000.
- No FLAML/TabPFN.

All are recorded in PROGRESS.md.

## Health
- **Tests:** `.venv/bin/pytest tests/ -q` gives **51 passed**.
- **Fixed this stage:**
  - shard index parsing
  - pilot/full path collisions
  - dry-run writing the real progress file
  - stale `load_failures.csv` (now 5 genuine failures)
  - OpenML `factorize` bug
  - numeric cast crashing on 'Y'
  - landmarker stratification crashes
  - shard crash on singleton classes (now skipped and logged)
  - hard-coded k = 18 in regret-vs-k
- **Known limitations:**
  - LightGBM is deterministic across seeds (no bagging), so only Random varies across seeds.
  - Timed-out pairs take the task's minimum accuracy.
  - Meta-features are computed on uncapped tasks.
  - The project is not a git repository.

## Demo history
- **Where it is saved:** `history/history.json`, created on the first recommendation. It keeps the latest 100 entries and is written atomically. Set `METAPILOT_HISTORY` to use another path; tests and the smoke test use temporary files.
- **What it holds:** summaries only (dataset file name, size, target, method, picks, safe list, tested accuracies). It never stores data values or other column names.
- **If the file breaks:** a corrupt file is moved aside and history starts empty.
- **CSV export:** neutralises spreadsheet formulas.
- **Git:** the project has no `.gitignore` because it isn't a git repository. Add `history/` to one if you create it.
- **UI:** the LLM placeholder has been removed.

## Next step
Run `notebooks/phase2_qlora_colab.ipynb` on a Colab T4 with `phase2_bundle.zip` (see PROGRESS Stage D). Then put the 9 prediction JSONs in `results_full/phase2/predictions/` and run `.venv/bin/python -m src.llm_eval --mode full`.
