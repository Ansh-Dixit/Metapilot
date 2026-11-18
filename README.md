# MetaPilot

**Pipeline selection for tabular classification as a learning problem.**

MetaPilot runs a fixed portfolio of ML pipelines on many classification tasks, describes each task with meta-features, and trains a meta-model to predict the best pipeline(s) for unseen datasets.

## Quick Start

```bash
make setup          # create venv and install dependencies
make test           # run pytest
make pilot-all      # run full pilot pipeline (data → portfolio → meta-features)
```

## Individual Steps

```bash
make pilot-data       # generate 30 parent datasets, 120 tasks
make pilot-portfolio  # run 18 pipelines × 120 tasks with 3-fold CV
make pilot-meta       # extract 14 meta-features + text metadata
```

## Project Structure

```
configs/         – YAML configuration (pilot.yaml, full.yaml)
data/raw/        – raw parent datasets
data/tasks/      – one .npz per task (subsampled)
data/meta/       – performance matrix, meta-features, folds, text metadata
results/         – tables, figures, plots
logs/            – run logs
src/             – all source modules
tests/           – pytest test suite
docs/            – SPEC.md, PROGRESS.md
```

## Modes

- `--mode pilot` – 30 parents, 120 tasks, 18 pipelines (~5 min on 1 core)
- `--mode full`  – ~150 parents from OpenML-CC18 + synthetics, ~40 pipelines

## Configuration

All parameters live in `configs/pilot.yaml` (or `full.yaml`). Nothing is hard-coded.
Global seed: 42.
