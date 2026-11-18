.PHONY: setup test pilot-data pilot-portfolio pilot-meta pilot-all clean full-tasks full-portfolio full-merge full-meta full-eval full-all demo

PYTHON = .venv/bin/python
PIP = .venv/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	.venv/bin/pytest tests/ -v

pilot-data:
	$(PYTHON) -m src.data_loading --mode pilot
	$(PYTHON) -m src.augmentation --mode pilot

pilot-portfolio:
	$(PYTHON) -m src.run_portfolio --mode pilot

pilot-meta:
	$(PYTHON) -m src.metafeatures --mode pilot
	$(PYTHON) -m src.meta_dataset --mode pilot

pilot-all: pilot-data pilot-portfolio pilot-meta

# ---- Full mode (paths: data/tasks_full, data/meta_full, results_full) ----
# SHARDS parallel single-threaded workers (default cores - 1); no oversubscription.
SHARDS ?= $(shell echo $$(( $$(sysctl -n hw.ncpu 2>/dev/null || nproc) - 1 )))
THREAD_ENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

full-tasks:
	$(PYTHON) -m src.augmentation --mode full

full-portfolio:
	mkdir -p logs
	for i in $$(seq 0 $$(( $(SHARDS) - 1 ))); do \
		$(THREAD_ENV) caffeinate -i $(PYTHON) -m src.run_portfolio --mode full --shard $$i/$(SHARDS) > logs/full_shard_$$i.log 2>&1 & \
	done; wait

full-merge:
	$(PYTHON) -m src.merge_shards --mode full

full-meta:
	$(PYTHON) -m src.metafeatures --mode full
	$(PYTHON) -m src.meta_dataset --mode full
	$(PYTHON) -m src.splits --mode full

full-eval:
	$(PYTHON) -m src.evaluation --mode full

full-all: full-tasks full-portfolio full-merge full-meta full-eval

clean:
	rm -rf data/raw data/tasks data/meta results logs
	mkdir -p data/raw data/tasks data/meta results logs

# ---- Demo (local only: http://127.0.0.1:7860) ----
models/metapilot_models.joblib:
	$(PYTHON) -m src.train_final --mode full

demo: models/metapilot_models.joblib
	$(PYTHON) app/gradio_demo.py
