# TerraModel Warehouse Cybersecurity - Make targets
# Robert B. Thompson, Cesar Noriega
.PHONY: setup fetch data baseline eda clean-all test all phase03

PYTHON ?= python3
PYRUN ?= PYTHONPATH=. $(PYTHON)

setup:
	$(PYTHON) -m pip install -r requirements.txt

fetch:
	$(PYTHON) collectors/fetch_attack.py
	$(PYTHON) collectors/fetch_cisa.py

generate:
	$(PYTHON) collectors/generate_synthetic.py

data: fetch generate
	$(PYTHON) pipeline/clean.py
	$(PYTHON) pipeline/features.py
	$(PYTHON) pipeline/split.py

baseline:
	$(PYTHON) agents/agent1_anomaly/baseline.py
	$(PYTHON) agents/agent2_assessment/baseline.py
	$(PYTHON) agents/agent3_integrity/baseline.py

eda:
	$(PYTHON) notebooks/06_baseline_eda.py

phase03:
	$(PYRUN) agents/agent1_anomaly/phase03_supervised.py
	$(PYRUN) agents/agent3_integrity/phase03_supervised.py
	$(PYRUN) agents/agent2_assessment/phase03_learned.py
	$(PYRUN) agents/orchestrator/phase03_pipeline.py
	$(PYRUN) -c "from experiments.tracker import consolidate; consolidate()"

phase03-fixes:
	$(PYTHON) collectors/generate_synthetic_realistic.py
	$(PYTHON) pipeline/clean.py
	$(PYTHON) pipeline/features.py
	$(PYRUN) pipeline/features_augmented.py
	$(PYTHON) pipeline/split.py
	$(PYRUN) pipeline/split_augmented.py
	$(PYTHON) collectors/fetch_kev.py
	$(PYRUN) agents/agent1_anomaly/phase03_augmented.py
	$(PYRUN) agents/agent2_assessment/phase03_kev.py
	$(PYRUN) agents/orchestrator/phase03_pipeline.py
	$(PYRUN) -c "from experiments.tracker import consolidate; consolidate()"

multi-seed:
	$(PYRUN) experiments/multi_seed_eval.py --n-seeds 10

multi-seed-fresh:
	$(PYRUN) experiments/multi_seed_eval.py --n-seeds 5 --fresh-data

live:
	$(PYRUN) experiments/live_eval.py

live-no-jitter:
	$(PYRUN) experiments/live_eval.py --jitter 0.0

test:
	$(PYRUN) -m pytest tests/ -v

all: data baseline eda phase03

clean-all:
	rm -rf data/raw/*.parquet data/processed/*.parquet data/processed/*.csv
	rm -rf data/processed/*.json data/threat/attack/*.json data/threat/cisa/*.json
	rm -rf data/threat/cisa/*.csv models/*.joblib reports/*.json reports/*.csv reports/*.png
	rm -rf reports/phase03/*.json reports/phase03/*.png reports/phase03/*.csv
	rm -rf reports/phase03/experiments/*.json
