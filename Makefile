# TerraModel Warehouse Cybersecurity - Make targets
# Robert B. Thompson, Cesar Noriega
.PHONY: setup fetch data baseline eda clean-all test all

PYTHON ?= python3

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

test:
	$(PYTHON) -m pytest tests/ -v

all: data baseline eda

clean-all:
	rm -rf data/raw/*.parquet data/processed/*.parquet data/processed/*.csv
	rm -rf data/processed/*.json data/threat/attack/*.json data/threat/cisa/*.json
	rm -rf data/threat/cisa/*.csv models/*.joblib reports/*.json reports/*.csv reports/*.png
