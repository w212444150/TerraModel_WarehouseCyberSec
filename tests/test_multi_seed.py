"""
Multi-seed evaluation test: verify variance report is produced and metrics
show real-but-small standard deviation (proves the pipeline actually trains
rather than reporting fixed numbers).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_multi_seed_summary_exists_or_skip():
    """If the summary exists, metrics should carry real variance."""
    p = REPO / "reports/phase03/multi_seed_summary.json"
    if not p.exists():
        pytest.skip("Run `make multi-seed` first")
    data = json.loads(p.read_text())
    assert data["n_seeds"] >= 3

    # Agent 3 F1 should be high AND show some variance (NOT 0.0 variance - that
    # would mean the pipeline is reporting hardcoded numbers)
    a3_f1 = data["agent3"]["f1"]
    assert a3_f1["mean"] > 0.70, f"Agent 3 mean F1 unexpectedly low: {a3_f1['mean']}"
    # Honest std bounds: too tight (<0.0001) means we're not actually training per seed
    assert a3_f1["std"] >= 0.0001, \
        f"Agent 3 F1 std too tight ({a3_f1['std']}) - pipeline may not be re-training per seed"
    # Not too wild either - above 0.2 std would mean the model is unstable
    assert a3_f1["std"] < 0.2, f"Agent 3 F1 std too wild: {a3_f1['std']}"


def test_multi_seed_agent1_has_variance():
    p = REPO / "reports/phase03/multi_seed_summary.json"
    if not p.exists():
        pytest.skip("Run `make multi-seed` first")
    data = json.loads(p.read_text())
    a1_f1 = data["agent1"]["f1"]
    assert a1_f1["std"] >= 0.0001, \
        f"Agent 1 F1 std too tight ({a1_f1['std']}) - pipeline may not vary per seed"
