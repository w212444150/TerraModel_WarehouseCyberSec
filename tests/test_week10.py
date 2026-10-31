"""
Week 10 refinement tests: verify the ensemble and final-optimization artifacts
exist and the promotion decisions were made honestly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Required artifact missing: {path}")


def test_ensemble_produced_both_agents():
    p = REPO / "reports/phase03/ensemble_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    assert "agent1" in data and "agent3" in data
    for a in ("agent1", "agent3"):
        assert "random_forest" in data[a]
        assert "xgboost" in data[a]
        assert "soft_ensemble" in data[a]
        assert "production_pick" in data[a]
        # production_pick must be one of the two valid values
        assert data[a]["production_pick"] in ("soft_ensemble", "best_single")


def test_ensemble_honest_comparison():
    """Promotion must be based on PR-AUC delta, not just always-promote."""
    p = REPO / "reports/phase03/ensemble_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    for a in ("agent1", "agent3"):
        delta = data[a]["ensemble_pr_auc_delta_vs_best_single"]
        pick = data[a]["production_pick"]
        if delta > 0:
            assert pick == "soft_ensemble"
        else:
            assert pick == "best_single", \
                f"{a}: ensemble lost (delta {delta}) but was promoted anyway"


def test_final_optimization_report_present():
    p = REPO / "reports/phase03/final_optimization.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    assert data["trials_per_agent"] >= 50
    assert "agent1" in data and "agent3" in data
    for a in ("agent1", "agent3"):
        assert "best_params" in data[a]
        assert "test_metrics" in data[a]
        assert "promotion_decision" in data[a]
