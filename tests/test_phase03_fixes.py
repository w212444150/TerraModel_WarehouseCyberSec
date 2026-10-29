"""
Phase 03 fix tests: verify the three limitation fixes produce the claimed lift.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Required artifact missing: {path}. Run the Phase 03 fix pipeline first.")


def test_limitation1_agent1_augmentation_improves_f1():
    """Agent 1 + PLC features should beat network-only Agent 1."""
    p = REPO / "reports/phase03/agent1_augmented_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    delta = data["augmentation_delta"]
    assert delta["f1_delta"] > 0.0, \
        f"PLC augmentation should raise F1, got delta {delta['f1_delta']}"
    assert delta["recall_delta"] > 0.0, \
        f"PLC augmentation should raise recall, got delta {delta['recall_delta']}"


def test_limitation2_kev_label_is_external():
    """Agent 2 KEV model uses an external label, not derived from asset_type input."""
    p = REPO / "reports/phase03/agent2_kev_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    assert data["label_source"] == "CISA Known Exploited Vulnerabilities (KEV) catalog"
    # Precision@k on real labels should be reported
    assert "precision_at_top_k" in data
    assert 0.0 <= data["precision_at_top_k"] <= 1.0


def test_limitation2_kev_cv_not_tautological():
    """When CV runs (positives >= 3), ROC-AUC should NOT be 1.0 like the old version."""
    p = REPO / "reports/phase03/agent2_kev_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    if data.get("cv_roc_auc_mean") is not None:
        # Real external labels should give a less-than-perfect CV score
        assert data["cv_roc_auc_mean"] < 0.99, \
            f"CV ROC-AUC of {data['cv_roc_auc_mean']} looks tautological; expected real-label behavior"


def test_limitation3_realistic_data_still_trains():
    """Realistic data generator produces data the pipeline can still train on."""
    p = REPO / "reports/phase03/agent3_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    champ_metrics = data[f"test_{data['champion']}"]
    # Even after realism injections, Agent 3 should hold strong signal
    assert champ_metrics["f1"] > 0.7, \
        f"Agent 3 F1 on realistic data dropped below 0.7: {champ_metrics['f1']}"


def test_hai_loader_module_present():
    """HAI loader for real-data Colab path should import cleanly."""
    import importlib
    mod = importlib.import_module("collectors.load_hai")
    assert hasattr(mod, "map_hai_to_terramodel")
