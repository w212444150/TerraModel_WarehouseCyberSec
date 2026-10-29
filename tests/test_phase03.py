"""
Phase 03 tests: verify supervised models beat baselines and framework mapping is complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Required artifact missing: {path}. Run `make phase03` first.")


def test_agent1_phase03_beats_baseline():
    p = REPO / "reports/phase03/agent1_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    champion = data["champion"]
    champion_test = data[f"test_{champion}"]
    baseline = data["baseline_isolation_forest_phase02"]
    assert champion_test["pr_auc"] > baseline["pr_auc"], \
        f"Phase 03 champion PR-AUC {champion_test['pr_auc']} should beat baseline {baseline['pr_auc']}"
    assert champion_test["f1"] > baseline["f1"]


def test_agent3_phase03_beats_baseline():
    p = REPO / "reports/phase03/agent3_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    champion = data["champion"]
    champion_test = data[f"test_{champion}"]
    baseline = data["baseline_mahalanobis_phase02"]
    # Agent 3 should at least double F1 vs baseline
    assert champion_test["f1"] >= 2 * baseline["f1"], \
        f"Phase 03 Agent 3 F1 {champion_test['f1']} should at least double baseline {baseline['f1']}"
    assert champion_test["recall"] > 0.6, \
        f"Phase 03 Agent 3 recall should be above 0.6, got {champion_test['recall']}"


def test_agent2_phase03_prioritizes_more_ot():
    p = REPO / "reports/phase03/agent2_metrics.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    assert data["top_k_ot_critical_count"] >= 8, \
        f"Phase 03 should put at least 8 OT-critical items in top-10, got {data['top_k_ot_critical_count']}"


def test_framework_mapping_complete():
    from agents.orchestrator.framework_mapper import FRAMEWORK_MAP
    expected = {"T0836", "T0855", "T0832", "T0814", "T0859", "T0866", "T0863"}
    assert set(FRAMEWORK_MAP.keys()) == expected
    for tid, mapping in FRAMEWORK_MAP.items():
        assert "nist_csf_2_0" in mapping
        assert "iec_62443_3_3" in mapping
        assert "nist_sp_800_82r3" in mapping
        assert "mitre_d3fend" in mapping
        for k in ("nist_csf_2_0", "iec_62443_3_3", "nist_sp_800_82r3", "mitre_d3fend"):
            assert len(mapping[k]) >= 1, f"{tid}:{k} is empty"


def test_consolidated_alerts_enriched():
    p = REPO / "reports/phase03/consolidated_alerts.json"
    _skip_if_missing(p)
    data = json.loads(p.read_text())
    assert data["total_sample_alerts_recorded"] > 0
    for alert in data["sample_alerts"]:
        assert "attack_mappings" in alert
        for am in alert["attack_mappings"]:
            fw = am.get("frameworks", {})
            assert "nist_csf_2_0" in fw
            assert "iec_62443_3_3" in fw
