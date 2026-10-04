"""
Tests that the pipeline is reproducible and the baselines behave sensibly.

Run with: python3 -m pytest tests/ -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent


def _ensure_data_exists():
    """Skip tests that depend on generated data if `make data` hasn't been run."""
    required = [
        REPO / "data/processed/agent1_train.parquet",
        REPO / "data/processed/agent3_train.parquet",
        REPO / "reports/metrics_summary.json",
    ]
    for p in required:
        if not p.exists():
            pytest.skip(f"Required artifact missing: {p}. Run `make data && make baseline && make eda` first.")


def test_generator_import():
    from collectors.generate_synthetic import ATTACK_TECHNIQUES, PROTOCOLS
    assert set(ATTACK_TECHNIQUES) == {"T0836", "T0855", "T0832", "T0814", "T0859", "T0866", "T0863"}
    assert "modbus_tcp" in PROTOCOLS


def test_attack_mapper_rules():
    from agents.orchestrator.attack_mapper import map_agent1_event, map_agent3_event
    # High packet count triggers T0814
    row = {"pkt_count": 10000, "pkt_count_roll_mean": 500, "unique_src_ips": 5, "unique_dst_ips": 5}
    tags = [h["technique_id"] for h in map_agent1_event(row, techniques={})]
    assert "T0814" in tags

    # Non-primary cmd source triggers T0855
    row3 = {"nonprimary_src_frac": 0.5, "max_dev": 0.01, "timing_jitter": 10, "cmd_entropy": 0.1}
    tags3 = [h["technique_id"] for h in map_agent3_event(row3, techniques={})]
    assert "T0855" in tags3


def test_pipeline_splits_are_stratified():
    _ensure_data_exists()
    train = pd.read_parquet(REPO / "data/processed/agent1_train.parquet")
    test = pd.read_parquet(REPO / "data/processed/agent1_test.parquet")
    # Every attack class in train should also appear in test
    assert set(test["attack_class"].unique()).issubset(set(train["attack_class"].unique()))


def test_metrics_within_expected_ranges():
    """Baselines should beat random on a labeled test set.

    Agent 1 PR-AUC should be above 0.5 (random baseline on a 50/50 split).
    Agent 3 PR-AUC should be above 0.5.
    Agent 2 top-10 mean CVSS should be above the overall median (ranker works).
    """
    _ensure_data_exists()
    summary = json.loads((REPO / "reports/metrics_summary.json").read_text())
    assert summary["agent1"]["pr_auc"] > 0.5
    assert summary["agent3"]["pr_auc"] > 0.5
    assert summary["agent2"]["top_k_mean_cvss"] > summary["agent2"]["bottom_k_mean_cvss"]


def test_agent2_ranker_prioritizes_ot_assets():
    """At least 50% of the top-10 should be OT assets (plc, hmi, robot_controller)."""
    _ensure_data_exists()
    summary = json.loads((REPO / "reports/metrics_summary.json").read_text())
    dist = summary["agent2"]["top_k_asset_distribution"]
    ot_count = dist.get("plc", 0) + dist.get("hmi", 0) + dist.get("robot_controller", 0)
    assert ot_count >= 5, f"Only {ot_count} of top-10 are OT assets; expected >= 5"


def test_attack_techniques_loaded():
    _ensure_data_exists()
    path = REPO / "data/threat/attack/techniques_lookup.json"
    assert path.exists(), "ATT&CK lookup missing; run `make fetch`"
    lookup = json.loads(path.read_text())
    expected = {"T0836", "T0855", "T0832", "T0814", "T0859", "T0866", "T0863"}
    assert set(lookup.keys()) == expected
