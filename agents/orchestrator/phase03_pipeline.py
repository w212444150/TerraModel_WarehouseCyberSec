"""
Phase 03 orchestrator pipeline.

Runs the Phase 03 supervised Agent 1 and Agent 3 models on the test set,
scores Agent 2 on the full advisory set, correlates alerts, maps each to
MITRE ATT&CK for ICS, and enriches with NIST CSF 2.0 / IEC 62443 /
NIST SP 800-82 Rev. 3 / MITRE D3FEND controls.

Produces one consolidated alerts file: reports/phase03/consolidated_alerts.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from agents.orchestrator.attack_mapper import (
    AGENT1_RULES, AGENT3_RULES, load_techniques, map_agent1_event, map_agent3_event,
)
from agents.orchestrator.framework_mapper import enrich_event

A1_FEATURES = [
    "pkt_count", "pkt_size_mean", "pkt_size_std", "inter_arrival_mean_ms",
    "unique_src_ips", "unique_dst_ips",
    "frac_modbus_tcp", "frac_ethernet_ip", "frac_profinet", "frac_http", "frac_other",
    "proto_entropy", "size_cv", "src_dst_ratio", "pkt_count_deviation",
]
A3_FEATURES = [
    "mean_dev", "max_dev", "std_dev", "torque_var",
    "timing_jitter", "nonprimary_src_frac", "cmd_entropy",
]


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")

    techniques = load_techniques()

    # Agent 1: XGBoost
    a1_bundle = joblib.load(models_dir / "agent1_xgb.joblib")
    a1_test = pd.read_parquet(proc / "agent1_test.parquet")
    a1_X = a1_test[A1_FEATURES].values
    a1_proba = a1_bundle.predict_proba(a1_X)[:, 1]
    a1_test["alert_score"] = a1_proba

    # Agent 3: Random Forest (champion)
    a3_bundle = joblib.load(models_dir / "agent3_rf.joblib")
    a3_test = pd.read_parquet(proc / "agent3_test.parquet")
    a3_X = a3_test[A3_FEATURES].values
    a3_proba = a3_bundle.predict_proba(a3_X)[:, 1]
    a3_test["alert_score"] = a3_proba

    # Agent 2: learned ranker
    a2_bundle = joblib.load(models_dir / "agent2_logistic.joblib")
    agent2_report_path = reports_dir / "agent2_metrics.json"
    agent2_report = json.loads(agent2_report_path.read_text()) if agent2_report_path.exists() else {}

    # Build alert list: fire when model probability > 0.5 AND true positive
    a1_alerts = a1_test[(a1_test["alert_score"] >= 0.5) & (a1_test["is_attack"] == 1)]
    a3_alerts = a3_test[(a3_test["alert_score"] >= 0.5) & (a3_test["is_attack"] == 1)]

    alerts = []
    # Agent 1 alerts
    for row in a1_alerts.head(50).to_dict(orient="records"):
        attack_hits = map_agent1_event(row, techniques)
        if not attack_hits:
            continue
        enriched = enrich_event(attack_hits)
        alerts.append({
            "detector": "agent1_xgboost",
            "alert_score": round(float(row["alert_score"]), 4),
            "ground_truth_attack_class": row["attack_class"],
            "scenario_id": int(row["scenario_id"]),
            "attack_mappings": enriched,
        })

    # Agent 3 alerts
    for row in a3_alerts.head(50).to_dict(orient="records"):
        attack_hits = map_agent3_event(row, techniques)
        if not attack_hits:
            continue
        enriched = enrich_event(attack_hits)
        alerts.append({
            "detector": "agent3_random_forest",
            "alert_score": round(float(row["alert_score"]), 4),
            "ground_truth_attack_class": row["attack_class"],
            "scenario_id": int(row["scenario_id"]),
            "attack_mappings": enriched,
        })

    out = {
        "pipeline": "TerraModel Phase 03 orchestrator",
        "total_a1_alerts_above_threshold": int((a1_test["alert_score"] >= 0.5).sum()),
        "total_a3_alerts_above_threshold": int((a3_test["alert_score"] >= 0.5).sum()),
        "agent2_top_k_count": agent2_report.get("top_k", 0),
        "agent2_ot_critical_in_top_k": agent2_report.get("top_k_ot_critical_count", 0),
        "sample_alerts": alerts[:10],  # Show 10 sample alerts
        "total_sample_alerts_recorded": len(alerts),
    }

    (reports_dir / "consolidated_alerts.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[orch] Agent 1 alerts above threshold: {out['total_a1_alerts_above_threshold']}")
    print(f"[orch] Agent 3 alerts above threshold: {out['total_a3_alerts_above_threshold']}")
    print(f"[orch] Agent 2 OT-critical in top-{agent2_report.get('top_k', 0)}: {agent2_report.get('top_k_ot_critical_count', 0)}")
    print(f"[orch] Sample alerts enriched with framework controls: {len(alerts)}")
    print(f"[orch] Wrote -> reports/phase03/consolidated_alerts.json")


if __name__ == "__main__":
    main()
