"""
Phase 03 Week 10 model demonstration script.

Shows the end-to-end TerraModel pipeline reacting to a cyber attack in real
time. Designed to be run live in front of an audience (instructor, class,
reviewer). Takes about 15 seconds.

Walkthrough:
  1. Load the trained Agent 1 (XGBoost) and Agent 3 (Random Forest ensemble)
  2. Pick a random attack window from the test set (unseen during training)
  3. Score it with both agents, show the raw probabilities
  4. Run the orchestrator attack mapper to assign a MITRE ATT&CK for ICS ID
  5. Enrich with NIST CSF 2.0, IEC 62443, NIST SP 800-82 Rev. 3, D3FEND
  6. Print a formatted operator-ready alert

Usage:
    PYTHONPATH=. python experiments/demo.py
    PYTHONPATH=. python experiments/demo.py --technique T0855     # pick a specific attack
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from agents.orchestrator.attack_mapper import (
    load_techniques, map_agent1_event, map_agent3_event,
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


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", default=None,
                        help="ATT&CK technique ID to demo (e.g., T0814, T0855, T0836)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])

    section("TerraModel Phase 03 Live Demo")
    print("  Team TerraModel - Robert B. Thompson and Cesar Noriega")
    print("  Multi-agent AI cybersecurity for AI-powered warehouse automation")

    section("Step 1: Load trained models")
    a1_bundle = joblib.load(models_dir / "agent1_xgb.joblib")
    a3_bundle = joblib.load(models_dir / "agent3_rf.joblib")
    print(f"  Loaded Agent 1 XGBoost model")
    print(f"  Loaded Agent 3 Random Forest model")
    techniques = load_techniques()
    print(f"  Loaded MITRE ATT&CK for ICS v18 ({len(techniques)} techniques)")

    section("Step 2: Simulate an incoming attack")
    a1_test = pd.read_parquet(proc / "agent1_test.parquet")
    a3_test = pd.read_parquet(proc / "agent3_test.parquet")

    if args.technique:
        a1_attacks = a1_test[a1_test["attack_class"] == args.technique]
        a3_attacks = a3_test[a3_test["attack_class"] == args.technique]
        if len(a1_attacks) == 0 and len(a3_attacks) == 0:
            print(f"  No {args.technique} windows in test set. Using any attack.")
            a1_attacks = a1_test[a1_test["is_attack"] == 1]
            a3_attacks = a3_test[a3_test["is_attack"] == 1]
    else:
        a1_attacks = a1_test[a1_test["is_attack"] == 1]
        a3_attacks = a3_test[a3_test["is_attack"] == 1]

    # Pick ONE attack scenario
    scenario_pool = sorted(set(a1_attacks["scenario_id"]) & set(a3_attacks["scenario_id"]))
    if not scenario_pool:
        scenario_pool = sorted(set(a1_attacks["scenario_id"]) | set(a3_attacks["scenario_id"]))
    scenario = random.choice(scenario_pool)

    a1_slice = a1_attacks[a1_attacks["scenario_id"] == scenario].head(1)
    a3_slice = a3_attacks[a3_attacks["scenario_id"] == scenario].head(1)

    ground_truth = (a1_slice["attack_class"].iloc[0] if len(a1_slice)
                    else a3_slice["attack_class"].iloc[0])
    print(f"  Attack scenario #{scenario}")
    print(f"  Ground truth technique: {ground_truth}")
    print(f"  (This is what the attacker actually did; model does not see this label.)")

    a1_proba = None; a3_proba = None
    section("Step 3: Agent 1 scores the network traffic window")
    if len(a1_slice):
        X = a1_slice[A1_FEATURES].values
        a1_proba = float(a1_bundle.predict_proba(X)[0, 1])
        print(f"  Agent 1 (XGBoost) alert probability: {a1_proba:.4f}")
        row = a1_slice.iloc[0].to_dict()
        print(f"  Window features Agent 1 saw:")
        print(f"    pkt_count          = {row['pkt_count']:,.0f}  (normal baseline ~600)")
        print(f"    inter_arrival_ms   = {row['inter_arrival_mean_ms']:.2f}  (normal ~1.6)")
        print(f"    unique_src_ips     = {row['unique_src_ips']:,.0f}")
        print(f"    pkt_count_dev      = {row['pkt_count_deviation']:+,.1f}  (deviation from 10s baseline)")
        a1_hits = map_agent1_event(row, techniques) if a1_proba >= 0.5 else []
    else:
        print(f"  (No Agent 1 window for this scenario.)")
        a1_hits = []

    section("Step 4: Agent 3 scores the robot integrity window")
    if len(a3_slice):
        X = a3_slice[A3_FEATURES].values
        a3_proba = float(a3_bundle.predict_proba(X)[0, 1])
        print(f"  Agent 3 (Random Forest) alert probability: {a3_proba:.4f}")
        row = a3_slice.iloc[0].to_dict()
        print(f"  Window features Agent 3 saw:")
        print(f"    mean_dev           = {row['mean_dev']:.4f}  (|commanded - actual| per joint)")
        print(f"    max_dev            = {row['max_dev']:.4f}")
        print(f"    nonprimary_src_frac= {row['nonprimary_src_frac']:.4f}  (unauthorized cmd source rate)")
        print(f"    cmd_entropy        = {row['cmd_entropy']:.4f}")
        a3_hits = map_agent3_event(row, techniques) if a3_proba >= 0.5 else []
    else:
        print(f"  (No Agent 3 window for this scenario.)")
        a3_hits = []

    section("Step 5: Orchestrator correlates and tags ATT&CK technique")
    all_hits = a1_hits + a3_hits
    if not all_hits:
        print(f"  No alerts fired. (This is a false negative; the attack slipped past.)")
        return
    # Deduplicate by technique_id
    seen = {}
    for h in all_hits:
        seen.setdefault(h["technique_id"], h)
    tagged = list(seen.values())
    for h in tagged:
        print(f"  Tagged: {h['technique_id']} - {h.get('name', '')}")
        print(f"          {h.get('url', '')}")

    section("Step 6: Enrich with defensive cybersecurity frameworks")
    enriched = enrich_event(tagged)
    for h in enriched:
        print(f"\n  [{h['technique_id']}] {h.get('name', '')}")
        fw = h.get("frameworks", {})
        print(f"    ATT&CK tactic: {fw.get('attack_tactic', 'n/a')}")
        print(f"    NIST CSF 2.0 subcategories:")
        for c in fw.get("nist_csf_2_0", []):
            print(f"      - {c}")
        print(f"    IEC 62443-3-3 System Requirements:")
        for c in fw.get("iec_62443_3_3", []):
            print(f"      - {c}")
        print(f"    NIST SP 800-82 Rev. 3 sections:")
        for c in fw.get("nist_sp_800_82r3", []):
            print(f"      - {c}")
        print(f"    MITRE D3FEND defensive countermeasures:")
        for c in fw.get("mitre_d3fend", []):
            print(f"      - {c}")

    section("Step 7: Operator alert (what gets sent to the SOC)")
    alert = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ground_truth_attack_class": ground_truth,
        "detected_techniques": [h["technique_id"] for h in tagged],
        "detection_correct": (ground_truth in [h["technique_id"] for h in tagged]
                              if ground_truth != "normal" else False),
        "detectors": {
            "agent1_xgboost": {"fired": bool(a1_hits), "probability": a1_proba},
            "agent3_random_forest": {"fired": bool(a3_hits), "probability": a3_proba},
        },
        "framework_mapping": enriched,
    }
    print(json.dumps({k: v for k, v in alert.items() if k != "framework_mapping"},
                     indent=2, default=str))
    print(f"\n  Full enriched alert (with framework details) saved to:")
    out = Path("reports/phase03/demo_alert.json")
    out.write_text(json.dumps(alert, indent=2, default=str))
    print(f"    {out}")

    section("Demo complete")
    print(f"  Ground truth attack:  {ground_truth}")
    print(f"  Detected and tagged:  {[h['technique_id'] for h in tagged]}")
    correct = ground_truth in [h["technique_id"] for h in tagged]
    print(f"  Correct detection:    {'YES' if correct else 'NO (model limitation)'}")
    print()


if __name__ == "__main__":
    main()
