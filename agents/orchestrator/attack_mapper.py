"""
Orchestrator ATT&CK mapper.

Loads the MITRE ATT&CK for ICS techniques lookup produced by collectors/fetch_attack.py.
For each flagged event from any agent, assigns one or more ATT&CK technique IDs based
on which features triggered.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

# Feature-pattern -> ATT&CK technique mapping. One feature can map to multiple techniques.
AGENT1_RULES = [
    # (feature_predicate, technique_id)
    (lambda row: row.get("pkt_count", 0) > 3 * row.get("pkt_count_roll_mean", 1), "T0814"),
    (lambda row: row.get("unique_src_ips", 0) > 15, "T0855"),
    (lambda row: row.get("unique_dst_ips", 0) > 25, "T0866"),
]
AGENT3_RULES = [
    (lambda row: row.get("nonprimary_src_frac", 0) > 0.3, "T0855"),
    (lambda row: row.get("max_dev", 0) > 0.05, "T0836"),
    (lambda row: row.get("timing_jitter", 0) < 1.5, "T0863"),
    (lambda row: row.get("cmd_entropy", 0) > 1.0, "T0832"),
]


def load_techniques() -> dict:
    path = Path("data/threat/attack/techniques_lookup.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def map_agent1_event(row: dict, techniques: dict) -> list[dict]:
    hits = []
    for pred, tid in AGENT1_RULES:
        if pred(row):
            info = techniques.get(tid, {})
            hits.append({"technique_id": tid, "name": info.get("name", ""),
                         "url": info.get("url", "")})
    return hits


def map_agent3_event(row: dict, techniques: dict) -> list[dict]:
    hits = []
    for pred, tid in AGENT3_RULES:
        if pred(row):
            info = techniques.get(tid, {})
            hits.append({"technique_id": tid, "name": info.get("name", ""),
                         "url": info.get("url", "")})
    return hits


def correlate(agent1_hits: list[dict], agent3_hits: list[dict]) -> list[dict]:
    """Deduplicate by technique ID and combine confidences."""
    combined: dict[str, dict] = {}
    for h in agent1_hits + agent3_hits:
        tid = h["technique_id"]
        if tid not in combined:
            combined[tid] = {**h, "detector_count": 1}
        else:
            combined[tid]["detector_count"] += 1
    # Sort: multi-detector hits first, then by technique_id for stability
    return sorted(combined.values(), key=lambda x: (-x["detector_count"], x["technique_id"]))


if __name__ == "__main__":
    # Smoke test
    import pandas as pd
    techniques = load_techniques()
    print(f"[orchestrator] Loaded {len(techniques)} ATT&CK techniques: {sorted(techniques.keys())}")

    a1 = pd.read_parquet("data/processed/agent1_test.parquet")
    a3 = pd.read_parquet("data/processed/agent3_test.parquet")

    # Take first 50 attack rows from each and map
    a1_sample = a1[a1["is_attack"] == 1].head(50)
    a3_sample = a3[a3["is_attack"] == 1].head(50)

    a1_tags = [tag["technique_id"] for row in a1_sample.to_dict(orient="records")
               for tag in map_agent1_event(row, techniques)]
    a3_tags = [tag["technique_id"] for row in a3_sample.to_dict(orient="records")
               for tag in map_agent3_event(row, techniques)]

    from collections import Counter
    print(f"[orchestrator] Agent 1 attack rows tagged: {Counter(a1_tags)}")
    print(f"[orchestrator] Agent 3 attack rows tagged: {Counter(a3_tags)}")
