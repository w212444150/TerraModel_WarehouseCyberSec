"""
Download the official MITRE ATT&CK for ICS STIX bundle.

Source: https://github.com/mitre-attack/attack-stix-data (CC BY 4.0 license).

Writes the raw bundle plus a flat lookup JSON of the techniques TerraModel targets:
  {technique_id: {name, description, url, tactics}}
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
import yaml

TARGET_TECHNIQUES = {"T0836", "T0855", "T0832", "T0814", "T0859", "T0866", "T0863"}


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    url = cfg["threat_sources"]["attack_stix_url"]
    threat_dir = Path(cfg["paths"]["threat_dir"]) / "attack"
    threat_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_attack] Downloading {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    bundle = resp.json()
    bundle_path = threat_dir / "ics-attack.json"
    bundle_path.write_text(json.dumps(bundle, indent=2))
    print(f"[fetch_attack] Wrote full STIX bundle -> {bundle_path} ({bundle_path.stat().st_size // 1024} KB)")

    # Build flat lookup
    lookup = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id", "")
                if tid in TARGET_TECHNIQUES:
                    lookup[tid] = {
                        "name": obj.get("name", ""),
                        "description": (obj.get("description", "") or "").split("\n")[0][:400],
                        "url": ref.get("url", ""),
                        "tactics": [p.get("phase_name") for p in obj.get("kill_chain_phases", [])],
                    }
                break

    lookup_path = threat_dir / "techniques_lookup.json"
    lookup_path.write_text(json.dumps(lookup, indent=2))
    print(f"[fetch_attack] Wrote target-technique lookup -> {lookup_path}")
    print(f"[fetch_attack] Techniques resolved: {sorted(lookup.keys())}")
    missing = TARGET_TECHNIQUES - set(lookup.keys())
    if missing:
        print(f"[fetch_attack] WARNING: missing technique IDs {sorted(missing)}")


if __name__ == "__main__":
    main()
