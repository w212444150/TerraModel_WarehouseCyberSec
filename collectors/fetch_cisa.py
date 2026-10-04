"""
Fetch CISA ICS Advisories as a structured dataset for Agent 2.

CISA publishes advisories at https://www.cisa.gov/news-events/ics-advisories.
The JSON index is at /cybersecurity-advisories/ics-advisories.json. If that is unreachable,
this script falls back to a bundled sample dataset so the pipeline still runs end to end.

Writes data/threat/cisa/advisories.json plus a flat CSV of (advisory_id, product, vendor,
cvss_v3, exposure_hint, age_days, title, url).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

FALLBACK_SAMPLE = [
    # Representative sample derived from CISA 2024-2025 ICS advisory patterns
    # (used only if network is unavailable at run time). Shape matches live feed fields.
    {"id": "ICSA-25-105-01", "title": "Siemens SIMATIC CP", "vendor": "Siemens", "product": "SIMATIC CP", "cvss_v3": 9.1, "released": "2025-04-15", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-105-01"},
    {"id": "ICSA-25-105-02", "title": "Rockwell FactoryTalk View", "vendor": "Rockwell Automation", "product": "FactoryTalk View", "cvss_v3": 8.6, "released": "2025-04-15", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-105-02"},
    {"id": "ICSA-25-098-03", "title": "Schneider Electric Modicon PLC", "vendor": "Schneider Electric", "product": "Modicon PLC", "cvss_v3": 7.5, "released": "2025-04-08", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-098-03"},
    {"id": "ICSA-25-091-01", "title": "ABB Robotics IRC5", "vendor": "ABB", "product": "IRC5 Robot Controller", "cvss_v3": 8.1, "released": "2025-04-01", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-091-01"},
    {"id": "ICSA-25-084-02", "title": "Mitsubishi MELSEC HMI", "vendor": "Mitsubishi Electric", "product": "MELSEC HMI", "cvss_v3": 6.8, "released": "2025-03-25", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-084-02"},
    {"id": "ICSA-25-077-01", "title": "FANUC CNC Robot", "vendor": "FANUC", "product": "CNC Robot Interface", "cvss_v3": 7.2, "released": "2025-03-18", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-077-01"},
    {"id": "ICSA-25-070-04", "title": "Omron NX1P2 PLC", "vendor": "Omron", "product": "NX1P2 PLC", "cvss_v3": 8.8, "released": "2025-03-11", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-070-04"},
    {"id": "ICSA-25-063-02", "title": "Siemens SCALANCE Switch", "vendor": "Siemens", "product": "SCALANCE Switch", "cvss_v3": 7.0, "released": "2025-03-04", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-063-02"},
    {"id": "ICSA-25-056-01", "title": "Rockwell ControlLogix", "vendor": "Rockwell Automation", "product": "ControlLogix PLC", "cvss_v3": 9.3, "released": "2025-02-25", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-056-01"},
    {"id": "ICSA-25-049-03", "title": "KUKA KR C4 Controller", "vendor": "KUKA", "product": "KR C4 Robot Controller", "cvss_v3": 7.8, "released": "2025-02-18", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-049-03"},
    {"id": "ICSA-25-042-01", "title": "GE Fanuc PACSystem", "vendor": "GE Fanuc", "product": "PACSystem RX3i PLC", "cvss_v3": 8.2, "released": "2025-02-11", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-042-01"},
    {"id": "ICSA-25-035-02", "title": "Phoenix Contact Industrial Switch", "vendor": "Phoenix Contact", "product": "FL Switch", "cvss_v3": 6.5, "released": "2025-02-04", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-035-02"},
    {"id": "ICSA-25-028-01", "title": "Hitachi Energy MicroSCADA", "vendor": "Hitachi Energy", "product": "MicroSCADA Pro", "cvss_v3": 7.5, "released": "2025-01-28", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-028-01"},
    {"id": "ICSA-25-021-02", "title": "Yokogawa STARDOM Controller", "vendor": "Yokogawa", "product": "STARDOM FCN Controller", "cvss_v3": 8.0, "released": "2025-01-21", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-021-02"},
    {"id": "ICSA-25-014-01", "title": "Beckhoff TwinCAT PLC Runtime", "vendor": "Beckhoff", "product": "TwinCAT PLC Runtime", "cvss_v3": 8.5, "released": "2025-01-14", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-014-01"},
    {"id": "ICSA-25-007-03", "title": "Honeywell Experion PKS", "vendor": "Honeywell", "product": "Experion PKS", "cvss_v3": 7.4, "released": "2025-01-07", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-25-007-03"},
    {"id": "ICSA-24-355-01", "title": "Universal Robots UR Controller", "vendor": "Universal Robots", "product": "UR Robot Controller", "cvss_v3": 9.0, "released": "2024-12-20", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-355-01"},
    {"id": "ICSA-24-348-02", "title": "Delta Electronics ISPSoft", "vendor": "Delta Electronics", "product": "ISPSoft Programming Software", "cvss_v3": 7.3, "released": "2024-12-13", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-348-02"},
    {"id": "ICSA-24-341-04", "title": "WAGO Controller 750-xxx", "vendor": "WAGO", "product": "750-xxx Controller", "cvss_v3": 8.6, "released": "2024-12-06", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-341-04"},
    {"id": "ICSA-24-334-01", "title": "Red Lion Crimson HMI", "vendor": "Red Lion", "product": "Crimson HMI", "cvss_v3": 6.9, "released": "2024-11-29", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-334-01"},
    {"id": "ICSA-24-327-03", "title": "Yaskawa Motoman Robot Controller", "vendor": "Yaskawa", "product": "Motoman Robot Controller", "cvss_v3": 7.9, "released": "2024-11-22", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-327-03"},
    {"id": "ICSA-24-320-02", "title": "Keyence KV-8000 PLC", "vendor": "Keyence", "product": "KV-8000 PLC", "cvss_v3": 8.3, "released": "2024-11-15", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-320-02"},
    {"id": "ICSA-24-313-01", "title": "Moxa EDS Switch", "vendor": "Moxa", "product": "EDS Industrial Switch", "cvss_v3": 7.1, "released": "2024-11-08", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-313-01"},
    {"id": "ICSA-24-306-02", "title": "Emerson Ovation Controller", "vendor": "Emerson", "product": "Ovation Controller", "cvss_v3": 7.6, "released": "2024-11-01", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-306-02"},
    {"id": "ICSA-24-299-04", "title": "Advantech WebAccess SCADA", "vendor": "Advantech", "product": "WebAccess SCADA", "cvss_v3": 8.9, "released": "2024-10-25", "url": "https://www.cisa.gov/news-events/ics-advisories/icsa-24-299-04"},
]

# Asset-type keyword rules for classifying products (used by Agent 2 later)
ASSET_RULES = [
    ("robot", "robot_controller"),
    ("plc", "plc"),
    ("hmi", "hmi"),
    ("scada", "hmi"),
    ("switch", "network_switch"),
    ("workstation", "engineering_workstation"),
    ("programming", "engineering_workstation"),
    ("runtime", "engineering_workstation"),
]


def classify_asset(product: str) -> str:
    p = product.lower()
    for keyword, asset_type in ASSET_RULES:
        if keyword in p:
            return asset_type
    return "other"


def compute_exposure_hint(title: str, product: str) -> int:
    """Rough internet-exposure heuristic. 1 if remote/web/network terms appear."""
    s = (title + " " + product).lower()
    exposure_keywords = ["web", "remote", "cloud", "scada", "runtime"]
    return int(any(k in s for k in exposure_keywords))


def normalize_record(rec: dict, today: datetime) -> dict:
    released_str = rec.get("released", "")
    try:
        released = datetime.strptime(released_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        released = today
    age_days = (today - released).days
    product = rec.get("product", "unknown")
    return {
        "advisory_id": rec.get("id", ""),
        "title": rec.get("title", ""),
        "vendor": rec.get("vendor", "unknown"),
        "product": product,
        "asset_type": classify_asset(product),
        "cvss_v3": float(rec.get("cvss_v3", 0.0)),
        "released": released.strftime("%Y-%m-%d"),
        "age_days": age_days,
        "exposure_hint": compute_exposure_hint(rec.get("title", ""), product),
        "url": rec.get("url", ""),
    }


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cisa_dir = Path(cfg["paths"]["threat_dir"]) / "cisa"
    cisa_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc)
    url = cfg["threat_sources"]["cisa_advisories_url"]

    records = []
    try:
        print(f"[fetch_cisa] Trying live feed: {url}")
        resp = requests.get(url, timeout=20, headers={"User-Agent": "TerraModel-capstone/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "advisories" in data:
                records = data["advisories"]
            print(f"[fetch_cisa] Live fetch succeeded: {len(records)} advisories")
        else:
            print(f"[fetch_cisa] Live feed returned HTTP {resp.status_code}; using fallback sample")
    except Exception as exc:
        print(f"[fetch_cisa] Live fetch failed ({exc}); using fallback sample")

    if not records:
        records = FALLBACK_SAMPLE
        print(f"[fetch_cisa] Using bundled fallback sample ({len(records)} advisories)")

    normalized = [normalize_record(r, today) for r in records]

    # Save raw and flat
    (cisa_dir / "advisories.json").write_text(json.dumps(records, indent=2))
    df = pd.DataFrame(normalized)
    df.to_csv(cisa_dir / "advisories_flat.csv", index=False)

    print(f"[fetch_cisa] Wrote {len(df)} advisories -> {cisa_dir}/advisories_flat.csv")
    print(f"[fetch_cisa] Asset type distribution:")
    for asset, cnt in df["asset_type"].value_counts().items():
        print(f"              {asset:28s} {cnt:4d}  ({cnt / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
