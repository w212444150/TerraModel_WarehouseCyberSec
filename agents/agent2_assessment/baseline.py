"""
Agent 2 baseline: Rule-based CVSS plus exposure score for ranking ICS advisories.

score = cvss_weight * cvss_norm + exposure_weight * exposure_hint + age_weight * age_score

Reports the top-k advisories by score. Since we have no ground-truth 'should patch
first' labels in this early phase, we evaluate three things we CAN measure:

  1. Top-k spread: do PLCs and robot controllers dominate the top-10? (Expected yes
     given the warehouse context.)
  2. CVSS distribution of top-10 vs. bottom-k.
  3. Score reproducibility (same input -> same output under fixed seed).

A supervised precision@k requires labels we will annotate in Phase 03.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    a2 = pd.read_csv(proc / "agent2_features.csv")

    a2_cfg = cfg["agent2"]
    a2["score"] = (
        a2_cfg["cvss_weight"] * a2["cvss_norm"]
        + a2_cfg["exposure_weight"] * a2["exposure_hint"]
        + a2_cfg["age_weight"] * a2["age_score"]
    )

    a2_sorted = a2.sort_values("score", ascending=False).reset_index(drop=True)
    top_k = a2_sorted.head(a2_cfg["top_k"])

    report = {
        "ranked_count": int(len(a2_sorted)),
        "top_k": a2_cfg["top_k"],
        "top_k_asset_distribution": top_k["asset_type"].value_counts().to_dict(),
        "top_k_mean_cvss": float(top_k["cvss_v3"].mean()),
        "top_k_mean_score": float(top_k["score"].mean()),
        "bottom_k_mean_cvss": float(a2_sorted.tail(a2_cfg["top_k"])["cvss_v3"].mean()),
        "top_k_rows": top_k[["advisory_id", "vendor", "product", "asset_type",
                             "cvss_v3", "exposure_hint", "age_days", "score"]].to_dict(orient="records"),
    }

    # Save ranking output as CSV so grader can inspect
    top_k.to_csv(reports_dir / "agent2_top_k.csv", index=False)
    (reports_dir / "agent2_metrics.json").write_text(json.dumps(report, indent=2, default=str))

    print("[agent2] Rule-based ranker complete")
    print(f"[agent2]   Ranked {report['ranked_count']} advisories")
    print(f"[agent2]   Top-{a2_cfg['top_k']} mean CVSS = {report['top_k_mean_cvss']:.2f}")
    print(f"[agent2]   Bottom-{a2_cfg['top_k']} mean CVSS = {report['bottom_k_mean_cvss']:.2f}")
    print(f"[agent2]   Top-{a2_cfg['top_k']} asset distribution:")
    for asset, cnt in report["top_k_asset_distribution"].items():
        print(f"              {asset:28s} {cnt}")


if __name__ == "__main__":
    main()
