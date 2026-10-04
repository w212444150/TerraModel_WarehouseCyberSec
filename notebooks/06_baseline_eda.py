"""
EDA + baseline visualization script.

Produces three PNG charts under reports/:
  - reports/agent1_proto_entropy.png - protocol entropy distribution by class
  - reports/agent1_pkt_count_by_class.png - packet count distribution by class
  - reports/agent3_mean_dev.png - robot trajectory deviation by class

Also runs the three baselines in sequence and consolidates a reports/metrics_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def plot_distribution_by_class(df: pd.DataFrame, col: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in df["attack_class"].unique():
        subset = df[df["attack_class"] == cls]
        ax.hist(subset[col], bins=40, alpha=0.5, label=cls)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[eda] Wrote {out}")


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    a1 = pd.read_parquet(proc / "agent1_features.parquet")
    a3 = pd.read_parquet(proc / "agent3_features.parquet")

    plot_distribution_by_class(a1, "pkt_count", "Agent 1: packet count by scenario class",
                               reports / "agent1_pkt_count_by_class.png")
    plot_distribution_by_class(a1, "proto_entropy", "Agent 1: protocol entropy by scenario class",
                               reports / "agent1_proto_entropy.png")
    plot_distribution_by_class(a3, "mean_dev", "Agent 3: mean |cmd - actual| by scenario class",
                               reports / "agent3_mean_dev.png")

    # Collect metric JSONs produced by the baselines
    summary = {}
    for name in ("agent1_metrics.json", "agent2_metrics.json", "agent3_metrics.json"):
        path = reports / name
        if path.exists():
            summary[name.replace("_metrics.json", "")] = json.loads(path.read_text())
    (reports / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[eda] Wrote consolidated metrics -> {reports / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
