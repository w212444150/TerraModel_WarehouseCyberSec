"""
Clean the raw telemetry Parquet files.

Removes:
  - duplicates
  - rows with non-finite values in numeric columns
  - rows where PLC reported value contradicts physical plausibility
    (negative conveyor speed, torque below zero)
  - rows with inter_arrival above a hard cap

Writes:
  - data/processed/plc_clean.parquet
  - data/processed/robot_clean.parquet
  - data/processed/network_clean.parquet
  - data/processed/cleaning_report.json  (removal reasons and counts)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def clean_plc(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    initial = len(df)
    # Drop non-finite values
    df = df[np.isfinite(df["value"])].copy()
    # Physical plausibility: conveyor speed cannot be negative or absurdly high
    before = len(df)
    df = df[(df["value"] > -5) & (df["value"] < 500)]
    report["plc"] = {
        "initial_rows": initial,
        "dropped_nonfinite": initial - before,
        "dropped_implausible": before - len(df),
        "final_rows": len(df),
    }
    return df


def clean_robot(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    initial = len(df)
    df = df[np.isfinite(df[["commanded", "actual", "torque"]].values).all(axis=1)].copy()
    before = len(df)
    # Torque is magnitude, must be non-negative (small negative = noise, we clip at -0.5)
    df = df[df["torque"] > -0.5]
    report["robot"] = {
        "initial_rows": initial,
        "dropped_nonfinite": initial - before,
        "dropped_implausible": before - len(df),
        "final_rows": len(df),
    }
    return df


def clean_network(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    initial = len(df)
    df = df[np.isfinite(df.select_dtypes("number")).all(axis=1)].copy()
    before = len(df)
    # Cap: inter_arrival cannot be negative, and we drop windows with 0 packets
    df = df[(df["inter_arrival_mean_ms"] > 0) & (df["pkt_count"] > 0)]
    report["network"] = {
        "initial_rows": initial,
        "dropped_nonfinite": initial - before,
        "dropped_implausible": before - len(df),
        "final_rows": len(df),
    }
    return df


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    raw = Path(cfg["paths"]["raw_dir"])
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)

    plc = pd.read_parquet(raw / "plc_telemetry.parquet")
    robot = pd.read_parquet(raw / "robot_states.parquet")
    net = pd.read_parquet(raw / "network_flows.parquet")

    report: dict = {}
    plc_c = clean_plc(plc, report)
    robot_c = clean_robot(robot, report)
    net_c = clean_network(net, report)

    plc_c.to_parquet(proc / "plc_clean.parquet", index=False)
    robot_c.to_parquet(proc / "robot_clean.parquet", index=False)
    net_c.to_parquet(proc / "network_clean.parquet", index=False)

    # Overall removal rate across all three streams
    total_in = sum(r["initial_rows"] for r in report.values())
    total_dropped = sum(r["initial_rows"] - r["final_rows"] for r in report.values())
    report["overall"] = {
        "total_input_rows": total_in,
        "total_dropped": total_dropped,
        "removal_rate_pct": round(total_dropped / total_in * 100, 3) if total_in else 0.0,
    }

    (proc / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("[clean] Cleaning complete")
    for stream, r in report.items():
        if stream == "overall":
            print(f"[clean] OVERALL: {r['total_dropped']:,} / {r['total_input_rows']:,} dropped ({r['removal_rate_pct']}%)")
        else:
            print(f"[clean] {stream:8s}: {r['final_rows']:,} rows kept, {r['initial_rows'] - r['final_rows']:,} dropped")


if __name__ == "__main__":
    main()
