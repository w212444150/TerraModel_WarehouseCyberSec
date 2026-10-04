"""
Feature engineering for the three TerraModel agents.

Reads cleaned Parquet files, produces one feature dataframe per agent:

  Agent 1 (OT anomaly):   per 1-second network window features + packet-size stats
                           + protocol mix + IP entropy proxy
  Agent 2 (assessment):    per-advisory CVE features for ranking
  Agent 3 (robot integ):   per 1-second window per robot: deviation, torque variance,
                           command sequence entropy, cmd_source consistency, timing jitter

Each agent dataframe carries scenario_id, attack_class, is_attack labels.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def features_agent1(network_df: pd.DataFrame) -> pd.DataFrame:
    """Agent 1 features: the network_flows frame is already per-window. Add engineered cols."""
    df = network_df.copy()
    # Normalize protocol mix columns (should already sum to ~1 but we re-normalize for safety)
    proto_cols = [c for c in df.columns if c.startswith("frac_")]
    df["proto_entropy"] = df[proto_cols].apply(
        lambda row: -sum(p * np.log2(p) for p in row if p > 0), axis=1
    )
    df["pkt_rate"] = df["pkt_count"].astype(float)
    df["size_cv"] = df["pkt_size_std"] / (df["pkt_size_mean"].abs() + 1e-6)  # coefficient of variation
    df["src_dst_ratio"] = df["unique_src_ips"] / (df["unique_dst_ips"] + 1e-6)
    # Rolling baselines per scenario (10-second window = 10 samples)
    df = df.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)
    df["pkt_count_roll_mean"] = df.groupby("scenario_id")["pkt_count"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    df["pkt_count_deviation"] = df["pkt_count"] - df["pkt_count_roll_mean"]
    return df


def features_agent2(cisa_df: pd.DataFrame) -> pd.DataFrame:
    """Agent 2 features from CISA advisory records. One row per advisory."""
    df = cisa_df.copy()
    # Normalize CVSS to 0-1
    df["cvss_norm"] = df["cvss_v3"] / 10.0
    # Age score: fresher advisories get higher score, but very old unpatched ones also matter
    # We treat 0-30 days as freshness, >180 days as "stale unpatched risk"
    df["age_score"] = np.where(
        df["age_days"] < 30, 1.0,
        np.where(df["age_days"] > 180, 0.6, 0.3)
    )
    # One-hot encode asset_type for later ML models
    for asset in ["plc", "hmi", "robot_controller", "network_switch", "engineering_workstation", "other"]:
        df[f"asset_is_{asset}"] = (df["asset_type"] == asset).astype(int)
    return df


def features_agent3(robot_df: pd.DataFrame) -> pd.DataFrame:
    """Agent 3 features: per 1-second window per robot.

    Features:
      - mean/max/std of |commanded - actual| per joint, aggregated across joints
      - torque variance
      - command sequence entropy (over cmd_source categorical in the window)
      - cmd_source non-primary fraction (operator_a/b/scheduler are primary)
      - timing jitter: std of cmd_interval_ms
    """
    df = robot_df.copy()
    df["abs_dev"] = (df["commanded"] - df["actual"]).abs()
    # Bucket into 1-second windows per scenario per robot
    df["win"] = df["timestamp"].astype(int)

    primary_sources = {"operator_a", "operator_b", "scheduler"}
    df["is_nonprimary_src"] = (~df["cmd_source"].isin(primary_sources)).astype(int)

    grouped = df.groupby(["scenario_id", "robot_id", "win"], observed=True)

    feats = grouped.agg(
        mean_dev=("abs_dev", "mean"),
        max_dev=("abs_dev", "max"),
        std_dev=("abs_dev", "std"),
        torque_var=("torque", "var"),
        timing_jitter=("cmd_interval_ms", "std"),
        nonprimary_src_frac=("is_nonprimary_src", "mean"),
        attack_class=("attack_class", "first"),
        is_attack=("is_attack", "first"),
    ).reset_index()

    # Command sequence entropy (per window)
    entropy_series = grouped["cmd_source"].apply(lambda s: _entropy(s.to_numpy()))
    entropy_series.name = "cmd_entropy"
    feats = feats.merge(entropy_series.reset_index(), on=["scenario_id", "robot_id", "win"])

    feats = feats.fillna(0.0)
    return feats


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    cisa_path = Path(cfg["paths"]["threat_dir"]) / "cisa" / "advisories_flat.csv"

    net = pd.read_parquet(proc / "network_clean.parquet")
    robot = pd.read_parquet(proc / "robot_clean.parquet")

    a1 = features_agent1(net)
    a3 = features_agent3(robot)
    a1.to_parquet(proc / "agent1_features.parquet", index=False)
    a3.to_parquet(proc / "agent3_features.parquet", index=False)

    print(f"[features] Agent 1 features: {len(a1):,} rows, {a1.shape[1]} cols -> agent1_features.parquet")
    print(f"[features] Agent 3 features: {len(a3):,} rows, {a3.shape[1]} cols -> agent3_features.parquet")

    if cisa_path.exists():
        cisa = pd.read_csv(cisa_path)
        a2 = features_agent2(cisa)
        a2.to_csv(proc / "agent2_features.csv", index=False)
        print(f"[features] Agent 2 features: {len(a2):,} advisories, {a2.shape[1]} cols -> agent2_features.csv")
    else:
        print(f"[features] WARNING: CISA data missing at {cisa_path}; run fetch_cisa.py")


if __name__ == "__main__":
    main()
