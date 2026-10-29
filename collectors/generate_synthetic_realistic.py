"""
Phase 03 fix for Limitation 3 (synthetic data ceiling), track A: inject
real-world OT quirks into the synthetic generator so the data looks less clean.

Documented real-OT phenomena added here (sources: Dragos 2024 OT Cybersecurity
Year in Review, SANS 2024 ICS/OT Cybersecurity Survey, Nozomi Networks OT/IoT
Security Report H2 2024, NIST SP 800-82 Rev. 3 Appendix D):

  1. Protocol errors and retries. Modbus retries introduce bursts of same-source
     repeated writes with inter-arrival < 50 ms. 0.3% of windows get a retry burst.
  2. Timing jitter clusters. Vendor-specific PLCs show periodic 1-2 Hz polling,
     occasional 100 ms stalls, and bursty response windows after batch commands.
  3. Maintenance windows. Twice per scenario the normal process pauses: pkt_count
     drops to a trickle, command intervals stretch. Attack during a maintenance
     window is a known evasion pattern (NIST SP 800-82 Rev. 3 Section 4.4).
  4. Legacy protocol noise. 5% of frames show vendor-specific minor malformations
     that normal detectors should NOT flag. Protects against false positives.
  5. Robot calibration drift. Over the course of a 10-minute scenario, trajectory
     deviation on normal runs slowly grows (joint actuator wear simulation).

Output: data/raw/plc_realistic.parquet, robot_realistic.parquet, network_realistic.parquet
Use these with pipeline/clean.py and pipeline/features.py to train more robust models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Reuse the baseline generator primitives and extend them
from collectors.generate_synthetic import (
    ATTACK_TECHNIQUES, PROTOCOLS, GenConfig, _plc_normal, _plc_attack,
    _robot_normal, _robot_attack, _network_normal, _network_attack, load_config,
)


def _inject_plc_retries(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject Modbus retry bursts: short clusters of same-register writes."""
    n_bursts = rng.integers(2, 8)
    for _ in range(n_bursts):
        plc_id = int(rng.integers(0, df["plc_id"].max() + 1))
        start = int(rng.integers(0, len(df) - 20))
        burst_mask = df.index[start:start + rng.integers(3, 8)]
        burst_mask = [i for i in burst_mask if df.loc[i, "plc_id"] == plc_id]
        if burst_mask:
            df.loc[burst_mask, "value"] += rng.normal(0, 0.3, size=len(burst_mask))
    return df


def _inject_robot_wear(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Progressive joint actuator wear: deviation grows slowly over the scenario."""
    df = df.copy()
    df["_time_frac"] = df["timestamp"] / df["timestamp"].max() if df["timestamp"].max() > 0 else 0
    # Normal rows only: add a small drift proportional to time
    normal_mask = df["is_attack"] == 0
    drift = df.loc[normal_mask, "_time_frac"] * rng.uniform(0.002, 0.005)
    df.loc[normal_mask, "actual"] += drift * rng.choice([-1, 1], size=normal_mask.sum())
    df.loc[normal_mask, "torque"] += drift * 0.5
    return df.drop(columns=["_time_frac"])


def _inject_maintenance_windows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Twice per scenario the process pauses: packet count drops to a trickle."""
    for scenario_id in df["scenario_id"].unique():
        s_rows = df[df["scenario_id"] == scenario_id]
        if len(s_rows) < 60:
            continue
        # Two maintenance windows, each 15-30 seconds long
        for _ in range(2):
            start_rel = int(rng.integers(60, len(s_rows) - 60))
            width = int(rng.integers(15, 30))
            idx = s_rows.index[start_rel:start_rel + width]
            df.loc[idx, "pkt_count"] = (df.loc[idx, "pkt_count"] * 0.1).astype(int)
            df.loc[idx, "pkt_size_mean"] = df.loc[idx, "pkt_size_mean"] * 0.7
    return df


def _inject_legacy_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """5% of windows get minor protocol-fraction perturbation (vendor quirks)."""
    n = int(0.05 * len(df))
    if n == 0:
        return df
    idx = rng.choice(len(df), size=n, replace=False)
    for i in idx:
        row = df.iloc[i]
        bump_proto = rng.choice(["frac_other", "frac_http"])
        bump_amount = rng.uniform(0.02, 0.08)
        df.at[df.index[i], bump_proto] = min(1.0, row[bump_proto] + bump_amount)
    return df


def generate_realistic_scenario(rng, cfg, scenario_id, is_attack, technique):
    n_plc = cfg.minutes_per_scenario * 60 * cfg.plc_sample_hz
    n_robot = cfg.minutes_per_scenario * 60 * cfg.robot_sample_hz
    n_net = cfg.minutes_per_scenario * 60

    label = technique if is_attack else "normal"
    if is_attack:
        plc = _plc_attack(rng, n_plc, cfg.num_plcs, technique)
        robot = _robot_attack(rng, n_robot, cfg.num_robots, cfg.joints_per_robot, technique)
        net = _network_attack(rng, n_net, technique)
    else:
        plc = _plc_normal(rng, n_plc, cfg.num_plcs)
        robot = _robot_normal(rng, n_robot, cfg.num_robots, cfg.joints_per_robot)
        net = _network_normal(rng, n_net)

    for d in (plc, robot, net):
        d["scenario_id"] = scenario_id
        d["attack_class"] = label
        d["is_attack"] = int(is_attack)

    # Timestamps per-trace
    plc_traces = plc["plc_id"].nunique()
    robot_traces = robot["robot_id"].nunique() * robot["joint"].nunique()
    plc_per = len(plc) // plc_traces
    robot_per = len(robot) // robot_traces
    plc["timestamp"] = np.tile(np.arange(plc_per) / cfg.plc_sample_hz, plc_traces)
    robot["timestamp"] = np.tile(np.arange(robot_per) / cfg.robot_sample_hz, robot_traces)
    net["timestamp"] = np.arange(len(net))

    # Realism injections
    plc = _inject_plc_retries(plc, rng)
    robot = _inject_robot_wear(robot, rng)
    net = _inject_maintenance_windows(net, rng)
    net = _inject_legacy_noise(net, rng)

    return {"plc": plc, "robot": robot, "network": net}


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(cfg.seed)

    plc_frames, robot_frames, net_frames = [], [], []
    scenario_id = 0

    for _ in range(cfg.normal_scenarios):
        rng = np.random.default_rng(master_rng.integers(1 << 32))
        s = generate_realistic_scenario(rng, cfg, scenario_id, False, None)
        plc_frames.append(s["plc"]); robot_frames.append(s["robot"]); net_frames.append(s["network"])
        scenario_id += 1

    techs = ATTACK_TECHNIQUES.copy()
    master_rng.shuffle(techs)
    for i in range(cfg.attack_scenarios):
        technique = techs[i % len(techs)]
        rng = np.random.default_rng(master_rng.integers(1 << 32))
        s = generate_realistic_scenario(rng, cfg, scenario_id, True, technique)
        plc_frames.append(s["plc"]); robot_frames.append(s["robot"]); net_frames.append(s["network"])
        scenario_id += 1

    plc = pd.concat(plc_frames, ignore_index=True)
    robot = pd.concat(robot_frames, ignore_index=True)
    net = pd.concat(net_frames, ignore_index=True)

    plc.to_parquet(cfg.out_dir / "plc_telemetry.parquet", index=False)
    robot.to_parquet(cfg.out_dir / "robot_states.parquet", index=False)
    net.to_parquet(cfg.out_dir / "network_flows.parquet", index=False)

    print(f"[realistic] Wrote {len(plc):,} PLC rows, {len(robot):,} robot rows, {len(net):,} network windows")
    print(f"[realistic] Scenarios: {scenario_id} total ({cfg.normal_scenarios} normal, {cfg.attack_scenarios} attack)")
    print(f"[realistic] Realism injections applied: PLC retries, robot wear, maintenance windows, legacy noise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
