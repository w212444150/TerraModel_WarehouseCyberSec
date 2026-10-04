"""
Synthetic warehouse OT telemetry generator for TerraModel.

Replaces (for Phase 02 / Phase 03 baseline work) the Factory I/O + OpenPLC + ROS 2
simulation stack described in the project proposal. Produces three parallel data
streams that mirror the shape of real OT telemetry:

  1. PLC register samples (Modbus TCP style)
  2. Robot joint-state samples (commanded vs actual position, torque)
  3. Network flow summaries (per-second packet counts, size distribution, protocol mix)

Each scenario is either 'normal' or 'attack'. Attack scenarios inject one of the
seven MITRE ATT&CK for ICS techniques TerraModel targets:

  T0836 Modify Parameter          - PLC setpoint pushed outside safe range
  T0855 Unauthorized Command Msg  - Spoofed Modbus writes to robot controller
  T0832 Manipulation of View      - PLC reports normal while sensor drifts
  T0814 Denial of Service         - Packet flood overwhelms a PLC
  T0859 Valid Accounts            - Elevated auth events from engineering workstation
  T0866 Exploit Remote Services   - Repeated connection attempts to remote service
  T0863 User Execution            - Workstation-origin anomalous command burst

Output: Parquet files in data/raw/.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ATT&CK ICS techniques injected during attack scenarios
ATTACK_TECHNIQUES = [
    "T0836",  # Modify Parameter
    "T0855",  # Unauthorized Command Message
    "T0832",  # Manipulation of View
    "T0814",  # Denial of Service
    "T0859",  # Valid Accounts
    "T0866",  # Exploitation of Remote Services
    "T0863",  # User Execution
]

PROTOCOLS = ["modbus_tcp", "ethernet_ip", "profinet", "http", "other"]


@dataclass
class GenConfig:
    seed: int
    normal_scenarios: int
    attack_scenarios: int
    minutes_per_scenario: int
    plc_sample_hz: int
    robot_sample_hz: int
    network_sample_hz: int
    num_plcs: int
    num_robots: int
    joints_per_robot: int
    out_dir: Path


def load_config(path: str = "config.yaml") -> GenConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    g = cfg["data_generation"]
    return GenConfig(
        seed=cfg["seed"],
        normal_scenarios=g["normal_scenarios"],
        attack_scenarios=g["attack_scenarios"],
        minutes_per_scenario=g["minutes_per_scenario"],
        plc_sample_hz=g["plc_sample_hz"],
        robot_sample_hz=g["robot_sample_hz"],
        network_sample_hz=g["network_sample_hz"],
        num_plcs=g["num_plcs"],
        num_robots=g["num_robots"],
        joints_per_robot=g["joints_per_robot"],
        out_dir=Path(cfg["paths"]["raw_dir"]),
    )


def _plc_normal(rng: np.random.Generator, n_samples: int, num_plcs: int) -> pd.DataFrame:
    """Normal PLC telemetry: register values with mild noise around a conveyor setpoint."""
    rows = []
    base_setpoints = rng.uniform(40, 80, size=num_plcs)  # conveyor speeds, say
    for plc_id in range(num_plcs):
        setpoint = base_setpoints[plc_id]
        noise = rng.normal(0, 1.5, size=n_samples)
        values = setpoint + noise
        rows.append(pd.DataFrame({
            "plc_id": plc_id,
            "register": "conveyor_speed",
            "value": values,
            "setpoint": setpoint,
        }))
    return pd.concat(rows, ignore_index=True)


def _plc_attack(rng: np.random.Generator, n_samples: int, num_plcs: int, technique: str) -> pd.DataFrame:
    """Attack-affected PLC telemetry. The attack shape depends on technique."""
    df = _plc_normal(rng, n_samples, num_plcs)
    if technique == "T0836":  # Modify Parameter - setpoint pushed out of band
        affected_plc = rng.integers(0, num_plcs)
        mask = (df["plc_id"] == affected_plc)
        # Push the first half of the samples to an unsafe value
        half = n_samples // 2
        idx = df[mask].index[:half]
        df.loc[idx, "value"] = df.loc[idx, "setpoint"] * 1.6 + rng.normal(0, 1.5, size=half)
    elif technique == "T0832":  # Manipulation of View - sensor drifts away from register
        affected_plc = rng.integers(0, num_plcs)
        mask = (df["plc_id"] == affected_plc)
        drift = rng.normal(5.0, 1.0, size=mask.sum())
        df.loc[mask, "value"] = df.loc[mask, "value"] + drift
    # Other techniques do not affect PLC register stream directly in this baseline
    return df


def _robot_normal(rng: np.random.Generator, n_samples: int, num_robots: int, joints: int) -> pd.DataFrame:
    """Normal robot joint states. commanded ~= actual, small timing jitter."""
    rows = []
    t = np.arange(n_samples) / 20.0  # time in seconds at 20 Hz
    for robot_id in range(num_robots):
        for joint in range(joints):
            # A gentle sinusoidal commanded path
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.3, 0.8)
            freq = rng.uniform(0.1, 0.4)
            commanded = amp * np.sin(2 * np.pi * freq * t + phase)
            # Actual tracks commanded with small noise and tiny lag
            actual = np.roll(commanded, 1) + rng.normal(0, 0.005, size=n_samples)
            torque = np.abs(commanded) * 2.5 + rng.normal(0, 0.08, size=n_samples)
            cmd_source = rng.choice(["operator_a", "operator_b", "scheduler"],
                                    size=n_samples, p=[0.4, 0.4, 0.2])
            rows.append(pd.DataFrame({
                "robot_id": robot_id,
                "joint": joint,
                "commanded": commanded,
                "actual": actual,
                "torque": torque,
                "cmd_source": cmd_source,
                "cmd_interval_ms": rng.normal(50, 2, size=n_samples),
            }))
    return pd.concat(rows, ignore_index=True)


def _robot_attack(rng: np.random.Generator, n_samples: int, num_robots: int,
                  joints: int, technique: str) -> pd.DataFrame:
    """Robot stream under attack."""
    df = _robot_normal(rng, n_samples, num_robots, joints)
    if technique == "T0855":  # Unauthorized Command Message - spoofed writes
        affected = rng.integers(0, num_robots)
        mask = (df["robot_id"] == affected)
        # Attack affects a middle window of samples
        affected_idx = df[mask].index
        window = affected_idx[len(affected_idx) // 4: len(affected_idx) // 4 * 3]
        # Spoofed command: pushes actual away from commanded, odd cmd_source
        df.loc[window, "actual"] = df.loc[window, "commanded"] + rng.normal(0.08, 0.02, size=len(window))
        df.loc[window, "cmd_source"] = "unknown_src"
        df.loc[window, "cmd_interval_ms"] = rng.normal(5, 1, size=len(window))  # burst timing
    elif technique == "T0863":  # User Execution burst
        # Random bursts of commands from workstation source
        burst_count = rng.integers(20, 60)
        burst_idx = rng.choice(len(df), size=burst_count, replace=False)
        df.loc[burst_idx, "cmd_source"] = "engineering_workstation"
        df.loc[burst_idx, "cmd_interval_ms"] = rng.normal(3, 0.5, size=burst_count)
    return df


def _network_normal(rng: np.random.Generator, n_windows: int) -> pd.DataFrame:
    """Normal 1-second network flow summaries."""
    base_pkt_count = rng.normal(600, 60, size=n_windows)
    pkt_count = np.maximum(base_pkt_count, 50).astype(int)
    pkt_size_mean = rng.normal(220, 20, size=n_windows)
    pkt_size_std = rng.normal(60, 8, size=n_windows)
    inter_arrival_mean = rng.normal(1.6, 0.2, size=n_windows)  # ms
    unique_srcs = rng.integers(4, 10, size=n_windows)
    unique_dsts = rng.integers(4, 12, size=n_windows)

    # Protocol mix (Modbus dominates, HTTP small)
    proto_mix = rng.dirichlet([5.0, 2.0, 1.5, 0.5, 0.3], size=n_windows)

    df = pd.DataFrame({
        "pkt_count": pkt_count,
        "pkt_size_mean": pkt_size_mean,
        "pkt_size_std": pkt_size_std,
        "inter_arrival_mean_ms": inter_arrival_mean,
        "unique_src_ips": unique_srcs,
        "unique_dst_ips": unique_dsts,
    })
    for i, p in enumerate(PROTOCOLS):
        df[f"frac_{p}"] = proto_mix[:, i]
    return df


def _network_attack(rng: np.random.Generator, n_windows: int, technique: str) -> pd.DataFrame:
    df = _network_normal(rng, n_windows)
    affected_idx = range(n_windows // 3, 2 * n_windows // 3)

    if technique == "T0814":  # Denial of Service - packet flood
        df.loc[affected_idx, "pkt_count"] = (df.loc[affected_idx, "pkt_count"] * rng.uniform(5, 9, size=len(affected_idx))).astype(int)
        df.loc[affected_idx, "pkt_size_mean"] = rng.normal(90, 10, size=len(affected_idx))
        df.loc[affected_idx, "inter_arrival_mean_ms"] = rng.normal(0.2, 0.05, size=len(affected_idx))
    elif technique == "T0855":  # Unauthorized Command - unusual src IP entropy
        df.loc[affected_idx, "unique_src_ips"] = df.loc[affected_idx, "unique_src_ips"] + rng.integers(5, 15, size=len(affected_idx))
        # Boost Modbus share
        df.loc[affected_idx, "frac_modbus_tcp"] = np.clip(df.loc[affected_idx, "frac_modbus_tcp"] + 0.2, 0, 1)
    elif technique == "T0866":  # Exploit Remote Services - repeated connections
        df.loc[affected_idx, "unique_dst_ips"] = df.loc[affected_idx, "unique_dst_ips"] + rng.integers(10, 25, size=len(affected_idx))
        df.loc[affected_idx, "frac_other"] = np.clip(df.loc[affected_idx, "frac_other"] + 0.15, 0, 1)
    elif technique == "T0859":  # Valid Accounts - subtle - shift src distribution
        df.loc[affected_idx, "unique_src_ips"] = df.loc[affected_idx, "unique_src_ips"] + rng.integers(1, 4, size=len(affected_idx))
    return df


def generate_scenario(rng: np.random.Generator, cfg: GenConfig, scenario_id: int,
                      is_attack: bool, technique: str | None) -> dict[str, pd.DataFrame]:
    n_plc = cfg.minutes_per_scenario * 60 * cfg.plc_sample_hz
    n_robot = cfg.minutes_per_scenario * 60 * cfg.robot_sample_hz
    n_net = cfg.minutes_per_scenario * 60  # 1-second windows

    label = technique if is_attack else "normal"

    if is_attack:
        plc_df = _plc_attack(rng, n_plc, cfg.num_plcs, technique)
        robot_df = _robot_attack(rng, n_robot, cfg.num_robots, cfg.joints_per_robot, technique)
        net_df = _network_attack(rng, n_net, technique)
    else:
        plc_df = _plc_normal(rng, n_plc, cfg.num_plcs)
        robot_df = _robot_normal(rng, n_robot, cfg.num_robots, cfg.joints_per_robot)
        net_df = _network_normal(rng, n_net)

    # Tag every row with scenario metadata
    for df in (plc_df, robot_df, net_df):
        df["scenario_id"] = scenario_id
        df["attack_class"] = label
        df["is_attack"] = int(is_attack)

    # Add timestamps. Each (plc_id, register) or (robot_id, joint) trace is one time series,
    # so timestamps repeat per trace.
    plc_traces = plc_df["plc_id"].nunique()
    robot_traces = robot_df["robot_id"].nunique() * robot_df["joint"].nunique()
    plc_t_per_trace = len(plc_df) // plc_traces
    robot_t_per_trace = len(robot_df) // robot_traces
    plc_df["timestamp"] = np.tile(np.arange(plc_t_per_trace) / cfg.plc_sample_hz, plc_traces)
    robot_df["timestamp"] = np.tile(np.arange(robot_t_per_trace) / cfg.robot_sample_hz, robot_traces)
    net_df["timestamp"] = np.arange(len(net_df))

    return {"plc": plc_df, "robot": robot_df, "network": net_df}


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(cfg.seed)

    plc_frames, robot_frames, net_frames = [], [], []
    scenario_id = 0

    # Normal scenarios
    for _ in range(cfg.normal_scenarios):
        rng = np.random.default_rng(master_rng.integers(1 << 32))
        s = generate_scenario(rng, cfg, scenario_id, is_attack=False, technique=None)
        plc_frames.append(s["plc"])
        robot_frames.append(s["robot"])
        net_frames.append(s["network"])
        scenario_id += 1

    # Attack scenarios - cycle through techniques so each gets at least one scenario
    techniques_shuffled = ATTACK_TECHNIQUES.copy()
    master_rng.shuffle(techniques_shuffled)
    for i in range(cfg.attack_scenarios):
        technique = techniques_shuffled[i % len(techniques_shuffled)]
        rng = np.random.default_rng(master_rng.integers(1 << 32))
        s = generate_scenario(rng, cfg, scenario_id, is_attack=True, technique=technique)
        plc_frames.append(s["plc"])
        robot_frames.append(s["robot"])
        net_frames.append(s["network"])
        scenario_id += 1

    plc_all = pd.concat(plc_frames, ignore_index=True)
    robot_all = pd.concat(robot_frames, ignore_index=True)
    net_all = pd.concat(net_frames, ignore_index=True)

    plc_path = cfg.out_dir / "plc_telemetry.parquet"
    robot_path = cfg.out_dir / "robot_states.parquet"
    net_path = cfg.out_dir / "network_flows.parquet"
    plc_all.to_parquet(plc_path, index=False)
    robot_all.to_parquet(robot_path, index=False)
    net_all.to_parquet(net_path, index=False)

    print(f"[generate_synthetic] Wrote {len(plc_all):,} PLC rows -> {plc_path}")
    print(f"[generate_synthetic] Wrote {len(robot_all):,} robot rows -> {robot_path}")
    print(f"[generate_synthetic] Wrote {len(net_all):,} network windows -> {net_path}")
    print(f"[generate_synthetic] Scenarios: {scenario_id} total "
          f"({cfg.normal_scenarios} normal, {cfg.attack_scenarios} attack)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
