"""
Phase 03 fix for Limitation 1: give Agent 1 PLC visibility.

Phase 02/early-Phase 03 Agent 1 saw only network flow summaries. Attacks that
hit the PLC without changing network behavior (T0836 Modify Parameter, T0832
Manipulation of View) stayed invisible, capping Agent 1 F1 near 0.49.

This module joins per-second PLC statistics into the Agent 1 feature frame.
New features added:

  plc_val_mean         - mean PLC register value in the window across all PLCs
  plc_val_std          - std of PLC register values
  plc_val_range        - max minus min
  plc_setpoint_delta   - mean absolute deviation from setpoint (catches T0836 directly)
  plc_outofband_frac   - fraction of PLC samples outside the physical safe band

Output: data/processed/agent1_features_augmented.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SAFE_BAND = (10.0, 150.0)  # Conveyor speed band in the synthetic generator


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])

    plc = pd.read_parquet(proc / "plc_clean.parquet")
    a1 = pd.read_parquet(proc / "agent1_features.parquet")

    plc["win"] = plc["timestamp"].astype(int)
    plc["abs_delta"] = (plc["value"] - plc["setpoint"]).abs()
    plc["out_of_band"] = (
        (plc["value"] < SAFE_BAND[0]) | (plc["value"] > SAFE_BAND[1])
    ).astype(int)

    stats = plc.groupby(["scenario_id", "win"], observed=True).agg(
        plc_val_mean=("value", "mean"),
        plc_val_std=("value", "std"),
        plc_val_range=("value", lambda s: float(s.max() - s.min())),
        plc_setpoint_delta=("abs_delta", "mean"),
        plc_outofband_frac=("out_of_band", "mean"),
    ).reset_index().rename(columns={"win": "timestamp"})

    augmented = a1.merge(stats, on=["scenario_id", "timestamp"], how="left").fillna(0.0)
    augmented.to_parquet(proc / "agent1_features_augmented.parquet", index=False)

    added = [c for c in augmented.columns if c not in a1.columns]
    print(f"[augment] Agent 1 features {a1.shape[1]} -> {augmented.shape[1]} columns")
    print(f"[augment] Added: {added}")
    print(f"[augment] Rows: {len(augmented):,}")
    print(f"[augment] Saved to data/processed/agent1_features_augmented.parquet")


if __name__ == "__main__":
    main()
