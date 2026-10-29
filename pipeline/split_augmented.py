"""
Stratified train/val/test split for the augmented Agent 1 features.
Mirrors pipeline/split.py but uses agent1_features_augmented.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    seed = cfg["seed"]
    s = cfg["split"]

    df = pd.read_parquet(proc / "agent1_features_augmented.parquet")
    stratify = df[s["stratify_col"]]
    train, temp = train_test_split(df, train_size=s["train_frac"],
                                    stratify=stratify, random_state=seed)
    val_rel = s["val_frac"] / (s["val_frac"] + s["test_frac"])
    val, test = train_test_split(temp, train_size=val_rel,
                                  stratify=temp[s["stratify_col"]], random_state=seed)

    train.to_parquet(proc / "agent1_augmented_train.parquet", index=False)
    val.to_parquet(proc / "agent1_augmented_val.parquet", index=False)
    test.to_parquet(proc / "agent1_augmented_test.parquet", index=False)
    print(f"[split-aug] train={len(train):,} val={len(val):,} test={len(test):,}")


if __name__ == "__main__":
    main()
