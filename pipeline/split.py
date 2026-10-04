"""
Train / validation / test split for Agent 1 and Agent 3 feature frames.

Stratified by attack_class so every attack technique appears in train, val, and test.
Random seed from config.yaml.

Agent 2 (CISA advisories) has only 25 rows in the fallback dataset, so we do not
split it; the whole set is used for the ranker's top-k evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg["split"]
    seed = cfg["seed"]
    stratify = df[split_cfg["stratify_col"]]
    train, temp = train_test_split(
        df, train_size=split_cfg["train_frac"], stratify=stratify, random_state=seed,
    )
    # Split temp into val + test by relative sizes
    val_rel = split_cfg["val_frac"] / (split_cfg["val_frac"] + split_cfg["test_frac"])
    val, test = train_test_split(
        temp, train_size=val_rel, stratify=temp[split_cfg["stratify_col"]], random_state=seed,
    )
    return train, val, test


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])

    report = {}
    for name, path in [("agent1", proc / "agent1_features.parquet"),
                       ("agent3", proc / "agent3_features.parquet")]:
        df = pd.read_parquet(path)
        train, val, test = stratified_split(df, cfg)
        train.to_parquet(proc / f"{name}_train.parquet", index=False)
        val.to_parquet(proc / f"{name}_val.parquet", index=False)
        test.to_parquet(proc / f"{name}_test.parquet", index=False)
        report[name] = {
            "train": len(train), "val": len(val), "test": len(test),
            "class_balance_train": train["attack_class"].value_counts().to_dict(),
            "class_balance_test": test["attack_class"].value_counts().to_dict(),
        }
        print(f"[split] {name}: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    (proc / "split_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
