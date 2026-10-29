"""
Agent 1 Phase 03: supervised models for OT network anomaly detection.

Phase 02 used an unsupervised Isolation Forest trained on normal data only. That
gave F1 0.24 on the test set, which is too low to ship. Phase 03 replaces it with
supervised classifiers trained on labeled attack windows:

  - RandomForestClassifier (bagging baseline)
  - XGBoostClassifier (gradient boosted, Optuna-tuned)

We also keep the Isolation Forest baseline score for comparison in the final report.

Outputs:
  - models/agent1_rf.joblib
  - models/agent1_xgb.joblib
  - reports/phase03/agent1_metrics.json
  - reports/phase03/agent1_confusion_matrix.png
  - reports/phase03/agent1_pr_curve.png
  - reports/phase03/agent1_shap_summary.png
  - reports/phase03/experiments/*.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from experiments.tracker import ExperimentRun

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "pkt_count", "pkt_size_mean", "pkt_size_std", "inter_arrival_mean_ms",
    "unique_src_ips", "unique_dst_ips",
    "frac_modbus_tcp", "frac_ethernet_ip", "frac_profinet", "frac_http", "frac_other",
    "proto_entropy", "size_cv", "src_dst_ratio", "pkt_count_deviation",
]


def load_splits(proc_dir: Path):
    train = pd.read_parquet(proc_dir / "agent1_train.parquet")
    val = pd.read_parquet(proc_dir / "agent1_val.parquet")
    test = pd.read_parquet(proc_dir / "agent1_test.parquet")
    return train, val, test


def xy(df):
    return df[FEATURE_COLS].values, df["is_attack"].values


def eval_classifier(model, X, y, needs_scaling=False, scaler=None):
    if needs_scaling and scaler is not None:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        y_proba = y_pred.astype(float)
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }, y_proba


def train_random_forest(X_train, y_train, X_val, y_val, seed):
    run = ExperimentRun("agent1_random_forest")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    model.fit(X_train, y_train)
    metrics, _ = eval_classifier(model, X_val, y_val)
    run.log_params(model="RandomForestClassifier", n_estimators=300, min_samples_leaf=2, class_weight="balanced")
    run.log_metrics(**{f"val_{k}": v for k, v in metrics.items() if isinstance(v, float)})
    run.save()
    print(f"[agent1][rf]  val F1={metrics['f1']:.4f} PR-AUC={metrics['pr_auc']:.4f}")
    return model


def tune_xgboost(X_train, y_train, X_val, y_val, seed, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = xgb.XGBClassifier(
            **params, eval_metric="aucpr", tree_method="hist",
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"[agent1][xgb] best val PR-AUC = {study.best_value:.4f}")
    print(f"[agent1][xgb] best params = {best}")

    final = xgb.XGBClassifier(
        **best, eval_metric="aucpr", tree_method="hist",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    final.fit(X_train, y_train)

    run = ExperimentRun("agent1_xgboost_optuna")
    run.log_params(model="XGBClassifier", n_trials=n_trials, **best)
    val_metrics, _ = eval_classifier(final, X_val, y_val)
    run.log_metrics(**{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)})
    run.save()
    return final, study


def plot_confusion(cm, out_path, title="Agent 1 Confusion Matrix (XGBoost, test)"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true, y_score, out_path, title="Agent 1 PR Curve (XGBoost, test)"):
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recalls, precisions, color="#1E2761", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_shap_summary(model, X_sample, feature_names, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=(8, 5))
    plt.title("Agent 1 Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    train, val, test = load_splits(proc)
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)

    print(f"[agent1] train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"[agent1] attack rate train/test = {y_train.mean():.3f} / {y_test.mean():.3f}")

    # Combine train+val for the final fit after model selection
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    rf = train_random_forest(X_train, y_train, X_val, y_val, seed)
    xgb_model, study = tune_xgboost(X_train, y_train, X_val, y_val, seed, n_trials=30)

    # Pick champion on validation set
    rf_val, _ = eval_classifier(rf, X_val, y_val)
    xgb_val, _ = eval_classifier(xgb_model, X_val, y_val)
    champion_name = "xgboost" if xgb_val["pr_auc"] >= rf_val["pr_auc"] else "random_forest"
    champion = xgb_model if champion_name == "xgboost" else rf

    # Refit champion on train+val before test eval
    champion.fit(X_all, y_all)

    # Evaluate ALL candidates on test
    rf_test, rf_proba = eval_classifier(rf, X_test, y_test)
    xgb_test, xgb_proba = eval_classifier(xgb_model, X_test, y_test)

    # Save models
    joblib.dump(rf, models_dir / "agent1_rf.joblib")
    joblib.dump(xgb_model, models_dir / "agent1_xgb.joblib")

    # Confusion matrix + PR curve + SHAP for the champion
    champion_proba = xgb_proba if champion_name == "xgboost" else rf_proba
    champion_metrics = xgb_test if champion_name == "xgboost" else rf_test
    champion_pred = (champion_proba >= 0.5).astype(int)
    cm_path = reports_dir / "agent1_confusion_matrix.png"
    pr_path = reports_dir / "agent1_pr_curve.png"
    shap_path = reports_dir / "agent1_shap_summary.png"
    plot_confusion(confusion_matrix(y_test, champion_pred), cm_path,
                   title=f"Agent 1 Confusion Matrix ({champion_name}, test)")
    plot_pr_curve(y_test, champion_proba, pr_path,
                  title=f"Agent 1 PR Curve ({champion_name}, test)")

    # SHAP works on tree models directly; use XGBoost or RF
    try:
        plot_shap_summary(champion, X_test, FEATURE_COLS, shap_path)
    except Exception as exc:
        print(f"[agent1][shap] skipped: {exc}")

    # Compare to Phase 02 baseline (if present)
    baseline_path = Path("reports/agent1_metrics.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    out = {
        "champion": champion_name,
        "best_optuna_val_pr_auc": float(study.best_value),
        "best_optuna_params": study.best_params,
        "test_random_forest": rf_test,
        "test_xgboost": xgb_test,
        "baseline_isolation_forest_phase02": baseline,
        "lift_over_baseline": {
            "f1_delta": round(champion_metrics["f1"] - baseline.get("f1", 0), 4),
            "pr_auc_delta": round(champion_metrics["pr_auc"] - baseline.get("pr_auc", 0), 4),
        } if baseline else "baseline not found",
        "feature_cols": FEATURE_COLS,
    }
    (reports_dir / "agent1_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\n[agent1][TEST] champion = {champion_name}")
    print(f"[agent1][TEST] rf  : F1={rf_test['f1']:.4f}  PR-AUC={rf_test['pr_auc']:.4f}  Recall={rf_test['recall']:.4f}")
    print(f"[agent1][TEST] xgb : F1={xgb_test['f1']:.4f}  PR-AUC={xgb_test['pr_auc']:.4f}  Recall={xgb_test['recall']:.4f}")
    if baseline:
        print(f"[agent1][TEST] Phase 02 baseline IF : F1={baseline.get('f1', 0):.4f}  PR-AUC={baseline.get('pr_auc', 0):.4f}")


if __name__ == "__main__":
    main()
