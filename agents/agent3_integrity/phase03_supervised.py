"""
Agent 3 Phase 03: supervised robot integrity classifier.

Phase 02 used Mahalanobis distance at p95 (unsupervised). F1=0.3742, PR-AUC=0.7116,
Recall=0.2444. The high precision + low recall said: the model was too conservative.

Phase 03 trains supervised classifiers:
  - Random Forest
  - XGBoost (Optuna-tuned)

Outputs: models/agent3_rf.joblib, models/agent3_xgb.joblib,
         reports/phase03/agent3_metrics.json + confusion + PR curve + SHAP summary.
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

from experiments.tracker import ExperimentRun

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "mean_dev", "max_dev", "std_dev", "torque_var",
    "timing_jitter", "nonprimary_src_frac", "cmd_entropy",
]


def xy(df):
    return df[FEATURE_COLS].values, df["is_attack"].values


def eval_classifier(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }, y_proba


def tune_xgboost(X_train, y_train, X_val, y_val, seed, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = xgb.XGBClassifier(**params, eval_metric="aucpr", tree_method="hist",
                                   random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    final = xgb.XGBClassifier(**best, eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
    final.fit(X_train, y_train)

    run = ExperimentRun("agent3_xgboost_optuna")
    run.log_params(model="XGBClassifier", n_trials=n_trials, **best)
    val_metrics, _ = eval_classifier(final, X_val, y_val)
    run.log_metrics(**{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)})
    run.save()
    return final, study


def plot_confusion(cm, out_path, title):
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
    fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_pr_curve(y_true, y_score, out_path, title):
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recalls, precisions, color="#1E2761", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02]); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_shap(model, X_sample, feature_names, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=(8, 5))
    plt.title("Agent 3 Feature Importance (SHAP)")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close("all")


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")
    seed = cfg["seed"]

    train = pd.read_parquet(proc / "agent3_train.parquet")
    val = pd.read_parquet(proc / "agent3_val.parquet")
    test = pd.read_parquet(proc / "agent3_test.parquet")

    X_train, y_train = xy(train); X_val, y_val = xy(val); X_test, y_test = xy(test)
    print(f"[agent3] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    rf_val, _ = eval_classifier(rf, X_val, y_val)
    print(f"[agent3][rf]  val F1={rf_val['f1']:.4f} PR-AUC={rf_val['pr_auc']:.4f}")

    xgb_model, study = tune_xgboost(X_train, y_train, X_val, y_val, seed, n_trials=30)
    xgb_val, _ = eval_classifier(xgb_model, X_val, y_val)
    print(f"[agent3][xgb] val F1={xgb_val['f1']:.4f} PR-AUC={xgb_val['pr_auc']:.4f}  best_val_PRAUC={study.best_value:.4f}")

    champion_name = "xgboost" if xgb_val["pr_auc"] >= rf_val["pr_auc"] else "random_forest"

    # Refit champion on train+val
    X_all = np.vstack([X_train, X_val]); y_all = np.concatenate([y_train, y_val])
    (xgb_model if champion_name == "xgboost" else rf).fit(X_all, y_all)

    rf_test, rf_proba = eval_classifier(rf, X_test, y_test)
    xgb_test, xgb_proba = eval_classifier(xgb_model, X_test, y_test)

    joblib.dump(rf, models_dir / "agent3_rf.joblib")
    joblib.dump(xgb_model, models_dir / "agent3_xgb.joblib")

    champion = xgb_model if champion_name == "xgboost" else rf
    champion_proba = xgb_proba if champion_name == "xgboost" else rf_proba
    champion_metrics = xgb_test if champion_name == "xgboost" else rf_test
    champion_pred = (champion_proba >= 0.5).astype(int)

    plot_confusion(confusion_matrix(y_test, champion_pred),
                   reports_dir / "agent3_confusion_matrix.png",
                   f"Agent 3 Confusion Matrix ({champion_name}, test)")
    plot_pr_curve(y_test, champion_proba,
                  reports_dir / "agent3_pr_curve.png",
                  f"Agent 3 PR Curve ({champion_name}, test)")
    try:
        plot_shap(champion, X_test, FEATURE_COLS, reports_dir / "agent3_shap_summary.png")
    except Exception as exc:
        print(f"[agent3][shap] skipped: {exc}")

    baseline_path = Path("reports/agent3_metrics.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    out = {
        "champion": champion_name,
        "best_optuna_val_pr_auc": float(study.best_value),
        "best_optuna_params": study.best_params,
        "test_random_forest": rf_test,
        "test_xgboost": xgb_test,
        "baseline_mahalanobis_phase02": baseline,
        "lift_over_baseline": {
            "f1_delta": round(champion_metrics["f1"] - baseline.get("f1", 0), 4),
            "pr_auc_delta": round(champion_metrics["pr_auc"] - baseline.get("pr_auc", 0), 4),
        } if baseline else "baseline not found",
        "feature_cols": FEATURE_COLS,
    }
    (reports_dir / "agent3_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\n[agent3][TEST] champion = {champion_name}")
    print(f"[agent3][TEST] rf  : F1={rf_test['f1']:.4f}  PR-AUC={rf_test['pr_auc']:.4f}  Recall={rf_test['recall']:.4f}")
    print(f"[agent3][TEST] xgb : F1={xgb_test['f1']:.4f}  PR-AUC={xgb_test['pr_auc']:.4f}  Recall={xgb_test['recall']:.4f}")
    if baseline:
        print(f"[agent3][TEST] Phase 02 baseline Mahalanobis : F1={baseline.get('f1', 0):.4f}  PR-AUC={baseline.get('pr_auc', 0):.4f}")


if __name__ == "__main__":
    main()
