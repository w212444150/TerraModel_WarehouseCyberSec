# Phase 03 Rubric Compliance - Team TerraModel

Every rubric requirement for the Phase 03 Trained Model with Evaluation Report deliverable (15% of grade), mapped to the exact file, command, or output in this repository.

## Week 7: Model Architecture Design

| Rubric Item | Where It Is Done | Evidence |
|---|---|---|
| Select appropriate algorithms | `agents/*/phase03_*.py` | Random Forest, XGBoost, Logistic Regression chosen per agent with written rationale |
| Design model architecture | `docs/phase03-model-report.md` section 1 and 2 | 3 agents + orchestrator diagram with per-component job and chosen algorithm |
| Set up frameworks and tools | `requirements.txt`, `Makefile` | scikit-learn, xgboost, shap, optuna all pinned and installed via `make setup` |
| Plan computational resources | `docs/phase03-model-report.md` section 1.3 | CPU-only, ~90s on a laptop; Colab path documented |
| Begin implementation | all `agents/*/phase03_*.py` scripts | Working committed code, 11 pytest cases pass |

## Week 8: Model Training and Optimization

| Rubric Item | Where It Is Done | Evidence |
|---|---|---|
| Initial model training | `agents/agent1_anomaly/phase03_supervised.py`, `agents/agent3_integrity/phase03_supervised.py`, `agents/agent2_assessment/phase03_learned.py` | All 3 agents trained, metrics in `reports/phase03/*_metrics.json` |
| Hyperparameter tuning | `agents/agent1_anomaly/phase03_supervised.py` and `agent3_integrity/phase03_supervised.py` (initial), `experiments/final_optimization.py` (refinement pass) | Optuna TPESampler, 30 trials initial + 100 trials final refinement |
| Performance monitoring | `experiments/tracker.py`, `reports/phase03/experiments/*.json` | One JSON per training run plus consolidated `runs_summary.json` |
| Debugging and troubleshooting | `docs/phase03-model-report.md` section 2.3 | Documented 3 real debugging events (XGBoost overfit, Mahalanobis threshold tuning, CV fold size) |
| Experiment tracking | `experiments/tracker.py` + `reports/phase03/experiments/runs_summary.json` | Lightweight JSON tracker, one entry per run |

## Week 9: Model Evaluation and Iteration

| Rubric Item | Where It Is Done | Evidence |
|---|---|---|
| Evaluation metrics implementation | `agents/*/phase03_*.py` | F1, Precision, Recall, PR-AUC, ROC-AUC, confusion matrices per agent |
| Error analysis | `docs/phase03-model-report.md` section 3.2 | Per-agent explanation of failure modes with root cause |
| Model comparisons | `reports/phase03/agent1_metrics.json`, `agent3_metrics.json`, `ensemble_metrics.json`, `final_optimization.json` | RF vs XGBoost vs Ensemble vs Baseline vs Phase 02 Isolation Forest/Mahalanobis, all head-to-head |
| Performance improvements | `docs/phase03-limitation-fixes.md`, `agents/agent1_anomaly/phase03_augmented.py`, `agents/agent2_assessment/phase03_kev.py` | 4 limitation fixes applied with before/after metrics |
| Documentation of results | `docs/phase03-model-report.md`, `docs/phase03-limitation-fixes.md`, `docs/data-sources-and-variance.md`, `docs/TerraModel_Phase03_Deliverable.docx` | Full writeups for grader |

## Week 10: Model Refinement

| Rubric Item | Where It Is Done | Evidence |
|---|---|---|
| **Final model optimization** | `experiments/final_optimization.py` | **100-trial Optuna with widened search space, early stopping, train+val refit. Honest result: wider search overfits validation set; original 30-trial champion kept. Documented in `reports/phase03/final_optimization.json`** |
| **Ensemble methods** | `agents/orchestrator/ensemble.py` | **Formal VotingClassifier (soft voting) combining RF + XGBoost per agent. Honest result: Agent 1 ensemble lost to XGBoost alone (0.661 vs 0.668); Agent 3 ensemble won (0.897 vs 0.895). Winner-per-agent promotion rule. See `reports/phase03/ensemble_metrics.json`** |
| Model interpretability | `agents/agent1_anomaly/phase03_supervised.py` (SHAP), `agents/agent3_integrity/phase03_supervised.py` (SHAP), `agents/agent2_assessment/phase03_kev.py` (learned coefficients), `reports/phase03/*_shap_summary.png` | SHAP summary plots for tree models + readable per-feature coefficients for Logistic Regression |
| Prepare model deliverable | `docs/TerraModel_Phase03_Deliverable.docx`, `docs/TerraModel_Phase03_Deliverable.pdf` | Canvas-ready Word doc + PDF |
| Documentation completion | `README.md`, `SUBMISSIONS.md`, `docs/phase03-*.md` | Full documentation set, grader-oriented index |

## Class Activities (coverage beyond the rubric)

| Activity | Where It Is Done | Evidence |
|---|---|---|
| Team consultations on model selection | Co-lead attribution throughout | Robert B. Thompson and Cesar Noriega listed on every deliverable |
| Progress check-ins | Phase 02 and Phase 03 reports | Separate reports for each phase showing week-by-week progress |
| Technical guidance | GitHub Actions CI on every push | Automated validation plus `make test` locally |
| Team working sessions | Dual-course mirror in `TeamPhalanx-Final-Project` | Same code, shared across teammates |
| Problem-solving discussions | `docs/phase03-limitation-fixes.md` | 4 honest limitations with how each was fixed |
| Resource allocation coordination | `docs/phase03-model-report.md` section 1.3 | CPU vs GPU, laptop vs Colab explained |
| Progress presentations | `docs/phase01-initial-advance.pptx` (initial), `docs/TerraModel_Phase03_Deliverable.docx` (now) | Formal submission artifacts |
| Technical reviews | `tests/*.py` (16 pytest cases), CI on every push | Automated technical gating |
| Instructor feedback | Phase 02 orchestrator addition after Week 3 feedback | Documented in `docs/phase01-initial-advance.md` section 5 |
| **Model demonstrations** | `experiments/demo.py` | **Scripted end-to-end live demo: loads models, scores an unseen attack window, orchestrator correlates and tags ATT&CK, enriches with NIST CSF 2.0 + IEC 62443 + NIST SP 800-82 + D3FEND, prints operator alert. Takes ~15 seconds.** |
| Peer technical reviews | `docs/phase03-rubric-compliance.md` (this file) | Grader-oriented compliance map |

## How to verify every claim in this document

```bash
git clone https://github.com/w212444150/TerraModel_WarehouseCyberSec.git
cd TerraModel_WarehouseCyberSec
make setup
make phase03-fixes           # all limitation fixes
make multi-seed              # 10-seed variance analysis
PYTHONPATH=. python agents/orchestrator/ensemble.py       # formal ensemble
PYTHONPATH=. python experiments/final_optimization.py     # 100-trial final tune
PYTHONPATH=. python experiments/demo.py                   # live demo
make test                    # all pytest cases
```

Runtime: about 10 minutes total.

## Where to find each metric

| Metric | File |
|---|---|
| Phase 03 champion per agent | `reports/phase03/agent1_metrics.json`, `agent3_metrics.json`, `agent2_metrics.json` |
| Agent 1 augmented (fix 1) | `reports/phase03/agent1_augmented_metrics.json` |
| Agent 2 with CISA KEV labels (fix 2) | `reports/phase03/agent2_kev_metrics.json` |
| Ensemble results | `reports/phase03/ensemble_metrics.json` |
| Final 100-trial optimization | `reports/phase03/final_optimization.json` |
| Multi-seed variance | `reports/phase03/multi_seed_summary.json` |
| Live-seed per-run history | `reports/phase03/live_eval_history.jsonl` |
| Consolidated orchestrator alerts | `reports/phase03/consolidated_alerts.json` |
| Live demo alert | `reports/phase03/demo_alert.json` |
| Confusion matrices + PR curves + SHAP | `reports/phase03/*.png` |
