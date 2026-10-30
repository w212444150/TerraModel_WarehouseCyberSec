# Capstone Submissions - Team TerraModel

Robert B. Thompson and Cesar Noriega.

Single index of every graded capstone deliverable. For a grader this is the fastest path from zero to full picture.

## Phase 01 - Initial Advance (Weeks 1 to 3)

Grade weight: 10% (Project Proposal)

- **Primary**: [docs/phase01-initial-advance.pptx](docs/phase01-initial-advance.pptx) - original 5-slide PowerPoint submission
- Supporting write-up: [docs/phase01-initial-advance.md](docs/phase01-initial-advance.md)

Covers: problem statement (OT blind spots, unknown vulnerabilities, physical safety), the three-agent solution, project scope and technical approach, Weeks 1 to 3 deliverables.

## Phase 02 - Data Pipeline and Exploratory Analysis (Weeks 4 to 6)

Grade weight: 15% (Data Report and Preprocessed Dataset)

- **Primary**: [docs/phase02-data-report.md](docs/phase02-data-report.md)
- Reproducibility: `make data && make baseline && make eda` writes every artifact referenced in the report.

Covers: data acquisition (synthetic OT generator plus live MITRE ATT&CK for ICS plus CISA advisories), preprocessing and feature engineering per agent, 70/15/15 stratified split, exploratory analysis, baseline models (Isolation Forest, rule-based ranker, Mahalanobis distance), privacy and ethics, reproducibility checklist.

## Phase 03 - Trained Model with Evaluation (Weeks 7 to 10)

Grade weight: 15% (Trained Model with Evaluation Report)

- **Primary (upload to Canvas)**: [docs/TerraModel_Phase03_Deliverable.docx](docs/TerraModel_Phase03_Deliverable.docx) - Word format
- **Primary (grading preview)**: [docs/TerraModel_Phase03_Deliverable.pdf](docs/TerraModel_Phase03_Deliverable.pdf) - PDF rendering of the same content
- Supporting: [docs/phase03-model-report.md](docs/phase03-model-report.md) - full model report
- Supporting: [docs/phase03-limitation-fixes.md](docs/phase03-limitation-fixes.md) - how we fixed the four honest limitations
- Reproducibility: `make phase03 && make phase03-fixes && make test` writes every artifact.

Covers: supervised Agent 1 (XGBoost), Agent 2 (Logistic Regression with real CISA KEV labels), Agent 3 (Random Forest), orchestrator with ATT&CK plus NIST CSF 2.0 plus IEC 62443-3-3 plus NIST SP 800-82 Rev. 3 plus MITRE D3FEND framework mapping, SHAP interpretability, PR curves, confusion matrices, experiment tracking, 16 pytest cases.

Live run evidence: on October 29, 2026, the Google Colab run of this pipeline pulled 1,764 real actively-exploited CVEs from the live CISA KEV feed.

## Phase 04 - Final Delivery (Weeks 11 to 12, planned)

Grade weight: TBD

Planned deliverables: FastAPI orchestrator service, Streamlit operator dashboard, PyTorch LSTM for Agent 1 upgrade, real HAI data integration, red team dry run, 10-minute scripted demo video, final deck.

## Repository and verification

- Repository: https://github.com/w212444150/TerraModel_WarehouseCyberSec
- Latest Phase 03 commit at submission: `e4c9bea`
- Random seed: 42 (pinned in `config.yaml`)
- CI: GitHub Actions runs full pipeline + tests on every push to main

## Team

- Robert B. Thompson (Brandon) - co-lead
- Cesar Noriega - co-lead
