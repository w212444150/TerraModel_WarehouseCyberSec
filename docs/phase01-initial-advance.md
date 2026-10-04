# Phase 01 Initial Advance - Team TerraModel

**Cybersecurity in AI-Powered Warehouse Automation**
**Project Proposal, Weeks 1 through 3**

Prepared by Robert B. Thompson and Cesar Noriega.

Original slide deck: [`phase01-initial-advance.pptx`](phase01-initial-advance.pptx) (5 slides, PowerPoint).

## Focus

Protect robots, PLCs, sensors, and warehouse operations from cyber threats.

## 1. Problem We Want to Solve

Cybersecurity risks in an automated warehouse surface in three places.

### OT Blind Spots

Traditional IT security may not continuously monitor robots, PLCs, and industrial sensors. An attacker can remain unnoticed until the operation is disrupted.

### Unknown Vulnerabilities

Automation may have default passwords, open remote access, missing patches, or poor network segmentation. These weaknesses can remain open until an attacker finds them.

### Physical Safety

A compromised robot can receive a manipulated command or leave its safe path. A cyberattack can therefore become an operational and worker-safety problem.

## 2. Proposed Solution

Three AI agents working together.

### Agent 1 - OT Anomaly Detection

Monitors activity in the operational technology network. Detects unusual behavior involving robots, PLCs, and sensors. Goal: identify an attack while it is happening.

### Agent 2 - Assessment / Hardening

Finds security weaknesses in the warehouse environment. Creates a ranked list of recommended fixes. Goal: reduce exposure before an attacker exploits it.

### Agent 3 - Robot Integrity Monitor

Checks robot commands and actual movement. Detects spoofed commands or unsafe deviations. Goal: protect workers, equipment, and inventory.

## 3. Project Scope and Technical Approach

Initial concept, suitable for a simulated warehouse environment.

### What We Plan to Build

- A simulated automated warehouse environment
- Synthetic data from robots, PLCs, and sensors
- AI-based anomaly detection
- Vulnerability assessment and prioritized recommendations
- Robot command and movement integrity monitoring
- Alerts or a simple dashboard for the operator

### Basic Workflow

1. Collect data
2. Detect abnormal behavior
3. Assess security weaknesses
4. Prioritize the most important fixes
5. Verify robot behavior
6. Generate an alert for the operator

## 4. Weeks 1 to 3 Deliverables

This is an initial project advance, not the final system.

### Week 1

- Form the team
- Confirm the cybersecurity problem
- Define the warehouse environment
- Identify initial data and tools
- Discuss project scope

### Week 2

- Analyze the problem
- Refine the scope
- Research OT cybersecurity
- Identify data, tools, and computing resources
- Begin technical planning

### Week 3

- Finalize the proposal
- Define technical approach
- Establish success metrics
- Assess risks and ethics
- Present proposal and receive instructor feedback

## 5. Post-Week-3 Instructor Feedback

Instructor feedback at the end of Week 3 asked us to add a coordination layer across the three agents and to speak the industry's own vocabulary when describing detected events. We added an **orchestrator** that pulls agent outputs together and maps each finding to a MITRE ATT&CK for ICS technique ID. The orchestrator is implemented in [`agents/orchestrator/attack_mapper.py`](../agents/orchestrator/attack_mapper.py) and uses the official MITRE ATT&CK for ICS v18 STIX bundle fetched by [`collectors/fetch_attack.py`](../collectors/fetch_attack.py).

Techniques currently covered: T0836, T0855, T0832, T0814, T0859, T0866, T0863.

## 6. Transition to Phase 02

Phase 02 (Weeks 4 through 6) delivered the data pipeline, feature engineering, and baseline models for the three agents plus the orchestrator. See [`phase02-data-report.md`](phase02-data-report.md) for the full Phase 02 report and the actual metrics from the pipeline run.
