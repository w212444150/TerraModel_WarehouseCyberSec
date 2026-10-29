"""
Cybersecurity framework mapping for TerraModel detections.

For each MITRE ATT&CK for ICS technique our agents detect, we also map to:
  - NIST Cybersecurity Framework 2.0 (CSF 2.0) functions and subcategories
  - IEC 62443 security requirements (industrial automation and control systems)
  - NIST SP 800-82 Rev. 3 (Guide to OT Security) relevant controls
  - MITRE D3FEND defensive countermeasures

This matters because warehouse automation lives inside a manufacturing OT
cybersecurity context. Reviewers and future employers expect findings mapped
to the standards their teams audit against, not just an attacker vocabulary.

References:
  - NIST CSF 2.0: https://doi.org/10.6028/NIST.CSWP.29
  - IEC 62443-3-3: System Security Requirements and Security Levels
  - NIST SP 800-82 Rev. 3: https://doi.org/10.6028/NIST.SP.800-82r3
  - MITRE D3FEND: https://d3fend.mitre.org/
"""
from __future__ import annotations

# Each entry: ATT&CK ID -> defensive framework mapping.
# NIST CSF 2.0 subcategory IDs follow the official CSF 2.0 naming (DE.CM-01, etc.)
# IEC 62443-3-3 SRs follow their published catalog.
# NIST SP 800-82 Rev. 3 references cite the control family and subsection.
# MITRE D3FEND IDs are the published technique IDs on d3fend.mitre.org.

FRAMEWORK_MAP = {
    "T0836": {  # Modify Parameter
        "attack_tactic": "Impair Process Control",
        "nist_csf_2_0": [
            "DE.CM-01 (Networks and network services are monitored)",
            "DE.CM-09 (Computing hardware and software are monitored)",
            "PR.DS-05 (Protections against data leaks are implemented)",
            "PR.PS-06 (Secure configurations)",
        ],
        "iec_62443_3_3": ["SR 3.4 (Software and information integrity)",
                           "SR 4.1 (Information confidentiality)",
                           "SR 7.6 (Network and security configuration settings)"],
        "nist_sp_800_82r3": ["5.1 (ICS Security Risk Assessment)",
                              "6.2.4 (Change management for ICS)",
                              "6.2.7 (Monitoring and reporting)"],
        "mitre_d3fend": ["D3-NTA (Network Traffic Analysis)",
                          "D3-PA (Process Analysis)",
                          "D3-FBA (Function Baseline Analysis)"],
    },
    "T0855": {  # Unauthorized Command Message
        "attack_tactic": "Impair Process Control",
        "nist_csf_2_0": [
            "DE.AE-02 (Potentially adverse events are analyzed)",
            "DE.CM-01 (Networks monitored)",
            "PR.AA-01 (Identities and credentials are managed)",
        ],
        "iec_62443_3_3": ["SR 1.1 (Human user identification and authentication)",
                           "SR 1.2 (Software process and device identification)",
                           "SR 3.1 (Communication integrity)"],
        "nist_sp_800_82r3": ["6.2.3 (Access control)",
                              "6.2.5 (Authentication and authorization)",
                              "6.2.7 (Monitoring and reporting)"],
        "mitre_d3fend": ["D3-MFA (Multi-factor Authentication)",
                          "D3-NTA (Network Traffic Analysis)",
                          "D3-PCSV (Process Code Segment Verification)"],
    },
    "T0832": {  # Manipulation of View
        "attack_tactic": "Impair Process Control",
        "nist_csf_2_0": [
            "DE.AE-03 (Information is correlated from multiple sources)",
            "DE.CM-09 (Computing hardware and software monitored)",
            "PR.DS-06 (Integrity checking mechanisms)",
        ],
        "iec_62443_3_3": ["SR 3.4 (Software and information integrity)",
                           "SR 3.8 (Session integrity)"],
        "nist_sp_800_82r3": ["6.2.7 (Monitoring and reporting)",
                              "6.2.8 (Incident response)"],
        "mitre_d3fend": ["D3-FBA (Function Baseline Analysis)",
                          "D3-PA (Process Analysis)"],
    },
    "T0814": {  # Denial of Service
        "attack_tactic": "Inhibit Response Function",
        "nist_csf_2_0": [
            "DE.CM-01 (Networks monitored)",
            "DE.AE-02 (Potentially adverse events analyzed)",
            "PR.IR-04 (Capacity and performance resources)",
        ],
        "iec_62443_3_3": ["SR 7.1 (Denial of service protection)",
                           "SR 7.2 (Resource management)"],
        "nist_sp_800_82r3": ["6.2.7 (Monitoring and reporting)",
                              "6.2.9 (Contingency and continuity)"],
        "mitre_d3fend": ["D3-NTA (Network Traffic Analysis)",
                          "D3-RTSD (Resource Access Pattern Analysis)"],
    },
    "T0859": {  # Valid Accounts
        "attack_tactic": "Lateral Movement",
        "nist_csf_2_0": [
            "PR.AA-01 (Identities and credentials managed)",
            "PR.AA-05 (Access permissions and authorizations managed)",
            "DE.CM-03 (Personnel activity monitored)",
        ],
        "iec_62443_3_3": ["SR 1.1 (Human user authentication)",
                           "SR 1.5 (Authenticator management)",
                           "SR 2.1 (Authorization enforcement)"],
        "nist_sp_800_82r3": ["6.2.3 (Access control)",
                              "6.2.5 (Authentication and authorization)"],
        "mitre_d3fend": ["D3-MFA (Multi-factor Authentication)",
                          "D3-DAM (Domain Account Monitoring)",
                          "D3-UGLPA (User-Group Local Policy Analysis)"],
    },
    "T0866": {  # Exploitation of Remote Services
        "attack_tactic": "Initial Access",
        "nist_csf_2_0": [
            "ID.RA-01 (Vulnerabilities are identified)",
            "PR.PS-02 (Software is maintained)",
            "DE.CM-01 (Networks monitored)",
        ],
        "iec_62443_3_3": ["SR 2.4 (Mobile code)",
                           "SR 3.2 (Protection from malicious code)",
                           "SR 7.8 (Auditable events)"],
        "nist_sp_800_82r3": ["5.3 (Vulnerability management)",
                              "6.2.1 (System hardening)"],
        "mitre_d3fend": ["D3-NTA (Network Traffic Analysis)",
                          "D3-RFEM (Remote File Access Mediation)",
                          "D3-PH (Process Hardening)"],
    },
    "T0863": {  # User Execution
        "attack_tactic": "Execution",
        "nist_csf_2_0": [
            "PR.AT-01 (Personnel awareness and training)",
            "DE.CM-09 (Computing hardware and software monitored)",
        ],
        "iec_62443_3_3": ["SR 3.2 (Protection from malicious code)",
                           "SR 7.7 (Least functionality)"],
        "nist_sp_800_82r3": ["6.2.1 (System hardening)",
                              "6.2.6 (Awareness and training)"],
        "mitre_d3fend": ["D3-UA (User Analysis)",
                          "D3-FCA (File Content Analysis)"],
    },
}


def map_technique(attack_id: str) -> dict:
    """Return the full framework mapping for a given ATT&CK for ICS technique."""
    return FRAMEWORK_MAP.get(attack_id, {})


def enrich_event(attack_hits: list[dict]) -> list[dict]:
    """Attach framework context to each ATT&CK hit from the orchestrator."""
    enriched = []
    for hit in attack_hits:
        tid = hit.get("technique_id", "")
        mapping = map_technique(tid)
        enriched.append({**hit, "frameworks": mapping})
    return enriched


if __name__ == "__main__":
    # Smoke test and print the mapping table
    for tid, mapping in FRAMEWORK_MAP.items():
        print(f"\n{tid}  tactic: {mapping['attack_tactic']}")
        for framework, controls in mapping.items():
            if framework == "attack_tactic":
                continue
            print(f"  {framework}:")
            for c in controls:
                print(f"    - {c}")
