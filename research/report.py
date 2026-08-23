"""report.py — the professional assessment report (spec #18), built from the findings model.

Turns a list of research.findings Hypotheses + the authorized session into the report an operator
expects: Executive Summary, Scope, Methodology, Attack Surface, Findings (each with severity,
confidence, evidence, reasoning, validation, impact, remediation, references), Timeline, Limitations,
Appendix. Two disciplines are enforced by construction:

  * EVERY finding is traceable to evidence — its evidence sources are printed; a finding with no
    evidence is shown UNCONFIRMED and says so.
  * A HYPOTHESIS is never rendered as a confirmed vulnerability — the status (CONFIRMED / LIKELY /
    POSSIBLE / UNCONFIRMED) is derived from evidence (findings.derive_status), and an unvalidated
    severity is marked ASSERTED.

Pure text generation, no model. Ordering: confirmed first, then by severity.
"""
from __future__ import annotations

import time

from research.findings import SEVERITY, STATUS_ORDER, Hypothesis

_SEV_RANK = {s: i for i, s in enumerate(reversed(SEVERITY))}   # CRITICAL=0 ... INFO=4
_REFS = {
    "sqli": "CWE-89; OWASP A03:2021 Injection", "sql_injection": "CWE-89; OWASP A03:2021 Injection",
    "xss": "CWE-79; OWASP A03:2021 Injection", "ssrf": "CWE-918; OWASP A10:2021",
    "idor": "CWE-639; OWASP A01:2021 Broken Access Control",
    "command_injection": "CWE-78; OWASP A03:2021", "code_execution": "CWE-94",
    "path_traversal": "CWE-22", "insecure_deserialization": "CWE-502; OWASP A08:2021",
    "hardcoded_secret": "CWE-798", "ssti": "CWE-1336", "xxe": "CWE-611",
    "crypto_misuse": "CWE-327", "misconfiguration": "OWASP A05:2021 Security Misconfiguration",
}


def _refs(vuln_class: str) -> str:
    return _REFS.get((vuln_class or "").lower(), "—")


def _order(findings):
    # confirmed first (highest status), then most severe, then highest test-priority
    return sorted(findings, key=lambda h: (-STATUS_ORDER.index(h.status),
                                           _SEV_RANK.get(h.severity, 9), -h.priority()))


def _validated(h: Hypothesis) -> bool:
    return h.status in ("CONFIRMED", "LIKELY")


def executive_summary(findings) -> str:
    by_status = {s: 0 for s in STATUS_ORDER}
    by_sev = {s: 0 for s in SEVERITY}
    for h in findings:
        by_status[h.status] = by_status.get(h.status, 0) + 1
        by_sev[h.severity] = by_sev.get(h.severity, 0) + 1
    conf = by_status.get("CONFIRMED", 0)
    likely = by_status.get("LIKELY", 0)
    hyp = by_status.get("POSSIBLE", 0) + by_status.get("UNCONFIRMED", 0)
    L = [f"{len(findings)} candidate finding(s): **{conf} CONFIRMED**, {likely} LIKELY, {hyp} "
         "unvalidated (POSSIBLE/UNCONFIRMED).",
         "Severity spread (as ASSERTED — validated only where status is CONFIRMED/LIKELY): "
         + ", ".join(f"{by_sev[s]} {s}" for s in SEVERITY if by_sev[s]) + ".",
         "Note: unvalidated findings are HYPOTHESES pending a validating test — not confirmed "
         "vulnerabilities."]
    return "\n".join(L)


def render_finding(h: Hypothesis, n: int) -> str:
    sev = h.severity + ("" if _validated(h) else "  (ASSERTED — unvalidated)")
    L = [f"### {n}. {h.title}",
         f"- **Severity:** {sev}",
         f"- **Confidence / Status:** {h.status}",
         f"- **Affected component:** {h.affected_component or '—'}",
         f"- **Vulnerability class:** {h.vuln_class or '—'}",
         "- **Evidence:**"]
    if h.evidence:
        for e in h.evidence:
            L.append(f"    - [{e.kind}, conf {e.confidence}] {e.detail or ''} (source: {e.source})")
    else:
        L.append("    - none gathered — this finding is UNCONFIRMED")
    L += [f"- **Reasoning:** {h.why_it_matters or h.observation or '—'}",
          f"- **Validation:** " + ("validated by a test (status CONFIRMED)" if h.status == "CONFIRMED"
                                    else f"NOT yet validated — next test: {h.next_test or 'define one'}"),
          f"- **Alternative explanation:** {h.alternative_explanation or '—'}",
          f"- **Impact:** {'demonstrated' if _validated(h) else 'potential (pending validation)'}",
          f"- **Remediation:** {_remediation(h.vuln_class)}",
          f"- **References:** {_refs(h.vuln_class)}"]
    return "\n".join(L)


def _remediation(vuln_class: str) -> str:
    from research.findings import _remediation_hint
    return _remediation_hint(vuln_class)


def assessment_report_json(*, objective: str, scope, findings, operator: str = "operator",
                           limitations=None) -> dict:
    """The same graded assessment as assessment_report(), but as a machine-readable structure:
    summary counts + per-finding severity/status/evidence/next-test/remediation/refs, ranked
    confirmed-first then by severity. For programmatic use, diffing runs, and the UI findings view."""
    findings = _order(list(findings))
    by_status = {s: 0 for s in STATUS_ORDER}
    by_sev = {s: 0 for s in SEVERITY}
    for h in findings:
        by_status[h.status] = by_status.get(h.status, 0) + 1
        by_sev[h.severity] = by_sev.get(h.severity, 0) + 1
    out = []
    for i, h in enumerate(findings, 1):
        out.append({
            "rank": i, "title": h.title, "severity": h.severity,
            "severity_asserted": not _validated(h),        # unvalidated severity is ASSERTED, not proven
            "status": h.status, "affected_component": h.affected_component or None,
            "vuln_class": h.vuln_class or None,
            "evidence": [{"kind": e.kind, "detail": e.detail, "source": e.source,
                          "confidence": e.confidence} for e in h.evidence],
            "reasoning": h.why_it_matters or h.observation or None,
            "validated": _validated(h),
            "next_test": None if h.status == "CONFIRMED" else (h.next_test or "define a validating test"),
            "impact": "demonstrated" if _validated(h) else "potential (pending validation)",
            "remediation": _remediation(h.vuln_class), "references": _refs(h.vuln_class)})
    return {
        "objective": objective, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "operator": operator,
        "scope": list(scope) if scope else [],
        "summary": {"total": len(findings), "confirmed": by_status.get("CONFIRMED", 0),
                    "likely": by_status.get("LIKELY", 0),
                    "unvalidated": by_status.get("POSSIBLE", 0) + by_status.get("UNCONFIRMED", 0),
                    "by_status": {s: by_status[s] for s in STATUS_ORDER if by_status[s]},
                    "by_severity": {s: by_sev[s] for s in SEVERITY if by_sev[s]}},
        "findings": out,
        "limitations": limitations or [
            "Unvalidated findings are hypotheses; confirm with a validating test before relying on them.",
            "Static-analysis findings are intraprocedural leads, not proof of exploitability.",
            "Absence of a finding is not proof of absence of a vulnerability."]}


def assessment_report(*, objective: str, scope, findings, methodology: str = "",
                      timeline=None, limitations=None, attack_surface=None,
                      operator: str = "operator") -> str:
    findings = _order(list(findings))
    ts = time.strftime("%Y-%m-%d %H:%M")
    L = [f"# Security Assessment Report",
         f"_Objective:_ {objective}    _Generated:_ {ts}", "",
         "## Executive Summary", executive_summary(findings), "",
         "## Scope",
         "Authorized targets: " + (", ".join(scope) if scope else "(local source/artifact review)"),
         "> Only targets explicitly authorized by the operator were assessed.", "",
         "## Methodology",
         methodology or ("Evidence -> inference -> hypothesis -> validation -> conclusion. A banner / "
                         "version / filename is treated as EVIDENCE, never as proof. Findings are "
                         "graded CONFIRMED / LIKELY / POSSIBLE / UNCONFIRMED by the evidence, and no "
                         "hypothesis is reported as a confirmed vulnerability without a validating "
                         "test."), ""]
    if attack_surface:
        L += ["## Attack Surface"] + [f"- {a}" for a in attack_surface] + [""]
    L.append("## Findings")
    if findings:
        for i, h in enumerate(findings, 1):
            L.append(render_finding(h, i))
            L.append("")
    else:
        L += ["No findings.", ""]
    if timeline:
        L += ["## Timeline"] + [f"- {t}" for t in timeline] + [""]
    L += ["## Limitations",
          "\n".join(f"- {x}" for x in (limitations or [
              "Unvalidated findings are hypotheses; confirm with a validating test before relying on "
              "them.",
              "Static-analysis findings are intraprocedural leads, not proof of exploitability.",
              "Absence of a finding is not proof of absence of a vulnerability."])), "",
          "## Appendix",
          f"Report produced for {operator} from {len(findings)} finding record(s); every finding above "
          "cites its evidence sources for traceability."]
    return "\n".join(L)
