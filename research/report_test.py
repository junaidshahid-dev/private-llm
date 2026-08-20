"""report_test.py — the assessment report is evidence-traceable and never overstates a hypothesis.

    python research/report_test.py
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from research.findings import Evidence, Hypothesis                             # noqa: E402
from research.report import assessment_report, executive_summary, render_finding  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("ASSESSMENT REPORT — evidence-traceable, hypotheses never rendered as confirmed")
    print("=" * 74)

    confirmed = Hypothesis(
        title="Path traversal reads arbitrary files", vuln_class="path_traversal",
        affected_component="web-target:/download?file=", severity="HIGH",
        why_it_matters="?file=../../etc/passwd returned the file contents",
        evidence=[Evidence("http_get ?file=../../etc/passwd", "validated_test", "root:x:0:0 returned", 0.95)])
    possible = Hypothesis(
        title="Exposed /config may disclose credentials", vuln_class="idor",
        affected_component="web-target:/config", severity="HIGH",
        why_it_matters="/config returned 200",
        next_test="retrieve /config and inspect for credentials",
        evidence=[Evidence("ffuf", "tool_output", "/config found (200)", 0.7)])
    unconf = Hypothesis(title="Server banner suggests an old Apache", vuln_class="misconfiguration",
                        severity="LOW")

    findings = [possible, unconf, confirmed]
    rep = assessment_report(objective="Assess the authorized lab web app",
                            scope=["lab.local"], findings=findings)

    print("\n1. required sections present")
    for sec in ("# Security Assessment Report", "## Executive Summary", "## Scope", "## Methodology",
                "## Findings", "## Limitations", "## Appendix"):
        check(f"has section {sec!r}", sec in rep)

    print("\n2. discipline: confirmed vs hypothesis")
    check("executive summary counts 1 CONFIRMED", "1 CONFIRMED" in rep, )
    check("a validated finding shows CONFIRMED", "Status:** CONFIRMED" in rep)
    check("an unvalidated finding's severity is marked ASSERTED", "ASSERTED" in rep)
    check("an unvalidated finding shows POSSIBLE/UNCONFIRMED, not confirmed",
          "POSSIBLE" in rep and "UNCONFIRMED" in rep)

    print("\n3. evidence traceability")
    check("confirmed finding cites its evidence source",
          "http_get ?file=../../etc/passwd" in rep)
    check("possible finding cites its ffuf evidence", "ffuf" in rep and "/config found" in rep)
    check("a no-evidence finding says UNCONFIRMED explicitly",
          "none gathered" in rep or "UNCONFIRMED" in render_finding(unconf, 1))

    print("\n4. ordering: confirmed first")
    idx_conf = rep.index("Path traversal reads arbitrary files")
    idx_poss = rep.index("Exposed /config")
    check("the CONFIRMED finding is listed before the POSSIBLE one", idx_conf < idx_poss)

    print("\n5. references + remediation are populated")
    check("path traversal references a CWE", "CWE-22" in rep)
    check("remediation text present for a finding", "Remediation:" in rep)

    print("\n6. exec summary on an empty set does not crash")
    check("empty findings -> a summary string", isinstance(executive_summary([]), str))
    check("empty report renders", "No findings." in assessment_report(
        objective="x", scope=[], findings=[]))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL REPORT TESTS PASS — sections complete, evidence traceable, hypotheses stay hypotheses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
