"""findings_test.py — the research reasoning core cannot overclaim, and ranks by value-to-test.

    python research/findings_test.py
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

from research.findings import (Evidence, Hypothesis, derive_status, audit_claim, rank,   # noqa: E402
                               exploitation_gate, render_report, render_reasoning)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("RESEARCH FINDINGS — status is earned by evidence; overclaims are downgraded")
    print("=" * 74)

    print("\n1. status is DERIVED from evidence (CONFIRMED needs a validating test)")
    none = Hypothesis(title="endpoint may disclose config")
    check("no evidence -> UNCONFIRMED", none.status == "UNCONFIRMED", none.status)

    inferred = Hypothesis(title="SSRF via url param", evidence=[
        Evidence("model", "inference", "the param looks like a URL", 0.8)])
    check("only inference -> POSSIBLE at most", inferred.status in ("POSSIBLE", "UNCONFIRMED"),
          inferred.status)

    strong = Hypothesis(title="config exposed", evidence=[
        Evidence("http://t/config", "tool_output", "200 with db creds", 0.9),
        Evidence("http://t/server-status", "tool_output", "server-status open", 0.8)])
    check("two real tool results -> LIKELY (not CONFIRMED without a test)",
          strong.status == "LIKELY", strong.status)

    confirmed = Hypothesis(title="path traversal reads /etc/passwd", evidence=[
        Evidence("http_get ?file=../../etc/passwd", "validated_test", "root:x:0:0 returned", 0.95)])
    check("a validating test -> CONFIRMED", confirmed.status == "CONFIRMED", confirmed.status)

    print("\n2. CONFIRMED is UNREACHABLE without a validating test, however strong the rest")
    piled = Hypothesis(title="rce", evidence=[Evidence(f"s{i}", "tool_output", "x", 1.0)
                                              for i in range(6)])
    check("six tool_outputs still cap at LIKELY", piled.status == "LIKELY", piled.status)

    print("\n3. audit_claim downgrades an overclaim and says why")
    ok, actual, reason = audit_claim("CONFIRMED", inferred.evidence)
    check("claiming CONFIRMED on inference is rejected", ok is False and actual != "CONFIRMED")
    check("reason names it an overclaim", "OVERCLAIM" in reason, reason)
    ok2, actual2, _ = audit_claim("POSSIBLE", inferred.evidence)
    check("a claim within evidence is accepted", ok2 is True and actual2 == "POSSIBLE")
    ok3, _, _ = audit_claim("LIKELY", confirmed.evidence)
    check("under-claiming is fine (LIKELY <= CONFIRMED)", ok3 is True)

    print("\n4. rank orders by value-to-test-next (info gain + impact − cost)")
    cheap_high = Hypothesis(title="A cheap high-impact possible", impact=0.9, exploitability=0.8,
                            cost_to_verify=0.1, evidence=[Evidence("s", "observed", "x", 0.4)])
    done = Hypothesis(title="B already confirmed", impact=0.9, exploitability=0.9,
                      evidence=[Evidence("t", "validated_test", "x", 0.9)])
    trivial = Hypothesis(title="C low impact", impact=0.1, exploitability=0.1, cost_to_verify=0.9)
    order = [h.title[0] for h in rank([done, trivial, cheap_high])]
    check("untested high-impact cheap test ranks first", order[0] == "A", str(order))
    check("a CONFIRMED finding (no info gain) is not first", order[0] != "B", str(order))

    print("\n5. discovery -> validation -> exploitation staging")
    can, why = exploitation_gate(inferred)
    check("exploitation refused while only POSSIBLE", can is False and "validate" in why.lower())
    can2, why2 = exploitation_gate(confirmed)
    check("exploitation only PERMITS a proposal, still needs human+authorized",
          can2 is True and "human approval" in why2.lower() and "authorized target" in why2.lower())

    print("\n6. evidence-first report never presents a hypothesis as confirmed")
    rep = render_report(inferred)
    check("unvalidated report is labelled a HYPOTHESIS", "HYPOTHESIS, not a confirmed" in rep)
    check("unvalidated severity is marked ASSERTED", "ASSERTED" in rep)
    rep_c = render_report(confirmed)
    check("a CONFIRMED report shows CONFIRMED, no hypothesis warning",
          "STATUS: CONFIRMED" in rep_c and "not a confirmed" not in rep_c)

    print("\n7. reasoning block carries the full schema")
    r = render_reasoning(strong)
    check("has all point-1 fields", all(k in r for k in (
        "OBSERVATION", "EVIDENCE", "HYPOTHESIS", "WHY IT MATTERS", "CONFIDENCE", "NEXT TEST",
        "EXPECTED RESULT", "ALTERNATIVE EXPLANATION")))

    print("\n8. evidence requires provenance + weighs by kind")
    check("validating kind outweighs inference",
          Evidence("t", "validated_test", "", 0.5).strength()
          > Evidence("m", "inference", "", 1.0).strength())

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL FINDINGS TESTS PASS — status is earned, overclaims are downgraded, exploitation is")
    print("staged behind validation, and a hypothesis is never rendered as a confirmed vulnerability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
