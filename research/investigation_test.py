"""investigation_test.py — long-horizon state + the three stop conditions.

    python research/investigation_test.py
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
from research.investigation import (Investigation, action_signature,            # noqa: E402
                                    authorization_state, expected_information_gain)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def _res(tool, args, ok=True):
    return {"tool": tool, "arguments": args, "result": {"ok": ok}}


def main() -> int:
    print("=" * 74)
    print("INVESTIGATION — long-horizon state, info gain, and the three stop conditions")
    print("=" * 74)

    print("\n1. action signature is stable and order-independent")
    check("same args in any order -> same signature",
          action_signature("nmap_scan", {"target": "a", "flags": "-sV"})
          == action_signature("nmap_scan", {"flags": "-sV", "target": "a"}))
    check("different args -> different signature",
          action_signature("fs_read", {"path": "a"}) != action_signature("fs_read", {"path": "b"}))

    print("\n2. record_round tracks NEW vs REPEAT information")
    inv = Investigation(objective="assess lab host", max_rounds=8)
    g1 = inv.record_round([_res("nmap_scan", {"target": "lab"})])
    check("first round with a new ok action -> new_info 1", g1["new_info"] == 1, str(g1))
    g2 = inv.record_round([_res("nmap_scan", {"target": "lab"})])     # identical -> repeat
    check("re-running the same action -> new_info 0", g2["new_info"] == 0, str(g2))

    print("\n3. repeated-action detection")
    check("repeated action detected (round 2 re-ran round 1's action)",
          inv.repeated_action_detected() is True)
    esc, why = inv.should_escalate()
    check("escalates on repeated action", esc is True and "repeated" in why.lower(), why)

    print("\n4. diminishing-returns detection (2 consecutive no-gain rounds)")
    inv2 = Investigation(objective="x", max_rounds=8, stall_rounds=2)
    inv2.record_round([_res("fs_read", {"path": "a"})])               # gain 1
    inv2.record_round([_res("fs_read", {"path": "a"}, ok=False)])     # errored -> gain 0
    check("one no-gain round is not yet diminishing", inv2.diminishing_returns() is False)
    inv2.record_round([_res("fs_read", {"path": "a"})])               # repeat ok -> gain 0
    check("two consecutive no-gain rounds -> diminishing", inv2.diminishing_returns() is True)

    print("\n5. max-rounds stop condition")
    inv3 = Investigation(objective="x", max_rounds=1)
    inv3.record_round([_res("fs_read", {"path": "a"})])
    e, w = inv3.should_escalate()
    check("hitting max rounds escalates", e is True and "max rounds" in w.lower(), w)

    print("\n6. next_action picks the highest-value UNTESTED hypothesis")
    inv4 = Investigation(objective="web app assessment")
    inv4.add_hypothesis(Hypothesis(title="exposed /config discloses creds", impact=0.9,
                                   exploitability=0.7, cost_to_verify=0.1,
                                   next_test="GET /config and inspect the response",
                                   evidence=[Evidence("ffuf", "tool_output", "/config found", 0.7)]))
    inv4.add_hypothesis(Hypothesis(title="favicon quirk", impact=0.1, exploitability=0.1,
                                   next_test="inspect the favicon"))
    nxt = inv4.next_action()
    check("chooses the high-impact config hypothesis", "config" in nxt["hypothesis"], str(nxt))
    check("reports expected information gain", nxt["expected_information_gain"] > 0)
    check("config test is classified ACTIVE (a GET request needs authorization)",
          nxt["authorization"]["active"] is True, str(nxt["authorization"]))

    inv4.mark_tested("exposed /config discloses creds", confirmed=True)
    check("tested hypothesis moves out of remaining",
          "config" not in (inv4.next_action() or {}).get("hypothesis", ""))
    check("tested list reflects it", len(inv4.tested_hypotheses()) == 1)

    print("\n7. authorization classification of a proposed test")
    check("a source-code review is passive", authorization_state("review the source for the sink")["active"] is False)
    check("a scan is active", authorization_state("nmap scan the host")["active"] is True)

    print("\n8. expected information gain falls to ~0 once CONFIRMED")
    confirmed = Hypothesis(title="pt", evidence=[Evidence("t", "validated_test", "root:x", 0.9)])
    open_h = Hypothesis(title="op")
    check("CONFIRMED has ~0 expected info gain", expected_information_gain(confirmed) == 0.0)
    check("an open hypothesis has high expected info gain",
          expected_information_gain(open_h) > expected_information_gain(confirmed))

    print("\n9. render carries all the per-iteration fields")
    r = inv4.render()
    check("render has the required state fields", all(k in r for k in (
        "OBJECTIVE", "ROUND", "KNOWN EVIDENCE", "TESTED HYPOTHESES", "REMAINING HYPOTHESES",
        "PROPOSED NEXT ACTION", "authorization", "ESCALATE TO HUMAN")))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL INVESTIGATION TESTS PASS — the loop knows its state, ranks the next action, and")
    print("escalates to a human on repeated actions, diminishing returns, or the round cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
