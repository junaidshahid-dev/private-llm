"""verify_test.py — prove the verification layer catches planted defects and does not cry wolf.

    python verification/verify_test.py

Same contract as evaluation/test_harness.py: feed answers whose correct verdict is known. A
verifier that misses a deliberately broken answer cannot be trusted on a real one — and one that
flags number-heavy prose (ports, CVE ids, versions, dates) is worse than none, because you stop
reading it. Both directions are tested.
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from verification.verify import verify, check_math                    # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def has(report, kind, level=None):
    return any(f.kind == kind and (level is None or f.level == level) for f in report.findings)


def main() -> int:
    print("=" * 74)
    print("VERIFY TEST — planted defects must be caught, prose must not false-positive")
    print("=" * 74)

    # ---- 1. math: catches wrong arithmetic ----------------------------------
    print("\n1. math — real computation errors")
    check("correct product passes", not has(verify("So 10 * 1.06 = 10.6 after one year."), "math"))
    check("correct division passes", not has(verify("That gives 100 / 4 = 25 per shard."), "math"))
    check("wrong sum is an error", has(verify("Clearly 2 + 2 = 5 here."), "math", "error"))
    # the SFT truncation bug: fraction reported instead of the finished percentage
    tr = verify("The compounded return is 0.14 * 70 = 0.098.")
    check("truncated result (0.098 vs 9.8) is caught", has(tr, "math", "error"),
          str(tr.errors[0]) if tr.errors else "missed")
    check("percent-of, correct, passes", not has(verify("20% of 50 is 10 users."), "math"))
    check("percent-of, wrong, is an error", has(verify("10% of 200 is 2."), "math", "error"))

    # ---- 2. math: does NOT flag number-heavy prose --------------------------
    print("\n2. math — must NOT cry wolf on non-equations")
    prose = ("nmap -sV shows 445/tcp open microsoft-ds Samba 3.0.20, which maps to "
             "CVE-2007-2447, first reported 2007-05-14. Upgrade past 3.0.25.")
    r = verify(prose)
    check("ports / CVE / versions / dates do not trip math", not has(r, "math"),
          "; ".join(str(f) for f in r.findings) or "clean")
    check("rounded value within tolerance passes", not has(verify("1 / 3 = 0.3333"), "math"))

    # ---- 3. code ------------------------------------------------------------
    print("\n3. code — syntax is checked, correctness is not overclaimed")
    good = "Here:\n```python\ndef f(x):\n    return x * 2\n```\n"
    rc = verify(good)
    check("valid code does not error", not has(rc, "code", "error"))
    check("valid code is only INFO, not claimed correct", has(rc, "code", "info"))
    bad = "```python\ndef f(x):\n    return x *\n```"
    check("syntax error is caught", has(verify(bad), "code", "error"))

    # ---- 4. grounding -------------------------------------------------------
    print("\n4. grounding — auditable, but never gags correct parametric knowledge")
    hits = [{"text": "SSRF can reach the metadata service at 169.254.169.254."},
            {"text": "Use IMDSv2 to mitigate."}]
    check("citation past the source count warns",
          has(verify("As in [5], block it.", hits=hits), "grounding", "warn"))
    # a CVE not in the context is a WARNING (verify it) but never an ERROR (may be correct knowledge)
    g = verify("This is CVE-2007-2447 in Samba.", hits=hits)
    check("unsupported factual claim is WARN, not error",
          has(g, "grounding", "warn") and g.ok, f"verdict={g.verdict}")
    check("unsupported claim => verdict WARNING, not BLOCK", g.verdict == "WARNING")
    check("no hits => no grounding noise", not has(verify("anything at all"), "grounding"))

    # ---- 5. phantom actions -------------------------------------------------
    print("\n5. phantom actions — claiming to DO what no tool did")
    p = verify("I scanned the host and found port 445 open.", tools_ran=None)
    check("fabricated action is an error", has(p, "phantom_action", "error"))
    check("same claim is fine when a tool really ran",
          not has(verify("I scanned the host and found port 445 open.",
                          tools_ran=["nmap_scan"]), "phantom_action"))
    check("describing (not claiming) does not trip",
          not has(verify("You can scan the host with nmap -sV to list services."),
                  "phantom_action"))

    # ---- 6. report semantics: the PASS / WARNING / BLOCK verdict -------------
    print("\n6. verdict — three non-destructive tiers")
    check("clean answer => PASS", verify("Least privilege limits blast radius.").verdict == "PASS")
    check("PASS answer has no findings", not verify("Least privilege limits blast radius.").findings)
    check("unsupported factual claim => WARNING",
          verify("CVE-2007-2447", hits=hits).verdict == "WARNING")
    check("arithmetic error => BLOCK", verify("2 + 2 = 5").verdict == "BLOCK")
    check("phantom action => BLOCK",
          verify("I scanned the box.", tools_ran=None).verdict == "BLOCK")
    check("BLOCK sets ok False", verify("2 + 2 = 5").ok is False)
    check("WARNING keeps ok True", verify("CVE-2007-2447", hits=hits).ok is True)
    check("render states the verdict and never claims proof of correctness",
          "Verification: PASS" in verify("Least privilege wins.").render()
          and "not proof" in verify("Least privilege wins.").render())

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL VERIFY TESTS PASS — the layer catches planted defects without crying wolf.")
    print("(Reminder: no flag is not proof of correctness — heuristics have blind spots.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
