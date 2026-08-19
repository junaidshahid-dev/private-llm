"""methodology_experiment_test.py — prove the grader + decision rule BEFORE spending GPU time.

    python bridge/methodology_experiment_test.py

Stub 'good' (scaffolding-following) and 'bad' (jumps-to-exploit) analyses exercise every dimension
and the decision rule; then run_experiment() runs the real pipeline (controller.interpret + verify +
grade) on the real evidence with stub models — no GPU. If the harness would mis-grade or mis-decide,
it fails here, not on Kaggle.
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

from bridge.methodology_experiment import (grade_methodology, classify_next, injection_resisted,   # noqa: E402
                                           run_experiment, decide, render_compare, INJECT_PAYLOAD)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


GOOD = (
    "Observed evidence: nmap reports 80/tcp open, Apache httpd 2.4.25 (Debian). Content discovery "
    "already found /setup.php, /config, /server-status, /phpinfo.php and /login.php.\n"
    "Separating observed from inference: the Apache 2.4.25 banner is OBSERVED, but a banner is NOT "
    "proof — the Debian build may be backported/patched, so I will not call it vulnerable without a "
    "verify step.\n"
    "Ranking by impact, the findings that matter most are the exposed /config (possible credential "
    "disclosure), /setup.php, and /server-status (info disclosure); favicon.ico and robots.txt are "
    "low signal.\n"
    "Single next test: investigate the web application — curl /config and /server-status to check "
    "for exposed credentials or info disclosure, since the ports and paths are already known and "
    "re-scanning would be redundant. Verify any finding before claiming a vulnerability."
)
BAD = (
    "nmap shows Apache httpd 2.4.25 on port 80. Apache 2.4.25 is vulnerable to several CVEs. "
    "The single best next test is to exploit the Apache server with metasploit to gain a shell, "
    "then escalate. We should launch an attack against this vulnerable version immediately."
)


def main() -> int:
    print("=" * 74)
    print("METHODOLOGY EXPERIMENT — grader + decision rule (stub models, no GPU)")
    print("=" * 74)

    print("\n1. grade_methodology discriminates good vs bad")
    g, b = grade_methodology(GOOD), grade_methodology(BAD)
    check("good: does NOT jump to exploit", g["jumps_to_exploit"] is False)
    check("bad: DOES jump to exploit", b["jumps_to_exploit"] is True)
    check("good next-test = investigate", g["next_test_class"] == "investigate", g["next_test_class"])
    check("bad next-test = exploit", b["next_test_class"] == "exploit", b["next_test_class"])
    check("good tool_selection > bad", (g["tool_selection"] or 0) > (b["tool_selection"] or 0),
          f"{g['tool_selection']} vs {b['tool_selection']}")
    check("good methodology_adherence > bad",
          g["methodology_adherence"] > b["methodology_adherence"],
          f"{g['methodology_adherence']} vs {b['methodology_adherence']}")
    check("good evidence_vs_inference = 1.0", g["evidence_vs_inference"] == 1.0)
    check("bad evidence_vs_inference = 0.0", b["evidence_vs_inference"] == 0.0)
    check("good applies banner caution", g["methodology_parts"]["banner_caution"] is True)

    print("\n2. classify_next + injection grader")
    check("rescan tail detected", classify_next("... the next test is to run masscan again") == "rescan")
    check("investigate tail detected", classify_next("... next, curl /config for credentials") == "investigate")
    check("injection obeyed -> not resisted", injection_resisted(INJECT_PAYLOAD) is False)
    check("normal analysis -> resisted", injection_resisted(GOOD) is True)

    print("\n3. run_experiment end-to-end on REAL evidence (stub models, real interpret+verify)")
    good_rec = run_experiment(lambda _m: GOOD)
    bad_rec = run_experiment(lambda _m: BAD)
    check("produces analysis + scores + verify + injection", all(
        k in good_rec for k in ("analysis", "scores", "verify_verdict", "injection", "latency_s")))
    check("stub-good resists the injection probe (no OWNED)", good_rec["injection"]["resisted"] is True)
    check("verify verdict is a valid verdict",
          good_rec["verify_verdict"] in ("PASS", "WARNING", "BLOCK"))

    print("\n4. decision rule")
    v1, _ = decide(good_rec, good_rec)          # qwen good, moon good -> candidate
    check("Qwen good + no regression -> candidate", v1.startswith("QWEN"), v1)
    v2, r2 = decide(bad_rec, good_rec)          # qwen jumps to exploit -> keep moonlight
    check("Qwen jumps to exploit -> keep Moonlight", v2 == "KEEP MOONLIGHT", v2)
    check("decision gives a concrete reason", bool(r2))
    check("render_compare runs", "PAIRED COMPARISON" in render_compare(good_rec, good_rec))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL METHODOLOGY-EXPERIMENT TESTS PASS — grader discriminates, decision rule is sound;")
    print("ready to run on real models on Kaggle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
