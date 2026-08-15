"""relevance_test.py — prove the relevance-gate logic (parse + filter) on CPU.

    python rag/relevance_test.py

A scripted judge stands in for the model, so the filtering behaviour — keep the named passages,
keep NONE on 'none', and FAIL CLOSED on anything unparseable — is tested without a GPU. The
judgment QUALITY (does the real model correctly call the XXE web-app doc irrelevant) needs the model
and is measured by run_secv3 --rag on the GPU.
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from rag.relevance import gate_relevance, parse_relevant_indices, build_relevance_prompt  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("RELEVANCE-GATE TEST — parse + filter, fail-closed")
    print("=" * 74)

    hits = [{"text": "web app security: XSS, CSRF, SQL injection, parameterized queries, CSP."},
            {"text": "network recon: nmap -sV banner interpretation."},
            {"text": "privilege escalation: sudo, SUID, GTFOBins."}]

    # ---- parsing --------------------------------------------------------------
    print("\n1. parse_relevant_indices")
    check("single number keeps that passage", parse_relevant_indices("1", 3) == [0])
    check("comma list", parse_relevant_indices("1,3", 3) == [0, 2])
    check("spaces and prose around numbers", parse_relevant_indices("passages 1 and 3", 3) == [0, 2])
    check("NONE => keep nothing", parse_relevant_indices("NONE", 3) == [])
    check("lowercase none => nothing", parse_relevant_indices("none", 3) == [])
    check("empty => nothing (fail closed)", parse_relevant_indices("", 3) == [])
    check("out-of-range number ignored => nothing", parse_relevant_indices("5", 3) == [])
    check("in-range survives even with junk", parse_relevant_indices("maybe 2?", 3) == [1])

    # ---- gating with a scripted judge ----------------------------------------
    print("\n2. gate_relevance")
    keep_first = gate_relevance("q", hits, judge_fn=lambda p: "1")
    check("keeps the passage the judge named", [h["text"] for h in keep_first] == [hits[0]["text"]])

    # the v3 XXE case: the judge correctly says none of these off-topic docs help
    xxe_keep = gate_relevance("How does XSD schema validation fail to stop XXE?", hits,
                              judge_fn=lambda p: "NONE")
    check("off-topic docs gated out => empty (defer to memory)", xxe_keep == [])

    # a genuinely relevant hit is kept
    ir = gate_relevance("privilege escalation via sudo?", hits, judge_fn=lambda p: "3")
    check("genuinely relevant passage kept", ir == [hits[2]])

    # fail closed on an unparseable judge reply
    fc = gate_relevance("q", hits, judge_fn=lambda p: "hmm, hard to say")
    check("unparseable judge reply => keep nothing (fail closed)", fc == [])

    check("no hits => no call, empty", gate_relevance("q", [], judge_fn=lambda p: "1") == [])

    # ---- prompt shape ---------------------------------------------------------
    print("\n3. build_relevance_prompt")
    p = build_relevance_prompt("what is XXE?", hits)
    check("prompt carries the question", "what is XXE?" in p)
    check("prompt lists numbered passages", "[1]" in p and "[3]" in p)
    check("prompt states related-but-different is NOT relevant", "NOT relevant" in p)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL RELEVANCE-GATE TESTS PASS (logic only; judgment quality needs the model on GPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
