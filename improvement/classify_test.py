"""classify_test.py — the failure classifier routes each class to the right layer.

    python improvement/classify_test.py

Every case here is one the project actually hit. The classifier must send each to the layer that
should be fixed — and must diagnose evaluator/tool problems BEFORE blaming the model, so we never
repeat the SFT mistake of retraining when the fault was in RAG or evaluation.
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

from improvement.classify import classify_failure                            # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def primary(record, tools=None):
    return classify_failure(record, tools)["primary"]


def main() -> int:
    print("=" * 74)
    print("FAILURE CLASSIFIER TEST — route each real failure class to its layer")
    print("=" * 74)

    # evaluator first: an unreliable grader must not be read as a model failure
    check("unparsed judge => evaluator",
          primary({"score": None, "judge_detail": "judge output did not parse"}) == "evaluator")
    check("judge/deterministic divergence => evaluator",
          primary({"score": 1.0, "det_score": 0.25, "divergence": 0.75}) == "evaluator")

    # tool execution beats a model diagnosis (the a1b2c3d4 / roots case)
    check("a tool error => tool_execution",
          primary({"score": 0.0, "tool_results": [{"tool": "fs_read",
                   "result": {"ok": False, "error": "not a file"}}]}) == "tool_execution")

    # fabrication / phantom => prompt_policy (honesty), not weights
    check("fabricated output => prompt_policy",
          primary({"score": 0.2, "verify_findings": ["[ERROR:tool_grounding] appears to FABRICATE"],
                   "tool_results": [{"tool": "x", "result": {"ok": True}}]}) == "prompt_policy")

    # retrieval: grounded and still failed (the XXE case)
    check("grounded-but-failed => retrieval",
          primary({"score": 0.17, "grounded": True, "kept_after_gate": 1,
                   "verify_findings": ["[warn:grounding] off-topic"]}) == "retrieval")

    # missing knowledge: failed, nothing relevant retrieved, no tools
    check("failed with no relevant retrieval => missing_knowledge",
          primary({"score": 0.2, "grounded": False}) == "missing_knowledge")

    # refusal => prompt_policy
    check("refusal => prompt_policy",
          primary({"score": 0.0, "output": "I cannot access or read files directly."})
          == "prompt_policy")

    # tool selection: proposed a tool that does not exist
    check("proposed a non-existent tool => tool_selection",
          primary({"score": 0.0, "proposed_tools": ["repo_inspector"]},
                  tools={"fs_read", "fs_list"}) == "tool_selection")

    # pure reasoning failure: no other signal -> model_reasoning (NOT retrain blindly)
    r = classify_failure({"score": 0.3, "grounded": None, "verify_verdict": "PASS"})
    check("bare low score => model_reasoning", r["primary"] == "model_reasoning")
    check("model_reasoning recommendation warns against blanket SFT",
          "NOT blanket fine-tuning" in r["recommendation"] or "stronger base" in r["recommendation"])
    check("verification appears as a SECONDARY note, not primary",
          any(l["layer"] == "verification" for l in r["layers"]) and r["primary"] != "verification")

    # a good result is not a failure
    ok = classify_failure({"score": 1.0, "verify_verdict": "PASS"})
    check("high score => not a failure", ok["failure"] is False)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL CLASSIFIER TESTS PASS — failures route to the right layer; the model is blamed last.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
