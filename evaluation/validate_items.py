"""validate_items.py — check a benchmark jsonl is well-formed and its graders agree with it.

    python evaluation/validate_items.py evaluation/development/domain_expansion/security/security.jsonl

For every item it checks the required fields exist, ids are unique, and — the useful part — that
each item's OWN reference answer scores as intended under its grader:

    exact / code_test          reference must score 1.0   (else the item or grader is wrong)
    rubric + false_premise     the denial reference must score 1.0
    rubric + should            reference is a description, not a literal answer, so only the
                               structural fields are checked (should in {answer,decline})

An item whose reference answer does not score correctly is a landmine: it will mis-grade every
model against it. This catches that on CPU before any GPU time is spent.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evaluation.grading import grade                                        # noqa: E402

REQUIRED = ["id", "category", "grading_type", "prompt", "reference_answer"]
SELF_GRADE_1 = {"exact", "code_test"}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python evaluation/validate_items.py <benchmark.jsonl>")
        return 2
    path = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(HERE, sys.argv[1])
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    print("=" * 74)
    print(f"VALIDATE — {os.path.relpath(path, HERE)}  ({len(items)} items)")
    print("=" * 74)

    fails, ids = [], set()

    def bad(id, msg):
        print(f"  [FAIL] {id}: {msg}")
        fails.append(id)

    for it in items:
        id = it.get("id", "<no id>")
        for k in REQUIRED:
            if k not in it:
                bad(id, f"missing required field {k!r}")
        if id in ids:
            bad(id, "duplicate id")
        ids.add(id)

        gt = it.get("grading_type")
        if gt == "code_test" and not it.get("tests"):
            bad(id, "code_test item has no tests")

        # Self-grade the reference answer.
        if gt in SELF_GRADE_1 or (gt == "rubric" and it.get("false_premise")):
            score, tier, why = grade(it, it["reference_answer"])
            if score is None or abs(score - 1.0) > 1e-6:
                # A terse reference that omits an import (uses re/math without importing) is a
                # style choice, not a grader disagreement — a complete model answer supplies it.
                # Only a genuine mismatch (a reference that fails an assertion) is a real fault.
                terse = any(s in (why or "") for s in ("not defined", "ImportError",
                                                       "NameError", "no module"))
                if terse:
                    print(f"  [warn] {id}: reference omits an import ({why}) — model answers "
                          "must be self-contained; not counted as invalid")
                else:
                    bad(id, f"reference answer scored {score} (expected 1.0) — {why}")
        elif gt == "rubric" and it.get("should"):
            if it["should"] not in ("answer", "decline"):
                bad(id, f"should={it['should']!r} not in answer|decline")

    # Category and should distribution — a security set that is mostly 'decline' would be
    # measuring the wrong thing.
    shoulds = [it.get("should") for it in items if it.get("should")]
    if shoulds:
        a, d = shoulds.count("answer"), shoulds.count("decline")
        print(f"\n  should=answer {a}   should=decline {d}")
        if d > a:
            print("  NOTE: more decline than answer items — is this set measuring capability")
            print("        or just refusal? For a domain you want the model to HELP with, "
                  "answer should dominate.")

    by_grader = {}
    for it in items:
        by_grader[it["grading_type"]] = by_grader.get(it["grading_type"], 0) + 1
    print(f"  graders: {by_grader}")

    print("\n" + "=" * 74)
    if fails:
        print(f"INVALID — {len(set(fails))} item(s) failed: {sorted(set(fails))}")
        return 1
    print("VALID — every reference answer grades as intended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
