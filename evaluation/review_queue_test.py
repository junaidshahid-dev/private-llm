"""review_queue_test.py — the triage flags the uncertain items and spares the confident ones.

    python evaluation/review_queue_test.py
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

from evaluation.review_queue import build_queue, review_reasons          # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def row(id, score, verdict="PASS", detail="", output="an answer"):
    return {"id": id, "category": "x", "score": score, "verify_verdict": verdict,
            "judge_detail": detail, "output": output}


def main() -> int:
    print("=" * 74)
    print("REVIEW-QUEUE TEST — flag the uncertain, spare the confident")
    print("=" * 74)

    rows = [
        row("clean", 1.0, "PASS", "4/4 points shown"),                       # confident: skip
        row("partial", 0.75, "PASS", "3/4 points shown"),                    # contradiction-like
        row("low", 0.17, "PASS", "2/3 points; HARMFUL must-not #[1]"),       # XXE-like + harmful
        row("disagree", 1.0, "BLOCK", "4/4 points shown"),                   # smuggling-like
        row("unparsed", None, "PASS", "judge output did not parse"),         # no score
    ]
    q = build_queue(rows)
    ids = {i["id"] for i in q}

    check("confident 1.0/PASS item is NOT flagged", "clean" not in ids)
    check("partial 0.75 flagged (the judge-miss band)", "partial" in ids)
    check("low 0.17 flagged", "low" in ids)
    check("judge-vs-verify disagreement flagged", "disagree" in ids)
    check("unparsed judge flagged", "unparsed" in ids)
    check("exactly the 4 uncertain items flagged", ids == {"partial", "low", "disagree", "unparsed"},
          str(sorted(ids)))

    check("unparsed sorts first (priority 0)", q[0]["id"] == "unparsed", q[0]["id"])
    dis = next(i for i in q if i["id"] == "disagree")
    check("disagreement reason names both signals",
          any("verification=BLOCK" in r and "judge=1.00" in r for r in dis["reasons"]))
    low = next(i for i in q if i["id"] == "low")
    check("harmful-flagged item carries the high-stakes reason",
          any("forbidden/harmful" in r for r in low["reasons"]))

    # divergence between the judge and the deterministic grader is a review trigger
    div_row = {"id": "div", "category": "x", "score": 0.75, "verify_verdict": "PASS",
               "judge_detail": "3/4", "det_score": 0.25, "divergence": 0.50, "output": "a"}
    check("judge-vs-deterministic divergence is flagged (P1)",
          any("DIVERGENCE" in msg for _, msg in review_reasons(div_row)))
    agree_row = dict(div_row, id="agree", score=1.0, det_score=1.0, divergence=0.0)
    check("agreement between the two graders is NOT flagged for divergence",
          not any("DIVERGENCE" in msg for _, msg in review_reasons(agree_row)))

    check("empty results => empty queue", build_queue([]) == [])
    check("a lone clean item yields no reasons", review_reasons(row("c", 1.0)) == [])

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL REVIEW-QUEUE TESTS PASS — triage surfaces the few items worth a human's time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
