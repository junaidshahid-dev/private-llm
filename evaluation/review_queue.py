"""review_queue.py — make human spot-checks FIRST-CLASS: surface exactly which graded items a
human should re-check, instead of trusting the self-judge or re-reading all of them.

    python evaluation/review_queue.py evaluation/results/secv3_rag/results.json

The self-judge is a signal, not a verdict (it already under-scored a good contradiction answer
0.75 and rubber-stamped a base run 1.000). This tool ranks items by how much CONFIDENCE we can put
in their score, using disagreements between independent signals we already record:

  * judge output UNPARSED            -> there is no score to trust at all
  * judge flagged a HARMFUL claim    -> high stakes; confirm the answer really is dangerous/wrong
  * judge vs VERIFICATION disagree    -> e.g. verify BLOCK/WARNING while the judge scored high
  * LOW judge score (<0.5)            -> real failure, or a judge miss? (v3 XXE was real; confirm)
  * PARTIAL judge score (0.5-<1.0)    -> borderline; the exact band where the judge erred before

Nothing here re-grades. It produces a short, ranked "check these by hand" list so a human verifies
the few uncertain items rather than all of them — deterministic triage over an unreliable judge.
CPU-only; runs on results.json files that already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (severity, reason) — lower severity number = check first
def review_reasons(r: dict) -> list[tuple[int, str]]:
    score = r.get("score")
    detail = r.get("judge_detail", "") or ""
    verdict = r.get("verify_verdict", "PASS")
    out: list[tuple[int, str]] = []
    if score is None:
        out.append((0, "judge output UNPARSED — no score to trust"))
    if "HARMFUL" in detail:
        out.append((1, "judge flagged a forbidden/harmful claim — confirm it is real, high stakes"))
    if verdict in ("WARNING", "BLOCK") and score is not None and score >= 0.8:
        out.append((1, f"DISAGREEMENT: verification={verdict} but judge={score:.2f} — one is wrong"))
    det, div = r.get("det_score"), r.get("divergence")
    if div is not None and div >= 0.34:
        out.append((1, f"DIVERGENCE: judge={score:.2f} vs deterministic={det:.2f} — the two "
                    "independent graders disagree; the judge may be wrong"))
    if score is not None and score < 0.5:
        out.append((2, f"LOW judge score {score:.2f} — real failure or a judge miss? confirm"))
    elif score is not None and 0.5 <= score < 1.0:
        out.append((3, f"PARTIAL judge score {score:.2f} — borderline; spot-check"))
    return out


def build_queue(rows: list[dict]) -> list[dict]:
    q = []
    for r in rows:
        reasons = review_reasons(r)
        if reasons:
            q.append({"id": r.get("id"), "category": r.get("category"),
                      "score": r.get("score"), "verdict": r.get("verify_verdict"),
                      "priority": min(p for p, _ in reasons),
                      "reasons": [msg for _, msg in sorted(reasons)],
                      "snippet": (r.get("output") or "").strip().replace("\n", " ")[:160]})
    return sorted(q, key=lambda x: x["priority"])


def render(path: str, rows: list[dict], q: list[dict]) -> str:
    L = [f"HUMAN REVIEW QUEUE — {os.path.relpath(path, HERE)}",
         f"{len(q)} of {len(rows)} items need a human check "
         f"(the rest agree across judge + verification).", "=" * 74]
    if not q:
        L.append("nothing flagged — but remember a clean pass is not proof; "
                 "the self-judge can still be uniformly wrong (spot-check a random item anyway).")
    for item in q:
        sc = "—" if item["score"] is None else f"{item['score']:.2f}"
        L.append(f"\n[P{item['priority']}] {item['id']}  ({item['category']})  "
                 f"score={sc}  verify={item['verdict']}")
        for reason in item["reasons"]:
            L.append(f"    - {reason}")
        L.append(f"    answer: {item['snippet']}…")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(HERE, "evaluation", "results", "secv3_rag", "results.json"),
                    help="a results.json from run_secv3 / run_secv2")
    args = ap.parse_args()
    if not os.path.exists(args.results):
        sys.exit(f"no results file at {args.results}")
    data = json.load(open(args.results, encoding="utf-8"))
    rows = data.get("results", [])
    print(render(args.results, rows, build_queue(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
