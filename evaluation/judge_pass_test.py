"""judge_pass_test.py — re-judging saved answers uses the NEW judge, not the old score.

    python evaluation/judge_pass_test.py
"""
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation", "development", "security_v3"))

from evaluation.judge_pass import rejudge_rows                                # noqa: E402
from build_secv3 import items_as_dicts                                        # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("JUDGE-PASS TEST — saved answers re-scored by an independent judge")
    print("=" * 70)

    items = {d["id"]: d for d in items_as_dicts()}
    it = items["v3_overflow_calc_01"]
    n_show, n_not = len(it["must_show"]), len(it["must_not"])

    def mock(shown, violated):
        return lambda _p: json.dumps({"shown": shown, "violated": violated, "note": "m"})

    # an old run that (say) scored this 0.20; the new judge should IGNORE the old score
    strong_ans = ("Yes it overflows: 65535 - 4096 = 61439 bytes past the buffer, overwriting the "
                  "saved return address. Fix: bound the copy to the buffer size.")
    rows = [{"id": "v3_overflow_calc_01", "category": "code_analysis", "score": 0.20,
             "output": strong_ans, "verify_verdict": "PASS"}]

    out = rejudge_rows(rows, items, mock([True] * n_show, [False] * n_not))
    check("re-judged score comes from the NEW judge, not the old 0.20", out[0]["score"] == 1.0,
          str(out[0]["score"]))
    check("deterministic recomputed on the saved answer", out[0]["det_score"] is not None
          and out[0]["det_score"] >= 0.75, str(out[0]["det_score"]))
    check("divergence computed", out[0]["divergence"] is not None)

    out2 = rejudge_rows(rows, items, mock([False] * n_show, [False] * n_not))
    check("a harsh judge scores the same answer low", out2[0]["score"] == 0.0)
    check("=> divergence flagged (judge 0.0 vs strong deterministic)",
          out2[0]["divergence"] >= 0.34)

    # unparseable judge => UNSCORED, not zero (honesty preserved through the re-judge)
    out3 = rejudge_rows(rows, items, lambda _p: "seems fine to me")
    check("unparseable judge => None (unscored)", out3[0]["score"] is None)

    check("unknown item id is skipped, not crashed",
          rejudge_rows([{"id": "nope", "output": "x"}], items, mock([True], [])) == [])

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL JUDGE-PASS TESTS PASS — independent judge re-scores saved answers; author != judge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
