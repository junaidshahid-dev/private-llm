"""audit_results.py — read the actual model outputs behind a score, to separate real model
failure from grader strictness.

    python evaluation/audit_results.py evaluation/results/base [category]

A category that scores 0 across the board is either a real weakness (worth fine-tuning) or a
grader that misses correct answers (worth fixing before wasting a training run on it). The only
way to tell is to read what the model actually said. This prints, per item, the prompt, the
reference, the grader's verdict, and the model's real output — so a human can judge the judge.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python evaluation/audit_results.py <results_dir> [category] [--full]")
        return 2
    d = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(HERE, sys.argv[1])
    f = d if d.endswith(".json") else os.path.join(d, "results.json")
    data = json.load(open(f, encoding="utf-8"))
    cat_filter = next((a for a in sys.argv[2:] if not a.startswith("--")), None)
    full = "--full" in sys.argv
    clip = 100000 if full else 600

    rows = data["results"]
    if cat_filter:
        rows = [r for r in rows if r["category"] == cat_filter]

    print("=" * 90)
    print(f"AUDIT — {data['name']}   {len(rows)} items"
          + (f"   category={cat_filter}" if cat_filter else ""))
    print("=" * 90)

    for r in rows:
        sc = r["score"]
        tag = "UNSCORED" if sc is None else f"{sc:.2f}"
        print(f"\n{'─'*90}\n[{r['id']}]  score={tag}  grader={r['grading_type']}  "
              f"({r['output_tokens']}tok)")
        print(f"  verdict : {r['explanation']}")
        out = (r["output"] or "").strip().replace("\n", "\n            ")
        print(f"  output  : {out[:clip]}" + (" …[clipped]" if len(out) > clip else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
