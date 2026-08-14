"""diff_outputs.py — read the ACTUAL text change from base to fine-tuned, item by item.

    python evaluation/diff_outputs.py --base evaluation/results/base \
        --candidate evaluation/results/experiment-003 --category factuality

compare.py answers "did the score move?". This answers "HOW did the model change?" — it prints,
for each item, the prompt, the base model's output, the fine-tuned output, and the score delta,
side by side. A score of 0->1 on fact_001 is data; seeing "invents a fake module" become
"correctly says it doesn't exist" is understanding.

Use it on the categories that matter after a run: factuality (did it stop hallucinating?),
security/behavior (did it stay capable and stop over-refusing?), and any category compare.py
flagged as regressed (what actually broke?).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1", "benchmark.jsonl")


def load_results(p):
    p = p if os.path.isabs(p) else os.path.join(HERE, p)
    f = p if p.endswith(".json") else os.path.join(p, "results.json")
    if not os.path.exists(f):
        sys.exit(f"no results at {f}")
    data = json.load(open(f, encoding="utf-8"))
    return data, {r["id"]: r for r in data["results"]}


def wrap(text, indent):
    return ("\n" + " " * indent).join((text or "").strip().splitlines()) or "(empty)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--category", help="only this category")
    ap.add_argument("--ids", help="comma-separated item ids")
    ap.add_argument("--only", choices=["improved", "regressed", "changed", "all"],
                    default="changed", help="which items to show (default: score changed)")
    ap.add_argument("--clip", type=int, default=500, help="max chars of each output shown")
    args = ap.parse_args()

    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}
    bdata, b = load_results(args.base)
    cdata, c = load_results(args.candidate)

    ids = [i.strip() for i in args.ids.split(",")] if args.ids else \
        [i for i in b if i in c]
    if args.category:
        ids = [i for i in ids if b.get(i, {}).get("category") == args.category]

    print("=" * 90)
    print(f"OUTPUT DIFF — base '{bdata['name']}'  vs  candidate '{cdata['name']}'"
          + (f"   category={args.category}" if args.category else ""))
    print("=" * 90)

    shown = 0
    for i in ids:
        if i not in b or i not in c:
            continue
        sb, sc = b[i]["score"], c[i]["score"]
        if sb is not None and sc is not None:
            d = sc - sb
            if args.only == "improved" and not d > 1e-9:
                continue
            if args.only == "regressed" and not d < -1e-9:
                continue
            if args.only == "changed" and abs(d) <= 1e-9:
                continue
            delta = f"{sb:.2f} -> {sc:.2f}  ({d:+.2f})"
        else:
            if args.only in ("improved", "regressed", "changed"):
                continue
            delta = f"{sb} -> {sc}"

        item = prompts.get(i, {})
        print(f"\n{'─'*90}\n[{i}]  {b[i]['category']}   score {delta}")
        if item.get("prompt"):
            print(f"  prompt : {wrap(item['prompt'][:300], 11)}")
        if item.get("reference_answer"):
            print(f"  expect : {str(item['reference_answer'])[:150]}")
        print(f"  BASE   : {wrap(b[i]['output'][:args.clip], 11)}")
        print(f"  TUNED  : {wrap(c[i]['output'][:args.clip], 11)}")
        shown += 1

    print(f"\n{'='*90}\n{shown} items shown"
          + (f" (filter: {args.only})" if args.only != "all" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
