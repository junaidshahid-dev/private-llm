"""build_baseline_docs.py — turn results/base/results.json into a permanent, honest record.

    python evaluation/build_baseline_docs.py

Writes, next to results.json:
    metrics.json   machine-readable summary, per category and tier
    README.md      human summary with the caveats stated up front

WHY A "HEADLINE" THAT EXCLUDES A CATEGORY
The audit of the base run found that tool_calling scores ~0 not because the model is bad at
tool use, but because the benchmark harness never gives it a tool schema — the prompt just asks
"what's the weather in Lahore?" with no list of available tools, so the model answers
conversationally. That is a harness limitation, not a model weakness, so a headline that folds
those ten zeros in understates the model by ~0.10. The category is recorded, flagged invalid,
and excluded from the headline until the benchmark provides a tool schema (a v2 task).

Nothing here edits the frozen benchmark or its hash. This only documents a run against it.
"""
from __future__ import annotations

import io
import json
import os
import sys
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categories whose current benchmark items do not validly measure the model. Documented, not
# hidden — excluded from the headline with the reason attached.
KNOWN_INVALID = {
    "tool_calling": "harness provides no tool schema in the prompt, so the model answers "
                    "conversationally instead of emitting a tool call; measures the harness, "
                    "not the model. Needs a v2 with tool definitions injected.",
}


def main() -> int:
    base = os.path.join(HERE, "evaluation", "results", "base")
    data = json.load(open(os.path.join(base, "results.json"), encoding="utf-8"))
    rows = data["results"]

    def tier_of(gt):
        return "objective" if gt in ("exact", "code_test", "structural") else \
               "rubric" if gt == "rubric" else "judge"

    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)

    cat_metrics = {}
    for cat, rs in sorted(cats.items()):
        scored = [r for r in rs if r["score"] is not None]
        mean = round(sum(r["score"] for r in scored) / len(scored), 4) if scored else None
        cat_metrics[cat] = {
            "n": len(rs), "scored": len(scored), "mean": mean,
            "valid": cat not in KNOWN_INVALID,
            "note": KNOWN_INVALID.get(cat, ""),
        }

    def tier_mean(tier, exclude_invalid):
        rs = [r for r in rows if tier_of(r["grading_type"]) == tier and r["score"] is not None
              and not (exclude_invalid and r["category"] in KNOWN_INVALID)]
        return (round(sum(r["score"] for r in rs) / len(rs), 4), len(rs)) if rs else (None, 0)

    obj_all, n_all = tier_mean("objective", exclude_invalid=False)
    obj_valid, n_valid = tier_mean("objective", exclude_invalid=True)
    rub_all, rn = tier_mean("rubric", exclude_invalid=False)

    metrics = {
        "model": data["model"],
        "model_revision": data["model_revision"],
        "benchmark_sha256": data["benchmark_sha256"],
        "decode": data["decode"],
        "environment": data["environment"],
        "cost": data["cost"],
        "headline_objective_valid": obj_valid,
        "headline_objective_n": n_valid,
        "raw_objective_all": obj_all,
        "raw_objective_n": n_all,
        "rubric_mean": rub_all,
        "rubric_n": rn,
        "known_invalid_categories": KNOWN_INVALID,
        "categories": cat_metrics,
        "unscored_items": sum(1 for r in rows if r["score"] is None),
    }
    with open(os.path.join(base, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- README ----
    lines = []
    A = lines.append
    A(f"# Base benchmark — {data['model']}\n")
    A(f"Untuned Moonlight-16B-A3B-Instruct against frozen benchmark "
      f"`{data['benchmark_sha256'][:16]}`. This is the baseline every fine-tune is measured "
      f"against; it is committed so comparisons never re-measure a constant.\n")
    A("## Headline\n")
    A(f"- **Objective (valid categories): {obj_valid} over {n_valid} items** — the number to beat.")
    A(f"- Raw objective incl. known-invalid: {obj_all} over {n_all} (see caveat).")
    A(f"- Rubric (heuristic): {rub_all} over {rn}.")
    A(f"- Unscored (judge / prose / RAG-abstain): {metrics['unscored_items']}.\n")
    A("## Per category\n")
    A("| category | n | mean | valid | note |")
    A("|---|--:|--:|:--:|---|")
    for cat, m in cat_metrics.items():
        A(f"| {cat} | {m['n']} | {m['mean']} | {'yes' if m['valid'] else 'NO'} | {m['note']} |")
    A("\n## Caveats that change what to fine-tune\n")
    A("- **tool_calling ~0 is not a model weakness.** The harness sends no tool schema, so the "
      "model answers in prose instead of calling a tool. Fixing this is a benchmark v2 task "
      "(inject tool definitions), not a fine-tuning target. Excluded from the headline.")
    A("- **factuality 0/10 IS real.** The model invents nonexistent modules/flags/papers and "
      "states confidently wrong facts (e.g. 16B active params instead of 3B, self-attention "
      "instead of MLA, a fictional 'Kaggle K2' GPU). A legitimate, high-value fine-tuning "
      "target.")
    A(f"\n## Conditions (must match for any comparison)\n")
    A(f"- decode: greedy, max_new_tokens={data['decode']['max_new_tokens']}, "
      f"temperature={data['decode']['temperature']}")
    A(f"- transformers {data['environment']['transformers']}, "
      f"device {data['environment']['cuda_device']}")
    A(f"- cost: {data['cost']['wall_seconds']/60:.0f} min, "
      f"{data['cost']['mean_tokens_per_s']} tok/s, peak {data['cost']['peak_vram_gb']} GB\n")
    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("wrote evaluation/results/base/metrics.json and README.md")
    print(f"\n  headline objective (valid):  {obj_valid}  over {n_valid}")
    print(f"  raw objective (all):         {obj_all}  over {n_all}")
    print(f"  tool_calling (excluded):     {cat_metrics['tool_calling']['mean']}  — harness flaw")
    print(f"  factuality (real weakness):  {cat_metrics['factuality']['mean']}  — fine-tune this")
    return 0


if __name__ == "__main__":
    sys.exit(main())
