"""compare.py — base vs fine-tuned, per item, per category, per tier.

    python evaluation/compare.py --base evaluation/results/base \
                                 --candidate evaluation/results/experiment-004

THE POINT OF THIS FILE IS TO MAKE A FLATTERING NUMBER HARD TO REPORT.

A single "score went up" hides the thing that actually matters: what did the fine-tune break?
LoRA moves the whole model, not only the part you meant to improve. A run that gains 8% on
coding and loses 9% on factuality has not improved — it has traded, and the trade may be bad.

So:

  * items are matched by id and compared INDIVIDUALLY, then aggregated
  * improved / regressed / unchanged are counted separately and always printed together
  * any category dropping past REGRESSION_THRESHOLD is flagged as a blocker
  * objective, rubric and judge tiers are never blended into one headline
  * the run refuses outright if the two sides were not measured under the same conditions

That last one matters most. Two runs with different decode settings, benchmark hashes, model
revisions or patch sets produce a delta that measures the difference in conditions, not in the
model. That comparison is worse than none, because it looks like evidence.
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

REGRESSION_THRESHOLD = 0.05      # category mean drop that counts as a blocker
EPSILON = 1e-9                   # below this a difference is not a change

# Conditions that must match for a delta to mean anything.
MUST_MATCH = [
    ("benchmark_sha256", "benchmark contents"),
    ("model_revision", "base model revision"),
]
MUST_MATCH_DECODE = ["do_sample", "temperature", "top_p", "max_new_tokens",
                     "repetition_penalty", "system_prompt"]


def load(p):
    p = p if os.path.isabs(p) else os.path.join(HERE, p)
    f = p if p.endswith(".json") else os.path.join(p, "results.json")
    if not os.path.exists(f):
        sys.exit(f"no results at {f}")
    return json.load(open(f, encoding="utf-8"))


def bar(v, width=22):
    """Signed bar. Left of centre is a regression."""
    half = width // 2
    n = min(half, int(round(abs(v) * half / 0.25)))     # full scale at ±0.25
    return (" " * (half - n) + "#" * n + "|" + " " * half) if v < 0 else \
           (" " * half + "|" + "#" * n + " " * (half - n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="compare anyway despite differing conditions (records the override)")
    args = ap.parse_args()

    b, c = load(args.base), load(args.candidate)

    print("=" * 76)
    print("BASE vs FINE-TUNED")
    print(f"  base       {b['name']:<28} {b.get('adapter') or 'no adapter'}")
    print(f"  candidate  {c['name']:<28} {c.get('adapter') or 'no adapter'}")
    print("=" * 76)

    # ---- condition guard -----------------------------------------------------
    problems = []
    for key, label in MUST_MATCH:
        if b.get(key) != c.get(key):
            problems.append(f"{label}: {str(b.get(key))[:16]} vs {str(c.get(key))[:16]}")
    for k in MUST_MATCH_DECODE:
        if b.get("decode", {}).get(k) != c.get("decode", {}).get(k):
            problems.append(f"decode.{k}: {b['decode'].get(k)} vs {c['decode'].get(k)}")
    if b.get("upstream_patches") != c.get("upstream_patches"):
        problems.append(f"patches: {b.get('upstream_patches')} vs {c.get('upstream_patches')}")
    if b.get("environment", {}).get("transformers") != c.get("environment", {}).get("transformers"):
        problems.append(f"transformers: {b['environment'].get('transformers')} vs "
                        f"{c['environment'].get('transformers')}")

    if problems:
        print("\n  CONDITIONS DIFFER — a delta measured across these is not about the model:")
        for p in problems:
            print(f"    - {p}")
        if not args.allow_mismatch:
            print("\n  Refusing to compare. Re-run one side under matching conditions, or pass")
            print("  --allow-mismatch if you genuinely intend to measure the difference in")
            print("  conditions rather than in the model.")
            return 1
        print("\n  --allow-mismatch given; continuing under protest. Recorded in the report.")
    else:
        print(f"\n  conditions match: benchmark {b['benchmark_sha256'][:12]}, "
              f"greedy, max_new_tokens={b['decode']['max_new_tokens']}")

    # ---- per-item ------------------------------------------------------------
    bi = {r["id"]: r for r in b["results"]}
    ci = {r["id"]: r for r in c["results"]}
    common = [i for i in bi if i in ci]
    only_b, only_c = set(bi) - set(ci), set(ci) - set(bi)
    if only_b or only_c:
        print(f"  WARNING: {len(only_b)} items only in base, {len(only_c)} only in candidate")

    improved, regressed, unchanged, unscored = [], [], [], []
    for i in common:
        sb, sc = bi[i]["score"], ci[i]["score"]
        if sb is None or sc is None:
            unscored.append(i)
            continue
        d = sc - sb
        (improved if d > EPSILON else regressed if d < -EPSILON else unchanged).append((i, sb, sc, d))

    # ---- tiers ---------------------------------------------------------------
    def tier_of(r):
        gt = r["grading_type"]
        return "objective" if gt in ("exact", "code_test", "structural") else \
               "rubric" if gt == "rubric" else "judge"

    print("\n" + "-" * 76)
    print("  TIER                    n      base    tuned     delta")
    print("-" * 76)
    tier_rows = {}
    for tier in ("objective", "rubric", "judge"):
        ids = [i for i in common if tier_of(bi[i]) == tier
               and bi[i]["score"] is not None and ci[i]["score"] is not None]
        if not ids:
            n_tot = sum(1 for i in common if tier_of(bi[i]) == tier)
            print(f"  {tier:<20} {n_tot:>4}         —        —         —   (unscored)")
            tier_rows[tier] = None
            continue
        mb = sum(bi[i]["score"] for i in ids) / len(ids)
        mc = sum(ci[i]["score"] for i in ids) / len(ids)
        print(f"  {tier:<20} {len(ids):>4}    {mb:>6.3f}   {mc:>6.3f}   {mc-mb:>+7.3f}")
        tier_rows[tier] = {"n": len(ids), "base": round(mb, 4), "candidate": round(mc, 4),
                           "delta": round(mc - mb, 4)}
    print("-" * 76)
    print("  The objective row is the headline. Rubric is heuristic and supports it.")
    print("  Judge items are unscored until a judge is explicitly configured.")

    # ---- categories ----------------------------------------------------------
    cats = sorted({bi[i]["category"] for i in common})
    print("\n" + "-" * 76)
    print(f"  CATEGORY              n    base   tuned    delta   {'regression':^24}")
    print("-" * 76)
    cat_rows, blockers = {}, []
    for cat in cats:
        ids = [i for i in common if bi[i]["category"] == cat
               and bi[i]["score"] is not None and ci[i]["score"] is not None]
        if not ids:
            n_tot = sum(1 for i in common if bi[i]["category"] == cat)
            print(f"  {cat:<20} {n_tot:>3}      —       —        —   all items unscored")
            cat_rows[cat] = None
            continue
        mb = sum(bi[i]["score"] for i in ids) / len(ids)
        mc = sum(ci[i]["score"] for i in ids) / len(ids)
        d = mc - mb
        flag = ""
        if d <= -REGRESSION_THRESHOLD:
            flag = "  <-- REGRESSION"
            blockers.append((cat, d))
        print(f"  {cat:<20} {len(ids):>3}  {mb:>6.3f}  {mc:>6.3f}  {d:>+7.3f}  {bar(d)}{flag}")
        cat_rows[cat] = {"n": len(ids), "base": round(mb, 4), "candidate": round(mc, 4),
                         "delta": round(d, 4), "regression": d <= -REGRESSION_THRESHOLD}
    print("-" * 76)

    # ---- movement ------------------------------------------------------------
    print("\n  ITEM MOVEMENT")
    print(f"    improved   {len(improved):>4}")
    print(f"    regressed  {len(regressed):>4}")
    print(f"    unchanged  {len(unchanged):>4}")
    print(f"    unscored   {len(unscored):>4}")

    if regressed:
        print("\n  WORST REGRESSIONS (item, base -> tuned):")
        for i, sb, sc, d in sorted(regressed, key=lambda x: x[3])[:8]:
            print(f"    {d:>+6.2f}  {i:<24} {sb:.2f} -> {sc:.2f}  [{bi[i]['category']}]")
            print(f"            was: {bi[i]['explanation'][:62]}")
            print(f"            now: {ci[i]['explanation'][:62]}")

    # ---- cost ----------------------------------------------------------------
    print("\n  COST")
    print(f"    {'':<22}{'base':>12}{'tuned':>12}")
    for k, label, unit in (("mean_latency_s", "mean latency", "s"),
                           ("mean_tokens_per_s", "tokens/sec", ""),
                           ("peak_vram_gb", "peak VRAM", "GB"),
                           ("total_output_tokens", "output tokens", "")):
        vb, vc = b["cost"].get(k), c["cost"].get(k)
        print(f"    {label:<22}{vb:>12}{vc:>12}  {unit}")

    # ---- verdict -------------------------------------------------------------
    obj = tier_rows.get("objective")
    print("\n" + "=" * 76)
    if obj is None:
        print("  NO VERDICT — no objectively scored items in common.")
        verdict = "no_objective_items"
    elif blockers:
        print(f"  MIXED — objective {obj['delta']:+.3f}, but {len(blockers)} "
              f"categor{'y' if len(blockers)==1 else 'ies'} regressed past "
              f"{REGRESSION_THRESHOLD:.0%}:")
        for cat, d in sorted(blockers, key=lambda x: x[1]):
            print(f"      {cat} {d:+.3f}")
        print("  A net gain with category regressions is a trade. Decide whether you want it")
        print("  before shipping this adapter.")
        verdict = "mixed_with_regressions"
    elif obj["delta"] > EPSILON:
        print(f"  IMPROVED — objective {obj['base']:.3f} -> {obj['candidate']:.3f} "
              f"({obj['delta']:+.3f}), no category regressed past {REGRESSION_THRESHOLD:.0%}.")
        print(f"  {len(regressed)} individual items still regressed; check they are noise.")
        verdict = "improved"
    elif obj["delta"] < -EPSILON:
        print(f"  WORSE — objective {obj['delta']:+.3f}. The fine-tune hurt the model.")
        verdict = "worse"
    else:
        print("  NO CHANGE on objective items.")
        verdict = "unchanged"
    print("=" * 76)

    report = {
        "base": b["name"], "candidate": c["name"],
        "base_adapter": b.get("adapter"), "candidate_adapter": c.get("adapter"),
        "benchmark_sha256": b["benchmark_sha256"],
        "conditions_matched": not problems,
        "condition_problems": problems,
        "mismatch_override": bool(problems and args.allow_mismatch),
        "verdict": verdict,
        "tiers": tier_rows, "categories": cat_rows,
        "regression_threshold": REGRESSION_THRESHOLD,
        "blocking_regressions": [{"category": c_, "delta": round(d, 4)} for c_, d in blockers],
        "movement": {"improved": len(improved), "regressed": len(regressed),
                     "unchanged": len(unchanged), "unscored": len(unscored)},
        "regressed_items": [{"id": i, "base": sb, "candidate": sc, "delta": round(d, 4),
                             "category": bi[i]["category"]}
                            for i, sb, sc, d in sorted(regressed, key=lambda x: x[3])],
        "improved_items": [{"id": i, "base": sb, "candidate": sc, "delta": round(d, 4),
                            "category": bi[i]["category"]}
                           for i, sb, sc, d in sorted(improved, key=lambda x: -x[3])],
        "cost": {"base": b["cost"], "candidate": c["cost"]},
    }
    out = args.out or os.path.join(HERE, "evaluation", "results",
                                   f"compare_{b['name']}_vs_{c['name']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
