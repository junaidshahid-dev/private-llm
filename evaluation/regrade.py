"""regrade.py — re-score saved model outputs with the current graders. No GPU, no re-run.

    python evaluation/regrade.py evaluation/results/base

results.json stores every model OUTPUT, not just its score. So when a grader is fixed — tool_010
should have been unscored, not 0 — the whole run can be re-scored in seconds from the saved text
instead of spending another 2.5 hours regenerating identical outputs.

It writes results.json back in place, updates the summary, and keeps a one-line record of the
change under "regrade_history" so a score can always be traced to the grader version that
produced it. The raw outputs are never touched, so this is reversible: fix the grader again and
re-run.
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BENCH = os.path.join(HERE, "evaluation", "frozen", "benchmark_v1", "benchmark.jsonl")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python evaluation/regrade.py <results_dir>")
        return 2
    from evaluation.grading import grade

    d = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(HERE, sys.argv[1])
    f = d if d.endswith(".json") else os.path.join(d, "results.json")
    data = json.load(open(f, encoding="utf-8"))
    items = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}

    print("=" * 74)
    print(f"REGRADE — {data['name']}  ({len(data['results'])} items)")
    print("=" * 74)

    changed = []
    for r in data["results"]:
        item = items.get(r["id"])
        if item is None:                       # dev-set item not in the frozen file; leave as is
            continue
        old = r["score"]
        score, tier, why = grade(item, r["output"])
        if score != old:
            changed.append((r["id"], old, score))
            r["score"], r["tier"], r["explanation"] = score, tier, why

    scored = [r for r in data["results"] if r["score"] is not None]
    obj = [r for r in scored if r["grading_type"] in ("exact", "code_test", "structural")]
    rub = [r for r in scored if r["grading_type"] == "rubric"]
    data["summary"] = {
        "objective_n": len(obj),
        "objective_mean": round(sum(r["score"] for r in obj) / len(obj), 4) if obj else None,
        "rubric_n": len(rub),
        "rubric_mean": round(sum(r["score"] for r in rub) / len(rub), 4) if rub else None,
        "unscored_n": len(data["results"]) - len(scored),
    }
    data.setdefault("regrade_history", []).append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "changed": len(changed),
    })

    if changed:
        print(f"\n{len(changed)} score(s) changed:")
        for id, old, new in changed:
            print(f"  {id:22} {old} -> {new}")
    else:
        print("\nno scores changed — graders agree with the saved run")

    s = data["summary"]
    print(f"\n  objective  {s['objective_n']:>3}   mean {s['objective_mean']}")
    print(f"  rubric     {s['rubric_n']:>3}   mean {s['rubric_mean']}")
    print(f"  unscored   {s['unscored_n']:>3}")

    json.dump(data, open(f, "w", encoding="utf-8"), indent=2)
    print(f"\nwritten back: {os.path.relpath(f, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
