"""compare_v3_test.py — the head-to-head compare surfaces divergences and counts wins.

    python evaluation/compare_v3_test.py
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

from evaluation.compare_v3 import render, _cats, _mean                        # noqa: E402
from serving.model_spec import resolve                                        # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def data(model, scores):
    return {"model": model, "cost": {"mean_latency_s": 100, "peak_vram_gb": 9.9},
            "results": [{"id": i, "category": c, "score": s, "verify_verdict": "PASS"}
                        for i, c, s in scores]}


def main() -> int:
    print("=" * 70)
    print("COMPARE-V3 TEST — divergences surface, wins counted")
    print("=" * 70)

    # Moonlight loses tool_selection (0 vs 1), ties elsewhere — the real Qwen finding
    moon = data("moonshotai/Moonlight", [("v3_toolselect_03", "tool_selection", 0.0),
                                         ("v3_deser_01", "code_analysis", 1.0),
                                         ("v3_smuggling_01", "multi_step", 1.0)])
    qwen = data("Qwen/Qwen2.5-Coder-14B", [("v3_toolselect_03", "tool_selection", 1.0),
                                           ("v3_deser_01", "code_analysis", 1.0),
                                           ("v3_smuggling_01", "multi_step", 0.5)])

    check("category means computed", _cats(moon["results"])["tool_selection"] == 0.0)
    check("mean skips None", _mean([1.0, None, 0.0]) == 0.5)

    out = render(moon, qwen)
    check("table has Overall row", "Overall" in out)
    check("table has the tool_selection category", "tool_selection" in out)
    check("shows both model names", "Moonlight" in out and "Qwen2.5-Coder-14B" in out)
    check("surfaces the tool-selection divergence line", "v3_toolselect_03" in out)
    check("has Latency and VRAM rows", "Latency" in out and "Peak VRAM" in out)
    # Moonlight wins smuggling (1.0 vs 0.5); Qwen wins toolselect (0 vs 1); deser ties
    check("per-item wins counted both ways", "tie 1" in out and " 1 " in out.replace("|", " "))

    print("\n--- rendered (for eyeballing) ---")
    print(out)

    # alias resolution for --model
    print("\nalias resolution:")
    check("moonlight -> baseline lock", resolve("moonlight") == "MODEL_SPEC.lock.json")
    check("qwen -> coder lock", resolve("qwen") == "MODEL_SPEC.qwen25-coder-14b.lock.json")
    check("qwen-14b -> general lock", resolve("qwen-14b") == "MODEL_SPEC.qwen25-14b.lock.json")
    check("unknown name -> MODEL_SPEC.<name>.lock.json",
          resolve("foo") == "MODEL_SPEC.foo.lock.json")

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL COMPARE TESTS PASS — head-to-head surfaces per-item divergences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
