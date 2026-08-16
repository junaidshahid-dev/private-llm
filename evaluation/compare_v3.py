"""compare_v3.py — paired head-to-head of two Security-Benchmark-v3 runs (e.g. Moonlight vs Qwen).

    python evaluation/compare_v3.py moonlight qwen25-coder-14b
    python evaluation/compare_v3.py evaluation/results/secv3_moonlight_base/results.json <other.json>

Same benchmark, prompts, RAG, decode, verification, judge, tools — only the MODEL differs (that is
what run_secv3 --model changes). This reads two results.json and reports the averages AND, more
importantly, the PAIRED per-item view (where the two models actually diverge), because an average
hides the thing that matters — Qwen won the tool-selection case Moonlight missed, and missed the
banner caution Moonlight had. CPU-only.
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
RESULTS = os.path.join(HERE, "evaluation", "results")


def resolve(spec: str) -> str:
    """A results.json path, a results dir, or a model tag (secv3_<tag>_base) -> a results.json."""
    if os.path.isfile(spec):
        return spec
    for cand in (os.path.join(spec, "results.json"),
                 os.path.join(RESULTS, spec, "results.json"),
                 os.path.join(RESULTS, f"secv3_{spec}_base", "results.json"),
                 os.path.join(RESULTS, f"secv3_{spec}_rag", "results.json")):
        if os.path.isfile(cand):
            return cand
    sys.exit(f"no results.json for {spec!r} (looked for a file, a dir, and secv3_{spec}_base/rag)")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _cats(rows):
    out = {}
    for r in rows:
        out.setdefault(r["category"], []).append(r["score"])
    return {c: _mean(v) for c, v in out.items()}


def _fmt(x):
    return "  —  " if x is None else f"{x:5.2f}"


def render(a, b) -> str:
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    na = a.get("model", "A").split("/")[-1]
    nb = b.get("model", "B").split("/")[-1]
    ca, cb = _cats(a["results"]), _cats(b["results"])
    L = [f"SECURITY BENCHMARK v3 — HEAD-TO-HEAD   ({len(ra)} items, identical everything but model)",
         "=" * 78, f"{'':22}{na[:26]:>26}{nb[:26]:>26}", "-" * 78]

    def row(label, x, y):
        L.append(f"{label:22}{_fmt(x):>26}{_fmt(y):>26}")

    # TWO overalls: the LLM judge (home-field-biased when it grades its own family) and the
    # INDEPENDENT deterministic anchors (same rubric for both models). When these disagree, trust
    # the deterministic RANKING and a human spot-check — that is what the divergence flags are for.
    row("Overall (judge)", _mean([r["score"] for r in a["results"]]),
        _mean([r["score"] for r in b["results"]]))
    row("Overall (determ.)", _mean([r.get("det_score") for r in a["results"]]),
        _mean([r.get("det_score") for r in b["results"]]))
    for cat in sorted(set(ca) | set(cb)):
        row(cat, ca.get(cat), cb.get(cat))

    # verification + cost (not [0,1] scores, shown raw)
    def blocks(d):
        return sum(1 for r in d["results"] if r.get("verify_verdict") in ("BLOCK", "WARNING"))
    def diverged(d):     # items where judge and deterministic disagree by >= a full rubric point
        return sum(1 for r in d["results"]
                   if r.get("divergence") is not None and r["divergence"] >= 0.34)
    ac, bc = a.get("cost", {}), b.get("cost", {})
    L.append("-" * 78)
    L.append(f"{'Judge≠determ. (⚠)':22}{diverged(a):>26}{diverged(b):>26}")
    L.append(f"{'Verify non-PASS':22}{blocks(a):>26}{blocks(b):>26}")
    L.append(f"{'Latency s/item':22}{str(ac.get('mean_latency_s','?')):>26}"
             f"{str(bc.get('mean_latency_s','?')):>26}")
    L.append(f"{'Peak VRAM GB':22}{str(ac.get('peak_vram_gb','?')):>26}"
             f"{str(bc.get('peak_vram_gb','?')):>26}")

    # ---- paired per-item: where do they diverge? ----------------------------
    L.append("\nPAIRED ITEMS (divergences first — the average hides these):")
    L.append(f"{'id':22}{'cat':16}{na[:10]:>8}{nb[:10]:>8}  {'Δ':>6}  verdicts")
    L.append("-" * 78)
    rows = []
    for i in sorted(set(ra) | set(rb)):
        x = ra.get(i, {}).get("score")
        y = rb.get(i, {}).get("score")
        delta = (y - x) if (x is not None and y is not None) else None
        rows.append((i, ra.get(i, rb[i])["category"], x, y, delta,
                     ra.get(i, {}).get("verify_verdict", "?"),
                     rb.get(i, {}).get("verify_verdict", "?")))
    rows.sort(key=lambda t: (abs(t[4]) if t[4] is not None else -1), reverse=True)
    a_wins = b_wins = ties = 0
    for i, cat, x, y, d, va, vb in rows:
        mark = "" if not d else ("  <<B" if d > 0 else "  A>>")
        if d is not None:
            a_wins += d < 0
            b_wins += d > 0
            ties += d == 0
        ds = "  —  " if d is None else f"{d:+5.2f}"
        L.append(f"{i:22}{cat:16}{_fmt(x):>8}{_fmt(y):>8}  {ds:>6}  {va[:4]:>4}/{vb[:4]:<4}{mark}")
    L.append("-" * 78)
    L.append(f"per-item wins:  {na[:20]} {a_wins}   |   {nb[:20]} {b_wins}   |   tie {ties}")
    L.append("Read the divergent rows, not the average. v3_toolselect_03 (the masscan-vs-web case) "
             "and the banner-caution items are the ones to eyeball by hand.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="results.json / dir / model tag (e.g. moonlight)")
    ap.add_argument("b", help="results.json / dir / model tag (e.g. qwen25-coder-14b)")
    args = ap.parse_args()
    a = json.load(open(resolve(args.a), encoding="utf-8"))
    b = json.load(open(resolve(args.b), encoding="utf-8"))
    print(render(a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
